#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build a sparse occupancy map with the sum-only voxel pipeline.

Standalone offline conversion anchors its grid directly to the raw positions
in the 3DGS checkpoint, using the same canonical layout as TSDF conversion.
The on-the-fly entry point may instead receive an explicit in-memory reference
grid. Both paths share Gaussian filtering, support preparation, splatting, and
the occupied contribution threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GSPAR.mapping.gaussian_occupancy.voxel_preprocess import (
    GaussianSet,
    load_gaussian_arrays,
    preprocess_gaussian_arrays,
    transform_gaussians,
)
from GSPAR.mapping.map_frame_transform import (
    apply_transform,
    load_map_frame_transform,
    transform_metadata,
)
from GSPAR.mapping.gaussian_occupancy.voxel_splat_update import (
    splat_gaussians_dense_sum_gpu,
    voxel_idx_to_world_center,
)
from GSPAR.mapping.voxel_grid_layout import (
    build_grid_spec_from_positions,
    load_reference_grid,
    resolve_voxel_size,
)
from GSPAR.mapping.voxel_map_cli import normalize_output_prefix, resolve_existing_optional_path


@dataclass
class VoxelGridSpec:
    origin: np.ndarray
    scene_min: np.ndarray
    scene_max: np.ndarray
    voxel_dims: np.ndarray
    block_dims: np.ndarray | None = None
    block_size: int | None = None


@dataclass
class Config:
    # Input/output.
    checkpoint: Path = Path(
        "/home/chen/nerficg/output/mipnerf360-test_8/stump/checkpoints/final.pt"
    )
    output_prefix: Path = Path(
        "/home/chen/nerficg/GSPAR/outputs/mipnerf360-test_8/stump/voxel_map"
    )
    reference_occupancy: Path | None = None
    map_frame_transform: Path | None = None
    scene_type: str = "outdoor"
    max_gaussians: int | None = None

    # Grid layout. A reference occupancy overrides origin/dims/voxel_size.
    voxel_size: float = 0.05
    voxel_size_json: Path | None = None
    scene_padding: float = 0.02
    block_size: int = 8

    # Gaussian geometry filtering thresholds shared with on-the-fly conversion.
    geometry_filter: bool = True
    knn_k: int = 8
    density_cell_size_multiplier: float = 2.0
    floater_opacity_threshold: float = 0.04
    isolation_quantile: float = 0.97
    low_density_quantile: float = 0.10
    detail_guard: bool = False
    geometry_filter_max_remove_ratio: float = 0.20

    boundary_outlier_filter: bool = True
    boundary_quantile: float = 0.10
    boundary_low_density_quantile: float = 0.20
    indoor_boundary_quantile: float = 0.001
    indoor_boundary_low_density_quantile: float = 0.20
    boundary_axis_mode: str = "all"
    boundary_tail: str = "both"
    # Gaussian support and occupied-voxel thresholds.
    support_contribution_floor: float = 0.01
    support_radius_cap_quantile: float = 0.995
    max_support_radius_voxels: int = 32
    occupied_contribution_threshold: float = 0.35

    # Shared splat geometry.
    surface_splat_mode: str = "thin"
    normal_support_voxels: float = 1.0
    thin_min_anisotropy: float = 0.0
    voxel_footprint_scale: float = 0.20

    # Same execution defaults as the on-the-fly global builder.
    compute_device: str = "cuda"
    gaussian_chunk_size: int = 2048
    splat_max_pairs_per_batch: int = 500_000
    # Larger logical grids keep the same lattice but use sparse sum storage.
    gpu_dense_accumulator_max_mb: int = 1536


DEFAULT_CONFIG = Config()


def _start_stage(label: str) -> float:
    del label
    return time.perf_counter()


def _finish_stage(
    timings_sec: dict[str, float],
    key: str,
    started_at: float,
    label: str | None = None,
) -> float:
    elapsed = float(time.perf_counter() - started_at)
    timings_sec[key] = elapsed
    del label
    return elapsed


def _timing_summary(t0: float, timings_sec: dict[str, float]) -> dict[str, object]:
    loading_keys = {"load_checkpoint", "load_checkpoint_preprocess"}
    loading = sum(float(timings_sec.get(key, 0.0)) for key in loading_keys)
    saving = sum(
        float(timings_sec.get(key, 0.0))
        for key in ("save_npz", "save_json")
    )
    calculation = sum(
        float(value)
        for key, value in timings_sec.items()
        if key not in loading_keys | {"save_npz", "save_json"}
    )
    return {
        "timing_sec": {
            "loading": loading,
            "calculation": calculation,
            "saving": saving,
            "categorized_total": loading + calculation + saving,
            "wall_total": float(time.perf_counter() - t0),
        }
    }


def parse_args(config: Config) -> Config:
    parser = argparse.ArgumentParser(
        description="Build a sum-threshold sparse voxel map from a 3DGS checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, default=config.checkpoint)
    parser.add_argument("--output-prefix", type=Path, default=config.output_prefix)
    parser.add_argument("--reference-occupancy", type=Path, default=config.reference_occupancy)
    parser.add_argument(
        "--map-frame-transform",
        type=Path,
        default=config.map_frame_transform,
        help="Shared training-world to axis-aligned map-frame JSON.",
    )
    parser.add_argument(
        "--scene-type",
        choices=("indoor", "outdoor"),
        default=config.scene_type,
    )
    parser.add_argument("--max-gaussians", type=int, default=config.max_gaussians)
    parser.add_argument("--voxel-size", type=float, default=config.voxel_size)
    parser.add_argument("--voxel-size-json", type=Path, default=config.voxel_size_json)
    parser.add_argument("--scene-padding", type=float, default=config.scene_padding)
    parser.add_argument("--block-size", type=int, default=config.block_size)

    parser.add_argument(
        "--disable-geometry-filter",
        dest="geometry_filter",
        action="store_false",
        default=config.geometry_filter,
    )
    parser.add_argument(
        "--disable-boundary-outlier-filter",
        dest="boundary_outlier_filter",
        action="store_false",
        default=config.boundary_outlier_filter,
    )
    detail_guard_group = parser.add_mutually_exclusive_group()
    detail_guard_group.add_argument(
        "--enable-detail-guard",
        dest="detail_guard",
        action="store_true",
        help="Enable the optional scale/anisotropy detail guard.",
    )
    detail_guard_group.add_argument(
        "--disable-detail-guard",
        dest="detail_guard",
        action="store_false",
        help="Disable the detail guard (default).",
    )
    parser.set_defaults(detail_guard=config.detail_guard)
    parser.add_argument(
        "--floater-opacity-threshold",
        type=float,
        default=config.floater_opacity_threshold,
    )
    parser.add_argument(
        "--isolation-quantile", type=float, default=config.isolation_quantile
    )
    parser.add_argument(
        "--low-density-quantile", type=float, default=config.low_density_quantile
    )
    parser.add_argument(
        "--boundary-quantile", type=float, default=config.boundary_quantile
    )
    parser.add_argument(
        "--boundary-low-density-quantile",
        type=float,
        default=config.boundary_low_density_quantile,
    )
    parser.add_argument(
        "--geometry-filter-max-remove-ratio",
        type=float,
        default=config.geometry_filter_max_remove_ratio,
    )

    parser.add_argument(
        "--support-contribution-floor",
        type=float,
        default=config.support_contribution_floor,
    )
    parser.add_argument(
        "--support-radius-cap-quantile",
        type=float,
        default=config.support_radius_cap_quantile,
    )
    parser.add_argument(
        "--max-support-radius-voxels",
        type=int,
        default=config.max_support_radius_voxels,
    )
    parser.add_argument(
        "--occupied-contribution-threshold",
        type=float,
        default=config.occupied_contribution_threshold,
    )
    parser.add_argument(
        "--surface-splat-mode",
        choices=("volume", "thin"),
        default=config.surface_splat_mode,
    )
    parser.add_argument(
        "--normal-support-voxels",
        type=float,
        default=config.normal_support_voxels,
    )
    parser.add_argument(
        "--thin-min-anisotropy",
        type=float,
        default=config.thin_min_anisotropy,
    )
    parser.add_argument(
        "--voxel-footprint-scale",
        type=float,
        default=config.voxel_footprint_scale,
    )
    parser.add_argument(
        "--gaussian-chunk-size", type=int, default=config.gaussian_chunk_size
    )
    parser.add_argument(
        "--splat-max-pairs-per-batch",
        type=int,
        default=config.splat_max_pairs_per_batch,
    )
    parser.add_argument(
        "--gpu-dense-accumulator-max-mb",
        type=int,
        default=config.gpu_dense_accumulator_max_mb,
    )

    args = parser.parse_args()
    return replace(
        config,
        checkpoint=args.checkpoint,
        output_prefix=normalize_output_prefix(args.output_prefix),
        reference_occupancy=resolve_existing_optional_path(
            args.reference_occupancy,
            default_suffix=".npz",
        ),
        map_frame_transform=args.map_frame_transform,
        scene_type=args.scene_type,
        max_gaussians=args.max_gaussians,
        voxel_size=args.voxel_size,
        voxel_size_json=args.voxel_size_json,
        scene_padding=args.scene_padding,
        block_size=args.block_size,
        geometry_filter=args.geometry_filter,
        boundary_outlier_filter=args.boundary_outlier_filter,
        detail_guard=args.detail_guard,
        floater_opacity_threshold=args.floater_opacity_threshold,
        isolation_quantile=args.isolation_quantile,
        low_density_quantile=args.low_density_quantile,
        boundary_quantile=args.boundary_quantile,
        boundary_low_density_quantile=args.boundary_low_density_quantile,
        geometry_filter_max_remove_ratio=args.geometry_filter_max_remove_ratio,
        support_contribution_floor=args.support_contribution_floor,
        support_radius_cap_quantile=args.support_radius_cap_quantile,
        max_support_radius_voxels=args.max_support_radius_voxels,
        occupied_contribution_threshold=args.occupied_contribution_threshold,
        surface_splat_mode=args.surface_splat_mode,
        normal_support_voxels=args.normal_support_voxels,
        thin_min_anisotropy=args.thin_min_anisotropy,
        voxel_footprint_scale=args.voxel_footprint_scale,
        gaussian_chunk_size=args.gaussian_chunk_size,
        splat_max_pairs_per_batch=args.splat_max_pairs_per_batch,
        gpu_dense_accumulator_max_mb=args.gpu_dense_accumulator_max_mb,
    )


def _apply_scene_type_overrides(cfg: Config) -> Config:
    scene_type = str(cfg.scene_type).strip().lower()
    if scene_type not in {"indoor", "outdoor"}:
        raise ValueError(f"scene_type must be indoor/outdoor, got {cfg.scene_type!r}.")
    if scene_type == "indoor":
        return replace(
            cfg,
            scene_type=scene_type,
            boundary_quantile=cfg.indoor_boundary_quantile,
            boundary_low_density_quantile=cfg.indoor_boundary_low_density_quantile,
        )
    return replace(cfg, scene_type=scene_type)


def _clip_quantile(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}.")
    return float(np.clip(value, 0.0, 1.0))


def _safe_quantile(values: np.ndarray, quantile: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.quantile(arr, _clip_quantile(quantile, "quantile")))


def _subset_gaussians(gaussians: GaussianSet, mask: np.ndarray) -> GaussianSet:
    keep = np.asarray(mask, dtype=bool).reshape(-1)
    return GaussianSet(
        positions=gaussians.positions[keep].astype(np.float32),
        scales=gaussians.scales[keep].astype(np.float32),
        rotations=gaussians.rotations[keep].astype(np.float32),
        opacities=gaussians.opacities[keep].astype(np.float32),
    )


def _fallback_hash_density(
    positions: np.ndarray,
    voxel_size: float,
    density_cell_size_multiplier: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    cell_size = max(
        float(voxel_size) * max(float(density_cell_size_multiplier), 1e-3),
        1e-6,
    )
    origin = positions.min(axis=0)
    cells = np.floor((positions - origin[None, :]) / cell_size).astype(np.int64)
    unique_cells, inverse, counts = np.unique(
        cells,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    count_by_cell = {
        tuple(cell.tolist()): int(count)
        for cell, count in zip(unique_cells, counts)
    }
    offsets = np.array(
        [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
        ],
        dtype=np.int64,
    )
    neighbor_counts_unique = np.empty((unique_cells.shape[0],), dtype=np.float32)
    for idx, cell in enumerate(unique_cells):
        total = 0
        for offset in offsets:
            total += count_by_cell.get(tuple((cell + offset).tolist()), 0)
        neighbor_counts_unique[idx] = float(total)
    neighbor_counts = neighbor_counts_unique[inverse]
    density = neighbor_counts / max((3.0 * cell_size) ** 3, 1e-18)
    mean_distance = cell_size / np.cbrt(
        np.maximum(neighbor_counts - 1.0, 1.0)
    )
    return (
        mean_distance.astype(np.float32),
        density.astype(np.float32),
        "hash_grid_27cell",
    )


def _compute_local_geometry(
    positions: np.ndarray,
    voxel_size: float,
    knn_k: int,
    density_cell_size_multiplier: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    n = int(positions.shape[0])
    if n <= 1:
        return (
            np.zeros((n,), dtype=np.float32),
            np.ones((n,), dtype=np.float32),
            "degenerate",
        )
    k = max(1, min(int(knn_k), n - 1))
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(positions.astype(np.float64))
        try:
            distances, _ = tree.query(
                positions.astype(np.float64),
                k=k + 1,
                workers=-1,
            )
        except TypeError:
            distances, _ = tree.query(
                positions.astype(np.float64),
                k=k + 1,
            )
        distances = np.asarray(distances, dtype=np.float64)
        if distances.ndim == 1:
            distances = distances[:, None]
        neighbor_distances = distances[:, 1:]
        mean_distance = np.mean(neighbor_distances, axis=1)
        kth_distance = np.maximum(neighbor_distances[:, -1], 1e-8)
        density = float(k) / (
            (4.0 / 3.0) * np.pi * np.power(kth_distance, 3.0)
        )
        return (
            mean_distance.astype(np.float32),
            density.astype(np.float32),
            "scipy_cKDTree",
        )
    except Exception:
        return _fallback_hash_density(
            positions=positions,
            voxel_size=voxel_size,
            density_cell_size_multiplier=density_cell_size_multiplier,
        )


def _filter_gaussians_geometry_aware_with_mask(
    gaussians: GaussianSet,
    cfg: Config,
    voxel_size: float,
) -> tuple[GaussianSet, np.ndarray, dict[str, object]]:
    n = int(gaussians.positions.shape[0])
    stats: dict[str, object] = {
        "enabled": bool(cfg.geometry_filter),
        "scene_type": str(cfg.scene_type),
        "input_gaussians": n,
        "method": "opacity_density_isolation_boundary_filter",
        "config": {
            "knn_k": int(cfg.knn_k),
            "density_cell_size_multiplier": float(
                cfg.density_cell_size_multiplier
            ),
            "floater_opacity_threshold": float(
                cfg.floater_opacity_threshold
            ),
            "isolation_quantile": float(cfg.isolation_quantile),
            "low_density_quantile": float(cfg.low_density_quantile),
            "detail_guard": bool(cfg.detail_guard),
            "geometry_filter_max_remove_ratio": float(
                cfg.geometry_filter_max_remove_ratio
            ),
            "boundary_outlier_filter": bool(cfg.boundary_outlier_filter),
            "boundary_quantile": float(cfg.boundary_quantile),
            "boundary_low_density_quantile": float(
                cfg.boundary_low_density_quantile
            ),
            "boundary_axis_mode": str(cfg.boundary_axis_mode),
            "boundary_tail": str(cfg.boundary_tail),
        },
    }
    if (not cfg.geometry_filter) or n <= 2:
        keep_mask = np.ones((n,), dtype=bool)
        stats.update(
            {
                "num_removed_total": 0,
                "num_output_gaussians": n,
                "local_geometry_method": "skipped",
            }
        )
        return gaussians, keep_mask, stats

    opacities = np.asarray(gaussians.opacities, dtype=np.float32).reshape(-1)
    scales = np.asarray(gaussians.scales, dtype=np.float32).reshape(-1, 3)
    scale_max = np.max(scales, axis=1)
    scale_min = np.maximum(np.min(scales, axis=1), 1e-12)
    anisotropy = scale_max / scale_min
    mean_knn_distance, local_density, local_method = _compute_local_geometry(
        positions=gaussians.positions.astype(np.float32),
        voxel_size=voxel_size,
        knn_k=cfg.knn_k,
        density_cell_size_multiplier=cfg.density_cell_size_multiplier,
    )

    floater_opacity_th = max(
        0.0,
        min(1.0, float(cfg.floater_opacity_threshold)),
    )
    isolation_th = _safe_quantile(mean_knn_distance, cfg.isolation_quantile)
    low_density_th = _safe_quantile(local_density, cfg.low_density_quantile)
    opacity_low_th = max(floater_opacity_th, _safe_quantile(opacities, 0.25))
    scale_boundary_th = _safe_quantile(scale_max, 0.75)
    scale_large_th = _safe_quantile(scale_max, 0.985)
    anisotropy_high_th = _safe_quantile(anisotropy, 0.95)
    density_weak_th = _safe_quantile(local_density, 0.20)

    floater_candidate = (
        (opacities <= floater_opacity_th)
        & (mean_knn_distance >= isolation_th)
        & (local_density <= low_density_th)
    )

    boundary_candidate = np.zeros((n,), dtype=bool)
    boundary_density_th = None
    boundary_lo = None
    boundary_hi = None
    boundary_axes = None
    if bool(cfg.boundary_outlier_filter):
        boundary_q = _clip_quantile(cfg.boundary_quantile, "boundary_quantile")
        if 0.0 < boundary_q <= 0.5:
            boundary_lo = np.quantile(gaussians.positions, boundary_q, axis=0)
            boundary_hi = np.quantile(
                gaussians.positions,
                1.0 - boundary_q,
                axis=0,
            )
            if cfg.boundary_axis_mode == "auto_up":
                robust_lo = np.quantile(gaussians.positions, 0.05, axis=0)
                robust_hi = np.quantile(gaussians.positions, 0.95, axis=0)
                boundary_axes = [int(np.argmin(robust_hi - robust_lo))]
            elif cfg.boundary_axis_mode == "all":
                boundary_axes = [0, 1, 2]
            else:
                raise ValueError(
                    f"Unsupported boundary_axis_mode: {cfg.boundary_axis_mode}"
                )

            boundary_mask = np.zeros((n,), dtype=bool)
            for axis in boundary_axes:
                if cfg.boundary_tail in ("lower", "both"):
                    boundary_mask |= (
                        gaussians.positions[:, axis] < boundary_lo[axis]
                    )
                if cfg.boundary_tail in ("upper", "both"):
                    boundary_mask |= (
                        gaussians.positions[:, axis] > boundary_hi[axis]
                    )
                if cfg.boundary_tail not in ("upper", "lower", "both"):
                    raise ValueError(
                        f"Unsupported boundary_tail: {cfg.boundary_tail}"
                    )
            boundary_density_th = _safe_quantile(
                local_density,
                cfg.boundary_low_density_quantile,
            )
            boundary_candidate = (
                boundary_mask
                & (local_density <= boundary_density_th)
                & (
                    (opacities <= opacity_low_th)
                    | (scale_max >= scale_boundary_th)
                )
            )

    large_scale_delete = (
        (scale_max >= scale_large_th)
        & (local_density <= density_weak_th)
        & (
            (anisotropy >= anisotropy_high_th)
            | (opacities <= opacity_low_th)
        )
    )

    candidate_raw = (
        floater_candidate
        | boundary_candidate
        | large_scale_delete
    )

    stable_mask = ~candidate_raw
    stable_scale_min = scale_min[stable_mask] if np.any(stable_mask) else scale_min
    stable_scale_max = scale_max[stable_mask] if np.any(stable_mask) else scale_max
    stable_anisotropy = (
        anisotropy[stable_mask] if np.any(stable_mask) else anisotropy
    )
    thin_scale_th = _safe_quantile(stable_scale_min, 0.10)
    anisotropy_th = _safe_quantile(stable_anisotropy, 0.90)
    detail_max_scale_th = _safe_quantile(stable_scale_max, 0.75)
    detail_guard = np.zeros((n,), dtype=bool)
    if bool(cfg.detail_guard):
        thin_detail = scale_min <= thin_scale_th
        anisotropic_detail = (
            (anisotropy >= anisotropy_th)
            & (scale_max <= detail_max_scale_th)
        )
        detail_guard = candidate_raw & (thin_detail | anisotropic_detail)

    candidate = candidate_raw & (~detail_guard)
    eps = 1e-8
    opacity_score = np.maximum(
        (floater_opacity_th - opacities) / (abs(floater_opacity_th) + eps),
        0.0,
    )
    isolation_score = mean_knn_distance / max(isolation_th, eps)
    density_score = low_density_th / np.maximum(local_density, eps)
    removal_score = opacity_score + isolation_score + density_score

    max_remove_ratio = float(
        np.clip(cfg.geometry_filter_max_remove_ratio, 0.0, 1.0)
    )
    max_remove = int(np.floor(max_remove_ratio * n))
    candidate_idx = np.flatnonzero(candidate)
    candidate_count = int(candidate_idx.size)
    remove_mask = np.zeros((n,), dtype=bool)
    if max_remove > 0 and candidate_idx.size > 0:
        if candidate_idx.size > max_remove:
            order = np.argsort(removal_score[candidate_idx])[::-1]
            candidate_idx = candidate_idx[order[:max_remove]]
        remove_mask[candidate_idx] = True

    keep_mask = ~remove_mask
    if not np.any(keep_mask):
        keep_mask[int(np.argmax(opacities))] = True
        remove_mask = ~keep_mask

    filtered = _subset_gaussians(gaussians, keep_mask)
    stats.update(
        {
            "local_geometry_method": local_method,
            "num_candidates_total": int(np.count_nonzero(candidate_raw)),
            "num_candidates_after_detail_guard": candidate_count,
            "num_candidates_after_safety_cap": int(candidate_idx.size),
            "num_floater_candidates": int(
                np.count_nonzero(floater_candidate)
            ),
            "num_boundary_outlier_candidates": int(
                np.count_nonzero(boundary_candidate)
            ),
            "num_large_scale_candidates": int(
                np.count_nonzero(large_scale_delete)
            ),
            "num_detail_guarded_candidates": int(np.count_nonzero(detail_guard)),
            "num_removed_total": int(np.count_nonzero(remove_mask)),
            "num_removed_low_opacity_isolated": int(
                np.count_nonzero(remove_mask & floater_candidate)
            ),
            "num_removed_boundary_sparse": int(
                np.count_nonzero(remove_mask & boundary_candidate)
            ),
            "num_removed_large_scale": int(
                np.count_nonzero(remove_mask & large_scale_delete)
            ),
            "num_output_gaussians": int(filtered.positions.shape[0]),
            "max_remove_ratio": max_remove_ratio,
            "thresholds": {
                "floater_opacity": float(floater_opacity_th),
                "mean_knn_distance_isolated": float(isolation_th),
                "local_density_low": float(low_density_th),
                "detail_thin_scale": float(thin_scale_th),
                "detail_anisotropy": float(anisotropy_th),
                "detail_max_scale": float(detail_max_scale_th),
                "boundary_density_low": (
                    float(boundary_density_th)
                    if boundary_density_th is not None
                    else None
                ),
                "boundary_low": (
                    boundary_lo.astype(np.float32).tolist()
                    if boundary_lo is not None
                    else None
                ),
                "boundary_high": (
                    boundary_hi.astype(np.float32).tolist()
                    if boundary_hi is not None
                    else None
                ),
                "boundary_axes": boundary_axes,
                "boundary_tail": cfg.boundary_tail,
                "opacity_low": float(opacity_low_th),
                "scale_boundary": float(scale_boundary_th),
                "scale_large": float(scale_large_th),
                "anisotropy_high": float(anisotropy_high_th),
                "density_weak": float(density_weak_th),
            },
        }
    )
    return filtered, keep_mask, stats


def filter_gaussians_geometry_aware(
    gaussians: GaussianSet,
    *,
    scene_type: str,
    voxel_size: float,
    enabled: bool = True,
) -> tuple[GaussianSet, np.ndarray, dict[str, object]]:
    """Apply the default sparse-map geometry filter and return its keep mask.

    The keep mask lets renderers apply the same selection to non-geometric
    per-Gaussian state such as spherical-harmonic features.  Scene-specific
    indoor/outdoor overrides are resolved here so callers cannot accidentally
    bypass them.
    """

    cfg = _apply_scene_type_overrides(
        replace(
            DEFAULT_CONFIG,
            scene_type=str(scene_type),
            geometry_filter=bool(enabled),
        )
    )
    return _filter_gaussians_geometry_aware_with_mask(
        gaussians=gaussians,
        cfg=cfg,
        voxel_size=float(voxel_size),
    )


def _filter_gaussians_geometry_aware(
    gaussians: GaussianSet,
    cfg: Config,
    voxel_size: float,
) -> tuple[GaussianSet, dict[str, object]]:
    """Backward-compatible wrapper used by the sparse voxel builders."""

    filtered, _, stats = _filter_gaussians_geometry_aware_with_mask(
        gaussians=gaussians,
        cfg=cfg,
        voxel_size=voxel_size,
    )
    return filtered, stats


def _compute_contribution_support_radii(
    gaussians: GaussianSet,
    contribution_floor: float,
) -> np.ndarray:
    floor_value = float(contribution_floor)
    if not np.isfinite(floor_value) or floor_value <= 0.0:
        raise ValueError(
            "support_contribution_floor must be > 0, "
            f"got {contribution_floor}."
        )
    opacities = np.asarray(gaussians.opacities, dtype=np.float64).reshape(-1)
    scales = np.asarray(gaussians.scales, dtype=np.float64).reshape(-1, 3)
    safe_alpha = np.clip(opacities, 1e-12, 1.0)
    rho_cut = np.maximum(2.0 * np.log(safe_alpha / floor_value), 0.0)
    return (scales * np.sqrt(rho_cut)[:, None]).astype(np.float32)


def _cap_support_radii(
    support_radii: np.ndarray,
    cfg: Config,
    voxel_size: float,
) -> tuple[np.ndarray, dict[str, object]]:
    radii = np.asarray(support_radii, dtype=np.float32).reshape(-1, 3)
    if radii.shape[0] == 0:
        return radii, {
            "enabled": False,
            "radius_cap": None,
            "num_clipped_gaussians": 0,
        }

    caps: list[float] = []
    cap_sources: dict[str, float] = {}
    q = float(cfg.support_radius_cap_quantile)
    if np.isfinite(q) and 0.0 < q < 1.0:
        q_cap = float(np.quantile(np.max(radii, axis=1), q))
        if np.isfinite(q_cap) and q_cap > 0.0:
            caps.append(q_cap)
            cap_sources["quantile"] = q_cap

    max_voxels = int(cfg.max_support_radius_voxels)
    if max_voxels > 0:
        voxel_cap = float(max_voxels) * float(voxel_size)
        if np.isfinite(voxel_cap) and voxel_cap > 0.0:
            caps.append(voxel_cap)
            cap_sources["max_voxels"] = voxel_cap

    if not caps:
        return radii, {
            "enabled": False,
            "radius_cap": None,
            "num_clipped_gaussians": 0,
            "sources": cap_sources,
        }

    cap = float(min(caps))
    clipped = np.minimum(radii, cap).astype(np.float32)
    return clipped, {
        "enabled": True,
        "radius_cap": cap,
        "support_radius_cap_quantile": q,
        "max_support_radius_voxels": max_voxels,
        "num_clipped_gaussians": int(
            np.count_nonzero(np.any(radii > cap, axis=1))
        ),
        "sources": cap_sources,
    }


def _support_aabb_half(
    gaussians: GaussianSet,
    support_radii: np.ndarray,
) -> np.ndarray:
    r2 = np.square(np.asarray(support_radii, dtype=np.float32))
    rot2 = np.square(gaussians.rotations).astype(np.float32)
    return np.sqrt(np.sum(rot2 * r2[:, None, :], axis=2)).astype(np.float32)


def _build_grid_spec_from_gaussian_support(
    gaussians: GaussianSet,
    support_radii: np.ndarray,
    voxel_size: float,
    scene_padding: float,
    voxel_footprint_scale: float = 0.0,
) -> VoxelGridSpec:
    if gaussians.positions.shape[0] == 0:
        raise RuntimeError(
            "No gaussians remain for filtered-support grid layout."
        )
    aabb_half = _support_aabb_half(gaussians, support_radii)
    aabb_half = aabb_half + (
        0.5
        * float(voxel_size)
        * max(0.0, float(voxel_footprint_scale))
    )
    scene_min = (
        (gaussians.positions - aabb_half).min(axis=0).astype(np.float32)
        - float(scene_padding)
    )
    scene_max = (
        (gaussians.positions + aabb_half).max(axis=0).astype(np.float32)
        + float(scene_padding)
    )
    extent = np.maximum(
        scene_max - scene_min,
        np.full((3,), float(voxel_size), dtype=np.float32),
    )
    voxel_dims = np.maximum(
        np.ceil(extent / float(voxel_size)).astype(np.int32),
        np.ones((3,), dtype=np.int32),
    )
    return VoxelGridSpec(
        origin=scene_min.astype(np.float32),
        scene_min=scene_min.astype(np.float32),
        scene_max=scene_max.astype(np.float32),
        voxel_dims=voxel_dims.astype(np.int32),
    )


def _build_grid_spec_from_reference(
    reference_grid: dict[str, object],
) -> VoxelGridSpec:
    block_dims = reference_grid.get("block_dims")
    block_size = reference_grid.get("block_size")
    return VoxelGridSpec(
        origin=np.asarray(reference_grid["origin"], dtype=np.float32).reshape(3),
        scene_min=np.asarray(
            reference_grid["scene_min"],
            dtype=np.float32,
        ).reshape(3),
        scene_max=np.asarray(
            reference_grid["scene_max"],
            dtype=np.float32,
        ).reshape(3),
        voxel_dims=np.asarray(
            reference_grid["voxel_dims"],
            dtype=np.int32,
        ).reshape(3),
        block_dims=(
            None
            if block_dims is None
            else np.asarray(block_dims, dtype=np.int32).reshape(3)
        ),
        block_size=None if block_size is None else int(block_size),
    )


def _build_canonical_grid_spec(
    positions: np.ndarray,
    voxel_size: float,
    block_size: int,
    scene_padding: float,
) -> tuple[VoxelGridSpec, int]:
    """Build the same checkpoint-anchored lattice used by the TSDF converter."""

    shared_grid = build_grid_spec_from_positions(
        positions=positions,
        voxel_size=float(voxel_size),
        block_size=int(block_size),
        scene_padding=float(scene_padding),
    )
    return VoxelGridSpec(
        origin=shared_grid.origin.astype(np.float32),
        scene_min=shared_grid.scene_min.astype(np.float32),
        scene_max=shared_grid.scene_max.astype(np.float32),
        voxel_dims=shared_grid.voxel_dims.astype(np.int32),
        block_dims=shared_grid.block_dims.astype(np.int32),
        block_size=int(shared_grid.block_size),
    ), int(positions.shape[0])


def _radius_summary(radii: np.ndarray) -> dict[str, list[float]]:
    arr = np.asarray(radii, dtype=np.float32).reshape(-1, 3)
    if arr.shape[0] == 0:
        zero = [0.0, 0.0, 0.0]
        return {"min": zero, "max": zero, "mean": zero}
    return {
        "min": np.min(arr, axis=0).astype(np.float32).tolist(),
        "max": np.max(arr, axis=0).astype(np.float32).tolist(),
        "mean": np.mean(arr, axis=0).astype(np.float32).tolist(),
    }


def build_sparse_voxel_payload(
    gaussians: GaussianSet,
    *,
    config: Config,
    reference_grid: Mapping[str, object] | None,
    reset_cuda_peak_memory_stats: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Build a global occupancy payload from an already loaded Gaussian set.

    This entry point performs no checkpoint/reference-grid I/O and does not
    persist the result.  Callers that need a reference-aligned grid must pass
    its metadata in memory.  Checkpoint loading and output serialization can
    therefore stay outside the reported build time.
    """

    started = time.perf_counter()
    stage_timings_sec: dict[str, float] = {}
    cfg = _apply_scene_type_overrides(config)

    if int(gaussians.positions.shape[0]) == 0:
        raise RuntimeError("No gaussians were provided for voxelization.")
    num_gaussians_loaded = int(gaussians.positions.shape[0])

    if reference_grid is not None:
        reference_values = dict(reference_grid)
        missing = [
            key
            for key in ("origin", "voxel_size", "voxel_dims")
            if key not in reference_values
        ]
        if missing:
            raise ValueError(f"reference_grid missing keys: {missing}")
        voxel_size = float(reference_values["voxel_size"])
        origin = np.asarray(reference_values["origin"], dtype=np.float32).reshape(3)
        voxel_dims = np.asarray(reference_values["voxel_dims"], dtype=np.int32).reshape(3)
        reference_values.setdefault("scene_min", origin)
        reference_values.setdefault(
            "scene_max",
            origin + voxel_dims.astype(np.float32) * voxel_size,
        )
    else:
        if cfg.voxel_size_json is not None:
            raise ValueError(
                "build_sparse_voxel_payload does not read voxel_size_json; "
                "resolve it before the call and set config.voxel_size."
            )
        reference_values = None
        voxel_size = float(cfg.voxel_size)

    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError(f"voxel_size must be positive and finite, got {voxel_size}.")

    stage_start = _start_stage("geometry_filter")
    filtered, geometry_filter_info = _filter_gaussians_geometry_aware(
        gaussians=gaussians,
        cfg=cfg,
        voxel_size=voxel_size,
    )
    geometry_filter_elapsed = _finish_stage(
        stage_timings_sec,
        "geometry_filter",
        stage_start,
    )
    geometry_filter_info["elapsed_sec"] = geometry_filter_elapsed
    if filtered.positions.shape[0] == 0:
        raise RuntimeError("No gaussians remain after geometry filtering.")

    stage_start = _start_stage("support_radius_preparation")
    support_radii = _compute_contribution_support_radii(
        gaussians=filtered,
        contribution_floor=cfg.support_contribution_floor,
    )
    support_mask = np.any(support_radii > 0.0, axis=1)
    num_gaussians_without_support = int(np.count_nonzero(~support_mask))
    if not np.any(support_mask):
        raise RuntimeError(
            "No gaussian can reach support_contribution_floor even at its center."
        )
    if num_gaussians_without_support > 0:
        filtered = _subset_gaussians(filtered, support_mask)
        support_radii = support_radii[support_mask].astype(np.float32)
    support_radii_uncapped = np.array(support_radii, copy=True)
    support_radii, support_cap_info = _cap_support_radii(
        support_radii=support_radii,
        cfg=cfg,
        voxel_size=voxel_size,
    )
    _finish_stage(
        stage_timings_sec,
        "support_radius_preparation",
        stage_start,
    )

    stage_start = _start_stage("grid_layout")
    if reference_values is not None:
        grid = _build_grid_spec_from_reference(reference_values)
        grid_source = "reference_grid"
    else:
        grid = _build_grid_spec_from_gaussian_support(
            gaussians=filtered,
            support_radii=support_radii,
            voxel_size=voxel_size,
            scene_padding=cfg.scene_padding,
            voxel_footprint_scale=cfg.voxel_footprint_scale,
        )
        grid_source = "filtered_support"
    _finish_stage(stage_timings_sec, "grid_layout", stage_start)

    stage_start = _start_stage("voxel_splat")
    occupied_idx, occupied_contribution, voxelization_info = (
        splat_gaussians_dense_sum_gpu(
            gaussians=filtered,
            support_radii=support_radii,
            origin=grid.origin,
            voxel_dims=grid.voxel_dims,
            voxel_size=voxel_size,
            contribution_floor=cfg.support_contribution_floor,
            occupied_threshold=cfg.occupied_contribution_threshold,
            voxel_footprint_scale=cfg.voxel_footprint_scale,
            surface_splat_mode=cfg.surface_splat_mode,
            normal_support_voxels=cfg.normal_support_voxels,
            thin_min_anisotropy=cfg.thin_min_anisotropy,
            requested_device=cfg.compute_device,
            gaussian_chunk_size=cfg.gaussian_chunk_size,
            max_pairs_per_batch=cfg.splat_max_pairs_per_batch,
            max_accumulator_bytes=max(
                1,
                int(cfg.gpu_dense_accumulator_max_mb),
            )
            * 1024
            * 1024,
            reset_peak_memory_stats=bool(reset_cuda_peak_memory_stats),
        )
    )
    voxelization_info = dict(voxelization_info)
    voxelization_info["support_radius_cap"] = support_cap_info
    _finish_stage(stage_timings_sec, "voxel_splat", stage_start)

    payload: dict[str, np.ndarray] = {
        "payload_format": np.asarray(["on_the_fly_compact_occupancy_v2"]),
        "voxel_size": np.asarray([voxel_size], dtype=np.float32),
        "origin": grid.origin.astype(np.float32),
        "scene_min": grid.scene_min.astype(np.float32),
        "scene_max": grid.scene_max.astype(np.float32),
        "voxel_dims": grid.voxel_dims.astype(np.int32),
        "occupied_voxel_idx": np.ascontiguousarray(
            np.asarray(occupied_idx, dtype=np.int32).reshape(-1, 3)
        ),
        "occupied_contribution": np.ascontiguousarray(
            np.asarray(occupied_contribution, dtype=np.float32).reshape(-1)
        ),
    }
    if grid.block_size is not None and grid.block_dims is not None:
        payload["block_size"] = np.asarray([int(grid.block_size)], dtype=np.int32)
        payload["block_dims"] = grid.block_dims.astype(np.int32)

    support_radius_stats = _radius_summary(support_radii)
    support_radius_uncapped_stats = _radius_summary(support_radii_uncapped)
    stats: dict[str, object] = {
        "grid_source": grid_source,
        "voxelization": "contribution_sum_gaussian_splatting",
        "voxelization_info": voxelization_info,
        "geometry_filter": geometry_filter_info,
        "support_radius_cap": support_cap_info,
        "num_gaussians_loaded": num_gaussians_loaded,
        "num_gaussians_without_contribution_support": num_gaussians_without_support,
        "num_gaussians_filtered": int(filtered.positions.shape[0]),
        "gaussian_retention_ratio": float(
            filtered.positions.shape[0] / num_gaussians_loaded
        ),
        "num_gaussians_removed": int(
            num_gaussians_loaded - filtered.positions.shape[0]
        ),
        "num_occupied_voxels": int(payload["occupied_voxel_idx"].shape[0]),
        "num_voxels_total": int(np.prod(grid.voxel_dims.astype(np.int64))),
        "voxel_size": voxel_size,
        "origin": grid.origin.tolist(),
        "scene_min": grid.scene_min.tolist(),
        "scene_max": grid.scene_max.tolist(),
        "voxel_dims": grid.voxel_dims.tolist(),
        "support_radius_min": support_radius_stats["min"],
        "support_radius_max": support_radius_stats["max"],
        "support_radius_mean": support_radius_stats["mean"],
        "support_radius_uncapped_min": support_radius_uncapped_stats["min"],
        "support_radius_uncapped_max": support_radius_uncapped_stats["max"],
        "support_radius_uncapped_mean": support_radius_uncapped_stats["mean"],
        "stage_timings_sec": dict(stage_timings_sec),
        "pipeline_elapsed_sec": float(time.perf_counter() - started),
    }
    return payload, stats


def run_pipeline(cfg: Config) -> dict:
    t0 = time.perf_counter()
    stage_timings_sec: dict[str, float] = {}
    cfg = _apply_scene_type_overrides(cfg)

    checkpoint_path = cfg.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    output_prefix = normalize_output_prefix(cfg.output_prefix).expanduser().resolve()
    map_transform, map_transform_info = load_map_frame_transform(cfg.map_frame_transform)
    map_transform_meta = transform_metadata(map_transform, map_transform_info)

    stage_start = _start_stage("resolve_voxel_size")
    reference_grid = (
        load_reference_grid(cfg.reference_occupancy)
        if cfg.reference_occupancy is not None
        else None
    )
    if reference_grid is not None:
        voxel_size = float(reference_grid["voxel_size"])
        voxel_size_json = (
            str(cfg.voxel_size_json.expanduser().resolve())
            if cfg.voxel_size_json is not None
            else None
        )
    else:
        voxel_size, voxel_size_json = resolve_voxel_size(
            cfg.voxel_size,
            cfg.voxel_size_json,
        )
    _finish_stage(stage_timings_sec, "resolve_voxel_size", stage_start)

    stage_start = _start_stage("load_checkpoint")
    gaussian_arrays = load_gaussian_arrays(checkpoint_path)
    _finish_stage(stage_timings_sec, "load_checkpoint", stage_start)

    stage_start = _start_stage("preprocess_gaussians")
    gaussians = preprocess_gaussian_arrays(
        gaussian_arrays,
        max_gaussians=cfg.max_gaussians,
    )
    gaussians = transform_gaussians(gaussians, map_transform)
    grid_checkpoint_positions = apply_transform(gaussian_arrays[0], map_transform)
    _finish_stage(
        stage_timings_sec,
        "preprocess_gaussians",
        stage_start,
    )
    if gaussians.positions.shape[0] == 0:
        raise RuntimeError("No gaussians were loaded from the checkpoint.")
    num_gaussians_loaded = int(gaussians.positions.shape[0])

    stage_start = _start_stage("geometry_filter")
    gaussians, geometry_filter_info = _filter_gaussians_geometry_aware(
        gaussians=gaussians,
        cfg=cfg,
        voxel_size=voxel_size,
    )
    geometry_filter_elapsed = _finish_stage(
        stage_timings_sec,
        "geometry_filter",
        stage_start,
    )
    geometry_filter_info["elapsed_sec"] = geometry_filter_elapsed
    if gaussians.positions.shape[0] == 0:
        raise RuntimeError("No gaussians remain after geometry filtering.")

    stage_start = _start_stage("support_radius_preparation")
    support_radii = _compute_contribution_support_radii(
        gaussians=gaussians,
        contribution_floor=cfg.support_contribution_floor,
    )
    support_mask = np.any(support_radii > 0.0, axis=1)
    num_gaussians_without_support = int(np.count_nonzero(~support_mask))
    if not np.any(support_mask):
        raise RuntimeError(
            "No gaussian can reach support_contribution_floor even at its center."
        )
    if num_gaussians_without_support > 0:
        gaussians = _subset_gaussians(gaussians, support_mask)
        support_radii = support_radii[support_mask].astype(np.float32)
    support_radii_uncapped = np.array(support_radii, copy=True)
    support_radii, support_cap_info = _cap_support_radii(
        support_radii=support_radii,
        cfg=cfg,
        voxel_size=voxel_size,
    )
    _finish_stage(
        stage_timings_sec,
        "support_radius_preparation",
        stage_start,
    )

    stage_start = _start_stage("grid_layout")
    grid_num_checkpoint_positions: int | None = None
    if reference_grid is not None:
        grid = _build_grid_spec_from_reference(reference_grid)
        grid_source = "reference_occupancy"
    else:
        grid, grid_num_checkpoint_positions = (
            _build_canonical_grid_spec(
                positions=grid_checkpoint_positions,
                voxel_size=voxel_size,
                block_size=cfg.block_size,
                scene_padding=cfg.scene_padding,
            )
        )
        grid_source = "checkpoint_raw_positions_canonical"
    _finish_stage(stage_timings_sec, "grid_layout", stage_start)

    stage_start = _start_stage("voxel_splat")
    occupied_idx, occupied_contribution, voxelization_info = (
        splat_gaussians_dense_sum_gpu(
            gaussians=gaussians,
            support_radii=support_radii,
            origin=grid.origin,
            voxel_dims=grid.voxel_dims,
            voxel_size=voxel_size,
            contribution_floor=cfg.support_contribution_floor,
            occupied_threshold=cfg.occupied_contribution_threshold,
            voxel_footprint_scale=cfg.voxel_footprint_scale,
            surface_splat_mode=cfg.surface_splat_mode,
            normal_support_voxels=cfg.normal_support_voxels,
            thin_min_anisotropy=cfg.thin_min_anisotropy,
            requested_device=cfg.compute_device,
            gaussian_chunk_size=cfg.gaussian_chunk_size,
            max_pairs_per_batch=cfg.splat_max_pairs_per_batch,
            max_accumulator_bytes=max(
                1,
                int(cfg.gpu_dense_accumulator_max_mb),
            )
            * 1024
            * 1024,
        )
    )
    voxelization_info["support_radius_cap"] = support_cap_info
    _finish_stage(stage_timings_sec, "voxel_splat", stage_start)

    stage_start = _start_stage("world_coordinate_conversion")
    occupied_world = voxel_idx_to_world_center(
        occupied_idx,
        origin=grid.origin,
        voxel_size=voxel_size,
    )
    _finish_stage(
        stage_timings_sec,
        "world_coordinate_conversion",
        stage_start,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = output_prefix.with_suffix(".npz")
    npz_payload = {
        "payload_format": np.asarray(
            ["on_the_fly_compact_occupancy_v2"]
        ),
        "voxel_size": np.asarray([voxel_size], dtype=np.float32),
        "origin": grid.origin.astype(np.float32),
        "scene_min": grid.scene_min.astype(np.float32),
        "scene_max": grid.scene_max.astype(np.float32),
        "voxel_dims": grid.voxel_dims.astype(np.int32),
        "occupied_voxel_idx": occupied_idx.astype(np.int32),
        "occupied_voxel_world": occupied_world.astype(np.float32),
        "occupied_contribution": occupied_contribution.astype(np.float32),
        "map_from_training_world": map_transform.astype(np.float32),
        "training_world_from_map": np.linalg.inv(map_transform).astype(np.float32),
    }
    if grid.block_size is not None and grid.block_dims is not None:
        npz_payload["block_size"] = np.asarray(
            [int(grid.block_size)],
            dtype=np.int32,
        )
        npz_payload["block_dims"] = grid.block_dims.astype(np.int32)

    stage_start = _start_stage("save_npz")
    np.savez_compressed(npz_path, **npz_payload)
    _finish_stage(stage_timings_sec, "save_npz", stage_start)

    support_radius_stats = _radius_summary(support_radii)
    support_radius_uncapped_stats = _radius_summary(support_radii_uncapped)
    json_path = output_prefix.with_suffix(".json")
    stats = {
        "config": {
            **asdict(cfg),
            "checkpoint": str(checkpoint_path),
            "reference_occupancy": (
                str(cfg.reference_occupancy)
                if cfg.reference_occupancy is not None
                else None
            ),
            "map_frame_transform": (
                str(cfg.map_frame_transform.expanduser().resolve())
                if cfg.map_frame_transform is not None
                else None
            ),
            "voxel_size": float(voxel_size),
            "voxel_size_json": voxel_size_json,
            "output_prefix": str(output_prefix),
        },
        "grid_source": grid_source,
        "map_frame": map_transform_meta,
        "grid_num_checkpoint_positions": grid_num_checkpoint_positions,
        "voxelization": "contribution_sum_gaussian_splatting",
        "support_contribution_floor": float(
            cfg.support_contribution_floor
        ),
        "support_radius_cap": support_cap_info,
        "occupied_contribution_threshold": float(
            cfg.occupied_contribution_threshold
        ),
        "surface_splat_mode": cfg.surface_splat_mode,
        "normal_support_voxels": float(cfg.normal_support_voxels),
        "thin_min_anisotropy": float(cfg.thin_min_anisotropy),
        "voxel_footprint_scale": float(cfg.voxel_footprint_scale),
        "voxelization_info": voxelization_info,
        "geometry_filter": geometry_filter_info,
        "num_gaussians_loaded": num_gaussians_loaded,
        "num_gaussians_without_contribution_support": (
            num_gaussians_without_support
        ),
        "num_gaussians_filtered": int(gaussians.positions.shape[0]),
        "gaussian_retention_ratio": float(
            gaussians.positions.shape[0] / num_gaussians_loaded
        ),
        "num_gaussians_removed": int(
            num_gaussians_loaded - gaussians.positions.shape[0]
        ),
        "num_occupied_voxels": int(occupied_idx.shape[0]),
        "num_voxels_total": int(
            np.prod(np.asarray(grid.voxel_dims, dtype=np.int64))
        ),
        "origin": grid.origin.tolist(),
        "scene_min": grid.scene_min.tolist(),
        "scene_max": grid.scene_max.tolist(),
        "voxel_dims": grid.voxel_dims.tolist(),
        "block_size": (
            int(grid.block_size) if grid.block_size is not None else None
        ),
        "block_dims": (
            grid.block_dims.astype(np.int32).tolist()
            if grid.block_dims is not None
            else None
        ),
        "support_radius_min": support_radius_stats["min"],
        "support_radius_max": support_radius_stats["max"],
        "support_radius_mean": support_radius_stats["mean"],
        "support_radius_uncapped_min": support_radius_uncapped_stats["min"],
        "support_radius_uncapped_max": support_radius_uncapped_stats["max"],
        "support_radius_uncapped_mean": support_radius_uncapped_stats["mean"],
        "output_npz": str(npz_path),
        "output_json": str(json_path),
    }
    stats.update(_timing_summary(t0, stage_timings_sec))

    stage_start = _start_stage("save_json")
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2, ensure_ascii=False)
    _finish_stage(stage_timings_sec, "save_json", stage_start)
    stats.update(_timing_summary(t0, stage_timings_sec))
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2, ensure_ascii=False)
    return stats


def main() -> None:
    cfg = parse_args(DEFAULT_CONFIG)
    stats = run_pipeline(cfg)
    values = stats["timing_sec"]
    print(
        "[SparseVoxelMap] done: "
        f"gaussians={stats['num_gaussians_filtered']}/{stats['num_gaussians_loaded']}, "
        f"occupied_voxels={stats['num_occupied_voxels']}, "
        f"calculation={values['calculation']:.3f}s, "
        f"total={values['wall_total']:.3f}s"
    )
    print(
        f"[SparseVoxelMap] output: {stats['output_npz']} "
        f"(stats: {stats['output_json']})"
    )


if __name__ == "__main__":
    main()
