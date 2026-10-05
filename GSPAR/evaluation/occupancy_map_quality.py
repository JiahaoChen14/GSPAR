#!/usr/bin/env python3
"""Evaluate a 3DGS voxel map against an nvblox reference."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GSPAR.mapping.map_frame_transform import validate_rigid_transform
from GSPAR.evaluation.resource_usage import ResourceMonitor


DEFAULT_SCENE_DIR = PROJECT_ROOT / "GSPAR/outputs/RaR_persp/O_windmill"
RESULT_NAME = "sparse_vs_nvblox_quality.json"
SURFACE_THRESHOLDS_M = (0.05, 0.10)
TRUNCATED_CHAMFER_THRESHOLD_M = 0.10
MAX_SURFACE_POINTS = 1_000_000
DISTANCE_BATCH_SIZE = 500_000


def resolve_npz(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    path = path if path.suffix == ".npz" else path.with_suffix(".npz")
    if not path.is_file():
        raise FileNotFoundError(f"Voxel map not found: {path}")
    return path


def pack_idx(indices: np.ndarray, dims: np.ndarray) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    return idx[:, 0] + dims[0] * (idx[:, 1] + dims[1] * idx[:, 2])


def unpack_idx(keys: np.ndarray, dims: np.ndarray) -> np.ndarray:
    keys = np.asarray(keys, dtype=np.int64).reshape(-1)
    x = keys % dims[0]
    yz = keys // dims[0]
    return np.stack((x, yz % dims[1], yz // dims[1]), axis=1).astype(np.int32)


def load_keys(data: np.lib.npyio.NpzFile, name: str, dims: np.ndarray) -> np.ndarray:
    if name not in data or np.asarray(data[name]).size == 0:
        return np.empty(0, dtype=np.int64)
    idx = np.asarray(data[name], dtype=np.int64).reshape(-1, 3)
    valid = np.logical_and(idx >= 0, idx < dims[None]).all(axis=1)
    return np.unique(pack_idx(idx[valid], dims))


def intersection_count(first: np.ndarray, second: np.ndarray) -> int:
    if not first.size or not second.size:
        return 0
    return int(np.intersect1d(first, second, assume_unique=True).size)


def contains(sorted_keys: np.ndarray, query: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.int64).reshape(-1)
    result = np.zeros(query.size, dtype=bool)
    if not query.size or not sorted_keys.size:
        return result
    positions = np.searchsorted(sorted_keys, query)
    valid = positions < sorted_keys.size
    result[valid] = sorted_keys[positions[valid]] == query[valid]
    return result


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


@dataclass(frozen=True)
class VoxelMap:
    path: Path
    voxel_size: float
    origin: np.ndarray
    dims: np.ndarray
    occupied: np.ndarray
    free: np.ndarray
    surface: np.ndarray | None
    implicit_free: bool
    map_from_training_world: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "VoxelMap":
        path = resolve_npz(path)
        with np.load(path) as data:
            required = ("voxel_size", "origin", "voxel_dims", "occupied_voxel_idx")
            missing = [name for name in required if name not in data]
            if missing:
                raise ValueError(f"{path} missing required keys: {missing}")
            dims = np.asarray(data["voxel_dims"], dtype=np.int64).reshape(3)
            occupied = load_keys(data, "occupied_voxel_idx", dims)
            free = load_keys(data, "free_voxel_idx", dims)
            surface = (
                load_keys(data, "surface_voxel_idx", dims)
                if "surface_voxel_idx" in data
                else None
            )
            unknown = load_keys(data, "unknown_voxel_idx", dims)
            implicit_free = (
                bool(np.asarray(data["implicit_free"]).reshape(-1)[0])
                if "implicit_free" in data
                else not free.size and not unknown.size
            )
            result = cls(
                path=path,
                voxel_size=float(np.asarray(data["voxel_size"]).reshape(-1)[0]),
                origin=np.asarray(data["origin"], dtype=np.float64).reshape(3),
                dims=dims,
                occupied=occupied,
                free=free,
                surface=surface,
                implicit_free=implicit_free,
                map_from_training_world=validate_rigid_transform(
                    np.asarray(
                        data["map_from_training_world"]
                        if "map_from_training_world" in data
                        else np.eye(4),
                        dtype=np.float32,
                    )
                ),
            )
        if overlap := intersection_count(result.occupied, result.free):
            raise ValueError(f"{path} labels {overlap} voxels as occupied and free")
        return result

    @property
    def total_count(self) -> int:
        return int(np.prod(self.dims))

    @property
    def labeled(self) -> np.ndarray:
        if self.implicit_free:
            raise ValueError("Implicit-free maps have no compact labeled array")
        return np.union1d(self.occupied, self.free)

    @property
    def labeled_count(self) -> int:
        return self.total_count if self.implicit_free else int(self.occupied.size + self.free.size)

    @property
    def free_count(self) -> int:
        return self.total_count - int(self.occupied.size) if self.implicit_free else int(self.free.size)

    def labels_at(self, keys: np.ndarray) -> np.ndarray:
        labels = np.full(np.asarray(keys).size, 1 if self.implicit_free else 0, dtype=np.uint8)
        if not self.implicit_free:
            labels[contains(self.free, keys)] = 1
        labels[contains(self.occupied, keys)] = 2
        return labels


def validate_same_grid(reference: VoxelMap, prediction: VoxelMap) -> None:
    if not np.isclose(reference.voxel_size, prediction.voxel_size, rtol=1e-4, atol=1e-8):
        raise ValueError("Voxel maps use different voxel sizes")
    if not np.allclose(reference.origin, prediction.origin):
        raise ValueError("Voxel maps use different origins")
    if not np.array_equal(reference.dims, prediction.dims):
        raise ValueError("Voxel maps use different dimensions")
    if not np.allclose(
        reference.map_from_training_world,
        prediction.map_from_training_world,
        atol=1e-6,
    ):
        raise ValueError("Voxel maps use different map-frame transforms")


def surface_boundary(occupied: np.ndarray, dims: np.ndarray) -> np.ndarray:
    if not occupied.size:
        return occupied
    idx = unpack_idx(occupied, dims).astype(np.int64)
    interior = np.ones(occupied.size, dtype=bool)
    for axis in range(3):
        for step in (-1, 1):
            neighbor = idx.copy()
            neighbor[:, axis] += step
            valid = np.logical_and(neighbor >= 0, neighbor < dims[None]).all(axis=1)
            present = np.zeros(occupied.size, dtype=bool)
            present[valid] = contains(occupied, pack_idx(neighbor[valid], dims))
            interior &= present
    return occupied[~interior]


def surface_voxel_keys(voxel_map: VoxelMap) -> tuple[np.ndarray, str]:
    """Return the map's geometry surface, with a legacy-map fallback.

    New maps persist mesh-derived ``surface_voxel_idx`` separately from their
    planning occupancy. Older maps fall back to their free-facing boundary so
    the negative TSDF band's hidden/back boundary is never scored as geometry.
    """

    if voxel_map.surface is not None:
        return (
            voxel_map.surface,
            "explicit mesh-derived surface_voxel_idx on the common grid",
        )
    if voxel_map.implicit_free or not voxel_map.free.size:
        return (
            surface_boundary(voxel_map.occupied, voxel_map.dims),
            "occupied 6-neighbor boundary against implicit free space",
        )
    occupied_idx = unpack_idx(voxel_map.occupied, voxel_map.dims).astype(np.int64)
    exposed = np.zeros(voxel_map.occupied.size, dtype=bool)
    for axis in range(3):
        for step in (-1, 1):
            neighbor = occupied_idx.copy()
            neighbor[:, axis] += step
            valid = np.logical_and(
                neighbor >= 0,
                neighbor < voxel_map.dims[None],
            ).all(axis=1)
            if np.any(valid):
                exposed[valid] |= contains(
                    voxel_map.free,
                    pack_idx(neighbor[valid], voxel_map.dims),
                )
    return (
        voxel_map.occupied[exposed],
        "occupied voxels face-adjacent to explicit free voxels; unknown boundary excluded",
    )


def world_centers(keys: np.ndarray, voxel_map: VoxelMap) -> np.ndarray:
    idx = unpack_idx(keys, voxel_map.dims).astype(np.float64)
    return (voxel_map.origin[None] + (idx + 0.5) * voxel_map.voxel_size).astype(np.float32)


def nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if not source.shape[0]:
        return np.empty(0, dtype=np.float32)
    if not target.shape[0]:
        return np.full(source.shape[0], np.inf, dtype=np.float32)
    tree = cKDTree(target.astype(np.float64, copy=False))
    result = np.empty(source.shape[0], dtype=np.float32)
    for start in range(0, source.shape[0], DISTANCE_BATCH_SIZE):
        stop = min(start + DISTANCE_BATCH_SIZE, source.shape[0])
        result[start:stop] = tree.query(source[start:stop], workers=-1)[0]
    return result


def distance_summary(distances: np.ndarray) -> dict[str, float | None]:
    values = distances[np.isfinite(distances)]
    return {
        "mean_m": None if not values.size else float(values.mean()),
        "p95_m": None if not values.size else float(np.quantile(values, 0.95)),
    }


def truncated_distance_mean(
    distances: np.ndarray,
    threshold_m: float = TRUNCATED_CHAMFER_THRESHOLD_M,
) -> float | None:
    """Return mean nearest-surface distance after clipping at ``threshold_m``."""

    values = distances[np.isfinite(distances)]
    if not values.size:
        return None
    return float(np.minimum(values, float(threshold_m)).mean())


def timing(path: Path) -> dict[str, float | str | None]:
    fields = (
        "loading",
        "calculation",
        "slam",
        "tsdf",
        "esdf",
        "tsdf_esdf",
        "saving",
        "categorized_total",
        "wall_total",
    )
    result: dict[str, float | str | None] = {"source": str(path), **dict.fromkeys(fields)}
    if not path.is_file():
        return result
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data.get("timing_sec", {})
    for field in fields:
        value = values.get(field)
        if isinstance(value, (int, float)):
            result[field] = float(value)
    tsdf = result["tsdf"]
    esdf = result["esdf"]
    if isinstance(tsdf, float) and isinstance(esdf, float):
        # Derive the evaluation value from the two allowed stages instead of
        # trusting a broader aggregate from the input file.
        result["tsdf_esdf"] = tsdf + esdf
    return result


def map_representation_stats(prediction: VoxelMap) -> dict[str, int | float | str | None]:
    """Read build counts and report persisted planning and surface sizes."""

    stats_path = prediction.path.with_suffix(".json")
    metadata: dict[str, Any] = {}
    if stats_path.is_file():
        loaded = json.loads(stats_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            metadata = loaded
    original = metadata.get("num_gaussians_loaded")
    filtered = metadata.get("num_gaussians_filtered")
    original_count = int(original) if isinstance(original, (int, float)) else None
    filtered_count = int(filtered) if isinstance(filtered, (int, float)) else None
    retention = (
        None
        if original_count is None or filtered_count is None or original_count <= 0
        else float(filtered_count / original_count)
    )
    return {
        "source": str(stats_path),
        "original_gaussian_count": original_count,
        "filtered_gaussian_count": filtered_count,
        "gaussian_retention_ratio": retention,
        "final_occupied_voxel_count": int(prediction.occupied.size),
        "geometry_surface_voxel_count": (
            None if prediction.surface is None else int(prediction.surface.size)
        ),
    }


def _rendered_depth_timing(manifest_path: Path | None) -> dict[str, float | str | None] | None:
    if manifest_path is None:
        return None
    result: dict[str, float | str | None] = {"source": str(manifest_path)}
    if not manifest_path.is_file():
        return result
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for group in (payload.get("timing_sec", {}), payload.get("render", {}).get("timing_sec", {})):
        if isinstance(group, dict):
            for name, value in group.items():
                if isinstance(value, (int, float)):
                    result[name] = float(value)
    return result


def efficiency(
    scene_dir: Path,
    prediction_npz: Path,
    *,
    sdf_dir: Path | None = None,
    pipeline_manifest: Path | None = None,
) -> dict[str, object]:
    sdf_dir = sdf_dir or scene_dir / "sdf"
    nvblox = timing(sdf_dir / "timing_stats.json")
    gaussian = timing(prediction_npz.with_suffix(".json"))
    # Compare only nvblox's TSDF integration plus ESDF update. In particular,
    # do not include external SLAM, input loading, initialization, or saving.
    nvblox_calc = nvblox["tsdf_esdf"]
    gaussian_calc = gaussian["calculation"]
    difference = ratio = None
    if isinstance(nvblox_calc, float) and isinstance(gaussian_calc, float):
        difference = gaussian_calc - nvblox_calc
        ratio = None if gaussian_calc <= 0.0 else nvblox_calc / gaussian_calc

    def size(path: Path) -> int | None:
        return path.stat().st_size if path.is_file() else None

    esdf_path = sdf_dir / "esdf_observed.ply"
    esdf_bytes = size(esdf_path)
    voxel_bytes = size(prediction_npz)
    stored_difference = (
        None
        if esdf_bytes is None or voxel_bytes is None
        else int(voxel_bytes - esdf_bytes)
    )
    stored_ratio = (
        None
        if esdf_bytes is None or voxel_bytes is None or voxel_bytes <= 0
        else float(esdf_bytes / voxel_bytes)
    )
    planning_total = (
        None
        if esdf_bytes is None or voxel_bytes is None
        else int(esdf_bytes + voxel_bytes)
    )

    return {
        "timing_sec": {
            "nvblox": nvblox,
            "gaussian_to_voxel": gaussian,
            "rendered_depth_pipeline": _rendered_depth_timing(pipeline_manifest),
            "nvblox_counted_field": "tsdf_esdf",
            "comparison": {
                "gaussian_minus_nvblox_tsdf_esdf": difference,
                "nvblox_tsdf_esdf_over_gaussian": ratio,
            },
        },
        "stored_map_size_bytes": {
            "nvblox_esdf_ply": esdf_bytes,
            "gaussian_voxel_npz": voxel_bytes,
            "comparison": {
                "gaussian_voxel_minus_nvblox_esdf": stored_difference,
                "nvblox_esdf_over_gaussian_voxel": stored_ratio,
            },
        },
        "planning_map_payload_bytes": {
            "esdf_ply": esdf_bytes,
            "voxel_npz": voxel_bytes,
            "combined": planning_total,
            "files": {
                "esdf_ply": str(esdf_path),
                "voxel_npz": str(prediction_npz),
            },
        },
    }


def threshold_metrics(pred_to_ref: np.ndarray, ref_to_pred: np.ndarray) -> list[dict]:
    rows = []
    for threshold in SURFACE_THRESHOLDS_M:
        prediction_matched = int(np.count_nonzero(pred_to_ref <= threshold))
        reference_matched = int(np.count_nonzero(ref_to_pred <= threshold))
        precision = safe_div(prediction_matched, int(pred_to_ref.size))
        recall = safe_div(reference_matched, int(ref_to_pred.size))
        rows.append({
            "threshold_m": float(threshold),
            "threshold_cm": float(threshold * 100.0),
            "prediction_matched_count": prediction_matched,
            "reference_matched_count": reference_matched,
            "precision": precision,
            "recall": recall,
            "fscore": safe_div(2 * precision * recall, precision + recall),
        })
    return rows


def zero_tolerance_surface_metrics(
    reference_surface: np.ndarray,
    prediction_surface: np.ndarray,
) -> dict[str, int | float | str]:
    """Compute exact overlap only between the unified surface voxel sets."""

    intersection = intersection_count(reference_surface, prediction_surface)
    union = int(reference_surface.size + prediction_surface.size - intersection)
    precision = safe_div(intersection, int(prediction_surface.size))
    recall = safe_div(intersection, int(reference_surface.size))
    return {
        "method": "exact surface voxel overlap on the common grid",
        "intersection_count": intersection,
        "union_count": union,
        "surface_iou": safe_div(intersection, union),
        "precision": precision,
        "recall": recall,
        "fscore": safe_div(2.0 * precision * recall, precision + recall),
    }


def evaluate(
    scene_dir: Path,
    prediction_path: Path,
    reference_path: Path,
    *,
    sdf_dir: Path | None = None,
    pipeline_manifest: Path | None = None,
) -> dict[str, object]:
    with ResourceMonitor() as resource_monitor:
        started_at = time.perf_counter()
        prediction = VoxelMap.load(prediction_path)
        reference = VoxelMap.load(reference_path)
        validate_same_grid(reference, prediction)
        pred_keys, prediction_surface_rule = surface_voxel_keys(prediction)
        ref_keys, reference_surface_rule = surface_voxel_keys(reference)
        pred_points = world_centers(pred_keys, prediction)
        ref_points = world_centers(ref_keys, reference)
        if pred_points.shape[0] > MAX_SURFACE_POINTS:
            pred_points = pred_points[
                np.linspace(
                    0,
                    pred_points.shape[0] - 1,
                    MAX_SURFACE_POINTS,
                    dtype=int,
                )
            ]
        if ref_points.shape[0] > MAX_SURFACE_POINTS:
            ref_points = ref_points[
                np.linspace(
                    0,
                    ref_points.shape[0] - 1,
                    MAX_SURFACE_POINTS,
                    dtype=int,
                )
            ]
        pred_to_ref = nearest_distances(pred_points, ref_points)
        ref_to_pred = nearest_distances(ref_points, pred_points)
        finite_pred = pred_to_ref[np.isfinite(pred_to_ref)]
        finite_ref = ref_to_pred[np.isfinite(ref_to_pred)]
        chamfer = None
        if finite_pred.size and finite_ref.size:
            chamfer = 0.5 * (
                float(finite_pred.mean())
                + float(finite_ref.mean())
            )
        truncated_pred_to_ref = truncated_distance_mean(pred_to_ref)
        truncated_ref_to_pred = truncated_distance_mean(ref_to_pred)
        truncated_chamfer = None
        if (
            truncated_pred_to_ref is not None
            and truncated_ref_to_pred is not None
        ):
            truncated_chamfer = 0.5 * (
                truncated_pred_to_ref + truncated_ref_to_pred
            )
        result = {
            "protocol": {
                "comparison": "geometry_surface_only",
                "surface_grid": (
                    "common origin, voxel size, dimensions, and map frame"
                ),
                "surface_thresholds_m": list(SURFACE_THRESHOLDS_M),
                "truncated_chamfer_threshold_m": TRUNCATED_CHAMFER_THRESHOLD_M,
                "truncated_chamfer_definition": (
                    "0.5 * (mean(min(prediction_to_reference, 0.10 m)) + "
                    "mean(min(reference_to_prediction, 0.10 m)))"
                ),
            },
            "config": {
                "scene_dir": str(scene_dir),
                "prediction_map": str(prediction.path),
                "reference_map": str(reference.path),
            },
            "map_representation": map_representation_stats(prediction),
            "surface_metrics": {
                "surface_definition": {
                    "reference": reference_surface_rule,
                    "prediction": prediction_surface_rule,
                },
                "counts": {
                    "reference_surface_voxels": int(ref_keys.size),
                    "prediction_surface_voxels": int(pred_keys.size),
                },
                "zero_tolerance": zero_tolerance_surface_metrics(
                    ref_keys,
                    pred_keys,
                ),
                "prediction_to_reference": distance_summary(pred_to_ref),
                "reference_to_prediction": distance_summary(ref_to_pred),
                "chamfer_l1_m": chamfer,
                "truncated_chamfer_l1_m": truncated_chamfer,
                "truncated_chamfer_threshold_m": TRUNCATED_CHAMFER_THRESHOLD_M,
                "truncated_prediction_to_reference_mean_m": (
                    truncated_pred_to_ref
                ),
                "truncated_reference_to_prediction_mean_m": (
                    truncated_ref_to_pred
                ),
                "by_threshold": threshold_metrics(pred_to_ref, ref_to_pred),
            },
            "efficiency": efficiency(
                scene_dir,
                prediction.path,
                sdf_dir=sdf_dir,
                pipeline_manifest=pipeline_manifest,
            ),
        }
        evaluation_sec = float(time.perf_counter() - started_at)
    result["evaluation_time_sec"] = evaluation_sec
    result["resource_usage"] = resource_monitor.to_dict()
    result["resource_usage_policy"] = {
        "cpu": "sampled process RSS; absolute and incremental peaks",
        "cuda": (
            "PyTorch allocated/reserved absolute and incremental peaks; "
            "driver and non-PyTorch allocations excluded"
        ),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--prediction-map", type=Path, default=None)
    parser.add_argument("--reference-map", type=Path, default=None)
    parser.add_argument(
        "--sdf-dir",
        type=Path,
        default=None,
        help="TSDF/ESDF directory associated with --prediction-map.",
    )
    parser.add_argument(
        "--pipeline-manifest",
        type=Path,
        default=None,
        help="Rendered-depth pipeline manifest whose stage timings are recorded.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    args.scene_dir = args.scene_dir.expanduser().resolve()
    args.prediction_map = (
        args.prediction_map or args.scene_dir / "voxel_map.npz"
    ).expanduser().resolve()
    args.reference_map = (
        args.reference_map or args.scene_dir / "voxel_map_tsdf.npz"
    ).expanduser().resolve()
    args.output = (
        args.output or args.scene_dir / "eval" / RESULT_NAME
    ).expanduser().resolve()
    if args.sdf_dir is not None:
        args.sdf_dir = args.sdf_dir.expanduser().resolve()
    if args.pipeline_manifest is not None:
        args.pipeline_manifest = args.pipeline_manifest.expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()
    result = evaluate(
        args.scene_dir,
        args.prediction_map,
        args.reference_map,
        sdf_dir=args.sdf_dir,
        pipeline_manifest=args.pipeline_manifest,
    )
    evaluation_sec = float(result["evaluation_time_sec"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    threshold_rows = result["surface_metrics"]["by_threshold"]
    exact = result["surface_metrics"]["zero_tolerance"]
    truncated_chamfer = result["surface_metrics"]["truncated_chamfer_l1_m"]
    timing_sec = result["efficiency"]["timing_sec"]
    nvblox_sec = timing_sec["nvblox"]["tsdf_esdf"]
    gaussian_sec = timing_sec["gaussian_to_voxel"]["calculation"]
    nvblox_text = "n/a" if nvblox_sec is None else f"{nvblox_sec:.3f}s"
    gaussian_text = "n/a" if gaussian_sec is None else f"{gaussian_sec:.3f}s"
    resources = result["resource_usage"]
    cpu_peak_mib = float(resources["cpu_rss_peak_sampled_bytes"]) / (1024.0**2)
    cuda_peak = resources["cuda_torch_allocated_peak_bytes"]
    cuda_text = "n/a" if cuda_peak is None else f"{float(cuda_peak) / (1024.0**2):.1f}MiB"
    print(
        "[OccupancyEval] done: "
        f"IoU/P/R/F@0={exact['surface_iou']:.4f}/{exact['precision']:.4f}/"
        f"{exact['recall']:.4f}/{exact['fscore']:.4f}, "
        f"F@5cm={threshold_rows[0]['fscore']:.4f}, "
        f"F@10cm={threshold_rows[1]['fscore']:.4f}, "
        f"truncated_chamfer@10cm="
        f"{'n/a' if truncated_chamfer is None else f'{truncated_chamfer:.4f}m'}, "
        f"evaluation={evaluation_sec:.3f}s, "
        f"3dgs_to_voxel={gaussian_text}, nvblox_tsdf_esdf={nvblox_text}, "
        f"cpu_rss_peak={cpu_peak_mib:.1f}MiB, cuda_allocated_peak={cuda_text}"
    )
    print(f"[OccupancyEval] output: {args.output}")


if __name__ == "__main__":
    main()
