#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ESDF global path planning with A* in nerficg world coordinates."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import heapq
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from GSPAR.path.global_path.astar_world_path import _grid_distance_heuristic, _neighbor_offsets


PLANNER_MAP_TYPE = "ESDF"
DEFAULT_UNKNOWN_DISTANCE_M = 1.0
UNKNOWN_POLICY_CLEARANCE = "clearance"
UNKNOWN_POLICY_OCCUPIED = "occupied"
UNKNOWN_POLICY_FREE = "free"
UNKNOWN_POLICIES = {UNKNOWN_POLICY_CLEARANCE, UNKNOWN_POLICY_OCCUPIED, UNKNOWN_POLICY_FREE}


def resolve_esdf_path(path: str | Path) -> Path:
    """Resolve an ESDF PLY from a file, an ``sdf`` directory, or a scene directory."""

    requested = Path(path).expanduser().resolve()
    candidates = [requested]
    if requested.is_dir():
        candidates = [
            requested / "esdf_observed.ply",
            requested / "sdf" / "esdf_observed.ply",
        ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"ESDF file not found from {requested}; expected esdf_observed.ply "
        "or sdf/esdf_observed.ply"
    )


@dataclass
class EsdfGridMetadata:
    origin: np.ndarray
    voxel_size: float
    voxel_dims: np.ndarray
    source_to_world_affine: np.ndarray | None
    source_voxel_size: float
    distance_scale: float
    max_esdf_distance_m: float
    grid_path: Path | None = None
    grid_json_path: Path | None = None


@dataclass
class EsdfGridGraph:
    """Sparse ESDF graph aligned to the target nerficg voxel grid."""

    origin: np.ndarray
    voxel_size: float
    voxel_dims: np.ndarray
    esdf_idx: np.ndarray
    esdf_distance_m: np.ndarray
    occupied_idx: np.ndarray | None = None
    free_idx: np.ndarray | None = None
    # ESDF PLY exports only observed ESDF voxels. Missing grid cells are
    # unknown, not necessarily free. unknown_policy controls whether they are
    # blocked, treated as free, or assigned this proxy clearance.
    unknown_clearance_m: float = DEFAULT_UNKNOWN_DISTANCE_M
    unknown_policy: str = UNKNOWN_POLICY_CLEARANCE

    def __post_init__(self) -> None:
        self.origin = np.asarray(self.origin, dtype=np.float32).reshape(3)
        self.voxel_size = float(self.voxel_size)
        self.voxel_dims = np.asarray(self.voxel_dims, dtype=np.int32).reshape(3)
        self.esdf_idx = np.asarray(self.esdf_idx, dtype=np.int32).reshape(-1, 3)
        self.esdf_distance_m = np.asarray(self.esdf_distance_m, dtype=np.float32).reshape(-1)
        if self.esdf_idx.shape[0] != self.esdf_distance_m.shape[0]:
            raise ValueError("esdf_idx and esdf_distance_m must have the same length")
        occupied = np.zeros((0, 3), dtype=np.int32) if self.occupied_idx is None else self.occupied_idx
        self.occupied_idx = np.asarray(occupied, dtype=np.int32).reshape(-1, 3)
        free = np.zeros((0, 3), dtype=np.int32) if self.free_idx is None else self.free_idx
        self.free_idx = np.asarray(free, dtype=np.int32).reshape(-1, 3)
        self.unknown_clearance_m = float(self.unknown_clearance_m)
        self.unknown_policy = str(self.unknown_policy).strip().lower()
        if self.unknown_policy not in UNKNOWN_POLICIES:
            raise ValueError(f"unknown_policy must be one of {sorted(UNKNOWN_POLICIES)}, got {self.unknown_policy!r}")
        self._dims64 = self.voxel_dims.astype(np.int64)
        esdf_keys = self._pack(self.esdf_idx).astype(np.int64, copy=False)
        esdf_order = np.argsort(esdf_keys, kind="mergesort")
        self._esdf_keys = esdf_keys[esdf_order]
        self.esdf_idx = self.esdf_idx[esdf_order]
        self.esdf_distance_m = self.esdf_distance_m[esdf_order]
        occupied_keys = self._pack(self.occupied_idx).astype(np.int64, copy=False)
        self._occupied_keys = np.unique(occupied_keys) if occupied_keys.shape[0] else np.zeros((0,), dtype=np.int64)
        free_keys = self._pack(self.free_idx).astype(np.int64, copy=False)
        self._free_keys = np.unique(free_keys) if free_keys.shape[0] else np.zeros((0,), dtype=np.int64)
        build_unknown_clearance_tree = (
            self.unknown_policy == UNKNOWN_POLICY_CLEARANCE and self.occupied_idx.shape[0] > 0
        )
        self._should_build_occupied_tree = bool(build_unknown_clearance_tree)
        self._occupied_tree: cKDTree | None = None
        self._safety_cache: dict[tuple[int, float], bool] = {}
        self._guard_clearance_cache: dict[int, float] = {}

    def _pack(self, idx: np.ndarray) -> np.ndarray:
        idx64 = np.asarray(idx, dtype=np.int64).reshape(-1, 3)
        return idx64[:, 0] + self._dims64[0] * (idx64[:, 1] + self._dims64[1] * idx64[:, 2])

    def _ensure_occupied_tree(self) -> cKDTree | None:
        if not self._should_build_occupied_tree:
            return None
        if self._occupied_tree is None:
            self._occupied_tree = cKDTree(self.occupied_idx.astype(np.float32))
        return self._occupied_tree

    @staticmethod
    def _contains_sorted(sorted_keys: np.ndarray, keys: np.ndarray) -> np.ndarray:
        keys_arr = np.asarray(keys, dtype=np.int64).reshape(-1)
        if keys_arr.shape[0] == 0 or sorted_keys.shape[0] == 0:
            return np.zeros((keys_arr.shape[0],), dtype=bool)
        pos = np.searchsorted(sorted_keys, keys_arr)
        mask = pos < sorted_keys.shape[0]
        if mask.any():
            mask[mask] = sorted_keys[pos[mask]] == keys_arr[mask]
        return mask

    def _esdf_distances_for_keys(self, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keys_arr = np.asarray(keys, dtype=np.int64).reshape(-1)
        distances = np.full((keys_arr.shape[0],), np.nan, dtype=np.float32)
        if keys_arr.shape[0] == 0 or self._esdf_keys.shape[0] == 0:
            return distances, np.zeros((keys_arr.shape[0],), dtype=bool)
        pos = np.searchsorted(self._esdf_keys, keys_arr)
        has_esdf = pos < self._esdf_keys.shape[0]
        if has_esdf.any():
            has_esdf[has_esdf] = self._esdf_keys[pos[has_esdf]] == keys_arr[has_esdf]
            distances[has_esdf] = self.esdf_distance_m[pos[has_esdf]]
        return distances, has_esdf

    def esdf_distance_for_idx_batch(self, idx: np.ndarray) -> np.ndarray:
        idx_arr = np.asarray(idx, dtype=np.int32).reshape(-1, 3)
        distances = np.full((idx_arr.shape[0],), -np.inf, dtype=np.float32)
        if idx_arr.shape[0] == 0:
            return distances
        valid = np.logical_and(idx_arr >= 0, idx_arr < self.voxel_dims[None, :]).all(axis=1)
        if not valid.any():
            return distances

        valid_rows = np.flatnonzero(valid)
        keys = self._pack(idx_arr[valid])
        occupied = self._contains_sorted(self._occupied_keys, keys)
        free_rows = valid_rows[~occupied]
        if free_rows.shape[0] == 0:
            return distances

        free_keys = keys[~occupied]
        esdf_distances, has_esdf = self._esdf_distances_for_keys(free_keys)
        distances[free_rows[has_esdf]] = esdf_distances[has_esdf]
        missing_rows = free_rows[~has_esdf]
        if missing_rows.shape[0]:
            if self.unknown_policy == UNKNOWN_POLICY_CLEARANCE:
                distances[missing_rows] = float(self.unknown_clearance_m)
            elif self.unknown_policy == UNKNOWN_POLICY_FREE:
                distances[missing_rows] = float("inf")
        return distances

    def distance_and_safety_for_idx_batch(
        self,
        idx: np.ndarray,
        safety_clearance_m: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        idx_arr = np.asarray(idx, dtype=np.int32).reshape(-1, 3)
        distances = np.full((idx_arr.shape[0],), -np.inf, dtype=np.float32)
        safe = np.zeros((idx_arr.shape[0],), dtype=bool)
        if idx_arr.shape[0] == 0:
            return distances, safe

        valid = np.logical_and(idx_arr >= 0, idx_arr < self.voxel_dims[None, :]).all(axis=1)
        if not valid.any():
            return distances, safe

        valid_rows = np.flatnonzero(valid)
        keys = self._pack(idx_arr[valid])
        occupied = self._contains_sorted(self._occupied_keys, keys)
        candidate_rows = valid_rows[~occupied]
        if candidate_rows.shape[0] == 0:
            return distances, safe

        candidate_keys = keys[~occupied]
        esdf_distances, has_esdf = self._esdf_distances_for_keys(candidate_keys)
        if has_esdf.any():
            rows = candidate_rows[has_esdf]
            distances[rows] = esdf_distances[has_esdf]
            safe[rows] = esdf_distances[has_esdf].astype(np.float64) >= float(safety_clearance_m)

        missing_rows = candidate_rows[~has_esdf]
        if missing_rows.shape[0]:
            if self.unknown_policy == UNKNOWN_POLICY_FREE:
                distances[missing_rows] = float("inf")
                safe[missing_rows] = True
            elif self.unknown_policy == UNKNOWN_POLICY_CLEARANCE:
                distances[missing_rows] = float(self.unknown_clearance_m)
                missing_safe = np.full(
                    (missing_rows.shape[0],),
                    float(self.unknown_clearance_m) >= float(safety_clearance_m),
                    dtype=bool,
                )
                occupied_tree = self._ensure_occupied_tree()
                if occupied_tree is not None and safety_clearance_m > 0.0 and missing_safe.any():
                    safety_clearance_voxels = float(safety_clearance_m) / max(self.voxel_size, 1.0e-12)
                    guard_distances = occupied_tree.query(idx_arr[missing_rows].astype(np.float32), k=1)[0]
                    missing_safe &= guard_distances >= safety_clearance_voxels
                safe[missing_rows] = missing_safe
        return distances, safe

    def step_costs_for_idx_batch(
        self,
        idx: np.ndarray,
        move_costs: np.ndarray,
        safety_clearance_m: float,
        clearance_cost_weight: float,
    ) -> np.ndarray:
        distances, safe = self.distance_and_safety_for_idx_batch(
            idx,
            safety_clearance_m=safety_clearance_m,
        )
        move = np.asarray(move_costs, dtype=np.float64).reshape(-1)
        costs = np.full((distances.shape[0],), np.inf, dtype=np.float64)
        if not safe.any():
            return costs
        if clearance_cost_weight <= 0.0:
            costs[safe] = move[safe]
            return costs

        free_inf = np.logical_and(safe, np.isposinf(distances))
        if free_inf.any():
            costs[free_inf] = move[free_inf]
        finite_safe = np.logical_and(safe, np.isfinite(distances))
        if finite_safe.any():
            clearance_vox = np.maximum(
                (distances[finite_safe].astype(np.float64) - float(safety_clearance_m)) / self.voxel_size,
                1.0e-3,
            )
            costs[finite_safe] = move[finite_safe] * (1.0 + float(clearance_cost_weight) / clearance_vox)
        return costs

    def is_valid_idx(self, idx: np.ndarray) -> bool:
        idx = np.asarray(idx, dtype=np.int32).reshape(3)
        return bool(np.logical_and(idx >= 0, idx < self.voxel_dims).all())

    def world_to_voxel_idx(self, point_world: np.ndarray) -> np.ndarray:
        point_world = np.asarray(point_world, dtype=np.float32).reshape(3)
        idx = np.floor((point_world - self.origin) / self.voxel_size).astype(np.int32)
        return np.clip(idx, 0, self.voxel_dims - 1)

    def _world_to_voxel_idx_unclipped(self, point_world: np.ndarray) -> np.ndarray:
        point_world = np.asarray(point_world, dtype=np.float32).reshape(3)
        return np.floor((point_world - self.origin) / self.voxel_size).astype(np.int32)

    def voxel_idx_to_world_center(self, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.float32).reshape(-1, 3)
        return self.origin[None, :] + (idx + 0.5) * self.voxel_size

    def esdf_distance_at_idx(self, idx: np.ndarray) -> float:
        """Return signed ESDF distance in meters, or the configured unknown value."""

        distances = self.esdf_distance_for_idx_batch(np.asarray(idx, dtype=np.int32).reshape(1, 3))
        return float(distances[0])

    def occupancy_guard_clearance_voxels(self, idx: np.ndarray) -> float:
        if not self.is_valid_idx(idx):
            return 0.0
        occupied_tree = self._ensure_occupied_tree()
        if occupied_tree is None:
            return float("inf")
        idx = np.asarray(idx, dtype=np.int32).reshape(3)
        key = int(self._pack(idx.reshape(1, 3))[0])
        cached = self._guard_clearance_cache.get(key)
        if cached is not None:
            return cached
        distance = float(occupied_tree.query(idx.astype(np.float32), k=1)[0])
        self._guard_clearance_cache[key] = distance
        return distance

    def is_safe_idx(self, idx: np.ndarray, safety_clearance_m: float = 0.0) -> bool:
        if not self.is_valid_idx(idx):
            return False
        safety_clearance_m = float(safety_clearance_m)
        key = int(self._pack(np.asarray(idx, dtype=np.int32).reshape(1, 3))[0])
        cache_key = (key, safety_clearance_m)
        cached = self._safety_cache.get(cache_key)
        if cached is not None:
            return cached
        _, safe_arr = self.distance_and_safety_for_idx_batch(
            np.asarray(idx, dtype=np.int32).reshape(1, 3),
            safety_clearance_m=safety_clearance_m,
        )
        safe = bool(safe_arr[0])
        self._safety_cache[cache_key] = safe
        return safe

    def snap_to_safe_idx(
        self,
        point_world: np.ndarray,
        max_radius: int = 8,
        safety_clearance_m: float = 0.0,
    ) -> np.ndarray:
        best_idx = self.find_safe_idx_near_world(
            point_world,
            max_radius=max_radius,
            safety_clearance_m=safety_clearance_m,
        )
        if best_idx is not None:
            return best_idx
        target = self._world_to_voxel_idx_unclipped(point_world)
        raise ValueError(
            f"no ESDF-safe voxel within radius {int(max_radius)} of target_idx {target.tolist()} "
            f"and clearance {float(safety_clearance_m):.3f}m"
        )

    def find_safe_idx_near_world(
        self,
        point_world: np.ndarray,
        max_radius: int = 8,
        safety_clearance_m: float = 0.0,
    ) -> np.ndarray | None:
        target = self._world_to_voxel_idx_unclipped(point_world)
        if self.is_valid_idx(target) and self.is_safe_idx(target, safety_clearance_m=safety_clearance_m):
            return target

        max_radius_i = max(0, int(max_radius))
        target_f = target.astype(np.float32)
        best_idx: np.ndarray | None = None
        best_dist2 = float("inf")
        for radius in range(1, max_radius_i + 1):
            lo = np.maximum(target - radius, 0)
            hi = np.minimum(target + radius, self.voxel_dims - 1)
            if np.any(lo > hi):
                continue
            xs = np.arange(int(lo[0]), int(hi[0]) + 1, dtype=np.int32)
            ys = np.arange(int(lo[1]), int(hi[1]) + 1, dtype=np.int32)
            zs = np.arange(int(lo[2]), int(hi[2]) + 1, dtype=np.int32)
            candidates = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
            _, safe = self.distance_and_safety_for_idx_batch(
                candidates,
                safety_clearance_m=safety_clearance_m,
            )
            if not safe.any():
                continue
            safe_candidates = candidates[safe]
            dist2 = np.sum((safe_candidates.astype(np.float32) - target_f[None, :]) ** 2, axis=1)
            best_local = int(np.argmin(dist2))
            local_dist2 = float(dist2[best_local])
            if local_dist2 < best_dist2:
                best_idx = safe_candidates[best_local].astype(np.int32, copy=True)
                best_dist2 = local_dist2
            if best_idx is not None:
                return best_idx
        return None


def _parse_ply_header(path: Path) -> tuple[str, int, int, int, dict[str, int], list[tuple[str, str]]]:
    vertex_count: int | None = None
    properties: list[tuple[str, str]] = []
    in_vertex = False
    header_lines = 0
    header_bytes = 0
    ply_format: str | None = None
    with path.open("rb") as f:
        first_bytes = f.readline()
        header_bytes += len(first_bytes)
        first = first_bytes.decode("ascii").strip()
        header_lines += 1
        if first != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        for line_bytes in f:
            header_bytes += len(line_bytes)
            header_lines += 1
            line = line_bytes.decode("ascii")
            s = line.strip()
            if s == "end_header":
                break
            if s.startswith("format "):
                parts = s.split()
                if len(parts) < 3:
                    raise ValueError(f"Malformed PLY format line: {s}")
                ply_format = parts[1]
                if ply_format not in {"ascii", "binary_little_endian"}:
                    raise ValueError(f"Unsupported PLY format: {s}")
            if s.startswith("element "):
                parts = s.split()
                in_vertex = len(parts) >= 3 and parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
                    properties = []
            elif in_vertex and s.startswith("property "):
                parts = s.split()
                if len(parts) != 3:
                    raise ValueError(f"Unsupported PLY property line: {s}")
                properties.append((parts[2], parts[1]))
        else:
            raise ValueError(f"PLY header missing end_header: {path}")
    if vertex_count is None:
        raise ValueError(f"PLY vertex element not found: {path}")
    if ply_format is None:
        raise ValueError(f"PLY format line not found: {path}")
    return ply_format, vertex_count, header_lines, header_bytes, {name: idx for idx, (name, _) in enumerate(properties)}, properties


def _resolve_sdf_value_property(prop: dict[str, int], path: Path) -> str:
    for name in ("intensity", "distance", "sdf", "esdf"):
        if name in prop:
            return name
    raise ValueError(
        f"PLY missing ESDF value property in {path}; expected one of "
        "['intensity', 'distance', 'sdf', 'esdf']."
    )


def load_esdf_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = resolve_esdf_path(path)
    ply_format, vertex_count, header_lines, header_bytes, prop, properties = _parse_ply_header(path)
    missing = [name for name in ("x", "y", "z") if name not in prop]
    if missing:
        raise ValueError(f"PLY missing required properties: {missing}")
    sdf_prop = _resolve_sdf_value_property(prop, path)
    if ply_format == "ascii":
        data = np.loadtxt(
            path,
            skiprows=header_lines,
            usecols=[prop["x"], prop["y"], prop["z"], prop[sdf_prop]],
            dtype=np.float32,
        )
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data[:, :3].astype(np.float32), data[:, 3].astype(np.float32)

    ply_to_numpy = {
        "char": "i1",
        "uchar": "u1",
        "int8": "i1",
        "uint8": "u1",
        "short": "<i2",
        "ushort": "<u2",
        "int16": "<i2",
        "uint16": "<u2",
        "int": "<i4",
        "uint": "<u4",
        "int32": "<i4",
        "uint32": "<u4",
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "float64": "<f8",
    }
    try:
        dtype = np.dtype([(name, ply_to_numpy[prop_type]) for name, prop_type in properties])
    except KeyError as exc:
        raise ValueError(f"Unsupported binary PLY property type: {exc.args[0]}") from exc

    with path.open("rb") as f:
        f.seek(header_bytes)
        data = np.fromfile(f, dtype=dtype, count=vertex_count)
    if data.shape[0] != vertex_count:
        raise ValueError(f"Binary PLY vertex data is truncated: {path}")
    points = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    return points, data[sdf_prop].astype(np.float32)


def _read_summary_value(summary_path: Path, keys: set[str], default: float | None = None) -> float:
    if not summary_path.is_file():
        if default is None:
            raise FileNotFoundError(f"SDF summary not found: {summary_path}")
        return float(default)
    with summary_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] in keys:
                return float(parts[1])
    if default is None:
        raise ValueError(f"Could not find {sorted(keys)} in summary: {summary_path}")
    return float(default)


def infer_voxel_size(points: np.ndarray, max_samples: int = 500000) -> float:
    sample = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if sample.shape[0] > max_samples:
        idx = np.linspace(0, sample.shape[0] - 1, num=max_samples, dtype=np.int64)
        sample = sample[idx]
    candidates: list[float] = []
    for axis in range(3):
        vals = np.unique(np.round(sample[:, axis], decimals=6))
        if vals.size < 2:
            continue
        diffs = np.diff(vals)
        diffs = diffs[diffs > 1e-6]
        if diffs.size > 0:
            candidates.append(float(diffs.min()))
    if not candidates:
        raise ValueError("Failed to infer voxel size from ESDF points.")
    return float(np.median(np.asarray(candidates, dtype=np.float32)))


def apply_affine(points: np.ndarray, affine_4x4: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    affine = np.asarray(affine_4x4, dtype=np.float32).reshape(4, 4)
    return (pts @ affine[:3, :3].T + affine[None, :3, 3]).astype(np.float32)


def voxel_center_to_idx(points: np.ndarray, origin: np.ndarray, voxel_size: float) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.int32)
    idx_float = (points - origin[None, :]) / float(voxel_size) - 0.5
    return np.rint(idx_float).astype(np.int32)


def voxel_idx_to_world_center(voxel_idx: np.ndarray, origin: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_idx.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return origin[None, :] + (voxel_idx.astype(np.float32) + 0.5) * float(voxel_size)


def _pack_idx(idx: np.ndarray, dims: np.ndarray) -> np.ndarray:
    idx64 = np.asarray(idx, dtype=np.int64).reshape(-1, 3)
    dims64 = np.asarray(dims, dtype=np.int64).reshape(3)
    return idx64[:, 0] + dims64[0] * (idx64[:, 1] + dims64[1] * idx64[:, 2])


def _source_grid_from_voxel_centers(points: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    scene_min = points.min(axis=0).astype(np.float32) - 0.5 * float(voxel_size)
    scene_max = points.max(axis=0).astype(np.float32) + 0.5 * float(voxel_size)
    extent = np.maximum(scene_max - scene_min, np.full((3,), voxel_size, dtype=np.float32))
    voxel_dims = np.rint(extent / float(voxel_size)).astype(np.int32)
    voxel_dims = np.maximum(voxel_dims, np.ones((3,), dtype=np.int32))
    return scene_min.astype(np.float32), voxel_dims.astype(np.int32)


def _candidate_grid_paths_from_esdf(esdf_path: Path) -> list[Path]:
    candidates: list[Path] = []
    if esdf_path.parent.name == "sdf":
        scene_dir = esdf_path.parent.parent
        candidates.extend(
            [
                scene_dir / "voxel_map_tsdf.npz",
                scene_dir / "voxel_map_tsdf.json",
                scene_dir / "voxel_map.npz",
                scene_dir / "voxel_map.json",
            ]
        )
    candidates.extend(
        [
            esdf_path.with_name("voxel_map_tsdf.npz"),
            esdf_path.with_name("voxel_map_tsdf.json"),
        ]
    )
    return candidates


def resolve_esdf_grid_path(esdf_path: str | Path, grid_metadata_path: str | Path | None = None) -> Path | None:
    if grid_metadata_path is not None and str(grid_metadata_path).strip():
        path = Path(grid_metadata_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ESDF grid metadata file not found: {path}")
        return path
    esdf = resolve_esdf_path(esdf_path)
    for candidate in _candidate_grid_paths_from_esdf(esdf):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _companion_grid_files(grid_path: Path | None) -> tuple[Path | None, Path | None]:
    if grid_path is None:
        return None, None
    if grid_path.suffix.lower() == ".npz":
        npz_path = grid_path
        json_path = grid_path.with_suffix(".json")
    elif grid_path.suffix.lower() == ".json":
        json_path = grid_path
        npz_path = grid_path.with_suffix(".npz")
    else:
        npz_path = grid_path if grid_path.suffix.lower() == ".npz" else None
        json_path = grid_path if grid_path.suffix.lower() == ".json" else None
    return (npz_path if npz_path is not None and npz_path.is_file() else None,
            json_path if json_path is not None and json_path.is_file() else None)


def _load_transform_json(transform_json: Path) -> np.ndarray:
    payload = json.load(transform_json.open("r", encoding="utf-8"))
    if "transform_4x4" in payload:
        return np.asarray(payload["transform_4x4"], dtype=np.float32).reshape(4, 4)
    if all(k in payload for k in ("scale", "rotation", "translation")):
        scale = float(payload["scale"])
        rot = np.asarray(payload["rotation"], dtype=np.float32).reshape(3, 3)
        trans = np.asarray(payload["translation"], dtype=np.float32).reshape(3)
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = scale * rot
        mat[:3, 3] = trans
        return mat
    raise ValueError(f"Transform json missing transform_4x4 or sim3 fields: {transform_json}")


def _metadata_from_json(json_path: Path | None) -> dict[str, Any]:
    if json_path is None or not json_path.is_file():
        return {}
    return json.load(json_path.open("r", encoding="utf-8"))


def load_esdf_grid_metadata(
    esdf_path: str | Path,
    grid_metadata_path: str | Path | None = None,
) -> EsdfGridMetadata:
    esdf_path = resolve_esdf_path(esdf_path)
    grid_path = resolve_esdf_grid_path(esdf_path, grid_metadata_path)
    npz_path, json_path = _companion_grid_files(grid_path)
    meta_json = _metadata_from_json(json_path)

    if npz_path is not None:
        data = np.load(npz_path)
        origin = np.asarray(data["origin"], dtype=np.float32).reshape(3)
        voxel_size = float(np.asarray(data["voxel_size"], dtype=np.float32).reshape(-1)[0])
        voxel_dims = np.asarray(data["voxel_dims"], dtype=np.int32).reshape(3)
    elif meta_json:
        origin = np.asarray(meta_json["origin"], dtype=np.float32).reshape(3)
        voxel_size = float(meta_json["transformed_source_voxel_size"])
        voxel_dims = np.asarray(meta_json["voxel_dims"], dtype=np.int32).reshape(3)
    else:
        points_raw, _ = load_esdf_ply(esdf_path)
        summary_path = esdf_path.parent / "summary.txt"
        source_voxel_size = _read_summary_value(
            summary_path,
            {"voxel_size_m", "voxel_size"},
            default=infer_voxel_size(points_raw),
        )
        origin, voxel_dims = _source_grid_from_voxel_centers(points_raw, source_voxel_size)
        return EsdfGridMetadata(
            origin=origin,
            voxel_size=float(source_voxel_size),
            voxel_dims=voxel_dims,
            source_to_world_affine=None,
            source_voxel_size=float(source_voxel_size),
            distance_scale=1.0,
            max_esdf_distance_m=_read_summary_value(
                summary_path,
                {"max_esdf_distance_m", "max_esdf_distance"},
                default=DEFAULT_UNKNOWN_DISTANCE_M,
            ),
            grid_path=None,
            grid_json_path=None,
        )

    affine = None
    if meta_json.get("dataset_affine") is not None:
        affine = np.asarray(meta_json["dataset_affine"], dtype=np.float32).reshape(4, 4)
    elif meta_json.get("transform_json"):
        transform_path = Path(str(meta_json["transform_json"])).expanduser()
        if transform_path.is_file():
            affine = _load_transform_json(transform_path.resolve())

    source_voxel_size = float(meta_json.get("source_voxel_size", voxel_size))
    summary_path = esdf_path.parent / "summary.txt"
    if "source_voxel_size" not in meta_json and summary_path.is_file():
        source_voxel_size = _read_summary_value(summary_path, {"voxel_size_m", "voxel_size"}, default=source_voxel_size)

    if affine is None:
        distance_scale = 1.0
    else:
        axis_lengths = np.linalg.norm(np.asarray(affine, dtype=np.float32)[:3, :3], axis=0)
        distance_scale = float(np.median(axis_lengths.astype(np.float32)))
    if "transformed_source_voxel_size" in meta_json and source_voxel_size > 0.0:
        distance_scale = float(meta_json["transformed_source_voxel_size"]) / float(source_voxel_size)

    max_esdf_distance_m = _read_summary_value(
        summary_path,
        {"max_esdf_distance_m", "max_esdf_distance"},
        default=DEFAULT_UNKNOWN_DISTANCE_M,
    ) * float(distance_scale)

    return EsdfGridMetadata(
        origin=origin,
        voxel_size=float(voxel_size),
        voxel_dims=voxel_dims,
        source_to_world_affine=affine,
        source_voxel_size=float(source_voxel_size),
        distance_scale=float(distance_scale),
        max_esdf_distance_m=float(max_esdf_distance_m),
        grid_path=npz_path or grid_path,
        grid_json_path=json_path,
    )


def _load_occupied_idx_from_grid(npz_path: Path | None) -> np.ndarray:
    if npz_path is None or not npz_path.is_file():
        return np.zeros((0, 3), dtype=np.int32)
    with np.load(npz_path) as data:
        if "occupied_voxel_idx" not in data:
            return np.zeros((0, 3), dtype=np.int32)
        return np.asarray(data["occupied_voxel_idx"], dtype=np.int32).reshape(-1, 3)


def _load_free_idx_from_grid(npz_path: Path | None) -> np.ndarray:
    if npz_path is None or not npz_path.is_file():
        return np.zeros((0, 3), dtype=np.int32)
    with np.load(npz_path) as data:
        if "free_voxel_idx" not in data:
            return np.zeros((0, 3), dtype=np.int32)
        return np.asarray(data["free_voxel_idx"], dtype=np.int32).reshape(-1, 3)


def _file_signature(path: Path | None) -> tuple[int, int]:
    if path is None or not path.is_file():
        return 0, 0
    stat = path.stat()
    return int(stat.st_size), int(stat.st_mtime_ns)


def _aligned_esdf_cache_path(esdf_path: Path, grid_path: Path | None) -> Path:
    grid_stem = "native" if grid_path is None else grid_path.stem
    return esdf_path.with_name(f"{esdf_path.stem}.{grid_stem}.aligned_cache.npz")


def _cache_signature_payload(esdf_path: Path, metadata: EsdfGridMetadata) -> dict[str, np.ndarray]:
    grid_path = metadata.grid_path
    json_path = metadata.grid_json_path
    summary_path = esdf_path.parent / "summary.txt"
    esdf_size, esdf_mtime = _file_signature(esdf_path)
    grid_size, grid_mtime = _file_signature(grid_path)
    json_size, json_mtime = _file_signature(json_path)
    summary_size, summary_mtime = _file_signature(summary_path)
    return {
        "cache_version": np.asarray([3], dtype=np.int32),
        "esdf_size": np.asarray([esdf_size], dtype=np.int64),
        "esdf_mtime_ns": np.asarray([esdf_mtime], dtype=np.int64),
        "grid_size": np.asarray([grid_size], dtype=np.int64),
        "grid_mtime_ns": np.asarray([grid_mtime], dtype=np.int64),
        "json_size": np.asarray([json_size], dtype=np.int64),
        "json_mtime_ns": np.asarray([json_mtime], dtype=np.int64),
        "summary_size": np.asarray([summary_size], dtype=np.int64),
        "summary_mtime_ns": np.asarray([summary_mtime], dtype=np.int64),
        "origin": metadata.origin.astype(np.float32),
        "voxel_size": np.asarray([metadata.voxel_size], dtype=np.float32),
        "voxel_dims": metadata.voxel_dims.astype(np.int32),
        "source_voxel_size": np.asarray([metadata.source_voxel_size], dtype=np.float32),
        "distance_scale": np.asarray([metadata.distance_scale], dtype=np.float32),
    }


def _cache_signature_matches(cache_data: np.lib.npyio.NpzFile, signature: dict[str, np.ndarray]) -> bool:
    for key, expected in signature.items():
        if key not in cache_data:
            return False
        if not np.array_equal(np.asarray(cache_data[key]), np.asarray(expected)):
            return False
    return True


def _try_load_aligned_esdf_cache(
    cache_path: Path,
    signature: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray] | None:
    if not cache_path.is_file():
        return None
    try:
        with np.load(cache_path) as data:
            if not _cache_signature_matches(data, signature):
                return None
            esdf_idx = np.asarray(data["esdf_idx"], dtype=np.int32).reshape(-1, 3)
            esdf_distance_m = np.asarray(data["esdf_distance_m"], dtype=np.float32).reshape(-1)
            if esdf_idx.shape[0] != esdf_distance_m.shape[0]:
                return None
            return esdf_idx, esdf_distance_m
    except Exception:
        return None


def _write_aligned_esdf_cache(
    cache_path: Path,
    signature: dict[str, np.ndarray],
    esdf_idx: np.ndarray,
    esdf_distance_m: np.ndarray,
) -> None:
    try:
        payload = dict(signature)
        payload["esdf_idx"] = np.asarray(esdf_idx, dtype=np.int32).reshape(-1, 3)
        payload["esdf_distance_m"] = np.asarray(esdf_distance_m, dtype=np.float32).reshape(-1)
        np.savez(cache_path, **payload)
    except Exception:
        return


def _source_esdf_lookup(
    target_idx: np.ndarray,
    *,
    metadata: EsdfGridMetadata,
    inverse_affine: np.ndarray | None,
    source_origin: np.ndarray,
    source_dims: np.ndarray,
    source_keys_sorted: np.ndarray,
    source_distance_sorted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.asarray(target_idx, dtype=np.int32).reshape(-1, 3)
    distances = np.full((idx.shape[0],), np.nan, dtype=np.float32)
    if idx.shape[0] == 0:
        return distances, np.zeros((0,), dtype=bool)

    target_centers = voxel_idx_to_world_center(idx, metadata.origin, metadata.voxel_size)
    source_points = target_centers if inverse_affine is None else apply_affine(target_centers, inverse_affine)
    source_idx = voxel_center_to_idx(source_points, source_origin, metadata.source_voxel_size)
    in_source = np.logical_and(source_idx >= 0, source_idx < source_dims[None, :]).all(axis=1)
    observed = np.zeros((idx.shape[0],), dtype=bool)
    if not in_source.any():
        return distances, observed

    rows = np.flatnonzero(in_source)
    source_keys = _pack_idx(source_idx[in_source], source_dims)
    pos = np.searchsorted(source_keys_sorted, source_keys)
    hit = pos < source_keys_sorted.shape[0]
    if hit.any():
        hit[hit] = source_keys_sorted[pos[hit]] == source_keys[hit]
    if hit.any():
        hit_rows = rows[hit]
        distances[hit_rows] = source_distance_sorted[pos[hit]].astype(np.float32) * float(metadata.distance_scale)
        observed[hit_rows] = True
    miss_rows = rows[~hit]
    if miss_rows.shape[0]:
        best_distance = np.full((miss_rows.shape[0],), np.nan, dtype=np.float32)
        best_distance_sq = np.full((miss_rows.shape[0],), np.inf, dtype=np.float32)
        miss_source_points = source_points[miss_rows]
        miss_nearest_idx = source_idx[miss_rows]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    offset = np.array([dx, dy, dz], dtype=np.int32)
                    candidate_source_idx = miss_nearest_idx + offset[None, :]
                    candidate_valid = np.logical_and(
                        candidate_source_idx >= 0,
                        candidate_source_idx < source_dims[None, :],
                    ).all(axis=1)
                    if not candidate_valid.any():
                        continue
                    candidate_rows = np.flatnonzero(candidate_valid)
                    candidate_keys = _pack_idx(candidate_source_idx[candidate_valid], source_dims)
                    candidate_pos = np.searchsorted(source_keys_sorted, candidate_keys)
                    candidate_hit = candidate_pos < source_keys_sorted.shape[0]
                    if candidate_hit.any():
                        candidate_hit[candidate_hit] = (
                            source_keys_sorted[candidate_pos[candidate_hit]] == candidate_keys[candidate_hit]
                        )
                    if not candidate_hit.any():
                        continue
                    local_rows = candidate_rows[candidate_hit]
                    centers = voxel_idx_to_world_center(
                        candidate_source_idx[candidate_valid][candidate_hit],
                        source_origin,
                        metadata.source_voxel_size,
                    )
                    dist_sq = np.sum((centers - miss_source_points[local_rows]) ** 2, axis=1)
                    improve = dist_sq < best_distance_sq[local_rows]
                    if improve.any():
                        improved_rows = local_rows[improve]
                        best_distance_sq[improved_rows] = dist_sq[improve].astype(np.float32)
                        best_distance[improved_rows] = (
                            source_distance_sorted[candidate_pos[candidate_hit][improve]].astype(np.float32)
                            * float(metadata.distance_scale)
                        )
        fallback_hit = np.isfinite(best_distance_sq)
        if fallback_hit.any():
            fallback_rows = miss_rows[fallback_hit]
            distances[fallback_rows] = best_distance[fallback_hit]
            observed[fallback_rows] = True
    return distances, observed


def _backward_fill_enclosed_slice_holes(
    unique_idx: np.ndarray,
    reduced_distance: np.ndarray,
    *,
    metadata: EsdfGridMetadata,
    inverse_affine: np.ndarray | None,
    source_origin: np.ndarray,
    source_dims: np.ndarray,
    source_keys_sorted: np.ndarray,
    source_distance_sorted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if unique_idx.shape[0] == 0:
        return unique_idx, reduced_distance

    x_dim = int(metadata.voxel_dims[0])
    y_dim = int(metadata.voxel_dims[1])
    if x_dim <= 0 or y_dim <= 0:
        return unique_idx, reduced_distance

    fill_idx_parts: list[np.ndarray] = []
    fill_distance_parts: list[np.ndarray] = []
    structure = ndimage.generate_binary_structure(2, 2)
    present_z = np.unique(unique_idx[:, 2])
    for z_value in present_z:
        plane_rows = np.flatnonzero(unique_idx[:, 2] == int(z_value))
        if plane_rows.shape[0] < 4:
            continue
        plane_idx = unique_idx[plane_rows]
        mask = np.zeros((x_dim, y_dim), dtype=bool)
        mask[plane_idx[:, 0], plane_idx[:, 1]] = True
        holes = np.logical_and(ndimage.binary_fill_holes(mask, structure=structure), ~mask)
        hole_count = int(np.count_nonzero(holes))
        if hole_count == 0:
            continue
        xy = np.column_stack(np.nonzero(holes)).astype(np.int32, copy=False)
        candidate_idx = np.empty((xy.shape[0], 3), dtype=np.int32)
        candidate_idx[:, 0] = xy[:, 0]
        candidate_idx[:, 1] = xy[:, 1]
        candidate_idx[:, 2] = int(z_value)
        candidate_distance, observed = _source_esdf_lookup(
            candidate_idx,
            metadata=metadata,
            inverse_affine=inverse_affine,
            source_origin=source_origin,
            source_dims=source_dims,
            source_keys_sorted=source_keys_sorted,
            source_distance_sorted=source_distance_sorted,
        )
        if observed.any():
            fill_idx_parts.append(candidate_idx[observed])
            fill_distance_parts.append(candidate_distance[observed])

    if not fill_idx_parts:
        return unique_idx, reduced_distance

    fill_idx = np.concatenate(fill_idx_parts, axis=0).astype(np.int32, copy=False)
    fill_distance = np.concatenate(fill_distance_parts, axis=0).astype(np.float32, copy=False)
    combined_idx = np.concatenate([unique_idx, fill_idx], axis=0).astype(np.int32, copy=False)
    combined_distance = np.concatenate([reduced_distance, fill_distance], axis=0).astype(np.float32, copy=False)
    combined_keys = _pack_idx(combined_idx, metadata.voxel_dims)
    unique_keys, inverse = np.unique(combined_keys, return_inverse=True)
    final_distance = np.full((unique_keys.shape[0],), np.inf, dtype=np.float32)
    np.minimum.at(final_distance, inverse, combined_distance)
    final_idx = np.empty((unique_keys.shape[0], 3), dtype=np.int32)
    dims64 = metadata.voxel_dims.astype(np.int64)
    final_idx[:, 0] = (unique_keys % dims64[0]).astype(np.int32)
    yz = unique_keys // dims64[0]
    final_idx[:, 1] = (yz % dims64[1]).astype(np.int32)
    final_idx[:, 2] = (yz // dims64[1]).astype(np.int32)
    return final_idx.astype(np.int32, copy=False), final_distance.astype(np.float32, copy=False)


def _align_esdf_samples_to_grid(
    points_raw: np.ndarray,
    distance_raw: np.ndarray,
    metadata: EsdfGridMetadata,
) -> tuple[np.ndarray, np.ndarray]:
    source_origin, source_dims = _source_grid_from_voxel_centers(points_raw, metadata.source_voxel_size)
    source_idx = voxel_center_to_idx(points_raw, source_origin, metadata.source_voxel_size)
    source_keys = _pack_idx(source_idx, source_dims)
    source_order = np.argsort(source_keys, kind="mergesort")
    source_keys_sorted = source_keys[source_order]
    source_distance_sorted = distance_raw[source_order].astype(np.float32, copy=False)
    source_centers = voxel_idx_to_world_center(source_idx, source_origin, metadata.source_voxel_size)
    points_world = (
        source_centers
        if metadata.source_to_world_affine is None
        else apply_affine(source_centers, metadata.source_to_world_affine)
    )
    target_idx_all = voxel_center_to_idx(points_world, metadata.origin, metadata.voxel_size)
    in_grid = np.logical_and(target_idx_all >= 0, target_idx_all < metadata.voxel_dims[None, :]).all(axis=1)
    target_idx = target_idx_all[in_grid].astype(np.int32, copy=False)
    target_distance = (distance_raw[in_grid].astype(np.float32, copy=False) * float(metadata.distance_scale)).astype(np.float32)
    if target_idx.shape[0] == 0:
        raise ValueError("No ESDF voxels fall inside the target grid")

    dims64 = metadata.voxel_dims.astype(np.int64)
    target_keys = _pack_idx(target_idx, metadata.voxel_dims)
    unique_keys, inverse = np.unique(target_keys, return_inverse=True)
    reduced_distance = np.full((unique_keys.shape[0],), np.inf, dtype=np.float32)
    np.minimum.at(reduced_distance, inverse, target_distance)
    unique_idx = np.empty((unique_keys.shape[0], 3), dtype=np.int32)
    unique_idx[:, 0] = (unique_keys % dims64[0]).astype(np.int32)
    yz = unique_keys // dims64[0]
    unique_idx[:, 1] = (yz % dims64[1]).astype(np.int32)
    unique_idx[:, 2] = (yz // dims64[1]).astype(np.int32)
    inverse_affine = (
        None
        if metadata.source_to_world_affine is None
        else np.linalg.inv(metadata.source_to_world_affine.astype(np.float64)).astype(np.float32)
    )
    unique_idx, reduced_distance = _backward_fill_enclosed_slice_holes(
        unique_idx.astype(np.int32, copy=False),
        reduced_distance.astype(np.float32, copy=False),
        metadata=metadata,
        inverse_affine=inverse_affine,
        source_origin=source_origin,
        source_dims=source_dims,
        source_keys_sorted=source_keys_sorted,
        source_distance_sorted=source_distance_sorted,
    )
    return unique_idx.astype(np.int32), reduced_distance.astype(np.float32)


def load_aligned_esdf_samples(
    esdf_path: str | Path,
    grid_metadata_path: str | Path | None = None,
    *,
    cache_aligned: bool = False,
) -> tuple[EsdfGridMetadata, np.ndarray, np.ndarray]:
    """Load aligned explicit ESDF samples without constructing a planner graph."""

    resolved_esdf_path = resolve_esdf_path(esdf_path)
    metadata = load_esdf_grid_metadata(resolved_esdf_path, grid_metadata_path)
    grid_npz_path, _ = _companion_grid_files(metadata.grid_path)
    cache_path = _aligned_esdf_cache_path(resolved_esdf_path, grid_npz_path)
    signature = _cache_signature_payload(resolved_esdf_path, metadata)
    cached = _try_load_aligned_esdf_cache(cache_path, signature) if cache_aligned else None
    if cached is None:
        points_raw, distance_raw = load_esdf_ply(resolved_esdf_path)
        esdf_idx, esdf_distance_m = _align_esdf_samples_to_grid(
            points_raw,
            distance_raw,
            metadata,
        )
        del points_raw, distance_raw
        if cache_aligned:
            _write_aligned_esdf_cache(
                cache_path,
                signature,
                esdf_idx,
                esdf_distance_m,
            )
    else:
        esdf_idx, esdf_distance_m = cached
    return metadata, esdf_idx, esdf_distance_m


def load_esdf_grid(
    esdf_path: str | Path,
    grid_metadata_path: str | Path | None = None,
    *,
    use_occupancy_guard: bool = True,
    load_free_idx: bool = False,
    cache_aligned: bool = False,
    unknown_policy: str = UNKNOWN_POLICY_CLEARANCE,
    allow_unknown: bool | None = None,
    unknown_clearance_m: float | None = None,
) -> EsdfGridGraph:
    """Load ESDF samples aligned to a target grid.

    Missing ESDF samples are unknown cells. unknown_policy chooses whether
    unknown cells are blocked, treated as free, or assigned unknown_clearance_m
    as proxy clearance. TSDF occupied guard cells remain blocked exactly when
    use_occupancy_guard is true. A KDTree is built only for the clearance
    unknown policy, where unknown cells need a nearest occupied voxel estimate.
    """

    esdf_path = resolve_esdf_path(esdf_path)
    metadata, esdf_idx, esdf_distance_m = load_aligned_esdf_samples(
        esdf_path,
        grid_metadata_path,
        cache_aligned=cache_aligned,
    )
    grid_npz_path, _ = _companion_grid_files(metadata.grid_path)
    occupied_idx = _load_occupied_idx_from_grid(grid_npz_path) if use_occupancy_guard else np.zeros((0, 3), dtype=np.int32)
    free_idx = _load_free_idx_from_grid(grid_npz_path) if load_free_idx else np.zeros((0, 3), dtype=np.int32)
    unknown_clearance = metadata.max_esdf_distance_m if unknown_clearance_m is None else float(unknown_clearance_m)
    if allow_unknown is not None:
        unknown_policy = UNKNOWN_POLICY_CLEARANCE if bool(allow_unknown) else UNKNOWN_POLICY_OCCUPIED
    return EsdfGridGraph(
        origin=metadata.origin,
        voxel_size=metadata.voxel_size,
        voxel_dims=metadata.voxel_dims,
        esdf_idx=esdf_idx,
        esdf_distance_m=esdf_distance_m,
        occupied_idx=occupied_idx,
        free_idx=free_idx,
        unknown_clearance_m=unknown_clearance,
        unknown_policy=unknown_policy,
    )


def _esdf_step_cost(
    graph: EsdfGridGraph,
    idx: np.ndarray,
    move_cost: float,
    safety_clearance_m: float,
    clearance_cost_weight: float,
) -> float:
    return float(
        graph.step_costs_for_idx_batch(
            np.asarray(idx, dtype=np.int32).reshape(1, 3),
            np.asarray([move_cost], dtype=np.float64),
            safety_clearance_m=float(safety_clearance_m),
            clearance_cost_weight=float(clearance_cost_weight),
        )[0]
    )


def esdf_astar_voxel_indices(
    graph: EsdfGridGraph,
    start_world: np.ndarray,
    goal_world: np.ndarray,
    *,
    connectivity: int = 26,
    max_expansions: int = 500000,
    snap_search_radius: int = 8,
    safety_clearance_m: float = 0.0,
    clearance_cost_weight: float = 1.0,
    heuristic_weight: float = 1.0,
    max_planning_time_sec: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    goal_idx = graph.find_safe_idx_near_world(goal_world, snap_search_radius, safety_clearance_m)
    if goal_idx is None:
        goal_target_idx = graph._world_to_voxel_idx_unclipped(goal_world)
        raise ValueError(
            f"no ESDF-safe goal voxel within radius {int(snap_search_radius)} "
            f"of goal_idx {goal_target_idx.tolist()} and clearance {float(safety_clearance_m):.3f}m; "
            "aborting before A* search"
        )
    start_idx = graph.find_safe_idx_near_world(start_world, snap_search_radius, safety_clearance_m)
    if start_idx is None:
        start_target_idx = graph._world_to_voxel_idx_unclipped(start_world)
        raise ValueError(
            f"no ESDF-safe start voxel within radius {int(snap_search_radius)} "
            f"of start_idx {start_target_idx.tolist()} and clearance {float(safety_clearance_m):.3f}m; "
            "aborting before A* search"
        )
    offsets, offset_costs = _neighbor_offsets(connectivity)
    heuristic_weight = max(1.0, float(heuristic_weight))
    clearance_cost_weight = max(0.0, float(clearance_cost_weight))

    start_key = int(graph._pack(start_idx.reshape(1, 3))[0])
    goal_key = int(graph._pack(goal_idx.reshape(1, 3))[0])
    if start_key == goal_key:
        return start_idx.reshape(1, 3), start_idx, goal_idx

    goal_f = goal_idx.astype(np.float32)
    open_heap: list[tuple[float, float, int]] = []
    g_score: dict[int, float] = {start_key: 0.0}
    parent: dict[int, int] = {}
    key_to_idx: dict[int, np.ndarray] = {start_key: start_idx.copy()}
    closed: set[int] = set()
    deadline = None
    if max_planning_time_sec is not None and float(max_planning_time_sec) > 0.0:
        deadline = time.perf_counter() + float(max_planning_time_sec)

    start_h = heuristic_weight * _grid_distance_heuristic(start_idx, goal_f, connectivity)
    heapq.heappush(open_heap, (start_h, 0.0, start_key))
    expansions = 0

    while open_heap:
        _, current_g, current_key = heapq.heappop(open_heap)
        if current_key in closed:
            continue
        if current_g > g_score.get(current_key, float("inf")) + 1e-8:
            continue

        current_idx = key_to_idx[current_key]
        if current_key == goal_key:
            break

        expansions += 1
        if deadline is not None and expansions % 256 == 0 and time.perf_counter() > deadline:
            raise RuntimeError(f"ESDF A* exceeded max planning time ({float(max_planning_time_sec):.3f}s)")
        if expansions > int(max_expansions):
            raise RuntimeError(f"ESDF A* exceeded max expansions ({int(max_expansions)})")
        closed.add(current_key)

        neighbor_idx = current_idx[None, :] + offsets
        valid = np.logical_and(neighbor_idx >= 0, neighbor_idx < graph.voxel_dims[None, :]).all(axis=1)
        neighbor_idx = neighbor_idx[valid]
        neighbor_costs = offset_costs[valid]
        if neighbor_idx.shape[0] == 0:
            continue

        neighbor_keys = graph._pack(neighbor_idx)
        open_mask = np.fromiter(
            (int(key) not in closed for key in neighbor_keys),
            dtype=bool,
            count=neighbor_keys.shape[0],
        )
        if not open_mask.any():
            continue
        neighbor_idx = neighbor_idx[open_mask]
        neighbor_costs = neighbor_costs[open_mask]
        neighbor_keys = neighbor_keys[open_mask]
        step_costs = graph.step_costs_for_idx_batch(
            neighbor_idx,
            neighbor_costs.astype(np.float64, copy=False),
            safety_clearance_m=float(safety_clearance_m),
            clearance_cost_weight=clearance_cost_weight,
        )

        for n_idx, n_key_value, step_cost in zip(neighbor_idx, neighbor_keys, step_costs):
            if not np.isfinite(step_cost):
                continue
            n_key = int(n_key_value)
            tentative_g = current_g + step_cost
            if tentative_g >= g_score.get(n_key, float("inf")):
                continue

            g_score[n_key] = tentative_g
            parent[n_key] = current_key
            key_to_idx[n_key] = n_idx.astype(np.int32, copy=True)
            h = heuristic_weight * _grid_distance_heuristic(n_idx, goal_f, connectivity)
            heapq.heappush(open_heap, (tentative_g + h, tentative_g, n_key))
    else:
        raise RuntimeError("ESDF A* could not find a path between the selected points")

    path_keys = [goal_key]
    while path_keys[-1] != start_key:
        prev = parent.get(path_keys[-1])
        if prev is None:
            raise RuntimeError("ESDF A* path reconstruction failed")
        path_keys.append(prev)
    path_keys.reverse()

    path_idx = np.stack([key_to_idx[key] for key in path_keys], axis=0).astype(np.int32)
    return path_idx, start_idx.astype(np.int32), goal_idx.astype(np.int32)


def esdf_astar_world_path(
    graph: EsdfGridGraph,
    start_world: np.ndarray,
    goal_world: np.ndarray,
    *,
    connectivity: int = 26,
    max_expansions: int = 500000,
    snap_search_radius: int = 8,
    safety_clearance_m: float = 0.0,
    clearance_cost_weight: float = 1.0,
    heuristic_weight: float = 1.0,
    max_planning_time_sec: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path_idx, start_idx, goal_idx = esdf_astar_voxel_indices(
        graph,
        start_world,
        goal_world,
        connectivity=connectivity,
        max_expansions=max_expansions,
        snap_search_radius=snap_search_radius,
        safety_clearance_m=safety_clearance_m,
        clearance_cost_weight=clearance_cost_weight,
        heuristic_weight=heuristic_weight,
        max_planning_time_sec=max_planning_time_sec,
    )
    path_world = graph.voxel_idx_to_world_center(path_idx).astype(np.float32)
    return path_world, start_idx, goal_idx
