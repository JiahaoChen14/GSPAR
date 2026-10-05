#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EGO-style local trajectory optimization on Occupancy Map voxels.

This module adapts the ESDF-free collision term from EGO-Planner to the
static Occupancy Map representation used by this project.  The optimizer uses
the coarse global path as the collision-free guiding path and only queries
occupied voxels when collision information is needed; it does not build or
consume an ESDF.
"""

from __future__ import annotations

import argparse
import heapq
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from scipy.sparse import csr_array
from scipy.spatial import cKDTree

from GSPAR.mapping.voxel_map_cli import resolve_existing_path
from GSPAR.path.global_path.astar_world_path import _grid_distance_heuristic, _neighbor_offsets


LOCAL_EGO_METHOD = "EGO (Occupancy Map)"


@dataclass
class CoarsePath:
    """Parsed coarse path with optional header metadata."""

    points: np.ndarray
    header: dict[str, str] = field(default_factory=dict)
    raw_header_lines: list[str] = field(default_factory=list)
    path: Path | None = None


@dataclass
class EgoPlannerConfig:
    """Configuration for the occupancy-map EGO local planner."""

    control_point_spacing_m: float = 0.30
    output_spacing_m: float = 0.05
    collision_sample_spacing_m: float = 0.05
    safety_clearance_m: float | None = None
    safety_clearance_voxels: float | None = None
    default_safety_clearance_voxels: float = 2.5
    inherit_input_clearance: bool = True
    treat_unknown_as_occupied: bool = False
    pin_endpoint_control_points: int = 2
    rebound_iterations: int = 8
    smooth_weight: float = 1.0
    jerk_weight: float = 0.1
    collision_weight: float = 200.0
    fitness_weight: float = 100.0
    fitness_axis_weight: float = 0.05
    fitness_radial_weight: float = 1.0
    optimizer_maxiter: int = 80
    optimizer_ftol: float = 1e-9
    optimizer_gtol: float = 1e-6
    guide_astar_fallback: bool = True
    guide_astar_margin_voxels: int = 32
    guide_astar_max_expansions: int = 80000
    guide_astar_time_limit_sec: float = 0.25
    connectivity: int = 26
    max_velocity_mps: float | None = None
    max_acceleration_mps2: float | None = None
    max_jerk_mps3: float | None = None
    initial_dt: float = 0.20
    feasibility_weight: float = 0.0
    safe_fallback_to_input: bool = True

    def input_safety_clearance_voxels(self, coarse_path: CoarsePath | None = None) -> float | None:
        if coarse_path is not None:
            for key in ("safety_clearance_voxels", "obstacle_safety_margin_voxels"):
                value = coarse_path.header.get(key)
                if value is None:
                    continue
                try:
                    return max(0.0, float(value.split()[0]))
                except ValueError:
                    continue
        return None

    def input_safety_clearance_m(self, coarse_path: CoarsePath | None = None) -> float | None:
        if coarse_path is not None:
            for key in ("safety_clearance_m", "obstacle_safety_margin_m", "clearance_m"):
                value = coarse_path.header.get(key)
                if value is None:
                    continue
                try:
                    return max(0.0, float(value.split()[0]))
                except ValueError:
                    continue
        return None

    def resolved_safety_clearance_voxels(
        self,
        coarse_path: CoarsePath | None = None,
        *,
        voxel_size_m: float | None = None,
    ) -> float:
        voxel_size = None if voxel_size_m is None else max(float(voxel_size_m), 1e-12)
        if self.safety_clearance_m is not None:
            if voxel_size is None:
                raise ValueError("voxel_size_m is required when safety_clearance_m is set")
            return max(0.0, float(self.safety_clearance_m)) / voxel_size
        if self.safety_clearance_voxels is not None:
            return max(0.0, float(self.safety_clearance_voxels))
        input_clearance_m = self.input_safety_clearance_m(coarse_path)
        if self.inherit_input_clearance and input_clearance_m is not None:
            if voxel_size is None:
                raise ValueError("voxel_size_m is required for meter-based input clearance")
            return input_clearance_m / voxel_size
        input_clearance = self.input_safety_clearance_voxels(coarse_path)
        if self.inherit_input_clearance and input_clearance is not None:
            return input_clearance
        if input_clearance is not None:
            return min(input_clearance, max(0.0, float(self.default_safety_clearance_voxels)))
        return max(0.0, float(self.default_safety_clearance_voxels))


@dataclass
class EgoPlannerResult:
    """Result returned by :func:`ego_occupancy_path`."""

    path_world: np.ndarray
    control_points_world: np.ndarray
    reference_control_points_world: np.ndarray
    safety_clearance_voxels: float
    safety_clearance_m: float
    dt: float
    iterations: int
    obstacle_pair_count: int
    unsafe_sample_count: int
    min_clearance_voxels: float
    min_clearance_m: float
    length_m: float
    success: bool
    message: str
    stats: dict[str, Any] = field(default_factory=dict)


class OccupancyGrid:
    """Voxel occupancy queries for local trajectory optimization."""

    def __init__(
        self,
        origin: np.ndarray,
        voxel_size: float,
        voxel_dims: np.ndarray,
        occupied_idx: np.ndarray,
        unknown_idx: np.ndarray | None = None,
        *,
        treat_unknown_as_occupied: bool = False,
    ) -> None:
        self.origin = np.asarray(origin, dtype=np.float64).reshape(3)
        self.voxel_size = float(voxel_size)
        self.voxel_dims = np.asarray(voxel_dims, dtype=np.int32).reshape(3)
        self.occupied_idx = np.asarray(occupied_idx, dtype=np.int32).reshape(-1, 3)
        self.unknown_idx = (
            np.asarray(unknown_idx, dtype=np.int32).reshape(-1, 3)
            if unknown_idx is not None
            else np.zeros((0, 3), dtype=np.int32)
        )
        if treat_unknown_as_occupied and self.unknown_idx.shape[0]:
            self.blocked_idx = np.concatenate([self.occupied_idx, self.unknown_idx], axis=0)
        else:
            self.blocked_idx = self.occupied_idx.copy()
        self.treat_unknown_as_occupied = bool(treat_unknown_as_occupied)
        self._dims64 = self.voxel_dims.astype(np.int64)
        self._blocked_keys = set(self._pack(self.blocked_idx).tolist())
        self._blocked_tree = (
            cKDTree(self.blocked_idx.astype(np.float32))
            if self.blocked_idx.shape[0]
            else None
        )

    def _pack(self, idx: np.ndarray) -> np.ndarray:
        idx64 = np.asarray(idx, dtype=np.int64).reshape(-1, 3)
        return idx64[:, 0] + self._dims64[0] * (idx64[:, 1] + self._dims64[1] * idx64[:, 2])

    @property
    def world_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = self.origin + 0.5 * self.voxel_size
        hi = self.origin + (self.voxel_dims.astype(np.float64) - 0.5) * self.voxel_size
        return lo, hi

    def is_valid_idx(self, idx: np.ndarray) -> bool:
        idx = np.asarray(idx, dtype=np.int32).reshape(3)
        return bool(np.logical_and(idx >= 0, idx < self.voxel_dims).all())

    def world_to_voxel_idx(self, points_world: np.ndarray) -> np.ndarray:
        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        idx = np.floor((points - self.origin[None, :]) / self.voxel_size).astype(np.int32)
        return idx.reshape(np.asarray(points_world).shape)

    def voxel_idx_to_world_center(self, idx: np.ndarray) -> np.ndarray:
        idx_arr = np.asarray(idx, dtype=np.float64).reshape(-1, 3)
        points = self.origin[None, :] + (idx_arr + 0.5) * self.voxel_size
        return points.reshape(np.asarray(idx).shape).astype(np.float32)

    def is_blocked_idx(self, idx: np.ndarray) -> bool:
        if not self.is_valid_idx(idx):
            return True
        key = int(self._pack(np.asarray(idx, dtype=np.int32).reshape(1, 3))[0])
        return key in self._blocked_keys

    def clearance_voxels_for_idx(self, idx: np.ndarray) -> float:
        idx = np.asarray(idx, dtype=np.int32).reshape(3)
        if not self.is_valid_idx(idx):
            return 0.0
        if self._blocked_tree is None:
            return float("inf")
        return float(self._blocked_tree.query(idx.astype(np.float32), k=1)[0])

    def clearance_voxels_for_points(self, points_world: np.ndarray) -> np.ndarray:
        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        if points.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        idx = self.world_to_voxel_idx(points).reshape(-1, 3)
        valid = np.logical_and(idx >= 0, idx < self.voxel_dims[None, :]).all(axis=1)
        clearances = np.zeros((points.shape[0],), dtype=np.float32)
        if self._blocked_tree is None:
            clearances[valid] = np.inf
            return clearances
        if valid.any():
            clearances[valid] = self._blocked_tree.query(idx[valid].astype(np.float32), k=1)[0].astype(np.float32)
        return clearances

    def is_safe_idx(self, idx: np.ndarray, safety_clearance_voxels: float) -> bool:
        if self.is_blocked_idx(idx):
            return False
        if float(safety_clearance_voxels) <= 0.0:
            return True
        return self.clearance_voxels_for_idx(idx) >= float(safety_clearance_voxels)

    def is_safe_point(self, point_world: np.ndarray, safety_clearance_voxels: float) -> bool:
        idx = self.world_to_voxel_idx(np.asarray(point_world, dtype=np.float64).reshape(1, 3))[0]
        return self.is_safe_idx(idx, safety_clearance_voxels)

    def snap_to_safe_idx(
        self,
        point_world: np.ndarray,
        safety_clearance_voxels: float,
        *,
        max_radius: int = 12,
    ) -> np.ndarray:
        target = self.world_to_voxel_idx(np.asarray(point_world, dtype=np.float64).reshape(1, 3))[0]
        target = np.clip(target, 0, self.voxel_dims - 1).astype(np.int32)
        if self.is_safe_idx(target, safety_clearance_voxels):
            return target

        best_idx: np.ndarray | None = None
        best_dist2 = float("inf")
        for radius in range(1, max(0, int(max_radius)) + 1):
            lo = np.maximum(target - radius, 0)
            hi = np.minimum(target + radius, self.voxel_dims - 1)
            for x in range(int(lo[0]), int(hi[0]) + 1):
                for y in range(int(lo[1]), int(hi[1]) + 1):
                    for z in range(int(lo[2]), int(hi[2]) + 1):
                        idx = np.array([x, y, z], dtype=np.int32)
                        if not self.is_safe_idx(idx, safety_clearance_voxels):
                            continue
                        dist2 = float(np.sum((idx.astype(np.float32) - target.astype(np.float32)) ** 2))
                        if dist2 < best_dist2:
                            best_idx = idx
                            best_dist2 = dist2
            if best_idx is not None:
                return best_idx
        raise ValueError(f"no safe voxel found within radius {int(max_radius)}")


def load_occupancy_grid(
    npz_path: str | Path,
    *,
    treat_unknown_as_occupied: bool = False,
) -> OccupancyGrid:
    path = resolve_existing_path(npz_path, default_suffix=".npz")
    if not path.is_file():
        raise FileNotFoundError(f"occupancy map not found: {path}")
    data = np.load(path)
    required = ("origin", "voxel_size", "voxel_dims", "occupied_voxel_idx")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"occupancy map missing keys: {missing}")
    unknown = np.asarray(data["unknown_voxel_idx"], dtype=np.int32) if "unknown_voxel_idx" in data else None
    return OccupancyGrid(
        origin=np.asarray(data["origin"], dtype=np.float64),
        voxel_size=float(np.asarray(data["voxel_size"]).reshape(-1)[0]),
        voxel_dims=np.asarray(data["voxel_dims"], dtype=np.int32),
        occupied_idx=np.asarray(data["occupied_voxel_idx"], dtype=np.int32),
        unknown_idx=unknown,
        treat_unknown_as_occupied=treat_unknown_as_occupied,
    )


def read_coarse_path(path: str | Path) -> CoarsePath:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"coarse path not found: {resolved}")
    if resolved.suffix.lower() == ".npy":
        points = np.asarray(np.load(resolved), dtype=np.float32)
        return CoarsePath(points=_validate_path_points(points), path=resolved)

    header: dict[str, str] = {}
    raw_header_lines: list[str] = []
    with resolved.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped.startswith("#"):
                continue
            content = stripped[1:].strip()
            raw_header_lines.append(content)
            if ":" in content:
                key, value = content.split(":", 1)
                header[key.strip()] = value.strip()

    points = np.asarray(np.loadtxt(resolved, comments="#", delimiter=None), dtype=np.float32)
    return CoarsePath(
        points=_validate_path_points(points),
        header=header,
        raw_header_lines=raw_header_lines,
        path=resolved,
    )


def _validate_path_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 1:
        if points.shape[0] != 3:
            raise ValueError("path file must contain Nx3 coordinates")
        points = points.reshape(1, 3)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("path file must contain Nx3 coordinates")
    if points.shape[0] < 2:
        raise ValueError("path must contain at least two points")
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if points.shape[0] < 2:
        raise ValueError("path must contain at least two finite points")
    keep = [0]
    for i in range(1, points.shape[0]):
        if np.linalg.norm(points[i] - points[keep[-1]]) > 1e-7:
            keep.append(i)
    if len(keep) < 2:
        raise ValueError("path degenerates to a single point")
    return points[np.asarray(keep, dtype=np.int32)].astype(np.float32)


def path_length(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def resample_polyline(points: np.ndarray, spacing_m: float) -> np.ndarray:
    points = _validate_path_points(points)
    spacing = max(float(spacing_m), 1e-6)
    seg = np.linalg.norm(np.diff(points.astype(np.float64), axis=0), axis=1)
    total = float(seg.sum())
    if total <= spacing:
        return points.copy()
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    count = max(2, int(math.ceil(total / spacing)) + 1)
    targets = np.linspace(0.0, total, count)
    out = np.empty((count, 3), dtype=np.float32)
    j = 0
    for i, s in enumerate(targets):
        while j < seg.shape[0] - 1 and cumulative[j + 1] < s:
            j += 1
        denom = max(float(seg[j]), 1e-12)
        u = float((s - cumulative[j]) / denom)
        out[i] = (1.0 - u) * points[j] + u * points[j + 1]
    out[0] = points[0]
    out[-1] = points[-1]
    return out


def _clamped_uniform_knots(num_control_points: int, degree: int = 3) -> np.ndarray:
    if num_control_points <= degree:
        raise ValueError("num_control_points must be greater than spline degree")
    interior = num_control_points - degree - 1
    if interior > 0:
        inner = np.linspace(0.0, 1.0, interior + 2, dtype=np.float64)[1:-1]
        return np.concatenate([
            np.zeros(degree + 1, dtype=np.float64),
            inner,
            np.ones(degree + 1, dtype=np.float64),
        ])
    return np.concatenate([
        np.zeros(degree + 1, dtype=np.float64),
        np.ones(degree + 1, dtype=np.float64),
    ])


def _design_matrix(num_control_points: int, sample_count: int, degree: int = 3) -> tuple[np.ndarray, csr_array]:
    sample_count = max(2, int(sample_count))
    if num_control_points <= degree:
        params = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
        rows = np.zeros((sample_count, num_control_points), dtype=np.float64)
        if num_control_points == 1:
            rows[:, 0] = 1.0
        else:
            x = params * (num_control_points - 1)
            lo = np.floor(x).astype(np.int32)
            lo = np.clip(lo, 0, num_control_points - 2)
            u = x - lo
            rows[np.arange(sample_count), lo] = 1.0 - u
            rows[np.arange(sample_count), lo + 1] = u
        return params, csr_array(rows)
    knots = _clamped_uniform_knots(num_control_points, degree)
    params = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
    matrix = BSpline.design_matrix(params, knots, degree, extrapolate=False)
    return params, csr_array(matrix)


def sample_control_spline(
    control_points: np.ndarray,
    *,
    spacing_m: float,
    degree: int = 3,
) -> tuple[np.ndarray, np.ndarray, csr_array]:
    q = np.asarray(control_points, dtype=np.float64).reshape(-1, 3)
    approx_length = path_length(q)
    sample_count = max(2, int(math.ceil(max(approx_length, spacing_m) / max(spacing_m, 1e-6))) + 1)
    params, basis = _design_matrix(q.shape[0], sample_count, degree=degree)
    samples = basis @ q
    return np.asarray(samples, dtype=np.float32), params, basis


def _reference_tangents(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    tangents = np.zeros_like(points)
    if points.shape[0] == 1:
        tangents[:, 0] = 1.0
        return tangents
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    if points.shape[0] > 2:
        tangents[1:-1] = points[2:] - points[:-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    tangents = np.where(norms > 1e-12, tangents / np.maximum(norms, 1e-12), fallback)
    return tangents


def _runs_from_indices(indices: set[int], count: int, *, expand: int = 1) -> list[tuple[int, int]]:
    if not indices:
        return []
    expanded: set[int] = set()
    for idx in indices:
        for j in range(max(0, idx - expand), min(count, idx + expand + 1)):
            expanded.add(j)
    ordered = sorted(expanded)
    runs: list[tuple[int, int]] = []
    begin = ordered[0]
    prev = ordered[0]
    for idx in ordered[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((begin, prev))
        begin = prev = idx
    runs.append((begin, prev))
    return runs


def _nearest_segment_point(point: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    point64 = np.asarray(point, dtype=np.float64).reshape(3)
    path = np.asarray(polyline, dtype=np.float64).reshape(-1, 3)
    if path.shape[0] == 1:
        return path[0].copy()
    best_point = path[0]
    best_dist2 = float("inf")
    for a, b in zip(path[:-1], path[1:]):
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            candidate = a
        else:
            u = float(np.clip(np.dot(point64 - a, ab) / denom, 0.0, 1.0))
            candidate = a + u * ab
        dist2 = float(np.sum((candidate - point64) ** 2))
        if dist2 < best_dist2:
            best_point = candidate
            best_dist2 = dist2
    return best_point.copy()


def _plane_intersection_or_nearest(
    point: np.ndarray,
    tangent: np.ndarray,
    polyline: np.ndarray,
) -> np.ndarray:
    point64 = np.asarray(point, dtype=np.float64).reshape(3)
    normal = np.asarray(tangent, dtype=np.float64).reshape(3)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-12:
        return _nearest_segment_point(point64, polyline)
    normal = normal / normal_norm
    path = np.asarray(polyline, dtype=np.float64).reshape(-1, 3)
    if path.shape[0] < 2:
        return path[0].copy()

    best_point: np.ndarray | None = None
    best_dist2 = float("inf")
    signed = (path - point64[None, :]) @ normal
    for i, (a, b) in enumerate(zip(path[:-1], path[1:])):
        da = float(signed[i])
        db = float(signed[i + 1])
        denom = db - da
        if abs(denom) <= 1e-12:
            if abs(da) > 1e-6:
                continue
            u = 0.0
        else:
            u = -da / denom
            if u < -1e-7 or u > 1.0 + 1e-7:
                continue
            u = float(np.clip(u, 0.0, 1.0))
        candidate = a + u * (b - a)
        dist2 = float(np.sum((candidate - point64) ** 2))
        if dist2 < best_dist2:
            best_point = candidate
            best_dist2 = dist2

    if best_point is not None:
        return best_point
    return _nearest_segment_point(point64, path)


def _slice_reference_guide(
    reference_path: np.ndarray,
    start_point: np.ndarray,
    end_point: np.ndarray,
) -> np.ndarray:
    reference = np.asarray(reference_path, dtype=np.float64).reshape(-1, 3)
    if reference.shape[0] <= 2:
        return reference.copy()
    start_idx = int(np.argmin(np.linalg.norm(reference - np.asarray(start_point)[None, :], axis=1)))
    end_idx = int(np.argmin(np.linalg.norm(reference - np.asarray(end_point)[None, :], axis=1)))
    if start_idx <= end_idx:
        guide = reference[start_idx:end_idx + 1]
    else:
        guide = reference[end_idx:start_idx + 1][::-1]
    if guide.shape[0] < 2:
        lo = max(0, min(start_idx, end_idx) - 2)
        hi = min(reference.shape[0], max(start_idx, end_idx) + 3)
        guide = reference[lo:hi]
    return guide.copy()


def _local_astar_guide(
    grid: OccupancyGrid,
    start_world: np.ndarray,
    goal_world: np.ndarray,
    *,
    safety_clearance_voxels: float,
    margin_voxels: int,
    max_expansions: int,
    time_limit_sec: float,
    connectivity: int,
) -> np.ndarray:
    start_idx = grid.snap_to_safe_idx(start_world, safety_clearance_voxels, max_radius=max(8, margin_voxels // 2))
    goal_idx = grid.snap_to_safe_idx(goal_world, safety_clearance_voxels, max_radius=max(8, margin_voxels // 2))
    if np.array_equal(start_idx, goal_idx):
        return grid.voxel_idx_to_world_center(start_idx.reshape(1, 3))

    margin = max(1, int(margin_voxels))
    lo = np.maximum(np.minimum(start_idx, goal_idx) - margin, 0).astype(np.int32)
    hi = np.minimum(np.maximum(start_idx, goal_idx) + margin + 1, grid.voxel_dims).astype(np.int32)
    offsets, costs = _neighbor_offsets(connectivity)
    start_key = int(grid._pack(start_idx.reshape(1, 3))[0])
    goal_key = int(grid._pack(goal_idx.reshape(1, 3))[0])
    goal_f = goal_idx.astype(np.float32)
    open_heap: list[tuple[float, float, int]] = []
    g_score: dict[int, float] = {start_key: 0.0}
    parent: dict[int, int] = {}
    key_to_idx: dict[int, np.ndarray] = {start_key: start_idx.copy()}
    closed: set[int] = set()
    heapq.heappush(open_heap, (_grid_distance_heuristic(start_idx, goal_f, connectivity), 0.0, start_key))
    deadline = time.perf_counter() + max(0.0, float(time_limit_sec))
    expansions = 0

    while open_heap:
        if time_limit_sec > 0.0 and time.perf_counter() > deadline:
            raise RuntimeError("local guide A* timed out")
        _, current_g, current_key = heapq.heappop(open_heap)
        if current_key in closed:
            continue
        if current_key == goal_key:
            break
        closed.add(current_key)
        current_idx = key_to_idx[current_key]
        expansions += 1
        if max_expansions > 0 and expansions > max_expansions:
            raise RuntimeError("local guide A* exceeded expansion limit")

        for offset, step_cost in zip(offsets, costs):
            n_idx = (current_idx + offset).astype(np.int32)
            if np.any(n_idx < lo) or np.any(n_idx >= hi):
                continue
            if not grid.is_safe_idx(n_idx, safety_clearance_voxels):
                continue
            n_key = int(grid._pack(n_idx.reshape(1, 3))[0])
            if n_key in closed:
                continue
            tentative_g = current_g + float(step_cost)
            if tentative_g >= g_score.get(n_key, float("inf")):
                continue
            g_score[n_key] = tentative_g
            parent[n_key] = current_key
            key_to_idx[n_key] = n_idx.copy()
            h = _grid_distance_heuristic(n_idx, goal_f, connectivity)
            heapq.heappush(open_heap, (tentative_g + h, tentative_g, n_key))
    else:
        raise RuntimeError("local guide A* could not find a path")

    keys = [goal_key]
    while keys[-1] != start_key:
        prev = parent.get(keys[-1])
        if prev is None:
            raise RuntimeError("local guide A* reconstruction failed")
        keys.append(prev)
    keys.reverse()
    idx_path = np.stack([key_to_idx[key] for key in keys], axis=0)
    return grid.voxel_idx_to_world_center(idx_path)


def _is_polyline_safe(
    grid: OccupancyGrid,
    points: np.ndarray,
    *,
    safety_clearance_voxels: float,
    spacing_m: float,
) -> bool:
    dense = resample_polyline(points, max(float(spacing_m), grid.voxel_size))
    clearances = grid.clearance_voxels_for_points(dense)
    return bool(np.all(clearances >= float(safety_clearance_voxels)))


def _collision_cost_value_grad(
    q: np.ndarray,
    obstacle_pairs: list[list[tuple[np.ndarray, np.ndarray]]],
    sample_obstacle_pairs: list[tuple[int, np.ndarray, np.ndarray]],
    sample_basis: csr_array | None,
    *,
    clearance_m: float,
    weight: float,
) -> tuple[float, np.ndarray]:
    cost = 0.0
    grad = np.zeros_like(q, dtype=np.float64)
    sf = max(float(clearance_m), 1e-9)
    for i, pairs in enumerate(obstacle_pairs):
        qi = q[i]
        for p, v in pairs:
            d = float(np.dot(qi - p, v))
            c = sf - d
            if c <= 0.0:
                continue
            if c <= sf:
                local_cost = c ** 3
                dcost_dc = 3.0 * c ** 2
            else:
                local_cost = 3.0 * sf * c ** 2 - 3.0 * sf ** 2 * c + sf ** 3
                dcost_dc = 6.0 * sf * c - 3.0 * sf ** 2
            cost += weight * local_cost
            grad[i] -= weight * dcost_dc * v
    if sample_obstacle_pairs and sample_basis is not None:
        basis_csr = sample_basis.tocsr()
        for row, p, v in sample_obstacle_pairs:
            start, end = basis_csr.indptr[row], basis_csr.indptr[row + 1]
            cols = basis_csr.indices[start:end]
            vals = basis_csr.data[start:end]
            if cols.shape[0] == 0:
                continue
            sample = vals @ q[cols]
            d = float(np.dot(sample - p, v))
            c = sf - d
            if c <= 0.0:
                continue
            if c <= sf:
                local_cost = c ** 3
                dcost_dc = 3.0 * c ** 2
            else:
                local_cost = 3.0 * sf * c ** 2 - 3.0 * sf ** 2 * c + sf ** 3
                dcost_dc = 6.0 * sf * c - 3.0 * sf ** 2
            cost += weight * local_cost
            sample_grad = -weight * dcost_dc * v
            grad[cols] += vals[:, None] * sample_grad[None, :]
    return cost, grad


def _smoothness_cost_value_grad(q: np.ndarray, *, smooth_weight: float, jerk_weight: float) -> tuple[float, np.ndarray]:
    cost = 0.0
    grad = np.zeros_like(q, dtype=np.float64)
    if q.shape[0] >= 3 and smooth_weight > 0.0:
        acc = q[2:] - 2.0 * q[1:-1] + q[:-2]
        cost += smooth_weight * float(np.sum(acc ** 2))
        scaled = 2.0 * smooth_weight * acc
        grad[:-2] += scaled
        grad[1:-1] -= 2.0 * scaled
        grad[2:] += scaled
    if q.shape[0] >= 4 and jerk_weight > 0.0:
        jerk = q[3:] - 3.0 * q[2:-1] + 3.0 * q[1:-2] - q[:-3]
        cost += jerk_weight * float(np.sum(jerk ** 2))
        scaled = 2.0 * jerk_weight * jerk
        grad[:-3] -= scaled
        grad[1:-2] += 3.0 * scaled
        grad[2:-1] -= 3.0 * scaled
        grad[3:] += scaled
    return cost, grad


def _fitness_cost_value_grad(
    q: np.ndarray,
    reference_q: np.ndarray,
    reference_tangents: np.ndarray,
    *,
    weight: float,
    axis_weight: float,
    radial_weight: float,
) -> tuple[float, np.ndarray]:
    if weight <= 0.0:
        return 0.0, np.zeros_like(q, dtype=np.float64)
    delta = q - reference_q
    tangents = reference_tangents
    axial = np.sum(delta * tangents, axis=1, keepdims=True)
    radial = delta - axial * tangents
    cost = weight * float(axis_weight * np.sum(axial ** 2) + radial_weight * np.sum(radial ** 2))
    grad = 2.0 * weight * (axis_weight * axial * tangents + radial_weight * radial)
    return cost, grad


def _feasibility_cost_value_grad(
    q: np.ndarray,
    *,
    dt: float,
    weight: float,
    max_velocity_mps: float | None,
    max_acceleration_mps2: float | None,
    max_jerk_mps3: float | None,
) -> tuple[float, np.ndarray]:
    if weight <= 0.0:
        return 0.0, np.zeros_like(q, dtype=np.float64)
    dt = max(float(dt), 1e-6)
    cost = 0.0
    grad = np.zeros_like(q, dtype=np.float64)

    def add_hinge(values: np.ndarray, coeffs: list[tuple[int, float]], limit: float | None) -> None:
        nonlocal cost, grad
        if limit is None or float(limit) <= 0.0:
            return
        limit_f = float(limit)
        excess_pos = values - limit_f
        excess_neg = -limit_f - values
        mask_pos = excess_pos > 0.0
        mask_neg = excess_neg > 0.0
        local_grad = np.zeros_like(values)
        if mask_pos.any():
            cost += weight * float(np.sum(excess_pos[mask_pos] ** 2))
            local_grad[mask_pos] += 2.0 * weight * excess_pos[mask_pos]
        if mask_neg.any():
            cost += weight * float(np.sum(excess_neg[mask_neg] ** 2))
            local_grad[mask_neg] -= 2.0 * weight * excess_neg[mask_neg]
        if not np.any(local_grad):
            return
        for offset, coeff in coeffs:
            grad[offset:offset + local_grad.shape[0]] += coeff * local_grad

    if q.shape[0] >= 2:
        vel = (q[1:] - q[:-1]) / dt
        add_hinge(vel, [(0, -1.0 / dt), (1, 1.0 / dt)], max_velocity_mps)
    if q.shape[0] >= 3:
        acc = (q[2:] - 2.0 * q[1:-1] + q[:-2]) / (dt ** 2)
        add_hinge(acc, [(0, 1.0 / dt ** 2), (1, -2.0 / dt ** 2), (2, 1.0 / dt ** 2)], max_acceleration_mps2)
    if q.shape[0] >= 4:
        jerk = (q[3:] - 3.0 * q[2:-1] + 3.0 * q[1:-2] - q[:-3]) / (dt ** 3)
        add_hinge(
            jerk,
            [(0, -1.0 / dt ** 3), (1, 3.0 / dt ** 3), (2, -3.0 / dt ** 3), (3, 1.0 / dt ** 3)],
            max_jerk_mps3,
        )
    return cost, grad


def _limit_exceed_ratio(
    q: np.ndarray,
    *,
    dt: float,
    max_velocity_mps: float | None,
    max_acceleration_mps2: float | None,
    max_jerk_mps3: float | None,
) -> float:
    dt = max(float(dt), 1e-6)
    ratio = 1.0
    if max_velocity_mps is not None and max_velocity_mps > 0.0 and q.shape[0] >= 2:
        vel = np.abs((q[1:] - q[:-1]) / dt)
        ratio = max(ratio, float(np.max(vel / float(max_velocity_mps))))
    if max_acceleration_mps2 is not None and max_acceleration_mps2 > 0.0 and q.shape[0] >= 3:
        acc = np.abs((q[2:] - 2.0 * q[1:-1] + q[:-2]) / (dt ** 2))
        ratio = max(ratio, float(np.sqrt(np.max(acc / float(max_acceleration_mps2)))))
    if max_jerk_mps3 is not None and max_jerk_mps3 > 0.0 and q.shape[0] >= 4:
        jerk = np.abs((q[3:] - 3.0 * q[2:-1] + 3.0 * q[1:-2] - q[:-3]) / (dt ** 3))
        ratio = max(ratio, float(np.cbrt(np.max(jerk / float(max_jerk_mps3)))))
    return max(1.0, ratio)


def _free_indices(num_control_points: int, pin_count: int) -> np.ndarray:
    pin = max(1, min(int(pin_count), max(1, num_control_points // 2 - 1)))
    if num_control_points <= 2 * pin:
        return np.arange(1, max(1, num_control_points - 1), dtype=np.int32)
    return np.arange(pin, num_control_points - pin, dtype=np.int32)


def _optimize_control_points(
    q_initial: np.ndarray,
    reference_q: np.ndarray,
    reference_tangents: np.ndarray,
    obstacle_pairs: list[list[tuple[np.ndarray, np.ndarray]]],
    sample_obstacle_pairs: list[tuple[int, np.ndarray, np.ndarray]],
    sample_basis: csr_array | None,
    grid: OccupancyGrid,
    config: EgoPlannerConfig,
    *,
    safety_clearance_voxels: float,
    dt: float,
) -> tuple[np.ndarray, Any]:
    q0 = np.asarray(q_initial, dtype=np.float64).reshape(-1, 3)
    free = _free_indices(q0.shape[0], config.pin_endpoint_control_points)
    if free.shape[0] == 0:
        return q0, None
    lo, hi = grid.world_bounds
    bounds = [(float(lo[axis]), float(hi[axis])) for _ in free for axis in range(3)]
    x0 = q0[free].reshape(-1)
    clearance_m = max(float(safety_clearance_voxels) * grid.voxel_size, grid.voxel_size)

    def unpack(x: np.ndarray) -> np.ndarray:
        q = q0.copy()
        q[free] = x.reshape(-1, 3)
        return q

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        q = unpack(x)
        total = 0.0
        grad = np.zeros_like(q, dtype=np.float64)

        value, local_grad = _smoothness_cost_value_grad(
            q,
            smooth_weight=float(config.smooth_weight),
            jerk_weight=float(config.jerk_weight),
        )
        total += value
        grad += local_grad

        value, local_grad = _fitness_cost_value_grad(
            q,
            reference_q,
            reference_tangents,
            weight=float(config.fitness_weight),
            axis_weight=float(config.fitness_axis_weight),
            radial_weight=float(config.fitness_radial_weight),
        )
        total += value
        grad += local_grad

        value, local_grad = _collision_cost_value_grad(
            q,
            obstacle_pairs,
            sample_obstacle_pairs,
            sample_basis,
            clearance_m=clearance_m,
            weight=float(config.collision_weight),
        )
        total += value
        grad += local_grad

        value, local_grad = _feasibility_cost_value_grad(
            q,
            dt=dt,
            weight=float(config.feasibility_weight),
            max_velocity_mps=config.max_velocity_mps,
            max_acceleration_mps2=config.max_acceleration_mps2,
            max_jerk_mps3=config.max_jerk_mps3,
        )
        total += value
        grad += local_grad
        return float(total), grad[free].reshape(-1)

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "maxiter": int(config.optimizer_maxiter),
            "ftol": float(config.optimizer_ftol),
            "gtol": float(config.optimizer_gtol),
            "maxls": 30,
        },
    )
    return unpack(result.x), result


def _detect_unsafe_control_indices(
    grid: OccupancyGrid,
    q: np.ndarray,
    *,
    safety_clearance_voxels: float,
    sample_spacing_m: float,
    sample_basis: csr_array | None = None,
) -> tuple[set[int], set[int], set[int], int, float]:
    direct_unsafe: set[int] = set()
    unsafe: set[int] = set()
    control_clearance = grid.clearance_voxels_for_points(q)
    for idx, clearance in enumerate(control_clearance):
        if clearance < safety_clearance_voxels:
            direct_unsafe.add(idx)
            unsafe.add(idx)

    if sample_basis is None:
        samples, _, basis = sample_control_spline(q, spacing_m=sample_spacing_m)
    else:
        basis = sample_basis
        samples = np.asarray(basis @ q, dtype=np.float32)
    sample_clearance = grid.clearance_voxels_for_points(samples)
    unsafe_sample_rows = np.flatnonzero(sample_clearance < safety_clearance_voxels)
    basis_csr = basis.tocsr()
    for row in unsafe_sample_rows:
        start, end = basis_csr.indptr[row], basis_csr.indptr[row + 1]
        cols = basis_csr.indices[start:end]
        vals = basis_csr.data[start:end]
        if cols.shape[0] == 0:
            continue
        unsafe.add(int(cols[int(np.argmax(vals))]))
        for col in cols[vals > 1e-3]:
            unsafe.add(int(col))
    min_clearance = float(np.min(sample_clearance)) if sample_clearance.shape[0] else float("inf")
    return direct_unsafe, unsafe, set(int(row) for row in unsafe_sample_rows), int(unsafe_sample_rows.shape[0]), min_clearance


def _add_obstacle_info(
    grid: OccupancyGrid,
    q: np.ndarray,
    obstacle_pairs: list[list[tuple[np.ndarray, np.ndarray]]],
    reference_path: np.ndarray,
    reference_tangents: np.ndarray,
    unsafe_indices: set[int],
    config: EgoPlannerConfig,
    *,
    safety_clearance_voxels: float,
) -> int:
    runs = _runs_from_indices(unsafe_indices, q.shape[0], expand=1)
    added = 0
    clearance_m = max(float(safety_clearance_voxels) * grid.voxel_size, grid.voxel_size)
    for begin, end in runs:
        start_idx = max(0, begin - 1)
        end_idx = min(q.shape[0] - 1, end + 1)
        guide = _slice_reference_guide(reference_path, q[start_idx], q[end_idx])
        if (
            config.guide_astar_fallback
            and not _is_polyline_safe(
                grid,
                guide,
                safety_clearance_voxels=safety_clearance_voxels,
                spacing_m=config.collision_sample_spacing_m,
            )
        ):
            try:
                guide = _local_astar_guide(
                    grid,
                    q[start_idx],
                    q[end_idx],
                    safety_clearance_voxels=safety_clearance_voxels,
                    margin_voxels=int(config.guide_astar_margin_voxels),
                    max_expansions=int(config.guide_astar_max_expansions),
                    time_limit_sec=float(config.guide_astar_time_limit_sec),
                    connectivity=int(config.connectivity),
                )
            except Exception:
                guide = _slice_reference_guide(reference_path, q[start_idx], q[end_idx])

        for ctrl_idx in range(begin, end + 1):
            if obstacle_pairs[ctrl_idx]:
                distances = [float(np.dot(q[ctrl_idx] - p, v)) for p, v in obstacle_pairs[ctrl_idx]]
                if min(distances) < clearance_m:
                    continue
            tangent = reference_tangents[ctrl_idx]
            p = _plane_intersection_or_nearest(q[ctrl_idx], tangent, guide)
            direction = p - q[ctrl_idx]
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-8:
                p = _nearest_segment_point(q[ctrl_idx], reference_path)
                direction = p - q[ctrl_idx]
                norm = float(np.linalg.norm(direction))
            if norm <= 1e-8:
                continue
            v = direction / norm
            duplicate = False
            for old_p, old_v in obstacle_pairs[ctrl_idx]:
                if np.linalg.norm(old_p - p) < 0.25 * grid.voxel_size and float(np.dot(old_v, v)) > 0.98:
                    duplicate = True
                    break
            if duplicate:
                continue
            obstacle_pairs[ctrl_idx].append((p.astype(np.float64), v.astype(np.float64)))
            added += 1
    return added


def _add_sample_obstacle_info(
    grid: OccupancyGrid,
    q: np.ndarray,
    sample_basis: csr_array,
    sample_obstacle_pairs: list[tuple[int, np.ndarray, np.ndarray]],
    reference_path: np.ndarray,
    unsafe_sample_rows: set[int],
    *,
    safety_clearance_voxels: float,
) -> int:
    if not unsafe_sample_rows:
        return 0
    samples = np.asarray(sample_basis @ q, dtype=np.float64)
    clearance_m = max(float(safety_clearance_voxels) * grid.voxel_size, grid.voxel_size)
    added = 0
    for row in sorted(unsafe_sample_rows):
        existing = [(p, v) for old_row, p, v in sample_obstacle_pairs if old_row == row]
        if existing:
            distances = [float(np.dot(samples[row] - p, v)) for p, v in existing]
            if min(distances) < clearance_m:
                continue
        p = _nearest_segment_point(samples[row], reference_path)
        direction = p - samples[row]
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            continue
        v = direction / norm
        duplicate = False
        for old_p, old_v in existing:
            if np.linalg.norm(old_p - p) < 0.25 * grid.voxel_size and float(np.dot(old_v, v)) > 0.98:
                duplicate = True
                break
        if duplicate:
            continue
        sample_obstacle_pairs.append((int(row), p.astype(np.float64), v.astype(np.float64)))
        added += 1
    return added


def validate_ego_path(
    grid: OccupancyGrid,
    path_world: np.ndarray,
    *,
    safety_clearance_voxels: float,
    sample_spacing_m: float,
) -> dict[str, Any]:
    dense = resample_polyline(path_world, max(float(sample_spacing_m), 1e-6))
    clearances = grid.clearance_voxels_for_points(dense)
    unsafe_mask = clearances < float(safety_clearance_voxels)
    return {
        "points": int(np.asarray(path_world).reshape(-1, 3).shape[0]),
        "dense_samples": int(dense.shape[0]),
        "length_m": path_length(path_world),
        "min_clearance_voxels": float(np.min(clearances)) if clearances.shape[0] else float("inf"),
        "min_clearance_m": (
            float(np.min(clearances) * grid.voxel_size)
            if clearances.shape[0]
            else float("inf")
        ),
        "unsafe_sample_count": int(np.count_nonzero(unsafe_mask)),
        "safety_clearance_voxels": float(safety_clearance_voxels),
    }


def ego_occupancy_path(
    grid: OccupancyGrid,
    coarse_path: CoarsePath | np.ndarray,
    config: EgoPlannerConfig | None = None,
) -> EgoPlannerResult:
    """Optimize a coarse path with an EGO-style ESDF-free local planner."""

    local_planning_started = time.perf_counter()
    cfg = config or EgoPlannerConfig()
    coarse = coarse_path if isinstance(coarse_path, CoarsePath) else CoarsePath(points=_validate_path_points(coarse_path))
    safety_clearance = cfg.resolved_safety_clearance_voxels(
        coarse,
        voxel_size_m=grid.voxel_size,
    )
    reference_path = resample_polyline(
        coarse.points,
        max(min(float(cfg.output_spacing_m), float(cfg.control_point_spacing_m)), grid.voxel_size),
    ).astype(np.float64)
    q_ref = resample_polyline(coarse.points, float(cfg.control_point_spacing_m)).astype(np.float64)
    if q_ref.shape[0] < 4:
        q_ref = resample_polyline(coarse.points, max(path_length(coarse.points) / 3.0, grid.voxel_size)).astype(np.float64)
    q = q_ref.copy()
    reference_tangents = _reference_tangents(q_ref)
    obstacle_pairs: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in range(q.shape[0])]
    sample_obstacle_pairs: list[tuple[int, np.ndarray, np.ndarray]] = []
    collision_sample_count = max(
        2,
        int(math.ceil(max(path_length(q_ref), grid.voxel_size) / max(float(cfg.collision_sample_spacing_m), 1e-6))) + 1,
    )
    _, collision_basis = _design_matrix(q.shape[0], collision_sample_count)
    dt = max(float(cfg.initial_dt), 1e-6)

    feasibility_ratio = _limit_exceed_ratio(
        q,
        dt=dt,
        max_velocity_mps=cfg.max_velocity_mps,
        max_acceleration_mps2=cfg.max_acceleration_mps2,
        max_jerk_mps3=cfg.max_jerk_mps3,
    )
    if feasibility_ratio > 1.0:
        dt *= feasibility_ratio

    optimize_results: list[Any] = []
    q, opt_result = _optimize_control_points(
        q,
        q_ref,
        reference_tangents,
        obstacle_pairs,
        sample_obstacle_pairs,
        collision_basis,
        grid,
        cfg,
        safety_clearance_voxels=safety_clearance,
        dt=dt,
    )
    optimize_results.append(opt_result)

    last_unsafe_count = 0
    min_clearance = float("inf")
    iterations = 0
    for iteration in range(max(0, int(cfg.rebound_iterations))):
        direct_unsafe, unsafe, unsafe_rows, unsafe_count, min_clearance = _detect_unsafe_control_indices(
            grid,
            q,
            safety_clearance_voxels=safety_clearance,
            sample_spacing_m=max(float(cfg.collision_sample_spacing_m), 1e-6),
            sample_basis=collision_basis,
        )
        last_unsafe_count = unsafe_count
        iterations = iteration + 1
        if not unsafe:
            break
        added_samples = _add_sample_obstacle_info(
            grid,
            q,
            collision_basis,
            sample_obstacle_pairs,
            reference_path,
            unsafe_rows,
            safety_clearance_voxels=safety_clearance,
        )
        added = _add_obstacle_info(
            grid,
            q,
            obstacle_pairs,
            reference_path,
            reference_tangents,
            direct_unsafe,
            cfg,
            safety_clearance_voxels=safety_clearance,
        )
        if added + added_samples == 0:
            cfg_local = EgoPlannerConfig(**cfg.__dict__)
            cfg_local.collision_weight = float(cfg.collision_weight) * 2.0
        else:
            cfg_local = cfg
        q, opt_result = _optimize_control_points(
            q,
            q_ref,
            reference_tangents,
            obstacle_pairs,
            sample_obstacle_pairs,
            collision_basis,
            grid,
            cfg_local,
            safety_clearance_voxels=safety_clearance,
            dt=dt,
        )
        optimize_results.append(opt_result)
    path_world, _, _ = sample_control_spline(q, spacing_m=max(float(cfg.output_spacing_m), 1e-6))
    path_world[0] = coarse.points[0]
    path_world[-1] = coarse.points[-1]
    local_planning_time = float(time.perf_counter() - local_planning_started)
    validation_started = time.perf_counter()
    validation = validate_ego_path(
        grid,
        path_world,
        safety_clearance_voxels=safety_clearance,
        sample_spacing_m=max(float(cfg.collision_sample_spacing_m), 1e-6),
    )
    last_unsafe_count = int(validation["unsafe_sample_count"])
    min_clearance = float(validation["min_clearance_voxels"])
    used_safety_fallback = False
    success = last_unsafe_count == 0
    message = "collision-free local EGO path" if success else "local EGO path still has unsafe samples"
    if not success and cfg.safe_fallback_to_input:
        fallback_validation = validate_ego_path(
            grid,
            coarse.points,
            safety_clearance_voxels=safety_clearance,
            sample_spacing_m=max(float(cfg.collision_sample_spacing_m), 1e-6),
        )
        if int(fallback_validation["unsafe_sample_count"]) == 0:
            path_world = coarse.points.astype(np.float32, copy=True)
            validation = fallback_validation
            last_unsafe_count = 0
            min_clearance = float(validation["min_clearance_voxels"])
            success = True
            used_safety_fallback = True
            message = "EGO smoothing unsafe; returned collision-free input path"
    planner_validation_time = float(time.perf_counter() - validation_started)
    pair_count = sum(len(pairs) for pairs in obstacle_pairs) + len(sample_obstacle_pairs)
    stats = {
        "optimizer_success": [bool(getattr(result, "success", True)) for result in optimize_results if result is not None],
        "optimizer_messages": [str(getattr(result, "message", "")) for result in optimize_results if result is not None],
        "control_points": int(q.shape[0]),
        "reference_points": int(reference_path.shape[0]),
        "treat_unknown_as_occupied": bool(grid.treat_unknown_as_occupied),
        "blocked_voxels": int(grid.blocked_idx.shape[0]),
        "used_safety_fallback": bool(used_safety_fallback),
        "safety_clearance_m": float(safety_clearance * grid.voxel_size),
        "min_clearance_m": float(min_clearance * grid.voxel_size),
        "output_spacing_m": float(cfg.output_spacing_m),
        "collision_sample_spacing_m": float(cfg.collision_sample_spacing_m),
        "timings_sec": {
            "local_planning": local_planning_time,
            "planner_validation": planner_validation_time,
        },
    }
    return EgoPlannerResult(
        path_world=path_world.astype(np.float32),
        control_points_world=q.astype(np.float32),
        reference_control_points_world=q_ref.astype(np.float32),
        safety_clearance_voxels=float(safety_clearance),
        safety_clearance_m=float(safety_clearance * grid.voxel_size),
        dt=float(dt),
        iterations=int(iterations),
        obstacle_pair_count=int(pair_count),
        unsafe_sample_count=last_unsafe_count,
        min_clearance_voxels=min_clearance,
        min_clearance_m=float(min_clearance * grid.voxel_size),
        length_m=float(validation["length_m"]),
        success=success,
        message=message,
        stats=stats,
    )


def _format_filename_value(value: float) -> str:
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p") or "0"


def default_output_path(input_path: str | Path, result: EgoPlannerResult, config: EgoPlannerConfig) -> Path:
    input_resolved = Path(input_path).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in input_resolved.stem).strip("_")
    if len(input_slug) > 72:
        input_slug = input_slug[:72].rstrip("_")
    filename = "_".join(
        [
            "ego",
            input_slug or "path",
            f"clr{_format_filename_value(result.safety_clearance_voxels)}",
            f"cp{_format_filename_value(config.control_point_spacing_m)}",
            f"it{int(config.rebound_iterations)}",
            timestamp,
        ]
    ) + ".txt"
    return input_resolved.parent / filename


def save_ego_path(
    output_path: str | Path,
    result: EgoPlannerResult,
    *,
    occupancy_map_path: str | Path,
    input_path: str | Path,
    config: EgoPlannerConfig,
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        stem = path.stem
        for i in range(2, 1000):
            candidate = path.with_name(f"{stem}_{i}{path.suffix}")
            if not candidate.exists():
                path = candidate
                break
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_clearance_value: float | None = None
    try:
        input_clearance_value = config.input_safety_clearance_voxels(read_coarse_path(input_path))
    except Exception:
        input_clearance_value = None
    header = [
        "NERFICG local path",
        f"method: {LOCAL_EGO_METHOD}",
        f"timestamp: {timestamp}",
        f"local_map: {Path(occupancy_map_path).expanduser().resolve()}",
        f"input_global_path: {Path(input_path).expanduser().resolve()}",
        f"points: {int(result.path_world.shape[0])}",
        f"control_points: {int(result.control_points_world.shape[0])}",
        f"safety_clearance_voxels: {result.safety_clearance_voxels:.3f}",
        f"safety_clearance_m: {result.safety_clearance_m:.6f}",
        f"input_safety_clearance_voxels: {input_clearance_value:.3f}" if input_clearance_value is not None else "input_safety_clearance_voxels: (unknown)",
        f"inherit_input_clearance: {bool(config.inherit_input_clearance)}",
        f"default_safety_clearance_voxels: {float(config.default_safety_clearance_voxels):.3f}",
        f"control_point_spacing_m: {float(config.control_point_spacing_m):.3f}",
        f"output_spacing_m: {float(config.output_spacing_m):.3f}",
        f"collision_sample_spacing_m: {float(config.collision_sample_spacing_m):.3f}",
        f"rebound_iterations: {int(result.iterations)}",
        f"obstacle_pairs: {int(result.obstacle_pair_count)}",
        f"min_clearance_voxels: {float(result.min_clearance_voxels):.3f}",
        f"min_clearance_m: {float(result.min_clearance_m):.6f}",
        f"unsafe_sample_count: {int(result.unsafe_sample_count)}",
        f"length_m: {float(result.length_m):.6f}",
        f"success: {bool(result.success)}",
        f"message: {result.message}",
        f"used_safety_fallback: {bool(result.stats.get('used_safety_fallback', False))}",
        "columns: x y z",
    ]
    np.savetxt(path, result.path_world, fmt="%.6f", delimiter=" ", header="\n".join(header), comments="# ")
    return path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EGO-style local planner for Occupancy Map voxel paths.")
    parser.add_argument("--map", required=True, type=Path, help="voxel_map.npz occupancy map.")
    parser.add_argument("--input-path", required=True, type=Path, help="Coarse global path .txt/.csv/.npy.")
    parser.add_argument("--output", type=Path, default=None, help="Output local path .txt. Defaults beside input path.")
    parser.add_argument("--safety-clearance-voxels", type=float, default=None, help="Explicit local EGO clearance override in voxels.")
    parser.add_argument("--safety-clearance-m", type=float, default=None, help="Explicit local EGO clearance in meters; converted using the map voxel size.")
    parser.add_argument(
        "--default-safety-clearance-voxels",
        type=float,
        default=2.5,
        help="Local EGO clearance used when no explicit override is passed.",
    )
    parser.add_argument(
        "--inherit-input-clearance",
        action="store_true",
        default=True,
        help="Use the coarse path clearance exactly (enabled by default).",
    )
    parser.add_argument(
        "--no-inherit-input-clearance",
        action="store_false",
        dest="inherit_input_clearance",
        help="Use --default-safety-clearance-voxels instead of the coarse path clearance.",
    )
    parser.add_argument("--control-point-spacing-m", type=float, default=0.30)
    parser.add_argument("--output-spacing-m", type=float, default=0.05)
    parser.add_argument("--collision-sample-spacing-m", type=float, default=0.05)
    parser.add_argument("--rebound-iterations", type=int, default=8)
    parser.add_argument("--optimizer-maxiter", type=int, default=80)
    parser.add_argument("--smooth-weight", type=float, default=1.0)
    parser.add_argument("--jerk-weight", type=float, default=0.1)
    parser.add_argument("--collision-weight", type=float, default=200.0)
    parser.add_argument("--fitness-weight", type=float, default=100.0)
    parser.add_argument("--fitness-axis-weight", type=float, default=0.05)
    parser.add_argument("--fitness-radial-weight", type=float, default=1.0)
    parser.add_argument("--pin-endpoint-control-points", type=int, default=2)
    parser.add_argument("--treat-unknown-as-occupied", action="store_true")
    parser.add_argument("--no-guide-astar-fallback", action="store_true")
    parser.add_argument("--guide-astar-margin-voxels", type=int, default=32)
    parser.add_argument("--guide-astar-time-limit-sec", type=float, default=0.25)
    parser.add_argument("--max-velocity-mps", type=float, default=None)
    parser.add_argument("--max-acceleration-mps2", type=float, default=None)
    parser.add_argument("--max-jerk-mps3", type=float, default=None)
    parser.add_argument("--feasibility-weight", type=float, default=0.0)
    parser.add_argument("--initial-dt", type=float, default=0.20)
    parser.add_argument("--disable-safe-fallback", action="store_true")
    parser.add_argument("--verify-only", action="store_true", help="Only validate the input path against the map.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    coarse = read_coarse_path(args.input_path)
    config = EgoPlannerConfig(
        control_point_spacing_m=float(args.control_point_spacing_m),
        output_spacing_m=float(args.output_spacing_m),
        collision_sample_spacing_m=float(args.collision_sample_spacing_m),
        safety_clearance_m=args.safety_clearance_m,
        safety_clearance_voxels=args.safety_clearance_voxels,
        default_safety_clearance_voxels=float(args.default_safety_clearance_voxels),
        inherit_input_clearance=bool(args.inherit_input_clearance),
        treat_unknown_as_occupied=bool(args.treat_unknown_as_occupied),
        pin_endpoint_control_points=int(args.pin_endpoint_control_points),
        rebound_iterations=int(args.rebound_iterations),
        smooth_weight=float(args.smooth_weight),
        jerk_weight=float(args.jerk_weight),
        collision_weight=float(args.collision_weight),
        fitness_weight=float(args.fitness_weight),
        fitness_axis_weight=float(args.fitness_axis_weight),
        fitness_radial_weight=float(args.fitness_radial_weight),
        optimizer_maxiter=int(args.optimizer_maxiter),
        guide_astar_fallback=not bool(args.no_guide_astar_fallback),
        guide_astar_margin_voxels=int(args.guide_astar_margin_voxels),
        guide_astar_time_limit_sec=float(args.guide_astar_time_limit_sec),
        max_velocity_mps=args.max_velocity_mps,
        max_acceleration_mps2=args.max_acceleration_mps2,
        max_jerk_mps3=args.max_jerk_mps3,
        feasibility_weight=float(args.feasibility_weight),
        initial_dt=float(args.initial_dt),
        safe_fallback_to_input=not bool(args.disable_safe_fallback),
    )
    grid = load_occupancy_grid(args.map, treat_unknown_as_occupied=config.treat_unknown_as_occupied)
    safety = config.resolved_safety_clearance_voxels(coarse, voxel_size_m=grid.voxel_size)
    if args.verify_only:
        validation = validate_ego_path(
            grid,
            coarse.points,
            safety_clearance_voxels=safety,
            sample_spacing_m=max(float(config.collision_sample_spacing_m), 1e-6),
        )
        print(
            "verify input: "
            f"points={validation['points']} "
            f"length_m={validation['length_m']:.6f} "
            f"min_clearance_voxels={validation['min_clearance_voxels']:.3f} "
            f"unsafe_sample_count={validation['unsafe_sample_count']} "
            f"safety_clearance_voxels={safety:.3f}"
        )
        return 0 if validation["unsafe_sample_count"] == 0 else 2

    result = ego_occupancy_path(grid, coarse, config)
    output = args.output if args.output is not None else default_output_path(args.input_path, result, config)
    saved = save_ego_path(output, result, occupancy_map_path=args.map, input_path=args.input_path, config=config)
    print(
        f"saved {saved}\n"
        f"success={result.success} message={result.message}\n"
        f"points={result.path_world.shape[0]} control_points={result.control_points_world.shape[0]} "
        f"length_m={result.length_m:.6f} min_clearance_voxels={result.min_clearance_voxels:.3f} "
        f"unsafe_sample_count={result.unsafe_sample_count} obstacle_pairs={result.obstacle_pair_count}"
    )
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
