#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Occupancy Map global path planning with voxel A*."""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from GSPAR.mapping.voxel_map_cli import resolve_existing_path


PLANNER_MAP_TYPE = "Occupancy Map"


@dataclass
class VoxelGridGraph:
    """Voxel graph backed by an Occupancy Map."""

    origin: np.ndarray
    voxel_size: float
    voxel_dims: np.ndarray
    occupied_idx: np.ndarray

    def __post_init__(self) -> None:
        self.origin = np.asarray(self.origin, dtype=np.float32).reshape(3)
        self.voxel_size = float(self.voxel_size)
        self.voxel_dims = np.asarray(self.voxel_dims, dtype=np.int32).reshape(3)
        self.occupied_idx = np.asarray(self.occupied_idx, dtype=np.int32).reshape(-1, 3)
        self._dims64 = self.voxel_dims.astype(np.int64)
        self._occupied_keys = set(self._pack(self.occupied_idx).tolist())
        self._occupied_tree = cKDTree(self.occupied_idx.astype(np.float32)) if self.occupied_idx.shape[0] else None
        self._clearance_cache: dict[int, float] = {}
        self._safety_cache: dict[tuple[int, float], bool] = {}

    def _pack(self, idx: np.ndarray) -> np.ndarray:
        idx64 = np.asarray(idx, dtype=np.int64).reshape(-1, 3)
        return idx64[:, 0] + self._dims64[0] * (idx64[:, 1] + self._dims64[1] * idx64[:, 2])

    def is_valid_idx(self, idx: np.ndarray) -> bool:
        idx = np.asarray(idx, dtype=np.int32).reshape(3)
        return bool(np.logical_and(idx >= 0, idx < self.voxel_dims).all())

    def world_to_voxel_idx(self, point_world: np.ndarray) -> np.ndarray:
        point_world = np.asarray(point_world, dtype=np.float32).reshape(3)
        idx = np.floor((point_world - self.origin) / self.voxel_size).astype(np.int32)
        return np.clip(idx, 0, self.voxel_dims - 1)

    def voxel_idx_to_world_center(self, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.float32).reshape(-1, 3)
        return self.origin[None, :] + (idx + 0.5) * self.voxel_size

    def is_occupied_idx(self, idx: np.ndarray) -> bool:
        if not self.is_valid_idx(idx):
            return True
        key = int(self._pack(np.asarray(idx, dtype=np.int32).reshape(1, 3))[0])
        return key in self._occupied_keys

    def clearance_to_occupied(self, idx: np.ndarray) -> float:
        idx = np.asarray(idx, dtype=np.int32).reshape(3)
        if not self.is_valid_idx(idx):
            return 0.0
        if self._occupied_tree is None:
            return float("inf")
        key = int(self._pack(idx.reshape(1, 3))[0])
        cached = self._clearance_cache.get(key)
        if cached is not None:
            return cached
        distance = float(self._occupied_tree.query(idx.astype(np.float32), k=1)[0])
        self._clearance_cache[key] = distance
        return distance

    def is_safe_idx(self, idx: np.ndarray, safety_clearance_voxels: float = 0.0) -> bool:
        if self.is_occupied_idx(idx):
            return False
        safety_clearance_voxels = float(safety_clearance_voxels)
        if safety_clearance_voxels <= 0.0:
            return True

        idx = np.asarray(idx, dtype=np.int32).reshape(3)
        key = int(self._pack(idx.reshape(1, 3))[0])
        cached_clearance = self._clearance_cache.get(key)
        if cached_clearance is not None:
            return cached_clearance >= safety_clearance_voxels

        cache_key = (key, safety_clearance_voxels)
        cached_safe = self._safety_cache.get(cache_key)
        if cached_safe is not None:
            return cached_safe
        if self._occupied_tree is None:
            self._safety_cache[cache_key] = True
            return True

        query_radius = float(np.nextafter(safety_clearance_voxels, 0.0))
        distance = float(self._occupied_tree.query(idx.astype(np.float32), k=1, distance_upper_bound=query_radius)[0])
        safe = not np.isfinite(distance)
        self._safety_cache[cache_key] = safe
        return safe

    def snap_to_safe_idx(
        self,
        point_world: np.ndarray,
        max_radius: int = 8,
        safety_clearance_voxels: float = 0.0,
    ) -> np.ndarray:
        target = self.world_to_voxel_idx(point_world)
        if self.is_safe_idx(target, safety_clearance_voxels=safety_clearance_voxels):
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
                        if not self.is_safe_idx(idx, safety_clearance_voxels=safety_clearance_voxels):
                            continue
                        dist2 = float(np.sum((idx.astype(np.float32) - target.astype(np.float32)) ** 2))
                        if dist2 < best_dist2:
                            best_idx = idx
                            best_dist2 = dist2
            if best_idx is not None:
                return best_idx

        raise ValueError(
            f"no safe voxel within radius {int(max_radius)} "
            f"and clearance {float(safety_clearance_voxels):.3f}"
        )


def load_voxel_graph(npz_path: str | Path) -> VoxelGridGraph:
    """Load an Occupancy Map .npz file into a voxel graph."""

    path = resolve_existing_path(npz_path, default_suffix=".npz")
    if not path.is_file():
        raise FileNotFoundError(f"occupancy file not found: {path}")
    if path.suffix.lower() != ".npz":
        raise ValueError("Occupancy Map planner requires a .npz occupancy file")

    data = np.load(path)
    required = ("origin", "voxel_size", "voxel_dims", "occupied_voxel_idx")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"occupancy npz missing keys: {missing}")

    return VoxelGridGraph(
        origin=np.asarray(data["origin"], dtype=np.float32),
        voxel_size=float(np.asarray(data["voxel_size"]).reshape(-1)[0]),
        voxel_dims=np.asarray(data["voxel_dims"], dtype=np.int32),
        occupied_idx=np.asarray(data["occupied_voxel_idx"], dtype=np.int32),
    )


def _neighbor_offsets(connectivity: int) -> tuple[np.ndarray, np.ndarray]:
    if connectivity not in (6, 18, 26):
        raise ValueError("connectivity must be 6, 18, or 26")

    offsets = []
    costs = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                l1 = abs(dx) + abs(dy) + abs(dz)
                if connectivity == 6 and l1 != 1:
                    continue
                if connectivity == 18 and l1 > 2:
                    continue
                offset = np.array([dx, dy, dz], dtype=np.int32)
                offsets.append(offset)
                costs.append(float(np.linalg.norm(offset.astype(np.float32))))

    return np.stack(offsets, axis=0), np.asarray(costs, dtype=np.float32)


def _grid_distance_heuristic(idx: np.ndarray, goal_idx: np.ndarray, connectivity: int) -> float:
    d = np.abs(
        np.asarray(idx, dtype=np.float32).reshape(3)
        - np.asarray(goal_idx, dtype=np.float32).reshape(3)
    )

    if connectivity == 6:
        return float(d.sum())

    if connectivity == 18:
        total = float(d.sum())
        max_axis = float(d.max())
        diagonal_steps = min(np.floor(total * 0.5), total - max_axis)
        axis_steps = total - 2.0 * diagonal_steps
        return float(axis_steps + diagonal_steps * np.sqrt(2.0))

    if connectivity == 26:
        d_sorted = np.sort(d)
        diagonal3 = float(d_sorted[0])
        diagonal2 = float(d_sorted[1] - d_sorted[0])
        axis_steps = float(d_sorted[2] - d_sorted[1])
        return float(axis_steps + diagonal2 * np.sqrt(2.0) + diagonal3 * np.sqrt(3.0))

    raise ValueError("connectivity must be 6, 18, or 26")


def astar_voxel_indices(
    graph: VoxelGridGraph,
    start_world: np.ndarray,
    goal_world: np.ndarray,
    *,
    connectivity: int = 26,
    max_expansions: int = 100000,
    snap_search_radius: int = 1,
    safety_clearance_voxels: float = 2.5,
    heuristic_weight: float = 10.0,
    max_planning_time_sec: float | None = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plan on an Occupancy Map and return voxel indices."""

    offsets, offset_costs = _neighbor_offsets(connectivity)
    start_idx = graph.snap_to_safe_idx(start_world, snap_search_radius, safety_clearance_voxels)
    goal_idx = graph.snap_to_safe_idx(goal_world, snap_search_radius, safety_clearance_voxels)
    heuristic_weight = max(1.0, float(heuristic_weight))

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
            raise RuntimeError(f"A* exceeded max planning time ({float(max_planning_time_sec):.3f}s)")
        if expansions > int(max_expansions):
            raise RuntimeError(f"A* exceeded max expansions ({int(max_expansions)})")
        closed.add(current_key)

        neighbor_idx = current_idx[None, :] + offsets
        valid = np.logical_and(neighbor_idx >= 0, neighbor_idx < graph.voxel_dims[None, :]).all(axis=1)
        neighbor_idx = neighbor_idx[valid]
        neighbor_costs = offset_costs[valid]

        for n_idx, n_cost in zip(neighbor_idx, neighbor_costs):
            if not graph.is_safe_idx(n_idx, safety_clearance_voxels=safety_clearance_voxels):
                continue
            n_key = int(graph._pack(n_idx.reshape(1, 3))[0])
            if n_key in closed:
                continue
            tentative_g = current_g + float(n_cost)
            if tentative_g >= g_score.get(n_key, float("inf")):
                continue

            g_score[n_key] = tentative_g
            parent[n_key] = current_key
            key_to_idx[n_key] = n_idx.astype(np.int32, copy=True)
            h = heuristic_weight * _grid_distance_heuristic(n_idx, goal_f, connectivity)
            heapq.heappush(open_heap, (tentative_g + h, tentative_g, n_key))
    else:
        raise RuntimeError("A* could not find a path between the selected points")

    path_keys = [goal_key]
    while path_keys[-1] != start_key:
        prev = parent.get(path_keys[-1])
        if prev is None:
            raise RuntimeError("A* path reconstruction failed")
        path_keys.append(prev)
    path_keys.reverse()

    path_idx = np.stack([key_to_idx[key] for key in path_keys], axis=0).astype(np.int32)
    return path_idx, start_idx.astype(np.int32), goal_idx.astype(np.int32)


def astar_world_path(
    graph: VoxelGridGraph,
    start_world: np.ndarray,
    goal_world: np.ndarray,
    *,
    connectivity: int = 26,
    max_expansions: int = 100000,
    snap_search_radius: int = 1,
    safety_clearance_voxels: float = 2.5,
    heuristic_weight: float = 10.0,
    max_planning_time_sec: float | None = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plan a global path in world coordinates from an Occupancy Map."""

    path_idx, start_idx, goal_idx = astar_voxel_indices(
        graph,
        start_world,
        goal_world,
        connectivity=connectivity,
        max_expansions=max_expansions,
        snap_search_radius=snap_search_radius,
        safety_clearance_voxels=safety_clearance_voxels,
        heuristic_weight=heuristic_weight,
        max_planning_time_sec=max_planning_time_sec,
    )
    path_world = graph.voxel_idx_to_world_center(path_idx).astype(np.float32)
    return path_world, start_idx, goal_idx
