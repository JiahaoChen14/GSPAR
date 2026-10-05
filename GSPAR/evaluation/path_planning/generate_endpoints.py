#!/usr/bin/env python3
"""Generate connected, reference-valid endpoint pairs without running a planner."""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from GSPAR.path.global_path.esdf_astar_world_path import (
    load_aligned_esdf_samples,
    resolve_esdf_path,
)
from .metrics import VoxelReference


SEED = 0
CONNECTED_COMPONENT_CONNECTIVITY = 26
CONNECTED_COMPONENT_SLAB_DEPTH = 32
CONNECTED_COMPONENT_TARGET_WORKING_BYTES = 96 * 1024 * 1024
MAX_GOAL_ATTEMPTS_PER_START = 50
MAX_START_ATTEMPTS_PER_PAIR = 200
MIN_ENDPOINT_DISTANCE_VOXELS = 3.0
REFERENCE_SAFETY_MARGIN_M = 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-map",
        required=True,
        type=Path,
        help="Aligned TSDF or converted GT voxel reference NPZ",
    )
    parser.add_argument(
        "--esdf-map",
        type=Path,
        default=None,
        help=(
            "nvblox ESDF PLY, sdf directory, or scene directory; inferred from "
            "the reference-map directory by default"
        ),
    )
    parser.add_argument(
        "--esdf-grid-metadata",
        type=Path,
        default=None,
        help=(
            "Existing TSDF voxel NPZ/JSON used to align ESDF samples; defaults "
            "to --reference-map, or its sibling voxel_map_tsdf for GT"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=100,
        help="Number of endpoint pairs to generate (default: 100)",
    )
    parser.add_argument("--robot-radius-m", type=float, default=0.08)
    parser.add_argument(
        "--min-separation-m",
        type=float,
        default=None,
        help=(
            "Optional additional inclusive minimum separation. Every pair is "
            "always strictly farther apart than three voxels."
        ),
    )
    parser.add_argument(
        "--max-separation-m",
        type=float,
        default=None,
        help="Optional inclusive maximum start-goal distance in metres",
    )
    return parser


def _resolve_npz_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() and resolved.suffix == "":
        resolved = resolved.with_suffix(".npz")
    if not resolved.is_file():
        raise FileNotFoundError(f"voxel map not found: {resolved}")
    if resolved.suffix.lower() != ".npz":
        raise ValueError(f"voxel reference must be an NPZ file: {resolved}")
    return resolved


def _load_grid_metadata(path: str | Path) -> tuple[Path, np.ndarray, float, np.ndarray]:
    resolved = _resolve_npz_path(path)
    with np.load(resolved) as data:
        required = ("origin", "voxel_size", "voxel_dims")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"voxel map missing grid metadata: {missing}")
        origin = np.asarray(data["origin"], dtype=np.float64).reshape(3)
        voxel_size = float(np.asarray(data["voxel_size"]).reshape(-1)[0])
        voxel_dims = np.asarray(data["voxel_dims"], dtype=np.int32).reshape(3)
    if not np.isfinite(origin).all() or not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("voxel grid origin and voxel_size must be finite and valid")
    if np.any(voxel_dims <= 0):
        raise ValueError("voxel_dims must be positive")
    return resolved, origin, voxel_size, voxel_dims


def validate_separation_bounds(
    min_separation_m: float | None,
    max_separation_m: float | None,
) -> tuple[float | None, float | None]:
    """Validate optional user-specified endpoint-separation bounds."""

    minimum = None if min_separation_m is None else float(min_separation_m)
    if minimum is not None and (not np.isfinite(minimum) or minimum < 0.0):
        raise ValueError("min_separation_m must be finite and non-negative")
    maximum = None if max_separation_m is None else float(max_separation_m)
    if maximum is not None and (not np.isfinite(maximum) or maximum <= 0.0):
        raise ValueError("max_separation_m must be finite and positive")
    if minimum is not None and maximum is not None and maximum < minimum:
        raise ValueError("max_separation_m must be greater than or equal to min_separation_m")
    return minimum, maximum


def _pack_voxel_indices(voxel_indices: np.ndarray, voxel_dims: np.ndarray) -> np.ndarray:
    indices = np.asarray(voxel_indices, dtype=np.int64).reshape(-1, 3)
    dims = np.asarray(voxel_dims, dtype=np.int64).reshape(3)
    return indices[:, 0] + dims[0] * (indices[:, 1] + dims[1] * indices[:, 2])


def _unpack_voxel_keys(keys: np.ndarray, voxel_dims: np.ndarray) -> np.ndarray:
    packed = np.asarray(keys, dtype=np.int64).reshape(-1)
    dims = np.asarray(voxel_dims, dtype=np.int64).reshape(3)
    indices = np.empty((packed.shape[0], 3), dtype=np.int32)
    indices[:, 0] = (packed % dims[0]).astype(np.int32)
    yz = packed // dims[0]
    indices[:, 1] = (yz % dims[1]).astype(np.int32)
    indices[:, 2] = (yz // dims[1]).astype(np.int32)
    return indices


def _voxel_centers(
    voxel_indices: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    indices = np.asarray(voxel_indices, dtype=np.float64).reshape(-1, 3)
    return (
        np.asarray(origin, dtype=np.float64).reshape(1, 3)
        + (indices + 0.5) * float(voxel_size)
    )


def _component_edges_across_slab_boundary(
    previous_plane: np.ndarray,
    current_plane: np.ndarray,
) -> np.ndarray:
    """Return unique provisional-component edges for all 9 cross-plane moves."""

    height, width = previous_plane.shape
    edge_codes: list[np.ndarray] = []
    for delta_y in (-1, 0, 1):
        if delta_y >= 0:
            previous_y = slice(0, height - delta_y)
            current_y = slice(delta_y, height)
        else:
            previous_y = slice(-delta_y, height)
            current_y = slice(0, height + delta_y)
        for delta_x in (-1, 0, 1):
            if delta_x >= 0:
                previous_x = slice(0, width - delta_x)
                current_x = slice(delta_x, width)
            else:
                previous_x = slice(-delta_x, width)
                current_x = slice(0, width + delta_x)
            left = previous_plane[previous_y, previous_x]
            right = current_plane[current_y, current_x]
            valid = np.logical_and(left > 0, right > 0)
            if not valid.any():
                continue
            codes = (
                left[valid].astype(np.uint64) << np.uint64(32)
            ) | right[valid].astype(np.uint64)
            edge_codes.append(np.unique(codes))
    if not edge_codes:
        return np.zeros((0,), dtype=np.uint64)
    return np.unique(np.concatenate(edge_codes))


def _component_roots(parent: np.ndarray, component_ids: np.ndarray) -> np.ndarray:
    roots = np.asarray(component_ids, dtype=np.int32).reshape(-1).copy()
    while roots.shape[0]:
        next_roots = parent[roots]
        if np.array_equal(next_roots, roots):
            break
        roots = next_roots
    return roots


def _merge_component_edges(parent: np.ndarray, edge_codes: np.ndarray) -> None:
    """Merge one slab boundary immediately so edge storage stays bounded."""

    if edge_codes.shape[0] == 0:
        return
    left = ((edge_codes >> np.uint64(32)).astype(np.int64) - 1).astype(np.int32)
    right = (
        (edge_codes & np.uint64(0xFFFFFFFF)).astype(np.int64) - 1
    ).astype(np.int32)
    left = _component_roots(parent, left)
    right = _component_roots(parent, right)
    distinct = left != right
    if not distinct.any():
        return
    left = left[distinct]
    right = right[distinct]
    nodes, inverse = np.unique(
        np.concatenate([left, right]),
        return_inverse=True,
    )
    edge_count = left.shape[0]
    adjacency = coo_matrix(
        (
            np.ones((edge_count,), dtype=np.uint8),
            (inverse[:edge_count], inverse[edge_count:]),
        ),
        shape=(nodes.shape[0], nodes.shape[0]),
    )
    count, labels = connected_components(
        adjacency,
        directed=False,
        return_labels=True,
    )
    canonical = np.full((count,), np.iinfo(np.int32).max, dtype=np.int32)
    np.minimum.at(canonical, labels, nodes)
    parent[nodes] = canonical[labels]


def _label_sorted_voxel_keys(
    sorted_keys: np.ndarray,
    voxel_dims: np.ndarray,
    *,
    slab_depth: int,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray, int]:
    """Label a sparse 3D mask with bounded-memory z-slab 26-connectivity CCL."""

    keys = np.asarray(sorted_keys, dtype=np.int64).reshape(-1)
    dims = np.asarray(voxel_dims, dtype=np.int64).reshape(3)
    if keys.shape[0] == 0:
        raise RuntimeError("no traversable voxels are available")

    bbox_min = np.full((3,), np.iinfo(np.int32).max, dtype=np.int32)
    bbox_max = np.full((3,), np.iinfo(np.int32).min, dtype=np.int32)
    batch_size = 2_000_000
    for begin in range(0, keys.shape[0], batch_size):
        indices = _unpack_voxel_keys(keys[begin : begin + batch_size], dims)
        bbox_min = np.minimum(bbox_min, indices.min(axis=0))
        bbox_max = np.maximum(bbox_max, indices.max(axis=0))

    width = int(bbox_max[0] - bbox_min[0] + 1)
    height = int(bbox_max[1] - bbox_min[1] + 1)
    plane_voxels = width * height
    target_cells = max(1, CONNECTED_COMPONENT_TARGET_WORKING_BYTES // 5)
    effective_depth = max(
        1,
        min(max(1, int(slab_depth)), max(1, target_cells // max(1, plane_voxels))),
    )
    z_begin = int(bbox_min[2])
    z_end = int(bbox_max[2]) + 1
    slab_count = int(np.ceil((z_end - z_begin) / effective_depth))
    provisional_ids = np.empty((keys.shape[0],), dtype=np.int32)
    parent = np.empty((keys.shape[0],), dtype=np.int32)
    structure = np.ones((3, 3, 3), dtype=bool)
    previous_last_plane = np.zeros((height, width), dtype=np.int32)
    provisional_count = 0
    xy_stride = int(dims[0] * dims[1])

    for slab_z_begin in range(z_begin, z_end, effective_depth):
        slab_z_end = min(z_end, slab_z_begin + effective_depth)
        row_begin = int(np.searchsorted(keys, slab_z_begin * xy_stride, side="left"))
        row_end = int(np.searchsorted(keys, slab_z_end * xy_stride, side="left"))
        slab_keys = keys[row_begin:row_end]
        depth = slab_z_end - slab_z_begin
        mask = np.zeros((depth, height, width), dtype=bool)
        if slab_keys.shape[0]:
            indices = _unpack_voxel_keys(slab_keys, dims)
            local_z = indices[:, 2].astype(np.int64) - slab_z_begin
            local_y = indices[:, 1].astype(np.int64) - int(bbox_min[1])
            local_x = indices[:, 0].astype(np.int64) - int(bbox_min[0])
            mask[local_z, local_y, local_x] = True
        else:
            local_z = np.zeros((0,), dtype=np.int64)
            local_y = np.zeros((0,), dtype=np.int64)
            local_x = np.zeros((0,), dtype=np.int64)

        local_labels = np.empty(mask.shape, dtype=np.int32)
        local_count = int(ndimage.label(mask, structure=structure, output=local_labels))
        if provisional_count + local_count >= np.iinfo(np.int32).max:
            raise RuntimeError("too many provisional connected components for int32 labels")
        parent[provisional_count : provisional_count + local_count] = np.arange(
            provisional_count,
            provisional_count + local_count,
            dtype=np.int32,
        )
        if slab_keys.shape[0]:
            provisional_ids[row_begin:row_end] = (
                local_labels[local_z, local_y, local_x] + provisional_count
            )

        current_first_plane = local_labels[0].copy()
        current_last_plane = local_labels[-1].copy()
        for plane in (current_first_plane, current_last_plane):
            nonzero = plane > 0
            plane[nonzero] += provisional_count
        edges = _component_edges_across_slab_boundary(
            previous_last_plane,
            current_first_plane,
        )
        _merge_component_edges(parent, edges)
        previous_last_plane = current_last_plane
        provisional_count += local_count
        if slab_keys.shape[0]:
            del indices
        del (
            current_first_plane,
            edges,
            local_labels,
            local_x,
            local_y,
            local_z,
            mask,
        )

    if provisional_count <= 0:
        raise RuntimeError("connected-component labeling found no traversable voxels")
    for begin in range(0, provisional_ids.shape[0], batch_size):
        values = provisional_ids[begin : begin + batch_size]
        values -= 1
        while values.shape[0]:
            next_values = parent[values]
            if np.array_equal(next_values, values):
                break
            values[:] = next_values
    unique_roots = np.unique(provisional_ids)
    component_count = int(unique_roots.shape[0])
    for begin in range(0, provisional_ids.shape[0], batch_size):
        values = provisional_ids[begin : begin + batch_size]
        values[:] = np.searchsorted(unique_roots, values) + 1
    return (
        provisional_ids,
        component_count,
        bbox_min,
        bbox_max,
        slab_count,
    )


@dataclass
class ConnectedComponentCache:
    """Sorted traversable voxel keys and their precomputed component IDs."""

    sorted_keys: np.ndarray
    component_ids: np.ndarray
    voxel_dims: np.ndarray
    component_count: int
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    slab_count: int

    def __post_init__(self) -> None:
        self.sorted_keys = np.asarray(self.sorted_keys, dtype=np.int64).reshape(-1)
        self.component_ids = np.asarray(self.component_ids, dtype=np.int32).reshape(-1)
        self.voxel_dims = np.asarray(self.voxel_dims, dtype=np.int32).reshape(3)
        self.bbox_min = np.asarray(self.bbox_min, dtype=np.int32).reshape(3)
        self.bbox_max = np.asarray(self.bbox_max, dtype=np.int32).reshape(3)
        if self.sorted_keys.shape[0] != self.component_ids.shape[0]:
            raise ValueError("sorted_keys and component_ids must have matching lengths")

    @classmethod
    def from_sorted_keys(
        cls,
        sorted_keys: np.ndarray,
        voxel_dims: np.ndarray,
        *,
        slab_depth: int = CONNECTED_COMPONENT_SLAB_DEPTH,
        assume_sorted_unique: bool = False,
    ) -> "ConnectedComponentCache":
        keys = np.asarray(sorted_keys, dtype=np.int64).reshape(-1)
        if not assume_sorted_unique:
            keys = np.unique(keys)
        else:
            batch_size = 2_000_000
            previous = None
            for begin in range(0, keys.shape[0], batch_size):
                values = keys[begin : begin + batch_size]
                if values.shape[0] == 0:
                    continue
                if previous is not None and int(values[0]) <= previous:
                    raise ValueError("assumed sorted voxel keys are not strictly increasing")
                if values.shape[0] > 1 and np.any(values[1:] <= values[:-1]):
                    raise ValueError("assumed sorted voxel keys are not strictly increasing")
                previous = int(values[-1])
        labels, count, bbox_min, bbox_max, slab_count = _label_sorted_voxel_keys(
            keys,
            voxel_dims,
            slab_depth=slab_depth,
        )
        return cls(
            sorted_keys=keys,
            component_ids=labels,
            voxel_dims=voxel_dims,
            component_count=count,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            slab_count=slab_count,
        )

    @classmethod
    def from_voxel_indices(
        cls,
        voxel_indices: np.ndarray,
        voxel_dims: np.ndarray,
        *,
        slab_depth: int = CONNECTED_COMPONENT_SLAB_DEPTH,
    ) -> "ConnectedComponentCache":
        indices = np.asarray(voxel_indices, dtype=np.int32).reshape(-1, 3)
        dims = np.asarray(voxel_dims, dtype=np.int32).reshape(3)
        valid = np.logical_and(indices >= 0, indices < dims[None, :]).all(axis=1)
        if not valid.all():
            raise ValueError("traversable voxel indices must lie inside voxel_dims")
        return cls.from_sorted_keys(
            _pack_voxel_indices(indices, dims),
            dims,
            slab_depth=slab_depth,
        )

    @property
    def voxel_indices(self) -> np.ndarray:
        return _unpack_voxel_keys(self.sorted_keys, self.voxel_dims)

    def voxel_idx_at(self, row: int) -> np.ndarray:
        return _unpack_voxel_keys(
            self.sorted_keys[int(row) : int(row) + 1],
            self.voxel_dims,
        )[0]

    def component_id_for_idx(self, voxel_idx: np.ndarray) -> int:
        index = np.asarray(voxel_idx, dtype=np.int64).reshape(3)
        if not np.logical_and(index >= 0, index < self.voxel_dims.astype(np.int64)).all():
            return 0
        key = int(_pack_voxel_indices(index.reshape(1, 3), self.voxel_dims)[0])
        position = int(np.searchsorted(self.sorted_keys, key))
        if position >= self.sorted_keys.shape[0] or int(self.sorted_keys[position]) != key:
            return 0
        return int(self.component_ids[position])


def ground_truth_reference_valid_mask(
    voxel_indices: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    voxel_dims: np.ndarray,
    reference: VoxelReference,
    robot_radius_m: float,
) -> np.ndarray:
    """Match final GT collision and common-grid-boundary semantics at voxel centers."""

    indices = np.asarray(voxel_indices, dtype=np.int32).reshape(-1, 3)
    points = _voxel_centers(indices, origin, voxel_size)
    radius = float(robot_radius_m)
    distances, in_reference = reference.obstacle_distances(points)
    world_min = np.asarray(origin, dtype=np.float64).reshape(3)
    world_max = world_min + np.asarray(voxel_dims, dtype=np.float64) * float(voxel_size)
    face_distances = np.minimum(
        points - world_min[None, :],
        world_max[None, :] - points,
    )
    footprint_in_bounds = np.all(
        face_distances * face_distances > radius * radius + 1.0e-12,
        axis=1,
    )
    return np.logical_and.reduce(
        (
            in_reference,
            footprint_in_bounds,
            distances > radius + REFERENCE_SAFETY_MARGIN_M + 1.0e-9,
        )
    )


def _collect_traversable_keys(
    esdf_idx: np.ndarray,
    esdf_free: np.ndarray,
    voxel_dims: np.ndarray,
    *,
    origin: np.ndarray,
    voxel_size: float,
    reference: VoxelReference | None,
    robot_radius_m: float,
) -> np.ndarray:
    """Pack ESDF-free, optionally GT-valid voxels without full-size index copies."""

    indices = np.asarray(esdf_idx, dtype=np.int32).reshape(-1, 3)
    free = np.asarray(esdf_free, dtype=bool).reshape(-1)
    if indices.shape[0] != free.shape[0]:
        raise ValueError("esdf_idx and esdf_free must have matching lengths")
    keys = np.empty((int(np.count_nonzero(free)),), dtype=np.int64)
    output_count = 0
    batch_size = 100_000 if reference is not None else 2_000_000
    for begin in range(0, indices.shape[0], batch_size):
        rows = np.flatnonzero(free[begin : begin + batch_size])
        if rows.shape[0] == 0:
            continue
        candidates = indices[begin + rows]
        if reference is not None:
            valid = ground_truth_reference_valid_mask(
                candidates,
                origin,
                voxel_size,
                voxel_dims,
                reference,
                robot_radius_m,
            )
            candidates = candidates[valid]
        count = int(candidates.shape[0])
        if count == 0:
            continue
        keys[output_count : output_count + count] = _pack_voxel_indices(
            candidates,
            voxel_dims,
        )
        output_count += count
    keys.resize((output_count,), refcheck=False)
    return keys


def explicit_esdf_free_mask(
    esdf_distance_m: np.ndarray,
    robot_radius_m: float,
) -> np.ndarray:
    """Apply ``ESDF >= robot_radius`` to explicitly stored samples."""

    distances = np.asarray(esdf_distance_m, dtype=np.float32).reshape(-1)
    result = np.zeros((distances.shape[0],), dtype=bool)
    batch_size = 2_000_000
    for begin in range(0, distances.shape[0], batch_size):
        values = distances[begin : begin + batch_size]
        result[begin : begin + batch_size] = np.logical_and(
            ~np.isnan(values),
            values.astype(np.float64) >= float(robot_radius_m),
        )
    return result


def _random_log_uniform_distance(
    rng: np.random.Generator,
    minimum: float,
    maximum: float,
) -> float:
    if maximum <= minimum:
        return float(minimum)
    return float(np.exp(rng.uniform(np.log(minimum), np.log(maximum))))


def sample_endpoint_pairs(
    cache: ConnectedComponentCache,
    *,
    origin: np.ndarray,
    voxel_size: float,
    rng: np.random.Generator,
    num_pairs: int,
    robot_radius_m: float = 0.0,
    min_separation_m: float | None = None,
    max_separation_m: float | None = None,
    goal_attempts_per_start: int = MAX_GOAL_ATTEMPTS_PER_START,
    max_start_attempts: int | None = None,
) -> list[dict[str, object]]:
    """Use bounded direction/distance rejection sampling on precomputed components."""

    requested_pairs = int(num_pairs)
    if requested_pairs <= 0:
        raise ValueError("num_pairs must be positive")
    if cache.sorted_keys.shape[0] < 2:
        raise RuntimeError("fewer than two traversable endpoint voxels are available")
    minimum, maximum = validate_separation_bounds(
        min_separation_m,
        max_separation_m,
    )
    voxel_size_value = float(voxel_size)
    strict_minimum = MIN_ENDPOINT_DISTANCE_VOXELS * voxel_size_value
    proposal_minimum = max(strict_minimum, 0.0 if minimum is None else minimum)
    proposal_minimum = float(np.nextafter(proposal_minimum, np.inf))
    bbox_extent = (
        cache.bbox_max.astype(np.float64) - cache.bbox_min.astype(np.float64)
    ) * voxel_size_value
    available_maximum = float(np.linalg.norm(bbox_extent))
    proposal_maximum = available_maximum if maximum is None else min(maximum, available_maximum)
    if not np.isfinite(proposal_maximum) or proposal_maximum <= proposal_minimum:
        raise RuntimeError(
            "traversable bounds cannot contain endpoints with the requested separation: "
            f"required > {strict_minimum:g} m, available <= {available_maximum:g} m"
        )

    attempts_per_start = max(1, int(goal_attempts_per_start))
    start_attempt_limit = (
        max(1000, requested_pairs * MAX_START_ATTEMPTS_PER_PAIR)
        if max_start_attempts is None
        else max(1, int(max_start_attempts))
    )
    grid_origin = np.asarray(origin, dtype=np.float64).reshape(3)
    dims = cache.voxel_dims.astype(np.int64)
    pairs: list[dict[str, object]] = []
    start_attempts = 0
    goal_attempts = 0

    while len(pairs) < requested_pairs and start_attempts < start_attempt_limit:
        start_attempts += 1
        start_row = int(rng.integers(cache.sorted_keys.shape[0]))
        start_idx = cache.voxel_idx_at(start_row)
        start_component = int(cache.component_ids[start_row])
        start_world = _voxel_centers(start_idx.reshape(1, 3), grid_origin, voxel_size_value)[0]

        for _ in range(attempts_per_start):
            goal_attempts += 1
            direction = rng.normal(size=3)
            norm = float(np.linalg.norm(direction))
            if not np.isfinite(norm) or norm <= 1.0e-12:
                continue
            direction /= norm
            sampled_distance = _random_log_uniform_distance(
                rng,
                proposal_minimum,
                proposal_maximum,
            )
            proposed_goal = start_world + direction * sampled_distance
            goal_idx = np.floor(
                (proposed_goal - grid_origin) / voxel_size_value
            ).astype(np.int64)
            if not np.logical_and(goal_idx >= 0, goal_idx < dims).all():
                continue
            goal_component = cache.component_id_for_idx(goal_idx)
            if goal_component == 0 or goal_component != start_component:
                continue
            goal_world = _voxel_centers(
                goal_idx.reshape(1, 3),
                grid_origin,
                voxel_size_value,
            )[0]
            separation = float(np.linalg.norm(goal_world - start_world))
            if separation <= strict_minimum:
                continue
            if minimum is not None and separation < minimum:
                continue
            if maximum is not None and separation > maximum:
                continue

            pairs.append(
                {
                    "trial_id": len(pairs),
                    "start_world": start_world.astype(float).tolist(),
                    "goal_world": goal_world.astype(float).tolist(),
                    "start_voxel_idx": start_idx.astype(int).tolist(),
                    "goal_voxel_idx": goal_idx.astype(int).tolist(),
                    "separation_m": separation,
                    "euclidean_distance_m": separation,
                    "sampled_distance_m": sampled_distance,
                    "component_id": start_component,
                    "start_component_id": start_component,
                    "goal_component_id": goal_component,
                    "robot_radius_m": float(robot_radius_m),
                }
            )
            break

    if len(pairs) < requested_pairs:
        raise RuntimeError(
            f"sampled only {len(pairs)}/{requested_pairs} endpoint pairs after "
            f"{start_attempts} start attempts and {goal_attempts} bounded goal attempts"
        )
    return pairs


def _resolve_esdf_inputs(
    reference_path: Path,
    esdf_map: Path | None,
    esdf_grid_metadata: Path | None,
    *,
    ground_truth_reference: bool,
) -> tuple[Path, Path]:
    esdf_source = reference_path.parent if esdf_map is None else esdf_map
    esdf_path = resolve_esdf_path(esdf_source)
    if esdf_grid_metadata is not None:
        grid_path = esdf_grid_metadata.expanduser().resolve()
    elif ground_truth_reference:
        json_path = reference_path.parent / "voxel_map_tsdf.json"
        npz_path = reference_path.parent / "voxel_map_tsdf.npz"
        grid_path = json_path if json_path.is_file() else npz_path
    else:
        grid_path = reference_path
    if not grid_path.is_file():
        raise FileNotFoundError(f"ESDF grid metadata not found: {grid_path}")
    return esdf_path, grid_path


def _validate_common_grid(
    reference_origin: np.ndarray,
    reference_voxel_size: float,
    reference_dims: np.ndarray,
    esdf_metadata,
) -> None:
    if not np.array_equal(
        reference_dims,
        np.asarray(esdf_metadata.voxel_dims, dtype=np.int32),
    ):
        raise ValueError(
            "reference and aligned ESDF voxel_dims differ: "
            f"{reference_dims.tolist()} vs "
            f"{np.asarray(esdf_metadata.voxel_dims).tolist()}"
        )
    if not np.isclose(
        reference_voxel_size,
        float(esdf_metadata.voxel_size),
        rtol=1.0e-6,
        atol=1.0e-9,
    ):
        raise ValueError(
            "reference and aligned ESDF voxel_size differ: "
            f"{reference_voxel_size:g} vs {float(esdf_metadata.voxel_size):g}"
        )
    if not np.allclose(
        reference_origin,
        np.asarray(esdf_metadata.origin, dtype=np.float64),
        rtol=0.0,
        atol=max(1.0e-7, reference_voxel_size * 1.0e-5),
    ):
        raise ValueError(
            "reference and aligned ESDF origins differ: "
            f"{reference_origin.tolist()} vs "
            f"{np.asarray(esdf_metadata.origin).tolist()}"
        )


def main(
    argv: list[str] | None = None,
    *,
    ground_truth_reference: bool = False,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        min_separation_m, max_separation_m = validate_separation_bounds(
            args.min_separation_m,
            args.max_separation_m,
        )
    except ValueError as exc:
        parser.error(str(exc))
    robot_radius_m = float(args.robot_radius_m)
    if not np.isfinite(robot_radius_m) or robot_radius_m <= 0.0:
        parser.error("robot_radius_m must be finite and positive")

    reference_path, origin, voxel_size, voxel_dims = _load_grid_metadata(
        args.reference_map
    )
    esdf_path, esdf_grid_path = _resolve_esdf_inputs(
        reference_path,
        args.esdf_map,
        args.esdf_grid_metadata,
        ground_truth_reference=ground_truth_reference,
    )
    preprocessing_started = time.perf_counter()
    esdf_metadata, esdf_idx, esdf_distance_m = load_aligned_esdf_samples(
        esdf_path,
        esdf_grid_path,
        cache_aligned=True,
    )
    _validate_common_grid(origin, voxel_size, voxel_dims, esdf_metadata)
    esdf_free = explicit_esdf_free_mask(
        esdf_distance_m,
        robot_radius_m,
    )
    known_esdf_free_count = int(np.count_nonzero(esdf_free))
    if known_esdf_free_count < 2:
        raise RuntimeError(
            "ESDF has fewer than two explicit voxels with clearance >= robot_radius"
        )

    if ground_truth_reference:
        reference = VoxelReference.from_map(
            reference_path,
            label=str(reference_path),
        )
        traversable_keys = _collect_traversable_keys(
            esdf_idx,
            esdf_free,
            voxel_dims,
            origin=origin,
            voxel_size=voxel_size,
            reference=reference,
            robot_radius_m=robot_radius_m,
        )
        reference_valid_count = int(traversable_keys.shape[0])
        del reference
    else:
        traversable_keys = _collect_traversable_keys(
            esdf_idx,
            esdf_free,
            voxel_dims,
            origin=origin,
            voxel_size=voxel_size,
            reference=None,
            robot_radius_m=robot_radius_m,
        )
        reference_valid_count = int(traversable_keys.shape[0])

    del esdf_free, esdf_idx, esdf_distance_m, esdf_metadata
    gc.collect()
    cache = ConnectedComponentCache.from_sorted_keys(
        traversable_keys,
        voxel_dims,
        slab_depth=CONNECTED_COMPONENT_SLAB_DEPTH,
        assume_sorted_unique=True,
    )
    preprocessing_time_sec = float(time.perf_counter() - preprocessing_started)
    rng = np.random.default_rng(SEED)
    sampling_started = time.perf_counter()
    pairs = sample_endpoint_pairs(
        cache,
        origin=origin,
        voxel_size=voxel_size,
        rng=rng,
        num_pairs=int(args.num_pairs),
        robot_radius_m=robot_radius_m,
        min_separation_m=min_separation_m,
        max_separation_m=max_separation_m,
    )
    sampling_time_sec = float(time.perf_counter() - sampling_started)
    manifest = {
        "format": "nerficg_precomputed_component_endpoint_manifest_v1",
        "occupancy_map": str(reference_path),
        "reference_map": str(reference_path),
        "esdf_map": str(esdf_path),
        "esdf_grid_metadata": str(esdf_grid_path),
        "reference_type": (
            "ground_truth_mesh_voxel_reference"
            if ground_truth_reference
            else "observed_esdf_reference"
        ),
        "endpoint_source": "precomputed_connected_component_rejection_sampling",
        # This legacy flag refers to the reference NPZ's explicit-free array.
        # Endpoint validity is defined by explicit ESDF samples here, so a
        # reference-unknown voxel can still be a valid endpoint.
        "unknown_endpoints_allowed": True,
        "esdf_unknown_endpoints_allowed": False,
        "reference_observed_free_robot_footprint_required": False,
        "reference_traversable_policy": (
            "explicit_esdf_clearance_and_gt_voxel_cube_collision_free"
            if ground_truth_reference
            else "explicit_esdf_clearance"
        ),
        "num_pairs": len(pairs),
        "seed": SEED,
        "robot_radius_m": robot_radius_m,
        "reference_safety_margin_m": REFERENCE_SAFETY_MARGIN_M,
        "esdf_clearance_threshold_m": robot_radius_m,
        "esdf_clearance_comparison": ">=",
        "minimum_endpoint_distance_voxels_exclusive": MIN_ENDPOINT_DISTANCE_VOXELS,
        "minimum_endpoint_distance_m_exclusive": MIN_ENDPOINT_DISTANCE_VOXELS * voxel_size,
        "min_separation_m": min_separation_m,
        "max_separation_m": max_separation_m,
        "distance_sampling": "log_uniform_random_distance_and_uniform_3d_direction",
        "goal_attempts_per_start": MAX_GOAL_ATTEMPTS_PER_START,
        "connected_component_connectivity": CONNECTED_COMPONENT_CONNECTIVITY,
        "connected_component_graph_semantics": "26_neighbor_voxel_center_adjacency",
        "connected_component_labeling": "bounded_memory_z_slab_then_boundary_merge",
        "known_esdf_free_voxel_count": known_esdf_free_count,
        "reference_valid_esdf_free_voxel_count": reference_valid_count,
        "sampled_safe_free_candidate_count": int(cache.sorted_keys.shape[0]),
        "traversable_voxel_count": int(cache.sorted_keys.shape[0]),
        "connected_component_count": int(cache.component_count),
        "connected_component_slab_count": int(cache.slab_count),
        "traversable_bbox_min_voxel_idx": cache.bbox_min.astype(int).tolist(),
        "traversable_bbox_max_voxel_idx": cache.bbox_max.astype(int).tolist(),
        "preprocessing_time_sec": preprocessing_time_sec,
        "sampling_time_sec": sampling_time_sec,
        "pairs": pairs,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"saved {len(pairs)} endpoint pairs from {cache.sorted_keys.shape[0]} "
        f"traversable voxels (precompute {preprocessing_time_sec:.3f}s, "
        f"sampling {sampling_time_sec:.3f}s): {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
