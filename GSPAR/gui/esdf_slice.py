#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build axis-aligned ESDF slices on the same grid as a voxel map."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GSPAR.path.global_path.esdf_astar_world_path import (  # noqa: E402
    UNKNOWN_POLICY_OCCUPIED,
    EsdfGridGraph,
    load_esdf_grid,
)


AXIS_NAMES = ("x", "y", "z")
AXIS_TO_INDEX = {name: idx for idx, name in enumerate(AXIS_NAMES)}
DEFAULT_UNKNOWN_COLOR = np.array([0.20, 0.38, 0.95, 1.0], dtype=np.float32)
DEFAULT_OCCUPIED_COLOR = np.array([1.00, 0.04, 0.02, 1.0], dtype=np.float32)


@dataclass
class EsdfSlice:
    axis: str
    coordinate: float
    plane_index: int
    plane_coordinate: float
    origin: np.ndarray
    voxel_size: float
    voxel_dims: np.ndarray
    voxel_idx: np.ndarray
    centers_world: np.ndarray
    distance_m: np.ndarray
    color_rgba: np.ndarray
    is_unknown: np.ndarray
    is_occupied: np.ndarray
    is_observed: np.ndarray
    esdf_path: Path | None = None
    grid_path: Path | None = None

    @property
    def num_voxels(self) -> int:
        return int(self.voxel_idx.shape[0])

    @property
    def num_unknown(self) -> int:
        return int(np.count_nonzero(self.is_unknown))

    @property
    def num_occupied(self) -> int:
        return int(np.count_nonzero(self.is_occupied))

    @property
    def num_observed(self) -> int:
        return int(np.count_nonzero(self.is_observed))


def normalize_axis(axis: str) -> str:
    value = str(axis).strip().lower()
    if value not in AXIS_TO_INDEX:
        raise ValueError(f"axis must be one of {AXIS_NAMES}, got {axis!r}")
    return value


def voxel_map_supports_esdf_slice(voxel_map_path: str | Path) -> bool:
    """Return true only for TSDF-derived grid maps that can align ESDF PLY samples."""

    path = Path(voxel_map_path).expanduser()
    if path.suffix == "":
        npz_candidate = path.with_suffix(".npz")
        json_candidate = path.with_suffix(".json")
        if npz_candidate.is_file() or json_candidate.is_file():
            path = npz_candidate if npz_candidate.is_file() else json_candidate
    path = path.resolve()
    json_path = path.with_suffix(".json") if path.suffix.lower() != ".json" else path
    if not json_path.is_file():
        return False
    try:
        payload = json.load(json_path.open("r", encoding="utf-8"))
    except Exception:
        return False
    return bool(payload.get("input_tsdf_ply"))


def infer_esdf_path_from_voxel_map(voxel_map_path: str | Path) -> Path | None:
    """Infer scene/sdf/esdf_observed.ply from a loaded voxel map path."""

    if not voxel_map_supports_esdf_slice(voxel_map_path):
        return None

    path = Path(voxel_map_path).expanduser()
    if path.suffix == "":
        npz_candidate = path.with_suffix(".npz")
        if npz_candidate.is_file():
            path = npz_candidate
    path = path.resolve()
    candidates: list[Path] = []
    if path.parent.name == "sdf":
        candidates.append(path.parent / "esdf_observed.ply")
        candidates.append(path.parent.parent / "sdf" / "esdf_observed.ply")
    else:
        candidates.append(path.parent / "sdf" / "esdf_observed.ply")
        candidates.append(path.with_name("esdf_observed.ply"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _plane_index_for_coordinate(
    *,
    coordinate: float,
    axis_index: int,
    origin: np.ndarray,
    voxel_size: float,
    voxel_dims: np.ndarray,
) -> tuple[int, float]:
    raw = np.floor((float(coordinate) - float(origin[axis_index])) / float(voxel_size))
    plane_index = int(np.clip(raw, 0, int(voxel_dims[axis_index]) - 1))
    plane_coordinate = float(origin[axis_index] + (plane_index + 0.5) * float(voxel_size))
    return plane_index, plane_coordinate


def _build_plane_indices(voxel_dims: np.ndarray, axis_index: int, plane_index: int) -> np.ndarray:
    other_axes = [idx for idx in range(3) if idx != axis_index]
    first = np.arange(int(voxel_dims[other_axes[0]]), dtype=np.int32)
    second = np.arange(int(voxel_dims[other_axes[1]]), dtype=np.int32)
    first_grid, second_grid = np.meshgrid(first, second, indexing="ij")
    voxel_idx = np.empty((first_grid.size, 3), dtype=np.int32)
    voxel_idx[:, axis_index] = int(plane_index)
    voxel_idx[:, other_axes[0]] = first_grid.reshape(-1)
    voxel_idx[:, other_axes[1]] = second_grid.reshape(-1)
    return voxel_idx


def esdf_distances_to_rgba(
    distances_m: np.ndarray,
    *,
    max_distance_m: float,
    alpha: float = 1.0,
) -> np.ndarray:
    """Map distance to red near obstacles and green far from obstacles."""

    distances = np.asarray(distances_m, dtype=np.float32).reshape(-1)
    max_distance = max(float(max_distance_m), 1.0e-6)
    t = np.clip(np.maximum(distances, 0.0) / max_distance, 0.0, 1.0).astype(np.float32)

    red = np.array([1.00, 0.04, 0.02], dtype=np.float32)
    yellow = np.array([1.00, 0.82, 0.04], dtype=np.float32)
    green = np.array([0.08, 0.86, 0.18], dtype=np.float32)
    rgb = np.empty((distances.shape[0], 3), dtype=np.float32)

    first_half = t <= 0.5
    if first_half.any():
        local_t = (t[first_half] * 2.0)[:, None]
        rgb[first_half] = red[None, :] * (1.0 - local_t) + yellow[None, :] * local_t
    if (~first_half).any():
        local_t = ((t[~first_half] - 0.5) * 2.0)[:, None]
        rgb[~first_half] = yellow[None, :] * (1.0 - local_t) + green[None, :] * local_t

    colors = np.empty((distances.shape[0], 4), dtype=np.float32)
    colors[:, :3] = rgb
    colors[:, 3] = float(np.clip(alpha, 0.0, 1.0))
    return colors


def build_esdf_slice_from_graph(
    graph: EsdfGridGraph,
    *,
    axis: str,
    coordinate: float,
    max_distance_m: float,
    include_observed: bool = True,
    include_unknown: bool = True,
    include_occupied: bool = True,
    unknown_color: np.ndarray = DEFAULT_UNKNOWN_COLOR,
    occupied_color: np.ndarray = DEFAULT_OCCUPIED_COLOR,
    esdf_path: str | Path | None = None,
    grid_path: str | Path | None = None,
) -> EsdfSlice:
    """Build a sparse colored slice for one world-coordinate plane."""

    axis_name = normalize_axis(axis)
    axis_index = AXIS_TO_INDEX[axis_name]
    plane_index, plane_coordinate = _plane_index_for_coordinate(
        coordinate=float(coordinate),
        axis_index=axis_index,
        origin=graph.origin,
        voxel_size=graph.voxel_size,
        voxel_dims=graph.voxel_dims,
    )

    idx_parts: list[np.ndarray] = []
    distance_parts: list[np.ndarray] = []
    unknown_parts: list[np.ndarray] = []
    occupied_parts: list[np.ndarray] = []
    observed_parts: list[np.ndarray] = []

    esdf_plane_mask = graph.esdf_idx[:, axis_index] == int(plane_index)
    esdf_plane_idx = graph.esdf_idx[esdf_plane_mask]
    esdf_plane_distance = graph.esdf_distance_m[esdf_plane_mask]
    if esdf_plane_idx.shape[0]:
        esdf_plane_keys = graph._pack(esdf_plane_idx)
        esdf_plane_occupied = graph._contains_sorted(graph._occupied_keys, esdf_plane_keys)
        observed_idx = esdf_plane_idx[~esdf_plane_occupied]
        observed_distance = esdf_plane_distance[~esdf_plane_occupied]
    else:
        observed_idx = np.zeros((0, 3), dtype=np.int32)
        observed_distance = np.zeros((0,), dtype=np.float32)

    occupied_idx = graph.occupied_idx[graph.occupied_idx[:, axis_index] == int(plane_index)]
    if occupied_idx.shape[0]:
        occupied_idx = np.unique(occupied_idx, axis=0).astype(np.int32, copy=False)

    if include_observed and observed_idx.shape[0]:
        idx_parts.append(observed_idx.astype(np.int32, copy=False))
        distance_parts.append(observed_distance.astype(np.float32, copy=False))
        unknown_parts.append(np.zeros((observed_idx.shape[0],), dtype=bool))
        occupied_parts.append(np.zeros((observed_idx.shape[0],), dtype=bool))
        observed_parts.append(np.ones((observed_idx.shape[0],), dtype=bool))

    if include_occupied and occupied_idx.shape[0]:
        idx_parts.append(occupied_idx.astype(np.int32, copy=False))
        distance_parts.append(np.zeros((occupied_idx.shape[0],), dtype=np.float32))
        unknown_parts.append(np.zeros((occupied_idx.shape[0],), dtype=bool))
        occupied_parts.append(np.ones((occupied_idx.shape[0],), dtype=bool))
        observed_parts.append(np.zeros((occupied_idx.shape[0],), dtype=bool))

    if include_unknown:
        plane_idx = _build_plane_indices(graph.voxel_dims, axis_index, plane_index)
        keys = graph._pack(plane_idx)
        _, has_esdf = graph._esdf_distances_for_keys(keys)
        is_occupied = graph._contains_sorted(graph._occupied_keys, keys)
        unknown_idx = plane_idx[np.logical_and(~has_esdf, ~is_occupied)]
        if unknown_idx.shape[0]:
            idx_parts.append(unknown_idx.astype(np.int32, copy=False))
            distance_parts.append(np.full((unknown_idx.shape[0],), np.nan, dtype=np.float32))
            unknown_parts.append(np.ones((unknown_idx.shape[0],), dtype=bool))
            occupied_parts.append(np.zeros((unknown_idx.shape[0],), dtype=bool))
            observed_parts.append(np.zeros((unknown_idx.shape[0],), dtype=bool))

    if idx_parts:
        selected_idx = np.concatenate(idx_parts, axis=0)
        selected_distance = np.concatenate(distance_parts, axis=0)
        selected_unknown = np.concatenate(unknown_parts, axis=0)
        selected_occupied = np.concatenate(occupied_parts, axis=0)
        selected_observed = np.concatenate(observed_parts, axis=0)
        order = np.argsort(graph._pack(selected_idx), kind="mergesort")
        selected_idx = selected_idx[order]
        selected_distance = selected_distance[order]
        selected_unknown = selected_unknown[order]
        selected_occupied = selected_occupied[order]
        selected_observed = selected_observed[order]
    else:
        selected_idx = np.zeros((0, 3), dtype=np.int32)
        selected_distance = np.zeros((0,), dtype=np.float32)
        selected_unknown = np.zeros((0,), dtype=bool)
        selected_occupied = np.zeros((0,), dtype=bool)
        selected_observed = np.zeros((0,), dtype=bool)

    colors = np.zeros((selected_idx.shape[0], 4), dtype=np.float32)
    observed_rows = np.flatnonzero(selected_observed)
    if observed_rows.shape[0]:
        colors[observed_rows] = esdf_distances_to_rgba(
            selected_distance[observed_rows],
            max_distance_m=max_distance_m,
            alpha=1.0,
        )
    occupied_rows = np.flatnonzero(selected_occupied)
    if occupied_rows.shape[0]:
        colors[occupied_rows] = np.asarray(occupied_color, dtype=np.float32).reshape(4)
    unknown_rows = np.flatnonzero(selected_unknown)
    if unknown_rows.shape[0]:
        colors[unknown_rows] = np.asarray(unknown_color, dtype=np.float32).reshape(4)

    centers_world = graph.voxel_idx_to_world_center(selected_idx).astype(np.float32)
    return EsdfSlice(
        axis=axis_name,
        coordinate=float(coordinate),
        plane_index=int(plane_index),
        plane_coordinate=float(plane_coordinate),
        origin=graph.origin.astype(np.float32, copy=True),
        voxel_size=float(graph.voxel_size),
        voxel_dims=graph.voxel_dims.astype(np.int32, copy=True),
        voxel_idx=selected_idx.astype(np.int32, copy=False),
        centers_world=centers_world,
        distance_m=selected_distance.astype(np.float32, copy=False),
        color_rgba=colors.astype(np.float32, copy=False),
        is_unknown=selected_unknown.astype(bool, copy=False),
        is_occupied=selected_occupied.astype(bool, copy=False),
        is_observed=selected_observed.astype(bool, copy=False),
        esdf_path=Path(esdf_path).expanduser().resolve() if esdf_path is not None else None,
        grid_path=Path(grid_path).expanduser().resolve() if grid_path is not None else None,
    )


def build_esdf_slice(
    esdf_path: str | Path,
    *,
    grid_metadata_path: str | Path | None,
    axis: str,
    coordinate: float,
    max_distance_m: float,
    include_observed: bool = True,
    include_unknown: bool = True,
    include_occupied: bool = True,
    cache_aligned: bool = True,
) -> EsdfSlice:
    """Load an ESDF grid and return a colored axis-aligned slice."""

    if grid_metadata_path is not None and not voxel_map_supports_esdf_slice(grid_metadata_path):
        raise ValueError("ESDF slicing requires a TSDF-derived voxel_map_tsdf.npz grid.")

    graph = load_esdf_grid(
        esdf_path,
        grid_metadata_path,
        use_occupancy_guard=True,
        load_free_idx=False,
        cache_aligned=bool(cache_aligned),
        unknown_policy=UNKNOWN_POLICY_OCCUPIED,
    )
    return build_esdf_slice_from_graph(
        graph,
        axis=axis,
        coordinate=coordinate,
        max_distance_m=max_distance_m,
        include_observed=include_observed,
        include_unknown=include_unknown,
        include_occupied=include_occupied,
        esdf_path=esdf_path,
        grid_path=grid_metadata_path,
    )


def _uniform_pick(indices: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    if count >= values.shape[0]:
        return values
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    pick = np.linspace(0, values.shape[0] - 1, num=int(count), dtype=np.int64)
    return values[pick]


def _balanced_group_quotas(group_sizes: list[int], max_count: int) -> list[int]:
    total = int(sum(group_sizes))
    if total <= int(max_count):
        return [int(size) for size in group_sizes]
    nonempty = [idx for idx, size in enumerate(group_sizes) if int(size) > 0]
    quotas = [0 for _ in group_sizes]
    if not nonempty or max_count <= 0:
        return quotas

    available = int(max_count)
    for idx in nonempty:
        quotas[idx] = 1
    available -= len(nonempty)
    if available < 0:
        quotas = [0 for _ in group_sizes]
        for idx in nonempty[:max_count]:
            quotas[idx] = 1
        return quotas

    raw_extra = []
    for idx, size in enumerate(group_sizes):
        if size <= 0:
            raw_extra.append(0.0)
            continue
        share = float(size) / float(total)
        raw_extra.append(max(0.0, share * float(max_count) - float(quotas[idx])))

    floors = [min(int(np.floor(raw_extra[idx])), max(0, int(group_sizes[idx]) - quotas[idx])) for idx in range(len(group_sizes))]
    for idx, amount in enumerate(floors):
        quotas[idx] += amount
        available -= amount

    while available > 0:
        candidates = [
            idx
            for idx in nonempty
            if quotas[idx] < int(group_sizes[idx])
        ]
        if not candidates:
            break
        best = max(candidates, key=lambda idx: raw_extra[idx] - np.floor(raw_extra[idx]))
        quotas[best] += 1
        available -= 1
        raw_extra[best] = 0.0
    return quotas


def downsample_esdf_slice(slice_data: EsdfSlice, max_voxels: int) -> EsdfSlice:
    """Limit display size while keeping known ESDF detail and a visible unknown sample."""

    max_count = int(max_voxels)
    if max_count <= 0 or slice_data.num_voxels <= max_count:
        return slice_data

    occupied = np.flatnonzero(slice_data.is_occupied)
    observed = np.flatnonzero(slice_data.is_observed)
    unknown = np.flatnonzero(slice_data.is_unknown)
    known_count = int(occupied.shape[0] + observed.shape[0])

    if known_count == 0:
        selected = _uniform_pick(unknown, max_count)
    elif unknown.shape[0] == 0:
        quotas = _balanced_group_quotas([int(occupied.shape[0]), int(observed.shape[0])], max_count)
        selected = np.concatenate([
            _uniform_pick(occupied, quotas[0]),
            _uniform_pick(observed, quotas[1]),
        ])
    else:
        unknown_quota = min(int(unknown.shape[0]), max(1, int(round(max_count * 0.20))))
        known_budget = max(0, max_count - unknown_quota)
        if known_count <= known_budget:
            known_selected = np.concatenate([occupied, observed])
            unknown_quota = min(int(unknown.shape[0]), max_count - int(known_selected.shape[0]))
            selected = np.concatenate([known_selected, _uniform_pick(unknown, unknown_quota)])
        else:
            quotas = _balanced_group_quotas(
                [int(occupied.shape[0]), int(observed.shape[0])],
                known_budget,
            )
            selected = np.concatenate([
                _uniform_pick(occupied, quotas[0]),
                _uniform_pick(observed, quotas[1]),
                _uniform_pick(unknown, unknown_quota),
            ])
    selected = np.sort(selected.astype(np.int64, copy=False))
    return replace(
        slice_data,
        voxel_idx=slice_data.voxel_idx[selected],
        centers_world=slice_data.centers_world[selected],
        distance_m=slice_data.distance_m[selected],
        color_rgba=slice_data.color_rgba[selected],
        is_unknown=slice_data.is_unknown[selected],
        is_occupied=slice_data.is_occupied[selected],
        is_observed=slice_data.is_observed[selected],
    )


def save_esdf_slice_npz(slice_data: EsdfSlice, output_path: str | Path) -> Path:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        axis=np.asarray(slice_data.axis),
        coordinate=np.asarray([slice_data.coordinate], dtype=np.float32),
        plane_index=np.asarray([slice_data.plane_index], dtype=np.int32),
        plane_coordinate=np.asarray([slice_data.plane_coordinate], dtype=np.float32),
        origin=slice_data.origin.astype(np.float32),
        voxel_size=np.asarray([slice_data.voxel_size], dtype=np.float32),
        voxel_dims=slice_data.voxel_dims.astype(np.int32),
        voxel_idx=slice_data.voxel_idx.astype(np.int32),
        centers_world=slice_data.centers_world.astype(np.float32),
        distance_m=slice_data.distance_m.astype(np.float32),
        color_rgba=slice_data.color_rgba.astype(np.float32),
        is_unknown=slice_data.is_unknown.astype(bool),
        is_occupied=slice_data.is_occupied.astype(bool),
        is_observed=slice_data.is_observed.astype(bool),
        esdf_path=np.asarray(str(slice_data.esdf_path or "")),
        grid_path=np.asarray(str(slice_data.grid_path or "")),
    )
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voxel-map", "--grid", dest="grid_path", required=True, help="Target voxel_map_tsdf.npz/.json")
    parser.add_argument("--esdf", default="", help="ESDF PLY. Defaults to scene/sdf/esdf_observed.ply next to voxel map.")
    parser.add_argument("--axis", choices=AXIS_NAMES, default="z")
    parser.add_argument("--coord", type=float, required=True, help="World coordinate of the requested slice plane.")
    parser.add_argument("--max-distance-m", type=float, default=1.0, help="Distance mapped to green.")
    parser.add_argument("--no-observed", action="store_true")
    parser.add_argument("--no-unknown", action="store_true")
    parser.add_argument("--no-occupied", action="store_true")
    parser.add_argument("--no-cache", action="store_true", help="Do not use/create aligned ESDF cache.")
    parser.add_argument("--max-output-voxels", type=int, default=0, help="Optional uniform cap for saved/displayed voxels.")
    parser.add_argument("--output", default="", help="Optional .npz output path.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    grid_path = Path(args.grid_path).expanduser().resolve()
    esdf_path = Path(args.esdf).expanduser().resolve() if args.esdf.strip() else infer_esdf_path_from_voxel_map(grid_path)
    if esdf_path is None:
        raise FileNotFoundError(f"Could not infer ESDF path from voxel map: {grid_path}")
    slice_data = build_esdf_slice(
        esdf_path,
        grid_metadata_path=grid_path,
        axis=args.axis,
        coordinate=float(args.coord),
        max_distance_m=float(args.max_distance_m),
        include_observed=not bool(args.no_observed),
        include_unknown=not bool(args.no_unknown),
        include_occupied=not bool(args.no_occupied),
        cache_aligned=not bool(args.no_cache),
    )
    if int(args.max_output_voxels) > 0:
        slice_data = downsample_esdf_slice(slice_data, int(args.max_output_voxels))
    if args.output.strip():
        output = save_esdf_slice_npz(slice_data, args.output)
        print(f"saved {output}")
    print(
        "slice "
        f"axis={slice_data.axis} requested={slice_data.coordinate:.4f} "
        f"plane_index={slice_data.plane_index} plane_coord={slice_data.plane_coordinate:.4f} "
        f"voxels={slice_data.num_voxels} observed={slice_data.num_observed} "
        f"occupied={slice_data.num_occupied} unknown={slice_data.num_unknown}"
    )


if __name__ == "__main__":
    main()
