#!/usr/bin/env python3
"""Convert an aligned ground-truth mesh to the project's voxel-map format.

The mesh supplies occupied surface voxels.  An optional observed occupancy map
supplies explicit free voxels for endpoint sampling and unknown-space metrics;
free voxels that overlap the ground-truth surface are removed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GSPAR.mapping.voxel_grid_layout import load_reference_grid
from GSPAR.mapping.map_frame_transform import (
    apply_transform,
    load_map_frame_transform,
    transform_metadata,
    validate_rigid_transform,
)
from GSPAR.mapping.voxel_map_cli import normalize_output_prefix, resolve_existing_path


PLY_TYPES = {
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


def _log(message: str) -> None:
    print(f"[GTVoxelMap] {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True, help="Aligned GT mesh in training/world coordinates.")
    parser.add_argument(
        "--grid-map",
        type=Path,
        required=True,
        help="Existing voxel map whose origin, resolution, and dimensions define the common grid.",
    )
    parser.add_argument(
        "--free-space-map",
        type=Path,
        default=None,
        help="Optional observed map whose explicit free voxels are copied into the GT reference.",
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument(
        "--map-frame-transform",
        type=Path,
        default=None,
        help="Shared training-world to axis-aligned map-frame JSON used by every map builder.",
    )
    parser.add_argument(
        "--surface-mode",
        choices=("auto", "triangle", "vertices"),
        default="auto",
        help=(
            "triangle uses Open3D surface voxelization; vertices maps mesh vertices directly; "
            "auto prefers triangle and falls back to vertices when Open3D is unavailable"
        ),
    )
    return parser


def _parse_ply_vertex_header(
    path: Path,
) -> tuple[str, int, int, int, list[tuple[str, str]]]:
    """Return PLY format, vertex count, header lines/bytes, and vertex fields."""

    ply_format: str | None = None
    vertex_count: int | None = None
    vertex_properties: list[tuple[str, str]] = []
    in_vertex = False
    header_lines = 0
    header_bytes = 0
    with path.open("rb") as file:
        first = file.readline()
        header_lines += 1
        header_bytes += len(first)
        if first.decode("ascii").strip() != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        for raw in file:
            header_lines += 1
            header_bytes += len(raw)
            line = raw.decode("ascii").strip()
            if line == "end_header":
                break
            if line.startswith("format "):
                ply_format = line.split()[1]
            elif line.startswith("element "):
                parts = line.split()
                in_vertex = len(parts) >= 3 and parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
                    vertex_properties = []
            elif in_vertex and line.startswith("property "):
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(f"List properties are not supported in the PLY vertex element: {line}")
                vertex_properties.append((parts[2], parts[1]))
        else:
            raise ValueError(f"PLY header has no end_header: {path}")
    if ply_format not in {"ascii", "binary_little_endian"}:
        raise ValueError(f"Unsupported PLY format {ply_format!r}: {path}")
    if vertex_count is None:
        raise ValueError(f"PLY has no vertex element: {path}")
    names = {name for name, _ in vertex_properties}
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"PLY vertex element lacks x/y/z: {path}")
    return ply_format, vertex_count, header_lines, header_bytes, vertex_properties


def load_ply_vertices(path: Path) -> np.ndarray:
    """Load only XYZ vertices, without reading a potentially large face element."""

    ply_format, count, header_lines, header_bytes, properties = _parse_ply_vertex_header(path)
    property_to_col = {name: index for index, (name, _) in enumerate(properties)}
    if ply_format == "ascii":
        data = np.loadtxt(
            path,
            skiprows=header_lines,
            max_rows=count,
            usecols=[property_to_col[name] for name in ("x", "y", "z")],
            dtype=np.float32,
        )
        return np.asarray(data, dtype=np.float32).reshape(-1, 3)

    try:
        dtype = np.dtype([(name, PLY_TYPES[field_type]) for name, field_type in properties])
    except KeyError as exc:
        raise ValueError(f"Unsupported PLY vertex type {exc.args[0]!r}: {path}") from exc
    with path.open("rb") as file:
        file.seek(header_bytes)
        data = np.fromfile(file, dtype=dtype, count=count)
    if data.shape[0] != count:
        raise ValueError(f"Incomplete PLY vertex data: {path}")
    return np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)


def voxelize_surface_vertices(
    vertices: np.ndarray,
    grid: dict[str, object],
    transform: np.ndarray | None = None,
) -> np.ndarray:
    """Voxelize mesh vertices on the common grid used by every method."""

    points = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    if transform is not None:
        points = apply_transform(points, validate_rigid_transform(transform))
    points = points.astype(np.float64, copy=False)
    origin = np.asarray(grid["origin"], dtype=np.float64).reshape(3)
    voxel_size = float(grid["voxel_size"])
    indices = np.floor((points - origin[None, :]) / voxel_size).astype(np.int32)
    dims = np.asarray(grid["voxel_dims"], dtype=np.int32).reshape(3)
    valid = np.logical_and(indices >= 0, indices < dims[None, :]).all(axis=1)
    if not valid.any():
        return np.zeros((0, 3), dtype=np.int32)
    return np.unique(indices[valid], axis=0).astype(np.int32)


def _triangle_surface_voxels(
    mesh_path: Path,
    grid: dict[str, object],
    map_frame_transform: np.ndarray,
) -> tuple[np.ndarray, dict[str, int | str]]:
    try:
        import open3d as o3d
    except ModuleNotFoundError as exc:
        raise RuntimeError("Open3D is unavailable for triangle surface voxelization") from exc

    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    mesh.transform(np.asarray(map_frame_transform, dtype=np.float64))
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if vertices.shape[0] == 0 or triangles.shape[0] == 0:
        raise ValueError(f"Mesh has no triangle geometry: {mesh_path}")
    _log(f"Open3D voxelizing {triangles.shape[0]} triangles")
    surface = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
        mesh,
        voxel_size=float(grid["voxel_size"]),
    )
    voxels = surface.get_voxels()
    if not voxels:
        occupied = np.zeros((0, 3), dtype=np.int32)
    else:
        local_idx = np.asarray([voxel.grid_index for voxel in voxels], dtype=np.float64)
        centers = (
            np.asarray(surface.origin, dtype=np.float64)[None, :]
            + (local_idx + 0.5) * float(surface.voxel_size)
        )
        occupied = voxelize_surface_vertices(centers, grid)
    return occupied, {
        "surface_mode": "triangle",
        "num_input_vertices": int(vertices.shape[0]),
        "num_input_triangles": int(triangles.shape[0]),
    }


def _vertex_surface_voxels(
    mesh_path: Path,
    grid: dict[str, object],
    map_frame_transform: np.ndarray,
) -> tuple[np.ndarray, dict[str, int | str]]:
    if mesh_path.suffix.lower() != ".ply":
        raise ValueError("vertices mode currently supports PLY input only")
    _log("Mapping dense mesh vertices to the common voxel grid")
    vertices = load_ply_vertices(mesh_path)
    if vertices.shape[0] == 0:
        raise ValueError(f"Mesh has no vertices: {mesh_path}")
    return voxelize_surface_vertices(vertices, grid, map_frame_transform), {
        "surface_mode": "vertices",
        "num_input_vertices": int(vertices.shape[0]),
        "num_input_triangles": -1,
    }


def voxelize_mesh(
    mesh_path: Path,
    grid: dict[str, object],
    surface_mode: str,
    map_frame_transform: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int | str]]:
    transform = (
        np.eye(4, dtype=np.float32)
        if map_frame_transform is None
        else validate_rigid_transform(map_frame_transform)
    )
    if surface_mode == "triangle":
        return _triangle_surface_voxels(mesh_path, grid, transform)
    if surface_mode == "vertices":
        return _vertex_surface_voxels(mesh_path, grid, transform)
    try:
        return _triangle_surface_voxels(mesh_path, grid, transform)
    except RuntimeError as exc:
        if "Open3D is unavailable" not in str(exc):
            raise
        _log("Open3D is unavailable; using dense mesh vertices as the surface fallback")
        return _vertex_surface_voxels(mesh_path, grid, transform)


def _pack_idx(indices: np.ndarray, dims: np.ndarray) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    dims = np.asarray(dims, dtype=np.int64).reshape(3)
    return idx[:, 0] + dims[0] * (idx[:, 1] + dims[1] * idx[:, 2])


def load_explicit_free(
    source_path: Path | None,
    grid: dict[str, object],
    occupied_idx: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Copy in-grid explicit free voxels and remove GT occupied conflicts."""

    if source_path is None:
        return np.zeros((0, 3), dtype=np.int32), 0
    source_path = resolve_existing_path(source_path, default_suffix=".npz")
    if not source_path.is_file():
        raise FileNotFoundError(f"Free-space map not found: {source_path}")
    source_grid = load_reference_grid(source_path)
    for name in ("origin", "voxel_dims"):
        if not np.array_equal(np.asarray(grid[name]), np.asarray(source_grid[name])):
            raise ValueError(f"Free-space map uses a different {name}: {source_path}")
    if not np.isclose(float(grid["voxel_size"]), float(source_grid["voxel_size"])):
        raise ValueError(f"Free-space map uses a different voxel_size: {source_path}")
    first_transform = np.asarray(
        grid.get("map_from_training_world", np.eye(4)), dtype=np.float32
    )
    second_transform = np.asarray(
        source_grid.get("map_from_training_world", np.eye(4)), dtype=np.float32
    )
    if not np.allclose(first_transform, second_transform, atol=1e-6):
        raise ValueError(f"Free-space map uses a different map-frame transform: {source_path}")

    with np.load(source_path) as data:
        if "free_voxel_idx" not in data:
            raise ValueError(f"Free-space map has no explicit free_voxel_idx: {source_path}")
        free_idx = np.asarray(data["free_voxel_idx"], dtype=np.int32).reshape(-1, 3)
    dims = np.asarray(grid["voxel_dims"], dtype=np.int32)
    valid = np.logical_and(free_idx >= 0, free_idx < dims[None, :]).all(axis=1)
    free_idx = np.unique(free_idx[valid], axis=0).astype(np.int32)
    before_conflict_filter = int(free_idx.shape[0])
    if free_idx.shape[0] and occupied_idx.shape[0]:
        occupied_keys = np.unique(_pack_idx(occupied_idx, dims))
        free_keys = _pack_idx(free_idx, dims)
        positions = np.searchsorted(occupied_keys, free_keys)
        conflict = positions < occupied_keys.shape[0]
        rows = np.flatnonzero(conflict)
        conflict[rows] = occupied_keys[positions[rows]] == free_keys[rows]
        free_idx = free_idx[~conflict]
    return np.ascontiguousarray(free_idx, dtype=np.int32), before_conflict_filter - int(free_idx.shape[0])


def _voxel_centers(indices: np.ndarray, grid: dict[str, object]) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.float32).reshape(-1, 3)
    return (
        np.asarray(grid["origin"], dtype=np.float32)[None, :]
        + (indices + 0.5) * float(grid["voxel_size"])
    ).astype(np.float32)


def save_voxel_map(
    output_prefix: Path,
    grid: dict[str, object],
    occupied_idx: np.ndarray,
    free_idx: np.ndarray,
    metadata: dict[str, object],
    map_frame_transform: np.ndarray,
) -> tuple[Path, Path]:
    output_prefix = normalize_output_prefix(output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = output_prefix.with_suffix(".npz")
    json_path = output_prefix.with_suffix(".json")

    block_size = int(grid["block_size"])
    active_block_idx = (
        np.unique(np.floor_divide(occupied_idx, block_size), axis=0).astype(np.int32)
        if occupied_idx.shape[0]
        else np.zeros((0, 3), dtype=np.int32)
    )
    active_block_world = (
        np.asarray(grid["origin"], dtype=np.float32)[None, :]
        + (active_block_idx.astype(np.float32) + 0.5)
        * (float(grid["voxel_size"]) * block_size)
    ).astype(np.float32)
    np.savez_compressed(
        npz_path,
        voxel_size=np.asarray([grid["voxel_size"]], dtype=np.float32),
        block_size=np.asarray([grid["block_size"]], dtype=np.int32),
        origin=np.asarray(grid["origin"], dtype=np.float32),
        scene_min=np.asarray(grid["scene_min"], dtype=np.float32),
        scene_max=np.asarray(grid["scene_max"], dtype=np.float32),
        voxel_dims=np.asarray(grid["voxel_dims"], dtype=np.int32),
        block_dims=np.asarray(grid["block_dims"], dtype=np.int32),
        active_block_idx=active_block_idx,
        active_block_world=active_block_world,
        occupied_voxel_idx=occupied_idx.astype(np.int32),
        occupied_voxel_world=_voxel_centers(occupied_idx, grid),
        # Geometry evaluation consumes this explicit surface representation,
        # independently from occupied/free labels used by path planning.
        surface_voxel_idx=occupied_idx.astype(np.int32),
        surface_voxel_world=_voxel_centers(occupied_idx, grid),
        free_voxel_idx=free_idx.astype(np.int32),
        free_voxel_world=_voxel_centers(free_idx, grid),
        # Occupancy evaluation treats every non-mesh voxel as GT free.  The
        # explicit free array remains a narrower observed-space support mask
        # used by endpoint sampling and the path unknown-as-occupied policy.
        implicit_free=np.asarray([1], dtype=np.uint8),
        map_from_training_world=np.asarray(map_frame_transform, dtype=np.float32),
        training_world_from_map=np.linalg.inv(
            np.asarray(map_frame_transform, dtype=np.float32)
        ).astype(np.float32),
    )
    metadata = dict(metadata)
    metadata.update({"output_npz": str(npz_path), "output_json": str(json_path)})
    json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return npz_path, json_path


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    mesh_path = args.mesh.expanduser().resolve()
    grid_path = resolve_existing_path(args.grid_map, default_suffix=".npz")
    if not mesh_path.is_file():
        raise FileNotFoundError(f"GT mesh not found: {mesh_path}")
    if not grid_path.is_file():
        raise FileNotFoundError(f"Grid map not found: {grid_path}")

    grid = load_reference_grid(grid_path)
    requested_transform, requested_transform_info = load_map_frame_transform(
        args.map_frame_transform
    )
    grid_transform_value = grid.get("map_from_training_world")
    if grid_transform_value is None:
        if not np.allclose(requested_transform, np.eye(4), atol=1e-6):
            raise ValueError(
                "Non-identity map-frame transform requires a grid map built with the same transform"
            )
        map_transform = requested_transform
    else:
        grid_transform = validate_rigid_transform(grid_transform_value)
        if args.map_frame_transform is not None and not np.allclose(
            requested_transform, grid_transform, atol=1e-6
        ):
            raise ValueError("GT map-frame transform differs from the grid-map transform")
        map_transform = grid_transform
        if args.map_frame_transform is None:
            requested_transform_info = {
                "enabled": not bool(np.allclose(map_transform, np.eye(4), atol=1e-6)),
                "source": f"{grid_path}:map_from_training_world",
                "format": "nerficg_map_frame_transform_v1",
            }
    _log(
        f"Common grid dims={tuple(np.asarray(grid['voxel_dims']).tolist())}, "
        f"voxel_size={float(grid['voxel_size']):.6f}"
    )
    occupied_idx, geometry_stats = voxelize_mesh(
        mesh_path,
        grid,
        args.surface_mode,
        map_frame_transform=map_transform,
    )
    free_idx, removed_conflicts = load_explicit_free(
        args.free_space_map,
        grid,
        occupied_idx,
    )
    has_observed_free_support = args.free_space_map is not None
    dims = np.asarray(grid["voxel_dims"], dtype=np.int64)
    metadata: dict[str, object] = {
        "format": "nerficg_ground_truth_mesh_voxel_v1",
        "reference_semantics": {
            "occupied": "aligned ground-truth mesh surface",
            "surface": "aligned ground-truth mesh voxelized on the common grid",
            "free": (
                "implicit complement of the GT surface inside the common grid"
            ),
            "path_observed_free_support": (
                "free_voxel_idx copied from free_space_map, excluding GT occupied voxels"
                if has_observed_free_support
                else "not available"
            ),
            "path_unknown": "voxels outside explicit path_observed_free_support",
        },
        "mesh": str(mesh_path),
        "grid_map": str(grid_path),
        "free_space_map": (
            None if args.free_space_map is None else str(resolve_existing_path(args.free_space_map, default_suffix=".npz"))
        ),
        "map_frame": transform_metadata(map_transform, requested_transform_info),
        **geometry_stats,
        "surface_voxelizer": "mesh_vertices_floor_to_common_grid",
        "voxel_size": float(grid["voxel_size"]),
        "block_size": int(grid["block_size"]),
        "origin": np.asarray(grid["origin"], dtype=np.float32).tolist(),
        "scene_min": np.asarray(grid["scene_min"], dtype=np.float32).tolist(),
        "scene_max": np.asarray(grid["scene_max"], dtype=np.float32).tolist(),
        "voxel_dims": np.asarray(grid["voxel_dims"], dtype=np.int32).tolist(),
        "block_dims": np.asarray(grid["block_dims"], dtype=np.int32).tolist(),
        "num_voxels_total": int(np.prod(dims)),
        "num_occupied_voxels": int(occupied_idx.shape[0]),
        "num_free_voxels": int(free_idx.shape[0]),
        "num_free_gt_conflicts_removed": int(removed_conflicts),
        "implicit_free": True,
        "has_observed_free_support": has_observed_free_support,
        "timing_sec": {"wall_total": float(time.perf_counter() - started)},
    }
    npz_path, json_path = save_voxel_map(
        args.output_prefix,
        grid,
        occupied_idx,
        free_idx,
        metadata,
        map_transform,
    )
    metadata["output_npz"] = str(npz_path)
    metadata["output_json"] = str(json_path)
    _log(
        f"Saved occupied={occupied_idx.shape[0]}, free={free_idx.shape[0]}, "
        f"removed_free_conflicts={removed_conflicts}: {npz_path}"
    )
    return metadata


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
