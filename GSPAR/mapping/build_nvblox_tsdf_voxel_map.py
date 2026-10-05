#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convert an nvblox TSDF PLY export to the project's ``voxel_map_tsdf`` format.

This script implements only the workflow required by this project:
1. Read a TSDF PLY in the nvblox source frame, optionally axis-aligned before fusion.
2. Transform it to the training/map frame using ``training_config`` or an explicit affine.
3. Derive a canonical grid independently from a 3DGS checkpoint and target voxel size.
4. Query backward only in target blocks that may overlap observed source blocks.
5. Mark TSDF hits <= 0 as occupied, hits > 0 as free, and misses as unknown.
6. Interpolate TSDF on the target grid and extract a mesh and surface voxels at zero crossings.
7. Preserve occupied/free voxels for planning and surface voxels for geometry evaluation.

No camera-matching alignment is attempted; only transforms specified by the training
configuration are applied.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GSPAR.mapping.alignment_utils import ensure_project_paths, load_training_dataset_affine
from GSPAR.mapping.map_frame_transform import (
    load_map_frame_transform,
    transform_metadata,
)
from GSPAR.mapping.voxel_grid_layout import build_grid_spec_from_checkpoint_positions
from GSPAR.mapping.voxel_map_cli import normalize_output_prefix


PROJECT_ROOT = ensure_project_paths(__file__)
DEFAULT_BLOCK_SIZE = 8
DEFAULT_SCENE_PADDING = 0.02
NVBLOX_BLOCK_SIZE = 8
OCCUPIED_DISTANCE_THRESHOLD = 0.0
FREE_DISTANCE_THRESHOLD = 0.0
BACKWARD_BATCH_VOXELS = 500_000
SOURCE_BLOCK_CHUNK = 50_000
TRILINEAR_WEIGHT_EPSILON = 1.0e-5
SURFACE_GRID_SNAP_EPSILON = 1.0e-3
CUBE_CORNER_OFFSETS = np.asarray(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    dtype=np.int32,
)
# Six tetrahedra around the 0-7 cube diagonal. Marching tetrahedra keeps the
# TSDF mesh extraction dependency-free while preserving the zero level set.
CUBE_TETRAHEDRA = (
    (0, 1, 3, 7),
    (0, 3, 2, 7),
    (0, 2, 6, 7),
    (0, 6, 4, 7),
    (0, 4, 5, 7),
    (0, 5, 1, 7),
)


def parse_args() -> argparse.Namespace:
    """Parse only the inputs that must vary between scenes."""

    parser = argparse.ArgumentParser(description="把 nvblox TSDF PLY 转成 voxel_map_tsdf。")
    parser.add_argument("--tsdf-ply", type=Path, required=True, help="输入 tsdf_observed.ply。")
    parser.add_argument("--grid-checkpoint", type=Path, required=True, help="3DGS checkpoint，用来确定共同体素网格。")
    parser.add_argument("--training-config", type=Path, required=True, help="训练配置，用来复现 world transform/PCA。")
    parser.add_argument(
        "--source-voxel-size",
        type=float,
        required=True,
        help="nvblox 原始 metric/COLMAP 空间的 voxel size。",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        required=True,
        help="最终 3DGS 训练坐标空间的 canonical voxel size。",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help="目标 grid 每个 block 单轴包含的 voxel 数。",
    )
    parser.add_argument(
        "--scene-padding",
        type=float,
        default=DEFAULT_SCENE_PADDING,
        help="checkpoint Gaussian 中心包围盒两侧 padding。",
    )
    parser.add_argument(
        "--source-coordinate-transform",
        choices=("auto", "xyz", "xnzy"),
        default="auto",
        help="源 COLMAP/TSDF 到训练 world 的轴变换；auto 使用当前 dataset loader 的 xyz 约定。",
    )
    parser.add_argument(
        "--source-to-training-affine",
        type=Path,
        default=None,
        help=(
            "可选 JSON，显式提供输入 TSDF frame 到 training world 的 4x4 affine；"
            "设置后不再从 training_config 推导该变换。未设置时，如果输入位于 "
            "<render-output>/sdf/，会优先自动使用相邻的 "
            "<render-output>/nvblox_frame_alignment.json。"
        ),
    )
    parser.add_argument(
        "--map-frame-transform",
        type=Path,
        default=None,
        help="Shared training-world to axis-aligned map-frame JSON.",
    )
    parser.add_argument("--output-prefix", type=Path, required=True, help="输出前缀，会生成 .npz 和 .json。")
    parser.add_argument(
        "--surface-mesh-output",
        type=Path,
        default=None,
        help="TSDF 零交叉 mesh 输出；默认写到 <output-prefix>_surface_mesh.ply。",
    )
    return parser.parse_args()


def _parse_ply_header(path: Path) -> tuple[str, int, int, int, dict[str, int], list[tuple[str, str]]]:
    """Read a PLY header to obtain the vertex count, property order, and format."""

    vertex_count: int | None = None
    properties: list[tuple[str, str]] = []
    in_vertex = False
    header_lines = 0
    header_bytes = 0
    ply_format: str | None = None

    with path.open("rb") as f:
        first_bytes = f.readline()
        header_bytes += len(first_bytes)
        header_lines += 1
        if first_bytes.decode("ascii").strip() != "ply":
            raise ValueError(f"不是 PLY 文件: {path}")

        for line_bytes in f:
            header_bytes += len(line_bytes)
            header_lines += 1
            line = line_bytes.decode("ascii").strip()
            if line == "end_header":
                break

            if line.startswith("format "):
                parts = line.split()
                if len(parts) < 3:
                    raise ValueError(f"PLY format 行格式错误: {line}")
                ply_format = parts[1]
                if ply_format not in {"ascii", "binary_little_endian"}:
                    raise ValueError(f"不支持的 PLY 格式: {line}")
                continue

            if line.startswith("element "):
                parts = line.split()
                in_vertex = len(parts) >= 3 and parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
                    properties = []
                continue

            if in_vertex and line.startswith("property "):
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(f"不支持的 PLY property 行: {line}")
                properties.append((parts[2], parts[1]))
        else:
            raise ValueError(f"PLY header 缺少 end_header: {path}")

    if vertex_count is None:
        raise ValueError(f"PLY 中没有 vertex element: {path}")
    if ply_format is None:
        raise ValueError(f"PLY 中没有 format 行: {path}")

    property_to_col = {name: idx for idx, (name, _) in enumerate(properties)}
    return ply_format, int(vertex_count), int(header_lines), int(header_bytes), property_to_col, properties


def _tsdf_value_property(property_to_col: dict[str, int], path: Path) -> str:
    """Find the TSDF property, which nvblox may name ``intensity`` or ``distance``."""

    for name in ("intensity", "distance"):
        if name in property_to_col:
            return name
    raise ValueError(f"PLY 缺少 TSDF 值字段 intensity/distance: {path}")


def load_tsdf_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read TSDF sample coordinates and values from a PLY file."""

    ply_format, vertex_count, header_lines, header_bytes, prop, properties = _parse_ply_header(path)
    missing = [name for name in ("x", "y", "z") if name not in prop]
    if missing:
        raise ValueError(f"PLY 缺少坐标字段: {missing}")

    tsdf_prop = _tsdf_value_property(prop, path)
    if ply_format == "ascii":
        data = np.loadtxt(
            path,
            skiprows=header_lines,
            usecols=[prop["x"], prop["y"], prop["z"], prop[tsdf_prop]],
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
        raise ValueError(f"不支持的二进制 PLY 字段类型: {exc.args[0]}") from exc

    with path.open("rb") as f:
        f.seek(header_bytes)
        data = np.fromfile(f, dtype=dtype, count=vertex_count)
    if data.shape[0] != vertex_count:
        raise ValueError(f"二进制 PLY 顶点数据不完整: {path}")

    points = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    return points, data[tsdf_prop].astype(np.float32)


def _load_source_to_training_affine(
    training_config: Path,
    source_coordinate_transform: str = "auto",
) -> tuple[np.ndarray, dict[str, object]]:
    """Load the deterministic coordinate transform from the training configuration."""

    training_config = training_config.expanduser().resolve()
    if not training_config.is_file():
        raise FileNotFoundError(f"找不到 training_config: {training_config}")
    return load_training_dataset_affine(
        training_config,
        PROJECT_ROOT,
        world_transform_override=source_coordinate_transform,
    )


def _load_source_to_training_affine_json(
    path: Path,
) -> tuple[np.ndarray, dict[str, object]]:
    """Load and strictly validate an explicit source-to-training affine."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"找不到 source-to-training affine JSON: {resolved}")
    with resolved.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        value = payload
        matrix_key = "top_level"
    elif isinstance(payload, dict):
        supported_keys = [
            key
            for key in ("source_to_training_affine", "training_from_metric_affine")
            if key in payload
        ]
        if len(supported_keys) != 1:
            raise ValueError(
                "source-to-training affine JSON object must contain exactly one of "
                "source_to_training_affine/training_from_metric_affine"
            )
        matrix_key = supported_keys[0]
        value = payload[matrix_key]
    else:
        raise ValueError(
            "source-to-training affine JSON must be a top-level 4x4 list or an object"
        )

    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("source-to-training affine must contain numeric values") from exc
    if matrix.shape != (4, 4):
        raise ValueError(
            f"source-to-training affine must be 4x4, got {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("source-to-training affine contains non-finite values")
    if not np.allclose(
        matrix[3],
        [0.0, 0.0, 0.0, 1.0],
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise ValueError("source-to-training affine has an invalid homogeneous last row")
    with np.errstate(over="ignore", invalid="ignore"):
        matrix_float32 = matrix.astype(np.float32)
    if not np.isfinite(matrix_float32).all():
        raise ValueError("source-to-training affine is not representable as finite float32")
    try:
        linear_inverse = np.linalg.inv(matrix_float32[:3, :3].astype(np.float64))
    except np.linalg.LinAlgError as exc:
        raise ValueError("source-to-training affine linear part is not invertible") from exc
    if not np.isfinite(linear_inverse).all():
        raise ValueError("source-to-training affine linear part is not invertible")

    return matrix_float32, {
        "source": "source_to_training_affine_json",
        "path": str(resolved),
        "matrix_key": matrix_key,
    }


def _discover_source_to_training_affine(tsdf_ply: Path) -> Path | None:
    """Find the frame sidecar emitted by the rendered-depth nvblox pipeline."""

    resolved = tsdf_ply.expanduser().resolve()
    if resolved.parent.name != "sdf":
        return None
    candidate = resolved.parent.parent / "nvblox_frame_alignment.json"
    return candidate if candidate.is_file() else None


def _transformed_voxel_size(source_voxel_size: float, affine: np.ndarray) -> tuple[float, list[float]]:
    """Convert the source TSDF voxel size to trained 3DGS space."""

    linear = np.asarray(affine, dtype=np.float32)[:3, :3]
    axis_scales = np.linalg.norm(linear, axis=0).astype(np.float32)
    # Training uses uniform PCA scaling; the median suppresses floating-point noise.
    return float(source_voxel_size) * float(np.median(axis_scales)), axis_scales.tolist()


def _grid_from_checkpoint(
    checkpoint_path: Path,
    voxel_size: float,
    block_size: int,
    scene_padding: float,
    map_frame_transform: np.ndarray | None = None,
) -> tuple[dict[str, object], int]:
    """Build the shared voxel grid from Gaussian positions in a checkpoint."""

    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到 grid checkpoint: {checkpoint_path}")

    grid, num_positions = build_grid_spec_from_checkpoint_positions(
        checkpoint_path=checkpoint_path,
        voxel_size=float(voxel_size),
        block_size=int(block_size),
        scene_padding=float(scene_padding),
        map_frame_transform=map_frame_transform,
    )
    return {
        "origin": grid.origin.astype(np.float32),
        "scene_min": grid.scene_min.astype(np.float32),
        "scene_max": grid.scene_max.astype(np.float32),
        "voxel_dims": grid.voxel_dims.astype(np.int32),
        "block_dims": grid.block_dims.astype(np.int32),
        "voxel_size": float(grid.voxel_size),
        "block_size": int(grid.block_size),
    }, int(num_positions)


def _voxel_center_to_idx(points: np.ndarray, origin: np.ndarray, voxel_size: float) -> np.ndarray:
    """Map TSDF samples to their nearest voxel-center indices."""

    if points.size == 0:
        return np.zeros((0, 3), dtype=np.int32)
    idx_float = (points.astype(np.float32) - origin[None, :]) / float(voxel_size) - 0.5
    return np.rint(idx_float).astype(np.int32)


def _sample_sparse_tsdf_trilinear(
    points: np.ndarray,
    source_origin: np.ndarray,
    source_voxel_size: float,
    source_dims: np.ndarray,
    source_keys: np.ndarray,
    source_tsdf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Trilinearly sample sparse TSDF; incomplete neighbourhoods stay unknown."""

    query = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    values = np.zeros((query.shape[0],), dtype=np.float32)
    matched = np.zeros((query.shape[0],), dtype=bool)
    if query.shape[0] == 0 or source_keys.shape[0] == 0:
        return values, matched

    origin = np.asarray(source_origin, dtype=np.float32).reshape(3)
    dims = np.asarray(source_dims, dtype=np.int32).reshape(3)
    continuous_idx = (
        (query - origin[None, :]) / float(source_voxel_size) - 0.5
    ).astype(np.float32)
    rounded_idx = np.rint(continuous_idx)
    continuous_idx = np.where(
        np.abs(continuous_idx - rounded_idx) <= TRILINEAR_WEIGHT_EPSILON,
        rounded_idx,
        continuous_idx,
    ).astype(np.float32)
    base_idx = np.floor(continuous_idx).astype(np.int32)
    fraction = (continuous_idx - base_idx.astype(np.float32)).astype(np.float32)
    corner_idx = base_idx[:, None, :] + CUBE_CORNER_OFFSETS[None, :, :]
    corner_in_bounds = np.logical_and(
        corner_idx >= 0,
        corner_idx < dims[None, None, :],
    ).all(axis=2)
    clipped_corner_idx = np.clip(
        corner_idx,
        np.zeros((1, 1, 3), dtype=np.int32),
        (dims - 1)[None, None, :],
    )
    corner_keys = _pack_voxel_idx(
        clipped_corner_idx.reshape(-1, 3), dims
    ).reshape(-1, 8)
    corner_positions = np.searchsorted(source_keys, corner_keys)
    safe_corner_positions = np.minimum(corner_positions, source_keys.shape[0] - 1)
    corner_observed = (
        corner_in_bounds
        & (corner_positions < source_keys.shape[0])
        & (source_keys[safe_corner_positions] == corner_keys)
    )

    corner_fraction = np.where(
        CUBE_CORNER_OFFSETS[None, :, :].astype(bool),
        fraction[:, None, :],
        1.0 - fraction[:, None, :],
    )
    corner_weights = np.prod(corner_fraction, axis=2, dtype=np.float32)
    required_corner = corner_weights > TRILINEAR_WEIGHT_EPSILON
    can_interpolate = np.all(corner_observed | ~required_corner, axis=1)
    if np.any(can_interpolate):
        corner_values = source_tsdf[safe_corner_positions]
        effective_weights = np.where(
            required_corner[can_interpolate],
            corner_weights[can_interpolate],
            0.0,
        ).astype(np.float32)
        weight_sum = np.sum(effective_weights, axis=1, dtype=np.float32)
        values[can_interpolate] = np.sum(
            effective_weights * corner_values[can_interpolate],
            axis=1,
            dtype=np.float32,
        ) / weight_sum
        matched[can_interpolate] = True

    return values, matched


def _voxelize_target_surface_vertices(
    vertices: np.ndarray,
    grid: dict[str, object],
) -> np.ndarray:
    """Voxelize target-frame mesh vertices with stable grid-plane tie handling."""

    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.int32)
    origin = np.asarray(grid["origin"], dtype=np.float64).reshape(3)
    voxel_size = float(grid["voxel_size"])
    relative = (points - origin[None, :]) / voxel_size
    rounded = np.rint(relative)
    relative = np.where(
        np.abs(relative - rounded) <= SURFACE_GRID_SNAP_EPSILON,
        rounded,
        relative,
    )
    indices = np.floor(relative).astype(np.int32)
    dims = np.asarray(grid["voxel_dims"], dtype=np.int32).reshape(3)
    valid = np.logical_and(indices >= 0, indices < dims[None, :]).all(axis=1)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.int32)
    return np.unique(indices[valid], axis=0).astype(np.int32)


def _voxel_idx_to_world_center(voxel_idx: np.ndarray, origin: np.ndarray, voxel_size: float) -> np.ndarray:
    """Convert voxel indices to center points in world coordinates."""

    voxel_idx = np.asarray(voxel_idx, dtype=np.float32).reshape(-1, 3)
    if voxel_idx.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return (np.asarray(origin, dtype=np.float32)[None, :] + (voxel_idx + 0.5) * float(voxel_size)).astype(np.float32)


def _block_idx_to_world_center(block_idx: np.ndarray, grid: dict[str, object]) -> np.ndarray:
    """Convert block indices to center points in world coordinates."""

    return _voxel_idx_to_world_center(
        block_idx,
        origin=np.asarray(grid["origin"], dtype=np.float32),
        voxel_size=float(grid["voxel_size"]) * int(grid["block_size"]),
    )


def _source_grid_from_centers(points: np.ndarray, voxel_size: float) -> dict[str, np.ndarray | float]:
    """Recover the source sparse-grid origin and dimensions from nvblox TSDF centers."""

    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    scene_min = (points.min(axis=0) - 0.5 * float(voxel_size)).astype(np.float32)
    scene_max = (points.max(axis=0) + 0.5 * float(voxel_size)).astype(np.float32)
    extent = np.maximum(scene_max - scene_min, np.full((3,), float(voxel_size), dtype=np.float32))
    voxel_dims = np.rint(extent / float(voxel_size)).astype(np.int32)
    voxel_dims = np.maximum(voxel_dims, np.ones((3,), dtype=np.int32))
    return {
        "origin": scene_min.astype(np.float32),
        "voxel_dims": voxel_dims.astype(np.int32),
        "voxel_size": float(voxel_size),
    }


def _pack_voxel_idx(voxel_idx: np.ndarray, voxel_dims: np.ndarray) -> np.ndarray:
    """Flatten 3D voxel indices into keys for deduplication."""

    idx64 = np.asarray(voxel_idx, dtype=np.int64).reshape(-1, 3)
    dims64 = np.asarray(voxel_dims, dtype=np.int64).reshape(3)
    # Linear index: x + dim_x * (y + dim_y * z).
    return idx64[:, 0] + dims64[0] * (idx64[:, 1] + dims64[1] * idx64[:, 2])


def _reduce_voxels(
    voxel_idx: np.ndarray,
    tsdf_values: np.ndarray,
    voxel_dims: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep the sample nearest the surface when multiple TSDF samples share a voxel."""

    if voxel_idx.shape[0] == 0:
        empty_idx = np.zeros((0, 3), dtype=np.int32)
        empty_val = np.zeros((0,), dtype=np.float32)
        return empty_idx, empty_val, empty_val, empty_val

    keys = _pack_voxel_idx(voxel_idx, voxel_dims)

    # Minimum |TSDF| is nearest the surface and best represents voxel occupancy.
    order = np.argsort(np.abs(tsdf_values), kind="stable")
    ordered_keys = keys[order]
    ordered_idx = voxel_idx[order]
    ordered_tsdf = tsdf_values[order]
    representative_keys, first_pos = np.unique(ordered_keys, return_index=True)
    representative_idx = ordered_idx[first_pos].astype(np.int32)
    representative_tsdf = ordered_tsdf[first_pos].astype(np.float32)

    # Min/max reveal whether a voxel contains both positive and negative TSDF samples.
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    min_tsdf = np.full((unique_keys.shape[0],), np.inf, dtype=np.float32)
    max_tsdf = np.full((unique_keys.shape[0],), -np.inf, dtype=np.float32)
    np.minimum.at(min_tsdf, inverse, tsdf_values)
    np.maximum.at(max_tsdf, inverse, tsdf_values)

    if not np.array_equal(representative_keys, unique_keys):
        raise RuntimeError("TSDF 体素合并时 key 顺序不一致。")
    return representative_idx, representative_tsdf, min_tsdf, max_tsdf


def _interpolate_zero_crossing(
    first_points: np.ndarray,
    second_points: np.ndarray,
    first_values: np.ndarray,
    second_values: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate the TSDF zero crossing along mesh edges."""

    denominator = first_values - second_values
    weight = np.divide(
        first_values,
        denominator,
        out=np.full_like(first_values, 0.5, dtype=np.float32),
        where=np.abs(denominator) > 1.0e-12,
    )
    weight = np.clip(weight, 0.0, 1.0)
    return (
        first_points
        + weight[:, None] * (second_points - first_points)
    ).astype(np.float32)


def _triangulate_tetrahedra(
    cube_points: np.ndarray,
    cube_values: np.ndarray,
) -> list[np.ndarray]:
    """Return zero-level triangles for complete observed TSDF cubes."""

    triangle_parts: list[np.ndarray] = []
    for tetrahedron in CUBE_TETRAHEDRA:
        tetra_idx = np.asarray(tetrahedron, dtype=np.int32)
        points = cube_points[:, tetra_idx]
        values = cube_values[:, tetra_idx]
        inside = values < 0.0
        inside_count = inside.sum(axis=1)

        # One vertex on one side of the surface produces one triangle. The
        # three-inside case is symmetric, so its lone outside vertex is used.
        for lone_is_inside in (True, False):
            target_count = 1 if lone_is_inside else 3
            lone_mask = inside if lone_is_inside else ~inside
            for lone in range(4):
                rows = (inside_count == target_count) & lone_mask[:, lone]
                if not np.any(rows):
                    continue
                other = [index for index in range(4) if index != lone]
                triangle_parts.append(
                    np.stack(
                        [
                            _interpolate_zero_crossing(
                                points[rows, lone],
                                points[rows, endpoint],
                                values[rows, lone],
                                values[rows, endpoint],
                            )
                            for endpoint in other
                        ],
                        axis=1,
                    )
                )

        # Two vertices on each side produce a quad, split consistently into
        # two triangles. Enumerating the six possible inside pairs is small
        # and avoids a per-cube Python loop.
        for first_inside in range(3):
            for second_inside in range(first_inside + 1, 4):
                rows = (
                    (inside_count == 2)
                    & inside[:, first_inside]
                    & inside[:, second_inside]
                )
                if not np.any(rows):
                    continue
                outside = [
                    index
                    for index in range(4)
                    if index not in (first_inside, second_inside)
                ]
                p00 = _interpolate_zero_crossing(
                    points[rows, first_inside],
                    points[rows, outside[0]],
                    values[rows, first_inside],
                    values[rows, outside[0]],
                )
                p01 = _interpolate_zero_crossing(
                    points[rows, first_inside],
                    points[rows, outside[1]],
                    values[rows, first_inside],
                    values[rows, outside[1]],
                )
                p10 = _interpolate_zero_crossing(
                    points[rows, second_inside],
                    points[rows, outside[0]],
                    values[rows, second_inside],
                    values[rows, outside[0]],
                )
                p11 = _interpolate_zero_crossing(
                    points[rows, second_inside],
                    points[rows, outside[1]],
                    values[rows, second_inside],
                    values[rows, outside[1]],
                )
                triangle_parts.extend(
                    (
                        np.stack((p00, p01, p11), axis=1),
                        np.stack((p00, p11, p10), axis=1),
                    )
                )
    return triangle_parts


def extract_tsdf_surface_mesh(
    source_idx: np.ndarray,
    source_tsdf: np.ndarray,
    source_grid: dict[str, np.ndarray | float],
) -> tuple[np.ndarray, np.ndarray, dict[str, int | str]]:
    """Extract the observed TSDF zero level set as a triangle mesh."""

    source_idx = np.asarray(source_idx, dtype=np.int32).reshape(-1, 3)
    source_tsdf = np.asarray(source_tsdf, dtype=np.float32).reshape(-1)
    source_dims = np.asarray(source_grid["voxel_dims"], dtype=np.int32)
    source_keys = _pack_voxel_idx(source_idx, source_dims)
    order = np.argsort(source_keys, kind="stable")
    source_keys = source_keys[order]
    source_tsdf = source_tsdf[order]
    candidate = source_idx[
        np.logical_and(source_idx >= 0, source_idx < (source_dims - 1)[None]).all(axis=1)
    ]
    triangle_parts: list[np.ndarray] = []
    complete_cube_count = 0
    surface_cube_count = 0
    source_origin = np.asarray(source_grid["origin"], dtype=np.float32)
    source_voxel_size = float(source_grid["voxel_size"])

    for start in range(0, candidate.shape[0], SOURCE_BLOCK_CHUNK):
        base_idx = candidate[start:start + SOURCE_BLOCK_CHUNK]
        corner_idx = base_idx[:, None, :] + CUBE_CORNER_OFFSETS[None, :, :]
        query_keys = _pack_voxel_idx(corner_idx.reshape(-1, 3), source_dims).reshape(-1, 8)
        positions = np.searchsorted(source_keys, query_keys)
        observed = positions < source_keys.shape[0]
        safe_positions = np.minimum(positions, max(0, source_keys.shape[0] - 1))
        if source_keys.shape[0]:
            observed &= source_keys[safe_positions] == query_keys
        complete = observed.all(axis=1)
        if not np.any(complete):
            continue
        complete_cube_count += int(np.count_nonzero(complete))
        complete_idx = base_idx[complete]
        complete_values = source_tsdf[positions[complete]]
        negative = complete_values < 0.0
        crosses_surface = negative.any(axis=1) & (~negative).any(axis=1)
        if not np.any(crosses_surface):
            continue
        surface_cube_count += int(np.count_nonzero(crosses_surface))
        surface_idx = complete_idx[crosses_surface]
        cube_points = (
            source_origin[None, None, :]
            + (
                surface_idx[:, None, :].astype(np.float32)
                + CUBE_CORNER_OFFSETS[None].astype(np.float32)
                + 0.5
            )
            * source_voxel_size
        ).astype(np.float32)
        triangle_parts.extend(
            _triangulate_tetrahedra(
                cube_points,
                complete_values[crosses_surface],
            )
        )

    if not triangle_parts:
        raise RuntimeError("TSDF has no complete observed cubes crossing the zero level set.")
    triangle_vertices = np.concatenate(triangle_parts, axis=0).reshape(-1, 3)
    vertices, inverse = np.unique(triangle_vertices, axis=0, return_inverse=True)
    triangles = inverse.reshape(-1, 3).astype(np.int32)
    return vertices.astype(np.float32), triangles, {
        "surface_extraction": "marching_tetrahedra_zero_level",
        "num_complete_observed_cubes": int(complete_cube_count),
        "num_surface_cubes": int(surface_cube_count),
        "num_surface_mesh_vertices": int(vertices.shape[0]),
        "num_surface_mesh_triangles": int(triangles.shape[0]),
    }


def write_triangle_mesh_ply(
    path: Path,
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> Path:
    """Write a compact binary PLY triangle mesh."""

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype="<f4").reshape(-1, 3)
    triangles = np.asarray(triangles, dtype="<i4").reshape(-1, 3)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {vertices.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {triangles.shape[0]}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    face_dtype = np.dtype([("count", "u1"), ("vertices", "<i4", (3,))])
    faces = np.empty(triangles.shape[0], dtype=face_dtype)
    faces["count"] = 3
    faces["vertices"] = triangles
    with path.open("wb") as output:
        output.write(header)
        vertices.tofile(output)
        faces.tofile(output)
    return path


def _apply_affine(points: np.ndarray, affine_4x4: np.ndarray) -> np.ndarray:
    """Apply a 4x4 affine to a point batch without repeating homogeneous-coordinate logic."""

    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    return (points_h @ np.asarray(affine_4x4, dtype=np.float32).T)[:, :3].astype(np.float32)


def _candidate_target_blocks_from_source_blocks(
    *,
    source_idx: np.ndarray,
    source_grid: dict[str, np.ndarray | float],
    target_grid: dict[str, object],
    affine: np.ndarray,
) -> np.ndarray:
    """Transform observed source-block corners to find candidate target blocks."""

    source_dims = np.asarray(source_grid["voxel_dims"], dtype=np.int32)
    source_origin = np.asarray(source_grid["origin"], dtype=np.float32)
    source_voxel_size = float(source_grid["voxel_size"])
    target_origin = np.asarray(target_grid["origin"], dtype=np.float32)
    target_voxel_size = float(target_grid["voxel_size"])
    target_block_size = int(target_grid["block_size"])
    target_block_dims = np.asarray(target_grid["block_dims"], dtype=np.int32)

    source_blocks = np.unique(
        np.floor_divide(np.asarray(source_idx, dtype=np.int32), NVBLOX_BLOCK_SIZE),
        axis=0,
    ).astype(np.int32)
    if source_blocks.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.int32)

    corner_offsets = np.asarray([
        [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
        [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
    ], dtype=np.float32)
    candidate_parts: list[np.ndarray] = []

    for start in range(0, source_blocks.shape[0], SOURCE_BLOCK_CHUNK):
        block_chunk = source_blocks[start:start + SOURCE_BLOCK_CHUNK]
        source_vmin = block_chunk * NVBLOX_BLOCK_SIZE
        source_vmax = np.minimum(
            (block_chunk + 1) * NVBLOX_BLOCK_SIZE,
            source_dims[None, :],
        )
        source_min_world = source_origin[None, :] + source_vmin.astype(np.float32) * source_voxel_size
        source_max_world = source_origin[None, :] + source_vmax.astype(np.float32) * source_voxel_size
        source_corners = (
            source_min_world[:, None, :]
            + corner_offsets[None, :, :] * (source_max_world - source_min_world)[:, None, :]
        ).reshape(-1, 3)
        target_corners = _apply_affine(source_corners, affine).reshape(-1, 8, 3)
        target_min = target_corners.min(axis=1)
        target_max = target_corners.max(axis=1)

        # Convert the AABB to target voxels, then blocks, and finally apply block padding.
        target_vmin = np.floor((target_min - target_origin[None, :]) / target_voxel_size).astype(np.int32)
        target_vmax = np.floor((target_max - target_origin[None, :]) / target_voxel_size).astype(np.int32)
        target_bmin = np.floor_divide(target_vmin, target_block_size) - 1
        target_bmax = np.floor_divide(target_vmax, target_block_size) + 1
        target_bmin = np.maximum(target_bmin, np.zeros((1, 3), dtype=np.int32))
        target_bmax = np.minimum(target_bmax, (target_block_dims - 1)[None, :])

        for bmin, bmax in zip(target_bmin, target_bmax):
            if np.any(bmin > bmax):
                continue
            xs = np.arange(int(bmin[0]), int(bmax[0]) + 1, dtype=np.int32)
            ys = np.arange(int(bmin[1]), int(bmax[1]) + 1, dtype=np.int32)
            zs = np.arange(int(bmin[2]), int(bmax[2]) + 1, dtype=np.int32)
            candidate_parts.append(np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3))

    if not candidate_parts:
        return np.zeros((0, 3), dtype=np.int32)
    return np.unique(np.concatenate(candidate_parts, axis=0), axis=0).astype(np.int32)


def _target_voxels_from_blocks(block_idx: np.ndarray, target_grid: dict[str, object]) -> np.ndarray:
    """Expand target blocks into valid target-voxel indices."""

    block_idx = np.asarray(block_idx, dtype=np.int32).reshape(-1, 3)
    if block_idx.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.int32)
    block_size = int(target_grid["block_size"])
    target_dims = np.asarray(target_grid["voxel_dims"], dtype=np.int32)
    local = np.stack(np.indices((block_size, block_size, block_size), dtype=np.int32), axis=-1).reshape(-1, 3)
    voxel_idx = (block_idx[:, None, :] * block_size + local[None, :, :]).reshape(-1, 3)
    valid = np.logical_and(voxel_idx >= 0, voxel_idx < target_dims[None, :]).all(axis=1)
    return voxel_idx[valid].astype(np.int32)


def _sample_target_grid_from_source(
    *,
    target_grid: dict[str, object],
    candidate_target_blocks: np.ndarray,
    source_grid: dict[str, np.ndarray | float],
    source_keys: np.ndarray,
    source_tsdf: np.ndarray,
    inverse_affine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Backward-sample target centers from the continuous sparse source TSDF."""

    target_dims = np.asarray(target_grid["voxel_dims"], dtype=np.int32)
    total_voxels = int(np.prod(target_dims.astype(np.int64)))
    source_origin = np.asarray(source_grid["origin"], dtype=np.float32)
    source_dims = np.asarray(source_grid["voxel_dims"], dtype=np.int32)
    source_voxel_size = float(source_grid["voxel_size"])
    observed_idx_parts: list[np.ndarray] = []
    observed_tsdf_parts: list[np.ndarray] = []
    matched_count = 0
    candidate_voxel_count = 0

    blocks_per_batch = max(1, BACKWARD_BATCH_VOXELS // (int(target_grid["block_size"]) ** 3))
    for start in range(0, candidate_target_blocks.shape[0], blocks_per_batch):
        block_chunk = candidate_target_blocks[start:start + blocks_per_batch]
        target_idx = _target_voxels_from_blocks(block_chunk, target_grid)
        if target_idx.shape[0] == 0:
            continue
        candidate_voxel_count += int(target_idx.shape[0])
        target_world = _voxel_idx_to_world_center(
            target_idx,
            np.asarray(target_grid["origin"], dtype=np.float32),
            float(target_grid["voxel_size"]),
        )
        source_world = _apply_affine(target_world, inverse_affine)
        sampled_tsdf, matched = _sample_sparse_tsdf_trilinear(
            source_world,
            source_origin,
            source_voxel_size,
            source_dims,
            source_keys,
            source_tsdf,
        )
        if not np.any(matched):
            continue

        matched_count += int(np.count_nonzero(matched))
        matched_target_idx = target_idx[matched]
        matched_tsdf = sampled_tsdf[matched]
        observed_idx_parts.append(matched_target_idx.astype(np.int32))
        observed_tsdf_parts.append(matched_tsdf.astype(np.float32))

    observed_idx = (
        np.concatenate(observed_idx_parts, axis=0)
        if observed_idx_parts
        else np.zeros((0, 3), dtype=np.int32)
    )
    observed_tsdf = (
        np.concatenate(observed_tsdf_parts, axis=0)
        if observed_tsdf_parts
        else np.zeros((0,), dtype=np.float32)
    )
    occupied_idx = observed_idx[
        observed_tsdf <= OCCUPIED_DISTANCE_THRESHOLD
    ].astype(np.int32)
    free_idx = observed_idx[
        observed_tsdf > FREE_DISTANCE_THRESHOLD
    ].astype(np.int32)
    return occupied_idx, free_idx, observed_idx, observed_tsdf, {
        "num_target_voxels": total_voxels,
        "num_candidate_target_blocks": int(candidate_target_blocks.shape[0]),
        "num_candidate_target_voxels": int(candidate_voxel_count),
        "num_backward_matched_voxels": int(matched_count),
        "num_backward_unknown_candidate_voxels": int(candidate_voxel_count - matched_count),
        "num_backward_unknown_voxels": int(total_voxels - matched_count),
    }


def _active_blocks(occupied_idx: np.ndarray, free_idx: np.ndarray, grid: dict[str, object]) -> np.ndarray:
    """Build active blocks over observed space, the union of occupied and free voxels."""

    parts = []
    block_size = int(grid["block_size"])
    if occupied_idx.shape[0] > 0:
        parts.append(np.floor_divide(occupied_idx, block_size).astype(np.int32))
    if free_idx.shape[0] > 0:
        parts.append(np.floor_divide(free_idx, block_size).astype(np.int32))
    if not parts:
        return np.zeros((0, 3), dtype=np.int32)
    return np.unique(np.concatenate(parts, axis=0), axis=0).astype(np.int32)


def _save_outputs(
    output_prefix: Path,
    grid: dict[str, object],
    occupied_idx: np.ndarray,
    free_idx: np.ndarray,
    surface_idx: np.ndarray,
    active_block_idx: np.ndarray,
    stats: dict[str, object],
    map_frame_transform: np.ndarray,
) -> tuple[Path, Path]:
    """Save the NPZ map and JSON statistics."""

    output_prefix = normalize_output_prefix(output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = output_prefix.with_suffix(".npz")
    json_path = output_prefix.with_suffix(".json")

    active_block_world = _block_idx_to_world_center(active_block_idx, grid)
    occupied_world = _voxel_idx_to_world_center(occupied_idx, grid["origin"], float(grid["voxel_size"]))
    free_world = _voxel_idx_to_world_center(free_idx, grid["origin"], float(grid["voxel_size"]))
    surface_world = _voxel_idx_to_world_center(
        surface_idx,
        grid["origin"],
        float(grid["voxel_size"]),
    )

    np.savez_compressed(
        npz_path,
        voxel_size=np.array([grid["voxel_size"]], dtype=np.float32),
        block_size=np.array([grid["block_size"]], dtype=np.int32),
        origin=np.asarray(grid["origin"], dtype=np.float32),
        scene_min=np.asarray(grid["scene_min"], dtype=np.float32),
        scene_max=np.asarray(grid["scene_max"], dtype=np.float32),
        voxel_dims=np.asarray(grid["voxel_dims"], dtype=np.int32),
        block_dims=np.asarray(grid["block_dims"], dtype=np.int32),
        active_block_idx=active_block_idx.astype(np.int32),
        active_block_world=active_block_world.astype(np.float32),
        occupied_voxel_idx=occupied_idx.astype(np.int32),
        occupied_voxel_world=occupied_world.astype(np.float32),
        free_voxel_idx=free_idx.astype(np.int32),
        free_voxel_world=free_world.astype(np.float32),
        # This mesh-derived zero level set is the only nvblox representation
        # used for geometry evaluation. The thick occupied TSDF band above is
        # retained independently for planning.
        surface_voxel_idx=surface_idx.astype(np.int32),
        surface_voxel_world=surface_world.astype(np.float32),
        implicit_free=np.array([0], dtype=np.uint8),
        map_from_training_world=np.asarray(map_frame_transform, dtype=np.float32),
        training_world_from_map=np.linalg.inv(
            np.asarray(map_frame_transform, dtype=np.float32)
        ).astype(np.float32),
    )

    stats = dict(stats)
    stats["output_npz"] = str(npz_path)
    stats["output_json"] = str(json_path)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    return npz_path, json_path


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    """Run the complete TSDF PLY to ``voxel_map_tsdf`` conversion."""

    started_at = time.perf_counter()
    source_voxel_size = float(args.source_voxel_size)
    target_voxel_size = float(args.voxel_size)
    block_size = int(args.block_size)
    scene_padding = float(args.scene_padding)
    if not np.isfinite(source_voxel_size) or source_voxel_size <= 0.0:
        raise ValueError(
            f"source_voxel_size must be positive and finite, got {source_voxel_size}."
        )
    if not np.isfinite(target_voxel_size) or target_voxel_size <= 0.0:
        raise ValueError(
            f"voxel_size must be positive and finite, got {target_voxel_size}."
        )
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    if not np.isfinite(scene_padding) or scene_padding < 0.0:
        raise ValueError(
            f"scene_padding must be non-negative and finite, got {scene_padding}."
        )

    tsdf_ply = args.tsdf_ply.expanduser().resolve()
    if not tsdf_ply.is_file():
        raise FileNotFoundError(f"找不到 TSDF PLY: {tsdf_ply}")

    loading_started_at = time.perf_counter()
    points, tsdf_values = load_tsdf_ply(tsdf_ply)
    loading_sec = float(time.perf_counter() - loading_started_at)
    calculation_started_at = time.perf_counter()

    source_grid = _source_grid_from_centers(points, source_voxel_size)
    source_dims = np.asarray(source_grid["voxel_dims"], dtype=np.int32)
    source_idx_all = _voxel_center_to_idx(
        points,
        np.asarray(source_grid["origin"], dtype=np.float32),
        source_voxel_size,
    )
    source_in_grid = np.logical_and(source_idx_all >= 0, source_idx_all < source_dims[None, :]).all(axis=1)
    source_idx = source_idx_all[source_in_grid]
    source_tsdf_values = tsdf_values[source_in_grid]
    source_idx, source_tsdf_values, source_min_tsdf, source_max_tsdf = _reduce_voxels(
        source_idx,
        source_tsdf_values,
        source_dims,
    )
    source_keys = _pack_voxel_idx(source_idx, source_dims)
    source_order = np.argsort(source_keys, kind="stable")
    source_keys = source_keys[source_order]
    source_idx = source_idx[source_order]
    source_tsdf_values = source_tsdf_values[source_order]
    source_conflicting = np.logical_and(
        source_min_tsdf <= OCCUPIED_DISTANCE_THRESHOLD,
        source_max_tsdf > FREE_DISTANCE_THRESHOLD,
    )
    source_to_training_affine_path = getattr(
        args,
        "source_to_training_affine",
        None,
    )
    if source_to_training_affine_path is None:
        source_to_training_affine_path = _discover_source_to_training_affine(
            tsdf_ply
        )
    if source_to_training_affine_path is None:
        training_affine, affine_info = _load_source_to_training_affine(
            args.training_config,
            args.source_coordinate_transform,
        )
    else:
        training_affine, affine_info = _load_source_to_training_affine_json(
            source_to_training_affine_path
        )
        if getattr(args, "source_to_training_affine", None) is None:
            affine_info["source"] = "auto_discovered_source_to_training_affine_json"
    map_transform, map_transform_info = load_map_frame_transform(
        getattr(args, "map_frame_transform", None)
    )
    affine = (
        np.asarray(map_transform, dtype=np.float32)
        @ np.asarray(training_affine, dtype=np.float32)
    ).astype(np.float32)
    transformed_source_voxel_size, affine_axis_scales = _transformed_voxel_size(
        source_voxel_size,
        affine,
    )
    grid, grid_num_positions = _grid_from_checkpoint(
        args.grid_checkpoint,
        target_voxel_size,
        block_size,
        scene_padding,
        map_frame_transform=map_transform,
    )
    voxel_dims = np.asarray(grid["voxel_dims"], dtype=np.int32)
    total_voxels = int(np.prod(voxel_dims.astype(np.int64)))
    candidate_target_blocks = _candidate_target_blocks_from_source_blocks(
        source_idx=source_idx,
        source_grid=source_grid,
        target_grid=grid,
        affine=affine,
    )
    (
        occupied_idx,
        free_idx,
        target_observed_idx,
        target_observed_tsdf,
        backward_stats,
    ) = _sample_target_grid_from_source(
        target_grid=grid,
        candidate_target_blocks=candidate_target_blocks,
        source_grid=source_grid,
        source_keys=source_keys,
        source_tsdf=source_tsdf_values,
        inverse_affine=np.linalg.inv(
            np.asarray(affine, dtype=np.float32)
        ).astype(np.float32),
    )
    surface_mesh_vertices, surface_mesh_triangles, surface_mesh_stats = (
        extract_tsdf_surface_mesh(
            target_observed_idx,
            target_observed_tsdf,
            grid,
        )
    )
    surface_idx = _voxelize_target_surface_vertices(surface_mesh_vertices, grid)
    active_block_idx = _active_blocks(occupied_idx, free_idx, grid)

    calculation_sec = float(time.perf_counter() - calculation_started_at)
    stats = {
        "input_tsdf_ply": str(tsdf_ply),
        "grid_checkpoint": str(args.grid_checkpoint.expanduser().resolve()),
        "training_config": str(args.training_config.expanduser().resolve()),
        # Kept under the legacy key because ESDF/path consumers read it as the
        # complete source-TSDF to map-frame transform.
        "dataset_affine": np.asarray(affine, dtype=np.float32).tolist(),
        "training_dataset_affine": np.asarray(training_affine, dtype=np.float32).tolist(),
        "dataset_affine_info": affine_info,
        "map_frame": transform_metadata(map_transform, map_transform_info),
        "grid_source": "checkpoint_raw_positions_canonical",
        "grid_num_checkpoint_positions": int(grid_num_positions),
        "voxel_size": float(grid["voxel_size"]),
        "source_voxel_size": source_voxel_size,
        "transformed_source_voxel_size": float(transformed_source_voxel_size),
        "target_voxel_size": target_voxel_size,
        "transformed_axis_scales": affine_axis_scales,
        "block_size": int(grid["block_size"]),
        "scene_padding": scene_padding,
        "occupied_distance_threshold": float(OCCUPIED_DISTANCE_THRESHOLD),
        "free_distance_threshold": float(FREE_DISTANCE_THRESHOLD),
        "sampling": "source_blocks_to_target_candidates_then_trilinear_backward",
        "geometry_surface": (
            "zero-level mesh extracted from the interpolated target-grid TSDF"
        ),
        "surface_voxelizer": "target_mesh_vertices_snap_then_floor_to_common_grid",
        **surface_mesh_stats,
        "num_surface_voxels": int(surface_idx.shape[0]),
        "backward_batch_voxels": int(BACKWARD_BATCH_VOXELS),
        "source_block_chunk": int(SOURCE_BLOCK_CHUNK),
        "num_tsdf_samples_raw": int(points.shape[0]),
        "num_source_samples_inside_grid": int(np.count_nonzero(source_in_grid)),
        "num_source_samples_outside_grid": int(source_in_grid.size - np.count_nonzero(source_in_grid)),
        "num_source_observed_voxels": int(source_idx.shape[0]),
        "num_source_conflicting_voxels": int(np.count_nonzero(source_conflicting)),
        **backward_stats,
        "num_occupied_voxels": int(occupied_idx.shape[0]),
        "num_free_voxels": int(free_idx.shape[0]),
        "num_explicit_free_voxels_saved": int(free_idx.shape[0]),
        "num_active_blocks": int(active_block_idx.shape[0]),
        "num_voxels_total": int(total_voxels),
        "origin": np.asarray(grid["origin"], dtype=np.float32).tolist(),
        "scene_min": np.asarray(grid["scene_min"], dtype=np.float32).tolist(),
        "scene_max": np.asarray(grid["scene_max"], dtype=np.float32).tolist(),
        "voxel_dims": voxel_dims.astype(np.int32).tolist(),
        "block_dims": np.asarray(grid["block_dims"], dtype=np.int32).tolist(),
        "implicit_free": False,
    }
    saving_started_at = time.perf_counter()
    normalized_output_prefix = normalize_output_prefix(
        args.output_prefix
    ).expanduser().resolve()
    surface_mesh_path = (
        args.surface_mesh_output.expanduser().resolve()
        if args.surface_mesh_output is not None
        else normalized_output_prefix.with_name(
            f"{normalized_output_prefix.name}_surface_mesh.ply"
        )
    )
    write_triangle_mesh_ply(
        surface_mesh_path,
        surface_mesh_vertices,
        surface_mesh_triangles,
    )
    stats["surface_mesh_ply"] = str(surface_mesh_path)
    stats["surface_mesh_coordinate_frame"] = "map_frame"
    npz_path, json_path = _save_outputs(
        args.output_prefix,
        grid,
        occupied_idx,
        free_idx,
        surface_idx,
        active_block_idx,
        stats,
        map_transform,
    )
    saving_sec = float(time.perf_counter() - saving_started_at)
    wall_total_sec = float(time.perf_counter() - started_at)
    stats["output_npz"] = str(npz_path)
    stats["output_json"] = str(json_path)
    stats["timing_sec"] = {
        "loading": loading_sec,
        "calculation": calculation_sec,
        "saving": saving_sec,
        "categorized_total": loading_sec + calculation_sec + saving_sec,
        "wall_total": wall_total_sec,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    return stats


def main() -> None:
    """Run the command-line entry point."""

    stats = run_pipeline(parse_args())
    timing = stats["timing_sec"]
    print(
        "[TSDFVoxelMap] done: "
        f"samples={stats['num_tsdf_samples_raw']}, "
        f"occupied_voxels={stats['num_occupied_voxels']}, "
        f"surface_voxels={stats['num_surface_voxels']}, "
        f"free_voxels={stats['num_free_voxels']}, "
        f"calculation={timing['calculation']:.3f}s, "
        f"total={timing['wall_total']:.3f}s"
    )
    print(
        f"[TSDFVoxelMap] output: {stats['output_npz']} "
        f"(stats: {stats['output_json']})"
    )


if __name__ == "__main__":
    main()
