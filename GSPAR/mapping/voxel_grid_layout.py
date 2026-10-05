#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Voxel-grid layout helpers shared by map-building scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from GSPAR.mapping.gaussian_occupancy.voxel_preprocess import load_gaussian_positions
from GSPAR.mapping.map_frame_transform import apply_transform


@dataclass
class GridSpec:
    origin: np.ndarray
    scene_min: np.ndarray
    scene_max: np.ndarray
    voxel_dims: np.ndarray
    block_dims: np.ndarray
    voxel_size: float
    block_size: int


def _make_grid(scene_min: np.ndarray, voxel_dims: np.ndarray, voxel_size: float, block_size: int) -> GridSpec:
    """Construct a grid from its minimum corner, dimensions, and voxel size."""

    scene_min = np.asarray(scene_min, dtype=np.float32).reshape(3)
    voxel_dims = np.asarray(voxel_dims, dtype=np.int32).reshape(3)
    block_size = int(block_size)
    return GridSpec(
        origin=scene_min,
        scene_min=scene_min,
        scene_max=(scene_min + voxel_dims.astype(np.float32) * float(voxel_size)).astype(np.float32),
        voxel_dims=voxel_dims,
        block_dims=np.floor_divide(voxel_dims + block_size - 1, block_size).astype(np.int32),
        voxel_size=float(voxel_size),
        block_size=block_size,
    )


def load_reference_grid(npz_path: Path) -> dict[str, object]:
    """Load a reference voxel grid so 3DGS output aligns with it exactly."""

    with np.load(npz_path) as data:
        grid = {
            "origin": np.asarray(data["origin"], dtype=np.float32).reshape(3),
            "scene_min": np.asarray(data["scene_min"], dtype=np.float32).reshape(3),
            "scene_max": np.asarray(data["scene_max"], dtype=np.float32).reshape(3),
            "voxel_dims": np.asarray(data["voxel_dims"], dtype=np.int32).reshape(3),
            "block_dims": np.asarray(data["block_dims"], dtype=np.int32).reshape(3),
            "voxel_size": float(np.asarray(data["voxel_size"]).reshape(-1)[0]),
            "block_size": int(np.asarray(data["block_size"]).reshape(-1)[0]),
        }
        if "map_from_training_world" in data:
            grid["map_from_training_world"] = np.asarray(
                data["map_from_training_world"], dtype=np.float32
            ).reshape(4, 4)
        return grid


def resolve_voxel_size(voxel_size: float | None, voxel_size_json: Path | None) -> tuple[float, str | None]:
    """Resolve the voxel size directly or from transformed TSDF metadata."""

    if voxel_size_json is None:
        return float(voxel_size), None
    json_path = voxel_size_json.expanduser().resolve()
    with json_path.open("r", encoding="utf-8") as f:
        value = json.load(f)["transformed_source_voxel_size"]
    return float(value), str(json_path)


def build_grid_spec_from_positions(
    positions: np.ndarray,
    voxel_size: float,
    block_size: int,
    scene_padding: float,
) -> GridSpec:
    """Build a canonical grid by quantizing point-cloud bounds.

    Each origin component is an integer multiple of ``voxel_size``. Builders
    using the same positions, voxel size, and padding therefore produce the
    same index-to-world mapping without reading each other's voxel maps.
    """

    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    voxel_size = float(voxel_size)
    scene_padding = float(scene_padding)
    block_size = int(block_size)
    if positions.shape[0] == 0:
        raise ValueError("positions must not be empty.")
    if not np.isfinite(positions).all():
        raise ValueError("positions contain non-finite values.")
    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError(f"voxel_size must be positive and finite, got {voxel_size}.")
    if not np.isfinite(scene_padding) or scene_padding < 0.0:
        raise ValueError(f"scene_padding must be non-negative and finite, got {scene_padding}.")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    bounds_min = positions.astype(np.float64).min(axis=0) - scene_padding
    bounds_max = positions.astype(np.float64).max(axis=0) + scene_padding
    # Anchor the finite grid to the infinite training-space lattice v * Z^3.
    # Quantization also avoids sub-voxel origin changes caused by different bounds.
    origin = np.floor(bounds_min / voxel_size) * voxel_size
    voxel_dims = np.maximum(
        np.ceil((bounds_max - origin) / voxel_size).astype(np.int32),
        1,
    )
    return _make_grid(origin.astype(np.float32), voxel_dims, voxel_size, block_size)


def build_grid_spec_from_checkpoint_positions(
    checkpoint_path: Path,
    voxel_size: float,
    block_size: int,
    scene_padding: float,
    map_frame_transform: np.ndarray | None = None,
) -> tuple[GridSpec, int]:
    """Build a canonical TSDF/3DGS grid from raw Gaussian centers."""

    positions = load_gaussian_positions(checkpoint_path)
    if positions.shape[0] == 0:
        raise RuntimeError(f"checkpoint 中没有 Gaussian 位置: {checkpoint_path}")
    if map_frame_transform is not None:
        positions = apply_transform(positions, map_frame_transform)
    grid = build_grid_spec_from_positions(positions, voxel_size, block_size, scene_padding)
    return grid, int(positions.shape[0])
