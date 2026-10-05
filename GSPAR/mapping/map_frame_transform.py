#!/usr/bin/env python3
"""Load and apply a shared rigid transform for voxel-map construction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


MAP_FRAME_FORMAT = "nerficg_map_frame_transform_v1"


def validate_rigid_transform(transform: np.ndarray) -> np.ndarray:
    """Return a float32 4x4 rigid transform or raise a clear error."""

    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"map-frame transform must be 4x4, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("map-frame transform contains non-finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("map-frame transform has an invalid homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("map-frame transform rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError("map-frame transform rotation must have determinant +1")
    return matrix.astype(np.float32)


def load_map_frame_transform(path: Path | None) -> tuple[np.ndarray, dict[str, object]]:
    """Load ``map_from_training_world``; ``None`` means the identity frame."""

    if path is None:
        return np.eye(4, dtype=np.float32), {
            "enabled": False,
            "source": None,
            "format": MAP_FRAME_FORMAT,
        }

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"map-frame transform not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"map-frame transform must be a JSON object: {resolved}")
    value = payload.get("map_from_training_world", payload.get("transform_4x4"))
    if value is None:
        raise ValueError(
            f"map-frame transform lacks map_from_training_world/transform_4x4: {resolved}"
        )
    matrix = validate_rigid_transform(np.asarray(value, dtype=np.float64))
    return matrix, {
        "enabled": True,
        "source": str(resolved),
        "format": str(payload.get("format", MAP_FRAME_FORMAT)),
        "axis_yaw_deg": payload.get("axis_yaw_deg"),
        "applied_yaw_deg": payload.get("applied_yaw_deg"),
        "rotation_center": payload.get("rotation_center"),
    }


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 column-vector transform to an ``Nx3`` point array."""

    values = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if values.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    matrix = validate_rigid_transform(transform)
    return (
        values @ matrix[:3, :3].T
        + matrix[:3, 3][None, :]
    ).astype(np.float32)


def transform_metadata(transform: np.ndarray, info: dict[str, object]) -> dict[str, object]:
    """Build JSON-safe metadata shared by all voxel map outputs."""

    matrix = validate_rigid_transform(transform)
    return {
        **dict(info),
        "map_from_training_world": matrix.tolist(),
        "training_world_from_map": np.linalg.inv(matrix).astype(np.float32).tolist(),
    }
