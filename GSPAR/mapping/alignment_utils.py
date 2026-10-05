#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Coordinate transforms shared by the current map-building scripts.

Read COLMAP cameras and the training configuration, then reproduce NerfICG's
world transform, PCA, and rescaling as a 4x4 affine from COLMAP to 3DGS space.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np


AXIS_TRANSFORMS = {
    "xyz": (1, 2, 3),
    "xnzy": (1, -3, 2),
}


def ensure_project_paths(file_path: str | Path) -> Path:
    """Find the project root and add its shared source directories to ``sys.path``."""

    file_path = Path(file_path).resolve()
    for parent in [file_path.parent] + list(file_path.parents):
        if (parent / "src").is_dir():
            for path in (parent, parent / "src", parent / "GSPAR" / "mapping"):
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))
            return parent
    raise RuntimeError(f"找不到项目根目录: {file_path}")


def world_transform_matrix_from_name(name: str) -> np.ndarray:
    """Represent a dataset axis permutation and flip as a 4x4 matrix."""

    key = name.strip().lower()
    spec = AXIS_TRANSFORMS[key]
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = 0.0
    for row, axis_spec in enumerate(spec):
        axis = abs(axis_spec) - 1
        mat[row, axis] = 1.0 if axis_spec > 0 else -1.0
    return mat


def apply_affine(points: np.ndarray, affine_4x4: np.ndarray) -> np.ndarray:
    """Apply a 4x4 affine transform to an ``N x 3`` point array."""

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    return (pts_h @ np.asarray(affine_4x4, dtype=np.float32).T)[:, :3].astype(np.float32)


def _read_next_bytes(fid, num_bytes: int, fmt: str):
    return struct.unpack("<" + fmt, fid.read(num_bytes))


def read_colmap_images_binary(images_bin: Path) -> list[dict[str, Any]]:
    """Read the quaternion, translation, and name of each COLMAP image."""

    images: list[dict[str, Any]] = []
    with images_bin.open("rb") as fid:
        num_images = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            props = _read_next_bytes(fid, 64, "idddddddi")
            name = ""
            char = _read_next_bytes(fid, 1, "c")[0]
            while char != b"\x00":
                name += char.decode("utf-8")
                char = _read_next_bytes(fid, 1, "c")[0]
            num_points2d = _read_next_bytes(fid, 8, "Q")[0]
            fid.read(24 * num_points2d)
            images.append({
                "qvec": np.asarray(props[1:5], dtype=np.float64),
                "tvec": np.asarray(props[5:8], dtype=np.float64),
                "name": name,
            })
    return images


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert a COLMAP quaternion to a rotation matrix."""

    q0, q1, q2, q3 = np.asarray(qvec, dtype=np.float64).tolist()
    return np.array([
        [1 - 2 * q2 * q2 - 2 * q3 * q3, 2 * q1 * q2 - 2 * q0 * q3, 2 * q3 * q1 + 2 * q0 * q2],
        [2 * q1 * q2 + 2 * q0 * q3, 1 - 2 * q1 * q1 - 2 * q3 * q3, 2 * q2 * q3 - 2 * q0 * q1],
        [2 * q3 * q1 - 2 * q0 * q2, 2 * q2 * q3 + 2 * q0 * q1, 1 - 2 * q1 * q1 - 2 * q2 * q2],
    ], dtype=np.float64)


def colmap_w2c_to_c2w(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert a COLMAP world-to-camera pose to camera-to-world."""

    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = qvec_to_rotmat(qvec)
    w2c[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return np.linalg.inv(w2c)


def _transform_poses_pca(poses: np.ndarray, rescale: bool) -> np.ndarray:
    """Reproduce ``src/Datasets/utils.py::transform_poses_pca``."""

    poses = np.asarray(poses, dtype=np.float64)
    colmap_to_opengl = np.diag([1.0, -1.0, -1.0, 1.0])
    poses_gl = poses @ colmap_to_opengl
    centers = poses_gl[:, :3, 3]
    center_mean = centers.mean(axis=0)
    centered = centers - center_mean

    eigval, eigvec = np.linalg.eig(centered.T @ centered)
    eigvec = eigvec[:, np.argsort(eigval)[::-1]]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag([1.0, 1.0, -1.0]) @ rot

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot
    transform[:3, 3] = rot @ -center_mean
    recentered = (transform @ poses_gl)[:, :3, :4]

    if recentered.mean(axis=0)[2, 1] < 0:
        flip = np.diag([1.0, -1.0, -1.0, 1.0])
        transform = flip @ transform
        recentered = flip[:3, :3] @ recentered

    if rescale:
        scale = 1.0 / np.max(np.abs(recentered[:, :3, 3]))
        transform = np.diag([scale, scale, scale, 1.0]) @ transform

    aligned_to_colmap = world_transform_matrix_from_name("XnZY").astype(np.float64)
    return (aligned_to_colmap @ transform).astype(np.float32)


def load_training_dataset_affine(
    training_config: Path,
    project_root: Path,
    world_transform_override: str | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Derive the affine from raw COLMAP space to trained 3DGS space."""

    import yaml

    training_config = training_config.expanduser().resolve()
    with training_config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_type = str(cfg["GLOBAL"]["DATASET_TYPE"])
    if world_transform_override is None or world_transform_override.strip().lower() == "auto":
        world_name = "XYZ"
        transform_source = "current_dataset_convention"
    else:
        override_key = world_transform_override.strip().lower()
        if override_key not in AXIS_TRANSFORMS:
            raise ValueError(
                f"不支持的 world transform: {world_transform_override}; "
                f"可选值为 auto/{'/'.join(AXIS_TRANSFORMS)}"
            )
        world_name = override_key.upper() if override_key == "xyz" else "XnZY"
        transform_source = "cli_override"
    configured_dataset_path = Path(str(cfg["DATASET"]["PATH"])).expanduser()
    if configured_dataset_path.is_absolute():
        dataset_path = configured_dataset_path
    else:
        dataset_path = project_root.expanduser().resolve() / configured_dataset_path
    images_bin = dataset_path / "sparse" / "0" / "images.bin"
    images = read_colmap_images_binary(images_bin)
    if not images:
        raise RuntimeError(f"COLMAP images.bin 没有注册图像: {images_bin}")

    world_mat = world_transform_matrix_from_name(world_name)
    affine = world_mat.copy()

    apply_pca = bool(cfg["DATASET"].get("APPLY_PCA", False))
    apply_pca_rescale = bool(cfg["DATASET"].get("APPLY_PCA_RESCALE", False))
    if apply_pca:
        c2ws = []
        for image in images:
            c2w = colmap_w2c_to_c2w(image["qvec"], image["tvec"])
            c2ws.append(c2w)
        affine = _transform_poses_pca(
            np.stack(c2ws, axis=0),
            rescale=apply_pca_rescale,
        )

    return affine.astype(np.float32), {
        "training_config": str(training_config),
        "dataset_type": dataset_type,
        "dataset_path": str(dataset_path),
        "world_transform": world_name,
        "world_transform_source": transform_source,
        "apply_pca": apply_pca,
        "apply_pca_rescale": apply_pca_rescale,
        "num_colmap_images": int(len(images)),
    }
