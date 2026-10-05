#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from GSPAR.mapping.map_frame_transform import apply_transform, validate_rigid_transform


@dataclass
class GaussianSet:
    positions: np.ndarray  # [N, 3]
    scales: np.ndarray  # [N, 3], physical scales
    rotations: np.ndarray  # [N, 3, 3], local->world rotation matrix
    opacities: np.ndarray  # [N], physical alpha in [0,1]


def transform_gaussians(gaussians: GaussianSet, transform: np.ndarray) -> GaussianSet:
    """Apply a rigid map-frame transform to positions and local-to-world rotations."""

    matrix = validate_rigid_transform(transform)
    return GaussianSet(
        positions=apply_transform(gaussians.positions, matrix),
        scales=np.asarray(gaussians.scales, dtype=np.float32),
        rotations=np.einsum(
            "ij,njk->nik",
            matrix[:3, :3],
            np.asarray(gaussians.rotations, dtype=np.float32),
        ).astype(np.float32),
        opacities=np.asarray(gaussians.opacities, dtype=np.float32),
    )


def ensure_project_src_path() -> None:
    file_path = Path(__file__).resolve()
    for parent in [file_path.parent] + list(file_path.parents):
        src = parent / "src"
        if src.is_dir():
            if str(src) not in sys.path:
                sys.path.insert(0, str(src))
            return
    raise RuntimeError(f"Could not locate project src from {file_path}")


def _find_state_key(state_dict: dict, suffixes: Iterable[str]) -> str:
    keys = list(state_dict.keys())
    for suffix in suffixes:
        for key in keys:
            if key.endswith(suffix):
                return key
    raise KeyError(f"Cannot find key suffix in checkpoint: {tuple(suffixes)}")


def _load_checkpoint_raw(checkpoint_path: Path) -> dict:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to read .pt checkpoint.") from exc

    ensure_project_src_path()
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"You are using `torch\.load` with `weights_only=False`",
                category=FutureWarning,
            )
            return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def _load_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for voxel-map preprocessing.") from exc
    return torch


def _preferred_device(torch_module, requested_device: str | None = None):
    if requested_device is not None:
        device = torch_module.device(requested_device)
        if device.type == "cuda" and not torch_module.cuda.is_available():
            raise RuntimeError(f"requested CUDA device is unavailable: {device}")
        return device
    return torch_module.device(
        "cuda" if torch_module.cuda.is_available() else "cpu"
    )


def _decode_physical_gaussian_params(torch_module, scale_raw_t, opa_raw_t):
    scales_are_physical = bool(torch_module.all(scale_raw_t > 0.0).item())
    opacities_are_physical = bool(torch_module.all((opa_raw_t >= 0.0) & (opa_raw_t <= 1.0)).item())
    if scales_are_physical and opacities_are_physical:
        return scale_raw_t.clamp_min(1e-12), opa_raw_t.reshape(-1).clamp_(0.0, 1.0)
    return torch_module.exp(scale_raw_t), torch_module.sigmoid(opa_raw_t).reshape(-1).clamp_(0.0, 1.0)


def _quat_wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    q = quat_wxyz.astype(np.float64)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    q = q / n
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - z * w)
    r02 = 2.0 * (x * z + y * w)
    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - x * w)
    r20 = 2.0 * (x * z - y * w)
    r21 = 2.0 * (y * z + x * w)
    r22 = 1.0 - 2.0 * (x * x + y * y)

    rot = np.stack([
        np.stack([r00, r01, r02], axis=1),
        np.stack([r10, r11, r12], axis=1),
        np.stack([r20, r21, r22], axis=1),
    ], axis=1)
    return rot.astype(np.float32)


def _to_rotation_matrix(rot_raw: np.ndarray) -> np.ndarray:
    if rot_raw.ndim == 3 and rot_raw.shape[1:] == (3, 3):
        return rot_raw.astype(np.float32)
    if rot_raw.ndim == 2 and rot_raw.shape[1] == 4:
        return _quat_wxyz_to_rotmat(rot_raw)
    raise ValueError(f"Unsupported rotation format: shape={rot_raw.shape}")


def _collect_raw_arrays(
    checkpoint_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ckpt = _load_checkpoint_raw(checkpoint_path)
    if "model_state_dict" not in ckpt:
        raise RuntimeError("Invalid checkpoint: missing 'model_state_dict'.")
    state = ckpt["model_state_dict"]

    pos_key = _find_state_key(state, ("._positions",))
    scale_key = _find_state_key(state, ("._scales",))
    rot_key = _find_state_key(state, ("._rotations",))
    opa_key = _find_state_key(state, ("._opacities",))
    pos = state[pos_key].detach().cpu().numpy().astype(np.float32)
    scale_raw = state[scale_key].detach().cpu().numpy().astype(np.float32)
    rot_raw = state[rot_key].detach().cpu().numpy().astype(np.float32)
    opa_raw = state[opa_key].detach().cpu().numpy().astype(np.float32)
    return pos, scale_raw, rot_raw, opa_raw


def load_gaussian_positions(checkpoint_path: Path) -> np.ndarray:
    pos, _, _, _ = _collect_raw_arrays(checkpoint_path)
    return pos.astype(np.float32)


def load_gaussian_arrays(
    checkpoint_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the four raw Gaussian arrays from a checkpoint."""

    return _collect_raw_arrays(checkpoint_path)


def preprocess_gaussian_arrays(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    max_gaussians: int | None,
    device: str | None = None,
) -> GaussianSet:
    torch = _load_torch()
    device = _preferred_device(torch, device)
    pos, scale_raw, rot_raw, opa_raw = arrays
    rotations = _to_rotation_matrix(rot_raw)
    with torch.no_grad():
        pos_t = torch.as_tensor(pos, dtype=torch.float32, device=device)
        scale_raw_t = torch.as_tensor(scale_raw, dtype=torch.float32, device=device)
        opa_raw_t = torch.as_tensor(opa_raw, dtype=torch.float32, device=device)
        rotations_t = torch.as_tensor(rotations, dtype=torch.float32, device=device)

        # final.pt usually stores physical scales and opacity; intermediate checkpoints may use log-scale and logits.
        scales_t, opacities_t = _decode_physical_gaussian_params(torch, scale_raw_t, opa_raw_t)
        idx_t = torch.arange(pos_t.shape[0], dtype=torch.int64, device=device)

        if max_gaussians is not None and max_gaussians > 0 and int(idx_t.numel()) > max_gaussians:
            _, top_idx_t = torch.topk(opacities_t.index_select(0, idx_t), k=max_gaussians, sorted=True)
            idx_t = idx_t.index_select(0, top_idx_t)

        gauss = GaussianSet(
            positions=pos_t.index_select(0, idx_t).cpu().numpy().astype(np.float32),
            scales=scales_t.index_select(0, idx_t).cpu().numpy().astype(np.float32),
            rotations=rotations_t.index_select(0, idx_t).cpu().numpy().astype(np.float32),
            opacities=opacities_t.index_select(0, idx_t).cpu().numpy().astype(np.float32),
        )

    return gauss


def preprocess_gaussians(
    checkpoint_path: Path,
    max_gaussians: int | None,
    device: str | None = None,
) -> GaussianSet:
    return preprocess_gaussian_arrays(
        load_gaussian_arrays(checkpoint_path),
        max_gaussians,
        device=device,
    )
