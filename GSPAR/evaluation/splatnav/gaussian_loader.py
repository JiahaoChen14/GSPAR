"""Load NerfICG Gaussian checkpoints without importing Nerfstudio."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PosixPath
from typing import Any, Iterable

import numpy as np
import torch

from GSPAR.mapping.map_frame_transform import validate_rigid_transform


# NerfICG stores several configuration values in final.pt using this dict-like
# class. Recreating only its qualified name is sufficient to safely unpickle
# the metadata; the planner consumes model_state_dict tensors only.
_CheckpointConfigParameterList = type("ConfigParameterList", (dict,), {})
_CheckpointConfigParameterList.__module__ = "Framework"


@dataclass
class SplatNavGaussians:
    """Tensor attributes consumed by the original SplatNav planner."""

    means: torch.Tensor
    rots: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    colors: torch.Tensor
    covs: torch.Tensor
    covs_inv: torch.Tensor


def _find_state_key(state_dict: dict[str, Any], suffixes: Iterable[str]) -> str:
    for suffix in suffixes:
        for key in state_dict:
            if key.endswith(suffix):
                return key
    raise KeyError(f"cannot find a checkpoint key ending with {tuple(suffixes)}")


def _load_state_dict(checkpoint_path: Path) -> dict[str, Any]:
    try:
        safe_globals = getattr(torch.serialization, "safe_globals", None)
        allowed_globals = [PosixPath, _CheckpointConfigParameterList]
        add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
        if safe_globals is not None:
            with safe_globals(allowed_globals):
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )
        elif add_safe_globals is not None:
            get_safe_globals = getattr(torch.serialization, "get_safe_globals", None)
            clear_safe_globals = getattr(torch.serialization, "clear_safe_globals", None)
            previous = [] if get_safe_globals is None else list(get_safe_globals())
            add_safe_globals(allowed_globals)
            try:
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )
            finally:
                if clear_safe_globals is not None:
                    clear_safe_globals()
                    add_safe_globals(previous)
        else:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
    except TypeError as error:
        raise RuntimeError(
            "this loader requires torch.load(..., weights_only=True) support"
        ) from error
    except Exception as error:
        raise RuntimeError(
            f"could not safely load NerfICG checkpoint {checkpoint_path}: "
            f"{type(error).__name__}: {error}"
        ) from error
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model_state_dict"), dict
    ):
        raise ValueError("checkpoint does not contain model_state_dict")
    return checkpoint["model_state_dict"]


def _quaternion_to_rotation_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    quaternions = quaternions / torch.linalg.vector_norm(
        quaternions,
        dim=1,
        keepdim=True,
    ).clamp_min(1.0e-12)
    w, x, y, z = quaternions.unbind(dim=1)
    matrices = torch.empty(
        (quaternions.shape[0], 3, 3),
        dtype=quaternions.dtype,
        device=quaternions.device,
    )
    matrices[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrices[:, 0, 1] = 2.0 * (x * y - z * w)
    matrices[:, 0, 2] = 2.0 * (x * z + y * w)
    matrices[:, 1, 0] = 2.0 * (x * y + z * w)
    matrices[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrices[:, 1, 2] = 2.0 * (y * z - x * w)
    matrices[:, 2, 0] = 2.0 * (x * z - y * w)
    matrices[:, 2, 1] = 2.0 * (y * z + x * w)
    matrices[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrices


def _compute_covariances(
    quaternions: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    rotations = _quaternion_to_rotation_matrix(quaternions)
    transform = torch.bmm(rotations, torch.diag_embed(scales))
    return torch.bmm(transform, transform.transpose(1, 2))


def _rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert one proper 3x3 rotation matrix to a normalized quaternion."""

    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quat = np.asarray([
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        ])
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            quat = np.asarray([
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            ])
        elif axis == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            quat = np.asarray([
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            ])
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            quat = np.asarray([
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            ])
    return (quat / np.linalg.norm(quat)).astype(np.float32)


def _quaternion_multiply_wxyz(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Broadcasted Hamilton product ``left * right`` in wxyz order."""

    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dim=-1,
    )


def load_nerficg_gaussians(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cuda",
    min_opacity: float = 0.0,
    min_scale: float = 1.0e-6,
    map_frame_transform: np.ndarray | None = None,
) -> SplatNavGaussians:
    """Convert a tensor-only NerfICG ``final.pt`` into SplatNav's layout."""

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"3DGS checkpoint not found: {checkpoint_path}")
    if not 0.0 <= min_opacity <= 1.0:
        raise ValueError("min_opacity must be in [0, 1]")

    state = _load_state_dict(checkpoint_path)
    position_key = _find_state_key(state, ("._positions", "_positions"))
    scale_key = _find_state_key(state, ("._scales", "_scales"))
    rotation_key = _find_state_key(state, ("._rotations", "_rotations"))
    opacity_key = _find_state_key(state, ("._opacities", "_opacities"))

    means = torch.as_tensor(state[position_key]).detach().float().reshape(-1, 3)
    raw_scales = torch.as_tensor(state[scale_key]).detach().float().reshape(-1, 3)
    rots = torch.as_tensor(state[rotation_key]).detach().float().reshape(-1, 4)
    raw_opacities = torch.as_tensor(state[opacity_key]).detach().float().reshape(-1)
    row_counts = {
        means.shape[0],
        raw_scales.shape[0],
        rots.shape[0],
        raw_opacities.shape[0],
    }
    if len(row_counts) != 1:
        raise ValueError("Gaussian checkpoint tensors have different row counts")

    values_are_physical = bool(torch.all(raw_scales > 0.0)) and bool(
        torch.all((raw_opacities >= 0.0) & (raw_opacities <= 1.0))
    )
    if values_are_physical:
        scales = raw_scales
        opacities = raw_opacities
    else:
        scales = torch.exp(raw_scales)
        opacities = torch.sigmoid(raw_opacities)

    quaternion_norms = torch.linalg.vector_norm(rots, dim=1)
    valid = torch.isfinite(means).all(dim=1)
    valid &= torch.isfinite(scales).all(dim=1) & (scales > 0.0).all(dim=1)
    valid &= torch.isfinite(rots).all(dim=1) & (quaternion_norms > 1.0e-12)
    valid &= torch.isfinite(opacities) & (opacities >= min_opacity)
    if not bool(valid.any()):
        raise ValueError("no valid Gaussians remain after loading and filtering")

    target_device = torch.device(device)
    means = means[valid].contiguous().to(target_device)
    scales = scales[valid].clamp_min(min_scale).contiguous().to(target_device)
    rots = rots[valid].contiguous().to(target_device)
    rots = rots / torch.linalg.vector_norm(
        rots,
        dim=1,
        keepdim=True,
    ).clamp_min(1.0e-12)
    opacities = opacities[valid].clamp(0.0, 1.0).reshape(-1, 1).to(target_device)
    colors = torch.full_like(means, 0.5)

    if map_frame_transform is not None:
        matrix = validate_rigid_transform(map_frame_transform)
        rotation = torch.as_tensor(
            matrix[:3, :3], dtype=means.dtype, device=target_device
        )
        translation = torch.as_tensor(
            matrix[:3, 3], dtype=means.dtype, device=target_device
        )
        means = means @ rotation.T + translation
        map_quaternion = torch.as_tensor(
            _rotation_matrix_to_quaternion_wxyz(matrix[:3, :3]),
            dtype=rots.dtype,
            device=target_device,
        ).reshape(1, 4)
        rots = _quaternion_multiply_wxyz(map_quaternion, rots)
        rots = rots / torch.linalg.vector_norm(
            rots, dim=1, keepdim=True
        ).clamp_min(1.0e-12)

    return SplatNavGaussians(
        means=means,
        rots=rots,
        scales=scales,
        opacities=opacities,
        colors=colors,
        covs=_compute_covariances(rots, scales),
        covs_inv=_compute_covariances(rots, scales.reciprocal()),
    )
