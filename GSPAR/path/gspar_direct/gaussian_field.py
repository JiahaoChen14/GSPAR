"""Normalize offline NerfICG Gaussian geometry for path planning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def _to_tensor(value: Any, *, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(value), device=device, dtype=dtype)


def _value(value: Any) -> Any:
    if callable(value):
        value = value()
    return value


def _mapping_value(mapping: dict[str, Any], names: tuple[str, ...]) -> tuple[Any | None, bool]:
    for name in names:
        if name not in mapping:
            continue
        value = mapping[name]
        if isinstance(value, dict) and "val" in value:
            return value["val"], True
        return value, False
    return None, False


def _decode_physical(scales: torch.Tensor, opacities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode log-scale/logit inputs while preserving baked final checkpoints."""

    scales_are_physical = bool(torch.all(scales > 0.0).item())
    opacity_flat = opacities.reshape(-1)
    opacities_are_physical = bool(
        torch.all((opacity_flat >= 0.0) & (opacity_flat <= 1.0)).item()
    )
    if scales_are_physical and opacities_are_physical:
        return scales, opacity_flat
    return torch.exp(scales), torch.sigmoid(opacity_flat)


class GaussianField:
    """Gaussian centers, physical axes, rotations, and opacities.

    Accepted inputs include a NerfICG ``final.pt`` path, ``GaussianSet``, a
    built NerfICG model/``Gaussians`` object, or a mapping containing physical
    ``positions/scales/rotations/opacities`` arrays.
    """

    def __init__(
        self,
        *,
        device: str | torch.device | None = None,
        active_only: bool = False,
        min_radius: float = 1.0e-4,
        opacity_min: float = 0.0,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.active_only = bool(active_only)
        self.min_radius = float(min_radius)
        self.opacity_min = float(opacity_min)
        self.means = torch.empty((0, 3), device=self.device)
        self.scales = torch.empty((0, 3), device=self.device)
        self.rotations = torch.empty((0, 3, 3), device=self.device)
        self.radii = torch.empty((0,), device=self.device)
        self.opacities = torch.empty((0,), device=self.device)

    @property
    def count(self) -> int:
        return int(self.means.shape[0])

    def update(
        self,
        raw_gaussian_map: Any,
        *,
        query_points: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        del query_points  # Offline checkpoints have no live anchor window.
        means, scales, rotations, opacities = self._normalize_source(raw_gaussian_map)
        means = means.reshape(-1, 3)
        scales = self._scaling_to_axes(scales)
        rotations = self._rotation_to_matrix(rotations, means.shape[0])
        opacities = opacities.reshape(-1)
        if not (means.shape[0] == scales.shape[0] == rotations.shape[0] == opacities.shape[0]):
            raise ValueError(
                "Gaussian field tensors must have matching row counts: "
                f"means={means.shape}, scales={scales.shape}, "
                f"rotations={rotations.shape}, opacities={opacities.shape}"
            )

        finite = torch.isfinite(means).all(dim=1)
        finite &= torch.isfinite(scales).all(dim=1) & (scales > 0.0).all(dim=1)
        finite &= torch.isfinite(rotations).all(dim=(1, 2)) & torch.isfinite(opacities)
        if self.opacity_min > 0.0:
            finite &= opacities >= self.opacity_min
        self.means = means[finite].contiguous()
        self.scales = torch.clamp(scales[finite], min=self.min_radius).contiguous()
        self.rotations = rotations[finite].contiguous()
        self.radii = torch.amax(self.scales, dim=1).contiguous()
        self.opacities = torch.clamp(opacities[finite], 0.0, 1.0).contiguous()

    def _normalize_source(
        self, source: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(source, (str, Path)):
            return self._from_checkpoint(Path(source))
        if isinstance(source, dict):
            return self._from_mapping(source)
        if hasattr(source, "gaussians") and getattr(source, "gaussians") is not None:
            return self._normalize_source(getattr(source, "gaussians"))
        return self._from_attrs(source)

    def _from_checkpoint(
        self, checkpoint_path: Path
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        path = checkpoint_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Gaussian checkpoint not found: {path}")
        from GSPAR.mapping.gaussian_occupancy.voxel_preprocess import (
            _collect_raw_arrays,
            _to_rotation_matrix,
        )

        positions, scale_raw, rotation_raw, opacity_raw = _collect_raw_arrays(path)
        means = _to_tensor(positions, device=self.device)
        scales = _to_tensor(scale_raw, device=self.device)
        opacities = _to_tensor(opacity_raw, device=self.device)
        scales, opacities = _decode_physical(scales, opacities)
        rotations = _to_tensor(_to_rotation_matrix(rotation_raw), device=self.device)
        return means, scales, rotations, opacities

    def _from_mapping(
        self, mapping: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if "gaussian_params" in mapping and isinstance(mapping["gaussian_params"], dict):
            mapping = mapping["gaussian_params"]
        means_value, means_raw = _mapping_value(
            mapping, ("means", "centers", "center", "xyz", "positions", "_positions")
        )
        scales_value, scales_raw = _mapping_value(
            mapping, ("scaling", "scalings", "scales", "scale", "_scales")
        )
        radii_value, _ = _mapping_value(mapping, ("radii", "radius"))
        rotations_value, _ = _mapping_value(
            mapping, ("rotation", "rotations", "quaternion", "quaternions", "_rotations")
        )
        opacity_value, opacity_raw = _mapping_value(mapping, ("opacity", "opacities", "_opacities"))
        if means_value is None or opacity_value is None or (scales_value is None and radii_value is None):
            raise ValueError(
                "Gaussian mapping requires positions/means, scales/radii, and opacities"
            )
        means = _to_tensor(means_value, device=self.device).reshape(-1, 3)
        if radii_value is not None:
            scales = _to_tensor(radii_value, device=self.device).reshape(-1, 1).repeat(1, 3)
        else:
            scales = _to_tensor(scales_value, device=self.device)
        opacities = _to_tensor(opacity_value, device=self.device).reshape(-1)
        if scales_raw or opacity_raw or means_raw:
            scales, opacities = _decode_physical(scales, opacities)
        if rotations_value is None:
            rotations = torch.eye(3, device=self.device).reshape(1, 3, 3).repeat(means.shape[0], 1, 1)
        else:
            rotations = _to_tensor(rotations_value, device=self.device)
        return means, scales, rotations, opacities

    def _from_attrs(
        self, value: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        names = (
            ("get_positions", "positions", "xyz", "means", "_positions"),
            ("get_scales", "scales", "scaling", "_scales"),
            ("get_rotations", "rotations", "rotation", "_rotations"),
            ("get_opacities", "opacities", "opacity", "_opacities"),
        )
        values: list[Any] = []
        selected_names: list[str] = []
        for alternatives in names:
            for name in alternatives:
                if hasattr(value, name):
                    values.append(_value(getattr(value, name)))
                    selected_names.append(name)
                    break
            else:
                raise TypeError(
                    "Gaussian source must be a checkpoint, mapping, NerfICG model, "
                    "or object exposing positions/scales/rotations/opacities"
                )
        means = _to_tensor(values[0], device=self.device)
        scales = _to_tensor(values[1], device=self.device)
        rotations = _to_tensor(values[2], device=self.device)
        opacities = _to_tensor(values[3], device=self.device)
        if any(name.startswith("_") for name in selected_names):
            scales, opacities = _decode_physical(scales, opacities)
        return means, scales, rotations, opacities

    @staticmethod
    def _scaling_to_axes(scaling: torch.Tensor) -> torch.Tensor:
        if scaling.shape[0] == 0:
            return torch.empty((0, 3), device=scaling.device, dtype=scaling.dtype)
        scaling = scaling.reshape(scaling.shape[0], -1)
        if scaling.shape[1] == 1:
            return scaling.repeat(1, 3)
        if scaling.shape[1] < 3:
            raise ValueError(f"Gaussian scaling must have 1 or at least 3 columns, got {scaling.shape}")
        return scaling[:, :3]

    @staticmethod
    def _rotation_to_matrix(rotation: torch.Tensor, count: int) -> torch.Tensor:
        if rotation.numel() == 0 and count == 0:
            return torch.empty((0, 3, 3), device=rotation.device, dtype=rotation.dtype)
        if rotation.ndim >= 2 and rotation.shape[-2:] == (3, 3):
            matrix = rotation.reshape(-1, 3, 3)
        else:
            quat = rotation.reshape(-1, 4)
            quat = quat / torch.linalg.vector_norm(quat, dim=1, keepdim=True).clamp_min(1.0e-12)
            w, x, y, z = quat.unbind(dim=1)
            matrix = torch.empty((quat.shape[0], 3, 3), device=quat.device, dtype=quat.dtype)
            matrix[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
            matrix[:, 0, 1] = 2.0 * (x * y - z * w)
            matrix[:, 0, 2] = 2.0 * (x * z + y * w)
            matrix[:, 1, 0] = 2.0 * (x * y + z * w)
            matrix[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
            matrix[:, 1, 2] = 2.0 * (y * z - x * w)
            matrix[:, 2, 0] = 2.0 * (x * z - y * w)
            matrix[:, 2, 1] = 2.0 * (y * z + x * w)
            matrix[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
        if matrix.shape[0] == 1 and count != 1:
            matrix = matrix.repeat(count, 1, 1)
        if matrix.shape[0] != count:
            raise ValueError(f"rotation count {matrix.shape[0]} does not match Gaussian count {count}")
        valid = torch.isfinite(matrix).all(dim=(1, 2)) & (torch.linalg.det(matrix).abs() > 1.0e-6)
        if not bool(valid.all()):
            matrix = matrix.clone()
            matrix[~valid] = torch.eye(3, device=matrix.device, dtype=matrix.dtype)
        return matrix

    def get_means(self) -> torch.Tensor:
        return self.means

    def get_radii(self) -> torch.Tensor:
        return self.radii

    def get_scales(self) -> torch.Tensor:
        return self.scales

    def get_rotations(self) -> torch.Tensor:
        return self.rotations

    def get_opacities(self) -> torch.Tensor:
        return self.opacities

    def max_radius(self) -> float:
        return 0.0 if self.count == 0 else float(torch.max(self.radii).detach().cpu())

    def select_near_path(
        self,
        trajectory_points: torch.Tensor | np.ndarray,
        margin: float,
        *,
        max_selected: int | None = None,
        representative_points: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        if self.count == 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)
        points = _to_tensor(trajectory_points, device=self.device).reshape(-1, 3)
        if points.shape[0] == 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)
        margin = max(float(margin), 0.0)
        box_min = torch.amin(points, dim=0) - margin
        box_max = torch.amax(points, dim=0) + margin
        mask = ((self.means >= box_min) & (self.means <= box_max)).all(dim=1)
        indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
        if max_selected is not None and 0 <= int(max_selected) < indices.numel():
            keep = int(max_selected)
            if keep == 0:
                return torch.empty((0,), dtype=torch.long, device=self.device)
            reference = (
                points
                if representative_points is None
                else _to_tensor(representative_points, device=self.device).reshape(-1, 3)
            )
            if reference.shape[0] == 0:
                reference = points
            if reference.shape[0] > 64:
                ids = torch.linspace(0, reference.shape[0] - 1, 64, device=self.device).round().long()
                reference = reference[ids]
            scores = torch.empty(indices.shape[0], device=self.device)
            for begin in range(0, indices.numel(), 32768):
                local = indices[begin : begin + 32768]
                distance = torch.cdist(self.means[local], reference).amin(dim=1)
                safe_radius = 3.0 * self.radii[local] + max(0.0, margin - 3.0 * self.max_radius())
                normalized = distance / safe_radius.clamp_min(self.min_radius)
                scores[begin : begin + local.numel()] = (
                    torch.log(self.opacities[local].clamp_min(1.0e-8)) - 0.5 * normalized**2
                )
            indices = indices[torch.topk(scores, k=keep, largest=True).indices]
        return indices.contiguous()
