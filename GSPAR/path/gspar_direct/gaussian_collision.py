"""Differentiable anisotropic Gaussian collision cost used by GSPAR-Direct."""

from __future__ import annotations

import torch

from .gaussian_field import GaussianField


class GaussianCollisionCost:
    """Evaluate the GSPAR-Direct Gaussian collision constraint."""

    def __init__(
        self,
        *,
        robot_radius: float = 0.5,
        sigma_rule: float = 3.0,
        eps: float = 1.0e-8,
        chunk_size: int = 4096,
    ) -> None:
        self.robot_radius = max(float(robot_radius), 0.0)
        self.sigma_rule = max(float(sigma_rule), 1.0e-3)
        self.eps = float(eps)
        self.chunk_size = max(1, int(chunk_size))

    def _geometry_terms(
        self,
        points: torch.Tensor,
        means: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        points = points.reshape(-1, 3).to(device=means.device, dtype=means.dtype)
        means = means.reshape(-1, 3)
        scales = scales.reshape(scales.shape[0], -1)
        if scales.shape[1] == 1:
            scales = scales.repeat(1, 3)
        scales = torch.clamp(scales[:, :3], min=self.eps)
        world_offset = points[:, None, :] - means[None, :, :]
        local_offset = world_offset
        if rotations is not None:
            # rotation is local-to-world; row-vector multiplication gives local coordinates.
            local_offset = torch.einsum(
                "pmc,mcd->pmd", world_offset, rotations.reshape(-1, 3, 3)
            )
        mahalanobis2 = torch.sum((local_offset / scales[None, :, :]) ** 2, dim=-1)
        distance2 = torch.sum(world_offset * world_offset, dim=-1)
        return mahalanobis2, distance2, scales

    def _threshold_from_geometry(
        self,
        mahalanobis2: torch.Tensor,
        distance2: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        eps2 = self.eps * self.eps
        inverse_directional_scale2 = mahalanobis2 / distance2.clamp_min(eps2)
        center_fallback = torch.amax(scales.reciprocal().square(), dim=1)[None, :]
        inverse_directional_scale2 = torch.where(
            distance2 > eps2, inverse_directional_scale2, center_fallback
        )
        safe_mahalanobis = self.sigma_rule + self.robot_radius * torch.sqrt(
            inverse_directional_scale2.clamp_min(0.0)
        )
        return torch.exp(-0.5 * safe_mahalanobis.square())

    def collision_threshold(
        self,
        points: torch.Tensor,
        means: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mahalanobis2, distance2, axes = self._geometry_terms(
            points, means, scales, rotations
        )
        return self._threshold_from_geometry(mahalanobis2, distance2, axes)

    def compute_alpha_and_threshold(
        self,
        points: torch.Tensor,
        means: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        rotations: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mahalanobis2, distance2, axes = self._geometry_terms(
            points, means, scales, rotations
        )
        opacity = opacities.reshape(-1).to(device=means.device, dtype=means.dtype)
        alpha = opacity[None, :] * torch.exp(-0.5 * mahalanobis2)
        return alpha, self._threshold_from_geometry(mahalanobis2, distance2, axes)

    def compute_alpha(
        self,
        points: torch.Tensor,
        means: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        rotations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mahalanobis2, _, _ = self._geometry_terms(points, means, scales, rotations)
        opacity = opacities.reshape(-1).to(device=means.device, dtype=means.dtype)
        return opacity[None, :] * torch.exp(-0.5 * mahalanobis2)

    @staticmethod
    def _selected_tensors(
        field: GaussianField, nearby_indices: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tensors = (
            field.get_means(),
            field.get_scales(),
            field.get_rotations(),
            field.get_opacities(),
        )
        if nearby_indices is None:
            return tensors
        indices = nearby_indices.to(device=field.device, dtype=torch.long).reshape(-1)
        return tuple(value[indices] for value in tensors)  # type: ignore[return-value]

    def _violations(
        self,
        points: torch.Tensor,
        field: GaussianField,
        nearby_indices: torch.Tensor | None,
    ):
        means, scales, rotations, opacities = self._selected_tensors(field, nearby_indices)
        means = means.to(device=points.device, dtype=points.dtype)
        scales = scales.to(device=points.device, dtype=points.dtype)
        rotations = rotations.to(device=points.device, dtype=points.dtype)
        opacities = opacities.to(device=points.device, dtype=points.dtype)
        for begin in range(0, means.shape[0], self.chunk_size):
            end = min(begin + self.chunk_size, means.shape[0])
            alpha, threshold = self.compute_alpha_and_threshold(
                points,
                means[begin:end],
                scales[begin:end],
                opacities[begin:end],
                rotations[begin:end],
            )
            yield torch.relu(alpha - threshold)

    def compute_cost(
        self,
        trajectory_points: torch.Tensor,
        gaussian_field: GaussianField,
        nearby_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        points = trajectory_points.reshape(-1, 3)
        means, _, _, _ = self._selected_tensors(gaussian_field, nearby_indices)
        total = torch.zeros((), device=points.device, dtype=points.dtype)
        if points.shape[0] == 0 or means.shape[0] == 0:
            return total
        for violation in self._violations(points, gaussian_field, nearby_indices):
            total = total + violation.pow(3).sum()
        return total

    def compute_cost_per_point(
        self,
        trajectory_points: torch.Tensor,
        gaussian_field: GaussianField,
        nearby_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        points = trajectory_points.reshape(-1, 3)
        result = torch.zeros((points.shape[0],), device=points.device, dtype=points.dtype)
        means, _, _, _ = self._selected_tensors(gaussian_field, nearby_indices)
        if points.shape[0] == 0 or means.shape[0] == 0:
            return result
        for violation in self._violations(points, gaussian_field, nearby_indices):
            result = result + violation.pow(3).sum(dim=1)
        return result

    def max_violation_per_point(
        self,
        trajectory_points: torch.Tensor,
        gaussian_field: GaussianField,
        nearby_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        points = trajectory_points.reshape(-1, 3)
        result = torch.zeros((points.shape[0],), device=points.device, dtype=points.dtype)
        means, _, _, _ = self._selected_tensors(gaussian_field, nearby_indices)
        if points.shape[0] == 0 or means.shape[0] == 0:
            return result
        for violation in self._violations(points, gaussian_field, nearby_indices):
            result = torch.maximum(result, violation.amax(dim=1))
        return result

    def max_violation_per_gaussian(
        self,
        trajectory_points: torch.Tensor,
        gaussian_field: GaussianField,
        nearby_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        points = trajectory_points.reshape(-1, 3)
        means, _, _, _ = self._selected_tensors(gaussian_field, nearby_indices)
        if points.shape[0] == 0 or means.shape[0] == 0:
            return torch.zeros((means.shape[0],), device=points.device, dtype=points.dtype)
        return torch.cat(
            [violation.amax(dim=0) for violation in self._violations(points, gaussian_field, nearby_indices)],
            dim=0,
        )
