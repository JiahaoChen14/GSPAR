"""GSPAR-Direct Gaussian collision adapter for RRT."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class GSPARDirectRRTCollisionChecker:
    def __init__(
        self,
        gaussian_field: Any,
        collision_cost: Any,
        *,
        nearby_indices: torch.Tensor | None = None,
        collision_tolerance: float = 0.0,
        unknown_map: Any | None = None,
    ) -> None:
        self.gaussian_field = gaussian_field
        self.collision_cost = collision_cost
        self.nearby_indices = nearby_indices
        self.collision_tolerance = max(float(collision_tolerance), 0.0)
        self.unknown_map = unknown_map

    def points_are_free(self, points: np.ndarray) -> np.ndarray:
        points_t = torch.as_tensor(
            np.asarray(points), device=self.gaussian_field.device, dtype=torch.float32
        ).reshape(-1, 3)
        with torch.no_grad():
            violation = self.collision_cost.max_violation_per_point(
                points_t, self.gaussian_field, self.nearby_indices
            )
        gaussian_free = (violation <= self.collision_tolerance).cpu().numpy().astype(bool)
        if self.unknown_map is None or int(getattr(self.unknown_map, "observation_count", 0)) <= 0:
            return gaussian_free
        with torch.no_grad():
            unknown = self.unknown_map.unknown_mask(points_t)
        return gaussian_free & (~unknown.cpu().numpy().astype(bool))
