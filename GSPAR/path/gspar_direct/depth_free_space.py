"""Optional depth-observation free-space model for GSPAR-Direct."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _cpu_float(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.float32)
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)


class DepthFreeSpaceMap:
    def __init__(
        self,
        *,
        rotations_w2c: torch.Tensor | np.ndarray,
        translations_w2c: torch.Tensor | np.ndarray,
        inverse_depth: torch.Tensor | np.ndarray,
        focals: torch.Tensor | np.ndarray,
        image_sizes: torch.Tensor | np.ndarray,
        principal_points: torch.Tensor | np.ndarray | None = None,
        confidence: torch.Tensor | np.ndarray | None = None,
        device: str | torch.device = "cpu",
        min_depth: float = 0.05,
        max_depth: float = 20.0,
        surface_margin: float = 0.05,
        transition_width: float = 0.05,
        confidence_min: float = 0.05,
        free_threshold: float = 0.5,
    ) -> None:
        self.device = torch.device(device)
        self.rotations_w2c = torch.as_tensor(rotations_w2c, dtype=torch.float32, device=self.device).reshape(-1, 3, 3)
        self.translations_w2c = torch.as_tensor(translations_w2c, dtype=torch.float32, device=self.device).reshape(-1, 3)
        depth = torch.as_tensor(inverse_depth, dtype=torch.float32, device=self.device)
        if depth.ndim == 3:
            depth = depth[:, None]
        self.inverse_depth = depth.reshape(depth.shape[0], 1, depth.shape[-2], depth.shape[-1]).contiguous()
        self.focals = torch.as_tensor(focals, dtype=torch.float32, device=self.device).reshape(-1)
        self.image_sizes = torch.as_tensor(image_sizes, dtype=torch.float32, device=self.device).reshape(-1, 2)
        if principal_points is None:
            principal_points = torch.stack(
                [(self.image_sizes[:, 0] - 1.0) * 0.5, (self.image_sizes[:, 1] - 1.0) * 0.5],
                dim=1,
            )
        self.principal_points = torch.as_tensor(principal_points, dtype=torch.float32, device=self.device).reshape(-1, 2)
        if confidence is None:
            self.confidence = torch.ones_like(self.inverse_depth)
        else:
            conf = torch.as_tensor(confidence, dtype=torch.float32, device=self.device)
            if conf.ndim == 3:
                conf = conf[:, None]
            self.confidence = conf.reshape_as(self.inverse_depth).clamp(0.0, 1.0)
        count = int(self.inverse_depth.shape[0])
        for name, value in (
            ("rotations_w2c", self.rotations_w2c),
            ("translations_w2c", self.translations_w2c),
            ("focals", self.focals),
            ("image_sizes", self.image_sizes),
            ("principal_points", self.principal_points),
            ("confidence", self.confidence),
        ):
            if value.shape[0] != count:
                raise ValueError(f"{name} has {value.shape[0]} views, expected {count}")
        self.min_depth = max(float(min_depth), 1.0e-4)
        self.max_depth = max(float(max_depth), self.min_depth)
        self.surface_margin = max(float(surface_margin), 0.0)
        self.transition_width = max(float(transition_width), 1.0e-4)
        self.confidence_min = max(float(confidence_min), 0.0)
        self.free_threshold = float(np.clip(free_threshold, 0.0, 1.0))

    @property
    def observation_count(self) -> int:
        return int(self.inverse_depth.shape[0])

    @classmethod
    def empty(cls, *, device: str | torch.device = "cpu") -> "DepthFreeSpaceMap":
        return cls(
            rotations_w2c=torch.empty((0, 3, 3)),
            translations_w2c=torch.empty((0, 3)),
            inverse_depth=torch.empty((0, 1, 1, 1)),
            focals=torch.empty((0,)),
            image_sizes=torch.empty((0, 2)),
            device=device,
        )

    @classmethod
    def from_scene_model(
        cls,
        scene_model: Any,
        *,
        device: str | torch.device,
        max_keyframes: int = 24,
        depth_map_max_size: int = 160,
        **kwargs,
    ) -> "DepthFreeSpaceMap":
        keyframes = [
            frame
            for frame in list(getattr(scene_model, "keyframes", []) or [])
            if getattr(frame, "idepth_pyr", None)
        ]
        if len(keyframes) > max(1, int(max_keyframes)):
            ids = np.linspace(0, len(keyframes) - 1, max(1, int(max_keyframes)), dtype=np.int64)
            keyframes = [keyframes[int(index)] for index in ids]
        rotations = []
        translations = []
        depths = []
        confidences = []
        focals = []
        sizes = []
        centers = []
        target_max = max(8, int(depth_map_max_size))
        for frame in keyframes:
            try:
                inverse_depth = _cpu_float(frame.idepth_pyr[0]).squeeze()
                inverse_depth = inverse_depth * float(_cpu_float(frame.depth_scale).reshape(-1)[0])
                inverse_depth = inverse_depth + float(_cpu_float(frame.depth_offset).reshape(-1)[0])
                transform = _cpu_float(frame.get_Rt()).reshape(4, 4)
                height = int(getattr(frame, "height", inverse_depth.shape[-2]))
                width = int(getattr(frame, "width", inverse_depth.shape[-1]))
                focal = float(_cpu_float(frame.f).reshape(-1)[0])
            except (AttributeError, ValueError, RuntimeError, IndexError):
                continue
            ratio = min(1.0, target_max / float(max(inverse_depth.shape)))
            target = (
                max(2, int(round(inverse_depth.shape[-2] * ratio))),
                max(2, int(round(inverse_depth.shape[-1] * ratio))),
            )
            inverse_depth = F.interpolate(
                inverse_depth[None, None], size=target, mode="bilinear", align_corners=True
            )[0, 0]
            confidence = getattr(frame, "mono_depth_conf", None)
            if confidence is None:
                confidence = torch.ones_like(inverse_depth)
            else:
                confidence = F.interpolate(
                    _cpu_float(confidence).squeeze()[None, None],
                    size=target,
                    mode="bilinear",
                    align_corners=True,
                )[0, 0]
            center = getattr(frame, "centre", None)
            center = (
                torch.tensor([(width - 1.0) * 0.5, (height - 1.0) * 0.5])
                if center is None
                else _cpu_float(center).reshape(-1)[:2]
            )
            rotations.append(transform[:3, :3])
            translations.append(transform[:3, 3])
            depths.append(inverse_depth)
            confidences.append(confidence)
            focals.append(focal)
            sizes.append((float(width), float(height)))
            centers.append(center)
        if not depths:
            return cls.empty(device=device)
        return cls(
            rotations_w2c=torch.stack(rotations),
            translations_w2c=torch.stack(translations),
            inverse_depth=torch.stack(depths),
            confidence=torch.stack(confidences),
            focals=torch.tensor(focals),
            image_sizes=torch.tensor(sizes),
            principal_points=torch.stack(centers),
            device=device,
            **kwargs,
        )

    def free_probability(self, points: torch.Tensor) -> torch.Tensor:
        points = points.reshape(-1, 3).to(device=self.device, dtype=torch.float32)
        if self.observation_count == 0 or points.shape[0] == 0:
            return torch.zeros((points.shape[0],), device=points.device)
        camera = torch.einsum("vij,pj->vpi", self.rotations_w2c, points)
        camera = camera + self.translations_w2c[:, None, :]
        z = camera[..., 2]
        u = self.focals[:, None] * camera[..., 0] / z.clamp_min(self.min_depth) + self.principal_points[:, None, 0]
        v = self.focals[:, None] * camera[..., 1] / z.clamp_min(self.min_depth) + self.principal_points[:, None, 1]
        grid_x = 2.0 * u / (self.image_sizes[:, None, 0] - 1.0).clamp_min(1.0) - 1.0
        grid_y = 2.0 * v / (self.image_sizes[:, None, 1] - 1.0).clamp_min(1.0) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)
        idepth = F.grid_sample(self.inverse_depth, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[:, 0, :, 0]
        confidence = F.grid_sample(self.confidence, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[:, 0, :, 0]
        observed_depth = idepth.clamp_min(1.0e-8).reciprocal()
        valid = (z >= self.min_depth) & (z <= self.max_depth)
        valid &= (grid_x.abs() <= 1.0) & (grid_y.abs() <= 1.0)
        valid &= (idepth > 0.0) & (observed_depth <= self.max_depth)
        valid &= confidence >= self.confidence_min
        evidence = torch.sigmoid(
            (observed_depth - self.surface_margin - z) / self.transition_width
        ) * confidence
        return torch.where(valid, evidence, torch.zeros_like(evidence)).amax(dim=0)

    def compute_cost(self, points: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.free_threshold - self.free_probability(points)).pow(2).sum()

    def unknown_mask(self, points: torch.Tensor) -> torch.Tensor:
        return self.free_probability(points) < self.free_threshold
