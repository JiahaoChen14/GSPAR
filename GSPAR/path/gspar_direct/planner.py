"""Fixed A-to-B offline GSPAR-Direct entry point."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .gaussian_collision import GaussianCollisionCost
from .gaussian_field import GaussianField
from .trajectory_optimizer import TrajectoryOptimizer


@dataclass
class GSPARDirectPlannerParams:
    robot_radius: float = 0.08
    num_control_points: int = 6
    num_samples: int = 50
    optimization_steps: int = 80
    learning_rate: float = 0.03
    optimizer_type: str = "lbfgs"
    lbfgs_strong_wolfe: bool = True
    lbfgs_tolerance_grad: float = 1.0e-7
    lbfgs_tolerance_change: float = 1.0e-9
    w_collision: float = 10.0
    w_smooth: float = 1.0
    w_length: float = 0.1
    w_dynamic: float = 0.5
    enable_unknown_module: bool = False
    w_unknown: float = 5.0
    v_max: float = 1.0
    a_max: float = 2.0
    device: str | None = None
    active_only: bool = False
    min_radius: float = 1.0e-4
    opacity_min: float = 0.0
    selection_margin_m: float | None = None
    max_selected_gaussians: int = 20000
    collision_chunk_size: int = 4096
    collision_tolerance: float = 0.0
    rrt_collision_tolerance: float = 0.0
    optimization_collision_tolerance: float = 1.0e-4
    validation_spacing_m: float = 0.05
    rrt_max_iterations: int = 2000
    rrt_step_size_m: float = 0.3
    rrt_sampling_margin_m: float = 1.0
    rrt_random_seed: int = 0
    record_animation_trace: bool = False
    max_unknown_fraction: float = 0.05
    depth_max_keyframes: int = 24
    depth_map_max_size: int = 160
    depth_min_m: float = 0.05
    depth_max_m: float = 20.0
    depth_surface_margin_m: float = 0.05
    depth_transition_width_m: float = 0.05
    depth_confidence_min: float = 0.05
    depth_free_threshold: float = 0.5


class GSPARDirectPlanner:
    """Plan directly in the anisotropic Gaussian field using RRT and Hermite optimization."""

    def __init__(self, params: GSPARDirectPlannerParams | None = None) -> None:
        self.params = params if params is not None else GSPARDirectPlannerParams()
        device = torch.device(
            self.params.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.gaussian_field = GaussianField(
            device=device,
            active_only=self.params.active_only,
            min_radius=self.params.min_radius,
            opacity_min=self.params.opacity_min,
        )
        self.collision_cost = GaussianCollisionCost(
            robot_radius=self.params.robot_radius,
            chunk_size=self.params.collision_chunk_size,
        )
        self.trajectory_optimizer = TrajectoryOptimizer(self.params, self.collision_cost)

    def plan(
        self,
        start_position: np.ndarray,
        goal_position: np.ndarray,
        raw_gaussian_map: Any,
        unknown_map: Any | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        start = np.asarray(start_position, dtype=np.float32).reshape(3)
        goal = np.asarray(goal_position, dtype=np.float32).reshape(3)
        if self.gaussian_field.device.type == "cuda":
            torch.cuda.synchronize(self.gaussian_field.device)
        setup_started = time.perf_counter()
        self.gaussian_field.update(raw_gaussian_map, query_points=np.stack([start, goal]))
        if self.gaussian_field.device.type == "cuda":
            torch.cuda.synchronize(self.gaussian_field.device)
        gaussian_setup_time = float(time.perf_counter() - setup_started)
        unknown_enabled = bool(self.params.enable_unknown_module)
        if not unknown_enabled:
            unknown_map = None
        elif unknown_map is None and hasattr(raw_gaussian_map, "keyframes"):
            from .depth_free_space import DepthFreeSpaceMap

            unknown_map = DepthFreeSpaceMap.from_scene_model(
                raw_gaussian_map,
                device=self.gaussian_field.device,
                max_keyframes=self.params.depth_max_keyframes,
                depth_map_max_size=self.params.depth_map_max_size,
                min_depth=self.params.depth_min_m,
                max_depth=self.params.depth_max_m,
                surface_margin=self.params.depth_surface_margin_m,
                transition_width=self.params.depth_transition_width_m,
                confidence_min=self.params.depth_confidence_min,
                free_threshold=self.params.depth_free_threshold,
            )
        trajectory, debug = self.trajectory_optimizer.optimize(
            start=start,
            goal=goal,
            gaussian_field=self.gaussian_field,
            unknown_map=unknown_map,
        )
        debug.setdefault("timings_sec", {})["planner_setup"] = gaussian_setup_time
        debug["gaussian_count"] = self.gaussian_field.count
        debug["anisotropic_collision"] = True
        debug["collision_model"] = "anisotropic_alpha_adaptive_threshold"
        debug["unknown_module_enabled"] = unknown_enabled
        debug["unknown_space_policy"] = "depth_free_space" if unknown_enabled else "free"
        depth_available = int(debug.get("depth_observation_count", 0)) > 0
        debug["depth_required"] = unknown_enabled
        debug["depth_available"] = depth_available
        if unknown_enabled and not depth_available:
            debug["success"] = False
            debug["unknown_safe"] = False
            debug["failure_reason"] = (
                "Unknown-space module is enabled, but no aligned depth observation is available."
            )
        return trajectory, debug
