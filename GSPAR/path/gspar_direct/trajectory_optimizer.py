"""Differentiable fixed-endpoint trajectory optimization for GSPAR-Direct."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from .gaussian_collision import GaussianCollisionCost
from .gaussian_field import GaussianField
from .rrt import RRTConfig, RRTPlanner, RRTResult, make_sampling_bounds
from .rrt_collision import GSPARDirectRRTCollisionChecker


class TrajectoryOptimizer:
    def __init__(self, params: Any, collision_cost: GaussianCollisionCost | None) -> None:
        self.params = params
        self.collision_cost = collision_cost
        self.device = torch.device(
            getattr(params, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def initialize_control_points(self, start: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        count = max(2, int(self.params.num_control_points))
        weights = torch.linspace(0.0, 1.0, count, device=self.device).reshape(-1, 1)
        return (1.0 - weights) * start.reshape(1, 3) + weights * goal.reshape(1, 3)

    @staticmethod
    def _catmull_rom_tangents(control_points: torch.Tensor) -> torch.Tensor:
        count = int(control_points.shape[0])
        tangents = torch.zeros_like(control_points)
        if count < 2:
            return tangents
        tangents[0] = control_points[1] - control_points[0]
        tangents[-1] = control_points[-1] - control_points[-2]
        if count > 2:
            tangents[1:-1] = 0.5 * (control_points[2:] - control_points[:-2])
        return 0.3 * tangents

    def sample_trajectory(self, control_points: torch.Tensor) -> torch.Tensor:
        count = int(control_points.shape[0])
        sample_count = max(2, int(self.params.num_samples))
        if count == 1:
            return control_points.repeat(sample_count, 1)
        if count == 2:
            weights = torch.linspace(
                0.0, 1.0, sample_count, device=control_points.device, dtype=control_points.dtype
            ).reshape(-1, 1)
            return (1.0 - weights) * control_points[:1] + weights * control_points[-1:]

        segment_count = count - 1
        t = torch.linspace(
            0.0,
            float(segment_count),
            sample_count,
            device=control_points.device,
            dtype=control_points.dtype,
        )
        left = torch.floor(t).long().clamp(max=count - 2)
        u = (t - left.to(dtype=control_points.dtype)).reshape(-1, 1)
        p0 = control_points[left]
        p1 = control_points[left + 1]
        tangents = self._catmull_rom_tangents(control_points)
        m0 = tangents[left]
        m1 = tangents[left + 1]
        u2 = u * u
        u3 = u2 * u
        trajectory = (
            (2.0 * u3 - 3.0 * u2 + 1.0) * p0
            + (u3 - 2.0 * u2 + u) * m0
            + (-2.0 * u3 + 3.0 * u2) * p1
            + (u3 - u2) * m1
        )
        return torch.cat([control_points[:1], trajectory[1:-1], control_points[-1:]], dim=0)

    @staticmethod
    def compute_smoothness_cost(points: torch.Tensor) -> torch.Tensor:
        if points.shape[0] < 3:
            return torch.zeros((), device=points.device, dtype=points.dtype)
        second = points[2:] - 2.0 * points[1:-1] + points[:-2]
        return torch.sum(second * second)

    @staticmethod
    def compute_length_cost(points: torch.Tensor) -> torch.Tensor:
        if points.shape[0] < 2:
            return torch.zeros((), device=points.device, dtype=points.dtype)
        diff = points[1:] - points[:-1]
        return torch.sqrt(torch.sum(diff * diff, dim=1) + 1.0e-12).sum()

    def compute_dynamic_cost(self, points: torch.Tensor) -> torch.Tensor:
        if points.shape[0] < 2:
            return torch.zeros((), device=points.device, dtype=points.dtype)
        velocity = points[1:] - points[:-1]
        speed = torch.linalg.vector_norm(velocity, dim=1)
        cost = torch.relu(speed - float(self.params.v_max)).pow(2).sum()
        if velocity.shape[0] >= 2:
            acceleration = velocity[1:] - velocity[:-1]
            cost = cost + torch.relu(
                torch.linalg.vector_norm(acceleration, dim=1) - float(self.params.a_max)
            ).pow(2).sum()
        return cost

    def _selection_margin(self, field: GaussianField) -> float:
        configured = getattr(self.params, "selection_margin_m", None)
        if configured is not None:
            return max(0.0, float(configured))
        return max(0.0, float(self.params.robot_radius) + 3.0 * field.max_radius())

    @staticmethod
    def _selection_representative_points(
        start: torch.Tensor,
        goal: torch.Tensor,
        bounds_min: np.ndarray,
        bounds_max: np.ndarray,
    ) -> torch.Tensor:
        """Cover both the direct corridor and the full RRT sampling box."""

        bounds = torch.as_tensor(
            np.stack([bounds_min, bounds_max]),
            device=start.device,
            dtype=start.dtype,
        )
        fractions = torch.linspace(0.0, 1.0, 3, device=start.device, dtype=start.dtype)
        axes = [
            bounds[0, axis] + fractions * (bounds[1, axis] - bounds[0, axis])
            for axis in range(3)
        ]
        box_grid = torch.stack(
            torch.meshgrid(*axes, indexing="ij"), dim=-1
        ).reshape(-1, 3)

        # GaussianField keeps at most 64 ranking references. A 3^3 box grid
        # leaves 37 samples for the start-goal corridor without downsampling.
        line_count = max(2, 64 - int(box_grid.shape[0]))
        weights = torch.linspace(
            0.0, 1.0, line_count, device=start.device, dtype=start.dtype
        ).reshape(-1, 1)
        line = (1.0 - weights) * start.reshape(1, 3) + weights * goal.reshape(1, 3)
        return torch.cat([line, box_grid], dim=0)

    @staticmethod
    def _control_points_from_interior(
        start: torch.Tensor, goal: torch.Tensor, interior: torch.Tensor | None
    ) -> torch.Tensor:
        if interior is None or interior.numel() == 0:
            return torch.stack([start, goal], dim=0)
        return torch.cat([start[None], interior.reshape(-1, 3), goal[None]], dim=0)

    def _losses(
        self,
        control_points: torch.Tensor,
        field: GaussianField,
        nearby_indices: torch.Tensor,
        unknown_map: Any | None,
    ) -> dict[str, torch.Tensor]:
        if self.collision_cost is None:
            raise RuntimeError("collision_cost is required for optimization")
        trajectory = self.sample_trajectory(control_points)
        collision = self.collision_cost.compute_cost(trajectory, field, nearby_indices)
        smooth = self.compute_smoothness_cost(trajectory)
        length = self.compute_length_cost(trajectory)
        dynamic = self.compute_dynamic_cost(trajectory)
        if unknown_map is not None and int(getattr(unknown_map, "observation_count", 0)) > 0:
            unknown = unknown_map.compute_cost(trajectory).to(trajectory)
        else:
            unknown = torch.zeros((), device=trajectory.device, dtype=trajectory.dtype)
        total = (
            float(self.params.w_collision) * collision
            + float(self.params.w_smooth) * smooth
            + float(self.params.w_length) * length
            + float(self.params.w_dynamic) * dynamic
            + float(self.params.w_unknown) * unknown
        )
        return {
            "total": total,
            "collision": collision,
            "smoothness": smooth,
            "length": length,
            "dynamic": dynamic,
            "unknown": unknown,
        }

    @staticmethod
    def _densify_control_points(points: torch.Tensor, minimum_count: int) -> torch.Tensor:
        result = [point for point in points.reshape(-1, 3)]
        if not result:
            raise ValueError("RRT coarse path must contain at least one point")
        if len(result) == 1:
            result.append(result[0])
        while len(result) < max(2, int(minimum_count)):
            stacked = torch.stack(result)
            index = int(torch.argmax(torch.linalg.vector_norm(stacked[1:] - stacked[:-1], dim=1)))
            result.insert(index + 1, 0.5 * (result[index] + result[index + 1]))
        return torch.stack(result)

    def _rrt_config(self) -> RRTConfig:
        return RRTConfig(
            max_iterations=max(1, int(self.params.rrt_max_iterations)),
            step_size=max(float(self.params.rrt_step_size_m), 1.0e-4),
            goal_sample_rate=0.2,
            collision_check_spacing=max(float(self.params.validation_spacing_m), 1.0e-4),
            sampling_margin=max(float(self.params.rrt_sampling_margin_m), 1.0e-3),
            random_seed=max(0, int(self.params.rrt_random_seed)),
        )

    def _plan_rrt_coarse_path(
        self,
        start: torch.Tensor,
        goal: torch.Tensor,
        field: GaussianField,
        unknown_map: Any | None,
    ) -> tuple[torch.Tensor, RRTResult, torch.Tensor]:
        if self.collision_cost is None:
            raise RuntimeError("collision_cost is required for RRT")
        config = self._rrt_config()
        start_np = start.detach().cpu().numpy().astype(np.float64, copy=False)
        goal_np = goal.detach().cpu().numpy().astype(np.float64, copy=False)
        bounds_min, bounds_max = make_sampling_bounds(start_np, goal_np, config.sampling_margin)
        search_bounds = torch.as_tensor(
            np.stack([bounds_min, bounds_max]), device=field.device
        )
        representative_points = self._selection_representative_points(
            start,
            goal,
            bounds_min,
            bounds_max,
        )
        nearby_indices = field.select_near_path(
            search_bounds,
            self._selection_margin(field),
            max_selected=int(self.params.max_selected_gaussians),
            representative_points=representative_points,
        )
        checker = GSPARDirectRRTCollisionChecker(
            field,
            self.collision_cost,
            nearby_indices=nearby_indices,
            collision_tolerance=float(
                getattr(
                    self.params,
                    "rrt_collision_tolerance",
                    getattr(self.params, "collision_tolerance", 0.0),
                )
            ),
            unknown_map=unknown_map,
        )
        result = RRTPlanner(config).plan(
            start_np,
            goal_np,
            checker,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            record_trace=bool(getattr(self.params, "record_animation_trace", False)),
        )
        coarse = torch.as_tensor(result.path, device=self.device, dtype=torch.float32)
        return (
            self._densify_control_points(coarse, int(self.params.num_control_points)),
            result,
            nearby_indices,
        )

    @staticmethod
    def _resample_for_validation(points: torch.Tensor, spacing: float) -> torch.Tensor:
        if points.shape[0] < 2:
            return points
        lengths = torch.linalg.vector_norm(points[1:] - points[:-1], dim=1)
        cumulative = torch.cat([lengths.new_zeros(1), torch.cumsum(lengths, dim=0)])
        total = float(cumulative[-1].detach().cpu())
        if total <= 1.0e-8:
            return points[:1]
        count = min(10000, max(2, int(np.ceil(total / max(float(spacing), 1.0e-3))) + 1))
        query = torch.linspace(0.0, total, count, device=points.device, dtype=points.dtype)
        right = torch.searchsorted(cumulative, query, right=True).clamp(1, points.shape[0] - 1)
        left = right - 1
        denominator = (cumulative[right] - cumulative[left]).clamp_min(1.0e-8)
        weight = ((query - cumulative[left]) / denominator).reshape(-1, 1)
        return (1.0 - weight) * points[left] + weight * points[right]

    @staticmethod
    def _max_straight_line_deviation(points: torch.Tensor) -> float:
        path = points.reshape(-1, 3)
        if path.shape[0] <= 2:
            return 0.0
        start = path[0]
        direction = path[-1] - start
        denominator = torch.dot(direction, direction)
        if float(denominator.detach().cpu()) <= 1.0e-12:
            return float(torch.linalg.vector_norm(path - start, dim=1).amax().detach().cpu())
        weight = (torch.sum((path - start) * direction, dim=1) / denominator).clamp(0.0, 1.0)
        projection = start[None] + weight[:, None] * direction[None]
        return float(torch.linalg.vector_norm(path - projection, dim=1).amax().detach().cpu())

    def optimize(
        self,
        *,
        start: np.ndarray | torch.Tensor,
        goal: np.ndarray | torch.Tensor,
        gaussian_field: GaussianField,
        unknown_map: Any | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self.collision_cost is None:
            raise RuntimeError("collision_cost is required for optimization")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        global_started = time.perf_counter()
        start_t = torch.as_tensor(start, device=self.device, dtype=torch.float32).reshape(3)
        goal_t = torch.as_tensor(goal, device=self.device, dtype=torch.float32).reshape(3)
        straight_control = self.initialize_control_points(start_t, goal_t)
        initial_control, rrt_result, nearby_indices = self._plan_rrt_coarse_path(
            start_t, goal_t, gaussian_field, unknown_map
        )
        record_animation = bool(getattr(self.params, "record_animation_trace", False))
        optimization_trace: list[np.ndarray] = []
        optimization_candidate_controls: list[torch.Tensor] = []

        def record_control(control: torch.Tensor) -> None:
            with torch.no_grad():
                optimization_candidate_controls.append(control.detach().clone())
                if not record_animation:
                    return
                optimization_trace.append(
                    self.sample_trajectory(control).detach().cpu().numpy().astype(np.float32)
                )

        record_control(initial_control)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        global_planning_time = float(time.perf_counter() - global_started)
        local_started = time.perf_counter()
        with torch.no_grad():
            straight_losses = self._losses(straight_control, gaussian_field, nearby_indices, unknown_map)
            initial_losses = self._losses(initial_control, gaussian_field, nearby_indices, unknown_map)
            initial_collision = float(straight_losses["collision"].cpu())

        if not rrt_result.success or initial_control.shape[0] <= 2 or int(self.params.optimization_steps) <= 0:
            final_control = initial_control
            final_losses = initial_losses
            steps_run = 0
        else:
            interior = initial_control[1:-1].clone().detach().requires_grad_(True)
            optimizer_type = str(self.params.optimizer_type).strip().lower()
            if optimizer_type == "adam":
                optimizer = torch.optim.Adam([interior], lr=float(self.params.learning_rate))
                steps_run = max(0, int(self.params.optimization_steps))
                for _ in range(steps_run):
                    optimizer.zero_grad()
                    control = self._control_points_from_interior(start_t, goal_t, interior)
                    losses = self._losses(control, gaussian_field, nearby_indices, unknown_map)
                    losses["total"].backward()
                    optimizer.step()
                    record_control(self._control_points_from_interior(start_t, goal_t, interior))
            else:
                optimizer = torch.optim.LBFGS(
                    [interior],
                    lr=float(self.params.learning_rate),
                    max_iter=max(1, int(self.params.optimization_steps)),
                    line_search_fn="strong_wolfe" if bool(self.params.lbfgs_strong_wolfe) else None,
                    tolerance_grad=float(self.params.lbfgs_tolerance_grad),
                    tolerance_change=float(self.params.lbfgs_tolerance_change),
                )

                def closure() -> torch.Tensor:
                    optimizer.zero_grad()
                    control = self._control_points_from_interior(start_t, goal_t, interior)
                    losses = self._losses(control, gaussian_field, nearby_indices, unknown_map)
                    record_control(control)
                    losses["total"].backward()
                    return losses["total"]

                optimizer.step(closure)
                steps_run = max(1, int(self.params.optimization_steps))
            final_control = self._control_points_from_interior(start_t, goal_t, interior.detach())
            record_control(final_control)
            with torch.no_grad():
                final_losses = self._losses(final_control, gaussian_field, nearby_indices, unknown_map)

        unknown_active = unknown_map is not None and int(getattr(unknown_map, "observation_count", 0)) > 0
        tolerance = float(
            getattr(
                self.params,
                "optimization_collision_tolerance",
                getattr(self.params, "collision_tolerance", 0.0),
            )
        )
        unknown_limit = float(getattr(self.params, "max_unknown_fraction", 0.05))

        def validate(control: torch.Tensor):
            trajectory = self.sample_trajectory(control).detach()
            validation = self._resample_for_validation(
                trajectory, float(getattr(self.params, "validation_spacing_m", 0.05))
            )
            violation = self.collision_cost.max_violation_per_point(
                validation, gaussian_field, nearby_indices
            )
            maximum = float(violation.amax().cpu()) if violation.numel() else 0.0
            unknown_fraction = (
                float(unknown_map.unknown_mask(validation).float().mean().cpu())
                if unknown_active
                else 0.0
            )
            return trajectory, validation, maximum, unknown_fraction, maximum <= tolerance, (
                not unknown_active or unknown_fraction <= unknown_limit
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        local_planning_time = float(time.perf_counter() - local_started)
        validation_started = time.perf_counter()
        with torch.no_grad():
            trajectory, validation, maximum, unknown_fraction, collision_safe, unknown_safe = validate(final_control)
            fallback_index = -1
            safe_optimized_snapshot_index = -1
            if not (collision_safe and unknown_safe):
                scored_candidates = []
                for candidate_index, candidate_control in enumerate(optimization_candidate_controls):
                    candidate_losses = self._losses(
                        candidate_control, gaussian_field, nearby_indices, unknown_map
                    )
                    scored_candidates.append(
                        (
                            float(candidate_losses["total"].detach().cpu()),
                            candidate_index,
                            candidate_control,
                            candidate_losses,
                        )
                    )
                for _, candidate_index, candidate_control, candidate_losses in sorted(
                    scored_candidates, key=lambda item: item[0]
                ):
                    candidate_validation = validate(candidate_control)
                    if candidate_validation[4] and candidate_validation[5]:
                        final_control = candidate_control
                        final_losses = candidate_losses
                        (
                            trajectory,
                            validation,
                            maximum,
                            unknown_fraction,
                            collision_safe,
                            unknown_safe,
                        ) = candidate_validation
                        if candidate_index == 0:
                            fallback_index = 0
                        else:
                            safe_optimized_snapshot_index = candidate_index
                        break
            finite = bool(torch.isfinite(trajectory).all() and torch.isfinite(final_losses["total"]))
            endpoint_violation = self.collision_cost.max_violation_per_point(
                trajectory[[0, -1]], gaussian_field, nearby_indices
            )
            start_violation = float(endpoint_violation[0].cpu())
            goal_violation = float(endpoint_violation[-1].cpu())
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        planner_validation_time = float(time.perf_counter() - validation_started)
        failure_reason = ""
        if max(start_violation, goal_violation) > tolerance:
            failure_reason = "Start or goal lies inside the configured Gaussian collision clearance."
        elif not rrt_result.success:
            failure_reason = rrt_result.message
        elif not collision_safe:
            failure_reason = "Optimized path violates the Gaussian collision constraint."
        elif not unknown_safe:
            failure_reason = "Optimized path exceeds the allowed unknown-space fraction."

        debug: dict[str, Any] = {
            "success": bool(finite and rrt_result.success and collision_safe and unknown_safe),
            "initial_collision_loss": initial_collision,
            "safe_fallback_candidate_index": fallback_index,
            "safe_optimized_snapshot_index": safe_optimized_snapshot_index,
            "rrt_collision_tolerance": float(
                getattr(self.params, "rrt_collision_tolerance", 0.0)
            ),
            "optimization_collision_tolerance": tolerance,
            "coarse_planner": rrt_result.backend,
            "rrt_success": bool(rrt_result.success),
            "rrt_iterations": int(rrt_result.iterations),
            "rrt_node_count": int(rrt_result.node_count),
            "rrt_path_point_count": int(rrt_result.path.shape[0]),
            "rrt_random_seed": max(0, int(self.params.rrt_random_seed)),
            "rrt_message": str(rrt_result.message),
            "rrt_coarse_path": np.asarray(rrt_result.path, dtype=np.float32),
            "rrt_max_straight_deviation_m": self._max_straight_line_deviation(
                torch.as_tensor(rrt_result.path, device=self.device)
            ),
            "final_max_straight_deviation_m": self._max_straight_line_deviation(trajectory),
            "selected_gaussian_count": int(nearby_indices.numel()),
            "selected_gaussian_indices": nearby_indices.cpu().numpy().astype(np.int64),
            "optimization_steps": int(steps_run),
            "validation_sample_count": int(validation.shape[0]),
            "optimizer_type": str(self.params.optimizer_type),
            "trajectory_model": "piecewise_cubic_hermite",
            "max_collision_violation": maximum,
            "start_collision_violation": start_violation,
            "goal_collision_violation": goal_violation,
            "collision_safe": bool(collision_safe),
            "unknown_module_active": bool(unknown_active),
            "depth_observation_count": int(getattr(unknown_map, "observation_count", 0)),
            "unknown_fraction": unknown_fraction,
            "unknown_safe": bool(unknown_safe),
            "failure_reason": failure_reason,
        }
        if record_animation:
            debug["animation_rrt_nodes"] = np.asarray(
                rrt_result.tree_nodes if rrt_result.tree_nodes is not None else rrt_result.path,
                dtype=np.float32,
            )
            debug["animation_rrt_parents"] = np.asarray(
                rrt_result.tree_parents
                if rrt_result.tree_parents is not None
                else np.arange(rrt_result.path.shape[0], dtype=np.int32) - 1,
                dtype=np.int32,
            )
            debug["animation_rrt_solution"] = np.asarray(rrt_result.path, dtype=np.float32)
            if not optimization_trace or not np.array_equal(
                optimization_trace[-1], trajectory.detach().cpu().numpy()
            ):
                optimization_trace.append(trajectory.detach().cpu().numpy().astype(np.float32))
            debug["animation_optimization_paths"] = np.stack(optimization_trace, axis=0)
        for name, value in final_losses.items():
            debug[f"final_{name}_loss"] = float(value.detach().cpu())
        path = trajectory.cpu().numpy().astype(np.float32)
        debug["timings_sec"] = {
            "global_planning": global_planning_time,
            "local_planning": local_planning_time,
            "planner_validation": planner_validation_time,
        }
        return path, debug
