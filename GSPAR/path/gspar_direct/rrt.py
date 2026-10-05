"""Map-independent deterministic 3D RRT coarse planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class RRTCollisionChecker(Protocol):
    def points_are_free(self, points: np.ndarray) -> np.ndarray:
        """Return one boolean per 3D query point."""


@dataclass(frozen=True)
class RRTConfig:
    max_iterations: int = 2000
    step_size: float = 0.3
    goal_sample_rate: float = 0.2
    collision_check_spacing: float = 0.05
    sampling_margin: float = 1.0
    random_seed: int = 0


@dataclass(frozen=True)
class RRTResult:
    path: np.ndarray
    success: bool
    iterations: int
    node_count: int
    message: str
    backend: str = "numpy_rrt"
    tree_nodes: np.ndarray | None = None
    tree_parents: np.ndarray | None = None
    raw_solution_path: np.ndarray | None = None


def make_sampling_bounds(
    start: np.ndarray, goal: np.ndarray, margin: float
) -> tuple[np.ndarray, np.ndarray]:
    start = np.asarray(start, dtype=np.float64).reshape(3)
    goal = np.asarray(goal, dtype=np.float64).reshape(3)
    padding = max(float(margin), 1.0e-3)
    return np.minimum(start, goal) - padding, np.maximum(start, goal) + padding


def _segment_is_free(
    start: np.ndarray,
    goal: np.ndarray,
    checker: RRTCollisionChecker,
    spacing: float,
) -> bool:
    distance = float(np.linalg.norm(goal - start))
    count = max(1, int(np.ceil(distance / max(float(spacing), 1.0e-6))))
    weights = np.linspace(0.0, 1.0, count + 1, dtype=np.float64)[:, None]
    samples = (1.0 - weights) * start[None, :] + weights * goal[None, :]
    free = np.asarray(checker.points_are_free(samples), dtype=bool).reshape(-1)
    if free.shape[0] != samples.shape[0]:
        raise ValueError("RRT collision checker must return one bool per point")
    return bool(free.all())


def _simplify_path(
    path: np.ndarray,
    checker: RRTCollisionChecker,
    spacing: float,
) -> np.ndarray:
    points = np.asarray(path, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] <= 2:
        return points.astype(np.float32)
    selected = [0]
    current = 0
    while current < points.shape[0] - 1:
        next_index = current + 1
        for candidate in range(points.shape[0] - 1, current, -1):
            if _segment_is_free(points[current], points[candidate], checker, spacing):
                next_index = candidate
                break
        selected.append(next_index)
        current = next_index
    return points[np.asarray(selected, dtype=np.int64)].astype(np.float32)


class RRTPlanner:
    """Plan a coarse path without requiring an external OMPL build."""

    def __init__(self, config: RRTConfig | None = None) -> None:
        self.config = config if config is not None else RRTConfig()

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        collision_checker: RRTCollisionChecker,
        *,
        bounds_min: np.ndarray | None = None,
        bounds_max: np.ndarray | None = None,
        record_trace: bool = False,
    ) -> RRTResult:
        start = np.asarray(start, dtype=np.float64).reshape(3)
        goal = np.asarray(goal, dtype=np.float64).reshape(3)
        fallback = np.stack([start, goal]).astype(np.float32)
        if not np.isfinite(start).all() or not np.isfinite(goal).all():
            return RRTResult(fallback, False, 0, 0, "RRT start and goal must be finite.")
        if bounds_min is None or bounds_max is None:
            bounds_min, bounds_max = make_sampling_bounds(start, goal, self.config.sampling_margin)
        bounds_min = np.asarray(bounds_min, dtype=np.float64).reshape(3)
        bounds_max = np.asarray(bounds_max, dtype=np.float64).reshape(3)
        if not np.isfinite(bounds_min).all() or not np.isfinite(bounds_max).all():
            raise ValueError("RRT sampling bounds must be finite")
        if bool((bounds_max <= bounds_min).any()):
            raise ValueError("RRT bounds_max must exceed bounds_min on every axis")
        if bool((start < bounds_min).any() or (start > bounds_max).any()):
            raise ValueError("RRT start lies outside sampling bounds")
        if bool((goal < bounds_min).any() or (goal > bounds_max).any()):
            raise ValueError("RRT goal lies outside sampling bounds")

        endpoint_free = np.asarray(
            collision_checker.points_are_free(np.stack([start, goal])), dtype=bool
        ).reshape(-1)
        if endpoint_free.shape[0] != 2 or not bool(endpoint_free.all()):
            return RRTResult(fallback, False, 0, 1, "RRT start or goal is in collision.")
        if float(np.linalg.norm(goal - start)) <= 1.0e-12:
            return RRTResult(fallback, True, 0, 1, "RRT start already equals goal.")
        spacing = max(float(self.config.collision_check_spacing), 1.0e-4)
        if _segment_is_free(start, goal, collision_checker, spacing):
            return RRTResult(
                fallback,
                True,
                0,
                2,
                "RRT found a direct collision-free path.",
                tree_nodes=fallback.copy() if record_trace else None,
                tree_parents=np.asarray([-1, 0], dtype=np.int32) if record_trace else None,
                raw_solution_path=fallback.copy() if record_trace else None,
            )

        rng = np.random.default_rng(max(0, int(self.config.random_seed)))
        step_size = max(float(self.config.step_size), 1.0e-4)
        goal_rate = float(np.clip(self.config.goal_sample_rate, 0.0, 1.0))
        nodes: list[np.ndarray] = [start]
        parents: list[int] = [-1]
        max_iterations = max(1, int(self.config.max_iterations))
        goal_index = -1
        iterations = 0
        for iteration in range(1, max_iterations + 1):
            iterations = iteration
            sample = goal if rng.random() < goal_rate else rng.uniform(bounds_min, bounds_max)
            stacked = np.asarray(nodes, dtype=np.float64)
            nearest = int(np.argmin(np.sum((stacked - sample[None, :]) ** 2, axis=1)))
            direction = sample - nodes[nearest]
            distance = float(np.linalg.norm(direction))
            if distance <= 1.0e-12:
                continue
            new_node = nodes[nearest] + direction * min(1.0, step_size / distance)
            if not _segment_is_free(nodes[nearest], new_node, collision_checker, spacing):
                continue
            nodes.append(new_node)
            parents.append(nearest)
            new_index = len(nodes) - 1
            if (
                float(np.linalg.norm(goal - new_node)) <= step_size
                and _segment_is_free(new_node, goal, collision_checker, spacing)
            ):
                nodes.append(goal)
                parents.append(new_index)
                goal_index = len(nodes) - 1
                break
        if goal_index < 0:
            return RRTResult(
                fallback,
                False,
                iterations,
                len(nodes),
                f"RRT failed after {iterations} iterations.",
                tree_nodes=np.asarray(nodes, dtype=np.float32) if record_trace else None,
                tree_parents=np.asarray(parents, dtype=np.int32) if record_trace else None,
            )

        indices = [goal_index]
        while parents[indices[-1]] >= 0:
            indices.append(parents[indices[-1]])
        indices.reverse()
        raw_solution_path = np.asarray([nodes[index] for index in indices], dtype=np.float32)
        path = _simplify_path(raw_solution_path, collision_checker, spacing)
        path[0] = start
        path[-1] = goal
        return RRTResult(
            path,
            True,
            iterations,
            len(nodes),
            "RRT found a collision-free coarse path.",
            tree_nodes=np.asarray(nodes, dtype=np.float32) if record_trace else None,
            tree_parents=np.asarray(parents, dtype=np.int32) if record_trace else None,
            raw_solution_path=raw_solution_path if record_trace else None,
        )
