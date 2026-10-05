"""Unified offline interfaces for grid and GSPAR planning routes."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from GSPAR.path.gspar_direct import GSPARDirectPlannerParams
from GSPAR.path.method_names import GSPAR_DIRECT_LABEL


GLOBAL_WEIGHTED_ASTAR = "Weighted A* (Occupancy Map)"
GLOBAL_JPS3D = "JPS-3D (Occupancy Map)"
GLOBAL_ESDF_WEIGHTED_ASTAR = "Weighted A* (ESDF)"
GLOBAL_ESDF_JPS3D = "JPS-3D (ESDF)"
LOCAL_EGO = "EGO (Occupancy Map)"
LOCAL_STOMP_ESDF = "STOMP (ESDF)"
GSPAR_DIRECT_METHOD = GSPAR_DIRECT_LABEL
_GSPAR_DIRECT_DEFAULTS = GSPARDirectPlannerParams()
_EGO_COARSE_SHORTCUT_MAX_CHECKS = 50000


@dataclass
class GlobalPlanOptions:
    method: str = GLOBAL_WEIGHTED_ASTAR
    connectivity: int = 26
    max_expansions: int = 100000
    snap_search_radius: int = 1
    safety_clearance_voxels: float = 2.5
    heuristic_weight: float = 10.0
    max_planning_time_sec: float = 10.0
    jps_crop_margin_voxels: int = 64
    treat_unknown_as_occupied: bool = False


@dataclass
class EgoPlanOptions:
    safety_clearance_voxels: float = 2.5
    safety_clearance_m: float | None = None
    inherit_input_clearance: bool = True
    rebound_iterations: int = 8
    control_point_spacing_m: float = 0.30
    output_spacing_m: float = 0.05
    collision_sample_spacing_m: float = 0.05
    treat_unknown_as_occupied: bool = False


@dataclass
class EsdfGlobalPlanOptions:
    method: str = GLOBAL_ESDF_WEIGHTED_ASTAR
    connectivity: int = 26
    max_expansions: int = 500000
    snap_search_radius: int = 8
    safety_clearance_m: float = 0.08
    clearance_cost_weight: float = 1.0
    heuristic_weight: float = 1.0
    max_planning_time_sec: float = 10.0
    jps_crop_margin_voxels: int = 64


@dataclass
class StompEsdfPlanOptions:
    safety_clearance_m: float = 0.08
    control_point_spacing_m: float = 0.25
    output_spacing_m: float = 0.05
    collision_sample_spacing_m: float = 0.05
    iterations: int = 80
    rollouts: int = 32
    noise_std_m: float = 0.10
    noise_decay: float = 0.98
    update_step_size: float = 1.0
    temperature: float = 1.0
    smooth_weight: float = 1.0
    jerk_weight: float = 0.1
    collision_weight: float = 200.0
    clearance_weight: float = 0.0
    fitness_weight: float = 30.0
    length_weight: float = 0.0
    random_seed: int = 7
    stagnation_patience: int = 20
    min_improvement: float = 1.0e-5
    adaptive_control_point_spacing: bool = True
    adaptive_min_control_point_spacing_m: float = 0.05
    adaptive_max_control_points: int = 256
    shortcut_input: bool = True


@dataclass
class GSPARDirectPlanOptions:
    robot_radius: float = _GSPAR_DIRECT_DEFAULTS.robot_radius
    num_control_points: int = _GSPAR_DIRECT_DEFAULTS.num_control_points
    num_samples: int = _GSPAR_DIRECT_DEFAULTS.num_samples
    optimization_steps: int = _GSPAR_DIRECT_DEFAULTS.optimization_steps
    learning_rate: float = _GSPAR_DIRECT_DEFAULTS.learning_rate
    optimizer_type: str = _GSPAR_DIRECT_DEFAULTS.optimizer_type
    w_collision: float = _GSPAR_DIRECT_DEFAULTS.w_collision
    w_smooth: float = _GSPAR_DIRECT_DEFAULTS.w_smooth
    w_length: float = _GSPAR_DIRECT_DEFAULTS.w_length
    w_dynamic: float = _GSPAR_DIRECT_DEFAULTS.w_dynamic
    enable_unknown_module: bool = _GSPAR_DIRECT_DEFAULTS.enable_unknown_module
    w_unknown: float = _GSPAR_DIRECT_DEFAULTS.w_unknown
    v_max: float = _GSPAR_DIRECT_DEFAULTS.v_max
    a_max: float = _GSPAR_DIRECT_DEFAULTS.a_max
    device: str | None = _GSPAR_DIRECT_DEFAULTS.device
    active_only: bool = _GSPAR_DIRECT_DEFAULTS.active_only
    opacity_min: float = _GSPAR_DIRECT_DEFAULTS.opacity_min
    selection_margin_m: float | None = _GSPAR_DIRECT_DEFAULTS.selection_margin_m
    max_selected_gaussians: int = _GSPAR_DIRECT_DEFAULTS.max_selected_gaussians
    collision_tolerance: float = _GSPAR_DIRECT_DEFAULTS.collision_tolerance
    rrt_collision_tolerance: float = _GSPAR_DIRECT_DEFAULTS.rrt_collision_tolerance
    optimization_collision_tolerance: float = (
        _GSPAR_DIRECT_DEFAULTS.optimization_collision_tolerance
    )
    validation_spacing_m: float = _GSPAR_DIRECT_DEFAULTS.validation_spacing_m
    rrt_max_iterations: int = _GSPAR_DIRECT_DEFAULTS.rrt_max_iterations
    rrt_step_size_m: float = _GSPAR_DIRECT_DEFAULTS.rrt_step_size_m
    rrt_sampling_margin_m: float = _GSPAR_DIRECT_DEFAULTS.rrt_sampling_margin_m
    rrt_random_seed: int = _GSPAR_DIRECT_DEFAULTS.rrt_random_seed
    record_animation_trace: bool = False
    max_unknown_fraction: float = _GSPAR_DIRECT_DEFAULTS.max_unknown_fraction


@dataclass
class GlobalPlanResult:
    path_world: np.ndarray
    start_idx: np.ndarray
    goal_idx: np.ndarray
    start_world: np.ndarray
    goal_world: np.ndarray
    start_was_snapped: bool
    goal_was_snapped: bool
    method: str
    header: dict[str, str]


@dataclass
class EgoPlanResult:
    path_world: np.ndarray
    success: bool
    message: str
    stats: dict[str, Any]
    header: dict[str, str]
    raw_result: Any
    raw_config: Any


@dataclass
class StompEsdfPlanResult:
    path_world: np.ndarray
    success: bool
    message: str
    stats: dict[str, Any]
    header: dict[str, str]
    raw_result: Any
    raw_config: Any


@dataclass
class GSPARDirectPlanResult:
    path_world: np.ndarray
    success: bool
    debug_info: dict[str, Any]
    header: dict[str, str]
    gaussian_payload: dict[str, np.ndarray]


def load_voxel_payload(
    source: str | Path | Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], Path | None]:
    if isinstance(source, Mapping):
        payload = {str(key): np.asarray(value) for key, value in source.items()}
        path = None
    else:
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            if path.suffix == "":
                path = path.with_suffix(".npz")
            if not path.is_file():
                raise FileNotFoundError(f"occupancy map not found: {path}")
        with np.load(path) as data:
            payload = {key: np.asarray(data[key]) for key in data.files}
    required = ("origin", "voxel_size", "voxel_dims", "occupied_voxel_idx")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"occupancy map missing keys: {missing}")
    return payload, path


def load_esdf_graph(
    source: str | Path,
    grid_metadata: str | Path | None = None,
    *,
    use_occupancy_guard: bool = True,
    cache_aligned: bool = True,
    unknown_policy: str = "occupied",
    unknown_clearance_m: float | None = None,
):
    """Load nvblox ESDF from a PLY, an ``sdf`` directory, or a scene directory."""

    from GSPAR.path.global_path.esdf_astar_world_path import load_esdf_grid

    return load_esdf_grid(
        source,
        grid_metadata,
        use_occupancy_guard=bool(use_occupancy_guard),
        cache_aligned=bool(cache_aligned),
        unknown_policy=str(unknown_policy),
        unknown_clearance_m=unknown_clearance_m,
    )


def payload_to_voxel_graph(
    payload: Mapping[str, Any],
    *,
    treat_unknown_as_occupied: bool = False,
):
    from GSPAR.path.global_path.astar_world_path import VoxelGridGraph

    occupied = np.asarray(payload["occupied_voxel_idx"], dtype=np.int32).reshape(-1, 3)
    if treat_unknown_as_occupied and "unknown_voxel_idx" in payload:
        occupied = np.concatenate(
            [occupied, np.asarray(payload["unknown_voxel_idx"], dtype=np.int32).reshape(-1, 3)],
            axis=0,
        )
    return VoxelGridGraph(
        origin=np.asarray(payload["origin"], dtype=np.float32),
        voxel_size=float(np.asarray(payload["voxel_size"]).reshape(-1)[0]),
        voxel_dims=np.asarray(payload["voxel_dims"], dtype=np.int32),
        occupied_idx=occupied,
    )


def payload_to_occupancy_grid(
    payload: Mapping[str, Any],
    *,
    treat_unknown_as_occupied: bool = False,
):
    from GSPAR.path.local_path.ego_occupancy_path import OccupancyGrid

    return OccupancyGrid(
        origin=np.asarray(payload["origin"], dtype=np.float64),
        voxel_size=float(np.asarray(payload["voxel_size"]).reshape(-1)[0]),
        voxel_dims=np.asarray(payload["voxel_dims"], dtype=np.int32),
        occupied_idx=np.asarray(payload["occupied_voxel_idx"], dtype=np.int32),
        unknown_idx=(
            None
            if "unknown_voxel_idx" not in payload
            else np.asarray(payload["unknown_voxel_idx"], dtype=np.int32)
        ),
        treat_unknown_as_occupied=bool(treat_unknown_as_occupied),
    )


def _segment_samples(start: np.ndarray, end: np.ndarray, spacing_m: float) -> np.ndarray:
    start = np.asarray(start, dtype=np.float32).reshape(3)
    end = np.asarray(end, dtype=np.float32).reshape(3)
    count = max(
        2,
        int(np.ceil(float(np.linalg.norm(end - start)) / max(float(spacing_m), 1.0e-6))) + 1,
    )
    weight = np.linspace(0.0, 1.0, count, dtype=np.float32)[:, None]
    samples = (1.0 - weight) * start[None] + weight * end[None]
    samples[0] = start
    samples[-1] = end
    return samples


def _line_segment_is_safe_occupancy(
    grid,
    start: np.ndarray,
    end: np.ndarray,
    *,
    safety_clearance_voxels: float,
    sample_spacing_m: float,
) -> bool:
    voxel_size = float(getattr(grid, "voxel_size", 0.0))
    spacing = min(float(sample_spacing_m), 0.25 * voxel_size) if voxel_size > 0.0 else float(sample_spacing_m)
    samples = _segment_samples(start, end, max(spacing, 1.0e-6))
    clearance = np.asarray(grid.clearance_voxels_for_points(samples), dtype=np.float32)
    threshold = float(safety_clearance_voxels)
    # A zero clearance still means occupied voxels are forbidden.
    return bool(np.all(clearance > 0.0)) if threshold <= 0.0 else bool(np.all(clearance >= threshold))


def _shortcut_occupancy_path(
    grid,
    path_world: np.ndarray,
    *,
    safety_clearance_voxels: float,
    sample_spacing_m: float,
    max_checks: int = _EGO_COARSE_SHORTCUT_MAX_CHECKS,
) -> np.ndarray:
    points = np.asarray(path_world, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] <= 2:
        return points.copy()
    selected = [0]
    checks = 0
    current = 0
    last = points.shape[0] - 1
    while current < last:
        best = current + 1
        for candidate in range(last, current, -1):
            if checks >= int(max_checks):
                break
            checks += 1
            if _line_segment_is_safe_occupancy(
                grid,
                points[current],
                points[candidate],
                safety_clearance_voxels=safety_clearance_voxels,
                sample_spacing_m=sample_spacing_m,
            ):
                best = candidate
                break
        selected.append(best)
        current = best
        if checks >= int(max_checks):
            selected.extend(range(current + 1, points.shape[0]))
            break
    return points[np.asarray(selected, dtype=np.int32)]


def _world_to_valid_voxel_idx(graph, point_world: np.ndarray) -> np.ndarray | None:
    point = np.asarray(point_world, dtype=np.float32).reshape(3)
    origin = np.asarray(graph.origin, dtype=np.float32).reshape(3)
    dims = np.asarray(graph.voxel_dims, dtype=np.int32).reshape(3)
    voxel_size = float(graph.voxel_size)
    if voxel_size <= 0.0 or not np.isfinite(point).all():
        return None
    relative = (point - origin) / voxel_size
    if not bool(((relative >= 0.0) & (relative < dims)).all()):
        return None
    return np.floor(relative).astype(np.int32)


def _preserve_safe_requested_endpoints(
    graph,
    path_world: np.ndarray,
    requested_start: np.ndarray,
    requested_goal: np.ndarray,
    start_idx: np.ndarray,
    goal_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, bool]:
    path = np.asarray(path_world, dtype=np.float32).reshape(-1, 3)
    if path.shape[0] == 0:
        raise ValueError("grid planner returned an empty path")
    start = np.asarray(requested_start, dtype=np.float32).reshape(3)
    goal = np.asarray(requested_goal, dtype=np.float32).reshape(3)
    start_idx = np.asarray(start_idx, dtype=np.int32).reshape(3)
    goal_idx = np.asarray(goal_idx, dtype=np.int32).reshape(3)
    centers = graph.voxel_idx_to_world_center(np.stack([start_idx, goal_idx])).astype(np.float32)
    requested_start_idx = _world_to_valid_voxel_idx(graph, start)
    requested_goal_idx = _world_to_valid_voxel_idx(graph, goal)
    start_snapped = requested_start_idx is None or not np.array_equal(requested_start_idx, start_idx)
    goal_snapped = requested_goal_idx is None or not np.array_equal(requested_goal_idx, goal_idx)
    effective_start = centers[0] if start_snapped else start
    effective_goal = centers[1] if goal_snapped else goal
    if path.shape[0] == 1:
        path = np.stack([effective_start, effective_goal])
    else:
        if not start_snapped and float(np.linalg.norm(path[0] - start)) > 1.0e-6:
            path = np.concatenate([start[None], path], axis=0)
        if not goal_snapped and float(np.linalg.norm(path[-1] - goal)) > 1.0e-6:
            path = np.concatenate([path, goal[None]], axis=0)
    return path, effective_start, effective_goal, bool(start_snapped), bool(goal_snapped)


def run_global_path(
    payload: Mapping[str, Any],
    start_world: np.ndarray,
    goal_world: np.ndarray,
    options: GlobalPlanOptions | None = None,
    *,
    map_label: str = "<memory>",
) -> GlobalPlanResult:
    from GSPAR.path.global_path.astar_world_path import astar_world_path
    from GSPAR.path.global_path.jps3d_world_path import jps3d_world_path

    options = options if options is not None else GlobalPlanOptions()
    graph = payload_to_voxel_graph(
        payload,
        treat_unknown_as_occupied=options.treat_unknown_as_occupied,
    )
    start = np.asarray(start_world, dtype=np.float32).reshape(3)
    goal = np.asarray(goal_world, dtype=np.float32).reshape(3)
    if options.method == GLOBAL_JPS3D:
        effective_max_expansions = 0
        effective_heuristic_weight = 1.0
        path, start_idx, goal_idx = jps3d_world_path(
            graph,
            start,
            goal,
            max_expansions=0,
            snap_search_radius=int(options.snap_search_radius),
            safety_clearance_voxels=float(options.safety_clearance_voxels),
            heuristic_weight=effective_heuristic_weight,
            max_planning_time_sec=float(options.max_planning_time_sec),
            crop_margin_voxels=int(options.jps_crop_margin_voxels),
        )
        backend = "NerfICG JPS-3D (cropped occupancy grid)"
    elif options.method == GLOBAL_WEIGHTED_ASTAR:
        effective_max_expansions = int(options.max_expansions)
        effective_heuristic_weight = max(1.0, float(options.heuristic_weight))
        path, start_idx, goal_idx = astar_world_path(
            graph,
            start,
            goal,
            connectivity=int(options.connectivity),
            max_expansions=effective_max_expansions,
            snap_search_radius=int(options.snap_search_radius),
            safety_clearance_voxels=float(options.safety_clearance_voxels),
            heuristic_weight=effective_heuristic_weight,
            max_planning_time_sec=float(options.max_planning_time_sec),
        )
        backend = "NerfICG sparse weighted A*"
    else:
        raise ValueError(f"unsupported global method: {options.method!r}")
    path, effective_start, effective_goal, start_snapped, goal_snapped = (
        _preserve_safe_requested_endpoints(
            graph, path, start, goal, start_idx, goal_idx
        )
    )
    header = {
        "method": str(options.method),
        "search_backend": backend,
        "planner_map_type": "Occupancy Map",
        "global_map": str(map_label),
        "points": str(path.shape[0]),
        "connectivity": str(int(options.connectivity)),
        "max_expansions": str(effective_max_expansions),
        "heuristic_weight": f"{effective_heuristic_weight:.3f}",
        "requested_heuristic_weight": f"{float(options.heuristic_weight):.3f}",
        "safety_clearance_voxels": f"{float(options.safety_clearance_voxels):.3f}",
        "max_planning_time_sec": f"{float(options.max_planning_time_sec):.3f}",
        "jps_crop_margin_voxels": str(int(options.jps_crop_margin_voxels)),
        "treat_unknown_as_occupied": str(bool(options.treat_unknown_as_occupied)),
        "start_idx": str(np.asarray(start_idx).tolist()),
        "goal_idx": str(np.asarray(goal_idx).tolist()),
    }
    return GlobalPlanResult(
        path_world=np.asarray(path, dtype=np.float32),
        start_idx=np.asarray(start_idx, dtype=np.int32),
        goal_idx=np.asarray(goal_idx, dtype=np.int32),
        start_world=np.asarray(effective_start, dtype=np.float32),
        goal_world=np.asarray(effective_goal, dtype=np.float32),
        start_was_snapped=start_snapped,
        goal_was_snapped=goal_snapped,
        method=str(options.method),
        header=header,
    )


def run_esdf_global_path(
    graph,
    start_world: np.ndarray,
    goal_world: np.ndarray,
    options: EsdfGlobalPlanOptions | None = None,
    *,
    map_label: str = "<memory ESDF>",
) -> GlobalPlanResult:
    """Run A* or JPS directly on an nvblox ESDF graph."""

    from GSPAR.path.global_path.esdf_astar_world_path import esdf_astar_world_path
    from GSPAR.path.global_path.jps3d_world_path import esdf_jps3d_world_path

    options = options if options is not None else EsdfGlobalPlanOptions()
    start = np.asarray(start_world, dtype=np.float32).reshape(3)
    goal = np.asarray(goal_world, dtype=np.float32).reshape(3)
    if options.method == GLOBAL_ESDF_JPS3D:
        effective_max_expansions = 0
        effective_heuristic_weight = 1.0
        effective_clearance_cost_weight: float | None = None
        path, start_idx, goal_idx = esdf_jps3d_world_path(
            graph,
            start,
            goal,
            max_expansions=effective_max_expansions,
            snap_search_radius=int(options.snap_search_radius),
            safety_clearance_m=float(options.safety_clearance_m),
            heuristic_weight=effective_heuristic_weight,
            max_planning_time_sec=float(options.max_planning_time_sec),
            crop_margin_voxels=int(options.jps_crop_margin_voxels),
        )
        backend = "NerfICG JPS-3D (cropped ESDF safety grid)"
    elif options.method == GLOBAL_ESDF_WEIGHTED_ASTAR:
        effective_max_expansions = int(options.max_expansions)
        effective_heuristic_weight = max(1.0, float(options.heuristic_weight))
        effective_clearance_cost_weight = max(0.0, float(options.clearance_cost_weight))
        path, start_idx, goal_idx = esdf_astar_world_path(
            graph,
            start,
            goal,
            connectivity=int(options.connectivity),
            max_expansions=effective_max_expansions,
            snap_search_radius=int(options.snap_search_radius),
            safety_clearance_m=float(options.safety_clearance_m),
            clearance_cost_weight=effective_clearance_cost_weight,
            heuristic_weight=effective_heuristic_weight,
            max_planning_time_sec=float(options.max_planning_time_sec),
        )
        backend = "NerfICG sparse ESDF weighted A*"
    else:
        raise ValueError(f"unsupported ESDF global method: {options.method!r}")
    path, effective_start, effective_goal, start_snapped, goal_snapped = (
        _preserve_safe_requested_endpoints(
            graph,
            path,
            start,
            goal,
            start_idx,
            goal_idx,
        )
    )
    header = {
        "method": str(options.method),
        "search_backend": backend,
        "planner_map_type": "ESDF",
        "global_map": str(map_label),
        "points": str(path.shape[0]),
        "connectivity": str(int(options.connectivity)),
        "max_expansions": str(effective_max_expansions),
        "heuristic_weight": f"{effective_heuristic_weight:.3f}",
        "requested_heuristic_weight": f"{float(options.heuristic_weight):.3f}",
        "clearance_cost_weight": (
            "not_applicable"
            if effective_clearance_cost_weight is None
            else f"{effective_clearance_cost_weight:.3f}"
        ),
        "requested_clearance_cost_weight": f"{float(options.clearance_cost_weight):.3f}",
        "safety_clearance_m": f"{float(options.safety_clearance_m):.3f}",
        "unknown_policy": str(getattr(graph, "unknown_policy", "")),
        "unknown_clearance_m": f"{float(getattr(graph, 'unknown_clearance_m', 0.0)):.3f}",
        "max_planning_time_sec": f"{float(options.max_planning_time_sec):.3f}",
        "jps_crop_margin_voxels": str(int(options.jps_crop_margin_voxels)),
        "start_idx": str(np.asarray(start_idx).tolist()),
        "goal_idx": str(np.asarray(goal_idx).tolist()),
    }
    return GlobalPlanResult(
        path_world=np.asarray(path, dtype=np.float32),
        start_idx=np.asarray(start_idx, dtype=np.int32),
        goal_idx=np.asarray(goal_idx, dtype=np.int32),
        start_world=np.asarray(effective_start, dtype=np.float32),
        goal_world=np.asarray(effective_goal, dtype=np.float32),
        start_was_snapped=start_snapped,
        goal_was_snapped=goal_snapped,
        method=str(options.method),
        header=header,
    )


def run_ego_path(
    payload: Mapping[str, Any],
    coarse_path: Any,
    options: EgoPlanOptions | None = None,
    *,
    map_label: str = "<memory>",
) -> EgoPlanResult:
    from GSPAR.path.local_path.ego_occupancy_path import (
        CoarsePath,
        EgoPlannerConfig,
        ego_occupancy_path,
    )

    options = options if options is not None else EgoPlanOptions()
    grid = payload_to_occupancy_grid(
        payload,
        treat_unknown_as_occupied=options.treat_unknown_as_occupied,
    )
    config = EgoPlannerConfig(
        safety_clearance_m=options.safety_clearance_m,
        safety_clearance_voxels=(
            None
            if options.safety_clearance_m is not None or options.inherit_input_clearance
            else float(options.safety_clearance_voxels)
        ),
        default_safety_clearance_voxels=float(options.safety_clearance_voxels),
        inherit_input_clearance=bool(options.inherit_input_clearance),
        rebound_iterations=int(options.rebound_iterations),
        control_point_spacing_m=float(options.control_point_spacing_m),
        output_spacing_m=float(options.output_spacing_m),
        collision_sample_spacing_m=float(options.collision_sample_spacing_m),
        treat_unknown_as_occupied=bool(options.treat_unknown_as_occupied),
    )
    coarse = (
        coarse_path
        if isinstance(coarse_path, CoarsePath)
        else CoarsePath(points=np.asarray(coarse_path, dtype=np.float32).reshape(-1, 3))
    )
    points_before = int(coarse.points.shape[0])
    shortcut = _shortcut_occupancy_path(
        grid,
        coarse.points,
        safety_clearance_voxels=config.resolved_safety_clearance_voxels(
            coarse,
            voxel_size_m=grid.voxel_size,
        ),
        sample_spacing_m=max(float(config.collision_sample_spacing_m), 1e-6),
    )
    shortcut_applied = shortcut.shape[0] < points_before
    if shortcut_applied:
        coarse = CoarsePath(
            points=shortcut,
            header=dict(coarse.header),
            raw_header_lines=list(coarse.raw_header_lines),
            path=coarse.path,
        )
    result = ego_occupancy_path(grid, coarse, config)
    result.stats = {
        **dict(result.stats),
        "coarse_points_before_shortcut": points_before,
        "coarse_points_after_shortcut": int(coarse.points.shape[0]),
        "coarse_shortcut_applied": shortcut_applied,
    }
    header = {
        "method": LOCAL_EGO,
        "planner_map_type": "Occupancy Map",
        "local_map": str(map_label),
        "points": str(result.path_world.shape[0]),
        "safety_clearance_voxels": f"{float(result.safety_clearance_voxels):.3f}",
        "safety_clearance_m": f"{float(result.safety_clearance_m):.6f}",
        "rebound_iterations": str(int(result.iterations)),
        "min_clearance_voxels": f"{float(result.min_clearance_voxels):.3f}",
        "min_clearance_m": f"{float(result.min_clearance_m):.6f}",
        "unsafe_sample_count": str(int(result.unsafe_sample_count)),
        "length_m": f"{float(result.length_m):.6f}",
        "success": str(bool(result.success)),
        "message": str(result.message),
    }
    return EgoPlanResult(
        path_world=np.asarray(result.path_world, dtype=np.float32),
        success=bool(result.success),
        message=str(result.message),
        stats=dict(result.stats),
        header=header,
        raw_result=result,
        raw_config=config,
    )


def run_stomp_esdf_path(
    graph,
    coarse_path: Any,
    options: StompEsdfPlanOptions | None = None,
    *,
    map_label: str = "<memory ESDF>",
) -> StompEsdfPlanResult:
    """Run STOMP directly against continuous nvblox ESDF distances."""

    from GSPAR.path.local_path.stomp_esdf_path import (
        CoarsePath,
        StompPlannerConfig,
        stomp_esdf_path,
    )

    options = options if options is not None else StompEsdfPlanOptions()
    config = StompPlannerConfig(
        safety_clearance_m=float(options.safety_clearance_m),
        inherit_input_clearance=False,
        control_point_spacing_m=float(options.control_point_spacing_m),
        output_spacing_m=float(options.output_spacing_m),
        collision_sample_spacing_m=float(options.collision_sample_spacing_m),
        iterations=int(options.iterations),
        rollouts=int(options.rollouts),
        noise_std_m=float(options.noise_std_m),
        noise_decay=float(options.noise_decay),
        update_step_size=float(options.update_step_size),
        temperature=float(options.temperature),
        smooth_weight=float(options.smooth_weight),
        jerk_weight=float(options.jerk_weight),
        collision_weight=float(options.collision_weight),
        clearance_weight=float(options.clearance_weight),
        fitness_weight=float(options.fitness_weight),
        length_weight=float(options.length_weight),
        random_seed=int(options.random_seed),
        stagnation_patience=int(options.stagnation_patience),
        min_improvement=float(options.min_improvement),
        safe_fallback_to_input=True,
        safe_shortcut_fallback=False,
        adaptive_control_point_spacing=bool(options.adaptive_control_point_spacing),
        adaptive_min_control_point_spacing_m=float(options.adaptive_min_control_point_spacing_m),
        adaptive_max_control_points=int(options.adaptive_max_control_points),
        shortcut_input=bool(options.shortcut_input),
    )
    coarse = (
        coarse_path
        if isinstance(coarse_path, CoarsePath)
        else CoarsePath(points=np.asarray(coarse_path, dtype=np.float32).reshape(-1, 3))
    )
    result = stomp_esdf_path(graph, coarse, config)
    header = {
        "method": LOCAL_STOMP_ESDF,
        "planner_map_type": "ESDF",
        "local_map": str(map_label),
        "points": str(result.path_world.shape[0]),
        "safety_clearance_m": f"{float(result.safety_clearance_m):.6f}",
        "unknown_policy": str(result.stats.get("unknown_policy", "")),
        "min_clearance_m": f"{float(result.min_clearance_m):.6f}",
        "unsafe_sample_count": str(int(result.unsafe_sample_count)),
        "random_seed": str(int(options.random_seed)),
        "success": str(bool(result.success)),
        "message": str(result.message),
    }
    return StompEsdfPlanResult(
        path_world=np.asarray(result.path_world, dtype=np.float32),
        success=bool(result.success),
        message=str(result.message),
        stats=dict(result.stats),
        header=header,
        raw_result=result,
        raw_config=config,
    )


def run_gspar_direct_path(
    raw_gaussian_map: Any,
    start_world: np.ndarray,
    goal_world: np.ndarray,
    options: GSPARDirectPlanOptions | None = None,
    *,
    include_gaussian_payload: bool = True,
) -> GSPARDirectPlanResult:
    import torch

    from GSPAR.path.gspar_direct import GSPARDirectPlanner

    options = options if options is not None else GSPARDirectPlanOptions()
    params = GSPARDirectPlannerParams(
        robot_radius=float(options.robot_radius),
        num_control_points=int(options.num_control_points),
        num_samples=int(options.num_samples),
        optimization_steps=int(options.optimization_steps),
        learning_rate=float(options.learning_rate),
        optimizer_type=str(options.optimizer_type),
        w_collision=float(options.w_collision),
        w_smooth=float(options.w_smooth),
        w_length=float(options.w_length),
        w_dynamic=float(options.w_dynamic),
        enable_unknown_module=bool(options.enable_unknown_module),
        w_unknown=float(options.w_unknown),
        v_max=float(options.v_max),
        a_max=float(options.a_max),
        device=options.device,
        active_only=bool(options.active_only),
        opacity_min=float(options.opacity_min),
        selection_margin_m=options.selection_margin_m,
        max_selected_gaussians=max(1, int(options.max_selected_gaussians)),
        collision_tolerance=max(0.0, float(options.collision_tolerance)),
        rrt_collision_tolerance=max(0.0, float(options.rrt_collision_tolerance)),
        optimization_collision_tolerance=max(
            0.0, float(options.optimization_collision_tolerance)
        ),
        validation_spacing_m=max(1.0e-4, float(options.validation_spacing_m)),
        rrt_max_iterations=max(1, int(options.rrt_max_iterations)),
        rrt_step_size_m=max(1.0e-4, float(options.rrt_step_size_m)),
        rrt_sampling_margin_m=max(1.0e-3, float(options.rrt_sampling_margin_m)),
        rrt_random_seed=max(0, int(options.rrt_random_seed)),
        record_animation_trace=bool(options.record_animation_trace),
        max_unknown_fraction=float(options.max_unknown_fraction),
    )
    planner_device = torch.device(
        params.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if planner_device.type == "cuda":
        torch.cuda.synchronize(planner_device)
    constructor_started = time.perf_counter()
    planner = GSPARDirectPlanner(params)
    if planner_device.type == "cuda":
        torch.cuda.synchronize(planner_device)
    constructor_time = float(time.perf_counter() - constructor_started)
    path, debug = planner.plan(start_world, goal_world, raw_gaussian_map)
    debug = dict(debug)
    timings = debug.setdefault("timings_sec", {})
    timings["planner_setup"] = float(
        timings.get("planner_setup", 0.0)
    ) + constructor_time
    indices = np.asarray(
        debug.pop("selected_gaussian_indices", np.zeros((0,), dtype=np.int64)),
        dtype=np.int64,
    )
    field = planner.gaussian_field
    if include_gaussian_payload:
        tensor_indices = torch.as_tensor(indices, dtype=torch.long, device=field.device)
        payload = {
            "centers": field.get_means()[tensor_indices].cpu().numpy().astype(np.float32),
            "scales": field.get_scales()[tensor_indices].cpu().numpy().astype(np.float32),
            "rotations": field.get_rotations()[tensor_indices].cpu().numpy().astype(np.float32),
            "radii": field.get_radii()[tensor_indices].cpu().numpy().astype(np.float32),
            "opacities": field.get_opacities()[tensor_indices].cpu().numpy().astype(np.float32),
            "robot_radius": np.asarray([float(options.robot_radius)], dtype=np.float32),
        }
    else:
        payload = {}
    header = {
        "method": GSPAR_DIRECT_METHOD,
        "planner_map_type": "Gaussian",
        "coarse_planner_family": "RRT",
        "avoidance_backend": "Gaussian ellipsoid collision cost",
        "points": str(np.asarray(path).reshape(-1, 3).shape[0]),
        "robot_radius_m": f"{float(options.robot_radius):.3f}",
        "num_control_points": str(int(options.num_control_points)),
        "num_samples": str(int(options.num_samples)),
        "optimization_steps": str(int(options.optimization_steps)),
        "optimizer_type": str(options.optimizer_type),
        "coarse_planner": str(debug.get("coarse_planner", "numpy_rrt")),
        "rrt_success": str(bool(debug.get("rrt_success", False))),
        "rrt_random_seed": str(int(debug.get("rrt_random_seed", options.rrt_random_seed))),
        "collision_safe": str(bool(debug.get("collision_safe", False))),
        "rrt_collision_tolerance": f"{float(options.rrt_collision_tolerance):.9g}",
        "optimization_collision_tolerance": (
            f"{float(options.optimization_collision_tolerance):.9g}"
        ),
        "selected_gaussian_count": str(int(debug.get("selected_gaussian_count", 0))),
        "gaussian_count": str(int(debug.get("gaussian_count", 0))),
        "failure_reason": str(debug.get("failure_reason", "")),
        "success": str(bool(debug.get("success", False))),
    }
    return GSPARDirectPlanResult(
        path_world=np.asarray(path, dtype=np.float32).reshape(-1, 3),
        success=bool(debug.get("success", False)),
        debug_info=debug,
        header=header,
        gaussian_payload=payload,
    )


def save_path_txt(
    output: str | Path,
    points: np.ndarray,
    *,
    header: Mapping[str, Any] | None = None,
) -> Path:
    path = Path(output).expanduser().resolve()
    if path.suffix == "":
        path = path / f"path_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["NERFICG path"]
    lines.extend(f"{key}: {value}" for key, value in (header or {}).items())
    lines.append("columns: x y z")
    np.savetxt(
        path,
        np.asarray(points, dtype=np.float32).reshape(-1, 3),
        fmt="%.6f",
        header="\n".join(lines),
        comments="# ",
    )
    return path
