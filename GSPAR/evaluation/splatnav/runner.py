#!/usr/bin/env python3
"""Replay saved endpoint manifests with SplatNav and score the returned paths.

This is an additive experiment runner.  It does not modify SplatNav or the
existing GSPAR planners, and it writes results to a separate output tree.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import re
import sys
import time
import traceback
import types
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from GSPAR.evaluation.resource_usage import (
    ResourceMonitor,
    summarize_resource_usages,
)

from .gaussian_loader import SplatNavGaussians, load_nerficg_gaussians
from ..path_planning.metrics import (
    PathEvaluation,
    VoxelReference,
    aggregate_evaluations,
    evaluate_path_metrics,
    evaluate_strict_success,
    strict_failure_reasons,
)
from ..path_planning.radius_profiles import (
    DEFAULT_ADDITIONAL_ROBOT_RADII_M,
    DEFAULT_BASELINE_ROBOT_RADIUS_M,
    default_robot_radius_for_dataset,
    normalize_robot_radius_profiles,
    radius_profile_directory_name,
    validate_endpoint_manifest_radius,
)


METHOD_NAME = "SplatNav"
REFERENCE_GROUND_TRUTH = "ground_truth_mesh_voxel"
REFERENCE_TSDF = "observed_tsdf_voxel"
REFERENCE_SAMPLE_SPACING_M = 0.05
REFERENCE_SAFETY_MARGIN_M = 0.0
GOAL_TOLERANCE_M = 0.10
ENDPOINT_CONSTRAINT_TOLERANCE_M = 1.0e-3
ENDPOINT_POLICY = "requested_exact_or_reject_occupied_or_outside_fixed_bounds_v2"
PLANNING_BOUNDS_POLICY = "fixed_gsplat_quantile_bounds_v1"
DEFAULT_PLANNING_GRID_CELL_SIZE_M = 0.05
DEFAULT_MAX_PLANNING_GRID_CELL_SIZE_M = 0.10
DEFAULT_GRID_MEMORY_BUDGET_GIB = 8.0
DEFAULT_GRID_FREE_MEMORY_FRACTION = 0.5
BYTES_PER_GIB = 1024**3
COMMAND_FLAGS = {
    "checkpoint": "--checkpoint",
    "reference_map": "--reference-map",
    "endpoints": "--endpoints",
}


@dataclass(frozen=True)
class SceneSpec:
    dataset: str
    scene: str
    checkpoint: Path
    reference_map: Path | None
    endpoints: Path
    reference_kind: str = REFERENCE_TSDF


@dataclass(frozen=True)
class PlannerConfig:
    robot_radius: float = DEFAULT_BASELINE_ROBOT_RADIUS_M
    vmax: float = 0.1
    amax: float = 0.1
    target_cell_size_m: float = DEFAULT_PLANNING_GRID_CELL_SIZE_M
    maximum_cell_size_m: float = DEFAULT_MAX_PLANNING_GRID_CELL_SIZE_M
    grid_memory_budget_gib: float = DEFAULT_GRID_MEMORY_BUDGET_GIB
    grid_free_memory_fraction: float = DEFAULT_GRID_FREE_MEMORY_FRACTION
    global_connectivity: int = 26
    spline_degree: int = 6
    samples_per_section: int = 10
    min_opacity: float = 0.0
    bounds_quantile: float = 0.005
    bounds_padding_fraction: float = 0.05
    lower_bound: tuple[float, float, float] | None = None
    upper_bound: tuple[float, float, float] | None = None
    device: str = "cuda"


@dataclass(frozen=True)
class UnscoredReference:
    """Minimal reference descriptor used when reference scoring is disabled."""

    label: str
    path: Path | None = None


class ResolutionInsufficientError(RuntimeError):
    """The grid would have to be coarser than the accepted metric accuracy."""

    def __init__(self, diagnostics: dict[str, Any]) -> None:
        self.diagnostics = diagnostics
        if diagnostics.get("failure_reason") == "runtime_grid_build_oom_at_maximum_cell_size":
            detail = (
                "SplatNav grid construction still ran out of CUDA memory at "
                "the maximum accepted cell size: "
                f"maximum_acceptable_cell_size="
                f"{diagnostics['maximum_acceptable_cell_size']} m, "
                f"last_resolution={diagnostics['resolution_xyz']}, "
                f"voxel_count={diagnostics['voxel_count']}"
            )
        else:
            detail = (
                "SplatNav grid exceeds the voxel budget at the maximum accepted "
                "cell size: "
                f"target_cell_size={diagnostics['target_cell_size']} m, "
                f"maximum_acceptable_cell_size="
                f"{diagnostics['maximum_acceptable_cell_size']} m, "
                f"required_effective_cell_size="
                f"{diagnostics['effective_cell_size']} m, "
                f"voxel_count={diagnostics['voxel_count']}, "
                f"voxel_budget={diagnostics['voxel_budget']}"
            )
        super().__init__(detail)


def _choose_bounds(
    gaussians: SplatNavGaussians,
    config: PlannerConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose fixed scene bounds from the GSplat, independent of test queries.

    Splat-Plan takes lower and upper grid bounds as environment inputs. Endpoint
    replay must not enlarge that shared environment with an outlying test query;
    such a query is rejected individually before Dijkstra instead.
    """

    if not 0.0 <= config.bounds_quantile < 0.5:
        raise ValueError("bounds_quantile must be in [0, 0.5)")
    means = gaussians.means.detach().cpu().numpy()
    half_extents = (
        gaussians.covs.diagonal(dim1=1, dim2=2)
        .clamp_min(0.0)
        .sqrt()
        .detach()
        .cpu()
        .numpy()
    )
    gaussian_lower = means - half_extents
    gaussian_upper = means + half_extents
    quantile = float(config.bounds_quantile)
    if quantile == 0.0:
        lower = gaussian_lower.min(axis=0)
        upper = gaussian_upper.max(axis=0)
    else:
        lower = np.quantile(gaussian_lower, quantile, axis=0)
        upper = np.quantile(gaussian_upper, 1.0 - quantile, axis=0)
    span = upper - lower
    padding = max(
        2.0 * float(config.robot_radius),
        float(config.bounds_padding_fraction) * float(span.max()),
        1.0e-3,
    )
    lower = lower - padding
    upper = upper + padding
    if np.any(upper <= lower):
        raise ValueError(f"invalid planning bounds: lower={lower}, upper={upper}")
    return lower.astype(np.float32), upper.astype(np.float32)


def _resolution_for_cell_size(
    lower: np.ndarray,
    upper: np.ndarray,
    grid_cell_size_m: float,
) -> tuple[int, int, int]:
    span = upper - lower
    if not np.isfinite(grid_cell_size_m) or float(grid_cell_size_m) <= 0.0:
        raise ValueError("grid_cell_size_m must be finite and positive")
    values = np.maximum(2, np.ceil(span / float(grid_cell_size_m)).astype(int))
    return tuple(int(value) for value in values)


def _dense_grid_allocation_diagnostics(
    resolution: tuple[int, int, int],
) -> dict[str, int]:
    """Return lower-bound storage estimates for SplatNav's dense voxel grid."""

    voxel_count = math.prod(int(value) for value in resolution)
    grid_centers_bytes = voxel_count * 3 * np.dtype(np.float32).itemsize
    occupancy_grid_bytes = voxel_count * np.dtype(np.bool_).itemsize
    return {
        "voxel_count": voxel_count,
        "grid_centers_bytes": grid_centers_bytes,
        "occupancy_grid_bytes_each": occupancy_grid_bytes,
        # Grid construction retains centers and occupancy, and the boolean-shift
        # dilation creates a second occupancy-sized output.
        "minimum_dense_grid_build_bytes": (
            grid_centers_bytes + 2 * occupancy_grid_bytes
        ),
    }


def _grid_memory_budget_diagnostics(
    *,
    cuda_free_bytes: int,
    cuda_total_bytes: int,
    memory_budget_gib: float,
    free_memory_fraction: float,
) -> dict[str, int | float]:
    """Convert configured and currently available VRAM into a voxel budget."""

    if not np.isfinite(memory_budget_gib) or float(memory_budget_gib) <= 0.0:
        raise ValueError("grid_memory_budget_gib must be finite and positive")
    if not np.isfinite(free_memory_fraction) or not (
        0.0 < float(free_memory_fraction) <= 1.0
    ):
        raise ValueError("grid_free_memory_fraction must be in (0, 1]")
    if int(cuda_free_bytes) <= 0 or int(cuda_total_bytes) <= 0:
        raise ValueError("CUDA memory sizes must be positive")

    configured_budget_bytes = int(float(memory_budget_gib) * BYTES_PER_GIB)
    available_budget_bytes = int(
        int(cuda_free_bytes) * float(free_memory_fraction)
    )
    effective_budget_bytes = min(configured_budget_bytes, available_budget_bytes)
    bytes_per_voxel = _dense_grid_allocation_diagnostics((1, 1, 1))[
        "minimum_dense_grid_build_bytes"
    ]
    voxel_budget = effective_budget_bytes // bytes_per_voxel
    if voxel_budget < 8:
        raise MemoryError(
            "CUDA grid memory budget cannot hold the minimum 2 x 2 x 2 grid: "
            f"effective_budget_bytes={effective_budget_bytes}"
        )
    return {
        "configured_grid_memory_budget_gib": float(memory_budget_gib),
        "configured_grid_memory_budget_bytes": configured_budget_bytes,
        "grid_free_memory_fraction": float(free_memory_fraction),
        "free_fraction_budget_bytes": available_budget_bytes,
        "effective_grid_memory_budget_bytes": effective_budget_bytes,
        "estimated_bytes_per_voxel": bytes_per_voxel,
        "voxel_budget": int(voxel_budget),
        "cuda_free_bytes_before_grid_build": int(cuda_free_bytes),
        "cuda_total_bytes": int(cuda_total_bytes),
    }


def _minimum_budget_cell_size(
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    target_cell_size: float,
    voxel_budget: int,
) -> tuple[float, tuple[int, int, int]]:
    """Find the finest isotropic cell-size request that fits the voxel budget."""

    target_resolution = _resolution_for_cell_size(
        lower,
        upper,
        target_cell_size,
    )
    if math.prod(target_resolution) <= int(voxel_budget):
        return float(target_cell_size), target_resolution

    span = np.asarray(upper, dtype=np.float64) - np.asarray(
        lower, dtype=np.float64
    )
    low = float(target_cell_size)
    high = max(low, float(span.max()))
    best_resolution = _resolution_for_cell_size(
        lower,
        upper,
        high,
    )
    if math.prod(best_resolution) > int(voxel_budget):
        raise MemoryError(
            "voxel budget cannot hold the minimum 2 x 2 x 2 planning grid"
        )

    # Resolution changes only at cell-boundary crossings. Keeping the last
    # known fitting tuple avoids a floating-point crossing returning one extra
    # voxel on the final iteration.
    for _ in range(80):
        candidate = (low + high) / 2.0
        candidate_resolution = _resolution_for_cell_size(
            lower,
            upper,
            candidate,
        )
        if math.prod(candidate_resolution) <= int(voxel_budget):
            high = candidate
            best_resolution = candidate_resolution
        else:
            low = candidate
    return high, best_resolution


def _adaptive_grid_diagnostics(
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    target_cell_size: float,
    maximum_acceptable_cell_size: float,
    robot_radius: float,
    voxel_budget: int,
) -> dict[str, Any]:
    """Select the finest metric grid allowed by accuracy and memory limits."""

    if not np.isfinite(target_cell_size) or float(target_cell_size) <= 0.0:
        raise ValueError("target_cell_size must be finite and positive")
    if not np.isfinite(maximum_acceptable_cell_size) or (
        float(maximum_acceptable_cell_size) < float(target_cell_size)
    ):
        raise ValueError(
            "maximum_acceptable_cell_size must be finite and at least "
            "target_cell_size"
        )
    if not np.isfinite(robot_radius) or float(robot_radius) <= 0.0:
        raise ValueError("robot_radius must be finite and positive")
    if int(voxel_budget) < 8:
        raise ValueError("voxel_budget must be at least 8")

    target_resolution = _resolution_for_cell_size(
        lower,
        upper,
        float(target_cell_size),
    )
    target_voxel_count = math.prod(target_resolution)
    selection_cell_size, resolution = _minimum_budget_cell_size(
        lower,
        upper,
        target_cell_size=float(target_cell_size),
        voxel_budget=int(voxel_budget),
    )
    allocation = _dense_grid_allocation_diagnostics(resolution)
    span = np.asarray(upper, dtype=np.float64) - np.asarray(
        lower, dtype=np.float64
    )
    cell_size_xyz = span / np.asarray(resolution, dtype=np.float64)
    effective_cell_size = float(cell_size_xyz.max())
    maximum_resolution = _resolution_for_cell_size(
        lower,
        upper,
        float(maximum_acceptable_cell_size),
    )
    maximum_voxel_count = math.prod(maximum_resolution)
    memory_limited = target_voxel_count > int(voxel_budget)
    resolution_insufficient = maximum_voxel_count > int(voxel_budget)
    return {
        "policy": "adaptive_metric_cell_size_under_cuda_voxel_budget_v1",
        "target_cell_size": float(target_cell_size),
        "effective_cell_size": effective_cell_size,
        "selection_cell_size": float(selection_cell_size),
        "maximum_acceptable_cell_size": float(maximum_acceptable_cell_size),
        "resolution_xyz": [int(value) for value in resolution],
        "voxel_count": int(allocation["voxel_count"]),
        "voxel_budget": int(voxel_budget),
        "target_resolution_xyz": [int(value) for value in target_resolution],
        "target_voxel_count": int(target_voxel_count),
        "maximum_acceptable_resolution_xyz": [
            int(value) for value in maximum_resolution
        ],
        "maximum_acceptable_voxel_count": int(maximum_voxel_count),
        "memory_limited": bool(memory_limited),
        "effective_cell_size_over_robot_radius": (
            effective_cell_size / float(robot_radius)
        ),
        "resolution_insufficient": bool(resolution_insufficient),
        "cell_size_xyz": cell_size_xyz.astype(float).tolist(),
        "estimated_dense_grid_bytes": int(
            allocation["minimum_dense_grid_build_bytes"]
        ),
    }


def _next_coarser_grid(
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    current_cell_size: float,
    maximum_cell_size: float,
    current_resolution: tuple[int, int, int],
) -> tuple[float, tuple[int, int, int]] | None:
    """Return the next distinct 25%-coarser grid, capped by accuracy."""

    if current_cell_size >= maximum_cell_size - 1.0e-12:
        return None
    next_cell_size = min(maximum_cell_size, current_cell_size * 1.25)
    while True:
        next_resolution = _resolution_for_cell_size(
            lower, upper, next_cell_size
        )
        if next_resolution != current_resolution:
            return float(next_cell_size), next_resolution
        if next_cell_size >= maximum_cell_size:
            return None
        next_cell_size = min(maximum_cell_size, next_cell_size * 1.25)


def _peak_vram_mb(resource_summary: dict[str, Any]) -> float | None:
    """Return the observed PyTorch reserved-memory peak in MiB."""

    value = resource_summary.get("peak_cuda_torch_reserved_bytes")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return float(value) / (1024.0**2)


def _endpoint_bounds_diagnostics(
    start: np.ndarray,
    goal: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, Any]:
    """Describe whether one requested endpoint pair lies in fixed scene bounds."""

    start_array = np.asarray(start, dtype=np.float32).reshape(3)
    goal_array = np.asarray(goal, dtype=np.float32).reshape(3)
    lower_array = np.asarray(lower, dtype=np.float32).reshape(3)
    upper_array = np.asarray(upper, dtype=np.float32).reshape(3)
    tolerance = 1.0e-6

    def outside_axes(point: np.ndarray) -> list[int]:
        outside = np.logical_or(
            point < lower_array - tolerance,
            point > upper_array + tolerance,
        )
        return np.flatnonzero(outside).astype(int).tolist()

    start_outside_axes = outside_axes(start_array)
    goal_outside_axes = outside_axes(goal_array)
    return {
        "fixed_scene_bounds_policy": PLANNING_BOUNDS_POLICY,
        "fixed_scene_bounds": {
            "lower": lower_array.astype(float).tolist(),
            "upper": upper_array.astype(float).tolist(),
        },
        "start_outside_axes": start_outside_axes,
        "goal_outside_axes": goal_outside_axes,
        "start_in_fixed_scene_bounds": not start_outside_axes,
        "goal_in_fixed_scene_bounds": not goal_outside_axes,
        "both_endpoints_in_fixed_scene_bounds": not (
            start_outside_axes or goal_outside_axes
        ),
    }


def _safety_parameter_contract(
    *,
    robot_radius_m: float,
    vmax: float,
    amax: float,
) -> dict[str, Any]:
    """Describe the common clearance and SplatNav-only broadphase settings."""

    radius = float(robot_radius_m)
    speed = float(vmax)
    acceleration = float(amax)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("robot_radius_m must be finite and positive")
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError("vmax must be finite and positive")
    if not np.isfinite(acceleration) or acceleration <= 0.0:
        raise ValueError("amax must be finite and positive")
    braking_distance = speed * speed / (2.0 * acceleration)
    return {
        "robot_geometry": "sphere",
        "robot_radius_m": radius,
        "global_grid_gaussian_inflation_radius_m": radius,
        "local_spline_gaussian_clearance_radius_m": radius,
        "reference_evaluation_robot_radius_m": radius,
        "reference_safety_margin_m": REFERENCE_SAFETY_MARGIN_M,
        "common_physical_clearance_m": radius + REFERENCE_SAFETY_MARGIN_M,
        "collision_candidate_broadphase_braking_distance_m": braking_distance,
        "collision_candidate_broadphase_radius_m": radius + braking_distance,
        "broadphase_braking_distance_changes_required_clearance": False,
        "note": (
            "SplatNav uses radius + vmax^2/(2*amax) only to collect nearby "
            "Gaussian collision candidates; global-grid inflation, spline "
            "constraints, and reference scoring all use the same robot radius "
            "recorded above."
        ),
    }


def _path_output_is_nonempty(path: Any) -> bool:
    if path is None:
        return False
    try:
        return bool(np.asarray(path).size > 0)
    except (TypeError, ValueError):
        return False


def _prepare_optional_open3d() -> bool:
    """Allow planning-only SplatNav imports when Open3D is not installed."""

    try:
        importlib.import_module("open3d")
        return True
    except ModuleNotFoundError as error:
        if error.name != "open3d":
            raise
        sys.modules["open3d"] = types.ModuleType("open3d")
        return False


def _check_runtime_dependencies() -> None:
    required = (
        "torch",
        "numpy",
        "scipy",
        "sympy",
        "clarabel",
        "cvxpy",
        "dijkstra3d",
        "polytope",
    )
    # Do not execute package imports here. Some packages have optional or
    # environment-specific transitive imports; treating those failures as if
    # every top-level package were absent hides the actual error. The planner's
    # normal imports below will preserve a precise traceback for broken imports.
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            "missing SplatNav planning dependencies in "
            f"{sys.executable}: {', '.join(missing)}"
        )


def _patch_single_voxel_unfold(splatplan_module: Any) -> None:
    """Fix the original grid builder's singleton-kernel squeeze edge case.

    With a 1x1x1 robot kernel, ``tensor.squeeze()`` removes both the batch and
    kernel dimensions. The subsequent one-element boolean mask then indexes the
    flattened voxel dimension. Building the same raw Gaussian occupancy without
    the no-op 1x1x1 dilation preserves the intended result.
    """

    original_class = splatplan_module.GSplatVoxel
    if getattr(original_class, "_gspar_singleton_kernel_compatible", False):
        return

    class CompatibleGSplatVoxel(original_class):
        _gspar_singleton_kernel_compatible = True

        def create_navigable_grid(self, unfold=True, chunk=1_000_000):
            singleton_kernel = (
                unfold
                and self.robot_mask is not None
                and int(self.robot_mask.numel()) == 1
            )
            if not singleton_kernel:
                return super().create_navigable_grid(unfold=unfold, chunk=chunk)

            original_radius = self.radius
            self.radius = 0.0
            try:
                return super().create_navigable_grid(unfold=False, chunk=chunk)
            finally:
                self.radius = original_radius

    splatplan_module.GSplatVoxel = CompatibleGSplatVoxel


def _patch_global_connectivity(grid_utils_module: Any, connectivity: int) -> None:
    """Use the shared 3-D neighborhood while retaining SplatNav's Dijkstra."""

    value = int(connectivity)
    if value not in (6, 18, 26):
        raise ValueError("global_connectivity must be one of 6, 18, or 26")
    current = grid_utils_module.astar3D
    if getattr(current, "_gspar_connectivity", None) == value:
        return

    import dijkstra3d

    def aligned_dijkstra(field, source, target, feasible):
        indices = dijkstra3d.binary_dijkstra(
            field,
            source,
            target,
            connectivity=value,
            background_color=1,
        )
        if indices is None or np.asarray(indices).size == 0:
            empty_indices = np.empty((0, 3), dtype=np.int32)
            return np.empty((0, 3), dtype=np.float32), empty_indices
        indices = np.asarray(indices, dtype=np.int32).reshape(-1, 3)
        trajectory = feasible[indices[:, 0], indices[:, 1], indices[:, 2]]
        return trajectory, indices

    aligned_dijkstra._gspar_connectivity = value
    grid_utils_module.astar3D = aligned_dijkstra


def _as_numpy_vector(value: Any) -> np.ndarray:
    """Copy a NumPy or torch vector to a three-element CPU array."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float32).reshape(3)


def _patch_exact_endpoint_paths(splatplan_module: Any) -> None:
    """Make SplatPlan optimize to requested endpoints instead of voxel centers.

    The original voxel planner returns an A* path whose first and last points
    are voxel centers. It also silently projects occupied endpoints to another
    free voxel. SplatPlan subsequently constrains its Bezier curve to those A*
    endpoints, not to the requested points. For endpoint-replay evaluation we
    instead reject endpoints that SplatNav considers occupied and replace the
    A* path's first/last centers with the exact requested coordinates. This
    causes the original corridor builder to process the adjusted first and last
    segments before the spline optimizer applies its endpoint constraints.
    """

    original_class = splatplan_module.GSplatVoxel
    if getattr(original_class, "_gspar_exact_endpoint_compatible", False):
        return

    class ExactEndpointGSplatVoxel(original_class):
        _gspar_exact_endpoint_compatible = True

        def create_path(self, x0, xf):
            requested_start = _as_numpy_vector(x0)
            requested_goal = _as_numpy_vector(xf)
            source = self.get_indices(x0)
            target = self.get_indices(xf)
            source_index = _as_numpy_vector(source).astype(np.int64)
            target_index = _as_numpy_vector(target).astype(np.int64)
            start_center = _as_numpy_vector(
                self.grid_centers[tuple(int(value) for value in source_index)]
            )
            goal_center = _as_numpy_vector(
                self.grid_centers[tuple(int(value) for value in target_index)]
            )
            start_occupied = bool(
                self.non_navigable_grid[
                    tuple(int(value) for value in source_index)
                ].item()
            )
            goal_occupied = bool(
                self.non_navigable_grid[
                    tuple(int(value) for value in target_index)
                ].item()
            )
            diagnostics = {
                "policy": ENDPOINT_POLICY,
                "requested_start_world": requested_start.tolist(),
                "requested_goal_world": requested_goal.tolist(),
                "astar_start_voxel_index": source_index.tolist(),
                "astar_goal_voxel_index": target_index.tolist(),
                "astar_start_center_world": start_center.tolist(),
                "astar_goal_center_world": goal_center.tolist(),
                "start_to_voxel_center_distance_m": float(
                    np.linalg.norm(requested_start - start_center)
                ),
                "goal_to_voxel_center_distance_m": float(
                    np.linalg.norm(requested_goal - goal_center)
                ),
                "start_endpoint_occupied": start_occupied,
                "goal_endpoint_occupied": goal_occupied,
                "start_endpoint_projection_prevented": start_occupied,
                "goal_endpoint_projection_prevented": goal_occupied,
                "legacy_endpoint_projection_applied": False,
                "exact_endpoints_inserted": False,
            }
            self._gspar_endpoint_diagnostics = diagnostics

            occupied = [
                name
                for name, is_occupied in (
                    ("start", start_occupied),
                    ("goal", goal_occupied),
                )
                if is_occupied
            ]
            if occupied:
                diagnostics["status"] = "rejected_occupied_endpoint"
                raise ValueError(
                    "SplatNav rejected requested endpoint(s) classified occupied: "
                    + ", ".join(occupied)
                )

            center_path = super().create_path(x0, xf)
            if center_path is None:
                diagnostics["status"] = "astar_failed"
                raise RuntimeError("SplatNav A* did not return a path")
            center_path = np.asarray(center_path, dtype=np.float32).reshape(-1, 3)
            if center_path.shape[0] == 0:
                diagnostics["status"] = "astar_returned_empty_path"
                raise RuntimeError("SplatNav A* returned an empty path")

            # Replace the snapped endpoints, rather than appending them, to
            # avoid zero-length segments when a request is already at a center.
            exact_path = np.concatenate(
                [
                    requested_start[None, :],
                    center_path[1:-1],
                    requested_goal[None, :],
                ],
                axis=0,
            )
            diagnostics["status"] = "exact_endpoints_inserted"
            diagnostics["astar_center_path_point_count"] = int(center_path.shape[0])
            diagnostics["exact_endpoints_inserted"] = True
            return exact_path

    splatplan_module.GSplatVoxel = ExactEndpointGSplatVoxel


def _returned_endpoint_diagnostics(
    path: np.ndarray | None,
    start: np.ndarray,
    goal: np.ndarray,
) -> dict[str, Any]:
    if path is None or len(path) == 0:
        return {
            "returned_start_error_m": None,
            "returned_goal_error_m": None,
            "endpoint_contract_satisfied": False,
        }
    start_error = float(np.linalg.norm(np.asarray(path[0]) - np.asarray(start)))
    goal_error = float(np.linalg.norm(np.asarray(path[-1]) - np.asarray(goal)))
    return {
        "returned_start_error_m": start_error,
        "returned_goal_error_m": goal_error,
        "endpoint_contract_satisfied": bool(
            start_error <= ENDPOINT_CONSTRAINT_TOLERANCE_M
            and goal_error <= ENDPOINT_CONSTRAINT_TOLERANCE_M
        ),
    }


def _resolve_from_root(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project_root / path).resolve()


def _command_value(text: str, flag: str) -> str:
    match = re.search(rf"{re.escape(flag)}\s+([^\s\\]+)", text)
    if match is None:
        raise ValueError(f"command file does not contain {flag}")
    return match.group(1)


def _reference_kind(reference_map: Path | None) -> str:
    """Infer the evaluation protocol from the conventional voxel-map name."""

    if reference_map is not None and reference_map.stem == "voxel_map_gt":
        return REFERENCE_GROUND_TRUTH
    return REFERENCE_TSDF


def scene_from_command_file(path: Path, project_root: Path) -> SceneSpec:
    text = path.read_text(encoding="utf-8")
    dataset_match = re.search(r"^#\s*数据集:\s*(\S+)\s*$", text, re.MULTILINE)
    scene_match = re.search(r"^#\s*场景:\s*(\S+)\s*$", text, re.MULTILINE)
    dataset = dataset_match.group(1) if dataset_match else path.parent.name
    scene = scene_match.group(1) if scene_match else path.stem
    values = {
        name: _resolve_from_root(_command_value(text, flag), project_root)
        for name, flag in COMMAND_FLAGS.items()
    }
    return SceneSpec(
        dataset=dataset,
        scene=scene,
        reference_kind=_reference_kind(values["reference_map"]),
        **values,
    )


def scenes_from_command_source(source: Path, project_root: Path) -> list[SceneSpec]:
    files = [source] if source.is_file() else sorted(source.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no command txt files found under {source}")
    return [scene_from_command_file(path, project_root) for path in files]


def _parse_scene_names(block: str) -> list[str]:
    names: list[str] = []
    for match in re.finditer(r"^\s+scenes:\s*(.+?)\s*$", block, re.MULTILINE):
        value = match.group(1).rstrip(":：").strip()
        for name in value.split(","):
            normalized = name.strip().rstrip(":：")
            if normalized and normalized not in names:
                names.append(normalized)
    return names


def scenes_from_dataset_config(path: Path, project_root: Path) -> list[SceneSpec]:
    """Parse the project's human-readable dataset.txt format.

    All ``scenes:`` lines in a dataset block are unioned.  This intentionally
    includes the indoor override lines for m60/panther in the current file.
    """

    text = path.read_text(encoding="utf-8")
    headings = list(re.finditer(r"^(\S[^:\n]*):\s*$", text, re.MULTILINE))
    scenes: list[SceneSpec] = []
    for index, heading in enumerate(headings):
        dataset = heading.group(1).strip()
        begin = heading.end()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        block = text[begin:end]
        gt_match = re.search(r"^\s+gt\s*:\s*(.*?)\s*$", block, re.MULTILINE)
        has_gt = bool(
            gt_match
            and gt_match.group(1).strip().lower() not in {"none", "null"}
        )
        for scene in _parse_scene_names(block):
            scene_root = project_root / "GSPAR" / "outputs" / dataset / scene
            reference_kind = REFERENCE_GROUND_TRUTH if has_gt else REFERENCE_TSDF
            scenes.append(
                SceneSpec(
                    dataset=dataset,
                    scene=scene,
                    checkpoint=(
                        project_root / "output" / dataset / scene / "checkpoints/final.pt"
                    ).resolve(),
                    reference_map=(
                        scene_root
                        / ("voxel_map_gt.npz" if has_gt else "voxel_map_tsdf.npz")
                    ).resolve(),
                    endpoints=(
                        scene_root
                        / (
                            "path_evaluation_gt/endpoints.json"
                            if has_gt
                            else "path_evaluation/endpoints.json"
                        )
                    ).resolve(),
                    reference_kind=reference_kind,
                )
            )
    if not scenes:
        raise ValueError(f"no datasets/scenes found in {path}")
    return scenes


def discover_scenes(source: Path, project_root: Path) -> list[SceneSpec]:
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"scene source not found: {source}")
    if source.is_file() and source.name == "dataset.txt":
        return scenes_from_dataset_config(source, project_root)
    return scenes_from_command_source(source, project_root)


def _load_endpoint_manifest(
    path: Path,
    num_trials: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError(f"endpoint manifest has no pairs list: {path}")
    requested = int(num_trials)
    if requested <= 0:
        raise ValueError("num_trials must be positive")
    if len(pairs) < requested:
        raise ValueError(f"{path} contains {len(pairs)} pairs, requested {requested}")
    normalized = []
    for index, pair in enumerate(pairs[:requested]):
        normalized.append(
            {
                "trial_id": pair.get("trial_id", index),
                "start_world": np.asarray(pair["start_world"], dtype=np.float32).reshape(3),
                "goal_world": np.asarray(pair["goal_world"], dtype=np.float32).reshape(3),
            }
        )
    return manifest, normalized


def _validate_endpoints_against_reference(
    manifest: dict[str, Any],
    endpoints: list[dict[str, Any]],
    reference: VoxelReference,
    *,
    robot_radius_m: float,
) -> None:
    """Reject endpoint/reference combinations that cannot be scored fairly."""

    manifest_radius = manifest.get("robot_radius_m")
    if (
        manifest_radius is not None
        and float(manifest_radius) + 1.0e-9 < float(robot_radius_m)
    ):
        raise ValueError(
            "endpoint manifest radius is smaller than the SplatNav evaluation: "
            f"manifest={float(manifest_radius):g} m, "
            f"evaluation={float(robot_radius_m):g} m; regenerate endpoints for "
            "the largest evaluated radius"
        )

    requires_observed_free = manifest.get("unknown_endpoints_allowed") is False
    requires_free_footprint = bool(
        manifest.get("reference_observed_free_robot_footprint_required", False)
    )
    if not requires_observed_free and not requires_free_footprint:
        return

    points = np.stack(
        [
            pair[key]
            for pair in endpoints
            for key in ("start_world", "goal_world")
        ],
        axis=0,
    )
    valid = reference.observed_free_mask(points)
    if requires_free_footprint:
        footprint_free, footprint_in_bounds = reference.observed_free_footprint_mask(
            points, float(robot_radius_m)
        )
        valid &= footprint_free & footprint_in_bounds
    if bool(valid.all()):
        return

    invalid_labels = []
    for point_index in np.flatnonzero(~valid)[:8]:
        pair = endpoints[int(point_index) // 2]
        endpoint_name = "start" if int(point_index) % 2 == 0 else "goal"
        invalid_labels.append(f"trial {pair['trial_id']} {endpoint_name}")
    manifest_reference = manifest.get("reference_map") or manifest.get("occupancy_map")
    raise ValueError(
        "endpoint manifest does not match the supplied evaluation reference: "
        f"{int((~valid).sum())}/{len(valid)} selected endpoints are not explicitly "
        "observed-free with the required robot footprint; invalid examples: "
        f"{', '.join(invalid_labels)}. Manifest reference: {manifest_reference}; "
        f"supplied reference: {reference.label}"
    )


def _trial_seed(base_seed: int, trial_id: str | int | None) -> int:
    payload = f"{max(0, int(base_seed))}:{trial_id!r}".encode("utf-8")
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "little")


def _tensor_bytes(values: Iterable[Any]) -> int:
    seen: set[tuple[str, int]] = set()
    total = 0
    for value in values:
        if not hasattr(value, "numel") or not hasattr(value, "element_size"):
            continue
        try:
            key = (str(value.device), int(value.data_ptr()))
            if key in seen:
                continue
            seen.add(key)
            total += int(value.numel()) * int(value.element_size())
        except (AttributeError, RuntimeError, TypeError):
            continue
    return total


class ReusableSplatNavPlanner:
    """One checkpoint/grid initialization followed by multiple path queries."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        splatnav_root: Path,
        robot_radius: float,
        vmax: float,
        amax: float,
        target_cell_size_m: float,
        maximum_cell_size_m: float,
        grid_memory_budget_gib: float,
        grid_free_memory_fraction: float,
        global_connectivity: int,
        min_opacity: float,
        bounds_quantile: float,
        bounds_padding_fraction: float,
        spline_degree: int,
        samples_per_section: int,
        device: str,
        map_from_training_world: np.ndarray | None = None,
    ) -> None:
        root = splatnav_root.expanduser().resolve()
        if not (root / "splatplan/splatplan.py").is_file():
            raise FileNotFoundError(f"original SplatNav source not found under {root}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        _check_runtime_dependencies()
        open3d_available = _prepare_optional_open3d()
        import torch
        splatplan_module = importlib.import_module("splatplan.splatplan")
        grid_utils_module = importlib.import_module("initialization.grid_utils")
        from splatplan.spline_utils import SplinePlanner

        _patch_single_voxel_unfold(splatplan_module)
        _patch_exact_endpoint_paths(splatplan_module)
        _patch_global_connectivity(grid_utils_module, global_connectivity)
        SplatPlan = splatplan_module.SplatPlan

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("SplatNav currently requires an available CUDA device")
        if self.device.index is not None:
            torch.cuda.set_device(self.device)

        config = PlannerConfig(
            robot_radius=float(robot_radius),
            vmax=float(vmax),
            amax=float(amax),
            target_cell_size_m=float(target_cell_size_m),
            maximum_cell_size_m=float(maximum_cell_size_m),
            grid_memory_budget_gib=float(grid_memory_budget_gib),
            grid_free_memory_fraction=float(grid_free_memory_fraction),
            global_connectivity=int(global_connectivity),
            spline_degree=int(spline_degree),
            samples_per_section=int(samples_per_section),
            min_opacity=float(min_opacity),
            bounds_quantile=float(bounds_quantile),
            bounds_padding_fraction=float(bounds_padding_fraction),
            device=str(device),
        )
        with ResourceMonitor(cuda_device=str(self.device)) as load_monitor:
            load_started = time.perf_counter()
            self.gaussians = load_nerficg_gaussians(
                checkpoint,
                device=self.device,
                min_opacity=config.min_opacity,
                map_frame_transform=map_from_training_world,
            )
            self.checkpoint_load_time_sec = float(
                time.perf_counter() - load_started
            )
        self.checkpoint_load_resource_usage = load_monitor.to_dict()

        lower, upper = _choose_bounds(self.gaussians, config)
        fixed_config = replace(
            config,
            lower_bound=tuple(float(value) for value in lower),
            upper_bound=tuple(float(value) for value in upper),
        )
        cuda_free_bytes, cuda_total_bytes = torch.cuda.mem_get_info(self.device)
        memory_budget = _grid_memory_budget_diagnostics(
            cuda_free_bytes=int(cuda_free_bytes),
            cuda_total_bytes=int(cuda_total_bytes),
            memory_budget_gib=fixed_config.grid_memory_budget_gib,
            free_memory_fraction=fixed_config.grid_free_memory_fraction,
        )
        grid_selection = _adaptive_grid_diagnostics(
            lower,
            upper,
            target_cell_size=fixed_config.target_cell_size_m,
            maximum_acceptable_cell_size=fixed_config.maximum_cell_size_m,
            robot_radius=fixed_config.robot_radius,
            voxel_budget=int(memory_budget["voxel_budget"]),
        )
        grid_selection.update(memory_budget)
        resolved_resolution = tuple(grid_selection["resolution_xyz"])
        if grid_selection["resolution_insufficient"]:
            load_summary = summarize_resource_usages(
                {"checkpoint_load": self.checkpoint_load_resource_usage}
            )
            grid_selection["peak_vram_mb"] = _peak_vram_mb(load_summary)
            grid_selection["peak_vram_scope"] = "checkpoint_load_only"
            grid_selection["bounds"] = {
                "lower": lower.astype(float).tolist(),
                "upper": upper.astype(float).tolist(),
            }
            raise ResolutionInsufficientError(grid_selection)

        grid_allocation = _dense_grid_allocation_diagnostics(resolved_resolution)
        grid_allocation.update(memory_budget)
        if grid_allocation["voxel_count"] > int(memory_budget["voxel_budget"]):
            raise MemoryError(
                "adaptive SplatNav grid exceeds the configured CUDA voxel budget: "
                f"bounds=({lower.tolist()}, {upper.tolist()}), "
                f"resolution={resolved_resolution}, "
                f"voxel_count={grid_allocation['voxel_count']}, "
                f"voxel_budget={memory_budget['voxel_budget']}"
            )

        torch.manual_seed(0)
        np.random.seed(0)
        lower_tensor = torch.as_tensor(lower, dtype=torch.float32, device=self.device)
        upper_tensor = torch.as_tensor(upper, dtype=torch.float32, device=self.device)
        spline_planner = SplinePlanner(
            spline_deg=fixed_config.spline_degree,
            N_sec=fixed_config.samples_per_section,
            device=self.device,
            use_cvxpy=False,
        )
        runtime_build_failed = False
        build_attempts: list[dict[str, Any]] = []
        selection_cell_size = grid_selection.get("selection_cell_size")
        with ResourceMonitor(cuda_device=str(self.device)) as build_monitor:
            build_started = time.perf_counter()
            while True:
                resolution_tensor = torch.as_tensor(
                    resolved_resolution,
                    dtype=torch.int64,
                    device=self.device,
                )
                attempt = {
                    "selection_cell_size": selection_cell_size,
                    "resolution_xyz": list(resolved_resolution),
                    "voxel_count": math.prod(resolved_resolution),
                }
                build_attempts.append(attempt)
                try:
                    torch.manual_seed(0)
                    np.random.seed(0)
                    self.planner = SplatPlan(
                        gsplat=self.gaussians,
                        robot_config={
                            "radius": fixed_config.robot_radius,
                            "vmax": fixed_config.vmax,
                            "amax": fixed_config.amax,
                        },
                        env_config={
                            "lower_bound": lower_tensor,
                            "upper_bound": upper_tensor,
                            "resolution": resolution_tensor,
                        },
                        spline_planner=spline_planner,
                        device=self.device,
                    )
                    attempt["status"] = "succeeded"
                    if not open3d_available:
                        # Open3D is used only for optional mesh output. The
                        # original planner attempts a diagnostic mesh export.
                        self.planner.save_polytope = (
                            lambda *_args, **_kwargs: False
                        )
                    break
                except torch.OutOfMemoryError as error:
                    attempt["status"] = "cuda_out_of_memory"
                    attempt["error"] = f"{type(error).__name__}: {error}"
                    # A failed constructor can retain its partial voxel tensors
                    # through the exception traceback. Break those references
                    # before releasing the CUDA allocator cache and retrying.
                    error.__traceback__ = None
                    self.__dict__.pop("planner", None)
                    del resolution_tensor
                    gc.collect()
                    torch.cuda.empty_cache()

                    if selection_cell_size is None:
                        raise RuntimeError(
                            "adaptive grid retry has no current cell size"
                        )
                    next_grid = _next_coarser_grid(
                        lower,
                        upper,
                        current_cell_size=float(selection_cell_size),
                        maximum_cell_size=fixed_config.maximum_cell_size_m,
                        current_resolution=resolved_resolution,
                    )
                    if next_grid is None:
                        runtime_build_failed = True
                        break
                    selection_cell_size, resolved_resolution = next_grid
                    print(
                        "SplatNav grid build OOM; retrying with "
                        f"cell_size={selection_cell_size:g}, "
                        f"resolution={resolved_resolution}"
                    )
                    cell_size_xyz = (
                        upper.astype(np.float64) - lower.astype(np.float64)
                    ) / np.asarray(resolved_resolution, dtype=np.float64)
                    grid_allocation = _dense_grid_allocation_diagnostics(
                        resolved_resolution
                    )
                    grid_allocation.update(memory_budget)
                    grid_selection.update(
                        {
                            "selection_cell_size": selection_cell_size,
                            "effective_cell_size": float(cell_size_xyz.max()),
                            "resolution_xyz": list(resolved_resolution),
                            "voxel_count": int(grid_allocation["voxel_count"]),
                            "memory_limited": True,
                            "runtime_memory_limited": True,
                            "effective_cell_size_over_robot_radius": (
                                float(cell_size_xyz.max())
                                / fixed_config.robot_radius
                            ),
                            "cell_size_xyz": cell_size_xyz.astype(float).tolist(),
                            "estimated_dense_grid_bytes": int(
                                grid_allocation[
                                    "minimum_dense_grid_build_bytes"
                                ]
                            ),
                        }
                    )
            self.planner_build_time_sec = float(
                time.perf_counter() - build_started
            )

        self.build_resource_usage = build_monitor.to_dict()
        grid_selection["grid_build_attempts"] = build_attempts
        grid_selection["grid_build_retry_count"] = max(
            0, len(build_attempts) - 1
        )
        if runtime_build_failed:
            grid_selection["resolution_insufficient"] = True
            grid_selection["failure_reason"] = (
                "runtime_grid_build_oom_at_maximum_cell_size"
            )
            grid_selection["bounds"] = {
                "lower": lower.astype(float).tolist(),
                "upper": upper.astype(float).tolist(),
            }
            failed_build_summary = summarize_resource_usages(
                {
                    "checkpoint_load": self.checkpoint_load_resource_usage,
                    "planner_build": self.build_resource_usage,
                }
            )
            grid_selection["peak_vram_mb"] = _peak_vram_mb(
                failed_build_summary
            )
            grid_selection["peak_vram_scope"] = (
                "checkpoint_load_and_failed_grid_build_attempts"
            )
            raise ResolutionInsufficientError(grid_selection)

        self.config = fixed_config
        self.lower = lower
        self.upper = upper
        self.resolution = resolved_resolution
        self.cell_size = (upper - lower) / np.asarray(resolved_resolution, dtype=np.float32)
        self.grid_selection_diagnostics = grid_selection
        self.grid_allocation_diagnostics = grid_allocation
        self.open3d_available = open3d_available
        self.representation_tensor_bytes = _tensor_bytes(
            list(vars(self.gaussians).values())
            + list(vars(self.planner.gsplat_voxel).values())
        )

    def query(
        self,
        *,
        trial_id: str | int | None,
        start: np.ndarray,
        goal: np.ndarray,
        seed: int,
        reference: VoxelReference | UnscoredReference,
        compute_reference_metrics: bool,
    ) -> PathEvaluation:
        torch = self.torch
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        np.random.seed(int(seed))
        path = None
        output = None
        native_success = False
        global_planner_success = False
        local_planner_success = False
        failure_reason = ""
        timings: dict[str, float] = {}
        endpoint_diagnostics: dict[str, Any] = {}
        self.planner.gsplat_voxel._gspar_endpoint_diagnostics = {}
        with ResourceMonitor(cuda_device=str(self.device)) as query_monitor:
            started = time.perf_counter()
            try:
                bounds_diagnostics = _endpoint_bounds_diagnostics(
                    start,
                    goal,
                    self.lower,
                    self.upper,
                )
                endpoint_diagnostics.update(bounds_diagnostics)
                if not bounds_diagnostics["both_endpoints_in_fixed_scene_bounds"]:
                    endpoint_diagnostics["status"] = "rejected_outside_fixed_scene_bounds"
                    raise ValueError(
                        "SplatNav rejected requested endpoint(s) outside the fixed "
                        "GSplat planning bounds: "
                        f"start_outside_axes={bounds_diagnostics['start_outside_axes']}, "
                        f"goal_outside_axes={bounds_diagnostics['goal_outside_axes']}"
                    )

                start_tensor = torch.as_tensor(
                    start, dtype=torch.float32, device=self.device
                )
                goal_tensor = torch.as_tensor(
                    goal, dtype=torch.float32, device=self.device
                )
                output = self.planner.generate_path(start_tensor, goal_tensor)
                global_planner_success = _path_output_is_nonempty(output.get("path"))
                local_planner_success = bool(output.get("feasible", False))
                trajectory = np.asarray(output["traj"], dtype=np.float32)
                if trajectory.ndim != 2 or trajectory.shape[1] < 3:
                    raise RuntimeError(f"unexpected SplatNav trajectory shape: {trajectory.shape}")
                path = trajectory[:, :3]
                native_success = bool(
                    global_planner_success and local_planner_success
                )
                timings = {
                    "astar": float(output.get("times_astar", 0.0)),
                    "collision_set": float(output.get("times_collision_set", 0.0)),
                    "polytope": float(output.get("times_polytope", 0.0)),
                    "optimization": float(output.get("times_opt", 0.0)),
                }
            except Exception as error:
                failure_reason = f"{type(error).__name__}: {error}"
                out_of_memory_type = getattr(torch, "OutOfMemoryError", None)
                if out_of_memory_type is not None and isinstance(
                    error, out_of_memory_type
                ):
                    endpoint_diagnostics["status"] = "query_cuda_out_of_memory"
                    # Query-local intermediates are no longer useful. Release
                    # cached blocks so the next endpoint can use the shared grid.
                    torch.cuda.empty_cache()
            planner_endpoint_diagnostics = dict(
                getattr(
                    self.planner.gsplat_voxel,
                    "_gspar_endpoint_diagnostics",
                    {},
                )
            )
            endpoint_diagnostics.update(planner_endpoint_diagnostics)
            if endpoint_diagnostics.get("status") == "exact_endpoints_inserted":
                # Local corridor/spline construction can raise after Dijkstra
                # has already returned a valid global seed path.
                global_planner_success = True
            endpoint_diagnostics["spline_solver_success"] = (
                None if output is None else bool(output.get("feasible", False))
            )
            query_time = float(time.perf_counter() - started)

        returned_endpoint_diagnostics = _returned_endpoint_diagnostics(
            path, start, goal
        )
        endpoint_diagnostics.update(returned_endpoint_diagnostics)
        if native_success and not returned_endpoint_diagnostics[
            "endpoint_contract_satisfied"
        ]:
            native_success = False
            failure_reason = (
                "SplatNav endpoint constraint violation: "
                f"start_error={returned_endpoint_diagnostics['returned_start_error_m']} m, "
                f"goal_error={returned_endpoint_diagnostics['returned_goal_error_m']} m, "
                f"tolerance={ENDPOINT_CONSTRAINT_TOLERANCE_M} m"
            )

        effective_start = None if path is None or len(path) == 0 else path[0]
        effective_goal = None if path is None or len(path) == 0 else path[-1]
        return evaluate_path_metrics(
            method=METHOD_NAME,
            trial_id=trial_id,
            path_world=path,
            reference=reference,
            robot_radius=float(self.config.robot_radius),
            native_success=native_success,
            failure_reason=failure_reason,
            start_world=start,
            goal_world=goal,
            effective_start_world=effective_start,
            effective_goal_world=effective_goal,
            trajectory_type="splatnav_bezier",
            computation_time_sec=query_time,
            global_planning_time_sec=timings.get("astar"),
            local_planning_time_sec=sum(
                timings.get(name, 0.0)
                for name in ("collision_set", "polytope", "optimization")
            ),
            planner_validation_time_sec=0.0,
            global_planner_success=global_planner_success,
            local_planner_success=local_planner_success,
            resource_usage=query_monitor.to_dict(),
            compute_reference_metrics=compute_reference_metrics,
            reference_sample_spacing_m=REFERENCE_SAMPLE_SPACING_M,
            reference_safety_margin_m=REFERENCE_SAFETY_MARGIN_M,
            goal_tolerance_m=GOAL_TOLERANCE_M,
            smoothness_sample_spacing_m=REFERENCE_SAMPLE_SPACING_M,
            extra={
                "random_seed": int(seed),
                "splatnav_timings_sec": timings,
                "num_polytopes": None if output is None else int(output.get("num_polytopes", 0)),
                "astar_path": None if output is None else output.get("path"),
                "bounds": {"lower": self.lower.tolist(), "upper": self.upper.tolist()},
                "bounds_policy": PLANNING_BOUNDS_POLICY,
                "resolution": list(self.resolution),
                "cell_size_m": self.cell_size.tolist(),
                "global_planner": {
                    "algorithm": "dijkstra3d.binary_dijkstra",
                    "connectivity": int(self.config.global_connectivity),
                },
                "safety_parameter_contract": _safety_parameter_contract(
                    robot_radius_m=float(self.config.robot_radius),
                    vmax=float(self.config.vmax),
                    amax=float(self.config.amax),
                ),
                "endpoint_policy": ENDPOINT_POLICY,
                "endpoint_constraint_tolerance_m": ENDPOINT_CONSTRAINT_TOLERANCE_M,
                "endpoint_diagnostics": endpoint_diagnostics,
            },
        )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _load_records(path: Path) -> list[PathEvaluation]:
    records: list[PathEvaluation] = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                payload.setdefault("global_planner_success", None)
                payload.setdefault("local_planner_success", None)
                for policy in ("unknown_as_free", "unknown_as_occupied"):
                    reference_metrics_requested = (
                        payload.get(f"task_success_{policy}") is not None
                        or payload.get(f"strict_success_{policy}") is not None
                    )
                    canonical_success = evaluate_strict_success(
                        native_success=bool(payload["native_success"]),
                        fallback_used=bool(payload.get("fallback_used", False)),
                        start_reached=bool(payload["start_reached"]),
                        goal_reached=bool(payload["goal_reached"]),
                        reference_safe=bool(payload[f"safe_{policy}"]),
                        reference_metrics_computed=reference_metrics_requested,
                        global_planner_success=payload.get("global_planner_success"),
                        local_planner_success=payload.get("local_planner_success"),
                    )
                    canonical_reasons = strict_failure_reasons(
                        native_success=bool(payload["native_success"]),
                        fallback_used=bool(payload.get("fallback_used", False)),
                        start_reached=bool(payload["start_reached"]),
                        goal_reached=bool(payload["goal_reached"]),
                        reference_safe=bool(payload[f"safe_{policy}"]),
                        reference_metrics_computed=reference_metrics_requested,
                        global_planner_success=payload.get("global_planner_success"),
                        local_planner_success=payload.get("local_planner_success"),
                    )
                    payload[f"strict_success_{policy}"] = canonical_success
                    payload[f"task_success_{policy}"] = canonical_success
                    payload[f"strict_failure_reasons_{policy}"] = canonical_reasons
                    payload[f"task_failure_reasons_{policy}"] = list(canonical_reasons)
                    payload[f"failure_reason_{policy}"] = (
                        ""
                        if canonical_success is not False
                        else ", ".join(canonical_reasons) or "task_failed"
                    )
                records.append(PathEvaluation(**payload))
            except (json.JSONDecodeError, TypeError) as error:
                raise ValueError(f"invalid record at {path}:{line_number}: {error}") from error
    return records


def _append_record(path: Path, record: PathEvaluation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record.to_json_dict(), ensure_ascii=False) + "\n")
        file.flush()


def _validate_inputs(spec: SceneSpec, *, require_reference: bool) -> None:
    required_paths = [
        ("checkpoint", spec.checkpoint),
        ("endpoints", spec.endpoints),
    ]
    if require_reference:
        required_paths.append(("reference_map", spec.reference_map))
    missing = [
        f"{name}: {path}"
        for name, path in required_paths
        if path is None or not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("missing scene inputs:\n" + "\n".join(missing))


def _scene_output_dir(spec: SceneSpec, args: argparse.Namespace) -> Path:
    scene_root = args.output_root / spec.dataset / spec.scene
    if getattr(args, "radius_profile_role", "baseline") == "baseline":
        return scene_root
    return (
        scene_root
        / "radius_profiles"
        / radius_profile_directory_name(float(args.robot_radius_m))
    )


def _matching_baseline_summary_path(
    spec: SceneSpec,
    args: argparse.Namespace,
) -> Path:
    if getattr(args, "radius_profile_role", "baseline") == "baseline":
        return spec.endpoints.parent / "summary.json"
    return (
        spec.endpoints.parent
        / "radius_profiles"
        / radius_profile_directory_name(float(args.robot_radius_m))
        / "summary.json"
    )


def _scene_run_signature(
    spec: SceneSpec,
    args: argparse.Namespace,
    *,
    endpoint_manifest_robot_radius_m: float | None,
) -> dict[str, Any]:
    safety_contract = _safety_parameter_contract(
        robot_radius_m=float(args.robot_radius_m),
        vmax=float(args.vmax),
        amax=float(args.amax),
    )
    return {
        "checkpoint": str(spec.checkpoint),
        "reference_map": None if spec.reference_map is None else str(spec.reference_map),
        "reference_kind": spec.reference_kind,
        "endpoints": str(spec.endpoints),
        "num_trials": int(args.num_trials),
        "robot_radius_m": float(args.robot_radius_m),
        "radius_profile_role": getattr(args, "radius_profile_role", "baseline"),
        "all_robot_radius_profiles_m": list(
            getattr(args, "all_robot_radius_profiles_m", [args.robot_radius_m])
        ),
        "endpoint_manifest_robot_radius_m": endpoint_manifest_robot_radius_m,
        "vmax": float(args.vmax),
        "amax": float(args.amax),
        "target_cell_size_m": float(args.target_cell_size_m),
        "maximum_cell_size_m": float(args.maximum_cell_size_m),
        "grid_memory_budget_gib": float(args.grid_memory_budget_gib),
        "grid_free_memory_fraction": float(args.grid_free_memory_fraction),
        "grid_resolution_policy": (
            "adaptive_metric_cell_size_under_cuda_voxel_budget_v1"
        ),
        "global_planner_algorithm": "dijkstra3d.binary_dijkstra",
        "global_connectivity": int(args.global_connectivity),
        "global_max_expansions": None,
        "global_max_planning_time_sec": None,
        "min_opacity": float(args.min_opacity),
        "bounds_quantile": float(args.bounds_quantile),
        "bounds_padding_fraction": float(args.bounds_padding_fraction),
        "spline_degree": int(args.spline_degree),
        "samples_per_section": int(args.samples_per_section),
        "device": str(args.device),
        "base_seed": int(args.base_seed),
        "reference_metrics": not bool(args.timing_only),
        "reference_sample_spacing_m": REFERENCE_SAMPLE_SPACING_M,
        "reference_safety_margin_m": REFERENCE_SAFETY_MARGIN_M,
        "goal_tolerance_m": GOAL_TOLERANCE_M,
        "endpoint_policy": ENDPOINT_POLICY,
        "endpoint_constraint_tolerance_m": ENDPOINT_CONSTRAINT_TOLERANCE_M,
        "planning_bounds_policy": PLANNING_BOUNDS_POLICY,
        "safety_parameter_contract": safety_contract,
        "evaluation_contract": "aligned_radius_profiles_and_strict_success_v3",
    }


def run_scene(spec: SceneSpec, args: argparse.Namespace) -> tuple[Path, list[PathEvaluation]]:
    _validate_inputs(spec, require_reference=not args.timing_only)
    endpoint_manifest, endpoints = _load_endpoint_manifest(
        spec.endpoints, args.num_trials
    )
    manifest_radius_value = endpoint_manifest.get("robot_radius_m")
    manifest_radius = (
        None if manifest_radius_value is None else float(manifest_radius_value)
    )
    scene_output = _scene_output_dir(spec, args)
    (scene_output / "failure.json").unlink(missing_ok=True)
    records_path = scene_output / "records.jsonl"
    config_path = scene_output / "run_config.json"
    signature = _scene_run_signature(
        spec,
        args,
        endpoint_manifest_robot_radius_m=manifest_radius,
    )
    if args.overwrite:
        records_path.unlink(missing_ok=True)
        config_path.unlink(missing_ok=True)
    elif config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("run_signature") != signature:
            raise RuntimeError(
                f"existing run has different inputs/options: {scene_output}; "
                "choose another --output-root or pass --overwrite"
            )

    records = _load_records(records_path)
    completed = {str(record.trial_id) for record in records}
    run_config: dict[str, Any] = {
        "format": "splatnav_endpoint_replay_run_v7",
        "dataset": spec.dataset,
        "scene": spec.scene,
        "method": METHOD_NAME,
        "reference_kind": spec.reference_kind,
        "reference_map": None if spec.reference_map is None else str(spec.reference_map),
        "run_signature": signature,
        "splatnav_root": str(args.splatnav_root),
        "status": "initializing",
        "completed_trials": len(records),
        "parameter_alignment": {
            "shared_endpoint_trials": True,
            "shared_trial_seed_policy": "blake2s(base_seed, trial_id)",
            "robot_radius_profile_m": float(args.robot_radius_m),
            "all_robot_radius_profiles_m": list(
                getattr(args, "all_robot_radius_profiles_m", [args.robot_radius_m])
            ),
            "radius_profile_role": getattr(
                args, "radius_profile_role", "baseline"
            ),
            "endpoint_manifest_robot_radius_m": manifest_radius,
            "reference_sample_spacing_m": REFERENCE_SAMPLE_SPACING_M,
            "smoothness_sample_spacing_m": REFERENCE_SAMPLE_SPACING_M,
            "goal_tolerance_m": GOAL_TOLERANCE_M,
            "global_planner": {
                "algorithm": "dijkstra3d.binary_dijkstra",
                "connectivity": int(args.global_connectivity),
                "max_expansions": None,
                "max_planning_time_sec": None,
                "limitation": (
                    "The original SplatNav Dijkstra API exposes neither an "
                    "expansion cap nor a planning-time limit."
                ),
            },
            "safety": signature["safety_parameter_contract"],
            "method_native_parameters": [
                "vmax",
                "amax",
                "spline_degree",
                "samples_per_section",
                "min_opacity",
                "bounds_quantile",
                "bounds_padding_fraction",
            ],
        },
        "strict_success_definition": (
            "native_success AND global_planner_success AND "
            "local_planner_success AND NOT fallback_used AND start_reached AND "
            "goal_reached AND reference_safe"
        ),
        "task_success_definition": (
            "alias of strict_success_definition; retained for backward "
            "compatibility"
        ),
    }
    _write_json(config_path, run_config)

    reference: VoxelReference | UnscoredReference
    if args.timing_only:
        reference = UnscoredReference(
            label=(
                "reference metrics disabled"
                if spec.reference_map is None
                else str(spec.reference_map)
            )
        )
    else:
        if spec.reference_map is None:
            raise RuntimeError("reference_map is required unless --timing-only is used")
        reference = VoxelReference.from_map(spec.reference_map)
        _validate_endpoints_against_reference(
            endpoint_manifest,
            endpoints,
            reference,
            robot_radius_m=float(args.robot_radius_m),
        )
        if spec.reference_kind == REFERENCE_GROUND_TRUTH:
            metadata_path = spec.reference_map.with_suffix(".json")
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                run_config["ground_truth_reference"] = {
                    "metadata": str(metadata_path),
                    "mesh": metadata.get("mesh"),
                    "reference_semantics": metadata.get("reference_semantics"),
                    "free_space_map": metadata.get("free_space_map"),
                }

    map_from_training_world = (
        reference.map_from_training_world
        if isinstance(reference, VoxelReference)
        else np.eye(4, dtype=np.float32)
    )
    if isinstance(reference, UnscoredReference) and spec.reference_map is not None:
        with np.load(spec.reference_map) as data:
            if "map_from_training_world" in data:
                map_from_training_world = np.asarray(
                    data["map_from_training_world"], dtype=np.float32
                ).reshape(4, 4)

    try:
        planner = ReusableSplatNavPlanner(
            checkpoint=spec.checkpoint,
            splatnav_root=args.splatnav_root,
            robot_radius=args.robot_radius_m,
            vmax=args.vmax,
            amax=args.amax,
            target_cell_size_m=args.target_cell_size_m,
            maximum_cell_size_m=args.maximum_cell_size_m,
            grid_memory_budget_gib=args.grid_memory_budget_gib,
            grid_free_memory_fraction=args.grid_free_memory_fraction,
            global_connectivity=args.global_connectivity,
            min_opacity=args.min_opacity,
            bounds_quantile=args.bounds_quantile,
            bounds_padding_fraction=args.bounds_padding_fraction,
            spline_degree=args.spline_degree,
            samples_per_section=args.samples_per_section,
            device=args.device,
            map_from_training_world=map_from_training_world,
        )
    except ResolutionInsufficientError as error:
        run_config["status"] = "resolution_insufficient"
        run_config["scene_setup"] = {
            "bounds": error.diagnostics.get("bounds"),
            "bounds_policy": PLANNING_BOUNDS_POLICY,
            "grid_resolution_policy": error.diagnostics["policy"],
            "grid_selection_diagnostics": error.diagnostics,
            "target_cell_size": error.diagnostics["target_cell_size"],
            "effective_cell_size": error.diagnostics["effective_cell_size"],
            "resolution_xyz": error.diagnostics["resolution_xyz"],
            "voxel_count": error.diagnostics["voxel_count"],
            "voxel_budget": error.diagnostics["voxel_budget"],
            "memory_limited": error.diagnostics["memory_limited"],
            "effective_cell_size_over_robot_radius": error.diagnostics[
                "effective_cell_size_over_robot_radius"
            ],
            "resolution_insufficient": True,
            "peak_vram_mb": error.diagnostics.get("peak_vram_mb"),
            "peak_vram_scope": error.diagnostics.get("peak_vram_scope"),
        }
        _write_json(config_path, run_config)
        raise
    run_config["scene_setup"] = {
        "checkpoint_load_time_sec": planner.checkpoint_load_time_sec,
        "checkpoint_load_resource_usage": planner.checkpoint_load_resource_usage,
        "planner_build_time_sec": planner.planner_build_time_sec,
        "total_time_sec": planner.checkpoint_load_time_sec + planner.planner_build_time_sec,
        "build_resource_usage": planner.build_resource_usage,
        "num_gaussians": int(planner.gaussians.means.shape[0]),
        "representation_tensor_bytes": planner.representation_tensor_bytes,
        "bounds": {"lower": planner.lower.tolist(), "upper": planner.upper.tolist()},
        "bounds_policy": PLANNING_BOUNDS_POLICY,
        "resolution": list(planner.resolution),
        "cell_size_m": planner.cell_size.tolist(),
        "grid_selection_diagnostics": planner.grid_selection_diagnostics,
        "grid_allocation_diagnostics": planner.grid_allocation_diagnostics,
        "target_cell_size": planner.grid_selection_diagnostics[
            "target_cell_size"
        ],
        "effective_cell_size": planner.grid_selection_diagnostics[
            "effective_cell_size"
        ],
        "resolution_xyz": planner.grid_selection_diagnostics["resolution_xyz"],
        "voxel_count": planner.grid_selection_diagnostics["voxel_count"],
        "voxel_budget": planner.grid_selection_diagnostics["voxel_budget"],
        "memory_limited": planner.grid_selection_diagnostics["memory_limited"],
        "effective_cell_size_over_robot_radius": (
            planner.grid_selection_diagnostics[
                "effective_cell_size_over_robot_radius"
            ]
        ),
        "resolution_insufficient": planner.grid_selection_diagnostics[
            "resolution_insufficient"
        ],
        "maximum_acceptable_cell_size": planner.grid_selection_diagnostics[
            "maximum_acceptable_cell_size"
        ],
        "grid_resolution_policy": signature["grid_resolution_policy"],
        "global_planner_algorithm": signature["global_planner_algorithm"],
        "global_connectivity": signature["global_connectivity"],
        "open3d_available": planner.open3d_available,
        "shared_once_per_scene": True,
        "endpoint_policy": ENDPOINT_POLICY,
        "safety_parameter_contract": signature["safety_parameter_contract"],
    }
    setup_resource_usages = {
        "checkpoint_load": planner.checkpoint_load_resource_usage,
        "planner_build": planner.build_resource_usage,
    }
    setup_resource_summary = summarize_resource_usages(setup_resource_usages)
    run_config["scene_setup"]["resource_usage_summary"] = setup_resource_summary
    run_config["scene_setup"]["resource_usage_by_setup_stage"] = (
        setup_resource_usages
    )
    run_config["scene_setup"]["peak_vram_mb"] = _peak_vram_mb(
        setup_resource_summary
    )
    run_config["scene_setup"]["peak_vram_scope"] = "scene_setup"
    run_config["scene_setup"]["peak_vram_measurement"] = (
        "PyTorch max_memory_reserved; excludes non-PyTorch CUDA allocations"
    )
    run_config["status"] = "running"
    _write_json(config_path, run_config)

    for index, pair in enumerate(endpoints):
        trial_id = pair["trial_id"]
        if str(trial_id) in completed:
            continue
        print(f"[{spec.dataset}/{spec.scene}] trial {index + 1}/{len(endpoints)} id={trial_id}")
        record = planner.query(
            trial_id=trial_id,
            start=pair["start_world"],
            goal=pair["goal_world"],
            seed=_trial_seed(args.base_seed, trial_id),
            reference=reference,
            compute_reference_metrics=not args.timing_only,
        )
        _append_record(records_path, record)
        records.append(record)
        completed.add(str(trial_id))
        run_config["completed_trials"] = len(records)
        _write_json(config_path, run_config)

    method_summary = aggregate_evaluations(records)
    setup_total = float(run_config["scene_setup"]["total_time_sec"])
    amortized = None if not records else setup_total / len(records)
    method_summary["scene_splatnav_setup_time_sec"] = setup_total
    method_summary["amortized_splatnav_setup_time_sec_per_trial"] = amortized
    query_mean = method_summary.get("mean_computation_time_sec_all_trials")
    method_summary["mean_end_to_end_time_sec_including_amortized_scene_build"] = (
        None if query_mean is None or amortized is None else float(query_mean) + amortized
    )
    method_summary["method_representation_array_bytes"] = int(
        planner.representation_tensor_bytes
    )
    all_resource_usages = dict(setup_resource_usages)
    all_resource_usages.update(
        {
            f"trial_{record.trial_id}": record.resource_usage
            for record in records
        }
    )
    all_resource_summary = summarize_resource_usages(all_resource_usages)
    method_summary["resource_usage_summary_all_stages"] = all_resource_summary
    peak_vram_mb = _peak_vram_mb(all_resource_summary)
    method_summary["peak_vram_mb"] = peak_vram_mb
    run_config["scene_setup"]["peak_vram_mb"] = peak_vram_mb
    run_config["scene_setup"]["peak_vram_scope"] = "all_completed_stages"
    summary = {
        "format": "splatnav_endpoint_replay_summary_v6",
        "reference_kind": spec.reference_kind,
        "reference_map": None if spec.reference_map is None else str(spec.reference_map),
        "endpoint_policy": ENDPOINT_POLICY,
        "robot_radius_m": float(args.robot_radius_m),
        "radius_profile_role": getattr(args, "radius_profile_role", "baseline"),
        "endpoint_manifest_robot_radius_m": manifest_radius,
        "strict_path_quality_population": (
            "strict_success_only_per_unknown_policy; detour and curvature means "
            "additionally require detour_ratio > 1.0001"
        ),
        "safety_parameter_contract": signature["safety_parameter_contract"],
        "scene_setup": run_config["scene_setup"],
        "by_method": {METHOD_NAME: method_summary},
    }
    _write_json(scene_output / "summary.json", summary)

    baseline_summary_path = _matching_baseline_summary_path(spec, args)
    if baseline_summary_path.is_file():
        baseline = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
        comparison_methods = dict(baseline.get("by_method", {}))
        comparison_methods[METHOD_NAME] = method_summary
        _write_json(
            scene_output / "comparison_summary.json",
            {
                "format": "gspar_splatnav_aggregate_comparison_v5",
                "robot_radius_m": float(args.robot_radius_m),
                "radius_profile_role": getattr(
                    args, "radius_profile_role", "baseline"
                ),
                "by_method": comparison_methods,
                "paired_outcomes_by_esdf_method": baseline.get(
                    "paired_outcomes_by_esdf_method", {}
                ),
                "note": (
                    "SplatNav uses the same endpoint manifest, but no SplatNav paired "
                    "ESDF outcomes are added because the baseline records.jsonl is not required."
                ),
                "baseline_summary": str(baseline_summary_path),
                "splatnav_summary": str(scene_output / "summary.json"),
            },
        )

    run_config["status"] = "completed"
    run_config["completed_trials"] = len(records)
    _write_json(config_path, run_config)
    del reference, planner
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass
    return scene_output, records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m GSPAR.evaluation.splatnav",
        description=__doc__,
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="configs/dataset.txt, one voxel command txt, or a voxel_command_txts directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Direct single-scene checkpoint; use instead of --source",
    )
    parser.add_argument(
        "--endpoints",
        type=Path,
        help="Direct single-scene endpoint manifest; use instead of --source",
    )
    parser.add_argument(
        "--reference-map",
        type=Path,
        help="Direct GT or TSDF voxel reference NPZ; optional only with --timing-only",
    )
    parser.add_argument("--dataset-name", default="custom")
    parser.add_argument("--scene-name", default="scene")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Root containing output/, dataset/, and GSPAR/ (default: current directory)",
    )
    parser.add_argument(
        "--splatnav-root",
        type=Path,
        default=None,
        help="Sibling SplatNav repository (default: PROJECT_ROOT/../splatnav)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Separate result root (default: splatnav_evaluation, or "
            "splatnav_evaluation_gt when every selected scene uses GT)"
        ),
    )
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument(
        "--robot-radius-m",
        type=float,
        default=DEFAULT_BASELINE_ROBOT_RADIUS_M,
        help="Baseline robot radius in metres (default: 0.08)",
    )
    parser.add_argument(
        "--additional-robot-radii-m",
        type=float,
        nargs="*",
        default=list(DEFAULT_ADDITIONAL_ROBOT_RADII_M),
        help=(
            "Additional radius profiles evaluated on the same endpoints "
            "(default: none; pass values to run an explicit sweep)."
        ),
    )
    parser.add_argument("--vmax", type=float, default=0.1)
    parser.add_argument("--amax", type=float, default=0.1)
    parser.add_argument(
        "--target-cell-size-m",
        type=float,
        default=DEFAULT_PLANNING_GRID_CELL_SIZE_M,
        help=(
            "Target SplatNav grid cell size in scene coordinate units "
            "(default: 0.05); enlarged automatically only when required by "
            "the CUDA voxel budget"
        ),
    )
    parser.add_argument(
        "--maximum-cell-size-m",
        type=float,
        default=DEFAULT_MAX_PLANNING_GRID_CELL_SIZE_M,
        help=(
            "Maximum accepted adaptive cell size (default: 0.10); a scene that "
            "requires a coarser grid is marked resolution_insufficient"
        ),
    )
    parser.add_argument(
        "--grid-memory-budget-gib",
        type=float,
        default=DEFAULT_GRID_MEMORY_BUDGET_GIB,
        help=(
            "Maximum estimated CUDA memory assigned to the dense grid "
            "(default: 8 GiB; calibrate by measurement for the target GPU)"
        ),
    )
    parser.add_argument(
        "--grid-free-memory-fraction",
        type=float,
        default=DEFAULT_GRID_FREE_MEMORY_FRACTION,
        help=(
            "Maximum fraction of currently free CUDA memory assigned to the "
            "dense grid (default: 0.5)"
        ),
    )
    parser.add_argument(
        "--global-connectivity",
        type=int,
        choices=(6, 18, 26),
        default=26,
        help="SplatNav Dijkstra grid connectivity (default: 26, shared protocol)",
    )
    parser.add_argument("--min-opacity", type=float, default=0.0)
    parser.add_argument("--bounds-quantile", type=float, default=0.005)
    parser.add_argument("--bounds-padding-fraction", type=float, default=0.05)
    parser.add_argument("--spline-degree", type=int, default=6)
    parser.add_argument("--samples-per-section", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if (
            not np.isfinite(args.target_cell_size_m)
            or args.target_cell_size_m <= 0.0
        ):
            raise ValueError("--target-cell-size-m must be finite and positive")
        if (
            not np.isfinite(args.maximum_cell_size_m)
            or args.maximum_cell_size_m <= 0.0
        ):
            raise ValueError("--maximum-cell-size-m must be finite and positive")
        if (
            args.maximum_cell_size_m < args.target_cell_size_m
        ):
            raise ValueError(
                "--maximum-cell-size-m must be at least --target-cell-size-m"
            )
        if (
            not np.isfinite(args.grid_memory_budget_gib)
            or args.grid_memory_budget_gib <= 0.0
        ):
            raise ValueError("--grid-memory-budget-gib must be finite and positive")
        if (
            not np.isfinite(args.grid_free_memory_fraction)
            or not 0.0 < args.grid_free_memory_fraction <= 1.0
        ):
            raise ValueError("--grid-free-memory-fraction must be in (0, 1]")
    except ValueError as error:
        parser.error(str(error))
    output_root_was_default = args.output_root is None
    args.project_root = args.project_root.expanduser().resolve()
    args.splatnav_root = (
        (args.project_root.parent / "splatnav")
        if args.splatnav_root is None
        else args.splatnav_root
    ).expanduser().resolve()
    args.output_root = (
        (args.project_root / "GSPAR/outputs/splatnav_evaluation")
        if args.output_root is None
        else args.output_root
    ).expanduser().resolve()

    direct_values = (args.checkpoint, args.endpoints, args.reference_map)
    if args.source is not None and any(value is not None for value in direct_values):
        parser.error("--source cannot be combined with direct scene input options")
    if args.source is None:
        if args.checkpoint is None or args.endpoints is None:
            parser.error("provide --source, or both --checkpoint and --endpoints")
        if args.reference_map is None and not args.timing_only:
            parser.error("--reference-map is required unless --timing-only is used")
        specs = [
            SceneSpec(
                dataset=str(args.dataset_name),
                scene=str(args.scene_name),
                checkpoint=_resolve_from_root(args.checkpoint, args.project_root),
                reference_map=(
                    None
                    if args.reference_map is None
                    else _resolve_from_root(args.reference_map, args.project_root)
                ),
                endpoints=_resolve_from_root(args.endpoints, args.project_root),
                reference_kind=_reference_kind(
                    None
                    if args.reference_map is None
                    else _resolve_from_root(args.reference_map, args.project_root)
                ),
            )
        ]
    else:
        specs = discover_scenes(args.source, args.project_root)
    if args.datasets:
        selected = set(args.datasets)
        specs = [spec for spec in specs if spec.dataset in selected]
    if args.scenes:
        selected = set(args.scenes)
        specs = [spec for spec in specs if spec.scene in selected]
    if not specs:
        raise RuntimeError("no scenes remain after applying dataset/scene filters")
    if output_root_was_default and all(
        spec.reference_kind == REFERENCE_GROUND_TRUTH for spec in specs
    ):
        args.output_root = (
            args.project_root / "GSPAR/outputs/splatnav_evaluation_gt"
        ).resolve()

    try:
        # Dataset-source runs use the common 0.25 m robot expressed in each
        # dataset's coordinate scale. Direct single-scene calls retain the
        # explicit --robot-radius-m value (whose default is 0.08 m).
        radii_by_scene = {
            (spec.dataset, spec.scene): normalize_robot_radius_profiles(
                (
                    default_robot_radius_for_dataset(spec.dataset)
                    if args.source is not None
                    else args.robot_radius_m
                ),
                args.additional_robot_radii_m,
            )
            for spec in specs
        }
        endpoint_radius_by_scene = {
            key: validate_endpoint_manifest_radius(spec.endpoints, max(radii_by_scene[key]))
            for spec in specs
            for key in [(spec.dataset, spec.scene)]
        }
        for radii in radii_by_scene.values():
            for radius in radii:
                _safety_parameter_contract(
                    robot_radius_m=radius,
                    vmax=args.vmax,
                    amax=args.amax,
                )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    args.output_root.mkdir(parents=True, exist_ok=True)
    profile_groups: dict[float, dict[str, Any]] = {}
    baseline_records: list[PathEvaluation] = []
    baseline_scene_results: list[dict[str, Any]] = []
    scene_sweep_profiles: dict[tuple[str, str], list[dict[str, Any]]] = {
        (spec.dataset, spec.scene): [] for spec in specs
    }
    failed = False
    stop_requested = False
    for scene_index, spec in enumerate(specs):
        scene_key = (spec.dataset, spec.scene)
        scene_radii = radii_by_scene[scene_key]
        print(
            f"=== scene {scene_index + 1}/{len(specs)}: "
            f"{spec.dataset}/{spec.scene}; radii={scene_radii} m ==="
        )
        for profile_index, radius in enumerate(scene_radii):
            profile_values = vars(args).copy()
            profile_values.update(
                {
                    "robot_radius_m": float(radius),
                    "additional_robot_radii_m": [],
                    "radius_profile_role": (
                        "baseline" if profile_index == 0 else "additional"
                    ),
                    "all_robot_radius_profiles_m": list(scene_radii),
                }
            )
            profile_args = argparse.Namespace(**profile_values)
            group = profile_groups.setdefault(
                float(radius),
                {
                    "robot_radius_m": float(radius),
                    "roles": set(),
                    "records": [],
                    "scenes": [],
                },
            )
            group["roles"].add(profile_args.radius_profile_role)
            try:
                output, records = run_scene(spec, profile_args)
                group["records"].extend(records)
                summary_path = output / "summary.json"
                summary_payload = json.loads(
                    summary_path.read_text(encoding="utf-8")
                )
                scene_result = {
                    "dataset": spec.dataset,
                    "scene": spec.scene,
                    "reference_kind": spec.reference_kind,
                    "status": "completed",
                    "num_trials": len(records),
                    "robot_radius_m": float(radius),
                    "output_dir": str(output),
                    "summary": str(summary_path),
                }
                group["scenes"].append(scene_result)
                scene_sweep_profiles[scene_key].append(
                    {
                        "robot_radius_m": float(radius),
                        "role": profile_args.radius_profile_role,
                        **scene_result,
                        "by_method": summary_payload.get("by_method", {}),
                    }
                )
                if profile_index == 0:
                    baseline_records.extend(records)
                    baseline_scene_results.append(scene_result)
            except Exception as error:
                failed = True
                resolution_insufficient = isinstance(
                    error, ResolutionInsufficientError
                )
                failure = {
                    "dataset": spec.dataset,
                    "scene": spec.scene,
                    "reference_kind": spec.reference_kind,
                    "status": (
                        "resolution_insufficient"
                        if resolution_insufficient
                        else "failed"
                    ),
                    "robot_radius_m": float(radius),
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "resolution_insufficient": resolution_insufficient,
                }
                if resolution_insufficient:
                    failure["grid_selection_diagnostics"] = error.diagnostics
                    failure["peak_vram_mb"] = error.diagnostics.get(
                        "peak_vram_mb"
                    )
                group["scenes"].append(failure)
                scene_sweep_profiles[scene_key].append(
                    {
                        "robot_radius_m": float(radius),
                        "role": profile_args.radius_profile_role,
                        **failure,
                    }
                )
                if profile_index == 0:
                    baseline_scene_results.append(failure)
                failure_path = _scene_output_dir(spec, profile_args) / "failure.json"
                _write_json(failure_path, failure)
                print(
                    f"FAILED {spec.dataset}/{spec.scene} at radius {radius:g} m: "
                    f"{failure['error']}",
                    file=sys.stderr,
                )
                gc.collect()
                try:
                    import torch

                    torch.cuda.empty_cache()
                except (ImportError, RuntimeError):
                    pass
                if args.fail_fast:
                    stop_requested = True
                    break
        if stop_requested:
            break

    profile_batches: list[dict[str, Any]] = []
    for radius, group in sorted(profile_groups.items()):
        records = group.pop("records")
        roles = group.pop("roles")
        scene_results = group["scenes"]
        profile_batch: dict[str, Any] = {
            "robot_radius_m": radius,
            "role": next(iter(roles)) if len(roles) == 1 else "mixed",
            "requested_scene_count": len(scene_results),
            "completed_scene_count": sum(
                item["status"] == "completed" for item in scene_results
            ),
            "failed_scene_count": sum(
                item["status"] != "completed" for item in scene_results
            ),
            "resolution_insufficient_scene_count": sum(
                item["status"] == "resolution_insufficient"
                for item in scene_results
            ),
            "scenes": scene_results,
        }
        if records:
            profile_batch["overall_by_method"] = {
                METHOD_NAME: aggregate_evaluations(records)
            }
        profile_batches.append(profile_batch)

    for spec in specs:
        scene_root = args.output_root / spec.dataset / spec.scene
        _write_json(
            scene_root / "radius_sweep_summary.json",
            {
                "format": "splatnav_robot_radius_sweep_v1",
                "dataset": spec.dataset,
                "scene": spec.scene,
                "endpoint_manifest": str(spec.endpoints),
                "endpoint_manifest_robot_radius_m": endpoint_radius_by_scene[
                    (spec.dataset, spec.scene)
                ],
                "shared_endpoint_trials": True,
                "profiles": scene_sweep_profiles[(spec.dataset, spec.scene)],
            },
        )

    unique_radii = sorted(profile_groups)
    baseline_batch: dict[str, Any] = {
        "scenes": baseline_scene_results,
        "completed_scene_count": sum(
            item["status"] == "completed" for item in baseline_scene_results
        ),
        "failed_scene_count": sum(
            item["status"] != "completed" for item in baseline_scene_results
        ),
        "resolution_insufficient_scene_count": sum(
            item["status"] == "resolution_insufficient"
            for item in baseline_scene_results
        ),
    }
    if baseline_records:
        baseline_batch["overall_by_method"] = {
            METHOD_NAME: aggregate_evaluations(baseline_records)
        }
    batch_payload: dict[str, Any] = {
        "format": "splatnav_endpoint_replay_batch_v6",
        "source": (
            None if args.source is None else str(args.source.expanduser().resolve())
        ),
        "project_root": str(args.project_root),
        "splatnav_root": str(args.splatnav_root),
        "output_root": str(args.output_root),
        "endpoint_policy": ENDPOINT_POLICY,
        "robot_radius_profiles_m": unique_radii,
        "robot_radius_profiles_by_scene_m": {
            f"{dataset}/{scene}": radii
            for (dataset, scene), radii in sorted(radii_by_scene.items())
        },
        "shared_endpoint_trials_across_radius_profiles": True,
        "target_cell_size_m": args.target_cell_size_m,
        "maximum_cell_size_m": args.maximum_cell_size_m,
        "grid_memory_budget_gib": args.grid_memory_budget_gib,
        "grid_free_memory_fraction": args.grid_free_memory_fraction,
        "requested_scene_count": len(specs),
        "completed_scene_count": baseline_batch.get("completed_scene_count", 0),
        "failed_scene_count": baseline_batch.get("failed_scene_count", 0),
        "resolution_insufficient_scene_count": baseline_batch.get(
            "resolution_insufficient_scene_count", 0
        ),
        "scenes": baseline_batch.get("scenes", []),
        "radius_profiles": profile_batches,
    }
    if "overall_by_method" in baseline_batch:
        batch_payload["overall_by_method"] = baseline_batch["overall_by_method"]
    _write_json(args.output_root / "batch_summary.json", batch_payload)
    _write_json(
        args.output_root / "radius_sweep_summary.json",
        {
            "format": "splatnav_robot_radius_sweep_batch_v1",
            "robot_radius_profiles_m": unique_radii,
            "robot_radius_profiles_by_scene_m": {
                f"{dataset}/{scene}": radii
                for (dataset, scene), radii in sorted(radii_by_scene.items())
            },
            "shared_endpoint_trials": True,
            "profiles": profile_batches,
        },
    )
    print(f"batch summary: {args.output_root / 'batch_summary.json'}")
    print(f"radius sweep: {args.output_root / 'radius_sweep_summary.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
