"""Reference-based safety, success, path-quality, and timing metrics."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.spatial import cKDTree

from GSPAR.mapping.map_frame_transform import validate_rigid_transform


REFERENCE_NEIGHBOR_COUNT = 32
REFERENCE_QUERY_BATCH_SIZE = 50_000
NONSTRAIGHT_DETOUR_EXCESS = 1.0e-4
# Treat a residual this small as numerical drift above an exact sample-spacing
# multiple, rather than as a meaningful final path segment.
RESAMPLE_SHORT_TAIL_FRACTION = 1.0e-4


def path_length(points: np.ndarray) -> float:
    path = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if path.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def max_straight_line_deviation(points: np.ndarray) -> float | None:
    """Return the largest vertex distance from the path endpoint segment."""

    path = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if path.shape[0] == 0:
        return None
    if path.shape[0] == 1:
        return 0.0
    start = path[0]
    segment = path[-1] - start
    squared_length = float(np.dot(segment, segment))
    if squared_length <= 1.0e-24:
        return float(np.linalg.norm(path - start[None, :], axis=1).max())
    projection = np.clip(
        ((path - start[None, :]) @ segment) / squared_length,
        0.0,
        1.0,
    )
    closest = start[None, :] + projection[:, None] * segment[None, :]
    return float(np.linalg.norm(path - closest, axis=1).max())


def _finite_path(path_world: np.ndarray | None) -> np.ndarray:
    if path_world is None:
        return np.zeros((0, 3), dtype=np.float64)
    path = np.asarray(path_world, dtype=np.float64).reshape(-1, 3)
    return path if np.isfinite(path).all() else np.zeros((0, 3), dtype=np.float64)


def _deduplicate_consecutive(points: np.ndarray) -> np.ndarray:
    path = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if path.shape[0] <= 1:
        return path
    keep = np.concatenate(
        [np.asarray([True]), np.linalg.norm(np.diff(path, axis=0), axis=1) > 1.0e-9]
    )
    return path[keep]


def resample_polyline(points: np.ndarray, spacing_m: float) -> np.ndarray:
    """Resample a polyline in float64 at approximately uniform arc-length spacing."""

    path = _deduplicate_consecutive(points)
    if path.shape[0] <= 1:
        return path
    segment_length = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate([np.asarray([0.0]), np.cumsum(segment_length)])
    total = float(cumulative[-1])
    if total <= 1.0e-12:
        return path[:1]

    spacing = max(float(spacing_m), 1.0e-6)
    targets = np.arange(0.0, total, spacing, dtype=np.float64)
    tail_length = total - float(targets[-1])
    short_tail_tolerance = max(1.0e-9, spacing * RESAMPLE_SHORT_TAIL_FRACTION)
    if targets.shape[0] > 1 and tail_length <= short_tail_tolerance:
        # Replacing the preceding regular sample preserves the endpoint without
        # creating a near-zero final segment that makes curvature ill-conditioned.
        targets[-1] = total
    else:
        targets = np.concatenate([targets, np.asarray([total])])
    segment = np.searchsorted(cumulative, targets, side="right") - 1
    segment = np.clip(segment, 0, segment_length.shape[0] - 1)
    weight = (targets - cumulative[segment]) / np.maximum(segment_length[segment], 1.0e-12)
    samples = (1.0 - weight[:, None]) * path[segment] + weight[:, None] * path[segment + 1]
    samples[0] = path[0]
    samples[-1] = path[-1]
    return samples


@dataclass
class VoxelReference:
    """Fixed observed TSDF reference shared by every planner in one run."""

    origin: np.ndarray
    voxel_size: float
    voxel_dims: np.ndarray
    occupied_idx: np.ndarray
    free_idx: np.ndarray
    label: str = "observed TSDF reference"
    path: Path | None = None
    map_from_training_world: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float32)
    )

    def __post_init__(self) -> None:
        self.origin = np.asarray(self.origin, dtype=np.float64).reshape(3)
        self.voxel_size = float(self.voxel_size)
        self.voxel_dims = np.asarray(self.voxel_dims, dtype=np.int32).reshape(3)
        self.occupied_idx = np.asarray(self.occupied_idx, dtype=np.int32).reshape(-1, 3)
        self.free_idx = np.asarray(self.free_idx, dtype=np.int32).reshape(-1, 3)
        self.map_from_training_world = validate_rigid_transform(
            self.map_from_training_world
        )
        if not np.isfinite(self.origin).all() or self.voxel_size <= 0.0:
            raise ValueError("reference grid origin and voxel_size must be finite and valid")
        if np.any(self.voxel_dims <= 0):
            raise ValueError("reference voxel_dims must be positive")
        valid = np.logical_and(
            self.occupied_idx >= 0,
            self.occupied_idx < self.voxel_dims[None, :],
        ).all(axis=1)
        self.occupied_idx = np.ascontiguousarray(self.occupied_idx[valid])
        valid_free = np.logical_and(
            self.free_idx >= 0,
            self.free_idx < self.voxel_dims[None, :],
        ).all(axis=1)
        free_idx = np.ascontiguousarray(self.free_idx[valid_free])
        if free_idx.shape[0] == 0:
            raise ValueError("observed TSDF reference has no explicit free_voxel_idx")
        dims64 = self.voxel_dims.astype(np.int64)
        free_linear = (
            (free_idx[:, 0].astype(np.int64) * dims64[1] + free_idx[:, 1])
            * dims64[2]
            + free_idx[:, 2]
        )
        self._free_linear = np.unique(free_linear)
        self.free_voxel_count = int(self._free_linear.shape[0])
        # Membership queries only need sorted linear ids; releasing the Nx3 copy
        # matters for multi-million-voxel TSDF maps.
        self.free_idx = np.zeros((0, 3), dtype=np.int32)
        self._occupied_centers = (
            self.origin[None, :]
            + (self.occupied_idx.astype(np.float64) + 0.5) * self.voxel_size
        )
        self._occupied_tree = (
            cKDTree(self._occupied_centers) if self._occupied_centers.shape[0] else None
        )

    @classmethod
    def from_map(
        cls,
        source: str | Path | Mapping[str, Any],
        *,
        label: str | None = None,
    ) -> "VoxelReference":
        if isinstance(source, Mapping):
            payload = source
            path = None
        else:
            path = Path(source).expanduser().resolve()
            if not path.is_file():
                if path.suffix == "":
                    path = path.with_suffix(".npz")
                if not path.is_file():
                    raise FileNotFoundError(f"reference voxel map not found: {path}")
        required = (
            "origin",
            "voxel_size",
            "voxel_dims",
            "occupied_voxel_idx",
            "free_voxel_idx",
        )
        if not isinstance(source, Mapping):
            with np.load(path) as data:
                payload = {
                    key: np.asarray(data[key])
                    for key in required
                    if key in data
                }
                if "map_from_training_world" in data:
                    payload["map_from_training_world"] = np.asarray(
                        data["map_from_training_world"]
                    )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"reference voxel map missing keys: {missing}")
        resolved_label = label or (str(path) if path is not None else "in-memory observed TSDF")
        return cls(
            origin=np.asarray(payload["origin"], dtype=np.float64),
            voxel_size=float(np.asarray(payload["voxel_size"]).reshape(-1)[0]),
            voxel_dims=np.asarray(payload["voxel_dims"], dtype=np.int32),
            occupied_idx=np.asarray(payload["occupied_voxel_idx"], dtype=np.int32),
            free_idx=np.asarray(payload["free_voxel_idx"], dtype=np.int32),
            label=resolved_label,
            path=path,
            map_from_training_world=np.asarray(
                payload.get("map_from_training_world", np.eye(4)),
                dtype=np.float32,
            ),
        )

    @property
    def world_max(self) -> np.ndarray:
        return self.origin + self.voxel_dims.astype(np.float64) * self.voxel_size

    def in_bounds(self, points_world: np.ndarray) -> np.ndarray:
        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        return np.logical_and(
            points >= self.origin[None, :],
            points < self.world_max[None, :],
        ).all(axis=1)

    def observed_free_index_mask(self, voxel_indices: np.ndarray) -> np.ndarray:
        """Return explicit-free membership for voxel indices."""

        indices = np.asarray(voxel_indices, dtype=np.int64).reshape(-1, 3)
        valid = np.logical_and(
            indices >= 0,
            indices < self.voxel_dims.astype(np.int64)[None, :],
        ).all(axis=1)
        result = np.zeros((indices.shape[0],), dtype=bool)
        if not valid.any():
            return result
        rows = np.flatnonzero(valid)
        valid_idx = indices[rows]
        dims64 = self.voxel_dims.astype(np.int64)
        linear = (
            (valid_idx[:, 0] * dims64[1] + valid_idx[:, 1]) * dims64[2]
            + valid_idx[:, 2]
        )
        positions = np.searchsorted(self._free_linear, linear)
        found = positions < self._free_linear.shape[0]
        if found.any():
            found_rows = np.flatnonzero(found)
            found[found_rows] = self._free_linear[positions[found_rows]] == linear[found_rows]
        result[rows] = found
        return result

    def observed_free_mask(self, points_world: np.ndarray) -> np.ndarray:
        """Return true only when a point lies in an explicitly observed-free voxel."""

        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        indices = np.floor((points - self.origin[None, :]) / self.voxel_size).astype(
            np.int64
        )
        return self.observed_free_index_mask(indices)

    def observed_free_footprint_mask(
        self,
        points_world: np.ndarray,
        radius_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Check whether each spherical robot footprint is fully observed free."""

        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        radius = max(0.0, float(radius_m))
        if radius <= 1.0e-12:
            in_bounds = self.in_bounds(points)
            return self.observed_free_mask(points), in_bounds

        max_offset = int(np.ceil(radius / self.voxel_size)) + 1
        axis = np.arange(-max_offset, max_offset + 1, dtype=np.int64)
        offsets = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
        observed_free = np.ones((points.shape[0],), dtype=bool)
        footprint_in_bounds = np.ones((points.shape[0],), dtype=bool)
        radius_squared = radius * radius + 1.0e-12
        batch_size = 2_000
        dims64 = self.voxel_dims.astype(np.int64)
        for begin in range(0, points.shape[0], batch_size):
            batch_points = points[begin : begin + batch_size]
            base = np.floor(
                (batch_points - self.origin[None, :]) / self.voxel_size
            ).astype(np.int64)
            candidate_idx = base[:, None, :] + offsets[None, :, :]
            voxel_min = self.origin[None, None, :] + candidate_idx * self.voxel_size
            delta = np.maximum(
                np.maximum(
                    voxel_min - batch_points[:, None, :],
                    batch_points[:, None, :] - (voxel_min + self.voxel_size),
                ),
                0.0,
            )
            intersects = np.einsum("ijk,ijk->ij", delta, delta) <= radius_squared
            local_rows, offset_rows = np.nonzero(intersects)
            intersected_idx = candidate_idx[local_rows, offset_rows]
            valid = np.logical_and(
                intersected_idx >= 0,
                intersected_idx < dims64[None, :],
            ).all(axis=1)
            if np.any(~valid):
                footprint_in_bounds[begin + np.unique(local_rows[~valid])] = False
            explicit_free = self.observed_free_index_mask(intersected_idx)
            if np.any(~explicit_free):
                observed_free[begin + np.unique(local_rows[~explicit_free])] = False
        return observed_free, footprint_in_bounds

    def obstacle_distances(self, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return distance to occupied voxel cubes and an in-reference mask."""

        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        in_bounds = self.in_bounds(points)
        distances = np.zeros((points.shape[0],), dtype=np.float64)
        if not in_bounds.any():
            return distances, in_bounds
        if self._occupied_tree is None:
            distances[in_bounds] = np.inf
            return distances, in_bounds

        rows = np.flatnonzero(in_bounds)
        neighbors = min(REFERENCE_NEIGHBOR_COUNT, self._occupied_centers.shape[0])
        half = 0.5 * self.voxel_size
        for begin in range(0, rows.shape[0], REFERENCE_QUERY_BATCH_SIZE):
            batch_rows = rows[begin : begin + REFERENCE_QUERY_BATCH_SIZE]
            query_points = points[batch_rows]
            _, indices = self._occupied_tree.query(query_points, k=neighbors, workers=-1)
            if neighbors == 1:
                indices = np.asarray(indices, dtype=np.int64).reshape(-1, 1)
            centers = self._occupied_centers[np.asarray(indices, dtype=np.int64)]
            delta = np.maximum(np.abs(query_points[:, None, :] - centers) - half, 0.0)
            distances[batch_rows] = np.linalg.norm(delta, axis=2).min(axis=1)
        return distances, in_bounds


@dataclass(frozen=True)
class ReferencePathMetrics:
    safe_unknown_as_free: bool
    safe_unknown_as_occupied: bool
    minimum_obstacle_distance_m: float
    minimum_clearance_m: float
    sample_count: int
    unsafe_sample_count_unknown_as_free: int
    unsafe_sample_fraction_unknown_as_free: float
    unsafe_sample_count_unknown_as_occupied: int
    unsafe_sample_fraction_unknown_as_occupied: float
    occupied_collision_sample_count: int
    occupied_collision_sample_fraction: float
    out_of_bounds_sample_count: int
    unknown_sample_count: int
    unknown_sample_fraction: float


def evaluate_reference_safety(
    path_world: np.ndarray,
    reference: VoxelReference,
    *,
    robot_radius_m: float,
    safety_margin_m: float = 0.0,
    sample_spacing_m: float | None = None,
) -> ReferencePathMetrics:
    """Evaluate a swept path against observed TSDF free/occupied/unknown space."""

    path = _finite_path(path_world)
    if path.shape[0] == 0:
        return ReferencePathMetrics(
            False,
            False,
            0.0,
            -float(robot_radius_m),
            0,
            0,
            0.0,
            0,
            0.0,
            0,
            0.0,
            0,
            0,
            0.0,
        )
    spacing = (
        0.5 * reference.voxel_size
        if sample_spacing_m is None
        else max(float(sample_spacing_m), 1.0e-6)
    )
    uniform = resample_polyline(path, spacing)
    # Include original vertices so sharp corners are checked even if they fall between samples.
    samples = np.concatenate([uniform, path], axis=0)
    distances, _ = reference.obstacle_distances(samples)
    threshold = max(0.0, float(robot_radius_m)) + max(0.0, float(safety_margin_m))
    observed_free, footprint_in_bounds = reference.observed_free_footprint_mask(
        samples,
        threshold,
    )
    out_of_bounds = int(np.count_nonzero(~footprint_in_bounds))
    occupied_collision = distances <= threshold + 1.0e-9
    unknown = np.logical_and(
        footprint_in_bounds,
        ~observed_free & ~occupied_collision,
    )
    unsafe_unknown_as_free = np.logical_or(
        ~footprint_in_bounds,
        occupied_collision,
    )
    unsafe_unknown_as_occupied = np.logical_or(
        unsafe_unknown_as_free,
        ~observed_free,
    )
    unsafe_free_count = int(np.count_nonzero(unsafe_unknown_as_free))
    unsafe_occupied_count = int(np.count_nonzero(unsafe_unknown_as_occupied))
    occupied_collision_count = int(np.count_nonzero(occupied_collision))
    unknown_count = int(np.count_nonzero(unknown))
    minimum_distance = 0.0 if out_of_bounds else float(np.min(distances))
    minimum_clearance = float(minimum_distance - max(0.0, float(robot_radius_m)))
    return ReferencePathMetrics(
        safe_unknown_as_free=unsafe_free_count == 0,
        safe_unknown_as_occupied=unsafe_occupied_count == 0,
        minimum_obstacle_distance_m=minimum_distance,
        minimum_clearance_m=minimum_clearance,
        sample_count=int(samples.shape[0]),
        unsafe_sample_count_unknown_as_free=unsafe_free_count,
        unsafe_sample_fraction_unknown_as_free=float(
            unsafe_free_count / samples.shape[0]
        ),
        unsafe_sample_count_unknown_as_occupied=unsafe_occupied_count,
        unsafe_sample_fraction_unknown_as_occupied=float(
            unsafe_occupied_count / samples.shape[0]
        ),
        occupied_collision_sample_count=occupied_collision_count,
        occupied_collision_sample_fraction=float(
            occupied_collision_count / samples.shape[0]
        ),
        out_of_bounds_sample_count=out_of_bounds,
        unknown_sample_count=unknown_count,
        unknown_sample_fraction=float(unknown_count / samples.shape[0]),
    )


@dataclass(frozen=True)
class SmoothnessMetrics:
    max_curvature_rad_per_m: float
    mean_squared_curvature: float


def evaluate_geometric_smoothness(
    path_world: np.ndarray,
    *,
    sample_spacing_m: float,
) -> SmoothnessMetrics:
    """Measure geometry after common arc-length resampling, independent of planner type."""

    samples = resample_polyline(path_world, sample_spacing_m)
    if samples.shape[0] < 3:
        return SmoothnessMetrics(0.0, 0.0)
    vectors = np.diff(samples.astype(np.float64), axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    valid_segment = lengths > 1.0e-9
    if np.count_nonzero(valid_segment) < 2:
        return SmoothnessMetrics(0.0, 0.0)
    vectors = vectors[valid_segment]
    lengths = lengths[valid_segment]
    directions = vectors / lengths[:, None]
    cosine = np.clip(np.sum(directions[:-1] * directions[1:], axis=1), -1.0, 1.0)
    sine = np.linalg.norm(np.cross(directions[:-1], directions[1:]), axis=1)
    angles = np.arctan2(sine, cosine)
    local_length = 0.5 * (lengths[:-1] + lengths[1:])
    curvature = angles / np.maximum(local_length, 1.0e-9)
    weight_sum = float(local_length.sum())
    mean_squared = (
        0.0
        if weight_sum <= 1.0e-12
        else float(np.sum(curvature * curvature * local_length) / weight_sum)
    )
    return SmoothnessMetrics(
        max_curvature_rad_per_m=float(curvature.max(initial=0.0)),
        mean_squared_curvature=mean_squared,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass
class PathEvaluation:
    method: str
    trial_id: str | int | None
    native_success: bool
    path_available: bool
    start_reached: bool
    goal_reached: bool
    safe_unknown_as_free: bool
    safe_unknown_as_occupied: bool
    task_success_unknown_as_free: bool | None
    task_success_unknown_as_occupied: bool | None
    failure_reason_unknown_as_free: str
    failure_reason_unknown_as_occupied: str
    planner_message: str
    task_failure_reasons_unknown_as_free: list[str]
    task_failure_reasons_unknown_as_occupied: list[str]
    trajectory_type: str
    fallback_used: bool
    reference_map: str
    reference_metrics_computed: bool
    minimum_clearance_m: float | None
    reference_sample_count: int
    unsafe_sample_fraction_unknown_as_free: float | None
    unsafe_sample_fraction_unknown_as_occupied: float | None
    occupied_collision_sample_fraction: float | None
    unknown_sample_fraction: float | None
    path_length_m: float | None
    detour_ratio: float | None
    max_curvature_rad_per_m: float | None
    mean_squared_curvature: float | None
    computation_time_sec: float | None
    evaluation_time_sec: float | None
    global_planning_time_sec: float | None = None
    local_planning_time_sec: float | None = None
    planner_validation_time_sec: float | None = None
    global_planner_success: bool | None = None
    local_planner_success: bool | None = None
    strict_success_unknown_as_free: bool | None = None
    strict_success_unknown_as_occupied: bool | None = None
    strict_failure_reasons_unknown_as_free: list[str] = field(default_factory=list)
    strict_failure_reasons_unknown_as_occupied: list[str] = field(default_factory=list)
    resource_usage: dict[str, Any] = field(default_factory=dict)
    path_point_count: int = 0
    robot_radius_m: float = 0.0
    reference_safety_margin_m: float = 0.0
    start_world: list[float] | None = None
    goal_world: list[float] | None = None
    effective_start_world: list[float] | None = None
    effective_goal_world: list[float] | None = None
    path_world: list[list[float]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    requested_endpoint_distance_m: float | None = None
    actual_endpoint_distance_m: float | None = None
    start_endpoint_error_m: float | None = None
    goal_endpoint_error_m: float | None = None
    task_path_length_m_including_endpoint_gaps: float | None = None
    task_detour_ratio: float | None = None
    max_straight_line_deviation_m: float | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def evaluate_strict_success(
    *,
    native_success: bool,
    fallback_used: bool,
    start_reached: bool,
    goal_reached: bool,
    reference_safe: bool,
    reference_metrics_computed: bool,
    global_planner_success: bool | None = None,
    local_planner_success: bool | None = None,
) -> bool | None:
    """Return the common strict-success result used by every evaluator."""

    if not reference_metrics_computed:
        return None
    stages_succeeded = (
        global_planner_success is not False
        and local_planner_success is not False
    )
    return bool(
        native_success
        and stages_succeeded
        and not fallback_used
        and start_reached
        and goal_reached
        and reference_safe
    )


def strict_failure_reasons(
    *,
    native_success: bool,
    fallback_used: bool,
    start_reached: bool,
    goal_reached: bool,
    reference_safe: bool,
    reference_metrics_computed: bool,
    global_planner_success: bool | None = None,
    local_planner_success: bool | None = None,
) -> list[str]:
    """Explain every failed term in the common strict-success predicate."""

    if not reference_metrics_computed:
        return ["reference_metrics_not_computed"]
    reasons = []
    if not native_success:
        reasons.append("native_planner_failed")
    if global_planner_success is False:
        reasons.append("global_planner_failed")
    if local_planner_success is False:
        reasons.append("local_planner_failed")
    if fallback_used:
        reasons.append("fallback_used")
    if not start_reached:
        reasons.append("start_not_reached")
    if not goal_reached:
        reasons.append("goal_not_reached")
    if not reference_safe:
        reasons.append("reference_unsafe")
    return reasons


def evaluate_path_metrics(
    *,
    method: str,
    trial_id: str | int | None,
    path_world: np.ndarray | None,
    reference: VoxelReference,
    robot_radius: float,
    native_success: bool,
    failure_reason: str = "",
    start_world: np.ndarray | None = None,
    goal_world: np.ndarray | None = None,
    effective_start_world: np.ndarray | None = None,
    effective_goal_world: np.ndarray | None = None,
    trajectory_type: str = "polyline",
    fallback_used: bool = False,
    computation_time_sec: float | None = None,
    global_planning_time_sec: float | None = None,
    local_planning_time_sec: float | None = None,
    planner_validation_time_sec: float | None = None,
    global_planner_success: bool | None = None,
    local_planner_success: bool | None = None,
    resource_usage: dict[str, Any] | None = None,
    compute_reference_metrics: bool = True,
    reference_sample_spacing_m: float | None = None,
    reference_safety_margin_m: float = 0.0,
    goal_tolerance_m: float = 0.10,
    smoothness_sample_spacing_m: float = 0.05,
    extra: dict[str, Any] | None = None,
) -> PathEvaluation:
    path = _finite_path(path_world)
    path_available = bool(path.shape[0] > 0)
    start = None if start_world is None else np.asarray(start_world, dtype=np.float64).reshape(3)
    goal = None if goal_world is None else np.asarray(goal_world, dtype=np.float64).reshape(3)
    tolerance = max(0.0, float(goal_tolerance_m))
    start_reached = bool(path_available and (start is None or np.linalg.norm(path[0] - start) <= tolerance))
    goal_reached = bool(path_available and (goal is None or np.linalg.norm(path[-1] - goal) <= tolerance))

    length = path_length(path) if path_available else None
    requested_endpoint_distance = None
    actual_endpoint_distance = None
    start_endpoint_error = None
    goal_endpoint_error = None
    task_path_length = None
    detour = None
    task_detour = None
    line_deviation = max_straight_line_deviation(path) if path_available else None
    if path_available:
        actual_endpoint_distance = float(np.linalg.norm(path[-1] - path[0]))
        if length is not None and actual_endpoint_distance > 1.0e-12:
            # Polyline length cannot be shorter than the segment joining its own
            # endpoints. Clamp only floating-point noise below one.
            detour = max(1.0, float(length / actual_endpoint_distance))
    if start is not None and goal is not None:
        requested_endpoint_distance = float(np.linalg.norm(goal - start))
        if path_available and length is not None:
            start_endpoint_error = float(np.linalg.norm(path[0] - start))
            goal_endpoint_error = float(np.linalg.norm(path[-1] - goal))
            task_path_length = float(
                start_endpoint_error + length + goal_endpoint_error
            )
            if requested_endpoint_distance > 1.0e-12:
                task_detour = max(
                    1.0,
                    float(task_path_length / requested_endpoint_distance),
                )

    smoothness = None
    if path_available:
        smoothness = evaluate_geometric_smoothness(
            path,
            sample_spacing_m=max(float(smoothness_sample_spacing_m), 1.0e-6),
        )

    reference_result = None
    evaluation_time = None
    if path_available and compute_reference_metrics:
        evaluation_started = time.perf_counter()
        reference_result = evaluate_reference_safety(
            path,
            reference,
            robot_radius_m=float(robot_radius),
            safety_margin_m=float(reference_safety_margin_m),
            sample_spacing_m=reference_sample_spacing_m,
        )
        evaluation_time = float(time.perf_counter() - evaluation_started)
    safe_unknown_as_free = bool(
        reference_result.safe_unknown_as_free
    ) if reference_result is not None else False
    safe_unknown_as_occupied = bool(
        reference_result.safe_unknown_as_occupied
    ) if reference_result is not None else False

    strict_success_unknown_as_free = evaluate_strict_success(
        native_success=bool(native_success),
        fallback_used=bool(fallback_used),
        start_reached=start_reached,
        goal_reached=goal_reached,
        reference_safe=safe_unknown_as_free,
        reference_metrics_computed=bool(compute_reference_metrics),
        global_planner_success=global_planner_success,
        local_planner_success=local_planner_success,
    )
    strict_success_unknown_as_occupied = evaluate_strict_success(
        native_success=bool(native_success),
        fallback_used=bool(fallback_used),
        start_reached=start_reached,
        goal_reached=goal_reached,
        reference_safe=safe_unknown_as_occupied,
        reference_metrics_computed=bool(compute_reference_metrics),
        global_planner_success=global_planner_success,
        local_planner_success=local_planner_success,
    )
    strict_failure_reasons_unknown_as_free = strict_failure_reasons(
        native_success=bool(native_success),
        fallback_used=bool(fallback_used),
        start_reached=start_reached,
        goal_reached=goal_reached,
        reference_safe=safe_unknown_as_free,
        reference_metrics_computed=bool(compute_reference_metrics),
        global_planner_success=global_planner_success,
        local_planner_success=local_planner_success,
    )
    strict_failure_reasons_unknown_as_occupied = strict_failure_reasons(
        native_success=bool(native_success),
        fallback_used=bool(fallback_used),
        start_reached=start_reached,
        goal_reached=goal_reached,
        reference_safe=safe_unknown_as_occupied,
        reference_metrics_computed=bool(compute_reference_metrics),
        global_planner_success=global_planner_success,
        local_planner_success=local_planner_success,
    )
    # Keep the historical Task Success fields for JSON compatibility, but use
    # the strict predicate and its reasons as the single canonical definition.
    task_success_unknown_as_free = strict_success_unknown_as_free
    task_success_unknown_as_occupied = strict_success_unknown_as_occupied
    reasons_unknown_as_free = list(strict_failure_reasons_unknown_as_free)
    reasons_unknown_as_occupied = list(strict_failure_reasons_unknown_as_occupied)
    planner_message = str(failure_reason)
    failure_reason_unknown_as_free = (
        ""
        if task_success_unknown_as_free is not False
        else ", ".join(reasons_unknown_as_free) or "task_failed"
    )
    failure_reason_unknown_as_occupied = (
        ""
        if task_success_unknown_as_occupied is not False
        else ", ".join(reasons_unknown_as_occupied) or "task_failed"
    )

    def vector(value: np.ndarray | None) -> list[float] | None:
        return None if value is None else np.asarray(value, dtype=np.float64).reshape(3).tolist()

    result_extra = dict(extra or {})
    result_extra["reference_metrics_skipped"] = not bool(compute_reference_metrics)
    return PathEvaluation(
        method=str(method),
        trial_id=trial_id,
        native_success=bool(native_success),
        path_available=path_available,
        start_reached=start_reached,
        goal_reached=goal_reached,
        safe_unknown_as_free=safe_unknown_as_free,
        safe_unknown_as_occupied=safe_unknown_as_occupied,
        task_success_unknown_as_free=task_success_unknown_as_free,
        task_success_unknown_as_occupied=task_success_unknown_as_occupied,
        failure_reason_unknown_as_free=failure_reason_unknown_as_free,
        failure_reason_unknown_as_occupied=failure_reason_unknown_as_occupied,
        planner_message=planner_message,
        task_failure_reasons_unknown_as_free=reasons_unknown_as_free,
        task_failure_reasons_unknown_as_occupied=reasons_unknown_as_occupied,
        trajectory_type=str(trajectory_type),
        fallback_used=bool(fallback_used),
        reference_map=str(reference.path or reference.label),
        reference_metrics_computed=bool(compute_reference_metrics and path_available),
        minimum_clearance_m=(None if reference_result is None else reference_result.minimum_clearance_m),
        reference_sample_count=0 if reference_result is None else reference_result.sample_count,
        unsafe_sample_fraction_unknown_as_free=(
            None
            if reference_result is None
            else reference_result.unsafe_sample_fraction_unknown_as_free
        ),
        unsafe_sample_fraction_unknown_as_occupied=(
            None
            if reference_result is None
            else reference_result.unsafe_sample_fraction_unknown_as_occupied
        ),
        occupied_collision_sample_fraction=(
            None
            if reference_result is None
            else reference_result.occupied_collision_sample_fraction
        ),
        unknown_sample_fraction=(
            None if reference_result is None else reference_result.unknown_sample_fraction
        ),
        path_length_m=length,
        detour_ratio=detour,
        requested_endpoint_distance_m=requested_endpoint_distance,
        actual_endpoint_distance_m=actual_endpoint_distance,
        start_endpoint_error_m=start_endpoint_error,
        goal_endpoint_error_m=goal_endpoint_error,
        task_path_length_m_including_endpoint_gaps=task_path_length,
        task_detour_ratio=task_detour,
        max_straight_line_deviation_m=line_deviation,
        max_curvature_rad_per_m=(
            None if smoothness is None else smoothness.max_curvature_rad_per_m
        ),
        mean_squared_curvature=(
            None if smoothness is None else smoothness.mean_squared_curvature
        ),
        strict_success_unknown_as_free=strict_success_unknown_as_free,
        strict_success_unknown_as_occupied=strict_success_unknown_as_occupied,
        strict_failure_reasons_unknown_as_free=(
            strict_failure_reasons_unknown_as_free
        ),
        strict_failure_reasons_unknown_as_occupied=(
            strict_failure_reasons_unknown_as_occupied
        ),
        global_planner_success=(
            None if global_planner_success is None else bool(global_planner_success)
        ),
        local_planner_success=(
            None if local_planner_success is None else bool(local_planner_success)
        ),
        computation_time_sec=(
            None if computation_time_sec is None else float(computation_time_sec)
        ),
        evaluation_time_sec=evaluation_time,
        global_planning_time_sec=(
            None if global_planning_time_sec is None else float(global_planning_time_sec)
        ),
        local_planning_time_sec=(
            None if local_planning_time_sec is None else float(local_planning_time_sec)
        ),
        planner_validation_time_sec=(
            None
            if planner_validation_time_sec is None
            else float(planner_validation_time_sec)
        ),
        resource_usage=dict(resource_usage or {}),
        path_point_count=int(path.shape[0]),
        robot_radius_m=float(robot_radius),
        reference_safety_margin_m=float(reference_safety_margin_m),
        start_world=vector(start_world),
        goal_world=vector(goal_world),
        effective_start_world=vector(effective_start_world),
        effective_goal_world=vector(effective_goal_world),
        path_world=path.astype(float).tolist(),
        extra=result_extra,
    )


def _mean_optional(values: Iterable[float | None]) -> float | None:
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    return None if array.size == 0 else float(array.mean())


def _min_optional(values: Iterable[float | None]) -> float | None:
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    return None if array.size == 0 else float(array.min())


def _max_optional(values: Iterable[float | None]) -> float | None:
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    return None if array.size == 0 else float(array.max())


def _record_strict_success(record: PathEvaluation, policy: str) -> bool | None:
    stored = getattr(record, f"strict_success_{policy}")
    if stored is not None:
        return stored
    task_success = getattr(record, f"task_success_{policy}")
    return evaluate_strict_success(
        native_success=record.native_success,
        fallback_used=record.fallback_used,
        start_reached=record.start_reached,
        goal_reached=record.goal_reached,
        reference_safe=getattr(record, f"safe_{policy}"),
        reference_metrics_computed=task_success is not None,
        global_planner_success=record.global_planner_success,
        local_planner_success=record.local_planner_success,
    )


def _strict_path_metrics(
    records: Iterable[PathEvaluation],
    policy: str,
) -> dict[str, float | None]:
    strict_paths = [
        record for record in records if _record_strict_success(record, policy) is True
    ]
    nonstraight_paths = [
        record
        for record in strict_paths
        if record.detour_ratio is not None
        and record.detour_ratio > 1.0 + NONSTRAIGHT_DETOUR_EXCESS
    ]
    suffix = f"strict_success_{policy}"
    return {
        f"num_nonstraight_{suffix}": len(nonstraight_paths),
        f"mean_minimum_clearance_m_{suffix}": _mean_optional(
            record.minimum_clearance_m for record in strict_paths
        ),
        f"min_minimum_clearance_m_{suffix}": _min_optional(
            record.minimum_clearance_m for record in strict_paths
        ),
        f"mean_unsafe_sample_fraction_unknown_as_free_{suffix}": _mean_optional(
            record.unsafe_sample_fraction_unknown_as_free for record in strict_paths
        ),
        f"mean_unsafe_sample_fraction_unknown_as_occupied_{suffix}": _mean_optional(
            record.unsafe_sample_fraction_unknown_as_occupied for record in strict_paths
        ),
        f"mean_occupied_collision_sample_fraction_{suffix}": _mean_optional(
            record.occupied_collision_sample_fraction for record in strict_paths
        ),
        f"mean_unknown_sample_fraction_{suffix}": _mean_optional(
            record.unknown_sample_fraction for record in strict_paths
        ),
        f"mean_path_length_m_{suffix}": _mean_optional(
            record.path_length_m for record in strict_paths
        ),
        f"mean_detour_ratio_{suffix}": _mean_optional(
            record.detour_ratio for record in nonstraight_paths
        ),
        f"mean_task_detour_ratio_{suffix}": _mean_optional(
            record.task_detour_ratio for record in nonstraight_paths
        ),
        f"mean_start_endpoint_error_m_{suffix}": _mean_optional(
            record.start_endpoint_error_m for record in strict_paths
        ),
        f"mean_goal_endpoint_error_m_{suffix}": _mean_optional(
            record.goal_endpoint_error_m for record in strict_paths
        ),
        f"max_endpoint_error_m_{suffix}": _max_optional(
            max(
                value
                for value in (record.start_endpoint_error_m, record.goal_endpoint_error_m)
                if value is not None
            )
            if record.start_endpoint_error_m is not None
            or record.goal_endpoint_error_m is not None
            else None
            for record in strict_paths
        ),
        f"mean_max_straight_line_deviation_m_{suffix}": _mean_optional(
            record.max_straight_line_deviation_m for record in strict_paths
        ),
        f"mean_max_curvature_rad_per_m_{suffix}": _mean_optional(
            record.max_curvature_rad_per_m for record in nonstraight_paths
        ),
        f"mean_squared_curvature_{suffix}": _mean_optional(
            record.mean_squared_curvature for record in nonstraight_paths
        ),
    }


def _strict_failure_reason_counts(
    records: Iterable[PathEvaluation],
    policy: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        if _record_strict_success(record, policy) is not False:
            continue
        reasons = strict_failure_reasons(
            native_success=record.native_success,
            fallback_used=record.fallback_used,
            start_reached=record.start_reached,
            goal_reached=record.goal_reached,
            reference_safe=getattr(record, f"safe_{policy}"),
            reference_metrics_computed=(
                getattr(record, f"task_success_{policy}") is not None
            ),
            global_planner_success=record.global_planner_success,
            local_planner_success=record.local_planner_success,
        )
        for reason in reasons:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def aggregate_evaluations(records: Iterable[PathEvaluation]) -> dict[str, Any]:
    values = list(records)
    paths = [record for record in values if record.path_available]
    reference_scored = [record for record in paths if record.reference_metrics_computed]
    strict_scored = {
        policy: [
            record
            for record in values
            if _record_strict_success(record, policy) is not None
        ]
        for policy in ("unknown_as_free", "unknown_as_occupied")
    }
    strict_successful = {
        policy: [
            record
            for record in strict_scored[policy]
            if _record_strict_success(record, policy) is True
        ]
        for policy in strict_scored
    }
    # Aggregate legacy Task Success keys from the same canonical strict result.
    task_scored_unknown_as_free = strict_scored["unknown_as_free"]
    task_scored_unknown_as_occupied = strict_scored["unknown_as_occupied"]
    successful_unknown_as_free = strict_successful["unknown_as_free"]
    successful_unknown_as_occupied = strict_successful["unknown_as_occupied"]
    global_stage_scored = [
        record for record in values if record.global_planner_success is not None
    ]
    local_stage_scored = [
        record for record in values if record.local_planner_success is not None
    ]

    def nested(record: PathEvaluation, *keys: str) -> Any:
        value: Any = record.extra
        for key in keys:
            if not isinstance(value, Mapping):
                return None
            value = value.get(key)
        return value

    rrt_status_scored = [
        record
        for record in values
        if isinstance(nested(record, "debug_info", "rrt_message"), str)
    ]
    rrt_direct = [
        record
        for record in rrt_status_scored
        if nested(record, "debug_info", "rrt_message")
        == "RRT found a direct collision-free path."
    ]
    shortcut_scored = [
        record
        for record in values
        if isinstance(
            nested(record, "ego_stats", "coarse_points_after_shortcut"),
            (int, float),
        )
    ]
    shortcut_to_two = [
        record
        for record in shortcut_scored
        if int(nested(record, "ego_stats", "coarse_points_after_shortcut")) == 2
    ]
    strict_occupied_ids = {
        id(record) for record in strict_successful["unknown_as_occupied"]
    }
    result = {
        "num_trials": len(values),
        "num_native_success": sum(record.native_success for record in values),
        "num_native_failure": len(values) - sum(
            record.native_success for record in values
        ),
        "native_success_rate": (
            None if not values else float(sum(record.native_success for record in values) / len(values))
        ),
        "num_global_planner_scored": len(global_stage_scored),
        "num_global_planner_success": sum(
            bool(record.global_planner_success) for record in global_stage_scored
        ),
        "num_global_planner_failure": sum(
            not bool(record.global_planner_success) for record in global_stage_scored
        ),
        "global_planner_success_rate": (
            None
            if not global_stage_scored
            else float(
                sum(bool(record.global_planner_success) for record in global_stage_scored)
                / len(global_stage_scored)
            )
        ),
        "num_local_planner_scored": len(local_stage_scored),
        "num_local_planner_success": sum(
            bool(record.local_planner_success) for record in local_stage_scored
        ),
        "num_local_planner_failure": sum(
            not bool(record.local_planner_success) for record in local_stage_scored
        ),
        "local_planner_success_rate": (
            None
            if not local_stage_scored
            else float(
                sum(bool(record.local_planner_success) for record in local_stage_scored)
                / len(local_stage_scored)
            )
        ),
        "num_rrt_status_scored": len(rrt_status_scored),
        "num_rrt_direct_collision_free": len(rrt_direct),
        "rrt_direct_collision_free_rate": (
            None
            if not rrt_status_scored
            else float(len(rrt_direct) / len(rrt_status_scored))
        ),
        "num_rrt_direct_collision_free_strict_success_unknown_as_occupied": sum(
            id(record) in strict_occupied_ids for record in rrt_direct
        ),
        "rrt_direct_collision_free_rate_strict_success_unknown_as_occupied": (
            None
            if not strict_successful["unknown_as_occupied"] or not rrt_status_scored
            else float(
                sum(id(record) in strict_occupied_ids for record in rrt_direct)
                / len(strict_successful["unknown_as_occupied"])
            )
        ),
        "num_global_shortcut_scored": len(shortcut_scored),
        "num_global_shortcut_to_two_points": len(shortcut_to_two),
        "global_shortcut_to_two_points_rate": (
            None
            if not shortcut_scored
            else float(len(shortcut_to_two) / len(shortcut_scored))
        ),
        "num_global_shortcut_to_two_points_strict_success_unknown_as_occupied": sum(
            id(record) in strict_occupied_ids for record in shortcut_to_two
        ),
        "global_shortcut_to_two_points_rate_strict_success_unknown_as_occupied": (
            None
            if not strict_successful["unknown_as_occupied"] or not shortcut_scored
            else float(
                sum(id(record) in strict_occupied_ids for record in shortcut_to_two)
                / len(strict_successful["unknown_as_occupied"])
            )
        ),
        "num_task_scored": len(task_scored_unknown_as_free),
        "num_task_success_unknown_as_free": len(successful_unknown_as_free),
        "num_task_failure_unknown_as_free": (
            len(task_scored_unknown_as_free) - len(successful_unknown_as_free)
        ),
        "task_success_rate_unknown_as_free": (
            None
            if not task_scored_unknown_as_free
            else float(
                len(successful_unknown_as_free) / len(task_scored_unknown_as_free)
            )
        ),
        "num_task_success_unknown_as_occupied": len(
            successful_unknown_as_occupied
        ),
        "num_task_failure_unknown_as_occupied": (
            len(task_scored_unknown_as_occupied) - len(successful_unknown_as_occupied)
        ),
        "task_success_rate_unknown_as_occupied": (
            None
            if not task_scored_unknown_as_occupied
            else float(
                len(successful_unknown_as_occupied)
                / len(task_scored_unknown_as_occupied)
            )
        ),
        "num_strict_scored_unknown_as_free": len(strict_scored["unknown_as_free"]),
        "num_strict_success_unknown_as_free": len(strict_successful["unknown_as_free"]),
        "num_strict_failure_unknown_as_free": (
            len(strict_scored["unknown_as_free"])
            - len(strict_successful["unknown_as_free"])
        ),
        "strict_success_rate_unknown_as_free": (
            None
            if not strict_scored["unknown_as_free"]
            else float(
                len(strict_successful["unknown_as_free"])
                / len(strict_scored["unknown_as_free"])
            )
        ),
        "num_strict_scored_unknown_as_occupied": len(
            strict_scored["unknown_as_occupied"]
        ),
        "num_strict_success_unknown_as_occupied": len(
            strict_successful["unknown_as_occupied"]
        ),
        "num_strict_failure_unknown_as_occupied": (
            len(strict_scored["unknown_as_occupied"])
            - len(strict_successful["unknown_as_occupied"])
        ),
        "strict_success_rate_unknown_as_occupied": (
            None
            if not strict_scored["unknown_as_occupied"]
            else float(
                len(strict_successful["unknown_as_occupied"])
                / len(strict_scored["unknown_as_occupied"])
            )
        ),
        "num_paths": len(paths),
        "goal_reached_rate": (
            None if not values else float(sum(record.goal_reached for record in values) / len(values))
        ),
        "safe_rate_all_trials_unknown_as_free": (
            None
            if not task_scored_unknown_as_free
            else float(
                sum(record.safe_unknown_as_free for record in task_scored_unknown_as_free)
                / len(task_scored_unknown_as_free)
            )
        ),
        "safe_rate_paths_unknown_as_free": (
            None
            if not reference_scored
            else float(
                sum(record.safe_unknown_as_free for record in reference_scored)
                / len(reference_scored)
            )
        ),
        "safe_rate_all_trials_unknown_as_occupied": (
            None
            if not task_scored_unknown_as_occupied
            else float(
                sum(
                    record.safe_unknown_as_occupied
                    for record in task_scored_unknown_as_occupied
                )
                / len(task_scored_unknown_as_occupied)
            )
        ),
        "safe_rate_paths_unknown_as_occupied": (
            None
            if not reference_scored
            else float(
                sum(record.safe_unknown_as_occupied for record in reference_scored)
                / len(reference_scored)
            )
        ),
        "mean_computation_time_sec_all_trials": _mean_optional(
            record.computation_time_sec for record in values
        ),
        "mean_evaluation_time_sec_all_trials": _mean_optional(
            record.evaluation_time_sec for record in values
        ),
        "mean_global_planning_time_sec_all_trials": _mean_optional(
            record.global_planning_time_sec for record in values
        ),
        "mean_local_planning_time_sec_all_trials": _mean_optional(
            record.local_planning_time_sec for record in values
        ),
        "mean_planner_validation_time_sec_all_trials": _mean_optional(
            record.planner_validation_time_sec for record in values
        ),
        "mean_cpu_peak_rss_sampled_delta_bytes_all_trials": _mean_optional(
            record.resource_usage.get("cpu_rss_peak_sampled_delta_bytes") for record in values
        ),
        "max_cpu_peak_rss_sampled_delta_bytes_all_trials": _max_optional(
            record.resource_usage.get("cpu_rss_peak_sampled_delta_bytes") for record in values
        ),
        "mean_cpu_peak_rss_sampled_bytes_all_trials": _mean_optional(
            record.resource_usage.get("cpu_rss_peak_sampled_bytes") for record in values
        ),
        "max_cpu_peak_rss_sampled_bytes_all_trials": _max_optional(
            record.resource_usage.get("cpu_rss_peak_sampled_bytes") for record in values
        ),
        "mean_cuda_torch_peak_allocated_delta_bytes_all_trials": _mean_optional(
            record.resource_usage.get("cuda_torch_allocated_peak_delta_bytes") for record in values
        ),
        "max_cuda_torch_peak_allocated_delta_bytes_all_trials": _max_optional(
            record.resource_usage.get("cuda_torch_allocated_peak_delta_bytes") for record in values
        ),
        "mean_cuda_torch_peak_allocated_bytes_all_trials": _mean_optional(
            record.resource_usage.get("cuda_torch_allocated_peak_bytes") for record in values
        ),
        "max_cuda_torch_peak_allocated_bytes_all_trials": _max_optional(
            record.resource_usage.get("cuda_torch_allocated_peak_bytes") for record in values
        ),
        "mean_cuda_torch_peak_reserved_delta_bytes_all_trials": _mean_optional(
            record.resource_usage.get("cuda_torch_reserved_peak_delta_bytes") for record in values
        ),
        "max_cuda_torch_peak_reserved_delta_bytes_all_trials": _max_optional(
            record.resource_usage.get("cuda_torch_reserved_peak_delta_bytes") for record in values
        ),
        "mean_cuda_torch_peak_reserved_bytes_all_trials": _mean_optional(
            record.resource_usage.get("cuda_torch_reserved_peak_bytes") for record in values
        ),
        "max_cuda_torch_peak_reserved_bytes_all_trials": _max_optional(
            record.resource_usage.get("cuda_torch_reserved_peak_bytes") for record in values
        ),
    }
    for policy in ("unknown_as_free", "unknown_as_occupied"):
        result.update(_strict_path_metrics(values, policy))
        result[f"strict_failure_reason_counts_{policy}"] = (
            _strict_failure_reason_counts(values, policy)
        )

    # Conservative-policy aliases keep existing summary consumers working while
    # changing their population from all returned/completed paths to strict success.
    occupied_suffix = "strict_success_unknown_as_occupied"
    result.update(
        {
            "num_completed_paths": len(strict_successful["unknown_as_occupied"]),
            "num_nonstraight_completed_paths": result[
                f"num_nonstraight_{occupied_suffix}"
            ],
            "mean_minimum_clearance_m": result[
                f"mean_minimum_clearance_m_{occupied_suffix}"
            ],
            "min_minimum_clearance_m": result[
                f"min_minimum_clearance_m_{occupied_suffix}"
            ],
            "mean_unsafe_sample_fraction_unknown_as_free": result[
                f"mean_unsafe_sample_fraction_unknown_as_free_{occupied_suffix}"
            ],
            "mean_unsafe_sample_fraction_unknown_as_occupied": result[
                f"mean_unsafe_sample_fraction_unknown_as_occupied_{occupied_suffix}"
            ],
            "mean_occupied_collision_sample_fraction": result[
                f"mean_occupied_collision_sample_fraction_{occupied_suffix}"
            ],
            "mean_unknown_sample_fraction": result[
                f"mean_unknown_sample_fraction_{occupied_suffix}"
            ],
            "mean_path_length_m_completed_paths": result[
                f"mean_path_length_m_{occupied_suffix}"
            ],
            "mean_detour_ratio_completed_paths": result[
                f"mean_detour_ratio_{occupied_suffix}"
            ],
            "mean_task_detour_ratio_completed_paths": result[
                f"mean_task_detour_ratio_{occupied_suffix}"
            ],
            "mean_start_endpoint_error_m_completed_paths": result[
                f"mean_start_endpoint_error_m_{occupied_suffix}"
            ],
            "mean_goal_endpoint_error_m_completed_paths": result[
                f"mean_goal_endpoint_error_m_{occupied_suffix}"
            ],
            "max_endpoint_error_m_completed_paths": result[
                f"max_endpoint_error_m_{occupied_suffix}"
            ],
            "mean_max_straight_line_deviation_m_completed_paths": result[
                f"mean_max_straight_line_deviation_m_{occupied_suffix}"
            ],
            "mean_max_curvature_rad_per_m_completed_paths": result[
                f"mean_max_curvature_rad_per_m_{occupied_suffix}"
            ],
            "mean_squared_curvature_completed_paths": result[
                f"mean_squared_curvature_{occupied_suffix}"
            ],
        }
    )
    return result
