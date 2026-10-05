"""Reference-based metrics for offline path planning."""

from .metrics import (
    PathEvaluation,
    VoxelReference,
    aggregate_evaluations,
    evaluate_geometric_smoothness,
    evaluate_path_metrics,
    evaluate_reference_safety,
    evaluate_strict_success,
    path_length,
    resample_polyline,
    strict_failure_reasons,
)
from .radius_profiles import (
    DEFAULT_ADDITIONAL_ROBOT_RADII_M,
    DEFAULT_BASELINE_ROBOT_RADIUS_M,
    DATASET_REFERENCE_ROBOT_RADIUS_M,
    DATASET_ROBOT_RADIUS_SCALE,
    default_robot_radius_for_dataset,
    normalize_robot_radius_profiles,
    radius_profile_directory_name,
    robot_radius_scale_for_dataset,
    validate_endpoint_manifest_radius,
)

__all__ = [
    "PathEvaluation",
    "VoxelReference",
    "aggregate_evaluations",
    "evaluate_geometric_smoothness",
    "evaluate_path_metrics",
    "evaluate_reference_safety",
    "evaluate_strict_success",
    "path_length",
    "resample_polyline",
    "strict_failure_reasons",
    "DEFAULT_ADDITIONAL_ROBOT_RADII_M",
    "DEFAULT_BASELINE_ROBOT_RADIUS_M",
    "DATASET_REFERENCE_ROBOT_RADIUS_M",
    "DATASET_ROBOT_RADIUS_SCALE",
    "default_robot_radius_for_dataset",
    "normalize_robot_radius_profiles",
    "radius_profile_directory_name",
    "robot_radius_scale_for_dataset",
    "validate_endpoint_manifest_radius",
]
