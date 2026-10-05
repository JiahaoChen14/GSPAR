#!/usr/bin/env python3
"""Aggregation helpers used by the batch evaluation-suite summary."""

from __future__ import annotations

from statistics import mean
from typing import Any, Iterable

QUALITY_MEAN_FIELDS = (
    "surface_iou_zero_tolerance",
    "surface_precision_zero_tolerance",
    "surface_recall_zero_tolerance",
    "surface_fscore_zero_tolerance",
    "surface_precision_5cm",
    "surface_recall_5cm",
    "surface_fscore_5cm",
    "surface_precision_10cm",
    "surface_recall_10cm",
    "surface_fscore_10cm",
    "truncated_chamfer_l1_m",
    "chamfer_l1_m",
    "accuracy_mean_m",
    "accuracy_p95_m",
    "completeness_mean_m",
    "completeness_p95_m",
    "gaussian_retention_ratio",
    "nvblox_tsdf_sec",
    "nvblox_esdf_sec",
    "nvblox_tsdf_esdf_sec",
    "gaussian_to_voxel_calculation_sec",
    "nvblox_tsdf_esdf_over_gaussian_calculation_ratio",
    "nvblox_esdf_bytes",
    "gaussian_voxel_bytes",
    "nvblox_esdf_over_gaussian_voxel",
    "planning_map_payload_combined_bytes",
)

PATH_COUNT_FIELDS = (
    "num_trials",
    "num_native_success",
    "num_native_failure",
    "num_global_planner_scored",
    "num_global_planner_success",
    "num_global_planner_failure",
    "num_local_planner_scored",
    "num_local_planner_success",
    "num_local_planner_failure",
    "num_task_scored",
    "num_task_success_unknown_as_free",
    "num_task_failure_unknown_as_free",
    "num_task_success_unknown_as_occupied",
    "num_task_failure_unknown_as_occupied",
    "num_strict_scored_unknown_as_free",
    "num_strict_success_unknown_as_free",
    "num_strict_failure_unknown_as_free",
    "num_strict_scored_unknown_as_occupied",
    "num_strict_success_unknown_as_occupied",
    "num_strict_failure_unknown_as_occupied",
    "num_paths",
    "num_completed_paths",
    "num_nonstraight_strict_success_unknown_as_free",
    "num_nonstraight_strict_success_unknown_as_occupied",
    "num_nonstraight_completed_paths",
    "num_rrt_status_scored",
    "num_rrt_direct_collision_free",
    "num_rrt_direct_collision_free_strict_success_unknown_as_occupied",
    "num_global_shortcut_scored",
    "num_global_shortcut_to_two_points",
    "num_global_shortcut_to_two_points_strict_success_unknown_as_occupied",
)

PATH_ALL_TRIAL_MEAN_FIELDS = (
    "mean_computation_time_sec_all_trials",
    "mean_evaluation_time_sec_all_trials",
    "mean_global_planning_time_sec_all_trials",
    "mean_local_planning_time_sec_all_trials",
    "mean_planner_validation_time_sec_all_trials",
    "mean_cpu_peak_rss_sampled_delta_bytes_all_trials",
    "mean_cuda_torch_peak_allocated_delta_bytes_all_trials",
    "mean_cuda_torch_peak_reserved_delta_bytes_all_trials",
    "mean_end_to_end_time_sec_including_amortized_scene_build",
)

STRICT_POLICIES = ("unknown_as_free", "unknown_as_occupied")

PATH_STRICT_MEAN_BASE_FIELDS = (
    "mean_minimum_clearance_m",
    "mean_unsafe_sample_fraction_unknown_as_free",
    "mean_unsafe_sample_fraction_unknown_as_occupied",
    "mean_occupied_collision_sample_fraction",
    "mean_unknown_sample_fraction",
    "mean_path_length_m",
    "mean_detour_ratio",
    "mean_task_detour_ratio",
    "mean_start_endpoint_error_m",
    "mean_goal_endpoint_error_m",
    "mean_max_straight_line_deviation_m",
    "mean_max_curvature_rad_per_m",
    "mean_squared_curvature",
)

PATH_STRICT_MAX_BASE_FIELDS = (
    "max_endpoint_error_m",
)

PATH_MAX_FIELDS = (
    "max_cpu_peak_rss_sampled_delta_bytes_all_trials",
    "max_cuda_torch_peak_allocated_delta_bytes_all_trials",
    "max_cuda_torch_peak_reserved_delta_bytes_all_trials",
)

PATH_SCENE_MEAN_FIELDS = (
    "scene_global_map_build_time_sec",
    "scene_esdf_load_time_sec",
    "shared_scene_input_array_bytes",
    "method_representation_array_bytes",
)

def get(data: Any, *keys: str | int) -> Any:
    value = data
    for key in keys:
        if isinstance(key, int) and isinstance(value, list) and key < len(value):
            value = value[key]
        elif isinstance(key, str) and isinstance(value, dict):
            value = value.get(key)
        else:
            return None
    return value


def number(data: Any, *keys: str | int) -> int | float | None:
    value = get(data, *keys)
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def quality_row(dataset: str, scene: str, data: dict[str, Any]) -> dict[str, Any]:
    threshold_5cm = get(data, "surface_metrics", "by_threshold", 0) or {}
    threshold_10cm = get(data, "surface_metrics", "by_threshold", 1) or {}
    return {
        "dataset": dataset,
        "scene": scene,
        "reference_surface_voxels": number(data, "surface_metrics", "counts", "reference_surface_voxels"),
        "prediction_surface_voxels": number(data, "surface_metrics", "counts", "prediction_surface_voxels"),
        "surface_intersection_zero_tolerance": number(data, "surface_metrics", "zero_tolerance", "intersection_count"),
        "surface_union_zero_tolerance": number(data, "surface_metrics", "zero_tolerance", "union_count"),
        "surface_iou_zero_tolerance": number(data, "surface_metrics", "zero_tolerance", "surface_iou"),
        "surface_precision_zero_tolerance": number(data, "surface_metrics", "zero_tolerance", "precision"),
        "surface_recall_zero_tolerance": number(data, "surface_metrics", "zero_tolerance", "recall"),
        "surface_fscore_zero_tolerance": number(data, "surface_metrics", "zero_tolerance", "fscore"),
        "prediction_matched_5cm": number(threshold_5cm, "prediction_matched_count"),
        "reference_matched_5cm": number(threshold_5cm, "reference_matched_count"),
        "surface_precision_5cm": number(threshold_5cm, "precision"),
        "surface_recall_5cm": number(threshold_5cm, "recall"),
        "surface_fscore_5cm": number(threshold_5cm, "fscore"),
        "prediction_matched_10cm": number(threshold_10cm, "prediction_matched_count"),
        "reference_matched_10cm": number(threshold_10cm, "reference_matched_count"),
        "surface_precision_10cm": number(threshold_10cm, "precision"),
        "surface_recall_10cm": number(threshold_10cm, "recall"),
        "surface_fscore_10cm": number(threshold_10cm, "fscore"),
        "truncated_chamfer_l1_m": number(
            data, "surface_metrics", "truncated_chamfer_l1_m"
        ),
        "chamfer_l1_m": number(data, "surface_metrics", "chamfer_l1_m"),
        "accuracy_mean_m": number(data, "surface_metrics", "prediction_to_reference", "mean_m"),
        "accuracy_p95_m": number(data, "surface_metrics", "prediction_to_reference", "p95_m"),
        "completeness_mean_m": number(data, "surface_metrics", "reference_to_prediction", "mean_m"),
        "completeness_p95_m": number(data, "surface_metrics", "reference_to_prediction", "p95_m"),
        "original_gaussian_count": number(data, "map_representation", "original_gaussian_count"),
        "filtered_gaussian_count": number(data, "map_representation", "filtered_gaussian_count"),
        "gaussian_retention_ratio": number(data, "map_representation", "gaussian_retention_ratio"),
        "final_occupied_voxel_count": number(data, "map_representation", "final_occupied_voxel_count"),
        "nvblox_tsdf_sec": number(data, "efficiency", "timing_sec", "nvblox", "tsdf"),
        "nvblox_esdf_sec": number(data, "efficiency", "timing_sec", "nvblox", "esdf"),
        "nvblox_tsdf_esdf_sec": number(data, "efficiency", "timing_sec", "nvblox", "tsdf_esdf"),
        "gaussian_to_voxel_calculation_sec": number(data, "efficiency", "timing_sec", "gaussian_to_voxel", "calculation"),
        "nvblox_tsdf_esdf_over_gaussian_calculation_ratio": number(data, "efficiency", "timing_sec", "comparison", "nvblox_tsdf_esdf_over_gaussian"),
        "nvblox_esdf_bytes": number(data, "efficiency", "stored_map_size_bytes", "nvblox_esdf_ply"),
        "gaussian_voxel_bytes": number(data, "efficiency", "stored_map_size_bytes", "gaussian_voxel_npz"),
        "nvblox_esdf_over_gaussian_voxel": number(data, "efficiency", "stored_map_size_bytes", "comparison", "nvblox_esdf_over_gaussian_voxel"),
        "planning_map_payload_combined_bytes": number(data, "efficiency", "planning_map_payload_bytes", "combined"),
    }


def available_mean(rows: Iterable[dict[str, Any]], field: str) -> float | None:
    values = [row[field] for row in rows if isinstance(row.get(field), (int, float))]
    return mean(values) if values else None


def weighted_mean(rows: Iterable[dict[str, Any]], field: str, weight: str) -> float | None:
    pairs = [
        (float(row[field]), float(row[weight]))
        for row in rows
        if isinstance(row.get(field), (int, float))
        and isinstance(row.get(weight), (int, float))
        and row[weight] > 0
    ]
    total_weight = sum(item[1] for item in pairs)
    return None if not total_weight else sum(value * item_weight for value, item_weight in pairs) / total_weight


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if not denominator else float(numerator / denominator)


def aggregate_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sums = {
        field: sum(int(row.get(field) or 0) for row in rows)
        for field in (
            "reference_surface_voxels",
            "prediction_surface_voxels",
            "surface_intersection_zero_tolerance",
            "surface_union_zero_tolerance",
            "prediction_matched_5cm",
            "reference_matched_5cm",
            "prediction_matched_10cm",
            "reference_matched_10cm",
            "original_gaussian_count",
            "filtered_gaussian_count",
            "final_occupied_voxel_count",
            "nvblox_esdf_bytes",
            "gaussian_voxel_bytes",
            "planning_map_payload_combined_bytes",
        )
    }
    precision_5cm = ratio(sums["prediction_matched_5cm"], sums["prediction_surface_voxels"])
    recall_5cm = ratio(sums["reference_matched_5cm"], sums["reference_surface_voxels"])
    precision_10cm = ratio(sums["prediction_matched_10cm"], sums["prediction_surface_voxels"])
    recall_10cm = ratio(sums["reference_matched_10cm"], sums["reference_surface_voxels"])

    def fscore(precision: float | None, recall: float | None) -> float | None:
        if precision is None or recall is None or precision + recall == 0.0:
            return None
        return 2.0 * precision * recall / (precision + recall)

    precision_zero = ratio(
        sums["surface_intersection_zero_tolerance"],
        sums["prediction_surface_voxels"],
    )
    recall_zero = ratio(
        sums["surface_intersection_zero_tolerance"],
        sums["reference_surface_voxels"],
    )

    return {
        "scene_count": len(rows),
        "counts_and_totals": sums,
        "micro_surface_iou_zero_tolerance": ratio(
            sums["surface_intersection_zero_tolerance"],
            sums["surface_union_zero_tolerance"],
        ),
        "micro_surface_precision_zero_tolerance": precision_zero,
        "micro_surface_recall_zero_tolerance": recall_zero,
        "micro_surface_fscore_zero_tolerance": fscore(precision_zero, recall_zero),
        "micro_surface_precision_5cm": precision_5cm,
        "micro_surface_recall_5cm": recall_5cm,
        "micro_surface_fscore_5cm": fscore(precision_5cm, recall_5cm),
        "micro_surface_precision_10cm": precision_10cm,
        "micro_surface_recall_10cm": recall_10cm,
        "micro_surface_fscore_10cm": fscore(precision_10cm, recall_10cm),
        "micro_gaussian_retention_ratio": ratio(
            sums["filtered_gaussian_count"], sums["original_gaussian_count"]
        ),
        "macro_means": {field: available_mean(rows, field) for field in QUALITY_MEAN_FIELDS},
    }


def aggregate_path(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {field: sum(int(row.get(field) or 0) for row in rows) for field in PATH_COUNT_FIELDS}
    result: dict[str, Any] = {
        "scene_count": len(rows),
        **counts,
        "native_success_rate": ratio(counts["num_native_success"], counts["num_trials"]),
        "global_planner_success_rate": ratio(
            counts["num_global_planner_success"],
            counts["num_global_planner_scored"],
        ),
        "local_planner_success_rate": ratio(
            counts["num_local_planner_success"],
            counts["num_local_planner_scored"],
        ),
        "task_success_rate_unknown_as_free": ratio(
            counts["num_task_success_unknown_as_free"], counts["num_task_scored"]
        ),
        "task_success_rate_unknown_as_occupied": ratio(
            counts["num_task_success_unknown_as_occupied"], counts["num_task_scored"]
        ),
        "strict_success_rate_unknown_as_free": ratio(
            counts["num_strict_success_unknown_as_free"],
            counts["num_strict_scored_unknown_as_free"],
        ),
        "strict_success_rate_unknown_as_occupied": ratio(
            counts["num_strict_success_unknown_as_occupied"],
            counts["num_strict_scored_unknown_as_occupied"],
        ),
        "goal_reached_rate": weighted_mean(rows, "goal_reached_rate", "num_trials"),
        "safe_rate_all_trials_unknown_as_free": weighted_mean(
            rows, "safe_rate_all_trials_unknown_as_free", "num_task_scored"
        ),
        "safe_rate_all_trials_unknown_as_occupied": weighted_mean(
            rows, "safe_rate_all_trials_unknown_as_occupied", "num_task_scored"
        ),
        "rrt_direct_collision_free_rate": ratio(
            counts["num_rrt_direct_collision_free"],
            counts["num_rrt_status_scored"],
        ),
        "rrt_direct_collision_free_rate_strict_success_unknown_as_occupied": ratio(
            counts[
                "num_rrt_direct_collision_free_strict_success_unknown_as_occupied"
            ],
            counts["num_strict_success_unknown_as_occupied"],
        ),
        "global_shortcut_to_two_points_rate": ratio(
            counts["num_global_shortcut_to_two_points"],
            counts["num_global_shortcut_scored"],
        ),
        "global_shortcut_to_two_points_rate_strict_success_unknown_as_occupied": ratio(
            counts[
                "num_global_shortcut_to_two_points_strict_success_unknown_as_occupied"
            ],
            counts["num_strict_success_unknown_as_occupied"],
        ),
    }
    result.update(
        {field: weighted_mean(rows, field, "num_trials") for field in PATH_ALL_TRIAL_MEAN_FIELDS}
    )
    for policy in STRICT_POLICIES:
        suffix = f"strict_success_{policy}"
        for base_field in PATH_STRICT_MEAN_BASE_FIELDS:
            field = f"{base_field}_{suffix}"
            weight = (
                f"num_nonstraight_{suffix}"
                if base_field
                in {
                    "mean_detour_ratio",
                    "mean_task_detour_ratio",
                    "mean_max_curvature_rad_per_m",
                    "mean_squared_curvature",
                }
                else f"num_strict_success_{policy}"
            )
            result[field] = weighted_mean(rows, field, weight)
        for base_field in PATH_STRICT_MAX_BASE_FIELDS:
            field = f"{base_field}_{suffix}"
            result[field] = max(
                (
                    row[field]
                    for row in rows
                    if isinstance(row.get(field), (int, float))
                ),
                default=None,
            )
        min_field = f"min_minimum_clearance_m_{suffix}"
        result[min_field] = min(
            (
                row[min_field]
                for row in rows
                if isinstance(row.get(min_field), (int, float))
            ),
            default=None,
        )

    occupied_suffix = "strict_success_unknown_as_occupied"
    result.update(
        {
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
    result.update(
        {
            field: max(
                (row[field] for row in rows if isinstance(row.get(field), (int, float))),
                default=None,
            )
            for field in PATH_MAX_FIELDS
        }
    )
    result.update({f"mean_{field}": available_mean(rows, field) for field in PATH_SCENE_MEAN_FIELDS})
    return result


