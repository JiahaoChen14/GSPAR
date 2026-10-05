#!/usr/bin/env python3
"""Compare planners on shared endpoints using an observed TSDF as reference."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from GSPAR.mapping.gaussian_occupancy.build_sparse_voxel_map import (
    Config as SparseVoxelConfig,
    build_sparse_voxel_payload,
)
from GSPAR.mapping.gaussian_occupancy.voxel_preprocess import (
    preprocess_gaussians,
    transform_gaussians,
)

from GSPAR.path.planning import (
    GLOBAL_ESDF_WEIGHTED_ASTAR,
    GLOBAL_WEIGHTED_ASTAR,
    EgoPlanOptions,
    EsdfGlobalPlanOptions,
    GlobalPlanOptions,
    GSPARDirectPlanOptions,
    StompEsdfPlanOptions,
    load_esdf_graph,
    run_ego_path,
    run_esdf_global_path,
    run_global_path,
    run_gspar_direct_path,
    run_stomp_esdf_path,
)
from GSPAR.path.method_names import (
    GSPAR_DIRECT,
    GSPAR_DIRECT_LABEL,
    GSPAR_GLOBAL_ASTAR_EGO_LABEL,
    GSPAR_GLOBAL_ASTAR_EGO,
    METHOD_LABEL_BY_ID,
    NVBLOX_ESDF_ASTAR_STOMP,
    NVBLOX_ESDF_ASTAR_STOMP_LABEL,
    canonical_method_label,
)
from GSPAR.path.global_path.esdf_astar_world_path import (
    UNKNOWN_POLICY_OCCUPIED,
    UNKNOWN_POLICY_FREE,
    resolve_esdf_path,
)
from GSPAR.evaluation.resource_usage import (
    ResourceMonitor,
    summarize_resource_usages,
)

from .metrics import (
    PathEvaluation,
    VoxelReference,
    aggregate_evaluations,
    evaluate_path_metrics,
)
from .radius_profiles import (
    DEFAULT_ADDITIONAL_ROBOT_RADII_M,
    DEFAULT_BASELINE_ROBOT_RADIUS_M,
    normalize_robot_radius_profiles,
    radius_profile_directory_name,
    validate_endpoint_manifest_radius,
)

# Preserve the previous local helper names for callers that imported them.
_radius_profile_directory_name = radius_profile_directory_name
_validate_endpoint_radius = validate_endpoint_manifest_radius


def _optional_stage(timings: Mapping[str, Any], name: str) -> float | None:
    value = timings.get(name)
    return None if value is None else float(value)


def _nvblox_map_build_time(esdf_path: Path) -> float | None:
    stats_path = esdf_path.parent / "timing_stats.json"
    if not stats_path.is_file():
        return None
    timing = json.loads(stats_path.read_text(encoding="utf-8")).get("timing_sec", {})
    tsdf, esdf = timing.get("tsdf"), timing.get("esdf")
    if not isinstance(tsdf, (int, float)) or not isinstance(esdf, (int, float)):
        return None
    total = float(tsdf) + float(esdf)

    manifest_path = esdf_path.parent.parent / "depth_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        render = manifest.get("render", {}).get("timing_sec", {}).get(
            "render_inference"
        )
        if not isinstance(render, (int, float)):
            return None
        total += float(render)
    return total


def evaluate_gspar_direct_trial(
    *,
    raw_gaussian_map: Any,
    start_world: np.ndarray,
    goal_world: np.ndarray,
    options: GSPARDirectPlanOptions | None = None,
    reference: VoxelReference,
    robot_radius: float | None = None,
    trial_id: str | int | None = None,
    compute_reference_metrics: bool = True,
    reference_sample_spacing_m: float | None = None,
    reference_safety_margin_m: float = 0.0,
    goal_tolerance_m: float = 0.10,
    smoothness_sample_spacing_m: float = 0.05,
) -> PathEvaluation:
    options = options if options is not None else GSPARDirectPlanOptions()
    result = None
    with ResourceMonitor(cuda_device=options.device) as resource_monitor:
        started = time.perf_counter()
        try:
            result = run_gspar_direct_path(
                raw_gaussian_map,
                start_world,
                goal_world,
                options,
                include_gaussian_payload=False,
            )
            native_success = bool(result.success)
            # The Direct planner keeps a start-to-goal candidate internally for
            # diagnostics when RRT fails. It is not a valid returned solution and
            # must not be evaluated or serialized as an available task path.
            path = result.path_world if native_success else None
            failure_reason = str(result.debug_info.get("failure_reason", ""))
        except Exception as error:
            native_success = False
            path = None
            failure_reason = str(error)
        elapsed = time.perf_counter() - started
    debug = {} if result is None else dict(result.debug_info)
    planner_timing = dict(debug.get("timings_sec", {}))
    global_planning = _optional_stage(planner_timing, "global_planning")
    local_planning = _optional_stage(planner_timing, "local_planning")
    planner_validation = _optional_stage(planner_timing, "planner_validation")
    fallback = int(debug.get("safe_fallback_candidate_index", -1)) >= 0
    global_planner_success = bool(debug.get("rrt_success", native_success))
    local_planner_success = bool(native_success and not fallback)
    return evaluate_path_metrics(
        method=GSPAR_DIRECT_LABEL,
        trial_id=trial_id,
        path_world=path,
        reference=reference,
        robot_radius=float(options.robot_radius if robot_radius is None else robot_radius),
        native_success=native_success,
        failure_reason=failure_reason,
        start_world=start_world,
        goal_world=goal_world,
        effective_start_world=start_world,
        effective_goal_world=goal_world,
        trajectory_type="gspar_direct_piecewise_cubic_hermite_optimized",
        fallback_used=fallback,
        computation_time_sec=elapsed,
        global_planning_time_sec=global_planning,
        local_planning_time_sec=local_planning,
        planner_validation_time_sec=planner_validation,
        global_planner_success=global_planner_success,
        local_planner_success=local_planner_success,
        resource_usage=resource_monitor.to_dict(),
        compute_reference_metrics=bool(compute_reference_metrics),
        reference_sample_spacing_m=reference_sample_spacing_m,
        reference_safety_margin_m=float(reference_safety_margin_m),
        goal_tolerance_m=float(goal_tolerance_m),
        smoothness_sample_spacing_m=float(smoothness_sample_spacing_m),
        extra={
            "rrt_random_seed": int(
                debug.get("rrt_random_seed", options.rrt_random_seed)
            ),
            "planner_header": {} if result is None else dict(result.header),
            "debug_info": debug,
        },
    )


def evaluate_occupancy_trial(
    *,
    payload: Mapping[str, Any],
    start_world: np.ndarray,
    goal_world: np.ndarray,
    global_options: GlobalPlanOptions,
    reference: VoxelReference,
    robot_radius: float,
    ego_options: EgoPlanOptions | None = None,
    map_label: str = "offline occupancy map",
    method_name: str | None = None,
    trial_id: str | int | None = None,
    compute_reference_metrics: bool = True,
    reference_sample_spacing_m: float | None = None,
    reference_safety_margin_m: float = 0.0,
    goal_tolerance_m: float = 0.10,
    smoothness_sample_spacing_m: float = 0.05,
) -> PathEvaluation:
    timing = {"query_global": 0.0, "query_ego": 0.0, "query_total": 0.0}
    global_result = None
    ego_result = None
    path = None
    native_success = False
    global_planner_success = False
    local_planner_success = None if ego_options is None else False
    failure_reason = ""
    effective_start = np.asarray(start_world, dtype=np.float32).reshape(3)
    effective_goal = np.asarray(goal_world, dtype=np.float32).reshape(3)
    with ResourceMonitor(monitor_cuda=False) as resource_monitor:
        started = time.perf_counter()
        try:
            stage = time.perf_counter()
            try:
                global_result = run_global_path(
                    payload,
                    start_world,
                    goal_world,
                    global_options,
                    map_label=map_label,
                )
            finally:
                timing["query_global"] = time.perf_counter() - stage
            path = global_result.path_world
            effective_start = global_result.start_world
            effective_goal = global_result.goal_world
            native_success = True
            global_planner_success = True
            if ego_options is not None:
                stage = time.perf_counter()
                try:
                    ego_result = run_ego_path(
                        payload,
                        global_result.path_world,
                        ego_options,
                        map_label=map_label,
                    )
                finally:
                    timing["query_ego"] = time.perf_counter() - stage
                path = ego_result.path_world
                native_success = bool(ego_result.success)
                failure_reason = str(ego_result.message)
        except Exception as error:
            failure_reason = str(error)
            if ego_options is not None and global_result is not None:
                native_success = False
        timing["query_total"] = time.perf_counter() - started
    fallback = False
    if ego_result is not None:
        fallback = bool(
            ego_result.stats.get("used_safety_fallback", False)
            or ego_result.stats.get("used_exact_collision_fallback", False)
        )
        local_planner_success = bool(ego_result.success and not fallback)
    ego_validation = (
        0.0
        if ego_result is None
        else float(
            dict(ego_result.stats.get("timings_sec", {})).get(
                "planner_validation",
                0.0,
            )
        )
    )
    local_planning = max(0.0, float(timing["query_ego"]) - ego_validation)
    default_name = GSPAR_GLOBAL_ASTAR_EGO_LABEL
    return evaluate_path_metrics(
        method=canonical_method_label(method_name) if method_name else default_name,
        trial_id=trial_id,
        path_world=path,
        reference=reference,
        robot_radius=float(robot_radius),
        native_success=native_success,
        failure_reason=failure_reason,
        start_world=start_world,
        goal_world=goal_world,
        effective_start_world=effective_start,
        effective_goal_world=effective_goal,
        trajectory_type=(
            "ego_bspline"
            if ego_result is not None and not fallback
            else "global_grid_polyline"
        ),
        fallback_used=fallback,
        computation_time_sec=float(timing["query_total"]),
        global_planning_time_sec=float(timing["query_global"]),
        local_planning_time_sec=local_planning,
        planner_validation_time_sec=ego_validation,
        global_planner_success=global_planner_success,
        local_planner_success=local_planner_success,
        resource_usage=resource_monitor.to_dict(),
        compute_reference_metrics=bool(compute_reference_metrics),
        reference_sample_spacing_m=reference_sample_spacing_m,
        reference_safety_margin_m=float(reference_safety_margin_m),
        goal_tolerance_m=float(goal_tolerance_m),
        smoothness_sample_spacing_m=float(smoothness_sample_spacing_m),
        extra={
            "global_header": {} if global_result is None else dict(global_result.header),
            "ego_header": {} if ego_result is None else dict(ego_result.header),
            "ego_stats": {} if ego_result is None else dict(ego_result.stats),
        },
    )


def evaluate_esdf_trial(
    *,
    graph: Any,
    start_world: np.ndarray,
    goal_world: np.ndarray,
    global_options: EsdfGlobalPlanOptions,
    reference: VoxelReference,
    robot_radius: float,
    stomp_options: StompEsdfPlanOptions | None = None,
    map_label: str = "nvblox ESDF",
    method_name: str,
    trial_id: str | int | None = None,
    compute_reference_metrics: bool = True,
    reference_sample_spacing_m: float | None = None,
    reference_safety_margin_m: float = 0.0,
    goal_tolerance_m: float = 0.10,
    smoothness_sample_spacing_m: float = 0.05,
) -> PathEvaluation:
    """Evaluate one nvblox ESDF global route with optional local refinement."""

    global_result = None
    local_result = None
    local_planner = ""
    path = None
    native_success = False
    global_planner_success = False
    local_planner_success = (
        None if stomp_options is None else False
    )
    failure_reason = ""
    effective_start = np.asarray(start_world, dtype=np.float32).reshape(3)
    effective_goal = np.asarray(goal_world, dtype=np.float32).reshape(3)
    timing = {"query_global": 0.0, "query_local": 0.0, "query_total": 0.0}
    with ResourceMonitor(monitor_cuda=False) as resource_monitor:
        started = time.perf_counter()
        try:
            stage = time.perf_counter()
            try:
                global_result = run_esdf_global_path(
                    graph,
                    start_world,
                    goal_world,
                    global_options,
                    map_label=map_label,
                )
            finally:
                timing["query_global"] = time.perf_counter() - stage
            path = global_result.path_world
            effective_start = global_result.start_world
            effective_goal = global_result.goal_world
            native_success = True
            global_planner_success = True
            if stomp_options is not None:
                stage = time.perf_counter()
                try:
                    local_result = run_stomp_esdf_path(
                        graph,
                        global_result.path_world,
                        stomp_options,
                        map_label=map_label,
                    )
                finally:
                    timing["query_local"] = time.perf_counter() - stage
                local_planner = "stomp"
            if local_result is not None:
                path = local_result.path_world
                native_success = bool(local_result.success)
                failure_reason = str(local_result.message)
        except Exception as error:
            native_success = False
            failure_reason = str(error)
        timing["query_total"] = time.perf_counter() - started

    local_timings = (
        {}
        if local_result is None
        else dict(local_result.stats.get("timings_sec", {}))
    )
    local_planning_time = float(
        local_timings.get("local_planning", timing["query_local"])
    )
    planner_validation_time = float(local_timings.get("planner_validation", 0.0))
    fallback = bool(
        local_result is not None
        and local_result.stats.get("used_safety_fallback", False)
    )
    if local_result is not None:
        fallback = bool(
            fallback or local_result.stats.get("used_shortcut_fallback", False)
        )
        local_planner_success = bool(local_result.success and not fallback)
    return evaluate_path_metrics(
        method=canonical_method_label(method_name),
        trial_id=trial_id,
        path_world=path,
        reference=reference,
        robot_radius=float(robot_radius),
        native_success=native_success,
        failure_reason=failure_reason,
        start_world=start_world,
        goal_world=goal_world,
        effective_start_world=effective_start,
        effective_goal_world=effective_goal,
        trajectory_type=(
            f"esdf_{local_planner}_trajectory"
            if local_result is not None and not fallback
            else "esdf_global_grid_polyline"
        ),
        fallback_used=fallback,
        computation_time_sec=float(timing["query_total"]),
        global_planning_time_sec=float(timing["query_global"]),
        local_planning_time_sec=local_planning_time,
        planner_validation_time_sec=planner_validation_time,
        global_planner_success=global_planner_success,
        local_planner_success=local_planner_success,
        resource_usage=resource_monitor.to_dict(),
        compute_reference_metrics=bool(compute_reference_metrics),
        reference_sample_spacing_m=reference_sample_spacing_m,
        reference_safety_margin_m=float(reference_safety_margin_m),
        goal_tolerance_m=float(goal_tolerance_m),
        smoothness_sample_spacing_m=float(smoothness_sample_spacing_m),
        extra={
            "global_header": {} if global_result is None else dict(global_result.header),
            "local_planner": local_planner,
            "local_header": {} if local_result is None else dict(local_result.header),
            "local_stats": {} if local_result is None else dict(local_result.stats),
        },
    )



PlannerEvalFn = Callable[[str | int | None, np.ndarray, np.ndarray], PathEvaluation]


def endpoint_pairs_from_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    pairs = manifest.get("pairs", [])
    if not isinstance(pairs, list):
        raise ValueError("endpoint manifest must contain a list field named 'pairs'")
    normalized = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise ValueError(f"endpoint pair {index} is not a dict")
        normalized.append(
            {
                **pair,
                "trial_id": pair.get("trial_id", index),
                "start_world": np.asarray(pair["start_world"], dtype=np.float32).reshape(3),
                "goal_world": np.asarray(pair["goal_world"], dtype=np.float32).reshape(3),
            }
        )
    return normalized


def load_endpoint_pairs(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).expanduser().open("r", encoding="utf-8") as file:
        return endpoint_pairs_from_manifest(json.load(file))


def run_evaluation_trials(
    *,
    planner_fns: dict[str, PlannerEvalFn],
    endpoint_pairs: Iterable[dict[str, Any]],
) -> list[PathEvaluation]:
    records = []
    for pair in endpoint_pairs:
        trial_id = pair.get("trial_id")
        start = np.asarray(pair["start_world"], dtype=np.float32).reshape(3)
        goal = np.asarray(pair["goal_world"], dtype=np.float32).reshape(3)
        for name, planner_fn in planner_fns.items():
            record = planner_fn(trial_id, start, goal)
            if record.method != name:
                record.extra["runner_planner_name"] = name
            records.append(record)
    return records


def write_jsonl(path: str | Path, records: Iterable[PathEvaluation]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for record in records:
            payload = record.to_json_dict()
            payload["method"] = canonical_method_label(str(payload["method"]))
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return output


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read evaluation records and normalize project method labels."""

    records: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "method" in record:
                record["method"] = canonical_method_label(str(record["method"]))
            records.append(record)
    return records


def summarize_by_method(records: Iterable[PathEvaluation]) -> dict[str, Any]:
    grouped: dict[str, list[PathEvaluation]] = defaultdict(list)
    for record in records:
        grouped[canonical_method_label(record.method)].append(record)
    output = {}
    for method, method_records in sorted(grouped.items()):
        summary = aggregate_evaluations(method_records)
        summary["resource_usage_summary_all_trials"] = summarize_resource_usages(
            {
                f"trial_{record.trial_id}": record.resource_usage
                for record in method_records
            }
        )
        output[method] = summary
    return output


ESDF_METHOD_LABELS = {
    NVBLOX_ESDF_ASTAR_STOMP_LABEL,
}


def paired_outcomes_by_esdf_method(
    records: Iterable[PathEvaluation],
) -> dict[str, Any]:
    """Count paired success/failure outcomes on the same endpoint trial IDs."""

    by_method: dict[str, dict[str | int | None, PathEvaluation]] = defaultdict(dict)
    for record in records:
        by_method[canonical_method_label(record.method)][record.trial_id] = record
    esdf_methods = sorted(set(by_method) & ESDF_METHOD_LABELS)
    fields = (
        "native_success",
        "task_success_unknown_as_free",
        "task_success_unknown_as_occupied",
        "strict_success_unknown_as_free",
        "strict_success_unknown_as_occupied",
    )
    output: dict[str, Any] = {}
    for esdf_method in esdf_methods:
        reference_records = by_method[esdf_method]
        comparisons: dict[str, Any] = {}
        for other_method in sorted(method for method in by_method if method != esdf_method):
            other_records = by_method[other_method]
            common_ids = sorted(
                set(reference_records) & set(other_records),
                key=lambda value: (str(type(value)), str(value)),
            )
            field_counts: dict[str, Any] = {}
            for field in fields:
                counts = {
                    "esdf_success_other_success": 0,
                    "esdf_success_other_failure": 0,
                    "esdf_failure_other_success": 0,
                    "esdf_failure_other_failure": 0,
                    "num_paired_scored_trials": 0,
                }
                for trial_id in common_ids:
                    esdf_success = getattr(reference_records[trial_id], field)
                    other_success = getattr(other_records[trial_id], field)
                    if esdf_success is None or other_success is None:
                        continue
                    key = (
                        f"esdf_{'success' if bool(esdf_success) else 'failure'}_"
                        f"other_{'success' if bool(other_success) else 'failure'}"
                    )
                    counts[key] += 1
                    counts["num_paired_scored_trials"] += 1
                field_counts[field] = counts
            comparisons[other_method] = {
                "num_common_trial_ids": len(common_ids),
                "outcomes": field_counts,
            }
        output[esdf_method] = comparisons
    return output


def write_summary(
    path: str | Path,
    records: Iterable[PathEvaluation],
    *,
    scene_setup: dict[str, Any] | None = None,
) -> Path:
    values = list(records)
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    by_method = summarize_by_method(values)
    setup = dict(scene_setup or {})
    global_build = setup.get("global_map_build")
    if isinstance(global_build, dict):
        build_time = global_build.get("time_sec")
        consumers = global_build.get("consumer_methods", [])
        if isinstance(build_time, (int, float)):
            for method in consumers:
                if method not in by_method:
                    continue
                trials = int(by_method[method]["num_trials"])
                amortized = None if trials <= 0 else float(build_time) / trials
                by_method[method]["scene_global_map_build_time_sec"] = float(
                    build_time
                )
                by_method[method][
                    "amortized_global_map_build_time_sec_per_trial"
                ] = amortized
                query = by_method[method].get(
                    "mean_computation_time_sec_all_trials"
                )
                by_method[method][
                    "mean_computation_plus_amortized_build_time_sec"
                ] = (
                    None
                    if query is None or amortized is None
                    else float(query) + amortized
                )
    esdf_load = setup.get("nvblox_esdf_load")
    if isinstance(esdf_load, dict):
        load_time = esdf_load.get("time_sec_excluded_from_all_method_timings")
        build_time = esdf_load.get("map_build_time_sec")
        consumers = esdf_load.get("consumer_methods", [])
        for method in consumers:
            if method not in by_method:
                continue
            trials = int(by_method[method]["num_trials"])
            if isinstance(load_time, (int, float)):
                amortized = None if trials <= 0 else float(load_time) / trials
                by_method[method]["scene_esdf_load_time_sec"] = float(load_time)
                by_method[method][
                    "amortized_esdf_load_time_sec_per_trial"
                ] = amortized
            if isinstance(build_time, (int, float)):
                amortized = None if trials <= 0 else float(build_time) / trials
                query = by_method[method].get(
                    "mean_computation_time_sec_all_trials"
                )
                by_method[method]["scene_global_map_build_time_sec"] = float(
                    build_time
                )
                by_method[method][
                    "amortized_global_map_build_time_sec_per_trial"
                ] = amortized
                by_method[method][
                    "mean_computation_plus_amortized_build_time_sec"
                ] = (
                    None
                    if query is None or amortized is None
                    else float(query) + amortized
                )
    memory_policy = setup.get("memory_policy")
    shared_inputs = setup.get("shared_scene_inputs")
    representations = (
        memory_policy.get("representation_array_bytes_by_method", {})
        if isinstance(memory_policy, dict)
        else {}
    )
    shared_bytes = (
        shared_inputs.get("total_tracked_numpy_array_bytes")
        if isinstance(shared_inputs, dict)
        else None
    )
    for method, method_summary in by_method.items():
        end_to_end = method_summary.get(
            "mean_computation_plus_amortized_build_time_sec"
        )
        if end_to_end is None:
            end_to_end = method_summary.get("mean_computation_time_sec_all_trials")
        method_summary[
            "mean_end_to_end_time_sec_including_amortized_scene_build"
        ] = end_to_end
        if isinstance(shared_bytes, (int, float)):
            method_summary["shared_scene_input_array_bytes"] = int(shared_bytes)
        representation_bytes = representations.get(method)
        if isinstance(representation_bytes, (int, float)):
            method_summary["method_representation_array_bytes"] = int(
                representation_bytes
            )
        setup_resources = setup.get("resource_usage_by_setup_stage", {})
        method_resources = (
            {
                name: usage
                for name, usage in setup_resources.items()
                if name == "shared_input_load"
                or (name == "checkpoint_load" and method not in ESDF_METHOD_LABELS)
            }
            if isinstance(setup_resources, dict)
            else {}
        )
        if (
            isinstance(global_build, dict)
            and method in global_build.get("consumer_methods", [])
            and isinstance(setup_resources, dict)
            and "global_map_build" in setup_resources
        ):
            method_resources["global_map_build"] = setup_resources[
                "global_map_build"
            ]
        if (
            isinstance(esdf_load, dict)
            and method in esdf_load.get("consumer_methods", [])
            and isinstance(setup_resources, dict)
            and "nvblox_esdf_load" in setup_resources
        ):
            method_resources["nvblox_esdf_load"] = setup_resources[
                "nvblox_esdf_load"
            ]
        method_resources.update(
            {
                f"trial_{record.trial_id}": record.resource_usage
                for record in values
                if canonical_method_label(record.method) == method
            }
        )
        method_summary["resource_usage_summary_all_stages"] = (
            summarize_resource_usages(method_resources)
        )
    payload = {
        "format": "gspar_offline_path_evaluation_summary_v12",
        "path_quality_population": (
            "strict-success paths per unknown policy; detour and curvature means "
            "exclude near-straight paths"
        ),
        "nonstraight_path_definition": "detour_ratio > 1.0001",
        "scene_setup": setup,
        "by_method": by_method,
        "paired_outcomes_by_esdf_method": paired_outcomes_by_esdf_method(values),
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


METHODS = (
    GSPAR_GLOBAL_ASTAR_EGO,
    GSPAR_DIRECT,
    NVBLOX_ESDF_ASTAR_STOMP,
)
DEFAULT_METHODS = (
    GSPAR_GLOBAL_ASTAR_EGO,
    GSPAR_DIRECT,
    NVBLOX_ESDF_ASTAR_STOMP,
)
GLOBAL_MAP_METHODS = {
    GSPAR_GLOBAL_ASTAR_EGO,
}
ESDF_METHODS = {
    NVBLOX_ESDF_ASTAR_STOMP,
}
CHECKPOINT_METHODS = set(METHODS) - ESDF_METHODS


def _unique_numpy_array_bytes(values) -> int:
    seen: set[int] = set()
    total = 0
    for value in values:
        if not isinstance(value, np.ndarray) or id(value) in seen:
            continue
        seen.add(id(value))
        total += int(value.nbytes)
    return total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="NerfICG final.pt checkpoint")
    parser.add_argument(
        "--reference-map",
        required=True,
        type=Path,
        help="nvblox occupancy reference",
    )
    parser.add_argument("--endpoints", required=True, type=Path, help="Shared endpoint manifest JSON")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--num-trials",
        type=int,
        default=100,
        help="Number of shared endpoint pairs to evaluate (default: 100)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(DEFAULT_METHODS),
        help=(
            "GSPAR and nvblox ESDF method IDs; the default routes use "
            "GSPAR A*+EGO and nvblox A*+STOMP"
        ),
    )
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
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--esdf-map",
        type=Path,
        default=None,
        help="nvblox ESDF PLY, sdf directory, or scene directory; inferred from --reference-map by default",
    )
    parser.add_argument(
        "--esdf-grid-metadata",
        type=Path,
        default=None,
        help="Grid metadata for ESDF alignment; defaults to --reference-map",
    )
    parser.add_argument(
        "--esdf-unknown-policy",
        choices=[UNKNOWN_POLICY_FREE, UNKNOWN_POLICY_OCCUPIED],
        default=UNKNOWN_POLICY_FREE,
    )
    parser.add_argument("--esdf-unknown-clearance-m", type=float, default=None)
    parser.add_argument(
        "--rrt-base-seed",
        type=int,
        default=0,
        help="Base seed used to derive a reproducible GSPAR-Direct RRT seed per trial",
    )
    parser.add_argument(
        "--timing-only",
        action="store_true",
        help="Skip observed-TSDF path validation metrics",
    )
    return parser


def select_endpoint_trials(
    endpoint_pairs: list[dict[str, Any]],
    num_trials: int,
) -> list[dict[str, Any]]:
    requested = int(num_trials)
    if requested <= 0:
        raise ValueError("num_trials must be positive")
    if len(endpoint_pairs) < requested:
        raise ValueError(
            f"endpoint manifest contains only {len(endpoint_pairs)} pairs, but {requested} trials "
            "were requested; regenerate endpoints with --num-pairs "
            f"{requested} or lower --num-trials explicitly"
        )
    return endpoint_pairs[:requested]


def rrt_seed_for_trial(base_seed: int, trial_id: str | int | None) -> int:
    """Derive a stable uint32 seed while retaining the configured base seed."""

    payload = f"{max(0, int(base_seed))}:{trial_id!r}".encode("utf-8")
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "little")


def planning_map_payload(
    esdf_path: Path | None,
    voxel_path: Path,
) -> dict[str, Any]:
    """Report the on-disk map files carried into path-planning comparisons."""

    resolved_voxel = voxel_path.expanduser().resolve()
    esdf_bytes = (
        esdf_path.stat().st_size
        if esdf_path is not None and esdf_path.is_file()
        else None
    )
    voxel_bytes = resolved_voxel.stat().st_size if resolved_voxel.is_file() else None
    return {
        "esdf_ply": None if esdf_path is None else str(esdf_path),
        "voxel_npz": str(resolved_voxel),
        "esdf_ply_bytes": esdf_bytes,
        "voxel_npz_bytes": voxel_bytes,
        "combined_bytes": (
            None
            if esdf_bytes is None or voxel_bytes is None
            else int(esdf_bytes + voxel_bytes)
        ),
    }


def _run_single_evaluation(args: argparse.Namespace) -> int:
    args.methods = list(dict.fromkeys(args.methods))
    with ResourceMonitor(cuda_device=args.device) as shared_input_resource_monitor:
        endpoints = select_endpoint_trials(
            load_endpoint_pairs(args.endpoints),
            args.num_trials,
        )
        reference = VoxelReference.from_map(args.reference_map)

    esdf_consumers = [method for method in args.methods if method in ESDF_METHODS]
    esdf_graph = None
    esdf_path = None
    esdf_grid_path = None
    esdf_load_setup = None
    if esdf_consumers:
        esdf_source = args.esdf_map or args.reference_map.expanduser().resolve().parent
        esdf_path = resolve_esdf_path(esdf_source)
        if args.esdf_grid_metadata is not None:
            esdf_grid_path = args.esdf_grid_metadata.expanduser().resolve()
        else:
            reference_path = args.reference_map.expanduser().resolve()
            tsdf_grid_path = reference_path.parent / "voxel_map_tsdf.npz"
            esdf_grid_path = (
                tsdf_grid_path
                if reference_path.stem == "voxel_map_gt" and tsdf_grid_path.is_file()
                else reference_path
            )
        with ResourceMonitor(monitor_cuda=False) as esdf_resource_monitor:
            esdf_load_started = time.perf_counter()
            esdf_graph = load_esdf_graph(
                esdf_path,
                esdf_grid_path,
                use_occupancy_guard=True,
                cache_aligned=True,
                unknown_policy=args.esdf_unknown_policy,
                unknown_clearance_m=args.esdf_unknown_clearance_m,
            )
            esdf_load_time = float(time.perf_counter() - esdf_load_started)
        esdf_load_setup = {
            "path": str(esdf_path),
            "grid_metadata": str(esdf_grid_path),
            "time_sec_excluded_from_all_method_timings": esdf_load_time,
            "map_build_time_sec": _nvblox_map_build_time(esdf_path),
            "consumer_methods": [METHOD_LABEL_BY_ID[method] for method in esdf_consumers],
            "unknown_policy": str(args.esdf_unknown_policy),
            "unknown_clearance_m": float(esdf_graph.unknown_clearance_m),
            "voxel_size_m": float(esdf_graph.voxel_size),
            "known_esdf_voxels": int(esdf_graph.esdf_idx.shape[0]),
            "payload_array_bytes": _unique_numpy_array_bytes(vars(esdf_graph).values()),
            "resource_usage": esdf_resource_monitor.to_dict(),
        }

    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint_consumers = [
        method for method in args.methods if method in CHECKPOINT_METHODS
    ]
    gaussian_set = None
    gaussian_array_bytes = 0
    checkpoint_load_setup = None
    if checkpoint_consumers:
        with ResourceMonitor(cuda_device=args.device) as checkpoint_resource_monitor:
            checkpoint_load_started = time.perf_counter()
            gaussian_set = preprocess_gaussians(
                checkpoint_path=checkpoint_path,
                max_gaussians=None,
                device=args.device,
            )
            gaussian_set = transform_gaussians(
                gaussian_set,
                reference.map_from_training_world,
            )
            checkpoint_load_time = float(time.perf_counter() - checkpoint_load_started)
        gaussian_array_bytes = int(
            gaussian_set.positions.nbytes
            + gaussian_set.scales.nbytes
            + gaussian_set.rotations.nbytes
            + gaussian_set.opacities.nbytes
        )
        checkpoint_load_setup = {
            "path": str(checkpoint_path),
            "time_sec_excluded_from_all_method_timings": checkpoint_load_time,
            "num_gaussians": int(gaussian_set.positions.shape[0]),
            "shared_gaussian_array_bytes": gaussian_array_bytes,
            "shared_once_per_scene": True,
            "consumer_methods": [
                METHOD_LABEL_BY_ID[method] for method in checkpoint_consumers
            ],
            "resource_usage": checkpoint_resource_monitor.to_dict(),
        }
    reference_array_bytes = _unique_numpy_array_bytes(vars(reference).values())
    endpoint_array_bytes = _unique_numpy_array_bytes(
        value
        for pair in endpoints
        for value in (pair.get("start_world"), pair.get("goal_world"))
    )
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass

    global_consumers = [
        method for method in args.methods if method in GLOBAL_MAP_METHODS
    ]
    payload = None
    global_build_setup = None
    if global_consumers:
        if gaussian_set is None:
            raise RuntimeError("GSPAR global methods require a loaded checkpoint")
        reference_grid = {
            "origin": reference.origin.copy(),
            "scene_min": reference.origin.copy(),
            "scene_max": reference.world_max.copy(),
            "voxel_size": float(reference.voxel_size),
            "voxel_dims": reference.voxel_dims.copy(),
        }
        base_config = SparseVoxelConfig()
        build_config = replace(
            base_config,
            voxel_size=float(reference.voxel_size),
            compute_device=str(args.device or base_config.compute_device),
            reference_occupancy=None,
            voxel_size_json=None,
        )
        with ResourceMonitor(
            cuda_device=str(build_config.compute_device)
        ) as build_resource_monitor:
            build_started = time.perf_counter()
            payload, build_stats = build_sparse_voxel_payload(
                gaussian_set,
                config=build_config,
                reference_grid=reference_grid,
                reset_cuda_peak_memory_stats=False,
            )
            global_build_time = float(time.perf_counter() - build_started)
        payload_array_bytes = int(
            sum(
                value.nbytes
                for value in payload.values()
                if isinstance(value, np.ndarray)
            )
        )
        global_build_setup = {
            "mode": "runtime_in_memory_no_save",
            "time_sec": global_build_time,
            "amortized_time_sec_per_endpoint_pair": (
                None if not endpoints else global_build_time / len(endpoints)
            ),
            "consumer_methods": [METHOD_LABEL_BY_ID[method] for method in global_consumers],
            "payload_array_bytes": payload_array_bytes,
            "resource_usage": build_resource_monitor.to_dict(),
            "stats": build_stats,
            "config": {
                "scene_type": str(build_config.scene_type),
                "voxel_size": float(reference.voxel_size),
                "compute_device": str(build_config.compute_device),
                "geometry_filter": bool(build_config.geometry_filter),
                "occupied_contribution_threshold": float(
                    build_config.occupied_contribution_threshold
                ),
            },
        }
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass

    representation_array_bytes_by_method = {
        METHOD_LABEL_BY_ID[method]: (
            int(global_build_setup["payload_array_bytes"])
            if method in GLOBAL_MAP_METHODS and global_build_setup is not None
            else int(esdf_load_setup["payload_array_bytes"])
            if method in ESDF_METHODS and esdf_load_setup is not None
            else gaussian_array_bytes
        )
        for method in args.methods
    }
    scene_setup = {
        "checkpoint_load": checkpoint_load_setup,
        "global_map_build": global_build_setup,
        "nvblox_esdf_load": esdf_load_setup,
        "planning_map_payload": planning_map_payload(
            esdf_path,
            args.reference_map.expanduser().resolve().parent / "voxel_map.npz",
        ),
        "shared_scene_inputs": {
            "gaussian_numpy_array_bytes": gaussian_array_bytes,
            "reference_tracked_numpy_array_bytes": reference_array_bytes,
            "endpoint_numpy_array_bytes": endpoint_array_bytes,
            "total_tracked_numpy_array_bytes": (
                gaussian_array_bytes + reference_array_bytes + endpoint_array_bytes
            ),
            "load_resource_usage": shared_input_resource_monitor.to_dict(),
        },
        "timing_policy": {
            "checkpoint_load_excluded": True,
            "output_serialization_excluded": True,
            "global_map_built_at_runtime": bool(global_consumers),
            "global_map_saved": False,
            "esdf_load_excluded": bool(esdf_consumers),
        },
        "memory_policy": {
            "cpu": "sampled process RSS absolute and incremental peaks",
            "cuda": (
                "PyTorch allocated/reserved absolute and incremental peaks; "
                "driver and non-PyTorch allocations are excluded"
            ),
            "representation_array_bytes_by_method": representation_array_bytes_by_method,
        },
    }
    setup_resource_usages = {
        "shared_input_load": shared_input_resource_monitor.to_dict(),
    }
    if checkpoint_load_setup is not None:
        setup_resource_usages["checkpoint_load"] = checkpoint_load_setup[
            "resource_usage"
        ]
    if global_build_setup is not None:
        setup_resource_usages["global_map_build"] = global_build_setup[
            "resource_usage"
        ]
    if esdf_load_setup is not None:
        setup_resource_usages["nvblox_esdf_load"] = esdf_load_setup[
            "resource_usage"
        ]
    scene_setup["resource_usage_summary"] = summarize_resource_usages(
        setup_resource_usages
    )
    scene_setup["resource_usage_by_setup_stage"] = setup_resource_usages
    occupancy_voxel_size_m = (
        float(np.asarray(payload["voxel_size"]).reshape(-1)[0])
        if payload is not None
        else float(reference.voxel_size)
    )
    safety_clearance_voxels = float(args.robot_radius_m) / max(
        occupancy_voxel_size_m,
        1e-12,
    )
    if esdf_graph is not None and not np.isclose(
        float(esdf_graph.voxel_size),
        occupancy_voxel_size_m,
        rtol=1e-6,
        atol=1e-9,
    ):
        raise ValueError(
            "fair comparison requires matching occupancy/ESDF voxel resolution: "
            f"{occupancy_voxel_size_m:.9f} m vs {float(esdf_graph.voxel_size):.9f} m"
        )
    shared_connectivity = 26
    shared_astar_max_expansions = 500_000
    shared_astar_heuristic_weight = 1.0
    shared_snap_search_radius = 8
    shared_max_planning_time_sec = 10.0
    shared_control_point_spacing_m = 0.25
    shared_output_spacing_m = 0.05
    shared_collision_sample_spacing_m = 0.05
    astar = GlobalPlanOptions(
        method=GLOBAL_WEIGHTED_ASTAR,
        connectivity=shared_connectivity,
        max_expansions=shared_astar_max_expansions,
        snap_search_radius=shared_snap_search_radius,
        safety_clearance_voxels=safety_clearance_voxels,
        heuristic_weight=shared_astar_heuristic_weight,
        max_planning_time_sec=shared_max_planning_time_sec,
        treat_unknown_as_occupied=args.esdf_unknown_policy == UNKNOWN_POLICY_OCCUPIED,
    )
    ego = EgoPlanOptions(
        safety_clearance_m=float(args.robot_radius_m),
        safety_clearance_voxels=safety_clearance_voxels,
        inherit_input_clearance=False,
        control_point_spacing_m=shared_control_point_spacing_m,
        output_spacing_m=shared_output_spacing_m,
        collision_sample_spacing_m=shared_collision_sample_spacing_m,
        treat_unknown_as_occupied=args.esdf_unknown_policy == UNKNOWN_POLICY_OCCUPIED,
    )
    esdf_astar = EsdfGlobalPlanOptions(
        method=GLOBAL_ESDF_WEIGHTED_ASTAR,
        connectivity=shared_connectivity,
        max_expansions=shared_astar_max_expansions,
        snap_search_radius=shared_snap_search_radius,
        safety_clearance_m=float(args.robot_radius_m),
        clearance_cost_weight=0.0,
        heuristic_weight=shared_astar_heuristic_weight,
        max_planning_time_sec=shared_max_planning_time_sec,
    )
    esdf_stomp = StompEsdfPlanOptions(
        safety_clearance_m=float(args.robot_radius_m),
        control_point_spacing_m=shared_control_point_spacing_m,
        output_spacing_m=shared_output_spacing_m,
        collision_sample_spacing_m=shared_collision_sample_spacing_m,
        random_seed=7,
    )
    gspar_direct = GSPARDirectPlanOptions(
        robot_radius=float(args.robot_radius_m),
        validation_spacing_m=shared_collision_sample_spacing_m,
        device=args.device,
    )
    physical_clearances_m = {
        "GSPAR-Global": float(astar.safety_clearance_voxels)
        * occupancy_voxel_size_m,
        "GSPAR-EGO-meters": float(ego.safety_clearance_m),
        "GSPAR-EGO-voxels": float(ego.safety_clearance_voxels)
        * occupancy_voxel_size_m,
        "nvblox-ESDF-Global": float(esdf_astar.safety_clearance_m),
        "nvblox-ESDF-STOMP": float(esdf_stomp.safety_clearance_m),
    }
    if not all(
        np.isclose(
            clearance,
            float(args.robot_radius_m),
            rtol=1.0e-6,
            atol=1.0e-9,
        )
        for clearance in physical_clearances_m.values()
    ):
        raise ValueError(
            "planner safety-clearance mismatch: "
            + ", ".join(
                f"{name}={clearance:.9f}m"
                for name, clearance in physical_clearances_m.items()
            )
        )
    reference_metrics = not bool(args.timing_only)
    metric_options = {
        "compute_reference_metrics": reference_metrics,
        "reference_sample_spacing_m": shared_collision_sample_spacing_m,
        "smoothness_sample_spacing_m": shared_output_spacing_m,
    }
    map_label = "GSPAR-Global runtime in-memory Gaussian occupancy map"
    esdf_map_label = "" if esdf_path is None else str(esdf_path)
    planner_fns = {}

    def occupancy_fn(name: str, options: GlobalPlanOptions, ego_options: EgoPlanOptions | None):
        if payload is None:
            raise RuntimeError("global occupancy payload was not built")
        return lambda trial_id, start, goal: evaluate_occupancy_trial(
            payload=payload,
            start_world=start,
            goal_world=goal,
            global_options=options,
            ego_options=ego_options,
            reference=reference,
            robot_radius=float(args.robot_radius_m),
            map_label=map_label,
            method_name=name,
            trial_id=trial_id,
            **metric_options,
        )

    def esdf_fn(
        name: str,
        options: EsdfGlobalPlanOptions,
        stomp_options: StompEsdfPlanOptions | None = None,
    ):
        if esdf_graph is None:
            raise RuntimeError("nvblox ESDF graph was not loaded")
        return lambda trial_id, start, goal: evaluate_esdf_trial(
            graph=esdf_graph,
            start_world=start,
            goal_world=goal,
            global_options=options,
            stomp_options=stomp_options,
            reference=reference,
            robot_radius=float(args.robot_radius_m),
            map_label=esdf_map_label,
            method_name=name,
            trial_id=trial_id,
            **metric_options,
        )

    if GSPAR_GLOBAL_ASTAR_EGO in args.methods:
        label = METHOD_LABEL_BY_ID[GSPAR_GLOBAL_ASTAR_EGO]
        planner_fns[label] = occupancy_fn(label, astar, ego)
    if GSPAR_DIRECT in args.methods:
        if gaussian_set is None:
            raise RuntimeError("GSPAR Direct requires a loaded checkpoint")
        label = METHOD_LABEL_BY_ID[GSPAR_DIRECT]
        planner_fns[label] = lambda trial_id, start, goal: evaluate_gspar_direct_trial(
            raw_gaussian_map=gaussian_set,
            start_world=start,
            goal_world=goal,
            options=replace(
                gspar_direct,
                rrt_random_seed=rrt_seed_for_trial(args.rrt_base_seed, trial_id),
            ),
            reference=reference,
            robot_radius=float(args.robot_radius_m),
            trial_id=trial_id,
            **metric_options,
        )
    if NVBLOX_ESDF_ASTAR_STOMP in args.methods:
        label = METHOD_LABEL_BY_ID[NVBLOX_ESDF_ASTAR_STOMP]
        planner_fns[label] = esdf_fn(label, esdf_astar, esdf_stomp)
    records = run_evaluation_trials(planner_fns=planner_fns, endpoint_pairs=endpoints)
    output_dir = args.output_dir.expanduser().resolve()
    records_path = write_jsonl(output_dir / "records.jsonl", records)
    summary_path = write_summary(
        output_dir / "summary.json",
        records,
        scene_setup=scene_setup,
    )
    run_config = {
        "format": "gspar_offline_path_evaluation_run_v12",
        "checkpoint": str(checkpoint_path),
        "occupancy_map": map_label,
        "scene_setup": scene_setup,
        "reference_map": str(args.reference_map.expanduser().resolve()),
        "reference_type": "observed_tsdf_free_occupied_unknown",
        "unknown_policies": ["unknown_as_free", "unknown_as_occupied"],
        "strict_success_definition": (
            "native_success AND global_planner_success AND "
            "(local_planner_success OR local_stage_not_applicable) AND "
            "NOT fallback_used AND start_reached AND goal_reached AND reference_safe"
        ),
        "task_success_definition": (
            "alias of strict_success_definition; retained for backward "
            "compatibility"
        ),
        "detour_ratio_definition": (
            "path_length / distance(path_start, path_goal)"
        ),
        "task_detour_ratio_definition": (
            "(requested_start_to_path_start_gap + path_length + "
            "path_goal_to_requested_goal_gap) / "
            "distance(requested_start, requested_goal)"
        ),
        "primary_success_metrics": [
            "strict_success_unknown_as_free",
            "strict_success_unknown_as_occupied",
        ],
        "path_quality_population": (
            "strict-success paths under the corresponding unknown-space policy; "
            "detour and curvature means additionally exclude near-straight paths"
        ),
        "nonstraight_path_definition": "detour_ratio > 1.0001",
        "planning_unknown_semantics": (
            "unknown_as_free"
            if args.esdf_unknown_policy == UNKNOWN_POLICY_FREE
            else "unknown_as_occupied"
            if args.esdf_unknown_policy == UNKNOWN_POLICY_OCCUPIED
            else "unknown_with_proxy_clearance"
        ),
        "endpoints": str(args.endpoints.expanduser().resolve()),
        "num_trials": len(endpoints),
        "method_ids": list(args.methods),
        "methods": [METHOD_LABEL_BY_ID[method] for method in args.methods],
        "robot_radius_m": float(args.robot_radius_m),
        "physical_safety_clearance_m": float(args.robot_radius_m),
        "occupancy_safety_clearance_voxels": safety_clearance_voxels,
        "physical_clearance_by_planner_component_m": physical_clearances_m,
        "safety_clearance_contract": {
            "occupancy_global_and_ego": (
                "nearest blocked voxel-center distance >= robot_radius_m / voxel_size"
            ),
            "nvblox_esdf_global_and_local": (
                "ESDF distance in meters >= robot_radius_m"
            ),
            "reference_evaluation": (
                "distance to occupied voxel cubes > robot_radius_m + reference margin"
            ),
            "gspar_direct_exception": (
                "Gaussian ellipsoid collision model uses robot_radius_m but has "
                "method-specific geometry semantics"
            ),
        },
        "output_spacing_m": shared_output_spacing_m,
        "collision_sample_spacing_m": shared_collision_sample_spacing_m,
        "shared_global_parameters": {
            "connectivity": shared_connectivity,
            "astar_max_expansions": shared_astar_max_expansions,
            "astar_heuristic_weight": shared_astar_heuristic_weight,
            "snap_search_radius": shared_snap_search_radius,
            "max_planning_time_sec": shared_max_planning_time_sec,
        },
        "shared_local_parameters": {
            "control_point_spacing_m": shared_control_point_spacing_m,
            "output_spacing_m": shared_output_spacing_m,
            "collision_sample_spacing_m": shared_collision_sample_spacing_m,
        },
        "planner_options": {
            "gspar_global_astar": asdict(astar),
            "gspar_ego": asdict(ego),
            "gspar_direct": asdict(gspar_direct),
            "nvblox_esdf_astar": asdict(esdf_astar),
            "nvblox_esdf_stomp": asdict(esdf_stomp),
        },
        "radius_profile": {
            "role": str(getattr(args, "radius_profile_role", "single")),
            "all_radii_m": list(
                getattr(args, "all_robot_radius_profiles_m", [args.robot_radius_m])
            ),
            "endpoint_manifest_robot_radius_m": float(
                getattr(args, "endpoint_manifest_robot_radius_m", args.robot_radius_m)
            ),
            "output_root": str(
                getattr(args, "radius_sweep_output_root", output_dir)
            ),
        },
        "nvblox_esdf_map": None if esdf_path is None else str(esdf_path),
        "nvblox_esdf_grid_metadata": (
            None if esdf_grid_path is None else str(esdf_grid_path)
        ),
        "nvblox_esdf_unknown_policy": str(args.esdf_unknown_policy),
        "stomp_parameters": asdict(esdf_stomp),
        "rrt_seed": {
            "base_seed": max(0, int(args.rrt_base_seed)),
            "strategy": "blake2s_uint32(base_seed, trial_id)",
            "record_field": "extra.rrt_random_seed",
        },
        "reference_metrics": reference_metrics,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"records: {records_path}")
    print(f"summary: {summary_path}")
    return 0


def _write_radius_sweep_summary(
    output_root: Path,
    *,
    endpoint_manifest: Path,
    endpoint_radius_m: float,
    profiles: list[dict[str, Any]],
) -> Path:
    output = output_root / "radius_sweep_summary.json"
    payload = {
        "format": "gspar_robot_radius_sweep_v1",
        "endpoint_manifest": str(endpoint_manifest.expanduser().resolve()),
        "endpoint_manifest_robot_radius_m": float(endpoint_radius_m),
        "shared_endpoint_trials": True,
        "profiles": profiles,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        radii = normalize_robot_radius_profiles(
            args.robot_radius_m,
            args.additional_robot_radii_m,
        )
        endpoint_radius = validate_endpoint_manifest_radius(args.endpoints, max(radii))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    output_root = args.output_dir.expanduser().resolve()
    profile_summaries = []
    for index, radius in enumerate(radii):
        profile_values = vars(args).copy()
        profile_output = (
            output_root
            if index == 0
            else output_root / "radius_profiles" / radius_profile_directory_name(radius)
        )
        profile_values.update(
            {
                "robot_radius_m": float(radius),
                "additional_robot_radii_m": [],
                "output_dir": profile_output,
                "radius_profile_role": "baseline" if index == 0 else "additional",
                "all_robot_radius_profiles_m": list(radii),
                "endpoint_manifest_robot_radius_m": float(endpoint_radius),
                "radius_sweep_output_root": output_root,
            }
        )
        profile_args = argparse.Namespace(**profile_values)
        _run_single_evaluation(profile_args)
        summary_path = profile_output / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        profile_summaries.append(
            {
                "robot_radius_m": float(radius),
                "role": profile_values["radius_profile_role"],
                "output_dir": str(profile_output),
                "records": str(profile_output / "records.jsonl"),
                "summary": str(summary_path),
                "run_config": str(profile_output / "run_config.json"),
                "by_method": summary.get("by_method", {}),
            }
        )

    sweep_path = _write_radius_sweep_summary(
        output_root,
        endpoint_manifest=args.endpoints,
        endpoint_radius_m=endpoint_radius,
        profiles=profile_summaries,
    )
    print(f"radius sweep: {sweep_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
