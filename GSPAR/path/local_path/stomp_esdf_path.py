#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Naive STOMP local path optimization on ESDF voxels.

The optimizer keeps the input global path as a reference, fits a clamped
B-spline control polygon to it, and runs basic STOMP rollouts over the free
interior control points.  Collision cost is evaluated by sampling the spline
and querying the project's ESDF grid representation.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_array

from GSPAR.path.global_path.esdf_astar_world_path import (
    UNKNOWN_POLICY_CLEARANCE,
    UNKNOWN_POLICY_FREE,
    UNKNOWN_POLICY_OCCUPIED,
    EsdfGridGraph,
    load_esdf_grid,
)
from GSPAR.path.local_path.ego_occupancy_path import (
    CoarsePath,
    _design_matrix,
    _format_filename_value,
    _free_indices,
    _validate_path_points,
    path_length,
    read_coarse_path,
    resample_polyline,
    sample_control_spline,
)


LOCAL_STOMP_METHOD = "STOMP (ESDF B-Spline)"


class EsdfClearanceQuery:
    """Vectorized ESDF clearance lookup for repeated trajectory scoring."""

    def __init__(self, graph: EsdfGridGraph) -> None:
        self.graph = graph
        self.origin = np.asarray(graph.origin, dtype=np.float64).reshape(3)
        self.voxel_size = max(float(graph.voxel_size), 1e-12)
        self.voxel_dims = np.asarray(graph.voxel_dims, dtype=np.int32).reshape(3)
        self._dims64 = self.voxel_dims.astype(np.int64)

        graph_esdf_keys = getattr(graph, "_esdf_keys", None)
        if graph_esdf_keys is None:
            esdf_keys = self._pack(np.asarray(graph.esdf_idx, dtype=np.int32).reshape(-1, 3))
            order = np.argsort(esdf_keys, kind="mergesort")
            self.esdf_keys = esdf_keys[order]
            self.esdf_distance_m = np.asarray(graph.esdf_distance_m, dtype=np.float32).reshape(-1)[order]
        else:
            self.esdf_keys = graph_esdf_keys
            self.esdf_distance_m = np.asarray(graph.esdf_distance_m, dtype=np.float32).reshape(-1)
        self.occupied_keys = getattr(graph, "_occupied_keys", np.zeros((0,), dtype=np.int64))

    def _pack(self, idx: np.ndarray) -> np.ndarray:
        idx64 = np.asarray(idx, dtype=np.int64).reshape(-1, 3)
        return idx64[:, 0] + self._dims64[0] * (idx64[:, 1] + self._dims64[1] * idx64[:, 2])

    def clearance_m_for_points(self, points_world: np.ndarray) -> np.ndarray:
        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        clearances = np.full((points.shape[0],), -np.inf, dtype=np.float32)
        if points.shape[0] == 0:
            return clearances

        idx = np.floor((points - self.origin[None, :]) / self.voxel_size).astype(np.int32)
        valid = np.logical_and(idx >= 0, idx < self.voxel_dims[None, :]).all(axis=1)
        if not valid.any():
            return clearances

        valid_rows = np.flatnonzero(valid)
        keys = self._pack(idx[valid])
        unknown_policy = getattr(self.graph, "unknown_policy", UNKNOWN_POLICY_CLEARANCE)
        if unknown_policy == UNKNOWN_POLICY_CLEARANCE:
            clearances[valid_rows] = float(self.graph.unknown_clearance_m)
        elif unknown_policy == UNKNOWN_POLICY_FREE:
            clearances[valid_rows] = float("inf")

        esdf_pos = np.searchsorted(self.esdf_keys, keys)
        has_esdf = esdf_pos < self.esdf_keys.shape[0]
        if has_esdf.any():
            has_esdf[has_esdf] = self.esdf_keys[esdf_pos[has_esdf]] == keys[has_esdf]
        if has_esdf.any():
            clearances[valid_rows[has_esdf]] = self.esdf_distance_m[esdf_pos[has_esdf]]

        if self.occupied_keys.shape[0]:
            occ_pos = np.searchsorted(self.occupied_keys, keys)
            occupied = occ_pos < self.occupied_keys.shape[0]
            if occupied.any():
                occupied[occupied] = self.occupied_keys[occ_pos[occupied]] == keys[occupied]
            if occupied.any():
                clearances[valid_rows[occupied]] = -np.inf
        return clearances


def _get_esdf_query(graph: EsdfGridGraph) -> EsdfClearanceQuery:
    query = getattr(graph, "_stomp_clearance_query", None)
    if query is None:
        query = EsdfClearanceQuery(graph)
        setattr(graph, "_stomp_clearance_query", query)
    return query


@dataclass
class StompPlannerConfig:
    """Configuration for the naive ESDF/STOMP local planner."""

    control_point_spacing_m: float = 0.10
    output_spacing_m: float = 0.05
    collision_sample_spacing_m: float = 0.05
    safety_clearance_m: float | None = None
    default_safety_clearance_m: float = 0.50
    inherit_input_clearance: bool = True
    pin_endpoint_control_points: int = 2
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
    random_seed: int | None = 7
    stagnation_patience: int = 20
    min_improvement: float = 1.0e-5
    safe_fallback_to_input: bool = True
    safe_shortcut_fallback: bool = False
    safe_shortcut_max_checks: int = 50000
    safe_shortcut_min_length_improvement_m: float = 1.0e-3
    # Match the occupancy/JPS/EGO pipeline by simplifying the global grid path
    # before fitting STOMP's reference spline.
    shortcut_input: bool = True
    adaptive_control_point_spacing: bool = True
    adaptive_min_control_point_spacing_m: float = 0.05
    adaptive_max_control_points: int = 256
    no_change_position_tolerance_m: float = 1.0e-6
    no_change_length_tolerance_m: float = 1.0e-6

    def input_safety_clearance_m(self, coarse_path: CoarsePath | None = None) -> float | None:
        if coarse_path is not None:
            for key in ("safety_clearance_m", "obstacle_safety_margin_m", "clearance_m"):
                value = coarse_path.header.get(key)
                if value is None:
                    continue
                try:
                    return max(0.0, float(value.split()[0]))
                except ValueError:
                    continue
        return None

    def resolved_safety_clearance_m(self, coarse_path: CoarsePath | None = None) -> float:
        if self.safety_clearance_m is not None:
            return max(0.0, float(self.safety_clearance_m))
        input_clearance = self.input_safety_clearance_m(coarse_path)
        if self.inherit_input_clearance and input_clearance is not None:
            return input_clearance
        if input_clearance is not None:
            return min(input_clearance, max(0.0, float(self.default_safety_clearance_m)))
        return max(0.0, float(self.default_safety_clearance_m))


@dataclass
class StompPlannerResult:
    """Result returned by :func:`stomp_esdf_path`."""

    path_world: np.ndarray
    control_points_world: np.ndarray
    reference_control_points_world: np.ndarray
    safety_clearance_m: float
    iterations: int
    rollouts: int
    min_clearance_m: float
    unsafe_sample_count: int
    length_m: float
    success: bool
    message: str
    stats: dict[str, Any] = field(default_factory=dict)


def esdf_clearance_m_for_points(graph: EsdfGridGraph, points_world: np.ndarray) -> np.ndarray:
    """Return ESDF clearance in meters for each world point.

    Out-of-grid points and occupied-guard voxels are returned as ``-inf`` so
    they receive a hard collision penalty during optimization.
    """

    return _get_esdf_query(graph).clearance_m_for_points(points_world)


def _world_bounds(graph: EsdfGridGraph) -> tuple[np.ndarray, np.ndarray]:
    origin = np.asarray(graph.origin, dtype=np.float64).reshape(3)
    dims = np.asarray(graph.voxel_dims, dtype=np.float64).reshape(3)
    lo = origin + 0.5 * float(graph.voxel_size)
    hi = origin + (dims - 0.5) * float(graph.voxel_size)
    return lo, hi


def _make_reference_control_points(coarse: CoarsePath, spacing_m: float, voxel_size: float) -> np.ndarray:
    length = max(path_length(coarse.points), float(voxel_size))
    spacing = max(float(spacing_m), float(voxel_size), 1e-6)
    num_control = max(4, int(math.ceil(length / spacing)) + 1)
    fit_spacing = max(min(0.5 * spacing, float(voxel_size)), 1e-6)
    reference = resample_polyline(coarse.points, fit_spacing).astype(np.float64)
    sample_count = max(reference.shape[0], num_control + 2)
    if reference.shape[0] != sample_count:
        reference = resample_polyline(coarse.points, length / max(sample_count - 1, 1)).astype(np.float64)
    _, basis = _design_matrix(num_control, reference.shape[0])
    dense_basis = basis.toarray()
    q_ref = np.zeros((num_control, 3), dtype=np.float64)
    q_ref[0] = coarse.points[0]
    q_ref[-1] = coarse.points[-1]
    if num_control > 2:
        rhs = (
            reference
            - dense_basis[:, 0:1] * q_ref[0][None, :]
            - dense_basis[:, -1:] * q_ref[-1][None, :]
        )
        inner, *_ = np.linalg.lstsq(dense_basis[:, 1:-1], rhs, rcond=None)
        q_ref[1:-1] = inner
    return q_ref


def _control_spacing_candidates(
    coarse: CoarsePath,
    graph: EsdfGridGraph,
    cfg: StompPlannerConfig,
) -> list[float]:
    requested = max(float(cfg.control_point_spacing_m), float(graph.voxel_size), 1e-6)
    if not bool(cfg.adaptive_control_point_spacing):
        return [requested]

    length = max(path_length(coarse.points), float(graph.voxel_size))
    max_points = max(4, int(cfg.adaptive_max_control_points))
    min_spacing = max(
        float(cfg.adaptive_min_control_point_spacing_m),
        float(cfg.output_spacing_m),
        float(graph.voxel_size),
        length / float(max_points - 1),
        1e-6,
    )
    min_spacing = min(requested, min_spacing)
    candidates = [requested]
    spacing = requested
    while spacing > min_spacing * 1.001:
        spacing = max(min_spacing, 0.5 * spacing)
        if abs(spacing - candidates[-1]) > 1e-9:
            candidates.append(spacing)
        else:
            break
    return candidates


def _build_control_problem(
    graph: EsdfGridGraph,
    coarse: CoarsePath,
    cfg: StompPlannerConfig,
    spacing_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    q_ref = _make_reference_control_points(coarse, float(spacing_m), float(graph.voxel_size))
    q = _clip_control_points_to_world(graph, q_ref.copy())
    q[0] = coarse.points[0]
    q[-1] = coarse.points[-1]
    free = _free_indices(q.shape[0], int(cfg.pin_endpoint_control_points))
    collision_sample_count = max(
        2,
        int(
            math.ceil(
                max(path_length(q_ref), float(graph.voxel_size))
                / max(float(cfg.collision_sample_spacing_m), 1e-6)
            )
        )
        + 1,
    )
    _, collision_basis_sparse = _design_matrix(q.shape[0], collision_sample_count)
    return q_ref, q, free, collision_sample_count, collision_basis_sparse.toarray()


def _sample_polyline_by_count(points: np.ndarray, count: int) -> np.ndarray:
    path = _validate_path_points(points)
    sample_count = max(2, int(count))
    if path.shape[0] == 1:
        return np.repeat(path.astype(np.float64), sample_count, axis=0)

    seg_lengths = np.linalg.norm(np.diff(path.astype(np.float64), axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = float(cumulative[-1])
    if total <= 1e-12:
        return np.repeat(path[:1].astype(np.float64), sample_count, axis=0)

    targets = np.linspace(0.0, total, sample_count)
    seg_idx = np.searchsorted(cumulative, targets, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, path.shape[0] - 2)
    denom = np.maximum(seg_lengths[seg_idx], 1e-12)
    alpha = ((targets - cumulative[seg_idx]) / denom)[:, None]
    return (1.0 - alpha) * path[seg_idx] + alpha * path[seg_idx + 1]


def _path_change_stats(
    reference_path: np.ndarray,
    candidate_path: np.ndarray,
    *,
    sample_spacing_m: float,
    position_tolerance_m: float,
    length_tolerance_m: float,
) -> dict[str, Any]:
    reference = _validate_path_points(reference_path)
    candidate = _validate_path_points(candidate_path)
    reference_length = path_length(reference)
    candidate_length = path_length(candidate)
    spacing = max(float(sample_spacing_m), 1e-6)
    sample_count = max(2, int(math.ceil(max(reference_length, candidate_length) / spacing)) + 1)
    sample_count = min(sample_count, 4096)
    reference_samples = _sample_polyline_by_count(reference, sample_count)
    candidate_samples = _sample_polyline_by_count(candidate, sample_count)
    displacement = np.linalg.norm(candidate_samples - reference_samples, axis=1)
    max_displacement = float(np.max(displacement)) if displacement.shape[0] else 0.0
    mean_displacement = float(np.mean(displacement)) if displacement.shape[0] else 0.0
    length_delta = float(candidate_length - reference_length)
    changed = (
        max_displacement > float(position_tolerance_m)
        or abs(length_delta) > float(length_tolerance_m)
    )
    return {
        "changed_path": bool(changed),
        "input_length_m": float(reference_length),
        "output_length_delta_m": float(length_delta),
        "max_path_displacement_m": max_displacement,
        "mean_path_displacement_m": mean_displacement,
        "path_change_sample_count": int(sample_count),
    }


def _smoothness_cost(q: np.ndarray, *, smooth_weight: float, jerk_weight: float) -> float:
    cost = 0.0
    if q.shape[0] >= 3 and smooth_weight > 0.0:
        acc = q[2:] - 2.0 * q[1:-1] + q[:-2]
        cost += float(smooth_weight) * float(np.sum(acc ** 2))
    if q.shape[0] >= 4 and jerk_weight > 0.0:
        jerk = q[3:] - 3.0 * q[2:-1] + 3.0 * q[1:-2] - q[:-3]
        cost += float(jerk_weight) * float(np.sum(jerk ** 2))
    return cost


def _length_cost(samples: np.ndarray) -> float:
    if samples.shape[0] < 2:
        return 0.0
    seg = np.linalg.norm(np.diff(samples, axis=0), axis=1)
    return float(np.sum(seg))


def _trajectory_cost_batch(
    query: EsdfClearanceQuery,
    q_batch: np.ndarray,
    q_ref: np.ndarray,
    basis: np.ndarray,
    config: StompPlannerConfig,
    *,
    safety_clearance_m: float,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    q_arr = np.asarray(q_batch, dtype=np.float64)
    if q_arr.ndim == 2:
        q_arr = q_arr[None, :, :]
    samples = np.einsum("sc,rcd->rsd", basis, q_arr, optimize=True)
    flat_clearances = query.clearance_m_for_points(samples.reshape(-1, 3))
    clearances = flat_clearances.reshape(samples.shape[:2])
    finite_clearances = np.isfinite(clearances)
    deficit = np.maximum(float(safety_clearance_m) - clearances.astype(np.float64), 0.0)
    invalid = np.logical_or(np.isnan(clearances), np.isneginf(clearances))
    if invalid.any():
        deficit[invalid] = float(safety_clearance_m) + 2.0 * float(query.graph.voxel_size)
    # Do not average collision over the whole trajectory. A few unsafe samples
    # on a long path should still dominate smoothness/fitness, otherwise STOMP
    # accepts slightly colliding shortcuts and then falls back to the input path.
    collision = float(config.collision_weight) * np.sum(deficit ** 2, axis=1)

    clearance = np.zeros((q_arr.shape[0],), dtype=np.float64)
    if float(config.clearance_weight) > 0.0:
        margin = clearances.astype(np.float64) - float(safety_clearance_m)
        safe = np.logical_and(finite_clearances, margin > 0.0)
        inv_margin = np.zeros_like(margin, dtype=np.float64)
        inv_margin[safe] = 1.0 / np.maximum(margin[safe], 0.1 * float(query.graph.voxel_size))
        safe_count = np.maximum(np.count_nonzero(safe, axis=1), 1)
        clearance = float(config.clearance_weight) * np.sum(inv_margin, axis=1) / safe_count

    smooth = np.zeros((q_arr.shape[0],), dtype=np.float64)
    if q_arr.shape[1] >= 3 and float(config.smooth_weight) > 0.0:
        acc = q_arr[:, 2:] - 2.0 * q_arr[:, 1:-1] + q_arr[:, :-2]
        smooth += float(config.smooth_weight) * np.sum(acc ** 2, axis=(1, 2))
    if q_arr.shape[1] >= 4 and float(config.jerk_weight) > 0.0:
        jerk = q_arr[:, 3:] - 3.0 * q_arr[:, 2:-1] + 3.0 * q_arr[:, 1:-2] - q_arr[:, :-3]
        smooth += float(config.jerk_weight) * np.sum(jerk ** 2, axis=(1, 2))

    fitness = float(config.fitness_weight) * np.mean((q_arr - q_ref[None, :, :]) ** 2, axis=(1, 2))
    length_value = np.zeros((q_arr.shape[0],), dtype=np.float64)
    if float(config.length_weight) > 0.0 and samples.shape[1] >= 2:
        seg = np.linalg.norm(np.diff(samples, axis=1), axis=2)
        length_value = float(config.length_weight) * np.sum(seg, axis=1)
    total = collision + clearance + smooth + fitness + length_value
    parts = [
        {
            "collision": float(collision[i]),
            "clearance": float(clearance[i]),
            "smooth": float(smooth[i]),
            "fitness": float(fitness[i]),
            "length": float(length_value[i]),
            "min_clearance_m": float(np.min(clearances[i])) if clearances.shape[1] else float("inf"),
            "unsafe_sample_count": int(np.count_nonzero(clearances[i] < float(safety_clearance_m))),
        }
        for i in range(q_arr.shape[0])
    ]
    return total.astype(np.float64), parts


def _trajectory_cost(
    query: EsdfClearanceQuery,
    q: np.ndarray,
    q_ref: np.ndarray,
    basis: np.ndarray,
    config: StompPlannerConfig,
    *,
    safety_clearance_m: float,
) -> tuple[float, dict[str, float]]:
    costs, parts = _trajectory_cost_batch(
        query,
        np.asarray(q, dtype=np.float64).reshape(1, -1, 3),
        q_ref,
        basis,
        config,
        safety_clearance_m=safety_clearance_m,
    )
    return float(costs[0]), parts[0]


def _clip_control_points_to_world(graph: EsdfGridGraph, q: np.ndarray) -> np.ndarray:
    lo, hi = _world_bounds(graph)
    return np.clip(q, lo[None, :], hi[None, :])


def _sample_noise(
    rng: np.random.Generator,
    shape: tuple[int, int],
    free: np.ndarray,
    noise_std: float,
) -> np.ndarray:
    noise = np.zeros(shape, dtype=np.float64)
    if free.shape[0] > 0 and noise_std > 0.0:
        noise[free] = rng.normal(0.0, float(noise_std), size=(free.shape[0], 3))
    return noise


def validate_stomp_path(
    graph: EsdfGridGraph,
    path_world: np.ndarray,
    *,
    safety_clearance_m: float,
    sample_spacing_m: float,
) -> dict[str, Any]:
    dense = resample_polyline(path_world, max(float(sample_spacing_m), 1e-6))
    clearances = esdf_clearance_m_for_points(graph, dense)
    unsafe_mask = clearances < float(safety_clearance_m)
    return {
        "points": int(np.asarray(path_world).reshape(-1, 3).shape[0]),
        "dense_samples": int(dense.shape[0]),
        "length_m": path_length(path_world),
        "min_clearance_m": float(np.min(clearances)) if clearances.shape[0] else float("inf"),
        "unsafe_sample_count": int(np.count_nonzero(unsafe_mask)),
        "safety_clearance_m": float(safety_clearance_m),
    }


def _line_segment_is_safe(
    graph: EsdfGridGraph,
    a: np.ndarray,
    b: np.ndarray,
    *,
    safety_clearance_m: float,
    sample_spacing_m: float,
) -> bool:
    segment = np.stack(
        [
            np.asarray(a, dtype=np.float32).reshape(3),
            np.asarray(b, dtype=np.float32).reshape(3),
        ],
        axis=0,
    )
    dense = resample_polyline(segment, max(float(sample_spacing_m), 1e-6))
    clearances = esdf_clearance_m_for_points(graph, dense)
    return bool(np.all(clearances >= float(safety_clearance_m)))


def _safe_shortcut_path(
    graph: EsdfGridGraph,
    path_world: np.ndarray,
    *,
    safety_clearance_m: float,
    sample_spacing_m: float,
    max_checks: int,
) -> np.ndarray:
    points = _validate_path_points(path_world)
    if points.shape[0] <= 2:
        return points.copy()

    selected = [0]
    checks = 0
    i = 0
    last = points.shape[0] - 1
    while i < last:
        best = i + 1
        for j in range(last, i, -1):
            if checks >= int(max_checks):
                break
            checks += 1
            if _line_segment_is_safe(
                graph,
                points[i],
                points[j],
                safety_clearance_m=safety_clearance_m,
                sample_spacing_m=sample_spacing_m,
            ):
                best = j
                break
        selected.append(best)
        i = best
        if checks >= int(max_checks):
            selected.extend(range(i + 1, points.shape[0]))
            break
    return points[np.asarray(selected, dtype=np.int32)].astype(np.float32)


def stomp_esdf_path(
    graph: EsdfGridGraph,
    coarse_path: CoarsePath | np.ndarray,
    config: StompPlannerConfig | None = None,
) -> StompPlannerResult:
    """Optimize a coarse path with naive STOMP rollouts over a B-spline."""

    local_planning_started = time.perf_counter()
    cfg = config or StompPlannerConfig()
    coarse = coarse_path if isinstance(coarse_path, CoarsePath) else CoarsePath(points=_validate_path_points(coarse_path))
    safety_clearance = cfg.resolved_safety_clearance_m(coarse)
    collision_sample_spacing = max(float(cfg.collision_sample_spacing_m), 1e-6)
    coarse_points_before_shortcut = int(coarse.points.shape[0])
    if bool(cfg.shortcut_input) and coarse_points_before_shortcut > 2:
        # Dense sampling prevents a line-of-sight check from stepping over a
        # narrow unsafe ESDF voxel.
        shortcut_sample_spacing = min(
            collision_sample_spacing,
            max(0.25 * float(graph.voxel_size), 1e-6),
        )
        shortcut_points = _safe_shortcut_path(
            graph,
            coarse.points,
            safety_clearance_m=safety_clearance,
            sample_spacing_m=shortcut_sample_spacing,
            max_checks=max(1, int(cfg.safe_shortcut_max_checks)),
        )
        coarse = CoarsePath(
            points=shortcut_points,
            header=dict(coarse.header),
            raw_header_lines=list(coarse.raw_header_lines),
            path=coarse.path,
        )
    coarse_shortcut_applied = int(coarse.points.shape[0]) < coarse_points_before_shortcut
    input_validation = validate_stomp_path(
        graph,
        coarse.points,
        safety_clearance_m=safety_clearance,
        sample_spacing_m=collision_sample_spacing,
    )
    query = _get_esdf_query(graph)

    spacing_attempts: list[dict[str, Any]] = []
    selected_problem: tuple[float, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, float, dict[str, float]] | None = None
    best_problem: tuple[float, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, float, dict[str, float]] | None = None
    input_is_safe = int(input_validation["unsafe_sample_count"]) == 0
    for spacing in _control_spacing_candidates(coarse, graph, cfg):
        q_ref_try, q_try, free_try, collision_sample_count_try, collision_basis_try = _build_control_problem(
            graph,
            coarse,
            cfg,
            spacing,
        )
        initial_cost_try, initial_parts_try = _trajectory_cost(
            query,
            q_try,
            q_ref_try,
            collision_basis_try,
            cfg,
            safety_clearance_m=safety_clearance,
        )
        initial_unsafe = int(initial_parts_try.get("unsafe_sample_count", 0))
        spacing_attempts.append(
            {
                "spacing_m": float(spacing),
                "control_points": int(q_try.shape[0]),
                "initial_unsafe_sample_count": initial_unsafe,
                "initial_min_clearance_m": float(initial_parts_try.get("min_clearance_m", float("inf"))),
            }
        )
        problem = (
            float(spacing),
            q_ref_try,
            q_try,
            free_try,
            int(collision_sample_count_try),
            collision_basis_try,
            float(initial_cost_try),
            initial_parts_try,
        )
        if best_problem is None or initial_unsafe < int(best_problem[7].get("unsafe_sample_count", 0)):
            best_problem = problem
        selected_problem = problem
        if not (bool(cfg.adaptive_control_point_spacing) and input_is_safe):
            break
        if initial_unsafe == 0:
            break

    if best_problem is None:
        raise RuntimeError("failed to build STOMP control problem")
    if selected_problem is None or int(selected_problem[7].get("unsafe_sample_count", 0)) > 0:
        selected_problem = best_problem

    (
        effective_control_point_spacing,
        q_ref,
        q,
        free,
        collision_sample_count,
        collision_basis,
        best_cost,
        best_parts,
    ) = selected_problem
    rng = np.random.default_rng(cfg.random_seed)
    noise_std = max(0.0, float(cfg.noise_std_m))
    best_q = q.copy()
    accepted_updates = 0
    iterations_done = 0
    cost_history = [float(best_cost)]
    current_cost = float(best_cost)
    current_parts = best_parts
    last_improvement_cost = float(best_cost)
    last_improvement_iteration = 0
    early_stopped = False

    for iteration in range(max(0, int(cfg.iterations))):
        iterations_done = iteration + 1
        rollout_count = max(1, int(cfg.rollouts))
        noises = np.zeros((rollout_count, q.shape[0], 3), dtype=np.float64)
        if free.shape[0] > 0 and noise_std > 0.0:
            noises[:, free, :] = rng.normal(0.0, float(noise_std), size=(rollout_count, free.shape[0], 3))
        rollout_q = _clip_control_points_to_world(graph, q[None, :, :] + noises)
        rollout_q[:, 0, :] = coarse.points[0]
        rollout_q[:, -1, :] = coarse.points[-1]
        noises = rollout_q - q[None, :, :]
        costs, _ = _trajectory_cost_batch(
            query,
            rollout_q,
            q_ref,
            collision_basis,
            cfg,
            safety_clearance_m=safety_clearance,
        )
        min_idx = int(np.argmin(costs))
        min_cost = float(costs[min_idx])
        scale = max(float(cfg.temperature), 1e-9)
        shifted = costs - min_cost
        weights = np.exp(-shifted / scale)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-12 or not np.isfinite(weight_sum):
            weights = np.full_like(costs, 1.0 / float(costs.shape[0]))
        else:
            weights /= weight_sum

        delta = np.zeros_like(q)
        for weight, noise in zip(weights, noises):
            delta += float(weight) * noise
        updated = _clip_control_points_to_world(graph, q + float(cfg.update_step_size) * delta)
        updated[0] = coarse.points[0]
        updated[-1] = coarse.points[-1]
        update_cost, update_parts = _trajectory_cost(
            query,
            updated,
            q_ref,
            collision_basis,
            cfg,
            safety_clearance_m=safety_clearance,
        )

        if update_cost < current_cost:
            q = updated
            current_cost = float(update_cost)
            current_parts = update_parts
            accepted_updates += 1
        elif min_cost < current_cost:
            q = rollout_q[min_idx]
            current_cost = min_cost
            _, current_parts = _trajectory_cost(
                query,
                q,
                q_ref,
                collision_basis,
                cfg,
                safety_clearance_m=safety_clearance,
            )
            accepted_updates += 1

        if current_cost < best_cost:
            best_q = q.copy()
            best_cost = float(current_cost)
            best_parts = current_parts
            if last_improvement_cost - best_cost >= float(cfg.min_improvement):
                last_improvement_cost = float(best_cost)
                last_improvement_iteration = iterations_done
        cost_history.append(float(best_cost))
        noise_std *= float(cfg.noise_decay)
        if (
            int(cfg.stagnation_patience) > 0
            and iterations_done - last_improvement_iteration >= int(cfg.stagnation_patience)
        ):
            early_stopped = True
            break

    q = best_q
    path_world, _, _ = sample_control_spline(
        q,
        spacing_m=max(float(cfg.output_spacing_m), 1e-6),
    )
    path_world[0] = coarse.points[0]
    path_world[-1] = coarse.points[-1]
    local_planning_time = float(time.perf_counter() - local_planning_started)
    validation_started = time.perf_counter()
    validation = validate_stomp_path(
        graph,
        path_world,
        safety_clearance_m=safety_clearance,
        sample_spacing_m=collision_sample_spacing,
    )
    used_safety_fallback = False
    used_shortcut_fallback = False
    collision_free = int(validation["unsafe_sample_count"]) == 0
    success = collision_free
    message = "collision-free local STOMP path" if success else "local STOMP path still has unsafe samples"
    shortcut_validation: dict[str, Any] | None = None
    shortcut_points: np.ndarray | None = None
    shortcut_spacing = max(float(cfg.output_spacing_m), 1e-6)
    needs_shortcut_fallback = bool(cfg.safe_shortcut_fallback) and (not success or accepted_updates == 0)
    if needs_shortcut_fallback:
        shortcut_raw = _safe_shortcut_path(
            graph,
            coarse.points,
            safety_clearance_m=safety_clearance,
            sample_spacing_m=collision_sample_spacing,
            max_checks=max(1, int(cfg.safe_shortcut_max_checks)),
        )
        shortcut_points = resample_polyline(shortcut_raw, shortcut_spacing)
        shortcut_points[0] = coarse.points[0]
        shortcut_points[-1] = coarse.points[-1]
        shortcut_validation = validate_stomp_path(
            graph,
            shortcut_points,
            safety_clearance_m=safety_clearance,
            sample_spacing_m=collision_sample_spacing,
        )

    shortcut_is_safe_and_shorter = (
        shortcut_validation is not None
        and shortcut_points is not None
        and int(shortcut_validation["unsafe_sample_count"]) == 0
        and float(shortcut_validation["length_m"]) + float(cfg.safe_shortcut_min_length_improvement_m)
        < float(validation["length_m"])
    )
    if needs_shortcut_fallback and shortcut_is_safe_and_shorter:
        stomp_was_success = success
        path_world = shortcut_points.astype(np.float32, copy=True)
        validation = shortcut_validation
        collision_free = True
        success = True
        used_shortcut_fallback = True
        if not stomp_was_success:
            message = "STOMP path unsafe; returned collision-free shortcut path"
        elif accepted_updates == 0:
            message = "STOMP made no accepted update; returned collision-free shortcut path"
        else:
            message = "returned collision-free shortcut path"

    if not success and cfg.safe_fallback_to_input:
        fallback_validation = validate_stomp_path(
            graph,
            coarse.points,
            safety_clearance_m=safety_clearance,
            sample_spacing_m=collision_sample_spacing,
        )
        if int(fallback_validation["unsafe_sample_count"]) == 0:
            path_world = coarse.points.astype(np.float32, copy=True)
            validation = fallback_validation
            collision_free = True
            success = True
            used_safety_fallback = True
            message = "STOMP smoothing unsafe; returned collision-free input path"

    change_stats = _path_change_stats(
        coarse.points,
        path_world,
        sample_spacing_m=max(float(cfg.output_spacing_m), 1e-6),
        position_tolerance_m=float(cfg.no_change_position_tolerance_m),
        length_tolerance_m=float(cfg.no_change_length_tolerance_m),
    )
    changed_path = bool(change_stats["changed_path"])
    no_change_reason = ""
    failure_stage = ""
    if not changed_path:
        if used_safety_fallback:
            no_change_reason = (
                "optimized candidate was unsafe; returned the original collision-free input path"
            )
        elif accepted_updates == 0:
            no_change_reason = "no accepted STOMP update changed the path"
        elif not collision_free:
            failure_stage = "unsafe_output"
            no_change_reason = "candidate path remained unsafe and matched the input geometry"
        else:
            no_change_reason = "output geometry matched the input path within tolerance"
    if not collision_free:
        success = False
        failure_stage = "unsafe_output"

    planner_validation_time = float(time.perf_counter() - validation_started)

    stats = {
        "best_cost": float(best_cost),
        "best_cost_parts": best_parts,
        "cost_history": cost_history,
        "accepted_updates": int(accepted_updates),
        "early_stopped": bool(early_stopped),
        "stagnation_patience": int(cfg.stagnation_patience),
        "min_improvement": float(cfg.min_improvement),
        "final_noise_std_m": float(noise_std),
        "control_points": int(q.shape[0]),
        "collision_samples": int(collision_sample_count),
        "input_unsafe_sample_count": int(input_validation["unsafe_sample_count"]),
        "input_min_clearance_m": float(input_validation["min_clearance_m"]),
        "coarse_points_before_shortcut": coarse_points_before_shortcut,
        "coarse_points_after_shortcut": int(coarse.points.shape[0]),
        "coarse_shortcut_applied": bool(coarse_shortcut_applied),
        "requested_control_point_spacing_m": float(cfg.control_point_spacing_m),
        "effective_control_point_spacing_m": float(effective_control_point_spacing),
        "adaptive_control_spacing_used": bool(
            abs(float(effective_control_point_spacing) - float(cfg.control_point_spacing_m)) > 1e-9
        ),
        "control_spacing_attempts": spacing_attempts,
        "requested_initial_unsafe_sample_count": (
            int(spacing_attempts[0]["initial_unsafe_sample_count"]) if spacing_attempts else 0
        ),
        "effective_initial_unsafe_sample_count": int(best_parts.get("unsafe_sample_count", 0)),
        "unknown_policy": str(graph.unknown_policy),
        "unknown_clearance_m": float(graph.unknown_clearance_m),
        "occupied_guard_voxels": int(graph.occupied_idx.shape[0]),
        "collision_free": bool(collision_free),
        "changed_path": bool(changed_path),
        "no_change_reason": no_change_reason,
        "failure_stage": failure_stage,
        "used_safety_fallback": bool(used_safety_fallback),
        "used_shortcut_fallback": bool(used_shortcut_fallback),
        "random_seed": None if cfg.random_seed is None else int(cfg.random_seed),
        "output_spacing_m": float(cfg.output_spacing_m),
        "collision_sample_spacing_m": float(cfg.collision_sample_spacing_m),
        "timings_sec": {
            "local_planning": local_planning_time,
            "planner_validation": planner_validation_time,
        },
        "shortcut_points": int(shortcut_points.shape[0]) if shortcut_points is not None else 0,
        "shortcut_length_m": float(shortcut_validation["length_m"]) if shortcut_validation is not None else 0.0,
        "shortcut_unsafe_sample_count": (
            int(shortcut_validation["unsafe_sample_count"]) if shortcut_validation is not None else 0
        ),
    }
    stats.update(change_stats)
    return StompPlannerResult(
        path_world=np.asarray(path_world, dtype=np.float32),
        control_points_world=q.astype(np.float32),
        reference_control_points_world=q_ref.astype(np.float32),
        safety_clearance_m=float(safety_clearance),
        iterations=int(iterations_done),
        rollouts=int(cfg.rollouts),
        min_clearance_m=float(validation["min_clearance_m"]),
        unsafe_sample_count=int(validation["unsafe_sample_count"]),
        length_m=float(validation["length_m"]),
        success=bool(success),
        message=message,
        stats=stats,
    )


def default_output_path(input_path: str | Path, result: StompPlannerResult, config: StompPlannerConfig) -> Path:
    input_resolved = Path(input_path).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in input_resolved.stem).strip("_")
    if len(input_slug) > 72:
        input_slug = input_slug[:72].rstrip("_")
    filename = "_".join(
        [
            "stomp_esdf",
            input_slug or "path",
            f"clr{_format_filename_value(result.safety_clearance_m)}m",
            f"cp{_format_filename_value(config.control_point_spacing_m)}",
            f"it{int(config.iterations)}",
            timestamp,
        ]
    ) + ".txt"
    return input_resolved.parent / filename


def save_stomp_path(
    output_path: str | Path,
    result: StompPlannerResult,
    *,
    esdf_map_path: str | Path,
    input_path: str | Path,
    config: StompPlannerConfig,
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        stem = path.stem
        for i in range(2, 1000):
            candidate = path.with_name(f"{stem}_{i}{path.suffix}")
            if not candidate.exists():
                path = candidate
                break
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_clearance_value: float | None = None
    try:
        input_clearance_value = config.input_safety_clearance_m(read_coarse_path(input_path))
    except Exception:
        input_clearance_value = None
    header = [
        "NERFICG local path",
        f"method: {LOCAL_STOMP_METHOD}",
        f"timestamp: {timestamp}",
        f"local_map: {Path(esdf_map_path).expanduser().resolve()}",
        f"input_global_path: {Path(input_path).expanduser().resolve()}",
        f"points: {int(result.path_world.shape[0])}",
        f"control_points: {int(result.control_points_world.shape[0])}",
        f"safety_clearance_m: {result.safety_clearance_m:.3f}",
        f"input_safety_clearance_m: {input_clearance_value:.3f}" if input_clearance_value is not None else "input_safety_clearance_m: (unknown)",
        f"inherit_input_clearance: {bool(config.inherit_input_clearance)}",
        f"default_safety_clearance_m: {float(config.default_safety_clearance_m):.3f}",
        f"control_point_spacing_m: {float(config.control_point_spacing_m):.3f}",
        f"output_spacing_m: {float(config.output_spacing_m):.3f}",
        f"collision_sample_spacing_m: {float(config.collision_sample_spacing_m):.3f}",
        f"stomp_iterations: {int(result.iterations)}",
        f"stomp_rollouts: {int(result.rollouts)}",
        f"noise_std_m: {float(config.noise_std_m):.3f}",
        f"noise_decay: {float(config.noise_decay):.3f}",
        f"temperature: {float(config.temperature):.3f}",
        f"random_seed: {config.random_seed}",
        f"stagnation_patience: {int(config.stagnation_patience)}",
        f"min_improvement: {float(config.min_improvement):.6g}",
        f"min_clearance_m: {float(result.min_clearance_m):.3f}",
        f"unsafe_sample_count: {int(result.unsafe_sample_count)}",
        f"length_m: {float(result.length_m):.6f}",
        f"success: {bool(result.success)}",
        f"collision_free: {bool(result.stats.get('collision_free', result.unsafe_sample_count == 0))}",
        f"changed_path: {bool(result.stats.get('changed_path', True))}",
        f"no_change_reason: {result.stats.get('no_change_reason', '')}",
        f"failure_stage: {result.stats.get('failure_stage', '')}",
        f"message: {result.message}",
        f"input_length_m: {float(result.stats.get('input_length_m', 0.0)):.6f}",
        f"output_length_delta_m: {float(result.stats.get('output_length_delta_m', 0.0)):.6f}",
        f"max_path_displacement_m: {float(result.stats.get('max_path_displacement_m', 0.0)):.9f}",
        f"mean_path_displacement_m: {float(result.stats.get('mean_path_displacement_m', 0.0)):.9f}",
        f"requested_control_point_spacing_m: {float(result.stats.get('requested_control_point_spacing_m', config.control_point_spacing_m)):.3f}",
        f"effective_control_point_spacing_m: {float(result.stats.get('effective_control_point_spacing_m', config.control_point_spacing_m)):.3f}",
        f"adaptive_control_spacing_used: {bool(result.stats.get('adaptive_control_spacing_used', False))}",
        f"input_unsafe_sample_count: {int(result.stats.get('input_unsafe_sample_count', 0))}",
        f"requested_initial_unsafe_sample_count: {int(result.stats.get('requested_initial_unsafe_sample_count', 0))}",
        f"effective_initial_unsafe_sample_count: {int(result.stats.get('effective_initial_unsafe_sample_count', 0))}",
        f"used_safety_fallback: {bool(result.stats.get('used_safety_fallback', False))}",
        f"used_shortcut_fallback: {bool(result.stats.get('used_shortcut_fallback', False))}",
        f"local_planning_time_sec: {float(dict(result.stats.get('timings_sec', {})).get('local_planning', 0.0)):.9f}",
        f"planner_validation_time_sec: {float(dict(result.stats.get('timings_sec', {})).get('planner_validation', 0.0)):.9f}",
        f"shortcut_length_m: {float(result.stats.get('shortcut_length_m', 0.0)):.6f}",
        f"shortcut_unsafe_sample_count: {int(result.stats.get('shortcut_unsafe_sample_count', 0))}",
        "columns: x y z",
    ]
    np.savetxt(path, result.path_world, fmt="%.6f", delimiter=" ", header="\n".join(header), comments="# ")
    return path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Naive STOMP B-spline local planner for ESDF voxel paths.")
    parser.add_argument("--map", required=True, type=Path, help="ESDF PLY map.")
    parser.add_argument("--grid-metadata", type=Path, default=None, help="Optional companion voxel_map .npz/.json.")
    parser.add_argument("--input-path", required=True, type=Path, help="Coarse global path .txt/.csv/.npy.")
    parser.add_argument("--output", type=Path, default=None, help="Output local path .txt. Defaults beside input path.")
    parser.add_argument("--safety-clearance-m", type=float, default=None, help="Explicit local STOMP clearance override in meters.")
    parser.add_argument("--default-safety-clearance-m", type=float, default=0.50)
    parser.add_argument("--no-inherit-input-clearance", action="store_true")
    parser.add_argument("--control-point-spacing-m", type=float, default=0.10)
    parser.add_argument("--output-spacing-m", type=float, default=0.05)
    parser.add_argument("--collision-sample-spacing-m", type=float, default=0.05)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--noise-std-m", type=float, default=0.10)
    parser.add_argument("--noise-decay", type=float, default=0.98)
    parser.add_argument("--update-step-size", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--smooth-weight", type=float, default=1.0)
    parser.add_argument("--jerk-weight", type=float, default=0.1)
    parser.add_argument("--collision-weight", type=float, default=200.0)
    parser.add_argument("--clearance-weight", type=float, default=0.0)
    parser.add_argument("--fitness-weight", type=float, default=30.0)
    parser.add_argument("--length-weight", type=float, default=0.0)
    parser.add_argument("--pin-endpoint-control-points", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--stagnation-patience", type=int, default=20)
    parser.add_argument("--min-improvement", type=float, default=1.0e-5)
    parser.add_argument(
        "--allow-unknown",
        dest="allow_unknown",
        action="store_true",
        default=True,
        help="Allow missing ESDF samples as unknown-free proxy cells. This matches the GUI default.",
    )
    parser.add_argument(
        "--disallow-unknown",
        dest="allow_unknown",
        action="store_false",
        help="Treat missing ESDF samples as blocked.",
    )
    parser.add_argument("--unknown-clearance-m", type=float, default=None)
    parser.add_argument("--no-occupancy-guard", action="store_true")
    parser.add_argument("--disable-safe-fallback", action="store_true")
    parser.add_argument(
        "--enable-safe-shortcut-fallback",
        action="store_true",
        help="Enable the non-STOMP greedy shortcut fallback (disabled by default for fair evaluation).",
    )
    parser.add_argument("--safe-shortcut-max-checks", type=int, default=50000)
    parser.add_argument("--disable-adaptive-control-spacing", action="store_true")
    parser.add_argument("--adaptive-min-control-point-spacing-m", type=float, default=0.05)
    parser.add_argument("--adaptive-max-control-points", type=int, default=256)
    parser.add_argument("--no-change-position-tolerance-m", type=float, default=1.0e-6)
    parser.add_argument("--no-change-length-tolerance-m", type=float, default=1.0e-6)
    parser.add_argument("--verify-only", action="store_true", help="Only validate the input path against the ESDF map.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    coarse = read_coarse_path(args.input_path)
    config = StompPlannerConfig(
        control_point_spacing_m=float(args.control_point_spacing_m),
        output_spacing_m=float(args.output_spacing_m),
        collision_sample_spacing_m=float(args.collision_sample_spacing_m),
        safety_clearance_m=args.safety_clearance_m,
        default_safety_clearance_m=float(args.default_safety_clearance_m),
        inherit_input_clearance=not bool(args.no_inherit_input_clearance),
        pin_endpoint_control_points=int(args.pin_endpoint_control_points),
        iterations=int(args.iterations),
        rollouts=int(args.rollouts),
        noise_std_m=float(args.noise_std_m),
        noise_decay=float(args.noise_decay),
        update_step_size=float(args.update_step_size),
        temperature=float(args.temperature),
        smooth_weight=float(args.smooth_weight),
        jerk_weight=float(args.jerk_weight),
        collision_weight=float(args.collision_weight),
        clearance_weight=float(args.clearance_weight),
        fitness_weight=float(args.fitness_weight),
        length_weight=float(args.length_weight),
        random_seed=args.random_seed,
        stagnation_patience=int(args.stagnation_patience),
        min_improvement=float(args.min_improvement),
        safe_fallback_to_input=not bool(args.disable_safe_fallback),
        safe_shortcut_fallback=bool(args.enable_safe_shortcut_fallback),
        safe_shortcut_max_checks=int(args.safe_shortcut_max_checks),
        adaptive_control_point_spacing=not bool(args.disable_adaptive_control_spacing),
        adaptive_min_control_point_spacing_m=float(args.adaptive_min_control_point_spacing_m),
        adaptive_max_control_points=int(args.adaptive_max_control_points),
        no_change_position_tolerance_m=float(args.no_change_position_tolerance_m),
        no_change_length_tolerance_m=float(args.no_change_length_tolerance_m),
    )
    graph = load_esdf_grid(
        args.map,
        args.grid_metadata,
        use_occupancy_guard=not bool(args.no_occupancy_guard),
        cache_aligned=True,
        unknown_policy=UNKNOWN_POLICY_FREE if args.allow_unknown else UNKNOWN_POLICY_OCCUPIED,
        unknown_clearance_m=args.unknown_clearance_m,
    )
    safety = config.resolved_safety_clearance_m(coarse)
    if args.verify_only:
        validation = validate_stomp_path(
            graph,
            coarse.points,
            safety_clearance_m=safety,
            sample_spacing_m=max(float(config.collision_sample_spacing_m), 1e-6),
        )
        print(
            "verify input: "
            f"points={validation['points']} "
            f"length_m={validation['length_m']:.6f} "
            f"min_clearance_m={validation['min_clearance_m']:.3f} "
            f"unsafe_sample_count={validation['unsafe_sample_count']} "
            f"safety_clearance_m={safety:.3f}"
        )
        return 0 if validation["unsafe_sample_count"] == 0 else 2

    result = stomp_esdf_path(graph, coarse, config)
    output = args.output if args.output is not None else default_output_path(args.input_path, result, config)
    saved = save_stomp_path(output, result, esdf_map_path=args.map, input_path=args.input_path, config=config)
    print(
        f"saved {saved}\n"
        f"success={result.success} message={result.message}\n"
        f"points={result.path_world.shape[0]} control_points={result.control_points_world.shape[0]} "
        f"length_m={result.length_m:.6f} min_clearance_m={result.min_clearance_m:.3f} "
        f"unsafe_sample_count={result.unsafe_sample_count} "
        f"changed_path={result.stats.get('changed_path', True)} "
        f"effective_control_point_spacing_m={result.stats.get('effective_control_point_spacing_m', config.control_point_spacing_m):.3f} "
        f"accepted_updates={result.stats.get('accepted_updates', 0)} "
        f"early_stopped={result.stats.get('early_stopped', False)}"
    )
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
