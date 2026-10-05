#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Occupancy Map global path planning with JPS-3D."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from GSPAR.path.global_path.astar_world_path import VoxelGridGraph


def _default_jps3d_executable() -> Path:
    return Path(__file__).resolve().parents[1] / "jps3d" / "build" / "nerficg_jps3d_cli"


def _jps3d_crop_bounds(
    graph: VoxelGridGraph,
    start_idx: np.ndarray,
    goal_idx: np.ndarray,
    crop_margin_voxels: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if crop_margin_voxels is None or int(crop_margin_voxels) < 0:
        return np.zeros(3, dtype=np.int32), graph.voxel_dims.astype(np.int32, copy=True)

    margin = int(crop_margin_voxels)
    lo = np.minimum(start_idx, goal_idx).astype(np.int32) - margin
    hi = np.maximum(start_idx, goal_idx).astype(np.int32) + margin + 1
    lo = np.maximum(lo, 0).astype(np.int32)
    hi = np.minimum(hi, graph.voxel_dims).astype(np.int32)
    return lo, hi


def _build_jps3d_cmap(
    graph: VoxelGridGraph,
    safety_clearance_voxels: float,
    crop_lo: np.ndarray,
    crop_hi: np.ndarray,
) -> np.ndarray:
    crop_lo = np.asarray(crop_lo, dtype=np.int32).reshape(3)
    crop_hi = np.asarray(crop_hi, dtype=np.int32).reshape(3)
    crop_dims = (crop_hi - crop_lo).astype(np.int64)
    total = int(np.prod(crop_dims))
    safety_clearance_voxels = float(safety_clearance_voxels)

    if safety_clearance_voxels <= 0.0:
        cmap = np.zeros(total, dtype=np.int8)
        if graph.occupied_idx.shape[0]:
            in_crop = np.logical_and(
                graph.occupied_idx >= crop_lo[None, :],
                graph.occupied_idx < crop_hi[None, :],
            ).all(axis=1)
            occ = graph.occupied_idx[in_crop] - crop_lo[None, :]
            if occ.shape[0]:
                crop_dims64 = crop_dims.astype(np.int64)
                keys = (
                    occ[:, 0].astype(np.int64)
                    + crop_dims64[0] * (occ[:, 1].astype(np.int64) + crop_dims64[1] * occ[:, 2].astype(np.int64))
                )
                cmap[keys] = np.int8(100)
        return cmap

    from scipy.ndimage import distance_transform_edt

    pad = int(np.ceil(safety_clearance_voxels))
    edt_lo = np.maximum(crop_lo - pad, 0).astype(np.int32)
    edt_hi = np.minimum(crop_hi + pad, graph.voxel_dims).astype(np.int32)
    edt_dims = (edt_hi - edt_lo).astype(np.int64)

    occupied_zyx = np.zeros((int(edt_dims[2]), int(edt_dims[1]), int(edt_dims[0])), dtype=bool)
    if graph.occupied_idx.shape[0]:
        in_edt = np.logical_and(
            graph.occupied_idx >= edt_lo[None, :],
            graph.occupied_idx < edt_hi[None, :],
        ).all(axis=1)
        occ = (graph.occupied_idx[in_edt] - edt_lo[None, :]).astype(np.intp, copy=False)
        occupied_zyx[occ[:, 2], occ[:, 1], occ[:, 0]] = True
    distance_zyx = distance_transform_edt(~occupied_zyx)
    crop_offset = crop_lo - edt_lo
    unsafe_zyx = distance_zyx[
        int(crop_offset[2]):int(crop_offset[2] + crop_dims[2]),
        int(crop_offset[1]):int(crop_offset[1] + crop_dims[1]),
        int(crop_offset[0]):int(crop_offset[0] + crop_dims[0]),
    ] < safety_clearance_voxels
    return (unsafe_zyx.ravel(order="C").astype(np.int8, copy=False) * np.int8(100)).copy()


def _write_jps3d_input(
    path: Path,
    cmap: np.ndarray,
    dims: np.ndarray,
    start_idx: np.ndarray,
    goal_idx: np.ndarray,
    heuristic_weight: float,
    max_expansions: int,
) -> None:
    max_expand = int(max_expansions)
    if max_expand <= 0:
        max_expand = -1
    header = (
        "NERFICG_JPS3D_V1\n"
        f"{int(dims[0])} {int(dims[1])} {int(dims[2])}\n"
        f"{int(start_idx[0])} {int(start_idx[1])} {int(start_idx[2])}\n"
        f"{int(goal_idx[0])} {int(goal_idx[1])} {int(goal_idx[2])}\n"
        f"{max(1.0, float(heuristic_weight)):.9g} 1 {max_expand}\n"
        "DATA\n"
    )
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(np.ascontiguousarray(cmap, dtype=np.int8).tobytes(order="C"))


def _parse_jps3d_output(stdout: str) -> np.ndarray:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("JPS-3D returned no output")
    if "OK" not in lines:
        detail = "; ".join(lines[:4])
        raise RuntimeError(f"JPS-3D failed: {detail}")
    lines = lines[lines.index("OK"):]

    path_header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("path "):
            path_header_idx = i
            break
    if path_header_idx is None:
        raise RuntimeError("JPS-3D output missing path header")

    count = int(lines[path_header_idx].split()[1])
    coords = []
    for line in lines[path_header_idx + 1:path_header_idx + 1 + count]:
        parts = line.split()
        if len(parts) != 3:
            raise RuntimeError(f"invalid JPS-3D path row: {line}")
        coords.append([int(parts[0]), int(parts[1]), int(parts[2])])
    if len(coords) != count:
        raise RuntimeError(f"JPS-3D output path truncated: expected {count}, got {len(coords)}")
    return np.asarray(coords, dtype=np.int32)


def _densify_jps3d_path(path_idx: np.ndarray) -> np.ndarray:
    path_idx = np.asarray(path_idx, dtype=np.int32).reshape(-1, 3)
    if path_idx.shape[0] <= 1:
        return path_idx.copy()

    dense: list[np.ndarray] = [path_idx[0].copy()]
    for start, end in zip(path_idx[:-1], path_idx[1:]):
        delta = end.astype(np.int32) - start.astype(np.int32)
        steps = int(np.max(np.abs(delta)))
        if steps <= 0:
            continue
        moving = np.abs(delta) > 0
        if not np.all(np.abs(delta[moving]) == steps):
            dense.append(end.copy())
            continue
        step = np.sign(delta).astype(np.int32)
        for i in range(1, steps + 1):
            dense.append((start + step * i).astype(np.int32))
    return np.stack(dense, axis=0).astype(np.int32)


def _run_jps3d_cmap(
    cmap: np.ndarray,
    *,
    crop_lo: np.ndarray,
    crop_hi: np.ndarray,
    start_idx: np.ndarray,
    goal_idx: np.ndarray,
    heuristic_weight: float,
    max_expansions: int,
    max_planning_time_sec: float | None,
    executable: str | Path | None,
    densify: bool,
) -> np.ndarray:
    """Run the JPS executable on a precomputed cropped collision map."""

    exe_path = Path(executable) if executable is not None else _default_jps3d_executable()
    if not exe_path.is_file():
        raise FileNotFoundError(
            "JPS-3D executable not found. Build it with: "
            "cmake -S GSPAR/path/jps3d -B GSPAR/path/jps3d/build && "
            "cmake --build GSPAR/path/jps3d/build -j"
        )
    crop_lo = np.asarray(crop_lo, dtype=np.int32).reshape(3)
    crop_hi = np.asarray(crop_hi, dtype=np.int32).reshape(3)
    crop_dims = (crop_hi - crop_lo).astype(np.int32)
    local_start_idx = (np.asarray(start_idx, dtype=np.int32).reshape(3) - crop_lo).astype(np.int32)
    local_goal_idx = (np.asarray(goal_idx, dtype=np.int32).reshape(3) - crop_lo).astype(np.int32)
    crop_dims64 = crop_dims.astype(np.int64)
    start_key = int(
        local_start_idx[0]
        + crop_dims64[0] * (local_start_idx[1] + crop_dims64[1] * local_start_idx[2])
    )
    goal_key = int(
        local_goal_idx[0]
        + crop_dims64[0] * (local_goal_idx[1] + crop_dims64[1] * local_goal_idx[2])
    )
    cmap = np.ascontiguousarray(cmap, dtype=np.int8).reshape(-1)
    cmap[start_key] = np.int8(0)
    cmap[goal_key] = np.int8(0)

    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(prefix="nerficg_jps3d_", suffix=".bin", delete=False) as tmp:
            tmp_name = tmp.name
        tmp_path = Path(tmp_name)
        _write_jps3d_input(
            tmp_path,
            cmap,
            crop_dims,
            local_start_idx,
            local_goal_idx,
            heuristic_weight,
            max_expansions,
        )
        timeout = None
        if max_planning_time_sec is not None and float(max_planning_time_sec) > 0.0:
            timeout = float(max_planning_time_sec)
        completed = subprocess.run(
            [str(exe_path), str(tmp_path)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"JPS-3D exceeded max planning time ({float(max_planning_time_sec):.3f}s)"
        ) from exc
    finally:
        if tmp_name is not None:
            Path(tmp_name).unlink(missing_ok=True)

    if completed.returncode != 0:
        detail = completed.stdout.strip() or completed.stderr.strip()
        raise RuntimeError(f"JPS-3D failed: {detail}")
    path_idx = _parse_jps3d_output(completed.stdout)
    if densify:
        path_idx = _densify_jps3d_path(path_idx)
    return (path_idx + crop_lo[None, :]).astype(np.int32)


def _build_esdf_jps3d_cmap(
    graph,
    safety_clearance_m: float,
    crop_lo: np.ndarray,
    crop_hi: np.ndarray,
    *,
    batch_voxels: int = 500_000,
) -> np.ndarray:
    """Materialize a cropped JPS collision map from ESDF safety queries."""

    crop_lo = np.asarray(crop_lo, dtype=np.int32).reshape(3)
    crop_hi = np.asarray(crop_hi, dtype=np.int32).reshape(3)
    crop_dims = (crop_hi - crop_lo).astype(np.int64)
    total = int(np.prod(crop_dims))
    cmap = np.empty((total,), dtype=np.int8)
    xy = int(crop_dims[0] * crop_dims[1])
    chunk = max(1, int(batch_voxels))
    for begin in range(0, total, chunk):
        end = min(total, begin + chunk)
        linear = np.arange(begin, end, dtype=np.int64)
        local_idx = np.empty((linear.shape[0], 3), dtype=np.int32)
        local_idx[:, 0] = (linear % crop_dims[0]).astype(np.int32)
        local_idx[:, 1] = ((linear // crop_dims[0]) % crop_dims[1]).astype(np.int32)
        local_idx[:, 2] = (linear // xy).astype(np.int32)
        _, safe = graph.distance_and_safety_for_idx_batch(
            local_idx + crop_lo[None, :],
            safety_clearance_m=float(safety_clearance_m),
        )
        cmap[begin:end] = np.where(safe, np.int8(0), np.int8(100))
    return cmap


def jps3d_voxel_indices(
    graph: VoxelGridGraph,
    start_world: np.ndarray,
    goal_world: np.ndarray,
    *,
    max_expansions: int = 0,
    snap_search_radius: int = 1,
    safety_clearance_voxels: float = 2.5,
    heuristic_weight: float = 1.0,
    max_planning_time_sec: float | None = 10.0,
    executable: str | Path | None = None,
    densify: bool = True,
    crop_margin_voxels: int | None = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plan on an Occupancy Map with JPS-3D and return voxel indices."""

    start_idx = graph.snap_to_safe_idx(start_world, snap_search_radius, safety_clearance_voxels)
    goal_idx = graph.snap_to_safe_idx(goal_world, snap_search_radius, safety_clearance_voxels)
    crop_lo, crop_hi = _jps3d_crop_bounds(graph, start_idx, goal_idx, crop_margin_voxels)
    cmap = _build_jps3d_cmap(graph, safety_clearance_voxels, crop_lo, crop_hi)
    path_idx = _run_jps3d_cmap(
        cmap,
        crop_lo=crop_lo,
        crop_hi=crop_hi,
        start_idx=start_idx,
        goal_idx=goal_idx,
        heuristic_weight=heuristic_weight,
        max_expansions=max_expansions,
        max_planning_time_sec=max_planning_time_sec,
        executable=executable,
        densify=densify,
    )
    return path_idx, start_idx.astype(np.int32), goal_idx.astype(np.int32)


def esdf_jps3d_voxel_indices(
    graph,
    start_world: np.ndarray,
    goal_world: np.ndarray,
    *,
    max_expansions: int = 0,
    snap_search_radius: int = 8,
    safety_clearance_m: float = 0.0,
    heuristic_weight: float = 1.0,
    max_planning_time_sec: float | None = 10.0,
    executable: str | Path | None = None,
    densify: bool = True,
    crop_margin_voxels: int | None = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plan with JPS-3D using ESDF clearance and unknown-space policy."""

    start_idx = graph.snap_to_safe_idx(
        start_world,
        max_radius=int(snap_search_radius),
        safety_clearance_m=float(safety_clearance_m),
    )
    goal_idx = graph.snap_to_safe_idx(
        goal_world,
        max_radius=int(snap_search_radius),
        safety_clearance_m=float(safety_clearance_m),
    )
    crop_lo, crop_hi = _jps3d_crop_bounds(graph, start_idx, goal_idx, crop_margin_voxels)
    cmap = _build_esdf_jps3d_cmap(
        graph,
        safety_clearance_m,
        crop_lo,
        crop_hi,
    )
    path_idx = _run_jps3d_cmap(
        cmap,
        crop_lo=crop_lo,
        crop_hi=crop_hi,
        start_idx=start_idx,
        goal_idx=goal_idx,
        heuristic_weight=heuristic_weight,
        max_expansions=max_expansions,
        max_planning_time_sec=max_planning_time_sec,
        executable=executable,
        densify=densify,
    )
    return path_idx, start_idx.astype(np.int32), goal_idx.astype(np.int32)


def jps3d_world_path(
    graph: VoxelGridGraph,
    start_world: np.ndarray,
    goal_world: np.ndarray,
    *,
    max_expansions: int = 0,
    snap_search_radius: int = 1,
    safety_clearance_voxels: float = 2.5,
    heuristic_weight: float = 1.0,
    max_planning_time_sec: float | None = 10.0,
    executable: str | Path | None = None,
    densify: bool = True,
    crop_margin_voxels: int | None = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plan a global path in world coordinates from an Occupancy Map."""

    path_idx, start_idx, goal_idx = jps3d_voxel_indices(
        graph,
        start_world,
        goal_world,
        max_expansions=max_expansions,
        snap_search_radius=snap_search_radius,
        safety_clearance_voxels=safety_clearance_voxels,
        heuristic_weight=heuristic_weight,
        max_planning_time_sec=max_planning_time_sec,
        executable=executable,
        densify=densify,
        crop_margin_voxels=crop_margin_voxels,
    )
    path_world = graph.voxel_idx_to_world_center(path_idx).astype(np.float32)
    return path_world, start_idx, goal_idx


def esdf_jps3d_world_path(
    graph,
    start_world: np.ndarray,
    goal_world: np.ndarray,
    *,
    max_expansions: int = 0,
    snap_search_radius: int = 8,
    safety_clearance_m: float = 0.0,
    heuristic_weight: float = 1.0,
    max_planning_time_sec: float | None = 10.0,
    executable: str | Path | None = None,
    densify: bool = True,
    crop_margin_voxels: int | None = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plan a global JPS path directly against an ESDF graph."""

    path_idx, start_idx, goal_idx = esdf_jps3d_voxel_indices(
        graph,
        start_world,
        goal_world,
        max_expansions=max_expansions,
        snap_search_radius=snap_search_radius,
        safety_clearance_m=safety_clearance_m,
        heuristic_weight=heuristic_weight,
        max_planning_time_sec=max_planning_time_sec,
        executable=executable,
        densify=densify,
        crop_margin_voxels=crop_margin_voxels,
    )
    path_world = graph.voxel_idx_to_world_center(path_idx).astype(np.float32)
    return path_world, start_idx, goal_idx
