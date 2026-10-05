#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""GPU Gaussian splat with one dense or sparse contribution-sum accumulator."""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def _unpack_voxel_idx_torch(torch, packed, dims):
    x = packed % dims[0]
    yz = torch.div(packed, dims[0], rounding_mode="floor")
    y = yz % dims[1]
    z = torch.div(yz, dims[1], rounding_mode="floor")
    return torch.stack([x, y, z], dim=1).to(dtype=torch.int32)


def _offset_grid(torch, radius: int, device):
    coords = torch.arange(-int(radius), int(radius) + 1, dtype=torch.int32, device=device)
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1)


def _synchronize_current_cuda_stream(torch, device):
    """Wait only for the stream that owns the current splat conversion."""

    stream = torch.cuda.current_stream(device)
    stream.synchronize()
    return stream


def _select_sum_accumulator_backend(
    voxel_dims: np.ndarray,
    max_accumulator_bytes: int,
) -> tuple[np.ndarray, int, int, str]:
    """Choose dense storage only when its full float32 grid fits the budget."""

    dims = np.asarray(voxel_dims, dtype=np.int64).reshape(3)
    if np.any(dims <= 0):
        raise ValueError(f"voxel_dims must be positive, got {dims.tolist()}.")
    num_voxels = int(dims[0]) * int(dims[1]) * int(dims[2])
    dense_bytes = int(num_voxels * np.dtype(np.float32).itemsize)
    backend = (
        "gpu_dense_sum_only"
        if dense_bytes <= int(max_accumulator_bytes)
        else "gpu_sparse_sum_only"
    )
    return dims, num_voxels, dense_bytes, backend


def splat_gaussians_dense_sum_gpu(
    *,
    gaussians,
    support_radii: np.ndarray,
    origin: np.ndarray,
    voxel_dims: np.ndarray,
    voxel_size: float,
    contribution_floor: float,
    occupied_threshold: float,
    voxel_footprint_scale: float,
    surface_splat_mode: str = "volume",
    normal_support_voxels: float = 1.0,
    thin_min_anisotropy: float = 0.0,
    requested_device: str,
    gaussian_chunk_size: int,
    max_pairs_per_batch: int,
    max_accumulator_bytes: int,
    reset_peak_memory_stats: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Accumulate contribution sums and keep voxels at the occupied threshold.

    Dense storage is used while the complete float32 grid fits the configured
    budget. Larger logical grids automatically use packed sparse voxel indices,
    preserving the same origin, dimensions, contributions, and threshold.
    """

    import torch

    started_at = time.perf_counter()
    device = torch.device(str(requested_device))
    if device.type != "cuda":
        raise ValueError(f"GPU sum-only splat requires CUDA, got {device}.")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU sum-only splat requested, but CUDA is unavailable.")

    surface_mode = str(surface_splat_mode).strip().lower()
    if surface_mode not in {"volume", "thin"}:
        raise ValueError(f"surface_splat_mode must be volume/thin, got {surface_splat_mode!r}.")
    normal_support = float(normal_support_voxels)
    if not np.isfinite(normal_support) or normal_support <= 0.0:
        raise ValueError(f"normal_support_voxels must be positive and finite, got {normal_support_voxels}.")
    min_anisotropy = max(0.0, float(thin_min_anisotropy))

    dims_np, num_voxels, dense_accumulator_bytes, backend = (
        _select_sum_accumulator_backend(voxel_dims, max_accumulator_bytes)
    )
    use_dense_accumulator = backend == "gpu_dense_sum_only"

    positions_np = np.asarray(gaussians.positions, dtype=np.float32).reshape(-1, 3)
    rotations_np = np.asarray(gaussians.rotations, dtype=np.float32).reshape(-1, 3, 3)
    scales_np = np.asarray(gaussians.scales, dtype=np.float32).reshape(-1, 3)
    opacities_np = np.asarray(gaussians.opacities, dtype=np.float32).reshape(-1)
    radii_np = np.asarray(support_radii, dtype=np.float32).reshape(-1, 3)
    n = int(positions_np.shape[0])

    scale_min_np = np.maximum(np.min(scales_np, axis=1), 1e-12)
    anisotropy_np = np.max(scales_np, axis=1) / scale_min_np
    thin_enabled_np = np.zeros((n,), dtype=bool)
    if surface_mode == "thin":
        thin_enabled_np = (
            anisotropy_np >= min_anisotropy
            if min_anisotropy > 0.0
            else np.ones((n,), dtype=bool)
        )
    normal_axis_np = np.argmin(scales_np, axis=1).astype(np.int64)
    num_thin_enabled = int(np.count_nonzero(thin_enabled_np))

    voxel_size_f = float(voxel_size)
    voxel_half_margin = 0.5 * voxel_size_f * max(0.0, float(voxel_footprint_scale))
    origin_np = np.asarray(origin, dtype=np.float32).reshape(3)
    effective_radii_np = np.array(radii_np, copy=True)
    if num_thin_enabled > 0:
        thin_rows = np.flatnonzero(thin_enabled_np)
        effective_radii_np[thin_rows, normal_axis_np[thin_rows]] = np.minimum(
            effective_radii_np[thin_rows, normal_axis_np[thin_rows]],
            normal_support * voxel_size_f,
        )
    aabb_half_np = np.sqrt(
        np.sum(np.square(rotations_np) * np.square(effective_radii_np)[:, None, :], axis=2)
    ).astype(np.float32) + voxel_half_margin
    vmin_np = np.floor(
        (positions_np - aabb_half_np - origin_np[None, :]) / voxel_size_f
    ).astype(np.int64)
    vmax_np = np.floor(
        (positions_np + aabb_half_np - origin_np[None, :]) / voxel_size_f
    ).astype(np.int64)
    vmin_np = np.maximum(vmin_np, np.zeros((1, 3), dtype=np.int64))
    vmax_np = np.minimum(vmax_np, (dims_np - 1)[None, :])
    nonempty_np = np.all(vmin_np <= vmax_np, axis=1)
    center_idx_np = np.floor(
        (positions_np - origin_np[None, :]) / voxel_size_f
    ).astype(np.int64)
    lower_np = center_idx_np - vmin_np
    upper_np = vmax_np - center_idx_np
    max_offset_np = np.maximum(lower_np, upper_np).max(axis=1)
    max_offset_np = np.where(nonempty_np, np.maximum(max_offset_np, 0), -1)
    valid_ids_np = np.flatnonzero(nonempty_np)

    if valid_ids_np.size == 0:
        empty_idx = np.zeros((0, 3), dtype=np.int32)
        empty_score = np.zeros((0,), dtype=np.float32)
        return empty_idx, empty_score, {
            "device": str(device),
            "backend": backend,
            "num_input_gaussians": n,
            "num_splatted_gaussians": 0,
            "surface_splat_mode": surface_mode,
            "normal_support_voxels": normal_support,
            "thin_min_anisotropy": min_anisotropy,
            "num_thin_enabled_gaussians": num_thin_enabled,
            "logical_dense_accumulator_voxels": num_voxels,
            "logical_dense_accumulator_bytes": dense_accumulator_bytes,
            "dense_accumulator_allocated": False,
            "elapsed_sec": float(time.perf_counter() - started_at),
        }

    if reset_peak_memory_stats:
        torch.cuda.reset_peak_memory_stats(device)
    positions = torch.as_tensor(positions_np, dtype=torch.float32, device=device)
    rotations = torch.as_tensor(rotations_np, dtype=torch.float32, device=device)
    scales = torch.as_tensor(scales_np, dtype=torch.float32, device=device).clamp_min_(1e-8)
    opacities = torch.as_tensor(opacities_np, dtype=torch.float32, device=device).clamp_(0.0, 1.0)
    thin_enabled = torch.as_tensor(thin_enabled_np, dtype=torch.bool, device=device)
    vmin = torch.as_tensor(vmin_np, dtype=torch.int32, device=device)
    vmax = torch.as_tensor(vmax_np, dtype=torch.int32, device=device)
    center_idx = torch.as_tensor(center_idx_np, dtype=torch.int32, device=device)
    origin_t = torch.as_tensor(origin_np, dtype=torch.float32, device=device)
    dims_t = torch.as_tensor(dims_np, dtype=torch.int64, device=device)
    contribution_grid = (
        torch.zeros((num_voxels,), dtype=torch.float32, device=device)
        if use_dense_accumulator
        else None
    )
    sparse_pack_parts = []
    sparse_sum_parts = []

    chunk_limit = max(1, int(gaussian_chunk_size))
    max_pairs = max(1, int(max_pairs_per_batch))
    bucket_bounds = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    max_radius = int(max_offset_np[valid_ids_np].max())
    while bucket_bounds[-1] < max_radius:
        bucket_bounds.append(bucket_bounds[-1] * 2)

    num_batches = 0
    total_pair_slots = 0
    total_aabb_samples = 0
    total_normal_rejected_samples = 0
    total_valid_samples = 0
    prev_bound = -1
    with torch.no_grad():
        for bound in bucket_bounds:
            ids_np = valid_ids_np[
                (max_offset_np[valid_ids_np] > prev_bound)
                & (max_offset_np[valid_ids_np] <= bound)
            ]
            prev_bound = bound
            if ids_np.size == 0:
                continue
            offsets = _offset_grid(torch, bound, device)
            offset_count = int(offsets.shape[0])
            current_chunk_size = max(1, min(chunk_limit, max_pairs // max(offset_count, 1)))

            for start in range(0, int(ids_np.size), current_chunk_size):
                gids = torch.as_tensor(
                    ids_np[start : start + current_chunk_size],
                    dtype=torch.int64,
                    device=device,
                )
                candidate = center_idx.index_select(0, gids)[:, None, :] + offsets[None, :, :]
                vmin_g = vmin.index_select(0, gids)[:, None, :]
                vmax_g = vmax.index_select(0, gids)[:, None, :]
                nonzero = torch.nonzero(
                    torch.all((candidate >= vmin_g) & (candidate <= vmax_g), dim=2),
                    as_tuple=False,
                )
                num_batches += 1
                total_pair_slots += int(candidate.shape[0] * candidate.shape[1])
                total_aabb_samples += int(nonzero.shape[0])
                if int(nonzero.shape[0]) == 0:
                    continue

                local_gaussian = nonzero[:, 0]
                idx = candidate[local_gaussian, nonzero[:, 1], :]
                source_gids = gids.index_select(0, local_gaussian)
                centers = origin_t.unsqueeze(0) + (idx.to(dtype=torch.float32) + 0.5) * voxel_size_f
                rotation = rotations.index_select(0, source_gids)
                scale = scales.index_select(0, source_gids)
                opacity = opacities.index_select(0, source_gids)
                local = torch.einsum(
                    "nji,nj->ni",
                    rotation,
                    centers - positions.index_select(0, source_gids),
                )
                if surface_mode == "thin":
                    thin_pair = thin_enabled.index_select(0, source_gids)
                    if bool(torch.any(thin_pair).item()):
                        normal_axis = torch.argmin(scale, dim=1, keepdim=True)
                        normal_distance = torch.gather(torch.abs(local), 1, normal_axis).reshape(-1)
                        normal_keep = (~thin_pair) | (
                            normal_distance <= normal_support * voxel_size_f
                        )
                        total_normal_rejected_samples += int(
                            torch.count_nonzero(thin_pair & (~normal_keep)).item()
                        )
                        if not bool(torch.any(normal_keep).item()):
                            continue
                        idx = idx[normal_keep]
                        rotation = rotation[normal_keep]
                        scale = scale[normal_keep]
                        opacity = opacity[normal_keep]
                        local = local[normal_keep]
                if voxel_half_margin > 0.0:
                    margin = voxel_half_margin * torch.sum(torch.abs(rotation), dim=1)
                    local = torch.clamp(torch.abs(local) - margin, min=0.0)
                contribution = opacity * torch.exp(
                    -0.5 * torch.sum(torch.square(local / scale), dim=1)
                )
                keep = contribution >= float(contribution_floor)
                if not bool(torch.any(keep).item()):
                    continue
                idx = idx[keep].to(dtype=torch.int64)
                contribution = contribution[keep].to(dtype=torch.float32)
                packed = idx[:, 0] + dims_t[0] * (idx[:, 1] + dims_t[1] * idx[:, 2])
                if use_dense_accumulator:
                    contribution_grid.scatter_add_(0, packed, contribution)
                else:
                    unique_pack, inverse = torch.unique(packed, return_inverse=True)
                    batch_sum = torch.zeros(
                        (unique_pack.shape[0],),
                        dtype=torch.float32,
                        device=device,
                    )
                    batch_sum.scatter_add_(0, inverse, contribution)
                    sparse_pack_parts.append(unique_pack)
                    sparse_sum_parts.append(batch_sum)
                total_valid_samples += int(contribution.shape[0])

        offsets = gids = candidate = vmin_g = vmax_g = nonzero = None
        local_gaussian = idx = source_gids = centers = rotation = scale = opacity = local = None
        thin_pair = normal_axis = normal_distance = normal_keep = margin = contribution = keep = packed = None
        scan_chunk_voxels = None
        if use_dense_accumulator:
            scan_chunk_voxels = 8_000_000
            occupied_parts = []
            num_contributing_voxels = 0
            for scan_start in range(0, num_voxels, scan_chunk_voxels):
                scan_end = min(num_voxels, scan_start + scan_chunk_voxels)
                score_chunk = contribution_grid[scan_start:scan_end]
                num_contributing_voxels += int(torch.count_nonzero(score_chunk).item())
                occupied_local = torch.nonzero(
                    score_chunk >= float(occupied_threshold),
                    as_tuple=False,
                ).reshape(-1)
                if occupied_local.numel() > 0:
                    occupied_parts.append(occupied_local + int(scan_start))
            occupied_pack = (
                torch.cat(occupied_parts, dim=0)
                if occupied_parts
                else torch.empty((0,), dtype=torch.int64, device=device)
            )
            occupied_score = contribution_grid.index_select(0, occupied_pack)
        elif sparse_pack_parts:
            all_pack = torch.cat(sparse_pack_parts, dim=0)
            all_sum = torch.cat(sparse_sum_parts, dim=0)
            unique_pack, inverse = torch.unique(all_pack, return_inverse=True)
            contribution_sum = torch.zeros(
                (unique_pack.shape[0],),
                dtype=torch.float32,
                device=device,
            )
            contribution_sum.scatter_add_(0, inverse, all_sum)
            occupied_mask = contribution_sum >= float(occupied_threshold)
            occupied_pack = unique_pack[occupied_mask]
            occupied_score = contribution_sum[occupied_mask]
            num_contributing_voxels = int(unique_pack.shape[0])
        else:
            occupied_pack = torch.empty((0,), dtype=torch.int64, device=device)
            occupied_score = torch.empty((0,), dtype=torch.float32, device=device)
            num_contributing_voxels = 0
        occupied_idx = _unpack_voxel_idx_torch(torch, occupied_pack, dims_t)

    # Only wait for the stream that owns this conversion. A device-wide
    # synchronize also waits for unrelated online-reconstruction work and
    # makes a background conversion inherit its latency.
    conversion_stream = _synchronize_current_cuda_stream(torch, device)
    stats = {
        "device": str(device),
        "cuda_stream_id": int(conversion_stream.cuda_stream),
        "cuda_stream_priority": int(conversion_stream.priority),
        "backend": backend,
        "num_input_gaussians": n,
        "num_splatted_gaussians": int(valid_ids_np.size),
        "max_offset_voxels": max_radius,
        "num_batches": int(num_batches),
        "total_pair_slots": int(total_pair_slots),
        "total_aabb_samples": int(total_aabb_samples),
        "total_normal_rejected_samples": int(total_normal_rejected_samples),
        "total_valid_samples": int(total_valid_samples),
        "num_contributing_voxels": num_contributing_voxels,
        "num_occupied_voxels": int(occupied_idx.shape[0]),
        "logical_dense_accumulator_voxels": num_voxels,
        "logical_dense_accumulator_bytes": dense_accumulator_bytes,
        "dense_accumulator_allocated": bool(use_dense_accumulator),
        "accumulator_budget_bytes": int(max_accumulator_bytes),
        "threshold_scan_chunk_voxels": scan_chunk_voxels,
        "occupied_sum_threshold": float(occupied_threshold),
        "voxel_footprint_scale": float(voxel_footprint_scale),
        "surface_splat_mode": surface_mode,
        "normal_support_voxels": normal_support,
        "thin_min_anisotropy": min_anisotropy,
        "num_thin_enabled_gaussians": num_thin_enabled,
        "thin_enabled_gaussian_ratio": float(num_thin_enabled / max(n, 1)),
        "elapsed_sec": float(time.perf_counter() - started_at),
        "max_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_cuda_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    return (
        occupied_idx.cpu().numpy().astype(np.int32),
        occupied_score.cpu().numpy().astype(np.float32),
        stats,
    )


def voxel_idx_to_world_center(
    voxel_idx: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    return origin[None, :] + (voxel_idx.astype(np.float32) + 0.5) * float(voxel_size)
