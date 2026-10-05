#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build an isolated nvblox TSDF/ESDF from rendered 3DGS depth.

The rendered depth can either already be projective camera-Z in training-world
units or be normalized to [0, 1]. Normalized depth is first denormalized with the
view's near/far range. The pipeline then matches loaded training poses to the
COLMAP reconstruction used by nvblox, estimates their Sim(3), and scales depth
into the same coordinate units as those COLMAP poses.

Before TSDF integration, the COLMAP reconstruction is rigidly rewritten into
an axis-aligned fusion frame while retaining its original length unit. This
prevents a slanted source voxel lattice from baking staircase artifacts into
walls; the final converter applies only the remaining uniform scale.

An optional geometry-filter switch applies the same indoor/outdoor 3DGS filter
as the sparse occupancy pipeline before the renderer is created.

Outputs are deliberately namespaced below one new root::

    <output-root>/
      absolute_depth/<COLMAP image name>.geometric.bin
      axis_aligned_colmap_sparse/{cameras,images,points3D}.txt
      nvblox_frame_alignment.json
      sdf/{tsdf_observed.ply,esdf_observed.ply,...}
      depth_manifest.json
      voxel_map_tsdf.{npz,json}                 # unless skipped

The normal scene ``sdf/`` and ``voxel_map_tsdf`` are therefore never reused.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GSPAR.mapping.gaussian_occupancy.build_sparse_voxel_map import (
    filter_gaussians_geometry_aware,
)
from GSPAR.mapping.gaussian_occupancy.voxel_preprocess import GaussianSet
from GSPAR.mapping.map_frame_transform import apply_transform, load_map_frame_transform


DEFAULT_NVBLOX_BINARY = (
    PROJECT_ROOT / "GSPAR" / "nvblox_gpu_sdf" / "build" / "nvblox_gpu_sdf_offline"
)
DEPTH_SAMPLE_STRIDE = 24
MAX_DEPTH_PERCENTILE = 90.0
MINIMUM_MAX_DEPTH_M = 20.0
MAX_DEPTH_ROUND_UP_M = 5.0


@dataclass(frozen=True)
class PoseAlignment:
    """Sim(3) taking training-world coordinates to the nvblox COLMAP world."""

    source_per_training_unit: float
    rotation_source_from_training: np.ndarray
    translation_source_from_training: np.ndarray
    common_names: tuple[str, ...]
    center_rmse: float
    center_max_error: float
    center_spread: float
    rotation_mean_error_deg: float
    rotation_max_error_deg: float

    @property
    def source_from_training_affine(self) -> np.ndarray:
        affine = np.eye(4, dtype=np.float64)
        affine[:3, :3] = (
            float(self.source_per_training_unit)
            * np.asarray(self.rotation_source_from_training, dtype=np.float64)
        )
        affine[:3, 3] = np.asarray(
            self.translation_source_from_training, dtype=np.float64
        )
        return affine

    @property
    def training_from_source_affine(self) -> np.ndarray:
        return np.linalg.inv(self.source_from_training_affine)

    def as_manifest(self) -> dict[str, Any]:
        return {
            "source_per_training_unit": float(self.source_per_training_unit),
            "common_pose_count": len(self.common_names),
            "common_image_names": list(self.common_names),
            "center_rmse": float(self.center_rmse),
            "center_max_error": float(self.center_max_error),
            "center_spread": float(self.center_spread),
            "rotation_mean_error_deg": float(self.rotation_mean_error_deg),
            "rotation_max_error_deg": float(self.rotation_max_error_deg),
            "source_from_training_affine": self.source_from_training_affine.tolist(),
            "source_to_training_affine": self.training_from_source_affine.tolist(),
        }


def _write_axis_aligned_colmap_sparse(
    source_sparse: Path,
    output_sparse: Path,
    source_to_training_affine: np.ndarray,
    map_from_training_world: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rewrite COLMAP into the final map axes before TSDF integration."""

    import pycolmap

    source_to_training = np.asarray(source_to_training_affine, dtype=np.float64)
    map_from_training = np.asarray(map_from_training_world, dtype=np.float64)
    map_from_source = map_from_training @ source_to_training
    u, singular_values, vh = np.linalg.svd(map_from_source[:3, :3])
    rotation = u @ vh
    scale = float(np.mean(singular_values))
    if (
        scale <= 0.0
        or np.linalg.det(rotation) <= 0.0
        or not np.allclose(singular_values, scale, rtol=1.0e-5, atol=0.0)
    ):
        raise ValueError(
            "source-to-map transform must be an orientation-preserving similarity"
        )

    nvblox_from_source = np.eye(4, dtype=np.float64)
    nvblox_from_source[:3, :3] = rotation
    nvblox_from_source[:3, 3] = map_from_source[:3, 3] / scale
    training_from_nvblox = (
        source_to_training @ np.linalg.inv(nvblox_from_source)
    )
    map_from_nvblox = map_from_training @ training_from_nvblox
    if (
        not np.allclose(
            map_from_nvblox[:3, :3] / scale,
            np.eye(3),
            rtol=1.0e-5,
            atol=1.0e-5,
        )
        or not np.allclose(map_from_nvblox[:3, 3], 0.0, atol=1.0e-7)
    ):
        raise RuntimeError("Failed to remove rotation before nvblox integration")

    reconstruction = pycolmap.Reconstruction(source_sparse)
    num_images = int(reconstruction.num_images())
    num_points3d = int(reconstruction.num_points3D())
    if num_images <= 0:
        raise ValueError(f"COLMAP reconstruction has no images: {source_sparse}")
    reconstruction.transform(pycolmap.Sim3d(nvblox_from_source[:3, :]))
    output_sparse.mkdir(parents=True, exist_ok=False)
    reconstruction.write_text(output_sparse)
    return training_from_nvblox, {
        "map_units_per_source_unit": scale,
        "nvblox_from_source_affine": nvblox_from_source.tolist(),
        "output": str(output_sparse),
        "num_images": num_images,
        "num_points3D": num_points3d,
    }


@dataclass(frozen=True)
class MaxDepthStats:
    """Statistics matching ``nvblox_gpu_sdf_offline`` adaptive max depth."""

    selected_max_depth_m: float
    percentile_depth_m: float
    valid_sample_count: int
    fallback_used: bool
    percentile: float = MAX_DEPTH_PERCENTILE
    sample_stride: int = DEPTH_SAMPLE_STRIDE
    minimum_max_depth_m: float = MINIMUM_MAX_DEPTH_M
    round_up_multiple_m: float = MAX_DEPTH_ROUND_UP_M

    def as_manifest(self) -> dict[str, Any]:
        return {
            "mode": "explicit_precomputed",
            "selected_max_depth_m": float(self.selected_max_depth_m),
            "percentile_depth_m": float(self.percentile_depth_m),
            "valid_sample_count": int(self.valid_sample_count),
            "fallback_used": bool(self.fallback_used),
            "percentile": float(self.percentile),
            "sample_stride": int(self.sample_stride),
            "minimum_max_depth_m": float(self.minimum_max_depth_m),
            "round_up_multiple_m": float(self.round_up_multiple_m),
        }


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    value = float(
        np.clip((np.trace(np.asarray(rotation, dtype=np.float64)) - 1.0) * 0.5, -1.0, 1.0)
    )
    return math.degrees(math.acos(value))


def _validate_pose(name: str, pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape == (3, 4):
        pose = np.vstack([pose, np.asarray([0.0, 0.0, 0.0, 1.0])])
    if pose.shape != (4, 4):
        raise ValueError(f"Pose {name!r} must be 3x4 or 4x4, got {pose.shape}.")
    if not np.isfinite(pose).all():
        raise ValueError(f"Pose {name!r} contains non-finite values.")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-6, rtol=0.0):
        raise ValueError(f"Pose {name!r} has an invalid homogeneous last row.")
    return pose


def estimate_source_pose_alignment(
    training_poses: Mapping[str, np.ndarray],
    source_poses: Mapping[str, np.ndarray],
    *,
    max_center_rmse_m: float | None = None,
    max_rotation_error_deg: float | None = None,
) -> PoseAlignment:
    """Estimate ``source = scale * R * training + t`` from matched cameras.

    Camera centers determine the Umeyama similarity. Camera rotations are an
    independent validation that the matching names and coordinate conventions
    describe one global Sim(3).
    """

    common_names = tuple(sorted(set(training_poses) & set(source_poses)))
    if len(common_names) < 3:
        raise ValueError(
            "At least three common camera poses are required to recover source "
            f"scale, got {len(common_names)}."
        )

    training = [_validate_pose(name, training_poses[name]) for name in common_names]
    source = [_validate_pose(name, source_poses[name]) for name in common_names]
    src = np.stack([pose[:3, 3] for pose in training], axis=0)
    dst = np.stack([pose[:3, 3] for pose in source], axis=0)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    src_variance = float(np.mean(np.sum(src_centered * src_centered, axis=1)))
    if not np.isfinite(src_variance) or src_variance <= 1.0e-15:
        raise ValueError("Training camera centers have no usable baseline for scale recovery.")

    covariance = (dst_centered.T @ src_centered) / float(len(common_names))
    u, singular_values, vh = np.linalg.svd(covariance)
    sign = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vh) < 0.0:
        sign[-1, -1] = -1.0
    rotation = u @ sign @ vh
    scale = float(np.trace(np.diag(singular_values) @ sign) / src_variance)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Recovered source pose scale must be positive, got {scale}.")
    translation = dst_mean - scale * (rotation @ src_mean)

    fitted = (scale * (rotation @ src.T)).T + translation[None, :]
    center_errors = np.linalg.norm(fitted - dst, axis=1)
    center_rmse = float(np.sqrt(np.mean(center_errors * center_errors)))
    center_max = float(np.max(center_errors))
    center_spread = float(
        np.sqrt(np.mean(np.sum(dst_centered * dst_centered, axis=1)))
    )

    rotation_errors = []
    for training_pose, source_pose in zip(training, source):
        predicted_source_rotation = rotation @ training_pose[:3, :3]
        rotation_errors.append(
            _rotation_angle_deg(predicted_source_rotation.T @ source_pose[:3, :3])
        )
    rotation_errors_np = np.asarray(rotation_errors, dtype=np.float64)
    rotation_mean = float(np.mean(rotation_errors_np))
    rotation_max = float(np.max(rotation_errors_np))

    if max_center_rmse_m is not None and center_rmse > float(max_center_rmse_m):
        raise ValueError(
            "Training and source poses are not explained by one Sim(3): "
            f"center RMSE {center_rmse:.6g} exceeds {max_center_rmse_m:.6g}."
        )
    if (
        max_rotation_error_deg is not None
        and rotation_max > float(max_rotation_error_deg)
    ):
        raise ValueError(
            "Training and source poses have inconsistent rotations: "
            f"maximum error {rotation_max:.6g} deg exceeds "
            f"{max_rotation_error_deg:.6g} deg."
        )

    return PoseAlignment(
        source_per_training_unit=scale,
        rotation_source_from_training=rotation,
        translation_source_from_training=translation,
        common_names=common_names,
        center_rmse=center_rmse,
        center_max_error=center_max,
        center_spread=center_spread,
        rotation_mean_error_deg=rotation_mean,
        rotation_max_error_deg=rotation_max,
    )


def estimate_metric_pose_alignment(
    training_poses: Mapping[str, np.ndarray],
    metric_poses: Mapping[str, np.ndarray],
    *,
    max_center_rmse_m: float | None = None,
    max_rotation_error_deg: float | None = None,
) -> PoseAlignment:
    """Backward-compatible wrapper; the source poses need not be metric."""

    return estimate_source_pose_alignment(
        training_poses,
        metric_poses,
        max_center_rmse_m=max_center_rmse_m,
        max_rotation_error_deg=max_rotation_error_deg,
    )


def denormalize_projective_depth(
    normalized_depth: np.ndarray,
    *,
    near: float,
    far: float,
) -> np.ndarray:
    """Convert linear [0, 1] depth to projective camera-Z depth."""

    if not np.isfinite(near) or not np.isfinite(far) or far <= near:
        raise ValueError(f"Invalid depth normalization range: near={near}, far={far}.")
    depth = np.asarray(normalized_depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Depth must be HxW, got {depth.shape}.")
    valid = np.isfinite(depth) & (depth > 0.0)
    if np.any((depth[valid] < -1.0e-6) | (depth[valid] > 1.0 + 1.0e-6)):
        observed_min = float(np.min(depth[valid]))
        observed_max = float(np.max(depth[valid]))
        raise ValueError(
            "Normalized rendered depth must be in [0, 1], got "
            f"[{observed_min:.6g}, {observed_max:.6g}]. Use "
            "--rendered-depth-unit camera-z if the renderer already returns "
            "absolute camera-Z depth."
        )
    output = np.zeros(depth.shape, dtype=np.float32)
    output[valid] = (float(near) + np.clip(depth[valid], 0.0, 1.0) * (float(far) - float(near))).astype(np.float32)
    return output


def write_nvblox_dense_depth(path: Path, depth_m: np.ndarray) -> None:
    """Write the row-major float32 dense-depth format used by this nvblox CLI."""

    depth = np.asarray(depth_m)
    if depth.ndim != 2:
        raise ValueError(f"Depth must be HxW, got {depth.shape}.")
    if not np.issubdtype(depth.dtype, np.number):
        raise ValueError("Depth must contain numeric values.")
    height, width = depth.shape
    if height <= 0 or width <= 0:
        raise ValueError("Depth dimensions must be positive.")
    depth = np.asarray(depth, dtype=np.float32)
    if not np.isfinite(depth).all() or np.any(depth < 0.0):
        raise ValueError("Depth must contain only finite, non-negative values.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(f"{width}&{height}&1&".encode("ascii"))
        output.write(np.ascontiguousarray(depth, dtype="<f4").tobytes(order="C"))


def sample_depth_for_nvblox(
    depth_m: np.ndarray, *, stride: int = DEPTH_SAMPLE_STRIDE
) -> np.ndarray:
    """Return the same valid strided pixels sampled by the native CLI."""

    if int(stride) <= 0:
        raise ValueError(f"stride must be positive, got {stride}.")
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Depth must be HxW, got {depth.shape}.")
    sampled = depth[:: int(stride), :: int(stride)].reshape(-1)
    return sampled[np.isfinite(sampled) & (sampled > 0.0)]


def estimate_nvblox_max_depth(
    valid_samples_m: np.ndarray | Sequence[float],
    *,
    percentile: float = MAX_DEPTH_PERCENTILE,
    sample_stride: int = DEPTH_SAMPLE_STRIDE,
    minimum_m: float = MINIMUM_MAX_DEPTH_M,
    round_up_multiple_m: float = MAX_DEPTH_ROUND_UP_M,
) -> MaxDepthStats:
    """Reproduce the native nearest-rank p90/floor/rounding calculation."""

    if not 0.0 <= float(percentile) <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}.")
    if int(sample_stride) <= 0:
        raise ValueError("sample_stride must be positive.")
    if not np.isfinite(minimum_m) or float(minimum_m) <= 0.0:
        raise ValueError("minimum_m must be positive and finite.")
    if not np.isfinite(round_up_multiple_m) or float(round_up_multiple_m) <= 0.0:
        raise ValueError("round_up_multiple_m must be positive and finite.")

    samples = np.asarray(valid_samples_m, dtype=np.float32).reshape(-1)
    samples = samples[np.isfinite(samples) & (samples > 0.0)]
    if samples.size == 0:
        return MaxDepthStats(
            selected_max_depth_m=float(minimum_m),
            percentile_depth_m=0.0,
            valid_sample_count=0,
            fallback_used=True,
            percentile=float(percentile),
            sample_stride=int(sample_stride),
            minimum_max_depth_m=float(minimum_m),
            round_up_multiple_m=float(round_up_multiple_m),
        )

    rank = (float(percentile) / 100.0) * float(samples.size)
    index = int(math.ceil(rank)) - 1 if rank > 0.0 else 0
    index = min(max(index, 0), int(samples.size) - 1)
    percentile_depth = float(np.partition(samples.copy(), index)[index])
    rounded = (
        math.ceil(percentile_depth / float(round_up_multiple_m))
        * float(round_up_multiple_m)
    )
    selected = max(float(minimum_m), float(rounded))
    return MaxDepthStats(
        selected_max_depth_m=selected,
        percentile_depth_m=percentile_depth,
        valid_sample_count=int(samples.size),
        fallback_used=False,
        percentile=float(percentile),
        sample_stride=int(sample_stride),
        minimum_max_depth_m=float(minimum_m),
        round_up_multiple_m=float(round_up_multiple_m),
    )


def build_nvblox_command(
    *,
    executable: Path,
    colmap_sparse: Path,
    depth_dir: Path,
    output_dir: Path,
    source_voxel_size: float,
    truncation_distance: float,
    max_depth: float,
    max_esdf_distance: float,
) -> list[str]:
    """Build the native command; ``--max_depth`` is intentionally mandatory."""

    return [
        str(executable),
        "--colmap_sparse",
        str(colmap_sparse),
        "--depth_dir",
        str(depth_dir),
        "--output_dir",
        str(output_dir),
        "--voxel_size",
        f"{float(source_voxel_size):.9g}",
        "--truncation_distance",
        f"{float(truncation_distance):.9g}",
        "--max_depth",
        f"{float(max_depth):.9g}",
        "--max_esdf_distance",
        f"{float(max_esdf_distance):.9g}",
    ]


def _resolve_path(path: Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _image_key(name: str) -> str:
    return Path(str(name).replace("\\", "/")).name.casefold()


def _safe_depth_relative_path(image_name: str) -> Path:
    relative = Path(str(image_name).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or not relative.name:
        raise ValueError(f"Unsafe COLMAP image name: {image_name!r}.")
    return Path(f"{relative}.geometric.bin")


def _tensor_to_hw(tensor: Any, name: str) -> np.ndarray:
    array = tensor.detach().float().cpu().numpy()
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0]
    if array.ndim == 3 and array.shape[-1] == 1:
        return array[..., 0]
    raise ValueError(f"Rendered {name} must be HxW, 1xHxW, or HxWx1, got {array.shape}.")


def _get_view_image_name(view: Any) -> str:
    parameters, _ = view.get_parallel_load_helpers("rgb")
    return Path(parameters["path"]).name


def _all_dataset_views(dataset: Any) -> list[Any]:
    views: dict[int, Any] = {}
    for subset in dataset.subsets:
        for view in dataset.data[subset]:
            key = int(view.global_frame_idx)
            if key in views and views[key] is not view:
                raise ValueError(f"Duplicate dataset global_frame_idx: {key}.")
            views[key] = view
    return [views[key] for key in sorted(views)]


def _load_colmap_poses(colmap_sparse: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    import pycolmap

    reconstruction = pycolmap.Reconstruction(colmap_sparse)
    poses: dict[str, np.ndarray] = {}
    original_names: dict[str, str] = {}
    for image in reconstruction.images.values():
        key = _image_key(image.name)
        if key in poses:
            raise ValueError(
                f"COLMAP reconstruction contains duplicate image basename {key!r}; "
                "basename matching would be ambiguous."
            )
        poses[key] = _validate_pose(
            str(image.name), image.cam_from_world().inverse().matrix()
        )
        original_names[key] = str(image.name)
    if not poses:
        raise ValueError(f"COLMAP reconstruction has no images: {colmap_sparse}.")
    return poses, original_names


def _load_metric_colmap_poses(metric_sparse: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Backward-compatible wrapper around the source-coordinate pose loader."""

    return _load_colmap_poses(metric_sparse)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _validate_output_root(output_root: Path) -> None:
    if output_root.is_file():
        raise FileExistsError(f"Output root is a file: {output_root}.")
    if output_root.is_dir() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output root must be new or empty to avoid mixing maps: {output_root}."
        )


def _retain_model_gaussians(gaussians: Any, keep_mask: np.ndarray) -> dict[str, Any]:
    """Slice every foreground per-Gaussian tensor with one shared keep mask."""

    import torch

    positions = getattr(gaussians, "get_positions", None)
    if not isinstance(positions, torch.Tensor) or positions.ndim < 1:
        raise TypeError("Loaded model does not expose tensor-valued gaussians.get_positions.")
    input_count = int(positions.shape[0])
    keep = np.asarray(keep_mask, dtype=bool).reshape(-1)
    if keep.shape != (input_count,):
        raise ValueError(
            "Geometry-filter mask length does not match the loaded model: "
            f"{keep.shape[0]} vs {input_count}."
        )
    output_count = int(np.count_nonzero(keep))
    if output_count <= 0:
        raise RuntimeError("Geometry filtering cannot remove every Gaussian.")
    if getattr(gaussians, "optimizer", None) is not None:
        raise RuntimeError(
            "Rendered-depth geometry filtering expects an inference-only model "
            "without an attached Gaussian optimizer."
        )

    keep_t = torch.as_tensor(keep, dtype=torch.bool, device=positions.device)
    sliced_parameters = [
        name
        for name, parameter in gaussians._parameters.items()
        if parameter is not None
        and parameter.ndim >= 1
        and int(parameter.shape[0]) == input_count
    ]
    for name in sliced_parameters:
        parameter = gaussians._parameters[name]
        selected = parameter.detach()[keep_t].contiguous()
        # This is an inference-only module without an optimizer. Reusing each
        # Parameter object avoids retaining a second full model while slicing.
        parameter.data = selected

    required = {"_positions", "_scales", "_rotations", "_opacities"}
    missing = sorted(required - set(sliced_parameters))
    if missing:
        raise RuntimeError(
            "Loaded model has unsupported Gaussian storage; failed to slice "
            f"required parameters: {missing}."
        )

    sliced_buffers = [
        name
        for name, buffer in gaussians._buffers.items()
        if buffer is not None
        and buffer.ndim >= 1
        and int(buffer.shape[0]) == input_count
    ]
    for name in sliced_buffers:
        buffer = gaussians._buffers[name]
        setattr(gaussians, name, buffer.detach()[keep_t].contiguous())

    # HTGS keeps these inference/training auxiliaries as regular attributes,
    # not registered buffers.  Keep them consistent when present.
    for name in ("filter_3D",):
        value = getattr(gaussians, name, None)
        if (
            isinstance(value, torch.Tensor)
            and value.ndim >= 1
            and int(value.shape[0]) == input_count
        ):
            setattr(gaussians, name, value.detach()[keep_t].contiguous())
            sliced_buffers.append(name)
    densification_info = getattr(gaussians, "densification_info", None)
    if (
        isinstance(densification_info, torch.Tensor)
        and densification_info.ndim >= 2
        and int(densification_info.shape[1]) == input_count
    ):
        gaussians.densification_info = densification_info.detach()[:, keep_t].contiguous()
        sliced_buffers.append("densification_info")

    if int(gaussians.get_positions.shape[0]) != output_count:
        raise RuntimeError("Gaussian parameter slicing produced an inconsistent model.")
    return {
        "sliced_parameters": sorted(sliced_parameters),
        "sliced_auxiliary_tensors": sorted(set(sliced_buffers)),
    }


def _apply_render_geometry_filter(
    model: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Compute the sparse-map geometry mask, then apply it to the render model."""

    started = time.perf_counter()
    gaussians = getattr(model, "gaussians", None)
    if gaussians is None:
        raise TypeError("Loaded model does not expose a foreground 'gaussians' module.")
    positions_t = getattr(gaussians, "get_positions", None)
    if positions_t is None:
        raise TypeError("Loaded model does not expose gaussians.get_positions.")
    input_count = int(positions_t.shape[0])
    enabled = bool(args.geometry_filter)
    if not enabled:
        return {
            "enabled": False,
            "scene_type": str(args.scene_type),
            "method": "opacity_density_isolation_boundary_filter",
            "input_gaussians": input_count,
            "num_output_gaussians": input_count,
            "num_removed_total": 0,
            "retention_ratio": 1.0,
            "local_geometry_method": "skipped",
            "elapsed_sec": float(time.perf_counter() - started),
        }

    map_frame_path = (
        _resolve_path(args.map_frame_transform)
        if args.map_frame_transform is not None
        else None
    )
    map_transform, map_transform_info = load_map_frame_transform(map_frame_path)
    use_3d_filter = bool(getattr(gaussians, "use_3d_filter", False))
    scales_t = (
        gaussians.get_scales_with_3D_filter
        if use_3d_filter
        else gaussians.get_scales
    )
    opacities_t = (
        gaussians.get_opacities_with_3D_filter
        if use_3d_filter
        else gaussians.get_opacities
    )
    rotations_t = gaussians.get_rotations
    filter_input = GaussianSet(
        positions=apply_transform(
            positions_t.detach().float().cpu().numpy(),
            map_transform,
        ),
        scales=scales_t.detach().float().cpu().numpy(),
        rotations=rotations_t.detach().float().cpu().numpy(),
        opacities=opacities_t.detach().float().cpu().numpy().reshape(-1),
    )
    _, keep_mask, stats = filter_gaussians_geometry_aware(
        filter_input,
        scene_type=str(args.scene_type),
        voxel_size=float(args.voxel_size),
        enabled=True,
    )
    tensor_stats = _retain_model_gaussians(gaussians, keep_mask)
    output_count = int(np.count_nonzero(keep_mask))
    stats.update(
        {
            "map_frame": map_transform_info,
            "model_used_3d_filter": use_3d_filter,
            "retention_ratio": output_count / max(input_count, 1),
            "elapsed_sec": float(time.perf_counter() - started),
            **tensor_stats,
        }
    )
    return stats


def _render_absolute_depths(
    args: argparse.Namespace,
    *,
    training_config: Path,
    checkpoint: Path,
    colmap_sparse: Path,
    depth_dir: Path,
) -> tuple[PoseAlignment, MaxDepthStats, dict[str, Any]]:
    """Load NerfICG lazily, render depth, denormalize/scale it, and write files."""

    if str(PROJECT_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))

    import Framework
    from Implementations import Datasets as DI
    from Implementations import Methods as MI

    Framework.setup(config_path=str(training_config), require_custom_config=True)
    render_started = time.perf_counter()
    model = None
    renderer = None
    dataset = None
    try:
        configured_dataset_path = Path(str(Framework.config.DATASET.PATH)).expanduser()
        if not configured_dataset_path.is_absolute():
            configured_dataset_path = PROJECT_ROOT / configured_dataset_path
        dataset = DI.get_dataset(
            dataset_type=Framework.config.GLOBAL.DATASET_TYPE,
            path=str(configured_dataset_path.resolve()),
        )
        views = _all_dataset_views(dataset)
        training_poses: dict[str, np.ndarray] = {}
        views_by_key: dict[str, Any] = {}
        for view in views:
            image_name = _get_view_image_name(view)
            key = _image_key(image_name)
            if key in training_poses:
                raise ValueError(
                    f"Training dataset contains duplicate image basename {key!r}."
                )
            training_poses[key] = np.asarray(view.c2w_numpy, dtype=np.float64)
            views_by_key[key] = view

        source_poses, source_names = _load_colmap_poses(colmap_sparse)
        alignment = estimate_source_pose_alignment(
            training_poses,
            source_poses,
            max_center_rmse_m=float(args.max_pose_rmse_m),
            max_rotation_error_deg=float(args.max_pose_rotation_error_deg),
        )

        selected_keys = [
            _image_key(_get_view_image_name(view))
            for view in views
            if _image_key(_get_view_image_name(view)) in source_poses
        ]
        if args.max_frames is not None:
            selected_keys = selected_keys[: int(args.max_frames)]
        if not selected_keys:
            raise ValueError("No matched views remain for depth rendering.")

        model_renderer_load_started = time.perf_counter()
        model_load_started = time.perf_counter()
        model = MI.get_model(
            method=Framework.config.GLOBAL.METHOD_TYPE,
            checkpoint=str(checkpoint),
        ).eval()
        model_load_sec = float(time.perf_counter() - model_load_started)
        geometry_filter_stats = _apply_render_geometry_filter(model, args)
        renderer_setup_started = time.perf_counter()
        renderer = MI.get_renderer(
            method=Framework.config.GLOBAL.METHOD_TYPE,
            model=model,
        )
        if hasattr(renderer, "FORCE_OPTIMIZED_INFERENCE"):
            renderer.FORCE_OPTIMIZED_INFERENCE = False
        if args.depth_mode != "config":
            if not hasattr(renderer, "USE_MEDIAN_DEPTH"):
                raise RuntimeError(
                    f"Renderer {type(renderer).__name__} cannot select expected/median depth."
                )
            renderer.USE_MEDIAN_DEPTH = args.depth_mode == "median"
        effective_depth_mode = (
            "median"
            if bool(getattr(renderer, "USE_MEDIAN_DEPTH", False))
            else "expected"
        )
        renderer_setup_sec = float(time.perf_counter() - renderer_setup_started)
        model_renderer_load_sec = float(
            time.perf_counter() - model_renderer_load_started
        )

        sampled_chunks: list[np.ndarray] = []
        valid_pixel_count = 0
        total_pixel_count = 0
        min_depth = math.inf
        max_depth = 0.0
        frame_records = []
        render_inference_sec = 0.0
        depth_postprocess_write_sec = 0.0
        for key in selected_keys:
            view = views_by_key[key]
            inference_started = time.perf_counter()
            outputs = renderer.render_image(view, to_chw=False, benchmark=False)
            render_inference_sec += float(time.perf_counter() - inference_started)
            if "depth" not in outputs or outputs["depth"] is None:
                raise RuntimeError(
                    f"Renderer {type(renderer).__name__} does not expose raw depth; "
                    "this pipeline requires HTGS or HTGS_egsr camera-Z depth."
                )
            postprocess_started = time.perf_counter()
            rendered_depth = _tensor_to_hw(outputs["depth"], "depth")
            valid = np.isfinite(rendered_depth) & (rendered_depth > 0.0)
            alpha = outputs.get("alpha")
            if alpha is None and float(args.min_alpha) > 0.0:
                raise RuntimeError(
                    f"Renderer {type(renderer).__name__} has no alpha output, but "
                    f"--min-alpha={args.min_alpha} was requested."
                )
            if alpha is not None and float(args.min_alpha) > 0.0:
                alpha_hw = _tensor_to_hw(alpha, "alpha")
                if alpha_hw.shape != rendered_depth.shape:
                    raise ValueError(
                        f"Rendered alpha/depth shapes differ: {alpha_hw.shape} vs "
                        f"{rendered_depth.shape}."
                    )
                valid &= np.isfinite(alpha_hw) & (alpha_hw >= float(args.min_alpha))

            if args.rendered_depth_unit == "normalized":
                depth_training = denormalize_projective_depth(
                    np.where(valid, rendered_depth, 0.0),
                    near=float(view.camera.near_plane),
                    far=float(view.camera.far_plane),
                )
            else:
                depth_training = np.zeros(rendered_depth.shape, dtype=np.float32)
                depth_training[valid] = rendered_depth[valid].astype(np.float32)

            depth_source = np.zeros(depth_training.shape, dtype=np.float32)
            depth_source[valid] = (
                depth_training[valid].astype(np.float64)
                * float(alignment.source_per_training_unit)
            ).astype(np.float32)
            finite_source = np.isfinite(depth_source) & (depth_source > 0.0)
            depth_source[~finite_source] = 0.0
            frame_valid = int(np.count_nonzero(finite_source))
            if frame_valid:
                values = depth_source[finite_source]
                min_depth = min(min_depth, float(np.min(values)))
                max_depth = max(max_depth, float(np.max(values)))
            valid_pixel_count += frame_valid
            total_pixel_count += int(depth_source.size)
            sampled = sample_depth_for_nvblox(depth_source)
            if sampled.size:
                sampled_chunks.append(sampled)

            source_name = source_names[key]
            relative_output = _safe_depth_relative_path(source_name)
            depth_path = depth_dir / relative_output
            write_nvblox_dense_depth(depth_path, depth_source)
            depth_postprocess_write_sec += float(time.perf_counter() - postprocess_started)
            frame_records.append(
                {
                    "image_name": source_name,
                    "depth_path": str(depth_path),
                    "width": int(depth_source.shape[1]),
                    "height": int(depth_source.shape[0]),
                    "valid_pixels": frame_valid,
                    "near": float(view.camera.near_plane),
                    "far": float(view.camera.far_plane),
                }
            )

        if valid_pixel_count == 0:
            raise RuntimeError("All rendered absolute depth maps are empty after filtering.")
        samples = (
            np.concatenate(sampled_chunks)
            if sampled_chunks
            else np.empty((0,), dtype=np.float32)
        )
        max_depth_stats = estimate_nvblox_max_depth(samples)
        render_stats = {
            "renderer": type(renderer).__name__,
            "method_type": str(Framework.config.GLOBAL.METHOD_TYPE),
            "depth_semantics": "projective_camera_z",
            "rendered_depth_unit": str(args.rendered_depth_unit),
            "depth_unit": "colmap_source_coordinate_unit",
            "depth_mode": effective_depth_mode,
            "min_alpha": float(args.min_alpha),
            "geometry_filter": geometry_filter_stats,
            "frames_rendered": len(frame_records),
            "valid_pixels": valid_pixel_count,
            "total_pixels": total_pixel_count,
            "valid_fraction": valid_pixel_count / max(total_pixel_count, 1),
            "minimum_valid_depth": float(min_depth),
            "maximum_valid_depth": float(max_depth),
            "frame_records": frame_records,
            "timing_sec": {
                "dataset_pose_alignment": float(
                    model_renderer_load_started - render_started
                ),
                "model_load": model_load_sec,
                "geometry_filter": float(
                    geometry_filter_stats["elapsed_sec"]
                ),
                "renderer_setup": renderer_setup_sec,
                "model_renderer_load": model_renderer_load_sec,
                "render_inference": render_inference_sec,
                "depth_postprocess_and_write": depth_postprocess_write_sec,
                "render_total": float(time.perf_counter() - render_started),
            },
            # Kept for consumers of manifests written before stage timing existed.
            "render_sec": float(time.perf_counter() - render_started),
        }
        return alignment, max_depth_stats, render_stats
    finally:
        # The last render dictionary still owns CUDA tensors until this frame
        # exits. Drop those references before launching the separate nvblox
        # process so both stages do not unnecessarily compete for GPU memory.
        outputs = None
        alpha = None
        view = None
        renderer = None
        model = None
        dataset = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        Framework.teardown()


def _build_voxel_map_command(
    args: argparse.Namespace,
    *,
    checkpoint: Path,
    training_config: Path,
    sdf_dir: Path,
    output_root: Path,
    frame_alignment_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "GSPAR" / "mapping" / "build_nvblox_tsdf_voxel_map.py"),
        "--tsdf-ply",
        str(sdf_dir / "tsdf_observed.ply"),
        "--grid-checkpoint",
        str(checkpoint),
        "--training-config",
        str(training_config),
        "--source-voxel-size",
        f"{float(args.source_voxel_size):.9g}",
        "--voxel-size",
        f"{float(args.voxel_size):.9g}",
        "--block-size",
        str(int(args.block_size)),
        "--scene-padding",
        f"{float(args.scene_padding):.9g}",
        "--source-to-training-affine",
        str(frame_alignment_path),
        "--output-prefix",
        str(output_root / "voxel_map_tsdf"),
    ]
    if args.map_frame_transform is not None:
        command += ["--map-frame-transform", str(_resolve_path(args.map_frame_transform))]
    return command


def _validate_args(args: argparse.Namespace) -> None:
    positive_values = {
        "source_voxel_size": args.source_voxel_size,
        "voxel_size": args.voxel_size,
        "truncation_distance": args.truncation_distance,
        "max_esdf_distance": args.max_esdf_distance,
        "max_pose_rmse_m": args.max_pose_rmse_m,
        "max_pose_rotation_error_deg": args.max_pose_rotation_error_deg,
    }
    for name, value in positive_values.items():
        if not np.isfinite(value) or float(value) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite.")
    if not 0.0 <= float(args.min_alpha) <= 1.0:
        raise ValueError("--min-alpha must be in [0, 1].")
    if args.max_frames is not None and int(args.max_frames) <= 0:
        raise ValueError("--max-frames must be positive.")
    if int(args.block_size) <= 0:
        raise ValueError("--block-size must be positive.")
    if not np.isfinite(args.scene_padding) or float(args.scene_padding) < 0.0:
        raise ValueError("--scene-padding must be non-negative and finite.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render 3DGS depth, denormalize it when needed, "
            "and build an isolated nvblox TSDF/ESDF."
        )
    )
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--colmap-sparse",
        "--metric-sparse",
        dest="colmap_sparse",
        type=Path,
        required=True,
        help=(
            "COLMAP sparse model used by nvblox. Its pose units define the "
            "absolute depth units written by this pipeline."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New/empty isolated namespace root for this map variant.",
    )
    parser.add_argument(
        "--nvblox-binary", type=Path, default=DEFAULT_NVBLOX_BINARY
    )
    parser.add_argument("--source-voxel-size", type=float, required=True)
    parser.add_argument("--voxel-size", type=float, required=True)
    parser.add_argument("--truncation-distance", type=float, default=0.20)
    parser.add_argument("--max-esdf-distance", type=float, default=5.0)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--scene-padding", type=float, default=0.02)
    parser.add_argument("--map-frame-transform", type=Path, default=None)
    geometry_filter_group = parser.add_mutually_exclusive_group()
    geometry_filter_group.add_argument(
        "--enable-geometry-filter",
        "--geometry-filter",
        dest="geometry_filter",
        action="store_true",
        help=(
            "Filter foreground Gaussians with the sparse-map geometry filter "
            "before creating the renderer."
        ),
    )
    geometry_filter_group.add_argument(
        "--disable-geometry-filter",
        dest="geometry_filter",
        action="store_false",
        help="Render the original checkpoint without geometry filtering (default).",
    )
    parser.set_defaults(geometry_filter=False)
    parser.add_argument(
        "--scene-type",
        choices=("indoor", "outdoor"),
        default="outdoor",
        help=(
            "Select the indoor/outdoor thresholds used by the shared geometry "
            "filter."
        ),
    )
    parser.add_argument(
        "--depth-mode",
        choices=("config", "expected", "median"),
        default="config",
        help="Keep the renderer config or explicitly select expected/median depth.",
    )
    parser.add_argument(
        "--rendered-depth-unit",
        choices=("normalized", "camera-z"),
        default="normalized",
        help=(
            "Interpret renderer depth as linear [0, 1] depth to be "
            "denormalized by near/far, or as already-absolute camera-Z depth."
        ),
    )
    parser.add_argument(
        "--min-alpha",
        type=float,
        default=0.0,
        help="Reject rendered pixels below this alpha when alpha is available.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-pose-rmse-m", type=float, default=0.02)
    parser.add_argument(
        "--max-pose-rotation-error-deg", type=float, default=0.25
    )
    parser.add_argument(
        "--skip-voxel-map",
        action="store_true",
        help="Keep raw TSDF/ESDF only; do not build canonical voxel_map_tsdf.",
    )
    return parser.parse_args(argv)


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    training_config = _resolve_path(args.training_config)
    checkpoint = _resolve_path(args.checkpoint)
    colmap_sparse = _resolve_path(args.colmap_sparse)
    nvblox_binary = _resolve_path(args.nvblox_binary)
    output_root = _resolve_path(args.output_root)
    map_frame_path = (
        _resolve_path(args.map_frame_transform)
        if args.map_frame_transform is not None
        else None
    )
    map_transform, map_transform_info = load_map_frame_transform(map_frame_path)
    for path, description in (
        (training_config, "training config"),
        (checkpoint, "checkpoint"),
        (colmap_sparse, "COLMAP sparse model"),
        (nvblox_binary, "nvblox executable"),
    ):
        if description == "COLMAP sparse model":
            exists = path.is_dir()
        else:
            exists = path.is_file()
        if not exists:
            raise FileNotFoundError(f"Missing {description}: {path}.")

    _validate_output_root(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    depth_dir = output_root / "absolute_depth"
    sdf_dir = output_root / "sdf"
    manifest_path = output_root / "depth_manifest.json"
    aligned_sparse = output_root / "axis_aligned_colmap_sparse"
    frame_alignment_path = output_root / "nvblox_frame_alignment.json"
    depth_dir.mkdir(parents=True, exist_ok=False)
    sdf_dir.mkdir(parents=True, exist_ok=False)

    pipeline_started = time.perf_counter()
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "status": "rendering",
        "training_config": str(training_config),
        "checkpoint": str(checkpoint),
        "colmap_pose_source": str(colmap_sparse),
        "colmap_pose_contract": (
            "Depth files are written in the same coordinate units as these "
            "COLMAP camera translations."
        ),
        "output_root": str(output_root),
        "absolute_depth_dir": str(depth_dir),
        "sdf_dir": str(sdf_dir),
        "geometry_filter": {
            "enabled": bool(args.geometry_filter),
            "scene_type": str(args.scene_type),
            "status": "pending",
        },
    }
    _write_json(manifest_path, manifest)

    try:
        alignment, max_depth_stats, render_stats = _render_absolute_depths(
            args,
            training_config=training_config,
            checkpoint=checkpoint,
            colmap_sparse=colmap_sparse,
            depth_dir=depth_dir,
        )
        manifest.update(alignment.as_manifest())
        manifest["render"] = render_stats
        manifest["geometry_filter"] = render_stats["geometry_filter"]
        manifest["max_depth"] = max_depth_stats.as_manifest()
        adjusted_affine, frame_alignment = _write_axis_aligned_colmap_sparse(
            colmap_sparse,
            aligned_sparse,
            alignment.training_from_source_affine,
            map_transform,
        )
        _write_json(
            frame_alignment_path,
            {
                "source_to_training_affine": adjusted_affine.tolist(),
            },
        )
        manifest["nvblox_frame_alignment"] = {
            **frame_alignment,
            "affine_for_voxel_converter": str(frame_alignment_path),
        }
        manifest["status"] = "depth_ready"
        _write_json(manifest_path, manifest)

        nvblox_command = build_nvblox_command(
            executable=nvblox_binary,
            colmap_sparse=aligned_sparse,
            depth_dir=depth_dir,
            output_dir=sdf_dir,
            source_voxel_size=float(args.source_voxel_size),
            truncation_distance=float(args.truncation_distance),
            max_depth=float(max_depth_stats.selected_max_depth_m),
            max_esdf_distance=float(args.max_esdf_distance),
        )
        manifest["nvblox_command"] = nvblox_command
        manifest["status"] = "running_nvblox"
        _write_json(manifest_path, manifest)
        nvblox_started = time.perf_counter()
        subprocess.run(nvblox_command, cwd=PROJECT_ROOT, check=True)
        manifest["nvblox_wall_sec"] = float(time.perf_counter() - nvblox_started)

        native_timing_path = sdf_dir / "timing_stats.json"
        native_timing = json.loads(native_timing_path.read_text(encoding="utf-8"))
        native_max_depth = native_timing.get("max_depth", {})
        if native_max_depth.get("mode") != "explicit":
            raise RuntimeError("nvblox did not report explicit max_depth mode.")
        if not math.isclose(
            float(native_max_depth.get("max_depth_m", math.nan)),
            float(max_depth_stats.selected_max_depth_m),
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        ):
            raise RuntimeError("nvblox max_depth does not match the precomputed value.")
        manifest["nvblox_timing_stats"] = str(native_timing_path)
        manifest["status"] = "nvblox_ready"
        _write_json(manifest_path, manifest)

        if not args.skip_voxel_map:
            voxel_command = _build_voxel_map_command(
                args,
                checkpoint=checkpoint,
                training_config=training_config,
                sdf_dir=sdf_dir,
                output_root=output_root,
                frame_alignment_path=frame_alignment_path,
            )
            manifest["voxel_map_command"] = voxel_command
            manifest["status"] = "building_voxel_map"
            _write_json(manifest_path, manifest)
            voxel_started = time.perf_counter()
            subprocess.run(voxel_command, cwd=PROJECT_ROOT, check=True)
            manifest["voxel_map_wall_sec"] = float(time.perf_counter() - voxel_started)
            manifest["voxel_map_prefix"] = str(output_root / "voxel_map_tsdf")

        manifest["timing_sec"] = {
            **dict(manifest["render"]["timing_sec"]),
            "nvblox_wall": float(manifest["nvblox_wall_sec"]),
            "voxel_map_wall": float(manifest.get("voxel_map_wall_sec", 0.0)),
            "pipeline_wall": float(time.perf_counter() - pipeline_started),
        }
        manifest["status"] = "complete"
        manifest["pipeline_wall_sec"] = float(time.perf_counter() - pipeline_started)
        _write_json(manifest_path, manifest)
        return manifest
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["pipeline_wall_sec"] = float(time.perf_counter() - pipeline_started)
        _write_json(manifest_path, manifest)
        raise


def main(argv: Sequence[str] | None = None) -> None:
    manifest = run_pipeline(parse_args(argv))
    print(
        "[RenderedDepthNvblox] done: "
        f"frames={manifest['render']['frames_rendered']}, "
        f"source_scale={manifest['source_per_training_unit']:.9g}, "
        f"max_depth={manifest['max_depth']['selected_max_depth_m']:.6g}"
    )
    print(f"[RenderedDepthNvblox] output: {manifest['output_root']}")


if __name__ == "__main__":
    main()
