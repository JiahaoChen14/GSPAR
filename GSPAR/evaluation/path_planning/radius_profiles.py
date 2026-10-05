"""Shared robot-radius sweep protocol for path-planning evaluations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_BASELINE_ROBOT_RADIUS_M = 0.08
# The benchmark uses one physical safety-clearance profile by default. Extra
# profiles remain available only when explicitly requested on the command line.
DEFAULT_ADDITIONAL_ROBOT_RADII_M: tuple[float, ...] = ()
DATASET_REFERENCE_ROBOT_RADIUS_M = 0.25
DATASET_ROBOT_RADIUS_SCALE = {
    "scannetpp": 1.0,
    "rar_persp": 0.2,
    "mipnerf360": 1.0 / 3.0,
    "tanks_and_temples": 0.2,
}


def robot_radius_scale_for_dataset(
    dataset: str | None,
    *,
    default: float | None = 1.0,
) -> float | None:
    """Resolve a coordinate scale from a dataset name, loader name, or path."""

    normalized = "".join(
        character
        for character in ("" if dataset is None else str(dataset)).casefold()
        if character.isalnum()
    )
    for name, scale in DATASET_ROBOT_RADIUS_SCALE.items():
        key = "".join(character for character in name if character.isalnum())
        if key and key in normalized:
            return float(scale)
    return default


def default_robot_radius_for_dataset(dataset: str | None) -> float:
    """Return the benchmark radius in the coordinate units of ``dataset``.

    Unknown or standalone datasets retain the 0.08 m command-line default.
    Known datasets represent a 0.25 m robot in their respective coordinate
    scales.
    """

    scale = robot_radius_scale_for_dataset(dataset, default=None)
    if scale is None:
        return DEFAULT_BASELINE_ROBOT_RADIUS_M
    return DATASET_REFERENCE_ROBOT_RADIUS_M * scale


def normalize_robot_radius_profiles(
    baseline_radius_m: float,
    additional_radii_m: Iterable[float],
) -> list[float]:
    """Validate radius profiles while preserving baseline-first order."""

    radii = [float(baseline_radius_m), *(float(value) for value in additional_radii_m)]
    if any(not np.isfinite(radius) or radius <= 0.0 for radius in radii):
        raise ValueError("all robot radii must be finite and positive")
    unique = []
    for radius in radii:
        if not any(
            np.isclose(radius, existing, rtol=0.0, atol=1.0e-9)
            for existing in unique
        ):
            unique.append(radius)
    return unique


def radius_profile_directory_name(radius_m: float) -> str:
    """Return the common on-disk directory name for one radius profile."""

    value = f"{float(radius_m):.3f}".replace("-", "neg").replace(".", "p")
    return f"radius_{value}m"


def validate_endpoint_manifest_radius(
    path: str | Path,
    required_radius_m: float,
) -> float:
    """Require endpoints generated for at least the largest evaluated radius."""

    manifest_path = Path(path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_radius = manifest.get("robot_radius_m")
    if manifest_radius is None:
        raise ValueError(
            "endpoint manifest does not record robot_radius_m; regenerate it for "
            "the largest evaluation radius"
        )
    radius = float(manifest_radius)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("endpoint manifest robot_radius_m must be finite and positive")
    if radius + 1.0e-9 < float(required_radius_m):
        raise ValueError(
            "endpoint manifest is not safe for every requested radius: "
            f"manifest={radius:g} m, largest evaluation radius={required_radius_m:g} m; "
            "regenerate endpoints with --robot-radius-m set to the largest radius"
        )
    return radius
