#!/usr/bin/env python3
"""Estimate a Manhattan yaw from vertical GT mesh triangles.

The generated rigid transform changes only the map-building frame.  Every map
builder must consume the same JSON file; applying it only to GT would break
registration with the 3DGS and TSDF maps.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GSPAR.mapping.ground_truth.mesh_to_voxel import (
    PLY_TYPES,
    _parse_ply_vertex_header,
    load_ply_vertices,
)
from GSPAR.mapping.map_frame_transform import MAP_FRAME_FORMAT, validate_rigid_transform


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True, help="GT triangular PLY in training/world coordinates.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON transform.")
    parser.add_argument(
        "--max-face-samples",
        type=int,
        default=500_000,
        help="Maximum uniformly sampled triangles used for the estimate.",
    )
    parser.add_argument(
        "--max-wall-tilt-deg",
        type=float,
        default=15.0,
        help="Maximum wall-normal tilt away from the XY plane.",
    )
    return parser


def _face_count(path: Path) -> int:
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.decode("ascii").strip()
            if line.startswith("element face "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise ValueError(f"PLY has no face element: {path}")


def load_sampled_ply_triangles(path: Path, max_face_samples: int) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Load vertices and a deterministic uniform sample of fixed-width triangles."""

    ply_format, vertex_count, _, header_bytes, properties = _parse_ply_vertex_header(path)
    if ply_format != "binary_little_endian":
        raise ValueError("axis alignment currently requires a binary_little_endian triangular PLY")
    face_count = _face_count(path)
    try:
        vertex_dtype = np.dtype([(name, PLY_TYPES[field_type]) for name, field_type in properties])
    except KeyError as exc:
        raise ValueError(f"unsupported PLY vertex type {exc.args[0]!r}: {path}") from exc
    face_offset = int(header_bytes + vertex_count * vertex_dtype.itemsize)
    face_dtype = np.dtype([("count", "u1"), ("index", "<i4", (3,))])
    expected_size = face_offset + face_count * face_dtype.itemsize
    if path.stat().st_size != expected_size:
        raise ValueError(
            "axis alignment requires a PLY whose face element contains only triangular "
            f"vertex_indices; expected {expected_size} bytes, got {path.stat().st_size}"
        )
    faces = np.memmap(path, mode="r", dtype=face_dtype, offset=face_offset, shape=(face_count,))
    sample_limit = max(1, int(max_face_samples))
    stride = max(1, int(math.ceil(face_count / sample_limit)))
    sampled = faces[::stride]
    if not np.all(sampled["count"] == 3):
        raise ValueError(f"PLY contains non-triangular faces: {path}")
    triangles = np.asarray(sampled["index"][:sample_limit], dtype=np.int64)
    vertices = load_ply_vertices(path)
    if triangles.size and (triangles.min() < 0 or triangles.max() >= vertices.shape[0]):
        raise ValueError(f"PLY face contains an out-of-range vertex index: {path}")
    return vertices, triangles, {
        "num_input_vertices": int(vertex_count),
        "num_input_faces": int(face_count),
        "face_sample_stride": int(stride),
        "num_sampled_faces": int(triangles.shape[0]),
    }


def _wrap_90_degrees(values: np.ndarray | float) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) + 45.0) % 90.0 - 45.0


def estimate_manhattan_yaw(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    max_wall_tilt_deg: float = 15.0,
) -> tuple[float, dict[str, object]]:
    """Return the dominant wall-normal yaw, modulo 90 degrees."""

    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if points.shape[0] == 0 or faces.shape[0] == 0:
        raise ValueError("axis alignment requires non-empty vertices and triangles")
    a, b, c = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    cross = np.cross(b - a, c - a)
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > 1e-10
    normals = cross[valid] / double_area[valid, None]
    weights = double_area[valid]
    max_abs_z = math.sin(math.radians(float(max_wall_tilt_deg)))
    wall = np.abs(normals[:, 2]) <= max_abs_z
    if not np.any(wall):
        raise ValueError("no sufficiently vertical mesh triangles were found")
    wall_normals = normals[wall]
    wall_weights = weights[wall]
    folded = _wrap_90_degrees(
        np.degrees(np.arctan2(wall_normals[:, 1], wall_normals[:, 0]))
    )

    bin_width = 0.25
    edges = np.arange(-45.0, 45.0 + bin_width, bin_width)
    histogram, edges = np.histogram(folded, bins=edges, weights=wall_weights)
    peak_index = int(np.argmax(histogram))
    peak = float(0.5 * (edges[peak_index] + edges[peak_index + 1]))
    delta = _wrap_90_degrees(folded - peak)
    inlier = np.abs(delta) <= 2.0
    if not np.any(inlier):
        raise ValueError("dominant Manhattan wall direction has no inliers")
    phases = np.radians(4.0 * folded[inlier])
    resultant = np.sum(wall_weights[inlier] * np.exp(1j * phases))
    yaw = float(_wrap_90_degrees(np.degrees(np.angle(resultant)) / 4.0))
    inlier_fraction = float(wall_weights[inlier].sum() / wall_weights.sum())
    return yaw, {
        "max_wall_tilt_deg": float(max_wall_tilt_deg),
        "num_nondegenerate_sampled_faces": int(valid.sum()),
        "num_vertical_sampled_faces": int(wall.sum()),
        "dominant_peak_deg": peak,
        "dominant_wall_area_fraction": inlier_fraction,
    }


def yaw_transform(yaw_deg: float, center: np.ndarray) -> np.ndarray:
    """Rotate by ``-yaw_deg`` around ``center`` so walls align with X/Y."""

    center = np.asarray(center, dtype=np.float64).reshape(3)
    angle = math.radians(-float(yaw_deg))
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = center - rotation @ center
    return validate_rigid_transform(transform)


def run(args: argparse.Namespace) -> dict[str, object]:
    mesh = args.mesh.expanduser().resolve()
    if not mesh.is_file():
        raise FileNotFoundError(f"GT mesh not found: {mesh}")
    vertices, triangles, load_stats = load_sampled_ply_triangles(mesh, args.max_face_samples)
    axis_yaw_deg, estimate_stats = estimate_manhattan_yaw(
        vertices,
        triangles,
        max_wall_tilt_deg=args.max_wall_tilt_deg,
    )
    center = 0.5 * (vertices.min(axis=0).astype(np.float64) + vertices.max(axis=0).astype(np.float64))
    transform = yaw_transform(axis_yaw_deg, center)
    payload: dict[str, object] = {
        "format": MAP_FRAME_FORMAT,
        "method": "area_weighted_vertical_triangle_normals_manhattan_yaw",
        "mesh": str(mesh),
        "axis_yaw_deg": axis_yaw_deg,
        "applied_yaw_deg": -axis_yaw_deg,
        "rotation_center": center.astype(float).tolist(),
        "map_from_training_world": transform.tolist(),
        "training_world_from_map": np.linalg.inv(transform).astype(np.float32).tolist(),
        **load_stats,
        **estimate_stats,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[AxisAlignment] wall_yaw={axis_yaw_deg:.4f} deg, "
        f"applied={-axis_yaw_deg:.4f} deg: {output}"
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
