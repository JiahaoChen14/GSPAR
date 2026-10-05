#!/usr/bin/env python3
"""Evaluate rendered-depth nvblox maps with geometry filtering on and off."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GSPAR.evaluation.path_planning.radius_profiles import default_robot_radius_for_dataset
from GSPAR.evaluation.ground_truth.occupancy_quality import evaluate as evaluate_gt
from GSPAR.evaluation.occupancy_map_quality import evaluate as evaluate_occupancy
from GSPAR.batch._datasets import gspar_output_path
from GSPAR.batch._common import run


VARIANTS = (("geometry_filter_on", True), ("geometry_filter_off", False))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def completed(root: Path, enabled: bool, scene_type: str) -> bool:
    manifest_path = root / "depth_manifest.json"
    voxel_map = root / "voxel_map_tsdf.npz"
    if not manifest_path.is_file() or not voxel_map.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    geometry = manifest.get("geometry_filter", {})
    return (
        manifest.get("status") == "complete"
        and geometry.get("enabled") is enabled
        and geometry.get("scene_type") == scene_type
    )


def run_maps(
    record: dict,
    variant: str,
    enabled: bool,
    args: argparse.Namespace,
) -> None:
    dataset, scene = record["dataset"], record["scene"]
    fields = record["fields"]
    scene_dir = gspar_output_path(fields["output"])
    has_gt = fields.get("gt") is not None
    nvblox = fields.get("nvblox version") or {}
    scene_type = str(fields.get("scene type") or "outdoor").lower()
    source_voxel_size = fields.get("source voxel size") or fields.get("voxel size") or 0.05
    target_voxel_size = fields.get("target voxel size") or fields.get("voxel size") or 0.05
    output = args.rendered_root / variant / dataset / scene / "nvblox-rendered_depth"
    reference = scene_dir / ("voxel_map_gt.npz" if has_gt else "voxel_map_tsdf.npz")
    transform = scene_dir / "map_frame_transform.json" if has_gt else None

    required = [
        Path(fields["3dgs model"]),
        Path(fields["training_config"]),
        Path(nvblox.get("sparse") or fields["sparse"]),
        reference,
    ]
    if transform is not None:
        required.append(transform)
    if not args.dry_run:
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing inputs: " + ", ".join(missing))

    if args.dry_run or not completed(output, enabled, scene_type):
        if not args.dry_run and output.exists() and any(output.iterdir()):
            raise RuntimeError(f"Incomplete rendered-depth output: {output}")
        command: list[object] = [
            sys.executable,
            "GSPAR/mapping/build_rendered_depth_nvblox_map.py",
            "--training-config", fields["training_config"],
            "--checkpoint", fields["3dgs model"],
            "--colmap-sparse", nvblox.get("sparse") or fields["sparse"],
            "--output-root", output,
            "--source-voxel-size", source_voxel_size,
            "--voxel-size", target_voxel_size,
            "--truncation-distance", 0.20,
            "--max-esdf-distance", 5.0,
            "--block-size", 8,
            "--scene-padding", 0.02,
            "--rendered-depth-unit", "camera-z",
            "--scene-type", scene_type,
            "--enable-geometry-filter" if enabled else "--disable-geometry-filter",
        ]
        if transform is not None:
            command += ["--map-frame-transform", transform]
        run(command, args.dry_run)

    evaluation = output / "evaluation"
    map_report = evaluation / ("map_quality_vs_gt.json" if has_gt else "map_quality.json")
    prediction = output / "voxel_map_tsdf.npz"
    if args.dry_run:
        print(f"[evaluate rendered-depth map] {prediction} vs {reference}", flush=True)
    elif has_gt:
        _write_json(
            map_report,
            evaluate_gt(scene_dir, reference, [("nvblox_rendered_depth", prediction)]),
        )
    else:
        _write_json(
            map_report,
            evaluate_occupancy(
                scene_dir, prediction, reference,
                sdf_dir=output / "sdf",
                pipeline_manifest=output / "depth_manifest.json",
            ),
        )

    if has_gt:
        diagnostic = evaluation / "map_quality_vs_nvblox.json"
        if not args.dry_run:
            _write_json(
                diagnostic,
                evaluate_occupancy(
                    scene_dir, prediction, scene_dir / "voxel_map_tsdf.npz",
                    sdf_dir=output / "sdf",
                    pipeline_manifest=output / "depth_manifest.json",
                ),
            )


def run_paths(record: dict, variant: str, args: argparse.Namespace) -> None:
    dataset, scene = record["dataset"], record["scene"]
    fields = record["fields"]
    scene_dir = gspar_output_path(fields["output"])
    has_gt = fields.get("gt") is not None
    output = args.rendered_root / variant / dataset / scene / "nvblox-rendered_depth"
    reference = scene_dir / ("voxel_map_gt.npz" if has_gt else "voxel_map_tsdf.npz")
    endpoints = scene_dir / (
        "path_evaluation_gt/endpoints.json" if has_gt else "path_evaluation/endpoints.json"
    )

    path_dir = output / "evaluation" / "path_planning"
    argv = [
        "--checkpoint", str(fields["3dgs model"]),
        "--reference-map", str(reference),
        "--esdf-map", str(output / "sdf"),
        "--esdf-grid-metadata", str(output / "voxel_map_tsdf.npz"),
        "--endpoints", str(endpoints),
        "--num-trials", str(args.num_trials),
        "--methods", "nvblox_esdf_astar_stomp",
        "--robot-radius-m", str(default_robot_radius_for_dataset(dataset)),
        "--rrt-base-seed", "0",
        "--output-dir", str(path_dir),
    ]
    if args.dry_run:
        print("[evaluate rendered-depth paths] " + " ".join(argv), flush=True)
    elif has_gt:
        from GSPAR.evaluation.ground_truth.batch_compare import main
        main(argv)
    else:
        from GSPAR.evaluation.path_planning.batch_compare import main
        main(argv)
