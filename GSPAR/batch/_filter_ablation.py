#!/usr/bin/env python3
"""Build and evaluate GSPAR without geometry filtering."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GSPAR.batch._datasets import gspar_output_path
from GSPAR.batch._common import run
from GSPAR.evaluation.occupancy_map_quality import evaluate


VARIANT = "without_geometry_filtering"
RESULT = "sparse_vs_nvblox_quality.json"


def run_scene(record: dict, args: argparse.Namespace) -> None:
    dataset, scene = record["dataset"], record["scene"]
    fields = record["fields"]
    scene_dir = gspar_output_path(fields["output"])
    output = args.ablation_root / dataset / VARIANT / scene
    voxel_prefix = output / "voxel_map"
    report = output / "eval" / RESULT
    voxel_size = fields.get("target voxel size") or fields.get("voxel size") or 0.05

    command: list[object] = [
        sys.executable,
        "GSPAR/mapping/gaussian_occupancy/build_sparse_voxel_map.py",
        "--checkpoint", fields["3dgs model"],
        "--output-prefix", voxel_prefix,
        "--voxel-size", voxel_size,
        "--block-size", 8,
        "--scene-padding", 0.02,
        "--disable-geometry-filter",
    ]
    if fields.get("gt") is not None:
        command += ["--map-frame-transform", scene_dir / "map_frame_transform.json"]
    if fields.get("scene type"):
        command += ["--scene-type", fields["scene type"]]
    run(command, args.dry_run)

    if args.dry_run:
        print(f"[evaluate ablation map] {voxel_prefix.with_suffix('.npz')}", flush=True)
        return
    result = evaluate(
        scene_dir,
        voxel_prefix.with_suffix(".npz"),
        scene_dir / "voxel_map_tsdf.npz",
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
