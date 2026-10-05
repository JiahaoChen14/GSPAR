#!/usr/bin/env python3
"""Evaluate one or more voxel maps against a ground-truth mesh voxel map."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from GSPAR.evaluation.occupancy_map_quality import evaluate as evaluate_voxel_pair
from GSPAR.evaluation.occupancy_map_quality import resolve_npz
from GSPAR.evaluation.resource_usage import summarize_resource_usages


RESULT_NAME = "voxel_vs_gt_quality.json"


def _prediction(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("prediction must use LABEL=PATH syntax")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label or not raw_path.strip():
        raise argparse.ArgumentTypeError("prediction must use non-empty LABEL=PATH syntax")
    return label, Path(raw_path.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--reference-map", type=Path, required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        type=_prediction,
        required=True,
        metavar="LABEL=PATH",
        help="Prediction to compare; repeat for GSPAR, nvblox, or other maps.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _reference_metadata(reference_path: Path) -> dict[str, object]:
    metadata_path = reference_path.with_suffix(".json")
    if not metadata_path.is_file():
        return {"metadata": str(metadata_path), "reference_semantics": None}
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "metadata": str(metadata_path),
        "reference_semantics": data.get("reference_semantics"),
        "mesh": data.get("mesh"),
        "free_space_map": data.get("free_space_map"),
    }


def evaluate(
    scene_dir: Path,
    reference_path: Path,
    predictions: list[tuple[str, Path]],
) -> dict[str, object]:
    reference_path = resolve_npz(reference_path)
    labels = [label for label, _ in predictions]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Prediction labels must be unique: {labels}")

    results: dict[str, object] = {}
    for label, prediction_path in predictions:
        started = time.perf_counter()
        pair = evaluate_voxel_pair(scene_dir, prediction_path, reference_path)
        # GT evaluation reports geometry only; planning-map timing and payload
        # comparisons belong to the separate nvblox-reference protocol.
        pair.pop("efficiency", None)
        pair["protocol"] = {
            "reference": "ground_truth_mesh_surface_voxels",
            "prediction": "mesh_or_reconstructed_surface_voxels",
            "surface_grid": "same voxelizer and common voxel grid",
            "metrics": (
                "zero-tolerance surface IoU, precision, recall, and F-score; surface "
                "precision, recall, and F-score at 0.05 m and 0.10 m; "
                "bidirectional nearest-surface distances, Chamfer-L1, and "
                "10 cm truncated Chamfer-L1"
            ),
        }
        pair["evaluation_time_sec"] = float(time.perf_counter() - started)
        results[label] = pair

    resource_usage_by_prediction = {
        label: pair["resource_usage"]
        for label, pair in results.items()
    }
    return {
        "format": "nerficg_ground_truth_surface_evaluation_v3",
        "protocol": {
            "reference": "ground_truth_mesh_surface_voxels",
            "prediction_labels": labels,
            "note": (
                "GT and mesh-capable predictions are voxelized from their surfaces with the "
                "same common-grid mesh vertex voxelizer. Planning occupancy labels are not scored."
            ),
        },
        "scene_dir": str(scene_dir),
        "reference_map": str(reference_path),
        "reference": _reference_metadata(reference_path),
        "predictions": results,
        "resource_usage_summary": summarize_resource_usages(
            resource_usage_by_prediction
        ),
        "resource_usage_by_prediction": resource_usage_by_prediction,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scene_dir = args.scene_dir.expanduser().resolve()
    reference_path = args.reference_map.expanduser().resolve()
    predictions = [
        (label, path.expanduser().resolve())
        for label, path in args.prediction
    ]
    result = evaluate(scene_dir, reference_path, predictions)
    output = (
        args.output or scene_dir / "eval_gt" / RESULT_NAME
    ).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for label, pair in result["predictions"].items():
        rows = pair["surface_metrics"]["by_threshold"]
        exact = pair["surface_metrics"]["zero_tolerance"]
        truncated_chamfer = pair["surface_metrics"]["truncated_chamfer_l1_m"]
        resources = pair["resource_usage"]
        cpu_peak_mib = float(resources["cpu_rss_peak_sampled_bytes"]) / (1024.0**2)
        cuda_peak = resources["cuda_torch_allocated_peak_bytes"]
        cuda_text = (
            "n/a"
            if cuda_peak is None
            else f"{float(cuda_peak) / (1024.0**2):.1f}MiB"
        )
        print(
            f"[GTOccupancyEval] {label}: "
            f"IoU/P/R/F@0={exact['surface_iou']:.4f}/{exact['precision']:.4f}/"
            f"{exact['recall']:.4f}/{exact['fscore']:.4f}, "
            f"P/R/F@5cm={rows[0]['precision']:.4f}/{rows[0]['recall']:.4f}/{rows[0]['fscore']:.4f}, "
            f"P/R/F@10cm={rows[1]['precision']:.4f}/{rows[1]['recall']:.4f}/{rows[1]['fscore']:.4f}, "
            f"truncated_chamfer@10cm="
            f"{'n/a' if truncated_chamfer is None else f'{truncated_chamfer:.4f}m'}, "
            f"cpu_rss_peak={cpu_peak_mib:.1f}MiB, "
            f"cuda_allocated_peak={cuda_text}"
        )
    print(f"[GTOccupancyEval] output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
