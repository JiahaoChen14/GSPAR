#!/usr/bin/env python3
"""Merge baseline, rendered-depth, ablation, and SplatNav results."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GSPAR.evaluation.summary_aggregation import (
    aggregate_path,
    aggregate_quality,
    quality_row,
)
from GSPAR.batch._datasets import gspar_output_path, load_records


VARIANTS = ("geometry_filter_on", "geometry_filter_off")


def load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def add_quality(
    rows: list[dict[str, Any]],
    dataset: str,
    scene: str,
    method: str,
    pipeline: str,
    reference: str,
    report: Path,
    prediction: str | None = None,
) -> None:
    data = load(report)
    if prediction is not None:
        data = data.get("predictions", {}).get(prediction)
        if not isinstance(data, dict):
            return
    rows.append(
        {
            **quality_row(dataset, scene, data),
            "method": method,
            "pipeline": pipeline,
            "reference": reference,
            "report": str(report),
        }
    )


def add_paths(
    rows: list[dict[str, Any]],
    dataset: str,
    scene: str,
    pipeline: str,
    reference: str,
    report: Path,
    prefix: str = "",
) -> None:
    for method, metrics in load(report).get("by_method", {}).items():
        if isinstance(metrics, dict):
            rows.append(
                {
                    **metrics,
                    "dataset": dataset,
                    "scene": scene,
                    "method": prefix + method,
                    "pipeline": pipeline,
                    "reference": reference,
                    "report": str(report),
                }
            )


def aggregate(
    rows: list[dict[str, Any]],
    function: Callable[[list[dict[str, Any]]], dict[str, Any]],
) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["reference"], row["method"])].append(row)
    output: dict[str, Any] = {}
    for (reference, method), group in sorted(groups.items()):
        output.setdefault(reference, {})[method] = function(group)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-file", type=Path, default=Path("configs/dataset.txt"))
    parser.add_argument("--dataset", dest="datasets", action="append")
    parser.add_argument("--scenes", help="Comma-separated scene names")
    parser.add_argument(
        "--rendered-root",
        type=Path,
        default=Path("GSPAR/outputs/nvblox_rendered_depth_evaluation"),
    )
    parser.add_argument(
        "--ablation-root",
        type=Path,
        default=Path("GSPAR/outputs/filter_ablation"),
    )
    parser.add_argument(
        "--splatnav-root",
        type=Path,
        default=Path("GSPAR/outputs/splatnav_evaluation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("GSPAR/outputs/evaluation_suite_summary"),
    )
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    datasets = set(args.datasets or [])
    scenes = {item.strip() for item in args.scenes.split(",") if item.strip()} if args.scenes else set()
    records = [
        record for record in load_records(args.dataset_file)
        if (not datasets or record["dataset"] in datasets)
        and (not scenes or record["scene"] in scenes)
    ]
    if not records:
        parser.error("No scenes matched --dataset/--scenes")

    map_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for record in records:
        dataset, scene = record["dataset"], record["scene"]
        has_gt = record["fields"].get("gt") is not None
        scene_dir = gspar_output_path(record["fields"]["output"])
        path_reference = "ground_truth_mesh_voxel" if has_gt else "nvblox_tsdf_voxel"

        baseline_map = scene_dir / "eval" / "sparse_vs_nvblox_quality.json"
        if baseline_map.is_file():
            add_quality(
                map_rows, dataset, scene, "GSPAR", "baseline",
                "nvblox_tsdf_voxel", baseline_map,
            )
        else:
            missing.append(str(baseline_map))
        if has_gt:
            gt_map = scene_dir / "eval_gt" / "voxel_vs_gt_quality.json"
            if gt_map.is_file():
                for method in ("GSPAR", "nvblox"):
                    add_quality(
                        map_rows, dataset, scene, method, "baseline",
                        "ground_truth_mesh_voxel", gt_map, method,
                    )
            else:
                missing.append(str(gt_map))

        baseline_path = scene_dir / (
            "path_evaluation_gt/summary.json" if has_gt else "path_evaluation/summary.json"
        )
        if baseline_path.is_file():
            add_paths(path_rows, dataset, scene, "baseline", path_reference, baseline_path)
        else:
            missing.append(str(baseline_path))

        for variant in VARIANTS:
            evaluation = (
                args.rendered_root / variant / dataset / scene
                / "nvblox-rendered_depth" / "evaluation"
            )
            rendered_map = evaluation / ("map_quality_vs_gt.json" if has_gt else "map_quality.json")
            if rendered_map.is_file():
                add_quality(
                    map_rows,
                    dataset,
                    scene,
                    f"nvblox-rendered-depth[{variant}]",
                    "rendered_depth",
                    path_reference,
                    rendered_map,
                    "nvblox-rendered_depth" if has_gt else None,
                )
            diagnostic = evaluation / "map_quality_vs_nvblox.json"
            if has_gt and diagnostic.is_file():
                add_quality(
                    map_rows, dataset, scene,
                    f"nvblox-rendered-depth[{variant}]", "rendered_depth",
                    "nvblox_tsdf_voxel", diagnostic,
                )
            rendered_path = evaluation / "path_planning" / "summary.json"
            if rendered_path.is_file():
                add_paths(
                    path_rows, dataset, scene, "rendered_depth", path_reference,
                    rendered_path, f"rendered-depth[{variant}]/",
                )

        ablation = (
            args.ablation_root / dataset / "without_geometry_filtering" / scene
            / "eval" / "sparse_vs_nvblox_quality.json"
        )
        if ablation.is_file():
            add_quality(
                map_rows, dataset, scene, "GSPAR[without_geometry_filtering]",
                "filter_ablation", "nvblox_tsdf_voxel", ablation,
            )
        splatnav = args.splatnav_root / dataset / scene / "summary.json"
        if splatnav.is_file():
            add_paths(path_rows, dataset, scene, "splatnav", path_reference, splatnav)

    result = {
        "datasets": sorted({record["dataset"] for record in records}),
        "scene_count": len(records),
        "map_quality": {
            "aggregate_by_reference_and_method": aggregate(map_rows, aggregate_quality),
            "rows": map_rows,
        },
        "path_planning": {
            "aggregate_by_reference_and_method": aggregate(path_rows, aggregate_path),
            "rows": path_rows,
        },
        "missing_baseline_results": missing,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "combined_results.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[EvaluationSummary] map_rows={len(map_rows)}, path_rows={len(path_rows)}, output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
