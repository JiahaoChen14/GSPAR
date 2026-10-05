#!/usr/bin/env python3
"""Sweep 3DGS-to-voxel parameters over every scene in ``dataset.txt``.

Each candidate is voxelized on the already-built reference map's grid.  A
scene with ``gt`` in ``dataset.txt`` is evaluated against ``voxel_map_gt``;
all other scenes are evaluated against ``voxel_map_tsdf``.  The script never
rebuilds GT meshes or nvblox TSDF maps.

This is a three-stage validation protocol, not a Cartesian grid search:

1. ``knn`` compares K8/K16/K32 while every other parameter is fixed.
2. ``composite`` fixes the selected k and compares B0/C1/C2.
3. ``indoor-boundary`` fixes the selected general configuration and compares
   indoor boundary quantiles 0.001/0.005/0.010 on indoor scenes only.

The second and third stages require explicit selections from the preceding
stage, so the runner never silently chooses a speed/quality trade-off.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GSPAR.evaluation.occupancy_map_quality import evaluate as evaluate_voxel_pair
from GSPAR.batch._datasets import load_records
from GSPAR.mapping.gaussian_occupancy.build_sparse_voxel_map import (
    Config,
    build_sparse_voxel_payload,
)
from GSPAR.mapping.gaussian_occupancy.voxel_preprocess import (
    load_gaussian_arrays,
    preprocess_gaussian_arrays,
    transform_gaussians,
)
from GSPAR.mapping.map_frame_transform import load_map_frame_transform, transform_metadata
from GSPAR.mapping.voxel_grid_layout import load_reference_grid


DEFAULT_DATASET_FILE = Path("configs/dataset.txt")
DEFAULT_OUTPUT_ROOT = Path("GSPAR/outputs/gaussian_occupancy_sweep")
REFERENCE_GT = "ground_truth_mesh_voxel"
REFERENCE_TSDF = "nvblox_tsdf_voxel"

# Fixed throughout all three stages.  These preserve the current geometry
# filtering and splatting structure while isolating the requested variables.
FIXED_OVERRIDES: dict[str, Any] = {
    "geometry_filter": True,
    "boundary_outlier_filter": True,
    "detail_guard": False,
    "isolation_quantile": 0.97,
    "low_density_quantile": 0.10,
    "geometry_filter_max_remove_ratio": 0.20,
    "support_radius_cap_quantile": 0.995,
    "max_support_radius_voxels": 32,
    "thin_min_anisotropy": 0.0,
    "surface_splat_mode": "thin",
    "voxel_footprint_scale": 0.20,
}

BASELINE_VARIABLES: dict[str, Any] = {
    "knn_k": 16,
    "floater_opacity_threshold": 0.04,
    "support_contribution_floor": 0.010,
    "normal_support_voxels": 1.0,
    "occupied_contribution_threshold": 0.35,
}

COMPOSITE_VARIABLES: dict[str, dict[str, Any]] = {
    "B0": {
        "floater_opacity_threshold": 0.04,
        "support_contribution_floor": 0.010,
        "normal_support_voxels": 1.0,
        "occupied_contribution_threshold": 0.35,
    },
    "C1": {
        "floater_opacity_threshold": 0.05,
        "support_contribution_floor": 0.009,
        "normal_support_voxels": 1.1,
        "occupied_contribution_threshold": 0.34,
    },
    "C2": {
        "floater_opacity_threshold": 0.03,
        "support_contribution_floor": 0.011,
        "normal_support_voxels": 0.9,
        "occupied_contribution_threshold": 0.36,
    },
}

STAGE_DIRECTORY = {
    "knn": "stage_1_knn",
    "composite": "stage_2_composite",
    "indoor-boundary": "stage_3_indoor_boundary",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _absolute(path: Path) -> Path:
    """Resolve only for I/O checks; result paths remain project-relative."""

    return path if path.is_absolute() else PROJECT_ROOT / path


def _relative_to_project(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        return expanded
    try:
        return expanded.relative_to(PROJECT_ROOT)
    except ValueError:
        return expanded


def _variant(name: str, variables: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "overrides": {**FIXED_OVERRIDES, **BASELINE_VARIABLES, **variables},
    }


def stage_variants(
    stage: str,
    *,
    scene_type: str,
    selected_knn_k: int | None,
    selected_composite: str | None,
) -> list[dict[str, Any]]:
    """Return the small, pre-registered candidate set for one protocol stage."""

    if stage == "knn":
        return [_variant(f"K{k}", {"knn_k": k}) for k in (8, 16, 32)]
    if stage == "composite":
        if selected_knn_k is None:
            raise ValueError("Stage composite requires --selected-knn-k (8, 16, or 32)")
        return [
            _variant(name, {"knn_k": selected_knn_k, **variables})
            for name, variables in COMPOSITE_VARIABLES.items()
        ]
    if stage == "indoor-boundary":
        if selected_knn_k is None:
            raise ValueError("Stage indoor-boundary requires --selected-knn-k")
        if selected_composite not in COMPOSITE_VARIABLES:
            raise ValueError("Stage indoor-boundary requires --selected-composite B0, C1, or C2")
        if scene_type != "indoor":
            return []
        base = {"knn_k": selected_knn_k, **COMPOSITE_VARIABLES[selected_composite]}
        return [
            _variant(
                f"{selected_composite}_IndoorBoundaryQ{quantile:.3f}",
                {**base, "indoor_boundary_quantile": quantile},
            )
            for quantile in (0.001, 0.005, 0.010)
        ]
    raise ValueError(f"Unsupported stage: {stage}")


def selected_stages(stage: str, selected_knn_k: int | None, selected_composite: str | None) -> list[str]:
    if stage != "all":
        return [stage]
    if selected_knn_k is None or selected_composite is None:
        raise ValueError(
            "--stage all requires --selected-knn-k and --selected-composite; "
            "the script will not choose a best configuration automatically."
        )
    return ["knn", "composite", "indoor-boundary"]


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path


def _scene_output_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return path if path.parts[:1] == ("GSPAR",) else Path("GSPAR") / path


def scene_specs(
    dataset_file: Path,
    datasets: set[str] | None,
    scenes: set[str] | None,
) -> list[dict[str, Any]]:
    specs = []
    for record in load_records(dataset_file):
        dataset = str(record["dataset"])
        scene = str(record["scene"])
        if datasets is not None and dataset not in datasets:
            continue
        if scenes is not None and scene not in scenes:
            continue
        config = record["fields"]
        output_dir = _scene_output_path(config["output"])
        has_gt = config.get("gt") is not None
        reference_path = output_dir / ("voxel_map_gt.npz" if has_gt else "voxel_map_tsdf.npz")
        specs.append(
            {
                "dataset": dataset,
                "scene": scene,
                "scene_type": str(config.get("scene type") or "outdoor").lower(),
                "checkpoint": _project_path(config["3dgs model"]),
                "source_output_dir": output_dir,
                "reference_path": reference_path,
                "reference_type": REFERENCE_GT if has_gt else REFERENCE_TSDF,
                "map_frame_transform": output_dir / "map_frame_transform.json" if has_gt else None,
            }
        )
    return specs


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _metric_row(
    *,
    dataset: str,
    scene: str,
    scene_type: str,
    reference_type: str,
    reference_path: Path,
    stage: str,
    variant: str,
    overrides: dict[str, Any],
    map_path: Path,
    evaluation_path: Path,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    surface = evaluation["surface_metrics"]
    exact = surface["zero_tolerance"]
    by_threshold = {float(item["threshold_m"]): item for item in surface["by_threshold"]}
    metadata = json.loads(map_path.with_suffix(".json").read_text(encoding="utf-8"))
    geometry_filter = metadata.get("geometry_filter", {})
    timing_sec = metadata.get("timing_sec", {})
    return {
        "dataset": dataset,
        "scene": scene,
        "scene_type": scene_type,
        "reference_type": reference_type,
        "reference_map": str(reference_path),
        "stage": stage,
        "variant": variant,
        "overrides_json": json.dumps(_jsonable(overrides), sort_keys=True),
        "map": str(map_path),
        "evaluation": str(evaluation_path),
        "surface_iou_0": exact["surface_iou"],
        "precision_0": exact["precision"],
        "recall_0": exact["recall"],
        "fscore_0": exact["fscore"],
        "fscore_5cm": by_threshold[0.05]["fscore"],
        "fscore_10cm": by_threshold[0.10]["fscore"],
        "truncated_chamfer_l1_m": surface["truncated_chamfer_l1_m"],
        "occupied_voxels": metadata.get("num_occupied_voxels"),
        "filtered_gaussians": metadata.get("num_gaussians_filtered"),
        "removed_gaussians": metadata.get("num_gaussians_removed"),
        "gaussian_retention_ratio": metadata.get("gaussian_retention_ratio"),
        "geometry_filter_sec": geometry_filter.get("elapsed_sec"),
        "geometry_filter_candidates": geometry_filter.get("num_candidates_total"),
        "geometry_filter_removed": geometry_filter.get("num_removed_total"),
        "build_calculation_sec": timing_sec.get("calculation"),
        "total_conversion_sec": timing_sec.get("categorized_total"),
        "evaluation_sec": evaluation.get("evaluation_time_sec"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset", "scene", "scene_type", "reference_type", "reference_map", "stage", "variant",
        "overrides_json", "map", "evaluation", "surface_iou_0", "precision_0",
        "recall_0", "fscore_0", "fscore_5cm", "fscore_10cm",
        "truncated_chamfer_l1_m", "occupied_voxels", "filtered_gaussians", "removed_gaussians",
        "gaussian_retention_ratio", "geometry_filter_sec", "geometry_filter_candidates",
        "geometry_filter_removed", "build_calculation_sec", "total_conversion_sec", "evaluation_sec",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    result: dict[str, Any] = {"count": len(values)}
    for name in (
        "fscore_0", "fscore_5cm", "fscore_10cm", "truncated_chamfer_l1_m",
        "geometry_filter_sec", "build_calculation_sec", "total_conversion_sec",
        "removed_gaussians",
    ):
        finite = [float(row[name]) for row in values if row.get(name) is not None]
        result[f"mean_{name}"] = None if not finite else float(np.mean(finite))
    return result


def _save_voxel_map(
    *,
    output_prefix: Path,
    payload: dict[str, np.ndarray],
    stats: dict[str, Any],
    config: Config,
    reference_path: Path,
    transform: np.ndarray,
    transform_info: dict[str, Any],
    load_sec: float,
) -> Path:
    """Persist the same required fields as the standalone 3DGS builder."""

    started = time.perf_counter()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    map_path = output_prefix.with_suffix(".npz")
    save_payload = dict(payload)
    save_payload["map_from_training_world"] = transform.astype(np.float32)
    save_payload["training_world_from_map"] = np.linalg.inv(transform).astype(np.float32)
    np.savez_compressed(map_path, **save_payload)
    save_npz_sec = float(time.perf_counter() - started)
    metadata = {
        "config": _jsonable(asdict(config)),
        "reference_occupancy": str(reference_path),
        "map_frame_transform": transform_metadata(transform, transform_info),
        **stats,
        "timing_sec": {
            "loading": float(load_sec),
            "calculation": float(stats["pipeline_elapsed_sec"]),
            "saving": save_npz_sec,
            "categorized_total": float(load_sec + stats["pipeline_elapsed_sec"] + save_npz_sec),
            "wall_total": float(load_sec + stats["pipeline_elapsed_sec"] + save_npz_sec),
        },
    }
    _write_json(map_path.with_suffix(".json"), metadata)
    return map_path


def _prepare_scene(spec: dict[str, Any], device: str) -> tuple[Any, dict[str, object], np.ndarray, dict[str, Any], float]:
    reference_path = Path(spec["reference_path"])
    checkpoint = Path(spec["checkpoint"])
    transform_path = spec["map_frame_transform"]
    if not _absolute(reference_path).is_file():
        raise FileNotFoundError(f"Reference voxel map not found: {reference_path}")
    if not _absolute(checkpoint).is_file():
        raise FileNotFoundError(f"3DGS checkpoint not found: {checkpoint}")
    if transform_path is not None and not _absolute(Path(transform_path)).is_file():
        raise FileNotFoundError(f"GT scene requires map-frame transform: {transform_path}")
    started = time.perf_counter()
    transform, transform_info = load_map_frame_transform(transform_path)
    raw_arrays = load_gaussian_arrays(checkpoint)
    gaussians = preprocess_gaussian_arrays(raw_arrays, max_gaussians=None, device=device)
    gaussians = transform_gaussians(gaussians, transform)
    reference_grid = load_reference_grid(reference_path)
    return gaussians, reference_grid, transform, transform_info, float(time.perf_counter() - started)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-file", type=Path, default=DEFAULT_DATASET_FILE)
    parser.add_argument("--dataset", action="append", dest="datasets", metavar="NAME",
                        help="Restrict to a dataset; repeatable.")
    parser.add_argument("--scenes", default=None, help="Comma-separated scene names to run.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--stage",
        choices=("knn", "composite", "indoor-boundary", "all"),
        default="knn",
        help="Run one validation stage; all requires both prior-stage selections.",
    )
    parser.add_argument(
        "--selected-knn-k",
        type=int,
        choices=(8, 16, 32),
        default=None,
        help="Explicit Stage-1 conclusion required by composite/indoor-boundary.",
    )
    parser.add_argument(
        "--selected-composite",
        choices=tuple(COMPOSITE_VARIABLES),
        default=None,
        help="Explicit Stage-2 conclusion required by indoor-boundary.",
    )
    parser.add_argument("--compute-device", default="cuda")
    parser.add_argument("--force", action="store_true", help="Rebuild completed candidates.")
    parser.add_argument("--continue-on-error", action="store_true", default=True,
                        help="Continue after failures (the default; retained for command compatibility).")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop after the first failed scene or candidate.")
    parser.add_argument("--dry-run", action="store_true", help="List planned candidates without building maps.")
    return parser


def run_sweep(args: argparse.Namespace) -> int:
    if args.fail_fast:
        args.continue_on_error = False
    # Retain relative project paths in commands and result JSON.  The script
    # always executes its file I/O from PROJECT_ROOT.
    os.chdir(PROJECT_ROOT)
    dataset_file = _relative_to_project(args.dataset_file)
    output_root = _relative_to_project(args.output_root)
    datasets = set(args.datasets) if args.datasets else None
    scenes = {item.strip() for item in args.scenes.split(",") if item.strip()} if args.scenes else None
    specs = scene_specs(dataset_file, datasets, scenes)
    if not specs:
        raise ValueError("No scenes matched --dataset/--scenes")
    stages = selected_stages(args.stage, args.selected_knn_k, args.selected_composite)
    planned = sum(
        len(stage_variants(
            stage,
            scene_type=spec["scene_type"],
            selected_knn_k=args.selected_knn_k,
            selected_composite=args.selected_composite,
        ))
        for stage in stages
        for spec in specs
    )
    print(
        f"[GaussianOccupancySweep] stages={','.join(stages)}, scenes={len(specs)}, "
        f"candidates={planned}, output={output_root}"
    )
    if args.dry_run:
        for stage in stages:
            for spec in specs:
                variants = stage_variants(
                    stage,
                    scene_type=spec["scene_type"],
                    selected_knn_k=args.selected_knn_k,
                    selected_composite=args.selected_composite,
                )
                if variants:
                    print(
                        f"  {stage}: {spec['dataset']}/{spec['scene']} "
                        f"({spec['reference_type']}): "
                        + ", ".join(variant["name"] for variant in variants)
                    )
        return 0

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for stage in stages:
        for spec in specs:
            variants = stage_variants(
                stage,
                scene_type=spec["scene_type"],
                selected_knn_k=args.selected_knn_k,
                selected_composite=args.selected_composite,
            )
            if not variants:
                continue
            print(
                f"[GaussianOccupancySweep] stage={stage} scene={spec['dataset']}/{spec['scene']} "
                f"variants={len(variants)}"
            )
            try:
                gaussians, reference_grid, transform, transform_info, load_sec = _prepare_scene(spec, args.compute_device)
            except Exception as exc:
                failure = {"dataset": spec["dataset"], "scene": spec["scene"], "stage": stage, "error": repr(exc)}
                failures.append(failure)
                print(f"[GaussianOccupancySweep] FAILED prepare: {failure}", file=sys.stderr)
                if args.continue_on_error:
                    continue
                break

            for variant in variants:
                variant_dir = output_root / STAGE_DIRECTORY[stage] / spec["dataset"] / spec["scene"] / variant["name"]
                prefix = variant_dir / "voxel_map"
                map_path = prefix.with_suffix(".npz")
                evaluation_path = variant_dir / "evaluation.json"
                try:
                    if not (map_path.is_file() and evaluation_path.is_file()) or args.force:
                        config = replace(
                            Config(),
                            checkpoint=Path(spec["checkpoint"]),
                            output_prefix=prefix,
                            reference_occupancy=Path(spec["reference_path"]),
                            map_frame_transform=spec["map_frame_transform"],
                            scene_type=spec["scene_type"],
                            voxel_size=float(reference_grid["voxel_size"]),
                            block_size=int(reference_grid["block_size"]),
                            compute_device=args.compute_device,
                            **variant["overrides"],
                        )
                        payload, stats = build_sparse_voxel_payload(
                            gaussians,
                            config=config,
                            reference_grid=reference_grid,
                        )
                        map_path = _save_voxel_map(
                            output_prefix=prefix,
                            payload=payload,
                            stats=stats,
                            config=config,
                            reference_path=Path(spec["reference_path"]),
                            transform=transform,
                            transform_info=transform_info,
                            load_sec=load_sec,
                        )
                        evaluation = evaluate_voxel_pair(
                            Path(spec["source_output_dir"]), map_path, Path(spec["reference_path"])
                        )
                        evaluation["sweep"] = {
                            "dataset": spec["dataset"], "scene": spec["scene"],
                            "scene_type": spec["scene_type"], "reference_type": spec["reference_type"],
                            "stage": stage, "variant": variant["name"], "overrides": variant["overrides"],
                        }
                        _write_json(evaluation_path, evaluation)
                    else:
                        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
                        print(f"[GaussianOccupancySweep] resume {spec['dataset']}/{spec['scene']}/{variant['name']}")
                    row = _metric_row(
                        dataset=spec["dataset"], scene=spec["scene"], scene_type=spec["scene_type"],
                        reference_type=spec["reference_type"], reference_path=Path(spec["reference_path"]),
                        stage=stage,
                        variant=variant["name"], overrides=variant["overrides"], map_path=map_path,
                        evaluation_path=evaluation_path, evaluation=evaluation,
                    )
                    rows.append(row)
                    print(f"[GaussianOccupancySweep] {spec['dataset']}/{spec['scene']}/{variant['name']}: "
                          f"F@0={row['fscore_0']:.4f}, F@5cm={row['fscore_5cm']:.4f}, "
                          f"F@10cm={row['fscore_10cm']:.4f}")
                except Exception as exc:
                    failure = {
                        "dataset": spec["dataset"], "scene": spec["scene"], "variant": variant["name"],
                        "stage": stage, "operation": "build_or_evaluate", "error": repr(exc),
                    }
                    failures.append(failure)
                    print(f"[GaussianOccupancySweep] FAILED: {failure}", file=sys.stderr)
                    if not args.continue_on_error:
                        break
        if failures and not args.continue_on_error:
            break

    rows.sort(key=lambda row: (row["stage"], row["dataset"], row["scene"], row["variant"]))
    _write_csv(output_root / "results.csv", rows)
    grouped: dict[str, dict[str, Any]] = {}
    for stage in stages:
        grouped[stage] = {
            variant: _aggregate(
                row for row in rows if row["stage"] == stage and row["variant"] == variant
            )
            for variant in sorted({row["variant"] for row in rows if row["stage"] == stage})
        }
    summary = {
        "format": "gspar_gaussian_occupancy_parameter_sweep_v1",
        "dataset_file": str(dataset_file),
        "stages": stages,
        "selected_knn_k": args.selected_knn_k,
        "selected_composite": args.selected_composite,
        "fixed_overrides": FIXED_OVERRIDES,
        "protocol": {
            "stage_1": "K8/K16/K32 with all other requested parameters fixed",
            "stage_2": "B0/C1/C2 at explicitly selected k",
            "stage_3": "indoor-only boundary quantiles at explicitly selected B0/C1/C2",
        },
        "output_root": str(output_root),
        "planned_candidates": planned,
        "completed_candidates": len(rows),
        "failed_candidates": len(failures),
        "reference_policy": {
            "with_gt": "voxel_map_gt.npz", "without_gt": "voxel_map_tsdf.npz",
            "grid": "every 3DGS candidate uses its reference map as reference_occupancy",
        },
        "by_variant_macro_mean": grouped,
        "failures": failures,
        "rows": rows,
    }
    _write_json(output_root / "summary.json", summary)
    print(f"[GaussianOccupancySweep] complete={len(rows)}, failed={len(failures)}; "
          f"summary={output_root / 'summary.json'}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    return run_sweep(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
