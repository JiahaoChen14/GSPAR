#!/usr/bin/env python3
"""Run the standard planner comparison with GT mesh voxels as safety reference."""

from __future__ import annotations

import json
from pathlib import Path

from GSPAR.evaluation.path_planning.batch_compare import build_parser
from GSPAR.evaluation.path_planning.batch_compare import main as standard_main


def _mark_ground_truth_run(output_dir: Path) -> None:
    root = output_dir.expanduser().resolve()
    config_paths = [root / "run_config.json"]
    config_paths.extend(sorted((root / "radius_profiles").glob("*/run_config.json")))
    for config_path in config_paths:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        data.update({
            "format": "gspar_ground_truth_path_evaluation_run_v5",
            "reference_type": "ground_truth_mesh_surface_with_observed_free_support",
            "reference_semantics": {
                "collision_and_clearance": "ground-truth mesh occupied voxels",
                "unknown_as_free": "GT collision plus common-grid bounds",
                "unknown_as_occupied": "GT collision plus explicit observed-free support",
                "strict_success": (
                    "native, global, and applicable local planner success without "
                    "fallback, exact task completion, and safety under the selected "
                    "GT reference policy"
                ),
            },
        })
        config_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = standard_main(argv)
    _mark_ground_truth_run(args.output_dir)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
