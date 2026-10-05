#!/usr/bin/env python3
"""Generate endpoints from a GT voxel reference with explicit observed free space."""

from __future__ import annotations

import json
from pathlib import Path

from GSPAR.evaluation.path_planning.generate_endpoints import build_parser
from GSPAR.evaluation.path_planning.generate_endpoints import main as standard_main


def _mark_ground_truth_manifest(output: Path) -> None:
    output = output.expanduser().resolve()
    data = json.loads(output.read_text(encoding="utf-8"))
    data.update({
        "format": "nerficg_ground_truth_component_endpoint_manifest_v2",
        "reference_type": "converted_ground_truth_voxel_reference",
        "endpoint_source": "precomputed_connected_component_rejection_sampling",
        "safety_reference": (
            "converted GT occupied voxels intersected with explicit ESDF clearance"
        ),
    })
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # The shared generator reads the already converted/aligned GT voxel NPZ;
    # no mesh loading or voxelization is repeated here.
    result = standard_main(argv, ground_truth_reference=True)
    _mark_ground_truth_manifest(args.output)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
