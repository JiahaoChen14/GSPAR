#!/usr/bin/env python3
"""Evaluate voxel maps produced by the per-scene command files."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GSPAR.batch._common import parser_for, run_scenes, select_records, summarize
from GSPAR.batch._datasets import gspar_output_path
from GSPAR.evaluation.ground_truth.occupancy_quality import evaluate as evaluate_gt
from GSPAR.evaluation.occupancy_map_quality import evaluate as evaluate_occupancy


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def run_scene(record, args) -> None:
    scene_dir = gspar_output_path(record['fields']['output'])
    prediction = scene_dir / 'voxel_map.npz'
    tsdf = scene_dir / 'voxel_map_tsdf.npz'
    has_gt = record['fields'].get('gt') is not None
    required = [prediction, tsdf]
    if has_gt:
        required.append(scene_dir / 'voxel_map_gt.npz')
    if args.dry_run:
        print('[evaluate maps] ' + ', '.join(str(path) for path in required), flush=True)
        return
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            'Missing prebuilt map inputs; run the generated scene commands first: '
            + ', '.join(missing)
        )

    _write_json(
        scene_dir / 'eval' / 'sparse_vs_nvblox_quality.json',
        evaluate_occupancy(scene_dir, prediction, tsdf),
    )
    if has_gt:
        reference = scene_dir / 'voxel_map_gt.npz'
        _write_json(
            scene_dir / 'eval_gt' / 'voxel_vs_gt_quality.json',
            evaluate_gt(
                scene_dir,
                reference,
                [('GSPAR', prediction), ('nvblox', tsdf)],
            ),
        )


def main(argv: list[str] | None = None) -> int:
    parser = parser_for(__doc__)
    args = parser.parse_args(argv)
    records = select_records(args, parser)
    status = run_scenes(records, run_scene, args)
    if not (status and args.fail_fast):
        summarize(args, 'mapping')
    return status


if __name__ == '__main__':
    raise SystemExit(main())
