#!/usr/bin/env python3
"""Evaluate the four core planning pipelines and optional SplatNav."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GSPAR.batch._common import parser_for, run, run_scenes, select_records, summarize
from GSPAR.batch._datasets import gspar_output_path
from GSPAR.batch._rendered_depth import run_paths as run_rendered_paths
from GSPAR.evaluation.path_planning.radius_profiles import (
    DEFAULT_ADDITIONAL_ROBOT_RADII_M,
    default_robot_radius_for_dataset,
    validate_endpoint_manifest_radius,
)

BASELINE_METHODS = ('nvblox_esdf_astar_stomp', 'gspar_global_astar_ego', 'gspar_direct')


def scene_inputs(record: dict) -> dict:
    fields = record['fields']
    scene_dir = gspar_output_path(fields['output'])
    has_gt = fields.get('gt') is not None
    output = scene_dir / ('path_evaluation_gt' if has_gt else 'path_evaluation')
    return dict(scene_dir=scene_dir, output=output, endpoints=output / 'endpoints.json',
                reference=scene_dir / ('voxel_map_gt.npz' if has_gt else 'voxel_map_tsdf.npz'),
                module='GSPAR.evaluation.ground_truth' if has_gt else 'GSPAR.evaluation.path_planning',
                radius=default_robot_radius_for_dataset(record['dataset']))


def ensure_endpoints(record, args) -> None:
    inputs = scene_inputs(record)
    path = inputs['endpoints']
    radius = max([inputs['radius'], *DEFAULT_ADDITIONAL_ROBOT_RADII_M])
    if path.is_file() and not args.regenerate_endpoints:
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if len(manifest.get('pairs', [])) < args.num_trials:
            raise ValueError(f'{path} has fewer than {args.num_trials} pairs; use --regenerate-endpoints')
        validate_endpoint_manifest_radius(path, radius)
        print(f'[endpoints] reuse {path}', flush=True)
        return
    argv = ['--reference-map', str(inputs['reference']),
            '--esdf-map', str(inputs['scene_dir'] / 'sdf'),
            '--esdf-grid-metadata', str(inputs['scene_dir'] / 'voxel_map_tsdf.json'),
            '--output', str(path), '--num-pairs', str(args.num_trials),
            '--robot-radius-m', str(radius)]
    if args.dry_run:
        print('[generate endpoints] ' + ' '.join(argv), flush=True)
        return
    if inputs['module'].endswith('.ground_truth'):
        from GSPAR.evaluation.ground_truth.generate_endpoints import main
    else:
        from GSPAR.evaluation.path_planning.generate_endpoints import main
    main(argv)


def run_baseline(record, args) -> None:
    inputs = scene_inputs(record)
    argv = ['--checkpoint', str(record['fields']['3dgs model']),
            '--reference-map', str(inputs['reference']),
            '--esdf-map', str(inputs['scene_dir'] / 'sdf'),
            '--esdf-grid-metadata', str(inputs['scene_dir'] / 'voxel_map_tsdf.npz'),
            '--endpoints', str(inputs['endpoints']), '--num-trials', str(args.num_trials),
            '--robot-radius-m', str(inputs['radius']), '--rrt-base-seed', '0',
            '--methods', *BASELINE_METHODS, '--output-dir', str(inputs['output'])]
    if args.dry_run:
        print('[evaluate paths] ' + ' '.join(argv), flush=True)
        return
    if inputs['module'].endswith('.ground_truth'):
        from GSPAR.evaluation.ground_truth.batch_compare import main
    else:
        from GSPAR.evaluation.path_planning.batch_compare import main
    main(argv)


def run_splatnav(record, args) -> None:
    inputs = scene_inputs(record)
    command = ['conda', 'run', '--no-capture-output', '-n', args.splatnav_env, 'python',
               '-m', 'GSPAR.evaluation.splatnav',
               '--project-root', ROOT, '--checkpoint', record['fields']['3dgs model'],
               '--reference-map', inputs['reference'], '--endpoints', inputs['endpoints'],
               '--dataset-name', record['dataset'], '--scene-name', record['scene'],
               '--robot-radius-m', inputs['radius'], '--num-trials', args.num_trials,
               '--output-root', args.splatnav_output_root, '--overwrite']
    if args.splatnav_root is not None:
        command += ['--splatnav-root', args.splatnav_root]
    run(command, args.dry_run)


def run_scene(record, args) -> None:
    ensure_endpoints(record, args)
    run_baseline(record, args)
    rendered_map = (
        args.rendered_root / 'geometry_filter_on' / record['dataset'] /
        record['scene'] / 'nvblox-rendered_depth' / 'voxel_map_tsdf.npz'
    )
    if args.dry_run or rendered_map.is_file():
        run_rendered_paths(record, 'geometry_filter_on', args)
    else:
        print(f'[skip] optional rendered-depth map is not built: {rendered_map}', flush=True)
    if args.with_splatnav:
        run_splatnav(record, args)


def main(argv: list[str] | None = None) -> int:
    parser = parser_for(__doc__)
    parser.add_argument('--num-trials', type=int, default=100)
    parser.add_argument('--regenerate-endpoints', action='store_true',
                        help='Replace shared endpoints; rerun all pipelines for a comparable result set.')
    parser.add_argument('--splatnav-env', default='splatnav')
    parser.add_argument('--splatnav-root', type=Path)
    parser.add_argument('--with-splatnav', action='store_true',
                        help='Also run optional SplatNav evaluation (disabled by default).')
    args = parser.parse_args(argv)
    records = select_records(args, parser)
    status = run_scenes(records, run_scene, args)
    if not (status and args.fail_fast):
        summarize(args, 'paths')
    return status


if __name__ == '__main__':
    raise SystemExit(main())
