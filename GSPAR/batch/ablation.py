#!/usr/bin/env python3
"""Run geometry-filter ablations and optional Gaussian occupancy parameter sweeps."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GSPAR.batch._common import parser_for, run_scenes, select_records, summarize
from GSPAR.batch._occupancy_sweep import run_sweep
from GSPAR.batch._filter_ablation import run_scene as run_filter_ablation
from GSPAR.batch._rendered_depth import VARIANTS, run_maps, run_paths
from GSPAR.batch.paths import ensure_endpoints


def run_scene(record, args) -> None:
    run_filter_ablation(record, args)
    ensure_endpoints(record, args)
    for variant, enabled in VARIANTS:
        run_maps(record, variant, enabled, args)
        run_paths(record, variant, args)


def main(argv: list[str] | None = None) -> int:
    parser = parser_for(__doc__)
    parser.add_argument('--experiment', choices=('geometry-filter', 'occupancy-sweep'), default='geometry-filter')
    parser.add_argument('--num-trials', type=int, default=100)
    parser.add_argument('--regenerate-endpoints', action='store_true')
    parser.add_argument('--stage', choices=('knn', 'composite', 'indoor-boundary', 'all'), default='knn')
    parser.add_argument('--selected-knn-k', type=int, choices=(8, 16, 32))
    parser.add_argument('--selected-composite', choices=('B0', 'C1', 'C2'))
    parser.add_argument('--sweep-root', type=Path, default=Path('GSPAR/outputs/gaussian_occupancy_sweep'))
    args = parser.parse_args(argv)
    records = select_records(args, parser)
    if args.experiment == 'occupancy-sweep':
        if args.stage != 'knn' and args.selected_knn_k is None:
            parser.error('--selected-knn-k is required after the knn stage')
        if args.stage in ('indoor-boundary', 'all') and args.selected_composite is None:
            parser.error('--selected-composite is required for indoor-boundary/all')
        sweep_args = argparse.Namespace(
            dataset_file=args.dataset_file,
            datasets=args.datasets,
            scenes=args.scenes,
            output_root=args.sweep_root,
            stage=args.stage,
            selected_knn_k=args.selected_knn_k,
            selected_composite=args.selected_composite,
            compute_device='cuda',
            force=False,
            continue_on_error=not args.fail_fast,
            fail_fast=args.fail_fast,
            dry_run=args.dry_run,
        )
        return run_sweep(sweep_args)
    status = run_scenes(records, run_scene, args)
    if not (status and args.fail_fast):
        summarize(args, 'ablation')
    return status


if __name__ == '__main__':
    raise SystemExit(main())
