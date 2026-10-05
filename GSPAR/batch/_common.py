"""Shared CLI selection and command execution for the three batch modules."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

from GSPAR.batch._datasets import load_records

ROOT = Path(__file__).resolve().parents[2]
RENDERED_ROOT = Path('GSPAR/outputs/nvblox_rendered_depth_evaluation')
ABLATION_ROOT = Path('GSPAR/outputs/filter_ablation')
SPLATNAV_ROOT = Path('GSPAR/outputs/splatnav_evaluation')


def parser_for(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--dataset-file', type=Path, default=Path('configs/dataset.txt'))
    parser.add_argument('--dataset', dest='datasets', action='append', help='Repeat to select datasets; default: all.')
    parser.add_argument('--scenes', help='Comma-separated scene names; default: all.')
    parser.add_argument('--rendered-root', type=Path, default=RENDERED_ROOT)
    parser.add_argument('--ablation-root', type=Path, default=ABLATION_ROOT)
    parser.add_argument('--splatnav-output-root', type=Path, default=SPLATNAV_ROOT)
    parser.add_argument('--skip-summary', action='store_true')
    parser.add_argument('--fail-fast', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without writing files or running workloads.')
    return parser


def select_records(args, parser) -> list[dict]:
    os.chdir(ROOT)
    datasets = set(args.datasets or [])
    scenes = {item.strip() for item in (args.scenes or '').split(',') if item.strip()}
    records = [record for record in load_records(args.dataset_file)
               if (not datasets or record['dataset'] in datasets)
               and (not scenes or record['scene'] in scenes)]
    if not records:
        parser.error('No scenes matched --dataset/--scenes')
    if hasattr(args, 'num_trials') and args.num_trials <= 0:
        parser.error('--num-trials must be positive')
    return records


def run(command, dry_run: bool) -> None:
    command = [str(part) for part in command]
    print('$ ' + shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def run_scenes(records, action, args) -> int:
    failed = False
    for record in records:
        print(f"\n[{record['dataset']}/{record['scene']}]", flush=True)
        try:
            action(record, args)
        except Exception as error:
            failed = True
            print(f"[failed] {type(error).__name__}: {error}", file=sys.stderr)
            if args.fail_fast:
                return 1
    return int(failed)


def summarize(args, module: str) -> None:
    if args.skip_summary:
        return
    command = ['--dataset-file', str(args.dataset_file),
               '--rendered-root', args.rendered_root,
               '--ablation-root', args.ablation_root,
               '--splatnav-root', args.splatnav_output_root,
               '--output-dir', Path('GSPAR/outputs/batch_summary') / module]
    for dataset in args.datasets or []:
        command += ['--dataset', dataset]
    if args.scenes:
        command += ['--scenes', args.scenes]
    command = [str(part) for part in command]
    if args.dry_run:
        print('[summarize] ' + shlex.join(command), flush=True)
        return
    from GSPAR.evaluation.summarize_evaluation_suite import main as summarize_main

    summarize_main(command)
