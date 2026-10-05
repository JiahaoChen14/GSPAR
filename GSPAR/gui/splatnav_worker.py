#!/usr/bin/env python3
"""Run one SplatNav query in the dedicated ``splatnav`` conda environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from GSPAR.evaluation.splatnav.runner import (
    PlannerConfig,
    ReusableSplatNavPlanner,
    UnscoredReference,
)


def _vec3(value: str) -> np.ndarray:
    point = np.fromstring(value, sep=",", dtype=np.float32)
    if point.shape != (3,):
        raise argparse.ArgumentTypeError("expected x,y,z")
    return point


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--splatnav-root", required=True, type=Path)
    parser.add_argument("--start", required=True, type=_vec3)
    parser.add_argument("--goal", required=True, type=_vec3)
    parser.add_argument("--robot-radius", required=True, type=float)
    parser.add_argument("--map-metadata", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    transform = np.eye(4, dtype=np.float32)
    if args.map_metadata is not None and args.map_metadata.is_file():
        with np.load(args.map_metadata) as data:
            if "map_from_training_world" in data:
                transform = np.asarray(
                    data["map_from_training_world"], dtype=np.float32
                ).reshape(4, 4)

    config = PlannerConfig(robot_radius=float(args.robot_radius), device="cuda:0")
    planner = ReusableSplatNavPlanner(
        checkpoint=args.checkpoint,
        splatnav_root=args.splatnav_root,
        robot_radius=config.robot_radius,
        vmax=config.vmax,
        amax=config.amax,
        target_cell_size_m=config.target_cell_size_m,
        maximum_cell_size_m=config.maximum_cell_size_m,
        grid_memory_budget_gib=config.grid_memory_budget_gib,
        grid_free_memory_fraction=config.grid_free_memory_fraction,
        global_connectivity=config.global_connectivity,
        min_opacity=config.min_opacity,
        bounds_quantile=config.bounds_quantile,
        bounds_padding_fraction=config.bounds_padding_fraction,
        spline_degree=config.spline_degree,
        samples_per_section=config.samples_per_section,
        device=config.device,
        map_from_training_world=transform,
    )
    record = planner.query(
        trial_id="gui",
        start=args.start,
        goal=args.goal,
        seed=0,
        reference=UnscoredReference(label="GUI unscored path"),
        compute_reference_metrics=False,
    )
    if not record.native_success or not record.path_world:
        raise RuntimeError(record.planner_message or "SplatNav found no feasible path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, np.asarray(record.path_world, dtype=np.float32))


if __name__ == "__main__":
    main()
