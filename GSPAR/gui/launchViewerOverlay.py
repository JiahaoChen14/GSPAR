#! /usr/bin/env python3

"""Launch the ICGui viewer with path, voxel, and scene-visibility overlays."""

from __future__ import annotations

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_PROJECT_ROOT), str(_PROJECT_ROOT / 'src')]

import torch
import torch.multiprocessing as mp

import Framework
from Datasets.Base import BaseDataset
from ICGui.Applications import LaunchParser, Viewer
from ICGui.Backend import CheckpointRunner
from ICGui.State import LaunchConfig, SharedState
from Implementations import Datasets as DI
from Logging import Logger
from GSPAR.gui.model_visibility_extra import SceneVisibilityExtra
from GSPAR.gui.occupancy_overlay_extra import OccupancyOverlayExtra
from GSPAR.gui.path_overlay_extra import PathOverlayExtra
from GSPAR.gui.viewer_runner import (
    defer_overlays,
    install_input_gate,
    launch_custom_viewer,
)
from GSPAR.mapping.voxel_map_cli import resolve_existing_optional_path


class ViewerWithOverlay(Viewer):
    """ICGui viewer with mapping and path-planning overlays."""

    def __init__(
        self,
        *args,
        start: str = '',
        end: str = '',
        path: str = '',
        path_file: str = '',
        occupancy_file: str = '',
        extra_occupancy_file: str = '',
        checkpoint_file: str = '',
        esdf_file: str = '',
        esdf_grid_file: str = '',
        splatnav_root: str = '',
        dataset_identifier: str = '',
        robot_radius_m: float = 0.25,
        occupancy_max_points: int = 200000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        input_gate = install_input_gate(self)
        defer_overlays(
            self,
            [
                SceneVisibilityExtra,
                lambda: OccupancyOverlayExtra(
                    occupancy_file=occupancy_file,
                    extra_occupancy_file=extra_occupancy_file,
                    max_points=occupancy_max_points,
                ),
                lambda: PathOverlayExtra(
                    set_input_enabled=input_gate.set_enabled,
                    start=start,
                    end=end,
                    path=path,
                    path_file=path_file,
                    occupancy_file=occupancy_file,
                    checkpoint_file=checkpoint_file,
                    esdf_file=esdf_file,
                    esdf_grid_file=esdf_grid_file,
                    splatnav_root=splatnav_root,
                    dataset_identifier=dataset_identifier,
                    robot_radius_m=robot_radius_m,
                ),
            ],
            enabled={
                'Scene Controls': True,
                'Occupancy Overlay': True,
                'Path Overlay': True,
            },
        )
        flying_idx = self._input_handler.camera_controls.index('Flying')
        self._input_handler.choose_camera(flying_idx)


def _normalize_occupancy_arg(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return ''
    resolved = resolve_existing_optional_path(cleaned, default_suffix='.npz')
    return str(resolved) if resolved is not None else cleaned


def parse_overlay_args() -> Namespace:
    """Parse command-line arguments specific to the custom overlays."""
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--start', type=str, default='',
                        help='Start coordinate in world frame. Example: "0.0, 0.0, 0.0"')
    parser.add_argument('--end', type=str, default='',
                        help='End coordinate in world frame. Example: "1.0, 2.0, 3.0"')
    parser.add_argument('--path', type=str, default='',
                        help='Path points. Example: "0,0,0; 0.5,0,0; 1,1,0"')
    parser.add_argument('--path-file', type=str, default='',
                        help='Path file (.txt/.csv/.npy) with Nx3 coordinates.')
    parser.add_argument('--voxel-map', dest='occupancy_file', type=str, default='',
                        help='Voxel map for occupancy/path overlays. Accepts voxel_map or voxel_map.npz.')
    parser.add_argument('--extra-voxel-map', dest='extra_occupancy_file', type=str, default='',
                        help='Second voxel map shown by the occupancy overlay for alignment comparison.')
    parser.add_argument('--esdf-map', dest='esdf_file', type=str, default='',
                        help='nvblox ESDF PLY used by the compact nvblox planning route.')
    parser.add_argument('--esdf-grid-metadata', dest='esdf_grid_file', type=str, default='',
                        help='Voxel-map NPZ/JSON used to align the selected ESDF.')
    parser.add_argument('--splatnav-root', type=str, default='',
                        help='SplatNav repository root (defaults to the sibling ../splatnav).')
    parser.add_argument('--robot-radius-m', type=float, default=0.25,
                        help='Physical robot radius before dataset scaling (default: 0.25 m).')
    parser.add_argument('--occupancy-max-points', type=int, default=200000,
                        help='Maximum nearby voxels to render.')
    args = parser.parse_known_args()[0]
    args.occupancy_file = _normalize_occupancy_arg(args.occupancy_file)
    args.extra_occupancy_file = _normalize_occupancy_arg(args.extra_occupancy_file)
    return args


def launch_gui_with_overlay(
    gui_config: LaunchConfig,
    dataset: BaseDataset,
    *,
    start: str,
    end: str,
    path: str,
    path_file: str,
    occupancy_file: str,
    extra_occupancy_file: str,
    checkpoint_file: str,
    esdf_file: str,
    esdf_grid_file: str,
    splatnav_root: str,
    dataset_identifier: str,
    robot_radius_m: float,
    occupancy_max_points: int,
) -> tuple[SharedState, mp.Process]:
    return launch_custom_viewer(
        gui_config,
        dataset,
        ViewerWithOverlay,
        start=start,
        end=end,
        path=path,
        path_file=path_file,
        occupancy_file=occupancy_file,
        extra_occupancy_file=extra_occupancy_file,
        checkpoint_file=checkpoint_file,
        esdf_file=esdf_file,
        esdf_grid_file=esdf_grid_file,
        splatnav_root=splatnav_root,
        dataset_identifier=dataset_identifier,
        robot_radius_m=robot_radius_m,
        occupancy_max_points=occupancy_max_points,
    )


@torch.no_grad()
def main() -> None:
    Logger.set_mode(Logger.MODE_DEBUG)
    overlay_args = parse_overlay_args()
    config = LaunchParser.from_command_line(skip_gui_setup=True)

    Framework.setup(require_custom_config=True, config_path=config.training_config_path)
    dataset: BaseDataset = DI.get_dataset(
        dataset_type=Framework.config.GLOBAL.DATASET_TYPE,
        path=Framework.config.DATASET.PATH,
    )

    shared_state, gui_process = launch_gui_with_overlay(
        config,
        dataset,
        start=overlay_args.start,
        end=overlay_args.end,
        path=overlay_args.path,
        path_file=overlay_args.path_file,
        occupancy_file=overlay_args.occupancy_file,
        extra_occupancy_file=overlay_args.extra_occupancy_file,
        checkpoint_file=str(config.checkpoint_path),
        esdf_file=overlay_args.esdf_file,
        esdf_grid_file=overlay_args.esdf_grid_file,
        splatnav_root=overlay_args.splatnav_root,
        dataset_identifier=(
            f'{Framework.config.GLOBAL.DATASET_TYPE} {Framework.config.DATASET.PATH}'
        ),
        robot_radius_m=overlay_args.robot_radius_m,
        occupancy_max_points=overlay_args.occupancy_max_points,
    )

    model_runner = CheckpointRunner(
        dataset,
        shared_state,
        checkpoint_path=config.checkpoint_path,
        initial_resolution=config.initial_resolution,
        initial_resolution_factor=config.resolution_factor,
        rolling_average_size=config.fps_rolling_average_size,
    )
    model_runner.run(gui_process)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
