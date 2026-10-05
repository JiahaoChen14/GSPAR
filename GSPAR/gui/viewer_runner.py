"""Shared process launcher and viewer helpers for custom ICGui overlays."""

from __future__ import annotations

import atexit
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import torch
import torch.multiprocessing as mp

import Framework
from Cameras.Base import BaseCamera
from Datasets.Base import BaseDataset
from Datasets.utils import View
from ICGui.Applications import Viewer
from ICGui.State import LaunchConfig, SharedState
from ICGui.State.Volatile import GlobalState, OverlayState
from Logging import Logger


class InputGate:
    """Proxy an ICGui input handler while overlays temporarily own the mouse."""

    def __init__(self, input_handler):
        self._input_handler = input_handler
        self._enabled = True
        self._blocked_this_frame = False

    def __getattr__(self, name: str):
        return getattr(self._input_handler, name)

    def set_enabled(self, enabled: bool) -> None:
        if self._enabled == enabled:
            return
        self._enabled = enabled
        if enabled:
            return

        self._input_handler.backend_window.set_mouse_capture(False)
        self._input_handler._mouse_captured = False
        self._input_handler._is_mouse_rotating = False
        self._input_handler._is_mouse_panning = False
        self._input_handler.control_scheme.stop_rotation()
        self._input_handler.control_scheme.stop_panning()
        self._input_handler.control_scheme.stop_animation()

    def handle_inputs(self, delta_time: float) -> None:
        self._blocked_this_frame = not self._enabled
        if self._enabled:
            self._input_handler.handle_inputs(delta_time)
        else:
            # Overlay interactions renew the block while rendering. Reset here so
            # disabling an overlay cannot leave viewer controls permanently locked.
            self._enabled = True

    def is_pressed(self, action) -> bool:
        return not self._blocked_this_frame and self._input_handler.is_pressed(action)


def install_input_gate(viewer: Viewer) -> InputGate:
    """Install an input gate on an initialized ICGui Viewer."""
    gate = InputGate(viewer._input_handler)
    viewer._input_handler = gate
    GlobalState().input_manager = gate
    return gate


def defer_overlays(
    viewer: Viewer,
    factories: list[Callable[[], Any]],
    *,
    enabled: dict[str, bool],
) -> None:
    """Create OpenGL overlays after the viewer has initialized its GL context."""
    original_initialize = viewer._viewer.initialize

    def initialize_with_overlays() -> None:
        original_initialize()
        viewer._viewer.overlays.extend(factory() for factory in factories)

    viewer._viewer.initialize = initialize_with_overlays
    OverlayState().enabled.update(enabled)


def launch_custom_viewer(
    gui_config: LaunchConfig,
    dataset: BaseDataset,
    viewer_type: type[Viewer],
    **viewer_kwargs,
) -> tuple[SharedState, mp.Process]:
    """Launch a custom Viewer subclass using the current ICGui shared-state API."""
    shared_state = SharedState(start_as_training=False)
    shared_state.splits = [split for split in ('train', 'val', 'test') if split in dataset.subsets]
    shared_state.splits += [split for split in dataset.subsets if split not in shared_state.splits]

    dataset_poses: dict[str, list[View]] = OrderedDict()
    for split in shared_state.splits.copy():
        views = dataset.data[split]
        if not views:
            Logger.log_info(f'No poses found for split "{split}".')
            shared_state.splits.remove(split)
            continue
        dataset_poses[split] = [view.to_simple() for view in views]

    if not dataset_poses:
        raise Framework.GUIError('No poses found for any dataset split. Aborting.')

    default_view = dataset.default_view.to_simple()
    if not isinstance(default_view, View) or not isinstance(default_view.camera, BaseCamera):
        raise Framework.GUIError('No valid default view found. Aborting.')

    process = mp.Process(
        target=_launch_viewer,
        kwargs={
            'viewer_type': viewer_type,
            'gui_config': gui_config,
            'dataset_poses': dataset_poses,
            'default_view': default_view,
            'bbox_size': torch.mean(dataset.bounding_box.size).item(),
            'shared_state': shared_state,
            **viewer_kwargs,
        },
    )
    process.start()
    atexit.register(shared_state.renderer_exited)
    Logger.set_mode(Framework.config.GLOBAL.LOG_LEVEL)
    return shared_state, process


def _launch_viewer(
    *,
    viewer_type: type[Viewer],
    gui_config: LaunchConfig,
    dataset_poses: dict[str, list[View]],
    default_view: View,
    bbox_size: float,
    shared_state: SharedState,
    **viewer_kwargs,
) -> None:
    default_view.camera.shared_settings.background_color = (
        default_view.camera.shared_settings.background_color.cpu()
    )
    Framework.setup(require_custom_config=True, config_path=str(gui_config.training_config_path))
    viewer = viewer_type(
        config=gui_config,
        dataset_poses=dataset_poses,
        default_view=default_view,
        bbox_size=bbox_size,
        shared_state=shared_state,
        **viewer_kwargs,
    )
    viewer.run()
