"""Shared helpers for GUI overlay modules."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import OpenGL.GL as gl
import numpy as np
from OpenGL.arrays.vbo import VBO

from Cameras.Perspective import PerspectiveCamera
from Datasets.utils import View
from ICGui.State.Volatile import GlobalState


OVERLAY_DEPTH_OCCLUDER_KEY = 'overlayDepthOccluderDrawn'
OVERLAY_OCCUPANCY_FILE_KEY = 'overlayOccupancyFile'


def select_file_dialog(
    *,
    title: str,
    initial_path: str = '',
    filetypes: tuple[tuple[str, str], ...] = (('All Files', '*'),),
) -> str | None:
    """Open a native file picker in a helper process to avoid crashing the SDL/OpenGL GUI process."""
    initial = Path(initial_path).expanduser() if initial_path.strip() else None
    initialdir = None
    initialfile = None
    if initial is not None:
        if initial.is_dir():
            initialdir = str(initial)
        else:
            initialdir = str(initial.parent)
            initialfile = initial.name

    helper_code = """
import json
import sys

try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:
    print("")
    raise SystemExit(0)

payload = json.loads(sys.argv[1])
root = tk.Tk()
root.withdraw()
root.update_idletasks()

try:
    selected = filedialog.askopenfilename(
        title=payload["title"],
        initialdir=payload["initialdir"],
        initialfile=payload["initialfile"],
        filetypes=payload["filetypes"],
    )
finally:
    root.destroy()

print(selected or "")
"""
    payload = json.dumps(
        {
            'title': title,
            'initialdir': initialdir,
            'initialfile': initialfile,
            'filetypes': list(filetypes),
        }
    )
    try:
        result = subprocess.run(
            [sys.executable, '-c', helper_code, payload],
            check=False,
            capture_output=True,
            text=True,
            close_fds=True,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    selected = result.stdout.strip()
    return selected or None


def select_directory_dialog(*, title: str, initial_path: str = '') -> str | None:
    """Open a native directory picker without sharing Tk state with SDL."""
    initial = Path(initial_path).expanduser() if initial_path.strip() else None
    if initial is not None and not initial.is_dir():
        initial = initial.parent

    helper_code = """
import json
import sys

try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:
    print("")
    raise SystemExit(0)

payload = json.loads(sys.argv[1])
root = tk.Tk()
root.withdraw()
root.update_idletasks()

try:
    selected = filedialog.askdirectory(
        title=payload["title"],
        initialdir=payload["initialdir"],
        mustexist=True,
    )
finally:
    root.destroy()

print(selected or "")
"""
    payload = json.dumps(
        {
            'title': title,
            'initialdir': None if initial is None else str(initial),
        }
    )
    try:
        result = subprocess.run(
            [sys.executable, '-c', helper_code, payload],
            check=False,
            capture_output=True,
            text=True,
            close_fds=True,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    selected = result.stdout.strip()
    return selected or None


def update_vbo(current_vbo: VBO | None, vertices) -> VBO:
    """Create or update a VBO with the provided vertex array."""
    if current_vbo is None:
        return VBO(vertices, gl.GL_STATIC_DRAW)
    current_vbo.set_array(vertices)
    return current_vbo


def get_view_projection(view: View) -> tuple[np.ndarray, np.ndarray]:
    """Return matrices that project raw model-world points in OpenGL.

    ``view.w2c`` consumes the model's COLMAP-style world coordinates. OpenGL
    only needs the left-side camera-coordinate conversion. Conjugating with
    ``model_transform`` would require every overlay point to be transformed as
    well and otherwise flips Y/Z relative to the rendered 3DGS.
    """
    if not isinstance(view.camera, PerspectiveCamera):
        raise TypeError(f'GUI overlays require PerspectiveCamera, got {type(view.camera).__name__}')

    model_transform = GlobalState().input_manager.control_scheme.model_transform
    view_matrix = model_transform @ view.w2c_numpy
    projection = view.camera.get_projection_matrix(invert_z=True)
    return (
        view_matrix.astype(np.float32).copy(order='C'),
        projection.cpu().numpy().astype(np.float32).copy(order='C'),
    )


def restore_gl_bool_state(enabled: bool, cap: int) -> None:
    """Restore a boolean OpenGL capability flag to its previous state."""
    if enabled:
        gl.glEnable(cap)
    else:
        gl.glDisable(cap)
