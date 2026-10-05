"""Path/start-end and compact planning overlay module."""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np
import OpenGL.GL as gl
import torch
from imgui_bundle import imgui
from OpenGL.arrays.vbo import VBO
from OpenGL.GL.shaders import ShaderProgram, compileProgram, compileShader

from Cameras.Perspective import PerspectiveCamera
from Datasets.utils import View
from ICGui.ModelViewer.Overlays import BaseOverlay
from ICGui.ModelViewer.Shaders import ShaderProgram as ViewerShaderProgram
from ICGui.ModelViewer.Shaders.ContextManagers import (
    bind_shader_program,
    bind_textures,
    bind_vertex_attributes,
    bind_vertex_buffer,
)
from ICGui.ModelViewer.Shaders.utils import fill_uniforms
from ICGui.State.Volatile import CameraState, GlobalState

from GSPAR.gui.overlay_common import (
    OVERLAY_DEPTH_OCCLUDER_KEY,
    OVERLAY_OCCUPANCY_FILE_KEY,
    get_view_projection,
    restore_gl_bool_state,
    select_directory_dialog,
    select_file_dialog,
    update_vbo,
)
from GSPAR.evaluation.path_planning.radius_profiles import (
    DATASET_REFERENCE_ROBOT_RADIUS_M,
    robot_radius_scale_for_dataset,
)
from GSPAR.mapping.voxel_map_cli import resolve_existing_path
from GSPAR.path.local_path.ego_occupancy_path import (
    CoarsePath,
    read_coarse_path,
)
from GSPAR.path.planning import (
    EgoPlanOptions,
    EsdfGlobalPlanOptions,
    GLOBAL_ESDF_WEIGHTED_ASTAR,
    GLOBAL_JPS3D,
    GlobalPlanOptions,
    GSPARDirectPlanOptions,
    StompEsdfPlanOptions,
    load_esdf_graph,
    load_voxel_payload,
    run_ego_path,
    run_esdf_global_path,
    run_global_path,
    run_gspar_direct_path,
    run_stomp_esdf_path,
)
from GSPAR.path.method_names import GSPAR_DIRECT_LABEL


COMPACT_GSPAR_DIRECT_METHOD = 'gspar-direct'
COMPACT_GSPAR_GLOBAL_METHOD = 'gspar-global'
COMPACT_NVBLOX_ESDF_METHOD = 'nvblox-esdf (raw/rendered depth)'
COMPACT_SPLATNAV_METHOD = 'splatnav'
DEFAULT_PHYSICAL_ROBOT_RADIUS_M = DATASET_REFERENCE_ROBOT_RADIUS_M


class PathOverlayExtra(BaseOverlay):
    """OpenGL extra that shows start/end points and a polyline path in world coordinates."""

    def __init__(
        self,
        set_input_enabled: Callable[[bool], None],
        start: str = '',
        end: str = '',
        path: str = '',
        path_file: str = '',
        occupancy_file: str = '',
        checkpoint_file: str = '',
        esdf_file: str = '',
        esdf_grid_file: str = '',
        splatnav_root: str = '',
        dataset_identifier: str = '',
        robot_radius_m: float = DEFAULT_PHYSICAL_ROBOT_RADIUS_M,
    ):
        super().__init__()
        self._set_input_enabled = set_input_enabled
        self._point_shader_program: ShaderProgram = self._build_point_shader()
        self._line_shader_program: ShaderProgram = self._build_line_shader()
        self._point_vao_location = gl.glGenVertexArrays(1)
        self._line_vao_location = gl.glGenVertexArrays(1)

        self._uniform_locations: dict[str, dict[str, tuple[int, str, tuple]]] = {
            'point': {
                'view': (gl.glGetUniformLocation(self._point_shader_program, 'view'), 'mat4', (True, )),
                'projection': (gl.glGetUniformLocation(self._point_shader_program, 'projection'), 'mat4', (True, )),
                'pointSize': (gl.glGetUniformLocation(self._point_shader_program, 'pointSize'), 'float', tuple()),
                'viewportSize': (gl.glGetUniformLocation(self._point_shader_program, 'viewportSize'), 'vec2', tuple()),
                'near': (gl.glGetUniformLocation(self._point_shader_program, 'near'), 'float', tuple()),
                'far': (gl.glGetUniformLocation(self._point_shader_program, 'far'), 'float', tuple()),
                'color': (gl.glGetUniformLocation(self._point_shader_program, 'color'), 'vec4', tuple()),
                'sceneOcclusionEnabled': (
                    gl.glGetUniformLocation(self._point_shader_program, 'sceneOcclusionEnabled'),
                    'int',
                    tuple(),
                ),
                'occlusionAlphaThreshold': (
                    gl.glGetUniformLocation(self._point_shader_program, 'occlusionAlphaThreshold'),
                    'float',
                    tuple(),
                ),
            },
            'line': {
                'view': (gl.glGetUniformLocation(self._line_shader_program, 'viewMatrix'), 'mat4', (True, )),
                'projection': (gl.glGetUniformLocation(self._line_shader_program, 'projectionMatrix'), 'mat4', (True, )),
                'lineWidth': (gl.glGetUniformLocation(self._line_shader_program, 'lineWidth'), 'float', tuple()),
                'viewportSize': (gl.glGetUniformLocation(self._line_shader_program, 'viewportSize'), 'vec2', tuple()),
                'near': (gl.glGetUniformLocation(self._line_shader_program, 'near'), 'float', tuple()),
                'far': (gl.glGetUniformLocation(self._line_shader_program, 'far'), 'float', tuple()),
                'color': (gl.glGetUniformLocation(self._line_shader_program, 'color'), 'vec4', tuple()),
                'highlightColor': (gl.glGetUniformLocation(self._line_shader_program, 'highlightColor'), 'vec4', tuple()),
                'sceneOcclusionEnabled': (
                    gl.glGetUniformLocation(self._line_shader_program, 'sceneOcclusionEnabled'),
                    'int',
                    tuple(),
                ),
                'occlusionAlphaThreshold': (
                    gl.glGetUniformLocation(self._line_shader_program, 'occlusionAlphaThreshold'),
                    'float',
                    tuple(),
                ),
            },
        }

        self._show_points = True
        self._show_path = True
        self._point_size = 0.02
        self._line_width = 0.006
        self._enable_scene_occlusion = True
        self._occlusion_alpha_threshold = 0.5
        self._start_color = np.array([0.05, 0.95, 0.15, 1.0], dtype=np.float32)
        self._end_color = np.array([0.95, 0.15, 0.05, 1.0], dtype=np.float32)
        self._path_color = np.array([1.0, 0.95, 0.1, 1.0], dtype=np.float32)
        self._show_gizmo = True
        self._gizmo_target = 'start'
        self._gizmo_axis_length = 0.12
        self._gizmo_handle_point_size = 0.028
        self._gizmo_line_width = 0.008
        self._gizmo_hit_threshold_px = 14.0
        self._gizmo_colors = {
            'x': np.array([0.98, 0.18, 0.12, 1.0], dtype=np.float32),
            'y': np.array([0.10, 0.86, 0.22, 1.0], dtype=np.float32),
            'z': np.array([0.20, 0.52, 0.98, 1.0], dtype=np.float32),
        }
        self._gizmo_highlight_scale = 1.18

        self._start_input = start
        self._end_input = end
        self._path_input = path
        self._path_file_input = path_file
        self._path_export_file_input = ''
        self._path_loaded_file_path: str | None = None
        self._path_header: dict[str, str] = {}
        self._path_raw_header_lines: list[str] = []
        self._planner_occupancy_file_input = occupancy_file
        self._planner_checkpoint_file = checkpoint_file
        self._compact_planning_methods = [
            COMPACT_GSPAR_DIRECT_METHOD,
            COMPACT_GSPAR_GLOBAL_METHOD,
            COMPACT_NVBLOX_ESDF_METHOD,
            COMPACT_SPLATNAV_METHOD,
        ]
        self._compact_planning_method = self._compact_planning_methods[0]
        self._dataset_identifier = dataset_identifier.strip() or 'unknown'
        self._robot_radius_physical_m = float(robot_radius_m)
        self._robot_radius_scale = float(
            robot_radius_scale_for_dataset(self._dataset_identifier, default=1.0)
        )
        default_splatnav_root = Path(__file__).resolve().parents[2].parent / 'splatnav'
        self._splatnav_root_input = splatnav_root or str(default_splatnav_root)
        self._standalone_last_result = None
        self._planner_esdf_file_input = esdf_file
        self._planner_esdf_grid_file_input = esdf_grid_file or occupancy_file
        self._status = 'Set start/end/path and click "Apply".'

        self._start: np.ndarray | None = None
        self._end: np.ndarray | None = None
        self._path_points: np.ndarray | None = None

        self._start_vbo: VBO | None = None
        self._end_vbo: VBO | None = None
        self._path_vbo: VBO | None = None
        self._path_vertex_count = 0
        self._gizmo_line_vbos: dict[str, VBO | None] = {'x': None, 'y': None, 'z': None}
        self._gizmo_line_vertex_counts: dict[str, int] = {'x': 0, 'y': 0, 'z': 0}
        self._gizmo_handle_vbos: dict[str, VBO | None] = {'x': None, 'y': None, 'z': None}
        self._geometry_dirty = True
        self._dataset_split_for_points = 'train'
        self._dataset_start_idx = 0
        self._dataset_end_idx = 0
        self._planner_esdf_graph = None
        self._planner_esdf_cache_key: tuple[str, str] | None = None
        self._planner_esdf_loaded_file_path: str | None = None
        self._planner_last_start_idx: np.ndarray | None = None
        self._planner_last_goal_idx: np.ndarray | None = None
        self._planner_last_start_world: np.ndarray | None = None
        self._planner_last_goal_world: np.ndarray | None = None
        self._planner_last_start_was_snapped = False
        self._planner_last_goal_was_snapped = False
        self._pending_pick_target: str | None = None
        self._picker_framebuffer = gl.glGenFramebuffers(1)
        self._gizmo_drag_axis: str | None = None
        self._gizmo_drag_start_mouse: np.ndarray | None = None
        self._gizmo_drag_origin_point: np.ndarray | None = None
        self._gizmo_drag_world_axis: np.ndarray | None = None
        self._gizmo_drag_screen_dir: np.ndarray | None = None
        self._gizmo_drag_pixels_per_unit = 1.0

        self._try_apply_inputs()

    @property
    def name(self) -> str:
        return 'Path Overlay'

    @staticmethod
    def _viewer_shader_source(shader_path: str) -> str:
        with open(ViewerShaderProgram.SHADER_ROOT / shader_path, 'r', encoding='utf-8') as shader_file:
            return shader_file.read()

    @classmethod
    def _compile_viewer_shader(cls, shader_path: str) -> int:
        shader_type = ViewerShaderProgram.determine_shader_type(shader_path)
        return compileShader(cls._viewer_shader_source(shader_path), shader_type)

    @staticmethod
    def _occlusion_line_fragment_shader() -> str:
        return """
        #version 420

        layout(location=0) in float depth;
        layout(location=1) in float highlighted;

        out vec4 fragColor;

        uniform float near;
        uniform float far;
        uniform vec2 viewportSize;
        uniform vec4 color;
        uniform vec4 highlightColor;
        uniform bool sceneOcclusionEnabled;
        uniform float occlusionAlphaThreshold;

        layout(binding=0) uniform sampler2D depthSampler;
        layout(binding=1) uniform sampler2D alphaSampler;

        void main() {
            vec4 col = color;
            if (highlighted > 0.5) { col = highlightColor; }

            if (sceneOcclusionEnabled) {
                vec2 uv = gl_FragCoord.xy / viewportSize;
                uv.y = 1.0 - uv.y;

                float sceneDepth = texture(depthSampler, uv).r;
                float sceneAlpha = texture(alphaSampler, uv).r;
                if (!(sceneDepth < near || sceneDepth > depth)) {
                    if (sceneAlpha >= occlusionAlphaThreshold) {
                        discard;
                    }
                }
            }

            fragColor = col;
        }
        """

    @staticmethod
    def _occlusion_point_fragment_shader() -> str:
        return """
        #version 450

        in vec2 uv;
        in float depth;

        uniform float near;
        uniform float far;
        uniform vec2 viewportSize;
        uniform vec4 color;
        uniform bool sceneOcclusionEnabled;
        uniform float occlusionAlphaThreshold;

        layout(binding=0) uniform sampler2D depthSampler;
        layout(binding=1) uniform sampler2D alphaSampler;

        out vec4 fragColor;

        void main() {
            if (uv.x * uv.x + uv.y * uv.y > 1.0) {
                discard;
            }

            if (sceneOcclusionEnabled) {
                vec2 screenUv = gl_FragCoord.xy / viewportSize;
                screenUv.y = 1.0 - screenUv.y;

                float sceneDepth = texture(depthSampler, screenUv).r;
                float sceneAlpha = texture(alphaSampler, screenUv).r;
                if (!(sceneDepth < near || sceneDepth > depth)) {
                    if (sceneAlpha >= occlusionAlphaThreshold) {
                        discard;
                    }
                }
            }

            fragColor = color;
        }
        """

    @classmethod
    def _build_point_shader(cls) -> int:
        return compileProgram(
            cls._compile_viewer_shader('Point.vert'),
            compileShader(cls._occlusion_point_fragment_shader(), gl.GL_FRAGMENT_SHADER),
        )

    @classmethod
    def _build_line_shader(cls) -> int:
        return compileProgram(
            cls._compile_viewer_shader('Identity.vert'),
            cls._compile_viewer_shader('LineThickness.geom'),
            compileShader(cls._occlusion_line_fragment_shader(), gl.GL_FRAGMENT_SHADER),
        )

    @staticmethod
    def _parse_vec3(text: str) -> np.ndarray:
        cleaned = text.replace(',', ' ').strip()
        if cleaned == '':
            raise ValueError('empty coordinate')
        parts = [p for p in cleaned.split() if p]
        if len(parts) != 3:
            raise ValueError('expected 3 values: x y z')
        return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float32)

    @classmethod
    def _parse_path(cls, text: str) -> np.ndarray:
        cleaned = text.replace(';', '\n')
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if len(lines) == 0:
            raise ValueError('empty path')
        return np.stack([cls._parse_vec3(line) for line in lines], axis=0).astype(np.float32)

    @staticmethod
    def _header_dict_from_lines(lines: list[str]) -> dict[str, str]:
        header: dict[str, str] = {}
        for line in lines:
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            header[key.strip()] = value.strip()
        return header

    def _set_current_path(
        self,
        points: np.ndarray,
        *,
        source_path: str | None = None,
        header: dict[str, str] | None = None,
        raw_header_lines: list[str] | None = None,
        sync_endpoints: bool = True,
    ) -> None:
        arr = np.asarray(points, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError('path must contain Nx3 coordinates')
        if arr.shape[0] == 0:
            raise ValueError('path is empty')
        self._path_points = arr
        self._path_input = '\n'.join(f'{p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f}' for p in arr)
        if sync_endpoints:
            start = arr[0].astype(np.float32, copy=True)
            end = arr[-1].astype(np.float32, copy=True) if arr.shape[0] > 1 else None
            self._set_start_end(start, end)
        self._path_loaded_file_path = source_path
        self._path_header = dict(header or {})
        self._path_raw_header_lines = list(raw_header_lines or [])
        self._geometry_dirty = True


    def _load_path_file(self, file_path: str):
        path = Path(file_path).expanduser()
        if not path.is_file():
            raise ValueError(f'path file not found: {path}')
        header: dict[str, str] = {}
        raw_header_lines: list[str] = []
        if path.suffix.lower() == '.csv':
            try:
                arr = np.loadtxt(path, delimiter=',')
            except ValueError:
                arr = np.loadtxt(path, delimiter=None)
        else:
            try:
                coarse = read_coarse_path(path)
                arr = coarse.points
                header = coarse.header
                raw_header_lines = coarse.raw_header_lines
            except ValueError:
                arr = np.loadtxt(path, delimiter=',')
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != 3:
            raise ValueError('path file must contain Nx3 coordinates')
        self._set_current_path(
            arr,
            source_path=str(path.resolve()),
            header=header,
            raw_header_lines=raw_header_lines,
            sync_endpoints=True,
        )

    def _current_coarse_path(self) -> CoarsePath:
        points = self._compose_path_points()
        if points is None or points.shape[0] < 2:
            raise ValueError('current path must contain at least two points')
        return CoarsePath(
            points=points.astype(np.float32),
            header=dict(self._path_header),
            raw_header_lines=list(self._path_raw_header_lines),
            path=Path(self._path_loaded_file_path) if self._path_loaded_file_path else None,
        )

    def _default_path_export_path(self) -> Path:
        occupancy = self._planner_occupancy_file_input.strip()
        if occupancy:
            return Path(occupancy).expanduser().resolve().parent / 'paths' / 'planned_path.txt'
        if self._path_loaded_file_path:
            return Path(self._path_loaded_file_path).resolve().with_name('planned_path.txt')
        return Path.cwd() / 'path_exports' / 'planned_path.txt'

    def _export_current_path(self) -> None:
        points = self._compose_path_points()
        if points is None or points.shape[0] == 0:
            raise ValueError('no path to export')
        output = Path(self._path_export_file_input).expanduser() if self._path_export_file_input.strip() else self._default_path_export_path()
        if output.suffix.lower() not in {'.txt', '.csv', '.npy'}:
            output = output.with_suffix('.txt')
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            stem, suffix = output.stem, output.suffix
            index = 2
            while output.exists():
                output = output.with_name(f'{stem}_{index}{suffix}')
                index += 1
        if output.suffix.lower() == '.npy':
            np.save(output, points.astype(np.float32))
        else:
            header_lines = list(self._path_raw_header_lines)
            if not header_lines:
                header_lines = ['NERFICG path'] + [f'{key}: {value}' for key, value in self._path_header.items()]
            np.savetxt(
                output,
                points,
                fmt='%.6f',
                delimiter=',' if output.suffix.lower() == '.csv' else ' ',
                header='\n'.join(header_lines),
                comments='# ',
            )
        self._path_export_file_input = str(output)
        self._status = f'Exported path: {output} ({points.shape[0]} points).'






    def _fill_esdf_inputs_from_occupancy_map(self) -> None:
        occupancy_path = self._planner_occupancy_file_input.strip()
        if occupancy_path == '':
            return
        try:
            grid_path = resolve_existing_path(occupancy_path, default_suffix='.npz')
        except Exception:
            grid_path = Path(occupancy_path).expanduser()
        scene_dir = grid_path.resolve().parent
        esdf_path = scene_dir / 'sdf' / 'esdf_observed.ply'
        if grid_path.is_file() and self._planner_esdf_grid_file_input.strip() == '':
            self._planner_esdf_grid_file_input = str(grid_path.resolve())
        if self._planner_esdf_file_input.strip() == '' and esdf_path.is_file():
            self._planner_esdf_file_input = str(esdf_path.resolve())




    def _sync_current_occupancy_file(self, extra_params: dict[str, Any]) -> None:
        occupancy_file = str(extra_params.get(OVERLAY_OCCUPANCY_FILE_KEY, '') or '').strip()
        if occupancy_file == '':
            return
        try:
            resolved = str(resolve_existing_path(occupancy_file, default_suffix='.npz'))
        except Exception:
            resolved = str(Path(occupancy_file).expanduser().resolve())
        current = self._planner_occupancy_file_input.strip()
        try:
            current_resolved = str(resolve_existing_path(current, default_suffix='.npz')) if current else ''
        except Exception:
            current_resolved = str(Path(current).expanduser().resolve()) if current else ''
        if resolved == current_resolved:
            return
        self._planner_occupancy_file_input = resolved
        self._planner_esdf_graph = None
        self._planner_esdf_cache_key = None
        self._planner_esdf_loaded_file_path = None
        if self._planner_esdf_grid_file_input.strip() == '':
            self._planner_esdf_grid_file_input = resolved





    def _path_browse_initial_path(self) -> str:
        current = self._path_file_input.strip()
        if current:
            return current
        occupancy = self._planner_occupancy_file_input.strip()
        return str(Path(occupancy).expanduser().parent) if occupancy else ''

    def _global_map_browse_initial_path(self) -> str:
        current = self._planner_occupancy_file_input.strip()
        return current

    def _esdf_browse_initial_path(self) -> str:
        return self._planner_esdf_file_input.strip() or self._global_map_browse_initial_path()



    def _set_start_end(self, start: np.ndarray | None, end: np.ndarray | None):
        self._start = start
        self._end = end
        self._geometry_dirty = True
        if self._start is not None:
            self._start_input = f'{self._start[0]:.6f}, {self._start[1]:.6f}, {self._start[2]:.6f}'
        if self._end is not None:
            self._end_input = f'{self._end[0]:.6f}, {self._end[1]:.6f}, {self._end[2]:.6f}'

    def _set_endpoint(self, target: str, point: np.ndarray):
        if target == 'start':
            self._set_start_end(point.astype(np.float32), self._end)
            return
        if target == 'end':
            self._set_start_end(self._start, point.astype(np.float32))
            return
        raise ValueError(f'unknown endpoint target "{target}"')

    def _get_endpoint(self, target: str) -> np.ndarray | None:
        if target == 'start':
            return self._start
        if target == 'end':
            return self._end
        raise ValueError(f'unknown endpoint target "{target}"')

    @staticmethod
    def _axis_vector(axis_name: str) -> np.ndarray:
        if axis_name == 'x':
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if axis_name == 'y':
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if axis_name == 'z':
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        raise ValueError(f'unknown axis "{axis_name}"')

    @staticmethod
    def _axis_arrow_side(axis_name: str) -> np.ndarray:
        if axis_name == 'x':
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if axis_name == 'y':
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if axis_name == 'z':
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        raise ValueError(f'unknown axis "{axis_name}"')

    @staticmethod
    def _highlight_color(color: np.ndarray, scale: float) -> np.ndarray:
        rgba = color.copy()
        rgba[:3] = np.clip(rgba[:3] * scale, 0.0, 1.0)
        return rgba

    def _set_mouse_pick_enabled(self, enabled: bool):
        self._set_input_enabled(not enabled)

    def _try_apply_inputs(self):
        try:
            start = self._parse_vec3(self._start_input) if self._start_input.strip() else None
            end = self._parse_vec3(self._end_input) if self._end_input.strip() else None
            path_points = self._parse_path(self._path_input) if self._path_input.strip() else None
            if path_points is not None and start is None:
                start = path_points[0].astype(np.float32, copy=True)
            if path_points is not None and end is None and path_points.shape[0] > 1:
                end = path_points[-1].astype(np.float32, copy=True)
            self._set_start_end(start, end)
            self._path_points = path_points
            self._path_loaded_file_path = None
            self._path_header = {}
            self._path_raw_header_lines = []
            self._geometry_dirty = True
            self._status = 'Coordinates applied.'
        except ValueError as exc:
            self._status = f'Parse error: {exc}'

    def _try_apply_endpoint_inputs(self) -> bool:
        try:
            start = self._parse_vec3(self._start_input) if self._start_input.strip() else None
            end = self._parse_vec3(self._end_input) if self._end_input.strip() else None
            self._set_start_end(start, end)
            return True
        except ValueError as exc:
            self._status = f'Endpoint parse error: {exc}'
            return False






















    def _compose_path_points(self) -> np.ndarray | None:
        if self._path_points is None or self._path_points.shape[0] == 0:
            return None
        points: list[np.ndarray] = []
        if self._start is not None:
            points.append(self._start.astype(np.float32))
        points.extend(self._path_points.astype(np.float32))
        if self._end is not None:
            points.append(self._end.astype(np.float32))
        if len(points) == 0:
            return None

        merged = [points[0]]
        for point in points[1:]:
            if np.linalg.norm(point - merged[-1]) > 1e-6:
                merged.append(point)
        if len(merged) == 0:
            return None
        return np.stack(merged, axis=0).astype(np.float32)

    def _build_line_vertices(self, points: np.ndarray) -> np.ndarray:
        if points.shape[0] < 2:
            return np.zeros((0, 3), dtype=np.float32)
        segments = np.zeros(((points.shape[0] - 1) * 2, 3), dtype=np.float32)
        segments[0::2] = points[:-1]
        segments[1::2] = points[1:]
        return segments

    @staticmethod
    def _expand_point_billboards(points: np.ndarray) -> np.ndarray:
        if points.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return np.repeat(points.astype(np.float32), 6, axis=0)








    def _ensure_geometry(self):
        if not self._geometry_dirty:
            return

        self._start_vbo = None
        self._end_vbo = None
        self._path_vbo = None
        self._path_vertex_count = 0
        for axis_name in ('x', 'y', 'z'):
            self._gizmo_line_vbos[axis_name] = None
            self._gizmo_line_vertex_counts[axis_name] = 0
            self._gizmo_handle_vbos[axis_name] = None

        if self._start is not None:
            self._start_vbo = update_vbo(
                self._start_vbo,
                self._expand_point_billboards(self._start.reshape(1, 3)),
            )
        if self._end is not None:
            self._end_vbo = update_vbo(
                self._end_vbo,
                self._expand_point_billboards(self._end.reshape(1, 3)),
            )

        display_path_points = self._compose_path_points()
        if display_path_points is not None and display_path_points.shape[0] > 1:
            line_vertices = self._build_line_vertices(display_path_points)
            self._path_vertex_count = int(line_vertices.shape[0])
            if self._path_vertex_count > 0:
                self._path_vbo = update_vbo(self._path_vbo, line_vertices)

        gizmo_point = self._get_endpoint(self._gizmo_target) if self._show_gizmo else None
        if gizmo_point is not None:
            for axis_name in ('x', 'y', 'z'):
                line_vertices, handle_vertices = self._build_gizmo_axis_vertices(gizmo_point, axis_name)
                self._gizmo_line_vertex_counts[axis_name] = int(line_vertices.shape[0])
                if self._gizmo_line_vertex_counts[axis_name] > 0:
                    self._gizmo_line_vbos[axis_name] = update_vbo(self._gizmo_line_vbos[axis_name], line_vertices)
                if handle_vertices.shape[0] > 0:
                    self._gizmo_handle_vbos[axis_name] = update_vbo(self._gizmo_handle_vbos[axis_name], handle_vertices)

        self._geometry_dirty = False

    def _build_gizmo_axis_vertices(self, origin: np.ndarray, axis_name: str) -> tuple[np.ndarray, np.ndarray]:
        axis = self._axis_vector(axis_name)
        side = self._axis_arrow_side(axis_name)
        length = float(self._gizmo_axis_length)
        head_length = 0.22 * length
        head_width = 0.10 * length

        tip = origin + axis * length
        branch_a = tip - axis * head_length + side * head_width
        branch_b = tip - axis * head_length - side * head_width
        line_vertices = np.asarray(
            [
                origin, tip,
                tip, branch_a,
                tip, branch_b,
            ],
            dtype=np.float32,
        )
        handle_vertices = self._expand_point_billboards(tip.reshape(1, 3))
        return line_vertices, handle_vertices

    @staticmethod
    def _extract_position_from_pose(pose: View) -> np.ndarray:
        return pose.position_numpy.astype(np.float32)

    def _apply_start_from_current_camera(self):
        cam_pos = CameraState().current_view.position_numpy.astype(np.float32)
        self._start = cam_pos
        self._gizmo_target = 'start'
        self._start_input = f'{cam_pos[0]:.6f}, {cam_pos[1]:.6f}, {cam_pos[2]:.6f}'
        self._geometry_dirty = True
        self._status = 'Set start from current camera position.'

    def _apply_end_from_current_camera(self):
        cam_pos = CameraState().current_view.position_numpy.astype(np.float32)
        self._end = cam_pos
        self._gizmo_target = 'end'
        self._end_input = f'{cam_pos[0]:.6f}, {cam_pos[1]:.6f}, {cam_pos[2]:.6f}'
        self._geometry_dirty = True
        self._status = 'Set end from current camera position.'

    def _apply_points_from_dataset_poses(self):
        dataset_poses = CameraState().dataset_poses
        if self._dataset_split_for_points not in dataset_poses:
            self._status = f'Split "{self._dataset_split_for_points}" not available.'
            return
        poses = dataset_poses[self._dataset_split_for_points]
        if len(poses) == 0:
            self._status = f'Split "{self._dataset_split_for_points}" has no poses.'
            return

        start_idx = int(max(0, min(self._dataset_start_idx, len(poses) - 1)))
        end_idx = int(max(0, min(self._dataset_end_idx, len(poses) - 1)))
        self._dataset_start_idx = start_idx
        self._dataset_end_idx = end_idx

        self._start = self._extract_position_from_pose(poses[start_idx])
        self._end = self._extract_position_from_pose(poses[end_idx])
        self._start_input = f'{self._start[0]:.6f}, {self._start[1]:.6f}, {self._start[2]:.6f}'
        self._end_input = f'{self._end[0]:.6f}, {self._end[1]:.6f}, {self._end[2]:.6f}'
        self._geometry_dirty = True
        self._status = (
            f'Set start/end from split "{self._dataset_split_for_points}" poses '
            f'[{start_idx}] and [{end_idx}].'
        )

    @staticmethod
    def _read_texture_size(texture_location: int) -> tuple[int, int]:
        prev_texture = gl.glGetIntegerv(gl.GL_TEXTURE_BINDING_2D)
        try:
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture_location)
            width = int(gl.glGetTexLevelParameteriv(gl.GL_TEXTURE_2D, 0, gl.GL_TEXTURE_WIDTH))
            height = int(gl.glGetTexLevelParameteriv(gl.GL_TEXTURE_2D, 0, gl.GL_TEXTURE_HEIGHT))
        finally:
            gl.glBindTexture(gl.GL_TEXTURE_2D, int(prev_texture))
        return width, height

    def _read_scalar_texture(self, texture_location: int, x: int, y: int) -> float:
        prev_framebuffer = gl.glGetIntegerv(gl.GL_FRAMEBUFFER_BINDING)
        prev_read_buffer = gl.glGetIntegerv(gl.GL_READ_BUFFER)
        try:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._picker_framebuffer)
            gl.glFramebufferTexture2D(
                gl.GL_FRAMEBUFFER,
                gl.GL_COLOR_ATTACHMENT0,
                gl.GL_TEXTURE_2D,
                texture_location,
                0,
            )
            status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
            if status != gl.GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError(f'framebuffer incomplete: {status}')
            gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
            buffer = gl.glReadPixels(x, y, 1, 1, gl.GL_RED, gl.GL_FLOAT)
            value = np.frombuffer(buffer, dtype=np.float32, count=1)
            if value.size != 1:
                raise RuntimeError('failed to read texture pixel')
            return float(value[0])
        finally:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, int(prev_framebuffer))
            gl.glReadBuffer(int(prev_read_buffer))

    @staticmethod
    def _unproject_pixel_to_world(view: View, pixel_x: int, pixel_y: int, depth: float) -> np.ndarray:
        pixel = torch.tensor(
            [[pixel_x + 0.5, pixel_y + 0.5]],
            dtype=view.position.dtype,
            device=view.position.device,
        )
        point_world = view.unproject_points(pixel, depth)
        return point_world[0].detach().cpu().numpy().astype(np.float32)

    def _pick_world_point(self, view: View, extra_params: dict[str, Any]) -> tuple[np.ndarray, int, int, float]:
        camera = view.camera
        viewport_w, viewport_h = extra_params['viewportSize']
        if viewport_w <= 0 or viewport_h <= 0:
            raise ValueError('invalid viewport size')

        mouse = imgui.get_mouse_pos()
        mouse_x, mouse_y = mouse.x, mouse.y
        if mouse_x < 0 or mouse_y < 0 or mouse_x >= viewport_w or mouse_y >= viewport_h:
            raise ValueError('mouse is outside the viewer')

        depth_texture = extra_params['modelTextureLocations']['depth']
        alpha_texture = extra_params['modelTextureLocations']['alpha']
        tex_w, tex_h = self._read_texture_size(depth_texture)
        if tex_w <= 0 or tex_h <= 0:
            raise ValueError('depth texture is not initialized')

        sample_x = int(np.clip(mouse_x / float(viewport_w) * tex_w, 0, tex_w - 1))
        sample_y = int(np.clip(mouse_y / float(viewport_h) * tex_h, 0, tex_h - 1))
        pixel_x = int(np.clip(mouse_x / float(viewport_w) * camera.width, 0, camera.width - 1))
        pixel_y = int(np.clip(mouse_y / float(viewport_h) * camera.height, 0, camera.height - 1))

        depth = self._read_scalar_texture(depth_texture, sample_x, sample_y)
        alpha = self._read_scalar_texture(alpha_texture, sample_x, sample_y)
        if not np.isfinite(depth):
            raise ValueError('depth under cursor is invalid')
        if alpha <= 1e-4:
            raise ValueError('clicked background pixel')
        if depth <= float(camera.near_plane) or depth > float(camera.far_plane):
            raise ValueError(f'depth {depth:.4f} is outside [{camera.near_plane:.4f}, {camera.far_plane:.4f}]')

        point_world = self._unproject_pixel_to_world(view, pixel_x, pixel_y, depth)
        return point_world, pixel_x, pixel_y, depth

    def _handle_pending_pick(self, view: View, extra_params: dict[str, Any]):
        if self._pending_pick_target is None:
            if self._gizmo_drag_axis is None:
                self._set_mouse_pick_enabled(False)
            return
        self._set_mouse_pick_enabled(True)
        if imgui.is_mouse_clicked(imgui.MouseButton_.right) and not imgui.get_io().want_capture_mouse:
            target = self._pending_pick_target
            self._pending_pick_target = None
            self._set_mouse_pick_enabled(False)
            self._status = f'Cancelled picking {target}.'
            return
        if not imgui.is_mouse_clicked(imgui.MouseButton_.left) or imgui.get_io().want_capture_mouse:
            return

        target = self._pending_pick_target
        try:
            point_world, pixel_x, pixel_y, depth = self._pick_world_point(view, extra_params)
            self._set_endpoint(target, point_world)
            self._gizmo_target = target
            self._pending_pick_target = None
            self._set_mouse_pick_enabled(False)
            self._status = (
                f'Picked {target} at pixel ({pixel_x}, {pixel_y}), depth={depth:.4f}: '
                f'[{point_world[0]:.4f}, {point_world[1]:.4f}, {point_world[2]:.4f}]'
            )
        except Exception as exc:
            self._status = f'Pick {target} failed: {exc}'

    def _project_world_points(
        self,
        view: View,
        points_world: np.ndarray,
        viewport_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        if points_world.size == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)
        pts = torch.as_tensor(
            points_world.astype(np.float32),
            device=view.position.device,
            dtype=view.position.dtype,
        )
        pixels, _, valid_mask = view.project_points(pts)
        pixels_np = pixels.detach().cpu().numpy().astype(np.float32)
        valid_np = valid_mask.detach().cpu().numpy().astype(bool)
        viewport_w, viewport_h = viewport_size
        pixels_np[:, 0] *= float(viewport_w) / max(float(view.camera.width), 1.0)
        pixels_np[:, 1] *= float(viewport_h) / max(float(view.camera.height), 1.0)
        return pixels_np, valid_np

    @staticmethod
    def _distance_to_segment(point: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> tuple[float, float]:
        segment = seg_b - seg_a
        denom = float(np.dot(segment, segment))
        if denom <= 1e-8:
            return float(np.linalg.norm(point - seg_a)), 0.0
        t = float(np.clip(np.dot(point - seg_a, segment) / denom, 0.0, 1.0))
        closest = seg_a + segment * t
        return float(np.linalg.norm(point - closest)), t

    @staticmethod
    def _mouse_position() -> np.ndarray:
        mouse = imgui.get_mouse_pos()
        return np.array([mouse.x, mouse.y], dtype=np.float32)

    def _pick_gizmo_axis(self, view: View, extra_params: dict[str, Any]) -> str | None:
        point = self._get_endpoint(self._gizmo_target)
        if point is None:
            return None
        viewport_size = tuple(int(v) for v in extra_params['viewportSize'])
        mouse = self._mouse_position()
        best_axis = None
        best_distance = float(self._gizmo_hit_threshold_px)
        for axis_name in ('x', 'y', 'z'):
            tip = point + self._axis_vector(axis_name) * float(self._gizmo_axis_length)
            projected, valid = self._project_world_points(
                view,
                np.stack([point, tip], axis=0).astype(np.float32),
                viewport_size,
            )
            if projected.shape[0] != 2 or not bool(valid.all()):
                continue
            distance, _ = self._distance_to_segment(mouse, projected[0], projected[1])
            if distance <= best_distance:
                best_distance = distance
                best_axis = axis_name
        return best_axis

    def _begin_gizmo_drag(self, view: View, extra_params: dict[str, Any], axis_name: str):
        point = self._get_endpoint(self._gizmo_target)
        if point is None:
            return
        viewport_size = tuple(int(v) for v in extra_params['viewportSize'])
        axis_world = self._axis_vector(axis_name)
        tip = point + axis_world * float(self._gizmo_axis_length)
        projected, valid = self._project_world_points(
            view,
            np.stack([point, tip], axis=0).astype(np.float32),
            viewport_size,
        )
        if projected.shape[0] != 2 or not bool(valid.all()):
            self._status = f'Cannot drag {self._gizmo_target}: gizmo axis is not visible.'
            return
        screen_vec = projected[1] - projected[0]
        pixel_length = float(np.linalg.norm(screen_vec))
        if pixel_length <= 1e-5:
            self._status = f'Cannot drag {self._gizmo_target}: gizmo axis is too small on screen.'
            return
        self._gizmo_drag_axis = axis_name
        self._gizmo_drag_start_mouse = self._mouse_position()
        self._gizmo_drag_origin_point = point.astype(np.float32).copy()
        self._gizmo_drag_world_axis = axis_world.astype(np.float32)
        self._gizmo_drag_screen_dir = (screen_vec / pixel_length).astype(np.float32)
        self._gizmo_drag_pixels_per_unit = pixel_length / max(float(self._gizmo_axis_length), 1e-6)
        self._set_mouse_pick_enabled(True)
        self._status = f'Dragging {self._gizmo_target} along {axis_name.upper()} axis.'

    def _finish_gizmo_drag(self):
        if self._gizmo_drag_axis is None:
            return
        axis_name = self._gizmo_drag_axis
        self._gizmo_drag_axis = None
        self._gizmo_drag_start_mouse = None
        self._gizmo_drag_origin_point = None
        self._gizmo_drag_world_axis = None
        self._gizmo_drag_screen_dir = None
        self._gizmo_drag_pixels_per_unit = 1.0
        self._set_mouse_pick_enabled(False)
        self._status = f'Finished dragging {self._gizmo_target} along {axis_name.upper()} axis.'

    def _handle_gizmo_drag(self):
        if self._gizmo_drag_axis is None:
            return
        self._set_mouse_pick_enabled(True)
        mouse_down = bool(imgui.get_io().mouse_down[0])
        if imgui.is_mouse_released(imgui.MouseButton_.left) or not mouse_down:
            self._finish_gizmo_drag()
            return
        assert self._gizmo_drag_start_mouse is not None
        assert self._gizmo_drag_origin_point is not None
        assert self._gizmo_drag_world_axis is not None
        assert self._gizmo_drag_screen_dir is not None
        mouse_now = self._mouse_position()
        delta_screen = mouse_now - self._gizmo_drag_start_mouse
        delta_world = float(np.dot(delta_screen, self._gizmo_drag_screen_dir)) / max(self._gizmo_drag_pixels_per_unit, 1e-6)
        new_point = self._gizmo_drag_origin_point + self._gizmo_drag_world_axis * delta_world
        self._set_endpoint(self._gizmo_target, new_point.astype(np.float32))
        self._status = (
            f'Dragging {self._gizmo_target} along {self._gizmo_drag_axis.upper()} axis: '
            f'[{new_point[0]:.4f}, {new_point[1]:.4f}, {new_point[2]:.4f}]'
        )

    def _handle_gizmo_interaction(self, view: View, extra_params: dict[str, Any]):
        if self._pending_pick_target is not None:
            return
        if not self._show_gizmo:
            if self._gizmo_drag_axis is not None:
                self._finish_gizmo_drag()
            return
        if self._get_endpoint(self._gizmo_target) is None:
            if self._gizmo_drag_axis is not None:
                self._finish_gizmo_drag()
            return
        if self._gizmo_drag_axis is not None:
            self._handle_gizmo_drag()
            return
        if imgui.get_io().want_capture_mouse or not imgui.is_mouse_clicked(imgui.MouseButton_.left):
            return
        axis_name = self._pick_gizmo_axis(view, extra_params)
        if axis_name is None:
            return
        self._begin_gizmo_drag(view, extra_params, axis_name)

    def _render_path_section(self):
        imgui.text('Path Endpoints')
        imgui.text('World coordinates, format: x, y, z')
        _, self._start_input = imgui.input_text(f'Start##{self.name}', self._start_input)
        _, self._end_input = imgui.input_text(f'End##{self.name}', self._end_input)
        _, self._path_input = imgui.input_text_multiline(
            f'Path (one point per line)##{self.name}',
            self._path_input,
        )
        if imgui.button(f'Apply##{self.name}'):
            self._try_apply_inputs()
        imgui.same_line()
        if imgui.button(f'Clear Path##{self.name}'):
            self._path_input = ''
            self._path_points = None
            self._path_loaded_file_path = None
            self._path_header = {}
            self._path_raw_header_lines = []
            self._path_vertex_count = 0
            self._geometry_dirty = True
            self._status = 'Path cleared.'

        imgui.text('Current camera to endpoint')
        if imgui.button(f'Camera -> Start##{self.name}'):
            self._apply_start_from_current_camera()
        imgui.same_line()
        if imgui.button(f'Camera -> End##{self.name}'):
            self._apply_end_from_current_camera()

        imgui.text('Mouse pick on viewer')
        if imgui.button(f'Pick Start##{self.name}'):
            self._pending_pick_target = 'start'
            self._gizmo_target = 'start'
            self._set_mouse_pick_enabled(True)
            self._status = 'Picking start: left click on the viewer, right click to cancel.'
        imgui.same_line()
        if imgui.button(f'Pick End##{self.name}'):
            self._pending_pick_target = 'end'
            self._gizmo_target = 'end'
            self._set_mouse_pick_enabled(True)
            self._status = 'Picking end: left click on the viewer, right click to cancel.'
        if self._pending_pick_target is not None:
            imgui.same_line()
            if imgui.button(f'Cancel Pick##{self.name}'):
                target = self._pending_pick_target
                self._pending_pick_target = None
                self._set_mouse_pick_enabled(False)
                self._status = f'Cancelled picking {target}.'

        _, self._show_gizmo = imgui.checkbox(f'Show 3D Gizmo##{self.name}', self._show_gizmo)
        gizmo_targets = ['start', 'end']
        gizmo_labels = ['Start', 'End']
        gizmo_target_idx = gizmo_targets.index(self._gizmo_target)
        changed, gizmo_target_idx = imgui.combo(
            f'Gizmo Target##{self.name}',
            gizmo_target_idx,
            gizmo_labels,
        )
        if changed:
            if self._gizmo_drag_axis is not None:
                self._finish_gizmo_drag()
            self._gizmo_target = gizmo_targets[gizmo_target_idx]
        _, self._gizmo_axis_length = imgui.drag_float(
            f'Gizmo Length##{self.name}',
            float(self._gizmo_axis_length),
            0.001,
            0.01,
            10.0,
            '%.3f',
        )

        dataset_poses = CameraState().dataset_poses
        splits = [split for split in GlobalState().shared.splits if split in dataset_poses]
        if splits:
            if self._dataset_split_for_points not in splits:
                self._dataset_split_for_points = splits[0]
            split_idx = splits.index(self._dataset_split_for_points)
            changed, split_idx = imgui.combo(
                f'Dataset Split##{self.name}',
                split_idx,
                splits,
            )
            if changed:
                self._dataset_split_for_points = splits[split_idx]

            pose_count = len(dataset_poses[self._dataset_split_for_points])
            max_pose_idx = max(0, pose_count - 1)
            _, self._dataset_start_idx = imgui.drag_int(
                f'Start Pose Idx##{self.name}',
                self._dataset_start_idx,
                1.0,
                0,
                max_pose_idx,
            )
            _, self._dataset_end_idx = imgui.drag_int(
                f'End Pose Idx##{self.name}',
                self._dataset_end_idx,
                1.0,
                0,
                max_pose_idx,
            )
            if imgui.button(f'Use Dataset Poses##{self.name}'):
                self._apply_points_from_dataset_poses()

        _, self._path_file_input = imgui.input_text(
            f'Path File (.txt/.csv/.npy)##{self.name}',
            self._path_file_input,
        )
        if imgui.button(f'Browse##PathFile{self.name}'):
            selected = select_file_dialog(
                title='Select Path File',
                initial_path=self._path_browse_initial_path(),
                filetypes=(
                    ('Path Files', '*.txt *.csv *.npy'),
                    ('NumPy', '*.npy'),
                    ('Text/CSV', '*.txt *.csv'),
                    ('All Files', '*'),
                ),
            )
            if selected is not None:
                self._path_file_input = selected
        if imgui.button(f'Load Path File##{self.name}'):
            try:
                self._load_path_file(self._path_file_input)
                self._status = f'Loaded path file: {self._path_file_input}'
            except ValueError as exc:
                self._status = f'File error: {exc}'

        _, self._path_export_file_input = imgui.input_text(
            f'Export Path (.txt/.csv/.npy)##{self.name}',
            self._path_export_file_input,
        )
        if imgui.button(f'Browse Export Directory##{self.name}'):
            selected = select_directory_dialog(
                title='Select Path Export Directory',
                initial_path=str(self._default_path_export_path().parent),
            )
            if selected is not None:
                self._path_export_file_input = str(Path(selected) / 'planned_path.txt')
        imgui.same_line()
        if imgui.button(f'Export Path##{self.name}'):
            try:
                self._export_current_path()
            except Exception as exc:
                self._status = f'Export error: {exc}'


    def _run_standalone_planner(self) -> None:
        if not self._try_apply_endpoint_inputs():
            return
        if self._start is None or self._end is None:
            self._status = 'Standalone planner requires both start and end.'
            return
        checkpoint = Path(self._planner_checkpoint_file).expanduser()
        if not checkpoint.is_file():
            self._status = f'Gaussian checkpoint not found: {checkpoint}'
            return

        try:
            radius = self._planning_robot_radius()
            result = run_gspar_direct_path(
                checkpoint,
                self._start,
                self._end,
                GSPARDirectPlanOptions(robot_radius=radius),
            )
            if not result.success:
                reason = str(result.debug_info.get('failure_reason', 'no feasible path'))
                raise RuntimeError(reason)
            method_detail = 'RRT + Gaussian avoidance'

            header = dict(result.header)
            header['standalone_method'] = GSPAR_DIRECT_LABEL
            header['physical_robot_radius_m'] = f'{self._robot_radius_physical_m:.6f}'
            header['dataset_radius_scale'] = f'{self._robot_radius_scale:.9g}'
            raw_header_lines = ['NERFICG standalone path'] + [
                f'{key}: {value}' for key, value in header.items()
            ] + ['columns: x y z']
            self._set_current_path(
                result.path_world,
                header=header,
                raw_header_lines=raw_header_lines,
                sync_endpoints=True,
            )
            self._standalone_last_result = result
            self._status = (
                f'{GSPAR_DIRECT_LABEL} complete ({method_detail}); '
                f'path points={result.path_world.shape[0]}.'
            )
        except Exception as exc:
            self._standalone_last_result = None
            self._status = f'{GSPAR_DIRECT_LABEL} failed: {exc}'



    def _planning_robot_radius(self) -> float:
        physical = float(self._robot_radius_physical_m)
        scale = float(self._robot_radius_scale)
        if not np.isfinite(physical) or physical <= 0.0:
            raise ValueError('physical robot radius must be finite and positive')
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError('dataset radius scale must be finite and positive')
        return physical * scale

    def _compact_endpoints(self) -> tuple[np.ndarray, np.ndarray] | None:
        if not self._try_apply_endpoint_inputs():
            return None
        if self._start is None or self._end is None:
            self._status = 'Planning requires both start and end.'
            return None
        return self._start.copy(), self._end.copy()

    def _apply_compact_global_result(self, result, title: str) -> None:
        self._planner_last_start_idx = np.asarray(result.start_idx, dtype=np.int32)
        self._planner_last_goal_idx = np.asarray(result.goal_idx, dtype=np.int32)
        self._planner_last_start_world = np.asarray(result.start_world, dtype=np.float32)
        self._planner_last_goal_world = np.asarray(result.goal_world, dtype=np.float32)
        self._planner_last_start_was_snapped = bool(result.start_was_snapped)
        self._planner_last_goal_was_snapped = bool(result.goal_was_snapped)
        header = dict(result.header)
        header['physical_robot_radius_m'] = f'{self._robot_radius_physical_m:.6f}'
        header['dataset_radius_scale'] = f'{self._robot_radius_scale:.9g}'
        raw_header_lines = [title] + [f'{key}: {value}' for key, value in header.items()]
        self._set_current_path(
            result.path_world,
            header=header,
            raw_header_lines=raw_header_lines,
            sync_endpoints=True,
        )

    def _run_compact_gspar_global(self) -> None:
        endpoints = self._compact_endpoints()
        if endpoints is None:
            return
        try:
            payload, resolved_path = load_voxel_payload(self._planner_occupancy_file_input)
            voxel_size = float(np.asarray(payload['voxel_size']).reshape(-1)[0])
            radius = self._planning_robot_radius()
            options = GlobalPlanOptions(
                method=GLOBAL_JPS3D,
                connectivity=26,
                snap_search_radius=8,
                safety_clearance_voxels=radius / max(voxel_size, 1.0e-12),
                heuristic_weight=1.0,
                max_planning_time_sec=10.0,
                treat_unknown_as_occupied=False,
            )
            start, end = endpoints
            result = run_global_path(
                payload,
                start,
                end,
                options,
                map_label=str(resolved_path or '<memory>'),
            )
            self._apply_compact_global_result(result, 'GSPAR-Global JPS-3D path')
            self._status = f'gspar-global global planning complete: {result.path_world.shape[0]} points.'
        except Exception as exc:
            self._status = f'gspar-global global planning failed: {exc}'

    def _run_compact_gspar_local(self) -> None:
        try:
            coarse = self._current_coarse_path()
            payload, resolved_path = load_voxel_payload(self._planner_occupancy_file_input)
            voxel_size = float(np.asarray(payload['voxel_size']).reshape(-1)[0])
            radius = self._planning_robot_radius()
            options = EgoPlanOptions(
                safety_clearance_voxels=radius / max(voxel_size, 1.0e-12),
                safety_clearance_m=radius,
                inherit_input_clearance=False,
                control_point_spacing_m=0.5,
                output_spacing_m=0.05,
                collision_sample_spacing_m=0.05,
                treat_unknown_as_occupied=False,
            )
            result = run_ego_path(
                payload,
                coarse,
                options,
                map_label=str(resolved_path or '<memory>'),
            )
            if not result.success:
                raise RuntimeError(result.message)
            header = {
                **dict(coarse.header),
                **dict(result.header),
                'physical_robot_radius_m': f'{self._robot_radius_physical_m:.6f}',
                'dataset_radius_scale': f'{self._robot_radius_scale:.9g}',
            }
            self._set_current_path(
                result.path_world,
                header=header,
                raw_header_lines=list(coarse.raw_header_lines),
                sync_endpoints=True,
            )
            self._status = f'gspar-global EGO local planning complete: {result.path_world.shape[0]} points.'
        except Exception as exc:
            self._status = f'gspar-global local planning failed: {exc}'

    def _prepare_compact_esdf(self):
        if self._planner_esdf_file_input.strip() == '':
            self._fill_esdf_inputs_from_occupancy_map()
        source = self._planner_esdf_file_input.strip()
        metadata = self._planner_esdf_grid_file_input.strip()
        if source == '':
            self._status = 'ESDF input is required.'
            return None
        cache_key = (source, metadata)
        if self._planner_esdf_graph is None or self._planner_esdf_cache_key != cache_key:
            try:
                self._planner_esdf_graph = load_esdf_graph(
                    source,
                    metadata or None,
                    use_occupancy_guard=True,
                    unknown_policy='free',
                )
                self._planner_esdf_cache_key = cache_key
                self._planner_esdf_loaded_file_path = str(Path(source).expanduser().resolve())
            except Exception as exc:
                self._planner_esdf_graph = None
                self._status = f'ESDF load failed: {exc}'
                return None
        return self._planner_esdf_graph

    def _run_compact_nvblox_global(self) -> None:
        endpoints = self._compact_endpoints()
        if endpoints is None:
            return
        graph = self._prepare_compact_esdf()
        if graph is None:
            return
        try:
            radius = self._planning_robot_radius()
            options = EsdfGlobalPlanOptions(
                method=GLOBAL_ESDF_WEIGHTED_ASTAR,
                connectivity=26,
                snap_search_radius=8,
                safety_clearance_m=radius,
                clearance_cost_weight=0.0,
                heuristic_weight=1.0,
                max_planning_time_sec=10.0,
            )
            start, end = endpoints
            result = run_esdf_global_path(
                graph,
                start,
                end,
                options,
                map_label=self._planner_esdf_loaded_file_path or self._planner_esdf_file_input,
            )
            self._apply_compact_global_result(result, 'nvblox-ESDF A* path')
            self._status = f'nvblox-esdf global planning complete: {result.path_world.shape[0]} points.'
        except Exception as exc:
            self._status = f'nvblox-esdf global planning failed: {exc}'

    def _run_compact_nvblox_local(self) -> None:
        graph = self._prepare_compact_esdf()
        if graph is None:
            return
        try:
            coarse = self._current_coarse_path()
            radius = self._planning_robot_radius()
            result = run_stomp_esdf_path(
                graph,
                coarse,
                StompEsdfPlanOptions(
                    safety_clearance_m=radius,
                    control_point_spacing_m=0.25,
                    output_spacing_m=0.05,
                    collision_sample_spacing_m=0.05,
                    random_seed=7,
                ),
                map_label=self._planner_esdf_loaded_file_path or self._planner_esdf_file_input,
            )
            if not result.success:
                raise RuntimeError(result.message)
            header = {
                **dict(coarse.header),
                **dict(result.header),
                'physical_robot_radius_m': f'{self._robot_radius_physical_m:.6f}',
                'dataset_radius_scale': f'{self._robot_radius_scale:.9g}',
            }
            self._set_current_path(
                result.path_world,
                header=header,
                raw_header_lines=list(coarse.raw_header_lines),
                sync_endpoints=True,
            )
            self._status = f'nvblox-esdf STOMP local planning complete: {result.path_world.shape[0]} points.'
        except Exception as exc:
            self._status = f'nvblox-esdf local planning failed: {exc}'


    def _run_compact_splatnav(self) -> None:
        endpoints = self._compact_endpoints()
        if endpoints is None:
            return
        checkpoint = Path(self._planner_checkpoint_file).expanduser()
        root = Path(self._splatnav_root_input).expanduser()
        if not checkpoint.is_file():
            self._status = f'SplatNav checkpoint not found: {checkpoint}'
            return
        if not (root / 'splatplan' / 'splatplan.py').is_file():
            self._status = f'SplatNav repository not found: {root}'
            return
        conda = shutil.which('conda')
        if conda is None:
            self._status = 'SplatNav planning failed: conda executable not found.'
            return
        try:
            radius = self._planning_robot_radius()
            start, end = endpoints
            project_root = Path(__file__).resolve().parents[2]
            with tempfile.TemporaryDirectory(prefix='gspar_splatnav_') as temp_dir:
                output = Path(temp_dir) / 'path.npy'
                command = [
                    conda, 'run', '-n', 'splatnav', 'python', '-m',
                    'GSPAR.gui.splatnav_worker',
                    '--checkpoint', str(checkpoint.resolve()),
                    '--splatnav-root', str(root.resolve()),
                    '--start', ','.join(map(str, start.tolist())),
                    '--goal', ','.join(map(str, end.tolist())),
                    '--robot-radius', str(radius),
                    '--output', str(output),
                ]
                metadata = self._planner_occupancy_file_input.strip()
                if metadata and Path(metadata).expanduser().is_file():
                    command.extend(['--map-metadata', str(Path(metadata).expanduser().resolve())])
                env = os.environ.copy()
                python_paths = [str(project_root), str(project_root / 'src')]
                if env.get('PYTHONPATH'):
                    python_paths.append(env['PYTHONPATH'])
                env['PYTHONPATH'] = os.pathsep.join(python_paths)
                self._status = 'SplatNav is running in conda env "splatnav"...'
                completed = subprocess.run(
                    command,
                    cwd=project_root,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if completed.returncode != 0:
                    detail = (completed.stderr or completed.stdout).strip()
                    raise RuntimeError(detail or f'worker exited with code {completed.returncode}')
                path_world = np.asarray(np.load(output), dtype=np.float32)
            header = {
                'method': 'SplatNav',
                'planner_map_type': 'Gaussian',
                'physical_robot_radius_m': f'{self._robot_radius_physical_m:.6f}',
                'dataset_radius_scale': f'{self._robot_radius_scale:.9g}',
                'robot_radius_m': f'{radius:.6f}',
                'global_planner': 'dijkstra3d.binary_dijkstra',
                'local_planner': 'polytope corridor + Bezier spline',
            }
            self._set_current_path(path_world, header=header, sync_endpoints=True)
            self._status = f'SplatNav global + local planning complete: {path_world.shape[0]} points.'
        except Exception as exc:
            self._status = f'SplatNav planning failed: {exc}'

    def _render_compact_checkpoint_input(self) -> None:
        _, self._planner_checkpoint_file = imgui.input_text(
            f'Checkpoint##Compact{self.name}',
            self._planner_checkpoint_file,
        )
        if imgui.button(f'Browse##CompactCheckpoint{self.name}'):
            selected = select_file_dialog(
                title='Select NerfICG Checkpoint',
                initial_path=self._planner_checkpoint_file,
                filetypes=(('PyTorch Checkpoint', '*.pt'), ('All Files', '*')),
            )
            if selected is not None:
                self._planner_checkpoint_file = selected

    def _render_compact_occupancy_input(self) -> None:
        _, self._planner_occupancy_file_input = imgui.input_text(
            f'Voxel Map##Compact{self.name}',
            self._planner_occupancy_file_input,
        )
        if imgui.button(f'Browse##CompactVoxel{self.name}'):
            selected = select_file_dialog(
                title='Select GSPAR Voxel Map',
                initial_path=self._global_map_browse_initial_path(),
                filetypes=(('Voxel Map', '*.npz'), ('All Files', '*')),
            )
            if selected is not None:
                self._planner_occupancy_file_input = selected

    def _render_compact_esdf_inputs(self) -> None:
        if self._planner_esdf_file_input.strip() == '':
            self._fill_esdf_inputs_from_occupancy_map()
        _, self._planner_esdf_file_input = imgui.input_text(
            f'ESDF Map##Compact{self.name}',
            self._planner_esdf_file_input,
        )
        if imgui.button(f'Browse##CompactEsdf{self.name}'):
            selected = select_file_dialog(
                title='Select nvblox ESDF (raw or rendered depth)',
                initial_path=self._esdf_browse_initial_path(),
                filetypes=(('ESDF PLY', '*.ply'), ('All Files', '*')),
            )
            if selected is not None:
                self._planner_esdf_file_input = selected
                self._planner_esdf_graph = None
        _, self._planner_esdf_grid_file_input = imgui.input_text(
            f'Grid Metadata##Compact{self.name}',
            self._planner_esdf_grid_file_input,
        )
        if imgui.button(f'Browse##CompactEsdfGrid{self.name}'):
            selected = select_file_dialog(
                title='Select ESDF Grid Metadata',
                initial_path=self._planner_esdf_grid_file_input or self._global_map_browse_initial_path(),
                filetypes=(('Voxel Metadata', '*.npz *.json'), ('All Files', '*')),
            )
            if selected is not None:
                self._planner_esdf_grid_file_input = selected
                self._planner_esdf_graph = None
        imgui.text_wrapped('The same inputs accept nvblox maps built from sensor depth or rendered 3DGS depth.')

    def _render_compact_planning_section(self) -> None:
        imgui.text('Path Planning')
        method_idx = self._compact_planning_methods.index(self._compact_planning_method)
        changed, method_idx = imgui.combo(
            f'Method##Compact{self.name}',
            method_idx,
            self._compact_planning_methods,
        )
        if changed:
            self._compact_planning_method = self._compact_planning_methods[method_idx]
        method = self._compact_planning_method
        _, self._robot_radius_physical_m = imgui.drag_float(
            f'Physical Robot Radius (m)##{self.name}',
            float(self._robot_radius_physical_m),
            0.01,
            0.001,
            10.0,
            '%.3f',
        )
        _, self._robot_radius_scale = imgui.drag_float(
            f'Dataset Radius Scale##{self.name}',
            float(self._robot_radius_scale),
            0.01,
            0.001,
            10.0,
            '%.6f',
        )
        if imgui.button(f'Redetect Dataset Scale##{self.name}'):
            self._robot_radius_scale = float(
                robot_radius_scale_for_dataset(self._dataset_identifier, default=1.0)
            )
        imgui.text_wrapped(f'Dataset: {self._dataset_identifier}')
        try:
            planning_radius = self._planning_robot_radius()
            imgui.text_wrapped(
                f'Planner radius: {planning_radius:.6f} scene units '
                f'({self._robot_radius_physical_m:.3f} m x {self._robot_radius_scale:.6f}).'
            )
        except ValueError as exc:
            imgui.text_wrapped(f'Invalid robot radius: {exc}')

        if method == COMPACT_GSPAR_DIRECT_METHOD:
            self._render_compact_checkpoint_input()
            if imgui.button(f'Run GSPAR-Direct##{self.name}'):
                self._run_standalone_planner()
        elif method == COMPACT_GSPAR_GLOBAL_METHOD:
            self._render_compact_occupancy_input()
            if imgui.button(f'Run Global (JPS-3D)##GSPAR{self.name}'):
                self._run_compact_gspar_global()
            imgui.same_line()
            if imgui.button(f'Run Local (EGO)##GSPAR{self.name}'):
                self._run_compact_gspar_local()
        elif method == COMPACT_NVBLOX_ESDF_METHOD:
            self._render_compact_esdf_inputs()
            if imgui.button(f'Run Global (A*)##NVBLOX{self.name}'):
                self._run_compact_nvblox_global()
            imgui.same_line()
            if imgui.button(f'Run Local (STOMP)##NVBLOX{self.name}'):
                self._run_compact_nvblox_local()
        elif method == COMPACT_SPLATNAV_METHOD:
            self._render_compact_checkpoint_input()
            _, self._splatnav_root_input = imgui.input_text(
                f'SplatNav Root##{self.name}',
                self._splatnav_root_input,
            )
            if imgui.button(f'Browse##SplatNavRoot{self.name}'):
                selected = select_directory_dialog(
                    title='Select SplatNav Repository',
                    initial_path=self._splatnav_root_input,
                )
                if selected is not None:
                    self._splatnav_root_input = selected
            if imgui.button(f'Run SplatNav (Global + Local)##{self.name}'):
                self._run_compact_splatnav()
        imgui.text_wrapped(f'Planner status: {self._status}')



    def _render_display_section(self):
        imgui.text('Path Display')
        _, self._show_points = imgui.checkbox(f'Show Start/End##{self.name}', self._show_points)
        _, self._show_path = imgui.checkbox(f'Show Path##{self.name}', self._show_path)
        _, self._point_size = imgui.drag_float(
            f'Point Size##{self.name}',
            self._point_size,
            0.001,
            0.001,
            0.1,
            '%.4f',
        )
        _, self._line_width = imgui.drag_float(
            f'Line Width##{self.name}',
            self._line_width,
            0.0005,
            0.0005,
            0.05,
            '%.4f',
        )
        changed, c = imgui.color_edit4(f'Start Color##{self.name}', self._start_color.tolist())
        if changed:
            self._start_color = np.array(c, dtype=np.float32)
        changed, c = imgui.color_edit4(f'End Color##{self.name}', self._end_color.tolist())
        if changed:
            self._end_color = np.array(c, dtype=np.float32)
        changed, c = imgui.color_edit4(f'Path Color##{self.name}', self._path_color.tolist())
        if changed:
            self._path_color = np.array(c, dtype=np.float32)

    def _render_status_section(self):
        imgui.text(f'Path vertex count: {self._path_vertex_count}')
        imgui.text(f'Has start: {self._start is not None}')
        imgui.text(f'Has end: {self._end is not None}')
        imgui.text(f'Pending pick target: {self._pending_pick_target or "(none)"}')
        imgui.text(f'Gizmo target: {self._gizmo_target}')
        imgui.text(f'Gizmo drag axis: {self._gizmo_drag_axis or "(none)"}')
        imgui.text_wrapped(f'Selected planning route: {self._compact_planning_method}')
        if self._planner_last_start_idx is not None:
            imgui.text_wrapped(
                f'Planner start snap: idx={self._planner_last_start_idx.tolist()}, '
                f'snapped={self._planner_last_start_was_snapped}'
            )
        if self._planner_last_goal_idx is not None:
            imgui.text_wrapped(
                f'Planner end snap: idx={self._planner_last_goal_idx.tolist()}, '
                f'snapped={self._planner_last_goal_was_snapped}'
            )
        imgui.text_wrapped(self._status)

    def render_options(self):
        self._render_path_section()
        imgui.spacing()
        imgui.separator()
        self._render_compact_planning_section()
        imgui.spacing()
        imgui.separator()
        self._render_display_section()
        imgui.spacing()
        imgui.separator()
        self._render_status_section()

    @staticmethod
    def _current_depth_write_mask() -> bool:
        return bool(np.asarray(gl.glGetBooleanv(gl.GL_DEPTH_WRITEMASK)).reshape(-1)[0])

    def _draw_point(
        self,
        view: View,
        point_vbo: VBO,
        color: np.ndarray,
        extra_params: dict[str, Any],
        *,
        point_size: float | None = None,
    ):
        camera = view.camera
        view_matrix, projection = get_view_projection(view)
        scene_occlusion_enabled = self._enable_scene_occlusion and not bool(extra_params.get('sceneHidden', False))
        uniform_values = {
            'view': view_matrix,
            'projection': projection,
            'pointSize': float(self._point_size if point_size is None else point_size),
            'viewportSize': np.array(extra_params['viewportSize'], dtype='f4'),
            'near': camera.near_plane,
            'far': camera.far_plane,
            'color': color,
            'sceneOcclusionEnabled': int(scene_occlusion_enabled),
            'occlusionAlphaThreshold': float(self._occlusion_alpha_threshold),
        }
        overlay_depth_occlusion_enabled = bool(extra_params.get(OVERLAY_DEPTH_OCCLUDER_KEY, False))
        prev_depth_test = gl.glIsEnabled(gl.GL_DEPTH_TEST)
        prev_depth_mask = self._current_depth_write_mask()
        prev_depth_func = int(gl.glGetIntegerv(gl.GL_DEPTH_FUNC))
        prev_cull_face = gl.glIsEnabled(gl.GL_CULL_FACE)
        prev_blend = gl.glIsEnabled(gl.GL_BLEND)
        try:
            if overlay_depth_occlusion_enabled:
                gl.glEnable(gl.GL_DEPTH_TEST)
                gl.glDepthFunc(gl.GL_LEQUAL)
                gl.glDepthMask(gl.GL_FALSE)
            else:
                gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glDisable(gl.GL_CULL_FACE)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            with (
                bind_shader_program(self._point_shader_program),
                bind_textures(extra_params['modelTextureLocations']['depth'], extra_params['modelTextureLocations']['alpha']),
                bind_vertex_buffer(self._point_vao_location, point_vbo),
                bind_vertex_attributes(3, gl.GL_FLOAT, 0),
            ):
                fill_uniforms(self._uniform_locations['point'], uniform_values)
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        finally:
            gl.glDepthFunc(prev_depth_func)
            gl.glDepthMask(prev_depth_mask)
            restore_gl_bool_state(prev_depth_test, gl.GL_DEPTH_TEST)
            restore_gl_bool_state(prev_cull_face, gl.GL_CULL_FACE)
            restore_gl_bool_state(prev_blend, gl.GL_BLEND)

    def _draw_lines(
        self,
        view: View,
        path_vbo: VBO,
        vertex_count: int,
        color: np.ndarray,
        extra_params: dict[str, Any],
        *,
        line_width: float | None = None,
    ):
        camera = view.camera
        view_matrix, projection = get_view_projection(view)
        scene_occlusion_enabled = self._enable_scene_occlusion and not bool(extra_params.get('sceneHidden', False))
        uniform_values = {
            'view': view_matrix,
            'projection': projection,
            'lineWidth': float(self._line_width if line_width is None else line_width),
            'viewportSize': np.array(extra_params['viewportSize'], dtype='f4'),
            'near': camera.near_plane,
            'far': camera.far_plane,
            'color': color,
            'highlightColor': color,
            'sceneOcclusionEnabled': int(scene_occlusion_enabled),
            'occlusionAlphaThreshold': float(self._occlusion_alpha_threshold),
        }
        overlay_depth_occlusion_enabled = bool(extra_params.get(OVERLAY_DEPTH_OCCLUDER_KEY, False))
        prev_depth_test = gl.glIsEnabled(gl.GL_DEPTH_TEST)
        prev_depth_mask = self._current_depth_write_mask()
        prev_depth_func = int(gl.glGetIntegerv(gl.GL_DEPTH_FUNC))
        prev_cull_face = gl.glIsEnabled(gl.GL_CULL_FACE)
        prev_blend = gl.glIsEnabled(gl.GL_BLEND)
        try:
            if overlay_depth_occlusion_enabled:
                gl.glEnable(gl.GL_DEPTH_TEST)
                gl.glDepthFunc(gl.GL_LEQUAL)
                gl.glDepthMask(gl.GL_FALSE)
            else:
                gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glDisable(gl.GL_CULL_FACE)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            with (
                bind_shader_program(self._line_shader_program),
                bind_textures(extra_params['modelTextureLocations']['depth'], extra_params['modelTextureLocations']['alpha']),
                bind_vertex_buffer(self._line_vao_location, path_vbo),
                bind_vertex_attributes(3, gl.GL_FLOAT, 0),
            ):
                fill_uniforms(self._uniform_locations['line'], uniform_values)
                gl.glDrawArrays(gl.GL_LINES, 0, vertex_count)
        finally:
            gl.glDepthFunc(prev_depth_func)
            gl.glDepthMask(prev_depth_mask)
            restore_gl_bool_state(prev_depth_test, gl.GL_DEPTH_TEST)
            restore_gl_bool_state(prev_cull_face, gl.GL_CULL_FACE)
            restore_gl_bool_state(prev_blend, gl.GL_BLEND)

    def _draw_path(self, view: View, path_vbo: VBO, extra_params: dict[str, Any]):
        self._draw_lines(view, path_vbo, self._path_vertex_count, self._path_color, extra_params)




    def _draw_gizmo(self, view: View, extra_params: dict[str, Any]):
        if not self._show_gizmo or self._get_endpoint(self._gizmo_target) is None:
            return
        for axis_name in ('x', 'y', 'z'):
            line_vbo = self._gizmo_line_vbos[axis_name]
            handle_vbo = self._gizmo_handle_vbos[axis_name]
            vertex_count = self._gizmo_line_vertex_counts[axis_name]
            if line_vbo is None or handle_vbo is None or vertex_count <= 0:
                continue
            color = self._gizmo_colors[axis_name]
            if self._gizmo_drag_axis == axis_name:
                color = self._highlight_color(color, self._gizmo_highlight_scale)
            self._draw_lines(
                view,
                line_vbo,
                vertex_count,
                color,
                extra_params,
                line_width=self._gizmo_line_width,
            )
            self._draw_point(
                view,
                handle_vbo,
                color,
                extra_params,
                point_size=self._gizmo_handle_point_size,
            )

    def render(self, view: View, extra_params: dict[str, Any]):
        if not isinstance(view.camera, PerspectiveCamera):
            return
        self._sync_current_occupancy_file(extra_params)
        self._handle_pending_pick(view, extra_params)
        self._handle_gizmo_interaction(view, extra_params)
        self._ensure_geometry()
        self._draw_gizmo(view, extra_params)
        if (
            self._show_path
            and self._path_vertex_count > 0
            and self._path_vbo is not None
        ):
            self._draw_path(view, self._path_vbo, extra_params)
        # Draw endpoints last so path lines cannot cover their markers.
        if self._show_points:
            if self._start is not None and self._start_vbo is not None:
                self._draw_point(view, self._start_vbo, self._start_color, extra_params)
            if self._end is not None and self._end_vbo is not None:
                self._draw_point(view, self._end_vbo, self._end_color, extra_params)
