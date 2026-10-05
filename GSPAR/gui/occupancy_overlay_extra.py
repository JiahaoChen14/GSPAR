"""Voxel occupancy overlay module."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import OpenGL.GL as gl
from imgui_bundle import imgui
from OpenGL.arrays.vbo import VBO
from OpenGL.GL.shaders import compileProgram, compileShader

from Cameras.Perspective import PerspectiveCamera
from Datasets.utils import View
from ICGui.ModelViewer.Overlays import BaseOverlay
from ICGui.ModelViewer.Shaders.ContextManagers import bind_textures

from GSPAR.gui.overlay_common import (
    OVERLAY_DEPTH_OCCLUDER_KEY,
    OVERLAY_OCCUPANCY_FILE_KEY,
    get_view_projection,
    restore_gl_bool_state,
    select_file_dialog,
    update_vbo,
)
from GSPAR.mapping.map_frame_transform import apply_transform, validate_rigid_transform
from GSPAR.gui.esdf_slice import (
    AXIS_NAMES,
    EsdfSlice,
    build_esdf_slice_from_graph,
    downsample_esdf_slice,
    infer_esdf_path_from_voxel_map,
    voxel_map_supports_esdf_slice,
)
from GSPAR.mapping.voxel_map_cli import resolve_existing_path
from GSPAR.path.global_path.esdf_astar_world_path import UNKNOWN_POLICY_OCCUPIED, EsdfGridGraph, load_esdf_grid


class OccupancyOverlayExtra(BaseOverlay):
    """Displays occupied voxels with solid cube rendering."""

    def __init__(
        self,
        occupancy_file: str = "",
        extra_occupancy_file: str = "",
        max_points: int = 200000,
    ):
        super().__init__()
        self._cube_shader_program = self._build_cube_shader()
        self._cube_vao_location = gl.glGenVertexArrays(1)

        self._cube_view_loc = gl.glGetUniformLocation(self._cube_shader_program, "view")
        self._cube_projection_loc = gl.glGetUniformLocation(self._cube_shader_program, "projection")
        self._cube_viewport_size_loc = gl.glGetUniformLocation(self._cube_shader_program, "viewportSize")
        self._cube_near_loc = gl.glGetUniformLocation(self._cube_shader_program, "near")
        self._cube_color_loc = gl.glGetUniformLocation(self._cube_shader_program, "color")
        self._cube_use_instance_color_loc = gl.glGetUniformLocation(self._cube_shader_program, "useInstanceColor")
        self._cube_overlap_color_loc = gl.glGetUniformLocation(
            self._cube_shader_program, "overlapColor"
        )
        self._cube_use_overlap_flags_loc = gl.glGetUniformLocation(
            self._cube_shader_program, "useOverlapFlags"
        )
        self._cube_size_loc = gl.glGetUniformLocation(self._cube_shader_program, "cubeSize")
        self._cube_map_to_training_loc = gl.glGetUniformLocation(
            self._cube_shader_program, "mapToTrainingWorld"
        )
        self._cube_depth_sampler_loc = gl.glGetUniformLocation(self._cube_shader_program, "depthSampler")
        self._cube_alpha_sampler_loc = gl.glGetUniformLocation(self._cube_shader_program, "alphaSampler")
        self._cube_occlusion_enabled_loc = gl.glGetUniformLocation(self._cube_shader_program, "occlusionEnabled")
        self._cube_occlusion_alpha_threshold_loc = gl.glGetUniformLocation(
            self._cube_shader_program, "occlusionAlphaThreshold"
        )

        self._show_occupied = True
        self._voxel_size_world = 0.1
        self._voxel_size_scale = 1.0
        self._enable_scene_occlusion = True
        self._occlusion_alpha_threshold = 0.5
        self._nearby_only = False
        self._nearby_radius = 0.3
        self._occupied_color = np.array([0.95, 0.28, 0.18, 1.0], dtype=np.float32)
        self._extra_occupied_color = np.array([0.10, 0.62, 1.00, 0.45], dtype=np.float32)
        self._overlap_color = np.array([1.00, 0.85, 0.05, 1.0], dtype=np.float32)
        self._max_points = max(1, int(max_points))
        self._status = "Set occupancy file."
        self._extra_status = "Set extra occupancy file."
        self._occupancy_file_input = occupancy_file
        self._extra_occupancy_file_input = extra_occupancy_file
        self._grid_origin_world: np.ndarray | None = None
        self._extra_grid_origin_world: np.ndarray | None = None
        self._grid_voxel_dims: np.ndarray | None = None
        self._extra_grid_voxel_dims: np.ndarray | None = None
        self._map_from_training_world = np.eye(4, dtype=np.float32)
        self._training_world_from_map = np.eye(4, dtype=np.float32)
        self._extra_map_from_training_world = np.eye(4, dtype=np.float32)
        self._extra_training_world_from_map = np.eye(4, dtype=np.float32)

        self._occupied_points_world: np.ndarray | None = None
        self._extra_occupied_points_world: np.ndarray | None = None
        self._occupied_voxel_idx: np.ndarray | None = None
        self._extra_occupied_voxel_idx: np.ndarray | None = None
        self._occupied_center_vbo: VBO | None = None
        self._occupied_near_vbo: VBO | None = None
        self._occupied_overlap_vbo: VBO | None = None
        self._extra_occupied_near_vbo: VBO | None = None
        self._extra_occupied_overlap_vbo: VBO | None = None
        self._probe_occupied_vbo: VBO | None = None
        self._probe_extra_occupied_vbo: VBO | None = None
        self._cached_occ_points = np.zeros((0, 3), dtype=np.float32)
        self._occupied_overlap_flags = np.zeros(0, dtype=np.uint8)
        self._cached_occ_overlap_flags = np.zeros(0, dtype=np.uint8)
        self._extra_cached_occ_points = np.zeros((0, 3), dtype=np.float32)
        self._extra_occupied_overlap_flags = np.zeros(0, dtype=np.uint8)
        self._extra_cached_occ_overlap_flags = np.zeros(0, dtype=np.uint8)
        self._filter_cache_key: tuple | None = None
        self._extra_filter_cache_key: tuple | None = None
        self._loaded_file_path: str | None = None
        self._extra_loaded_file_path: str | None = None
        self._cube_vertex_vbo: VBO = VBO(self._unit_cube_vertices(), gl.GL_STATIC_DRAW)
        self._cube_normal_vbo: VBO = VBO(self._unit_cube_normals(), gl.GL_STATIC_DRAW)
        self._num_occupied = 0
        self._num_overlap = 0
        self._extra_num_occupied = 0
        self._overlap_status = "Load two aligned voxel maps to detect overlap."
        self._extra_voxel_size_world = self._voxel_size_world
        self._geometry_dirty = True
        self._show_extra_occupied = True
        self._highlight_overlap = True
        self._probe_voxel_enabled = False
        self._probe_coord_input = "0.0, 0.0, 0.0"
        self._probe_status = "Enable coordinate voxel check to compare the two maps."
        self._probe_occ_center = np.zeros((0, 3), dtype=np.float32)
        self._probe_extra_center = np.zeros((0, 3), dtype=np.float32)
        self._probe_geometry_dirty = True
        self._probe_cache_key: tuple | None = None
        self._probe_align_epsilon = 1e-5
        self._show_esdf_slice = False
        self._esdf_slice_axis_idx = 2
        self._esdf_slice_coord = 0.0
        self._esdf_slice_coord_initialized = False
        self._esdf_slice_file_input = ""
        self._esdf_slice_auto_file = True
        self._esdf_slice_show_observed = True
        self._esdf_slice_show_unknown = True
        self._esdf_slice_show_occupied = True
        self._esdf_slice_max_distance_m = 1.0
        self._esdf_slice_max_voxels = 500000
        self._esdf_slice_applied_auto_file = self._esdf_slice_auto_file
        self._esdf_slice_applied_file_input = self._esdf_slice_file_input
        self._esdf_slice_applied_show_observed = self._esdf_slice_show_observed
        self._esdf_slice_applied_show_unknown = self._esdf_slice_show_unknown
        self._esdf_slice_applied_show_occupied = self._esdf_slice_show_occupied
        self._esdf_slice_applied_max_distance_m = self._esdf_slice_max_distance_m
        self._esdf_slice_graph: EsdfGridGraph | None = None
        self._esdf_slice_graph_key: tuple[str, str] | None = None
        self._esdf_slice_graph_future: Future[EsdfGridGraph] | None = None
        self._esdf_slice_graph_future_key: tuple[str, str] | None = None
        self._esdf_slice_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="esdf-slice")
        self._esdf_slice_data: EsdfSlice | None = None
        self._esdf_slice_display: EsdfSlice | None = None
        self._esdf_slice_center_vbo: VBO | None = None
        self._esdf_slice_color_vbo: VBO | None = None
        self._esdf_slice_geometry_dirty = True
        self._esdf_slice_dirty = True
        self._esdf_slice_status = "Load voxel_map_tsdf.npz to enable ESDF slicing."
        self._nvidia_smi = shutil.which("nvidia-smi") or "nvidia-smi"
        self._stats_refresh_sec = 1.0
        self._stats_last_update = 0.0
        self._stats_cache: dict[str, object] = {
            "rss_bytes": 0,
            "overlay_bytes": 0,
            "gpu_text": "GPU stats unavailable",
            "gpu_proc_text": "GUI VRAM unavailable",
        }

        if occupancy_file.strip():
            self._try_load_occupancy(occupancy_file)
        if extra_occupancy_file.strip():
            self._try_load_extra_occupancy(extra_occupancy_file)

    @property
    def name(self) -> str:
        return "Occupancy Overlay"

    @staticmethod
    def _build_cube_shader() -> int:
        cube_vert = """
        #version 330
        layout(location=0) in vec3 cubeVertex;
        layout(location=1) in vec3 cubeNormal;
        layout(location=2) in vec3 centerWorld;
        layout(location=3) in vec4 instanceColor;
        layout(location=4) in float instanceOverlap;
        uniform mat4 view;
        uniform mat4 projection;
        uniform mat4 mapToTrainingWorld;
        uniform float cubeSize;
        out vec3 faceNormalAxis;
        out vec3 normalView;
        out vec3 viewPos3;
        out vec3 localPos;
        out vec4 vertexColor;
        flat out float vertexOverlap;
        out float depth;
        void main() {
            vec3 world = (mapToTrainingWorld * vec4(centerWorld + cubeVertex * cubeSize, 1.0)).xyz;
            vec3 normalWorld = normalize(mat3(mapToTrainingWorld) * cubeNormal);
            vec4 viewPos = view * vec4(world, 1.0);
            faceNormalAxis = normalize(cubeNormal);
            normalView = normalize((view * vec4(normalWorld, 0.0)).xyz);
            viewPos3 = viewPos.xyz;
            localPos = cubeVertex;
            vertexColor = instanceColor;
            vertexOverlap = instanceOverlap;
            depth = -viewPos.z;
            gl_Position = projection * viewPos;
        }
        """
        cube_frag = """
        #version 330
        in vec3 faceNormalAxis;
        in vec3 normalView;
        in vec3 viewPos3;
        in vec3 localPos;
        in vec4 vertexColor;
        flat in float vertexOverlap;
        in float depth;
        uniform vec2 viewportSize;
        uniform float near;
        uniform bool occlusionEnabled;
        uniform float occlusionAlphaThreshold;
        uniform vec4 color;
        uniform vec4 overlapColor;
        uniform bool useInstanceColor;
        uniform bool useOverlapFlags;
        uniform sampler2D depthSampler;
        uniform sampler2D alphaSampler;
        out vec4 fragColor;
        void main() {
            vec4 baseColor = useInstanceColor ? vertexColor : color;
            if (useOverlapFlags && vertexOverlap > 0.5) {
                baseColor = overlapColor;
            }
            float outAlpha = baseColor.a;
            if (occlusionEnabled) {
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

            vec3 viewDir = normalize(-viewPos3);
            vec3 faceNormal = normalize(normalView);
            if (dot(faceNormal, viewDir) < 0.0) {
                faceNormal = -faceNormal;
            }
            vec3 lightDir = normalize(vec3(0.35, 0.45, 0.82));
            vec3 fillLightDir = normalize(vec3(-0.52, 0.18, 0.70));
            float diffuse = max(dot(faceNormal, lightDir), 0.0);
            float fill = max(dot(faceNormal, fillLightDir), 0.0);
            float rim = pow(1.0 - max(dot(faceNormal, viewDir), 0.0), 3.0);

            vec3 absLocal = abs(localPos) * 2.0;
            vec3 edgeAxes = vec3(
                absLocal.x * (1.0 - abs(faceNormalAxis.x)),
                absLocal.y * (1.0 - abs(faceNormalAxis.y)),
                absLocal.z * (1.0 - abs(faceNormalAxis.z))
            );
            float faceEdge = max(max(edgeAxes.x, edgeAxes.y), edgeAxes.z);
            float edgeDarken = smoothstep(0.82, 0.98, faceEdge);

            float shade = 0.52 + 0.38 * diffuse + 0.16 * fill + 0.10 * rim;
            shade *= mix(1.0, 0.82, edgeDarken);
            fragColor = vec4(baseColor.rgb * clamp(shade, 0.45, 1.18), outAlpha);
        }
        """
        return compileProgram(
            compileShader(cube_vert, gl.GL_VERTEX_SHADER),
            compileShader(cube_frag, gl.GL_FRAGMENT_SHADER),
        )

    @staticmethod
    def _unit_cube_vertices() -> np.ndarray:
        return np.array([
            [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5],
            [-0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5],
            [-0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
            [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5],
            [0.5, -0.5, -0.5], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5],
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5],
            [-0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5],
            [-0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        ], dtype=np.float32)

    @staticmethod
    def _unit_cube_normals() -> np.ndarray:
        return np.array([
            [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        ], dtype=np.float32)

    @staticmethod
    def _indices_to_world_points(idx: np.ndarray, origin: np.ndarray, voxel_size: float) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.float32).reshape(-1, 3)
        origin = np.asarray(origin, dtype=np.float32).reshape(1, 3)
        return origin + (idx + 0.5) * float(voxel_size)

    @staticmethod
    def _world_points_to_indices(points: np.ndarray, origin: np.ndarray, voxel_size: float) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        origin = np.asarray(origin, dtype=np.float32).reshape(1, 3)
        idx_float = (points - origin) / float(voxel_size) - 0.5
        return np.rint(idx_float).astype(np.int32)

    @staticmethod
    def _voxel_index_keys(indices: np.ndarray) -> np.ndarray:
        """Return sortable row keys without retaining a Python set per voxel."""
        rows = np.ascontiguousarray(indices, dtype=np.int64).reshape(-1, 3)
        return rows.view(np.dtype((np.void, rows.dtype.itemsize * 3))).reshape(-1)

    def _rebuild_overlap_flags(self) -> None:
        """Mark common occupied cells using one byte per voxel."""
        self._occupied_overlap_flags = np.zeros(self._num_occupied, dtype=np.uint8)
        self._extra_occupied_overlap_flags = np.zeros(
            self._extra_num_occupied, dtype=np.uint8
        )
        self._num_overlap = 0
        self._invalidate_filtered_cache()
        self._extra_filter_cache_key = None
        self._extra_cached_occ_overlap_flags = np.zeros(0, dtype=np.uint8)

        if self._occupied_points_world is None or self._extra_occupied_points_world is None:
            self._overlap_status = "Load two aligned voxel maps to detect overlap."
            return
        if self._occupied_voxel_idx is None or self._extra_occupied_voxel_idx is None:
            self._overlap_status = "Overlap requires occupied_voxel_idx metadata in both maps."
            return
        if self._grid_origin_world is None or self._extra_grid_origin_world is None:
            self._overlap_status = "Overlap requires origin metadata in both maps."
            return

        voxel_size = float(self._voxel_size_world)
        size_tolerance = max(1.0e-6, abs(voxel_size) * 1.0e-5)
        if not np.isclose(
            voxel_size,
            float(self._extra_voxel_size_world),
            rtol=0.0,
            atol=size_tolerance,
        ):
            self._overlap_status = "Overlap requires matching voxel sizes."
            return

        # Express the extra grid in the main map frame. Integer translation is
        # supported without transforming or duplicating every voxel center.
        extra_map_to_main_map = (
            self._map_from_training_world.astype(np.float64)
            @ self._extra_training_world_from_map.astype(np.float64)
        )
        if not np.allclose(
            extra_map_to_main_map[:3, :3], np.eye(3), atol=1.0e-5, rtol=0.0
        ):
            self._overlap_status = "Overlap requires axis-aligned map grids."
            return
        extra_origin_in_main = (
            extra_map_to_main_map[:3, :3]
            @ self._extra_grid_origin_world.astype(np.float64)
            + extra_map_to_main_map[:3, 3]
        )
        offset_float = (
            extra_origin_in_main - self._grid_origin_world.astype(np.float64)
        ) / voxel_size
        offset = np.rint(offset_float).astype(np.int64)
        if not np.allclose(offset_float, offset, atol=1.0e-4, rtol=0.0):
            self._overlap_status = "Overlap requires voxel-aligned grid origins."
            return

        main_keys = self._voxel_index_keys(self._occupied_voxel_idx)
        extra_keys = self._voxel_index_keys(
            self._extra_occupied_voxel_idx.astype(np.int64) + offset[None, :]
        )
        _, main_matches, extra_matches = np.intersect1d(
            main_keys,
            extra_keys,
            assume_unique=True,
            return_indices=True,
        )
        self._occupied_overlap_flags[main_matches] = 1
        self._extra_occupied_overlap_flags[extra_matches] = 1
        self._num_overlap = int(main_matches.size)
        self._overlap_status = f"Overlapping occupied voxels: {self._num_overlap}."

    @staticmethod
    def _load_occupancy_arrays(
        path: Path,
        *,
        fallback_voxel_size: float,
    ) -> tuple[
        np.ndarray,
        np.ndarray | None,
        float,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray,
        np.ndarray,
    ]:
        occupied_points = None
        occupied_idx = None
        voxel_size = float(fallback_voxel_size)
        grid_origin = None
        grid_voxel_dims = None
        map_from_training_world = np.eye(4, dtype=np.float32)

        if path.suffix.lower() == ".npz":
            with np.load(path) as data:
                if "voxel_size" in data:
                    voxel_size = float(np.asarray(data["voxel_size"]).reshape(-1)[0])
                grid_origin = (
                    np.asarray(data["origin"], dtype=np.float32).reshape(3)
                    if "origin" in data
                    else None
                )
                grid_voxel_dims = (
                    np.asarray(data["voxel_dims"], dtype=np.int32).reshape(3)
                    if "voxel_dims" in data
                    else None
                )
                if "map_from_training_world" in data:
                    map_from_training_world = validate_rigid_transform(
                        np.asarray(data["map_from_training_world"], dtype=np.float32)
                    )
                occupied_idx = (
                    np.asarray(data["occupied_voxel_idx"], dtype=np.int32).reshape(-1, 3)
                    if "occupied_voxel_idx" in data
                    else None
                )
                if "occupied_voxel_world" in data:
                    occupied_points = np.asarray(data["occupied_voxel_world"], dtype=np.float32)
                elif occupied_idx is not None and grid_origin is not None:
                    occupied_points = OccupancyOverlayExtra._indices_to_world_points(
                        idx=occupied_idx.astype(np.float32, copy=False),
                        origin=grid_origin,
                        voxel_size=voxel_size,
                    ).astype(np.float32)
                else:
                    raise ValueError(
                        "NPZ missing required keys: occupied_voxel_world or (occupied_voxel_idx, origin, voxel_size)."
                    )
        elif path.suffix.lower() == ".npy":
            occupied_points = np.asarray(np.load(path), dtype=np.float32)
        else:
            occupied_points = np.asarray(np.loadtxt(path, delimiter=None), dtype=np.float32)

        if occupied_points is None:
            occupied_points = np.zeros((0, 3), dtype=np.float32)
        if occupied_points.ndim == 1:
            occupied_points = occupied_points.reshape(1, -1)
        if occupied_points.shape[1] != 3:
            raise ValueError("Occupied points must be Nx3.")
        if occupied_idx is None and grid_origin is not None and voxel_size > 0.0:
            occupied_idx = OccupancyOverlayExtra._world_points_to_indices(occupied_points, grid_origin, voxel_size)
        training_world_from_map = np.linalg.inv(map_from_training_world).astype(np.float32)
        return (
            occupied_points.astype(np.float32),
            occupied_idx,
            voxel_size,
            grid_origin,
            grid_voxel_dims,
            map_from_training_world,
            training_world_from_map,
        )

    def _try_load_occupancy(self, file_path: str) -> None:
        path = resolve_existing_path(file_path, default_suffix=".npz")
        if not path.is_file():
            self._status = f"File not found: {path}"
            return
        try:
            (
                occupied_points,
                occupied_idx,
                voxel_size,
                grid_origin,
                grid_voxel_dims,
                map_from_training_world,
                training_world_from_map,
            ) = self._load_occupancy_arrays(
                path,
                fallback_voxel_size=self._voxel_size_world,
            )
            self._voxel_size_world = voxel_size
            self._grid_origin_world = grid_origin
            self._grid_voxel_dims = grid_voxel_dims
            self._map_from_training_world = map_from_training_world
            self._training_world_from_map = training_world_from_map
            self._occupied_points_world = occupied_points
            self._occupied_voxel_idx = occupied_idx
            self._num_occupied = int(self._occupied_points_world.shape[0])
            self._geometry_dirty = True
            self._invalidate_filtered_cache()
            self._probe_cache_key = None
            self._loaded_file_path = str(path.resolve())
            self._reset_esdf_slice_for_loaded_map()
            self._status = (
                f"Loaded occupied={self._num_occupied}, "
                f"voxel_size={self._voxel_size_world:.4f}."
            )
            self._rebuild_overlap_flags()
        except Exception as exc:
            self._status = f"Load failed: {exc}"

    def _try_load_extra_occupancy(self, file_path: str) -> None:
        path = resolve_existing_path(file_path, default_suffix=".npz")
        if not path.is_file():
            self._extra_status = f"File not found: {path}"
            return
        try:
            (
                occupied_points,
                occupied_idx,
                voxel_size,
                grid_origin,
                grid_voxel_dims,
                map_from_training_world,
                training_world_from_map,
            ) = self._load_occupancy_arrays(
                path,
                fallback_voxel_size=self._extra_voxel_size_world,
            )
            self._extra_voxel_size_world = voxel_size
            self._extra_grid_origin_world = grid_origin
            self._extra_grid_voxel_dims = grid_voxel_dims
            self._extra_map_from_training_world = map_from_training_world
            self._extra_training_world_from_map = training_world_from_map
            self._extra_occupied_points_world = occupied_points
            self._extra_occupied_voxel_idx = occupied_idx
            self._extra_num_occupied = int(self._extra_occupied_points_world.shape[0])
            self._extra_loaded_file_path = str(path.resolve())
            self._extra_filter_cache_key = None
            self._extra_cached_occ_points = np.zeros((0, 3), dtype=np.float32)
            self._probe_cache_key = None
            self._extra_status = (
                f"Loaded extra occupied={self._extra_num_occupied}, "
                f"voxel_size={self._extra_voxel_size_world:.4f}."
            )
            self._rebuild_overlap_flags()
        except Exception as exc:
            self._extra_status = f"Extra load failed: {exc}"
            self._extra_occupied_points_world = None
            self._extra_occupied_voxel_idx = None
            self._extra_num_occupied = 0
            self._rebuild_overlap_flags()

    def _invalidate_filtered_cache(self) -> None:
        self._filter_cache_key = None
        self._cached_occ_points = np.zeros((0, 3), dtype=np.float32)
        self._cached_occ_overlap_flags = np.zeros(0, dtype=np.uint8)

    def _mark_esdf_slice_dirty(self, *, clear_display: bool = False) -> None:
        self._esdf_slice_dirty = True
        self._esdf_slice_geometry_dirty = True
        if clear_display:
            self._esdf_slice_data = None
            self._esdf_slice_display = None
            self._esdf_slice_center_vbo = None
            self._esdf_slice_color_vbo = None

    def _reset_esdf_slice_for_loaded_map(self) -> None:
        if self._esdf_slice_graph_future is not None:
            self._esdf_slice_graph_future.cancel()
            self._esdf_slice_graph_future = None
            self._esdf_slice_graph_future_key = None
        self._esdf_slice_graph = None
        self._esdf_slice_graph_key = None
        self._esdf_slice_data = None
        self._esdf_slice_display = None
        self._esdf_slice_center_vbo = None
        self._esdf_slice_color_vbo = None
        self._mark_esdf_slice_dirty()
        if self._loaded_file_path and self._loaded_map_supports_esdf_slice():
            inferred = infer_esdf_path_from_voxel_map(self._loaded_file_path)
            if inferred is not None and (self._esdf_slice_auto_file or self._esdf_slice_file_input.strip() == ""):
                self._esdf_slice_file_input = str(inferred)
                self._esdf_slice_auto_file = True
        elif self._loaded_file_path:
            self._esdf_slice_file_input = ""
            self._esdf_slice_auto_file = True
        self._initialize_esdf_slice_coordinate()
        if self._loaded_file_path and self._loaded_map_supports_esdf_slice():
            self._esdf_slice_status = "ESDF slice ready to load."
        elif self._loaded_file_path:
            self._esdf_slice_status = "ESDF slicing requires a TSDF-derived voxel_map_tsdf.npz."

    def _loaded_map_supports_esdf_slice(self) -> bool:
        return bool(self._loaded_file_path and voxel_map_supports_esdf_slice(self._loaded_file_path))

    def _initialize_esdf_slice_coordinate(self) -> None:
        if self._grid_origin_world is None or self._grid_voxel_dims is None:
            return
        axis_idx = int(np.clip(self._esdf_slice_axis_idx, 0, 2))
        lo = float(self._grid_origin_world[axis_idx])
        hi = lo + float(self._grid_voxel_dims[axis_idx]) * float(self._voxel_size_world)
        if not self._esdf_slice_coord_initialized or not (lo <= float(self._esdf_slice_coord) <= hi):
            self._esdf_slice_coord = 0.5 * (lo + hi)
            self._esdf_slice_coord_initialized = True

    def _commit_esdf_slice_controls(self) -> None:
        self._esdf_slice_applied_auto_file = bool(self._esdf_slice_auto_file)
        self._esdf_slice_applied_file_input = self._esdf_slice_file_input
        self._esdf_slice_applied_show_observed = bool(self._esdf_slice_show_observed)
        self._esdf_slice_applied_show_unknown = bool(self._esdf_slice_show_unknown)
        self._esdf_slice_applied_show_occupied = bool(self._esdf_slice_show_occupied)
        self._esdf_slice_applied_max_distance_m = max(0.001, float(self._esdf_slice_max_distance_m))

    def _has_pending_esdf_slice_controls(self) -> bool:
        if bool(self._esdf_slice_auto_file) != bool(self._esdf_slice_applied_auto_file):
            return True
        if not bool(self._esdf_slice_auto_file):
            if self._esdf_slice_file_input.strip() != self._esdf_slice_applied_file_input.strip():
                return True
        if bool(self._esdf_slice_show_observed) != bool(self._esdf_slice_applied_show_observed):
            return True
        if bool(self._esdf_slice_show_unknown) != bool(self._esdf_slice_applied_show_unknown):
            return True
        if bool(self._esdf_slice_show_occupied) != bool(self._esdf_slice_applied_show_occupied):
            return True
        return abs(float(self._esdf_slice_max_distance_m) - float(self._esdf_slice_applied_max_distance_m)) > 1e-9

    def _resolve_esdf_slice_path(self, *, use_applied_controls: bool = False) -> Path | None:
        auto_file = (
            self._esdf_slice_applied_auto_file if use_applied_controls else self._esdf_slice_auto_file
        )
        file_input = (
            self._esdf_slice_applied_file_input if use_applied_controls else self._esdf_slice_file_input
        )
        if auto_file and self._loaded_file_path:
            inferred = infer_esdf_path_from_voxel_map(self._loaded_file_path)
            if inferred is not None:
                if not use_applied_controls:
                    self._esdf_slice_file_input = str(inferred)
                return inferred
        candidate = file_input.strip()
        if candidate == "":
            return None
        path = Path(candidate).expanduser().resolve()
        return path if path.is_file() else None

    def _esdf_slice_axis_range_text(self) -> str:
        if self._grid_origin_world is None or self._grid_voxel_dims is None:
            return "Slice range unavailable until a grid npz is loaded."
        axis_idx = int(np.clip(self._esdf_slice_axis_idx, 0, 2))
        lo = float(self._grid_origin_world[axis_idx])
        hi = lo + float(self._grid_voxel_dims[axis_idx]) * float(self._voxel_size_world)
        return f"{AXIS_NAMES[axis_idx]} range: {lo:.4f} to {hi:.4f}"

    def _ensure_esdf_slice_graph(self, esdf_path: Path) -> EsdfGridGraph | None:
        if not self._loaded_file_path:
            self._esdf_slice_status = "Load voxel_map_tsdf.npz before ESDF slicing."
            return None
        if not self._loaded_map_supports_esdf_slice():
            self._esdf_slice_status = "ESDF slicing requires a TSDF-derived voxel_map_tsdf.npz."
            return None
        grid_path = str(Path(self._loaded_file_path).expanduser().resolve())
        key = (str(esdf_path.resolve()), grid_path)
        if self._esdf_slice_graph_key == key and self._esdf_slice_graph is not None:
            return self._esdf_slice_graph
        if self._esdf_slice_graph_future_key == key and self._esdf_slice_graph_future is not None:
            if not self._esdf_slice_graph_future.done():
                self._esdf_slice_status = "Loading ESDF grid in background..."
                return None
            self._esdf_slice_graph = self._esdf_slice_graph_future.result()
            self._esdf_slice_graph_key = key
            self._esdf_slice_graph_future = None
            self._esdf_slice_graph_future_key = None
            return self._esdf_slice_graph

        if self._esdf_slice_graph_future is not None:
            self._esdf_slice_graph_future.cancel()
        self._esdf_slice_graph_future = self._esdf_slice_executor.submit(
            load_esdf_grid,
            esdf_path,
            grid_path,
            use_occupancy_guard=True,
            load_free_idx=False,
            cache_aligned=True,
            unknown_policy=UNKNOWN_POLICY_OCCUPIED,
        )
        self._esdf_slice_graph_future_key = key
        self._esdf_slice_status = "Loading ESDF grid in background..."
        return None

    def _rebuild_esdf_slice(self, *, commit_controls: bool = True) -> None:
        if commit_controls:
            previous_source = (
                bool(self._esdf_slice_applied_auto_file),
                self._esdf_slice_applied_file_input.strip(),
            )
            next_source = (bool(self._esdf_slice_auto_file), self._esdf_slice_file_input.strip())
            if previous_source != next_source:
                self._esdf_slice_graph = None
                self._esdf_slice_graph_key = None
            self._commit_esdf_slice_controls()

        esdf_path = self._resolve_esdf_slice_path(use_applied_controls=True)
        if esdf_path is None:
            self._esdf_slice_status = "ESDF file not found. Expected scene/sdf/esdf_observed.ply."
            self._esdf_slice_data = None
            self._esdf_slice_display = None
            self._esdf_slice_geometry_dirty = True
            return
        try:
            graph = self._ensure_esdf_slice_graph(esdf_path)
            if graph is None:
                return
            axis = AXIS_NAMES[int(np.clip(self._esdf_slice_axis_idx, 0, 2))]
            slice_data = build_esdf_slice_from_graph(
                graph,
                axis=axis,
                coordinate=float(self._esdf_slice_coord),
                max_distance_m=float(self._esdf_slice_applied_max_distance_m),
                include_observed=bool(self._esdf_slice_applied_show_observed),
                include_unknown=bool(self._esdf_slice_applied_show_unknown),
                include_occupied=bool(self._esdf_slice_applied_show_occupied),
                esdf_path=esdf_path,
                grid_path=self._loaded_file_path,
            )
            display_data = downsample_esdf_slice(slice_data, int(self._esdf_slice_max_voxels))
            self._esdf_slice_data = slice_data
            self._esdf_slice_display = display_data
            self._esdf_slice_dirty = self._has_pending_esdf_slice_controls()
            self._esdf_slice_geometry_dirty = True
            clipped = "" if display_data.num_voxels == slice_data.num_voxels else f", displayed={display_data.num_voxels}"
            pending = " Display options pending; click Update ESDF Slice." if self._esdf_slice_dirty else ""
            self._esdf_slice_status = (
                f"Slice {axis}={slice_data.plane_coordinate:.4f}: "
                f"observed={slice_data.num_observed}, occupied={slice_data.num_occupied}, "
                f"unknown={slice_data.num_unknown}{clipped}.{pending}"
            )
        except Exception as exc:
            self._esdf_slice_status = f"ESDF slice load failed: {exc}"
            self._esdf_slice_data = None
            self._esdf_slice_display = None
            self._esdf_slice_dirty = True
            self._esdf_slice_geometry_dirty = True

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        value = float(max(num_bytes, 0))
        unit = units[0]
        for unit in units:
            if value < 1024.0 or unit == units[-1]:
                break
            value /= 1024.0
        return f"{value:.1f} {unit}"

    def _estimate_overlay_bytes(self) -> int:
        total = 0
        arrays = (
            self._occupied_points_world,
            self._extra_occupied_points_world,
            self._occupied_overlap_flags,
            self._extra_occupied_overlap_flags,
            self._cached_occ_points,
            self._cached_occ_overlap_flags,
            self._extra_cached_occ_points,
            self._extra_cached_occ_overlap_flags,
            self._probe_occ_center,
            self._probe_extra_center,
            self._esdf_slice_display.centers_world if self._esdf_slice_display is not None else None,
            self._esdf_slice_display.color_rgba if self._esdf_slice_display is not None else None,
        )
        for arr in arrays:
            if isinstance(arr, np.ndarray):
                total += int(arr.nbytes)
        return total

    def _query_nvidia_stats(self) -> tuple[str, str]:
        try:
            gpu_cmd = [
                self._nvidia_smi,
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
            gpu_out = subprocess.run(gpu_cmd, check=True, capture_output=True, text=True, timeout=1.0).stdout.strip()
            gpu_lines = [line.strip() for line in gpu_out.splitlines() if line.strip()]
            if gpu_lines:
                gpu_text = " | ".join(
                    f"GPU {parts[0].strip()} {parts[1].strip()}: {parts[2].strip()}/{parts[3].strip()} MB"
                    for line in gpu_lines
                    if len((parts := line.split(","))) >= 4
                )
            else:
                gpu_text = "GPU stats unavailable"

            proc_cmd = [
                self._nvidia_smi,
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ]
            proc_out = subprocess.run(proc_cmd, check=True, capture_output=True, text=True, timeout=1.0).stdout.strip()
            pid = os.getpid()
            proc_mb = []
            for line in proc_out.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == pid:
                    try:
                        proc_mb.append(int(float(parts[1])))
                    except ValueError:
                        continue
            gpu_proc_text = f"GUI VRAM: {sum(proc_mb)} MB" if proc_mb else "GUI VRAM: 0 MB or not tracked"
            return gpu_text, gpu_proc_text
        except Exception:
            return "GPU stats unavailable", "GUI VRAM unavailable"

    def _refresh_runtime_stats(self) -> None:
        now = time.monotonic()
        if now - self._stats_last_update < self._stats_refresh_sec:
            return
        self._stats_last_update = now
        rss_pages = int(Path("/proc/self/statm").read_text().split()[1])
        rss_bytes = rss_pages * os.sysconf("SC_PAGE_SIZE")
        overlay_bytes = self._estimate_overlay_bytes()
        gpu_text, gpu_proc_text = self._query_nvidia_stats()
        self._stats_cache = {
            "rss_bytes": rss_bytes,
            "overlay_bytes": overlay_bytes,
            "gpu_text": gpu_text,
            "gpu_proc_text": gpu_proc_text,
        }

    def _ensure_geometry(self) -> None:
        if not self._geometry_dirty:
            return
        if self._occupied_points_world is None or self._occupied_points_world.shape[0] == 0:
            self._occupied_center_vbo = None
            self._num_occupied = 0
        else:
            self._occupied_center_vbo = update_vbo(self._occupied_center_vbo, self._occupied_points_world)

        self._geometry_dirty = False

    def _ensure_esdf_slice_geometry(self) -> None:
        if not self._esdf_slice_geometry_dirty:
            return
        if self._esdf_slice_display is None or self._esdf_slice_display.num_voxels == 0:
            self._esdf_slice_center_vbo = None
            self._esdf_slice_color_vbo = None
        else:
            self._esdf_slice_center_vbo = update_vbo(
                self._esdf_slice_center_vbo,
                self._esdf_slice_display.centers_world.astype(np.float32, copy=False),
            )
            self._esdf_slice_color_vbo = update_vbo(
                self._esdf_slice_color_vbo,
                self._esdf_slice_display.color_rgba.astype(np.float32, copy=False),
            )
        self._esdf_slice_geometry_dirty = False

    def _ensure_loaded(self) -> None:
        candidate = self._occupancy_file_input.strip()
        if candidate == "":
            return
        try:
            resolved = str(resolve_existing_path(candidate, default_suffix=".npz"))
        except Exception:
            resolved = str(Path(candidate).expanduser().resolve())
        if self._loaded_file_path == resolved and self._occupied_points_world is not None:
            return
        self._try_load_occupancy(candidate)

    def _ensure_extra_loaded(self) -> None:
        candidate = self._extra_occupancy_file_input.strip()
        if candidate == "":
            return
        try:
            resolved = str(resolve_existing_path(candidate, default_suffix=".npz"))
        except Exception:
            resolved = str(Path(candidate).expanduser().resolve())
        if self._extra_loaded_file_path == resolved and self._extra_occupied_points_world is not None:
            return
        self._try_load_extra_occupancy(candidate)

    def _occupancy_browse_initial_path(self) -> str:
        current = self._occupancy_file_input.strip()
        if current:
            return current
        if self._loaded_file_path:
            return self._loaded_file_path
        return ""

    def _extra_occupancy_browse_initial_path(self) -> str:
        current = self._extra_occupancy_file_input.strip()
        if current:
            return current
        if self._loaded_file_path:
            return self._loaded_file_path
        return self._occupancy_file_input.strip()

    @staticmethod
    def _parse_world_coord(text: str) -> np.ndarray:
        parts = text.replace(",", " ").split()
        if len(parts) != 3:
            raise ValueError("expected x, y, z")
        return np.array([float(part) for part in parts], dtype=np.float32)

    @staticmethod
    def _voxel_for_world_coord(
        coord: np.ndarray,
        origin: np.ndarray | None,
        voxel_size: float,
        voxel_dims: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, bool] | None:
        if origin is None:
            return None
        voxel_size = float(voxel_size)
        if voxel_size <= 0.0:
            return None
        idx = np.floor((coord.astype(np.float32) - origin.astype(np.float32)) / voxel_size).astype(np.int32)
        in_grid = True
        if voxel_dims is not None:
            in_grid = bool(np.logical_and(idx >= 0, idx < voxel_dims.astype(np.int32)).all())
        center = OccupancyOverlayExtra._indices_to_world_points(idx.reshape(1, 3), origin, voxel_size)[0]
        return idx, center.astype(np.float32), in_grid

    @staticmethod
    def _contains_voxel_idx(occupied_idx: np.ndarray | None, idx: np.ndarray) -> bool | None:
        if occupied_idx is None:
            return None
        if occupied_idx.shape[0] == 0:
            return False
        idx = np.asarray(idx, dtype=np.int32).reshape(1, 3)
        return bool(np.any(np.all(occupied_idx == idx, axis=1)))

    @staticmethod
    def _format_vec(values: np.ndarray, precision: int = 6) -> str:
        arr = np.asarray(values).reshape(-1)
        return "(" + ", ".join(f"{float(v):.{precision}f}" for v in arr.tolist()) + ")"

    def _update_probe_voxels(self) -> None:
        key = (
            bool(self._probe_voxel_enabled),
            self._probe_coord_input.strip(),
            self._loaded_file_path,
            self._extra_loaded_file_path,
            None if self._grid_origin_world is None else tuple(np.round(self._grid_origin_world, 9).tolist()),
            None if self._extra_grid_origin_world is None else tuple(np.round(self._extra_grid_origin_world, 9).tolist()),
            round(float(self._voxel_size_world), 9),
            round(float(self._extra_voxel_size_world), 9),
        )
        if self._probe_cache_key == key:
            return
        self._probe_cache_key = key
        self._probe_occ_center = np.zeros((0, 3), dtype=np.float32)
        self._probe_extra_center = np.zeros((0, 3), dtype=np.float32)
        self._probe_geometry_dirty = True

        if not self._probe_voxel_enabled:
            self._probe_status = "Enable coordinate voxel check to compare the two maps."
            return
        if self._occupied_points_world is None or self._loaded_file_path is None:
            self._probe_status = "Load the main occupancy map first."
            return
        if self._extra_occupied_points_world is None or self._extra_loaded_file_path is None:
            self._probe_status = "Load the extra occupancy map first."
            return

        try:
            coord = self._parse_world_coord(self._probe_coord_input)
        except ValueError as exc:
            self._probe_status = f"Invalid coordinate: {exc}"
            return

        main_coord = apply_transform(
            coord.reshape(1, 3), self._map_from_training_world
        )[0]
        extra_coord = apply_transform(
            coord.reshape(1, 3), self._extra_map_from_training_world
        )[0]
        main_voxel = self._voxel_for_world_coord(
            main_coord,
            self._grid_origin_world,
            self._voxel_size_world,
            self._grid_voxel_dims,
        )
        extra_voxel = self._voxel_for_world_coord(
            extra_coord,
            self._extra_grid_origin_world,
            self._extra_voxel_size_world,
            self._extra_grid_voxel_dims,
        )
        if main_voxel is None or extra_voxel is None:
            self._probe_status = "Both maps need origin and voxel_size metadata for coordinate voxel check."
            return

        main_idx, main_center, main_in_grid = main_voxel
        extra_idx, extra_center, extra_in_grid = extra_voxel

        main_has_voxel = self._contains_voxel_idx(self._occupied_voxel_idx, main_idx)
        extra_has_voxel = self._contains_voxel_idx(self._extra_occupied_voxel_idx, extra_idx)
        if main_has_voxel is None or extra_has_voxel is None:
            self._probe_status = (
                "Coordinate voxel check requires occupied_voxel_idx in both maps; "
                f"main_idx_available={main_has_voxel is not None}, "
                f"extra_idx_available={extra_has_voxel is not None}."
            )
            return

        main_visible = bool(main_in_grid and main_has_voxel)
        extra_visible = bool(extra_in_grid and extra_has_voxel)
        if main_visible:
            self._probe_occ_center = main_center.reshape(1, 3).astype(np.float32)
        if extra_visible:
            self._probe_extra_center = extra_center.reshape(1, 3).astype(np.float32)

        main_label = f"main_idx={tuple(main_idx.tolist())}, occupied={main_has_voxel}, in_grid={main_in_grid}"
        extra_label = f"extra_idx={tuple(extra_idx.tolist())}, occupied={extra_has_voxel}, in_grid={extra_in_grid}"
        if not main_visible or not extra_visible:
            if main_visible and not extra_visible:
                self._probe_status = f"Only main map has an occupied voxel here; no comparison. {main_label}; {extra_label}."
            elif extra_visible and not main_visible:
                self._probe_status = f"Only extra map has an occupied voxel here; no comparison. {main_label}; {extra_label}."
            else:
                self._probe_status = f"No occupied voxel at this coordinate in either map; no comparison. {main_label}; {extra_label}."
            return

        main_center_training = apply_transform(
            main_center.reshape(1, 3), self._training_world_from_map
        )[0]
        extra_center_training = apply_transform(
            extra_center.reshape(1, 3), self._extra_training_world_from_map
        )[0]
        delta = extra_center_training.astype(np.float64) - main_center_training.astype(np.float64)
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm <= float(self._probe_align_epsilon):
            delta = np.zeros(3, dtype=np.float64)
            delta_norm = 0.0
            aligned_text = f"aligned (diff <= {self._probe_align_epsilon:g})"
        else:
            aligned_text = "NOT aligned"
        grid_text = ""
        if not main_in_grid or not extra_in_grid:
            grid_text = f" Main in grid={main_in_grid}, extra in grid={extra_in_grid}."
        self._probe_status = (
            f"{aligned_text}; delta={self._format_vec(delta)}, norm={delta_norm:.8f}. "
            f"{main_label}; {extra_label}.{grid_text}"
        )

    def _ensure_probe_geometry(self) -> None:
        if not self._probe_geometry_dirty:
            return
        self._probe_occupied_vbo = (
            update_vbo(self._probe_occupied_vbo, self._probe_occ_center)
            if self._probe_occ_center.shape[0] > 0
            else None
        )
        self._probe_extra_occupied_vbo = (
            update_vbo(self._probe_extra_occupied_vbo, self._probe_extra_center)
            if self._probe_extra_center.shape[0] > 0
            else None
        )
        self._probe_geometry_dirty = False

    def _render_data_section(self) -> None:
        imgui.text("Voxel Data")
        _, self._occupancy_file_input = imgui.input_text(
            f"Occupancy File (voxel_map/.npz/.npy/.txt)##{self.name}",
            self._occupancy_file_input,
        )
        if imgui.button(f"Browse##Occupancy{self.name}"):
            selected = select_file_dialog(
                title="Select Occupancy File",
                initial_path=self._occupancy_browse_initial_path(),
                filetypes=(
                    ("Occupancy Files", "*.npz *.npy *.txt"),
                    ("NumPy", "*.npz *.npy"),
                    ("Text", "*.txt"),
                    ("All Files", "*"),
                ),
            )
            if selected is not None:
                self._occupancy_file_input = selected
        _, self._extra_occupancy_file_input = imgui.input_text(
            f"Extra Occupancy File##{self.name}",
            self._extra_occupancy_file_input,
        )
        if imgui.button(f"Browse##ExtraOccupancy{self.name}"):
            selected = select_file_dialog(
                title="Select Extra Occupancy File",
                initial_path=self._extra_occupancy_browse_initial_path(),
                filetypes=(
                    ("Occupancy Files", "*.npz *.npy *.txt"),
                    ("NumPy", "*.npz *.npy"),
                    ("Text", "*.txt"),
                    ("All Files", "*"),
                ),
            )
            if selected is not None:
                self._extra_occupancy_file_input = selected

    def _render_display_section(self) -> None:
        imgui.text("Voxel Display")
        _, self._show_occupied = imgui.checkbox(f"Show Occupied Voxels##{self.name}", self._show_occupied)
        _, self._show_extra_occupied = imgui.checkbox(
            f"Show Extra Occupied Voxels##{self.name}",
            self._show_extra_occupied,
        )
        _, self._highlight_overlap = imgui.checkbox(
            f"Highlight Overlap##{self.name}",
            self._highlight_overlap,
        )
        _, self._max_points = imgui.drag_int(f"Max Nearby Voxels##{self.name}", self._max_points, 1000, 1000, 2000000)
        _, self._voxel_size_scale = imgui.drag_float(
            f"Voxel Size Scale##{self.name}",
            self._voxel_size_scale,
            0.01,
            0.1,
            3.0,
            "%.2f",
        )
        _, self._nearby_only = imgui.checkbox(f"Show Nearby Only##{self.name}", self._nearby_only)
        _, self._nearby_radius = imgui.drag_float(
            f"Nearby Radius(world)##{self.name}",
            self._nearby_radius,
            0.02,
            0.1,
            100.0,
            "%.2f",
        )
        changed, c = imgui.color_edit4(f"Occupied Color##{self.name}", self._occupied_color.tolist())
        if changed:
            self._occupied_color = np.array(c, dtype=np.float32)
        changed, c = imgui.color_edit4(
            f"Extra Occupied Color##{self.name}",
            self._extra_occupied_color.tolist(),
        )
        if changed:
            self._extra_occupied_color = np.array(c, dtype=np.float32)
        changed, c = imgui.color_edit4(
            f"Overlap Color##{self.name}",
            self._overlap_color.tolist(),
        )
        if changed:
            self._overlap_color = np.array(c, dtype=np.float32)

    def _render_probe_section(self) -> None:
        imgui.text("Coordinate Voxel Check")
        changed, self._probe_voxel_enabled = imgui.checkbox(
            f"Show Only Coordinate Voxels##{self.name}",
            self._probe_voxel_enabled,
        )
        if changed:
            self._probe_cache_key = None
        changed, self._probe_coord_input = imgui.input_text(
            f"World Coordinate##{self.name}",
            self._probe_coord_input,
        )
        if changed:
            self._probe_cache_key = None
        self._update_probe_voxels()
        imgui.text_wrapped(self._probe_status)

    def _render_esdf_slice_section(self) -> None:
        imgui.text("ESDF Slice")
        if not self._loaded_file_path:
            imgui.text_wrapped("Load voxel_map_tsdf.npz first.")
            imgui.text_wrapped(self._esdf_slice_status)
            return
        if not self._loaded_map_supports_esdf_slice():
            imgui.text_wrapped("ESDF slicing requires a TSDF-derived voxel_map_tsdf.npz.")
            imgui.text_wrapped(self._esdf_slice_status)
            return
        if self._grid_origin_world is None or self._grid_voxel_dims is None:
            imgui.text_wrapped("ESDF slicing requires a voxel_map_tsdf.npz-style grid with origin and voxel_dims.")
            return
        if self._show_esdf_slice and self._esdf_slice_dirty and self._esdf_slice_graph_future is not None:
            self._rebuild_esdf_slice(commit_controls=False)

        changed, self._show_esdf_slice = imgui.checkbox(f"Show ESDF Slice##{self.name}", self._show_esdf_slice)
        if changed and self._show_esdf_slice and self._esdf_slice_display is None:
            self._mark_esdf_slice_dirty(clear_display=True)

        changed, axis_idx = imgui.combo(
            f"Slice Axis##{self.name}",
            int(np.clip(self._esdf_slice_axis_idx, 0, 2)),
            list(AXIS_NAMES),
        )
        if changed:
            self._esdf_slice_axis_idx = axis_idx
            self._initialize_esdf_slice_coordinate()
            self._rebuild_esdf_slice(commit_controls=False)

        imgui.text(self._esdf_slice_axis_range_text())
        coord_changed, coord = imgui.drag_float(
            f"Slice Coord (world)##{self.name}",
            float(self._esdf_slice_coord),
            max(0.001, float(self._voxel_size_world)),
            -100000.0,
            100000.0,
            "%.4f",
        )
        if coord_changed:
            self._esdf_slice_coord = float(coord)
            self._rebuild_esdf_slice(commit_controls=False)

        changed, self._esdf_slice_auto_file = imgui.checkbox(
            f"Infer ESDF From Voxel Map##{self.name}",
            self._esdf_slice_auto_file,
        )
        if changed:
            self._esdf_slice_graph = None
            self._esdf_slice_graph_key = None
            self._mark_esdf_slice_dirty()
            self._esdf_slice_status = "ESDF file option changed; click Update ESDF Slice."

        changed, self._esdf_slice_file_input = imgui.input_text(
            f"ESDF File (esdf_observed.ply)##{self.name}",
            self._esdf_slice_file_input,
        )
        if changed:
            self._esdf_slice_auto_file = False
            self._esdf_slice_graph = None
            self._esdf_slice_graph_key = None
            self._mark_esdf_slice_dirty()
            self._esdf_slice_status = "ESDF file changed; click Update ESDF Slice."
        if imgui.button(f"Browse##ESDFSlice{self.name}"):
            selected = select_file_dialog(
                title="Select ESDF PLY",
                initial_path=self._esdf_slice_file_input,
                filetypes=(("ESDF PLY", "*.ply"), ("All Files", "*")),
            )
            if selected is not None:
                self._esdf_slice_file_input = selected
                self._esdf_slice_auto_file = False
                self._esdf_slice_graph = None
                self._esdf_slice_graph_key = None
                self._mark_esdf_slice_dirty()
                self._esdf_slice_status = "ESDF file changed; click Update ESDF Slice."

        changed, self._esdf_slice_show_observed = imgui.checkbox(
            f"Known ESDF Cells##{self.name}",
            self._esdf_slice_show_observed,
        )
        if changed:
            self._mark_esdf_slice_dirty()
            self._esdf_slice_status = "ESDF display options changed; click Update ESDF Slice."
        changed, self._esdf_slice_show_unknown = imgui.checkbox(
            f"Unknown Cells##{self.name}",
            self._esdf_slice_show_unknown,
        )
        if changed:
            self._mark_esdf_slice_dirty()
            self._esdf_slice_status = "ESDF display options changed; click Update ESDF Slice."
        changed, self._esdf_slice_show_occupied = imgui.checkbox(
            f"Occupied Guard##{self.name}",
            self._esdf_slice_show_occupied,
        )
        if changed:
            self._mark_esdf_slice_dirty()
            self._esdf_slice_status = "ESDF display options changed; click Update ESDF Slice."

        changed, value = imgui.drag_float(
            f"Green At Distance (m)##{self.name}",
            float(self._esdf_slice_max_distance_m),
            0.01,
            0.001,
            20.0,
            "%.3f",
        )
        if changed:
            self._esdf_slice_max_distance_m = max(0.001, float(value))
            self._mark_esdf_slice_dirty()
            self._esdf_slice_status = "ESDF distance color changed; click Update ESDF Slice."

        changed, value = imgui.drag_int(
            f"Max Slice Voxels##{self.name}",
            int(self._esdf_slice_max_voxels),
            10000,
            1000,
            3000000,
        )
        if changed:
            self._esdf_slice_max_voxels = max(1000, int(value))
            if self._esdf_slice_data is not None:
                self._esdf_slice_display = downsample_esdf_slice(self._esdf_slice_data, self._esdf_slice_max_voxels)
                self._esdf_slice_geometry_dirty = True

        if imgui.button(f"Update ESDF Slice##{self.name}"):
            self._rebuild_esdf_slice()
        imgui.text_wrapped(self._esdf_slice_status)

    def _render_stats_section(self) -> None:
        imgui.text("Voxel Status")
        imgui.text(f"Loaded occupied voxels: {self._num_occupied}")
        imgui.text_wrapped(f'Loaded file: {self._loaded_file_path or "(none)"}')
        imgui.text(f"Loaded extra occupied voxels: {self._extra_num_occupied}")
        imgui.text(f"Overlapping occupied voxels: {self._num_overlap}")
        imgui.text_wrapped(f'Loaded extra file: {self._extra_loaded_file_path or "(none)"}')
        imgui.text(f"Voxel size(world): {self._voxel_size_world:.4f}")
        imgui.text(f"Extra voxel size(world): {self._extra_voxel_size_world:.4f}")
        imgui.text(f"Nearby radius(world): {self._nearby_radius:.2f}")
        imgui.text("Voxel opacity: color alpha")
        imgui.separator()
        imgui.text(f'GUI RSS: {self._format_bytes(int(self._stats_cache["rss_bytes"]))}')
        imgui.text(f'Overlay Arrays: {self._format_bytes(int(self._stats_cache["overlay_bytes"]))}')
        imgui.text_wrapped(str(self._stats_cache["gpu_text"]))
        imgui.text(str(self._stats_cache["gpu_proc_text"]))
        imgui.text_wrapped(self._status)
        imgui.text_wrapped(self._extra_status)
        imgui.text_wrapped(self._overlap_status)

    def render_options(self) -> None:
        self._ensure_loaded()
        self._ensure_extra_loaded()
        self._refresh_runtime_stats()
        self._render_data_section()
        imgui.spacing()
        imgui.separator()
        self._render_display_section()
        imgui.spacing()
        imgui.separator()
        self._render_probe_section()
        imgui.spacing()
        imgui.separator()
        self._render_esdf_slice_section()
        imgui.spacing()
        imgui.separator()
        self._render_stats_section()

    @staticmethod
    def _camera_world_position(view: View) -> np.ndarray:
        return view.position_numpy.astype(np.float32)

    def _filter_nearby(
        self,
        points: np.ndarray | None,
        cam_pos: np.ndarray,
        flags: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if points is None or points.shape[0] == 0:
            empty_flags = np.zeros(0, dtype=np.uint8) if flags is not None else None
            return np.zeros((0, 3), dtype=np.float32), empty_flags
        if not self._nearby_only:
            return points, flags
        r = max(0.0, float(self._nearby_radius))
        if r <= 0.0:
            empty_flags = np.zeros(0, dtype=np.uint8) if flags is not None else None
            return np.zeros((0, 3), dtype=np.float32), empty_flags
        d2 = np.sum((points - cam_pos[None, :]) ** 2, axis=1)
        idx = np.where(d2 <= (r * r))[0]
        if idx.size == 0:
            empty_flags = np.zeros(0, dtype=np.uint8) if flags is not None else None
            return np.zeros((0, 3), dtype=np.float32), empty_flags
        if idx.size > self._max_points:
            shrink = (float(self._max_points) / float(idx.size)) ** (1.0 / 3.0)
            r_eff = max(0.05 * r, r * shrink)
            idx = np.where(d2 <= (r_eff * r_eff))[0]
            if idx.size > self._max_points:
                idx = idx[np.argsort(d2[idx])[:self._max_points]]
        return points[idx], None if flags is None else flags[idx]

    def _filter_cache_key_from_camera(self, cam_pos: np.ndarray, *, count: int, voxel_size: float) -> tuple:
        step = max(1e-4, float(voxel_size))
        cam_key = tuple(np.floor(cam_pos / step).astype(np.int64).tolist())
        return (
            bool(self._nearby_only),
            round(float(self._nearby_radius), 3),
            int(self._max_points),
            int(count),
            cam_key if self._nearby_only else None,
        )

    def _update_filtered_cache(self, cam_pos: np.ndarray) -> None:
        main_cam_pos = apply_transform(
            cam_pos.reshape(1, 3), self._map_from_training_world
        )[0]
        extra_cam_pos = apply_transform(
            cam_pos.reshape(1, 3), self._extra_map_from_training_world
        )[0]
        key = self._filter_cache_key_from_camera(
            main_cam_pos,
            count=self._num_occupied,
            voxel_size=self._voxel_size_world,
        )
        if self._filter_cache_key == key:
            pass
        else:
            points, flags = self._filter_nearby(
                self._occupied_points_world,
                main_cam_pos,
                self._occupied_overlap_flags,
            )
            self._cached_occ_points = points.astype(np.float32, copy=False)
            self._cached_occ_overlap_flags = np.asarray(flags, dtype=np.uint8)
            self._filter_cache_key = key

        extra_key = self._filter_cache_key_from_camera(
            extra_cam_pos,
            count=self._extra_num_occupied,
            voxel_size=self._extra_voxel_size_world,
        )
        if self._extra_filter_cache_key != extra_key:
            points, flags = self._filter_nearby(
                self._extra_occupied_points_world,
                extra_cam_pos,
                self._extra_occupied_overlap_flags,
            )
            self._extra_cached_occ_points = points.astype(np.float32, copy=False)
            self._extra_cached_occ_overlap_flags = np.asarray(flags, dtype=np.uint8)
            self._extra_filter_cache_key = extra_key

    @staticmethod
    def _solid_color(color: np.ndarray) -> np.ndarray:
        rgba = np.array(color, dtype=np.float32, copy=True)
        rgba[:3] = np.clip(rgba[:3], 0.0, 1.0)
        rgba[3] = np.clip(rgba[3], 0.0, 1.0)
        return rgba

    @staticmethod
    def _get_depth_mask() -> bool:
        return bool(np.asarray(gl.glGetBooleanv(gl.GL_DEPTH_WRITEMASK)).reshape(-1)[0])

    def _draw_instanced_cubes(
        self,
        view: View,
        centers_vbo: VBO,
        num_instances: int,
        color: np.ndarray,
        extra_params: dict[str, Any],
        *,
        colors_vbo: VBO | None = None,
        overlap_flags_vbo: VBO | None = None,
        occlusion_enabled: bool,
        voxel_size: float | None = None,
        training_world_from_map: np.ndarray | None = None,
        depth_write: bool = True,
    ) -> None:
        camera = view.camera
        cube_size = float(self._voxel_size_world if voxel_size is None else voxel_size) * float(self._voxel_size_scale)
        if cube_size <= 0.0:
            return
        model_tex_locations = extra_params["modelTextureLocations"]
        viewport_size = np.array(extra_params["viewportSize"], dtype="f4")
        view_matrix, projection = get_view_projection(view)
        fill_color = self._solid_color(color)
        map_to_training = (
            self._training_world_from_map
            if training_world_from_map is None
            else validate_rigid_transform(training_world_from_map)
        )

        prev_depth_test = gl.glIsEnabled(gl.GL_DEPTH_TEST)
        prev_cull_face = gl.glIsEnabled(gl.GL_CULL_FACE)
        prev_blend = gl.glIsEnabled(gl.GL_BLEND)
        prev_depth_mask = self._get_depth_mask()
        gl.glUseProgram(self._cube_shader_program)
        gl.glBindVertexArray(self._cube_vao_location)
        try:
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthMask(bool(depth_write))
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glDisable(gl.GL_CULL_FACE)
            self._cube_vertex_vbo.bind()
            gl.glEnableVertexAttribArray(0)
            gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
            gl.glVertexAttribDivisor(0, 0)

            self._cube_normal_vbo.bind()
            gl.glEnableVertexAttribArray(1)
            gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
            gl.glVertexAttribDivisor(1, 0)

            centers_vbo.bind()
            gl.glEnableVertexAttribArray(2)
            gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
            gl.glVertexAttribDivisor(2, 1)

            if colors_vbo is not None:
                colors_vbo.bind()
                gl.glEnableVertexAttribArray(3)
                gl.glVertexAttribPointer(3, 4, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
                gl.glVertexAttribDivisor(3, 1)

            if overlap_flags_vbo is not None:
                overlap_flags_vbo.bind()
                gl.glEnableVertexAttribArray(4)
                gl.glVertexAttribPointer(4, 1, gl.GL_UNSIGNED_BYTE, gl.GL_FALSE, 0, None)
                gl.glVertexAttribDivisor(4, 1)

            gl.glUniformMatrix4fv(self._cube_view_loc, 1, True, view_matrix)
            gl.glUniformMatrix4fv(self._cube_projection_loc, 1, True, projection)
            gl.glUniformMatrix4fv(
                self._cube_map_to_training_loc,
                1,
                True,
                np.asarray(map_to_training, dtype=np.float32),
            )
            gl.glUniform2fv(self._cube_viewport_size_loc, 1, viewport_size)
            gl.glUniform1f(self._cube_near_loc, camera.near_plane)
            gl.glUniform4fv(self._cube_color_loc, 1, fill_color)
            gl.glUniform1i(self._cube_use_instance_color_loc, int(colors_vbo is not None))
            gl.glUniform4fv(
                self._cube_overlap_color_loc,
                1,
                self._solid_color(self._overlap_color),
            )
            gl.glUniform1i(
                self._cube_use_overlap_flags_loc,
                int(overlap_flags_vbo is not None),
            )
            gl.glUniform1f(self._cube_size_loc, cube_size)
            gl.glUniform1i(self._cube_depth_sampler_loc, 0)
            gl.glUniform1i(self._cube_alpha_sampler_loc, 1)
            gl.glUniform1i(self._cube_occlusion_enabled_loc, int(occlusion_enabled))
            gl.glUniform1f(self._cube_occlusion_alpha_threshold_loc, float(self._occlusion_alpha_threshold))
            with bind_textures(model_tex_locations["depth"], model_tex_locations["alpha"]):
                gl.glDrawArraysInstanced(gl.GL_TRIANGLES, 0, 36, num_instances)
        finally:
            if overlap_flags_vbo is not None:
                gl.glDisableVertexAttribArray(4)
                overlap_flags_vbo.unbind()
            if colors_vbo is not None:
                gl.glDisableVertexAttribArray(3)
                colors_vbo.unbind()
            gl.glDisableVertexAttribArray(2)
            gl.glDisableVertexAttribArray(1)
            gl.glDisableVertexAttribArray(0)
            centers_vbo.unbind()
            self._cube_normal_vbo.unbind()
            self._cube_vertex_vbo.unbind()
            gl.glBindVertexArray(0)
            gl.glUseProgram(0)
            gl.glDepthMask(prev_depth_mask)
            restore_gl_bool_state(prev_depth_test, gl.GL_DEPTH_TEST)
            restore_gl_bool_state(prev_cull_face, gl.GL_CULL_FACE)
            restore_gl_bool_state(prev_blend, gl.GL_BLEND)

    def render(self, view: View, extra_params: dict[str, Any]) -> None:
        if not isinstance(view.camera, PerspectiveCamera):
            return
        self._ensure_loaded()
        self._ensure_extra_loaded()
        self._ensure_geometry()
        self._ensure_esdf_slice_geometry()
        if self._loaded_file_path:
            extra_params[OVERLAY_OCCUPANCY_FILE_KEY] = self._loaded_file_path
        cam_pos = self._camera_world_position(view)
        self._update_filtered_cache(cam_pos)
        self._update_probe_voxels()
        self._ensure_probe_geometry()
        occlusion_enabled = self._enable_scene_occlusion and not bool(extra_params.get("sceneHidden", False))

        probe_mode = bool(self._probe_voxel_enabled)
        draw_occupied = (
            not probe_mode
            and self._show_occupied
            and self._num_occupied > 0
            and self._cached_occ_points.shape[0] > 0
        )
        draw_extra_occupied = (
            not probe_mode
            and self._show_extra_occupied
            and self._extra_num_occupied > 0
            and self._extra_cached_occ_points.shape[0] > 0
        )
        draw_probe_occupied = (
            probe_mode
            and self._probe_occupied_vbo is not None
            and self._probe_occ_center.shape[0] > 0
        )
        draw_probe_extra_occupied = (
            probe_mode
            and self._probe_extra_occupied_vbo is not None
            and self._probe_extra_center.shape[0] > 0
        )
        draw_esdf_slice = (
            not probe_mode
            and self._show_esdf_slice
            and self._esdf_slice_display is not None
            and self._esdf_slice_display.num_voxels > 0
            and self._esdf_slice_center_vbo is not None
            and self._esdf_slice_color_vbo is not None
        )

        if draw_occupied or draw_extra_occupied or draw_probe_occupied or draw_probe_extra_occupied or draw_esdf_slice:
            gl.glClearDepth(1.0)
            gl.glClear(gl.GL_DEPTH_BUFFER_BIT)

        if draw_occupied:
            occupied_points = self._cached_occ_points
            self._occupied_near_vbo = update_vbo(self._occupied_near_vbo, occupied_points.astype(np.float32))
            overlap_flags_vbo = None
            if (
                self._highlight_overlap
                and self._num_overlap > 0
                and self._cached_occ_overlap_flags.shape[0] == occupied_points.shape[0]
            ):
                self._occupied_overlap_vbo = update_vbo(
                    self._occupied_overlap_vbo,
                    self._cached_occ_overlap_flags,
                )
                overlap_flags_vbo = self._occupied_overlap_vbo
            self._draw_instanced_cubes(
                view,
                self._occupied_near_vbo,
                int(occupied_points.shape[0]),
                self._occupied_color,
                extra_params,
                overlap_flags_vbo=overlap_flags_vbo,
                occlusion_enabled=occlusion_enabled,
            )
            extra_params[OVERLAY_DEPTH_OCCLUDER_KEY] = True

        if draw_extra_occupied:
            extra_points = self._extra_cached_occ_points
            self._extra_occupied_near_vbo = update_vbo(
                self._extra_occupied_near_vbo,
                extra_points.astype(np.float32),
            )
            overlap_flags_vbo = None
            if (
                self._highlight_overlap
                and self._num_overlap > 0
                and self._extra_cached_occ_overlap_flags.shape[0] == extra_points.shape[0]
            ):
                self._extra_occupied_overlap_vbo = update_vbo(
                    self._extra_occupied_overlap_vbo,
                    self._extra_cached_occ_overlap_flags,
                )
                overlap_flags_vbo = self._extra_occupied_overlap_vbo
            self._draw_instanced_cubes(
                view,
                self._extra_occupied_near_vbo,
                int(extra_points.shape[0]),
                self._extra_occupied_color,
                extra_params,
                overlap_flags_vbo=overlap_flags_vbo,
                occlusion_enabled=occlusion_enabled,
                voxel_size=self._extra_voxel_size_world,
                training_world_from_map=self._extra_training_world_from_map,
            )
            extra_params[OVERLAY_DEPTH_OCCLUDER_KEY] = True

        if draw_probe_occupied:
            self._draw_instanced_cubes(
                view,
                self._probe_occupied_vbo,
                int(self._probe_occ_center.shape[0]),
                self._occupied_color,
                extra_params,
                occlusion_enabled=occlusion_enabled,
                voxel_size=self._voxel_size_world,
                depth_write=False,
                training_world_from_map=self._training_world_from_map,
            )
            extra_params[OVERLAY_DEPTH_OCCLUDER_KEY] = True

        if draw_probe_extra_occupied:
            self._draw_instanced_cubes(
                view,
                self._probe_extra_occupied_vbo,
                int(self._probe_extra_center.shape[0]),
                self._extra_occupied_color,
                extra_params,
                occlusion_enabled=occlusion_enabled,
                voxel_size=self._extra_voxel_size_world,
                depth_write=False,
                training_world_from_map=self._extra_training_world_from_map,
            )
            extra_params[OVERLAY_DEPTH_OCCLUDER_KEY] = True

        if draw_esdf_slice:
            self._draw_instanced_cubes(
                view,
                self._esdf_slice_center_vbo,
                int(self._esdf_slice_display.num_voxels),
                np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                extra_params,
                colors_vbo=self._esdf_slice_color_vbo,
                occlusion_enabled=occlusion_enabled,
                training_world_from_map=self._training_world_from_map,
            )
            extra_params[OVERLAY_DEPTH_OCCLUDER_KEY] = True
