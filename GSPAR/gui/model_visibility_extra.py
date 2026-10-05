"""Overlay module for temporarily hiding the rendered 3DGS image."""

from __future__ import annotations

from typing import Any

import OpenGL.GL as gl
from imgui_bundle import imgui
from OpenGL.GL.shaders import compileProgram, compileShader

from Datasets.utils import View
from ICGui.ModelViewer.Overlays import BaseOverlay

from GSPAR.gui.overlay_common import restore_gl_bool_state


class SceneVisibilityExtra(BaseOverlay):
    """Renders a fullscreen overlay to hide the model output when desired."""

    def __init__(self):
        super().__init__()
        self._hide_model_output = False
        self._overlay_color = (0.0, 0.0, 0.0, 1.0)
        self._shader_program = self._build_shader()
        self._vao_location = gl.glGenVertexArrays(1)

    @property
    def name(self) -> str:
        return 'Scene Controls'

    @staticmethod
    def _build_shader() -> int:
        vert = """
        #version 330
        const vec2 positions[3] = vec2[](
            vec2(-1.0, -1.0),
            vec2( 3.0, -1.0),
            vec2(-1.0,  3.0)
        );
        void main() {
            gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
        }
        """
        frag = """
        #version 330
        uniform vec4 overlayColor;
        out vec4 fragColor;
        void main() {
            fragColor = overlayColor;
        }
        """
        return compileProgram(
            compileShader(vert, gl.GL_VERTEX_SHADER),
            compileShader(frag, gl.GL_FRAGMENT_SHADER),
        )

    def render_options(self):
        _, self._hide_model_output = imgui.checkbox(
            f'Hide 3DGS Result##{self.name}',
            self._hide_model_output,
        )
        imgui.text('When enabled, only overlays such as voxels/path remain visible.')

    def render(self, view: View, extra_params: dict[str, Any]):
        del view
        extra_params['sceneHidden'] = bool(self._hide_model_output)
        if not self._hide_model_output:
            return

        prev_blend = gl.glIsEnabled(gl.GL_BLEND)
        prev_depth_test = gl.glIsEnabled(gl.GL_DEPTH_TEST)
        prev_cull_face = gl.glIsEnabled(gl.GL_CULL_FACE)
        try:
            gl.glDisable(gl.GL_BLEND)
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glDisable(gl.GL_CULL_FACE)
            gl.glUseProgram(self._shader_program)
            gl.glBindVertexArray(self._vao_location)
            color_location = gl.glGetUniformLocation(self._shader_program, 'overlayColor')
            gl.glUniform4f(color_location, *self._overlay_color)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        finally:
            gl.glBindVertexArray(0)
            gl.glUseProgram(0)
            restore_gl_bool_state(prev_blend, gl.GL_BLEND)
            restore_gl_bool_state(prev_depth_test, gl.GL_DEPTH_TEST)
            restore_gl_bool_state(prev_cull_face, gl.GL_CULL_FACE)
