"""Renderer for HTGS with an optional environment background model."""

import torch

import Framework
from Cameras.Perspective import PerspectiveCamera
from Datasets.Base import BaseDataset
from Datasets.utils import View
from Logging import Logger
from Methods.Base.Model import BaseModel
from Methods.Base.Renderer import BaseRenderer
from Methods.HTGS_egsr.HTGSCudaBackend_egsr import HTGSRasterizer
from Methods.HTGS_egsr.Model import HTGSModel


@Framework.Configurable.configure(
    BLEND_MODE=0,
    K=16,
    SCALE_MODIFIER=1.0,
    DISABLE_SH0=False,
    DISABLE_SH1=False,
    DISABLE_SH2=False,
    DISABLE_SH3=False,
    USE_MEDIAN_DEPTH=False,
    FORCE_OPTIMIZED_INFERENCE=False,
    DISABLE_BG_MODEL_INFERENCE=False,
)
class HTGSRenderer(BaseRenderer):
    """Renders HTGS foreground Gaussians and composites an optional background."""

    def __init__(self, model: BaseModel) -> None:
        super().__init__(model, [HTGSModel])
        if not Framework.config.GLOBAL.GPU_INDICES:
            raise Framework.RendererError('renderer not implemented in CPU mode')
        if len(Framework.config.GLOBAL.GPU_INDICES) > 1:
            Logger.log_warning(
                f'renderer not implemented in multi-GPU mode: using GPU {Framework.config.GLOBAL.GPU_INDICES[0]}'
            )
        self.rasterizer = HTGSRasterizer()
        if self.BLEND_MODE not in [0, 1, 2, 3]:
            raise Framework.RendererError('Invalid blend mode')
        if self.BLEND_MODE < 2 and self.K not in [1, 2, 4, 8, 16, 32]:
            Logger.log_warning(f'unsupported K value for selected blend mode may lead to undefined behavior: {self.K}')

    def render_image(self, view: View, to_chw: bool = False, benchmark: bool = False) -> dict[str, torch.Tensor]:
        """Renders an image for a View from the current dataset API."""
        if benchmark or self.FORCE_OPTIMIZED_INFERENCE:
            return self.render_image_benchmark(view, to_chw=to_chw or benchmark)
        if self.model.training:
            raise Framework.RendererError('please directly call render_image_training() instead of render_image() during training')
        return self.render_image_inference(view, to_chw)

    def render_image_training(
        self,
        view: View,
        update_densification_info: bool,
        use_distance_scaling: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Renders RGB and alpha for optimization."""
        self._validate_view(view)
        rgb, alpha = self.rasterizer(
            positions=self.model.gaussians.get_positions,
            scales=self.model.gaussians.get_scales_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_scales,
            rotations=self.model.gaussians.get_rotations,
            opacities=self.model.gaussians.get_opacities_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_opacities,
            sh_0=self.model.gaussians.get_sh_0,
            sh_rest=self.model.gaussians.get_sh_rest,
            densification_info=self.model.gaussians.get_densification_info if update_densification_info else torch.empty(0),
            camera=view,
            mode=self.BLEND_MODE,
            K=self.K,
            active_sh_bases=self.model.gaussians.active_sh_bases,
            scale_modifier=1.0,
            use_distance_scaling=use_distance_scaling and update_densification_info,
        )
        if self.model.background_model is None:
            rgb = rgb + (1.0 - alpha) * view.camera.background_color[:, None, None]
        else:
            background = self.model.background_model(view).permute(2, 0, 1).to(rgb.device)
            rgb = rgb + (1.0 - alpha) * background
        return rgb, alpha

    @torch.no_grad()
    def render_image_inference(self, view: View, to_chw: bool) -> dict[str, torch.Tensor]:
        """Renders RGB, depth, and alpha for inference."""
        self._validate_view(view)
        sh_0 = self.model.gaussians.get_sh_0
        if self.DISABLE_SH0:
            sh_0 = torch.zeros_like(sh_0)
        sh_rest = self.model.gaussians.get_sh_rest
        if self.DISABLE_SH1 or self.DISABLE_SH2 or self.DISABLE_SH3:
            sh_rest = sh_rest.clone()
        if self.DISABLE_SH1:
            sh_rest[:, 0:3].zero_()
        if self.DISABLE_SH2:
            sh_rest[:, 3:8].zero_()
        if self.DISABLE_SH3:
            sh_rest[:, 8:15].zero_()

        rgb, depth, alpha = self.rasterizer.render(
            positions=self.model.gaussians.get_positions,
            scales=self.model.gaussians.get_scales_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_scales,
            rotations=self.model.gaussians.get_rotations,
            opacities=self.model.gaussians.get_opacities_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_opacities,
            sh_0=sh_0,
            sh_rest=sh_rest,
            camera=view,
            mode=self.BLEND_MODE,
            K=self.K,
            active_sh_bases=self.model.gaussians.active_sh_bases,
            scale_modifier=self.SCALE_MODIFIER,
            to_chw=to_chw,
            use_median_depth=self.USE_MEDIAN_DEPTH,
        )
        if self.model.background_model is None or self.DISABLE_BG_MODEL_INFERENCE:
            background = view.camera.background_color[:, None, None] if to_chw else view.camera.background_color
        else:
            background = self.model.background_model(view).to(rgb.device)
            if to_chw:
                background = background.permute(2, 0, 1)
        rgb = rgb + (1.0 - alpha) * background
        return {'rgb': rgb, 'depth': depth, 'alpha': alpha}

    @torch.inference_mode()
    def render_image_benchmark(self, view: View, to_chw: bool) -> dict[str, torch.Tensor]:
        """Uses the optimized RGB-only inference path."""
        self._validate_view(view)
        rgb = self.rasterizer.benchmark(
            positions=self.model.gaussians.get_positions,
            scales=self.model.gaussians.get_scales_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_scales,
            rotations=self.model.gaussians.get_rotations,
            opacities=self.model.gaussians.get_opacities_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_opacities,
            sh_0=self.model.gaussians.get_sh_0,
            sh_rest=self.model.gaussians.get_sh_rest,
            camera=view,
            mode=self.BLEND_MODE,
            K=self.K,
            active_sh_bases=self.model.gaussians.active_sh_bases,
            scale_modifier=self.SCALE_MODIFIER,
            to_chw=to_chw,
        )
        return {'rgb': rgb}

    def compute_max_weights(self, dataset: BaseDataset, threshold: float) -> torch.Tensor:
        """Computes the maximum blending weight of every foreground Gaussian."""
        positions = self.model.gaussians.get_positions
        scales = self.model.gaussians.get_scales_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_scales
        opacities = self.model.gaussians.get_opacities_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_opacities
        rotations = self.model.gaussians.get_rotations
        max_weights = torch.zeros(opacities.shape[0], device=opacities.device, dtype=opacities.dtype)
        for view in dataset:
            self._validate_view(view)
            self.rasterizer.update_max_weights(
                max_weights=max_weights,
                positions=positions,
                scales=scales,
                rotations=rotations,
                opacities=opacities,
                camera=view,
                mode=self.BLEND_MODE,
                K=self.K,
                active_sh_bases=0,
                scale_modifier=1.0,
                weight_threshold=threshold,
            )
        return max_weights

    @staticmethod
    def _validate_view(view: View) -> None:
        if not isinstance(view.camera, PerspectiveCamera):
            raise Framework.RendererError('HTGS_egsr currently only supports perspective cameras')
