# -- coding: utf-8 --

"""HTGS/Model.py: Implementation of the model for the HTGS method."""

import torch, math
from typing import List, Optional, Tuple
import Framework
from Cameras.Perspective import PerspectiveCamera
from Datasets.Base import BaseDataset
from Datasets.utils import BasicPointCloud, View
from Cameras.utils import quaternion_to_rotation_matrix
from Cameras.Base import BaseCamera
from Logging import Logger
from Methods.Base.Model import BaseModel
from Methods.GaussianSplatting.utils import rgb_to_sh0, sh0_to_rgb
from Optim.adam_utils import replace_param_group_data, prune_param_groups, extend_param_groups
from Optim.knn_utils import compute_root_mean_squared_knn_distances
from Optim.lr_utils import LRDecayPolicy
import Thirdparty.TinyCudaNN as tcnn
from CudaUtils.MortonEncoding import morton_encode
from Methods.HTGS_egsr.HTGSCudaBackend_egsr import update_3d_filter
from Methods.HTGS_egsr.HTGSCudaBackend_egsr import HTGSRasterizer
from Methods.HTGS_egsr.utils.gaussian_env_utils import fibonacci_sphere,fibonacci_sphere_with_latitude_scales,compute_max_visible_hits as _vis_hits,propagate_by_overlay as _propagate_overlay,knn_fix_opacity, interp_HTGS as interp


class NeuralEnvironmentMap(torch.nn.Module):
    """Environment map using spherical harmonics encoding and an MLP."""

    def __init__(self) -> None:
        """Initialize submodules."""
        super().__init__()
        self.net_with_encoding = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=3,
            encoding_config={
                'otype': 'SphericalHarmonics',
                'degree': 4
            },
            network_config={
                'otype': 'FullyFusedMLP',
                'activation': 'ReLU',
                'output_activation': 'Sigmoid',
                'n_neurons': 128,
                'n_hidden_layers': 4,
            },
            seed=Framework.config.GLOBAL.RANDOM_SEED
        )

    def get_optimizer_param_groups(self, max_iterations: int) -> tuple[list[dict], list[LRDecayPolicy]]:
        """Returns the parameter groups for the optimizer."""
        param_groups = [{'params': self.net_with_encoding.parameters(), 'lr': 1.0}]
        schedulers = [LRDecayPolicy(
            lr_init=1.0e-2,
            lr_final=1.0e-2,
            lr_delay_steps=0,
            lr_delay_mult=1.0,
            max_steps=max_iterations)
        ]
        return param_groups, schedulers

    def increase_used_sh_degree(self) -> None:
        """Increases the used SH degree."""
        pass

    def forward(self, view: View) -> torch.Tensor:
        """Returns environment map values for the given camera."""
        view_dirs_all = view.cam_to_world(view.camera.compute_local_ray_directions(), is_point=False)
        view_dirs_all = torch.nn.functional.normalize(view_dirs_all, p=2, dim=-1)
        view_dirs_all.mul_(0.5).add_(0.5)  
        out = self.net_with_encoding(view_dirs_all).reshape(view.camera.height, view.camera.width, 3)
        return out


class SHEnvironmentMap(torch.nn.Module):
    """Environment map using spherical harmonics encoding."""

    def __init__(self, sh_degree: int, pretrained: bool, disable_coarse_to_fine: bool) -> None:
        """Initialize submodules."""
        super().__init__()
        self.active_sh_degree = sh_degree if pretrained or disable_coarse_to_fine else 0
        self.active_sh_bases = (self.active_sh_degree + 1) ** 2
        self.max_sh_degree = sh_degree
        self.encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                'otype': 'SphericalHarmonics',
                'degree': sh_degree + 1,  # +1 because tcnn counts degrees differently
            },
            dtype=torch.float32,
            seed=Framework.config.GLOBAL.RANDOM_SEED
        )
        self.register_parameter('coefficients', torch.nn.Parameter(torch.zeros((1, self.encoding.n_output_dims, 3), dtype=torch.float32)))
        # TODO
        #   - compare different output activations
        #   - learning rate tuning
        #   - fused implementation
        #   - binary cross entropy loss tuning
        self.output_activation = torch.nn.Sigmoid()
        

    def get_optimizer_param_groups(self, max_iterations: int) -> tuple[list[dict], list[LRDecayPolicy]]:
        """Returns the parameter groups for the optimizer."""
        param_groups = [{'params': self.parameters(), 'lr': 1.0}]
        schedulers = [LRDecayPolicy(
            lr_init=1.0e-1,
            lr_final=1.0e-1,
            lr_delay_steps=0,
            lr_delay_mult=1.0,
            max_steps=max_iterations)
        ]
        return param_groups, schedulers

    def increase_used_sh_degree(self) -> None:
        """Increases the used SH degree."""
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            self.active_sh_bases = (self.active_sh_degree + 1) ** 2

    @torch.compile
    def eval_sh(self, view_directions: torch.Tensor) -> torch.Tensor:
        """Evaluates the SH coefficients for the given view directions."""
        sh_values = self.encoding.forward(view_directions)[:, :self.active_sh_bases, None]
        coefficients = self.coefficients[:, :self.active_sh_bases]
        return self.output_activation(sh_values.mul(coefficients).sum(dim=-2))

    def forward(self, view: View) -> torch.Tensor:
        """Returns environment map values for the given camera."""
        view_dirs_all = view.cam_to_world(view.camera.compute_local_ray_directions(), is_point=False)
        view_dirs_all = torch.nn.functional.normalize(view_dirs_all, p=2, dim=-1)
        view_dirs_all.mul_(0.5).add_(0.5)  
        out = self.eval_sh(view_dirs_all).reshape(view.camera.height, view.camera.width, 3)
        return out

class EnvironmentMap(torch.nn.Module):
    """Environment map."""

    def __init__(self) -> None:
        """Initialize submodules."""
        super().__init__()
        self.register_parameter('texture', torch.nn.Parameter(torch.full((1, 3, 1000, 2000), fill_value=0.5, dtype=torch.float32)))

    def cart2tex(self, view_directions: torch.Tensor) -> torch.Tensor:
        """Converts cartesian coordinates to texture coordinates via spherical coordinates."""
        # convert to spherical coordinates
        theta = torch.atan2(view_directions[..., 1:2], view_directions[..., 0:1])
        phi = torch.acos(view_directions[..., 2:3])
        x = (theta + torch.pi) / (2.0 * torch.pi)
        y = phi / torch.pi
        return torch.cat([x, y], dim=-1)[None]

    def get_optimizer_param_groups(self, max_iterations: int) -> tuple[list[dict], list[LRDecayPolicy]]:
        """Returns the parameter groups for the optimizer."""
        param_groups = [{'params': self.parameters(), 'lr': 1.0}]
        schedulers = [LRDecayPolicy(
            lr_init=1.0e-3,
            lr_final=1.0e-3 / 30.0,
            lr_delay_steps=500,
            lr_delay_mult=100.0,
            max_steps=max_iterations)
        ]
        return param_groups, schedulers

    def increase_used_sh_degree(self) -> None:
        """Increases the used SH degree."""
        pass

    def forward(self, view: View) -> torch.Tensor:
        """Returns environment map values for the given view directions."""
        view_directions = view.cam_to_world(view.camera.compute_local_ray_directions(), is_point=False)
        view_directions = torch.nn.functional.normalize(view_directions, p=2, dim=-1)
        view_directions = view_directions.reshape(view.camera.height, view.camera.width, 3)
        uvs = self.cart2tex(view_directions) * 2.0 - 1.0
        colors = torch.nn.functional.grid_sample(
            input=self.texture,  
            grid=uvs,  
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        ).squeeze()  
        out = colors.permute(1, 2, 0).sigmoid()
        return out




class MultiSphereEnvironmentMap(torch.nn.Module):
    """Multi-Sphere Image (MSI) background model."""

    def __init__(self) -> None:
        """Initialize submodules."""
        super().__init__()
        self.n_layers = 8
        self.register_parameter('texture', torch.nn.Parameter(torch.full((self.n_layers, 3, 1000, 2000), fill_value=0.5, dtype=torch.float32)))
        self.register_parameter('alpha', torch.nn.Parameter(torch.full((self.n_layers, 1, 1000, 2000), 10.0, dtype=torch.float32)))
 

    def cart2tex(self, view_directions: torch.Tensor) -> torch.Tensor:
        """Converts cartesian coordinates to texture coordinates via spherical coordinates."""
        theta = torch.atan2(view_directions[..., 1:2], view_directions[..., 0:1])
        phi = torch.acos(view_directions[..., 2:3])
        x = (theta + torch.pi) / (2.0 * torch.pi)
        y = phi / torch.pi
        return torch.cat([x, y], dim=-1)[None]

    def get_optimizer_param_groups(self, max_iterations: int) -> tuple[list[dict], list[LRDecayPolicy]]:
        """Returns the parameter groups for the optimizer."""
        param_groups = [{'params': self.parameters(), 'lr': 1.0}]
        schedulers = [LRDecayPolicy(
            lr_init=1.0e-3,
            lr_final=1.0e-3 / 30.0,
            lr_delay_steps=500,
            lr_delay_mult=100.0,
            max_steps=max_iterations)
        ]
        return param_groups, schedulers

    def increase_used_sh_degree(self) -> None:
        """Increases the used SH degree."""
        pass

    def forward(self, view: View) -> torch.Tensor:
        """Compute background color from MSI layers for given view directions."""
        view_directions = view.cam_to_world(view.camera.compute_local_ray_directions(), is_point=False)
        view_directions = torch.nn.functional.normalize(view_directions, p=2, dim=-1)
        view_directions = view_directions.reshape(view.camera.height, view.camera.width, 3)
        uvs = self.cart2tex(view_directions) * 2.0 - 1.0
        uvs = uvs.expand(self.n_layers, -1, -1, -1) 
        # Sample color and opacity from each spherical layer
        rgb_layers = torch.nn.functional.grid_sample(
            input=self.texture, grid=uvs,
            mode='bilinear', padding_mode='border',
            align_corners=False).squeeze(0)
        alpha_layers = torch.nn.functional.grid_sample(
            input=self.alpha, grid=uvs,
            mode='bilinear', padding_mode='border',
            align_corners=False).squeeze(0).sigmoid()
        rgb_layers = rgb_layers.sigmoid() 
        
        # Front-to-back compositing across layers
        H, W = view.camera.height, view.camera.width
        T = torch.ones((1, H, W), dtype=rgb_layers.dtype, device=rgb_layers.device)
        C = torch.zeros((3, H, W), dtype=rgb_layers.dtype, device=rgb_layers.device)
         
        for l in range(self.n_layers):
            w = alpha_layers[l]                # Per-layer opacity
            C = C + T * w * rgb_layers[l]      # Accumulate color
            T = T * (1.0 - w)                  # Update transmittance

        return C.permute(1, 2, 0) 




class LayeredGaussianEnvMap(torch.nn.Module):
    """Unified N-layer environment Gaussians (multi-shell sky dome)."""

    def __init__(
        self,
        base_gaussians: int,
        sh_degree: int,
        pretrained: bool,
        disable_coarse_to_fine: bool,
        num_layers: int,
        beta_interval: float,
        keep_ratio: float,
        min_scale_ratio: float,
        max_scale_ratio: float,
        auto_config:bool,
        half_ball:bool,
        knn_color: int
    ):
        super().__init__()
        self.rasterizer = HTGSRasterizer()
        # SH setup
        self.max_sh_degree = sh_degree
        self.active_sh_degree = sh_degree if (pretrained or disable_coarse_to_fine) else 0
        self.active_sh_bases = (self.active_sh_degree + 1) ** 2
        # Store config
        self.base_gaussians = base_gaussians
        self.sh_degree = sh_degree
        self.pretrained = pretrained
        self.disable_c2f = disable_coarse_to_fine
        self.num_layers = num_layers
        self.beta_interval = beta_interval
        self.keep_ratio = keep_ratio
        self.min_scale_ratio = min_scale_ratio
        self.max_scale_ratio = max_scale_ratio
        self.overlay_indices = None
        self.auto_config = auto_config
        self.half_ball = half_ball
        self.knn_color = knn_color


    @torch.no_grad()
    def build(self, scene_center: torch.Tensor, scene_radius: float,sky_complexity:float):
        """Initialize multi-shell background points and per-shell scales."""
        self.scene_center=scene_center
        self.scene_radius=scene_radius
        # Concentric shell radii: r = beta * scene_radius
        betas = [1.0 + i * self.beta_interval for i in range(self.num_layers)]
        print(f"scene info:\nscene_radius={scene_radius}\nsky_complexity={sky_complexity}")
        # Scene-adaptive config (capacity & scale vs. sky complexity/radius)
        if self.auto_config:
            # N = interp(sky_complexity, 0.201, 0.539, 9500, 26000)
            # self.base_gaussians  = int(N * (scene_radius / 24.18)*self.num_layers) 
            # self.min_scale_ratio = interp(sky_complexity, 0.201, 0.539, 0.01, 0.006)
            # self.max_scale_ratio = interp(sky_complexity, 0.201, 0.539, 0.08, 0.03) 
            N = interp(sky_complexity, 0.201, 0.539, 9500, 26000, k=6.0)
            radius_factor = (scene_radius / 24.18)**0.7
            self.base_gaussians = int(N * (scene_radius / 24.18)**0.7  * self.num_layers)
            self.min_scale_ratio = interp(sky_complexity, 0.201, 0.539, 0.01, 0.006, k=7.0)
            self.max_scale_ratio = interp(sky_complexity, 0.201, 0.539, 0.08, 0.03, k=7.0)
            
        pts_per_layer = self.base_gaussians // len(betas)
        all_means, all_scales = [], []

        if self.half_ball:
            print(f"auto config:\nbase_gaussians={self.base_gaussians}\nmin_scale_ratio={self.min_scale_ratio}\nmax_scale_ratio={self.max_scale_ratio}")
            for beta in betas:
                shell_radius = scene_radius * beta
                min_scale = shell_radius * self.min_scale_ratio
                max_scale = shell_radius * self.max_scale_ratio
                points, scales = fibonacci_sphere_with_latitude_scales(
                    pts_per_layer,
                    shell_radius=shell_radius,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    device="cuda",
                )
                all_means.append(points)
                all_scales.append(scales)
        else:
            self.scale_ratio=(self.max_scale_ratio+self.min_scale_ratio)/2
            print("Without using a hemisphere, all Gaussian sizes are identical. Using the average of max_scale_ratio and min_scale_ratio.")
            print(f"auto config:\nbase_gaussians={self.base_gaussians}\nscale_ratio={self.scale_ratio}")
            for beta in betas:
                shell_radius = scene_radius * beta
                points, scales = fibonacci_sphere(
                    pts_per_layer,
                    shell_radius=shell_radius,
                    scale=shell_radius *(self.max_scale_ratio+self.min_scale_ratio)/2,
                    device="cuda",
                )
                all_means.append(points)
                all_scales.append(scales)

        # merge layers
        points = torch.cat(all_means, dim=0)
        scales = torch.cat(all_scales, dim=0)

        # Translate sky dome to scene center
        points = points + scene_center.to(device=points.device, dtype=points.dtype).view(1, 3)
        rotations = torch.tensor([0.0, 0.0, 0.0, 1.0], device="cuda").expand(points.shape[0], 4).clone()

        # buffers
        self.register_buffer("sky_rotations", rotations)
        self.register_buffer("sky_means", points)
        self.register_buffer("sky_scales", scales)

        # SH parameters
        rgbs = torch.full_like(points, 0.5)
        n = points.shape[0]
        sh_all = torch.zeros((n, (self.max_sh_degree + 1) ** 2, 3), dtype=torch.float32, device=points.device)
        sh_all[:, 0] = rgb_to_sh0(rgbs)
        self.sky_sh_0 = torch.nn.Parameter(sh_all[:, 0:1].contiguous())
        self.sky_sh_rest = torch.nn.Parameter(sh_all[:, 1:].contiguous())

        # opacity in logit space
        initial_opacity = 0.1
        self.sky_opacity = torch.nn.Parameter(
            torch.logit(torch.full((n, 1), initial_opacity, device=points.device), eps=1e-6)
        )


    def get_opacities(self) -> torch.Tensor:
        """Matches main model's opacity activation."""
        return torch.sigmoid(self.sky_opacity).view(-1, 1)

    def forward(self, view: View) -> torch.Tensor:
        """Render envmap, returns (H, W, 3)."""
        rgb, alpha = self.rasterizer(
            positions=self.sky_means,
            scales=self.sky_scales,
            rotations=self.sky_rotations,
            opacities=self.get_opacities(),
            sh_0=self.sky_sh_0,
            sh_rest=self.sky_sh_rest,
            densification_info=torch.empty(0, device=self.sky_means.device),
            camera=view,
            mode=0,
            K=16,
            active_sh_bases=self.active_sh_bases,
            scale_modifier=1.0,
            use_distance_scaling=False,
        )
        return rgb.permute(1, 2, 0).clamp(0.0, 1.0)

    def increase_used_sh_degree(self) -> None:
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            self.active_sh_bases = (self.active_sh_degree + 1) ** 2

    def get_optimizer_param_groups(self, max_iterations: int):
        """Return param_groups and schedulers for the envmap only."""
        param_groups = [{"params": [self.sky_sh_0, self.sky_opacity], "lr": 5e-2}]
        schedulers = [
            LRDecayPolicy(
                lr_init=5e-2,
                lr_final=1e-3 / 30.0,
                lr_delay_steps=500,
                lr_delay_mult=100.0,
                max_steps=max_iterations,
            )
        ]
        return param_groups, schedulers

    @torch.no_grad()
    def compute_max_visible_hits(
        self,
        camera,
        alpha_2d: torch.Tensor,
        pixel_pad: int = 1,
        tau: float = 1e-3,
    ) -> torch.Tensor:
        """
        Identify background (sky) Gaussians located in unsupervised or unobserved regions.
        """
        return _vis_hits(self.sky_means, camera, alpha_2d, pixel_pad, tau)

    def propagation(self):
        """
        Propagate sky Gaussian colors into unsupervised regions.Implements the color diffusion step to fill missing background areas.
        """
        if self.overlay_indices is not None and self.overlay_indices.numel() > 0:
            self.sky_sh_0.data = _propagate_overlay(
                self.sky_means,
                self.sky_sh_0.data,
                self.overlay_indices,
                knn_k=2048,
                max_layer=2,
                paint_knn=self.knn_color,
                )

    @torch.no_grad()
    def correct_opacity(self, opacity_thresh: float = 1e-1, knn_k: int = 8) -> int:
        """
        Fix Gaussians with too low opacity (same threshold as initialization).
        'Good' points have stable opacity; 'bad' ones are corrected using nearby good points.
        """
        opacity = self.get_opacities()
        mask_low = opacity <= opacity_thresh
        mask_good = opacity > (opacity_thresh * 2)
        num_fixed = int(mask_low.sum().item())
        if num_fixed == 0 or mask_good.sum() == 0:
            return 
            
        new_opacity = knn_fix_opacity(
            self.sky_means,
            opacity,
            mask_low,
            mask_good,
            knn_k=knn_k,
        )

        bad_idx = torch.where(mask_low)[0]
        fixed = new_opacity[bad_idx].view(-1)
        self.sky_opacity.data[bad_idx, 0] = torch.log(fixed / (1 - fixed)).to(self.sky_opacity.device)



class Gaussians(torch.nn.Module):
    """Stores a set of points with 3D Gaussian extent."""

    GOF_DENSIFICATION_GRAD = True
    GOF_DENSIFICATION_CLONE = True

    def __init__(self, sh_degree: int, pretrained: bool) -> None:
        super().__init__()
        self.active_sh_degree = sh_degree if pretrained else 0
        self.active_sh_bases = (self.active_sh_degree + 1) ** 2
        self.max_sh_degree = sh_degree
        self.register_parameter('_positions', None)
        self.register_parameter('_sh_0', None)
        self.register_parameter('_sh_rest', None)
        self.register_parameter('_scales', None)
        self.register_parameter('_rotations', None)
        self.register_parameter('_opacities', None)
        self.densification_info = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.training_cameras_extent = 1.0
        self.filter_3D = torch.empty(0)
        self.use_3d_filter = False
        self.distance2filter = 0
        # activation functions
        self.scale_activation = torch.nn.Identity() if pretrained else torch.exp
        self.inverse_scale_activation = torch.nn.Identity() if pretrained else torch.log
        self.opacity_activation = torch.nn.Identity() if pretrained else torch.sigmoid
        self.inverse_opacity_activation = torch.nn.Identity() if pretrained else torch.special.logit
        self.rotation_activation = torch.nn.Identity() if pretrained else torch.nn.functional.normalize

    @property
    def get_scales(self) -> torch.Tensor:
        """Returns the Gaussians' scales."""
        return self.scale_activation(self._scales)

    @property
    def get_rotations(self) -> torch.Tensor:
        """Returns the Gaussians' rotations as quaternions."""
        return self.rotation_activation(self._rotations)

    @property
    def get_positions(self) -> torch.Tensor:
        """Returns the Gaussians' means."""
        return self._positions

    @property
    def get_sh_0(self) -> torch.Tensor:
        """Returns the Gaussians' 0-th degree SH features."""
        return self._sh_0

    @property
    def get_sh_rest(self) -> torch.Tensor:
        """Returns the Gaussians' SH features beyond the 0-th degree."""
        return self._sh_rest

    @property
    def get_opacities(self) -> torch.Tensor:
        """Returns the Gaussians' opacities."""
        return self.opacity_activation(self._opacities)

    @property
    def get_opacities_with_3D_filter(self) -> torch.Tensor:
        """Returns the Gaussians' opacities with the 3D filter applied."""
        # apply 3D filter
        scales = self.get_scales
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        scales_after_square = scales_square + torch.square(self.filter_3D)
        det2 = scales_after_square.prod(dim=1)
        coef = torch.sqrt(det1 / det2)
        return self.get_opacities * coef[..., None]

    @property
    def get_scales_with_3D_filter(self) -> torch.Tensor:
        """Returns the Gaussians' scales with the 3D filter applied."""
        scales = self.get_scales
        # apply 3D filter
        scales = torch.square(scales) + torch.square(self.filter_3D)
        scales = torch.sqrt(scales)
        return scales

    @property
    def get_densification_info(self) -> torch.Tensor:
        """Returns the current densification info buffers."""
        return self.densification_info

    def setup_3d_filter(self, dataset: 'BaseDataset', dilation: float = 0.2) -> None:
        """Sets up a 3D filter (see https://arxiv.org/abs/2311.16493)."""
        self.use_3d_filter = True
        max_focal = 1.0e-12
        for view in dataset:
            if not isinstance(view.camera, PerspectiveCamera):
                raise Framework.ModelError('3d filter only supports perspective cameras')
            if view.camera.distortion is not None:
                Logger.log_warning('3d filter ignores all distortion parameters')
            max_focal = max(max_focal, max(view.camera.focal_x, view.camera.focal_y))
        # assume max_focal is focal length of the highest resolution camera
        self.distance2filter = dilation ** 0.5 / max_focal
        self.compute_3d_filter(dataset)

    def compute_3d_filter(self, dataset: 'BaseDataset', clipping_tolerance: float = 0.15) -> None:
        """Computes the 3D filter."""
        positions = self.get_positions.contiguous()
        filter_3d = torch.full((positions.shape[0], 1), fill_value=torch.finfo(torch.float32).max, device=positions.device, dtype=torch.float32)
        visibility_mask = torch.zeros((positions.shape[0], 1), device=positions.device, dtype=torch.bool)
        for view in dataset:
            if not isinstance(view.camera, PerspectiveCamera):
                raise Framework.ModelError('3d filter only supports perspective cameras')
            if view.camera.distortion is not None:
                Logger.log_warning('3d filter ignores all distortion parameters')
            update_3d_filter(
                positions,
                view.w2c,
                filter_3d,
                visibility_mask,
                view.camera.width,
                view.camera.height,
                view.camera.focal_x,
                view.camera.focal_y,
                view.camera.center_x,
                view.camera.center_y,
                view.camera.near_plane,
                clipping_tolerance,
                self.distance2filter,
            )
        filter_3d_max = filter_3d[visibility_mask].max()
        filter_3d = torch.where(visibility_mask, filter_3d, filter_3d_max, out=filter_3d)
        self.filter_3D = filter_3d

    def increase_used_sh_degree(self) -> None:
        """Increases the used SH degree."""
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            self.active_sh_bases = (self.active_sh_degree + 1) ** 2

    def initialize_from_point_cloud(self, point_cloud: BasicPointCloud, training_cameras_extent: float) -> None:
        """Initializes the model from a point cloud."""
        self.training_cameras_extent = training_cameras_extent
        positions = point_cloud.positions.cuda()
        rgbs = torch.full_like(positions, fill_value=0.5) if point_cloud.colors is None else point_cloud.colors.cuda()
        n_initial_points = positions.shape[0]
        sh_all = torch.zeros((n_initial_points, (self.max_sh_degree + 1) ** 2, 3), dtype=torch.float32, device='cuda')
        sh_all[:, 0] = rgb_to_sh0(rgbs)

        Logger.log_info(f'Number of points at initialization: {n_initial_points:,}')

        distances = compute_root_mean_squared_knn_distances(positions)
        scales = self.inverse_scale_activation(distances)[..., None].repeat(1, 3)
        rotations = torch.zeros((n_initial_points, 4), dtype=torch.float32, device='cuda')
        rotations[:, 0] = 1.0

        opacities = self.inverse_opacity_activation(torch.full((n_initial_points, 1), fill_value=0.1, dtype=torch.float32, device='cuda'))

        self._positions = torch.nn.Parameter(positions.contiguous())
        self._sh_0 = torch.nn.Parameter(sh_all[:, 0:1].contiguous())
        self._sh_rest = torch.nn.Parameter(sh_all[:, 1:].contiguous())
        self._scales = torch.nn.Parameter(scales.contiguous())
        self._rotations = torch.nn.Parameter(rotations.contiguous())
        self._opacities = torch.nn.Parameter(opacities.contiguous())
        self.reset_densification_info()

    def training_setup(self, training_wrapper, dataset: 'BaseDataset') -> None:
        """Sets up the optimizer."""
        self.percent_dense = training_wrapper.PERCENT_DENSE

        param_groups = [
            {'params': [self._positions], 'lr': training_wrapper.LEARNING_RATE_POSITION_INIT * self.training_cameras_extent, 'name': 'positions'},
            {'params': [self._sh_0], 'lr': training_wrapper.LEARNING_RATE_FEATURE, 'name': 'sh_0'},
            {'params': [self._sh_rest], 'lr': training_wrapper.LEARNING_RATE_FEATURE / 20.0, 'name': 'sh_rest'},
            {'params': [self._opacities], 'lr': training_wrapper.LEARNING_RATE_OPACITY, 'name': 'opacities'},
            {'params': [self._scales], 'lr': training_wrapper.LEARNING_RATE_SCALING, 'name': 'scales'},
            {'params': [self._rotations], 'lr': training_wrapper.LEARNING_RATE_ROTATION, 'name': 'rotations'}
        ]

        try:
            from Thirdparty.Apex import FusedAdam
            # slightly faster than the PyTorch implementation
            self.optimizer = FusedAdam(param_groups, lr=0.0, eps=1e-15, adam_w_mode=False)
            Logger.log_info('using apex FusedAdam')
        except Framework.ExtensionError:
            Logger.log_warning('apex is not installed -> using the slightly slower PyTorch Adam instead')
            Logger.log_warning('apex can be installed using ./scripts/install.py -e src/Thirdparty/Apex.py')
            self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15, fused=True)

        self.position_lr_scheduler = LRDecayPolicy(
            lr_init=training_wrapper.LEARNING_RATE_POSITION_INIT * self.training_cameras_extent,
            lr_final=training_wrapper.LEARNING_RATE_POSITION_FINAL * self.training_cameras_extent,
            lr_delay_mult=training_wrapper.LEARNING_RATE_POSITION_DELAY_MULT,
            max_steps=training_wrapper.LEARNING_RATE_POSITION_MAX_STEPS)

        if training_wrapper.USE_3D_FILTER:
            self.setup_3d_filter(dataset)

    def update_learning_rate(self, iteration: int) -> None:
        """ Learning rate scheduling per step """
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'positions':
                lr = self.position_lr_scheduler(iteration)
                param_group['lr'] = lr

    def reset_opacities(self, max_opacity: float) -> None:
        """Resets the opacities to a fixed value."""
        current_opacities = self.get_opacities_with_3D_filter if self.use_3d_filter else self.get_opacities
        opacities_new = current_opacities.clamp_max(max_opacity)
        if self.use_3d_filter:
            # make sure that the current 3d filter has the same effect on the new opacities
            scales_square = torch.square(self.get_scales)
            det1 = scales_square.prod(dim=1)
            scales_after_square = scales_square + torch.square(self.filter_3D)
            det2 = scales_after_square.prod(dim=1)
            coef = torch.sqrt(det1 / det2)
            opacities_new = opacities_new / coef[..., None]
        opacities_new = self.inverse_opacity_activation(opacities_new)
        replace_param_group_data(self.optimizer, opacities_new, 'opacities')

    def decay_opacities(self, decay_factor: float):
        """Decays the opacities by a factor."""
        opacities_new = self.inverse_opacity_activation(self.get_opacities * decay_factor)
        replace_param_group_data(self.optimizer, opacities_new, 'opacities')
        # these lines match the behavior of reset_opacities, but aren't necessary as the decay is only a small change
        # current_opacities = self.get_opacities_with_3D_filter if self.use_3d_filter else self.get_opacities
        # opacities_new = current_opacities * decay_factor
        # if self.use_3d_filter:
        #     # make sure that the current 3d filter has the same effect on the new opacities
        #     scales = self.get_scales
        #     scales_square = torch.square(scales)
        #     det1 = scales_square.prod(dim=1)
        #     scales_after_square = scales_square + torch.square(self.filter_3D)
        #     det2 = scales_after_square.prod(dim=1)
        #     coef = torch.sqrt(det1 / det2)
        #     opacities_new = opacities_new / coef[..., None]
        # opacities_new = self.inverse_opacity_activation(opacities_new)
        # replace_param_group_data(self.optimizer, opacities_new, 'opacities')

    def prune_points(self, prune_mask: torch.Tensor) -> None:
        """Prunes points that are not visible or too large."""
        valid_mask = ~prune_mask
        optimizable_tensors = prune_param_groups(self.optimizer, valid_mask)

        self._positions = optimizable_tensors['positions']
        self._sh_0 = optimizable_tensors['sh_0']
        self._sh_rest = optimizable_tensors['sh_rest']
        self._opacities = optimizable_tensors['opacities']
        self._scales = optimizable_tensors['scales']
        self._rotations = optimizable_tensors['rotations']

    def densification_postfix(
            self,
            new_positions: torch.Tensor,
            new_sh_0: torch.Tensor,
            new_sh_rest: torch.Tensor,
            new_opacities: torch.Tensor,
            new_scales: torch.Tensor,
            new_rotations: torch.Tensor
    ) -> None:
        """Incorporate the changes from the densification step into the parameter groups."""
        optimizable_tensors = extend_param_groups(self.optimizer, {
            'positions': new_positions,
            'sh_0': new_sh_0,
            'sh_rest': new_sh_rest,
            'opacities': new_opacities,
            'scales': new_scales,
            'rotations': new_rotations
        })
        self._positions = optimizable_tensors['positions']
        self._sh_0 = optimizable_tensors['sh_0']
        self._sh_rest = optimizable_tensors['sh_rest']
        self._opacities = optimizable_tensors['opacities']
        self._scales = optimizable_tensors['scales']
        self._rotations = optimizable_tensors['rotations']

    def reset_densification_info(self):
        n_points = self._positions.shape[0]
        n_floats = 3 if Gaussians.GOF_DENSIFICATION_GRAD else 2
        self.densification_info = torch.zeros((n_floats, n_points, 1), dtype=torch.float32, device='cuda')

    def split(self, grads: torch.Tensor, grad_threshold: float, grads_abs: torch.Tensor | None, grad_abs_threshold: float | None) -> torch.Tensor:
        """Densify by splitting Gaussians that satisfy the gradient condition."""
        n_init_points = self.get_positions.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros(n_init_points, dtype=torch.float32, device='cuda')
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        if grads_abs is not None:
            padded_grad_abs = torch.zeros(n_init_points, dtype=torch.float32, device='cuda')
            padded_grad_abs[:grads_abs.shape[0]] = grads_abs.squeeze()
            selected_pts_mask |= torch.where(padded_grad_abs >= grad_abs_threshold, True, False)
        selected_pts_mask &= torch.max(self.get_scales, dim=1).values > self.percent_dense * self.training_cameras_extent

        stds = self.get_scales[selected_pts_mask].repeat(2, 1)
        samples = torch.normal(mean=0.0, std=stds)
        rots = quaternion_to_rotation_matrix(self._rotations[selected_pts_mask]).repeat(2, 1, 1)
        new_positions = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_positions[selected_pts_mask].repeat(2, 1)
        new_scales = self.inverse_scale_activation(self.get_scales[selected_pts_mask].repeat(2, 1) / 1.6)
        new_rotations = self._rotations[selected_pts_mask].repeat(2, 1)
        new_sh_0 = self._sh_0[selected_pts_mask].repeat(2, 1, 1)
        new_sh_rest = self._sh_rest[selected_pts_mask].repeat(2, 1, 1)
        new_opacities = self._opacities[selected_pts_mask].repeat(2, 1)

        self.densification_postfix(new_positions, new_sh_0, new_sh_rest, new_opacities, new_scales, new_rotations)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(2 * selected_pts_mask.sum().item(), device='cuda', dtype=torch.bool)))
        return prune_filter

    def duplicate(self, grads: torch.Tensor, grad_threshold: float, grads_abs: torch.Tensor | None, grad_abs_threshold: float | None) -> None:
        """Densify by duplicating Gaussians that satisfy the gradient condition."""
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(grads.flatten() >= grad_threshold, True, False)
        if grads_abs is not None:
            selected_pts_mask |= torch.where(grads_abs.flatten() >= grad_abs_threshold, True, False)
        selected_pts_mask &= torch.max(self.get_scales, dim=1).values <= self.percent_dense * self.training_cameras_extent

        if Gaussians.GOF_DENSIFICATION_CLONE:
            # sample a new gaussian instead of fixing position (from gof)
            stds = self.get_scales[selected_pts_mask]
            samples = torch.normal(mean=0.0, std=stds)
            rots = quaternion_to_rotation_matrix(self._rotations[selected_pts_mask])
            new_positions = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_positions[selected_pts_mask]
        else:
            new_positions = self._positions[selected_pts_mask]  # 3dgs

        new_sh_0 = self._sh_0[selected_pts_mask]
        new_sh_rest = self._sh_rest[selected_pts_mask]
        new_opacities = self._opacities[selected_pts_mask]
        new_scales = self._scales[selected_pts_mask]
        new_rotations = self._rotations[selected_pts_mask]

        self.densification_postfix(new_positions, new_sh_0, new_sh_rest, new_opacities, new_scales, new_rotations)

    def densify_and_prune(self, grad_threshold: float, min_opacity: float, prune_large_gaussians: bool) -> None:
        """Densifies the point cloud and prunes points that are not visible or too large."""
        denominator = self.densification_info[0].clamp_min(1.0)
        grads = self.densification_info[1] / denominator
        grads_abs, grad_abs_threshold = None, None
        if Gaussians.GOF_DENSIFICATION_GRAD:
            grads_abs = self.densification_info[2] / denominator
            ratio = (grads.flatten() >= grad_threshold).float().mean()
            grad_abs_threshold = torch.quantile(grads_abs.flatten(), 1.0 - ratio).item()

        self.duplicate(grads, grad_threshold, grads_abs, grad_abs_threshold)
        prune_mask = self.split(grads, grad_threshold, grads_abs, grad_abs_threshold)

        prune_mask |= self.get_opacities.flatten() < min_opacity
        if prune_large_gaussians:
            prune_mask |= self.get_scales.max(dim=1).values > 0.1 * self.training_cameras_extent
        self.prune_points(prune_mask)

        self.reset_densification_info()

        torch.cuda.empty_cache()

    def importance_pruning(self, max_blending_weights: torch.Tensor, threshold: float) -> None:
        """Prunes points based on the maximum blending weights."""
        mask = max_blending_weights < threshold
        self.prune_points(mask)
        if self.use_3d_filter:
            self.filter_3D = self.filter_3D[~mask].contiguous()

    def bake_activations(self):
        """Bakes relevant activation functions into the final parameters."""
        # bake activation functions into final parameters
        self._rotations.data = self.get_rotations
        self.rotation_activation = torch.nn.Identity()
        # Important: opacities must be baked before scales due to implementation of get_opacities_with_3D_filter
        self._opacities.data = self.get_opacities_with_3D_filter if self.use_3d_filter else self.get_opacities
        self.opacity_activation = torch.nn.Identity()
        self.inverse_opacity_activation = torch.nn.Identity()
        self._scales.data = self.get_scales_with_3D_filter if self.use_3d_filter else self.get_scales
        self.scale_activation = torch.nn.Identity()
        self.inverse_scale_activation = torch.nn.Identity()
        # 3d filter is baked into relevant parameters now
        self.use_3d_filter = False

        # prune points that would never be visible anyway
        self.prune_points((self._opacities < 0.00392156862).squeeze())  # 1/255

        # morton sort
        morton_encoding = morton_encode(self._positions)
        order = torch.argsort(morton_encoding)
        self._positions.data = self._positions[order].contiguous()
        self._rotations.data = self._rotations[order].contiguous()
        self._sh_0.data = self._sh_0[order].contiguous()
        self._sh_rest.data = self._sh_rest[order].contiguous()
        self._scales.data = self._scales[order].contiguous()
        self._opacities.data = self._opacities[order].contiguous()


@Framework.Configurable.configure(
    SH_DEGREE=3,
    BACKGROUND_MODEL='SH',
    SH_BACKGROUND_MODEL=Framework.ConfigParameterList(
        DEGREE=3,
        DISABLE_COARSE_TO_FINE=True,
    ),
    VET_BACKGROUND_MODEL=Framework.ConfigParameterList(
        DEGREE=3,
        DISABLE_COARSE_TO_FINE=True,
        BASE_GAUSSIANS=100000,
        NUM_LAYERS=3,
        BETA_INTERVAL=0.02,
        KEEP_RATIO=0.4,
        MIN_SCALE_RATIO=0.005,
        MAX_SCALE_RATIO=0.06, 
        USE_AUTO_CONFIG=True,
        HALF_BALL=True,
        KNN_COLOR =4,
    ),
    HG_BACKGROUND_MODEL=Framework.ConfigParameterList(
        DEGREE=3,
        DISABLE_COARSE_TO_FINE=True,
        BASE_GAUSSIANS=100000,
        BETA_INTERVAL=0.02,
        KEEP_RATIO=0.4,
        MIN_SCALE_RATIO=0.005,
        MAX_SCALE_RATIO=0.06, 
        USE_AUTO_CONFIG=True,
        HALF_BALL=True,
        KNN_COLOR =4,
    ),
)
class HTGSModel(BaseModel):
    """Defines the HTGS model."""

    def __init__(self, name: str = None) -> None:
        super().__init__(name)
        self.gaussians: Gaussians | None = None
        self.background_model: NeuralEnvironmentMap | None = None

    def build(self) -> 'HTGSModel':
        """Builds the model."""
        pretrained = self.num_iterations_trained > 0
        self.gaussians = Gaussians(self.SH_DEGREE, pretrained)
        match self.BACKGROUND_MODEL:
            case 'SH':
                self.background_model = SHEnvironmentMap(self.SH_BACKGROUND_MODEL.DEGREE, pretrained, self.SH_BACKGROUND_MODEL.DISABLE_COARSE_TO_FINE)
            case 'MLP':
                self.background_model = NeuralEnvironmentMap()
            case 'TEX':
                self.background_model = EnvironmentMap()
            case 'MSI':
                self.background_model = MultiSphereEnvironmentMap()
            case 'VET':
                self.background_model = LayeredGaussianEnvMap(
                    base_gaussians=self.VET_BACKGROUND_MODEL.BASE_GAUSSIANS,
                    sh_degree=self.VET_BACKGROUND_MODEL.DEGREE,
                    pretrained=pretrained,
                    disable_coarse_to_fine=self.VET_BACKGROUND_MODEL.DISABLE_COARSE_TO_FINE,
                    num_layers=self.VET_BACKGROUND_MODEL.NUM_LAYERS,
                    beta_interval=self.VET_BACKGROUND_MODEL.BETA_INTERVAL,
                    keep_ratio=self.VET_BACKGROUND_MODEL.KEEP_RATIO,
                    min_scale_ratio=self.VET_BACKGROUND_MODEL.MIN_SCALE_RATIO,
                    max_scale_ratio=self.VET_BACKGROUND_MODEL.MAX_SCALE_RATIO,
                    auto_config = self.VET_BACKGROUND_MODEL.USE_AUTO_CONFIG,
                    half_ball = self.VET_BACKGROUND_MODEL.HALF_BALL,
                    knn_color=  self.VET_BACKGROUND_MODEL.KNN_COLOR,
            )
            case 'HG':
                self.background_model = LayeredGaussianEnvMap(
                    base_gaussians=self.HG_BACKGROUND_MODEL.BASE_GAUSSIANS,
                    sh_degree=self.HG_BACKGROUND_MODEL.DEGREE,
                    pretrained=pretrained,
                    disable_coarse_to_fine=self.HG_BACKGROUND_MODEL.DISABLE_COARSE_TO_FINE,
                    num_layers=1,
                    beta_interval=self.HG_BACKGROUND_MODEL.BETA_INTERVAL,
                    keep_ratio=self.HG_BACKGROUND_MODEL.KEEP_RATIO,
                    min_scale_ratio=self.HG_BACKGROUND_MODEL.MIN_SCALE_RATIO,
                    max_scale_ratio=self.HG_BACKGROUND_MODEL.MAX_SCALE_RATIO,
                    auto_config = self.HG_BACKGROUND_MODEL.USE_AUTO_CONFIG,
                    half_ball = self.HG_BACKGROUND_MODEL.HALF_BALL,
                    knn_color=  self.HG_BACKGROUND_MODEL.KNN_COLOR,
            )

        return self

    def get_optimizer_param_groups(self, n_iterations: int) -> tuple[list[dict], list[LRDecayPolicy]]:
        """Returns the optimizer parameter groups and learning rate schedulers."""
        learnable_components = [self.background_model]
        param_groups, schedulers = zip(*(component.get_optimizer_param_groups(n_iterations) for component in learnable_components))
        return sum(param_groups, []), sum(schedulers, [])
