# -- coding: utf-8 --

"""HTGS/Trainer.py: Implementation of the trainer for HTGS."""
import torch
import Framework
from pathlib import Path
from Datasets.Base import BaseDataset
from Datasets.utils import BasicPointCloud
from Logging import Logger
from Methods.Base.GuiTrainer import GuiTrainer
from Methods.Base.utils import pre_training_callback, training_callback, post_training_callback
from Methods.HTGS_egsr.Loss import HTGSLoss
from Optim.Samplers.DatasetSamplers import DatasetSampler
from Methods.HTGS_egsr.Model import (LayeredGaussianEnvMap,EnvironmentMap,SHEnvironmentMap,
NeuralEnvironmentMap,MultiSphereEnvironmentMap)
from Methods.HTGS_egsr.Loss import background_color_regularization,sky_cone_floor_loss
from Methods.HTGS_egsr.utils.Sky_color_extractor import (
    compute_sky_mode_color_and_masks,
    compute_sky_hist_and_complexity,
    get_view_image_name,
    is_sky_cache_current,
    load_cached_sky_complexity,
    load_cached_sky_masks_and_color,
    resolve_sky_cache_dir,
    write_sky_cache_version,
)
from Methods.HTGS_egsr.utils.gaussian_env_utils import  compute_lambda
@Framework.Configurable.configure(
    NUM_ITERATIONS=30_000,
    LEARNING_RATE_POSITION_INIT=0.00016,
    LEARNING_RATE_POSITION_FINAL=0.0000016,
    LEARNING_RATE_POSITION_DELAY_MULT=0.01,
    LEARNING_RATE_POSITION_MAX_STEPS=30_000,
    LEARNING_RATE_FEATURE=0.0025,
    LEARNING_RATE_OPACITY=0.05,
    LEARNING_RATE_SCALING=0.005,
    LEARNING_RATE_ROTATION=0.001,
    PERCENT_DENSE=0.01,
    USE_3D_FILTER=True,
    USE_OPACITY_RESET=False,
    OPACITY_RESET_MAX_OPACITY=0.01,
    USE_OPACITY_DECAY=True,
    USE_VISIBILITY_PRUNING=True,
    VISIBILITY_PRUNING_THRESHOLD=0.01,
    USE_DISTANCE_SCALING=True,
    OPACITY_RESET_INTERVAL=3_000,
    OPACITY_THRESHOLD=0.005,
    DENSIFY_START_ITERATION=500,
    DENSIFY_END_ITERATION=15_000,
    DENSIFICATION_INTERVAL=100,
    DENSIFY_GRAD_THRESHOLD=0.0002,
    LOSS=Framework.ConfigParameterList(
        LAMBDA_L1=0.8,
        LAMBDA_DSSIM=0.2,
        LAMBDA_BCE=0.001,
        LAMBDA_FG_IN_SKY=0.01, 
    ),
)
class HTGSTrainer(GuiTrainer):
    """Defines the trainer for the HTGS method."""

    def __init__(self, **kwargs) -> None:
        super(HTGSTrainer, self).__init__(**kwargs)
        self.train_sampler = None
        self.loss = HTGSLoss(loss_config=self.LOSS)
        self.optimizer = None
        self.scheduler = None
        self._bg_frozen = False
        
    @pre_training_callback(priority=50)
    @torch.no_grad()
    def createSampler(self, _, dataset: 'BaseDataset') -> None:
        """Creates the sampler."""
        self.train_sampler = DatasetSampler(dataset=dataset.train(), random=True)

    @pre_training_callback(priority=40)
    @torch.no_grad()
    def setupGaussians(self, _, dataset: 'BaseDataset') -> None:
        """Sets up the model."""
        dataset.train()
        camera_centers = torch.stack([view.position for view in dataset])
        radius = (1.1 * torch.max(torch.linalg.norm(camera_centers - torch.mean(camera_centers, dim=0), dim=1))).item()
        Logger.log_info(f'Training cameras extent: {radius:.2f}')

        if dataset.point_cloud is not None:
            point_cloud = dataset.point_cloud
        else:
            n_random_points = 100_000
            min_bounds, max_bounds = dataset.bounding_box.min_max
            extent = max_bounds - min_bounds
            point_cloud = BasicPointCloud(torch.rand(n_random_points, 3, dtype=torch.float32, device=min_bounds.device) * extent + min_bounds)
        self.model.gaussians.initialize_from_point_cloud(point_cloud, radius)
        self.model.gaussians.training_setup(self, dataset)
        self.init_background_and_optimizer(dataset)


    @training_callback(priority=110, start_iteration=1000, iteration_stride=1000)
    @torch.no_grad()
    def increaseSHDegree(self, *_) -> None:
        """Increase the number of used SH coefficients up to a maximum degree."""
        self.model.gaussians.increase_used_sh_degree()
        if self.model.background_model is not None:
            self.model.background_model.increase_used_sh_degree()

    @training_callback(active='USE_VISIBILITY_PRUNING', priority=105, start_iteration=15000, iteration_stride=1000)
    @torch.no_grad()
    def importanceBasedPruning(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Pruning from RadSplat (see https://arxiv.org/abs/2403.13806)."""
        if iteration in [16000, 24000]:
            max_blending_weights = self.renderer.compute_max_weights(dataset.train(), threshold=self.VISIBILITY_PRUNING_THRESHOLD)
            self.model.gaussians.importance_pruning(max_blending_weights, threshold=self.VISIBILITY_PRUNING_THRESHOLD)

    @training_callback(priority=100)
    def trainingIteration(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Performs a training step without actually doing the optimizer step."""
        # init modes
        self.model.train()
        dataset.train()
        self.loss.train()
        # update learning rate
        self.model.gaussians.update_learning_rate(iteration + 1)
        # get random sample from dataset
        sample = self.train_sampler.get(dataset=dataset)
        view = sample['view']

        has_bg = getattr(self.model, "background_model", None) is not None
        training_enabled=(iteration < int(0.07 * self.NUM_ITERATIONS)) or (iteration >= int(0.95 * self.NUM_ITERATIONS))
        if iteration > int(0.99 * self.NUM_ITERATIONS) and isinstance(self.model.background_model, LayeredGaussianEnvMap):
            training_enabled=False
        # render sample
        image, alpha = self.renderer.render_image_training(
            view=view,
            update_densification_info=iteration <= self.DENSIFY_END_ITERATION,
            use_distance_scaling=self.USE_DISTANCE_SCALING,
        )
        # safe sky_mask
        # self.sky_masks is explicitly aligned with the training split during
        # initialization. Reuse the already sampled ID so image and mask match.
        sky_mask = None
        if has_bg and hasattr(self, 'sky_masks') and self.sky_masks is not None:
            sky_mask = self.sky_masks[sample['sample_id']]
        elif has_bg:
            sky_mask = getattr(view, 'sky_mask', None)
        if sky_mask is not None:
            sky_mask = sky_mask.to(device=alpha.device, dtype=torch.bool)
        if sky_mask is None:
            sky_mask = torch.zeros_like(alpha, dtype=torch.bool, device=alpha.device)
        # main loss
        loss = self.loss(image, view.rgb, alpha, sky_mask)

        if has_bg:
            if not training_enabled and not self._bg_frozen:
                # enter freeze window: disable background grads
                for p in self.model.background_model.parameters():
                    p.requires_grad = False
                self._bg_frozen = True
                self._rebuild_optimizer_excluding_frozen()
            elif training_enabled and self._bg_frozen:
                # leave freeze window: re-enable background grads
                for p in self.model.background_model.parameters():
                    p.requires_grad = True
                self._bg_frozen = False
                self._rebuild_optimizer_excluding_frozen()

        # background regularization
        if has_bg and training_enabled: 
            bg_model = self.model.background_model
            if isinstance(self.model.background_model, LayeredGaussianEnvMap):
                # color prior toward target sky tone
                color_lambda = self.color_lambda
                loss = loss + color_lambda * background_color_regularization(bg_model.sky_sh_0, self.sky_target_color)  
                if iteration == int(0.99 * self.NUM_ITERATIONS):
                    # color diffusion
                    bg_model.propagation()
                if iteration%200 == 0 and iteration!=0:
                    # opacity correction
                    bg_model.correct_opacity()

            elif  isinstance(self.model.background_model, SHEnvironmentMap):
                # SH cone floor toward plausible zenith brightness
                loss = loss + 5e-3 * sky_cone_floor_loss(bg_model,
                            target=(torch.tensor(self.sky_target_color)/255.0),
                            margin=0.05,  
                            cone_deg=35,
                            K=1024)

        loss.backward()


    @training_callback(priority=90, start_iteration='DENSIFY_START_ITERATION', end_iteration='DENSIFY_END_ITERATION', iteration_stride='DENSIFICATION_INTERVAL')
    @torch.no_grad()
    def densify(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Apply densification."""
        if iteration == self.DENSIFY_START_ITERATION:
            return
        self.model.gaussians.densify_and_prune(self.DENSIFY_GRAD_THRESHOLD, self.OPACITY_THRESHOLD, iteration > self.OPACITY_RESET_INTERVAL)

        if self.USE_3D_FILTER:
            self.model.gaussians.compute_3d_filter(dataset.train())

    @training_callback(active='USE_OPACITY_RESET', priority=80, start_iteration='OPACITY_RESET_INTERVAL', end_iteration='DENSIFY_END_ITERATION', iteration_stride='OPACITY_RESET_INTERVAL')
    @torch.no_grad()
    def resetOpacities(self, iteration: int, _) -> None:
        """Reset opacities."""
        if iteration == self.DENSIFY_END_ITERATION:
            return
        self.model.gaussians.reset_opacities(max_opacity=self.OPACITY_RESET_MAX_OPACITY)

    @training_callback(active='USE_OPACITY_DECAY', priority=80, start_iteration='DENSIFY_START_ITERATION', end_iteration='DENSIFY_END_ITERATION', iteration_stride=50)
    @torch.no_grad()
    def decayOpacities(self, iteration: int, _) -> None:
        """Decay opacities."""
        if iteration == self.DENSIFY_START_ITERATION:
            return
        self.model.gaussians.decay_opacities(decay_factor=0.9995)

    @training_callback(active='USE_3D_FILTER', priority=75, start_iteration='DENSIFY_END_ITERATION', iteration_stride=100)
    @torch.no_grad()
    def recompute3DFilter(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Recompute 3D filter."""
        if self.DENSIFY_END_ITERATION < iteration < self.NUM_ITERATIONS - 100:
            self.model.gaussians.compute_3d_filter(dataset.train())

    @training_callback(priority=70)
    @torch.no_grad()
    def performOptimizerStep(self, *_) -> None:
        """Update parameters."""
        self.model.gaussians.optimizer.step()
        self.model.gaussians.optimizer.zero_grad()
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

    @training_callback(active='WANDB.ACTIVATE', priority=10, iteration_stride='WANDB.INTERVAL')
    @torch.no_grad()
    def logWandB(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Adds primitive count to default Weights & Biases logging."""
        Framework.wandb.log({
            'n_primitives': self.model.gaussians.get_positions.shape[0]
        }, step=iteration)
        # default logging
        super().log_wandb(iteration, dataset)

    @post_training_callback(priority=1000)
    @torch.no_grad()
    def bakeActivations(self, *_) -> None:
        """Bake relevant activation functions after training."""
        self.model.gaussians.bake_activations()
        # delete optimizer to save memory
        self.model.gaussians.optimizer = None


    @pre_training_callback(priority=35)
    @torch.no_grad()
    def probe_no_gt_background(self, _, dataset: 'BaseDataset', tau: float = 1e-3):
        """Scan the dataset to mark background Gaussians with no foreground supervision."""
        if not hasattr(self.model, "background_model") or not isinstance(self.model.background_model, (LayeredGaussianEnvMap)):
            return

        bg = self.model.background_model
        dataset.train()

        n_bg = bg.sky_sh_0.shape[0]
        global_hits = torch.zeros(n_bg, dtype=torch.bool, device=bg.sky_sh_0.device)

        for view in dataset:

            # render foreground alpha
            _, alpha = self.renderer.render_image_training(
                view=view,
                update_densification_info=False,
                use_distance_scaling=self.USE_DISTANCE_SCALING,
            )
            alpha2d = alpha.squeeze(0)

            # visible if projected inside image and neighborhood has (1 - alpha) > tau
            hits = bg.compute_max_visible_hits(view, alpha2d, pixel_pad=1, tau=tau)
            global_hits |= hits

        no_gt_mask = ~global_hits
        bg.overlay_indices = torch.where(no_gt_mask)[0]  # indices with no GT supervision

        



    def init_background_and_optimizer(self, dataset: 'BaseDataset') -> None:
        """Initialize optimizer/scheduler and background-related tensors if background_model exists."""
        bg = getattr(self.model, "background_model", None)
        if bg is None:
            return
        dataset.train()
        # Precompute sky color and masks using the active training views. In the
        # GSPAR 2.0 data model each View owns its ImageData source path, so no
        # parallel dataset-level filename list is needed.
        with torch.no_grad():
            samples = [dataset[i] for i in range(len(dataset))]
            imgs = [sample.rgb for sample in samples]
            names = [get_view_image_name(sample) for sample in samples]

            explicit_cache_dir = getattr(dataset, 'mask_path', None)
            cache_dir = Path(explicit_cache_dir) if explicit_cache_dir is not None else resolve_sky_cache_dir(
                dataset.dataset_path,
                Framework.Directories.NERFICG_ROOT / 'dataset',
                Framework.Directories.OUTPUT_DIR,
            )
            cache_is_current = is_sky_cache_current(cache_dir)
            cached_sky_data = (
                load_cached_sky_masks_and_color(imgs, names, cache_dir)
                if cache_is_current else None
            )

            if cached_sky_data is not None:
                self.sky_target_color, self.sky_masks = cached_sky_data
            else:
                print("Precomputing background color for all train images...")
                self.sky_target_color, self.sky_masks = compute_sky_mode_color_and_masks(imgs, names, cache_dir)

            cached_complexity = load_cached_sky_complexity(cache_dir) if cached_sky_data is not None else None
            if cached_complexity is not None:
                self.sky_complexity = cached_complexity
            else:
                print("Computing sky complexity...")
                global_stats = compute_sky_hist_and_complexity(imgs, names, self.sky_masks, cache_dir)
                self.sky_complexity = global_stats['complexity']

            if cached_sky_data is None or cached_complexity is None:
                write_sky_cache_version(cache_dir)
        #Logger.logInfo("Background colors precomputed.")
        print("Background colors precomputed.")
        self.color_lambda = 1e-2
        if isinstance(bg, LayeredGaussianEnvMap) and bg.auto_config:
            self.color_lambda = compute_lambda(self.sky_complexity)
            print(f"auto color_lambda={self.color_lambda}")

        # Initialize background textures with sky color prior
        if isinstance(bg, EnvironmentMap) or isinstance(bg, MultiSphereEnvironmentMap)  :
            with torch.no_grad():
                tex = next(
                    p for n, p in bg.named_parameters()
                    if n.split('.')[-1] == 'texture' and p.ndim == 4 and p.shape[1] == 3
                )
                tex[:] = torch.logit(
                    torch.tensor(self.sky_target_color, device=tex.device, dtype=tex.dtype)
                        .div_(255.0)
                        .clamp_(1e-6, 1 - 1e-6)
                ).view(1, 3, 1, 1)
                # For MSI: create stable near-to-far alpha ramp to ensure smooth initialization
                if hasattr(bg, 'alpha'):
                    alp = next(p for n,p in bg.named_parameters()
                            if n.split('.')[-1] == 'alpha' and p.ndim == 4 and p.shape[1] == 1)
                    L = alp.shape[0]
                    a_near, a_far = -8.0, 4.0   # Near layers more transparent, far layers denser
                    ramp = torch.linspace(a_near, a_far, L, dtype=alp.dtype, device=alp.device).view(L,1,1,1)
                    alp[...] = ramp
        # Build Gaussian layers using scene stats and sky complexity
        elif isinstance(bg, LayeredGaussianEnvMap):
            with torch.no_grad():
                pos = dataset.point_cloud.positions.to(device="cuda", dtype=torch.float32)
                camera_centers = torch.stack([view.position for view in dataset]).to(device=pos.device, dtype=pos.dtype)
                center = camera_centers.mean(dim=0)

                # COLMAP point clouds can contain distant outliers. A min/max
                # bounding box makes those outliers create enormous background
                # Gaussians, so use a robust scene radius while ensuring every
                # training camera remains comfortably inside the background shell.
                point_radius = torch.quantile(
                    torch.linalg.norm(pos - center, dim=1), 0.95
                ).item()
                camera_radius = torch.linalg.norm(
                    camera_centers - center, dim=1
                ).amax().item()
                radius = max(1.1 * point_radius, 1.5 * camera_radius)
                Logger.log_info(
                    f'Background shell radius: {radius:.2f} '
                    f'(point q95: {point_radius:.2f}, camera radius: {camera_radius:.2f})'
                )
                bg.build(center, radius,self.sky_complexity)


        # --- optimizer & scheduler ---
        param_groups, schedulers = self.model.get_optimizer_param_groups(self.NUM_ITERATIONS)
        # TODO: for some reason using apex.FusedAdam here leads to no optimization at all
        # self.optimizer = torch.optim.AdamW(
        #     param_groups, lr=1.0, betas=(0.9, 0.99), eps=1.0e-15, weight_decay=0.0
        # )
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedulers)
        self._rebuild_optimizer_excluding_frozen()

    def _rebuild_optimizer_excluding_frozen(self):
        """
        Rebuild optimizer and scheduler, excluding parameters with requires_grad=False.
        Ensures frozen background parameters are ignored during training.
        """
        if self.optimizer is not None:
            # Clear old optimizer and scheduler to release filtered states
            del self.optimizer, self.scheduler
            self.optimizer = None
            self.scheduler = None
        # Request param groups from model and keep only trainable ones
        param_groups, schedulers = self.model.get_optimizer_param_groups(self.NUM_ITERATIONS)
        filtered = []
        for g in param_groups:
            params = [p for p in g['params'] if p.requires_grad]
            if len(params) > 0:
                ng = dict(g)
                ng['params'] = params
                filtered.append(ng)
        if len(filtered) == 0:
            return  
        self.optimizer = torch.optim.AdamW(filtered, lr=1.0, betas=(0.9, 0.99), eps=1e-15, weight_decay=0.0)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedulers)
