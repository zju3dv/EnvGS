import copy
import torch
import numpy as np
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F

from easyvolcap.engine import cfg
from easyvolcap.engine import SAMPLERS
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.models.networks.noop_network import NoopNetwork
from easyvolcap.models.samplers.gaussian2d_sampler import Gaussian2DSampler

from easyvolcap.utils.sh_utils import *
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.math_utils import normalize
from easyvolcap.utils.grid_utils import sample_points_subgrid
from easyvolcap.utils.colmap_utils import load_sfm_ply, save_sfm_ply
from easyvolcap.utils.net_utils import freeze_module, make_params, make_buffer
from easyvolcap.utils.gaussian2d_utils import GaussianModel, render, prepare_gaussian_camera
from easyvolcap.utils.data_utils import load_pts, export_pts, to_x, to_cuda, to_cpu, to_tensor, remove_batch


@SAMPLERS.register_module()
class EnvGSSampler(Gaussian2DSampler):
    def __init__(self,
                 # Legacy APIs
                 network: NoopNetwork = None,  # ignore this

                 # 3DGS-DR related configs
                 sh_start_iter: int = 10000,
                 densify_until_iter: int = 30000,
                 init_densification_interval: int = 100,
                 norm_densification_interval: int = 500,
                 normal_prop_until_iter: int = 24000,
                 normal_prop_interval: int = 1000,
                 opacity_lr0_interval: int = 200,
                 opacity_lr: float = 0.05,
                 color_sabotage_until_iter: int = 24000,
                 color_sabotage_interval: int = 1000,
                 reset_specular_all: bool = False,

                 # Gaussian configs
                 env_preload_gs: str = '',
                 env_bounds: List[List[float]] = [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
                 # SHs configs
                 env_sh_deg: int = 3,
                 env_init_sh_deg: int = 0,
                 env_sh_start_iter: int = 0,
                 env_sh_update_iter: int = 1000,
                 # Opacity and scale configs
                 env_init_occ: float = 0.1,
                 # Densify & pruning configs
                 env_densify_from_iter: int = 500,
                 env_densify_until_iter: int = 15000,
                 env_densification_interval: int = 100,
                 env_opacity_reset_interval: int = 3000,
                 env_densify_grad_threshold: float = 0.0002,
                 env_min_opacity: float = 0.05,
                 env_densify_size_threshold: float = 0.01,  # alias for `percent_dense` as in the original code, https://github.com/hbb1/2d-gaussian-splatting/blob/6d249deeec734ad07760496fc32be3b91ac236fc/scene/gaussian_model.py#L378
                 env_prune_large_gs: bool = True,
                 env_prune_visibility: bool = False,
                 env_max_scene_threshold: float = 0.1,  # default 0.1, same as the original 2DGS
                 env_max_screen_threshold: float = None,  # not used in the original 3DGS/2DGS, they wrote a bug, though `max_screen_threshold=20`
                 env_min_weight_threshold: float = None,
                 # EasyVolcap additional densify & pruning tricks
                 env_screen_until_iter: int = int(4000 / 60 * cfg.runner_cfg.epochs),
                 env_split_screen_threshold: float = None,
                 env_min_gradient: float = None,
                 # Rendering configs
                 env_white_bg: bool = False,  # always set to False !!!
                 env_bg_brightness: float = 0.0,  # used in the original renderer

                 # Reflection related parameters
                 render_reflection: bool = True,  # default is True here
                 render_reflection_start_iter: int = 3000,  # need a initial geometry to model reflection
                 detach: bool = False,  # detach the reflected rays for training the reflection model

                 # Ray tracing configs
                 use_optix_tracing: bool = True,
                 use_base_tracing: bool = False,
                 tracing_backend: str = 'cpp',
                 env_max_gs: int = 7e5,  # control the maximum number of gaussians
                 env_max_gs_threshold: float = 0.9,  # percentage of the visibility pruning
                 prune_visibility: bool = True,  # whether to prune the gaussians based on accumulated weights
                 max_trace_depth: int = 0,
                 specular_threshold: float = 0.0,  # specular threshold for reflection rendering
                 n_sample_dirs: int = 1, # number of sampled reflected directions
                 specular_filtering_start_iter: int = -1,  # start to filter pixels with large specular values
                 specular_filtering_percent: float = 0.75,  # percentage of pixels to be filtered
                 acc_filtering_start_iter: int = -1,  # start to filter pixels with large accumulated weights
                 multi_sampling_start_iter: int = -1,  # start to use multi-sample for reflection rendering

                 # Default parameters for Gaussian2DSampler
                 **kwargs,
                 ):
        # Inherit from the default `VolumetricVideoDataset`
        call_from_cfg(super().__init__,
                      kwargs,
                      network=network,
                      sh_start_iter=sh_start_iter,
                      densify_until_iter=densify_until_iter,
                      render_reflection=render_reflection,
                      use_optix_tracing=use_optix_tracing,
                      tracing_backend=tracing_backend,
                      prune_visibility=prune_visibility,
                      max_trace_depth=max_trace_depth,
                      specular_threshold=specular_threshold)

        # 3DGS-DR related configs
        self.init_densification_interval = init_densification_interval
        self.norm_densification_interval = norm_densification_interval
        self.normal_prop_until_iter = normal_prop_until_iter
        self.normal_prop_interval = normal_prop_interval
        self.opacity_lr0_interval = opacity_lr0_interval
        self.opacity_lr = opacity_lr
        self.color_sabotage_until_iter = color_sabotage_until_iter
        self.color_sabotage_interval = color_sabotage_interval
        self.reset_specular_all = reset_specular_all

        # Reflection related parameters
        self.use_base_tracing = use_base_tracing
        self.render_reflection_start_iter = render_reflection_start_iter
        self.n_sample_dirs = n_sample_dirs
        self.detach = detach
        self.specular_filtering_start_iter = specular_filtering_start_iter
        self.specular_filtering_percent = specular_filtering_percent
        self.acc_filtering_start_iter = acc_filtering_start_iter
        self.multi_sampling_start_iter = multi_sampling_start_iter

        # Environment Gaussian related parameters
        self.env_preload_gs = env_preload_gs
        self.env_bounds = env_bounds
        # Environment SH related parameters
        self.env_sh_deg = env_sh_deg
        self.env_init_sh_deg = env_init_sh_deg
        self.env_sh_start_iter = env_sh_start_iter
        self.env_sh_update_iter = env_sh_update_iter
        # Environment opacity and scale parameters
        self.env_init_occ = env_init_occ
        # Densify & pruning parameters
        self.env_densify_from_iter = env_densify_from_iter
        self.env_densify_until_iter = env_densify_until_iter
        self.env_densification_interval = env_densification_interval
        self.env_opacity_reset_interval = env_opacity_reset_interval
        self.env_densify_grad_threshold = env_densify_grad_threshold
        self.env_min_opacity = env_min_opacity
        self.env_densify_size_threshold = env_densify_size_threshold
        self.env_prune_large_gs = env_prune_large_gs
        self.env_prune_visibility = env_prune_visibility
        self.env_max_scene_threshold = env_max_scene_threshold
        self.env_max_screen_threshold = env_max_screen_threshold
        self.env_min_weight_threshold = env_min_weight_threshold
        # EasyVolcap additional densify & pruning tricks
        self.env_screen_until_iter = env_screen_until_iter
        self.env_split_screen_threshold = env_split_screen_threshold
        self.env_min_gradient = env_min_gradient
        self.env_max_gs = env_max_gs
        self.env_max_gs_threshold = env_max_gs_threshold
        # Store the last output for updating the gaussians
        self.last_output_env = None

        xyz, colors = self.init_env_points(self.env_preload_gs)
        # Create environment Gaussians
        self.env = GaussianModel(
            xyz=xyz,
            colors=colors,
            init_occ=self.env_init_occ,
            init_scale=None,
            sh_degree=self.env_sh_deg,
            init_sh_degree=self.env_init_sh_deg,
            spatial_scale=self.spatial_scale,
            xyz_lr_scheduler=self.xyz_lr_scheduler,
            render_reflection=False,
            max_gs=self.env_max_gs,
            max_gs_threshold=self.env_max_gs_threshold
        )

        # Update `self.pipe`
        self.pipe.convert_SHs_python = True  # enable SH -> RGB conversion in Python
        if self.use_base_tracing: self.pipe.convert_SHs_python = False
        self.pipe_env = copy.deepcopy(self.pipe)
        self.pipe_env.convert_SHs_python = False

        # Rendering configs of environment Gaussian
        self.env_white_bg = env_white_bg
        self.env_bg_brightness = 1. if env_white_bg else env_bg_brightness
        self.env_bg_channel = 3
        self.env_bg_color = make_buffer(torch.Tensor([self.env_bg_brightness] * self.env_bg_channel))

        # Time statistics
        self.times = []

    def init_env_points(self, ply_file: str = None, S: int = 32, N: int = 5):
        # Try to load the ply file
        try:
            xyz, rgb = load_sfm_ply(ply_file)  # (P, 3), (P, 3)
            log(yellow(f"Loaded the point cloud from {ply_file}."))
            xyz = torch.as_tensor(xyz, dtype=torch.float)
            rgb = torch.as_tensor(rgb, dtype=torch.float)  # already normalized to [0, 1]
        # If the file does not exist, generate random points and save them
        except:
            log(yellow(f"Failed to load the point cloud from {ply_file}, generating random points."))
            xyz = sample_points_subgrid(torch.as_tensor(self.env_bounds), S, N).float()  # (P, 3)
            rgb = torch.rand(xyz.shape, dtype=torch.float) / 255.0  # (P, 3)
            save_sfm_ply(ply_file, xyz.numpy(), rgb.numpy() * 255.0)

        return xyz, rgb

    @torch.no_grad()
    def update_dif_gaussians(self, batch: dotdict):
        if not self.training: return

        # Update the densification interval
        if batch.meta.iter < self.render_reflection_start_iter: self.densification_interval = self.init_densification_interval
        elif batch.meta.iter < self.normal_prop_until_iter: self.densification_interval = self.norm_densification_interval
        else: self.densification_interval = self.init_densification_interval

        # Prepare global variables
        iter: int = batch.meta.iter  # controls whether we're to update in this iteration
        output = self.last_output  # contains necessary information for updating gaussians
        optimizer: Adam = cfg.runner.optimizer

        # Log the total number of gaussians
        scalar_stats = batch.output.get('scalar_stats', dotdict())
        scalar_stats.num_pts = self.pcd.number
        batch.output.scalar_stats = scalar_stats
        # Log the last opacity reset iteration
        batch.output.last_opacity_reset_iter = self.opacity_reset_interval * (iter // self.opacity_reset_interval)

        # Update the learning rate
        self.pcd.update_learning_rate(iter.item(), optimizer, prefix='sampler.pcd.')

        # Increase the levels of SHs every `self.sh_update_iter=1000` iterations until a maximum degree
        if iter > 0 and iter < self.densify_until_iter and iter % self.sh_update_iter == 0 and self.sh_start_iter is not None and iter > self.sh_start_iter:
            changed = self.pcd.oneupSHdegree()
            if changed: log(yellow_slim(f'[ONEUP SH DEGREE] sh_deg: {self.pcd.active_sh_degree.item()}'))

        # Update only the rendered frame
        if iter > 0 and iter < self.densify_until_iter and output is not None:
            # Update all rendered gaussians in the batch
            pcd: GaussianModel = self.pcd

            # Preparing gaussian status for update
            visibility_filter = output.visibility_filter
            viewspace_point_tensor = output.viewspace_points  # no indexing, otherwise no grad # !: BATCH
            if output.viewspace_points.grad is None: return  # previous rendering was an evaluation
            if 'weight_accumulate' not in output: pcd.add_densification_stats(viewspace_point_tensor, visibility_filter)
            else: pcd.add_densification_stats(viewspace_point_tensor, visibility_filter, output.weight_accumulate)

            # Update gaussian splatting radii for update
            if not self.use_optix_tracing:
                radii = output.radii
                pcd.max_radii2D[visibility_filter] = torch.max(pcd.max_radii2D[visibility_filter], radii[visibility_filter])

            # Perform densification and pruning
            if iter > self.densify_from_iter and iter % self.densification_interval == 0:
                log(yellow_slim(f'Start updating gaussians of step: {iter:06d}'))
                # Iteration-related densification and pruning parameters
                split_screen_threshold = self.split_screen_threshold if iter < self.screen_until_iter else None
                max_screen_threshold = self.max_screen_threshold if iter > self.opacity_reset_interval else None
                # Perform actual densification and pruning
                pcd.densify_and_prune(
                    self.min_opacity,
                    self.min_gradient,
                    self.densify_grad_threshold,
                    self.densify_size_threshold,
                    split_screen_threshold,
                    self.max_scene_threshold,
                    max_screen_threshold,
                    self.min_weight_threshold,
                    self.prune_visibility,
                    optimizer,
                    self.prune_large_gs,
                    prefix='sampler.pcd.'
                )
                log(yellow_slim('Densification and pruning done! ' +
                                f'min opacity: {pcd.get_opacity.min().item():.4f} ' +
                                f'max opacity: {pcd.get_opacity.max().item():.4f} ' +
                                f'number of points: {pcd.get_xyz.shape[0]}'))

            opacity_reset_flag = False
            # Perform opacity reset
            if iter % self.opacity_reset_interval == 0:
                # Reset the opacity of the gaussians to 0.01 (default)
                pcd.reset_opacity(optimizer=optimizer, prefix='sampler.pcd.')
                log(yellow_slim('Resetting opacity done! ' +
                                f'min opacity: {pcd.get_opacity.min().item():.4f} ' +
                                f'max opacity: {pcd.get_opacity.max().item():.4f}'))
                opacity_reset_flag = True

                if iter > self.opacity_reset_interval and iter > self.render_reflection_start_iter:
                    # Reset the specular of the gaussians to 0.001 (default)
                    pcd.reset_specular(
                        reset_specular=self.init_specular,
                        reset_specular_all=self.reset_specular_all,
                        optimizer=optimizer,
                        prefix='sampler.pcd.'
                    )
                    log(yellow_slim('Resetting specular done! ' +
                                    f'min specular: {pcd.get_specular.min().item():.4f} ' +
                                    f'max specular: {pcd.get_specular.max().item():.4f}'))

            if self.opacity_lr0_interval > 0 and iter % self.opacity_lr0_interval == 0 and self.render_reflection_start_iter < iter <= self.normal_prop_until_iter:
                pcd.update_learning_rate_by_name(
                    name='_opacity',
                    lr=self.opacity_lr,
                    optimizer=optimizer,
                    prefix='sampler.pcd.'
                )

            # Make individual color sabotage
            if self.render_reflection_start_iter < iter <= self.color_sabotage_until_iter and iter % self.color_sabotage_interval == 0 and not opacity_reset_flag:
                pcd.distort_color(optimizer=optimizer, prefix='sampler.pcd.')

            if self.render_reflection_start_iter < iter <= self.normal_prop_until_iter and iter % self.normal_prop_interval == 0 and not opacity_reset_flag:
                # Reset the opacity of the gaussians to 0.9 (default)
                pcd.enlarge_opacity(optimizer=optimizer, prefix='sampler.pcd.')
                pcd.enlarge_scaling(optimizer=optimizer, prefix='sampler.pcd.')
                if self.opacity_lr0_interval > 0 and iter != self.normal_prop_until_iter:
                    pcd.update_learning_rate_by_name(
                        name='_opacity',
                        lr=0.0,
                        optimizer=optimizer,
                        prefix='sampler.pcd.'
                    )

    @torch.no_grad()
    def update_env_gaussians(self, batch: dotdict):
        if not self.training: return

        # Log the total number of gaussians
        scalar_stats = batch.output.get('scalar_stats', dotdict())
        scalar_stats.env_num_pts = self.env.number
        batch.output.scalar_stats = scalar_stats

        # Prepare global variables
        iter: int = batch.meta.iter  # controls whether we're to update in this iteration
        output = self.last_output_env  # contains necessary information for updating gaussians
        optimizer: Adam = cfg.runner.optimizer
        # Return if we're not in the update iteration
        if iter <= self.render_reflection_start_iter: return

        # Update the learning rate
        self.env.update_learning_rate(iter.item(), optimizer, prefix='sampler.env.')

        # Increase the levels of SHs every `self.sh_update_iter=1000` iterations until a maximum degree
        if iter > 0 and iter < self.env_densify_until_iter and iter % self.env_sh_update_iter == 0 and self.env_sh_start_iter is not None and iter > self.env_sh_start_iter:
            changed = self.env.oneupSHdegree()
            if changed: log(green_slim(f'[ONEUP SH DEGREE] sh_deg: {self.env.active_sh_degree.item()}'))

        # Update only the rendered frame
        if iter > 0 and iter < self.env_densify_until_iter and output is not None:
            # Update all rendered gaussians in the batch
            env: GaussianModel = self.env

            # Preparing gaussian status for update
            visibility_filter = output.visibility_filter
            viewspace_point_tensor = output.viewspace_points  # no indexing, otherwise no grad # !: BATCH
            if output.viewspace_points.grad is None: return  # previous rendering was an evaluation
            if 'weight_accumulate' not in output: env.add_densification_stats(viewspace_point_tensor, visibility_filter)
            else: env.add_densification_stats(viewspace_point_tensor, visibility_filter, output.weight_accumulate)

            # Perform densification and pruning
            if iter > self.env_densify_from_iter and iter % self.env_densification_interval == 0:
                log(green_slim(f'Start updating gaussians of step: {iter:06d}'))
                # Iteration-related densification and pruning parameters
                env_split_screen_threshold = self.env_split_screen_threshold if iter < self.env_screen_until_iter else None
                env_max_screen_threshold = self.env_max_screen_threshold if iter > self.env_opacity_reset_interval else None
                # Perform actual densification and pruning
                env.densify_and_prune(
                    self.env_min_opacity,
                    self.env_min_gradient,
                    self.env_densify_grad_threshold,
                    self.env_densify_size_threshold,
                    env_split_screen_threshold,
                    self.env_max_scene_threshold,
                    env_max_screen_threshold,
                    self.env_min_weight_threshold,
                    self.env_prune_visibility,
                    optimizer,
                    self.env_prune_large_gs,
                    prefix='sampler.env.'
                )
                log(green_slim('Densification and pruning done! ' +
                                f'min opacity: {env.get_opacity.min().item():.4f} ' +
                                f'max opacity: {env.get_opacity.max().item():.4f} ' +
                                f'number of points: {env.get_xyz.shape[0]}'))

            # Perform opacity reset
            if iter % self.env_opacity_reset_interval == 0:
                env.reset_opacity(optimizer=optimizer, prefix='sampler.env.')
                log(green_slim('Resetting opacity done! ' +
                                f'min opacity: {env.get_opacity.min().item():.4f} ' +
                                f'max opacity: {env.get_opacity.max().item():.4f}'))

    def store_dif_gaussian_output(self, middle: dotdict, batch: dotdict):
        # Reshape and permute the middle output
        middle = self.store_gaussian_output(middle, batch)

        output = dotdict()
        # Store the output for supervision and visualization
        output.acc_map       = middle.acc_map         # (B, P, 1)
        output.dpt_map       = middle.dpt_map         # (B, P, 1)
        output.norm_map      = middle.norm_map        # (B, P, 3)
        output.dist_map      = middle.dist_map        # (B, P, 1)
        output.surf_norm_map = middle.surf_norm_map   # (B, P, 3)
        output.bg_color      = torch.full_like(output.norm_map, self.bg_brightness)  # only for training and comparing with gt
        # Reflectance related outputs
        if self.render_reflection and 'specular' in middle:
            output.spec_map  = middle.spec_map        # (B, P, 1)
            output.rough_map = middle.rough_map       # (B, P, 1)
        # The diffuse RGB output
        output.dif_rgb_map   = middle.rgb_map.clone()  # (B, P, 3)
        output.rgb_map       = middle.rgb_map          # (B, P, 3)

        # Don't forget the iteration number for later supervision retrieval
        output.iter = batch.meta.iter
        return output

    def get_reflect_rays(self, ray_o: torch.Tensor, ray_d: torch.Tensor, coords: torch.Tensor,
                         output: dotdict, batch: dotdict):
        # Compute the reflected rays direction, -d+d' = -2(d·n)n -> d' = d - 2(d·n)n
        norm = normalize(output.norm_map)  # (B, P, 3)
        ref_d = ray_d - 2 * torch.sum(ray_d * norm, dim=-1, keepdim=True) * norm  # (B, P, 3)

        # Compute the surface coordinate as the intersection point
        ref_o = ray_o + ray_d * output.dpt_map  # (B, P, 3)

        # Store the reflected rays for later supervision
        output.ref_o = ref_o  # (B, P, 3)
        output.ref_d = ref_d  # (B, P, 3)

        # Prepare for multi-sampling and specular filtering
        is_specular_filtering = self.specular_filtering_start_iter > 0 and batch.meta.iter >= self.specular_filtering_start_iter
        is_acc_filtering = self.acc_filtering_start_iter > 0 and batch.meta.iter >= self.acc_filtering_start_iter
        H, W = batch.meta.H[0].item(), batch.meta.W[0].item()

        if is_specular_filtering or is_acc_filtering:
            # Only perform reflection tracing on pixels with high specular values or accumulated weights
            if is_specular_filtering:
                ref_msk = output.spec_map[..., 0] > torch.quantile(output.spec_map[..., 0], self.specular_filtering_percent)
            else:
                ref_msk = output.acc_map[..., 0] > 0.75
            ref_o = ref_o[ref_msk][None]  # (N, S, 3)
            ref_d = ref_d[ref_msk][None]  # (N, S, 3)
            # Store the specular mask for later scattering
            output.ref_msk = ref_msk  # (B, P)

        if not (is_specular_filtering or is_acc_filtering):
            # This branch is for compatibility with the original code
            ref_o = ref_o.reshape(H, W, 3)  # (H, W, 3)
            ref_d = ref_d.reshape(H, W, 3)  # (H, W, 3)

        if self.detach: return ref_o.detach(), ref_d.detach()
        else: return ref_o, ref_d

    def store_env_gaussian_output(self, middle: dotdict, output: dotdict, batch: dotdict):
        # Reshape and permute the middle output
        middle = self.store_gaussian_output(middle, batch)

        # Prepare for multi-sampling and specular filtering
        is_specular_filtering = self.specular_filtering_start_iter > 0 and batch.meta.iter >= self.specular_filtering_start_iter
        is_acc_filtering = self.acc_filtering_start_iter > 0 and batch.meta.iter >= self.acc_filtering_start_iter

        if is_specular_filtering or is_acc_filtering:
            # Update the RGB output with the specular or accumulated weight filtering
            rgb_map = middle.rgb_map[0]
            output.rgb_map[output.ref_msk] = (1 - output.spec_map[output.ref_msk]) * output.rgb_map[output.ref_msk] + output.spec_map[output.ref_msk] * rgb_map
            ref_rgb_map = torch.zeros_like(output.rgb_map)
            ref_rgb_map[output.ref_msk] = rgb_map
            output.ref_rgb_map = ref_rgb_map  # (B, P, 3)
        else:
            # Update the RGB output with the reflection
            output.rgb_map = (1 - output.spec_map) * output.rgb_map + output.spec_map * middle.rgb_map
            output.ref_rgb_map = middle.rgb_map  # (B, P, 3)

        # Store the environment Gaussian output for supervision
        output.env_opacity = self.env.get_opacity  # (P, 1)
        return output

    def forward(self, batch: dotdict):
        # Maybe update diffuse Gaussians: densification & pruning
        self.update_dif_gaussians(batch)
        # Maybe update environment Gaussians: densification & pruning
        self.update_env_gaussians(batch)

        # Prepare the camera transformation for Gaussian
        viewpoint_camera = to_x(prepare_gaussian_camera(batch), torch.float)

        # Compute the caemra ray origins and directions, and reflected rays
        ray_o, ray_d, coords, _, _, _ = self.get_camera_rays(
            batch,
            n_rays=self.n_rays,
            patch_size=self.patch_size
        )
        # Shape things
        H, W = batch.meta.H[0].item(), batch.meta.W[0].item()

        # Invoke hardware ray tracer
        if self.use_base_tracing:
            if self.tracing_backend == 'cpp':
                dif_output = self.diffop.render_gaussians(
                    viewpoint_camera,
                    ray_o.reshape(H, W, 3),
                    ray_d.reshape(H, W, 3),
                    self.pcd,
                    self.pipe,
                    self.bg_color,
                    0,
                    self.specular_threshold,
                    scaling_modifier=self.scale_mod,
                    override_color=None,
                    batch=batch
                )
            else:
                raise ValueError(f'Unknown tracing backend: {self.tracing_backend}')
        # Rasterize diffuse Gaussians to image, obtain their radii (on screen)
        else:
            dif_output = self.render_gaussians(
                viewpoint_camera,
                self.pcd,
                self.pipe,
                self.bg_color,
                self.scale_mod,
                override_color=None
            )

        # Retain diffuse Gaussian gradients after updates
        # Skip saving the output if not in training mode to avoid unexpected densification skipping caused by `None` gradient
        if self.training: self.last_output = dif_output
        # Prepare output for supervision and visualization
        output = self.store_dif_gaussian_output(dif_output, batch)

        if batch.meta.iter >= self.render_reflection_start_iter:
            # Compute the reflected rays origins and directions
            ref_o, ref_d = self.get_reflect_rays(ray_o, ray_d, coords, output, batch)

            # Invoke hardware ray tracer
            if self.tracing_backend == 'cpp':
                env_output = self.diffop.render_gaussians(
                    viewpoint_camera,
                    ref_o,
                    ref_d,
                    self.env,
                    self.pipe_env,
                    self.env_bg_color,
                    0,
                    start_from_first=False,
                    scaling_modifier=self.scale_mod,
                    override_color=None,
                    batch=batch
                )
            else:
                raise ValueError(f'Unknown tracing backend: {self.tracing_backend}')

            # Retain gradients after updates
            # Skip saving the output if not in training mode to avoid unexpected densification skipping caused by `None` gradient
            if self.training: self.last_output_env = env_output

            # Prepare output for supervision and visualization
            output = self.store_env_gaussian_output(env_output, output, batch)

        # Update the output to the batch
        batch.output.update(output)
