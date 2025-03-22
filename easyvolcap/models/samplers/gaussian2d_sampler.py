import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F

from easyvolcap.engine import cfg
from easyvolcap.engine import SAMPLERS
from easyvolcap.models.networks.noop_network import NoopNetwork

from easyvolcap.utils.sh_utils import *
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.chunk_utils import multi_gather
from easyvolcap.utils.patch_utils import sample_patch
from easyvolcap.utils.grid_utils import sample_points_bbox
from easyvolcap.utils.optix_utils import HardwareRendering
from easyvolcap.utils.ray_utils import get_rays, weighted_sample_rays
from easyvolcap.utils.bound_utils import get_near_far_aabb, monotonic_near_far
from easyvolcap.utils.gaussian2d_utils import GaussianModel, render, prepare_gaussian_camera
from easyvolcap.utils.net_utils import VolumetricVideoModule, typed, update_optimizer_state, make_buffer
from easyvolcap.utils.data_utils import load_pts, export_pts, to_x, to_cuda, to_cpu, to_tensor, remove_batch, save_image
from easyvolcap.utils.colmap_utils import read_points3D_binary_custom, read_points3D_text_custom, load_sfm_ply, save_sfm_ply


@SAMPLERS.register_module()
class Gaussian2DSampler(VolumetricVideoModule):
    def __init__(self,
                 # Legacy APIs
                 network: NoopNetwork = None,  # ignore this

                 # Gaussian configs
                 preload_gs: str = '',
                 xyz_lr_scheduler: dotdict = None,

                 # sh configs
                 sh_deg: int = 3,
                 init_sh_deg: int = 0,
                 sh_start_iter: int = 0,
                 sh_update_iter: int = 1000,

                 # Opacity and scale configs
                 init_occ: float = 0.1,
                 bounds: List[List[float]] = [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
                 spatial_scale: float = 1.0,

                 # Densify & pruning configs
                 densify_from_iter: int = 500,
                 densify_until_iter: int = 15000,
                 densification_interval: int = 100,
                 opacity_reset_interval: int = 3000,
                 densify_grad_threshold: float = 0.0002,
                 min_opacity: float = 0.05,
                 densify_size_threshold: float = 0.01,  # alias for `percent_dense` as in the original code, https://github.com/hbb1/2d-gaussian-splatting/blob/6d249deeec734ad07760496fc32be3b91ac236fc/scene/gaussian_model.py#L378
                 prune_large_gs: bool = True,
                 prune_visibility: bool = False,
                 max_scene_threshold: float = 0.1,  # default 0.1, same as the original 2DGS
                 max_screen_threshold: float = None,  # not used in the original 3DGS/2DGS, they wrote a bug, though `max_screen_threshold=20`
                 min_weight_threshold: float = None,
                 # EasyVolcap additional densify & pruning tricks
                 screen_until_iter: int = int(4000 / 60 * cfg.runner_cfg.epochs),
                 split_screen_threshold: float = None,
                 min_gradient: float = None,

                 # Reflection related parameters
                 render_reflection: bool = False,  # default is False here
                 specular_channels: int = 1,  # default is 1 here
                 init_specular: float = 1e-3,  # specular initialization
                 init_roughness: float = 0.5,  # roughness initialization
                 use_z_depth: bool = True,  # normalized direction since no sampling on the original ray
                 correct_pix: bool = True,
                 # Placed here only to remove warnings, default disabled, only enabled for debugging use
                 n_rays: int = -1,  # number of rays to sample for reflection
                 patch_size: List[int] = [-1, -1],  # patch size to sample for reflection
                 patch_exact: bool = True,

                 # Rendering configs
                 compute_cov3D_python: bool = False,  # used in the original renderer
                 convert_SHs_python: bool = False,  # used in the original renderer
                 debug: bool = False,  # used in the original renderer
                 depth_ratio: float = 0.0,  # used in the original renderer
                 white_bg: bool = False,  # used in the original renderer
                 bg_brightness: float = 0.0,  # used in the original renderer
                 scale_mod: float = 1.0,

                 # Ray tracing configs
                 use_optix_tracing: bool = False,
                 tracing_backend: str = 'cpp',
                 max_gs: int = 2e6,  # control the maximum number of gaussians
                 max_gs_threshold: float = 0.9,  # percentage of the visibility pruning
                 max_trace_depth: int = 0,
                 specular_threshold: float = 0.0,  # specular threshold for reflection rendering

                 # Housekeepings
                 **kwargs,
                 ):
        super().__init__(network)

        # Gaussian configs
        self.preload_gs = preload_gs
        self.xyz_lr_scheduler = xyz_lr_scheduler

        # SHs configs
        self.sh_deg = sh_deg
        self.init_sh_deg = init_sh_deg
        self.sh_start_iter = sh_start_iter
        self.sh_update_iter = sh_update_iter

        # Scale and opacity configs
        self.init_occ = init_occ
        self.bounds = bounds
        self.spatial_scale = spatial_scale

        # Densify configs
        self.densify_from_iter = densify_from_iter
        self.densify_until_iter = densify_until_iter
        self.densification_interval = densification_interval
        self.opacity_reset_interval = opacity_reset_interval
        self.densify_grad_threshold = densify_grad_threshold
        self.min_opacity = min_opacity
        self.densify_size_threshold = densify_size_threshold
        self.prune_large_gs = prune_large_gs
        self.prune_visibility = prune_visibility
        self.max_scene_threshold = max_scene_threshold
        self.max_screen_threshold = max_screen_threshold
        self.min_weight_threshold = min_weight_threshold
        self.max_gs = max_gs
        self.max_gs_threshold = max_gs_threshold
        # EasyVolcap additional densify & pruning tricks
        self.screen_until_iter = screen_until_iter
        self.split_screen_threshold = split_screen_threshold
        self.min_gradient = min_gradient
        # Store the last output for updating the gaussians
        self.last_output = None

        # Reflection rendering configs
        self.render_reflection = render_reflection
        self.specular_channels = specular_channels
        self.init_specular = init_specular
        self.init_roughness = init_roughness
        self.use_z_depth = use_z_depth
        self.correct_pix = correct_pix
        # Placed here only to remove warnings, default disabled, only enabled for debugging use
        self.n_rays = n_rays
        self.patch_size = patch_size
        self.patch_exact = patch_exact

        # Load the initial point cloud from the SfM point cloud
        xyz, colors = self.init_points(self.preload_gs)
        self.pcd = GaussianModel(
            xyz=xyz, colors=colors,
            init_occ=self.init_occ,
            init_scale=None,
            sh_degree=self.sh_deg,
            init_sh_degree=self.init_sh_deg,
            spatial_scale=self.spatial_scale,
            xyz_lr_scheduler=self.xyz_lr_scheduler,
            render_reflection=self.render_reflection,
            specular_channels=self.specular_channels,
            init_specular=self.init_specular,
            init_roughness=self.init_roughness,
            max_gs=self.max_gs,
            max_gs_threshold=self.max_gs_threshold
        )

        # Rendering configs
        self.pipe = dotdict({
            'convert_SHs_python': True if self.render_reflection and not use_optix_tracing else convert_SHs_python,
            'compute_cov3D_python': compute_cov3D_python,
            'debug': debug,
            'depth_ratio': depth_ratio,
        })
        self.white_bg = white_bg
        self.bg_brightness = 1. if white_bg else bg_brightness
        self.bg_channel = 3 + 2 * self.render_reflection * (1 - use_optix_tracing)
        self.bg_color = make_buffer(torch.Tensor([self.bg_brightness] * self.bg_channel))
        self.scale_mod = scale_mod

        # Rendering function alias
        self.render_gaussians = render

        # Ray tracing configs
        self.use_optix_tracing = use_optix_tracing
        if self.use_optix_tracing:
            self.diffop = HardwareRendering()
            # Disable the screen pruning since tracing does not have it
            self.max_screen_threshold = None
        self.tracing_backend = tracing_backend
        self.max_trace_depth = max_trace_depth
        self.specular_threshold = specular_threshold

        # Debug options
        self.scale_mult = 1.0
        self.alpha_mult = 1.0

    def init_points(self, ply_file: str = None, N: int = 100000):
        # Try to initialize from the SfM point cloud
        if ply_file is not None and not exists(ply_file) and (exists(ply_file.replace(".ply", ".bin")) or exists(ply_file.replace(".ply", ".txt"))):
            log(yellow("Converting point3d.bin to .ply, will happen only the first time you open the scene."))
            try: xyz, rgb, _ = read_points3D_binary_custom(ply_file.replace(".ply", ".bin"))
            except: xyz, rgb, _ = read_points3D_text_custom(ply_file.replace(".ply", ".txt"))
            save_sfm_ply(ply_file, xyz, rgb)

        try:
            # Try to load the ply data from the provided file
            xyz, rgb = load_sfm_ply(ply_file)
            log(yellow(f"Loaded the point cloud from {ply_file}."))
            xyz = torch.as_tensor(xyz, dtype=torch.float)
            rgb = torch.as_tensor(rgb, dtype=torch.float)  # already normalized to [0, 1]

        except:
            # If the file does not exist, generate random points and save them
            log(yellow(f"Failed to load the point cloud from {ply_file}, generating random points."))
            xyz = sample_points_bbox(torch.as_tensor(self.bounds), N).float()
            rgb = torch.rand(xyz.shape, dtype=torch.float)  # the initialization should be RGB, not SHs
            save_sfm_ply(ply_file, xyz.numpy(), rgb.numpy() * 255.0)  # convert SHs to RGB before saving

        return xyz, rgb

    def extract_input(self, batch: dotdict, n_rays: int = -1, patch_size: List[int] = [-1, -1]):
        bounds: torch.Tensor = batch.bounds  # (B, 2, 3)
        n: torch.Tensor = batch.n  # (B,)
        f: torch.Tensor = batch.f  # (B,)
        t: torch.Tensor = batch.t  # (B,)
        H, W = batch.meta.H[0].item(), batch.meta.W[0].item()  # !: BATCH

        def process_input(ray_o, ray_d):
            near, far = get_near_far_aabb(bounds, ray_o, ray_d)
            near, far = monotonic_near_far(near, far, n, f)
            time = t[..., None, None].expand(-1, *ray_o.shape[1:-1], 1)  # (B, P, 1)
            return near, far, time

        # Get different sampling conditions of current invoking
        should_sample_whole = not self.training or (n_rays < 0 and patch_size[0] < 0)
        should_sample_patch = self.training and patch_size[0] > 0
        should_sample_rands = self.training and n_rays > 0
        # Check the sampling mode, cannot be in two modes at the same time
        assert not (should_sample_rands and should_sample_patch), 'You must choose one mode between [rays or patch]'
        # Check if the rays are already computed, if so, we can skip the ray computation
        could_skip_rays = 'ray_o' in batch and ((should_sample_whole and batch.ray_o.shape[1] == H * W) or \
                                                (should_sample_rands and batch.ray_o.shape[1] == n_rays) or \
                                                (should_sample_patch and batch.ray_o.shape[1] == patch_size[0] * patch_size[1]))

        # Use the provided camera rays if available
        if could_skip_rays:
            ray_o, ray_d, coords = batch.ray_o, batch.ray_d, batch.coords
            # Doesn't need to recompute the near, far, time, if already computed
            if 'time' in batch: near, far, time = batch.near, batch.far, batch.time
            # Otherwise, recompute the near, far, time since we actually skip the `VolumetricVideoModel`
            else: near, far, time = process_input(ray_o, ray_d)

        # Compute the camera rays of the whole image
        elif should_sample_whole:
            K, R, T = batch.K, batch.R, batch.T  # !: BATCH
            ray_o, ray_d, coords = get_rays(H, W, K, R, T, z_depth=self.use_z_depth, correct_pix=self.correct_pix, ret_coord=True)  # (B, H, W, 3), (B, H, W, 3), (B, H, W, 2
            ray_o, ray_d, coords = ray_o.view(-1, H * W, 3), ray_d.view(-1, H * W, 3), coords.view(-1, H * W, 2)
            near, far, time = process_input(ray_o, ray_d)

        # Randomly sample a patch of size `patch_size`
        elif should_sample_patch:
            K, R, T = batch.K, batch.R, batch.T  # !: BATCH
            ray_o, ray_d, coords = get_rays(H, W, K, R, T, z_depth=self.use_z_depth, correct_pix=self.correct_pix, ret_coord=True)  # (B, H, W, 3), (B, H, W, 3), (B, H, W, 2
            x, y, w, h = sample_patch(H, W, patch_size=patch_size, exact=self.patch_exact)
            ray_o, ray_d, coords = ray_o[:, y:y + h, x:x + w], ray_d[:, y:y + h, x:x + w], coords[:, y:y + h, x:x + w]  # (B, h, w, 3), (B, h, w, 3), (B, h, w, 2)
            ray_o, ray_d, coords = ray_o.reshape(-1, h * w, 3), ray_d.reshape(-1, h * w, 3), coords.reshape(-1, h * w, 2)  # (B, P, 3), (B, P, 3), (B, P, 3)
            near, far, time = process_input(ray_o, ray_d)
            # Store the actual sampled patch size for supervision use
            batch.meta.patch_h, batch.meta.patch_w = torch.as_tensor(h)[None], torch.as_tensor(w)[None]  # NOTE: store patch size for later supervision if needed

        # Randomly sample `n_rays`
        elif should_sample_rands:
            wet = torch.ones((t.shape[0], H, W, 1), device=t.device)  # (B, H, W, 1)
            ray_o, ray_d, coords = weighted_sample_rays(wet, batch.K, batch.R, batch.T, n_rays, self.use_z_depth, self.correct_pix)  # (N, 3); (N, 3); (N, 3); (N, 2)
            ray_o, ray_d, coords = ray_o[None].expand(bounds.shape[0], -1, -1), ray_d[None].expand(bounds.shape[0], -1, -1), coords[None].expand(bounds.shape[0], -1, -1)
            near, far, time = process_input(ray_o, ray_d)

        # Store the computed rays for later use
        batch.ray_o, batch.ray_d, batch.coords = ray_o, ray_d, coords
        batch.near, batch.far = near, far
        batch.time = time

        return batch

    def get_camera_rays(self, batch: dotdict, n_rays: int = -1, patch_size: List[int] = [-1, -1]):
        # Extract the input data, including the camera rays origin, direction and shrinked near, far range
        # Since we are in `let_user_handle_input=True` mode here, so we need to extract the input data manually
        batch = self.extract_input(batch, n_rays=n_rays, patch_size=patch_size)

        # Fetch out the camera rays
        ray_o, ray_d, coords = batch.ray_o, batch.ray_d, batch.coords  # (B, P, 3), (B, P, 3), (B, P, 2)
        near, far, t = batch.near, batch.far, batch.time  # (B, P, 1), (B, P, 1), (B, P, 1)

        return ray_o, ray_d, coords, near, far, t

    @torch.no_grad()
    def update_gaussians(self, batch: dotdict):
        if not self.training: return

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

            # Perform opacity reset
            if iter % self.opacity_reset_interval == 0:
                pcd.reset_opacity(optimizer=optimizer, prefix='sampler.pcd.')
                log(yellow_slim('Resetting opacity done! ' +
                                f'min opacity: {pcd.get_opacity.min().item():.4f} ' +
                                f'max opacity: {pcd.get_opacity.max().item():.4f}'))

    def store_gaussian_output(self, output: dotdict, batch: dotdict):
        # Post-process mainly does two things: add the batch dimension and reshape to the desired (B, P, C) shape
        # Visualization and supervision results processing
        output.acc_map       = output.rend_alpha[None ].permute(0, 2, 3, 1).reshape(1, -1, output.rend_alpha.shape[0] )  # (B, H * W, 1)
        output.dpt_map       = output.surf_depth[None ].permute(0, 2, 3, 1).reshape(1, -1, output.surf_depth.shape[0] )  # (B, H * W, 1)
        output.norm_map      = output.rend_normal[None].permute(0, 2, 3, 1).reshape(1, -1, output.rend_normal.shape[0])  # (B, H * W, 3)
        # Supervision results processing
        output.dist_map      = output.rend_dist[None  ].permute(0, 2, 3, 1).reshape(1, -1, output.rend_dist.shape[0]  )  # (B, H * W, 3)
        output.surf_norm_map = output.surf_normal[None].permute(0, 2, 3, 1).reshape(1, -1, output.surf_normal.shape[0])  
        output.bg_color      = torch.full_like(output.norm_map, self.bg_brightness)  # only for training and comparing with gt

        # Reflection related processing
        if self.render_reflection and 'specular' in output:
            output.spec_map  = output.specular[None   ].permute(0, 2, 3, 1).reshape(1, -1, output.specular.shape[0]   )  # (B, H * W, 3)
            output.rough_map = output.roughness[None  ].permute(0, 2, 3, 1).reshape(1, -1, output.roughness.shape[0]  )  # (B, H * W, 1)

        # RGB color related processing
        rgb = output.render[None].permute(0, 2, 3, 1).reshape(1, -1, output.render.shape[0])  # (B, H * W, 3)
        output.rgb_map = rgb

        # Don't forget the iteration number for later supervision retrieval
        output.iter = batch.meta.iter
        return output

    def forward(self, batch: dotdict):
        # Maybe update: densification & pruning
        self.update_gaussians(batch)

        # Prepare the camera transformation for Gaussian
        viewpoint_camera = to_x(prepare_gaussian_camera(batch), torch.float)

        # Generate the ray origins and directions if performing ray tracing
        # or random ray/patch sampling training strategy when training
        if (self.training and (self.n_rays > 0 or self.patch_size[0] > 0)) or self.use_optix_tracing:
            ray_o, ray_d, coords, _, _, _ = self.get_camera_rays(
                batch,
                n_rays=self.n_rays,
                patch_size=self.patch_size
            )

        # Perform hardware Gaussian tracing
        if self.use_optix_tracing:
            # Reshape the rays to the original shape when perform ray tracing
            H, W = batch.meta.H[0].item(), batch.meta.W[0].item()

            # Invoke hardware ray tracer
            if self.tracing_backend == 'cpp':
                output = self.diffop.render_gaussians(
                    viewpoint_camera,
                    ray_o.reshape(H, W, 3),
                    ray_d.reshape(H, W, 3),
                    self.pcd,
                    self.pipe,
                    self.bg_color,
                    self.max_trace_depth,
                    self.specular_threshold,
                    scaling_modifier=self.scale_mod,
                    override_color=None,
                    batch=batch
                )
            else:
                raise ValueError(f'Unknown tracing backend: {self.tracing_backend}')

        # Rasterize visible Gaussians to image, obtain their radii (on screen)
        else:
            output = self.render_gaussians(
                viewpoint_camera,
                self.pcd,
                self.pipe,
                self.bg_color,
                self.scale_mod,
                override_color=None
            )

        # Retain gradients after updates
        # Skip saving the output if not in training mode to avoid unexpected densification skipping caused by `None` gradient
        if self.training: self.last_output = output

        # Prepare output for supervision and visualization
        output = self.store_gaussian_output(output, batch)

        # Update the output to the batch
        batch.output.update(output)
