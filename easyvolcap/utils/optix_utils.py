import math
import torch
from torch import nn
import torch.utils.cpp_extension

# Maybe lazy import this
from diff_surfel_tracing import SurfelTracer, SurfelTracingSettings

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.sh_utils import eval_sh
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.math_utils import normalize
from easyvolcap.utils.nerf_utils import volume_rendering
from easyvolcap.utils.gaussian2d_utils import GaussianModel, dpt2norm


class HardwareRendering(nn.Module):
    def __init__(self,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Create OptiX context
        self.tracer = SurfelTracer()

        # BVH indicator
        self.has_bvh = False

        # Some tracer global configurations, remember to synchrone with the CUDA code
        self.mid_channel = 16  # 3 ray_o + 3 ray_d + 1 dpt + 1 acc + 3 normal + AUX_CHANNELS + NUM_CHANNELS
        self.rayo_off = 0  # ray origin offset
        self.rayd_off = 3  # ray direction offset
        self.dpt_off = 6  # depth offset
        self.acc_off = 7  # accumulated weight offset
        self.norm_off = 8  # normal offset
        self.aux_off = 11  # auxiliary offset
        self.rgb_off = 13  # RGB offset

    def get_disks(self, pcd: GaussianModel):
        # Build the uv tangent plane to world transformation matrix, splat2world
        T = pcd.get_covariance()  # (P, 4, 4)
        T = T.permute(0, 2, 1)  # (P, 4, 4)
        T[..., 2] = 0  # (P, 4, 4)

        # Deal with nasty shapes
        P = T.shape[0]
        V = 4  # 1 2DGS <-> 2 triangles <-> 4 vertices

        # 3-sigma range in local uv splat coordiantes
        sigma3 = torch.as_tensor([[-1., 1.], [-1., -1.], [1., 1.], [1., -1.]], device=T.device) * 3  # (V, 2)
        # sigma3 = torch.as_tensor([[0., 1.], [0., -1.], [1., 0.], [-1., 0.]], device=T.device)  # (V, 2)
        sigma3 = torch.cat([sigma3, torch.ones_like(sigma3)], dim=-1)  # (V, 4)
        # Expand
        sigma3 = sigma3[None].repeat(P, 1, 1)  # (P, V, 4)
        # sigma3[..., :2] = sigma3[..., :2] * pcd.get_scaling[:, None]  # (P, V, 4)
        T = T[:, None].expand(-1, V, -1, -1)  # (P, V, 4, 4)

        # Convert the vertices to the world coordinate
        v = T.reshape(-1, 4, 4) @ sigma3.reshape(-1, 4, 1)  # (P * V, 4, 1)
        v = v[..., :3, 0]  # (P * V, 3)

        # Generate face indices
        indices = torch.arange(0, v.shape[0]).reshape(P, V).to(T.device)  # (P, V)
        f = torch.stack([indices[:, :3], indices[:, 1:]], dim=1).reshape(-1, 3).int()  # (P, 2, 3) -> (P * 2, 3)

        # NOTE: contiguous() is necessary for the following OptiX CUDA operations !!!
        v, f = v.contiguous(), f.contiguous()

        return v, f

    def build_bvh(self, pcd: GaussianModel, batch: dotdict, rebuild: int = 1):
        # Train-time, rebuild-time or the first test-time build
        if rebuild or self.training or (not self.training and not self.has_bvh):
            # Convert 2DGS to triangles
            v, f = self.get_disks(pcd)

            # Build the BVH
            self.tracer.build_acceleration_structure(v.detach().clone(), f.detach().clone(), rebuild=True)

            # Update the flag
            self.has_bvh = True if not self.training else False
        else:
            v, f = None, None

        return v, f

    def render_gaussians(self,
                         camera: dotdict,
                         ray_o: torch.Tensor,
                         ray_d: torch.Tensor,
                         pcd: GaussianModel,
                         pipe: dotdict,
                         bg_color: torch.Tensor,
                         max_trace_depth: int = 0,
                         specular_threshold: float = 0.0,
                         start_from_first: bool = True,
                         scaling_modifier: int = 1,
                         override_color: torch.Tensor = None,
                         batch: dotdict = None,
                         ):

        # Setup tracer configuration
        # NOTE: `.contiguous()` is important to avoid weird CUDA errors
        tracer_settings = SurfelTracingSettings(
            image_height=int(camera.image_height),
            image_width=int(camera.image_width),
            tanfovx=math.tan(camera.FoVx * 0.5),
            tanfovy=math.tan(camera.FoVy * 0.5),
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=camera.world_view_transform.contiguous(),
            projmatrix=camera.full_proj_transform.contiguous(),
            sh_degree=pcd.active_sh_degree,
            campos=camera.camera_center.contiguous(),
            prefiltered=False,
            debug=False,
            max_trace_depth=max_trace_depth,
            specular_threshold=specular_threshold
        )

        # Build the acceleration structure first
        v, f = self.build_bvh(pcd, batch, rebuild=self.training)

        # Ensure the ray direction is normalized, and the ray origin and direction are contiguous
        # NOTE: do not normalize the ray direction, it must be in z_depth
        ray_o = ray_o.contiguous()
        ray_d = ray_d.contiguous()

        # Get the means, opacities, and colors of the Gaussians
        means3D = pcd.get_xyz.contiguous()
        opacities = pcd.get_opacity.contiguous()

        # Create zero tensor, we will use it to make pytorch return gradients of the 3D means
        grads3D = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device=means3D.device) + 0
        try: grads3D.retain_grad()
        except: pass

        # If precomputed 3d covariance is provided, use it
        # If not, then it will be computed from scaling / rotation by the rasterizer
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            # NOTE: Currently don't support normal consistency loss if use precomputed covariance
            splat2world = pcd.get_covariance(scaling_modifier)
            W, H = camera.image_width, camera.image_height
            near, far = camera.znear, camera.zfar
            ndc2pix = torch.tensor([
                [W / 2, 0, 0, (W-1) / 2],
                [0, H / 2, 0, (H-1) / 2],
                [0, 0, far-near, near],
                [0, 0, 0, 1]]).float().cuda().T
            world2pix =  camera.full_proj_transform @ ndc2pix
            cov3D_precomp = (splat2world[:, [0, 1, 3]] @ world2pix[:, [0, 1, 3]]).permute(0, 2, 1).reshape(-1, 9).contiguous()  # `glm` is column major
        else:
            scales = pcd.get_scaling.contiguous()
            rotations = pcd.get_rotation.contiguous()

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer
        shs = None
        colors_precomp = None
        if override_color is None or (pcd.render_reflection and pcd.feature_splatting):
            if pipe.convert_SHs_python:
                shs_view = pcd.get_features.transpose(1, 2).view(-1, 3, (pcd.max_sh_degree+1)**2)
                dir_pp = (pcd.get_xyz - camera.camera_center.repeat(pcd.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pcd.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0).contiguous()
            else:
                shs = pcd.get_features.contiguous()
        else:
            colors_precomp = override_color.contiguous()

        # Maybe render other properties
        others_precomp = None
        if pcd.render_reflection:
            # This is a hack to make the reflection rendering work, only for debugging
            if pcd.get_specular.shape[0] != means3D.shape[0]:
                others_precomp = torch.cat([
                    torch.ones_like(opacities) * pcd.init_specular, torch.ones_like(opacities) * pcd.init_roughness], 
                dim=-1).contiguous()  # (P, 2)
            # For reflection rendering, we need to render specular and (roughness, no use for now)
            else:
                others_precomp = torch.cat([pcd.get_specular, pcd.get_roughness], dim=-1).contiguous()  # (P, 2)

        # Perform hardware ray tracing
        rgb, dpt, acc, norm, dist, aux, mid, wet = self.tracer(
            ray_o, ray_d, v,
            means3D=means3D,
            grads3D=grads3D,
            shs=shs,
            colors_precomp=colors_precomp,
            others_precomp=others_precomp,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            tracer_settings=tracer_settings,
            start_from_first=start_from_first
        )

        # Compute the visibility filter using camera projection
        with torch.no_grad():
            # Default account for the contributed Gaussians outside the image plane
            visibility_filter = (wet[..., 0] > 0.0)  # (P,)
            # If we start from the caemra space tracing, we need to account for
            # the contributed Gaussians inside the image plane
            if start_from_first:
                uvd = (batch.K @ (batch.R @ means3D[..., None] + batch.T))[..., 0]  # (P, 3)
                uvd[..., :2] = uvd[..., :2] / uvd[..., 2:]  # (P, 3)
                vis = (uvd[..., 2] >= 0.2) & (uvd[..., 0] >= 0.0) & (uvd[..., 0] <= camera.image_width) & (uvd[..., 1] >= 0.0) & (uvd[..., 1] <= camera.image_height)
                visibility_filter = visibility_filter | vis

        # Prepare output dict
        output = dotdict(
            viewspace_points=grads3D,  # (P, 3)
            visibility_filter=visibility_filter.detach().clone(),  # (P,)
            weight_accumulate=wet,  # (P, 1)
            render=rgb.permute(2, 0, 1),  # (3, H, W)
            rend_alpha=acc.permute(2, 0, 1),  # (1, H, W)
            rend_normal=norm.permute(2, 0, 1),  # (3, H, W)
            rend_dist=dist.permute(2, 0, 1),  # (1, H, W)
            surf_depth=dpt.permute(2, 0, 1)  # (1, H, W)
        )

        if start_from_first:
            # Assume the depth points form the 'surface' and generate psudo surface normal for regularizations
            surf_norm = dpt2norm(camera, dpt.permute(2, 0, 1))  # (H, W, 3)
            # Remember to multiply with accum_alpha since render_normal is unnormalized
            surf_norm = surf_norm * acc.detach()  # (H, W, 3)
            # Update the output
            output.surf_normal = surf_norm.permute(2, 0, 1)  # (3, H, W)
        else:
            output.surf_normal = torch.zeros_like(norm).permute(2, 0, 1)  # (3, H, W)

        # Return the reflection properties if needed
        if pcd.render_reflection:
            output.update(dotdict(
                specular=aux[..., :pcd.specular_channels].permute(2, 0, 1),  # (1, H, W)
                roughness=aux[..., pcd.specular_channels:pcd.specular_channels+1].permute(2, 0, 1))  # (1, H, W)
            )

        # Return the rendered RGB map of different tracing depths
        if max_trace_depth > 0:
            # Return the first-reflected ray direction and origin
            cur_off = self.mid_channel
            output.update(dotdict(
                ray_o=ray_o.permute(2, 0, 1),  # (3, H, W)
                ray_d=ray_d.permute(2, 0, 1),  # (3, H, W)
                ref_o=mid[..., self.rayo_off+cur_off:self.rayo_off+3+cur_off].permute(2, 0, 1),  # (3, H, W)
                ref_d=mid[..., self.rayd_off+cur_off:self.rayd_off+3+cur_off].permute(2, 0, 1))  # (3, H, W)
            )

            # Return the rendered results of different tracing depths
            stages = dotdict(rgb_map=[], dpt_map=[], acc_map=[], norm_map=[])
            # NOTE: The first stage is the original rendering
            for i in range(max_trace_depth + 1):
                cur_off = self.mid_channel * i
                stages.rgb_map.append( mid[..., self.rgb_off +cur_off:self.rgb_off +3+cur_off].permute(2, 0, 1))  # (3, H, W)
                stages.dpt_map.append( mid[..., self.dpt_off +cur_off:self.dpt_off +1+cur_off].permute(2, 0, 1))  # (1, H, W)
                stages.acc_map.append( mid[..., self.acc_off +cur_off:self.acc_off +1+cur_off].permute(2, 0, 1))  # (1, H, W)
                stages.norm_map.append(mid[..., self.norm_off+cur_off:self.norm_off+3+cur_off].permute(2, 0, 1))  # (3, H, W)
            # Return the stages
            output.stages = stages

        return output
