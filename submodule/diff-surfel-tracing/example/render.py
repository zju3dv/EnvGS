import os
import json
import math
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from os.path import join, exists

from diff_surfel_tracing import SurfelTracer, SurfelTracingSettings

from utils.ray_utils import get_rays
from utils.gaussian_utils import GaussianModel, Camera
from utils.data_utils import save_image, normalize_depth, visualize_normal, generate_video


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='data/trained_model/truck/latest.ply')
    parser.add_argument('--camera_dir', type=str, default='data/path/truck/path.json')
    parser.add_argument('--result_dir', type=str, default='data/result/truck')
    parser.add_argument('--max_sh_degree', type=int, default=3, help='Maximum spherical harmonics degree')
    parser.add_argument('--white_bg', action='store_true', help='Use white background')
    parser.add_argument('--scale_modifier', type=float, default=1.0, help='Scale modifier for the surfel tracing')
    parser.add_argument('--max_trace_depth', type=int, default=0, help='Maximum trace depth, default 0 means one trace no bounce')
    parser.add_argument('--specular_threshold', type=float, default=0.0, help='Threshold for specular reflection, default 0.0 means no specular reflection')
    parser.add_argument('--vis_ext', type=str, default='.jpg', help='Extension of the visualization images')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for the generated video')
    return parser.parse_args()


def load_nerf_camera(camera: dict):
    # Load camera parameters
    H, W = camera['h'], camera['w']
    K = torch.tensor([
        [camera['fl_x'], 0., camera['cx']],
        [0., camera['fl_y'], camera['cy']],
        [0., 0., 1.]
    ], dtype=torch.float32).cuda()
    c2w = torch.tensor(camera['transform_matrix'], dtype=torch.float32)
    c2w[:3, [1, 2]] *= -1  # Change the coordinate system from OpenCV to PyTorch
    w2c = torch.inverse(c2w)
    R, T = w2c[:3, :3].cuda(), w2c[:3, 3:].cuda()
    return H, W, K, R, T


def main():
    # Parse arguments
    args = parse_args()
    # Set the result directory
    rgb_dir = join(args.result_dir, 'RENDER')
    dpt_dir = join(args.result_dir, 'DEPTH')
    nrm_dir = join(args.result_dir, 'NORMAL')

    # Create the 2D Gaussian model and load the pre-trained model
    pcd = GaussianModel(args.max_sh_degree)
    pcd.load_ply(args.model_path)
    print(f"[INFO] Loaded number of Gaussians: {pcd.number}.")

    # Convert the 2D Gaussian primitives to covering triangles
    v, f = pcd.get_triangles()
    v, f = v.cuda().contiguous(), f.cuda().contiguous()  # .contiguous() is necessary for the following CUDA operations
    print(f"[INFO] Converted number of triangles: {f.shape[0]}")

    # Create the SurfelTracer
    tracer = SurfelTracer()
    # Build the acceleration structure
    tracer.build_acceleration_structure(v.detach().clone(), f.detach().clone(), rebuild=True)
    print('[INFO] Acceleration structure built.')

    # Prepare tracer input, including 2D Gaussian parameters
    means3D = pcd.get_xyz.cuda().contiguous()
    shs = pcd.get_features.cuda().contiguous()
    colors_precomp = None
    others_precomp = None
    opacities = pcd.get_opacity.cuda().contiguous()
    scales = pcd.get_scaling.cuda().contiguous()
    rotations = pcd.get_rotation.cuda().contiguous()
    cov3D_precomp = None
    start_from_first = True

    # Determine the background color
    bg_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).cuda()
    # Create dummy input to receive the gradient for densification
    grads3D = (
        torch.zeros_like(
            means3D, dtype=means3D.dtype, requires_grad=True, device=means3D.device
        ) + 0
    )
    try: grads3D.retain_grad()
    except: pass

    # Load the pre-defined camera paths
    # NOTE: The loaded camera extrinsics are w2c (world-to-camera) transformations
    cameras = json.load(open(args.camera_dir, 'r'))['frames']

    # Actual render
    for i, cam in tqdm(enumerate(cameras), total=len(cameras)):
        # Convert the NeRF format camera to the OpenCV format camera
        H, W, K, R, T = load_nerf_camera(cam)
        viewpoint_camera = Camera(H, W, K, R, T)

        # Generate the ray origins and directions
        ray_o, ray_d = get_rays(H, W, K, R, T)
        ray_o, ray_d = ray_o.cuda().contiguous(), ray_d.cuda().contiguous()

        # Set the surfel tracing settings
        tracer_settings = SurfelTracingSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
            tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
            bg=bg_color,
            scale_modifier=args.scale_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pcd.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
            max_trace_depth=args.max_trace_depth,
            specular_threshold=args.specular_threshold,
        )

        # Perform the ray tracing
        rgb, dpt, acc, norm, dist, aux, mid, wet = tracer(
            ray_o,  # (H, W, 3) or (B, P, 3)
            ray_d,  # (H, W, 3) or (B, P, 3)
            v,  # (P * 4, 3)
            means3D=means3D,  # (P, 3)
            grads3D=grads3D,  # (P, 3)
            shs=shs,
            colors_precomp=colors_precomp,
            others_precomp=others_precomp,
            opacities=opacities,  # (P, 1)
            scales=scales,  # (P, 2)
            rotations=rotations,  # (P, 4)
            cov3D_precomp=cov3D_precomp,
            tracer_settings=tracer_settings,
            start_from_first=start_from_first,
        )

        # Visualization saving
        save_image(join(rgb_dir, f'{i:04d}{args.vis_ext}'), rgb)
        save_image(join(dpt_dir, f'{i:04d}{args.vis_ext}'), normalize_depth(dpt))
        save_image(join(nrm_dir, f'{i:04d}{args.vis_ext}'), visualize_normal(norm, R))

    # Generate videos
    generate_video(f'"{rgb_dir}/*{args.vis_ext}"', f'{join(args.result_dir, "RENDER.mp4")}', fps=args.fps)
    generate_video(f'"{dpt_dir}/*{args.vis_ext}"', f'{join(args.result_dir, "DEPTH.mp4")}', fps=args.fps)
    generate_video(f'"{nrm_dir}/*{args.vis_ext}"', f'{join(args.result_dir, "NORMAL.mp4")}', fps=args.fps)

    # Finish testing
    print(f"[INFO] Rendering finished.")


if __name__ == '__main__':
    main()
