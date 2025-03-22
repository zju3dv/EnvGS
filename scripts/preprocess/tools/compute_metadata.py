import os
import cv2
import copy
import torch
import argparse
import numpy as np

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.easy_utils import read_camera
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.colmap_utils import read_points3D_binary_custom, read_points3D_text_custom, load_sfm_ply, save_sfm_ply


def calculate_bounding_box(xyz, lower_percentile=5, upper_percentile=95):
    # Convert to tensor
    xyz_tensor = torch.tensor(xyz)
    # Compute min and max values
    min_values = torch.quantile(xyz_tensor, lower_percentile / 100.0, dim=0)
    max_values = torch.quantile(xyz_tensor, upper_percentile / 100.0, dim=0)

    # Filter out points outside the range
    mask = ((xyz_tensor >= min_values) & (xyz_tensor <= max_values)).all(dim=1)
    filtered_xyz = xyz_tensor[mask]

    # Compute min and max values
    min_xyz = torch.min(filtered_xyz, dim=0).values
    max_xyz = torch.max(filtered_xyz, dim=0).values
    return min_xyz.numpy(), max_xyz.numpy()


def load_points(points_file: str = None):
    xyz = None
    if exists(points_file):
        xyz, _ = load_sfm_ply(points_file)
    else:
        try:
            xyz, _, _ = read_points3D_binary_custom(points_file.replace(".ply", ".bin"))
        except:
            xyz, _, _ = read_points3D_text_custom(points_file.replace(".ply", ".txt"))
    return xyz


@catch_throw
def main(args):
    # Define the per-scene processing function
    def process_scene(scene):
        log(f'{yellow("Processing scene")}: {cyan(scene)}')

        # Data paths
        data_root = join(args.data_root, scene)
        points_path = join(data_root, args.points_file)

        # Load all cameras
        cameras = read_camera(data_root)
        camera_names = sorted(cameras.keys())  # sort the camera names

        # Split the cameras into two groups if `args.eval` is True
        if args.eval:
            view_sample = [i for i in range(len(camera_names)) if i % args.skip != 0]
            val_view_sample = [i for i in range(len(camera_names)) if i % args.skip == 0]
        else:
            view_sample = [i for i in range(len(camera_names))]
            val_view_sample = []
        log(f'dataloader_cfg.dataset_cfg.view_sample: {view_sample}')
        log(f'val_dataloader_cfg.dataset_cfg.view_sample: {val_view_sample}')

        # Compute the center and radius for the training set
        Rs = np.stack([cameras[camera_names[i]]['R'] for i in view_sample], axis=0)  # (N, 3, 3)
        Ts = np.stack([cameras[camera_names[i]]['T'] for i in view_sample], axis=0)  # (N, 3, 1)
        Cs = -Rs.mT @ Ts  # (N, 3, 1)
        center = np.mean(Cs, axis=0)  # (3, 1)
        radius = np.linalg.norm(Cs - center[None], axis=1).max()  # scalar
        radius = radius * 1.1  # follow the original 3DGS
        log(f"model_cfg.sampler_cfg.spatial_scale: {radius}")

        # Load SfM points
        xyz = load_points(points_path)
        if xyz is not None:
            min_xyz, max_xyz = calculate_bounding_box(xyz, lower_percentile=args.lower_percentile, upper_percentile=args.upper_percentile)
        else:
            min_xyz = args.bounds[:3]
            max_xyz = args.bounds[3:]
        log(f"model_cfg.sampler_cfg.env_bounds: [[{min_xyz[0]}, {min_xyz[1]}, {min_xyz[2]}], [{max_xyz[0]}, {max_xyz[1]}, {max_xyz[2]}]]")
        log()

    # Find all scenes
    if len(args.scenes):
        scenes = [f for f in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, f)) and f in args.scenes]
    else:
        scenes = [f for f in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, f))]

    # Process each scene
    parallel_execution(scenes, action=process_scene, sequential=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/datasets/refnerf/ref_real')
    parser.add_argument('--images_dir', type=str, default='images')
    parser.add_argument('--points_file', type=str, default='sparse/0/points3D.ply')
    parser.add_argument('--scenes', nargs='+', default=[])
    parser.add_argument('--eval', action='store_true', default=True)
    parser.add_argument('--skip', type=int, default=8)
    parser.add_argument('--lower_percentile', type=float, default=0.5)
    parser.add_argument('--upper_percentile', type=float, default=99.5)
    parser.add_argument('--bounds', type=float, nargs='+', default=[-20., -20., -20., 20., 20., 20.])
    args = parser.parse_args()
    main(args)
