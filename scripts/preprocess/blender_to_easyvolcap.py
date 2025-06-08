import os
import cv2
import json
import argparse
import operator
import numpy as np
from PIL import Image
import imageio.v2 as imageio

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.math_utils import normalize
from easyvolcap.utils.easy_utils import write_camera
from easyvolcap.utils.data_utils import save_image, load_image
from easyvolcap.utils.parallel_utils import parallel_execution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--blender_root', type=str, default='data/datasets/original/refnerf/ref')
    parser.add_argument('--easyvolcap_root', type=str, default='data/datasets/refnerf/shiny_blender')
    parser.add_argument('--has_alpha', action='store_true')
    parser.add_argument('--has_normal', action='store_true')
    parser.add_argument('--black_bkgd', action='store_true')
    parser.add_argument('--ext', type=str, default='png')
    args = parser.parse_args()

    blender_root = args.blender_root
    easyvolcap_root = args.easyvolcap_root

    # Background color
    bg_color = np.array([0, 0, 0]) if args.black_bkgd else np.array([1, 1, 1])

    def process_camera_image(blender_path, easyvolcap_path, split, frames, camera_angle_x, H, W):
        # Define and create output image path and mask path
        img_out_dir = join(easyvolcap_path, split, f'images')
        msk_out_dir = join(easyvolcap_path, split, f'masks')
        if args.has_alpha: alpha_out_dir = join(easyvolcap_path, split, f'alphas')
        if args.has_normal: normal_out_dir = join(easyvolcap_path, split, f'normals_gt')

        cameras = dotdict()
        # Remove frames with the same timestamp
        for cnt, frame in enumerate(frames):
            # Create soft link for image
            img_blender_path = join(blender_path, frame['file_path'][2:] + f'.{args.ext}')
            img_easyvolcap_path = join(img_out_dir, f'{cnt:04d}', f'000000.{args.ext}')
            img = np.array(Image.open(img_blender_path).convert("RGBA")) / 255.0
            rgb = img[:, :, :3] * img[:, :, 3:] + bg_color * (1 - img[:, :, 3:])
            os.makedirs(os.path.dirname(img_easyvolcap_path), exist_ok=True)
            save_image(img_easyvolcap_path, rgb)

            # Create mask for the image
            msk = imageio.imread(img_blender_path).sum(axis=-1) > 0
            msk = msk.astype(np.uint8) * 255
            msk_easyvolcap_path = join(msk_out_dir, f'{cnt:04d}', f'000000.{args.ext}')
            os.makedirs(os.path.dirname(msk_easyvolcap_path), exist_ok=True)
            cv2.imwrite(msk_easyvolcap_path, msk)

            # Copy alpha and normal images
            if args.has_alpha:
                acc_easyvolcap_path = join(alpha_out_dir, f'{cnt:04d}', f'000000.{args.ext}')
                os.makedirs(os.path.dirname(acc_easyvolcap_path), exist_ok=True)
                os.system(f"cp -r {img_blender_path.replace(f'.{args.ext}', f'_alpha.{args.ext}')} {acc_easyvolcap_path}")

            # Fetch and store camera parameters
            c2w_opengl = np.array(frame['transform_matrix']).astype(np.float32)
            c2w_opencv = c2w_opengl @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            w2c_opencv = np.linalg.inv(c2w_opencv)
            cameras[f'{cnt:04d}'] = {
                'R': w2c_opencv[:3, :3],
                'T': w2c_opencv[:3, 3:],
                'K': np.array([[0.5 * W / np.tan(0.5 * camera_angle_x), 0, 0.5 * W],
                               [0, 0.5 * W / np.tan(0.5 * camera_angle_x), 0.5 * H],
                               [0, 0, 1]]),
                'D': np.zeros((1, 5)),
                'H': H, 'W': W,
            }
            cnt += 1

        return cameras

    def process_scene(scene):
        # Create soft link for scene
        blender_path = join(blender_root, scene)
        easyvolcap_path = join(easyvolcap_root, scene)

        sh = imageio.imread(join(blender_path, 'train', sorted(os.listdir(join(blender_path, 'train')))[1])).shape
        H, W = int(sh[0]), int(sh[1])

        # Load frames information of all splits
        splits = ['train', 'val', 'test']
        for split in splits:
            if not exists(join(blender_path, f'transforms_{split}.json')):
                continue
            # Load all frames information
            frames = json.load(open(join(blender_path, f'transforms_{split}.json')))['frames']
            camera_angle_x = json.load(open(join(blender_path, f'transforms_{split}.json')))['camera_angle_x']
            # Process camera parameters
            cameras = process_camera_image(blender_path, easyvolcap_path, split, frames, camera_angle_x, H, W)
            # Write camera parameters, treat dnerf dataset as one camera monocular video dataset
            write_camera(cameras, join(easyvolcap_path, split))
            log(yellow(f'Converted cameras saved to {blue(join(easyvolcap_path, split, "{intri.yml,extri.yml}"))}'))

    # Clean and restart
    # os.system(f'rm -rf {easyvolcap_root}')
    os.makedirs(easyvolcap_root, exist_ok=True)

    # Convert all scenes
    scenes = os.listdir(blender_root)
    scenes = sorted(scenes)
    parallel_execution(scenes, action=process_scene, sequential=True, print_progress=True)


if __name__ == '__main__':
    main()
