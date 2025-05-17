import os
import cv2
import copy
import argparse
import numpy as np

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.easy_utils import write_camera
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.colmap_utils import qvec2rotmat, read_model, read_cameras_text, read_images_text, detect_model_format, read_cameras_binary, read_images_binary


def detect_model_format(path, ext):
    if os.path.isfile(os.path.join(path, "cameras" + ext)) and \
       os.path.isfile(os.path.join(path, "images" + ext)):
        print("Detected model format: '" + ext + "'")
        return True

    return False


@catch_throw
def main(args):
    # Define the per-scene processing function
    def process_scene(scene):
        log(f'processing scene: {scene}')

        # Data paths
        data_root = join(args.data_root, scene)
        colmap = join(data_root, args.colmap)
        output = join(args.output, scene)

        # cameras, images, points3D = read_model(path=args.colmap)
        ext = ''
        if ext == "":
            if detect_model_format(colmap, ".bin"):
                ext = ".bin"
            elif detect_model_format(colmap, ".txt"):
                ext = ".txt"
            else:
                print("Provide model format: '.bin' or '.txt'")
                return

        if ext == '.bin':
            cameras = read_cameras_binary(join(colmap, "cameras" + ext))
            images = read_images_binary(join(colmap, "images" + ext))
        else:
            cameras = read_cameras_text(join(colmap, "cameras" + ext))
            images = read_images_text(join(colmap, "images" + ext))
        log(f"number of cameras: {len(cameras)}")
        log(f"number of images: {len(images)}")
        # log(f"number of points3D: {len(points3D)}")

        intrinsics = {}
        for key in cameras.keys():
            p = cameras[key].params
            if cameras[key].model == 'SIMPLE_RADIAL':
                f, cx, cy, k = p
                K = np.array([f, 0, cx, 0, f, cy, 0, 0, 1]).reshape(3, 3)
                dist = np.array([[k, 0, 0, 0, 0]])
            elif cameras[key].model == 'PINHOLE':
                K = np.array([[p[0], 0, p[2], 0, p[1], p[3], 0, 0, 1]]).reshape(3, 3)
                dist = np.array([[0., 0., 0., 0., 0.]])
            else:  # OPENCV
                K = np.array([[p[0], 0, p[2], 0, p[1], p[3], 0, 0, 1]]).reshape(3, 3)
                dist = np.array([[p[4], p[5], p[6], p[7], 0.]])
            H, W = cameras[key].height, cameras[key].width
            intrinsics[key] = {'K': K, 'dist': dist, 'H': H, 'W': W}

        cnt = 0
        evccams = {}
        for key, val in sorted(images.items(), key=lambda item: item[1].name)[::args.skip]:
            if args.sub in val.name:
                # Convert COLMAP camera to EasyVolCap camera
                log(f'preparing camera: {val.name}(#{val.camera_id})')
                cam = copy.deepcopy(intrinsics[val.camera_id])
                t = val.tvec.reshape(3, 1)
                R = qvec2rotmat(val.qvec)
                cam['Rvec'] = cv2.Rodrigues(R)[0]
                cam['R'] = R
                cam['T'] = t * args.scale
                evccams[f'{cnt:0{args.digit}d}'] = cam

                # Link images
                src_images_dir = join(data_root, args.src_images_dir)
                tar_images_dir = join(output, args.tar_images_dir, f'{cnt:0{args.digit}d}')
                os.makedirs(tar_images_dir, exist_ok=True)
                # Check if val.name is a file or a directory
                if not exists(join(src_images_dir, val.name)):
                    if exists(join(src_images_dir, val.name.replace('.jpg', '.JPG'))):
                        val.name = val.name.replace('.jpg', '.JPG')
                    elif exists(join(src_images_dir, val.name.replace('.JPG', '.jpg'))):
                        val.name = val.name.replace('.JPG', '.jpg')
                    elif exists(join(src_images_dir, val.name.replace('.jpg', '.png'))):
                        val.name = val.name.replace('.jpg', '.png')
                    elif exists(join(src_images_dir, val.name.replace('.JPG', '.png'))):
                        val.name = val.name.replace('.JPG', '.png')
                    else:
                        log(f'cannot find image: {val.name}', 'red')
                        continue
                os.system(f'ln -s {abspath(join(src_images_dir, val.name))} {tar_images_dir}/{0:06d}.{args.ext}')

                # Increment counter
                cnt += 1
            else:
                log(f'skipping camera: {val.name}(#{val.camera_id}) since {args.sub} not in {val.name}', 'yellow')

        # Dicts preserve insertion order in Python 3.7+. Same in CPython 3.6, but it's an implementation detail.
        evccams = dict(sorted(evccams.items(), key=lambda item: item[0]))
        write_camera(evccams, output)

        # Copy the original colmap model
        os.makedirs(f'{output}/sparse', exist_ok=True)
        os.system(f'cp -r {colmap} {output}/sparse')

    # Find all scenes
    if len(args.scenes):
        scenes = [
            f
            for f in os.listdir(args.data_root)
            if os.path.isdir(os.path.join(args.data_root, f)) and f in args.scenes
        ]
    else:
        scenes = [
            f
            for f in os.listdir(args.data_root)
            if os.path.isdir(os.path.join(args.data_root, f))
        ]

    # Process each scene
    parallel_execution(scenes, action=process_scene, sequential=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/datasets/original/refnerf/ref_real')
    parser.add_argument('--scenes', nargs='+', default=[])
    parser.add_argument('--colmap', type=str, default='sparse/0')
    parser.add_argument('--src_images_dir', type=str, default='images')
    parser.add_argument('--tar_images_dir', type=str, default='images')
    parser.add_argument('--output', type=str, default='data/datasets/refnerf/ref_real')
    parser.add_argument('--sub', type=str, default='')  # only camera name containing this string will be saved
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--digit', type=int, default=4)
    parser.add_argument('--ext', type=str, default='jpg')
    parser.add_argument('--skip', type=int, default=1)
    args = parser.parse_args()
    main(args)
