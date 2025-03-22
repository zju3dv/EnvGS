import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from stablenormal.pipeline_yoso_normal import YOSONormalsPipeline
from stablenormal.pipeline_stablenormal import StableNormalPipeline
from stablenormal.scheduler.heuristics_ddimsampler import HEURI_DDIMScheduler

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import save_image
from easyvolcap.utils.parallel_utils import parallel_execution


def create_pipeline(
    yoso_version: str = 'Stable-X/yoso-normal-v0-3',
    stable_version: str = 'Stable-X/stable-normal-v0-1'
):
    # Decide on the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the YOSO normal pipeline
    x_start_pipeline = YOSONormalsPipeline.from_pretrained(
        yoso_version,
        trust_remote_code=True,
        variant="fp16",
        torch_dtype=torch.float16
    ).to(device)

    # Create the Stable Normal pipeline
    pipe = StableNormalPipeline.from_pretrained(
        stable_version,
        trust_remote_code=True,
        variant="fp16",
        torch_dtype=torch.float16,
        scheduler=HEURI_DDIMScheduler(
            prediction_type='sample',
            beta_start=0.00085,
            beta_end=0.0120,
            beta_schedule="scaled_linear"
        )
    )

    # Set the x_start_pipeline
    pipe.x_start_pipeline = x_start_pipeline
    pipe.to(device)
    pipe.prior.to(device, torch.float16)
    try:
        import xformers
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass  # run without xformers
    return pipe


def resize_image(
    image: Image.Image,
    resolution: int,
    align: int = 64
):
    # Get image dimensions
    Wo, Ho = image.size

    # Calculate the scaling factor
    k = float(resolution) / float(min(Ho, Wo))

    # Determine new dimensions
    Hn = Ho * k
    Wn = Wo * k
    Hn = int(np.round(Hn / align)) * align
    Wn = int(np.round(Wn / align)) * align

    # Resize the image using PIL's resize method
    if Hn != Ho or Wn != Wo:
        image = image.resize(
            (Wn, Hn),
            Image.Resampling.LANCZOS
        )

    return image, Ho, Wo, Hn, Wn


def process_image(
    pipe,
    image_path: str,
    resolution: int = 1024,
    align: int = 64
):
    # Load the image
    image = Image.open(image_path)
    # Resize the image to the network input resolution
    image, Ho, Wo, Hn, Wn = resize_image(
        image,
        resolution=resolution,
        align=align
    )

    # Forward the image through the pipeline
    output = pipe(
        image,
        match_input_resolution=True,
        processing_resolution=max(image.size)
    )
    # Get the normal prediction
    normal = output.prediction[0, :, :]

    # Restore the image to its original size
    if Hn != Ho or Wn != Wo:
        normal = cv2.resize(
            normal,
            (Wo, Ho),
            interpolation=cv2.INTER_LINEAR
        )

    # Color the normal prediction for visualization
    normal_colored = pipe.image_processor.visualize_normals(normal)

    # Post-process the normal prediction
    normal = np.asarray(normal_colored)
    normal = normal / 255. * 2 - 1
    normal = normal * -1.
    normal = normal * 0.5 + 0.5

    return normal_colored, normal


def main(args):
    # Define the per-scene processing function
    def process_scene(scene):
        log(f'Processing scene: {scene}')

        # Data paths
        data_root = join(args.data_root, scene)
        images_dir = join(data_root, args.images_dir)
        normals_dir = join(data_root, args.normals_dir)

        # Loop through all the views
        for v in sorted(os.listdir(images_dir)):
            images_path = join(images_dir, v)
            normals_path = join(normals_dir, v)
            os.makedirs(normals_path, exist_ok=True)

            # Loop through all the views
            for f in sorted(os.listdir(images_path)):
                image_file = join(images_path, f)
                normal_file = join(normals_dir, f)

                # Check if the normal image already exists
                if os.path.exists(normal_file):
                    continue

                # Process the image
                normal_colored, normal = process_image(
                    pipe,
                    image_file,
                    resolution=args.resolution,
                    align=args.align
                )

                # Save the normal image
                save_image(normal_file, normal)

    # Create the pipeline
    pipe = create_pipeline(
        yoso_version=args.yoso_version,
        stable_version=args.stable_version
    )

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
    parser.add_argument('--yoso_version', type=str, default='Stable-X/yoso-normal-v0-3')
    parser.add_argument('--stable_version', type=str, default='Stable-X/stable-normal-v0-1')
    parser.add_argument('--data_root', type=str, default='data/datasets/original/refnerf/ref_real')
    parser.add_argument('--scenes', nargs='+', default=[])
    parser.add_argument('--images_dir', type=str, default='images')
    parser.add_argument('--normals_dir', type=str, default='normals')
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--align', type=int, default=64)
    args = parser.parse_args()
    main(args)
