"""
This file contains the utility functions for data and depth processing.
Copied from our EasyVolcap codebase, check the original file here:
https://github.com/zju3dv/EasyVolcap/blob/main/easyvolcap/utils/data_utils.py
https://github.com/zju3dv/EasyVolcap/blob/main/easyvolcap/utils/depth_utils.py
"""

import os
import cv2
import sys
import torch
import numpy as np
from os.path import join, dirname


def save_image(img_path: str, img: np.ndarray, jpeg_quality=75, png_compression=9, save_dtype=np.uint8):
    if isinstance(img, torch.Tensor): img = img.detach().cpu().numpy()  # convert to numpy arrays
    if img.ndim == 4: img = np.concatenate(img, axis=0)  # merge into one image along y axis
    if img.ndim == 2: img = img[..., None]  # append last dim
    if img.shape[0] < img.shape[-1] and (img.shape[0] == 3 or img.shape[0] == 4): img = np.transpose(img, (1, 2, 0))
    if np.issubdtype(img.dtype, np.integer):
        img = img / np.iinfo(img.dtype).max  # to float
    if img.shape[-1] >= 3:
        if not img.flags['WRITEABLE']:
            img = img.copy()  # avoid assignment only inputs
        img[..., :3] = img[..., [2, 1, 0]]
    if dirname(img_path):
        os.makedirs(dirname(img_path), exist_ok=True)
    if img_path.endswith('.png'):
        max = np.iinfo(save_dtype).max
        img = (img * max).clip(0, max).astype(save_dtype)
    elif img_path.endswith('.jpg'):
        img = img[..., :3]  # only color
        img = (img * 255).clip(0, 255).astype(np.uint8)
    elif img_path.endswith('.hdr'):
        img = img[..., :3]  # only color
    elif img_path.endswith('.exr'):
        # ... https://github.com/opencv/opencv/issues/21326
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    else:
        # should we try to discard alpha channel here?
        # exr could store alpha channel
        pass  # no transformation for other unspecified file formats
    # log(f'Writing image to: {img_path}')
    # breakpoint()
    return cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality,
                                       cv2.IMWRITE_PNG_COMPRESSION, png_compression,
                                       cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_PIZ])


@torch.jit.script
def normalize_depth(depth: torch.Tensor, p: float = 0.01):
    n = int(depth.numel() * p)
    near = depth.ravel().topk(n, largest=False)[0].max()  # a simple version of percentile
    far = depth.ravel().topk(n, largest=True)[0].min()  # a simple version of percentile
    depth = 1 - (depth - near) / (far - near)
    depth = depth.clip(0, 1)
    return depth


def visualize_normal(n: torch.Tensor, R: torch.Tensor, eps: float = 1e-8):
    # Normalize the normal map
    n = n / (torch.norm(n, dim=-1, keepdim=True) + eps)
    # Convert the normal map to camera space
    n = n @ R.mT
    # Better visualization
    n[..., [1, 2]] *= -1
    n = n * 0.5 + 0.5
    return n


def run(cmd,
        quite=False,
        dry_run=False,
        skip_failed=False,
        invocation=os.system,  # or subprocess.run
        ):
    """
    Run a shell command and print the command to the console.

    Args:
        cmd (str or list): The command to run. If a list, it will be joined with spaces.
        quite (bool): If True, suppress console output.
        dry_run (bool): If True, print the command but do not execute it.

    Raises:
        RuntimeError: If the command returns a non-zero exit code.

    Returns:
        None
    """
    if isinstance(cmd, list):
        cmd = ' '.join(list(map(str, cmd)))
    func = sys._getframe(1).f_code.co_name
    if not quite:
        cmd_color = 'cyan' if not cmd.startswith('rm') else 'red'
        cmd_color = 'green' if dry_run else cmd_color
        dry_msg = '[dry_run]: ' if dry_run else ''
        print(func, '->', invocation.__name__ + ":", dry_msg + cmd, cmd_color)
        # print(color(cmd, cmd_color), soft_wrap=False)
    if not dry_run:
        code = invocation(cmd)
    else:
        code = 0
    if code != 0 and not skip_failed:
        print(code, "<-", func + ":", cmd)
        # print(red(cmd), soft_wrap=True)
        raise RuntimeError(f'{code} <- {func}: {cmd}')
    else:
        return code  # or output


def generate_video(result_str: str,
                   output: str,
                   verbose: bool = False,
                   fps: int = 30,
                   crf: int = 17,
                   cqv: int = 19,
                   lookahead: int = 20,
                   hwaccel: str = 'cuda',
                   preset: str = 'p7',
                   tag: str = 'hvc1',
                   vcodec: str = 'hevc_nvenc',
                   pix_fmt: str = 'yuv420p',  # chrome friendly
                   ):
    cmd = [
        'ffmpeg',
        '-hwaccel', hwaccel,
    ] + ([
        '-hide_banner',
        '-loglevel', 'error',
    ] if not verbose else []) + ([
        '-framerate', fps,
    ] if fps > 0 else []) + ([
        '-f', 'image2',
        '-pattern_type', 'glob',
    ] if '*' in result_str else []) + ([
        '-r', fps,
    ] if fps > 0 else []) + [
        '-nostdin',  # otherwise you cannot chain commands together
        '-y',
        '-i', result_str,
        '-c:v', vcodec,
        '-preset', preset,
        '-cq:v', cqv,
        '-rc:v', 'vbr',
        '-tag:v', tag,
        '-crf', crf,
        '-pix_fmt', pix_fmt,
        '-rc-lookahead', lookahead,
        '-vf', '"pad=ceil(iw/2)*2:ceil(ih/2)*2"',  # avoid yuv420p odd number bug
        output,
    ]
    run(cmd)
    return output
