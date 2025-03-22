"""
Given images, output scalar metrics on CPU
Used for evaluation. For training, please check out loss_utils
"""

import torch
import numpy as np
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.loss_utils import mse as compute_mse
from easyvolcap.utils.loss_utils import lpips as compute_lpips
from skimage.metrics import structural_similarity as compare_ssim

from enum import Enum, auto


@torch.no_grad()
def psnr(x: torch.Tensor, y: torch.Tensor):
    mse = compute_mse(x, y).mean()
    psnr = (1 / mse.clip(1e-10)).log() * 10 / np.log(10)
    return psnr.item()  # tensor to scalar


@torch.no_grad()
def ssim(x: torch.Tensor, y: torch.Tensor, kernel_size=11, size_average=True) -> float:
    # Assume input shape (B, H, W, C) or (H, W, C)
    if x.ndim == 3: x = x.unsqueeze(0)
    if y.ndim == 3: y = y.unsqueeze(0)
    x = x.permute(0, 3, 1, 2)
    y = y.permute(0, 3, 1, 2)

    # Determine the number of channels in the input tensor
    groups = x.size(-3)
    weight = create_window(kernel_size, groups)
    # Move the kernel_size to the same device as the input tensor
    if x.is_cuda: weight = weight.cuda(x.get_device())
    weight = weight.type_as(x)

    # Compute means using Gaussian filter
    mu1 = F.conv2d(x, weight, padding=kernel_size//2, groups=groups)
    mu2 = F.conv2d(y, weight, padding=kernel_size//2, groups=groups)

    # Compute squared means and cross-term
    mu1_squ = mu1.pow(2)
    mu2_squ = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Compute variances and covariance
    var1 = F.conv2d(x * x, weight, padding=kernel_size // 2, groups=groups) - mu1_squ
    var2 = F.conv2d(y * y, weight, padding=kernel_size // 2, groups=groups) - mu2_squ
    var1_var2 = F.conv2d(x * y, weight, padding=kernel_size // 2, groups=groups) - mu1_mu2

    # Constants to stabilize division with weak denominator
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    # Compute SSIM map
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * var1_var2 + C2)) / ((mu1_squ + mu2_squ + C1) * (var1 + var2 + C2))

    # Return the average SSIM value
    if size_average: return ssim_map.mean().item()
    else: return ssim_map.mean(dim=[1, 2, 3]).item()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


# @torch.no_grad()
# def ssim(x: torch.Tensor, y: torch.Tensor):
#     return np.mean([
#         compare_ssim(
#             _x.detach().cpu().numpy(),
#             _y.detach().cpu().numpy(),
#             channel_axis=-1,
#             data_range=2.0
#         )
#         for _x, _y in zip(x, y)
#     ]).astype(float).item()


@torch.no_grad()
def lpips(x: torch.Tensor, y: torch.Tensor):
    if x.ndim == 3: x = x.unsqueeze(0)
    if y.ndim == 3: y = y.unsqueeze(0)
    x = x.permute(0, 3, 1, 2)
    y = y.permute(0, 3, 1, 2)
    return compute_lpips(x, y, net='vgg').item()


class Metrics(Enum):
    PSNR = psnr
    SSIM = ssim
    LPIPS = lpips
