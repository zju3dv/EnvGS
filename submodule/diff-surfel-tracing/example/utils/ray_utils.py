"""
This file contains the utility functions for math operations and ray generation.
Copied from our EasyVolcap codebase, check the original file here:
https://github.com/zju3dv/EasyVolcap/blob/main/easyvolcap/utils/math_utils.py
https://github.com/zju3dv/EasyVolcap/blob/main/easyvolcap/utils/ray_utils.py
"""

import torch
from torch import nn


@torch.jit.script
def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # channel last: normalization
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


@torch.jit.script
def torch_inverse_3x3(R: torch.Tensor, eps: float = torch.finfo(torch.float).eps):
    # n_batch, n_bones, 3, 3
    """
    a, b, c | m00, m01, m02
    d, e, f | m10, m11, m12
    g, h, i | m20, m21, m22
    """

    # convenient access
    r00 = R[..., 0, 0]
    r01 = R[..., 0, 1]
    r02 = R[..., 0, 2]
    r10 = R[..., 1, 0]
    r11 = R[..., 1, 1]
    r12 = R[..., 1, 2]
    r20 = R[..., 2, 0]
    r21 = R[..., 2, 1]
    r22 = R[..., 2, 2]

    M = torch.empty_like(R)

    # determinant of matrix minors
    # fmt: off
    M[..., 0, 0] =   r11 * r22 - r21 * r12
    M[..., 1, 0] = - r10 * r22 + r20 * r12
    M[..., 2, 0] =   r10 * r21 - r20 * r11
    M[..., 0, 1] = - r01 * r22 + r21 * r02
    M[..., 1, 1] =   r00 * r22 - r20 * r02
    M[..., 2, 1] = - r00 * r21 + r20 * r01
    M[..., 0, 2] =   r01 * r12 - r11 * r02
    M[..., 1, 2] = - r00 * r12 + r10 * r02
    M[..., 2, 2] =   r00 * r11 - r10 * r01
    # fmt: on

    # determinant of matrix
    D = r00 * M[..., 0, 0] + r01 * M[..., 1, 0] + r02 * M[..., 2, 0]

    # inverse of 3x3 matrix
    M = M / (D[..., None, None] + eps)

    return M


@torch.jit.script
def create_meshgrid(H: int, W: int, device: torch.device = torch.device('cuda'), indexing: str = 'ij', ndc: bool = False,
                    correct_pix: bool = True, dtype: torch.dtype = torch.float):
    # kornia has meshgrid, but not the best
    i = torch.arange(H, device=device, dtype=dtype)
    j = torch.arange(W, device=device, dtype=dtype)
    if correct_pix:
        i = i + 0.5
        j = j + 0.5
    if ndc:
        i = i / H * 2 - 1
        j = j / W * 2 - 1
    ij = torch.meshgrid(i, j, indexing=indexing)  # defaults to ij
    ij = torch.stack(ij, dim=-1)  # Ht, Wt, 2

    return ij


def get_rays(H: int, W: int, K: torch.Tensor, R: torch.Tensor, T: torch.Tensor, is_inv_K: bool = False,
             z_depth: bool = False, correct_pix: bool = True, ret_coord: bool = False):
    # calculate the world coodinates of pixels
    i, j = torch.meshgrid(torch.arange(H, dtype=R.dtype, device=R.device),
                          torch.arange(W, dtype=R.dtype, device=R.device),
                          indexing='ij')
    bss = K.shape[:-2]
    for _ in range(len(bss)): i, j = i[None], j[None]
    i, j = i.expand(bss + i.shape[len(bss):]), j.expand(bss + j.shape[len(bss):])
    # 0->H, 0->W
    return get_rays_from_ij(i, j, K, R, T, is_inv_K, z_depth, correct_pix, ret_coord)


def get_rays_from_ij(i: torch.Tensor, j: torch.Tensor,
                     K: torch.Tensor, R: torch.Tensor, T: torch.Tensor,
                     is_inv_K: bool = False, use_z_depth: bool = False,
                     correct_pix: bool = True, ret_coord: bool = False):
    # i: B, P or B, H, W or P or H, W
    # j: B, P or B, H, W or P or H, W
    # K: B, 3, 3
    # R: B, 3, 3
    # T: B, 3, 1
    nb_dim = len(K.shape[:-2])  # number of batch dimensions
    np_dim = len(i.shape[nb_dim:])  # number of points dimensions
    if not is_inv_K: invK = torch_inverse_3x3(K.float()).type(K.dtype)
    else: invK = K
    ray_o = - R.mT @ T  # B, 3, 1

    # Prepare the shapes
    for _ in range(np_dim): invK = invK.unsqueeze(-3)
    invK = invK.expand(i.shape + (3, 3))
    for _ in range(np_dim): R = R.unsqueeze(-3)
    R = R.expand(i.shape + (3, 3))
    for _ in range(np_dim): T = T.unsqueeze(-3)
    T = T.expand(i.shape + (3, 1))
    for _ in range(np_dim): ray_o = ray_o.unsqueeze(-3)
    ray_o = ray_o.expand(i.shape + (3, 1))

    # Pixel center correction
    if correct_pix: i, j = i + 0.5, j + 0.5
    else: i, j = i.float(), j.float()

    # 0->H, 0->W
    # int -> float; # B, H, W, 3, 1 or B, P, 3, 1 or P, 3, 1 or H, W, 3, 1
    xy1 = torch.stack([j, i, torch.ones_like(i)], dim=-1)[..., None]
    pixel_camera = invK @ xy1  # B, H, W, 3, 1 or B, P, 3, 1
    pixel_world = R.mT @ (pixel_camera - T)  # B, P, 3, 1

    # Calculate the ray direction
    pixel_world = pixel_world[..., 0]
    ray_o = ray_o[..., 0]
    ray_d = pixel_world - ray_o  # use pixel_world depth as is (no curving)
    if not use_z_depth: ray_d = normalize(ray_d)  # B, P, 3, 1

    if not ret_coord: return ray_o, ray_d
    elif correct_pix: return ray_o, ray_d, (torch.stack([i, j], dim=-1) - 0.5).long()  # B, P, 2
    else: return ray_o, ray_d, torch.stack([i, j], dim=-1).long()  # B, P, 2
