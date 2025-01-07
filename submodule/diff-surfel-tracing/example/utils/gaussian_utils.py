import math
import torch
import numpy as np
import torch.nn as nn
from plyfile import PlyData
import torch.nn.functional as F


class GaussianModel(nn.Module):
    def __init__(
        self,
        max_sh_degree: int = 3,
    ):
        super().__init__()
        self.setup_functions(
            scaling_activation=scaling_activation,
            scaling_inverse_activation=scaling_inverse_activation,
            opacity_activation=opacity_activation,
            inverse_opacity_activation=inverse_opacity_activation,
        )

        self.max_sh_degree = max_sh_degree

    def setup_functions(
        self,
        scaling_activation=torch.exp,
        scaling_inverse_activation=torch.log,
        opacity_activation=torch.sigmoid,
        inverse_opacity_activation=torch.logit,
        rotation_activation=F.normalize,
    ):
        self.scaling_activation = (
            getattr(torch, scaling_activation)
            if isinstance(scaling_activation, str)
            else scaling_activation
        )
        self.scaling_inverse_activation = (
            getattr(torch, scaling_inverse_activation)
            if isinstance(scaling_inverse_activation, str)
            else scaling_inverse_activation
        )
        self.opacity_activation = (
            getattr(torch, opacity_activation)
            if isinstance(opacity_activation, str)
            else opacity_activation
        )
        self.opacity_inverse_activation = (
            getattr(torch, inverse_opacity_activation)
            if isinstance(inverse_opacity_activation, str)
            else inverse_opacity_activation
        )
        self.rotation_activation = (
            getattr(torch, rotation_activation)
            if isinstance(rotation_activation, str)
            else rotation_activation
        )
        self.covariance_activation = build_cov

    @property
    def device(self):
        return self.get_xyz.device

    @property
    def number(self):
        return self._xyz.shape[0]

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_max_sh_channels(self):
        return (self.max_sh_degree + 1) ** 2

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_xyz, self.get_scaling, scaling_modifier, self.get_rotation
        )

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )

        self.active_sh_degree = torch.full((1,), self.max_sh_degree, dtype=torch.long)

    def get_triangles(self):
        # Build the uv tangent plane to world transformation matrix, splat2world
        T = self.get_covariance()  # (P, 4, 4)
        T = T.permute(0, 2, 1)  # (P, 4, 4)
        T[..., 2] = 0  # (P, 4, 4)

        # Deal with nasty shapes
        P, V = T.shape[0], 4  # 1 2DGS <-> 2 triangles <-> 4 vertices

        # 3-sigma range in local uv splat coordiantes
        sigma3 = (
            torch.as_tensor(
                [[-1.0, 1.0], [-1.0, -1.0], [1.0, 1.0], [1.0, -1.0]], device=T.device
            )
            * 3
        )  # (V, 2)
        sigma3 = torch.cat([sigma3, torch.ones_like(sigma3)], dim=-1)  # (V, 4)
        # Expand
        sigma3 = sigma3[None].repeat(P, 1, 1)  # (P, V, 4)
        T = T[:, None].expand(-1, V, -1, -1)  # (P, V, 4, 4)

        # Convert the vertices to the world coordinate
        v = T.reshape(-1, 4, 4) @ sigma3.reshape(-1, 4, 1)  # (P * V, 4, 1)
        v = v[..., :3, 0]  # (P * V, 3)

        # Generate face indices
        indices = torch.arange(0, v.shape[0]).reshape(P, V).to(T.device)  # (P, V)
        f = (
            torch.stack([indices[:, :3], indices[:, 1:]], dim=1).reshape(-1, 3).int()
        )  # (P, 2, 3) -> (P * 2, 3)

        # NOTE: `.contiguous()` is necessary for the following OptiX CUDA operations!
        v, f = v.contiguous(), f.contiguous()

        return v, f


@torch.jit.script
def build_rotation(r):
    """Build a rotation matrix from a quaternion, the
        default quaternion convention is (w, x, y, z).

    Args:
        r (torch.Tensor), (..., 4): the quaternion.

    Returns:
        R (torch.Tensor), (..., 3, 3): the rotation matrix
    """

    # Normalize the quaternion
    s = torch.norm(r, dim=-1)[..., None]  # (..., 1)
    q = r / s  # (..., 4)

    # Extract the quaternion components in (w, x, y, z) order
    r = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    # Build the rotation matrix, column-major
    R = torch.zeros(
        q.shape[:-1] + (3, 3), dtype=r.dtype, device=r.device
    )  # (..., 3, 3)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - r * z)
    R[..., 0, 2] = 2 * (x * z + r * y)
    R[..., 1, 0] = 2 * (x * y + r * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - r * x)
    R[..., 2, 0] = 2 * (x * z - r * y)
    R[..., 2, 1] = 2 * (y * z + r * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)

    return R


@torch.jit.script
def scaling_activation(x):
    return torch.exp(x)


@torch.jit.script
def scaling_inverse_activation(x):
    return torch.log(x.clamp(1e-6, 1e6))


@torch.jit.script
def opacity_activation(x):
    return torch.sigmoid(x)


@torch.jit.script
def inverse_opacity_activation(x):
    return torch.logit(torch.clamp(x, 1e-6, 1 - 1e-6))


@torch.jit.script
def build_scaling_rotation(s, r):
    L = torch.zeros(s.shape[:-1] + (3, 3), dtype=s.dtype, device=s.device)
    R = build_rotation(r)

    L[..., 0, 0] = s[..., 0]
    L[..., 1, 1] = s[..., 1]
    L[..., 2, 2] = s[..., 2]

    L = R @ L
    return L


@torch.jit.script
def build_cov(
    center: torch.Tensor, s: torch.Tensor, scaling_modifier: float, q: torch.Tensor
):
    L = build_scaling_rotation(
        torch.cat([s * scaling_modifier, torch.ones_like(s)], dim=-1), q
    ).permute(0, 2, 1)
    T = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device=L.device)
    T[:, :3, :3] = L
    T[:, 3, :3] = center
    T[:, 3, 3] = 1
    return T


def focal2fov(focal, pixels):
    return 2 * torch.atan(pixels / (2 * focal))


@torch.jit.script
def getWorld2View(R: torch.Tensor, t: torch.Tensor):
    """
    R: ..., 3, 3
    T: ..., 3, 1
    """
    sh = R.shape[:-2]
    T = torch.eye(4, dtype=R.dtype, device=R.device)  # 4, 4
    for i in range(len(sh)):
        T = T.unsqueeze(0)
    T = T.expand(sh + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3:] = t
    return T


@torch.jit.script
def getProjectionMatrix(fovx: torch.Tensor, fovy: torch.Tensor, znear: torch.Tensor, zfar: torch.Tensor):
    tanfovy = math.tan((fovy / 2))
    tanfovx = math.tan((fovx / 2))

    t = tanfovy * znear
    b = -t
    r = tanfovx * znear
    l = -r

    P = torch.zeros(4, 4, dtype=znear.dtype, device=znear.device)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (r - l)
    P[1, 1] = 2.0 * znear / (t - b)

    P[0, 2] = (r + l) / (r - l)
    P[1, 2] = (t + b) / (t - b)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)

    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P


class Camera(nn.Module):
    def __init__(self, H, W, K, R, T, n=0.01, f=100.0):
        super().__init__()
        self.H, self.W, self.K = H, W, K
        self.R, self.T = R, T
        self.image_height = H
        self.image_width = W
        self.FoVx = focal2fov(K[0, 0], W)
        self.FoVy = focal2fov(K[1, 1], H)
        self.world_view_transform = getWorld2View(R, T).transpose(0, 1)
        self.full_proj_transform = getProjectionMatrix(
            self.FoVx, self.FoVy,
            torch.tensor(n).to(self.R), torch.tensor(f).to(self.R)
        ).transpose(0, 1)
        self.full_proj_transform = torch.matmul(
            self.world_view_transform, self.full_proj_transform
        )
        self.camera_center = (-self.R.mT @ self.T)[..., 0]
        self.tanfovx = math.tan(self.FoVx * 0.5)
        self.tanfovy = math.tan(self.FoVy * 0.5)
