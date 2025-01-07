# Differential Surfel Tracing

This is a differentiable 2D Gaussian ray tracer, built on the foundation of [2DGS](https://surfsplatting.github.io/) and NVIDIA OptiX, tailored for differentiable optimization and rendering tasks. Key features of this tracer include:

- **Differentiability**: The tracer is entirely differentiable, encompassing the 2D Gaussian parameters and the input ray origins and directions, should the rays be optimized.
- **Path Tracing**: It supports path tracing with multiple bounces, which is beneficial for rendering complex materials and simulating intricate light transport phenomena.
- **Customizable Rendering**: The tracer allows for customized rendering, enabling you to incorporate additional precomputed parameters and outputs tailored to your specific requirements.

https://github.com/xbillowy/assets/diff-surfel-tracing/assets/bb3095a6-71ed-4f55-8b77-0effbc85af37


## 📦 Installation

The installation is similar to the installation of [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization/tree/main) and [diff-surfel-rasterization](https://github.com/hbb1/diff-surfel-rasterization) except that the [NVIDIA OptiX SDK](https://developer.nvidia.com/designworks/optix/download) is required.

Download the OptiX SDK from the [NVIDIA official website](https://developer.nvidia.com/designworks/optix/download). Note that **OptiX 7.7.0** and **CUDA 11.8** are recommended, OptiX 7.1.0 and lower are not compatible due to API changes. After installing the OptiX SDK, set the environment variable `OPTIX_HOME` to the download directory of the OptiX SDK to your `.zshrc` or `.bashrc` to expose related paths for compilation.

```bash
# CUDA related configs, you may need to change the path according to your installation
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME="/usr/local/cuda"
# Set the environment variable OPTIX_HOME to the installation directory of the OptiX SDK
export OPTIX_HOME=/path/to/optix
```

Then, install the tracer from local clone:

```bash
# Clone the repository
git clone https://github.com/zju3dv/EnvGS.git

# Install using pip from the local clone
cd EnvGS
pip install submodule/diff-surfel-tracing
```

Or the latest commit from GitHub:

```bash
pip install -v git+https://github.com/xbillowy/diff-surfel-tracing#subdirectory=submodule/diff-surfel-tracing
```


## 🛠️ Usage

We provide a simple example in the [`example/render.py`](example/render.py) to demonstrate how to use the tracer. To use the tracer, you can download our pre-trained 2DGS model and a pre-defined camera path from the [Google Drive](https://drive.google.com/file/d/1drKlXptpkht0ZVp6Ywh8ZSSXsi8RddKx/view?usp=sharing). Once you have the tracer installed and the example data downloaded, unzip the files to the root directory of this repository, and run the following commands to render the example scene:

```bash
# Install the dependencies
pip install -r example/requirements.txt

# Run the example to render the scene
python example/render.py
```

The rendered RGB images, depth maps, normal maps, and corresponding videos will be saved to the `data/result/` directory.

### Core Snippets

The usage of this tracer is quite similar to the use of [diff-surfel-rasterization](https://github.com/hbb1/diff-surfel-rasterization), here is an example of how to use this tracer, note that you may need a pre-trained [2DGS](https://github.com/hbb1/2d-gaussian-splatting) model first:

```python
import torch
from diff_surfel_tracing import SurfelTracer, SurfelTracingSettings

# Create a SurfelTracer
tracer = SurfelTracer()

# Convert 2D Gaussian primitives to triangle vertices and faces
v, f = get_triangles(pcd)

# Build the acceleration structure
tracer.build_acceleration_structure(v, f, rebuild=True)

# NOTE: To avoid weird behavior, it is recommended to manually invoke the
# NOTE: `.contiguous()` on every input tensor before passing them to the tracer.

# Set the surfel tracing settings
# Check the details of the parameters in the Parameters Explanation section
tracer_settings = SurfelTracingSettings(
    image_height=int(viewpoint_camera.image_height),
    image_width=int(viewpoint_camera.image_width),
    tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
    tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
    bg=bg_color,
    scale_modifier=scale_modifier,
    viewmatrix=viewpoint_camera.world_view_transform,
    projmatrix=viewpoint_camera.full_proj_transform,
    sh_degree=pcd.active_sh_degree,
    campos=viewpoint_camera.camera_center,
    prefiltered=False,
    debug=False,
    max_trace_depth=max_trace_depth,
    specular_threshold=specular_threshold,
)

# Create dummy input to receive the gradient for densification
grads3D = (
    torch.zeros_like(
        means3D, dtype=means3D.dtype, requires_grad=True, device=means3D.device
    )
    + 0
)
try:
    grads3D.retain_grad()
except:
    pass

# Perform the ray tracing
# Check the details of the inputs in the Parameters Explanation section
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
```

### 2DGS to Triangles

The `get_triangles` function is used to convert the 2DGS to vertices and faces in order to build the acceleration structure, here is an example of how to implement this function.

<details>

<summary>Example implementation of <code>get_triangles()</code></summary>

```python
def get_triangles(pcd: GaussianModel):
    # Build the uv tangent plane to world transformation matrix, splat2world
    T = pcd.get_covariance()  # (P, 4, 4)
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
```

</details>

### Parameters Explanation

Most parameters are consistent with [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) and [diff-surfel-rasterization](https://github.com/hbb1/diff-surfel-rasterization), here we provide details on the parameters that may be different or newly added.

<details>

<summary><code>SurfelTracingSettings</code> Parameters</summary>

- `viewmatrix`: no actual use in the ray tracing, only for consistency with the rasterizer.
- `projmatrix`: no actual use in the ray tracing, only for consistency with the rasterizer.
- `campos`: no actual use in the ray tracing, only for consistency with the rasterizer.
- **`max_trace_depth`**: number of path tracing bounces, default is 0, means only trace once.
- **`specular_threshold`**: the threshold for continuing the path tracing, default is 0.0. Ignore this if you are not using the path tracing or any BRDFs rendering.

</details>

<details>

<summary><code>SurfelTracer</code> Parameters</summary>

- **`ray_o`**: the origin of the rays, a 3-dimension Tensor of shape `(H, W, 3)` or `(B, P, 3)`.
- **`ray_d`**: the direction of the rays, a 3-dimension Tensor of shape `(H, W, 3)` or `(B, P, 3)`.
- **`v`**: the covering triangles of the 2D Gaussian splats, a 2-dimension Tensor of shape `(P, 3)`, which is used to support fully-differentiable backpropagation.
- `grads3D`: the gradient tensor for the densification, the same as [`means2D`](https://github.com/graphdeco-inria/gaussian-splatting/blob/54c035f7834b564019656c3e3fcc3646292f727d/gaussian_renderer/__init__.py#L55) in the original 3DGS rasterizer.
- `colors_precomp`: used for RGB only, since we use pixel ray direction rather than the Gaussian center minus camera center direction as the ray direction, which means the original precomputation of the color is not applicable.
- **`others_precomp`**: support custom rendering, you can add more. Remember to add the corresponding parameters and offsets in the [config.h](./optix_tracer/config.h).
- **`start_from_first`**: indicates whether the rays start from the camera or any other starting point (e.g., the bounce surface point), default is `True`.

</details>

### Outputs Explanation

In addition to the default output of [diff-surfel-rasterization](https://github.com/hbb1/2d-gaussian-splatting/blob/df1f6c684cc4e41a34937fd45a7847260e9c6cd7/gaussian_renderer/__init__.py#L97-L156), namely `rgb`, `dpt`, `acc`, `norm` for RGB image, depth map, accumulated opacity, and normal map, respectively, we also provide the following outputs: `dist`, `aux`, `mid`, `wet`, see the details below.

<details>

<summary>Additional Outputs</summary>

- `aux`: corresponding to the rendered `others_precomp` map in the input, used for custom rendering.
- `mid`: the middle rendering results for each path tracing bounce, e.g., the accumulated color, opacity, and normal of the first trace will be stored if you set `max_trace_depth` to 1.
- `wet`: the accumulated contribution weight for each 2D Gaussian splat.
- `dist`: invalid distortion map, all zeros for now, maybe implement in the future.

</details>


## 🚧 TODOs

- [x] TODO: Release the initial version.
- [ ] TODO: Test OptiX 8.0.0 and OptiX 8.1.0 compatibility.
- [ ] TODO: Apply more of the CUDA optimization techniques.


## 📚 Credits

We would like to acknowledge the following inspiring prior work:

- [3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes](https://gaussiantracer.github.io/)
- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://github.com/graphdeco-inria/gaussian-splatting)
- [2DGS: 2D Gaussian Splatting for Geometrically Accurate Radiance Fields](https://surfsplatting.github.io/)


## 📜 Citation

```bibtex
@article{xie2024envgs,
  title={EnvGS: Modeling View-Dependent Appearance with Environment Gaussian},
  author={Xie, Tao and Chen, Xi and Xu, Zhen and Xie, Yiman and Jin, Yudong and Shen, Yujun and Peng, Sida and Bao, Hujun and Zhou, Xiaowei},
  journal={arXiv preprint arXiv:2412.15215},
  year={2024}
}

@article{3dgrt2024,
    author = {Nicolas Moenne-Loccoz and Ashkan Mirzaei and Or Perel and Riccardo de Lutio and Janick Martinez Esturo and Gavriel State and Sanja Fidler and Nicholas Sharp and Zan Gojcic},
    title = {3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes},
    journal = {ACM Transactions on Graphics and SIGGRAPH Asia},
    year = {2024},
}
```
