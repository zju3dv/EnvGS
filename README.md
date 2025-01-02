# EnvGS: Modeling View-Dependent Appearance with Environment Gaussian

## [Project Page](https://zju3dv.github.io/envgs) | [Paper](https://arxiv.org/abs/2412.15215) | [arXiv](https://arxiv.org/abs/2412.15215)

<!-- ![python](https://img.shields.io/github/languages/top/zju3dv/EnvGS)
![star](https://img.shields.io/github/stars/zju3dv/EnvGS)
[![license](https://img.shields.io/badge/license-zju3dv-white)](LICENSE) -->

> EnvGS: Modeling View-Dependent Appearance with Environment Gaussian<br>
> [Tao Xie*](https://github.com/xbillowy), [Xi Chen*](https://github.com/Burningdust21), [Zhen Xu](https://zhenx.me), [Yiman Xie](https://zju3dv.github.io/envgs/), [Yudong Jin](https://github.com/krahets), [Yujun Shen](https://shenyujun.github.io), [Sida Peng](https://pengsida.net), [Hujun Bao](http://www.cad.zju.edu.cn/home/bao), [Xiaowei Zhou](https://xzhou.me)<br>
> arXiv 2024

![teaser](assets/teaser.png)

## Pipeline

![pipeline](assets/pipeline.png)

## Code

Our code and Gaussian tracer are coming soon, stay tuned!
- [ ] Release 2D Gaussian ray tracer.
- [ ] Release evaluation and training code.

## Acknowledgments

This work is implemented using our PyTorch framework, [EasyVolcap](https://github.com/zju3dv/EasyVolcap), feel free to explore it.

- [EasyVolcap: Accelerating Neural Volumetric Video Research](https://github.com/zju3dv/EasyVolcap)

We would also like to acknowledge the following inspiring prior work:

- [NeRF-Casting: Improved View-Dependent Appearance with Consistent Reflections](https://dorverbin.github.io/nerf-casting/)
- [3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes](https://gaussiantracer.github.io/)
- [Ref-NeRF: Structured View-Dependent Appearance for Neural Radiance Fields](https://dorverbin.github.io/refnerf/)
- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://github.com/graphdeco-inria/gaussian-splatting)
- [2DGS: 2D Gaussian Splatting for Geometrically Accurate Radiance Fields](https://surfsplatting.github.io/)

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@article{xie2024envgs,
  title={EnvGS: Modeling View-Dependent Appearance with Environment Gaussian},
  author={Xie, Tao and Chen, Xi and Xu, Zhen and Xie, Yiman and Jin, Yudong and Shen, Yujun and Peng, Sida and Bao, Hujun and Zhou, Xiaowei},
  journal={arXiv preprint arXiv:2412.15215},
  year={2024}
}
```
