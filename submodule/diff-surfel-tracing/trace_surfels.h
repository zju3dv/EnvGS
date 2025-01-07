#pragma once
#include <cstdio>
#include <tuple>
#include <string>
#include <torch/extension.h>

#include "optix_tracer/optix_wrapper.h"


void 
BuildAccelerationStructure(
    OptiXStateWrapper& stateWrapper,
    torch::Tensor& vertices,
    torch::Tensor& triangles,
    unsigned int rebuild);


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
TraceSurfelsCUDA(
    const OptiXStateWrapper& stateWrapper,
    const bool training,
    const bool start_from_first,
    const torch::Tensor& ray_o,
    const torch::Tensor& ray_d,
    const torch::Tensor& vertices,
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& shs,
    const int degree,
    const torch::Tensor& colors_precomp,
    const torch::Tensor& others_precomp,
    const torch::Tensor& opacities,
    const torch::Tensor& scales,
    const float scale_modifier,
    const torch::Tensor& rotations,
    const torch::Tensor& transMat_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const torch::Tensor& campos,
    const bool prefiltered,
    const bool debug,
    const int max_trace_depth,
    const float specular_threshold);


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
TraceSurfelsBackwardCUDA(
    const OptiXStateWrapper& stateWrapper,
    const bool start_from_first,
    const torch::Tensor& ray_o,
    const torch::Tensor& ray_d,
    const torch::Tensor& vertices,
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& shs,
    const int degree,
    const torch::Tensor& colors_precomp,
    const torch::Tensor& others_precomp,
    const torch::Tensor& opacities,
    const torch::Tensor& scales,
    const float scale_modifier,
    const torch::Tensor& rotations,
    const torch::Tensor& transMat_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const torch::Tensor& campos,
    const bool prefiltered,
    const bool debug,
    const int max_trace_depth,
    const float specular_threshold,
    const torch::Tensor& out_rgb,
    const torch::Tensor& out_dpt,
    const torch::Tensor& out_acc,
    const torch::Tensor& out_norm,
    const torch::Tensor& out_dist,
    const torch::Tensor& out_aux,
    const torch::Tensor& mid_val,
    const torch::Tensor& dL_dout_rgb,
    const torch::Tensor& dL_dout_dpt,
    const torch::Tensor& dL_dout_acc,
    const torch::Tensor& dL_dout_norm,
    const torch::Tensor& dL_dout_dist,
    const torch::Tensor& dL_dout_aux);
