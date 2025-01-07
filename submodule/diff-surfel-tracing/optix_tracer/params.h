#pragma once

#include <optix.h>
#include <cuda_runtime.h>

#include "config.h"


// Define the data types
typedef unsigned int uint32_t;
typedef unsigned long int uint64_t;


// Define the data structure used in the OptiX kernel
struct RayGenData {};
struct HitGroupData {};
struct MissData {};


// Define the global data structure used in the OptiX kernel
struct Params
{
    // OptiX handler
    OptixTraversableHandle handle;

    // Global parameters
    bool training;  // training or testing
    int max_trace_depth;  // maximum trace depth
    bool start_from_first;  // input current trace depth, default -1 as invalid
    float specular_threshold;  // threshold for specular reflection

    // Input parameters
    int P, H, W, D, M;  // Gaussian number, height, width, SHs number
    float3* ray_o;  // (H, W, 3), ray origin
    float3* ray_d;  // (H, W, 3), ray direction
    float3* vertices;  // (P * 2, 3), primitive vertices
    float* background;  // (3), background color
    float3* means3D;  // (P, 3), center coordinates
    float3* shs;  // (P, M, 3), SHs
    float* colors_precomp;  // (P, C), precomputed parameters
    float* others_precomp;  // (P, C), auxiliary parameters
    float* opacities;  // (P, 1), opacities
    float2* scales;  // (P, 2), scales
    float scale_modifier;
    float4* rotations;  // (P, 4), rotations
    float* transMat_precomp;
    float* viewmatrix;
    float* projmatrix;
    float3* campos;

    // Output forward results
    float* out_rgb;  // (H, W, C), RGB color or other features
    float* out_dpt;  // (H, W, 1), depth
    float* out_acc;  // (H, W, 1), accumulated weight
    float3* out_dist;  // (H, W, 3), distortion, M1, M2
    float3* out_norm;  // (H, W, 3), normal
    float* out_aux;  // (H, W, C), auxiliary parameters
    // Record middle result of each trace for backward pass
    float* mid_val;  // (H, W, (T + 1) * (C + 1 + 1)), traced middle results
    // Other accmulated parameters
    float* a_weights;  // (P, 1), per-Gaussian accumulated weights

    // Input upstream gradients
    float* dL_drgb;  // (H, W, C), gradient of RGB color or other features
    float* dL_ddpt;  // (H, W, 1), gradient of depth
    float* dL_dacc;  // (H, W, 1), gradient of accumulated weight
    float* dL_ddist;  // (H, W, 1), gradient of distortion
    float3* dL_dnorm;  // (H, W, 3), gradient of normal
    float* dL_daux;  // (H, W, C), gradient of auxiliary parameters

    // Output gradients
    float3* dL_dray_o;  // (H, W, 3), gradient of ray origin
    float3* dL_dray_d;  // (H, W, 3), gradient of ray direction
    float3* dL_dmeans3D;  // (P, 3), gradient of center coordinates
    float3* dL_dgrads3D;  // (P, 3), gradient for densification, same as dL_dmeans3D
    float3* dL_dgrads3D_abs;  // (P, 3), absolute version of dL_dgrads3D
    float3* dL_dshs;  // (P, M, 3), gradient of SHs
    float* dL_dcolors;  // (P, C), gradient of middle colors
    float* dL_dothers;  // (P, C), gradient of auxiliary parameters
    float* dL_dopacities;  // (P, 1), gradient of opacities
    float2* dL_dscales;  // (P, 2), gradient of scales
    float4* dL_drotations;  // (P, 4), gradient of rotations
    float* dL_dtransMat_precomp;  // (P, 9), gradient of trans matrix
};


// Define the primitive info
struct IntersectionInfo
{
    float tmx;  // t range along the ray
    uint32_t idx;  // intersection primitive ID
};
// Typedef
typedef struct IntersectionInfo IntersectionInfo;

// Define the ray payload.
// Ray pyaload is used to pass data between optixTrace
// and the programs invoked during ray traversal.
struct RayPayload
{
    float dpt;  // trace depth during the whole chunkify tracing
    uint32_t cnt;  // record number of intersections for one chunk
    IntersectionInfo* buffer;  // hit buffer for one chunk
};
// Typedef
typedef struct RayPayload RayPayload;

struct TracePayload
{
    float3 ray_o;  // ray origin of current trace
    float3 ray_d;  // ray direction of current trace
    float cur_trace_depth;  // current trace depth
};
// Typedef
typedef struct TracePayload TracePayload;
