#ifndef OPTIX_TRACER_CONFIG_H_INCLUDED
#define OPTIX_TRACER_CONFIG_H_INCLUDED

// Some global parameters
#define NUM_CHANNELS 3  // Default 3, RGB
#define AUX_CHANNELS 2  // NOTE: 1 specular + 1 roughness
#define MID_CHANNELS 16  // NOTE: 3 ray_o + 3 ray_d + 1 dpt + 1 acc + 3 normal + AUX_CHANNELS + NUM_CHANNELS
// Variables offset in `mid_val`
#define RAYO_MID_OFFSET 0  // ray origin offset in `mid_val`
#define RAYD_MID_OFFSET 3  // ray direction offset in `mid_val`
#define DPT_MID_OFFSET 6  // depth offset in `mid_val`
#define ACC_MID_OFFSET 7  // accumulated weight offset in `mid_val`
#define NORM_MID_OFFSET 8  // normal offset in `mid_val`
#define AUX_MID_OFFSET 11  // auxiliary offset in `mid_val`
#define RGB_MID_OFFSET 13  // RGB offset in `mid_val`
// Variables offset in `out_aux`
#define SPECULAR_OFFSET 0  // specular offset in `out_aux`
#define ROUGHNESS_OFFSET 1  // roughness offset in `out_aux`

#define CHUNK_SIZE 16  // Chunk size for one traversal
#define DEPTH_INFINTY 1e16f  // Default depth infinity
#define STEP_EPSILON 0.00001
#define MAX_INTERSECTION 256  // Usually around 200
#define MAX_SH_COEFFS 16  // Maximum SH coefficients

// Path tracing parameters
#define MAX_TRACE_DEPTH 1  // Maximum trace depth
// Secondary ray min trace depth
#define START_OFFSET 0.06
#define VISIBLE_THRESHOLD 0.6f

#define BLOCK_X 16
#define BLOCK_Y 16

#endif
