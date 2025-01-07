#define OPTIXU_MATH_DEFINE_IN_NAMESPACE

#include <optix.h>
#include <math_constants.h>

#include "params.h"
#include "auxiliary.h"


// Make the parameters available to the device code
extern "C" {
    __constant__ Params params;
}


// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color
__device__ float3 computeColorFromSH(int deg, const float3* sh, const float3& dir)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	float3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;
    return max(result, 0.0f);
}


// Compute a 2D-to-2D mapping matrix from world to splat space,
// given a 2D gaussian parameters
__device__ void compute_transmat_uv(
	const float3 p_orig,
	const float2 scale,
	float mod,
	const float4 rot,
    const float3 xyz,
	float4* world2splat,
	float3& normal,
    float2& uv
) {
    float3 R[3];
    // Convert the quaternion vector to rotation matrix, row-major, transposed version
    quat_to_rotmat_transpose(rot, R);
    float3 T = matmul33x3(R, p_orig);

	// Compute the world to splat transformation matrix
    world2splat[0] = make_float4(R[0].x, R[0].y, R[0].z, -T.x);
    world2splat[1] = make_float4(R[1].x, R[1].y, R[1].z, -T.y);
    world2splat[2] = make_float4(R[2].x, R[2].y, R[2].z, -T.z);
    world2splat[3] = make_float4(0.0f,   0.0f,   0.0f,   1.0f);

    // Return the normal in world coordinate system directly
	normal = make_float3(R[2].x, R[2].y, R[2].z);

    // Convert the intersection point from world to splat space
    float4 uv1 = matmul44x4(world2splat, make_float4(xyz.x, xyz.y, xyz.z, 1.0f));
    uv = make_float2(uv1.x / scale.x, uv1.y / scale.y);
}


// Unpack two 32-bit payload from a 64-bit pointer
static __forceinline__ __device__
void *unpackPointer(uint32_t i0, uint32_t i1) {
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr); 
    return ptr;
}
// Pack a 64-bit pointer from two 32-bit payload
static __forceinline__ __device__
void packPointer(void* ptr, uint32_t& i0, uint32_t& i1) {
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}
// Get the payload pointer
template<typename T>
static __forceinline__ __device__ T *getPayload() { 
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>(unpackPointer(u0, u1));
}


// Call optixTrace() to start a traversal
__device__ void traceStep(float3 ray_o, float3 ray_d, uint32_t payload_u0, uint32_t payload_u1)
{
    optixTrace(
        params.handle,
        ray_o,
        ray_d,
        0.0f,  // Min intersection distance
        1e16,  // Max intersection distance
        0.0f,  // rayTime, used for motion blur, disable
        OptixVisibilityMask(0xFF),
        OPTIX_RAY_FLAG_NONE,
        0,  // SBT offset
        0,  // SBT stride
        0,  // missSBTIndex
        payload_u0, payload_u1);
}


// Trace a single ray
__device__ void traceRay(
    // Trace parameters
    const float3& ray_o,  // ray direction
    const float3& ray_d,  // ray origin
    const float min_depth,  // minimum exclude depth
    const float max_depth,  // maximum exclude depth
    const float T_threshold,  // exit transmittance threshold
    // Trace depth indicator
    const int trace_depth,  // current trace depth
    // Trace output
    float* C,  // accumulated color
    float& D,  // accumulated distortion
    float& A,  // accumulated weight
    float3& N,  // accumulated normal
    float& dist,  // accumulated distortion
    float& M1,  // distortion utility
    float& M2,  // distortion utility
    float* O,  // accumulated auxiliary data
    float& T,  // terminal transmittance
    float3& E  // surface position
) {
    // Lookup current location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    uint32_t tidx = idx.x * dim.y + idx.y;

    // Set the changing ray origin and direction for the current trace
    float3 ray_ot = ray_o;
    float3 ray_dt = ray_d;

    // Creat and initialize the ray payload data
    RayPayload payload;
    IntersectionInfo buffer[CHUNK_SIZE];
    for (int i = 0; i < CHUNK_SIZE; i++) buffer[i].tmx = max_depth;
    payload.buffer = buffer;
    payload.dpt = 0.0f;
    payload.cnt = 0;
    // Pack the pointer, the values we store the payload pointer in
    uint32_t payload_u0, payload_u1;
    packPointer(&payload, payload_u0, payload_u1);

    // Initialize bookkeeping variables
    int last_gidx = -1;  // to avoid repeated contribution
	int contributor = 0;  // current trace contributor counter
    // Prepare rendering data
    float T_prev = 1.0f;
    float T_next = 1.0f;
    float dpt = 0.0f;
    float rho3d = 0.0f;
    float4 world2splat[4];
    float3 xyz;
    float3 normal;
    float2 uv;
    float3 result;

    while (1)
    {
        // Actual optixTrace
        traceStep(ray_ot, ray_dt, payload_u0, payload_u1);

        // Volume rendering
        for (int i = 0; i < CHUNK_SIZE; i++)
        {
            // Break if the intersection depth is invalid
            if (i >= payload.cnt)
                break;

            // Get the primitive index and Gaussian index
            int pidx = payload.buffer[i].idx;  // intersection primitive index
            int gidx = pidx / 2;  // Gaussian index is half of the primitive index
            // Skip the repeated Gaussian index
            if (gidx == last_gidx)
                continue;

            // Compute the actual intersection depth and coordinates in world space
            dpt = payload.buffer[i].tmx + payload.dpt;
            xyz = ray_o + dpt * ray_d;

            // Re-initialize payload data
            payload.buffer[i].tmx = max_depth;
            payload.buffer[i].idx = 0;

            // Build the world to splat transformation matrix
            // and compute the normal vector
            compute_transmat_uv(params.means3D[gidx], params.scales[gidx],
                                params.scale_modifier, params.rotations[gidx],
                                xyz, world2splat, normal, uv);
            rho3d = dot(uv, uv);

            // Adjust the normal direction
#if DUAL_VISIABLE
            float3 dir = ray_d;
            // float3 dir = params.means3D[gidx] - *params.campos;
            float cos = -sumf3(dir * normal);
            if (cos == 0) continue;
            normal = cos > 0 ? normal : -normal;
#endif

            // Exclude the Gaussian that is too close to the camera
            if (dpt < min_depth)
                continue;

            // Get weights
            float power = -0.5f * rho3d;
            if (power > 0.0f)
                continue;

            // Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
            float alpha = min(0.99f, params.opacities[gidx] * exp(power));
            if (alpha < 1.0f / 255.0f)
                continue;
            T_next = T_prev * (1 - alpha);
            if (T_next < T_threshold)
                break;

			contributor++;
            // Exit if the number of intersections exceeds the maximum
            if (contributor > MAX_INTERSECTION)
                break;
            // Keep track of the last contributed Gaussian index and intersection position
            last_gidx = gidx;

            // Compute the volume rendering weight for the current Gaussian
            float w = alpha * T_prev;

            // Render colors
            if (params.colors_precomp == nullptr)
            {
                result = computeColorFromSH(params.D, &params.shs[gidx * params.M], ray_d);
                C[0] += w * result.x;
                C[1] += w * result.y;
                C[2] += w * result.z;
            }
            else
            {
                for (int ch = 0; ch < NUM_CHANNELS; ch++)
                    C[ch] += w * params.colors_precomp[ch + NUM_CHANNELS * gidx];
            }
            // Render auxiliary data
            if (params.others_precomp != nullptr)
            {
                for (int ch = 0; ch < AUX_CHANNELS; ch++)
                    O[ch] += w * params.others_precomp[ch + AUX_CHANNELS * gidx];
            }
            // Render other componments
            D += w * dpt;
            N += w * normal;
            // TODO (xbillowy): maybe add distortion computation

            // Update transmittence
            T_prev = T_next;

            if (params.training)
            {
                // Accumulate the contribution weight for each Gaussian
                // TODO (xbillowy): profile the performance of atomicAdd()
                atomicAdd(&(params.a_weights[gidx]), w);
            }
        }

        if (T_next < T_threshold || payload.cnt < CHUNK_SIZE || contributor > MAX_INTERSECTION)
            break;

        // Re-initialize payload data
        payload.dpt = dpt + STEP_EPSILON;  // avoid self-intersection
        payload.cnt = 0;
        // Update ray origin
        ray_ot = ray_o + payload.dpt * ray_d;
    }

    // Return values
    for (int ch = 0; ch < NUM_CHANNELS; ch++)
        C[ch] += T_prev * params.background[ch];
    A = 1 - T_prev;
    T = T_prev;
    E = ray_o + D * ray_d;
}


__device__ void tracePath(
    const float3& ray_o,
    const float3& ray_d,
    const float max_trace_depth,
    float* out_rgb,
    float& out_dpt,
    float& out_acc,
    float3& out_norm,
    float3& out_dist,
    float* out_aux,
    float* mid_val
) {
    // Set the changing ray origin and direction for the current trace
    float3 ray_ot = ray_o;
    float3 ray_dt = ray_d;

    // Initialize the iterable variables
    float s_prod = 1.0f;

    // Perform path tracing
    for (int i = 0; i < max_trace_depth + 1; i++)
    {
        // Initialize the accumulated rendering data
        float C[NUM_CHANNELS] = {0.0f};
        float D = 0.0f;
        float A = 0.0f;
        float3 N = make_float3(0.0f, 0.0f, 0.0f);
        float dist = 0.0f;
        float M1 = 0.0f;
        float M2 = 0.0f;
        float O[AUX_CHANNELS] = {0.0f};
        float T = 1.0f;
        float3 E = make_float3(0.0f, 0.0f, 0.0f);

        // Prepare tracing parameters
        float min_depth = (i == 0 && params.start_from_first) ? near_n : (i == 0  && !params.start_from_first) ? 0.0f : START_OFFSET;
        float max_depth = DEPTH_INFINTY;
        float T_threshold = (i == 0 && params.start_from_first) ? 0.0001f : 0.0001f;

        // Perform the ray tracing
        traceRay(
            ray_ot,
            ray_dt,
            min_depth,
            max_depth,
            T_threshold,
            i,
            C,
            D,
            A,
            N,
            dist,
            M1,
            M2,
            O,
            T,
            E
        );

        // Update the accumulated rendering data to the output or the next trace
        if (i == 0)
        {
            // Use the first traced depth, accumulated weight, normal, and distortion as the output
            out_dpt = D;
            out_acc = A;
            out_norm = N;
            out_dist = make_float3(dist, M1, M2);
            for (int ch = 0; ch < AUX_CHANNELS; ch++)
                out_aux[ch] = O[ch];
        }
        // RGB output is much more complicated, c_o = (1 - s_i) * c_i + s_i * c_{i+1}
        float f_this = (i == max_trace_depth) ? 1.0f : 1 - O[SPECULAR_OFFSET];
        for (int ch = 0; ch < NUM_CHANNELS; ch++)
            out_rgb[ch] += C[ch] * f_this * s_prod;
        // Update the iterable specular weight
        s_prod *= O[SPECULAR_OFFSET];

        // Store the ray origin and direction for the current trace,
        // since the backward pass starts from the last trace
        mid_val[0 + RAYO_MID_OFFSET + i * MID_CHANNELS] = ray_ot.x;  // ray origin
        mid_val[1 + RAYO_MID_OFFSET + i * MID_CHANNELS] = ray_ot.y;
        mid_val[2 + RAYO_MID_OFFSET + i * MID_CHANNELS] = ray_ot.z;
        mid_val[0 + RAYD_MID_OFFSET + i * MID_CHANNELS] = ray_dt.x;  // ray direction
        mid_val[1 + RAYD_MID_OFFSET + i * MID_CHANNELS] = ray_dt.y;
        mid_val[2 + RAYD_MID_OFFSET + i * MID_CHANNELS] = ray_dt.z;
        // Record the traced results for backward access
        for (int ch = 0; ch < NUM_CHANNELS; ch++)
            mid_val[ch + RGB_MID_OFFSET + i * MID_CHANNELS] = C[ch];  // color
        mid_val[DPT_MID_OFFSET + i * MID_CHANNELS] = D;  // depth
        mid_val[ACC_MID_OFFSET + i * MID_CHANNELS] = A;  // accumulated weight
        mid_val[0 + NORM_MID_OFFSET + i * MID_CHANNELS] = N.x;  // normal
        mid_val[1 + NORM_MID_OFFSET + i * MID_CHANNELS] = N.y;
        mid_val[2 + NORM_MID_OFFSET + i * MID_CHANNELS] = N.z;
        for (int ch = 0; ch < AUX_CHANNELS; ch++)
            mid_val[ch + AUX_MID_OFFSET + i * MID_CHANNELS] = O[ch];  // auxiliaries

        // There will be no more rays to trace if the accumulated normal is zero,
        // or the accumulated specular weight is zero
        if (length(N) == 0.0f || O[SPECULAR_OFFSET] <= params.specular_threshold)
            break;

        // Update the ray origin and direction for the next trace
        float3 n = normalize(N);
        ray_dt = ray_dt - 2 * dot(n, ray_dt) * n;
        ray_ot = E + START_OFFSET * ray_dt;
    }
}


// Core __raygen__ program
extern "C" __global__ void __raygen__ot()
{
    // Lookup current location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    uint32_t tidx = idx.x * dim.y + idx.y;

    // Fetch the ray origin and direction of the current pixel
    float3 ray_o = params.ray_o[tidx];
    float3 ray_d = params.ray_d[tidx];

    // Initialize the output data
    float out_rgb[NUM_CHANNELS] = {0.0f};
    float out_dpt = 0.0f;
    float out_acc = 0.0f;
    float3 out_norm = make_float3(0.0f, 0.0f, 0.0f);
    float3 out_dist = make_float3(0.0f, 0.0f, 0.0f);
    float out_aux[AUX_CHANNELS] = {0.0f};
    float mid_val[MID_CHANNELS * (MAX_TRACE_DEPTH + 1)];

    // Perform path tracing
    tracePath(
        ray_o,
        ray_d,
        params.max_trace_depth,
        out_rgb,
        out_dpt,
        out_acc,
        out_norm,
        out_dist,
        out_aux,
        mid_val
    );

    // Write the output data to the global memory
    for (int ch = 0; ch < NUM_CHANNELS; ch++)
        params.out_rgb[tidx * NUM_CHANNELS + ch] = out_rgb[ch];
    params.out_dpt[tidx] = out_dpt;
    params.out_acc[tidx] = out_acc;
    params.out_norm[tidx] = out_norm;
    params.out_dist[tidx] = out_dist;
    for (int ch = 0; ch < AUX_CHANNELS; ch++)
        params.out_aux[tidx * AUX_CHANNELS + ch] = out_aux[ch];
    for (int i = 0; i < params.max_trace_depth + 1; i++)
    {
        for (int ch = 0; ch < MID_CHANNELS; ch++)
            params.mid_val[ch + i * MID_CHANNELS + (MAX_TRACE_DEPTH + 1) * MID_CHANNELS * tidx] =
                mid_val[ch + i * MID_CHANNELS];
    }
}


// Core __anyhit__ program
extern "C" __global__ void __anyhit__ot()
{
    // https://forums.developer.nvidia.com/t/some-confusion-on-anyhit-shader-in-optix/223336
    // Get the payload pointer
    RayPayload &payload = *getPayload<RayPayload>();

    // Get the intersection tmax and the primitive index
    float tmx = optixGetRayTmax();
    uint32_t idx = optixGetPrimitiveIndex();

    // Increment the number of intersections
    if (tmx < payload.buffer[CHUNK_SIZE - 1].tmx)
    {
        // Enter this branch means current intersection is closer, we need to update the buffer
        // Increment the counter, the counter only increases when the intersection is closer
        payload.cnt += 1;

        // Temporary variable for swapping
        float tmp_tmx;
        float cur_tmx = tmx;
        uint32_t tmp_idx;
        uint32_t cur_idx = idx;

        // Insert the new primitive into the ascending t sorted list
        for (int i = 0; i < CHUNK_SIZE; ++i)
        {
            // Swap if the new intersection is closer
            if (payload.buffer[i].tmx > cur_tmx)
            {
                // Store the original buffer info
                tmp_tmx = payload.buffer[i].tmx;
                tmp_idx = payload.buffer[i].idx;
                // Update the current intersection info
                payload.buffer[i].tmx = cur_tmx;
                payload.buffer[i].idx = cur_idx;
                // Swap
                cur_tmx = tmp_tmx;
                cur_idx = tmp_idx;
            }
        }
    }

    // Ignore the intersection to continue traversal
    optixIgnoreIntersection();
}
