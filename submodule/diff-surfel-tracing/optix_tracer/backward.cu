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
__device__ float3 computeColorFromSH(int deg, const float3* sh, const float3& dir, float* clamped)
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
	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[0] = (result.x < 0);
	clamped[1] = (result.y < 0);
	clamped[2] = (result.z < 0);
	return max(result, 0.0f);
}


// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian
__device__ void computeColorFromSHBackward(int deg, const float3* sh, const float3& dir, const float* clamped, const float* dL_dcolor, float3* dL_dsh, float3& dL_ddir)
{
	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	float3 dL_dRGB = make_float3(dL_dcolor[0], dL_dcolor[1], dL_dcolor[2]);
	dL_dRGB.x *= clamped[0] ? 0 : 1;
	dL_dRGB.y *= clamped[1] ? 0 : 1;
	dL_dRGB.z *= clamped[2] ? 0 : 1;

	float3 dRGBdx = make_float3(0, 0, 0);
	float3 dRGBdy = make_float3(0, 0, 0);
	float3 dRGBdz = make_float3(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}
	// The view direction may be an input to the computation
	dL_ddir = make_float3(dot(dRGBdx, dL_dRGB), dot(dRGBdy, dL_dRGB), dot(dRGBdz, dL_dRGB));
}


// Compute a 2D-to-2D mapping matrix from world to splat space,
// given a 2D gaussian parameters
__device__ void compute_transmat_uv_forward(
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


__device__ void compute_transmat_uv_backward(
	const float3 p_orig,
	const float2 scale, 
	float mod,
	const float4 rot,
    const float3 xyz,
	const float4* world2splat,
	const float normal_sign,
    const float2 uv,
	const float3 ray_o,
	const float3 ray_d,
	const float dpt,
	const float3 v1,
	const float3 v2,
	const float3 v3,
	const float3 h1,
	const float3 h2,
	const float3 h3,
	const float G,
	const float power_clamped,
	const float3 dL_dN,
	const float dL_dD,
	const float dL_dG,
	float3& dL_dray_o,
	float3& dL_dray_d,
	float2& dL_dscale,
	float4& dL_drot,
	float3& dL_dmean3D
)
{
	// Compute the gradient w.r.t. the uv
	float2 dL_duv = dL_dG * power_clamped * -G * uv;

	float3 dL_dR[3];
	// Compute the gradient w.r.t. the transposed rotation matrix
	dL_dR[0] = dL_duv.x * (xyz - p_orig) / scale.x;
	dL_dR[1] = dL_duv.y * (xyz - p_orig) / scale.y;
	dL_dR[2] = dL_dN * normal_sign;

	// Update the gradient w.r.t. the scale
	dL_dscale = dL_dG * power_clamped * (G * uv * uv / scale);

	// Update the gradient w.r.t. the mean3D
	float3 dG_dmean3D = G * (to_float3(world2splat[0]) * uv.x / scale.x + to_float3(world2splat[1]) * uv.y / scale.y);
	dL_dmean3D = dL_dG * power_clamped * dG_dmean3D;

	// Compute the gradient flow through the ray-triangle intersection
	float3 dL_dxyz = make_float3(
		dL_duv.x / scale.x * world2splat[0].x + dL_duv.y / scale.y * world2splat[1].x,
		dL_duv.x / scale.x * world2splat[0].y + dL_duv.y / scale.y * world2splat[1].y,
		dL_duv.x / scale.x * world2splat[0].z + dL_duv.y / scale.y * world2splat[1].z
	);
	float dL_dd = dL_dD + dot(dL_dxyz, ray_d);
	// Compute the gradient w.r.t. the triangle vertices
	// Define some useful middle variables
	float3 n = cross(v2 - v1, v3 - v1);
	float3 c = v1 - ray_o;
	// Numerator and denominator aliases
	float p = dot(n, c);
	float q = dot(n, ray_d);
	// Chain rule for the gradient
	float3 dL_dn = (c - p / q * ray_d) / q;
	float3 dL_dv1 = dL_dd * cross(v2 - v3, dL_dn) + dL_dd * n / q;
	float3 dL_dv2 = dL_dd * cross(v3 - v1, dL_dn);
	float3 dL_dv3 = dL_dd * cross(v1 - v2, dL_dn);

	// Update the gradient w.r.t. the ray origin and direction
	dL_dray_o += dL_dxyz + dL_dd * -n / q;
	dL_dray_d += dL_dxyz * dpt + dL_dd * -p / (q * q) * n;

	// Update the gradient w.r.t. the transposed rotation matrix R
	dL_dR[0].x += scale.x * (h1.x * dL_dv1.x + h2.x * dL_dv2.x + h3.x * dL_dv3.x);
	dL_dR[0].y += scale.x * (h1.x * dL_dv1.y + h2.x * dL_dv2.y + h3.x * dL_dv3.y);
	dL_dR[0].z += scale.x * (h1.x * dL_dv1.z + h2.x * dL_dv2.z + h3.x * dL_dv3.z);
	dL_dR[1].x += scale.y * (h1.y * dL_dv1.x + h2.y * dL_dv2.x + h3.y * dL_dv3.x);
	dL_dR[1].y += scale.y * (h1.y * dL_dv1.y + h2.y * dL_dv2.y + h3.y * dL_dv3.y);
	dL_dR[1].z += scale.y * (h1.y * dL_dv1.z + h2.y * dL_dv2.z + h3.y * dL_dv3.z);
	// Update gradient w.r.t. rotation
	dL_drot = quat_to_rotmat_vjp(rot, dL_dR);

	// Update the gradient w.r.t. the scale
	dL_dscale.x += dot(to_float3(world2splat[0]), h1.x * dL_dv1 + h2.x * dL_dv2 + h3.x * dL_dv3);
	dL_dscale.y += dot(to_float3(world2splat[1]), h1.y * dL_dv1 + h2.y * dL_dv2 + h3.y * dL_dv3);

	// Update the gradient w.r.t. the mean3D
	dL_dmean3D += dL_dv1 + dL_dv2 + dL_dv3;
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


// Call optixTrace to trace a single ray
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


__device__ void traceRay(
    // Trace parameters
    const float3& ray_o,
    const float3& ray_d,
    const float min_depth,
    const float max_depth,
    const float T_threshold,
    // Trace depth indicator
    const int trace_depth,
    // Accumulated results
    const float* out_rgb,
    const float& out_dpt,
    const float& out_acc,
    const float3& out_norm,
    const float3& out_dist,
    const float* out_aux,
	// Upstream gradient
	const float* dL_drgb,
	const float dL_ddpt,
	const float dL_dacc,
	const float3 dL_dnorm,
	const float dL_ddist,
	const float* dL_daux,
	// Trace output
    float* C,
	float* clamped,
    float& D,
    float& W,
    float3& N,
    float& dist,
    float& M1,
    float& M2,
    float* O,
	// Middle gradient
	float3& dL_dray_o,
	float3& dL_dray_d
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
	float c[NUM_CHANNELS] = {0.0f};
    float dpt = 0.0f;
    float rho3d = 0.0f;
    float4 world2splat[4];
    float3 xyz;
    float3 normal;
	float normal_sign = 1.0f;
    float2 uv;
    float3 result;

	float cutoff;
#if TIGHTBBOX
	// The effective extent maybe depend on the opacity of Gaussian
	cutoff = sqrtf(max(9.f + 2.f * logf(params.opacities[gidx]), 0.000001));
#else
	cutoff = 3.0f;
#endif

	// The accumulated transmittance is required via backward
	// TODO (xbillowy): maybe remove this?
	float T_final = 1.0f - out_acc;
	// Per-Gaussian output gradient
	float dL_dcolor[NUM_CHANNELS];
	float3 dL_dsh[MAX_SH_COEFFS];
	float2 dL_dscale;
	float4 dL_drot;
	float3 dL_dmean3D;

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
            compute_transmat_uv_forward(params.means3D[gidx], params.scales[gidx],
                                		params.scale_modifier, params.rotations[gidx],
										xyz, world2splat, normal, uv);
            rho3d = dot(uv, uv);

			// Adjust the normal direction and get the sign
#if DUAL_VISIABLE
			float3 dir = ray_d;
			// float3 dir = params.means3D[gidx] - *params.campos;
			float cos = -sumf3(dir * normal);
			if (cos == 0) continue;
			normal_sign = cos > 0 ? 1.0f : -1.0f;
			normal = normal_sign * normal;
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
			const float G = exp(power);
            float alpha = min(0.99f, params.opacities[gidx] * G);
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
                result = computeColorFromSH(params.D, &params.shs[gidx * params.M], ray_d, clamped);
                C[0] += w * result.x;
                C[1] += w * result.y;
                C[2] += w * result.z;
				c[0] = result.x;
				c[1] = result.y;
				c[2] = result.z;
            }
            else
            {
                for (int ch = 0; ch < NUM_CHANNELS; ch++)
				{
                    C[ch] += w * params.colors_precomp[ch + NUM_CHANNELS * gidx];
					c[ch] = params.colors_precomp[ch + NUM_CHANNELS * gidx];
				}
            }
            // Render auxiliary data
            if (params.others_precomp != nullptr)
            {
                for (int ch = 0; ch < AUX_CHANNELS; ch++)
                    O[ch] += w * params.others_precomp[ch + AUX_CHANNELS * gidx];
            }
			// We need the depth and normal to compute the reflection direction
			D += w * dpt;
			N += w * normal;
			// TODO (xbillowy): maybe add distortion computation

			// Backward pass here
			// NOTE: dL_dalpha = dL_dF * (T_{i-1} * f_i + (F - F_i) / (1 - alpha_i))
			float dL_dalpha = 0.0f;
			float numerator = 1.0f - alpha;  // better accuracy?

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel pair)
			for (int ch = 0; ch < NUM_CHANNELS; ch++)
			{
				const float channel = c[ch];
				const float dL_dchannel = dL_drgb[ch];
				dL_dcolor[ch] = dL_dchannel * w;
				// Update the gradients w.r.t. color of the Gaussian.
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian
				atomicAdd(&(params.dL_dcolors[ch + NUM_CHANNELS * gidx]), dL_dchannel * w);
				// Update the gradients w.r.t. the alpha through the color
				dL_dalpha += dL_dchannel * (T_prev * channel - (out_rgb[ch] - C[ch]) / numerator);
				// Account for fact that alpha also influences how much of
				// the background color is added if nothing left to blend
				dL_dalpha += dL_dchannel * params.background[ch] * (-T_final / numerator);
			}
			// Propagate gradients to per-Gaussian auxiliary data
			if (params.others_precomp != nullptr)
			{
				for (int ch = 0; ch < AUX_CHANNELS; ch++)
				{
					const float channel = params.others_precomp[ch + AUX_CHANNELS * gidx];
					const float dL_dchannel = dL_daux[ch];
					atomicAdd(&(params.dL_dothers[ch + AUX_CHANNELS * gidx]), dL_dchannel * w);
					dL_dalpha += dL_dchannel * (T_prev * channel - (out_aux[ch] - O[ch]) / numerator);
				}
			}
			// Propagate gradients to per-Gaussian ray-splat intersection depth
			float dL_dD = dL_ddpt * w;
			dL_dalpha += dL_ddpt * (T_prev * dpt - (out_dpt - D) / numerator);
			// Propagate gradients to per-Gaussian normal
			float3 dL_dN = dL_dnorm * w;
			dL_dalpha += sumf3(dL_dnorm * (T_prev * normal - (out_norm - N) / numerator));
			// Propagate gradients to per-Gaussian weight
			dL_dalpha += dL_dacc * (T_prev * 1.f - (out_acc - W) / numerator);
			// Other components are only used in the first trace
			// TODO (xbillowy): implement the distortion loss gradient

			// Helpful reusable temporary variables
			const float dL_dG = dL_dalpha * params.opacities[gidx];
			float power_clamped = (params.opacities[gidx] * G) > 0.99f ? 0.0f : 1.0f;

			// Update gradients w.r.t. opacity of the Gaussian
			float dL_dopacity = dL_dalpha * G * power_clamped;
			atomicAdd(&(params.dL_dopacities[gidx]), dL_dopacity);

			// Compute gradients w.r.t. scaling, rotation, position of the Gaussian
			float3 v1, v2, v3, h1, h2, h3;
			// Prepare primitive vertices and local coordinate
			if (pidx % 2 == 0)
			{
				v1 = params.vertices[gidx * 4 + 0];
				v2 = params.vertices[gidx * 4 + 1];
				v3 = params.vertices[gidx * 4 + 2];
				// This should be consistent with the bvh buliding process
				h1 = make_float3(-1.0f,  1.0f, 1.0f) * cutoff;
				h2 = make_float3(-1.0f, -1.0f, 1.0f) * cutoff;
				h3 = make_float3( 1.0f,  1.0f, 1.0f) * cutoff;
			}
			else
			{
				v1 = params.vertices[gidx * 4 + 1];
				v2 = params.vertices[gidx * 4 + 2];
				v3 = params.vertices[gidx * 4 + 3];
				// This should be consistent with the bvh buliding process
				h1 = make_float3(-1.0f, -1.0f, 1.0f) * cutoff;
				h2 = make_float3( 1.0f,  1.0f, 1.0f) * cutoff;
				h3 = make_float3( 1.0f, -1.0f, 1.0f) * cutoff;
			}
			compute_transmat_uv_backward(params.means3D[gidx], params.scales[gidx],
										 params.scale_modifier, params.rotations[gidx],
										 xyz, world2splat, normal_sign, uv, ray_o, ray_d, dpt, v1, v2, v3, h1, h2, h3,
										 G, power_clamped, dL_dN, dL_dD, dL_dG,
										 dL_dray_o, dL_dray_d, dL_dscale, dL_drot, dL_dmean3D);

			// Update gradients w.r.t. scaling
			atomicAdd(&(params.dL_dscales[gidx].x), dL_dscale.x);
			atomicAdd(&(params.dL_dscales[gidx].y), dL_dscale.y);
			// Update gradients w.r.t. rotation
			atomicAdd(&(params.dL_drotations[gidx].x), dL_drot.x);
			atomicAdd(&(params.dL_drotations[gidx].y), dL_drot.y);
			atomicAdd(&(params.dL_drotations[gidx].z), dL_drot.z);
			atomicAdd(&(params.dL_drotations[gidx].w), dL_drot.w);
			// Update gradients w.r.t. position of the Gaussian
			atomicAdd(&(params.dL_dmeans3D[gidx].x), dL_dmean3D.x);
			atomicAdd(&(params.dL_dmeans3D[gidx].y), dL_dmean3D.y);
			atomicAdd(&(params.dL_dmeans3D[gidx].z), dL_dmean3D.z);

			// Update the accumulated gradients for densification
			// NOTE: scale the gradients by half depth to avoid far distance pruning
			atomicAdd(&(params.dL_dgrads3D[gidx].x), dL_dmean3D.x * 0.5f * dpt);
			atomicAdd(&(params.dL_dgrads3D[gidx].y), dL_dmean3D.y * 0.5f * dpt);
			atomicAdd(&(params.dL_dgrads3D[gidx].z), dL_dmean3D.z * 0.5f * dpt);

			// Compute the gradient w.r.t. the SHs if they are present
			if (params.colors_precomp == nullptr)
			{
				computeColorFromSHBackward(params.D, &params.shs[gidx * params.M], ray_d, clamped,
										   dL_dcolor, dL_dsh, dL_dray_d);
				// Update gradients w.r.t. SHs
				for (int j = 0; j < (params.D + 1) * (params.D + 1); j++)
				{
					atomicAdd(&(params.dL_dshs[j + params.M * gidx].x), dL_dsh[j].x);
					atomicAdd(&(params.dL_dshs[j + params.M * gidx].y), dL_dsh[j].y);
					atomicAdd(&(params.dL_dshs[j + params.M * gidx].z), dL_dsh[j].z);
				}
			}

            // Update transmittence
            T_prev = T_next;
		}

		if (T_next < T_threshold || payload.cnt < CHUNK_SIZE || contributor > MAX_INTERSECTION)
            break;

        // Re-initialize payload data
        payload.dpt = dpt + STEP_EPSILON;  // avoid self-intersection
        payload.cnt = 0;
        // Update Ray origin
        ray_ot = ray_o + payload.dpt * ray_d;
	}
}


__device__ void tracePath(
    const float3& ray_o,
    const float3& ray_d,
    const float max_trace_depth,
    const float* out_rgb,
    const float& out_dpt,
    const float& out_acc,
    const float3& out_norm,
    const float3& out_dist,
    const float* out_aux,
    const float* mid_val,
	const float* dL_dout_rgb,
	const float dL_dout_dpt,
	const float dL_dout_acc,
	const float3 dL_dout_norm,
	const float dL_dout_dist,
	const float* dL_dout_aux,
	float3& dL_dray_o,
	float3& dL_dray_d
) {
    // Instantiate the changing ray origin and direction
    float3 ray_ot;
    float3 ray_dt;

    // Initialize the iterable variables
	float C_prev[NUM_CHANNELS];
	for (int ch = 0; ch < NUM_CHANNELS; ch++)
		C_prev[ch] = out_rgb[ch];
	float c_prev[NUM_CHANNELS] = {0.0f};
	float s_prod = 1.0f;
	// The backward pass is performed from last trace to the first trace,
	// so we need to set the iterable variables to the last trace
	for (int i = 0; i < max_trace_depth; i++)
	{
		// The upstream gradient from final color to the current color
		// and specular component are complicated
		for (int ch = 0; ch < NUM_CHANNELS; ch++)
		{
			// Update the gradient w.r.t. the specular component
			float c_curr = mid_val[ch + RGB_MID_OFFSET + i * MID_CHANNELS];
			C_prev[ch] = C_prev[ch] + s_prod * (c_prev[ch] - c_curr);
			c_prev[ch] = c_curr;
		}
		// Update the iterable specular
		s_prod *= mid_val[SPECULAR_OFFSET + AUX_MID_OFFSET + i * MID_CHANNELS];
	}

    // Perform path tracing for backward
    for (int i = max_trace_depth; i >= 0; i--)
    {
		// The ray origin and direction are stored in `mid_val`
		ray_ot = make_float3(
			mid_val[0 + RAYO_MID_OFFSET + i * MID_CHANNELS],
			mid_val[1 + RAYO_MID_OFFSET + i * MID_CHANNELS],
			mid_val[2 + RAYO_MID_OFFSET + i * MID_CHANNELS]
		);
		ray_dt = make_float3(
			mid_val[0 + RAYD_MID_OFFSET + i * MID_CHANNELS],
			mid_val[1 + RAYD_MID_OFFSET + i * MID_CHANNELS],
			mid_val[2 + RAYD_MID_OFFSET + i * MID_CHANNELS]
		);
		// Fetch the previous specular component first
		float s_prev = (i > 0) ? mid_val[SPECULAR_OFFSET + AUX_MID_OFFSET + (i - 1) * MID_CHANNELS] : 1.0f;

		// Skip the trace if the ray direction is invalid,
		// or the specular component is zero
		if (length(ray_dt) == 0.0f || s_prev <= params.specular_threshold)
			continue;

		// Store the accumulated results
		float rgb[NUM_CHANNELS] = {0.0f};
		float dpt = 0.0f;
		float acc = 0.0f;
		float3 norm = make_float3(0.0f, 0.0f, 0.0f);
		float3 dist = make_float3(0.0f, 0.0f, 0.0f);
		float aux[AUX_CHANNELS] = {0.0f};
		// The RGB, accumulated weight and auxiliaries outputs are
		// stored in `mid_val` during the forward pass
		for (int ch = 0; ch < NUM_CHANNELS; ch++)
			rgb[ch] = mid_val[ch + RGB_MID_OFFSET + i * MID_CHANNELS];
		dpt = mid_val[DPT_MID_OFFSET + i * MID_CHANNELS];
		acc = mid_val[ACC_MID_OFFSET + i * MID_CHANNELS];
		norm = make_float3(
			mid_val[0 + NORM_MID_OFFSET + i * MID_CHANNELS],
			mid_val[1 + NORM_MID_OFFSET + i * MID_CHANNELS],
			mid_val[2 + NORM_MID_OFFSET + i * MID_CHANNELS]
		);
		for (int ch = 0; ch < AUX_CHANNELS; ch++)
			aux[ch] = mid_val[ch + AUX_MID_OFFSET + i * MID_CHANNELS];

		// Store upstream gradients for current trace
		float dL_drgb[NUM_CHANNELS] = {0.0f};
		float dL_ddpt = 0.0f;
		float dL_dacc = 0.0f;
		float3 dL_dnorm = make_float3(0.0f, 0.0f, 0.0f);
		float dL_ddist = 0.0f;
		float dL_daux[AUX_CHANNELS] = {0.0f};
		// The upstream gradient from final color to the current color
		// and specular component are much more complicated
		float f_this = (i == max_trace_depth) ? 1.0f : 1 - aux[SPECULAR_OFFSET];
		// Update the gradient w.r.t. the current color
		for (int ch = 0; ch < NUM_CHANNELS; ch++)
			dL_drgb[ch] = dL_dout_rgb[ch] * f_this * s_prod;
		// Update the iterative specular weight
		s_prod /= s_prev;
		// For traces other than the last one, there are many more gradients,
		// 1. gradients from output color to the current specular component
		// 2. gradients from last trace to the current depth and normal through ray
		if (i != max_trace_depth)
		{
			// Update the gradient w.r.t. the specular component
			for (int ch = 0; ch < NUM_CHANNELS; ch++)
			{
				dL_daux[SPECULAR_OFFSET] += dL_dout_rgb[ch] * (C_prev[ch] / aux[SPECULAR_OFFSET]);
				c_prev[ch] = (i > 0) ? mid_val[ch + RGB_MID_OFFSET + (i - 1) * MID_CHANNELS] : 0.0f;
				C_prev[ch] = C_prev[ch] - s_prod * (c_prev[ch] - rgb[ch]);
			}

			// `dL_dray_o` is the gradient w.r.t. the ray origin of last trace i+1
			// `ray_dt` is the direction of the current trace i
			if (length(norm) != 0)
			{
				// The upstream gradient of d_{i+1} has a component from dL_o_{i+1},
				// since o_{i+1} = o_i + dpt * d_i + STEP_EPSILON * d_{i+1}
				dL_dray_d = dL_dray_d + dL_dray_o * STEP_EPSILON;
				// Update the gradient w.r.t. the depth and normal of the current trace i
				float3 n = normalize(norm);
				dL_ddpt += sumf3(dL_dray_o * ray_dt);
				dL_dnorm += dnormvdv(norm, ddotndndn(n, ray_dt, dL_dray_d * -2.0f));
				// Update the gradient w.r.t. the ray origin and direction of the current trace i
				dL_dray_o = dL_dray_o;
				dL_dray_d = dL_dray_o * dpt + dL_dray_d + ddotndndd(n, ray_dt, dL_dray_d * -2.0f);
			}
		}
		// The first trace is special
		if (i == 0)
		{
			// Upstream gradient for depth, accumulated weight, normal
			dL_ddpt += dL_dout_dpt;
			dL_dacc += dL_dout_acc;
			dL_dnorm += dL_dout_norm;
			dL_ddist += dL_dout_dist;
			for (int ch = 0; ch < AUX_CHANNELS; ch++)
				dL_daux[ch] += dL_dout_aux[ch];
		}

        // Initialize the accumulated rendering data
        float C[NUM_CHANNELS] = {0.0f};
		float clamped[3] = {0.0f};
        float D = 0.0f;
        float W = 0.0f;
        float3 N = make_float3(0.0f, 0.0f, 0.0f);
        float DS = 0.0f;
        float M1 = 0.0f;
        float M2 = 0.0f;
        float O[AUX_CHANNELS] = {0.0f};

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
            rgb,
			dpt,
			acc,
			norm,
			dist,
			aux,
			dL_drgb,
			dL_ddpt,
			dL_dacc,
			dL_dnorm,
			dL_ddist,
			dL_daux,
			C,
			clamped,
			D,
			W,
			N,
			DS,
			M1,
			M2,
			O,
			dL_dray_o,
			dL_dray_d
        );
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

    // Fetch the accumulated rendering results
    float out_rgb[NUM_CHANNELS];
	for (int ch = 0; ch < NUM_CHANNELS; ch++)
		out_rgb[ch] = params.out_rgb[ch + NUM_CHANNELS * tidx];
	float out_dpt = params.out_dpt[tidx];
	float out_acc = params.out_acc[tidx];
	float3 out_norm = params.out_norm[tidx];
	float3 out_dist = params.out_dist[tidx];
	float out_aux[AUX_CHANNELS];
	for (int ch = 0; ch < AUX_CHANNELS; ch++)
		out_aux[ch] = params.out_aux[ch + AUX_CHANNELS * tidx];
	float mid_val[MID_CHANNELS * (MAX_TRACE_DEPTH + 1)];
	for (int i = 0; i < params.max_trace_depth + 1; i++)
	{
		for (int ch = 0; ch < MID_CHANNELS; ch++)
			mid_val[ch + i * MID_CHANNELS] =
				params.mid_val[ch + i * MID_CHANNELS + MID_CHANNELS * (MAX_TRACE_DEPTH + 1) * tidx];
	}

	// Fetch the upstream gradient from torch
	float dL_drgb[NUM_CHANNELS];
	for (int ch = 0; ch < NUM_CHANNELS; ch++)
		dL_drgb[ch] = params.dL_drgb[ch + NUM_CHANNELS * tidx];
	float dL_ddpt = params.dL_ddpt[tidx];
	float dL_dacc = params.dL_dacc[tidx];
	float3 dL_dnorm = params.dL_dnorm[tidx];
	float dL_ddist = params.dL_ddist[tidx];
	float dL_daux[AUX_CHANNELS];
	for (int ch = 0; ch < AUX_CHANNELS; ch++)
		dL_daux[ch] = params.dL_daux[ch + AUX_CHANNELS * tidx];

	// Initialize the middle gradient w.r.t. the ray origin and direction
	float3 dL_dray_o = make_float3(0.0f, 0.0f, 0.0f);
	float3 dL_dray_d = make_float3(0.0f, 0.0f, 0.0f);

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
        mid_val,
		dL_drgb,
		dL_ddpt,
		dL_dacc,
		dL_dnorm,
		dL_ddist,
		dL_daux,
		dL_dray_o,
		dL_dray_d
    );

	// Store the gradients w.r.t. the ray origin and direction
	params.dL_dray_o[tidx] = dL_dray_o;
	params.dL_dray_d[tidx] = dL_dray_d;
}


// Core __anyhit__ program
// Same as the forward pass, we need to sort the intersections
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
