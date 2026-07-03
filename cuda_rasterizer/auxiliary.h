/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)
#define FilterSize 0.7071067811865476
#define FilterInvSquare 1/(FilterSize*FilterSize)

#define TIGHTBBOX 0
#define RENDER_AXUTILITY 1
#define DEPTH_OFFSET 0
#define ALPHA_OFFSET 1
#define NORMAL_OFFSET 2 
#define MIDDEPTH_OFFSET 5
#define DISTORTION_OFFSET 6
#define MEDIAN_WEIGHT_OFFSET 7

// distortion helper macros
#define BACKFACE_CULL 1
#define DUAL_VISIABLE 1
#define NEAR_PLANE 0.2
#define FAR_PLANE 100.0
#define DETACH_WEIGHT 0

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float3 crossProduct(float3 a, float3 b) {
	float3 result;
	result.x = a.y * b.z - a.z * b.y;
    result.y = a.z * b.x - a.x * b.z;
    result.z = a.x * b.y - a.y * b.x;
    return result;
}

// ============================================================================
// Analytic compact tile bounding (SnugBox + AccuTile) for 2D Gaussian surfels.
//
// Instead of a loose square box + a brute-force per-tile double loop, we bound
// each splat by the *exact* screen-space conic of its rho3d footprint, unioned
// with the low-pass filter disk (rho2d), and enumerate only the tiles the
// footprint actually intersects by computing per-row tile spans in closed form.
//
// The preprocess (counting) and duplication (emitting) kernels both go through
// the same inline helpers below with identical inputs, so the tile count and
// the emitted key count are guaranteed to agree.
// ============================================================================

// Effective Mahalanobis cutoff radius (in sigmas) of a splat's footprint.
__forceinline__ __device__ float truncatedR(float opacity)
{
#if TIGHTBBOX
	// The effective extent depends on the opacity of the gaussian.
	return sqrtf(max(9.f + logf(opacity), 0.000001f));
#else
	return 3.f;
#endif
}

// Screen-space center and per-axis half-extent of the 2D Gaussian (the AABB of
// the rho3d == 1 ellipse). See Eq. (9) in the 2DGS paper. Shared by the
// preprocess and duplication kernels so both derive the exact same fallback.
__forceinline__ __device__ bool computeCenterExtent(const float* transMat, float2& center, float2& extent)
{
	glm::mat4x3 T = glm::mat4x3(
		transMat[0], transMat[1], transMat[2],
		transMat[3], transMat[4], transMat[5],
		transMat[6], transMat[7], transMat[8],
		transMat[6], transMat[7], transMat[8]
	);

	float d = glm::dot(glm::vec3(1.0, 1.0, -1.0), T[3] * T[3]);
	if (d == 0.0f) return false;
	glm::vec3 f = glm::vec3(1.0, 1.0, -1.0) * (1.0f / d);
	glm::vec3 p = glm::vec3(
		glm::dot(f, T[0] * T[3]),
		glm::dot(f, T[1] * T[3]),
		glm::dot(f, T[2] * T[3]));
	glm::vec3 h0 = p * p -
		glm::vec3(
			glm::dot(f, T[0] * T[0]),
			glm::dot(f, T[1] * T[1]),
			glm::dot(f, T[2] * T[2])
		);
	glm::vec3 h = sqrt(max(glm::vec3(0.0), h0)) + glm::vec3(0.0, 0.0, 1e-2);
	center = { p.x, p.y };
	extent = { h.x, h.y };
	return true;
}

// Precomputed analytic bounding box for one splat.
struct TileBBox
{
	uint2 rect_min;      // touched tile range [rect_min, rect_max)
	uint2 rect_max;
	bool  ellipse;       // true: bounded ellipse -> use per-row analytic spans

	// Screen-space conic of the rho3d == trunc_R^2 footprint:
	//   Q(x,y) = a x^2 + 2 b x y + c y^2 + 2 d x + 2 e y + f  <= 0
	float a, b, c, d, e, f;
	float x_lo, x_hi;        // ellipse global x-extremes (pixels)
	float y_at_xlo, y_at_xhi;// pixel rows at which those extremes occur

	// Low-pass filter disk (rho2d): center + radius.
	float cx, cy, R_lp, R_lp2;
};

// Convert a continuous pixel-space AABB to a tile range, using the exact same
// rounding as getRect() so the fallback path matches the original box.
__forceinline__ __device__ void pixelBoxToTileRect(
	float px_lo, float px_hi, float py_lo, float py_hi,
	dim3 grid, uint2& rect_min, uint2& rect_max)
{
	rect_min = {
		min(grid.x, max((int)0, (int)(px_lo / BLOCK_X))),
		min(grid.y, max((int)0, (int)(py_lo / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)ceilf(px_hi / BLOCK_X))),
		min(grid.y, max((int)0, (int)ceilf(py_hi / BLOCK_Y)))
	};
}

// Build the compact bounding box (SnugBox). Returns false if the footprint
// touches no tile (the splat can be culled).
__forceinline__ __device__ bool buildTileBBox(
	const float* transMat, float2 center, float2 extent,
	float trunc_R, dim3 grid, TileBBox& bb)
{
	const float trunc_R2 = trunc_R * trunc_R;
	bb.cx = center.x; bb.cy = center.y;
	bb.R_lp = trunc_R * FilterSize;
	bb.R_lp2 = bb.R_lp * bb.R_lp;

	// rho3d(x,y) = (p.x^2 + p.y^2) / p.z^2, where p(x,y) is *linear* in the pixel:
	//   p = x*A + y*B + Cc,  A = Tv x Tw,  B = Tw x Tu,  Cc = Tu x Tv.
	// Hence "rho3d <= trunc_R^2" is the conic p.x^2 + p.y^2 - trunc_R^2 p.z^2 <= 0.
	const float3 Tu = { transMat[0], transMat[1], transMat[2] };
	const float3 Tv = { transMat[3], transMat[4], transMat[5] };
	const float3 Tw = { transMat[6], transMat[7], transMat[8] };
	const float3 A  = crossProduct(Tv, Tw);
	const float3 B  = crossProduct(Tw, Tu);
	const float3 Cc = crossProduct(Tu, Tv);

	const float a = A.x * A.x + A.y * A.y - trunc_R2 * A.z * A.z;
	const float c = B.x * B.x + B.y * B.y - trunc_R2 * B.z * B.z;
	const float b = A.x * B.x + A.y * B.y - trunc_R2 * A.z * B.z;
	const float d = A.x * Cc.x + A.y * Cc.y - trunc_R2 * A.z * Cc.z;
	const float e = B.x * Cc.x + B.y * Cc.y - trunc_R2 * B.z * Cc.z;
	const float f = Cc.x * Cc.x + Cc.y * Cc.y - trunc_R2 * Cc.z * Cc.z;
	const float det2 = a * c - b * b;

	float px_lo, px_hi, py_lo, py_hi;
	bb.ellipse = false;

	// A bounded ellipse requires a positive-definite quadratic part.
	if (a > 0.0f && c > 0.0f && det2 > 0.0f)
	{
		// Tight AABB: extreme x where dQ/dy = 0, extreme y where dQ/dx = 0.
		//   (a c - b^2) x^2 + 2(d c - b e) x + (f c - e^2) = 0
		//   (a c - b^2) y^2 + 2(e a - b d) y + (f a - d^2) = 0
		const float Bx = d * c - b * e, Cx = f * c - e * e;
		const float By = e * a - b * d, Cy = f * a - d * d;
		const float disc_x = Bx * Bx - det2 * Cx;
		const float disc_y = By * By - det2 * Cy;
		if (disc_x > 0.0f && disc_y > 0.0f)
		{
			const float sx = sqrtf(disc_x), sy = sqrtf(disc_y);
			px_lo = (-Bx - sx) / det2; px_hi = (-Bx + sx) / det2;
			py_lo = (-By - sy) / det2; py_hi = (-By + sy) / det2;

			bb.ellipse = true;
			bb.a = a; bb.b = b; bb.c = c; bb.d = d; bb.e = e; bb.f = f;
			bb.x_lo = px_lo; bb.x_hi = px_hi;
			bb.y_at_xlo = -(b * px_lo + e) / c;
			bb.y_at_xhi = -(b * px_hi + e) / c;

			// Union with the low-pass disk (matters for thin / sub-pixel splats).
			px_lo = min(px_lo, center.x - bb.R_lp); px_hi = max(px_hi, center.x + bb.R_lp);
			py_lo = min(py_lo, center.y - bb.R_lp); py_hi = max(py_hi, center.y + bb.R_lp);
		}
	}

	if (!bb.ellipse)
	{
		// Degenerate footprint (grazing / edge-on splat): fall back to the loose
		// square used by the original code (which already covers the disk via the
		// FilterSize floor). Never worse than before.
		const float r = ceilf(trunc_R * max(max(extent.x, extent.y), (float)FilterSize));
		px_lo = center.x - r; px_hi = center.x + r;
		py_lo = center.y - r; py_hi = center.y + r;
	}

	pixelBoxToTileRect(px_lo, px_hi, py_lo, py_hi, grid, bb.rect_min, bb.rect_max);
	return (bb.rect_max.x > bb.rect_min.x) && (bb.rect_max.y > bb.rect_min.y);
}

// Analytic per-row tile span (AccuTile): for tile row ty, return the tile-x
// range [x_lo, x_hi) the footprint actually intersects. No per-tile tests.
__forceinline__ __device__ void tileRowSpanX(
	const TileBBox& bb, uint32_t ty, dim3 grid, uint32_t& x_lo, uint32_t& x_hi)
{
	if (!bb.ellipse)
	{
		// Degenerate fallback: emit the full (already-loose) rect row.
		x_lo = bb.rect_min.x;
		x_hi = bb.rect_max.x;
		return;
	}

	const float ya = (float)(ty * BLOCK_Y);
	const float yb = (float)(ty * BLOCK_Y + BLOCK_Y);

	float exlo = 1e30f, exhi = -1e30f;

	// Ellipse coverage over the horizontal strip [ya, yb]. Solving Q(x,y)=0 for x
	// at a fixed row gives a x^2 + 2(b y + d) x + (c y^2 + 2 e y + f) = 0. The
	// extreme x over the strip is attained at a strip boundary or at the ellipse's
	// global x-extreme (if its row lies inside the strip).
	#pragma unroll
	for (int it = 0; it < 2; ++it)
	{
		const float yy = (it == 0) ? ya : yb;
		const float Bc = bb.b * yy + bb.d;
		const float Cc2 = bb.c * yy * yy + 2.0f * bb.e * yy + bb.f;
		const float disc = Bc * Bc - bb.a * Cc2;
		if (disc >= 0.0f)
		{
			const float s = sqrtf(disc);
			exlo = min(exlo, (-Bc - s) / bb.a);
			exhi = max(exhi, (-Bc + s) / bb.a);
		}
	}
	if (bb.y_at_xlo >= ya && bb.y_at_xlo <= yb) exlo = min(exlo, bb.x_lo);
	if (bb.y_at_xhi >= ya && bb.y_at_xhi <= yb) exhi = max(exhi, bb.x_hi);

	// Low-pass disk coverage over the strip.
	const float dy = max(0.0f, max(ya - bb.cy, bb.cy - yb));
	if (dy < bb.R_lp)
	{
		const float half = sqrtf(bb.R_lp2 - dy * dy);
		exlo = min(exlo, bb.cx - half);
		exhi = max(exhi, bb.cx + half);
	}

	if (exlo > exhi)
	{
		// No coverage in this row.
		x_lo = bb.rect_min.x;
		x_hi = bb.rect_min.x;
		return;
	}

	int tlo = (int)(exlo / BLOCK_X);
	int thi = (int)ceilf(exhi / BLOCK_X);
	if (tlo < (int)bb.rect_min.x) tlo = (int)bb.rect_min.x;
	if (thi > (int)bb.rect_max.x) thi = (int)bb.rect_max.x;
	if (thi < tlo) thi = tlo;
	x_lo = (uint32_t)tlo;
	x_hi = (uint32_t)thi;
}

// Count the tiles the footprint touches: a single loop over rows with analytic
// spans (no nested per-tile iteration).
__forceinline__ __device__ uint32_t countTilesBBox(const TileBBox& bb, dim3 grid)
{
	uint32_t n = 0;
	for (uint32_t ty = bb.rect_min.y; ty < bb.rect_max.y; ++ty)
	{
		uint32_t x_lo, x_hi;
		tileRowSpanX(bb, ty, grid, x_lo, x_hi);
		n += (x_hi - x_lo);
	}
	return n;
}

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

// adopt from gsplat: https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/cuda/csrc/forward.cu
inline __device__ glm::mat3 quat_to_rotmat(const glm::vec4 quat) {
	// quat to rotation matrix
	float s = rsqrtf(
		quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
	);
	float w = quat.x * s;
	float x = quat.y * s;
	float y = quat.z * s;
	float z = quat.w * s;

	// glm matrices are column-major
	return glm::mat3(
		1.f - 2.f * (y * y + z * z),
		2.f * (x * y + w * z),
		2.f * (x * z - w * y),
		2.f * (x * y - w * z),
		1.f - 2.f * (x * x + z * z),
		2.f * (y * z + w * x),
		2.f * (x * z + w * y),
		2.f * (y * z - w * x),
		1.f - 2.f * (x * x + y * y)
	);
}


inline __device__ glm::vec4
quat_to_rotmat_vjp(const glm::vec4 quat, const glm::mat3 v_R) {
	float s = rsqrtf(
		quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
	);
	float w = quat.x * s;
	float x = quat.y * s;
	float y = quat.z * s;
	float z = quat.w * s;

	glm::vec4 v_quat;
	// v_R is COLUMN MAJOR
	// w element stored in x field
	v_quat.x =
		2.f * (
				  // v_quat.w = 2.f * (
				  x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
				  z * (v_R[0][1] - v_R[1][0])
			  );
	// x element in y field
	v_quat.y =
		2.f *
		(
			// v_quat.x = 2.f * (
			-2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
			z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1])
		);
	// y element in z field
	v_quat.z =
		2.f *
		(
			// v_quat.y = 2.f * (
			x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
			z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2])
		);
	// z element in w field
	v_quat.w =
		2.f *
		(
			// v_quat.z = 2.f * (
			x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
			2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0])
		);
	return v_quat;
}


inline __device__ glm::mat3
scale_to_mat(const float3 scale, const float glob_scale) {
	glm::mat3 S = glm::mat3(1.f);
	S[0][0] = glob_scale * scale.x;
	S[1][1] = glob_scale * scale.y;
	S[2][2] = glob_scale * scale.z;
	return S;
}



#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif
