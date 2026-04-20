#include <metal_stdlib>

using namespace metal;

struct PreprocessBackwardKernelParams {
  uint n;
  uint image_width;
  uint image_height;
  uint use_cov3d_precomp;
};

inline float3 read_packed_float3(const device float* arr, uint idx) {
  return float3(arr[3 * idx], arr[3 * idx + 1], arr[3 * idx + 2]);
}

inline float4 mul_mat4_vec3_h(const device float* m, float3 p) {
  return float4(
      m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12],
      m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13],
      m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14],
      m[3] * p.x + m[7] * p.y + m[11] * p.z + m[15]);
}

inline float3 read_packed_float3(const device float* arr, uint idx, uint stride) {
  return float3(arr[stride * idx], arr[stride * idx + 1], arr[stride * idx + 2]);
}

inline float4 read_packed_float4(const device float* arr, uint idx, uint stride) {
  return float4(arr[stride * idx], arr[stride * idx + 1], arr[stride * idx + 2], arr[stride * idx + 3]);
}

kernel void fastgs_preprocess_backward_kernel(
    constant PreprocessBackwardKernelParams& params [[buffer(0)]],
    const device float* means3d [[buffer(1)]],
    const device float* scales [[buffer(2)]],
    const device float* quats [[buffer(3)]],
    const device float* viewmat [[buffer(4)]],
    const device float* projmat [[buffer(5)]],
    const device float* dL_dcov3d [[buffer(6)]],
    const device float* dL_dxys [[buffer(7)]],
    const device float* dL_ddepths [[buffer(8)]],
    const device float* dL_dconic_opacity [[buffer(9)]],
    const device float* dL_dviewspace_out [[buffer(10)]],
    device float* dL_dmeans3d [[buffer(11)]],
    device float* dL_dopacities [[buffer(12)]],
    device float* dL_dscales [[buffer(13)]],
    device float* dL_dquats [[buffer(14)]],
    device float* dL_dviewspace_in [[buffer(15)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= params.n) {
    return;
  }

  const float3 p = read_packed_float3(means3d, tid);
  const float4 ph = mul_mat4_vec3_h(projmat, p);

  const float gx = dL_dxys[2 * tid + 0];
  const float gy = dL_dxys[2 * tid + 1];
  const float gz = dL_ddepths[tid];

  // Approximate d(ndc)/d(pixel): x_pix = ((ndc_x + 1) * W - 1)/2
  // so dL/dndc_x = dL/dx_pix * W/2, similarly for y.
  const float dL_dndc_x = gx * (0.5f * float(params.image_width));
  const float dL_dndc_y = gy * (0.5f * float(params.image_height));

  const float w = (fabs(ph.w) < 1e-6f) ? 1e-6f : ph.w;
  const float invw = 1.0f / w;

  float dphx = dL_dndc_x * invw;
  float dphy = dL_dndc_y * invw;
  float dphw = -(dL_dndc_x * ph.x + dL_dndc_y * ph.y) * (invw * invw);

  // Backprop through projection homogeneous transform + depth term via view matrix z row.
  const float dmx =
      projmat[0] * dphx + projmat[1] * dphy + projmat[3] * dphw + viewmat[2] * gz;
  const float dmy =
      projmat[4] * dphx + projmat[5] * dphy + projmat[7] * dphw + viewmat[6] * gz;
  const float dmz =
      projmat[8] * dphx + projmat[9] * dphy + projmat[11] * dphw + viewmat[10] * gz;

  dL_dmeans3d[3 * tid + 0] = dmx;
  dL_dmeans3d[3 * tid + 1] = dmy;
  dL_dmeans3d[3 * tid + 2] = dmz;
  dL_dopacities[tid] = dL_dconic_opacity[4 * tid + 3];

  // Backprop cov3d( scale, quat ) -> d_scales, d_quats
  if (params.use_cov3d_precomp == 0u) {
    const float3 scale = read_packed_float3(scales, tid, 3u);
    const float4 q = read_packed_float4(quats, tid, 4u);
    const float r = q.x;
    const float x = q.y;
    const float y = q.z;
    const float z = q.w;

    float3x3 R = float3x3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));

    float3 s = scale;
    float3x3 S = float3x3(1.0f);
    S[0][0] = s.x;
    S[1][1] = s.y;
    S[2][2] = s.z;

    float3x3 M = S * R;

    const uint cbase = 6u * tid;
    const float c0 = dL_dcov3d[cbase + 0];
    const float c1 = dL_dcov3d[cbase + 1];
    const float c2 = dL_dcov3d[cbase + 2];
    const float c3 = dL_dcov3d[cbase + 3];
    const float c4 = dL_dcov3d[cbase + 4];
    const float c5 = dL_dcov3d[cbase + 5];

    float3x3 dSigma = float3x3(
        c0, 0.5f * c1, 0.5f * c2,
        0.5f * c1, c3, 0.5f * c4,
        0.5f * c2, 0.5f * c4, c5);

    float3x3 dM = 2.0f * M * dSigma;
    float3x3 Rt = transpose(R);
    float3x3 dMt = transpose(dM);

    float dsx = dot(Rt[0], dMt[0]);
    float dsy = dot(Rt[1], dMt[1]);
    float dsz = dot(Rt[2], dMt[2]);
    dL_dscales[3 * tid + 0] = dsx;
    dL_dscales[3 * tid + 1] = dsy;
    dL_dscales[3 * tid + 2] = dsz;

    dMt[0] *= s.x;
    dMt[1] *= s.y;
    dMt[2] *= s.z;

    float4 d_q;
    d_q.x = 2 * z * (dMt[0][1] - dMt[1][0]) + 2 * y * (dMt[2][0] - dMt[0][2]) + 2 * x * (dMt[1][2] - dMt[2][1]);
    d_q.y = 2 * y * (dMt[1][0] + dMt[0][1]) + 2 * z * (dMt[2][0] + dMt[0][2]) + 2 * r * (dMt[1][2] - dMt[2][1]) - 4 * x * (dMt[2][2] + dMt[1][1]);
    d_q.z = 2 * x * (dMt[1][0] + dMt[0][1]) + 2 * r * (dMt[2][0] - dMt[0][2]) + 2 * z * (dMt[1][2] + dMt[2][1]) - 4 * y * (dMt[2][2] + dMt[0][0]);
    d_q.w = 2 * r * (dMt[0][1] - dMt[1][0]) + 2 * x * (dMt[2][0] + dMt[0][2]) + 2 * y * (dMt[1][2] + dMt[2][1]) - 4 * z * (dMt[1][1] + dMt[0][0]);

    dL_dquats[4 * tid + 0] = d_q.x;
    dL_dquats[4 * tid + 1] = d_q.y;
    dL_dquats[4 * tid + 2] = d_q.z;
    dL_dquats[4 * tid + 3] = d_q.w;
  }

  // preprocess forward passes viewspace_points through; backward is identity.
  dL_dviewspace_in[4 * tid + 0] = dL_dviewspace_out[4 * tid + 0];
  dL_dviewspace_in[4 * tid + 1] = dL_dviewspace_out[4 * tid + 1];
  dL_dviewspace_in[4 * tid + 2] = dL_dviewspace_out[4 * tid + 2];
  dL_dviewspace_in[4 * tid + 3] = dL_dviewspace_out[4 * tid + 3];
}
