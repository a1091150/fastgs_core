#include <metal_stdlib>

using namespace metal;

struct PreprocessBackwardKernelParams {
  uint n;
  uint image_width;
  uint image_height;
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

kernel void fastgs_preprocess_backward_kernel(
    constant PreprocessBackwardKernelParams& params [[buffer(0)]],
    const device float* means3d [[buffer(1)]],
    const device float* viewmat [[buffer(2)]],
    const device float* projmat [[buffer(3)]],
    const device float* dL_dxys [[buffer(4)]],
    const device float* dL_ddepths [[buffer(5)]],
    const device float* dL_dviewspace_out [[buffer(6)]],
    device float* dL_dmeans3d [[buffer(7)]],
    device float* dL_dviewspace_in [[buffer(8)]],
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

  // preprocess forward passes viewspace_points through; backward is identity.
  dL_dviewspace_in[4 * tid + 0] = dL_dviewspace_out[4 * tid + 0];
  dL_dviewspace_in[4 * tid + 1] = dL_dviewspace_out[4 * tid + 1];
  dL_dviewspace_in[4 * tid + 2] = dL_dviewspace_out[4 * tid + 2];
  dL_dviewspace_in[4 * tid + 3] = dL_dviewspace_out[4 * tid + 3];
}
