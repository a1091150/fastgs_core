#include <metal_stdlib>

using namespace metal;

struct PreprocessKernelParams {
  int degree;
  int max_sh_coeffs;
  float scale_modifier;
  float mult;
  float tan_fovx;
  float tan_fovy;
  uint image_width;
  uint image_height;
  uint tile_bounds_x;
  uint tile_bounds_y;
  uint tile_bounds_z;
  uint prefiltered;
  uint use_cov3d_precomp;
  uint use_colors_precomp;
};

inline float3 read_packed_float3(const device float* arr, uint idx) {
  return float3(arr[3 * idx], arr[3 * idx + 1], arr[3 * idx + 2]);
}

inline void write_packed_float2(device float* arr, uint idx, float2 val) {
  arr[2 * idx] = val.x;
  arr[2 * idx + 1] = val.y;
}

inline void write_packed_float3(device float* arr, uint idx, float3 val) {
  arr[3 * idx] = val.x;
  arr[3 * idx + 1] = val.y;
  arr[3 * idx + 2] = val.z;
}

inline void write_packed_float4(device float* arr, uint idx, float4 val) {
  arr[4 * idx] = val.x;
  arr[4 * idx + 1] = val.y;
  arr[4 * idx + 2] = val.z;
  arr[4 * idx + 3] = val.w;
}

kernel void fastgs_preprocess_forward_kernel(
    constant int& n [[buffer(0)]],
    constant PreprocessKernelParams& p [[buffer(1)]],
    device const float* means3d [[buffer(2)]],
    device const float* dc [[buffer(3)]],
    device const float* sh [[buffer(4)]],
    device const float* colors_precomp [[buffer(5)]],
    device const float* opacities [[buffer(6)]],
    device const float* scales [[buffer(7)]],
    device const float* quats [[buffer(8)]],
    device const float* cov3d_precomp [[buffer(9)]],
    device const float* viewmat [[buffer(10)]],
    device const float* projmat [[buffer(11)]],
    device const float* cam_pos [[buffer(12)]],
    device const float* viewspace_points_in [[buffer(13)]],
    device int* radii [[buffer(14)]],
    device float* xys [[buffer(15)]],
    device float* depths [[buffer(16)]],
    device float* cov3d [[buffer(17)]],
    device float* rgbs [[buffer(18)]],
    device float* conic_opacity [[buffer(19)]],
    device uint* tiles_touched [[buffer(20)]],
    device bool* clamped [[buffer(21)]],
    device float* viewspace_points_out [[buffer(22)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= static_cast<uint>(n)) {
    return;
  }

  float3 m = read_packed_float3(means3d, tid);
  radii[tid] = 1;
  write_packed_float2(xys, tid, float2(m.x, m.y));
  depths[tid] = m.z;

  write_packed_float3(cov3d, tid * 2u, float3(1.0f, 0.0f, 0.0f));
  cov3d[6 * tid + 3] = 1.0f;
  cov3d[6 * tid + 4] = 0.0f;
  cov3d[6 * tid + 5] = 1.0f;

  float3 rgb = float3(0.0f);
  if (p.use_colors_precomp != 0u) {
    rgb = read_packed_float3(colors_precomp, tid);
  } else if (dc != nullptr) {
    rgb = read_packed_float3(dc, tid);
  }
  write_packed_float3(rgbs, tid, max(rgb, float3(0.0f)));

  conic_opacity[4 * tid + 0] = 1.0f;
  conic_opacity[4 * tid + 1] = 0.0f;
  conic_opacity[4 * tid + 2] = 1.0f;
  conic_opacity[4 * tid + 3] = opacities[tid];

  tiles_touched[tid] = 1u;
  clamped[3 * tid + 0] = false;
  clamped[3 * tid + 1] = false;
  clamped[3 * tid + 2] = false;

  write_packed_float4(viewspace_points_out, tid,
                      float4(viewspace_points_in[4 * tid + 0],
                             viewspace_points_in[4 * tid + 1],
                             viewspace_points_in[4 * tid + 2],
                             viewspace_points_in[4 * tid + 3]));

  (void)sh;
  (void)scales;
  (void)quats;
  (void)cov3d_precomp;
  (void)viewmat;
  (void)projmat;
  (void)cam_pos;
}
