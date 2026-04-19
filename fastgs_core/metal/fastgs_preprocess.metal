#include <metal_stdlib>

using namespace metal;

#define BLOCK_X 16
#define BLOCK_Y 16

constant float SH_C0 = 0.28209479177387814f;
constant float SH_C1 = 0.4886025119029199f;
constant float SH_C2[] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f,
};
constant float SH_C3[] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f,
};

struct PreprocessKernelParams {
  int degree;
  int max_sh_coeffs;
  float scale_modifier;
  float mult;
  float tan_fovx;
  float tan_fovy;
  float focal_x;
  float focal_y;
  uint image_width;
  uint image_height;
  uint tile_bounds_x;
  uint tile_bounds_y;
  uint tile_bounds_z;
  uint prefiltered;
  uint use_cov3d_precomp;
  uint use_colors_precomp;
};

inline float ndc2pix(float v, int s) {
  return ((v + 1.0f) * s - 1.0f) * 0.5f;
}

inline float3 read_packed_float3(const device float* arr, uint idx) {
  return float3(arr[3 * idx], arr[3 * idx + 1], arr[3 * idx + 2]);
}

inline float4 read_packed_float4(const device float* arr, uint idx) {
  return float4(arr[4 * idx], arr[4 * idx + 1], arr[4 * idx + 2], arr[4 * idx + 3]);
}

inline float3 read_sh_coeff(const device float* shs, uint idx, int max_coeffs, uint coeff_idx) {
  uint off = idx * uint(max_coeffs) * 3u + coeff_idx * 3u;
  return float3(shs[off], shs[off + 1], shs[off + 2]);
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

inline float3 transform_point_4x3(const float3 p, const device float* matrix) {
  return float3(
      matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
      matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
      matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14]);
}

inline float4 transform_point_4x4(const float3 p, const device float* matrix) {
  return float4(
      matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
      matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
      matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
      matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]);
}

inline bool in_frustum(uint idx,
                       const device float* orig_points,
                       const device float* viewmatrix,
                       const device float* projmatrix,
                       bool prefiltered,
                       thread float3& p_view) {
  float3 p_orig = read_packed_float3(orig_points, idx);
  float4 p_hom = transform_point_4x4(p_orig, projmatrix);
  float p_w = 1.0f / (p_hom.w + 1.0e-7f);
  float3 p_proj = float3(p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w);
  p_view = transform_point_4x3(p_orig, viewmatrix);
  if (p_view.z <= 0.2f) {
    if (prefiltered) {
      return false;
    }
    return false;
  }
  (void)p_proj;
  return true;
}

inline float3 compute_color_from_sh(uint idx,
                                    int deg,
                                    int max_coeffs,
                                    const device float* means,
                                    float3 campos,
                                    const device float* dc,
                                    const device float* shs,
                                    device bool* clamped) {
  float3 pos = read_packed_float3(means, idx);
  float3 dir = normalize(pos - campos);

  uint base = 3 * idx;
  float3 result = SH_C0 * float3(dc[base], dc[base + 1], dc[base + 2]);

  if (deg > 0) {
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;
    result = result - SH_C1 * y * read_sh_coeff(shs, idx, max_coeffs, 0u) +
             SH_C1 * z * read_sh_coeff(shs, idx, max_coeffs, 1u) -
             SH_C1 * x * read_sh_coeff(shs, idx, max_coeffs, 2u);

    if (deg > 1) {
      float xx = x * x;
      float yy = y * y;
      float zz = z * z;
      float xy = x * y;
      float yz = y * z;
      float xz = x * z;
      result += SH_C2[0] * xy * read_sh_coeff(shs, idx, max_coeffs, 3u) +
                SH_C2[1] * yz * read_sh_coeff(shs, idx, max_coeffs, 4u) +
                SH_C2[2] * (2.0f * zz - xx - yy) * read_sh_coeff(shs, idx, max_coeffs, 5u) +
                SH_C2[3] * xz * read_sh_coeff(shs, idx, max_coeffs, 6u) +
                SH_C2[4] * (xx - yy) * read_sh_coeff(shs, idx, max_coeffs, 7u);

      if (deg > 2) {
        result += SH_C3[0] * y * (3.0f * xx - yy) * read_sh_coeff(shs, idx, max_coeffs, 8u) +
                  SH_C3[1] * xy * z * read_sh_coeff(shs, idx, max_coeffs, 9u) +
                  SH_C3[2] * y * (4.0f * zz - xx - yy) * read_sh_coeff(shs, idx, max_coeffs, 10u) +
                  SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * read_sh_coeff(shs, idx, max_coeffs, 11u) +
                  SH_C3[4] * x * (4.0f * zz - xx - yy) * read_sh_coeff(shs, idx, max_coeffs, 12u) +
                  SH_C3[5] * z * (xx - yy) * read_sh_coeff(shs, idx, max_coeffs, 13u) +
                  SH_C3[6] * x * (xx - 3.0f * yy) * read_sh_coeff(shs, idx, max_coeffs, 14u);
      }
    }
  }

  result += 0.5f;
  clamped[3 * idx + 0] = result.x < 0.0f;
  clamped[3 * idx + 1] = result.y < 0.0f;
  clamped[3 * idx + 2] = result.z < 0.0f;
  return max(result, float3(0.0f));
}

inline float3 compute_cov2d(const float3 mean,
                            float focal_x,
                            float focal_y,
                            float tan_fovx,
                            float tan_fovy,
                            const thread float* cov3d,
                            const device float* viewmatrix) {
  float3 t = transform_point_4x3(mean, viewmatrix);
  float limx = 1.3f * tan_fovx;
  float limy = 1.3f * tan_fovy;
  float txtz = t.x / t.z;
  float tytz = t.y / t.z;
  t.x = min(limx, max(-limx, txtz)) * t.z;
  t.y = min(limy, max(-limy, tytz)) * t.z;

  float3x3 j = float3x3(
      focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
      0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
      0.0f, 0.0f, 0.0f);

  float3x3 w = float3x3(
      viewmatrix[0], viewmatrix[1], viewmatrix[2],
      viewmatrix[4], viewmatrix[5], viewmatrix[6],
      viewmatrix[8], viewmatrix[9], viewmatrix[10]);

  float3x3 t_mat = w * j;
  float3x3 vrk = float3x3(
      cov3d[0], cov3d[1], cov3d[2],
      cov3d[1], cov3d[3], cov3d[4],
      cov3d[2], cov3d[4], cov3d[5]);
  float3x3 cov = transpose(t_mat) * transpose(vrk) * t_mat;
  cov[0][0] += 0.3f;
  cov[1][1] += 0.3f;
  return float3(cov[0][0], cov[0][1], cov[1][1]);
}

inline void compute_cov3d(float3 scale, float mod, float4 rot, device float* cov3d) {
  float3x3 s = float3x3(1.0f);
  s[0][0] = mod * scale.x;
  s[1][1] = mod * scale.y;
  s[2][2] = mod * scale.z;

  float r = rot.x;
  float x = rot.y;
  float y = rot.z;
  float z = rot.w;

  float3x3 rmat = float3x3(
      1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
      2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
      2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));

  float3x3 m = s * rmat;
  float3x3 sigma = transpose(m) * m;
  cov3d[0] = sigma[0][0];
  cov3d[1] = sigma[0][1];
  cov3d[2] = sigma[0][2];
  cov3d[3] = sigma[1][1];
  cov3d[4] = sigma[1][2];
  cov3d[5] = sigma[2][2];
}

inline void get_rect(float2 p, int max_radius, thread uint2& rect_min,
                     thread uint2& rect_max, uint3 grid) {
  rect_min = uint2(
      min(grid.x, (uint)max(0, (int)((p.x - max_radius) / BLOCK_X))),
      min(grid.y, (uint)max(0, (int)((p.y - max_radius) / BLOCK_Y))));
  rect_max = uint2(
      min(grid.x, (uint)max(0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
      min(grid.y, (uint)max(0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y))));
}

inline float evaluate_opacity_factor(float dx, float dy, float4 co) {
  return 0.5f * (co.x * dx * dx + co.z * dy * dy) + co.y * dx * dy;
}

inline float2 compute_ellipse_intersection(float4 con_o,
                                           float disc,
                                           float t,
                                           float2 p,
                                           bool is_y,
                                           float coord) {
  float p_u = is_y ? p.y : p.x;
  float p_v = is_y ? p.x : p.y;
  float coeff = is_y ? con_o.x : con_o.z;
  float h = coord - p_u;
  float sqrt_term = sqrt(disc * h * h + t * coeff);
  return float2(
      (-con_o.y * h - sqrt_term) / coeff + p_v,
      (-con_o.y * h + sqrt_term) / coeff + p_v);
}

inline uint process_tiles(float4 con_o,
                          float disc,
                          float t,
                          float2 p,
                          float2 bbox_min,
                          float2 bbox_max,
                          float2 bbox_argmin,
                          float2 bbox_argmax,
                          int2 rect_min,
                          int2 rect_max,
                          uint3 grid,
                          bool is_y) {
  float block_u = is_y ? BLOCK_Y : BLOCK_X;
  float block_v = is_y ? BLOCK_X : BLOCK_Y;

  if (is_y) {
    rect_min = int2(rect_min.y, rect_min.x);
    rect_max = int2(rect_max.y, rect_max.x);
    bbox_min = float2(bbox_min.y, bbox_min.x);
    bbox_max = float2(bbox_max.y, bbox_max.x);
    bbox_argmin = float2(bbox_argmin.y, bbox_argmin.x);
    bbox_argmax = float2(bbox_argmax.y, bbox_argmax.x);
  }

  uint tiles_count = 0;
  float2 intersect_min_line;
  float2 intersect_max_line = float2(bbox_max.y, bbox_min.y);
  float ellipse_min;
  float ellipse_max;
  float min_line = rect_min.x * block_u;

  if (bbox_min.x <= min_line) {
    intersect_min_line =
        compute_ellipse_intersection(con_o, disc, t, p, is_y, rect_min.x * block_u);
  } else {
    intersect_min_line = intersect_max_line;
  }

  for (int u = rect_min.x; u < rect_max.x; ++u) {
    float max_line = min_line + block_u;
    if (max_line <= bbox_max.x) {
      intersect_max_line =
          compute_ellipse_intersection(con_o, disc, t, p, is_y, max_line);
    }

    if (min_line <= bbox_argmin.y && bbox_argmin.y < max_line) {
      ellipse_min = bbox_min.y;
    } else {
      ellipse_min = min(intersect_min_line.x, intersect_max_line.x);
    }

    if (min_line <= bbox_argmax.y && bbox_argmax.y < max_line) {
      ellipse_max = bbox_max.y;
    } else {
      ellipse_max = max(intersect_min_line.y, intersect_max_line.y);
    }

    int min_tile_v = max(rect_min.y, min(rect_max.y, (int)(ellipse_min / block_v)));
    int max_tile_v =
        min(rect_max.y, max(rect_min.y, (int)(ellipse_max / block_v + 1)));
    tiles_count += uint(max_tile_v - min_tile_v);

    intersect_min_line = intersect_max_line;
    min_line = max_line;
  }
  return tiles_count;
}

inline uint duplicate_to_tiles_touched(float2 p, float4 con_o, uint3 grid, float mult) {
  float disc = con_o.y * con_o.y - con_o.x * con_o.z;
  if (con_o.x <= 0.0f || con_o.z <= 0.0f || disc >= 0.0f) {
    return 0u;
  }

  float t = 2.0f * log(con_o.w * 255.0f);
  t = mult * t;

  float x_term = sqrt(-(con_o.y * con_o.y * t) / (disc * con_o.x));
  x_term = (con_o.y < 0.0f) ? x_term : -x_term;
  float y_term = sqrt(-(con_o.y * con_o.y * t) / (disc * con_o.z));
  y_term = (con_o.y < 0.0f) ? y_term : -y_term;

  float2 bbox_argmin = float2(p.y - y_term, p.x - x_term);
  float2 bbox_argmax = float2(p.y + y_term, p.x + x_term);

  float2 bbox_min = float2(
      compute_ellipse_intersection(con_o, disc, t, p, true, bbox_argmin.x).x,
      compute_ellipse_intersection(con_o, disc, t, p, false, bbox_argmin.y).x);
  float2 bbox_max = float2(
      compute_ellipse_intersection(con_o, disc, t, p, true, bbox_argmax.x).y,
      compute_ellipse_intersection(con_o, disc, t, p, false, bbox_argmax.y).y);

  int2 rect_min = int2(
      max(0, min((int)grid.x, (int)(bbox_min.x / BLOCK_X))),
      max(0, min((int)grid.y, (int)(bbox_min.y / BLOCK_Y))));
  int2 rect_max = int2(
      max(0, min((int)grid.x, (int)(bbox_max.x / BLOCK_X + 1))),
      max(0, min((int)grid.y, (int)(bbox_max.y / BLOCK_Y + 1))));

  int y_span = rect_max.y - rect_min.y;
  int x_span = rect_max.x - rect_min.x;
  if (y_span * x_span == 0) {
    return 0u;
  }

  bool is_y = y_span < x_span;
  return process_tiles(
      con_o,
      disc,
      t,
      p,
      bbox_min,
      bbox_max,
      bbox_argmin,
      bbox_argmax,
      rect_min,
      rect_max,
      grid,
      is_y);
}

kernel void fastgs_preprocess_forward_kernel(
    constant int& n [[buffer(0)]],
    constant PreprocessKernelParams& p [[buffer(1)]],
    device const float* means3d [[buffer(2)]],
    device const float* dc [[buffer(3)]],
    device const float* shs [[buffer(4)]],
    device const float* colors_precomp [[buffer(5)]],
    device const float* opacities [[buffer(6)]],
    device const float* scales [[buffer(7)]],
    device const float* rotations [[buffer(8)]],
    device const float* cov3d_precomp [[buffer(9)]],
    device const float* viewmatrix [[buffer(10)]],
    device const float* projmatrix [[buffer(11)]],
    device const float* cam_pos [[buffer(12)]],
    device const float* viewspace_points_in [[buffer(13)]],
    device int* radii [[buffer(14)]],
    device float* points_xy_image [[buffer(15)]],
    device float* depths [[buffer(16)]],
    device float* cov3ds [[buffer(17)]],
    device float* rgb [[buffer(18)]],
    device float* conic_opacity [[buffer(19)]],
    device uint* tiles_touched [[buffer(20)]],
    device bool* clamped [[buffer(21)]],
    device float* viewspace_points_out [[buffer(22)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= static_cast<uint>(n)) {
    return;
  }

  radii[tid] = 0;
  tiles_touched[tid] = 0u;

  float3 p_view;
  if (!in_frustum(tid, means3d, viewmatrix, projmatrix, bool(p.prefiltered), p_view)) {
    return;
  }

  float3 p_orig = read_packed_float3(means3d, tid);
  float4 p_hom = transform_point_4x4(p_orig, projmatrix);
  float p_w = 1.0f / (p_hom.w + 1.0e-7f);
  float3 p_proj = float3(p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w);

  thread float local_cov3d[6];
  if (p.use_cov3d_precomp != 0u) {
    for (uint i = 0; i < 6; ++i) {
      local_cov3d[i] = cov3d_precomp[tid * 6 + i];
      cov3ds[tid * 6 + i] = local_cov3d[i];
    }
  } else {
    compute_cov3d(read_packed_float3(scales, tid), p.scale_modifier, read_packed_float4(rotations, tid),
                  cov3ds + tid * 6);
    for (uint i = 0; i < 6; ++i) {
      local_cov3d[i] = cov3ds[tid * 6 + i];
    }
  }

  float3 cov = compute_cov2d(p_orig, p.focal_x, p.focal_y, p.tan_fovx, p.tan_fovy, local_cov3d, viewmatrix);

  float det = cov.x * cov.z - cov.y * cov.y;
  if (det == 0.0f) {
    return;
  }
  float det_inv = 1.f / det;
  float3 conic = float3(cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv);

  float mid = 0.5f * (cov.x + cov.z);
  float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
  float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
  float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
  float2 point_image = float2(ndc2pix(p_proj.x, int(p.image_width)),
                              ndc2pix(p_proj.y, int(p.image_height)));

  float4 con_o = float4(conic.x, conic.y, conic.z, opacities[tid]);
  uint3 grid = uint3(p.tile_bounds_x, p.tile_bounds_y, p.tile_bounds_z);
  uint tiles_count = duplicate_to_tiles_touched(point_image, con_o, grid, p.mult);
  if (tiles_count == 0u) {
    return;
  }

  if (p.use_colors_precomp != 0u) {
    write_packed_float3(rgb, tid, read_packed_float3(colors_precomp, tid));
    clamped[3 * tid + 0] = false;
    clamped[3 * tid + 1] = false;
    clamped[3 * tid + 2] = false;
  } else {
    float3 result = compute_color_from_sh(
        tid, p.degree, p.max_sh_coeffs, means3d, read_packed_float3(cam_pos, 0), dc, shs, clamped);
    write_packed_float3(rgb, tid, result);
  }

  depths[tid] = p_view.z;
  radii[tid] = int(my_radius);
  write_packed_float2(points_xy_image, tid, point_image);
  write_packed_float4(conic_opacity, tid, con_o);
  tiles_touched[tid] = tiles_count;

  write_packed_float4(
      viewspace_points_out,
      tid,
      float4(p_view.x, p_view.y, p_view.z, viewspace_points_in[4 * tid + 3]));
}
