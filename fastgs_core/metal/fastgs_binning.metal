#include <metal_stdlib>

using namespace metal;

#define BLOCK_X 16
#define BLOCK_Y 16

struct BinningKernelParams {
  float mult;
  uint tile_bounds_x;
  uint tile_bounds_y;
  uint tile_bounds_z;
};

inline float2 read_packed_float2(const device float* arr, uint idx) {
  return float2(arr[2 * idx], arr[2 * idx + 1]);
}

inline float4 read_packed_float4(const device float* arr, uint idx) {
  return float4(arr[4 * idx], arr[4 * idx + 1], arr[4 * idx + 2], arr[4 * idx + 3]);
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

inline void process_tiles(float4 con_o,
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
                          bool is_y,
                          uint idx,
                          uint off,
                          float depth,
                          device ulong* gaussian_keys_unsorted,
                          device uint* gaussian_values_unsorted) {
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

    int min_tile_v = max(rect_min.y, min(rect_max.y, int(ellipse_min / block_v)));
    int max_tile_v = min(rect_max.y, max(rect_min.y, int(ellipse_max / block_v + 1)));

    for (int v = min_tile_v; v < max_tile_v; ++v) {
      ulong key = is_y ? ulong(u * int(grid.x) + v) : ulong(v * int(grid.x) + u);
      key <<= 32;
      key |= ulong(as_type<uint>(depth));
      gaussian_keys_unsorted[off] = key;
      gaussian_values_unsorted[off] = idx;
      off++;
    }

    intersect_min_line = intersect_max_line;
    min_line = max_line;
  }
}

inline void duplicate_with_keys(float2 p,
                                float4 con_o,
                                uint3 grid,
                                float mult,
                                uint idx,
                                uint off,
                                float depth,
                                device ulong* gaussian_keys_unsorted,
                                device uint* gaussian_values_unsorted) {
  float disc = con_o.y * con_o.y - con_o.x * con_o.z;
  if (con_o.x <= 0.0f || con_o.z <= 0.0f || disc >= 0.0f) {
    return;
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
      max(0, min(int(grid.x), int(bbox_min.x / BLOCK_X))),
      max(0, min(int(grid.y), int(bbox_min.y / BLOCK_Y))));
  int2 rect_max = int2(
      max(0, min(int(grid.x), int(bbox_max.x / BLOCK_X + 1))),
      max(0, min(int(grid.y), int(bbox_max.y / BLOCK_Y + 1))));

  int y_span = rect_max.y - rect_min.y;
  int x_span = rect_max.x - rect_min.x;
  if (y_span * x_span == 0) {
    return;
  }

  bool is_y = y_span < x_span;
  process_tiles(
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
      is_y,
      idx,
      off,
      depth,
      gaussian_keys_unsorted,
      gaussian_values_unsorted);
}

kernel void fastgs_duplicate_with_keys_kernel(
    constant int& p [[buffer(0)]],
    constant BinningKernelParams& params [[buffer(1)]],
    device float* points_xy [[buffer(2)]],
    device float* depths [[buffer(3)]],
    device uint* point_offsets [[buffer(4)]],
    device float* conic_opacity [[buffer(5)]],
    device uint* tiles_touched [[buffer(6)]],
    device ulong* gaussian_keys_unsorted [[buffer(7)]],
    device uint* gaussian_values_unsorted [[buffer(8)]],
    uint3 gp [[thread_position_in_grid]]) {
  uint idx = gp.x;
  if (idx >= uint(p)) {
    return;
  }

  if (tiles_touched[idx] == 0u) {
    return;
  }

  uint off = (idx == 0u) ? 0u : point_offsets[idx - 1];
  duplicate_with_keys(
      read_packed_float2(points_xy, idx),
      read_packed_float4(conic_opacity, idx),
      uint3(params.tile_bounds_x, params.tile_bounds_y, params.tile_bounds_z),
      params.mult,
      idx,
      off,
      depths[idx],
      gaussian_keys_unsorted,
      gaussian_values_unsorted);
}
