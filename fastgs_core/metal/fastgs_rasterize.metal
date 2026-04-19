#include <metal_stdlib>

using namespace metal;

struct RasterizeKernelParams {
  uint image_width;
  uint image_height;
  uint block_x;
  uint block_y;
  uint num_channels;
  uint num_tiles;
  uint bucket_sum;
  uint get_flag;
};

inline float2 read_packed_float2(const device float* arr, uint idx) {
  return float2(arr[2 * idx], arr[2 * idx + 1]);
}

inline float4 read_packed_float4(const device float* arr, uint idx) {
  return float4(arr[4 * idx], arr[4 * idx + 1], arr[4 * idx + 2], arr[4 * idx + 3]);
}

kernel void fastgs_render_forward_kernel(
    constant RasterizeKernelParams& params [[buffer(0)]],
    device uint* ranges [[buffer(1)]],
    device uint* point_list [[buffer(2)]],
    device uint* per_tile_bucket_offset [[buffer(3)]],
    device float* means2d [[buffer(4)]],
    device float* colors [[buffer(5)]],
    device float* conic_opacity [[buffer(6)]],
    device float* background [[buffer(7)]],
    device int* radii [[buffer(8)]],
    device int* metric_map [[buffer(9)]],
    device uint* bucket_to_tile [[buffer(10)]],
    device float* sampled_t [[buffer(11)]],
    device float* sampled_ar [[buffer(12)]],
    device float* final_t [[buffer(13)]],
    device uint* n_contrib [[buffer(14)]],
    device uint* max_contrib [[buffer(15)]],
    device float* pixel_colors [[buffer(16)]],
    device float* out_color [[buffer(17)]],
    device int* metric_count [[buffer(18)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tgp [[threadgroup_position_in_grid]]) {
  uint pix_x = gid.x;
  uint pix_y = gid.y;
  uint tile_x = tgp.x;
  uint tile_y = tgp.y;

  if (tile_x >= (params.image_width + params.block_x - 1) / params.block_x ||
      tile_y >= (params.image_height + params.block_y - 1) / params.block_y) {
    return;
  }

  uint tile_id = tile_y * ((params.image_width + params.block_x - 1) / params.block_x) +
                 tile_x;
  uint2 range = uint2(ranges[2 * tile_id], ranges[2 * tile_id + 1]);
  int to_do = int(range.y - range.x);
  int num_buckets = (to_do + 31) / 32;
  uint bbm = (tile_id == 0) ? 0u : per_tile_bucket_offset[tile_id - 1];

  uint local_rank = tid.y * params.block_x + tid.x;
  for (int i = 0;
       i < (num_buckets + int(params.block_x * params.block_y) - 1) /
               int(params.block_x * params.block_y);
       ++i) {
    int bucket_idx = i * int(params.block_x * params.block_y) + int(local_rank);
    if (bucket_idx < num_buckets) {
      bucket_to_tile[bbm + uint(bucket_idx)] = tile_id;
    }
  }

  if (pix_x >= params.image_width || pix_y >= params.image_height) {
    return;
  }

  uint pix_id = pix_y * params.image_width + pix_x;
  float2 pixf = float2(float(pix_x), float(pix_y));
  float t_val = 1.0f;
  uint contributor = 0u;
  uint last_contributor = 0u;

  constexpr uint kMaxChannels = 3;
  float c_accum[kMaxChannels] = {0.0f, 0.0f, 0.0f};

  for (uint idx = range.x; idx < range.y; ++idx) {
    if (((idx - range.x) % 32u) == 0u) {
      sampled_t[bbm * (params.block_x * params.block_y) + local_rank] = t_val;
      for (uint ch = 0; ch < min(params.num_channels, kMaxChannels); ++ch) {
        sampled_ar[(bbm * params.num_channels * (params.block_x * params.block_y)) +
                   ch * (params.block_x * params.block_y) + local_rank] = c_accum[ch];
      }
      bbm++;
    }

    contributor++;
    uint coll_id = point_list[idx];
    float2 xy = read_packed_float2(means2d, coll_id);
    float2 d = xy - pixf;
    float4 con_o = read_packed_float4(conic_opacity, coll_id);
    float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) -
                  con_o.y * d.x * d.y;
    if (power > 0.0f) {
      continue;
    }

    float alpha = min(0.99f, con_o.w * exp(power));
    if (alpha < 1.0f / 255.0f) {
      continue;
    }

    float test_t = t_val * (1.0f - alpha);
    if (test_t < 0.0001f) {
      break;
    }

    for (uint ch = 0; ch < min(params.num_channels, kMaxChannels); ++ch) {
      c_accum[ch] += colors[coll_id * params.num_channels + ch] * alpha * t_val;
    }

    if (params.get_flag != 0u && metric_map[pix_id] == 1) {
      atomic_fetch_add_explicit(
          (device atomic_int*)&metric_count[coll_id], 1, memory_order_relaxed);
    }

    t_val = test_t;
    last_contributor = contributor;
  }

  final_t[pix_id] = t_val;
  n_contrib[pix_id] = last_contributor;
  max_contrib[tile_id] = max(max_contrib[tile_id], last_contributor);
  for (uint ch = 0; ch < min(params.num_channels, kMaxChannels); ++ch) {
    pixel_colors[ch * params.image_height * params.image_width + pix_id] = c_accum[ch];
    out_color[ch * params.image_height * params.image_width + pix_id] =
        c_accum[ch] + t_val * background[ch];
  }

  (void)radii;
}
