#include <metal_stdlib>

using namespace metal;

struct RasterizeBackwardKernelParams {
  uint image_width;
  uint image_height;
  uint block_x;
  uint block_y;
  uint num_channels;
  uint num_tiles;
  uint bucket_sum;
  uint block_size;
};

inline float2 read_packed_float2(const device float* arr, uint idx) {
  return float2(arr[2 * idx], arr[2 * idx + 1]);
}

inline float4 read_packed_float4(const device float* arr, uint idx) {
  return float4(arr[4 * idx], arr[4 * idx + 1], arr[4 * idx + 2], arr[4 * idx + 3]);
}

kernel void fastgs_render_backward_kernel(
    constant RasterizeBackwardKernelParams& params [[buffer(0)]],
    const device uint* ranges [[buffer(1)]],
    const device uint* point_list [[buffer(2)]],
    const device uint* per_tile_bucket_offset [[buffer(3)]],
    const device float* means2d [[buffer(4)]],
    const device float* colors [[buffer(5)]],
    const device float* conic_opacity [[buffer(6)]],
    const device float* background [[buffer(7)]],
    const device float* dL_dout_color [[buffer(8)]],
    const device uint* bucket_to_tile [[buffer(9)]],
    const device float* sampled_t [[buffer(10)]],
    const device float* sampled_ar [[buffer(11)]],
    const device float* final_t [[buffer(12)]],
    const device uint* n_contrib [[buffer(13)]],
    const device uint* max_contrib [[buffer(14)]],
    const device float* pixel_colors [[buffer(15)]],
    device atomic_float* dL_dmeans2d [[buffer(16)]],
    device atomic_float* dL_dcolors [[buffer(17)]],
    device atomic_float* dL_dconic_opacity [[buffer(18)]],
    device atomic_float* dL_dviewspace_points [[buffer(19)]],
    uint lane [[thread_index_in_simdgroup]],
    uint tid_in_tg [[thread_index_in_threadgroup]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint simd_size [[threads_per_simdgroup]]) {
  const uint C = min(params.num_channels, 3u);
  const uint bucket = tgp.x;
  if (bucket >= params.bucket_sum) {
    return;
  }
  if (simd_size != 32u) {
    return;
  }

  const uint tile_id = bucket_to_tile[bucket];
  if (tile_id >= params.num_tiles) {
    return;
  }

  const uint2 range = uint2(ranges[2 * tile_id], ranges[2 * tile_id + 1]);
  const int num_splats_in_tile = int(range.y - range.x);
  const uint bbm = (tile_id == 0u) ? 0u : per_tile_bucket_offset[tile_id - 1u];
  const int bucket_idx_in_tile = int(bucket - bbm);
  const int splat_idx_in_tile = bucket_idx_in_tile * 32 + int(lane);
  const int splat_idx_global = int(range.x) + splat_idx_in_tile;
  const bool valid_splat = (splat_idx_in_tile < num_splats_in_tile);

  if (bucket_idx_in_tile * 32 >= int(max_contrib[tile_id])) {
    return;
  }

  int gaussian_idx = 0;
  float2 xy = float2(0.0f);
  float4 con_o = float4(0.0f);
  float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f;
  if (valid_splat) {
    gaussian_idx = int(point_list[splat_idx_global]);
    xy = read_packed_float2(means2d, uint(gaussian_idx));
    con_o = read_packed_float4(conic_opacity, uint(gaussian_idx));
    if (C > 0u) c0 = colors[uint(gaussian_idx) * params.num_channels + 0u];
    if (C > 1u) c1 = colors[uint(gaussian_idx) * params.num_channels + 1u];
    if (C > 2u) c2 = colors[uint(gaussian_idx) * params.num_channels + 2u];
  }

  float reg_dmean_x = 0.0f;
  float reg_dmean_y = 0.0f;
  float reg_dmean_z = 0.0f;
  float reg_dmean_w = 0.0f;
  float reg_dconic_x = 0.0f;
  float reg_dconic_y = 0.0f;
  float reg_dconic_w = 0.0f;
  float reg_dopacity = 0.0f;
  float reg_dcolor0 = 0.0f, reg_dcolor1 = 0.0f, reg_dcolor2 = 0.0f;

  const uint horizontal_blocks = (params.image_width + params.block_x - 1u) / params.block_x;
  const uint2 tile = uint2(tile_id % horizontal_blocks, tile_id / horizontal_blocks);
  const uint2 pix_min = uint2(tile.x * params.block_x, tile.y * params.block_y);
  const float ddelx_dx = 0.5f * float(params.image_width);
  const float ddely_dy = 0.5f * float(params.image_height);

  float T = 0.0f;
  float T_final = 0.0f;
  float last_contributor = 0.0f;
  float ar0 = 0.0f, ar1 = 0.0f, ar2 = 0.0f;
  float dL_dpixel0 = 0.0f, dL_dpixel1 = 0.0f, dL_dpixel2 = 0.0f;

  threadgroup float shared_sampled_ar[32 * 3 + 1];
  threadgroup float shared_pixels[32 * 3];
  const device float* sampled_ar_bucket = sampled_ar + bucket * params.block_size * params.num_channels;

  for (int i = 0; i < int(params.block_size) + 31; ++i) {
    if ((i % 32) == 0) {
      if (C > 0u) {
        int shift0 = i + int(tid_in_tg);
        shared_sampled_ar[0 * 32 + tid_in_tg] = sampled_ar_bucket[shift0];
      }
      if (C > 1u) {
        int shift1 = int(params.block_size) + i + int(tid_in_tg);
        shared_sampled_ar[1 * 32 + tid_in_tg] = sampled_ar_bucket[shift1];
      }
      if (C > 2u) {
        int shift2 = 2 * int(params.block_size) + i + int(tid_in_tg);
        shared_sampled_ar[2 * 32 + tid_in_tg] = sampled_ar_bucket[shift2];
      }

      const uint local_id = uint(i) + tid_in_tg;
      const uint2 pix = uint2(
          pix_min.x + (local_id % params.block_x),
          pix_min.y + (local_id / params.block_x));
      const bool pix_valid = pix.x < params.image_width && pix.y < params.image_height;
      const uint id = pix_valid ? (params.image_width * pix.y + pix.x) : 0u;

      if (C > 0u) {
        shared_pixels[0 * 32 + tid_in_tg] =
            pix_valid ? pixel_colors[0u * params.image_height * params.image_width + id] : 0.0f;
      }
      if (C > 1u) {
        shared_pixels[1 * 32 + tid_in_tg] =
            pix_valid ? pixel_colors[1u * params.image_height * params.image_width + id] : 0.0f;
      }
      if (C > 2u) {
        shared_pixels[2 * 32 + tid_in_tg] =
            pix_valid ? pixel_colors[2u * params.image_height * params.image_width + id] : 0.0f;
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    T = simd_shuffle_up(T, 1);
    last_contributor = simd_shuffle_up(last_contributor, 1);
    T_final = simd_shuffle_up(T_final, 1);
    ar0 = simd_shuffle_up(ar0, 1);
    ar1 = simd_shuffle_up(ar1, 1);
    ar2 = simd_shuffle_up(ar2, 1);
    dL_dpixel0 = simd_shuffle_up(dL_dpixel0, 1);
    dL_dpixel1 = simd_shuffle_up(dL_dpixel1, 1);
    dL_dpixel2 = simd_shuffle_up(dL_dpixel2, 1);

    const int idx = i - int(lane);
    const uint2 pix = uint2(
        pix_min.x + (uint(idx) % params.block_x),
        pix_min.y + (uint(idx) / params.block_x));
    const bool valid_pixel = pix.x < params.image_width && pix.y < params.image_height;
    const uint pix_id = valid_pixel ? (params.image_width * pix.y + pix.x) : 0u;
    const float2 pixf = float2(float(pix.x), float(pix.y));

    if (valid_splat && valid_pixel && lane == 0u && idx < int(params.block_size)) {
      T = sampled_t[bucket * params.block_size + uint(idx)];
      const int ii = i % 32;
      if (C > 0u) ar0 = -shared_pixels[0 * 32 + ii] + shared_sampled_ar[0 * 32 + ii];
      if (C > 1u) ar1 = -shared_pixels[1 * 32 + ii] + shared_sampled_ar[1 * 32 + ii];
      if (C > 2u) ar2 = -shared_pixels[2 * 32 + ii] + shared_sampled_ar[2 * 32 + ii];
      T_final = final_t[pix_id];
      last_contributor = float(n_contrib[pix_id]);
      if (C > 0u) dL_dpixel0 = dL_dout_color[0u * params.image_height * params.image_width + pix_id];
      if (C > 1u) dL_dpixel1 = dL_dout_color[1u * params.image_height * params.image_width + pix_id];
      if (C > 2u) dL_dpixel2 = dL_dout_color[2u * params.image_height * params.image_width + pix_id];
    }

    if (valid_splat && valid_pixel && 0 <= idx && idx < int(params.block_size)) {
      if (splat_idx_in_tile >= int(last_contributor)) {
        continue;
      }

      const float2 d = float2(xy.x - pixf.x, xy.y - pixf.y);
      const float power =
          -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
      if (power > 0.0f) {
        continue;
      }
      const float G = exp(power);
      const float alpha = min(0.99f, con_o.w * G);
      if (alpha < (1.0f / 255.0f)) {
        continue;
      }

      const float dchannel_dcolor = alpha * T;
      const float one_minus_alpha_reci = 1.0f / (1.0f - alpha);
      float dL_dalpha = 0.0f;

      if (C > 0u) {
        ar0 += dchannel_dcolor * c0;
        reg_dcolor0 += dchannel_dcolor * dL_dpixel0;
        dL_dalpha += (c0 * T + one_minus_alpha_reci * ar0) * dL_dpixel0;
      }
      if (C > 1u) {
        ar1 += dchannel_dcolor * c1;
        reg_dcolor1 += dchannel_dcolor * dL_dpixel1;
        dL_dalpha += (c1 * T + one_minus_alpha_reci * ar1) * dL_dpixel1;
      }
      if (C > 2u) {
        ar2 += dchannel_dcolor * c2;
        reg_dcolor2 += dchannel_dcolor * dL_dpixel2;
        dL_dalpha += (c2 * T + one_minus_alpha_reci * ar2) * dL_dpixel2;
      }

      float bg_dot_dpixel = 0.0f;
      if (C > 0u) bg_dot_dpixel += background[0] * dL_dpixel0;
      if (C > 1u) bg_dot_dpixel += background[1] * dL_dpixel1;
      if (C > 2u) bg_dot_dpixel += background[2] * dL_dpixel2;
      dL_dalpha += (-T_final * one_minus_alpha_reci) * bg_dot_dpixel;
      T *= (1.0f - alpha);

      const float dL_dG = con_o.w * dL_dalpha;
      const float gdx = G * d.x;
      const float gdy = G * d.y;
      const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
      const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

      const float tmp_x = dL_dG * dG_ddelx * ddelx_dx;
      const float tmp_y = dL_dG * dG_ddely * ddely_dy;
      reg_dmean_x += tmp_x;
      reg_dmean_y += tmp_y;
      reg_dmean_z += fabs(tmp_x);
      reg_dmean_w += fabs(tmp_y);

      reg_dconic_x += -0.5f * gdx * d.x * dL_dG;
      reg_dconic_y += -0.5f * gdx * d.y * dL_dG;
      reg_dconic_w += -0.5f * gdy * d.y * dL_dG;
      reg_dopacity += G * dL_dalpha;
    }
  }

  if (valid_splat) {
    const uint g = uint(gaussian_idx);
    atomic_fetch_add_explicit(&dL_dmeans2d[2 * g + 0], reg_dmean_x, memory_order_relaxed);
    atomic_fetch_add_explicit(&dL_dmeans2d[2 * g + 1], reg_dmean_y, memory_order_relaxed);
    atomic_fetch_add_explicit(&dL_dviewspace_points[4 * g + 0], reg_dmean_x, memory_order_relaxed);
    atomic_fetch_add_explicit(&dL_dviewspace_points[4 * g + 1], reg_dmean_y, memory_order_relaxed);
    atomic_fetch_add_explicit(&dL_dviewspace_points[4 * g + 2], reg_dmean_z, memory_order_relaxed);
    atomic_fetch_add_explicit(&dL_dviewspace_points[4 * g + 3], reg_dmean_w, memory_order_relaxed);

    atomic_fetch_add_explicit(&dL_dconic_opacity[4 * g + 0], reg_dconic_x, memory_order_relaxed);
    atomic_fetch_add_explicit(&dL_dconic_opacity[4 * g + 1], reg_dconic_y, memory_order_relaxed);
    atomic_fetch_add_explicit(&dL_dconic_opacity[4 * g + 2], reg_dconic_w, memory_order_relaxed);
    atomic_fetch_add_explicit(&dL_dconic_opacity[4 * g + 3], reg_dopacity, memory_order_relaxed);

    if (C > 0u) {
      atomic_fetch_add_explicit(
          &dL_dcolors[g * params.num_channels + 0u], reg_dcolor0, memory_order_relaxed);
    }
    if (C > 1u) {
      atomic_fetch_add_explicit(
          &dL_dcolors[g * params.num_channels + 1u], reg_dcolor1, memory_order_relaxed);
    }
    if (C > 2u) {
      atomic_fetch_add_explicit(
          &dL_dcolors[g * params.num_channels + 2u], reg_dcolor2, memory_order_relaxed);
    }
  }
}
