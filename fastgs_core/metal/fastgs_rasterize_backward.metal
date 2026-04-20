#include <metal_stdlib>

using namespace metal;

struct RasterizeBackwardKernelParams {
  uint image_width;
  uint image_height;
  uint num_channels;
  uint max_contrib_per_pixel;
};

inline float2 read_packed_float2(const device float* arr, uint idx) {
  return float2(arr[2 * idx], arr[2 * idx + 1]);
}

inline float4 read_packed_float4(const device float* arr, uint idx) {
  return float4(arr[4 * idx], arr[4 * idx + 1], arr[4 * idx + 2], arr[4 * idx + 3]);
}

inline float dot3(const float3 a, const float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

kernel void fastgs_render_backward_kernel(
    constant RasterizeBackwardKernelParams& params [[buffer(0)]],
    const device uint* ranges [[buffer(1)]],
    const device uint* point_list [[buffer(2)]],
    const device float* means2d [[buffer(3)]],
    const device float* colors [[buffer(4)]],
    const device float* conic_opacity [[buffer(5)]],
    const device float* background [[buffer(6)]],
    const device float* dL_dout_color [[buffer(7)]],
    device atomic_float* dL_dmeans2d [[buffer(8)]],
    device atomic_float* dL_dcolors [[buffer(9)]],
    device atomic_float* dL_dconic_opacity [[buffer(10)]],
    device atomic_float* dL_dviewspace_points [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tgp [[threadgroup_position_in_grid]]) {
  const uint pix_x = gid.x;
  const uint pix_y = gid.y;
  if (pix_x >= params.image_width || pix_y >= params.image_height) {
    return;
  }

  const uint tiles_x = (params.image_width + 15u) / 16u;
  const uint tile_id = tgp.y * tiles_x + tgp.x;
  const uint2 range = uint2(ranges[2 * tile_id], ranges[2 * tile_id + 1]);
  const uint pix_id = pix_y * params.image_width + pix_x;

  float3 g = float3(0.0f);
  const uint num_channels = min(params.num_channels, 3u);
  for (uint ch = 0; ch < num_channels; ++ch) {
    g[ch] = dL_dout_color[ch * params.image_height * params.image_width + pix_id];
  }

  const float2 pixf = float2(float(pix_x), float(pix_y));

  constexpr uint kMaxLocal = 1024u;
  uint ids[kMaxLocal];
  float alphas[kMaxLocal];
  float prefix_t[kMaxLocal];

  float t_val = 1.0f;
  uint m = 0u;
  const uint max_steps = min(params.max_contrib_per_pixel, range.y - range.x);

  for (uint step = 0; step < max_steps; ++step) {
    const uint list_idx = range.x + step;
    if (list_idx >= range.y || m >= kMaxLocal) {
      break;
    }

    const uint gid_gauss = point_list[list_idx];
    const float2 xy = read_packed_float2(means2d, gid_gauss);
    const float2 d = xy - pixf;
    const float4 con_o = read_packed_float4(conic_opacity, gid_gauss);

    const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) -
                        con_o.y * d.x * d.y;
    if (power > 0.0f) {
      continue;
    }

    const float gexp = exp(power);
    const float alpha_raw = con_o.w * gexp;
    const float alpha = min(0.99f, alpha_raw);
    if (alpha < (1.0f / 255.0f)) {
      continue;
    }

    ids[m] = gid_gauss;
    alphas[m] = alpha;
    prefix_t[m] = t_val;

    const float next_t = t_val * (1.0f - alpha);
    t_val = next_t;
    ++m;

    if (t_val < 0.0001f) {
      break;
    }
  }

  float3 c_next = float3(background[0], background[1], background[2]);

  for (int j = int(m) - 1; j >= 0; --j) {
    const uint gid_gauss = ids[j];
    const float alpha = alphas[j];
    const float Ti = prefix_t[j];

    float3 color = float3(0.0f);
    for (uint ch = 0; ch < num_channels; ++ch) {
      color[ch] = colors[gid_gauss * params.num_channels + ch];
      const float dcolor = g[ch] * alpha * Ti;
      atomic_fetch_add_explicit(&dL_dcolors[gid_gauss * params.num_channels + ch],
                                dcolor,
                                memory_order_relaxed);
    }

    const float d_alpha = dot3(g, (color - c_next)) * Ti;

    const float2 xy = read_packed_float2(means2d, gid_gauss);
    const float2 d = xy - pixf;
    const float4 con_o = read_packed_float4(conic_opacity, gid_gauss);
    const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) -
                        con_o.y * d.x * d.y;
    const float gexp = exp(power);
    const float alpha_raw = con_o.w * gexp;
    const bool clipped = alpha_raw > 0.99f;

    float d_opacity = 0.0f;
    float d_a = 0.0f;
    float d_b = 0.0f;
    float d_c = 0.0f;
    float d_mx = 0.0f;
    float d_my = 0.0f;

    if (!clipped) {
      const float d_gexp = d_alpha * con_o.w;
      const float d_power = d_gexp * gexp;
      d_opacity = d_alpha * gexp;
      d_a = d_power * (-0.5f * d.x * d.x);
      d_b = d_power * (-d.x * d.y);
      d_c = d_power * (-0.5f * d.y * d.y);
      d_mx = d_power * (-(con_o.x * d.x + con_o.y * d.y));
      d_my = d_power * (-(con_o.z * d.y + con_o.y * d.x));
    }

    atomic_fetch_add_explicit(&dL_dconic_opacity[4 * gid_gauss + 0], d_a, memory_order_relaxed);
    atomic_fetch_add_explicit(&dL_dconic_opacity[4 * gid_gauss + 1], d_b, memory_order_relaxed);
    atomic_fetch_add_explicit(&dL_dconic_opacity[4 * gid_gauss + 2], d_c, memory_order_relaxed);
    atomic_fetch_add_explicit(&dL_dconic_opacity[4 * gid_gauss + 3], d_opacity, memory_order_relaxed);

    atomic_fetch_add_explicit(&dL_dmeans2d[2 * gid_gauss + 0], d_mx, memory_order_relaxed);
    atomic_fetch_add_explicit(&dL_dmeans2d[2 * gid_gauss + 1], d_my, memory_order_relaxed);
    atomic_fetch_add_explicit(&dL_dviewspace_points[4 * gid_gauss + 0], d_mx, memory_order_relaxed);
    atomic_fetch_add_explicit(&dL_dviewspace_points[4 * gid_gauss + 1], d_my, memory_order_relaxed);

    c_next = alpha * color + (1.0f - alpha) * c_next;
  }
}
