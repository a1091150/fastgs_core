#include <metal_stdlib>

using namespace metal;

struct PreprocessBackwardKernelParams {
  uint n;
  uint image_width;
  uint image_height;
  uint use_cov3d_precomp;
  uint use_colors_precomp;
  int degree;
  int max_sh_coeffs;
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

inline float3 dnormvdv(float3 v, float3 g) {
  float len = max(length(v), 1e-6f);
  float3 n = v / len;
  return (g - n * dot(n, g)) / len;
}

kernel void fastgs_preprocess_backward_kernel(
    constant PreprocessBackwardKernelParams& params [[buffer(0)]],
    const device float* means3d [[buffer(1)]],
    const device float* scales [[buffer(2)]],
    const device float* quats [[buffer(3)]],
    const device float* viewmat [[buffer(4)]],
    const device float* projmat [[buffer(5)]],
    const device float* cam_pos [[buffer(6)]],
    const device float* sh [[buffer(7)]],
    const device float* dL_dcov3d [[buffer(8)]],
    const device float* dL_drgb [[buffer(9)]],
    const device float* dL_dxys [[buffer(10)]],
    const device float* dL_ddepths [[buffer(11)]],
    const device float* dL_dconic_opacity [[buffer(12)]],
    const device float* dL_dviewspace_out [[buffer(13)]],
    const device bool* clamped [[buffer(14)]],
    device float* dL_dmeans3d [[buffer(15)]],
    device float* dL_ddc [[buffer(16)]],
    device float* dL_dsh [[buffer(17)]],
    device float* dL_dcolors_precomp [[buffer(18)]],
    device float* dL_dopacities [[buffer(19)]],
    device float* dL_dscales [[buffer(20)]],
    device float* dL_dquats [[buffer(21)]],
    device float* dL_dviewspace_in [[buffer(22)]],
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

  if (params.use_colors_precomp != 0u) {
    dL_dcolors_precomp[3 * tid + 0] = dL_drgb[3 * tid + 0];
    dL_dcolors_precomp[3 * tid + 1] = dL_drgb[3 * tid + 1];
    dL_dcolors_precomp[3 * tid + 2] = dL_drgb[3 * tid + 2];
  } else {
    float3 dL_dRGB = float3(
        clamped[3 * tid + 0] ? 0.0f : dL_drgb[3 * tid + 0],
        clamped[3 * tid + 1] ? 0.0f : dL_drgb[3 * tid + 1],
        clamped[3 * tid + 2] ? 0.0f : dL_drgb[3 * tid + 2]);

    // degree-0 dc term
    constexpr float SH_C0 = 0.28209479177387814f;
    dL_ddc[3 * tid + 0] = SH_C0 * dL_dRGB[0];
    dL_ddc[3 * tid + 1] = SH_C0 * dL_dRGB[1];
    dL_ddc[3 * tid + 2] = SH_C0 * dL_dRGB[2];
    dL_dcolors_precomp[3 * tid + 0] = 0.0f;
    dL_dcolors_precomp[3 * tid + 1] = 0.0f;
    dL_dcolors_precomp[3 * tid + 2] = 0.0f;

    if (params.degree > 0 && params.max_sh_coeffs >= 3) {
      constexpr float SH_C1 = 0.4886025119029199f;
      constexpr float SH_C2[5] = {
          1.0925484305920792f,
          -1.0925484305920792f,
          0.31539156525252005f,
          -1.0925484305920792f,
          0.5462742152960396f,
      };
      constexpr float SH_C3[7] = {
          -0.5900435899266435f,
          2.890611442640554f,
          -0.4570457994644658f,
          0.3731763325901154f,
          -0.4570457994644658f,
          1.445305721320277f,
          -0.5900435899266435f,
      };
      float3 pos = read_packed_float3(means3d, tid, 3u);
      float3 campos = float3(cam_pos[0], cam_pos[1], cam_pos[2]);
      float3 dir_orig = pos - campos;
      float3 dir = normalize(dir_orig);
      float x = dir.x;
      float y = dir.y;
      float z = dir.z;

      uint base = (tid * uint(params.max_sh_coeffs)) * 3u;
      float3 sh0 = float3(sh[base + 0], sh[base + 1], sh[base + 2]);
      float3 sh1 = float3(sh[base + 3], sh[base + 4], sh[base + 5]);
      float3 sh2 = float3(sh[base + 6], sh[base + 7], sh[base + 8]);

      float3 g0 = (-SH_C1 * y) * dL_dRGB;
      float3 g1 = ( SH_C1 * z) * dL_dRGB;
      float3 g2 = (-SH_C1 * x) * dL_dRGB;
      dL_dsh[base + 0] = g0.x; dL_dsh[base + 1] = g0.y; dL_dsh[base + 2] = g0.z;
      dL_dsh[base + 3] = g1.x; dL_dsh[base + 4] = g1.y; dL_dsh[base + 5] = g1.z;
      dL_dsh[base + 6] = g2.x; dL_dsh[base + 7] = g2.y; dL_dsh[base + 8] = g2.z;

      float3 dRGBdx = -SH_C1 * sh2;
      float3 dRGBdy = -SH_C1 * sh0;
      float3 dRGBdz =  SH_C1 * sh1;

      if (params.degree > 1 && params.max_sh_coeffs >= 8) {
        float xx = x * x, yy = y * y, zz = z * z;
        float xy = x * y, yz = y * z, xz = x * z;
        uint b3 = base + 9u;   // coeff 3
        uint b4 = base + 12u;  // coeff 4
        uint b5 = base + 15u;  // coeff 5
        uint b6 = base + 18u;  // coeff 6
        uint b7 = base + 21u;  // coeff 7

        float3 sh3 = float3(sh[b3 + 0], sh[b3 + 1], sh[b3 + 2]);
        float3 sh4 = float3(sh[b4 + 0], sh[b4 + 1], sh[b4 + 2]);
        float3 sh5 = float3(sh[b5 + 0], sh[b5 + 1], sh[b5 + 2]);
        float3 sh6 = float3(sh[b6 + 0], sh[b6 + 1], sh[b6 + 2]);
        float3 sh7 = float3(sh[b7 + 0], sh[b7 + 1], sh[b7 + 2]);

        float3 g3 = (SH_C2[0] * xy) * dL_dRGB;
        float3 g4 = (SH_C2[1] * yz) * dL_dRGB;
        float3 g5 = (SH_C2[2] * (2.0f * zz - xx - yy)) * dL_dRGB;
        float3 g6 = (SH_C2[3] * xz) * dL_dRGB;
        float3 g7 = (SH_C2[4] * (xx - yy)) * dL_dRGB;
        dL_dsh[b3 + 0] = g3.x; dL_dsh[b3 + 1] = g3.y; dL_dsh[b3 + 2] = g3.z;
        dL_dsh[b4 + 0] = g4.x; dL_dsh[b4 + 1] = g4.y; dL_dsh[b4 + 2] = g4.z;
        dL_dsh[b5 + 0] = g5.x; dL_dsh[b5 + 1] = g5.y; dL_dsh[b5 + 2] = g5.z;
        dL_dsh[b6 + 0] = g6.x; dL_dsh[b6 + 1] = g6.y; dL_dsh[b6 + 2] = g6.z;
        dL_dsh[b7 + 0] = g7.x; dL_dsh[b7 + 1] = g7.y; dL_dsh[b7 + 2] = g7.z;

        dRGBdx += SH_C2[0] * y * sh3 + SH_C2[2] * 2.f * -x * sh5 + SH_C2[3] * z * sh6 + SH_C2[4] * 2.f * x * sh7;
        dRGBdy += SH_C2[0] * x * sh3 + SH_C2[1] * z * sh4 + SH_C2[2] * 2.f * -y * sh5 + SH_C2[4] * 2.f * -y * sh7;
        dRGBdz += SH_C2[1] * y * sh4 + SH_C2[2] * 4.f * z * sh5 + SH_C2[3] * x * sh6;

        if (params.degree > 2 && params.max_sh_coeffs >= 15) {
          uint b8 = base + 24u;
          uint b9 = base + 27u;
          uint b10 = base + 30u;
          uint b11 = base + 33u;
          uint b12 = base + 36u;
          uint b13 = base + 39u;
          uint b14 = base + 42u;

          float3 sh8 = float3(sh[b8 + 0], sh[b8 + 1], sh[b8 + 2]);
          float3 sh9 = float3(sh[b9 + 0], sh[b9 + 1], sh[b9 + 2]);
          float3 sh10 = float3(sh[b10 + 0], sh[b10 + 1], sh[b10 + 2]);
          float3 sh11 = float3(sh[b11 + 0], sh[b11 + 1], sh[b11 + 2]);
          float3 sh12 = float3(sh[b12 + 0], sh[b12 + 1], sh[b12 + 2]);
          float3 sh13 = float3(sh[b13 + 0], sh[b13 + 1], sh[b13 + 2]);
          float3 sh14 = float3(sh[b14 + 0], sh[b14 + 1], sh[b14 + 2]);

          float3 g8 = (SH_C3[0] * y * (3.f * xx - yy)) * dL_dRGB;
          float3 g9 = (SH_C3[1] * xy * z) * dL_dRGB;
          float3 g10 = (SH_C3[2] * y * (4.f * zz - xx - yy)) * dL_dRGB;
          float3 g11 = (SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy)) * dL_dRGB;
          float3 g12 = (SH_C3[4] * x * (4.f * zz - xx - yy)) * dL_dRGB;
          float3 g13 = (SH_C3[5] * z * (xx - yy)) * dL_dRGB;
          float3 g14 = (SH_C3[6] * x * (xx - 3.f * yy)) * dL_dRGB;
          dL_dsh[b8 + 0] = g8.x; dL_dsh[b8 + 1] = g8.y; dL_dsh[b8 + 2] = g8.z;
          dL_dsh[b9 + 0] = g9.x; dL_dsh[b9 + 1] = g9.y; dL_dsh[b9 + 2] = g9.z;
          dL_dsh[b10 + 0] = g10.x; dL_dsh[b10 + 1] = g10.y; dL_dsh[b10 + 2] = g10.z;
          dL_dsh[b11 + 0] = g11.x; dL_dsh[b11 + 1] = g11.y; dL_dsh[b11 + 2] = g11.z;
          dL_dsh[b12 + 0] = g12.x; dL_dsh[b12 + 1] = g12.y; dL_dsh[b12 + 2] = g12.z;
          dL_dsh[b13 + 0] = g13.x; dL_dsh[b13 + 1] = g13.y; dL_dsh[b13 + 2] = g13.z;
          dL_dsh[b14 + 0] = g14.x; dL_dsh[b14 + 1] = g14.y; dL_dsh[b14 + 2] = g14.z;

          dRGBdx += (
              SH_C3[0] * sh8 * 6.f * xy +
              SH_C3[1] * sh9 * yz +
              SH_C3[2] * sh10 * -2.f * xy +
              SH_C3[3] * sh11 * -6.f * xz +
              SH_C3[4] * sh12 * (-3.f * xx + 4.f * zz - yy) +
              SH_C3[5] * sh13 * 2.f * xz +
              SH_C3[6] * sh14 * 3.f * (xx - yy));

          dRGBdy += (
              SH_C3[0] * sh8 * 3.f * (xx - yy) +
              SH_C3[1] * sh9 * xz +
              SH_C3[2] * sh10 * (-3.f * yy + 4.f * zz - xx) +
              SH_C3[3] * sh11 * -6.f * yz +
              SH_C3[4] * sh12 * -2.f * xy +
              SH_C3[5] * sh13 * -2.f * yz +
              SH_C3[6] * sh14 * -6.f * xy);

          dRGBdz += (
              SH_C3[1] * sh9 * xy +
              SH_C3[2] * sh10 * 8.f * yz +
              SH_C3[3] * sh11 * 3.f * (2.f * zz - xx - yy) +
              SH_C3[4] * sh12 * 8.f * xz +
              SH_C3[5] * sh13 * (xx - yy));
        }
      }
      float3 dL_ddir = float3(dot(dRGBdx, dL_dRGB), dot(dRGBdy, dL_dRGB), dot(dRGBdz, dL_dRGB));
      float3 dL_dmean_sh = dnormvdv(dir_orig, dL_ddir);
      dL_dmeans3d[3 * tid + 0] += dL_dmean_sh.x;
      dL_dmeans3d[3 * tid + 1] += dL_dmean_sh.y;
      dL_dmeans3d[3 * tid + 2] += dL_dmean_sh.z;
    }
  }

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
