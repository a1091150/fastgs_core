#include "include/fastgs_preprocess.h"
#include "include/helper.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace fastgs_core {

namespace {
struct PreprocessKernelParams {
  int degree;
  int max_sh_coeffs;
  float scale_modifier;
  float mult;
  float tan_fovx;
  float tan_fovy;
  float focal_x;
  float focal_y;
  uint32_t image_width;
  uint32_t image_height;
  uint32_t tile_bounds_x;
  uint32_t tile_bounds_y;
  uint32_t tile_bounds_z;
  uint32_t prefiltered;
  uint32_t use_cov3d_precomp;
  uint32_t use_colors_precomp;
};
}  // namespace

std::vector<mx::array> fastgs_preprocess(const PreprocessInput& input) {
  auto prim = std::make_shared<FastGSPreprocess>(to_stream(input.s), input.params);

  const int n = input.means3d.shape(0);
  std::vector<mx::Shape> output_shapes = {
      {n},
      {n, 2},
      {n},
      {n, 6},
      {n, 3},
      {n, 4},
      {n},
      {n, 3},
      {n, 4},
  };
  std::vector<mx::Dtype> output_types = {
      mx::int32,
      mx::float32,
      mx::float32,
      mx::float32,
      mx::float32,
      mx::float32,
      mx::uint32,
      mx::bool_,
      input.viewspace_points.dtype(),
  };

  std::vector<mx::array> inputs = {
      mx::contiguous(input.means3d),       mx::contiguous(input.dc),
      mx::contiguous(input.sh),            mx::contiguous(input.colors_precomp),
      mx::contiguous(input.opacities),     mx::contiguous(input.scales),
      mx::contiguous(input.quats),         mx::contiguous(input.cov3d_precomp),
      mx::contiguous(input.viewmat),       mx::contiguous(input.projmat),
      mx::contiguous(input.cam_pos),       mx::contiguous(input.viewspace_points),
  };

  return mx::array::make_arrays(output_shapes, output_types, prim, inputs);
}

#ifdef _METAL_
void FastGSPreprocess::eval_gpu(const std::vector<mx::array>& inputs,
                                std::vector<mx::array>& outputs) {
  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    std::memset(out.data<void>(), 0, out.nbytes());
  }

  auto& means3d = inputs[0];
  auto& dc = inputs[1];
  auto& sh = inputs[2];
  auto& colors_precomp = inputs[3];
  auto& opacities = inputs[4];
  auto& scales = inputs[5];
  auto& quats = inputs[6];
  auto& cov3d_precomp = inputs[7];
  auto& viewmat = inputs[8];
  auto& projmat = inputs[9];
  auto& cam_pos = inputs[10];
  auto& viewspace_points = inputs[11];

  auto& radii = outputs[0];
  auto& xys = outputs[1];
  auto& depths = outputs[2];
  auto& cov3d = outputs[3];
  auto& rgbs = outputs[4];
  auto& conic_opacity = outputs[5];
  auto& tiles_touched = outputs[6];
  auto& clamped = outputs[7];
  auto& viewspace_points_out = outputs[8];

  const int n = means3d.shape(0);

  PreprocessKernelParams kernel_params = {
      .degree = params_.degree,
      .max_sh_coeffs = params_.max_sh_coeffs,
      .scale_modifier = params_.scale_modifier,
      .mult = params_.mult,
      .tan_fovx = params_.tan_fovx,
      .tan_fovy = params_.tan_fovy,
      .focal_x = static_cast<float>(params_.image_width) / (2.0f * params_.tan_fovx),
      .focal_y = static_cast<float>(params_.image_height) / (2.0f * params_.tan_fovy),
      .image_width = static_cast<uint32_t>(params_.image_width),
      .image_height = static_cast<uint32_t>(params_.image_height),
      .tile_bounds_x = static_cast<uint32_t>(std::get<0>(params_.tile_bounds)),
      .tile_bounds_y = static_cast<uint32_t>(std::get<1>(params_.tile_bounds)),
      .tile_bounds_z = static_cast<uint32_t>(std::get<2>(params_.tile_bounds)),
      .prefiltered = static_cast<uint32_t>(params_.prefiltered),
      .use_cov3d_precomp = static_cast<uint32_t>(params_.use_cov3d_precomp),
      .use_colors_precomp = static_cast<uint32_t>(params_.use_colors_precomp),
  };

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto lib = d.get_library("fastgs_core", current_binary_dir());
  auto kernel = d.get_kernel("fastgs_preprocess_forward_kernel", lib);

  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_bytes(n, 0);
  compute_encoder.set_bytes(kernel_params, 1);
  compute_encoder.set_input_array(means3d, 2);
  compute_encoder.set_input_array(dc, 3);
  compute_encoder.set_input_array(sh, 4);
  compute_encoder.set_input_array(colors_precomp, 5);
  compute_encoder.set_input_array(opacities, 6);
  compute_encoder.set_input_array(scales, 7);
  compute_encoder.set_input_array(quats, 8);
  compute_encoder.set_input_array(cov3d_precomp, 9);
  compute_encoder.set_input_array(viewmat, 10);
  compute_encoder.set_input_array(projmat, 11);
  compute_encoder.set_input_array(cam_pos, 12);
  compute_encoder.set_input_array(viewspace_points, 13);
  compute_encoder.set_output_array(radii, 14);
  compute_encoder.set_output_array(xys, 15);
  compute_encoder.set_output_array(depths, 16);
  compute_encoder.set_output_array(cov3d, 17);
  compute_encoder.set_output_array(rgbs, 18);
  compute_encoder.set_output_array(conic_opacity, 19);
  compute_encoder.set_output_array(tiles_touched, 20);
  compute_encoder.set_output_array(clamped, 21);
  compute_encoder.set_output_array(viewspace_points_out, 22);

  const size_t max_threads = kernel->maxTotalThreadsPerThreadgroup();
  size_t tgp_size = std::min(static_cast<size_t>(n), max_threads);
  MTL::Size group_size = MTL::Size(tgp_size, 1, 1);
  MTL::Size grid_size = MTL::Size(n, 1, 1);
  compute_encoder.dispatch_threads(grid_size, group_size);
}
#else
void FastGSPreprocess::eval_gpu(const std::vector<mx::array>&,
                                std::vector<mx::array>&) {
  throw std::runtime_error("FastGSPreprocess has no GPU implementation.");
}
#endif

void FastGSPreprocess::eval_cpu(const std::vector<mx::array>&,
                                std::vector<mx::array>& outputs) {
  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    std::memset(out.data<void>(), 0, out.nbytes());
  }
}

std::vector<mx::array> FastGSPreprocess::jvp(const std::vector<mx::array>&,
                                             const std::vector<mx::array>&,
                                             const std::vector<int>&) {
  throw std::runtime_error("FastGSPreprocess jvp is not implemented.");
}

std::vector<mx::array> FastGSPreprocess::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
  throw std::runtime_error("FastGSPreprocess vjp is not implemented yet.");
}

std::pair<std::vector<mx::array>, std::vector<int>> FastGSPreprocess::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("FastGSPreprocess vmap is not implemented.");
}

bool FastGSPreprocess::is_equivalent(const mx::Primitive& other) const {
  return name() == other.name();
}

}  // namespace fastgs_core
