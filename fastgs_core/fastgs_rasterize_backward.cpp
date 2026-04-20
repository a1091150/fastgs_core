#include "include/fastgs_rasterize.h"
#include "include/helper.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace fastgs_core {

namespace {
constexpr int kRasterizeNumInputs = 10;
constexpr int kRasterizeNumOutputs = 9;

struct RasterizeBackwardKernelParams {
  uint32_t image_width;
  uint32_t image_height;
  uint32_t num_channels;
  uint32_t max_contrib_per_pixel;
};
}  // namespace

std::vector<mx::array> fastgs_rasterize_backward(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& cotangents,
    const RasterizeParams& params,
    mx::StreamOrDevice s) {
  if (primals.size() != kRasterizeNumInputs) {
    throw std::invalid_argument("fastgs_rasterize_backward expects 10 primals.");
  }
  if (cotangents.size() != kRasterizeNumOutputs) {
    throw std::invalid_argument("fastgs_rasterize_backward expects 9 cotangents.");
  }

  auto prim = std::make_shared<FastGSRasterizeBackward>(to_stream(s), params);
  std::vector<mx::Shape> output_shapes;
  std::vector<mx::Dtype> output_types;
  output_shapes.reserve(primals.size());
  output_types.reserve(primals.size());
  for (const auto& primal : primals) {
    output_shapes.push_back(primal.shape());
    output_types.push_back(primal.dtype());
  }

  std::vector<mx::array> inputs;
  inputs.reserve(primals.size() + cotangents.size());
  for (const auto& primal : primals) {
    inputs.push_back(mx::contiguous(primal));
  }
  for (const auto& cotangent : cotangents) {
    inputs.push_back(mx::contiguous(cotangent));
  }

  return mx::array::make_arrays(output_shapes, output_types, prim, inputs);
}

void FastGSRasterizeBackward::eval_cpu(const std::vector<mx::array>&,
                                       std::vector<mx::array>& outputs) {
  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    std::memset(out.data<void>(), 0, out.nbytes());
  }
}

void FastGSRasterizeBackward::eval_gpu(const std::vector<mx::array>& inputs,
                                       std::vector<mx::array>& outputs) {
  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    std::memset(out.data<void>(), 0, out.nbytes());
  }

#ifdef _METAL_
  // primals
  const auto& ranges = inputs[0];
  const auto& point_list = inputs[1];
  const auto& means2d = inputs[3];
  const auto& colors = inputs[4];
  const auto& conic_opacity = inputs[5];
  const auto& background = inputs[6];
  // cotangents
  const auto& out_color_cot = inputs[kRasterizeNumInputs + kOutColor];

  // grads wrt primals
  auto& dmeans2d = outputs[3];
  auto& dcolors = outputs[4];
  auto& dconic_opacity = outputs[5];
  auto& dviewspace_points = outputs[9];

  RasterizeBackwardKernelParams kernel_params = {
      .image_width = static_cast<uint32_t>(params_.image_width),
      .image_height = static_cast<uint32_t>(params_.image_height),
      .num_channels = static_cast<uint32_t>(params_.num_channels),
      .max_contrib_per_pixel = 1024u,
  };

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto lib = d.get_library("fastgs_core", fastgs_core::current_binary_dir());
  auto kernel = d.get_kernel("fastgs_render_backward_kernel", lib);

  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_bytes(kernel_params, 0);
  compute_encoder.set_input_array(ranges, 1);
  compute_encoder.set_input_array(point_list, 2);
  compute_encoder.set_input_array(means2d, 3);
  compute_encoder.set_input_array(colors, 4);
  compute_encoder.set_input_array(conic_opacity, 5);
  compute_encoder.set_input_array(background, 6);
  compute_encoder.set_input_array(out_color_cot, 7);
  compute_encoder.set_output_array(dmeans2d, 8);
  compute_encoder.set_output_array(dcolors, 9);
  compute_encoder.set_output_array(dconic_opacity, 10);
  compute_encoder.set_output_array(dviewspace_points, 11);

  const size_t bx = 16;
  const size_t by = 16;
  MTL::Size group_size = MTL::Size(bx, by, 1);
  MTL::Size grid_size = MTL::Size(
      static_cast<size_t>(params_.image_width + static_cast<int>(bx) - 1) /
          static_cast<int>(bx) * bx,
      static_cast<size_t>(params_.image_height + static_cast<int>(by) - 1) /
          static_cast<int>(by) * by,
      1);
  compute_encoder.dispatch_threads(grid_size, group_size);
#endif
}

std::vector<mx::array> FastGSRasterizeBackward::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("FastGSRasterizeBackward jvp is not implemented.");
}

std::vector<mx::array> FastGSRasterizeBackward::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
  throw std::runtime_error("FastGSRasterizeBackward vjp is not implemented.");
}

std::pair<std::vector<mx::array>, std::vector<int>> FastGSRasterizeBackward::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("FastGSRasterizeBackward vmap is not implemented.");
}

bool FastGSRasterizeBackward::is_equivalent(const mx::Primitive& other) const {
  return name() == other.name();
}

}  // namespace fastgs_core
