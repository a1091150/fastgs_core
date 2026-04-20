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
  uint32_t block_x;
  uint32_t block_y;
  uint32_t num_channels;
  uint32_t num_tiles;
  uint32_t bucket_sum;
  uint32_t block_size;
};
}  // namespace

std::vector<mx::array> fastgs_rasterize_backward(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& cotangents,
    const std::vector<mx::array>& forward_outputs,
    const RasterizeParams& params,
    mx::StreamOrDevice s) {
  if (primals.size() != kRasterizeNumInputs) {
    throw std::invalid_argument("fastgs_rasterize_backward expects 10 primals.");
  }
  if (cotangents.size() != kRasterizeNumOutputs) {
    throw std::invalid_argument("fastgs_rasterize_backward expects 9 cotangents.");
  }
  if (forward_outputs.size() != kRasterizeNumOutputs) {
    throw std::invalid_argument(
        "fastgs_rasterize_backward expects 9 forward outputs.");
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
  inputs.reserve(primals.size() + cotangents.size() + forward_outputs.size());
  for (const auto& primal : primals) {
    inputs.push_back(mx::contiguous(primal));
  }
  for (const auto& cotangent : cotangents) {
    inputs.push_back(mx::contiguous(cotangent));
  }
  for (const auto& out : forward_outputs) {
    inputs.push_back(mx::contiguous(out));
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
  const auto& per_tile_bucket_offset = inputs[2];
  const auto& means2d = inputs[3];
  const auto& colors = inputs[4];
  const auto& conic_opacity = inputs[5];
  const auto& background = inputs[6];
  // cotangents
  const auto& out_color_cot = inputs[kRasterizeNumInputs + kOutColor];
  // forward outputs
  const size_t fo = static_cast<size_t>(kRasterizeNumInputs + kRasterizeNumOutputs);
  const auto& bucket_to_tile = inputs[fo + kBucketToTile];
  const auto& sampled_t = inputs[fo + kSampledT];
  const auto& sampled_ar = inputs[fo + kSampledAr];
  const auto& final_t = inputs[fo + kFinalT];
  const auto& n_contrib = inputs[fo + kNContrib];
  const auto& max_contrib = inputs[fo + kMaxContrib];
  const auto& pixel_colors = inputs[fo + kPixelColors];

  // grads wrt primals
  auto& dmeans2d = outputs[3];
  auto& dcolors = outputs[4];
  auto& dconic_opacity = outputs[5];
  auto& dviewspace_points = outputs[9];

  RasterizeBackwardKernelParams kernel_params = {
      .image_width = static_cast<uint32_t>(params_.image_width),
      .image_height = static_cast<uint32_t>(params_.image_height),
      .block_x = static_cast<uint32_t>(std::max(1, params_.block_x)),
      .block_y = static_cast<uint32_t>(std::max(1, params_.block_y)),
      .num_channels = static_cast<uint32_t>(params_.num_channels),
      .num_tiles = static_cast<uint32_t>(std::max(1, params_.num_tiles)),
      .bucket_sum = static_cast<uint32_t>(std::max(0, params_.bucket_sum)),
      .block_size =
          static_cast<uint32_t>(std::max(1, params_.block_x * params_.block_y)),
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
  compute_encoder.set_input_array(per_tile_bucket_offset, 3);
  compute_encoder.set_input_array(means2d, 4);
  compute_encoder.set_input_array(colors, 5);
  compute_encoder.set_input_array(conic_opacity, 6);
  compute_encoder.set_input_array(background, 7);
  compute_encoder.set_input_array(out_color_cot, 8);
  compute_encoder.set_input_array(bucket_to_tile, 9);
  compute_encoder.set_input_array(sampled_t, 10);
  compute_encoder.set_input_array(sampled_ar, 11);
  compute_encoder.set_input_array(final_t, 12);
  compute_encoder.set_input_array(n_contrib, 13);
  compute_encoder.set_input_array(max_contrib, 14);
  compute_encoder.set_input_array(pixel_colors, 15);
  compute_encoder.set_output_array(dmeans2d, 16);
  compute_encoder.set_output_array(dcolors, 17);
  compute_encoder.set_output_array(dconic_opacity, 18);
  compute_encoder.set_output_array(dviewspace_points, 19);

  constexpr size_t kThreadsPerGroup = 32;
  const size_t launch_groups =
      (static_cast<size_t>(std::max(0, params_.bucket_sum)) * kThreadsPerGroup +
       kThreadsPerGroup - 1) /
      kThreadsPerGroup;
  MTL::Size group_size = MTL::Size(kThreadsPerGroup, 1, 1);
  MTL::Size grid_size = MTL::Size(launch_groups * kThreadsPerGroup, 1, 1);
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
  if (name() != other.name()) {
    return false;
  }
  auto other_ptr = dynamic_cast<const FastGSRasterizeBackward*>(&other);
  if (!other_ptr) {
    return false;
  }
  const auto& p = params_;
  const auto& q = other_ptr->params_;
  return p.image_width == q.image_width && p.image_height == q.image_height &&
         p.block_x == q.block_x && p.block_y == q.block_y &&
         p.num_channels == q.num_channels && p.num_tiles == q.num_tiles &&
         p.bucket_sum == q.bucket_sum && p.get_flag == q.get_flag;
}

}  // namespace fastgs_core
