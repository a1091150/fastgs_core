#include "include/fastgs_binning.h"
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
struct BinningKernelParams {
  float mult;
  uint32_t tile_bounds_x;
  uint32_t tile_bounds_y;
  uint32_t tile_bounds_z;
};
}  // namespace

std::vector<mx::array> fastgs_binning(const BinningInput& input) {
  if (input.params.num_rendered == 0) {
    return {
        mx::zeros({0}, mx::uint64, input.s),
        mx::zeros({0}, mx::uint32, input.s),
    };
  }

  auto prim = std::make_shared<FastGSBinning>(to_stream(input.s), input.params);

  std::vector<mx::Shape> output_shapes = {
      {input.params.num_rendered},
      {input.params.num_rendered},
  };
  std::vector<mx::Dtype> output_types = {
      mx::uint64,
      mx::uint32,
  };

  std::vector<mx::array> inputs = {
      mx::contiguous(input.xys),
      mx::contiguous(input.depths),
      mx::contiguous(input.point_offsets),
      mx::contiguous(input.conic_opacity),
      mx::contiguous(input.tiles_touched),
  };

  return mx::array::make_arrays(output_shapes, output_types, prim, inputs);
}

#ifdef _METAL_
void FastGSBinning::eval_gpu(const std::vector<mx::array>& inputs,
                             std::vector<mx::array>& outputs) {
  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    std::memset(out.data<void>(), 0, out.nbytes());
  }

  auto& xys = inputs[0];
  auto& depths = inputs[1];
  auto& point_offsets = inputs[2];
  auto& conic_opacity = inputs[3];
  auto& tiles_touched = inputs[4];
  auto& gaussian_keys_unsorted = outputs[0];
  auto& gaussian_values_unsorted = outputs[1];

  BinningKernelParams kernel_params = {
      .mult = params_.mult,
      .tile_bounds_x = static_cast<uint32_t>(std::get<0>(params_.tile_bounds)),
      .tile_bounds_y = static_cast<uint32_t>(std::get<1>(params_.tile_bounds)),
      .tile_bounds_z = static_cast<uint32_t>(std::get<2>(params_.tile_bounds)),
  };

  const int p = xys.shape(0);

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto lib = d.get_library("fastgs_core", fastgs_core::current_binary_dir());
  auto kernel = d.get_kernel("fastgs_duplicate_with_keys_kernel", lib);

  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_bytes(p, 0);
  compute_encoder.set_bytes(kernel_params, 1);
  compute_encoder.set_input_array(xys, 2);
  compute_encoder.set_input_array(depths, 3);
  compute_encoder.set_input_array(point_offsets, 4);
  compute_encoder.set_input_array(conic_opacity, 5);
  compute_encoder.set_input_array(tiles_touched, 6);
  compute_encoder.set_output_array(gaussian_keys_unsorted, 7);
  compute_encoder.set_output_array(gaussian_values_unsorted, 8);

  const size_t max_threads = kernel->maxTotalThreadsPerThreadgroup();
  size_t tgp_size = std::min(static_cast<size_t>(p), max_threads);
  MTL::Size group_size = MTL::Size(tgp_size, 1, 1);
  MTL::Size grid_size = MTL::Size(p, 1, 1);
  compute_encoder.dispatch_threads(grid_size, group_size);
}
#else
void FastGSBinning::eval_gpu(const std::vector<mx::array>&,
                             std::vector<mx::array>&) {
  throw std::runtime_error("FastGSBinning has no GPU implementation.");
}
#endif

void FastGSBinning::eval_cpu(const std::vector<mx::array>&,
                             std::vector<mx::array>& outputs) {
  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    std::memset(out.data<void>(), 0, out.nbytes());
  }
}

std::vector<mx::array> FastGSBinning::jvp(const std::vector<mx::array>&,
                                          const std::vector<mx::array>&,
                                          const std::vector<int>&) {
  throw std::runtime_error("FastGSBinning jvp is not implemented.");
}

std::vector<mx::array> FastGSBinning::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
  throw std::runtime_error("FastGSBinning vjp is not implemented.");
}

std::pair<std::vector<mx::array>, std::vector<int>> FastGSBinning::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("FastGSBinning vmap is not implemented.");
}

bool FastGSBinning::is_equivalent(const mx::Primitive& other) const {
  return name() == other.name();
}

}  // namespace fastgs_core
