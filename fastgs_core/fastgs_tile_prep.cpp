#include "include/fastgs_tile_prep.h"
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
struct TilePrepKernelParams {
  uint32_t num_rendered;
  uint32_t num_tiles;
};
}  // namespace

std::vector<mx::array> fastgs_tile_prep(const TilePrepInput& input) {
  if (input.params.num_tiles == 0) {
    return {
        mx::zeros({0, 2}, mx::uint32, input.s),
        mx::zeros({0}, mx::uint32, input.s),
    };
  }

  auto prim = std::make_shared<FastGSTilePrep>(to_stream(input.s), input.params);
  std::vector<mx::Shape> output_shapes = {
      {input.params.num_tiles, 2},
      {input.params.num_tiles},
  };
  std::vector<mx::Dtype> output_types = {
      mx::uint32,
      mx::uint32,
  };
  std::vector<mx::array> inputs = {mx::contiguous(input.point_list_keys)};
  return mx::array::make_arrays(output_shapes, output_types, prim, inputs);
}

#ifdef _METAL_
void FastGSTilePrep::eval_gpu(const std::vector<mx::array>& inputs,
                              std::vector<mx::array>& outputs) {
  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    std::memset(out.data<void>(), 0, out.nbytes());
  }

  auto& point_list_keys = inputs[0];
  auto& ranges = outputs[kRanges];
  auto& bucket_count = outputs[kBucketCount];

  TilePrepKernelParams kernel_params = {
      .num_rendered = static_cast<uint32_t>(params_.num_rendered),
      .num_tiles = static_cast<uint32_t>(params_.num_tiles),
  };

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto lib = d.get_library("fastgs_core", fastgs_core::current_binary_dir());

  auto identify_kernel = d.get_kernel("fastgs_identify_tile_ranges_kernel", lib);
  auto bucket_kernel = d.get_kernel("fastgs_per_tile_bucket_count_kernel", lib);
  auto& compute_encoder = d.get_command_encoder(s.index);

  compute_encoder.set_compute_pipeline_state(identify_kernel);
  compute_encoder.set_bytes(kernel_params, 0);
  compute_encoder.set_input_array(point_list_keys, 1);
  compute_encoder.set_output_array(ranges, 2);
  {
    const int n = params_.num_rendered;
    if (n > 0) {
      const size_t max_threads = identify_kernel->maxTotalThreadsPerThreadgroup();
      size_t tgp_size = std::min(static_cast<size_t>(n), max_threads);
      MTL::Size group_size = MTL::Size(tgp_size, 1, 1);
      MTL::Size grid_size = MTL::Size(n, 1, 1);
      compute_encoder.dispatch_threads(grid_size, group_size);
    }
  }

  compute_encoder.set_compute_pipeline_state(bucket_kernel);
  compute_encoder.set_bytes(kernel_params, 0);
  compute_encoder.set_input_array(ranges, 1);
  compute_encoder.set_output_array(bucket_count, 2);
  {
    const int n = params_.num_tiles;
    const size_t max_threads = bucket_kernel->maxTotalThreadsPerThreadgroup();
    size_t tgp_size = std::min(static_cast<size_t>(n), max_threads);
    MTL::Size group_size = MTL::Size(tgp_size, 1, 1);
    MTL::Size grid_size = MTL::Size(n, 1, 1);
    compute_encoder.dispatch_threads(grid_size, group_size);
  }
}
#else
void FastGSTilePrep::eval_gpu(const std::vector<mx::array>&,
                              std::vector<mx::array>&) {
  throw std::runtime_error("FastGSTilePrep has no GPU implementation.");
}
#endif

void FastGSTilePrep::eval_cpu(const std::vector<mx::array>&,
                              std::vector<mx::array>& outputs) {
  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    std::memset(out.data<void>(), 0, out.nbytes());
  }
}

std::vector<mx::array> FastGSTilePrep::jvp(const std::vector<mx::array>&,
                                           const std::vector<mx::array>&,
                                           const std::vector<int>&) {
  throw std::runtime_error("FastGSTilePrep jvp is not implemented.");
}

std::vector<mx::array> FastGSTilePrep::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
  throw std::runtime_error("FastGSTilePrep vjp is not implemented.");
}

std::pair<std::vector<mx::array>, std::vector<int>> FastGSTilePrep::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("FastGSTilePrep vmap is not implemented.");
}

bool FastGSTilePrep::is_equivalent(const mx::Primitive& other) const {
  return name() == other.name();
}

}  // namespace fastgs_core
