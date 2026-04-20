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
struct RasterizeKernelParams {
  uint32_t image_width;
  uint32_t image_height;
  uint32_t block_x;
  uint32_t block_y;
  uint32_t num_channels;
  uint32_t num_tiles;
  uint32_t bucket_sum;
  uint32_t get_flag;
};
}  // namespace

std::vector<mx::array> fastgs_rasterize(const RasterizeInput& input) {
  if (input.means2d.ndim() != 2 || input.means2d.shape(1) != 2) {
    throw std::invalid_argument("FastGSRasterize expects means2d with shape (N, 2).");
  }
  if (input.viewspace_points.ndim() != 2 || input.viewspace_points.shape(1) != 4) {
    throw std::invalid_argument(
        "FastGSRasterize expects viewspace_points with shape (N, 4).");
  }
  if (input.means2d.shape(0) != input.viewspace_points.shape(0)) {
    throw std::invalid_argument(
        "FastGSRasterize expects means2d and viewspace_points to share the same N.");
  }

  auto prim = std::make_shared<FastGSRasterize>(to_stream(input.s), input.params);

  const int num_pixels = input.params.image_width * input.params.image_height;
  const int sample_size =
      input.params.bucket_sum * input.params.block_x * input.params.block_y;

  std::vector<mx::Shape> output_shapes = {
      {sample_size},
      {sample_size},
      {input.params.num_channels, sample_size},
      {num_pixels},
      {num_pixels},
      {input.params.num_tiles},
      {input.params.num_channels, num_pixels},
      {input.params.num_channels, num_pixels},
      input.metric_count.shape(),
  };
  std::vector<mx::Dtype> output_types = {
      mx::uint32,
      mx::float32,
      mx::float32,
      mx::float32,
      mx::uint32,
      mx::uint32,
      mx::float32,
      mx::float32,
      input.metric_count.dtype(),
  };

  std::vector<mx::array> inputs = {
      mx::contiguous(input.ranges),
      mx::contiguous(input.point_list),
      mx::contiguous(input.per_tile_bucket_offset),
      mx::contiguous(input.means2d),
      mx::contiguous(input.colors),
      mx::contiguous(input.conic_opacity),
      mx::contiguous(input.background),
      mx::contiguous(input.radii),
      mx::contiguous(input.metric_map),
      mx::contiguous(input.viewspace_points),
  };

  return mx::array::make_arrays(output_shapes, output_types, prim, inputs);
}

#ifdef _METAL_
void FastGSRasterize::eval_gpu(const std::vector<mx::array>& inputs,
                               std::vector<mx::array>& outputs) {
  for (size_t i = 0; i < outputs.size(); ++i) {
    outputs[i].set_data(mx::allocator::malloc(outputs[i].nbytes()));
    std::memset(outputs[i].data<void>(), 0, outputs[i].nbytes());
  }

  auto& ranges = inputs[0];
  auto& point_list = inputs[1];
  auto& per_tile_bucket_offset = inputs[2];
  auto& means2d = inputs[3];
  auto& colors = inputs[4];
  auto& conic_opacity = inputs[5];
  auto& background = inputs[6];
  auto& radii = inputs[7];
  auto& metric_map = inputs[8];

  auto& bucket_to_tile = outputs[kBucketToTile];
  auto& sampled_t = outputs[kSampledT];
  auto& sampled_ar = outputs[kSampledAr];
  auto& final_t = outputs[kFinalT];
  auto& n_contrib = outputs[kNContrib];
  auto& max_contrib = outputs[kMaxContrib];
  auto& pixel_colors = outputs[kPixelColors];
  auto& out_color = outputs[kOutColor];
  auto& metric_count = outputs[kMetricCount];

  RasterizeKernelParams kernel_params = {
      .image_width = static_cast<uint32_t>(params_.image_width),
      .image_height = static_cast<uint32_t>(params_.image_height),
      .block_x = static_cast<uint32_t>(params_.block_x),
      .block_y = static_cast<uint32_t>(params_.block_y),
      .num_channels = static_cast<uint32_t>(params_.num_channels),
      .num_tiles = static_cast<uint32_t>(params_.num_tiles),
      .bucket_sum = static_cast<uint32_t>(params_.bucket_sum),
      .get_flag = static_cast<uint32_t>(params_.get_flag),
  };

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto lib = d.get_library("fastgs_core", fastgs_core::current_binary_dir());
  auto kernel = d.get_kernel("fastgs_render_forward_kernel", lib);

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
  compute_encoder.set_input_array(radii, 8);
  compute_encoder.set_input_array(metric_map, 9);
  compute_encoder.set_output_array(bucket_to_tile, 10);
  compute_encoder.set_output_array(sampled_t, 11);
  compute_encoder.set_output_array(sampled_ar, 12);
  compute_encoder.set_output_array(final_t, 13);
  compute_encoder.set_output_array(n_contrib, 14);
  compute_encoder.set_output_array(max_contrib, 15);
  compute_encoder.set_output_array(pixel_colors, 16);
  compute_encoder.set_output_array(out_color, 17);
  compute_encoder.set_output_array(metric_count, 18);

  const size_t bx = static_cast<size_t>(params_.block_x);
  const size_t by = static_cast<size_t>(params_.block_y);
  MTL::Size group_size = MTL::Size(bx, by, 1);
  MTL::Size grid_size = MTL::Size(
      static_cast<size_t>(params_.image_width + params_.block_x - 1) /
          params_.block_x *
          bx,
      static_cast<size_t>(params_.image_height + params_.block_y - 1) /
          params_.block_y *
          by,
      1);
  compute_encoder.dispatch_threads(grid_size, group_size);
}
#else
void FastGSRasterize::eval_gpu(const std::vector<mx::array>&,
                               std::vector<mx::array>&) {
  throw std::runtime_error("FastGSRasterize has no GPU implementation.");
}
#endif

void FastGSRasterize::eval_cpu(const std::vector<mx::array>&,
                               std::vector<mx::array>& outputs) {
  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    std::memset(out.data<void>(), 0, out.nbytes());
  }
}

std::vector<mx::array> FastGSRasterize::jvp(const std::vector<mx::array>&,
                                            const std::vector<mx::array>&,
                                            const std::vector<int>&) {
  throw std::runtime_error("FastGSRasterize jvp is not implemented.");
}

std::vector<mx::array> FastGSRasterize::vjp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mx::array>& outputs) {
  auto grads = fastgs_rasterize_backward(primals, cotangents, outputs, params_, stream());
  std::vector<mx::array> selected;
  selected.reserve(argnums.size());
  for (int argnum : argnums) {
    if (argnum < 0 || argnum >= static_cast<int>(grads.size())) {
      throw std::out_of_range("FastGSRasterize vjp argnum out of range.");
    }
    selected.push_back(grads[argnum]);
  }
  return selected;
}

std::pair<std::vector<mx::array>, std::vector<int>> FastGSRasterize::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("FastGSRasterize vmap is not implemented.");
}

bool FastGSRasterize::is_equivalent(const mx::Primitive& other) const {
  if (name() != other.name()) {
    return false;
  }
  auto other_ptr = dynamic_cast<const FastGSRasterize*>(&other);
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
