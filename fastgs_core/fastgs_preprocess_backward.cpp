#include "include/fastgs_preprocess.h"
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
constexpr int kPreprocessNumInputs = 12;
constexpr int kPreprocessNumOutputs = 9;

struct PreprocessBackwardKernelParams {
  uint32_t n;
  uint32_t image_width;
  uint32_t image_height;
  uint32_t use_cov3d_precomp;
  uint32_t use_colors_precomp;
  int32_t degree;
  int32_t max_sh_coeffs;
};
}  // namespace

std::vector<mx::array> fastgs_preprocess_backward(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& cotangents,
    const std::vector<mx::array>& forward_outputs,
    const PreprocessParams& params,
    mx::StreamOrDevice s) {
  if (primals.size() != kPreprocessNumInputs) {
    throw std::invalid_argument("fastgs_preprocess_backward expects 12 primals.");
  }
  if (cotangents.size() != kPreprocessNumOutputs) {
    throw std::invalid_argument("fastgs_preprocess_backward expects 9 cotangents.");
  }
  if (forward_outputs.size() != kPreprocessNumOutputs) {
    throw std::invalid_argument("fastgs_preprocess_backward expects 9 forward outputs.");
  }

  auto prim =
      std::make_shared<FastGSPreprocessBackward>(to_stream(s), params);
  std::vector<mx::Shape> output_shapes;
  std::vector<mx::Dtype> output_types;
  output_shapes.reserve(primals.size());
  output_types.reserve(primals.size());
  for (const auto& primal : primals) {
    output_shapes.push_back(primal.shape());
    output_types.push_back(primal.dtype());
  }

  std::vector<mx::array> inputs;
  inputs.reserve(primals.size() + cotangents.size() + 1);
  for (const auto& primal : primals) {
    inputs.push_back(mx::contiguous(primal));
  }
  for (const auto& cotangent : cotangents) {
    inputs.push_back(mx::contiguous(cotangent));
  }
  inputs.push_back(mx::contiguous(forward_outputs[kClamped]));

  return mx::array::make_arrays(output_shapes, output_types, prim, inputs);
}

void FastGSPreprocessBackward::eval_cpu(const std::vector<mx::array>&,
                                        std::vector<mx::array>& outputs) {
  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    std::memset(out.data<void>(), 0, out.nbytes());
  }
}

void FastGSPreprocessBackward::eval_gpu(const std::vector<mx::array>& inputs,
                                        std::vector<mx::array>& outputs) {
  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    std::memset(out.data<void>(), 0, out.nbytes());
  }

#ifdef _METAL_
  const auto& means3d = inputs[0];
  const auto& sh = inputs[2];
  const auto& scales = inputs[5];
  const auto& quats = inputs[6];
  const auto& cam_pos = inputs[10];
  const auto& viewmat = inputs[8];
  const auto& projmat = inputs[9];
  const auto& d_cov3d = inputs[kPreprocessNumInputs + kCov3d];
  const auto& d_rgb = inputs[kPreprocessNumInputs + kRgb];
  const auto& d_xys = inputs[kPreprocessNumInputs + kXys];
  const auto& d_depths = inputs[kPreprocessNumInputs + kDepths];
  const auto& d_conic_opacity = inputs[kPreprocessNumInputs + kConicOpacity];
  const auto& d_viewspace = inputs[kPreprocessNumInputs + kViewspacePoints];
  const auto& clamped = inputs[kPreprocessNumInputs + kPreprocessNumOutputs];

  auto& d_means3d = outputs[0];
  auto& d_dc = outputs[1];
  auto& d_sh = outputs[2];
  auto& d_colors_precomp = outputs[3];
  auto& d_opacities = outputs[4];
  auto& d_scales = outputs[5];
  auto& d_quats = outputs[6];
  auto& d_viewspace_in = outputs[11];

  const uint32_t n = static_cast<uint32_t>(means3d.shape(0));
  PreprocessBackwardKernelParams params = {
      .n = n,
      .image_width = static_cast<uint32_t>(params_.image_width),
      .image_height = static_cast<uint32_t>(params_.image_height),
      .use_cov3d_precomp = static_cast<uint32_t>(params_.use_cov3d_precomp),
      .use_colors_precomp = static_cast<uint32_t>(params_.use_colors_precomp),
      .degree = params_.degree,
      .max_sh_coeffs = params_.max_sh_coeffs,
  };

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto lib = d.get_library("fastgs_core", fastgs_core::current_binary_dir());
  auto kernel = d.get_kernel("fastgs_preprocess_backward_kernel", lib);

  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_bytes(params, 0);
  compute_encoder.set_input_array(means3d, 1);
  compute_encoder.set_input_array(scales, 2);
  compute_encoder.set_input_array(quats, 3);
  compute_encoder.set_input_array(viewmat, 4);
  compute_encoder.set_input_array(projmat, 5);
  compute_encoder.set_input_array(cam_pos, 6);
  compute_encoder.set_input_array(sh, 7);
  compute_encoder.set_input_array(d_cov3d, 8);
  compute_encoder.set_input_array(d_rgb, 9);
  compute_encoder.set_input_array(d_xys, 10);
  compute_encoder.set_input_array(d_depths, 11);
  compute_encoder.set_input_array(d_conic_opacity, 12);
  compute_encoder.set_input_array(d_viewspace, 13);
  compute_encoder.set_input_array(clamped, 14);
  compute_encoder.set_output_array(d_means3d, 15);
  compute_encoder.set_output_array(d_dc, 16);
  compute_encoder.set_output_array(d_sh, 17);
  compute_encoder.set_output_array(d_colors_precomp, 18);
  compute_encoder.set_output_array(d_opacities, 19);
  compute_encoder.set_output_array(d_scales, 20);
  compute_encoder.set_output_array(d_quats, 21);
  compute_encoder.set_output_array(d_viewspace_in, 22);

  const size_t max_threads = kernel->maxTotalThreadsPerThreadgroup();
  size_t tgp_size = std::min(static_cast<size_t>(n), max_threads);
  if (tgp_size == 0) {
    return;
  }
  MTL::Size group_size = MTL::Size(tgp_size, 1, 1);
  MTL::Size grid_size = MTL::Size(n, 1, 1);
  compute_encoder.dispatch_threads(grid_size, group_size);
#endif
}

std::vector<mx::array> FastGSPreprocessBackward::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("FastGSPreprocessBackward jvp is not implemented.");
}

std::vector<mx::array> FastGSPreprocessBackward::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
  throw std::runtime_error("FastGSPreprocessBackward vjp is not implemented.");
}

std::pair<std::vector<mx::array>, std::vector<int>> FastGSPreprocessBackward::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("FastGSPreprocessBackward vmap is not implemented.");
}

bool FastGSPreprocessBackward::is_equivalent(const mx::Primitive& other) const {
  if (name() != other.name()) {
    return false;
  }
  auto other_ptr = dynamic_cast<const FastGSPreprocessBackward*>(&other);
  if (!other_ptr) {
    return false;
  }
  const auto& p = params_;
  const auto& q = other_ptr->params_;
  return p.degree == q.degree && p.max_sh_coeffs == q.max_sh_coeffs &&
         p.scale_modifier == q.scale_modifier && p.tan_fovx == q.tan_fovx &&
         p.tan_fovy == q.tan_fovy && p.image_height == q.image_height &&
         p.image_width == q.image_width && p.tile_bounds == q.tile_bounds &&
         p.mult == q.mult && p.prefiltered == q.prefiltered &&
         p.use_cov3d_precomp == q.use_cov3d_precomp &&
         p.use_colors_precomp == q.use_colors_precomp;
}

}  // namespace fastgs_core
