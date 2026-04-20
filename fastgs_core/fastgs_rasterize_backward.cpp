#include "include/fastgs_rasterize.h"

#include <cstring>
#include <stdexcept>

namespace fastgs_core {

namespace {
constexpr int kRasterizeNumInputs = 10;
constexpr int kRasterizeNumOutputs = 9;
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

void FastGSRasterizeBackward::eval_gpu(const std::vector<mx::array>&,
                                       std::vector<mx::array>& outputs) {
  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    std::memset(out.data<void>(), 0, out.nbytes());
  }
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
