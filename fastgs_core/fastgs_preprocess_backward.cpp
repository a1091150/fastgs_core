#include "include/fastgs_preprocess.h"

#include <cstring>
#include <stdexcept>

namespace fastgs_core {

namespace {
constexpr int kPreprocessNumInputs = 12;
constexpr int kPreprocessNumOutputs = 9;
}  // namespace

std::vector<mx::array> fastgs_preprocess_backward(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& cotangents,
    const PreprocessParams& params,
    mx::StreamOrDevice s) {
  if (primals.size() != kPreprocessNumInputs) {
    throw std::invalid_argument("fastgs_preprocess_backward expects 12 primals.");
  }
  if (cotangents.size() != kPreprocessNumOutputs) {
    throw std::invalid_argument("fastgs_preprocess_backward expects 9 cotangents.");
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
  inputs.reserve(primals.size() + cotangents.size());
  for (const auto& primal : primals) {
    inputs.push_back(mx::contiguous(primal));
  }
  for (const auto& cotangent : cotangents) {
    inputs.push_back(mx::contiguous(cotangent));
  }

  return mx::array::make_arrays(output_shapes, output_types, prim, inputs);
}

void FastGSPreprocessBackward::eval_cpu(const std::vector<mx::array>&,
                                        std::vector<mx::array>& outputs) {
  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    std::memset(out.data<void>(), 0, out.nbytes());
  }
}

void FastGSPreprocessBackward::eval_gpu(const std::vector<mx::array>&,
                                        std::vector<mx::array>& outputs) {
  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    std::memset(out.data<void>(), 0, out.nbytes());
  }
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
  return name() == other.name();
}

}  // namespace fastgs_core
