#pragma once

#include <tuple>
#include <vector>

#include "mlx/mlx.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace fastgs_core {
namespace mx = mlx::core;

struct BinningParams {
  float mult;
  std::tuple<int, int, int> tile_bounds;
  int num_rendered;
};

struct BinningInput {
  mx::array xys;
  mx::array depths;
  mx::array point_offsets;
  mx::array conic_opacity;
  mx::array tiles_touched;
  mx::StreamOrDevice s;
  BinningParams params;
};

enum BinningOutputIndex {
  kPointListKeysUnsorted = 0,
  kPointListUnsorted = 1,
};

std::vector<mx::array> fastgs_binning(const BinningInput& input);

class FastGSBinning : public mx::Primitive {
 public:
  FastGSBinning(mx::Stream stream, BinningParams params)
      : mx::Primitive(stream), params_(params) {}

  void eval_cpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;
  void eval_gpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;

  std::vector<mx::array> jvp(const std::vector<mx::array>& primals,
                             const std::vector<mx::array>& tangents,
                             const std::vector<int>& argnums) override;

  std::vector<mx::array> vjp(const std::vector<mx::array>& primals,
                             const std::vector<mx::array>& cotangents,
                             const std::vector<int>& argnums,
                             const std::vector<mx::array>& outputs) override;

  std::pair<std::vector<mx::array>, std::vector<int>> vmap(
      const std::vector<mx::array>& inputs,
      const std::vector<int>& axes) override;

  const char* name() const override { return "FastGSBinning"; }

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  BinningParams params_;
};

}  // namespace fastgs_core
