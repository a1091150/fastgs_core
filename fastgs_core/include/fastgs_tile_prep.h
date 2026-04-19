#pragma once

#include <vector>

#include "mlx/mlx.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace fastgs_core {
namespace mx = mlx::core;

struct TilePrepParams {
  int num_rendered;
  int num_tiles;
};

struct TilePrepInput {
  mx::array point_list_keys;
  mx::StreamOrDevice s;
  TilePrepParams params;
};

enum TilePrepOutputIndex {
  kRanges = 0,
  kBucketCount = 1,
};

std::vector<mx::array> fastgs_tile_prep(const TilePrepInput& input);

class FastGSTilePrep : public mx::Primitive {
 public:
  FastGSTilePrep(mx::Stream stream, TilePrepParams params)
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

  const char* name() const override { return "FastGSTilePrep"; }

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  TilePrepParams params_;
};

}  // namespace fastgs_core
