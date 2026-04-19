#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "mlx/mlx.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace fastgs_core {
namespace mx = mlx::core;

struct RasterizeParams {
  int image_width;
  int image_height;
  int block_x;
  int block_y;
  int num_channels;
  int num_tiles;
  int bucket_sum;
  bool get_flag;
};

struct RasterizeInput {
  mx::array ranges;
  mx::array point_list;
  mx::array per_tile_bucket_offset;
  mx::array means2d;
  mx::array colors;
  mx::array conic_opacity;
  mx::array background;
  mx::array radii;
  mx::array metric_map;
  mx::array metric_count;
  mx::array viewspace_points;
  mx::StreamOrDevice s;
  RasterizeParams params;
};

enum RasterizeOutputIndex {
  kBucketToTile = 0,
  kSampledT = 1,
  kSampledAr = 2,
  kFinalT = 3,
  kNContrib = 4,
  kMaxContrib = 5,
  kPixelColors = 6,
  kOutColor = 7,
  kMetricCount = 8,
};

std::vector<mx::array> fastgs_rasterize(const RasterizeInput& input);

class FastGSRasterize : public mx::Primitive {
 public:
  FastGSRasterize(mx::Stream stream, RasterizeParams params)
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

  const char* name() const override { return "FastGSRasterize"; }

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  RasterizeParams params_;
};

}  // namespace fastgs_core
