#pragma once

#include <tuple>
#include <vector>

#include "mlx/mlx.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace fastgs_core {
namespace mx = mlx::core;

struct PreprocessParams {
  int degree;
  int max_sh_coeffs;
  float scale_modifier;
  float tan_fovx;
  float tan_fovy;
  int image_height;
  int image_width;
  std::tuple<int, int, int> tile_bounds;
  float mult;
  bool prefiltered;
  bool use_cov3d_precomp;
  bool use_colors_precomp;
};

struct PreprocessInput {
  mx::array means3d;
  mx::array dc;
  mx::array sh;
  mx::array colors_precomp;
  mx::array opacities;
  mx::array scales;
  mx::array quats;
  mx::array cov3d_precomp;
  mx::array viewmat;
  mx::array projmat;
  mx::array cam_pos;
  mx::array viewspace_points;
  mx::StreamOrDevice s;
  PreprocessParams params;
};

enum PreprocessOutputIndex {
  kRadii = 0,
  kXys = 1,
  kDepths = 2,
  kCov3d = 3,
  kRgb = 4,
  kConicOpacity = 5,
  kTilesTouched = 6,
  kClamped = 7,
  kViewspacePoints = 8,
};

std::vector<mx::array> fastgs_preprocess(const PreprocessInput& input);

class FastGSPreprocess : public mx::Primitive {
 public:
  FastGSPreprocess(mx::Stream stream, PreprocessParams params)
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

  const char* name() const override { return "FastGSPreprocess"; }

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  PreprocessParams params_;
};

}  // namespace fastgs_core
