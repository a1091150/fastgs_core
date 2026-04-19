#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>

#include <unordered_map>

#include <mlx/array.h>
#include <mlx/mlx.h>
#include <mlx/ops.h>

#include "../include/dummy.h"
#include "../include/fastgs_preprocess.h"

namespace nb = nanobind;
namespace mx = mlx::core;
using namespace nb::literals;

namespace {

nb::dict preprocess_forward(
    const std::unordered_map<std::string, mx::array>& inputs,
    int image_width,
    int image_height,
    int block_x,
    int block_y,
    float tan_fovx,
    float tan_fovy,
    int degree,
    float scale_modifier,
    float mult,
    bool prefiltered) {
  auto required = {
      inputs.find("means3d"),
      inputs.find("dc"),
      inputs.find("sh"),
      inputs.find("colors_precomp"),
      inputs.find("opacities"),
      inputs.find("scales"),
      inputs.find("quats"),
      inputs.find("cov3d_precomp"),
      inputs.find("viewmat"),
      inputs.find("projmat"),
      inputs.find("cam_pos"),
      inputs.find("viewspace_points"),
  };
  for (auto it : required) {
    if (it == inputs.end()) {
      throw std::runtime_error("preprocess_forward missing required input tensor");
    }
  }

  const auto& means3d = inputs.at("means3d");
  const auto& dc = inputs.at("dc");
  const auto& sh = inputs.at("sh");
  const auto& colors_precomp = inputs.at("colors_precomp");
  const auto& opacities = inputs.at("opacities");
  const auto& scales = inputs.at("scales");
  const auto& quats = inputs.at("quats");
  const auto& cov3d_precomp = inputs.at("cov3d_precomp");
  const auto& viewmat = inputs.at("viewmat");
  const auto& projmat = inputs.at("projmat");
  const auto& cam_pos = inputs.at("cam_pos");
  const auto& viewspace_points = inputs.at("viewspace_points");

  const int max_sh_coeffs = (sh.ndim() >= 2) ? static_cast<int>(sh.shape(1)) : 0;

  fastgs_core::PreprocessParams preprocess_params = {
      .degree = degree,
      .max_sh_coeffs = max_sh_coeffs,
      .scale_modifier = scale_modifier,
      .tan_fovx = tan_fovx,
      .tan_fovy = tan_fovy,
      .image_height = image_height,
      .image_width = image_width,
      .tile_bounds = std::make_tuple(
          (image_width + block_x - 1) / block_x,
          (image_height + block_y - 1) / block_y,
          1),
      .mult = mult,
      .prefiltered = prefiltered,
      .use_cov3d_precomp = cov3d_precomp.size() != 0,
      .use_colors_precomp = colors_precomp.size() != 0,
  };

  fastgs_core::PreprocessInput preprocess_input = {
      .means3d = means3d,
      .dc = dc,
      .sh = sh,
      .colors_precomp = colors_precomp,
      .opacities = opacities,
      .scales = scales,
      .quats = quats,
      .cov3d_precomp = cov3d_precomp,
      .viewmat = viewmat,
      .projmat = projmat,
      .cam_pos = cam_pos,
      .viewspace_points = viewspace_points,
      .s = mx::Device::gpu,
      .params = preprocess_params,
  };

  auto outputs = fastgs_core::fastgs_preprocess(preprocess_input);

  nb::dict result;
  result["radii"] = outputs[fastgs_core::PreprocessOutputIndex::kRadii];
  result["xys"] = outputs[fastgs_core::PreprocessOutputIndex::kXys];
  result["depths"] = outputs[fastgs_core::PreprocessOutputIndex::kDepths];
  result["cov3d"] = outputs[fastgs_core::PreprocessOutputIndex::kCov3d];
  result["rgb"] = outputs[fastgs_core::PreprocessOutputIndex::kRgb];
  result["conic_opacity"] =
      outputs[fastgs_core::PreprocessOutputIndex::kConicOpacity];
  result["tiles_touched"] =
      outputs[fastgs_core::PreprocessOutputIndex::kTilesTouched];
  result["clamped"] = outputs[fastgs_core::PreprocessOutputIndex::kClamped];
  result["viewspace_points"] =
      outputs[fastgs_core::PreprocessOutputIndex::kViewspacePoints];
  return result;
}

}  // namespace

NB_MODULE(_fastgs_core, m) {
  nb::module_::import_("mlx.core");
  m.def("dummy_add", &fastgs_core::dummy_add, "a"_a, "b"_a);
  m.def(
      "dummy_array_size",
      [](const mx::array& a) { return static_cast<int>(a.size()); },
      "a"_a);

  m.def(
      "preprocess_forward",
      &preprocess_forward,
      "inputs"_a,
      "image_width"_a,
      "image_height"_a,
      "block_x"_a = 16,
      "block_y"_a = 16,
      "tan_fovx"_a = 1.0f,
      "tan_fovy"_a = 1.0f,
      "degree"_a = 0,
      "scale_modifier"_a = 1.0f,
      "mult"_a = 1.0f,
      "prefiltered"_a = false);
}
