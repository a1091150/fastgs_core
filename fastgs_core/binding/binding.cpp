#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <mlx/mlx.h>
#include <mlx/ops.h>
#include <mlx/array.h>
#include "../include/dummy.h"
#include "../include/fastgs_preprocess.h"

namespace nb = nanobind;
namespace mx = mlx::core;
using namespace nb::literals;

namespace
{

  nb::dict preprocess_forward(
      const mx::array &means3d,
      const mx::array &dc,
      const mx::array &sh,
      const mx::array &colors_precomp,
      const mx::array &opacities,
      const mx::array &scales,
      const mx::array &quats,
      const mx::array &cov3d_precomp,
      const mx::array &viewmat,
      const mx::array &projmat,
      const mx::array &cam_pos,
      const mx::array &viewspace_points,
      int image_width,
      int image_height,
      int block_x,
      int block_y,
      float tan_fovx,
      float tan_fovy,
      int degree,
      float scale_modifier,
      float mult,
      bool prefiltered)
  {
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

} // namespace

NB_MODULE(_fastgs_core, m)
{
  nb::module_::import_("mlx.core");
  m.def("dummy_add", &fastgs_core::dummy_add, "a"_a, "b"_a);
  m.def("dummy_array_size_obj", [](nb::object a)
        { return nb::cast<int>(a.attr("size")); });
  m.def(
      "dummy_array_size",
      [](const mx::array &a)
      { return static_cast<int>(a.size()); },
      "a"_a);
  m.def(
      "preprocess_forward",
      &preprocess_forward,
      "means3d"_a,
      "dc"_a,
      "sh"_a,
      "colors_precomp"_a,
      "opacities"_a,
      "scales"_a,
      "quats"_a,
      "cov3d_precomp"_a,
      "viewmat"_a,
      "projmat"_a,
      "cam_pos"_a,
      "viewspace_points"_a,
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
