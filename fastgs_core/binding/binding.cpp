#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>

#include <unordered_map>

#include <mlx/array.h>
#include <mlx/mlx.h>
#include <mlx/ops.h>

#include "../include/dummy.h"
#include "../include/fastgs_preprocess.h"
#include "../include/fastgs_binning.h"
#include "../include/fastgs_tile_prep.h"
#include "../include/fastgs_rasterize.h"

namespace nb = nanobind;
namespace mx = mlx::core;
using namespace nb::literals;

namespace {




nb::dict rasterize_forward(
    const mx::array& ranges,
    const mx::array& point_list,
    const mx::array& per_tile_bucket_offset,
    const mx::array& means2d,
    const mx::array& colors,
    const mx::array& conic_opacity,
    const mx::array& background,
    const mx::array& radii,
    const mx::array& metric_map,
    const mx::array& metric_count,
    const mx::array& viewspace_points,
    int image_width,
    int image_height,
    int block_x,
    int block_y,
    int num_channels,
    int num_tiles,
    int bucket_sum,
    bool get_flag) {
  fastgs_core::RasterizeInput input = {
      .ranges = ranges,
      .point_list = point_list,
      .per_tile_bucket_offset = per_tile_bucket_offset,
      .means2d = means2d,
      .colors = colors,
      .conic_opacity = conic_opacity,
      .background = background,
      .radii = radii,
      .metric_map = metric_map,
      .metric_count = metric_count,
      .viewspace_points = viewspace_points,
      .s = mx::Device::gpu,
      .params = {
          .image_width = image_width,
          .image_height = image_height,
          .block_x = block_x,
          .block_y = block_y,
          .num_channels = num_channels,
          .num_tiles = num_tiles,
          .bucket_sum = bucket_sum,
          .get_flag = get_flag,
      },
  };

  auto outputs = fastgs_core::fastgs_rasterize(input);
  nb::dict result;
  result["bucket_to_tile"] = outputs[fastgs_core::RasterizeOutputIndex::kBucketToTile];
  result["sampled_t"] = outputs[fastgs_core::RasterizeOutputIndex::kSampledT];
  result["sampled_ar"] = outputs[fastgs_core::RasterizeOutputIndex::kSampledAr];
  result["final_t"] = outputs[fastgs_core::RasterizeOutputIndex::kFinalT];
  result["n_contrib"] = outputs[fastgs_core::RasterizeOutputIndex::kNContrib];
  result["max_contrib"] = outputs[fastgs_core::RasterizeOutputIndex::kMaxContrib];
  result["pixel_colors"] = outputs[fastgs_core::RasterizeOutputIndex::kPixelColors];
  result["out_color"] = outputs[fastgs_core::RasterizeOutputIndex::kOutColor];
  result["metric_count"] = outputs[fastgs_core::RasterizeOutputIndex::kMetricCount];
  return result;
}

nb::dict binning_forward(
    const mx::array& xys,
    const mx::array& depths,
    const mx::array& point_offsets,
    const mx::array& conic_opacity,
    const mx::array& tiles_touched,
    float mult,
    int tile_bounds_x,
    int tile_bounds_y,
    int tile_bounds_z,
    int num_rendered) {
  fastgs_core::BinningInput input = {
      .xys = xys,
      .depths = depths,
      .point_offsets = point_offsets,
      .conic_opacity = conic_opacity,
      .tiles_touched = tiles_touched,
      .s = mx::Device::gpu,
      .params = {
          .mult = mult,
          .tile_bounds = std::make_tuple(tile_bounds_x, tile_bounds_y, tile_bounds_z),
          .num_rendered = num_rendered,
      },
  };

  auto outputs = fastgs_core::fastgs_binning(input);
  nb::dict result;
  result["point_list_keys_unsorted"] = outputs[fastgs_core::BinningOutputIndex::kPointListKeysUnsorted];
  result["point_list_unsorted"] = outputs[fastgs_core::BinningOutputIndex::kPointListUnsorted];
  return result;
}

nb::dict tile_prep_forward(
    const mx::array& point_list_keys,
    int num_rendered,
    int num_tiles) {
  fastgs_core::TilePrepInput input = {
      .point_list_keys = point_list_keys,
      .s = mx::Device::gpu,
      .params = {
          .num_rendered = num_rendered,
          .num_tiles = num_tiles,
      },
  };

  auto outputs = fastgs_core::fastgs_tile_prep(input);
  nb::dict result;
  result["ranges"] = outputs[fastgs_core::TilePrepOutputIndex::kRanges];
  result["bucket_count"] = outputs[fastgs_core::TilePrepOutputIndex::kBucketCount];
  return result;
}

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
      "rasterize_forward",
      &rasterize_forward,
      "ranges"_a,
      "point_list"_a,
      "per_tile_bucket_offset"_a,
      "means2d"_a,
      "colors"_a,
      "conic_opacity"_a,
      "background"_a,
      "radii"_a,
      "metric_map"_a,
      "metric_count"_a,
      "viewspace_points"_a,
      "image_width"_a,
      "image_height"_a,
      "block_x"_a,
      "block_y"_a,
      "num_channels"_a,
      "num_tiles"_a,
      "bucket_sum"_a,
      "get_flag"_a = false);

  m.def(
      "binning_forward",
      &binning_forward,
      "xys"_a,
      "depths"_a,
      "point_offsets"_a,
      "conic_opacity"_a,
      "tiles_touched"_a,
      "mult"_a,
      "tile_bounds_x"_a,
      "tile_bounds_y"_a,
      "tile_bounds_z"_a,
      "num_rendered"_a);

  m.def(
      "tile_prep_forward",
      &tile_prep_forward,
      "point_list_keys"_a,
      "num_rendered"_a,
      "num_tiles"_a);

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
