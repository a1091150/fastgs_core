#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>

#include <cstdint>
#include <string>
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

uint32_t read_last_u32(const mx::array& a) {
  mx::eval(a);
  if (a.size() == 0) {
    return 0;
  }
  return a.data<uint32_t>()[a.size() - 1];
}

nb::dict rasterize_gaussians_forward(
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
    bool prefiltered,
    bool get_flag) {
  auto require_key = [&](const char* key) -> const mx::array& {
    auto it = inputs.find(key);
    if (it == inputs.end()) {
      throw std::runtime_error(std::string("rasterize_gaussians_forward missing required input tensor: ") + key);
    }
    return it->second;
  };
  auto get_or_empty = [&](const char* key) -> mx::array {
    auto it = inputs.find(key);
    if (it == inputs.end()) {
      return mx::zeros({0}, mx::float32);
    }
    return it->second;
  };

  const auto& means3d = require_key("means3d");
  const auto& background = require_key("background");
  const auto& opacities = require_key("opacities");
  const auto& metric_map = require_key("metric_map");
  const auto& viewmatrix = require_key("viewmatrix");
  const auto& projmatrix = require_key("projmatrix");
  const auto& campos = require_key("campos");

  mx::array colors_precomp = get_or_empty("colors_precomp");
  if (colors_precomp.size() == 0) {
    colors_precomp = get_or_empty("colors");
  }
  const bool has_dc_input = inputs.find("dc") != inputs.end();
  const bool has_sh_input = inputs.find("sh") != inputs.end();
  mx::array dc = has_dc_input ? inputs.at("dc") : mx::zeros({0}, mx::float32);
  mx::array sh = has_sh_input ? inputs.at("sh") : mx::zeros({0, 0, 3}, mx::float32);
  const bool has_colors_precomp = colors_precomp.size() != 0;
  const bool has_sh_path = has_dc_input && has_sh_input;
  if (has_colors_precomp == has_sh_path) {
    throw std::runtime_error(
        "rasterize_gaussians_forward expects exactly one color path: "
        "either colors_precomp (or colors) OR dc+sh");
  }

  mx::array cov3d_precomp = get_or_empty("cov3d_precomp");
  mx::array scales = get_or_empty("scales");
  mx::array rotations = get_or_empty("rotations");
  const bool has_cov3d_precomp = cov3d_precomp.size() != 0;
  const bool has_scale_rot = (scales.size() != 0) && (rotations.size() != 0);
  if (has_cov3d_precomp == has_scale_rot) {
    throw std::runtime_error(
        "rasterize_gaussians_forward expects exactly one geometry path: "
        "either cov3d_precomp OR scales+rotations");
  }

  mx::array viewspace_points = mx::zeros({0}, means3d.dtype());
  if (auto it = inputs.find("viewspace_points"); it != inputs.end()) {
    viewspace_points = it->second;
  }

  const int num_points = means3d.shape(0);
  if (viewspace_points.size() == 0) {
    viewspace_points = mx::zeros({num_points, 4}, means3d.dtype(), mx::Device::gpu);
  }
  const int max_sh_coeffs = (sh.ndim() >= 2) ? static_cast<int>(sh.shape(1)) : 0;
  auto tile_bounds = std::make_tuple(
      (image_width + block_x - 1) / block_x,
      (image_height + block_y - 1) / block_y,
      1);

  fastgs_core::PreprocessParams preprocess_params = {
      .degree = degree,
      .max_sh_coeffs = max_sh_coeffs,
      .scale_modifier = scale_modifier,
      .tan_fovx = tan_fovx,
      .tan_fovy = tan_fovy,
      .image_height = image_height,
      .image_width = image_width,
      .tile_bounds = tile_bounds,
      .mult = mult,
      .prefiltered = prefiltered,
      .use_cov3d_precomp = has_cov3d_precomp,
      .use_colors_precomp = has_colors_precomp,
  };

  fastgs_core::PreprocessInput preprocess_input = {
      .means3d = means3d,
      .dc = dc,
      .sh = sh,
      .colors_precomp = colors_precomp,
      .opacities = opacities,
      .scales = scales,
      .quats = rotations,
      .cov3d_precomp = cov3d_precomp,
      .viewmat = viewmatrix,
      .projmat = projmatrix,
      .cam_pos = campos,
      .viewspace_points = viewspace_points,
      .s = mx::Device::gpu,
      .params = preprocess_params,
  };

  auto preprocess_outputs = fastgs_core::fastgs_preprocess(preprocess_input);
  auto xys = preprocess_outputs[fastgs_core::PreprocessOutputIndex::kXys];
  auto depths = preprocess_outputs[fastgs_core::PreprocessOutputIndex::kDepths];
  auto rgbs = preprocess_outputs[fastgs_core::PreprocessOutputIndex::kRgb];
  auto conic_opacity =
      preprocess_outputs[fastgs_core::PreprocessOutputIndex::kConicOpacity];
  auto tiles_touched =
      preprocess_outputs[fastgs_core::PreprocessOutputIndex::kTilesTouched];
  auto preprocess_viewspace_points =
      preprocess_outputs[fastgs_core::PreprocessOutputIndex::kViewspacePoints];
  auto point_offsets = mx::stop_gradient(mx::cumsum(tiles_touched, false, true));
  uint32_t num_rendered = read_last_u32(point_offsets);

  mx::array point_list_keys = mx::zeros({0}, mx::uint64);
  mx::array point_list = mx::zeros({0}, mx::uint32);
  mx::array ranges = mx::zeros({0, 2}, mx::uint32);
  mx::array bucket_count = mx::zeros({0}, mx::uint32);
  mx::array bucket_offsets = mx::zeros({0}, mx::uint32);
  mx::array bucket_to_tile = mx::zeros({0}, mx::uint32);
  mx::array sampled_t = mx::zeros({0}, mx::float32);
  mx::array sampled_ar = mx::zeros({0}, mx::float32);
  mx::array final_t = mx::zeros({0}, mx::float32);
  mx::array n_contrib = mx::zeros({0}, mx::uint32);
  mx::array max_contrib = mx::zeros({0}, mx::uint32);
  mx::array pixel_colors = mx::zeros({0}, mx::float32);
  mx::array out_color = mx::zeros({0}, mx::float32);
  mx::array metric_count =
      get_flag ? mx::zeros({num_points}, mx::int32) : mx::zeros({0}, mx::int32);

  const int num_tiles = std::get<0>(tile_bounds) * std::get<1>(tile_bounds);
  uint32_t bucket_sum = 0;

  if (num_rendered > 0) {
    fastgs_core::BinningInput binning_input = {
        .xys = mx::stop_gradient(xys),
        .depths = mx::stop_gradient(depths),
        .point_offsets = point_offsets,
        .conic_opacity = mx::stop_gradient(conic_opacity),
        .tiles_touched = mx::stop_gradient(tiles_touched),
        .s = mx::Device::gpu,
        .params = {
            .mult = mult,
            .tile_bounds = tile_bounds,
            .num_rendered = static_cast<int>(num_rendered),
        },
    };
    auto binning_outputs = fastgs_core::fastgs_binning(binning_input);
    auto point_list_keys_unsorted = mx::stop_gradient(
        binning_outputs[fastgs_core::BinningOutputIndex::kPointListKeysUnsorted]);
    auto point_list_unsorted = mx::stop_gradient(
        binning_outputs[fastgs_core::BinningOutputIndex::kPointListUnsorted]);

    auto sorted_indices = mx::argsort(point_list_keys_unsorted);
    point_list_keys = mx::stop_gradient(mx::take(point_list_keys_unsorted, sorted_indices));
    point_list = mx::stop_gradient(mx::take(point_list_unsorted, sorted_indices));

    fastgs_core::TilePrepInput tile_input = {
        .point_list_keys = point_list_keys,
        .s = mx::Device::gpu,
        .params = {
            .num_rendered = static_cast<int>(num_rendered),
            .num_tiles = num_tiles,
        },
    };
    auto tile_outputs = fastgs_core::fastgs_tile_prep(tile_input);
    ranges = mx::stop_gradient(tile_outputs[fastgs_core::TilePrepOutputIndex::kRanges]);
    bucket_count =
        mx::stop_gradient(tile_outputs[fastgs_core::TilePrepOutputIndex::kBucketCount]);
    bucket_offsets = mx::stop_gradient(mx::cumsum(bucket_count, false, true));
    bucket_sum = read_last_u32(bucket_offsets);

    if (bucket_sum > 0) {
      fastgs_core::RasterizeInput rasterize_input = {
          .ranges = ranges,
          .point_list = point_list,
          .per_tile_bucket_offset = bucket_offsets,
          .means2d = xys,
          .colors = rgbs,
          .conic_opacity = conic_opacity,
          .background = background,
          .radii = mx::stop_gradient(
              preprocess_outputs[fastgs_core::PreprocessOutputIndex::kRadii]),
          .metric_map = mx::stop_gradient(metric_map),
          .metric_count = metric_count,
          .viewspace_points = preprocess_viewspace_points,
          .s = mx::Device::gpu,
          .params = {
              .image_width = image_width,
              .image_height = image_height,
              .block_x = block_x,
              .block_y = block_y,
              .num_channels = 3,
              .num_tiles = num_tiles,
              .bucket_sum = static_cast<int>(bucket_sum),
              .get_flag = get_flag,
          },
      };

      auto rasterize_outputs = fastgs_core::fastgs_rasterize(rasterize_input);
      bucket_to_tile = rasterize_outputs[fastgs_core::RasterizeOutputIndex::kBucketToTile];
      sampled_t = rasterize_outputs[fastgs_core::RasterizeOutputIndex::kSampledT];
      sampled_ar = rasterize_outputs[fastgs_core::RasterizeOutputIndex::kSampledAr];
      final_t = rasterize_outputs[fastgs_core::RasterizeOutputIndex::kFinalT];
      n_contrib = rasterize_outputs[fastgs_core::RasterizeOutputIndex::kNContrib];
      max_contrib = rasterize_outputs[fastgs_core::RasterizeOutputIndex::kMaxContrib];
      pixel_colors = rasterize_outputs[fastgs_core::RasterizeOutputIndex::kPixelColors];
      out_color = rasterize_outputs[fastgs_core::RasterizeOutputIndex::kOutColor];
      metric_count = rasterize_outputs[fastgs_core::RasterizeOutputIndex::kMetricCount];
    }
  }

  nb::dict result;
  result["rendered"] = nb::int_(num_rendered);
  result["num_buckets"] = nb::int_(bucket_sum);
  result["out_color"] = out_color;
  result["radii"] = preprocess_outputs[fastgs_core::PreprocessOutputIndex::kRadii];
  result["point_offsets"] = point_offsets;
  result["ranges"] = ranges;
  result["bucket_count"] = bucket_count;
  result["bucket_offsets"] = bucket_offsets;
  result["point_list_keys"] = point_list_keys;
  result["point_list"] = point_list;
  result["metric_count"] = metric_count;
  result["viewspace_points"] = preprocess_viewspace_points;
  result["geom_means2d"] = xys;
  result["geom_depths"] = depths;
  result["geom_cov3d"] = preprocess_outputs[fastgs_core::PreprocessOutputIndex::kCov3d];
  result["geom_rgb"] = rgbs;
  result["geom_conic_opacity"] = conic_opacity;
  result["geom_tiles_touched"] = tiles_touched;
  result["geom_clamped"] = preprocess_outputs[fastgs_core::PreprocessOutputIndex::kClamped];
  result["sample_bucket_to_tile"] = bucket_to_tile;
  result["sample_t"] = sampled_t;
  result["sample_ar"] = sampled_ar;
  result["img_final_t"] = final_t;
  result["img_n_contrib"] = n_contrib;
  result["img_max_contrib"] = max_contrib;
  result["img_pixel_colors"] = pixel_colors;
  return result;
}



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
  auto require_key = [&](const char* key) -> const mx::array& {
    auto it = inputs.find(key);
    if (it == inputs.end()) {
      throw std::runtime_error(std::string("preprocess_forward missing required input tensor: ") + key);
    }
    return it->second;
  };
  auto get_or_empty = [&](const char* key) -> mx::array {
    auto it = inputs.find(key);
    if (it == inputs.end()) {
      return mx::zeros({0}, mx::float32);
    }
    return it->second;
  };

  const auto& means3d = require_key("means3d");
  const auto& opacities = require_key("opacities");
  const auto& viewmat = require_key("viewmat");
  const auto& projmat = require_key("projmat");
  const auto& cam_pos = require_key("cam_pos");
  const auto& viewspace_points = require_key("viewspace_points");

  mx::array colors_precomp = get_or_empty("colors_precomp");
  const bool has_dc_input = inputs.find("dc") != inputs.end();
  const bool has_sh_input = inputs.find("sh") != inputs.end();
  mx::array dc = has_dc_input ? inputs.at("dc") : mx::zeros({0}, mx::float32);
  mx::array sh = has_sh_input ? inputs.at("sh") : mx::zeros({0, 0, 3}, mx::float32);
  const bool has_colors_precomp = colors_precomp.size() != 0;
  const bool has_sh_path = has_dc_input && has_sh_input;
  if (has_colors_precomp == has_sh_path) {
    throw std::runtime_error(
        "preprocess_forward expects exactly one color path: "
        "either colors_precomp OR dc+sh");
  }

  mx::array cov3d_precomp = get_or_empty("cov3d_precomp");
  mx::array scales = get_or_empty("scales");
  mx::array quats = get_or_empty("quats");
  const bool has_cov3d_precomp = cov3d_precomp.size() != 0;
  const bool has_scale_rot = (scales.size() != 0) && (quats.size() != 0);
  if (has_cov3d_precomp == has_scale_rot) {
    throw std::runtime_error(
        "preprocess_forward expects exactly one geometry path: "
        "either cov3d_precomp OR scales+quats");
  }

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
      .use_cov3d_precomp = has_cov3d_precomp,
      .use_colors_precomp = has_colors_precomp,
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

  m.def(
      "rasterize_gaussians_forward",
      &rasterize_gaussians_forward,
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
      "prefiltered"_a = false,
      "get_flag"_a = false);

  m.def(
      "rasterize_gaussians",
      &rasterize_gaussians_forward,
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
      "prefiltered"_a = false,
      "get_flag"_a = false);
}
