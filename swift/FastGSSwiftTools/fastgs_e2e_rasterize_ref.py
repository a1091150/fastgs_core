#!/usr/bin/env python3

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np


def import_extension():
    try:
        from fastgs_core import _fastgs_core as ext
        return ext
    except Exception:
        repo_root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repo_root / "build"))
        import _fastgs_core as ext
        return ext


def arr(x):
    return np.array(x)


def main():
    ext = import_extension()
    n = 2
    image_width = 64
    image_height = 64
    block_x = 16
    block_y = 16
    num_tiles = 16
    inputs = {
        "means3d": mx.array([[0.0, 0.0, 1.0], [0.25, -0.25, 1.0]], dtype=mx.float32),
        "colors_precomp": mx.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=mx.float32),
        "opacities": mx.array([1.0, 1.0], dtype=mx.float32),
        "scales": mx.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=mx.float32),
        "quats": mx.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=mx.float32),
        "viewmat": mx.eye(4, dtype=mx.float32),
        "projmat": mx.eye(4, dtype=mx.float32),
        "cam_pos": mx.array([0.0, 0.0, 0.0], dtype=mx.float32),
        "viewspace_points": mx.array([[0.0, 0.0, 0.0, 7.0], [0.0, 0.0, 0.0, 9.0]], dtype=mx.float32),
    }
    prep = ext.preprocess_forward(
        inputs, image_width, image_height, block_x, block_y, 1.0, 1.0, 0, 1.0, 1.0, False
    )
    point_offsets = mx.cumsum(prep["tiles_touched"], reverse=False, inclusive=True)
    mx.eval(point_offsets)
    num_rendered = int(arr(point_offsets)[-1])
    binning = ext.binning_forward(
        prep["xys"],
        prep["depths"],
        point_offsets,
        prep["conic_opacity"],
        prep["tiles_touched"],
        1.0,
        4,
        4,
        1,
        num_rendered,
    )
    sorted_indices = mx.argsort(binning["point_list_keys_unsorted"])
    point_list_keys = mx.take(binning["point_list_keys_unsorted"], sorted_indices)
    point_list = mx.take(binning["point_list_unsorted"], sorted_indices)
    tile = ext.tile_prep_forward(point_list_keys, num_rendered, num_tiles)
    bucket_offsets = mx.cumsum(tile["bucket_count"], reverse=False, inclusive=True)
    mx.eval(bucket_offsets)
    bucket_sum = int(arr(bucket_offsets)[-1])
    background = mx.array([0.1, 0.2, 0.3], dtype=mx.float32)
    metric_map = mx.zeros((image_width * image_height,), dtype=mx.int32)
    metric_count = mx.zeros((n,), dtype=mx.int32)
    rast = ext.rasterize_forward(
        tile["ranges"],
        point_list,
        bucket_offsets,
        prep["xys"],
        prep["rgb"],
        prep["conic_opacity"],
        background,
        prep["radii"],
        metric_map,
        metric_count,
        prep["viewspace_points"],
        image_width,
        image_height,
        block_x,
        block_y,
        3,
        num_tiles,
        bucket_sum,
        False,
    )
    mx.eval(
        rast["bucket_to_tile"],
        rast["sampled_t"],
        rast["sampled_ar"],
        rast["final_t"],
        rast["n_contrib"],
        rast["max_contrib"],
        rast["pixel_colors"],
        rast["out_color"],
        rast["metric_count"],
    )
    out = arr(rast["out_color"]).astype(np.float32)
    pix = arr(rast["pixel_colors"]).astype(np.float32)
    ft = arr(rast["final_t"]).astype(np.float32)
    nc = arr(rast["n_contrib"]).astype(np.uint32)
    mc = arr(rast["max_contrib"]).astype(np.uint32)
    btt = arr(rast["bucket_to_tile"]).astype(np.uint32)
    st = arr(rast["sampled_t"]).astype(np.float32)
    sar = arr(rast["sampled_ar"]).astype(np.float32)
    sample_ids = [0, 31 + 31 * image_width, 32 + 32 * image_width, 39 + 23 * image_width, 63 + 63 * image_width]
    print("num_rendered", num_rendered)
    print("bucket_sum", bucket_sum)
    print("out_sums", out.sum(axis=1).tolist())
    print("pixel_sums", pix.sum(axis=1).tolist())
    print("final_t_sum", float(ft.sum()))
    print("n_contrib_sum", int(nc.sum()))
    print("max_contrib", mc.tolist())
    print("bucket_to_tile", btt.tolist())
    print("sampled_t_prefix", st[:32].tolist())
    print("sampled_ar_prefix", sar[:12].tolist())
    print("sample_ids", sample_ids)
    print("out_samples", out[:, sample_ids].reshape(-1).tolist())
    print("pixel_samples", pix[:, sample_ids].reshape(-1).tolist())
    print("final_t_samples", ft[sample_ids].tolist())
    print("n_contrib_samples", nc[sample_ids].tolist())


if __name__ == "__main__":
    main()
