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


def main():
    ext = import_extension()

    image_width = 80
    image_height = 48
    block_x = 16
    block_y = 16
    tiles_x = (image_width + block_x - 1) // block_x
    tiles_y = (image_height + block_y - 1) // block_y
    num_tiles = tiles_x * tiles_y

    means = [
        [-0.62, -0.45, 1.0],
        [-0.25, 0.25, 1.15],
        [0.20, -0.10, 0.92],
        [0.55, 0.38, 1.25],
        [0.05, 0.62, 1.45],
    ]
    colors = [
        [1.0, 0.16, 0.08],
        [0.10, 0.72, 1.0],
        [1.0, 0.95, 0.18],
        [0.22, 1.0, 0.28],
        [0.88, 0.24, 1.0],
    ]
    opacities = [0.82, 0.76, 0.68, 0.72, 0.58]
    scales = [
        [0.18, 0.28, 0.16],
        [0.26, 0.18, 0.18],
        [0.20, 0.20, 0.20],
        [0.30, 0.16, 0.18],
        [0.18, 0.22, 0.24],
    ]
    quats = [[1.0, 0.0, 0.0, 0.0]] * len(means)
    n = len(means)

    inputs = {
        "means3d": mx.array(means, dtype=mx.float32),
        "colors_precomp": mx.array(colors, dtype=mx.float32),
        "opacities": mx.array(opacities, dtype=mx.float32),
        "scales": mx.array(scales, dtype=mx.float32),
        "quats": mx.array(quats, dtype=mx.float32),
        "cov3d_precomp": mx.zeros((0,), dtype=mx.float32),
        "viewmat": mx.eye(4, dtype=mx.float32),
        "projmat": mx.eye(4, dtype=mx.float32),
        "cam_pos": mx.zeros((3,), dtype=mx.float32),
        "viewspace_points": mx.zeros((n, 4), dtype=mx.float32),
    }

    pre = ext.preprocess_forward(
        inputs,
        image_width,
        image_height,
        block_x,
        block_y,
        1.0,
        1.0,
        0,
        1.0,
        1.0,
        False,
    )
    mx.eval(pre["tiles_touched"])
    point_offsets = mx.cumsum(pre["tiles_touched"])
    mx.eval(point_offsets)
    num_rendered = int(np.array(point_offsets)[-1])

    binning = ext.binning_forward(
        pre["xys"],
        pre["depths"],
        point_offsets,
        pre["conic_opacity"],
        pre["tiles_touched"],
        1.0,
        tiles_x,
        tiles_y,
        1,
        num_rendered,
    )
    sorted_indices = mx.argsort(binning["point_list_keys_unsorted"])
    point_list_keys = mx.take(binning["point_list_keys_unsorted"], sorted_indices)
    point_list = mx.take(binning["point_list_unsorted"], sorted_indices)
    tile = ext.tile_prep_forward(point_list_keys, num_rendered, num_tiles)
    bucket_offsets = mx.cumsum(tile["bucket_count"])
    mx.eval(bucket_offsets)
    bucket_sum = int(np.array(bucket_offsets)[-1])

    background = mx.array([0.025, 0.03, 0.04], dtype=mx.float32)
    out = ext.rasterize_forward(
        tile["ranges"],
        point_list,
        bucket_offsets,
        pre["xys"],
        pre["rgb"],
        pre["conic_opacity"],
        background,
        pre["radii"],
        mx.zeros((image_width * image_height,), dtype=mx.int32),
        mx.zeros((n,), dtype=mx.int32),
        pre["viewspace_points"],
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
        pre["radii"],
        pre["xys"],
        pre["depths"],
        pre["tiles_touched"],
        point_offsets,
        point_list_keys,
        point_list,
        tile["ranges"],
        tile["bucket_count"],
        bucket_offsets,
        out["bucket_to_tile"],
        out["sampled_t"],
        out["sampled_ar"],
        out["final_t"],
        out["n_contrib"],
        out["max_contrib"],
        out["pixel_colors"],
        out["out_color"],
        out["metric_count"],
    )

    out_color = np.array(out["out_color"]).reshape(3, image_height, image_width)
    rgb = np.clip(np.transpose(out_color, (1, 2, 0)), 0, 1)
    ppm = (rgb * 255 + 0.5).astype(np.uint8)
    path = "/private/tmp/fastgs_large_rasterize_ref.ppm"
    with open(path, "wb") as f:
        f.write(f"P6\n{image_width} {image_height}\n255\n".encode("ascii"))
        f.write(ppm.tobytes())

    sample_ids = [
        0,
        7 + 5 * image_width,
        24 + 14 * image_width,
        40 + 24 * image_width,
        63 + 32 * image_width,
        image_width * image_height - 1,
    ]

    def py(x):
        return np.array(x).tolist()

    print("image", image_width, image_height, "tiles", tiles_x, tiles_y, "num_tiles", num_tiles)
    print("num_rendered", num_rendered)
    print("bucket_sum", bucket_sum)
    print("radii", py(pre["radii"]))
    print("xy", py(pre["xys"].reshape(-1)))
    print("depths", py(pre["depths"]))
    print("tiles_touched", py(pre["tiles_touched"]))
    print("point_offsets", py(point_offsets))
    print("bucket_count", py(tile["bucket_count"]))
    print("bucket_offsets", py(bucket_offsets))
    print("ranges", py(tile["ranges"].reshape(-1)))
    print("bucket_to_tile_prefix", py(out["bucket_to_tile"][:bucket_sum]))
    print("max_contrib", py(out["max_contrib"]))
    print("out_sums", np.array(out["out_color"]).reshape(3, -1).sum(axis=1).tolist())
    print("pixel_sums", np.array(out["pixel_colors"]).reshape(3, -1).sum(axis=1).tolist())
    print("final_t_sum", float(np.array(out["final_t"]).sum()))
    print("n_contrib_sum", int(np.array(out["n_contrib"]).sum()))
    print("sample_ids", sample_ids)
    print("out_samples", np.array(out["out_color"]).reshape(3, -1)[:, sample_ids].reshape(-1).tolist())
    print("pixel_samples", np.array(out["pixel_colors"]).reshape(3, -1)[:, sample_ids].reshape(-1).tolist())
    print("final_t_samples", np.array(out["final_t"])[sample_ids].tolist())
    print("n_contrib_samples", np.array(out["n_contrib"])[sample_ids].tolist())
    print("sampled_t_prefix", py(out["sampled_t"][:48]))
    print("sampled_ar_prefix", py(out["sampled_ar"][:24]))
    print("ppm", path)


if __name__ == "__main__":
    main()
