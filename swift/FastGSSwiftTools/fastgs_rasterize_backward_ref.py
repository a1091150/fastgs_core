#!/usr/bin/env python3

import argparse
import json
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


def summarize(array, sample_ids):
    values = np.array(array, dtype=np.float32).reshape(-1)
    safe_ids = [int(i) for i in sample_ids if int(i) < values.size]
    return {
        "shape": list(np.array(array).shape),
        "sum": float(values.sum(dtype=np.float64)),
        "absSum": float(np.abs(values).sum(dtype=np.float64)),
        "maxAbs": float(np.abs(values).max(initial=0.0)),
        "samples": values[safe_ids].astype(float).tolist(),
        "sampleIds": safe_ids,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate rasterize backward reference summary for FastGSSwift.")
    parser.add_argument("--out", default="/private/tmp/fastgs_rasterize_backward_ref.json")
    args = parser.parse_args()

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
    metric_map = mx.zeros((image_width * image_height,), dtype=mx.int32)
    metric_count = mx.zeros((n,), dtype=mx.int32)

    def loss_fn(means2d, colors_in, conic_in, viewspace_in):
        out = ext.rasterize_forward(
            tile["ranges"],
            point_list,
            bucket_offsets,
            means2d,
            colors_in,
            conic_in,
            background,
            pre["radii"],
            metric_map,
            metric_count,
            viewspace_in,
            image_width,
            image_height,
            block_x,
            block_y,
            3,
            num_tiles,
            bucket_sum,
            False,
        )
        return mx.sum(out["out_color"])

    value, grads = mx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3))(
        pre["xys"],
        pre["rgb"],
        pre["conic_opacity"],
        pre["viewspace_points"],
    )
    d_means2d, d_colors, d_conic, d_viewspace = grads
    mx.eval(value, d_means2d, d_colors, d_conic, d_viewspace)

    sample_ids = {
        "means2D": [0, 1, 4, 5, 8, 9],
        "colors": [0, 1, 2, 6, 7, 8, 12, 13, 14],
        "conicOpacity": [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19],
        "viewspacePoints": [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19],
    }

    report = {
        "imageWidth": image_width,
        "imageHeight": image_height,
        "blockX": block_x,
        "blockY": block_y,
        "numTiles": num_tiles,
        "numRendered": num_rendered,
        "bucketSum": bucket_sum,
        "loss": float(value.item()),
        "gradients": {
            "means2D": summarize(d_means2d, sample_ids["means2D"]),
            "colors": summarize(d_colors, sample_ids["colors"]),
            "conicOpacity": summarize(d_conic, sample_ids["conicOpacity"]),
            "viewspacePoints": summarize(d_viewspace, sample_ids["viewspacePoints"]),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
