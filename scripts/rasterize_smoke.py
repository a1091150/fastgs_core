#!/usr/bin/env python3

import os
import sys

import mlx.core as mx


def import_extension():
    try:
        from fastgs_core import _fastgs_core as ext
        return ext
    except Exception:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.insert(0, os.path.join(repo_root, "build"))
        import _fastgs_core as ext
        return ext


def main() -> None:
    ext = import_extension()

    image_width = 16
    image_height = 16
    block_x = 16
    block_y = 16
    num_channels = 3
    num_tiles = 1
    bucket_sum = 1

    n = 1
    ranges = mx.array([[0, 1]], dtype=mx.uint32)
    point_list = mx.array([0], dtype=mx.uint32)
    per_tile_bucket_offset = mx.array([1], dtype=mx.uint32)
    means2d = mx.array([[8.0, 8.0]], dtype=mx.float32)
    colors = mx.array([[1.0, 0.0, 0.0]], dtype=mx.float32)
    conic_opacity = mx.array([[1.0, 0.0, 1.0, 0.9]], dtype=mx.float32)
    background = mx.array([0.0, 0.0, 0.0], dtype=mx.float32)
    radii = mx.array([1], dtype=mx.int32)
    metric_map = mx.zeros((image_width * image_height,), dtype=mx.int32)
    metric_count = mx.zeros((n,), dtype=mx.int32)
    viewspace_points = mx.zeros((n, 4), dtype=mx.float32)

    out = ext.rasterize_forward(
        ranges,
        point_list,
        per_tile_bucket_offset,
        means2d,
        colors,
        conic_opacity,
        background,
        radii,
        metric_map,
        metric_count,
        viewspace_points,
        image_width,
        image_height,
        block_x,
        block_y,
        num_channels,
        num_tiles,
        bucket_sum,
        False,
    )

    mx.eval(out["out_color"], out["pixel_colors"], out["final_t"])

    print("rasterize_forward smoke ok")
    print("out_color shape:", out["out_color"].shape)
    print("pixel_colors shape:", out["pixel_colors"].shape)
    print("final_t shape:", out["final_t"].shape)


if __name__ == "__main__":
    main()
