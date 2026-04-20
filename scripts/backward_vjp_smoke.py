#!/usr/bin/env python3

import os
import sys
import traceback

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


def run_preprocess_vjp_smoke(ext) -> None:
    n = 8
    image_width = 16
    image_height = 16

    base_inputs = {
        "means3d": mx.zeros((n, 3), dtype=mx.float32, stream=mx.gpu),
        "opacities": mx.ones((n, 1), dtype=mx.float32, stream=mx.gpu),
        "cov3d_precomp": mx.ones((n, 6), dtype=mx.float32, stream=mx.gpu),
        "colors_precomp": mx.ones((n, 3), dtype=mx.float32, stream=mx.gpu),
        "viewmat": mx.eye(4, dtype=mx.float32, stream=mx.gpu),
        "projmat": mx.eye(4, dtype=mx.float32, stream=mx.gpu),
        "cam_pos": mx.zeros((3,), dtype=mx.float32, stream=mx.gpu),
        "viewspace_points": mx.zeros((n, 4), dtype=mx.float32, stream=mx.gpu),
    }

    def loss_fn(means3d):
        local = dict(base_inputs)
        local["means3d"] = means3d
        out = ext.preprocess_forward(
            local,
            image_width=image_width,
            image_height=image_height,
            block_x=16,
            block_y=16,
            tan_fovx=1.0,
            tan_fovy=1.0,
            degree=0,
            scale_modifier=1.0,
            mult=1.0,
            prefiltered=False,
        )
        return mx.sum(out["xys"])

    g = mx.grad(loss_fn)(base_inputs["means3d"])
    mx.eval(g)
    print("[PASS] preprocess vjp smoke")
    print("  grad shape:", g.shape, "dtype:", g.dtype)


def run_rasterize_vjp_smoke(ext) -> None:
    n = 1
    image_width = 16
    image_height = 16
    num_tiles = 1
    bucket_sum = 1

    ranges = mx.array([[0, 1]], dtype=mx.uint32)
    point_list = mx.array([0], dtype=mx.uint32)
    per_tile_bucket_offset = mx.array([1], dtype=mx.uint32)
    colors = mx.array([[1.0, 1.0, 1.0]], dtype=mx.float32)
    conic_opacity = mx.array([[1.0, 0.0, 1.0, 0.5]], dtype=mx.float32)
    background = mx.zeros((3,), dtype=mx.float32, stream=mx.gpu)
    radii = mx.array([1], dtype=mx.int32)
    metric_map = mx.zeros((image_width * image_height,), dtype=mx.int32, stream=mx.gpu)
    metric_count = mx.zeros((n,), dtype=mx.int32, stream=mx.gpu)
    viewspace_points = mx.zeros((n, 4), dtype=mx.float32, stream=mx.gpu)

    means2d_init = mx.array([[8.0, 8.0]], dtype=mx.float32)

    def loss_fn(means2d):
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
            16,
            16,
            3,
            num_tiles,
            bucket_sum,
            False,
        )
        return mx.sum(out["out_color"])

    g = mx.grad(loss_fn)(means2d_init)
    mx.eval(g)
    print("[PASS] rasterize vjp smoke")
    print("  grad shape:", g.shape, "dtype:", g.dtype)


def main() -> None:
    ext = import_extension()

    failures = []

    try:
        run_preprocess_vjp_smoke(ext)
    except Exception as ex:
        failures.append(("preprocess", ex, traceback.format_exc()))

    try:
        run_rasterize_vjp_smoke(ext)
    except Exception as ex:
        failures.append(("rasterize", ex, traceback.format_exc()))

    if failures:
        print("[FAIL] backward vjp smoke")
        for name, ex, tb in failures:
            print(f"--- {name} failed: {type(ex).__name__}: {ex}")
            print(tb)
        raise SystemExit(1)

    print("[OK] backward vjp smoke all passed")


if __name__ == "__main__":
    main()
