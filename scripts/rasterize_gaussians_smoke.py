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

    n = 8
    image_width = 64
    image_height = 64
    stream = mx.gpu

    inputs = {
        "background": mx.zeros((3,), dtype=mx.float32, stream=stream),
        "means3d": mx.zeros((n, 3), dtype=mx.float32, stream=stream),
        "colors": mx.ones((n, 3), dtype=mx.float32, stream=stream),
        "opacities": mx.ones((n,), dtype=mx.float32, stream=stream),
        "scales": mx.ones((n, 3), dtype=mx.float32, stream=stream),
        "rotations": mx.zeros((n, 4), dtype=mx.float32, stream=stream),
        "metric_map": mx.zeros((image_width * image_height,), dtype=mx.int32, stream=stream),
        "viewmatrix": mx.eye(4, dtype=mx.float32, stream=stream),
        "projmatrix": mx.eye(4, dtype=mx.float32, stream=stream),
        "dc": mx.zeros((n, 3), dtype=mx.float32, stream=stream),
        "sh": mx.zeros((n, 0, 3), dtype=mx.float32, stream=stream),
        "campos": mx.zeros((3,), dtype=mx.float32, stream=stream),
        "viewspace_points": mx.zeros((n, 4), dtype=mx.float32, stream=stream),
    }

    out = ext.rasterize_gaussians_forward(
        inputs,
        image_width,
        image_height,
        16,
        16,
        1.0,
        1.0,
        0,
        1.0,
        1.0,
        False,
        False,
    )

    mx.eval(
        out["out_color"],
        out["radii"],
        out["point_offsets"],
        out["ranges"],
        out["bucket_count"],
    )

    print("rasterize_gaussians_forward smoke ok")
    print("rendered:", out["rendered"])
    print("num_buckets:", out["num_buckets"])
    print("out_color shape:", out["out_color"].shape)
    print("radii shape:", out["radii"].shape)
    print("ranges shape:", out["ranges"].shape)


if __name__ == "__main__":
    main()
