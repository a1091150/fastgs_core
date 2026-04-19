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
    stream = mx.gpu

    inputs = {
        "means3d": mx.zeros((n, 3), dtype=mx.float32, stream=stream),
        "dc": mx.zeros((n, 3), dtype=mx.float32, stream=stream),
        "sh": mx.zeros((n, 0, 3), dtype=mx.float32, stream=stream),
        "colors_precomp": mx.zeros((n, 3), dtype=mx.float32, stream=stream),
        "opacities": mx.ones((n,), dtype=mx.float32, stream=stream),
        "scales": mx.ones((n, 3), dtype=mx.float32, stream=stream),
        "quats": mx.zeros((n, 4), dtype=mx.float32, stream=stream),
        "cov3d_precomp": mx.zeros((0,), dtype=mx.float32, stream=stream),
        "viewmat": mx.eye(4, dtype=mx.float32, stream=stream),
        "projmat": mx.eye(4, dtype=mx.float32, stream=stream),
        "cam_pos": mx.zeros((3,), dtype=mx.float32, stream=stream),
        "viewspace_points": mx.zeros((n, 4), dtype=mx.float32, stream=stream),
    }

    out = ext.preprocess_forward(
        inputs,
        64,
        64,
        16,
        16,
        1.0,
        1.0,
        0,
        1.0,
        1.0,
        False,
    )

    mx.eval(
        out["radii"],
        out["xys"],
        out["depths"],
        out["cov3d"],
        out["rgb"],
        out["conic_opacity"],
        out["tiles_touched"],
        out["clamped"],
        out["viewspace_points"],
    )

    print("preprocess_forward smoke ok")
    print("radii shape:", out["radii"].shape)
    print("xys shape:", out["xys"].shape)
    print("rgb shape:", out["rgb"].shape)


if __name__ == "__main__":
    main()
