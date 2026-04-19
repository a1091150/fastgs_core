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

    p = 8
    xys = mx.zeros((p, 2), dtype=mx.float32)
    depths = mx.arange(p, dtype=mx.float32)
    point_offsets = mx.arange(1, p + 1, dtype=mx.uint32)
    conic_opacity = mx.ones((p, 4), dtype=mx.float32)
    tiles_touched = mx.ones((p,), dtype=mx.uint32)

    out = ext.binning_forward(
        xys,
        depths,
        point_offsets,
        conic_opacity,
        tiles_touched,
        1.0,
        4,
        4,
        1,
        p,
    )

    mx.eval(out["point_list_keys_unsorted"], out["point_list_unsorted"])

    print("binning_forward smoke ok")
    print("point_list_keys_unsorted shape:", out["point_list_keys_unsorted"].shape)
    print("point_list_unsorted shape:", out["point_list_unsorted"].shape)


if __name__ == "__main__":
    main()
