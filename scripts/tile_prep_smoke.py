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

    num_rendered = 8
    num_tiles = 4
    # high 32-bit tile id, low 32-bit depth key (dummy ordering)
    keys_np = [
        (0 << 32) | 1,
        (0 << 32) | 2,
        (1 << 32) | 1,
        (1 << 32) | 2,
        (1 << 32) | 3,
        (3 << 32) | 1,
        (3 << 32) | 2,
        (3 << 32) | 3,
    ]
    point_list_keys = mx.array(keys_np, dtype=mx.uint64)

    out = ext.tile_prep_forward(point_list_keys, num_rendered, num_tiles)
    mx.eval(out["ranges"], out["bucket_count"])

    print("tile_prep_forward smoke ok")
    print("ranges shape:", out["ranges"].shape)
    print("bucket_count shape:", out["bucket_count"].shape)


if __name__ == "__main__":
    main()
