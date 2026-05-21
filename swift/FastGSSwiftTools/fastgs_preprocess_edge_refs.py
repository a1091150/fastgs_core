#!/usr/bin/env python3

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


def dump(ext, name, inputs, degree=0):
    out = ext.preprocess_forward(inputs, 64, 64, 16, 16, 1.0, 1.0, degree, 1.0, 1.0, False)
    mx.eval(*out.values())
    print("---", name)
    print(
        json.dumps(
            {
                k: {
                    "shape": list(v.shape),
                    "dtype": str(v.dtype),
                    "values": np.array(v).tolist(),
                }
                for k, v in out.items()
            },
            indent=2,
        )
    )


def main():
    ext = import_extension()
    base_mats = {
        "viewmat": mx.eye(4, dtype=mx.float32),
        "projmat": mx.eye(4, dtype=mx.float32),
        "cam_pos": mx.array([0, 0, 0], dtype=mx.float32),
    }

    inputs = dict(base_mats)
    inputs.update(
        {
            "means3d": mx.array([[0, 0, 0.1], [0.25, -0.25, 1]], dtype=mx.float32),
            "colors_precomp": mx.array([[1, 0, 0], [0, 1, 0]], dtype=mx.float32),
            "opacities": mx.array([1, 1], dtype=mx.float32),
            "scales": mx.array([[1, 1, 1], [1, 1, 1]], dtype=mx.float32),
            "quats": mx.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=mx.float32),
            "viewspace_points": mx.array([[0, 0, 0, 7], [0, 0, 0, 9]], dtype=mx.float32),
        }
    )
    dump(ext, "culling", inputs)

    inputs = dict(base_mats)
    inputs.update(
        {
            "means3d": mx.array([[0, 0, 1], [0.25, -0.25, 1]], dtype=mx.float32),
            "colors_precomp": mx.array([[1, 0, 0], [0, 1, 0]], dtype=mx.float32),
            "opacities": mx.array([1, 1], dtype=mx.float32),
            "cov3d_precomp": mx.array(
                [[0.5, 0.1, 0, 0.75, 0.05, 1.25], [1.5, -0.2, 0.1, 1.1, 0, 0.8]],
                dtype=mx.float32,
            ),
            "viewspace_points": mx.array([[0, 0, 0, 7], [0, 0, 0, 9]], dtype=mx.float32),
        }
    )
    dump(ext, "cov3d_precomp", inputs)

    inputs = dict(base_mats)
    inputs.update(
        {
            "means3d": mx.array([[0, 0, 1], [0.25, -0.25, 1]], dtype=mx.float32),
            "dc": mx.array([[-3.0, -2.0, -1.0], [0.2, 0.1, -4.0]], dtype=mx.float32),
            "sh": mx.array(np.zeros((2, 0, 3), dtype=np.float32), dtype=mx.float32),
            "opacities": mx.array([1, 1], dtype=mx.float32),
            "scales": mx.array([[1, 1, 1], [1, 1, 1]], dtype=mx.float32),
            "quats": mx.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=mx.float32),
            "viewspace_points": mx.array([[0, 0, 0, 7], [0, 0, 0, 9]], dtype=mx.float32),
        }
    )
    dump(ext, "sh_clamp", inputs, degree=0)


if __name__ == "__main__":
    main()
