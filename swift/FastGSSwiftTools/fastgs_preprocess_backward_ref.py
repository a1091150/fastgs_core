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
    parser = argparse.ArgumentParser(description="Generate preprocess backward reference summary for FastGSSwift.")
    parser.add_argument("--out", default="/private/tmp/fastgs_preprocess_backward_ref.json")
    args = parser.parse_args()

    ext = import_extension()

    means3d = mx.array(
        [
            [0.0, 0.0, 1.0],
            [0.25, -0.25, 1.0],
        ],
        dtype=mx.float32,
    )
    colors_precomp = mx.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=mx.float32,
    )
    opacities = mx.array([1.0, 1.0], dtype=mx.float32)
    scales = mx.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=mx.float32,
    )
    quats = mx.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=mx.float32,
    )
    viewspace_points = mx.array(
        [
            [0.0, 0.0, 0.0, 7.0],
            [0.0, 0.0, 0.0, 9.0],
        ],
        dtype=mx.float32,
    )

    base_inputs = {
        "means3d": means3d,
        "colors_precomp": colors_precomp,
        "opacities": opacities,
        "scales": scales,
        "quats": quats,
        "cov3d_precomp": mx.zeros((0,), dtype=mx.float32),
        "viewmat": mx.eye(4, dtype=mx.float32),
        "projmat": mx.eye(4, dtype=mx.float32),
        "cam_pos": mx.zeros((3,), dtype=mx.float32),
        "viewspace_points": viewspace_points,
    }

    def loss_fn(m3d, colors, opa, scl, quat, vsp):
        local = dict(base_inputs)
        local["means3d"] = m3d
        local["colors_precomp"] = colors
        local["opacities"] = opa
        local["scales"] = scl
        local["quats"] = quat
        local["viewspace_points"] = vsp
        out = ext.preprocess_forward(
            local,
            image_width=64,
            image_height=64,
            block_x=16,
            block_y=16,
            tan_fovx=1.0,
            tan_fovy=1.0,
            degree=0,
            scale_modifier=1.0,
            mult=1.0,
            prefiltered=False,
        )
        return (
            mx.sum(out["xys"])
            + 0.1 * mx.sum(out["depths"])
            + 0.2 * mx.sum(out["cov3d"])
            + 0.3 * mx.sum(out["rgb"])
            + 0.4 * mx.sum(out["conic_opacity"])
            + 0.5 * mx.sum(out["viewspace_points"])
        )

    value, grads = mx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5))(
        means3d,
        colors_precomp,
        opacities,
        scales,
        quats,
        viewspace_points,
    )
    d_means3d, d_colors, d_opacities, d_scales, d_quats, d_viewspace = grads
    mx.eval(value, d_means3d, d_colors, d_opacities, d_scales, d_quats, d_viewspace)

    sample_ids = {
        "means3D": [0, 1, 2, 3, 4, 5],
        "colorsPrecomputed": [0, 1, 2, 3, 4, 5],
        "opacities": [0, 1],
        "scales": [0, 1, 2, 3, 4, 5],
        "rotations": [0, 1, 2, 3, 4, 5, 6, 7],
        "viewspacePoints": [0, 1, 2, 3, 4, 5, 6, 7],
    }

    report = {
        "loss": float(value.item()),
        "gradients": {
            "means3D": summarize(d_means3d, sample_ids["means3D"]),
            "colorsPrecomputed": summarize(d_colors, sample_ids["colorsPrecomputed"]),
            "opacities": summarize(d_opacities, sample_ids["opacities"]),
            "scales": summarize(d_scales, sample_ids["scales"]),
            "rotations": summarize(d_quats, sample_ids["rotations"]),
            "viewspacePoints": summarize(d_viewspace, sample_ids["viewspacePoints"]),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
