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


def _max_abs(x: mx.array) -> float:
    return float(mx.max(mx.abs(x)).item())


def _all_finite(x: mx.array) -> bool:
    return bool(mx.all(mx.isfinite(x)).item())


def run_once(ext):
    n = 8
    image_width = 16
    image_height = 16

    means3d = mx.array(
        [
            [0.1, 0.2, 1.0],
            [0.2, 0.1, 1.1],
            [0.3, 0.0, 1.2],
            [0.0, 0.3, 1.3],
            [0.1, 0.1, 1.4],
            [0.2, 0.2, 1.5],
            [0.3, 0.3, 1.6],
            [0.4, 0.2, 1.7],
        ],
        dtype=mx.float32,
    )
    viewspace_points = mx.zeros((n, 4), dtype=mx.float32)

    base_inputs = {
        "means3d": means3d,
        "opacities": mx.ones((n, 1), dtype=mx.float32),
        "cov3d_precomp": mx.ones((n, 6), dtype=mx.float32),
        "colors_precomp": mx.ones((n, 3), dtype=mx.float32),
        "viewmat": mx.eye(4, dtype=mx.float32),
        "projmat": mx.eye(4, dtype=mx.float32),
        "cam_pos": mx.zeros((3,), dtype=mx.float32),
        "viewspace_points": viewspace_points,
    }

    def loss_fn(m3d, vsp):
        local = dict(base_inputs)
        local["means3d"] = m3d
        local["viewspace_points"] = vsp
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
        return mx.sum(out["xys"]) + 0.1 * mx.sum(out["depths"]) + 0.01 * mx.sum(out["viewspace_points"])

    vg = mx.value_and_grad(loss_fn, argnums=(0, 1))
    value, grads = vg(means3d, viewspace_points)
    d_means3d, d_viewspace = grads
    mx.eval(value, d_means3d, d_viewspace)

    return {
        "value": value,
        "d_means3d": d_means3d,
        "d_viewspace": d_viewspace,
    }


def validate_shapes_and_finite(result):
    checks = []
    checks.append((result["d_means3d"].shape == (8, 3), "d_means3d shape == (8,3)"))
    checks.append((result["d_viewspace"].shape == (8, 4), "d_viewspace shape == (8,4)"))
    checks.append((result["d_means3d"].dtype == mx.float32, "d_means3d dtype == float32"))
    checks.append((result["d_viewspace"].dtype == mx.float32, "d_viewspace dtype == float32"))
    checks.append((_all_finite(result["d_means3d"]), "d_means3d finite"))
    checks.append((_all_finite(result["d_viewspace"]), "d_viewspace finite"))

    ok = True
    for passed, label in checks:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {label}")
        ok = ok and passed

    print(f"[INFO] d_means3d max_abs={_max_abs(result['d_means3d']):.3e}")
    print(f"[INFO] d_viewspace max_abs={_max_abs(result['d_viewspace']):.3e}")
    return ok


def validate_repeatability(r1, r2, tol=1e-6):
    diffs = {
        "d_means3d": _max_abs(r1["d_means3d"] - r2["d_means3d"]),
        "d_viewspace": _max_abs(r1["d_viewspace"] - r2["d_viewspace"]),
    }
    ok = True
    for k, v in diffs.items():
        passed = v <= tol
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] repeatability {k}: max_abs_diff={v:.3e} (tol={tol:.1e})")
        ok = ok and passed
    return ok


def main():
    ext = import_extension()

    r1 = run_once(ext)
    print("[INFO] shape/dtype/finite checks")
    ok_contract = validate_shapes_and_finite(r1)

    r2 = run_once(ext)
    print("[INFO] repeatability checks")
    ok_repeat = validate_repeatability(r1, r2)

    if ok_contract and ok_repeat:
        print("[OK] preprocess backward validation passed")
        return

    print("[FAIL] preprocess backward validation failed")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
