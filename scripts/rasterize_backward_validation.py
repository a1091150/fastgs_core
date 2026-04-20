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
    n = 1
    image_width = 16
    image_height = 16
    num_tiles = 1
    bucket_sum = 1

    ranges = mx.array([[0, 1]], dtype=mx.uint32)
    point_list = mx.array([0], dtype=mx.uint32)
    per_tile_bucket_offset = mx.array([1], dtype=mx.uint32)
    colors = mx.array([[1.0, 0.5, 0.25]], dtype=mx.float32)
    conic_opacity = mx.array([[1.0, 0.0, 1.0, 0.5]], dtype=mx.float32)
    background = mx.array([0.0, 0.0, 0.0], dtype=mx.float32)
    radii = mx.array([1], dtype=mx.int32)
    metric_map = mx.zeros((image_width * image_height,), dtype=mx.int32)
    metric_count = mx.zeros((n,), dtype=mx.int32)
    viewspace_points = mx.zeros((n, 4), dtype=mx.float32)
    means2d_init = mx.array([[8.0, 8.0]], dtype=mx.float32)

    def loss_fn(means2d, colors_in, conic_in, viewspace_in):
        out = ext.rasterize_forward(
            ranges,
            point_list,
            per_tile_bucket_offset,
            means2d,
            colors_in,
            conic_in,
            background,
            radii,
            metric_map,
            metric_count,
            viewspace_in,
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

    vg = mx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3))
    value, grads = vg(means2d_init, colors, conic_opacity, viewspace_points)
    d_means2d, d_colors, d_conic, d_viewspace = grads
    mx.eval(value, d_means2d, d_colors, d_conic, d_viewspace)

    return {
        "value": value,
        "d_means2d": d_means2d,
        "d_colors": d_colors,
        "d_conic": d_conic,
        "d_viewspace": d_viewspace,
    }


def validate_shapes_and_finite(result):
    checks = []
    checks.append((result["d_means2d"].shape == (1, 2), "d_means2d shape == (1,2)"))
    checks.append((result["d_colors"].shape == (1, 3), "d_colors shape == (1,3)"))
    checks.append((result["d_conic"].shape == (1, 4), "d_conic shape == (1,4)"))
    checks.append((result["d_viewspace"].shape == (1, 4), "d_viewspace shape == (1,4)"))

    checks.append((result["d_means2d"].dtype == mx.float32, "d_means2d dtype == float32"))
    checks.append((result["d_colors"].dtype == mx.float32, "d_colors dtype == float32"))
    checks.append((result["d_conic"].dtype == mx.float32, "d_conic dtype == float32"))
    checks.append((result["d_viewspace"].dtype == mx.float32, "d_viewspace dtype == float32"))

    checks.append((_all_finite(result["d_means2d"]), "d_means2d finite"))
    checks.append((_all_finite(result["d_colors"]), "d_colors finite"))
    checks.append((_all_finite(result["d_conic"]), "d_conic finite"))
    checks.append((_all_finite(result["d_viewspace"]), "d_viewspace finite"))

    ok = True
    for passed, label in checks:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {label}")
        ok = ok and passed
    return ok


def validate_repeatability(r1, r2, tol=1e-6):
    diffs = {
        "d_means2d": _max_abs(r1["d_means2d"] - r2["d_means2d"]),
        "d_colors": _max_abs(r1["d_colors"] - r2["d_colors"]),
        "d_conic": _max_abs(r1["d_conic"] - r2["d_conic"]),
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
        print("[OK] rasterize backward validation passed")
        return

    print("[FAIL] rasterize backward validation failed")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
