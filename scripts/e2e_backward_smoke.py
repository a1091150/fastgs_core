#!/usr/bin/env python3

import os
import sys

import mlx.core as mx


def import_extension():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(repo_root, "python_package"))
    sys.path.insert(0, os.path.join(repo_root, "build"))
    try:
        from fastgs_core import _fastgs_core as ext
        return ext
    except Exception:
        import _fastgs_core as ext
        return ext


def all_finite(x: mx.array) -> bool:
    return bool(mx.all(mx.isfinite(x)).item())


def nonzero_ratio(x: mx.array, eps: float = 1e-12) -> float:
    nz = mx.sum(mx.abs(x) > eps)
    total = x.size
    return float(nz.item()) / float(total)


def build_inputs(n: int, image_width: int, image_height: int):
    return {
        "background": mx.array([0.0, 0.0, 0.0], dtype=mx.float32),
        "means3d": mx.array([[0.1, 0.1, 1.2 + 0.01 * i] for i in range(n)], dtype=mx.float32),
        "colors_precomp": mx.array([[1.0, 0.6, 0.2] for _ in range(n)], dtype=mx.float32),
        "opacities": mx.array([0.6 for _ in range(n)], dtype=mx.float32),
        "cov3d_precomp": mx.array([[1.0, 0.0, 0.0, 1.0, 0.0, 1.0] for _ in range(n)], dtype=mx.float32),
        "metric_map": mx.zeros((image_width * image_height,), dtype=mx.int32),
        "viewmatrix": mx.eye(4, dtype=mx.float32),
        "projmatrix": mx.eye(4, dtype=mx.float32),
        "dc": mx.zeros((n, 3), dtype=mx.float32),
        "sh": mx.zeros((n, 0, 3), dtype=mx.float32),
        "campos": mx.zeros((3,), dtype=mx.float32),
        "viewspace_points": mx.zeros((n, 4), dtype=mx.float32),
    }


def main():
    ext = import_extension()

    n = 16
    image_width = 32
    image_height = 32
    base = build_inputs(n, image_width, image_height)

    def loss_fn(means3d, viewspace_points):
        local = dict(base)
        local["means3d"] = means3d
        local["viewspace_points"] = viewspace_points
        out = ext.rasterize_gaussians(
            local,
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
        # Main path: image loss. Tiny auxiliary term keeps gradient query over returned
        # viewspace tensor explicit and catches accidental graph breaks.
        return mx.sum(out["out_color"]) + 0.0 * mx.sum(out["viewspace_points"])

    vg = mx.value_and_grad(loss_fn, argnums=(0, 1))
    value, grads = vg(base["means3d"], base["viewspace_points"])
    d_means3d, d_viewspace = grads
    mx.eval(value, d_means3d, d_viewspace)

    ok = True

    checks = [
        (d_viewspace.shape == (n, 4), f"d_viewspace shape == ({n},4)"),
        (d_viewspace.dtype == mx.float32, "d_viewspace dtype == float32"),
        (all_finite(d_viewspace), "d_viewspace finite"),
    ]

    for passed, label in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {label}")
        ok = ok and passed

    left = d_viewspace[:, :2]
    right = d_viewspace[:, 2:]
    left_ratio = nonzero_ratio(left)
    right_ratio = nonzero_ratio(right)

    print(f"[INFO] viewspace grad nonzero ratio [:2] = {left_ratio:.6f}")
    print(f"[INFO] viewspace grad nonzero ratio [2:] = {right_ratio:.6f}")

    # For current staged migration, enforce non-zero on :2 and finiteness on 2:.
    left_ok = left_ratio > 0.0
    right_ok = all_finite(right)
    print(f"[{'PASS' if left_ok else 'FAIL'}] viewspace grad [:2] non-zero")
    print(f"[{'PASS' if right_ok else 'FAIL'}] viewspace grad [2:] finite")
    ok = ok and left_ok and right_ok

    if ok:
        print("[OK] e2e backward smoke passed")
        return

    print("[FAIL] e2e backward smoke failed")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
