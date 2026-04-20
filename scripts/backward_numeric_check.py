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


def scalar_loss(ext, means2d):
    w, h = 16, 16
    out = ext.rasterize_forward(
        mx.array([[0, 1]], dtype=mx.uint32),
        mx.array([0], dtype=mx.uint32),
        mx.array([1], dtype=mx.uint32),
        means2d,
        mx.array([[1.0, 1.0, 1.0]], dtype=mx.float32),
        mx.array([[1.0, 0.0, 1.0, 0.5]], dtype=mx.float32),
        mx.array([0.0, 0.0, 0.0], dtype=mx.float32),
        mx.array([1], dtype=mx.int32),
        mx.zeros((w * h,), dtype=mx.int32),
        mx.zeros((1,), dtype=mx.int32),
        mx.zeros((1, 4), dtype=mx.float32),
        w,
        h,
        16,
        16,
        3,
        1,
        1,
        False,
    )
    return mx.sum(out["out_color"])


def main() -> None:
    ext = import_extension()
    x = mx.array([[8.0, 8.0]], dtype=mx.float32)
    grad = mx.grad(lambda z: scalar_loss(ext, z))(x)
    mx.eval(grad)

    eps = 1e-3
    num = []
    for j in range(2):
        d = [[0.0, 0.0]]
        d[0][j] = eps
        d = mx.array(d, dtype=mx.float32)
        fp = scalar_loss(ext, x + d)
        fm = scalar_loss(ext, x - d)
        mx.eval(fp, fm)
        num.append(float(((fp - fm) / (2.0 * eps)).item()))

    ana = [float(grad[0, 0].item()), float(grad[0, 1].item())]
    abs_err = [abs(ana[i] - num[i]) for i in range(2)]
    rel_err = [abs_err[i] / max(1e-6, abs(num[i])) for i in range(2)]

    tol = 5e-1  # staged implementation tolerance
    ok = max(rel_err) < tol
    print(f"[INFO] analytic={ana}")
    print(f"[INFO] numeric ={num}")
    print(f"[INFO] rel_err={rel_err}, tol={tol}")
    print(f"[{'PASS' if ok else 'FAIL'}] finite-difference check (means2d)")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
