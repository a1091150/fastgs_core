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
    return float(nz.item()) / float(x.size)


def main() -> None:
    ext = import_extension()
    n = 16
    w = 32
    h = 32
    base = {
        "background": mx.array([0.0, 0.0, 0.0], dtype=mx.float32),
        "means3d": mx.array([[0.1, 0.1, 1.2 + 0.01 * i] for i in range(n)], dtype=mx.float32),
        "colors_precomp": mx.array([[1.0, 0.6, 0.2] for _ in range(n)], dtype=mx.float32),
        "opacities": mx.array([0.6 for _ in range(n)], dtype=mx.float32),
        "cov3d_precomp": mx.array([[1.0, 0.0, 0.0, 1.0, 0.0, 1.0] for _ in range(n)], dtype=mx.float32),
        "metric_map": mx.zeros((w * h,), dtype=mx.int32),
        "viewmatrix": mx.eye(4, dtype=mx.float32),
        "projmatrix": mx.eye(4, dtype=mx.float32),
        "dc": mx.zeros((n, 3), dtype=mx.float32),
        "sh": mx.zeros((n, 0, 3), dtype=mx.float32),
        "campos": mx.zeros((3,), dtype=mx.float32),
        "viewspace_points": mx.zeros((n, 4), dtype=mx.float32),
    }

    def loss_fn(vsp):
        local = dict(base)
        local["viewspace_points"] = vsp
        out = ext.rasterize_gaussians(local, w, h, 16, 16, 1.0, 1.0, 0, 1.0, 1.0, False, False)
        return mx.sum(out["out_color"]) + 0.0 * mx.sum(out["viewspace_points"])

    d_vsp = mx.grad(loss_fn)(base["viewspace_points"])
    mx.eval(d_vsp)

    ok = True
    checks = [
        (d_vsp.shape == (n, 4), f"shape == ({n},4)"),
        (d_vsp.dtype == mx.float32, "dtype == float32"),
        (all_finite(d_vsp), "finite"),
    ]
    for passed, label in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] viewspace grad {label}")
        ok = ok and passed

    left_ratio = nonzero_ratio(d_vsp[:, :2])
    right_ratio = nonzero_ratio(d_vsp[:, 2:])
    print(f"[INFO] nonzero ratio [:2]={left_ratio:.6f}, [2:]={right_ratio:.6f}")

    left_ok = left_ratio > 0.0
    right_ok = all_finite(d_vsp[:, 2:])
    print(f"[{'PASS' if left_ok else 'FAIL'}] [:2] usable for densification")
    print(f"[{'PASS' if right_ok else 'FAIL'}] [2:] finite contract")
    ok = ok and left_ok and right_ok

    if ok:
        print("[OK] backward grad contract smoke passed")
        return
    print("[FAIL] backward grad contract smoke failed")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
