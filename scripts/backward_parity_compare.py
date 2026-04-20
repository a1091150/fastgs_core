#!/usr/bin/env python3

import argparse
import json
import os
import sys

import mlx.core as mx
import numpy as np


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


def capture_current(ext):
    from backward_grad_contract_smoke import import_extension as _  # noqa: F401
    n, w, h = 16, 32, 32
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

    def loss_fn(m3d, vsp):
        local = dict(base)
        local["means3d"] = m3d
        local["viewspace_points"] = vsp
        out = ext.rasterize_gaussians(local, w, h, 16, 16, 1.0, 1.0, 0, 1.0, 1.0, False, False)
        return mx.sum(out["out_color"])

    value, grads = mx.value_and_grad(loss_fn, argnums=(0, 1))(base["means3d"], base["viewspace_points"])
    d_means3d, d_viewspace = grads
    mx.eval(value, d_means3d, d_viewspace)
    return {
        "value": float(value.item()),
        "d_means3d": np.array(d_means3d),
        "d_viewspace": np.array(d_viewspace),
    }


def max_rel_err(a, b):
    denom = np.maximum(1e-6, np.abs(b))
    return float(np.max(np.abs(a - b) / denom))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ref", type=str, default="", help="Path to CUDA reference .npz")
    p.add_argument("--out", type=str, default="/tmp/fastgs_backward_current.npz", help="Path to save current snapshot")
    p.add_argument(
        "--report",
        type=str,
        default="/tmp/fastgs_backward_parity_report.json",
        help="Path to save parity report JSON",
    )
    p.add_argument("--tol", type=float, default=1e-2)
    args = p.parse_args()

    ext = import_extension()
    cur = capture_current(ext)
    np.savez(args.out, **cur)
    print(f"[INFO] saved current snapshot: {args.out}")
    print("[INFO] expected reference schema: keys={value, d_means3d, d_viewspace}")

    if not args.ref:
        print("[INFO] no --ref provided; snapshot generated for future parity compare")
        return

    ref = np.load(args.ref)
    keys = ["value", "d_means3d", "d_viewspace"]
    ok = True
    report = {"tol": args.tol, "ref": args.ref, "out": args.out, "results": {}}
    for k in keys:
        if k not in ref:
            print(f"[FAIL] missing key in ref: {k}")
            ok = False
            report["results"][k] = {"status": "missing"}
            continue
        if k == "value":
            err = abs(float(cur[k]) - float(ref[k]))
            passed = err <= args.tol
            print(f"[{'PASS' if passed else 'FAIL'}] {k} abs_err={err:.3e} tol={args.tol:.1e}")
            report["results"][k] = {"status": "pass" if passed else "fail", "abs_err": err}
            ok = ok and passed
            continue
        err = max_rel_err(cur[k], ref[k])
        passed = err <= args.tol
        print(f"[{'PASS' if passed else 'FAIL'}] {k} max_rel_err={err:.3e} tol={args.tol:.1e}")
        report["results"][k] = {"status": "pass" if passed else "fail", "max_rel_err": err}
        ok = ok and passed

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[INFO] parity report written: {args.report}")

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
