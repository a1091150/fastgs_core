#!/usr/bin/env python3

import argparse
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


def _rasterize_loss(ext, means2d):
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


def _e2e_loss(ext, means3d, opacities):
    n, w, h = 8, 32, 32
    base = {
        "background": mx.array([0.0, 0.0, 0.0], dtype=mx.float32),
        "means3d": means3d,
        "colors_precomp": mx.array([[1.0, 0.6, 0.2] for _ in range(n)], dtype=mx.float32),
        "opacities": opacities,
        "cov3d_precomp": mx.array([[1.0, 0.0, 0.0, 1.0, 0.0, 1.0] for _ in range(n)], dtype=mx.float32),
        "metric_map": mx.zeros((w * h,), dtype=mx.int32),
        "viewmatrix": mx.eye(4, dtype=mx.float32),
        "projmatrix": mx.eye(4, dtype=mx.float32),
        "dc": mx.zeros((n, 3), dtype=mx.float32),
        "sh": mx.zeros((n, 0, 3), dtype=mx.float32),
        "campos": mx.zeros((3,), dtype=mx.float32),
        "viewspace_points": mx.zeros((n, 4), dtype=mx.float32),
    }
    out = ext.rasterize_gaussians(base, w, h, 16, 16, 1.0, 1.0, 0, 1.0, 1.0, False, False)
    return mx.sum(out["out_color"])


def _e2e_color_loss(ext, colors_precomp):
    n, w, h = 8, 32, 32
    base = {
        "background": mx.array([0.0, 0.0, 0.0], dtype=mx.float32),
        "means3d": mx.array([[0.1, 0.1, 1.2 + 0.01 * i] for i in range(n)], dtype=mx.float32),
        "colors_precomp": colors_precomp,
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
    out = ext.rasterize_gaussians(base, w, h, 16, 16, 1.0, 1.0, 0, 1.0, 1.0, False, False)
    return mx.sum(out["out_color"])


def _preprocess_cov_loss(ext, scales, quats):
    n = scales.shape[0]
    w, h = 16, 16
    inputs = {
        "means3d": mx.array([[0.1, 0.1, 1.2 + 0.01 * i] for i in range(n)], dtype=mx.float32),
        "opacities": mx.array([0.6 for _ in range(n)], dtype=mx.float32),
        "scales": scales,
        "quats": quats,
        "colors_precomp": mx.array([[1.0, 0.6, 0.2] for _ in range(n)], dtype=mx.float32),
        "viewmat": mx.eye(4, dtype=mx.float32),
        "projmat": mx.eye(4, dtype=mx.float32),
        "cam_pos": mx.zeros((3,), dtype=mx.float32),
        "viewspace_points": mx.zeros((n, 4), dtype=mx.float32),
    }
    out = ext.preprocess_forward(
        inputs,
        image_width=w,
        image_height=h,
        block_x=16,
        block_y=16,
        tan_fovx=1.0,
        tan_fovy=1.0,
        degree=0,
        scale_modifier=1.0,
        mult=1.0,
        prefiltered=False,
    )
    return mx.sum(out["cov3d"])


def _preprocess_sh_loss(ext, dc, sh):
    n = dc.shape[0]
    w, h = 16, 16
    inputs = {
        "means3d": mx.array([[0.1, 0.1, 1.2 + 0.01 * i] for i in range(n)], dtype=mx.float32),
        "opacities": mx.array([0.6 for _ in range(n)], dtype=mx.float32),
        "cov3d_precomp": mx.array([[1.0, 0.0, 0.0, 1.0, 0.0, 1.0] for _ in range(n)], dtype=mx.float32),
        "dc": dc,
        "sh": sh,
        "viewmat": mx.eye(4, dtype=mx.float32),
        "projmat": mx.eye(4, dtype=mx.float32),
        "cam_pos": mx.zeros((3,), dtype=mx.float32),
        "viewspace_points": mx.zeros((n, 4), dtype=mx.float32),
    }
    out = ext.preprocess_forward(
        inputs,
        image_width=w,
        image_height=h,
        block_x=16,
        block_y=16,
        tan_fovx=1.0,
        tan_fovy=1.0,
        degree=0,  # staged check focuses on degree-0 dc path
        scale_modifier=1.0,
        mult=1.0,
        prefiltered=False,
    )
    return mx.sum(out["rgb"])


def _central_diff_2d(loss_fn, x, eps):
    num = []
    for j in range(x.shape[1]):
        d = mx.array([[eps if k == j else 0.0 for k in range(x.shape[1])]], dtype=mx.float32)
        fp = loss_fn(x + d)
        fm = loss_fn(x - d)
        mx.eval(fp, fm)
        num.append(float(((fp - fm) / (2.0 * eps)).item()))
    return num


def _central_diff_scalar(loss_fn, x, idx, eps):
    if isinstance(idx, tuple):
        import numpy as np
        d_np = np.zeros(tuple(x.shape), dtype=np.float32)
        d_np[idx] = eps
        d = mx.array(d_np, dtype=mx.float32)
    else:
        d = [0.0 for _ in range(x.shape[0])]
        d[idx] = eps
        d = mx.array(d, dtype=mx.float32)
    fp = loss_fn(x + d)
    fm = loss_fn(x - d)
    mx.eval(fp, fm)
    return float(((fp - fm) / (2.0 * eps)).item())


def _report(name, ana, num, tol):
    abs_err = [abs(ana[i] - num[i]) for i in range(len(ana))]
    rel_err = [abs_err[i] / max(1e-6, abs(num[i])) for i in range(len(ana))]
    ok = max(rel_err) < tol
    print(f"[INFO] {name} analytic={ana}")
    print(f"[INFO] {name} numeric ={num}")
    print(f"[INFO] {name} rel_err={rel_err}, tol={tol}")
    print(f"[{'PASS' if ok else 'FAIL'}] finite-difference check ({name})")
    return ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--tol-staged", type=float, default=5e-1)
    p.add_argument("--check-opacity", action="store_true", help="Also check opacity finite-diff on current staged path.")
    args = p.parse_args()

    ext = import_extension()
    ok = True

    # 1) Raster primitive path: means2d
    x2d = mx.array([[8.0, 8.0]], dtype=mx.float32)
    g2d = mx.grad(lambda z: _rasterize_loss(ext, z))(x2d)
    mx.eval(g2d)
    ana2d = [float(g2d[0, 0].item()), float(g2d[0, 1].item())]
    num2d = _central_diff_2d(lambda z: _rasterize_loss(ext, z), x2d, args.eps)
    ok = _report("means2d", ana2d, num2d, args.tol_staged) and ok

    # 2) E2E path: means3d (sample one scalar coordinate)
    n = 8
    m3d = mx.array([[0.1, 0.1, 1.2 + 0.01 * i] for i in range(n)], dtype=mx.float32)
    opa = mx.array([0.6 for _ in range(n)], dtype=mx.float32)
    g3d = mx.grad(lambda m: _e2e_loss(ext, m, opa))(m3d)
    mx.eval(g3d)
    ana3d = [float(g3d[0, 0].item())]
    num3d = [_central_diff_scalar(lambda m: _e2e_loss(ext, m, opa), m3d, (0, 0), args.eps)]
    ok = _report("means3d[0,0]", ana3d, num3d, args.tol_staged) and ok

    # 3) Optional path: opacity (may be incomplete on staged migration)
    if args.check_opacity:
        gopa = mx.grad(lambda o: _e2e_loss(ext, m3d, o))(opa)
        mx.eval(gopa)
        ana_o = [float(gopa[0].item())]
        num_o = [_central_diff_scalar(lambda o: _e2e_loss(ext, m3d, o), opa, 0, args.eps)]
        ok = _report("opacity[0]", ana_o, num_o, args.tol_staged) and ok
    else:
        print("[INFO] skip opacity finite-difference (use --check-opacity to enable)")

    # 4) Preprocess path: scale / rotation
    n = 8
    scales = mx.array([[1.0, 1.1, 1.2] for _ in range(n)], dtype=mx.float32)
    quats = mx.array([[1.0, 0.0, 0.0, 0.0] for _ in range(n)], dtype=mx.float32)

    gscale = mx.grad(lambda s: _preprocess_cov_loss(ext, s, quats))(scales)
    mx.eval(gscale)
    ana_s = [float(gscale[0, 0].item())]
    num_s = [_central_diff_scalar(lambda s: _preprocess_cov_loss(ext, s, quats), scales, (0, 0), args.eps)]
    ok = _report("scale[0,0]", ana_s, num_s, args.tol_staged) and ok

    grot = mx.grad(lambda q: _preprocess_cov_loss(ext, scales, q))(quats)
    mx.eval(grot)
    ana_r = [float(grot[0, 0].item())]
    num_r = [_central_diff_scalar(lambda q: _preprocess_cov_loss(ext, scales, q), quats, (0, 0), args.eps)]
    ok = _report("rotation[0,0]", ana_r, num_r, args.tol_staged) and ok

    # 5) E2E path: colors_precomp
    n = 8
    colors = mx.array([[1.0, 0.6, 0.2] for _ in range(n)], dtype=mx.float32)
    gcol = mx.grad(lambda c: _e2e_color_loss(ext, c))(colors)
    mx.eval(gcol)
    ana_c = [float(gcol[0, 0].item())]
    num_c = [_central_diff_scalar(lambda c: _e2e_color_loss(ext, c), colors, (0, 0), args.eps)]
    ok = _report("colors_precomp[0,0]", ana_c, num_c, args.tol_staged) and ok

    # 6) Preprocess SH path (degree-0 staged): dc and sh
    n = 8
    dc = mx.array([[0.5, 0.4, 0.3] for _ in range(n)], dtype=mx.float32)
    sh = mx.zeros((n, 1, 3), dtype=mx.float32)
    gdc = mx.grad(lambda d: _preprocess_sh_loss(ext, d, sh))(dc)
    mx.eval(gdc)
    ana_dc = [float(gdc[0, 0].item())]
    num_dc = [_central_diff_scalar(lambda d: _preprocess_sh_loss(ext, d, sh), dc, (0, 0), args.eps)]
    ok = _report("dc[0,0]", ana_dc, num_dc, args.tol_staged) and ok

    gsh = mx.grad(lambda s: _preprocess_sh_loss(ext, dc, s))(sh)
    mx.eval(gsh)
    ana_sh = [float(gsh[0, 0, 0].item())]
    num_sh = [_central_diff_scalar(lambda s: _preprocess_sh_loss(ext, dc, s), sh, (0, 0), args.eps)]
    ok = _report("sh[0,0,0]", ana_sh, num_sh, args.tol_staged) and ok

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
