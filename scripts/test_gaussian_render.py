#!/usr/bin/env python3

import argparse
import math
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import numpy as np

from train_scanner_fixed import (
    import_extension,
    init_model,
    logit,
    prepare_dataset,
    render_chw,
    save_side_by_side,
)

try:
    import spz
except Exception:
    spz = None


def quaternions_wxyz_to_rotation_matrices(quats: np.ndarray) -> np.ndarray:
    q = np.asarray(quats, dtype=np.float32)
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.clip(norms, 1.0e-8, None)

    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    rot = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    rot[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    rot[:, 0, 1] = 2.0 * (xy - wz)
    rot[:, 0, 2] = 2.0 * (xz + wy)
    rot[:, 1, 0] = 2.0 * (xy + wz)
    rot[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    rot[:, 1, 2] = 2.0 * (yz - wx)
    rot[:, 2, 0] = 2.0 * (xz - wy)
    rot[:, 2, 1] = 2.0 * (yz + wx)
    rot[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return rot


def rotation_matrices_to_quaternions_wxyz(rot: np.ndarray) -> np.ndarray:
    r = np.asarray(rot, dtype=np.float32)
    q = np.empty((r.shape[0], 4), dtype=np.float32)

    trace = r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2]
    mask = trace > 0.0

    if np.any(mask):
        s = np.sqrt(trace[mask] + 1.0) * 2.0
        q[mask, 0] = 0.25 * s
        q[mask, 1] = (r[mask, 2, 1] - r[mask, 1, 2]) / s
        q[mask, 2] = (r[mask, 0, 2] - r[mask, 2, 0]) / s
        q[mask, 3] = (r[mask, 1, 0] - r[mask, 0, 1]) / s

    mask_x = (~mask) & (r[:, 0, 0] > r[:, 1, 1]) & (r[:, 0, 0] > r[:, 2, 2])
    if np.any(mask_x):
        s = np.sqrt(1.0 + r[mask_x, 0, 0] - r[mask_x, 1, 1] - r[mask_x, 2, 2]) * 2.0
        q[mask_x, 0] = (r[mask_x, 2, 1] - r[mask_x, 1, 2]) / s
        q[mask_x, 1] = 0.25 * s
        q[mask_x, 2] = (r[mask_x, 0, 1] + r[mask_x, 1, 0]) / s
        q[mask_x, 3] = (r[mask_x, 0, 2] + r[mask_x, 2, 0]) / s

    mask_y = (~mask) & (~mask_x) & (r[:, 1, 1] > r[:, 2, 2])
    if np.any(mask_y):
        s = np.sqrt(1.0 + r[mask_y, 1, 1] - r[mask_y, 0, 0] - r[mask_y, 2, 2]) * 2.0
        q[mask_y, 0] = (r[mask_y, 0, 2] - r[mask_y, 2, 0]) / s
        q[mask_y, 1] = (r[mask_y, 0, 1] + r[mask_y, 1, 0]) / s
        q[mask_y, 2] = 0.25 * s
        q[mask_y, 3] = (r[mask_y, 1, 2] + r[mask_y, 2, 1]) / s

    mask_z = (~mask) & (~mask_x) & (~mask_y)
    if np.any(mask_z):
        s = np.sqrt(1.0 + r[mask_z, 2, 2] - r[mask_z, 0, 0] - r[mask_z, 1, 1]) * 2.0
        q[mask_z, 0] = (r[mask_z, 1, 0] - r[mask_z, 0, 1]) / s
        q[mask_z, 1] = (r[mask_z, 0, 2] + r[mask_z, 2, 0]) / s
        q[mask_z, 2] = (r[mask_z, 1, 2] + r[mask_z, 2, 1]) / s
        q[mask_z, 3] = 0.25 * s

    q /= np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1.0e-8, None)
    return q


def save_as_spz_local(filename: Path, model, sh_degree: int) -> bool:
    if spz is None:
        print("[WARN] spz is not available; skip spz export")
        return False

    mx.eval(
        model.means3d,
        model.log_scales,
        model.get_rotations,
        model.opacity_logits,
        model.features_dc,
        model.features_rest,
    )

    means = np.array(model.means3d, dtype=np.float32)
    means_spz = np.empty_like(means)
    means_spz[:, 0] = means[:, 0]
    means_spz[:, 1] = -means[:, 2]
    means_spz[:, 2] = means[:, 1]

    quats = np.array(model.get_rotations, dtype=np.float32)
    rot_mats = quaternions_wxyz_to_rotation_matrices(quats)
    axis3 = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    rot_mats_spz = axis3 @ rot_mats @ axis3.T
    quats_spz = rotation_matrices_to_quaternions_wxyz(rot_mats_spz)

    cloud = spz.GaussianCloud()
    cloud.antialiased = True
    cloud.positions = means_spz.flatten().astype(np.float32)
    cloud.scales = np.array(model.log_scales, dtype=np.float32).flatten()
    cloud.rotations = quats_spz.flatten().astype(np.float32)
    cloud.alphas = np.array(model.opacity_logits, dtype=np.float32).flatten()
    cloud.colors = np.array(model.features_dc, dtype=np.float32).flatten()
    cloud.sh_degree = int(sh_degree)
    cloud.sh = np.array(model.features_rest, dtype=np.float32).flatten()

    opts = spz.PackOptions()
    ok = spz.save_spz(cloud, opts, str(filename))
    if not ok:
        raise RuntimeError(f"failed to save spz to {filename}")
    print(f"saved spz: {filename}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load scanner point cloud, initialize gaussians, render without training, and export SBS image + SPZ."
    )
    parser.add_argument("--data", type=str, default="/Users/yangdunfu/Downloads/2026_03_01_16_36_14")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=30000000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extra-points-ratio", type=float, default=0.0)
    parser.add_argument("--extra-points-mode", type=str, default="surface-jitter")
    parser.add_argument("--extra-points-jitter-scale", type=float, default=0.01)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--eval-index", type=int, default=0)
    parser.add_argument("--render-all", action="store_true")
    parser.add_argument("--scale", type=float, default=0.02)
    parser.add_argument("--opacity", type=float, default=0.82)
    args = parser.parse_args()

    if args.sh_degree < 0 or args.sh_degree > 3:
        raise ValueError("--sh-degree must be between 0 and 3")
    if args.extra_points_ratio < 0.0:
        raise ValueError("--extra-points-ratio must be non-negative")
    if args.extra_points_mode not in {"surface-jitter", "bbox"}:
        raise ValueError("--extra-points-mode must be one of: surface-jitter, bbox")
    if args.extra_points_jitter_scale < 0.0:
        raise ValueError("--extra-points-jitter-scale must be non-negative")
    if args.scale <= 0.0:
        raise ValueError("--scale must be positive")
    if not (0.0 < args.opacity < 1.0):
        raise ValueError("--opacity must be between 0 and 1")

    dataset_dir = Path(args.data)
    if not dataset_dir.exists():
        raise RuntimeError(f"Dataset path does not exist: {dataset_dir}")

    ext = import_extension()
    cameras, targets, points, colors, base_point_count = prepare_dataset(
        dataset_dir=dataset_dir,
        width=args.width,
        height=args.height,
        max_frames=args.max_frames,
        frame_step=args.frame_step,
        start_index=args.start_index,
        max_points=args.max_points,
        seed=args.seed,
        extra_points_ratio=args.extra_points_ratio,
        extra_points_mode=args.extra_points_mode,
        extra_points_jitter_scale=args.extra_points_jitter_scale,
    )
    if not cameras:
        raise RuntimeError("No cameras loaded from dataset")

    eval_idx = max(0, min(args.eval_index, len(cameras) - 1))
    extra_point_count = int(points.shape[0] - base_point_count)
    model = init_model(points, colors, args.sh_degree)
    n = points.shape[0]
    model.log_scales = mx.array(
        np.full((n, 3), math.log(args.scale), dtype=np.float32),
        dtype=mx.float32,
    )
    model.opacity_logits = mx.array(
        logit(np.full((n,), args.opacity, dtype=np.float32)).astype(np.float32),
        dtype=mx.float32,
    )
    background = mx.array([0.0, 0.0, 0.0], dtype=mx.float32)
    render_means3d = model.means3d
    render_features_dc = model.features_dc
    render_features_rest = model.features_rest
    render_opacities = mx.sigmoid(model.opacity_logits)
    render_scales = mx.exp(model.log_scales)
    render_rotations = model.rotations / (mx.linalg.norm(model.rotations, axis=1, keepdims=True) + 1.0e-8)

    repo_root = Path(__file__).resolve().parent.parent
    date_dir = datetime.now().strftime("%Y%m%d_%H_%M")
    out_dir = repo_root / "training" / "output" / "test_gaussian_render" / date_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_sbs = out_dir / f"sbs_{eval_idx:04d}.png"
    out_spz = out_dir / "gaussians.spz"

    pred_eval = render_chw(
        ext=ext,
        means3d=render_means3d,
        features_dc=render_features_dc,
        features_rest=render_features_rest,
        opacities=render_opacities,
        scales=render_scales,
        rotations=render_rotations,
        camera=cameras[eval_idx],
        background=background,
        sh_degree=args.sh_degree,
    )
    save_side_by_side(targets[eval_idx], pred_eval, out_sbs)

    if args.render_all:
        out_all_dir = out_dir / "all_views"
        out_all_dir.mkdir(parents=True, exist_ok=True)
        for cam_idx, (camera, target_chw) in enumerate(zip(cameras, targets)):
            pred_camera = render_chw(
                ext=ext,
                means3d=render_means3d,
                features_dc=render_features_dc,
                features_rest=render_features_rest,
                opacities=render_opacities,
                scales=render_scales,
                rotations=render_rotations,
                camera=camera,
                background=background,
                sh_degree=args.sh_degree,
            )
            save_side_by_side(target_chw, pred_camera, out_all_dir / f"sbs_{cam_idx:04d}.png")

    save_as_spz_local(out_spz, model, args.sh_degree)

    print("[OK] test_gaussian_render done")
    print("frames:", len(cameras), "points:", points.shape[0])
    print("base_points:", points.shape[0] - extra_point_count, "extra_points:", extra_point_count)
    print("extra_points_mode:", args.extra_points_mode, "extra_points_ratio:", args.extra_points_ratio)
    print("render_scales:", "exp(log_scales)")
    print("scale:", args.scale, "opacity:", args.opacity)
    print("eval_index:", eval_idx)
    print("saved sbs:", out_sbs)
    if args.render_all:
        print("saved all views:", out_dir / "all_views")
    print("saved spz:", out_spz)


if __name__ == "__main__":
    main()
