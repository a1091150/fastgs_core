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
    save_as_spz,
    save_side_by_side,
)


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

    save_as_spz(out_spz, model, args.sh_degree)

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
