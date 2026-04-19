#!/usr/bin/env python3

import argparse
import os
import sys
from datetime import datetime
import math

import cv2
import numpy as np
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


def to_hwc_rgb(image_array: mx.array, h: int, w: int) -> np.ndarray:
    arr = np.array(image_array)
    if arr.ndim == 1 and arr.size == h * w * 3:
        arr = arr.reshape(h, w, 3)
    elif arr.ndim == 2 and arr.shape == (h * w, 3):
        arr = arr.reshape(h, w, 3)
    elif arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[1] == h and arr.shape[2] == w:
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.ndim == 2 and arr.shape[0] == 3 and arr.shape[1] == h * w:
        arr = arr.reshape(3, h, w)
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.ndim == 3 and arr.shape[0] == h and arr.shape[1] == w and arr.shape[2] == 3:
        pass
    else:
        raise RuntimeError(f"Unexpected out_color shape: {arr.shape}")
    return arr


def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1.0e-8)


def get_projection_matrix(znear: float, zfar: float, fovx: float, fovy: float) -> np.ndarray:
    tan_half_fovy = math.tan(fovy / 2.0)
    tan_half_fovx = math.tan(fovx / 2.0)

    top = tan_half_fovy * znear
    bottom = -top
    right = tan_half_fovx * znear
    left = -right

    p = np.zeros((4, 4), dtype=np.float32)
    z_sign = 1.0
    p[0, 0] = 2.0 * znear / (right - left)
    p[1, 1] = 2.0 * znear / (top - bottom)
    p[0, 2] = (right + left) / (right - left)
    p[1, 2] = (top + bottom) / (top - bottom)
    p[3, 2] = z_sign
    p[2, 2] = z_sign * zfar / (zfar - znear)
    p[2, 3] = -(zfar * znear) / (zfar - znear)
    return p


def build_look_at_camera(
    fovx: float,
    fovy: float,
    eye=(0.0, 0.0, 4.2),
    target=(0.0, 0.0, 0.0),
    up=(0.0, 1.0, 0.0),
):
    znear = 0.01
    zfar = 100.0

    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    forward = normalize(target - eye)
    right = normalize(np.cross(forward, up))
    true_up = normalize(np.cross(right, forward))

    world_to_view = np.eye(4, dtype=np.float32)
    world_to_view[0, :3] = right
    world_to_view[1, :3] = true_up
    world_to_view[2, :3] = forward
    world_to_view[:3, 3] = -world_to_view[:3, :3] @ eye

    world_view = world_to_view.T
    projection = get_projection_matrix(znear, zfar, fovx, fovy).T
    full_proj = world_view @ projection
    return world_view, full_proj, eye


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["default", "random"],
        default="default",
        help="default: fixed params, random: random means/colors with fixed scale",
    )
    args = parser.parse_args()

    ext = import_extension()

    n = 2048
    image_width = 768
    image_height = 768
    fovx = math.radians(50.0)
    fovy = math.radians(50.0)
    tan_fovx = math.tan(0.5 * fovx)
    tan_fovy = math.tan(0.5 * fovy)
    viewmatrix_np, projmatrix_np, eye_np = build_look_at_camera(fovx, fovy)

    if args.mode == "random":
        rng = np.random.default_rng()
        xy = rng.uniform(
            low=[64.0, 64.0],
            high=[image_width - 64.0, image_height - 64.0],
            size=(n, 2),
        ).astype(np.float32)
        z = np.full((n, 1), 1.0, dtype=np.float32)
        means3d_np = np.concatenate([xy, z], axis=1)
        colors_np = rng.uniform(0.0, 1.0, size=(n, 3)).astype(np.float32)
    else:
        grid_cols = 64
        grid_rows = 32
        xs = np.linspace(64.0, image_width - 64.0, grid_cols, dtype=np.float32)
        ys = np.linspace(64.0, image_height - 64.0, grid_rows, dtype=np.float32)
        xv, yv = np.meshgrid(xs, ys)
        xy = np.stack([xv.reshape(-1), yv.reshape(-1)], axis=1)[:n]
        z = np.full((n, 1), 1.0, dtype=np.float32)
        means3d_np = np.concatenate([xy, z], axis=1)

        colors_np = np.zeros((n, 3), dtype=np.float32)
        colors_np[:, 0] = xy[:, 0] / float(image_width)
        colors_np[:, 1] = xy[:, 1] / float(image_height)
        colors_np[:, 2] = 1.0 - colors_np[:, 0]

    scales_np = np.full((n, 3), 0.1, dtype=np.float32)
    rotations_np = np.zeros((n, 4), dtype=np.float32)
    rotations_np[:, 0] = 1.0
    opacities_np = np.full((n,), 0.85, dtype=np.float32)

    inputs = {
        "background": mx.array([0.02, 0.02, 0.03], dtype=mx.float32),
        "means3d": mx.array(means3d_np, dtype=mx.float32),
        "colors_precomp": mx.array(colors_np, dtype=mx.float32),
        "opacities": mx.array(opacities_np, dtype=mx.float32),
        "scales": mx.array(scales_np, dtype=mx.float32),
        "rotations": mx.array(rotations_np, dtype=mx.float32),
        "metric_map": mx.zeros((image_width * image_height,), dtype=mx.int32),
        "viewmatrix": mx.array(viewmatrix_np, dtype=mx.float32),
        "projmatrix": mx.array(projmatrix_np, dtype=mx.float32),
        "dc": mx.zeros((0,), dtype=mx.float32),
        "sh": mx.zeros((0,), dtype=mx.float32),
        "campos": mx.array(eye_np[None, :], dtype=mx.float32),
        "viewspace_points": mx.zeros((n, 4), dtype=mx.float32),
    }

    out = ext.rasterize_gaussians_forward(
        inputs,
        image_width,
        image_height,
        16,
        16,
        tan_fovx,
        tan_fovy,
        0,
        1.0,
        1.0,
        False,
        False,
    )

    mx.eval(out["out_color"])
    rgb = to_hwc_rgb(out["out_color"], image_height, image_width)
    rgb = np.clip(rgb, 0.0, 1.0)
    bgr_u8 = (rgb[:, :, ::-1] * 255.0).astype(np.uint8)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(repo_root, "training", "output" ,"forward_test")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{datetime.now().strftime('%Y%m%d%M')}_{args.mode}.png")
    ok = cv2.imwrite(out_path, bgr_u8)
    if not ok:
        raise RuntimeError(f"Failed to write image: {out_path}")

    print("render_2048_smoke ok")
    print("mode:", args.mode)
    print("rendered:", out["rendered"])
    print("num_buckets:", out["num_buckets"])
    print("saved:", out_path)


if __name__ == "__main__":
    main()
