#!/usr/bin/env python3

import argparse
import math
import os
import sys
from datetime import datetime

import cv2
import mlx.core as mx
import numpy as np


def import_extension():
    try:
        from fastgs_core import _fastgs_core as ext
        return ext
    except Exception:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.insert(0, os.path.join(repo_root, "build"))
        import _fastgs_core as ext
        return ext


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


def make_square_target(height: int, width: int, square_ratio: float = 0.38) -> np.ndarray:
    img = np.ones((height, width, 3), dtype=np.float32)
    side = int(min(height, width) * square_ratio)
    y0 = height // 8
    x0 = width // 8
    y1 = min(height, y0 + side)
    x1 = min(width, x0 + side)
    img[y0:y1, x0:x1, :] = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # red square

    y0b = max(0, height - (height // 8 + side))
    x0b = max(0, width - (width // 8 + side))
    y1b = min(height, y0b + side)
    x1b = min(width, x0b + side)
    img[y0b:y1b, x0b:x1b, :] = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # blue square
    return img


def to_chw_mx(out_color: mx.array, h: int, w: int) -> mx.array:
    shape = tuple(out_color.shape)
    if len(shape) == 1 and shape[0] == h * w * 3:
        return mx.transpose(mx.reshape(out_color, (h, w, 3)), (2, 0, 1))
    if len(shape) == 2 and shape == (h * w, 3):
        return mx.transpose(mx.reshape(out_color, (h, w, 3)), (2, 0, 1))
    if len(shape) == 3 and shape == (3, h, w):
        return out_color
    if len(shape) == 2 and shape == (3, h * w):
        return mx.reshape(out_color, (3, h, w))
    if len(shape) == 3 and shape == (h, w, 3):
        return mx.transpose(out_color, (2, 0, 1))
    raise RuntimeError(f"Unexpected out_color shape: {shape}")


def to_hwc_numpy(chw: mx.array) -> np.ndarray:
    mx.eval(chw)
    arr = np.array(chw)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise RuntimeError(f"Expected CHW with C=3, got {arr.shape}")
    return np.transpose(arr, (1, 2, 0))


def init_gaussians_grid(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cols = int(round(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    xs = np.linspace(-1.0, 1.0, cols, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, rows, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    xy = np.stack([xv.reshape(-1), yv.reshape(-1)], axis=1)[:n]
    z = np.zeros((n, 1), dtype=np.float32)
    means3d = np.concatenate([xy, z], axis=1)

    colors = np.full((n, 3), 0.5, dtype=np.float32)
    opacities = np.full((n,), 0.6, dtype=np.float32)
    scales = np.full((n, 3), 0.08, dtype=np.float32)
    return means3d, colors, opacities, scales


def render_chw(
    ext,
    means3d: mx.array,
    colors_precomp: mx.array,
    opacities: mx.array,
    scales: mx.array,
    rotations: mx.array,
    viewmatrix: mx.array,
    projmatrix: mx.array,
    campos: mx.array,
    image_width: int,
    image_height: int,
    tan_fovx: float,
    tan_fovy: float,
) -> mx.array:
    n = means3d.shape[0]
    inputs = {
        "background": mx.array([1.0, 1.0, 1.0], dtype=mx.float32),
        "means3d": means3d,
        "colors_precomp": colors_precomp,
        "opacities": opacities,
        "scales": scales,
        "rotations": rotations,
        "metric_map": mx.zeros((image_width * image_height,), dtype=mx.int32),
        "viewmatrix": viewmatrix,
        "projmatrix": projmatrix,
        "dc": mx.zeros((0,), dtype=mx.float32),
        "sh": mx.zeros((0,), dtype=mx.float32),
        "campos": campos,
        "viewspace_points": mx.zeros((n, 4), dtype=mx.float32),
    }
    out = ext.rasterize_gaussians(
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
    out_color = out["out_color"]
    if out_color.size == 0:
        return mx.ones((3, image_height, image_width), dtype=mx.float32)
    return to_chw_mx(out_color, image_height, image_width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--random-background", action="store_true")
    parser.add_argument("--lr-colors", type=float, default=5e-2)
    parser.add_argument("--lr-opacity", type=float, default=2e-2)
    parser.add_argument("--lr-means", type=float, default=5e-3)
    parser.add_argument("--lr-scales", type=float, default=2e-3)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    args = parser.parse_args()

    ext = import_extension()

    fovx = math.radians(50.0)
    fovy = math.radians(50.0)
    tan_fovx = math.tan(0.5 * fovx)
    tan_fovy = math.tan(0.5 * fovy)
    viewmatrix_np, projmatrix_np, eye_np = build_look_at_camera(fovx, fovy)

    means3d_np, colors_np, opacities_np, scales_np = init_gaussians_grid(args.n)
    rotations_np = np.zeros((args.n, 4), dtype=np.float32)
    rotations_np[:, 0] = 1.0

    means3d = mx.array(means3d_np, dtype=mx.float32)
    colors = mx.array(colors_np, dtype=mx.float32)
    opacities = mx.array(opacities_np, dtype=mx.float32)
    scales = mx.array(scales_np, dtype=mx.float32)
    rotations = mx.array(rotations_np, dtype=mx.float32)
    viewmatrix = mx.array(viewmatrix_np, dtype=mx.float32)
    projmatrix = mx.array(projmatrix_np, dtype=mx.float32)
    campos = mx.array(eye_np[None, :], dtype=mx.float32)

    target_np = make_square_target(args.height, args.width)
    target_chw = mx.array(np.transpose(target_np, (2, 0, 1)), dtype=mx.float32)

    base_bg = mx.array([1.0, 1.0, 1.0], dtype=mx.float32)

    def loss_fn(m: mx.array, c: mx.array, o: mx.array, s: mx.array, bg: mx.array):
        n = m.shape[0]
        inputs = {
            "background": bg,
            "means3d": m,
            "colors_precomp": c,
            "opacities": o,
            "scales": s,
            "rotations": rotations,
            "metric_map": mx.zeros((args.width * args.height,), dtype=mx.int32),
            "viewmatrix": viewmatrix,
            "projmatrix": projmatrix,
            "dc": mx.zeros((0,), dtype=mx.float32),
            "sh": mx.zeros((0,), dtype=mx.float32),
            "campos": campos,
            "viewspace_points": mx.zeros((n, 4), dtype=mx.float32),
        }
        out = ext.rasterize_gaussians(
            inputs,
            args.width,
            args.height,
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
        out_color = out["out_color"]
        if out_color.size == 0:
            pred = mx.ones((3, args.height, args.width), dtype=mx.float32)
        else:
            pred = to_chw_mx(out_color, args.height, args.width)
        # Follow old train.py style: L1 main loss.
        return mx.mean(mx.abs(pred - target_chw))

    grad_fn = mx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3))
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    date_dir = datetime.now().strftime("%Y%m%d%S")
    out_dir = os.path.join(repo_root, "training", "output", "train_square", date_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_npz = os.path.join(out_dir, "train_state.npz")

    ema_loss = 0.0
    losses = []

    for step in range(1, args.steps + 1):
        bg = mx.random.uniform(shape=(3,), low=0.0, high=1.0, dtype=mx.float32) if args.random_background else base_bg
        loss, (g_means, g_colors, g_opacity, g_scales) = grad_fn(
            means3d, colors, opacities, scales, bg
        )
        means3d = means3d - args.lr_means * g_means
        colors = mx.clip(colors - args.lr_colors * g_colors, 0.0, 1.0)
        opacities = mx.clip(opacities - args.lr_opacity * g_opacity, 0.0, 1.0)
        scales = mx.clip(scales - args.lr_scales * g_scales, 1.0e-4, 2.0)

        if step == 1:
            ema_loss = float(loss.item())
        else:
            ema_loss = 0.4 * float(loss.item()) + 0.6 * ema_loss

        if step % args.log_every == 0 or step == args.steps:
            mx.eval(loss)
            losses.append((step, float(loss.item()), ema_loss))
            print(
                f"[train] step={step:04d} loss={float(loss.item()):.6f} ema={ema_loss:.6f}"
            )

        if step % args.save_every == 0 or step == args.steps:
            pred_chw = render_chw(
                ext=ext,
                means3d=means3d,
                colors_precomp=colors,
                opacities=opacities,
                scales=scales,
                rotations=rotations,
                viewmatrix=viewmatrix,
                projmatrix=projmatrix,
                campos=campos,
                image_width=args.width,
                image_height=args.height,
                tan_fovx=tan_fovx,
                tan_fovy=tan_fovy,
            )
            pred_hwc = np.clip(to_hwc_numpy(pred_chw), 0.0, 1.0)
            target_hwc = np.clip(target_np, 0.0, 1.0)
            vis = np.concatenate([target_hwc, pred_hwc], axis=1)
            vis_bgr = (vis[:, :, ::-1] * 255.0).astype(np.uint8)
            out_img = os.path.join(out_dir, f"step_{step:04d}.png")
            if not cv2.imwrite(out_img, vis_bgr):
                raise RuntimeError(f"Failed to write image: {out_img}")

    pred_chw = render_chw(
        ext=ext,
        means3d=means3d,
        colors_precomp=colors,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        campos=campos,
        image_width=args.width,
        image_height=args.height,
        tan_fovx=tan_fovx,
        tan_fovy=tan_fovy,
    )
    pred_hwc = np.clip(to_hwc_numpy(pred_chw), 0.0, 1.0)
    target_hwc = np.clip(target_np, 0.0, 1.0)

    mx.eval(means3d, colors, opacities, scales)
    np.savez(
        out_npz,
        means3d=np.array(means3d),
        colors=np.array(colors),
        opacities=np.array(opacities),
        scales=np.array(scales),
        target=target_hwc,
        pred=pred_hwc,
        losses=np.array(losses, dtype=np.float32),
    )

    print("[OK] train_square done")
    print("saved state:", out_npz)

if __name__ == "__main__":
    main()
