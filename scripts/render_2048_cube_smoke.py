#!/usr/bin/env python3

import argparse
import math
import os
import sys
from datetime import datetime

import cv2
import mlx.core as mx
import numpy as np

try:
    import spz
except Exception:
    spz = None


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
    elif arr.ndim == 3 and arr.shape == (3, h, w):
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.ndim == 2 and arr.shape == (3, h * w):
        arr = np.transpose(arr.reshape(3, h, w), (1, 2, 0))
    elif arr.ndim == 3 and arr.shape == (h, w, 3):
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
    eye=(2.7, 2.2, 3.6),
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


def face_gradient_colors(face_name: str, uv: np.ndarray) -> np.ndarray:
    u = (uv[:, 0] + 1.0) * 0.5
    v = (uv[:, 1] + 1.0) * 0.5

    if face_name == "+x":
        c0 = np.array([1.0, 0.20, 0.15], dtype=np.float32)
        c1 = np.array([1.0, 0.85, 0.25], dtype=np.float32)
        c2 = np.array([1.0, 0.55, 0.75], dtype=np.float32)
    elif face_name == "-x":
        c0 = np.array([0.15, 0.55, 1.0], dtype=np.float32)
        c1 = np.array([0.10, 0.90, 0.85], dtype=np.float32)
        c2 = np.array([0.55, 0.35, 1.0], dtype=np.float32)
    elif face_name == "+y":
        c0 = np.array([1.0, 1.0, 0.25], dtype=np.float32)
        c1 = np.array([1.0, 0.75, 0.10], dtype=np.float32)
        c2 = np.array([0.95, 1.0, 0.65], dtype=np.float32)
    elif face_name == "-y":
        c0 = np.array([0.25, 0.95, 0.25], dtype=np.float32)
        c1 = np.array([0.05, 0.55, 0.20], dtype=np.float32)
        c2 = np.array([0.70, 1.0, 0.45], dtype=np.float32)
    elif face_name == "+z":
        c0 = np.array([1.0, 0.65, 0.10], dtype=np.float32)
        c1 = np.array([1.0, 0.30, 0.05], dtype=np.float32)
        c2 = np.array([1.0, 0.90, 0.55], dtype=np.float32)
    else:
        c0 = np.array([0.35, 0.20, 0.95], dtype=np.float32)
        c1 = np.array([0.85, 0.45, 1.0], dtype=np.float32)
        c2 = np.array([0.10, 0.15, 0.55], dtype=np.float32)

    colors = ((1.0 - u)[:, None] * c0 + u[:, None] * c1)
    colors = 0.6 * colors + 0.4 * ((1.0 - v)[:, None] * c2 + v[:, None] * 1.0)
    return np.clip(colors, 0.0, 1.0).astype(np.float32)


def build_hollow_cube_gaussians(n: int, cube_half_extent: float, face_inset: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    face_names = ["+x", "-x", "+y", "-y", "+z", "-z"]
    counts = [n // 6] * 6
    for i in range(n - sum(counts)):
        counts[i] += 1

    means_parts = []
    colors_parts = []
    face_pos = cube_half_extent - face_inset

    for face_name, count in zip(face_names, counts):
        uv = rng.uniform(-cube_half_extent, cube_half_extent, size=(count, 2)).astype(np.float32)
        if face_name == "+x":
            means = np.column_stack([
                np.full((count,), face_pos, dtype=np.float32),
                uv[:, 0],
                uv[:, 1],
            ])
        elif face_name == "-x":
            means = np.column_stack([
                np.full((count,), -face_pos, dtype=np.float32),
                uv[:, 0],
                uv[:, 1],
            ])
        elif face_name == "+y":
            means = np.column_stack([
                uv[:, 0],
                np.full((count,), face_pos, dtype=np.float32),
                uv[:, 1],
            ])
        elif face_name == "-y":
            means = np.column_stack([
                uv[:, 0],
                np.full((count,), -face_pos, dtype=np.float32),
                uv[:, 1],
            ])
        elif face_name == "+z":
            means = np.column_stack([
                uv[:, 0],
                uv[:, 1],
                np.full((count,), face_pos, dtype=np.float32),
            ])
        else:
            means = np.column_stack([
                uv[:, 0],
                uv[:, 1],
                np.full((count,), -face_pos, dtype=np.float32),
            ])

        means_parts.append(means.astype(np.float32))
        colors_parts.append(face_gradient_colors(face_name, uv))

    means3d = np.concatenate(means_parts, axis=0)
    colors = np.concatenate(colors_parts, axis=0)
    return means3d, colors


def save_as_spz(
    filename: str,
    means3d_np: np.ndarray,
    colors_np: np.ndarray,
    opacities_np: np.ndarray,
    log_scales_np: np.ndarray,
    rotations_np: np.ndarray,
) -> bool:
    if spz is None:
        print("[WARN] spz is not available; skip spz export")
        return False

    cloud = spz.GaussianCloud()
    cloud.antialiased = True
    sh_degree = 0
    rest_coeffs = (sh_degree + 1) ** 2 - 1
    features_rest = np.zeros((means3d_np.shape[0], rest_coeffs, 3), dtype=np.float32)
    cloud.positions = means3d_np.astype(np.float32).flatten()
    cloud.scales = log_scales_np.astype(np.float32).flatten()
    cloud.rotations = rotations_np.astype(np.float32).flatten()
    cloud.alphas = opacities_np.astype(np.float32).flatten()
    cloud.colors = colors_np.astype(np.float32).flatten()
    cloud.sh_degree = sh_degree
    cloud.sh = features_rest.transpose(0, 2, 1).flatten()

    opts = spz.PackOptions()
    ok = spz.save_spz(cloud, opts, filename)
    if not ok:
        raise RuntimeError(f"failed to save spz to {filename}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--image-width", type=int, default=768)
    parser.add_argument("--image-height", type=int, default=768)
    parser.add_argument("--cube-half-extent", type=float, default=1.0)
    parser.add_argument("--face-inset", type=float, default=0.02)
    parser.add_argument("--scale", type=float, default=0.04)
    parser.add_argument("--opacity", type=float, default=0.82)
    args = parser.parse_args()

    ext = import_extension()

    fovx = math.radians(50.0)
    fovy = math.radians(50.0)
    tan_fovx = math.tan(0.5 * fovx)
    tan_fovy = math.tan(0.5 * fovy)
    viewmatrix_np, projmatrix_np, eye_np = build_look_at_camera(fovx, fovy)

    means3d_np, colors_np = build_hollow_cube_gaussians(
        n=args.n,
        cube_half_extent=args.cube_half_extent,
        face_inset=args.face_inset,
    )
    n = means3d_np.shape[0]
    log_scales_np = np.full((n, 3), np.log(args.scale), dtype=np.float32)
    scales_np = np.exp(log_scales_np).astype(np.float32)
    rotations_np = np.zeros((n, 4), dtype=np.float32)
    rotations_np[:, 0] = 1.0
    opacities_np = np.full((n,), args.opacity, dtype=np.float32)

    inputs = {
        "background": mx.array([0.015, 0.016, 0.022], dtype=mx.float32),
        "means3d": mx.array(means3d_np, dtype=mx.float32),
        "colors_precomp": mx.array(colors_np, dtype=mx.float32),
        "opacities": mx.array(opacities_np, dtype=mx.float32),
        "scales": mx.array(scales_np, dtype=mx.float32),
        "rotations": mx.array(rotations_np, dtype=mx.float32),
        "metric_map": mx.zeros((args.image_width * args.image_height,), dtype=mx.int32),
        "viewmatrix": mx.array(viewmatrix_np, dtype=mx.float32),
        "projmatrix": mx.array(projmatrix_np, dtype=mx.float32),
        "dc": mx.zeros((0,), dtype=mx.float32),
        "sh": mx.zeros((0,), dtype=mx.float32),
        "campos": mx.array(eye_np[None, :], dtype=mx.float32),
        "viewspace_points": mx.zeros((n, 4), dtype=mx.float32),
    }

    out = ext.rasterize_gaussians_forward(
        inputs,
        args.image_width,
        args.image_height,
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

    if out["out_color"].size == 0:
        bg = np.array(inputs["background"], dtype=np.float32)
        rgb = np.broadcast_to(bg.reshape(1, 1, 3), (args.image_height, args.image_width, 3)).copy()
    else:
        mx.eval(out["out_color"])
        rgb = to_hwc_rgb(out["out_color"], args.image_height, args.image_width)

    rgb = np.clip(rgb, 0.0, 1.0)
    bgr_u8 = (rgb[:, :, ::-1] * 255.0).astype(np.uint8)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(repo_root, "training", "output", "forward_test")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_png = os.path.join(out_dir, f"{stamp}_cube.png")
    out_spz = os.path.join(out_dir, f"{stamp}_cube.spz")

    ok = cv2.imwrite(out_png, bgr_u8)
    if not ok:
        raise RuntimeError(f"Failed to write image: {out_png}")

    save_as_spz(
        out_spz,
        means3d_np=means3d_np,
        colors_np=colors_np,
        opacities_np=opacities_np,
        log_scales_np=log_scales_np,
        rotations_np=rotations_np,
    )

    print("render_2048_cube_smoke ok")
    print("gaussians:", n)
    print("saved image:", out_png)
    print("saved spz:", out_spz)


if __name__ == "__main__":
    main()
