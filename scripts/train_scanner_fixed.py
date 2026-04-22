#!/usr/bin/env python3

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image
from mlx.nn import value_and_grad
from mlx.optimizers import Adam

mx.set_cache_limit(limit=(1 << 31))

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


@dataclass
class ScannerFrame:
    index: int
    image_path: Path
    json_path: Path
    frame: dict | None = None


@dataclass
class TrainCamera:
    viewmatrix: mx.array
    projmatrix: mx.array
    campos: mx.array
    image_width: int
    image_height: int
    tan_fovx: float
    tan_fovy: float


def to_hwc_numpy(chw: mx.array) -> np.ndarray:
    mx.eval(chw)
    arr = np.array(chw)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise RuntimeError(f"Expected CHW with C=3, got {arr.shape}")
    return np.transpose(arr, (1, 2, 0))


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


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1.0e-6, 1.0 - 1.0e-6)
    return np.log(p / (1.0 - p))


def load_ply_positions_colors(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise ImportError(
            "Reading dataset point clouds requires the 'plyfile' package at runtime."
        ) from exc

    ply = PlyData.read(str(path))
    vertices = ply["vertex"]
    points = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)

    colors = None
    names = vertices.data.dtype.names or ()
    if {"red", "green", "blue"}.issubset(names):
        colors = np.stack([vertices["red"], vertices["green"], vertices["blue"]], axis=1).astype(np.float32)
        if colors.max() > 1.0:
            colors = colors / 255.0
        colors = np.clip(colors, 0.0, 1.0)
    return points, colors


def extract_frame_index(path: Path) -> int | None:
    m = re.search(r"frame_(\d+)", path.stem)
    if m is None:
        return None
    return int(m.group(1))


def collect_scanner_frames(
    dataset_dir: Path,
    max_frames: int,
    frame_step: int,
    start_index: int,
) -> list[ScannerFrame]:
    image_files = sorted(dataset_dir.glob("frame_*.jpg"))
    json_files = sorted(dataset_dir.glob("frame_*.json"))

    image_map = {}
    json_map = {}
    for p in image_files:
        idx = extract_frame_index(p)
        if idx is not None:
            image_map[idx] = p
    for p in json_files:
        idx = extract_frame_index(p)
        if idx is not None:
            json_map[idx] = p

    common = sorted(set(image_map.keys()) & set(json_map.keys()))
    common = [i for i in common if i >= start_index]
    if frame_step > 1:
        common = common[::frame_step]
    if max_frames > 0:
        common = common[:max_frames]

    frames = [ScannerFrame(i, image_map[i], json_map[i]) for i in common]
    if not frames:
        raise RuntimeError(f"No scanner frame pairs found in {dataset_dir}")
    return frames


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_axis_transform() -> tuple[np.ndarray, np.ndarray]:
    a = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    a4 = np.eye(4, dtype=np.float32)
    a4[:3, :3] = a
    return a, a4


def compute_normalization(camera_positions: list[np.ndarray]) -> tuple[np.ndarray, float]:
    centers_np = np.stack(camera_positions, axis=0).astype(np.float32)
    translation = centers_np.mean(axis=0)
    denom = np.max(np.abs(centers_np - translation[None, :]))
    scale = 1.0 / float(denom) if denom > 0.0 else 1.0
    return translation, scale


def build_camera_from_scanner_json(
    frame: dict,
    image_width: int,
    image_height: int,
    znear: float = 0.001,
    zfar: float = 1000.0,
) -> TrainCamera:
    width = float(image_width)
    height = float(image_height)
    raw_width = float(frame.get("w", image_width))
    raw_height = float(frame.get("h", image_height))
    sx = width / raw_width
    sy = height / raw_height

    fx = float(frame["fl_x"]) * sx
    fy = float(frame["fl_y"]) * sy
    cx = float(frame["cx"]) * sx
    cy = float(frame["cy"]) * sy

    c2w = np.array(frame["transform_matrix"], dtype=np.float32)
    r = c2w[:3, :3].astype(np.float32)
    t = c2w[:3, 3:4].astype(np.float32)
    r = r @ np.diag([1.0, -1.0, -1.0]).astype(np.float32)

    rinv = r.T
    tinv = (-rinv @ t).astype(np.float32)

    raw_viewmat = np.eye(4, dtype=np.float32)
    raw_viewmat[:3, :3] = rinv
    raw_viewmat[:3, 3:4] = tinv

    fovx = 2.0 * math.atan(width / (2.0 * fx))
    fovy = 2.0 * math.atan(height / (2.0 * fy))

    top = znear * math.tan(0.5 * fovy)
    bottom = -top
    right = znear * math.tan(0.5 * fovx)
    left = -right

    raw_projmat = np.array(
        [
            [2.0 * znear / (right - left), 0.0, (right + left) / (right - left), 0.0],
            [0.0, 2.0 * znear / (top - bottom), (top + bottom) / (top - bottom), 0.0],
            [0.0, 0.0, (zfar + znear) / (zfar - znear), -(zfar * znear) / (zfar - znear)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    raw_full_proj = raw_projmat @ raw_viewmat
    camera_position = t[:, 0].astype(np.float32)

    return TrainCamera(
        viewmatrix=mx.array(raw_viewmat.T, dtype=mx.float32),
        projmatrix=mx.array(raw_full_proj.T, dtype=mx.float32),
        campos=mx.array(camera_position[None, :], dtype=mx.float32),
        image_width=int(image_width),
        image_height=int(image_height),
        tan_fovx=float(math.tan(0.5 * fovx)),
        tan_fovy=float(math.tan(0.5 * fovy)),
    )


def load_target_image(path: Path, width: int, height: int) -> np.ndarray:
    image = Image.open(path)
    rgba = np.array(image.convert("RGBA"), dtype=np.float32) / 255.0
    if rgba.shape[1] != width or rgba.shape[0] != height:
        rgba = np.array(
            Image.fromarray((rgba * 255.0).astype(np.uint8), mode="RGBA").resize(
                (width, height), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0
    bg = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    return rgba[:, :, :3] * rgba[:, :, 3:4] + bg * (1.0 - rgba[:, :, 3:4])


def prepare_dataset(
    dataset_dir: Path,
    width: int,
    height: int,
    max_frames: int,
    frame_step: int,
    start_index: int,
    max_points: int,
    seed: int,
    extra_points_ratio: float,
    extra_points_mode: str,
    extra_points_jitter_scale: float,
) -> tuple[list[TrainCamera], list[mx.array], np.ndarray, np.ndarray, int]:
    a, a4 = make_axis_transform()
    frames = collect_scanner_frames(dataset_dir, max_frames, frame_step, start_index)
    points, colors = load_ply_positions_colors(dataset_dir / "points.ply")
    points = (a @ points.T).T

    camera_positions = []
    for frame in frames:
        raw = load_json(frame.json_path)
        with Image.open(frame.image_path) as img:
            raw_width, raw_height = img.size

        intrinsics = raw.get("intrinsics")
        if intrinsics is None or len(intrinsics) != 9:
            raise RuntimeError(f"Invalid intrinsics in {frame.json_path}")
        pose = raw.get("cameraPoseARFrame")
        if pose is None or len(pose) != 16:
            raise RuntimeError(f"Invalid cameraPoseARFrame in {frame.json_path}")

        c2w_src = np.array(pose, dtype=np.float32).reshape(4, 4)
        c2w = (a4 @ c2w_src).astype(np.float32)
        frame.frame = {
            "w": int(raw_width),
            "h": int(raw_height),
            "file_path": frame.image_path.name,
            "fl_x": float(intrinsics[0]),
            "fl_y": float(intrinsics[4]),
            "cx": float(intrinsics[2]),
            "cy": float(intrinsics[5]),
            "transform_matrix": c2w.tolist(),
        }
        camera_positions.append(c2w[:3, 3])

    translation, norm_scale = compute_normalization(camera_positions)
    points = (points - translation[None, :]) * norm_scale

    rng = np.random.default_rng(seed)
    if max_points > 0 and points.shape[0] > max_points:
        keep = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[keep]
        if colors is not None:
            colors = colors[keep]

    colors_np = colors.astype(np.float32) if colors is not None else np.full_like(points, 0.5, dtype=np.float32)
    base_point_count = int(points.shape[0])
    extra_points = int(round(points.shape[0] * extra_points_ratio))
    if extra_points > 0:
        if extra_points_mode == "surface-jitter":
            source_idx = rng.integers(0, points.shape[0], size=extra_points)
            bbox_min = points.min(axis=0)
            bbox_max = points.max(axis=0)
            diag = float(np.linalg.norm(bbox_max - bbox_min))
            jitter_std = extra_points_jitter_scale * diag
            jitter = rng.normal(loc=0.0, scale=jitter_std, size=(extra_points, 3)).astype(np.float32)
            extra_xyz = points[source_idx] + jitter
            extra_rgb = colors_np[source_idx]
        elif extra_points_mode == "bbox":
            bbox_min = points.min(axis=0)
            bbox_max = points.max(axis=0)
            extra_xyz = rng.uniform(low=bbox_min, high=bbox_max, size=(extra_points, 3)).astype(np.float32)
            source_idx = rng.integers(0, colors_np.shape[0], size=extra_points)
            extra_rgb = colors_np[source_idx]
        else:
            raise ValueError(f"Unsupported --extra-points-mode: {extra_points_mode}")

        points = np.concatenate([points, extra_xyz], axis=0).astype(np.float32)
        colors_np = np.concatenate([colors_np, extra_rgb.astype(np.float32)], axis=0)

    cameras = []
    targets = []
    for f in frames:
        if f.frame is None:
            raise RuntimeError(f"Missing normalized frame metadata for {f.json_path}")
        c2w = np.array(f.frame["transform_matrix"], dtype=np.float32)
        c2w[:3, 3] = (c2w[:3, 3] - translation) * norm_scale
        norm_frame = dict(f.frame)
        norm_frame["transform_matrix"] = c2w.tolist()
        camera = build_camera_from_scanner_json(
            frame=norm_frame,
            image_width=width,
            image_height=height,
        )
        target_hwc = load_target_image(f.image_path, width, height)
        target_chw = np.transpose(target_hwc, (2, 0, 1))
        cameras.append(camera)
        targets.append(mx.array(target_chw, dtype=mx.float32))

    return cameras, targets, points.astype(np.float32), colors_np, base_point_count


class ScannerTrainModel(nn.Module):
    def __init__(
        self,
        means3d: mx.array,
        features_dc: mx.array,
        features_rest: mx.array,
        opacity_logits: mx.array,
        log_scales: mx.array,
        rotations: mx.array,
    ):
        super().__init__()
        self.means3d = means3d
        self.features_dc = features_dc
        self.features_rest = features_rest
        self.opacity_logits = opacity_logits
        self.log_scales = log_scales
        self.rotations = rotations

    @property
    def get_opacities(self) -> mx.array:
        return mx.sigmoid(self.opacity_logits)

    @property
    def get_scales(self) -> mx.array:
        # Trainable scales are stored in log space for optimization stability.
        # Rendering/rasterization expects linear-space scales, so convert here.
        return mx.exp(self.log_scales)

    @property
    def get_rotations(self) -> mx.array:
        return self.rotations / (mx.linalg.norm(self.rotations, axis=1, keepdims=True) + 1.0e-8)


def render_chw(
    ext,
    means3d: mx.array,
    features_dc: mx.array,
    features_rest: mx.array,
    opacities: mx.array,
    scales: mx.array,
    rotations: mx.array,
    camera: TrainCamera,
    background: mx.array,
    sh_degree: int,
) -> mx.array:
    n = means3d.shape[0]
    inputs = {
        "background": background,
        "means3d": means3d,
        "dc": features_dc,
        "sh": features_rest,
        "opacities": opacities,
        "scales": scales,
        "rotations": rotations,
        "metric_map": mx.zeros((camera.image_width * camera.image_height,), dtype=mx.int32),
        "viewmatrix": camera.viewmatrix,
        "projmatrix": camera.projmatrix,
        "campos": camera.campos,
        "viewspace_points": mx.zeros((n, 4), dtype=mx.float32),
    }
    out = ext.rasterize_gaussians(
        inputs,
        camera.image_width,
        camera.image_height,
        16,
        16,
        camera.tan_fovx,
        camera.tan_fovy,
        sh_degree,
        1.0,
        1.0,
        False,
        False,
    )
    out_color = out["out_color"]
    if out_color.size == 0:
        bg = np.array(background, dtype=np.float32)
        return mx.array(
            np.broadcast_to(bg.reshape(3, 1, 1), (3, camera.image_height, camera.image_width)).copy(),
            dtype=mx.float32,
        )
    return to_chw_mx(out_color, camera.image_height, camera.image_width)


def save_side_by_side(target_chw: mx.array, pred_chw: mx.array, out_path: Path) -> None:
    target_hwc = np.clip(to_hwc_numpy(target_chw), 0.0, 1.0)
    pred_hwc = np.clip(to_hwc_numpy(pred_chw), 0.0, 1.0)
    h = target_hwc.shape[0]
    sep = np.zeros((h, 2, 3), dtype=np.float32)
    vis = np.concatenate([target_hwc, sep, pred_hwc], axis=1)
    vis_bgr = (vis[:, :, ::-1] * 255.0).astype(np.uint8)
    ok = cv2.imwrite(str(out_path), vis_bgr)
    if not ok:
        raise RuntimeError(f"Failed to write image: {out_path}")


def init_model(points: np.ndarray, colors: np.ndarray, sh_degree: int) -> ScannerTrainModel:
    n = points.shape[0]
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    # base_scale = max(1.0e-3, 0.01 * diag)
    base_scale = 0.02

    # Keep the trainable scale parameter in log space. Any render/rasterizer path
    # must convert back to linear scale via exp(log_scales).
    log_scales = np.full((n, 3), math.log(base_scale), dtype=np.float32)
    rotations = np.zeros((n, 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    opacity_logits = logit(np.full((n,), 0.02, dtype=np.float32)).astype(np.float32)

    sh_c0 = 0.28209479177387814
    features_dc = ((colors - 0.5) / sh_c0).astype(np.float32)
    rest_coeffs = max(0, (sh_degree + 1) ** 2 - 1)
    features_rest = np.zeros((n, rest_coeffs, 3), dtype=np.float32)

    return ScannerTrainModel(
        means3d=mx.array(points, dtype=mx.float32),
        features_dc=mx.array(features_dc, dtype=mx.float32),
        features_rest=mx.array(features_rest, dtype=mx.float32),
        opacity_logits=mx.array(opacity_logits, dtype=mx.float32),
        log_scales=mx.array(log_scales, dtype=mx.float32),
        rotations=mx.array(rotations, dtype=mx.float32),
    )


def save_as_spz(filename: Path, model: ScannerTrainModel, sh_degree: int) -> bool:
    if spz is None:
        print("[WARN] spz is not available; skip final.spz export")
        return False

    cloud = spz.GaussianCloud()
    cloud.antialiased = True

    # Match the legacy fastgs_mlx export path: SPZ stores the underlying
    # log-scale tensor instead of the linear scale used for rasterization.
    mx.eval(
        model.means3d,
        model.log_scales,
        model.get_rotations,
        model.get_opacities,
        model.features_dc,
        model.features_rest,
    )
    means = np.array(model.means3d, dtype=np.float32)
    means_spz = np.empty_like(means)
    # Only care about scaniverse app preview.
    means_spz[:, 0] = means[:, 0]
    means_spz[:, 1] = -means[:, 2]
    means_spz[:, 2] = means[:, 1]

    scales = np.array(model.log_scales, dtype=np.float32)
    quats = np.array(model.get_rotations, dtype=np.float32)
    opacities = np.array(model.get_opacities, dtype=np.float32)
    features_dc = np.array(model.features_dc, dtype=np.float32)
    features_rest = np.array(model.features_rest, dtype=np.float32)

    cloud.positions = means_spz.flatten().astype(np.float32)
    cloud.scales = scales.flatten().astype(np.float32)
    cloud.rotations = quats.flatten().astype(np.float32)
    cloud.alphas = opacities.flatten().astype(np.float32)
    cloud.colors = features_dc.flatten().astype(np.float32)
    cloud.sh_degree = int(sh_degree)
    cloud.sh = features_rest.flatten().astype(np.float32)

    opts = spz.PackOptions()
    ok = spz.save_spz(cloud, opts, str(filename))
    if not ok:
        raise RuntimeError(f"failed to save spz to {filename}")
    print(f"saved spz: {filename}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/Users/yangdunfu/Downloads/2026_03_01_16_36_14")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=161)
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
    parser.add_argument("--random-background", type=bool, default=False)
    parser.add_argument("--lr-colors", type=float, default=1e-3)
    parser.add_argument("--lr-opacity", type=float, default=1e-3)
    parser.add_argument("--lr-means", type=float, default=3e-3)
    parser.add_argument("--lr-scales", type=float, default=1e-3)
    parser.add_argument("--lr-rotations", type=float, default=1e-3)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.99)
    parser.add_argument("--stage-color-steps", type=int, default=0)
    parser.add_argument("--stage-means-steps", type=int, default=0)
    parser.add_argument("--stage-scales-steps", type=int, default=0)
    parser.add_argument("--stage-rotations-steps", type=int, default=0)
    parser.add_argument("--mse-until", type=int, default=600)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--sh-degree-interval", type=int, default=200)
    parser.add_argument("--debug-scales", action="store_true")
    parser.add_argument("--debug-scale-threshold", type=float, default=0.5)
    parser.add_argument("--debug-scale-growth-ratio", type=float, default=1.25)
    args = parser.parse_args()
    if args.sh_degree < 0 or args.sh_degree > 3:
        raise ValueError("--sh-degree must be between 0 and 3")
    if args.sh_degree_interval <= 0:
        raise ValueError("--sh-degree-interval must be positive")
    if args.extra_points_ratio < 0.0:
        raise ValueError("--extra-points-ratio must be non-negative")
    if args.extra_points_mode not in {"surface-jitter", "bbox"}:
        raise ValueError("--extra-points-mode must be one of: surface-jitter, bbox")
    if args.extra_points_jitter_scale < 0.0:
        raise ValueError("--extra-points-jitter-scale must be non-negative")

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
    if len(cameras) != len(targets):
        raise RuntimeError("Internal error: camera/target length mismatch")
    extra_point_count = int(points.shape[0] - base_point_count)

    model = init_model(points, colors, args.sh_degree)
    betas = (args.adam_beta1, args.adam_beta2)
    means_opt = Adam(learning_rate=args.lr_means, betas=betas)
    dc_opt = Adam(learning_rate=args.lr_colors, betas=betas)
    rest_opt = Adam(learning_rate=args.lr_colors, betas=betas)
    opacity_opt = Adam(learning_rate=args.lr_opacity, betas=betas)
    scales_opt = Adam(learning_rate=args.lr_scales, betas=betas)
    rotations_opt = Adam(learning_rate=args.lr_rotations, betas=betas)

    base_bg = mx.array([0.0, 0.0, 0.0], dtype=mx.float32)

    def loss_fn(model: ScannerTrainModel, camera: TrainCamera, target_chw: mx.array, bg: mx.array, use_l1: mx.array):
        # Rasterization expects linear-space scales, so pass get_scales here.
        pred = render_chw(
            ext=ext,
            means3d=model.means3d,
            features_dc=model.features_dc,
            features_rest=model.features_rest,
            opacities=model.get_opacities,
            scales=model.get_scales,
            rotations=model.get_rotations,
            camera=camera,
            background=bg,
            sh_degree=active_sh_degree,
        )
        diff = pred - target_chw
        l1 = mx.mean(mx.abs(diff))
        mse = mx.mean(diff * diff)
        return mx.where(use_l1, l1, mse)

    loss_and_grad_fn = value_and_grad(model=model, fn=loss_fn)

    repo_root = Path(__file__).resolve().parent.parent
    date_dir = datetime.now().strftime("%Y%m%d_%H_%M")
    out_dir = repo_root / "training" / "output" / "train_scanner_fixed" / date_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "train_state.npz"
    out_best = out_dir / "best_step.png"
    out_spz = out_dir / "final.spz"
    out_final_dir = out_dir / "final"
    out_final_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    best_loss = float("inf")
    best_step = -1
    ema_loss = 0.0
    losses = []
    prev_scale_mean = None
    active_sh_degree = 0

    def _arr_stats(arr: np.ndarray) -> tuple[float, float, float, float]:
        flat = arr.reshape(-1)
        return (
            float(np.min(flat)),
            float(np.max(flat)),
            float(np.mean(flat)),
            float(np.percentile(flat, 95.0)),
        )

    eval_idx = 0
    for step in range(1, args.steps + 1):
        active_sh_degree = min(step // args.sh_degree_interval, args.sh_degree)
        # idx = int(rng.integers(0, len(cameras)))
        idx = step % len(cameras)
        camera = cameras[idx]
        target_chw = targets[idx]

        bg = mx.random.uniform(shape=(3,), low=0.0, high=1.0, dtype=mx.float32) if args.random_background else base_bg
        use_l1 = mx.array(step > args.mse_until, dtype=mx.bool_)
        loss, grads = loss_and_grad_fn(model, camera, target_chw, bg, use_l1)

        
        opacity_opt.update(model, {"opacity_logits": grads["opacity_logits"]})

        if step > args.stage_color_steps:
            dc_opt.update(model, {"features_dc": grads["features_dc"]})
            rest_opt.update(model, {"features_rest": grads["features_rest"]})

        if step > args.stage_means_steps:
            means_opt.update(model, {"means3d": grads["means3d"]})

        if step > args.stage_scales_steps:
            scales_opt.update(model, {"log_scales": grads["log_scales"]})

        if step > args.stage_rotations_steps:
            rotations_opt.update(model, {"rotations": grads["rotations"]})

        mx.eval(loss)
        curr_loss = float(loss.item())

        if curr_loss < best_loss:
            best_loss = curr_loss
            best_step = step
            # Rasterization expects linear-space scales, so pass get_scales here.
            pred_best = render_chw(
                ext=ext,
                means3d=model.means3d,
                features_dc=model.features_dc,
                features_rest=model.features_rest,
                opacities=model.get_opacities,
                scales=model.get_scales,
                rotations=model.get_rotations,
                camera=cameras[eval_idx],
                background=base_bg,
                sh_degree=active_sh_degree,
            )
            save_side_by_side(targets[eval_idx], pred_best, out_best)

        if step == 1:
            ema_loss = curr_loss
        else:
            ema_loss = 0.4 * curr_loss + 0.6 * ema_loss

        if step % args.log_every == 0 or step == args.steps:
            losses.append((step, curr_loss, ema_loss))
            print(
                f"[train] step={step:04d} view={idx:03d} "
                f"sh_degree={active_sh_degree}/{args.sh_degree} "
                f"loss={curr_loss:.6f} ema={ema_loss:.6f}"
            )
            if args.debug_scales:
                mx.eval(model.log_scales, model.get_scales, grads["log_scales"], model.get_opacities)
                log_scales_np = np.array(model.log_scales, dtype=np.float32)
                scales_np = np.array(model.get_scales, dtype=np.float32)
                grads_scales_np = np.array(grads["log_scales"], dtype=np.float32)
                opacity_np = np.array(model.get_opacities, dtype=np.float32)
                ls_min, ls_max, ls_mean, ls_p95 = _arr_stats(log_scales_np)
                s_min, s_max, s_mean, s_p95 = _arr_stats(scales_np)
                g_min, g_max, g_mean, g_p95 = _arr_stats(grads_scales_np)
                o_min, o_max, o_mean, o_p95 = _arr_stats(opacity_np)
                growth = 1.0 if prev_scale_mean is None else (s_mean / (prev_scale_mean + 1.0e-12))
                print(
                    "[debug:scales] "
                    f"step={step:04d} "
                    f"log_scale[min={ls_min:.6f}, max={ls_max:.6f}, mean={ls_mean:.6f}, p95={ls_p95:.6f}] "
                    f"scale[min={s_min:.6f}, max={s_max:.6f}, mean={s_mean:.6f}, p95={s_p95:.6f}] "
                    f"log_grad[min={g_min:.6f}, max={g_max:.6f}, mean={g_mean:.6f}, p95={g_p95:.6f}] "
                    f"opacity[min={o_min:.6f}, max={o_max:.6f}, mean={o_mean:.6f}, p95={o_p95:.6f}] "
                    f"growth_vs_prev={growth:.4f}"
                )
                if s_max > args.debug_scale_threshold:
                    print(
                        "[warn:scales] "
                        f"step={step:04d} scale max {s_max:.6f} exceeds threshold {args.debug_scale_threshold:.6f}"
                    )
                if prev_scale_mean is not None and growth > args.debug_scale_growth_ratio:
                    print(
                        "[warn:scales] "
                        f"step={step:04d} mean scale growth {growth:.4f} exceeds ratio {args.debug_scale_growth_ratio:.4f}"
                    )
                prev_scale_mean = s_mean

        if step % args.save_every == 0 or step == args.steps or step == 0:
            # Rasterization expects linear-space scales, so pass get_scales here.
            pred_eval = render_chw(
                ext=ext,
                means3d=model.means3d,
                features_dc=model.features_dc,
                features_rest=model.features_rest,
                opacities=model.get_opacities,
                scales=model.get_scales,
                rotations=model.get_rotations,
                camera=cameras[eval_idx],
                background=base_bg,
                sh_degree=active_sh_degree,
            )
            out_img = out_dir / f"step_{step:04d}.png"
            save_side_by_side(targets[eval_idx], pred_eval, out_img)
        pass

    # Rasterization expects linear-space scales, so pass get_scales here.
    pred_final = render_chw(
        ext=ext,
        means3d=model.means3d,
        features_dc=model.features_dc,
        features_rest=model.features_rest,
        opacities=model.get_opacities,
        scales=model.get_scales,
        rotations=model.get_rotations,
        camera=cameras[eval_idx],
        background=base_bg,
        sh_degree=active_sh_degree,
    )

    for cam_idx, (camera, target_chw) in enumerate(zip(cameras, targets)):
        pred_camera = render_chw(
            ext=ext,
            means3d=model.means3d,
            features_dc=model.features_dc,
            features_rest=model.features_rest,
            opacities=model.get_opacities,
            scales=model.get_scales,
            rotations=model.get_rotations,
            camera=camera,
            background=base_bg,
            sh_degree=active_sh_degree,
        )
        save_side_by_side(target_chw, pred_camera, out_final_dir / f"final_{cam_idx:04d}.png")

    mx.eval(
        model.means3d,
        model.features_dc,
        model.features_rest,
        model.opacity_logits,
        model.get_opacities,
        model.log_scales,
        model.get_scales,
        model.rotations,
        model.get_rotations,
    )
    # np.savez(
    #     out_npz,
    #     means3d=np.array(model.means3d),
    #     features_dc=np.array(model.features_dc),
    #     features_rest=np.array(model.features_rest),
    #     opacity_logits=np.array(model.opacity_logits),
    #     opacities=np.array(model.get_opacities),
    #     log_scales=np.array(model.log_scales),
    #     scales=np.array(model.get_scales),
    #     rotations=np.array(model.rotations),
    #     normalized_rotations=np.array(model.get_rotations),
    #     losses=np.array(losses, dtype=np.float32),
    #     active_sh_degree=np.array([active_sh_degree], dtype=np.int32),
    #     max_sh_degree=np.array([args.sh_degree], dtype=np.int32),
    #     best_step=np.array([best_step], dtype=np.int32),
    #     best_loss=np.array([best_loss], dtype=np.float32),
    #     eval_target=np.array(targets[eval_idx]),
    #     eval_pred=np.array(pred_final),
    # )
    save_as_spz(out_spz, model, args.sh_degree)

    print("[OK] train_scanner_fixed done")
    print("frames:", len(cameras), "points:", points.shape[0])
    print("base_points:", points.shape[0] - extra_point_count, "extra_points:", extra_point_count)
    print("extra_points_mode:", args.extra_points_mode, "extra_points_ratio:", args.extra_points_ratio)
    print("saved state:", out_npz)
    print("saved best:", out_best)
    print("saved final dir:", out_final_dir)
    print("saved spz:", out_spz)


if __name__ == "__main__":
    main()
