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


def compute_camera_extent(camera_positions: list[np.ndarray]) -> float:
    centers_np = np.stack(camera_positions, axis=0).astype(np.float32)
    center = centers_np.mean(axis=0)
    dist = np.linalg.norm(centers_np - center[None, :], axis=1)
    radius = float(np.max(dist)) * 1.1
    return max(radius, 1.0e-6)


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
) -> tuple[list[TrainCamera], list[mx.array], np.ndarray, np.ndarray, int, float]:
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

    camera_radius = compute_camera_extent(camera_positions)

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

    return cameras, targets, points.astype(np.float32), colors_np, base_point_count, camera_radius


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
    opacity_logits = logit(np.full((n,), 0.82, dtype=np.float32)).astype(np.float32)

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
        model.opacity_logits,
        model.features_dc,
        model.features_rest,
    )
    means = np.array(model.means3d, dtype=np.float32)
    means_spz = np.empty_like(means)
    # Export directly in SPZ's internal RUB/Three.js coordinate system. The
    # scanner training basis is raw RDF with y/z permuted as [x, z, -y]; this
    # mapping matches the Swift Mac App SPZ export path and avoids asking SPZ to
    # apply an RDF conversion that cannot represent the y/z permutation.
    means_spz[:, 0] = -means[:, 0]
    means_spz[:, 1] = -means[:, 2]
    means_spz[:, 2] = -means[:, 1]

    scales = np.array(model.log_scales, dtype=np.float32)
    quats = np.array(model.get_rotations, dtype=np.float32)
    rot_mats = quaternions_wxyz_to_rotation_matrices(quats)
    spz_rub_from_training = np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    rot_mats_spz = spz_rub_from_training @ rot_mats
    quats_spz = rotation_matrices_to_quaternions_wxyz(rot_mats_spz)
    opacity_logits = np.array(model.opacity_logits, dtype=np.float32)
    features_dc = np.array(model.features_dc, dtype=np.float32)
    features_rest = np.array(model.features_rest, dtype=np.float32)

    cloud.positions = means_spz.flatten().astype(np.float32)
    cloud.scales = scales.flatten().astype(np.float32)
    cloud.rotations = quats_spz[:, [1, 2, 3, 0]].flatten().astype(np.float32)
    cloud.alphas = opacity_logits.flatten().astype(np.float32)
    cloud.colors = features_dc.flatten().astype(np.float32)
    cloud.sh_degree = int(sh_degree)
    cloud.sh = features_rest.flatten().astype(np.float32)

    opts = spz.PackOptions()
    ok = spz.save_spz(cloud, opts, str(filename))
    if not ok:
        raise RuntimeError(f"failed to save spz to {filename}")
    print(f"saved spz: {filename}")
    return True



@dataclass
class FastGSDensificationState:
    max_radii2d: np.ndarray
    xyz_grad_accum: np.ndarray
    xyz_grad_accum_abs: np.ndarray
    denom: np.ndarray
    tmp_radii: np.ndarray | None = None


@dataclass
class OptimizerPolicyConfig:
    means_lr: float
    dc_lr: float
    sh_lr: float
    opacity_lr: float
    scaling_lr: float
    rotation_lr: float
    position_lr_init: float | None = None
    position_lr_final: float = 1.6e-6
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30000
    spatial_lr_scale: float = 1.0
    betas: tuple[float, float] = (0.9, 0.99)
    sh_lr_divisor: float = 20.0


def get_expon_lr_func(
    lr_init: float,
    lr_final: float,
    lr_delay_steps: int = 0,
    lr_delay_mult: float = 1.0,
    max_steps: int = 1000000,
):
    def helper(step: int) -> float:
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1.0 - lr_delay_mult) * math.sin(
                0.5 * math.pi * min(max(step / lr_delay_steps, 0.0), 1.0)
            )
        else:
            delay_rate = 1.0
        t = min(max(step / max_steps, 0.0), 1.0)
        log_lerp = math.exp(math.log(lr_init) * (1.0 - t) + math.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


class ScannerFastGSOptimizerPolicy:
    def __init__(self, cfg: OptimizerPolicyConfig):
        self.cfg = cfg
        self.main_optimizers = {
            "means3d": Adam(learning_rate=cfg.means_lr, betas=cfg.betas),
            "features_dc": Adam(learning_rate=cfg.dc_lr, betas=cfg.betas),
            "opacity_logits": Adam(learning_rate=cfg.opacity_lr, betas=cfg.betas),
            "log_scales": Adam(learning_rate=cfg.scaling_lr, betas=cfg.betas),
            "rotations": Adam(learning_rate=cfg.rotation_lr, betas=cfg.betas),
        }
        self.sh_optimizer = Adam(
            learning_rate=cfg.sh_lr / cfg.sh_lr_divisor,
            betas=cfg.betas,
        )
        xyz_lr_init = cfg.position_lr_init if cfg.position_lr_init is not None else cfg.means_lr
        self.xyz_scheduler = get_expon_lr_func(
            lr_init=xyz_lr_init * cfg.spatial_lr_scale,
            lr_final=cfg.position_lr_final * cfg.spatial_lr_scale,
            lr_delay_mult=cfg.position_lr_delay_mult,
            max_steps=cfg.position_lr_max_steps,
        )

    @property
    def all_optimizers(self):
        return {**self.main_optimizers, "features_rest": self.sh_optimizer}

    def _take_rows(self, array: mx.array, indices: mx.array) -> mx.array:
        if indices.shape[0] == 0:
            empty_shape = list(array.shape)
            empty_shape[0] = 0
            return mx.zeros(tuple(empty_shape), array.dtype)
        return mx.take(array, indices, axis=0)

    def _resize_state_like(self, optimizer: Adam, name: str, indices: mx.array | None = None, appended: mx.array | None = None):
        state = optimizer.state.get(name)
        if not isinstance(state, dict):
            return
        for key in ("m", "v"):
            if key not in state:
                continue
            value = state[key]
            if indices is not None:
                value = self._take_rows(value, indices)
            if appended is not None:
                value = mx.concatenate([value, mx.zeros_like(appended)], axis=0)
            state[key] = value

    def prune_states_np(self, keep_mask: np.ndarray):
        keep_indices = mx.array(np.flatnonzero(keep_mask).astype(np.uint32), dtype=mx.uint32)
        for name, optimizer in self.main_optimizers.items():
            self._resize_state_like(optimizer, name, indices=keep_indices)
        self._resize_state_like(self.sh_optimizer, "features_rest", indices=keep_indices)

    def append_states_np(self, appended_tensors: dict[str, np.ndarray]):
        for name, optimizer in self.main_optimizers.items():
            appended = appended_tensors.get(name)
            if appended is not None and appended.shape[0] > 0:
                self._resize_state_like(optimizer, name, appended=mx.array(appended))
        appended_rest = appended_tensors.get("features_rest")
        if appended_rest is not None and appended_rest.shape[0] > 0:
            self._resize_state_like(self.sh_optimizer, "features_rest", appended=mx.array(appended_rest))

    def replace_state_np(self, name: str, new_value: np.ndarray):
        optimizer = self.main_optimizers.get(name)
        if optimizer is None and name == "features_rest":
            optimizer = self.sh_optimizer
        if optimizer is None:
            return
        state = optimizer.state.get(name)
        if not isinstance(state, dict):
            return
        new_mx = mx.array(new_value)
        if "m" in state:
            state["m"] = mx.zeros_like(new_mx)
        if "v" in state:
            state["v"] = mx.zeros_like(new_mx)

    def update_learning_rate(self, iteration: int) -> float:
        lr = self.xyz_scheduler(iteration)
        self.main_optimizers["means3d"].learning_rate = lr
        return lr

    def _should_step_main(self, iteration: int) -> bool:
        if iteration <= 15000:
            return True
        if iteration <= 20000:
            return iteration % 32 == 0
        return iteration % 64 == 0

    def _should_step_sh(self, iteration: int) -> bool:
        if iteration <= 15000:
            return iteration % 16 == 0
        if iteration <= 20000:
            return iteration % 32 == 0
        return iteration % 64 == 0

    def apply_gradients(self, model: ScannerTrainModel, grads: dict[str, mx.array], iteration: int):
        if self._should_step_main(iteration):
            for name, optimizer in self.main_optimizers.items():
                grad = grads.get(name)
                if grad is not None:
                    optimizer.update(model, {name: grad})
        if self._should_step_sh(iteration):
            grad = grads.get("features_rest")
            if grad is not None:
                self.sh_optimizer.update(model, {"features_rest": grad})


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def quat_to_rotmat_np(quat: np.ndarray) -> np.ndarray:
    q = quat / np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1.0e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rot = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    rot[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rot[:, 0, 1] = 2.0 * (x * y - z * w)
    rot[:, 0, 2] = 2.0 * (x * z + y * w)
    rot[:, 1, 0] = 2.0 * (x * y + z * w)
    rot[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rot[:, 1, 2] = 2.0 * (y * z - x * w)
    rot[:, 2, 0] = 2.0 * (x * z - y * w)
    rot[:, 2, 1] = 2.0 * (y * z + x * w)
    rot[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rot


def make_densification_state(num_points: int) -> FastGSDensificationState:
    return FastGSDensificationState(
        max_radii2d=np.zeros((num_points,), dtype=np.float32),
        xyz_grad_accum=np.zeros((num_points, 1), dtype=np.float32),
        xyz_grad_accum_abs=np.zeros((num_points, 1), dtype=np.float32),
        denom=np.zeros((num_points, 1), dtype=np.float32),
    )


def render_pkg(
    ext,
    model: ScannerTrainModel,
    camera: TrainCamera,
    background: mx.array,
    sh_degree: int,
    metric_map: mx.array | None = None,
    get_flag: bool = False,
) -> dict:
    n = model.means3d.shape[0]
    if metric_map is None:
        metric_map = mx.zeros((camera.image_width * camera.image_height,), dtype=mx.int32)
    inputs = {
        "background": background,
        "means3d": model.means3d,
        "dc": model.features_dc,
        "sh": model.features_rest,
        "opacities": model.get_opacities,
        "scales": model.get_scales,
        "rotations": model.get_rotations,
        "metric_map": metric_map,
        "viewmatrix": camera.viewmatrix,
        "projmatrix": camera.projmatrix,
        "campos": camera.campos,
        "viewspace_points": mx.zeros((n, 4), dtype=mx.float32),
    }
    return ext.rasterize_gaussians(
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
        get_flag,
    )


def l1_map_chw(pred: mx.array, target: mx.array) -> mx.array:
    return mx.mean(mx.abs(pred - target), axis=0)


def normalized_positive_map(x: mx.array) -> mx.array:
    x_min = mx.min(x)
    x_max = mx.max(x)
    denom = mx.maximum(x_max - x_min, mx.array(1.0e-6, dtype=x.dtype))
    return (x - x_min) / denom


def sample_camera_indices(rng: np.random.Generator, num_cameras: int, sample_count: int) -> np.ndarray:
    count = min(sample_count, num_cameras)
    return rng.choice(num_cameras, size=count, replace=False)


def compute_gaussian_scores_fastgs(
    ext,
    model: ScannerTrainModel,
    cameras: list[TrainCamera],
    targets: list[mx.array],
    camera_indices: np.ndarray,
    background: mx.array,
    sh_degree: int,
    loss_thresh: float,
    densify: bool,
) -> tuple[np.ndarray | None, np.ndarray]:
    full_metric_counts = None
    full_metric_score = None

    for idx in camera_indices.tolist():
        camera = cameras[idx]
        target = targets[idx]

        pred = render_chw(
            ext=ext,
            means3d=model.means3d,
            features_dc=model.features_dc,
            features_rest=model.features_rest,
            opacities=model.get_opacities,
            scales=model.get_scales,
            rotations=model.get_rotations,
            camera=camera,
            background=background,
            sh_degree=sh_degree,
        )
        loss_map = normalized_positive_map(l1_map_chw(pred, target))
        metric_map = mx.array(mx.reshape(loss_map > loss_thresh, (-1,)), dtype=mx.int32)

        second = render_pkg(ext, model, camera, background, sh_degree, metric_map=metric_map, get_flag=True)
        mx.eval(second["metric_count"])

        photometric_loss = float(mx.mean(mx.abs(pred - target)).item())
        metric_count = np.array(second["metric_count"], dtype=np.float32)

        if densify:
            if full_metric_counts is None:
                full_metric_counts = metric_count.copy()
            else:
                full_metric_counts += metric_count

        if full_metric_score is None:
            full_metric_score = photometric_loss * metric_count
        else:
            full_metric_score += photometric_loss * metric_count

        mx.eval(pred)

    if full_metric_score is None:
        zeros = np.zeros((model.means3d.shape[0],), dtype=np.float32)
        return (zeros if densify else None), zeros

    score_min = float(np.min(full_metric_score))
    score_max = float(np.max(full_metric_score))
    pruning_score = (full_metric_score - score_min) / max(score_max - score_min, 1.0e-6)
    importance_score = None
    if densify and full_metric_counts is not None:
        importance_score = np.floor(full_metric_counts / max(len(camera_indices), 1)).astype(np.float32)
    return importance_score, pruning_score.astype(np.float32)


def apply_param_arrays(
    model: ScannerTrainModel,
    means3d: np.ndarray,
    features_dc: np.ndarray,
    features_rest: np.ndarray,
    opacity_logits: np.ndarray,
    log_scales: np.ndarray,
    rotations: np.ndarray,
) -> None:
    model.means3d = mx.array(means3d, dtype=mx.float32)
    model.features_dc = mx.array(features_dc, dtype=mx.float32)
    model.features_rest = mx.array(features_rest, dtype=mx.float32)
    model.opacity_logits = mx.array(opacity_logits, dtype=mx.float32)
    model.log_scales = mx.array(log_scales, dtype=mx.float32)
    model.rotations = mx.array(rotations, dtype=mx.float32)


def capture_model_np(model: ScannerTrainModel) -> dict[str, np.ndarray]:
    mx.eval(
        model.means3d,
        model.features_dc,
        model.features_rest,
        model.opacity_logits,
        model.log_scales,
        model.rotations,
        model.get_opacities,
        model.get_scales,
        model.get_rotations,
    )
    return {
        "means3d": np.array(model.means3d, dtype=np.float32),
        "features_dc": np.array(model.features_dc, dtype=np.float32),
        "features_rest": np.array(model.features_rest, dtype=np.float32),
        "opacity_logits": np.array(model.opacity_logits, dtype=np.float32),
        "log_scales": np.array(model.log_scales, dtype=np.float32),
        "rotations": np.array(model.rotations, dtype=np.float32),
        "opacities": np.array(model.get_opacities, dtype=np.float32),
        "scales": np.array(model.get_scales, dtype=np.float32),
    }


class ScannerGaussianOps:
    def __init__(self, optimizer_policy: ScannerFastGSOptimizerPolicy | None = None):
        self.optimizer_policy = optimizer_policy

    def reset_densification_buffers(self, state: FastGSDensificationState, num_points: int) -> None:
        reset_densification_buffers(state, num_points)

    def update_densification_stats(
        self,
        state: FastGSDensificationState,
        radii_np: np.ndarray,
        d_viewspace_np: np.ndarray,
    ) -> None:
        visible = radii_np > 0
        state.max_radii2d[: visible.shape[0]][visible] = np.maximum(
            state.max_radii2d[: visible.shape[0]][visible],
            radii_np[visible],
        )
        state.xyz_grad_accum[visible] += np.linalg.norm(d_viewspace_np[visible, :2], axis=1, keepdims=True)
        state.xyz_grad_accum_abs[visible] += np.linalg.norm(d_viewspace_np[visible, 2:], axis=1, keepdims=True)
        state.denom[visible] += 1.0
        state.tmp_radii = radii_np.copy()

    def append_new_points(
        self,
        model: ScannerTrainModel,
        state: FastGSDensificationState,
        new_data: dict[str, np.ndarray],
    ) -> None:
        append_new_points(model, state, new_data, optimizer_policy=self.optimizer_policy)

    def prune_points(
        self,
        model: ScannerTrainModel,
        state: FastGSDensificationState,
        prune_mask: np.ndarray,
    ) -> None:
        prune_points(model, state, prune_mask, optimizer_policy=self.optimizer_policy)

    def reset_opacity_logits(self, model: ScannerTrainModel, reset_value: float) -> None:
        reset_opacity_logits(model, reset_value, optimizer_policy=self.optimizer_policy)

    def cap_opacity_logits(self, model: ScannerTrainModel, opacity_cap: float) -> None:
        cap_opacity_logits(model, opacity_cap, optimizer_policy=self.optimizer_policy)

    def densify_and_prune_fastgs(
        self,
        model: ScannerTrainModel,
        state: FastGSDensificationState,
        args,
        scene_extent: float,
        importance_score: np.ndarray,
        pruning_score: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        return densify_and_prune_fastgs(
            model,
            state,
            args,
            scene_extent,
            importance_score,
            pruning_score,
            rng,
            optimizer_policy=self.optimizer_policy,
        )

    def final_prune_fastgs(
        self,
        model: ScannerTrainModel,
        state: FastGSDensificationState,
        min_opacity: float,
        pruning_score: np.ndarray,
        score_thresh: float,
        min_gaussians: int,
        dry_run: bool = False,
    ) -> dict[str, int]:
        return final_prune_fastgs(
            model,
            state,
            min_opacity,
            pruning_score,
            score_thresh,
            min_gaussians,
            optimizer_policy=self.optimizer_policy,
            dry_run=dry_run,
        )


def reset_densification_buffers(state: FastGSDensificationState, num_points: int) -> None:
    state.max_radii2d = np.zeros((num_points,), dtype=np.float32)
    state.xyz_grad_accum = np.zeros((num_points, 1), dtype=np.float32)
    state.xyz_grad_accum_abs = np.zeros((num_points, 1), dtype=np.float32)
    state.denom = np.zeros((num_points, 1), dtype=np.float32)
    state.tmp_radii = None


def append_new_points(
    model: ScannerTrainModel,
    state: FastGSDensificationState,
    new_data: dict[str, np.ndarray],
    optimizer_policy: ScannerFastGSOptimizerPolicy | None = None,
) -> None:
    if new_data["means3d"].shape[0] == 0:
        return
    current = capture_model_np(model)
    apply_param_arrays(
        model,
        means3d=np.concatenate([current["means3d"], new_data["means3d"]], axis=0),
        features_dc=np.concatenate([current["features_dc"], new_data["features_dc"]], axis=0),
        features_rest=np.concatenate([current["features_rest"], new_data["features_rest"]], axis=0),
        opacity_logits=np.concatenate([current["opacity_logits"], new_data["opacity_logits"]], axis=0),
        log_scales=np.concatenate([current["log_scales"], new_data["log_scales"]], axis=0),
        rotations=np.concatenate([current["rotations"], new_data["rotations"]], axis=0),
    )
    if optimizer_policy is not None:
        optimizer_policy.append_states_np(
            {
                "means3d": new_data["means3d"],
                "features_dc": new_data["features_dc"],
                "features_rest": new_data["features_rest"],
                "opacity_logits": new_data["opacity_logits"],
                "log_scales": new_data["log_scales"],
                "rotations": new_data["rotations"],
            }
        )
    state.max_radii2d = np.concatenate([state.max_radii2d, new_data["tmp_radii"]], axis=0)
    reset_densification_buffers(state, model.means3d.shape[0])


def prune_points(
    model: ScannerTrainModel,
    state: FastGSDensificationState,
    prune_mask: np.ndarray,
    optimizer_policy: ScannerFastGSOptimizerPolicy | None = None,
) -> None:
    if prune_mask.size == 0 or not np.any(prune_mask):
        return
    keep = ~prune_mask
    current = capture_model_np(model)
    apply_param_arrays(
        model,
        means3d=current["means3d"][keep],
        features_dc=current["features_dc"][keep],
        features_rest=current["features_rest"][keep],
        opacity_logits=current["opacity_logits"][keep],
        log_scales=current["log_scales"][keep],
        rotations=current["rotations"][keep],
    )
    if optimizer_policy is not None:
        optimizer_policy.prune_states_np(keep)
    state.max_radii2d = state.max_radii2d[keep]
    state.xyz_grad_accum = state.xyz_grad_accum[keep]
    state.xyz_grad_accum_abs = state.xyz_grad_accum_abs[keep]
    state.denom = state.denom[keep]
    if state.tmp_radii is not None:
        state.tmp_radii = state.tmp_radii[keep]


def densify_and_clone_fastgs(
    model: ScannerTrainModel,
    state: FastGSDensificationState,
    source: dict[str, np.ndarray],
    source_tmp_radii: np.ndarray,
    metric_mask: np.ndarray,
    clone_filter: np.ndarray,
    optimizer_policy: ScannerFastGSOptimizerPolicy | None = None,
) -> int:
    selected = metric_mask & clone_filter
    if not np.any(selected):
        return 0
    append_new_points(
        model,
        state,
        {
            "means3d": source["means3d"][selected],
            "features_dc": source["features_dc"][selected],
            "features_rest": source["features_rest"][selected],
            "opacity_logits": source["opacity_logits"][selected],
            "log_scales": source["log_scales"][selected],
            "rotations": source["rotations"][selected],
            "tmp_radii": source_tmp_radii[selected],
        },
        optimizer_policy=optimizer_policy,
    )
    return int(np.sum(selected))


def densify_and_split_fastgs(
    model: ScannerTrainModel,
    state: FastGSDensificationState,
    source: dict[str, np.ndarray],
    source_tmp_radii: np.ndarray,
    metric_mask: np.ndarray,
    split_filter: np.ndarray,
    rng: np.random.Generator,
    split_factor: int,
    optimizer_policy: ScannerFastGSOptimizerPolicy | None = None,
) -> tuple[int, int]:
    selected = metric_mask & split_filter
    if not np.any(selected):
        return 0, 0
    means = source["means3d"][selected]
    scales = source["scales"][selected]
    log_scales = source["log_scales"][selected]
    rotations = source["rotations"][selected]
    rotmats = quat_to_rotmat_np(rotations)

    repeated_scales = np.repeat(scales, split_factor, axis=0)
    repeated_rotmats = np.repeat(rotmats, split_factor, axis=0)
    local_samples = rng.normal(loc=0.0, scale=repeated_scales).astype(np.float32)
    offsets = np.einsum("nij,nj->ni", repeated_rotmats, local_samples)

    repeated_means = np.repeat(means, split_factor, axis=0)
    repeated_log_scales = np.repeat(log_scales, split_factor, axis=0)
    new_scales = np.log(np.exp(repeated_log_scales) / (0.8 * float(split_factor)))

    append_new_points(
        model,
        state,
        {
            "means3d": repeated_means + offsets,
            "features_dc": np.repeat(source["features_dc"][selected], split_factor, axis=0),
            "features_rest": np.repeat(source["features_rest"][selected], split_factor, axis=0),
            "opacity_logits": np.repeat(source["opacity_logits"][selected], split_factor, axis=0),
            "log_scales": new_scales.astype(np.float32),
            "rotations": np.repeat(rotations, split_factor, axis=0),
            "tmp_radii": np.repeat(source_tmp_radii[selected], split_factor, axis=0),
        },
        optimizer_policy=optimizer_policy,
    )

    current_count = int(model.means3d.shape[0])
    source_count = int(selected.shape[0])
    selected_count = int(np.sum(selected))
    appended_count = selected_count * split_factor
    existing_extra = current_count - source_count - appended_count
    if existing_extra < 0:
        raise RuntimeError(
            f"Invalid split state sizes: current={current_count}, source={source_count}, appended={appended_count}"
        )
    prune_mask = np.concatenate(
        [selected, np.zeros((existing_extra + appended_count,), dtype=bool)],
        axis=0,
    )
    prune_points(model, state, prune_mask, optimizer_policy=optimizer_policy)
    return selected_count, appended_count


def cap_opacity_logits(
    model: ScannerTrainModel,
    opacity_cap: float,
    optimizer_policy: ScannerFastGSOptimizerPolicy | None = None,
) -> None:
    current = capture_model_np(model)
    capped = np.minimum(current["opacities"], opacity_cap).astype(np.float32)
    current["opacity_logits"] = logit(capped).astype(np.float32)
    apply_param_arrays(
        model,
        means3d=current["means3d"],
        features_dc=current["features_dc"],
        features_rest=current["features_rest"],
        opacity_logits=current["opacity_logits"],
        log_scales=current["log_scales"],
        rotations=current["rotations"],
    )
    if optimizer_policy is not None:
        optimizer_policy.replace_state_np("opacity_logits", current["opacity_logits"])


def reset_opacity_logits(
    model: ScannerTrainModel,
    reset_value: float,
    optimizer_policy: ScannerFastGSOptimizerPolicy | None = None,
) -> None:
    current = capture_model_np(model)
    capped = np.minimum(current["opacities"], reset_value).astype(np.float32)
    current["opacity_logits"] = logit(capped).astype(np.float32)
    apply_param_arrays(
        model,
        means3d=current["means3d"],
        features_dc=current["features_dc"],
        features_rest=current["features_rest"],
        opacity_logits=current["opacity_logits"],
        log_scales=current["log_scales"],
        rotations=current["rotations"],
    )
    if optimizer_policy is not None:
        optimizer_policy.replace_state_np("opacity_logits", current["opacity_logits"])


def save_step_preview(
    ext,
    model: ScannerTrainModel,
    cameras: list[TrainCamera],
    targets: list[mx.array],
    background: mx.array,
    sh_degree: int,
    eval_idx: int,
    out_path: Path,
) -> None:
    n = len(cameras)
    if n < 8:
        pred_eval = render_chw(
            ext=ext,
            means3d=model.means3d,
            features_dc=model.features_dc,
            features_rest=model.features_rest,
            opacities=model.get_opacities,
            scales=model.get_scales,
            rotations=model.get_rotations,
            camera=cameras[eval_idx],
            background=background,
            sh_degree=sh_degree,
        )
        save_side_by_side(targets[eval_idx], pred_eval, out_path)
        return

    sampled_indices = [min(n - 1, int(math.floor(k * n / 8))) for k in range(1, 9)]
    tiles = [np.clip(to_hwc_numpy(targets[0]), 0.0, 1.0)]
    for sampled_idx in sampled_indices:
        pred = render_chw(
            ext=ext,
            means3d=model.means3d,
            features_dc=model.features_dc,
            features_rest=model.features_rest,
            opacities=model.get_opacities,
            scales=model.get_scales,
            rotations=model.get_rotations,
            camera=cameras[sampled_idx],
            background=background,
            sh_degree=sh_degree,
        )
        tiles.append(np.clip(to_hwc_numpy(pred), 0.0, 1.0))

    rows = [np.concatenate(tiles[row * 3 : (row + 1) * 3], axis=1) for row in range(3)]
    grid = np.concatenate(rows, axis=0)
    grid_bgr = (grid[:, :, ::-1] * 255.0).astype(np.uint8)
    ok = cv2.imwrite(str(out_path), grid_bgr)
    if not ok:
        raise RuntimeError(f"Failed to write image: {out_path}")


def make_optimizer_policy(args, spatial_lr_scale: float) -> ScannerFastGSOptimizerPolicy:
    return ScannerFastGSOptimizerPolicy(
        OptimizerPolicyConfig(
            means_lr=args.lr_means,
            dc_lr=args.lr_colors,
            sh_lr=args.lr_colors,
            opacity_lr=args.lr_opacity,
            scaling_lr=args.lr_scales,
            rotation_lr=args.lr_rotations,
            position_lr_init=args.lr_means,
            position_lr_final=1.6e-6,
            position_lr_delay_mult=0.01,
            position_lr_max_steps=args.steps,
            spatial_lr_scale=spatial_lr_scale,
            betas=(args.adam_beta1, args.adam_beta2),
        )
    )


def densify_and_prune_fastgs(
    model: ScannerTrainModel,
    state: FastGSDensificationState,
    args,
    scene_extent: float,
    importance_score: np.ndarray,
    pruning_score: np.ndarray,
    rng: np.random.Generator,
    optimizer_policy: ScannerFastGSOptimizerPolicy | None = None,
) -> dict[str, int]:
    denom = np.maximum(state.denom, 1.0)
    grad_vars = state.xyz_grad_accum / denom
    grads_abs = state.xyz_grad_accum_abs / denom

    current = capture_model_np(model)
    source_tmp_radii = state.tmp_radii
    if source_tmp_radii is None or source_tmp_radii.shape[0] != current["means3d"].shape[0]:
        source_tmp_radii = np.zeros((current["means3d"].shape[0],), dtype=np.float32)
    grad_qualifiers = np.linalg.norm(grad_vars, axis=1) >= args.grad_thresh
    grad_qualifiers_abs = np.linalg.norm(grads_abs, axis=1) >= args.grad_abs_thresh
    max_scale = np.max(current["scales"], axis=1)
    clone_qualifiers = max_scale <= args.dense * scene_extent
    split_qualifiers = max_scale > args.dense * scene_extent
    metric_mask = importance_score > args.importance_score_threshold
    clone_candidates = int(np.sum(metric_mask & clone_qualifiers & grad_qualifiers))
    split_candidates = int(np.sum(metric_mask & split_qualifiers & grad_qualifiers_abs))

    cloned = densify_and_clone_fastgs(
        model,
        state,
        current,
        source_tmp_radii,
        metric_mask,
        clone_qualifiers & grad_qualifiers,
        optimizer_policy=optimizer_policy,
    )
    split_sources, split_children = densify_and_split_fastgs(
        model,
        state,
        current,
        source_tmp_radii,
        metric_mask,
        split_qualifiers & grad_qualifiers_abs,
        rng,
        args.split_factor,
        optimizer_policy=optimizer_policy,
    )

    current = capture_model_np(model)
    if pruning_score.size < current["opacities"].shape[0]:
        pruning_score = np.pad(
            pruning_score,
            (0, current["opacities"].shape[0] - pruning_score.size),
            mode="constant",
        )
    opacity_prune_mask = current["opacities"] < args.min_opacity
    prune_mask = opacity_prune_mask.copy()
    screen_prune_mask = np.zeros_like(prune_mask)
    if args.max_screen_size > 0:
        screen_prune_mask = state.max_radii2d > args.max_screen_size
        prune_mask = prune_mask | screen_prune_mask
    world_prune_mask = np.zeros_like(prune_mask)
    if args.max_world_scale_factor > 0.0:
        world_prune_mask = np.max(current["scales"], axis=1) > args.max_world_scale_factor * scene_extent
        prune_mask = prune_mask | world_prune_mask

    to_remove = int(np.sum(prune_mask))
    remove_budget = int(args.prune_budget_factor * to_remove)
    actual_removed = 0
    if not args.no_prune_gaussians and remove_budget > 0 and pruning_score.size > 0:
        weights = np.zeros_like(pruning_score, dtype=np.float32)
        weights[:] = 1.0 / (1.0e-6 + (1.0 - pruning_score))
        candidate_ids = np.flatnonzero(prune_mask)
        if candidate_ids.size > 0:
            cand_weights = weights[candidate_ids]
            cand_weights = cand_weights / max(float(np.sum(cand_weights)), 1.0e-6)
            chosen = rng.choice(candidate_ids, size=min(remove_budget, candidate_ids.size), replace=False, p=cand_weights)
            final_prune = np.zeros_like(prune_mask)
            final_prune[chosen] = True
            final_prune &= prune_mask
            actual_removed = int(np.sum(final_prune))
            prune_points(model, state, final_prune, optimizer_policy=optimizer_policy)

    cap_opacity_logits(model, args.opacity_cap_after_densify, optimizer_policy=optimizer_policy)
    return {
        "metric_hits": int(np.sum(metric_mask)),
        "clone_candidates": clone_candidates,
        "split_candidates": split_candidates,
        "cloned": cloned,
        "split_sources": split_sources,
        "split_children": split_children,
        "opacity_prune_candidates": int(np.sum(opacity_prune_mask)),
        "screen_prune_candidates": int(np.sum(screen_prune_mask)),
        "world_prune_candidates": int(np.sum(world_prune_mask)),
        "total_prune_candidates": to_remove,
        "prune_budget": remove_budget,
        "actual_removed": actual_removed,
    }


def final_prune_fastgs(
    model: ScannerTrainModel,
    state: FastGSDensificationState,
    min_opacity: float,
    pruning_score: np.ndarray,
    score_thresh: float,
    min_gaussians: int,
    optimizer_policy: ScannerFastGSOptimizerPolicy | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    current = capture_model_np(model)
    opacity_mask = current["opacities"] < min_opacity
    score_mask = pruning_score > score_thresh
    prune_mask = opacity_mask | score_mask

    total = int(prune_mask.size)
    requested_remove = int(np.sum(prune_mask))
    if total == 0:
        return {
            "total": 0,
            "opacity_hits": 0,
            "score_hits": 0,
            "requested_remove": 0,
            "actual_remove": 0,
            "kept": 0,
        }

    min_keep = min(max(int(min_gaussians), 0), total)
    if total - requested_remove < min_keep:
        keep_priority = current["opacities"] - pruning_score
        keep_order = np.argsort(keep_priority)[::-1]
        keep_indices = keep_order[:min_keep]
        adjusted_prune_mask = np.ones((total,), dtype=bool)
        adjusted_prune_mask[keep_indices] = False
        actual_prune_mask = adjusted_prune_mask
    else:
        actual_prune_mask = prune_mask
    actual_remove = int(np.sum(actual_prune_mask))
    if dry_run:
        actual_remove = 0
    else:
        prune_points(model, state, actual_prune_mask, optimizer_policy=optimizer_policy)
    return {
        "total": total,
        "opacity_hits": int(np.sum(opacity_mask)),
        "score_hits": int(np.sum(score_mask)),
        "requested_remove": requested_remove,
        "actual_remove": actual_remove,
        "kept": int(total - actual_remove),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/Users/yangdunfu/Downloads/2026_03_01_16_36_14")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
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
    parser.add_argument("--random-background", action="store_true")
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
    parser.add_argument("--sh-degree-interval", type=int, default=1000)
    parser.add_argument("--densify-from-step", type=int, default=500)
    parser.add_argument("--densify-until-step", type=int, default=15000)
    parser.add_argument("--densification-interval", type=int, default=500)
    parser.add_argument("--opacity-reset-interval", type=int, default=3000)
    parser.add_argument("--opacity-reset-value", type=float, default=0.82)
    parser.add_argument("--opacity-cap-after-densify", type=float, default=0.82)
    parser.add_argument("--grad-thresh", type=float, default=2.0e-4)
    parser.add_argument("--grad-abs-thresh", type=float, default=1.2e-3)
    parser.add_argument("--dense", type=float, default=0.01)
    parser.add_argument("--loss-thresh", type=float, default=0.06)
    parser.add_argument("--importance-score-threshold", type=float, default=5.0)
    parser.add_argument("--densify-camera-sample", type=int, default=10)
    parser.add_argument("--split-factor", type=int, default=2)
    parser.add_argument("--min-opacity", type=float, default=0.005)
    parser.add_argument("--final-prune-min-opacity", type=float, default=0.1)
    parser.add_argument("--final-prune-start", type=int, default=15000)
    parser.add_argument("--final-prune-end", type=int, default=30000)
    parser.add_argument("--final-prune-interval", type=int, default=3000)
    parser.add_argument("--final-prune-score-thresh", type=float, default=0.9)
    parser.add_argument("--final-prune-min-gaussians", type=int, default=64)
    parser.add_argument("--max-screen-size", type=float, default=20.0)
    parser.add_argument("--max-world-scale-factor", type=float, default=0.1)
    parser.add_argument("--prune-budget-factor", type=float, default=0.5)
    parser.add_argument("--no-prune-gaussians", action="store_true", help="Temporarily skip all Gaussian removal while keeping densification enabled.")
    parser.add_argument("--reset-optimizer", action="store_true", help="Reset the learning-rate schedule every --reset-optimizer-interval steps.")
    parser.add_argument("--reset-optimizer-interval", type=int, default=201)
    args = parser.parse_args()

    ext = import_extension()
    dataset_dir = Path(args.data)
    if not dataset_dir.exists():
        raise RuntimeError(f"Dataset path does not exist: {dataset_dir}")

    cameras, targets, points, colors, base_point_count, camera_radius = prepare_dataset(
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
    extra_point_count = int(points.shape[0] - base_point_count)

    model = init_model(points, colors, args.sh_degree)
    dens_state = make_densification_state(points.shape[0])
    scene_extent = camera_radius
    optimizer_policy = make_optimizer_policy(args, scene_extent)
    gaussian_ops = ScannerGaussianOps(optimizer_policy=optimizer_policy)

    repo_root = Path(__file__).resolve().parent.parent
    date_dir = datetime.now().strftime("%Y%m%d_%H_%M")
    out_dir = repo_root / "training" / "output" / ("train_scanner_" + "fastgs2") / date_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_best = out_dir / "best_step.png"
    out_spz = out_dir / "final.spz"
    out_final_dir = out_dir / "final"
    out_final_dir.mkdir(parents=True, exist_ok=True)

    base_bg = mx.array([0.0, 0.0, 0.0], dtype=mx.float32)
    rng = np.random.default_rng(args.seed)
    best_loss = float("inf")
    best_step = -1
    ema_loss = 0.0
    losses = []
    eval_idx = 0
    active_sh_degree = 0
    optimizer_lr_reset_step = 0
    viewpoint_stack = list(range(len(cameras)))
    save_step_preview(
        ext=ext,
        model=model,
        cameras=cameras,
        targets=targets,
        background=base_bg,
        sh_degree=active_sh_degree,
        eval_idx=eval_idx,
        out_path=out_dir / "step_00000.png",
    )

    def loss_fn(means3d, features_dc, features_rest, opacity_logits, log_scales, rotations, viewspace_points, camera, target_chw, bg, use_l1, sh_degree):
        local_model = ScannerTrainModel(
            means3d=means3d,
            features_dc=features_dc,
            features_rest=features_rest,
            opacity_logits=opacity_logits,
            log_scales=log_scales,
            rotations=rotations,
        )
        inputs = {
            "background": bg,
            "means3d": local_model.means3d,
            "dc": local_model.features_dc,
            "sh": local_model.features_rest,
            "opacities": local_model.get_opacities,
            "scales": local_model.get_scales,
            "rotations": local_model.get_rotations,
            "metric_map": mx.zeros((camera.image_width * camera.image_height,), dtype=mx.int32),
            "viewmatrix": camera.viewmatrix,
            "projmatrix": camera.projmatrix,
            "campos": camera.campos,
            "viewspace_points": viewspace_points,
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
            pred = mx.broadcast_to(
                mx.reshape(bg, (3, 1, 1)),
                (3, camera.image_height, camera.image_width),
            )
        else:
            pred = to_chw_mx(out_color, camera.image_height, camera.image_width)
        diff = pred - target_chw
        l1 = mx.mean(mx.abs(diff))
        mse = mx.mean(diff * diff)
        return mx.where(use_l1, l1, mse)

    grad_fn = mx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5, 6))

    for step in range(1, args.steps + 1):
        lr_step = step - optimizer_lr_reset_step
        xyz_lr = optimizer_policy.update_learning_rate(lr_step)
        if args.sh_degree_interval > 0 and step % args.sh_degree_interval == 0:
            active_sh_degree = min(active_sh_degree + 1, args.sh_degree)

        if not viewpoint_stack:
            viewpoint_stack = list(range(len(cameras)))
        rand_pos = int(rng.integers(0, len(viewpoint_stack)))
        idx = viewpoint_stack.pop(rand_pos)
        camera = cameras[idx]
        target_chw = targets[idx]
        bg = mx.random.uniform(shape=(3,), low=0.0, high=1.0, dtype=mx.float32) if args.random_background else base_bg
        use_l1 = mx.array(step > args.mse_until, dtype=mx.bool_)
        viewspace_seed = mx.zeros((model.means3d.shape[0], 4), dtype=mx.float32)

        loss, grads = grad_fn(
            model.means3d,
            model.features_dc,
            model.features_rest,
            model.opacity_logits,
            model.log_scales,
            model.rotations,
            viewspace_seed,
            camera,
            target_chw,
            bg,
            use_l1,
            active_sh_degree,
        )
        d_means3d, d_features_dc, d_features_rest, d_opacity_logits, d_log_scales, d_rotations, d_viewspace = grads
        grad_map = {"opacity_logits": d_opacity_logits}
        if step > args.stage_color_steps:
            grad_map["features_dc"] = d_features_dc
            grad_map["features_rest"] = d_features_rest
        if step > args.stage_means_steps:
            grad_map["means3d"] = d_means3d
        if step > args.stage_scales_steps:
            grad_map["log_scales"] = d_log_scales
        if step > args.stage_rotations_steps:
            grad_map["rotations"] = d_rotations

        mx.eval(loss, d_viewspace)
        curr_loss = float(loss.item())
        skip_optimizer_step = False

        if step < args.densify_until_step:
            stats_render = render_pkg(ext, model, camera, bg, active_sh_degree, get_flag=False)
            mx.eval(stats_render["radii"])
            radii_np = np.array(stats_render["radii"], dtype=np.float32)
            d_viewspace_np = np.array(d_viewspace, dtype=np.float32)
            gaussian_ops.update_densification_stats(dens_state, radii_np, d_viewspace_np)

        if step < args.densify_until_step:
            if (
                args.densification_interval > 0
                and step > args.densify_from_step
                and step % args.densification_interval == 0
            ):
                sample_ids = sample_camera_indices(rng, len(cameras), args.densify_camera_sample)
                importance_score, pruning_score = compute_gaussian_scores_fastgs(
                    ext=ext,
                    model=model,
                    cameras=cameras,
                    targets=targets,
                    camera_indices=sample_ids,
                    background=base_bg,
                    sh_degree=active_sh_degree,
                    loss_thresh=args.loss_thresh,
                    densify=True,
                )
                before = int(model.means3d.shape[0])
                densify_stats = gaussian_ops.densify_and_prune_fastgs(
                    model,
                    dens_state,
                    args,
                    scene_extent,
                    importance_score,
                    pruning_score,
                    rng,
                )
                after = int(model.means3d.shape[0])
                skip_optimizer_step = True
                print(
                    f"[fastgs] step={step:05d} densify/prune points {before} -> {after} "
                    f"(metric_hits={densify_stats['metric_hits']}, "
                    f"clone_candidates={densify_stats['clone_candidates']}, cloned={densify_stats['cloned']}, "
                    f"split_candidates={densify_stats['split_candidates']}, "
                    f"split_sources={densify_stats['split_sources']}, split_children={densify_stats['split_children']}, "
                    f"opacity_prune={densify_stats['opacity_prune_candidates']}, "
                    f"screen_prune={densify_stats['screen_prune_candidates']}, "
                    f"world_prune={densify_stats['world_prune_candidates']}, "
                    f"total_prune={densify_stats['total_prune_candidates']}, "
                    f"prune_budget={densify_stats['prune_budget']}, actual_removed={densify_stats['actual_removed']})"
                )

            if args.opacity_reset_interval > 0 and step % args.opacity_reset_interval == 0:
                gaussian_ops.reset_opacity_logits(model, args.opacity_reset_value)
                skip_optimizer_step = True
                print(f"[fastgs] step={step:05d} reset opacity to <= {args.opacity_reset_value:.4f}")

        if (
            args.final_prune_interval > 0
            and step % args.final_prune_interval == 0
            and step > args.final_prune_start
            and step < args.final_prune_end
        ):
            sample_ids = sample_camera_indices(rng, len(cameras), args.densify_camera_sample)
            _, pruning_score = compute_gaussian_scores_fastgs(
                ext=ext,
                model=model,
                cameras=cameras,
                targets=targets,
                camera_indices=sample_ids,
                background=base_bg,
                sh_degree=active_sh_degree,
                loss_thresh=args.loss_thresh,
                densify=False,
            )
            before = int(model.means3d.shape[0])
            prune_stats = gaussian_ops.final_prune_fastgs(
                model,
                dens_state,
                args.final_prune_min_opacity,
                pruning_score,
                args.final_prune_score_thresh,
                args.final_prune_min_gaussians,
                dry_run=args.no_prune_gaussians,
            )
            after = int(model.means3d.shape[0])
            if after != before:
                skip_optimizer_step = True
            print(
                f"[fastgs] step={step:05d} final prune points {before} -> {after} "
                f"(opacity_hits={prune_stats['opacity_hits']}, score_hits={prune_stats['score_hits']}, "
                f"requested_remove={prune_stats['requested_remove']}, actual_remove={prune_stats['actual_remove']}, "
                f"kept={prune_stats['kept']})"
            )

        if skip_optimizer_step:
            print(f"[fastgs] step={step:05d} skip optimizer step after parameter topology/state update")
        else:
            optimizer_policy.apply_gradients(model, grad_map, step)

        if args.reset_optimizer and args.reset_optimizer_interval > 0 and step % args.reset_optimizer_interval == 0:
            optimizer_lr_reset_step = step
            print(f"[fastgs] step={step:05d} reset learning-rate schedule")

        mx.eval(model.means3d)

        if curr_loss < best_loss:
            best_loss = curr_loss
            best_step = step
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

        ema_loss = curr_loss if step == 1 else (0.4 * curr_loss + 0.6 * ema_loss)
        if step % args.log_every == 0 or step == args.steps:
            losses.append((step, curr_loss, ema_loss, int(model.means3d.shape[0])))
            print(
                f"[train] step={step:05d} view={idx:03d} sh_degree={active_sh_degree}/{args.sh_degree} "
                f"loss={curr_loss:.6f} ema={ema_loss:.6f} points={int(model.means3d.shape[0])} xyz_lr={xyz_lr:.8f}"
            )

        if step % args.save_every == 0 or step == args.steps:
            out_img = out_dir / f"step_{step:05d}.png"
            save_step_preview(
                ext=ext,
                model=model,
                cameras=cameras,
                targets=targets,
                background=base_bg,
                sh_degree=active_sh_degree,
                eval_idx=eval_idx,
                out_path=out_img,
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

    save_as_spz(out_spz, model, args.sh_degree)

    print("[OK] scanner FastGS2 training done")
    print("frames:", len(cameras), "points:", int(model.means3d.shape[0]))
    print("base_points:", points.shape[0] - extra_point_count, "extra_points:", extra_point_count)
    print("best_step:", best_step, "best_loss:", f"{best_loss:.6f}")
    print("saved best:", out_best)
    print("saved final dir:", out_final_dir)
    print("saved spz:", out_spz)


if __name__ == "__main__":
    main()
