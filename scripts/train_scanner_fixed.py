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


def read_ply_positions_colors_ascii(path: Path) -> tuple[np.ndarray, np.ndarray]:
    header_lines = 0
    vertex_count = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            header_lines += 1
            s = line.strip()
            if s.startswith("element vertex"):
                parts = s.split()
                vertex_count = int(parts[-1])
            if s == "end_header":
                break

    if vertex_count is None:
        raise RuntimeError(f"Failed to parse vertex count from {path}")

    data = np.loadtxt(
        str(path),
        dtype=np.float32,
        skiprows=header_lines,
        usecols=(0, 1, 2, 3, 4, 5),
        max_rows=vertex_count,
    )
    if data.ndim == 1:
        data = data[None, :]

    points = data[:, :3].astype(np.float32)
    colors = data[:, 3:6].astype(np.float32)
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


def compute_normalization(frames: list[ScannerFrame], a4: np.ndarray) -> tuple[np.ndarray, float]:
    centers = []
    for frame in frames:
        raw = load_json(frame.json_path)
        pose = raw.get("cameraPoseARFrame")
        if pose is None or len(pose) != 16:
            raise RuntimeError(f"Invalid cameraPoseARFrame in {frame.json_path}")
        c2w_src = np.array(pose, dtype=np.float32).reshape(4, 4)
        c2w = a4 @ c2w_src
        centers.append(c2w[:3, 3])

    centers_np = np.stack(centers, axis=0).astype(np.float32)
    translation = centers_np.mean(axis=0)
    denom = np.max(np.abs(centers_np - translation[None, :]))
    scale = 1.0 / float(denom) if denom > 0.0 else 1.0
    return translation, scale


def build_camera_from_scanner_json(
    meta: dict,
    image_width: int,
    image_height: int,
    a4: np.ndarray,
    translation: np.ndarray,
    norm_scale: float,
    znear: float = 0.001,
    zfar: float = 1000.0,
) -> TrainCamera:
    intrinsics = meta.get("intrinsics")
    if intrinsics is None or len(intrinsics) != 9:
        raise RuntimeError("intrinsics must contain 9 elements")

    fx = float(intrinsics[0])
    fy = float(intrinsics[4])

    pose = meta.get("cameraPoseARFrame")
    if pose is None or len(pose) != 16:
        raise RuntimeError("cameraPoseARFrame must contain 16 elements")
    c2w_src = np.array(pose, dtype=np.float32).reshape(4, 4)
    c2w = (a4 @ c2w_src).astype(np.float32)
    c2w[:3, 3] = (c2w[:3, 3] - translation) * norm_scale

    r = c2w[:3, :3].astype(np.float32)
    t = c2w[:3, 3:4].astype(np.float32)
    r = r @ np.diag([1.0, -1.0, -1.0]).astype(np.float32)

    rinv = r.T
    tinv = (-rinv @ t).astype(np.float32)

    raw_viewmat = np.eye(4, dtype=np.float32)
    raw_viewmat[:3, :3] = rinv
    raw_viewmat[:3, 3:4] = tinv

    width = float(image_width)
    height = float(image_height)
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
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if img.shape[1] != width or img.shape[0] != height:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def prepare_dataset(
    dataset_dir: Path,
    width: int,
    height: int,
    max_frames: int,
    frame_step: int,
    start_index: int,
    max_points: int,
    seed: int,
) -> tuple[list[TrainCamera], list[mx.array], np.ndarray, np.ndarray]:
    a, a4 = make_axis_transform()
    frames = collect_scanner_frames(dataset_dir, max_frames, frame_step, start_index)
    translation, norm_scale = compute_normalization(frames, a4)

    points, colors = read_ply_positions_colors_ascii(dataset_dir / "points.ply")
    points = (a @ points.T).T
    points = (points - translation[None, :]) * norm_scale

    rng = np.random.default_rng(seed)
    if max_points > 0 and points.shape[0] > max_points:
        keep = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[keep]
        colors = colors[keep]

    cameras = []
    targets = []
    for f in frames:
        meta = load_json(f.json_path)
        camera = build_camera_from_scanner_json(
            meta=meta,
            image_width=width,
            image_height=height,
            a4=a4,
            translation=translation,
            norm_scale=norm_scale,
        )
        target_hwc = load_target_image(f.image_path, width, height)
        target_chw = np.transpose(target_hwc, (2, 0, 1))
        cameras.append(camera)
        targets.append(mx.array(target_chw, dtype=mx.float32))

    return cameras, targets, points.astype(np.float32), colors.astype(np.float32)


class ScannerTrainModel(nn.Module):
    def __init__(
        self,
        means3d: mx.array,
        colors: mx.array,
        opacities: mx.array,
        scales: mx.array,
        rotations: mx.array,
    ):
        super().__init__()
        self.means3d = means3d
        self.colors = colors
        self.opacities = opacities
        self.scales = scales
        self.rotations = rotations


def render_chw(
    ext,
    means3d: mx.array,
    colors_precomp: mx.array,
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
        "colors_precomp": colors_precomp,
        "opacities": opacities,
        "scales": scales,
        "rotations": rotations,
        "metric_map": mx.zeros((camera.image_width * camera.image_height,), dtype=mx.int32),
        "viewmatrix": camera.viewmatrix,
        "projmatrix": camera.projmatrix,
        "dc": mx.zeros((0,), dtype=mx.float32),
        "sh": mx.zeros((0,), dtype=mx.float32),
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


def init_model(points: np.ndarray, colors: np.ndarray) -> ScannerTrainModel:
    n = points.shape[0]
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    base_scale = max(1.0e-3, 0.01 * diag)

    scales = np.full((n, 3), base_scale, dtype=np.float32)
    rotations = np.zeros((n, 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    opacities = np.full((n,), 0.35, dtype=np.float32)

    return ScannerTrainModel(
        means3d=mx.array(points, dtype=mx.float32),
        colors=mx.array(colors, dtype=mx.float32),
        opacities=mx.array(opacities, dtype=mx.float32),
        scales=mx.array(scales, dtype=mx.float32),
        rotations=mx.array(rotations, dtype=mx.float32),
    )


def save_as_spz(filename: Path, model: ScannerTrainModel, sh_degree: int) -> bool:
    if spz is None:
        print("[WARN] spz is not available; skip final.spz export")
        return False

    cloud = spz.GaussianCloud()
    cloud.antialiased = True

    mx.eval(model.means3d, model.scales, model.rotations, model.opacities, model.colors)
    means = np.array(model.means3d, dtype=np.float32)
    scales = np.array(model.scales, dtype=np.float32)
    quats = np.array(model.rotations, dtype=np.float32)
    quats = quats / (np.linalg.norm(quats, axis=1, keepdims=True) + 1.0e-8)
    opacities = np.array(model.opacities, dtype=np.float32)
    colors = np.array(model.colors, dtype=np.float32)

    cloud.positions = means.flatten().astype(np.float32)
    cloud.scales = scales.flatten().astype(np.float32)
    cloud.rotations = quats.flatten().astype(np.float32)
    cloud.alphas = opacities.flatten().astype(np.float32)
    cloud.colors = colors.flatten().astype(np.float32)
    cloud.sh_degree = int(sh_degree)
    sh_coeffs_per_channel = max(0, (cloud.sh_degree + 1) ** 2 - 1)
    sh_len = means.shape[0] * 3 * sh_coeffs_per_channel
    cloud.sh = np.zeros((sh_len,), dtype=np.float32)

    opts = spz.PackOptions()
    ok = spz.save_spz(cloud, opts, str(filename))
    if not ok:
        raise RuntimeError(f"failed to save spz to {filename}")
    print(f"saved spz: {filename}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/Users/yangdunfu/Downloads/2026_03_01_16_36_14")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--frame-step", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-background", action="store_true")
    parser.add_argument("--lr-colors", type=float, default=2e-2)
    parser.add_argument("--lr-opacity", type=float, default=1e-2)
    parser.add_argument("--lr-means", type=float, default=3e-3)
    parser.add_argument("--lr-scales", type=float, default=1e-3)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.99)
    parser.add_argument("--stage-color-steps", type=int, default=200)
    parser.add_argument("--stage-means-steps", type=int, default=800)
    parser.add_argument("--mse-until", type=int, default=600)
    parser.add_argument("--sh-degree", type=int, default=2)
    args = parser.parse_args()

    dataset_dir = Path(args.data)
    if not dataset_dir.exists():
        raise RuntimeError(f"Dataset path does not exist: {dataset_dir}")

    ext = import_extension()
    cameras, targets, points, colors = prepare_dataset(
        dataset_dir=dataset_dir,
        width=args.width,
        height=args.height,
        max_frames=args.max_frames,
        frame_step=args.frame_step,
        start_index=args.start_index,
        max_points=args.max_points,
        seed=args.seed,
    )
    if len(cameras) != len(targets):
        raise RuntimeError("Internal error: camera/target length mismatch")

    model = init_model(points, colors)
    betas = (args.adam_beta1, args.adam_beta2)
    means_opt = Adam(learning_rate=args.lr_means, betas=betas)
    colors_opt = Adam(learning_rate=args.lr_colors, betas=betas)
    opacity_opt = Adam(learning_rate=args.lr_opacity, betas=betas)
    scales_opt = Adam(learning_rate=args.lr_scales, betas=betas)

    base_bg = mx.array([1.0, 1.0, 1.0], dtype=mx.float32)

    def loss_fn(model: ScannerTrainModel, camera: TrainCamera, target_chw: mx.array, bg: mx.array, use_l1: mx.array):
        pred = render_chw(
            ext=ext,
            means3d=model.means3d,
            colors_precomp=model.colors,
            opacities=model.opacities,
            scales=model.scales,
            rotations=model.rotations,
            camera=camera,
            background=bg,
            sh_degree=args.sh_degree,
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

    rng = np.random.default_rng(args.seed)
    best_loss = float("inf")
    best_step = -1
    ema_loss = 0.0
    losses = []

    eval_idx = 0
    for step in range(1, args.steps + 1):
        idx = int(rng.integers(0, len(cameras)))
        camera = cameras[idx]
        target_chw = targets[idx]

        bg = mx.random.uniform(shape=(3,), low=0.0, high=1.0, dtype=mx.float32) if args.random_background else base_bg
        use_l1 = mx.array(step > args.mse_until, dtype=mx.bool_)
        loss, grads = loss_and_grad_fn(model, camera, target_chw, bg, use_l1)

        colors_opt.update(model, {"colors": grads["colors"]})
        opacity_opt.update(model, {"opacities": grads["opacities"]})

        if step > args.stage_color_steps:
            means_opt.update(model, {"means3d": grads["means3d"]})

        if step > args.stage_means_steps:
            scales_opt.update(model, {"scales": grads["scales"]})

        model.colors = mx.clip(model.colors, 0.0, 1.0)
        model.opacities = mx.clip(model.opacities, 0.0, 1.0)
        model.scales = mx.clip(model.scales, 1.0e-5, 2.0)

        mx.eval(loss)
        curr_loss = float(loss.item())

        if curr_loss < best_loss:
            best_loss = curr_loss
            best_step = step
            pred_best = render_chw(
                ext=ext,
                means3d=model.means3d,
                colors_precomp=model.colors,
                opacities=model.opacities,
                scales=model.scales,
                rotations=model.rotations,
                camera=cameras[eval_idx],
                background=base_bg,
                sh_degree=args.sh_degree,
            )
            save_side_by_side(targets[eval_idx], pred_best, out_best)

        if step == 1:
            ema_loss = curr_loss
        else:
            ema_loss = 0.4 * curr_loss + 0.6 * ema_loss

        if step % args.log_every == 0 or step == args.steps:
            losses.append((step, curr_loss, ema_loss))
            print(f"[train] step={step:04d} view={idx:03d} loss={curr_loss:.6f} ema={ema_loss:.6f}")

        if step % args.save_every == 0 or step == args.steps:
            pred_eval = render_chw(
                ext=ext,
                means3d=model.means3d,
                colors_precomp=model.colors,
                opacities=model.opacities,
                scales=model.scales,
                rotations=model.rotations,
                camera=cameras[eval_idx],
                background=base_bg,
                sh_degree=args.sh_degree,
            )
            out_img = out_dir / f"step_{step:04d}.png"
            save_side_by_side(targets[eval_idx], pred_eval, out_img)
        pass

    pred_final = render_chw(
        ext=ext,
        means3d=model.means3d,
        colors_precomp=model.colors,
        opacities=model.opacities,
        scales=model.scales,
        rotations=model.rotations,
        camera=cameras[eval_idx],
        background=base_bg,
        sh_degree=args.sh_degree,
    )

    mx.eval(model.means3d, model.colors, model.opacities, model.scales)
    np.savez(
        out_npz,
        means3d=np.array(model.means3d),
        colors=np.array(model.colors),
        opacities=np.array(model.opacities),
        scales=np.array(model.scales),
        losses=np.array(losses, dtype=np.float32),
        best_step=np.array([best_step], dtype=np.int32),
        best_loss=np.array([best_loss], dtype=np.float32),
        eval_target=np.array(targets[eval_idx]),
        eval_pred=np.array(pred_final),
    )
    save_as_spz(out_spz, model, args.sh_degree)

    print("[OK] train_scanner_fixed done")
    print("frames:", len(cameras), "points:", points.shape[0])
    print("saved state:", out_npz)
    print("saved best:", out_best)
    print("saved spz:", out_spz)


if __name__ == "__main__":
    main()
