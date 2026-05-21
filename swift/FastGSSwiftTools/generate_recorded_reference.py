#!/usr/bin/env python3

import json
import math
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from train_scanner_fixed import (  # noqa: E402
    import_extension,
    init_model,
    logit,
    prepare_dataset,
    render_chw,
    save_side_by_side,
)

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Pillow is required") from exc


DATASET_DIR = Path("/Users/yangdunfu/Downloads/2026_05_04_16_51_29")
OUT_DIR = Path("/private/tmp/fastgs_recorded_reference")
WIDTH = 160
HEIGHT = 120
MAX_FRAMES = 8
MAX_POINTS = 4096
EVAL_INDEX = 0
SH_DEGREE = 3
SCALE = 0.02
OPACITY = 0.82


def chw_to_hwc(values: np.ndarray) -> np.ndarray:
    return np.transpose(values, (1, 2, 0))


def save_chw_png(chw: mx.array, path: Path) -> None:
    arr = np.array(chw, dtype=np.float32)
    hwc = np.clip(chw_to_hwc(arr), 0.0, 1.0)
    Image.fromarray((hwc * 255.0).astype(np.uint8), mode="RGB").save(path)


def channel_sums(chw: np.ndarray) -> list[float]:
    return [float(chw[channel].sum()) for channel in range(chw.shape[0])]


def sample_pixels(chw: np.ndarray, ids: list[int]) -> list[float]:
    flat = chw.reshape(3, -1)
    out = []
    for channel in range(3):
        for pixel_id in ids:
            out.append(float(flat[channel, pixel_id]))
    return out


def array_payload(array: mx.array) -> list[float]:
    return np.array(array, dtype=np.float32).reshape(-1).tolist()


def write_f32_buffer(values: np.ndarray, path: Path) -> dict:
    arr = np.ascontiguousarray(values, dtype=np.float32)
    arr.reshape(-1).tofile(path)
    return {
        "path": path.name,
        "dtype": "float32",
        "shape": list(arr.shape),
    }


def main() -> None:
    if not DATASET_DIR.exists():
        raise RuntimeError(f"missing dataset: {DATASET_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ext = import_extension()
    cameras, targets, points, colors, base_point_count = prepare_dataset(
        dataset_dir=DATASET_DIR,
        width=WIDTH,
        height=HEIGHT,
        max_frames=MAX_FRAMES,
        frame_step=1,
        start_index=0,
        max_points=MAX_POINTS,
        seed=42,
        extra_points_ratio=0.0,
        extra_points_mode="surface-jitter",
        extra_points_jitter_scale=0.01,
    )
    if not cameras:
        raise RuntimeError("no cameras loaded")

    eval_index = max(0, min(EVAL_INDEX, len(cameras) - 1))
    model = init_model(points, colors, SH_DEGREE)
    n = points.shape[0]
    model.log_scales = mx.array(np.full((n, 3), math.log(SCALE), dtype=np.float32), dtype=mx.float32)
    model.opacity_logits = mx.array(logit(np.full((n,), OPACITY, dtype=np.float32)).astype(np.float32), dtype=mx.float32)

    background = mx.array([0.0, 0.0, 0.0], dtype=mx.float32)
    rotations = model.rotations / (mx.linalg.norm(model.rotations, axis=1, keepdims=True) + 1.0e-8)
    pred = render_chw(
        ext=ext,
        means3d=model.means3d,
        features_dc=model.features_dc,
        features_rest=model.features_rest,
        opacities=mx.sigmoid(model.opacity_logits),
        scales=mx.exp(model.log_scales),
        rotations=rotations,
        camera=cameras[eval_index],
        background=background,
        sh_degree=SH_DEGREE,
    )
    target = targets[eval_index]

    pred_png = OUT_DIR / "recorded_pred.png"
    target_png = OUT_DIR / "recorded_target.png"
    sbs_png = OUT_DIR / "recorded_sbs.png"
    manifest_path = OUT_DIR / "recorded_manifest.json"
    means3d_path = OUT_DIR / "recorded_means3d.f32"
    colors_path = OUT_DIR / "recorded_colors.f32"
    save_chw_png(pred, pred_png)
    save_chw_png(target, target_png)
    save_side_by_side(target, pred, sbs_png)
    means3d_buffer = write_f32_buffer(points, means3d_path)
    colors_buffer = write_f32_buffer(colors, colors_path)

    pred_np = np.array(pred, dtype=np.float32)
    target_np = np.array(target, dtype=np.float32)
    sample_ids = [
        0,
        WIDTH // 2,
        WIDTH - 1,
        (HEIGHT // 2) * WIDTH + (WIDTH // 2),
        WIDTH * HEIGHT - 1,
    ]
    camera = cameras[eval_index]
    manifest = {
        "dataset": str(DATASET_DIR),
        "width": WIDTH,
        "height": HEIGHT,
        "maxFrames": MAX_FRAMES,
        "maxPoints": MAX_POINTS,
        "basePointCount": int(base_point_count),
        "pointCount": int(points.shape[0]),
        "evalIndex": eval_index,
        "shDegree": SH_DEGREE,
        "scale": SCALE,
        "opacity": OPACITY,
        "background": [0.0, 0.0, 0.0],
        "tanFovX": float(camera.tan_fovx),
        "tanFovY": float(camera.tan_fovy),
        "viewmatrix": array_payload(camera.viewmatrix),
        "projmatrix": array_payload(camera.projmatrix),
        "campos": array_payload(camera.campos),
        "means3dBuffer": means3d_buffer,
        "colorsBuffer": colors_buffer,
        "predChannelSums": channel_sums(pred_np),
        "targetChannelSums": channel_sums(target_np),
        "samplePixelIds": sample_ids,
        "predSamples": sample_pixels(pred_np, sample_ids),
        "targetSamples": sample_pixels(target_np, sample_ids),
        "predPng": str(pred_png),
        "targetPng": str(target_png),
        "sideBySidePng": str(sbs_png),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[OK] recorded reference generated")
    print("manifest:", manifest_path)
    print("pred:", pred_png)
    print("target:", target_png)
    print("sbs:", sbs_png)
    print("means3d:", means3d_path)
    print("colors:", colors_path)
    print("points:", points.shape[0], "frames:", len(cameras))
    print("predChannelSums:", manifest["predChannelSums"])


if __name__ == "__main__":
    main()
