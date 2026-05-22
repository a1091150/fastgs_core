#!/usr/bin/env python3

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a recorded FastGS reference for Swift parity tests.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--max-points", type=int, default=MAX_POINTS)
    parser.add_argument("--eval-index", type=int, default=EVAL_INDEX)
    parser.add_argument("--sh-degree", type=int, default=SH_DEGREE)
    parser.add_argument("--scale", type=float, default=SCALE)
    parser.add_argument("--opacity", type=float, default=OPACITY)
    return parser.parse_args()


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


def samples_flat(values: np.ndarray, ids: list[int]) -> list[float]:
    flat = values.reshape(-1)
    return [float(flat[pixel_id]) for pixel_id in ids if pixel_id < flat.shape[0]]


def samples_chw_flat(values: np.ndarray, ids: list[int], channels: int = 3) -> list[float]:
    flat = values.reshape(channels, -1)
    return [float(flat[channel, pixel_id]) for channel in range(channels) for pixel_id in ids if pixel_id < flat.shape[1]]


def array_prefix(values: np.ndarray, count: int = 16) -> list:
    return values.reshape(-1)[:count].tolist()


def stage_summaries(
    ext,
    model,
    camera,
    background: mx.array,
    rotations: mx.array,
    sh_degree: int,
    sample_ids: list[int],
) -> dict:
    n = int(model.means3d.shape[0])
    block_x = 16
    block_y = 16
    tiles_x = (camera.image_width + block_x - 1) // block_x
    tiles_y = (camera.image_height + block_y - 1) // block_y
    num_tiles = tiles_x * tiles_y
    inputs = {
        "means3d": model.means3d,
        "dc": model.features_dc,
        "sh": model.features_rest,
        "colors_precomp": mx.zeros((0, 3), dtype=mx.float32),
        "opacities": mx.sigmoid(model.opacity_logits),
        "scales": mx.exp(model.log_scales),
        "quats": rotations,
        "cov3d_precomp": mx.zeros((0,), dtype=mx.float32),
        "viewmat": camera.viewmatrix,
        "projmat": camera.projmatrix,
        "cam_pos": camera.campos,
        "viewspace_points": mx.zeros((n, 4), dtype=mx.float32),
    }
    pre = ext.preprocess_forward(
        inputs,
        camera.image_width,
        camera.image_height,
        block_x,
        block_y,
        camera.tan_fovx,
        camera.tan_fovy,
        sh_degree,
        1.0,
        1.0,
        False,
    )
    mx.eval(
        pre["radii"],
        pre["xys"],
        pre["depths"],
        pre["rgb"],
        pre["conic_opacity"],
        pre["tiles_touched"],
    )
    point_offsets = mx.cumsum(pre["tiles_touched"], reverse=False, inclusive=True)
    mx.eval(point_offsets)
    point_offsets_np = np.array(point_offsets, dtype=np.uint32)
    num_rendered = int(point_offsets_np[-1]) if point_offsets_np.size else 0
    binning = ext.binning_forward(
        pre["xys"],
        pre["depths"],
        point_offsets,
        pre["conic_opacity"],
        pre["tiles_touched"],
        1.0,
        tiles_x,
        tiles_y,
        1,
        num_rendered,
    )
    sorted_indices = mx.argsort(binning["point_list_keys_unsorted"])
    point_list_keys = mx.take(binning["point_list_keys_unsorted"], sorted_indices)
    point_list = mx.take(binning["point_list_unsorted"], sorted_indices)
    tile = ext.tile_prep_forward(point_list_keys, num_rendered, num_tiles)
    bucket_offsets = mx.cumsum(tile["bucket_count"], reverse=False, inclusive=True)
    mx.eval(
        binning["point_list_keys_unsorted"],
        binning["point_list_unsorted"],
        point_list_keys,
        point_list,
        tile["ranges"],
        tile["bucket_count"],
        bucket_offsets,
    )
    bucket_offsets_np = np.array(bucket_offsets, dtype=np.uint32)
    bucket_sum = int(bucket_offsets_np[-1]) if bucket_offsets_np.size else 0
    rast = ext.rasterize_forward(
        tile["ranges"],
        point_list,
        bucket_offsets,
        pre["xys"],
        pre["rgb"],
        pre["conic_opacity"],
        background,
        pre["radii"],
        mx.zeros((camera.image_width * camera.image_height,), dtype=mx.int32),
        mx.zeros((n,), dtype=mx.int32),
        pre["viewspace_points"],
        camera.image_width,
        camera.image_height,
        block_x,
        block_y,
        3,
        num_tiles,
        bucket_sum,
        False,
    )
    mx.eval(
        rast["bucket_to_tile"],
        rast["final_t"],
        rast["n_contrib"],
        rast["max_contrib"],
        rast["pixel_colors"],
        rast["out_color"],
    )

    radii = np.array(pre["radii"], dtype=np.int32)
    tiles_touched = np.array(pre["tiles_touched"], dtype=np.uint32)
    depths = np.array(pre["depths"], dtype=np.float32)
    xys = np.array(pre["xys"], dtype=np.float32)
    rgb = np.array(pre["rgb"], dtype=np.float32)
    conic = np.array(pre["conic_opacity"], dtype=np.float32)
    keys = np.array(point_list_keys, dtype=np.uint64)
    plist = np.array(point_list, dtype=np.uint32)
    ranges = np.array(tile["ranges"], dtype=np.uint32)
    bucket_count = np.array(tile["bucket_count"], dtype=np.uint32)
    out = np.array(rast["out_color"], dtype=np.float32)
    pixels = np.array(rast["pixel_colors"], dtype=np.float32)
    final_t = np.array(rast["final_t"], dtype=np.float32)
    n_contrib = np.array(rast["n_contrib"], dtype=np.uint32)
    max_contrib = np.array(rast["max_contrib"], dtype=np.uint32)

    visible = radii > 0
    return {
        "preprocess": {
            "visibleCount": int(visible.sum()),
            "radiiSum": int(radii.sum()),
            "tilesTouchedSum": int(tiles_touched.sum()),
            "depthSumVisible": float(depths[visible].sum()),
            "xysSumVisible": [float(xys[visible, axis].sum()) for axis in range(2)],
            "rgbSums": [float(rgb[:, channel].sum()) for channel in range(3)],
            "conicOpacitySums": [float(conic[:, channel].sum()) for channel in range(4)],
            "radiiPrefix": array_prefix(radii),
            "tilesTouchedPrefix": array_prefix(tiles_touched),
        },
        "binning": {
            "numRendered": num_rendered,
            "pointListKeyPrefix": array_prefix(keys),
            "pointListPrefix": array_prefix(plist),
            "pointListKeyChecksum": int(keys[: min(keys.size, 4096)].sum() & np.uint64(0xFFFFFFFFFFFFFFFF)),
            "pointListChecksum": int(plist[: min(plist.size, 4096)].sum()),
        },
        "tile": {
            "bucketSum": bucket_sum,
            "bucketCountSum": int(bucket_count.sum()),
            "bucketCountPrefix": array_prefix(bucket_count),
            "bucketOffsetPrefix": array_prefix(bucket_offsets_np),
            "rangesPrefix": array_prefix(ranges, 32),
        },
        "rasterize": {
            "outColorSums": [float(out[channel].sum()) for channel in range(3)],
            "pixelColorSums": [float(pixels[channel].sum()) for channel in range(3)],
            "finalTSum": float(final_t.sum()),
            "nContribSum": int(n_contrib.sum()),
            "maxContribSum": int(max_contrib.sum()),
            "outSamples": samples_chw_flat(out, sample_ids),
            "finalTSamples": samples_flat(final_t, sample_ids),
            "nContribSamples": [int(v) for v in np.array(n_contrib.reshape(-1)[sample_ids], dtype=np.uint32)],
        },
    }


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
    args = parse_args()
    if not args.dataset_dir.exists():
        raise RuntimeError(f"missing dataset: {args.dataset_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ext = import_extension()
    cameras, targets, points, colors, base_point_count = prepare_dataset(
        dataset_dir=args.dataset_dir,
        width=args.width,
        height=args.height,
        max_frames=args.max_frames,
        frame_step=1,
        start_index=0,
        max_points=args.max_points,
        seed=42,
        extra_points_ratio=0.0,
        extra_points_mode="surface-jitter",
        extra_points_jitter_scale=0.01,
    )
    if not cameras:
        raise RuntimeError("no cameras loaded")

    eval_index = max(0, min(args.eval_index, len(cameras) - 1))
    model = init_model(points, colors, args.sh_degree)
    n = points.shape[0]
    model.log_scales = mx.array(np.full((n, 3), math.log(args.scale), dtype=np.float32), dtype=mx.float32)
    model.opacity_logits = mx.array(logit(np.full((n,), args.opacity, dtype=np.float32)).astype(np.float32), dtype=mx.float32)

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
        sh_degree=args.sh_degree,
    )
    target = targets[eval_index]

    pred_png = args.out_dir / "recorded_pred.png"
    target_png = args.out_dir / "recorded_target.png"
    sbs_png = args.out_dir / "recorded_sbs.png"
    manifest_path = args.out_dir / "recorded_manifest.json"
    means3d_path = args.out_dir / "recorded_means3d.f32"
    colors_path = args.out_dir / "recorded_colors.f32"
    target_path = args.out_dir / "recorded_target.f32"
    save_chw_png(pred, pred_png)
    save_chw_png(target, target_png)
    save_side_by_side(target, pred, sbs_png)
    means3d_buffer = write_f32_buffer(points, means3d_path)
    colors_buffer = write_f32_buffer(colors, colors_path)

    pred_np = np.array(pred, dtype=np.float32)
    target_np = np.array(target, dtype=np.float32)
    target_buffer = write_f32_buffer(target_np.reshape(3, args.width * args.height), target_path)
    sample_ids = [
        0,
        args.width // 2,
        args.width - 1,
        (args.height // 2) * args.width + (args.width // 2),
        args.width * args.height - 1,
    ]
    summaries = stage_summaries(ext, model, cameras[eval_index], background, rotations, args.sh_degree, sample_ids)
    camera = cameras[eval_index]
    manifest = {
        "dataset": str(args.dataset_dir),
        "width": args.width,
        "height": args.height,
        "maxFrames": args.max_frames,
        "maxPoints": args.max_points,
        "basePointCount": int(base_point_count),
        "pointCount": int(points.shape[0]),
        "evalIndex": eval_index,
        "shDegree": args.sh_degree,
        "scale": args.scale,
        "opacity": args.opacity,
        "background": [0.0, 0.0, 0.0],
        "tanFovX": float(camera.tan_fovx),
        "tanFovY": float(camera.tan_fovy),
        "viewmatrix": array_payload(camera.viewmatrix),
        "projmatrix": array_payload(camera.projmatrix),
        "campos": array_payload(camera.campos),
        "means3dBuffer": means3d_buffer,
        "colorsBuffer": colors_buffer,
        "targetBuffer": target_buffer,
        "predChannelSums": channel_sums(pred_np),
        "targetChannelSums": channel_sums(target_np),
        "samplePixelIds": sample_ids,
        "predSamples": sample_pixels(pred_np, sample_ids),
        "targetSamples": sample_pixels(target_np, sample_ids),
        "stageSummaries": summaries,
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
    print("target f32:", target_path)
    print("points:", points.shape[0], "frames:", len(cameras))
    print("predChannelSums:", manifest["predChannelSums"])


if __name__ == "__main__":
    main()
