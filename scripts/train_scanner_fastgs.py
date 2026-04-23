#!/usr/bin/env python3

import argparse
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.optimizers import Adam

from train_scanner_fixed import (
    ScannerTrainModel,
    TrainCamera,
    import_extension,
    init_model,
    logit,
    prepare_dataset,
    render_chw,
    save_as_spz,
    save_side_by_side,
)

mx.set_cache_limit(limit=(1 << 31))


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


def compute_scene_extent(points: np.ndarray) -> float:
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    return max(float(np.linalg.norm(bbox_max - bbox_min)), 1.0e-6)


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
    ) -> dict[str, int]:
        return final_prune_fastgs(
            model,
            state,
            min_opacity,
            pruning_score,
            score_thresh,
            min_gaussians,
            optimizer_policy=self.optimizer_policy,
        )


def reset_densification_buffers(state: FastGSDensificationState, num_points: int) -> None:
    state.max_radii2d = np.zeros((num_points,), dtype=np.float32)
    state.xyz_grad_accum = np.zeros((num_points, 1), dtype=np.float32)
    state.xyz_grad_accum_abs = np.zeros((num_points, 1), dtype=np.float32)
    state.denom = np.zeros((num_points, 1), dtype=np.float32)


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
    metric_mask: np.ndarray,
    clone_filter: np.ndarray,
    optimizer_policy: ScannerFastGSOptimizerPolicy | None = None,
) -> int:
    selected = metric_mask & clone_filter
    if not np.any(selected):
        return 0
    current = capture_model_np(model)
    append_new_points(
        model,
        state,
        {
            "means3d": current["means3d"][selected],
            "features_dc": current["features_dc"][selected],
            "features_rest": current["features_rest"][selected],
            "opacity_logits": current["opacity_logits"][selected],
            "log_scales": current["log_scales"][selected],
            "rotations": current["rotations"][selected],
            "tmp_radii": state.tmp_radii[selected],
        },
        optimizer_policy=optimizer_policy,
    )
    return int(np.sum(selected))


def densify_and_split_fastgs(
    model: ScannerTrainModel,
    state: FastGSDensificationState,
    metric_mask: np.ndarray,
    split_filter: np.ndarray,
    rng: np.random.Generator,
    split_factor: int,
    optimizer_policy: ScannerFastGSOptimizerPolicy | None = None,
) -> tuple[int, int]:
    selected = metric_mask & split_filter
    if not np.any(selected):
        return 0, 0
    current = capture_model_np(model)
    means = current["means3d"][selected]
    scales = current["scales"][selected]
    log_scales = current["log_scales"][selected]
    rotations = current["rotations"][selected]
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
            "features_dc": np.repeat(current["features_dc"][selected], split_factor, axis=0),
            "features_rest": np.repeat(current["features_rest"][selected], split_factor, axis=0),
            "opacity_logits": np.repeat(current["opacity_logits"][selected], split_factor, axis=0),
            "log_scales": new_scales.astype(np.float32),
            "rotations": np.repeat(rotations, split_factor, axis=0),
            "tmp_radii": np.repeat(state.tmp_radii[selected], split_factor, axis=0),
        },
        optimizer_policy=optimizer_policy,
    )

    prune_mask = np.concatenate(
        [selected, np.zeros((int(np.sum(selected)) * split_factor,), dtype=bool)],
        axis=0,
    )
    prune_points(model, state, prune_mask, optimizer_policy=optimizer_policy)
    selected_count = int(np.sum(selected))
    return selected_count, int(selected_count * split_factor)


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
        metric_mask,
        clone_qualifiers & grad_qualifiers,
        optimizer_policy=optimizer_policy,
    )
    split_sources, split_children = densify_and_split_fastgs(
        model,
        state,
        metric_mask,
        split_qualifiers & grad_qualifiers_abs,
        rng,
        args.split_factor,
        optimizer_policy=optimizer_policy,
    )

    current = capture_model_np(model)
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
    if remove_budget > 0 and pruning_score.size > 0:
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
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
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
    args = parser.parse_args()

    ext = import_extension()
    dataset_dir = Path(args.data)
    if not dataset_dir.exists():
        raise RuntimeError(f"Dataset path does not exist: {dataset_dir}")

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
    extra_point_count = int(points.shape[0] - base_point_count)

    model = init_model(points, colors, args.sh_degree)
    dens_state = make_densification_state(points.shape[0])
    optimizer_policy = ScannerFastGSOptimizerPolicy(
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
            betas=(args.adam_beta1, args.adam_beta2),
        )
    )
    gaussian_ops = ScannerGaussianOps(optimizer_policy=optimizer_policy)
    scene_extent = compute_scene_extent(points)

    repo_root = Path(__file__).resolve().parent.parent
    date_dir = datetime.now().strftime("%Y%m%d_%H_%M")
    out_dir = repo_root / "training" / "output" / "train_scanner_fastgs" / date_dir
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
    viewpoint_stack = list(range(len(cameras)))

    def loss_fn(means3d, features_dc, features_rest, opacity_logits, log_scales, rotations, viewspace_points, camera, target_chw, bg, use_l1, sh_degree):
        local_model = ScannerTrainModel(
            means3d=means3d,
            features_dc=features_dc,
            features_rest=features_rest,
            opacity_logits=opacity_logits,
            log_scales=log_scales,
            rotations=rotations,
        )
        n = means3d.shape[0]
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
        pred = render_chw(
            ext=ext,
            means3d=local_model.means3d,
            features_dc=local_model.features_dc,
            features_rest=local_model.features_rest,
            opacities=local_model.get_opacities,
            scales=local_model.get_scales,
            rotations=local_model.get_rotations,
            camera=camera,
            background=bg,
            sh_degree=sh_degree,
        )
        diff = pred - target_chw
        l1 = mx.mean(mx.abs(diff))
        mse = mx.mean(diff * diff)
        return mx.where(use_l1, l1, mse) + 0.0 * mx.sum(out["viewspace_points"])

    grad_fn = mx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5, 6))

    for step in range(1, args.steps + 1):
        xyz_lr = optimizer_policy.update_learning_rate(step)
        if step % 1000 == 0:
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

        optimizer_policy.apply_gradients(model, grad_map, step)

        mx.eval(loss, d_viewspace, model.means3d)
        curr_loss = float(loss.item())

        if step < args.densify_until_step:
            stats_render = render_pkg(ext, model, camera, bg, active_sh_degree, get_flag=False)
            mx.eval(stats_render["radii"])
            radii_np = np.array(stats_render["radii"], dtype=np.float32)
            d_viewspace_np = np.array(d_viewspace, dtype=np.float32)
            gaussian_ops.update_densification_stats(dens_state, radii_np, d_viewspace_np)

            if step > args.densify_from_step and step % args.densification_interval == 0:
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

            if step % args.opacity_reset_interval == 0:
                gaussian_ops.reset_opacity_logits(model, args.opacity_reset_value)
                print(f"[fastgs] step={step:05d} reset opacity to <= {args.opacity_reset_value:.4f}")

        if (
            step % args.final_prune_interval == 0
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
            )
            after = int(model.means3d.shape[0])
            print(
                f"[fastgs] step={step:05d} final prune points {before} -> {after} "
                f"(opacity_hits={prune_stats['opacity_hits']}, score_hits={prune_stats['score_hits']}, "
                f"requested_remove={prune_stats['requested_remove']}, actual_remove={prune_stats['actual_remove']}, "
                f"kept={prune_stats['kept']})"
            )

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
            out_img = out_dir / f"step_{step:05d}.png"
            save_side_by_side(targets[eval_idx], pred_eval, out_img)

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

    print("[OK] train_scanner_fastgs done")
    print("frames:", len(cameras), "points:", int(model.means3d.shape[0]))
    print("base_points:", points.shape[0] - extra_point_count, "extra_points:", extra_point_count)
    print("best_step:", best_step, "best_loss:", f"{best_loss:.6f}")
    print("saved best:", out_best)
    print("saved final dir:", out_final_dir)
    print("saved spz:", out_spz)


if __name__ == "__main__":
    main()
