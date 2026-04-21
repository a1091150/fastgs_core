#!/usr/bin/env python3

import os
from datetime import datetime

import numpy as np

try:
    import spz
except Exception as exc:
    raise RuntimeError("spz module is required for this test") from exc


def main() -> None:
    n = 1 << 12
    sh_degree = 2
    rng = np.random.default_rng(0)

    means3d = rng.uniform(low=-20.0, high=20.0, size=(n, 3)).astype(np.float32)
    scales = np.full((n, 3), 0.02, dtype=np.float32)
    rotations = np.zeros((n, 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    alphas = np.full((n,), 0.85, dtype=np.float32)
    colors = rng.uniform(low=0.0, high=0.01, size=(n, 3)).astype(np.float32)
    rest_coeffs = (sh_degree + 1) ** 2 - 1
    features_rest = np.zeros((n, rest_coeffs, 3), dtype=np.float32)

    cloud = spz.GaussianCloud()
    cloud.antialiased = True
    cloud.positions = means3d.flatten()
    cloud.scales = scales.flatten()
    cloud.rotations = rotations.flatten()
    cloud.alphas = alphas.flatten()
    cloud.colors = colors.flatten()
    cloud.sh_degree = sh_degree
    cloud.sh = features_rest.transpose(0, 2, 1).flatten()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(repo_root, "training", "output", "spz_test")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_1000.spz")

    opts = spz.PackOptions()
    opts.from_coord = spz.CoordinateSystem.RUB
    ok = spz.save_spz(cloud, opts, out_path)
    if not ok:
        raise RuntimeError(f"failed to save spz to {out_path}")

    print("test_spz_export ok")
    print("gaussians:", n)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
