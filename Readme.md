# FastGS MLX

This project reimplements [FastGS](https://github.com/fastgs/FastGS) using MLX and Metal code. It supports both `forward` image rendering and `backward` model training.

FastGS differs from 3DGS. In 3DGS, `torch.Tensor.retain_grad()` is used to preserve `means2d`-related information for later training stages. MLX does not provide an equivalent to `retain_grad()`. FastGS instead uses `viewspace_points`(a dummy trainable parameter) to preserve `means2d.grad` and the required intermediate information, making it possible to obtain gradients in MLX as well.

### Features
1. Build the project with CMake.
2. Generate an Xcode project for easier Metal code testing and code completion.
3. Pure MLX-based 3DGS training and rendering.
4. Since there are still very few MLX C++ custom extension examples, this project serves as a useful template.

### Note
The project is still under development.

## Environment Setup

- Install Xcode from the App Store.
- Install CMake with `brew install cmake`.
- Install Conda. The `Makefile` assumes a Conda environment by default.

1. Create a `fastgs_core` virtual environment. Python 3.11 is used by default.

```shell
conda create -n fastgs_core python=3.11
```

2. Install the required pip packages.

```shell
pip install mlx==0.30.0 nanobind cmake opencv-python plyfile
```

3. Install `fastgs_core`.

```shell
pip install .
```

4. Install `spz`.

```shell
git submodule update --init --recursive
cd submodules/spz
git checkout ef094fd1a96ca6ff414d72d7904ee4f4f6d97be9
pip install .
```

Notes:
- The MLX version is not limited to `0.30.0`, but nanobind may produce `Incompatible function arguments`. If that happens, you may need to test compatible MLX and nanobind versions yourself.
- The latest `spz` version (`b2a63b9204c2989de713e4e426a28eeaa415643e`) has an issue on exporting spz.

## Quick Start

## Dataset

At the moment, only datasets scanned with the [3D Scanner iPhone App](https://3dscannerapp.com/) are supported for training.

1. Use an iPhone with LiDAR support.
2. Install the 3D Scanner App.
3. Capture in landscape orientation, with the front camera on the left side.
4. Choose the Point Cloud option for scanning.
5. Export using the "All Data" option and transfer it to your Mac with AirDrop.

## Commands

You can refer to the `Makefile`. The default Python environment is `CONDA_ENV ?= fastgs_core`. Update the `--data` path as needed so it points to your training dataset.

### Project Compilation and Build

- `env-check`: Verify the environment.
- `gen-primitive CLASS=Foo`: Create an MLX Primitive class named `Foo` and place it in the corresponding directory.
- `pyext-build`: Test building the Python MLX extension.
- `cmake-configure`, `test-build`, `test-run`: Test building the Xcode project using `build/`.
- `xcode-configure`, `xcode-build`: Test building the Xcode project using `build_xcode/`.
- `pip-install`, `pip-develop`, `pip-wheel`: Install the `fastgs` package.

### FastGS Training and Testing

#### FastGS Training

- `test-scanner`: Export the specified dataset as `spz` and side-by-side images.
- `train-scanner-fixed`: Train with a fixed number of gaussians.
- `train-scanner-fastgs`: Run the standard FastGS training pipeline.
- `train-scanner-fastgs-smoke`: Run a smaller FastGS training job for smoke testing.
- `train-scanner-fastgs-bbox`: Fill with a large number of gaussians and train.
- `train-scanner-fastgs-densify`, `train-scanner-fastgs-densify2`, `train-scanner-fastgs-densify3`: Train with different densification parameters.

`train-scanner-fastgs-densify3` is currently the recommended training command.

#### Scripts

The `scripts/` directory provides some Python scripts for testing:

- `scripts/train_square.py`: Train gaussians using a fixed image and export `spz` plus side-by-side images.
- `scripts/render_2048_cube_smoke.py`: Render a Rubik's Cube and export `spz`.

## Acknowledgements

MLX does not provide a training workflow equivalent to PyTorch's gradient retention behavior. By chance, I found that FastGS exposes the `means2d` gradient, which made the MLX port substantially easier. That removed the need to manually wire up a separate backward path just to retrieve gradients.

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://github.com/graphdeco-inria/gaussian-splatting)
- [FastGS: Training 3D Gaussian Splatting in 100 Seconds](https://github.com/fastgs/FastGS)

## Additional Notes

- Use the [SuperSplat SPZ Online Viewer](https://superspl.at/editor) to inspect `spz` files.
- This project contains code generated with Codex 5.3 and 5.4, so you may notice some weird implementations. Translating CUDA code to Metal code is not one of Codex's strengths, and it may produce simplified or incorrect results. The Python code has generally been much more reliable.
- MLX does not support boolean indexing; use ranges instead.
- For both 3DGS and FastGS, the quality of the trained gaussians depends heavily on the design of the post-training pipeline.
