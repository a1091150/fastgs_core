# Task 1

## Scope
- Build a minimal dummy baseline for `fastgs_core2` using root-level build entry.
- Keep `fastgs_core/` as code-only directory.
- Use conda env `fastgs_core`.
- Add `scripts/` for utility python or shell scripts.
- Add `python_package/` for pip package build/install artifacts (for example egg-info).
- Add pip package config files for `fastgs_core` (based on scratch reference).

## Completed
- [x] Remove previous docs-based workflow files (`docs/`).
- [x] Create code directory structure under `fastgs_core/`:
  - [x] `fastgs_core/include/`
  - [x] `fastgs_core/metal/`
  - [x] `fastgs_core/binding/`
  - [x] `fastgs_core/test/`
- [x] Create dummy source files:
  - [x] `fastgs_core/include/dummy.h`
  - [x] `fastgs_core/dummy.cpp`
  - [x] `fastgs_core/metal/dummy.metal`
  - [x] `fastgs_core/binding/binding.cpp`
  - [x] `fastgs_core/test/test.cpp`
- [x] Move build entry to repository root:
  - [x] `CMakeLists.txt` at `fastgs_core2/`
  - [x] `Makefile` at `fastgs_core2/`
- [x] Ensure build outputs are generated in root build folders:
  - [x] `build/`
  - [x] `build_xcode/` (configured by target)
- [x] Remove old subdirectory build entry to avoid confusion:
  - [x] Delete `fastgs_core/CMakeLists.txt`
  - [x] Clean old `fastgs_core/build*` outputs
- [x] Create `scripts/` directory for python/sh helpers.
- [x] Create and update `scripts/dummy.py` to call package function `fastgs_core.dummy_add`.
- [x] Create `python_package/` directory for package artifacts.
- [x] Create pip package files:
  - [x] `setup.py`
  - [x] `pyproject.toml`
  - [x] `python_package/fastgs_core/__init__.py`
- [x] Update `Makefile` with package build/install targets:
  - [x] `pip-install`
  - [x] `pip-develop`
  - [x] `pip-wheel`

## Validation Completed
- [x] `make cmake-configure`
- [x] `make pyext-build`
- [x] `make test-run`
- [x] Python smoke test:
  - [x] `import _fastgs_core`
  - [x] `_fastgs_core.dummy_add(1, 2) == 3`

## Notes
- This baseline is intentionally dummy-only for build/link/runtime validation.

---

# Task 2

## Scope
- Add primitive auto-generator script based on scratch reference.
- Make CMake auto-include newly generated `.cpp` and `.metal` files.
- Add Makefile command for primitive generation.
- Add helper utility files (`helper.h` / `helper.cpp`) with `fastgs_core` namespace.

## Completed
- [x] Add primitive generator script:
  - [x] `scripts/mlx_cxx_primitive_generate.py`
- [x] Align generator with this repo layout:
  - [x] Header output: `fastgs_core/include/<name>.h`
  - [x] CPP output: `fastgs_core/<name>.cpp`
  - [x] Metal output: `fastgs_core/metal/<name>.metal`
  - [x] Namespace changed to `fastgs_core`
- [x] Fix generator template formatting/runtime bugs:
  - [x] Escape braces for `str.format()` in template namespaces
  - [x] Fix `to_snake()` regex replacement
- [x] Add Makefile primitive generation target:
  - [x] `make gen-primitive CLASS=Foo [FORCE=1]`
- [x] Update root CMake to auto-compile generated files:
  - [x] `file(GLOB FASTGS_CPP_SOURCES CONFIGURE_DEPENDS "fastgs_core/*.cpp")`
  - [x] `file(GLOB FASTGS_METAL_SOURCES CONFIGURE_DEPENDS "fastgs_core/metal/*.metal")`
- [x] Add helper files (from scratch reference, namespace adjusted):
  - [x] `fastgs_core/include/helper.h`
  - [x] `fastgs_core/helper.cpp`

## Validation Completed
- [x] `python3 scripts/mlx_cxx_primitive_generate.py --help`
- [x] Generate test primitive successfully:
  - [x] `python3 scripts/mlx_cxx_primitive_generate.py TempPrimitive --force`
- [x] Root CMake reconfigure after auto-source updates:
  - [x] `make cmake-configure`

## Notes
- Generated test primitive files currently exist:
  - `fastgs_core/include/temp_primitive.h`
  - `fastgs_core/temp_primitive.cpp`
  - `fastgs_core/metal/temp_primitive.metal`
- Next step is to generate first real FastGS forward primitive and wire binding API.

---

# Task 3

## Scope
- Migrate `diff-gaussian-rasterization_fastgs` from CUDA/PyTorch extension to Metal + MLX Primitive.
- Keep Python training call path conceptually aligned with original FastGS flow.
- Build staged primitives and wire them through `_fastgs_core` binding.

## Source Call Path (FastGS Reference)
- `train.py` -> `render_fastgs(...)`
- `gaussian_renderer/__init__.py` -> `GaussianRasterizer(...)` -> `rasterize_gaussians(...)`
- `diff_gaussian_rasterization_fastgs/__init__.py` -> `_RasterizeGaussians.apply(...)`
- `_C.rasterize_gaussians(...)` / `_C.rasterize_gaussians_backward(...)`
- `ext.cpp` pybind exports -> `rasterize_points.cu` CUDA entry points

## Target Architecture (fastgs_core2)
- Namespace: `fastgs_core`
- Python module: `_fastgs_core`
- Planned primitive stages:
  - `fastgs_preprocess`
  - `fastgs_tile_prep`
  - `fastgs_binning`
  - `fastgs_rasterize`
  - `fastgs_preprocess_backward`
  - `fastgs_rasterize_backward`
- Planned file naming:
  - Header: `fastgs_core/include/fastgs_<stage>.h`
  - CPP: `fastgs_core/fastgs_<stage>.cpp`
  - Metal: `fastgs_core/metal/fastgs_<stage>.metal`

## Binding/API Plan
- Provide PyTorch-like entry naming on Python side for migration clarity:
  - `_fastgs_core.rasterize_gaussians(...)`
  - `_fastgs_core.rasterize_gaussians_backward(...)`
  - `_fastgs_core.mark_visible(...)` (later)
  - `_fastgs_core.adam_update(...)` (later)

## Validation Plan
- Forward-first validation with deterministic fixture and image output.
- Backward validation by finite-difference check on reduced tensor sizes.
- Smoke path: Python import -> binding call -> no crash -> expected output shape/types.

## Status
- [x] Task definition and migration chain analysis completed.
- [ ] Implementation started.

---

# Task 3.1 (Preprocess Migration)

## Scope
- Migrate preprocess stage first (forward path first, backward later).
- Use existing `fastgs_mlx/fastgs_core` preprocess implementation as reference.

## Planned Files
- `fastgs_core/include/fastgs_preprocess.h`
- `fastgs_core/fastgs_preprocess.cpp`
- `fastgs_core/metal/fastgs_preprocess.metal`
- (later) `fastgs_core/include/fastgs_preprocess_backward.h`
- (later) `fastgs_core/fastgs_preprocess_backward.cpp`
- (later) `fastgs_core/metal/fastgs_preprocess_backward.metal`

## Task 3.1 Steps
- [x] Define preprocess params/input/output contracts in header.
- [x] Implement MLX Primitive wrapper and `fastgs_preprocess(...)` API in cpp.
- [x] Implement Metal kernel `fastgs_preprocess_forward_kernel`.
- [x] Wire kernel dispatch arguments and output buffers.
- [ ] Expose callable binding route from `_fastgs_core`.
- [ ] Add minimal smoke test script in `scripts/` or `training/test/`.

## Task 3.1 Validation
- [x] `make pyext-build` passes.
- [ ] Preprocess binding call returns expected tensor shapes/dtypes.
- [ ] No Metal runtime error for basic fixture.

## Notes
- Keep forward correctness as first priority.
- Backward path for preprocess is tracked after forward stability.
