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
- Primitive-first migration:
  - Task 3.1: `fastgs_preprocess`
  - Task 3.2: `fastgs_tile_prep`
  - Task 3.3: `fastgs_binning`
  - Task 3.4: `fastgs_rasterize`
  - Task 3.5: external API wiring in `binding.cpp` (`rasterize_gaussians` etc.)

## Status
- [x] Task definition and migration chain analysis completed.
- [x] End-to-end forward path callable (smoke-level) completed.
- [ ] CUDA parity for forward path completed.
- [x] Forward parity check temporarily deferred for this iteration (user-approved).

---

# Task 3.1 (Preprocess Migration)

## Scope
- Migrate preprocess stage first (forward path first, backward later).
- Use existing `fastgs_mlx/fastgs_core` preprocess implementation as reference.

## Planned Files
- `fastgs_core/include/fastgs_preprocess.h`
- `fastgs_core/fastgs_preprocess.cpp`
- `fastgs_core/metal/fastgs_preprocess.metal`

## Task 3.1 Steps
- [x] Define preprocess params/input/output contracts in header.
- [x] Implement MLX Primitive wrapper and `fastgs_preprocess(...)` API in cpp.
- [x] Implement Metal kernel `fastgs_preprocess_forward_kernel`.
- [x] Wire kernel dispatch arguments and output buffers.
- [x] Expose callable binding route from `_fastgs_core` (dictionary input style).
- [x] Add minimal smoke test script in `scripts/preprocess_smoke.py`.

## Task 3.1 Validation
- [x] `make pyext-build` passes.
- [x] `python scripts/preprocess_smoke.py` passes.
- [x] Preprocess binding call returns expected tensor shapes/dtypes.
- [ ] Full parity with `forward.cu::preprocessCUDA` confirmed.
- [x] Forward parity validation deferred for now (temporary pass condition: render output acceptable on local run).

## Notes
- Forward preprocess is available; backward remains out of scope for this stage.
- Earlier implementation included temporary approximations (not CUDA-equivalent). Current direction is CUDA parity-first.
- Do not mark Task 3.1 complete for parity until numeric/statistical comparison with CUDA reference is added.

---

# Task 3.2 (Tile Prep Primitive)

## Scope
- Implement `fastgs_tile_prep` as one dedicated MLX Primitive.
- Input from sorted key stream; output per-tile ranges / bucket metadata.

## Planned Files
- `fastgs_core/include/fastgs_tile_prep.h`
- `fastgs_core/fastgs_tile_prep.cpp`
- `fastgs_core/metal/fastgs_tile_prep.metal`

## Steps
- [x] Define params/input/output contracts.
- [x] Implement primitive wrapper + GPU dispatch.
- [x] Implement Metal forward kernel.
- [x] Add smoke validation script/section.

## Validation
- [x] Build passes.
- [x] Output shapes/types match expected contracts.

---

# Task 3.3 (Binning Primitive)

## Scope
- Implement `fastgs_binning` as one dedicated MLX Primitive.
- Generate point list keys / unsorted lists from preprocess geometry outputs.

## Planned Files
- `fastgs_core/include/fastgs_binning.h`
- `fastgs_core/fastgs_binning.cpp`
- `fastgs_core/metal/fastgs_binning.metal`

## Steps
- [x] Define params/input/output contracts.
- [x] Implement primitive wrapper + GPU dispatch.
- [x] Implement Metal forward kernel.
- [x] Add smoke validation script/section.

## Validation
- [x] Build passes.
- [x] Output shapes/types match expected contracts.

---

# Task 3.4 (Rasterize Primitive)

## Scope
- Implement `fastgs_rasterize` as one dedicated MLX Primitive.
- Produce final image (`out_color`) and required intermediate accumulators.

## Planned Files
- `fastgs_core/include/fastgs_rasterize.h`
- `fastgs_core/fastgs_rasterize.cpp`
- `fastgs_core/metal/fastgs_rasterize.metal`

## Steps
- [x] Define params/input/output contracts.
- [x] Implement primitive wrapper + GPU dispatch.
- [x] Implement Metal forward kernel.
- [x] Add smoke validation script/section.

## Validation
- [x] Build passes.
- [x] `out_color` shape/dtype correct and non-empty for test fixture.

---

# Task 3.5 (External API Wiring)

## Scope
- Wire Task 3.1 ~ 3.4 primitives into outward-facing APIs in `binding.cpp`.
- Keep migration-friendly API style aligned with original FastGS flow.

## Planned APIs
- `_fastgs_core.rasterize_gaussians(...)` (forward)
- `_fastgs_core.preprocess_forward(...)` (already present)
- (later) backward APIs after forward completion

## Steps
- [x] Add end-to-end forward chain in binding (`preprocess -> binning -> tile_prep -> rasterize`).
- [x] Return forward outputs in stable dict schema.
- [x] Add end-to-end forward smoke script.

## Validation
- [x] Python smoke call works end-to-end (user local run).
- [x] Forward returns expected keys/shapes (smoke-level).
- [x] `make pyext-build` passes with `rasterize_gaussians_forward` linked.
- [ ] CUDA parity validation for end-to-end forward outputs.
- [x] Add render smoke script with fixed 2048 gaussians and image output (`scripts/render_2048_smoke.py`).

---

# Task 4 (Backward Path Migration, CUDA -> Metal Full Parity)

## Scope
- Migrate FastGS backward path from CUDA/PyTorch extension to Metal + MLX Primitive end-to-end.
- Keep call path behavior aligned with FastGS training loop (`loss.backward()` -> rasterizer backward).
- Adopt a strict primitive pairing architecture:
  - `fastgs_*` forward primitive in `<name>.cpp` / `<name>.metal`
  - matching backward primitive in `<name>_backward.cpp` / `<name>_backward.metal`
  - `vjp()` in forward primitive delegates to matching backward primitive.
- No transitional fallback implementation is allowed.
  - No `stop_gradient`-based temporary detours for backward-critical tensors.
  - No placeholder `vjp()` that throws at runtime for migrated backward path.
  - No partial backward that only returns subset gradients for final integration stage.

## FastGS Reference Backward Call Path (Authoritative)
- `train.py`
  - `render_fastgs(...)` returns `render`, `viewspace_points`, `radii`.
  - `loss.backward()` triggers autograd backward.
  - `gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)` consumes `viewspace_point_tensor.grad`.
- `gaussian_renderer/__init__.py`
  - `screenspace_points = torch.zeros((P, 4), requires_grad=True, device="cuda")`
  - `means2D = screenspace_points` (dummy trainable parameter for 2D/view-space gradient flow)
- `diff_gaussian_rasterization_fastgs/__init__.py`
  - `_RasterizeGaussians.apply(...)` forward
  - `_RasterizeGaussians.backward(...)` calls `_C.rasterize_gaussians_backward(...)`
  - Backward returns gradients including `grad_means2D`.
- CUDA binding and rasterizer
  - `rasterize_points.cu::RasterizeGaussiansBackwardCUDA(...)`
  - `CudaRasterizer::Rasterizer::backward(...)`
  - `BACKWARD::render(...)` then `BACKWARD::preprocess(...)`

## Backward Data/Gradient Contracts to Preserve
- `means2D/xys` gradient contract:
  - shape must remain `[P, 4]` (FastGS uses 4 channels and splits `:2` / `2:` in densification stats).
  - gradient must propagate to `viewspace_points` output tensor without graph break.
- Required backward gradients for parity path:
  - `dL_dmeans2D`, `dL_dmeans3D`, `dL_dopacity`, `dL_dcolors` (or `dL_ddc`/`dL_dsh` path),
  - `dL_dcov3D` (or `dL_dscale` + `dL_drot` path depending on geometry mode).
- Forward intermediates required by backward must be preserved in Metal-side saved buffers (equivalent role to CUDA `geomBuffer/binningBuffer/imgBuffer/sampleBuffer`).

## Task 4.1 (Autograd/VJP Plumbing in fastgs_core2)

### Scope
- Implement MLX-side backward plumbing so end-to-end `value_and_grad` / `vjp` works for rasterization chain.
- Keep `vjp()` thin: it should call dedicated backward primitive(s), not inline full backward math.
- Remove backward-breaking graph cuts in external API wiring for tensors that require gradients.

### Planned Files
- `fastgs_core/fastgs_preprocess.cpp`
- `fastgs_core/fastgs_preprocess_backward.cpp`
- `fastgs_core/fastgs_binning.cpp`
- `fastgs_core/fastgs_binning_backward.cpp` (if backward is required on training-critical path)
- `fastgs_core/fastgs_tile_prep.cpp`
- `fastgs_core/fastgs_tile_prep_backward.cpp` (if backward is required on training-critical path)
- `fastgs_core/fastgs_rasterize.cpp`
- `fastgs_core/fastgs_rasterize_backward.cpp`
- `fastgs_core/binding/binding.cpp`
- headers under `fastgs_core/include/` for backward signatures

### Steps
- [ ] Implement `vjp()` for migrated primitives needed on training critical path.
- [ ] Ensure each `vjp()` dispatches to corresponding `*_backward` primitive.
- [ ] Ensure `binding.cpp` does not apply `mx::stop_gradient(...)` on backward-critical tensors (`xys/means2d`, etc.).
- [ ] Add explicit backward API/output schema for debug parity dumps (optional runtime flag).
- [ ] Add shape/dtype assertions for backward outputs (`[P,4]` on means2D gradient is mandatory).

### Validation
- [ ] `make pyext-build` passes.
- [ ] Minimal autograd call (`mx::value_and_grad` or Python-facing equivalent) executes without `vjp not implemented` exceptions.

---

## Task 4.2 (Raster Backward: Pixel -> Gaussian Params)

### Scope
- Port CUDA backward render stage (`BACKWARD::render`) to Metal equivalent kernels.
- Reconstruct gradients from per-pixel loss to Gaussian-space intermediates.

### Planned Files
- `fastgs_core/fastgs_rasterize_backward.cpp`
- `fastgs_core/metal/fastgs_rasterize_backward.metal`
- `fastgs_core/fastgs_rasterize.cpp`
- `fastgs_core/include/fastgs_rasterize.h`

### Steps
- [ ] Implement Metal kernels for raster backward accumulation (`dL_dmean2D`, `dL_dconic`, `dL_dopacity`, `dL_dcolor`).
- [ ] Preserve CUDA-equivalent indexing semantics for tile/bucket traversal.
- [ ] Verify gradient buffer initialization/atomic accumulation semantics.

### Validation
- [ ] Backward output tensors have correct shapes/dtypes and finite values.
- [ ] Deterministic repeatability check (same seed/input -> stable gradients within tolerance).

---

## Task 4.3 (Preprocess Backward: 2D/Conic -> 3D Params)

### Scope
- Port CUDA backward preprocess stage (`BACKWARD::preprocess`) to Metal equivalent.
- Finish propagation from screen-space gradients to 3D Gaussian parameters.

### Planned Files
- `fastgs_core/fastgs_preprocess_backward.cpp`
- `fastgs_core/metal/fastgs_preprocess_backward.metal`
- `fastgs_core/fastgs_preprocess.cpp`
- `fastgs_core/include/fastgs_preprocess.h`

### Steps
- [ ] Implement gradients for covariance path and parameterized geometry path.
- [ ] Implement SH/color-related backward propagation (`dc/sh` path) with clamping-consistent rules.
- [ ] Ensure means3D gradient accumulation includes all required contributions.

### Validation
- [ ] Gradient outputs for means3D/opacities/scales/rotations/(dc,sh or colors) are finite and non-trivial.
- [ ] Small-case numerical gradient checks pass tolerance gates.

---

## Task 4.4 (End-to-End Backward Integration)

### Scope
- Wire full backward path into public rasterization API used by training loop equivalent.
- Make backward path consumable by densification logic requiring `viewspace_points.grad`.

### Planned Files
- `fastgs_core/binding/binding.cpp`
- test scripts under `scripts/`

### Steps
- [ ] Add end-to-end backward callable route from `_fastgs_core.rasterize_gaussians(...)` pipeline.
- [ ] Ensure output includes view-space tensor whose gradient is connected and retrievable.
- [ ] Add integration checks for densification-style gradient split:
  - `grad[:, :2]` and `grad[:, 2:]` both available and valid.

### Validation
- [ ] End-to-end backward smoke script passes.
- [ ] `viewspace_points.grad` exists, shape `[P,4]`, finite, and non-zero ratio above minimum threshold in fixture.

---

## Task 4.5 (Backward Test Matrix and Parity Criteria)

### Scope
- Provide robust, repeatable test strategy for backward path correctness and parity confidence.

### Planned Scripts
- `scripts/backward_smoke.py`
- `scripts/backward_grad_contract_smoke.py`
- `scripts/backward_numeric_check.py`
- `scripts/backward_parity_compare.py`

### Test Matrix
- Contract tests:
  - [ ] Verify mandatory gradient presence and shape constraints.
  - [ ] Verify `means2D/xys` gradient contract (`[P,4]`, split channels used by densification logic).
- Numerical tests:
  - [ ] Finite-difference checks on sampled parameters (`means3D`, `opacity`, `scale`, `rotation`, optional color path).
- Stability tests:
  - [ ] Repeat-run gradient consistency under fixed seeds.
- Reference parity tests:
  - [ ] Compare Metal gradients to CUDA reference snapshots for selected fixtures.

### Acceptance Criteria (Task 4 Exit)
- [ ] No `vjp not implemented` on training-critical backward path.
- [ ] No backward-critical `stop_gradient` graph breaks in final path.
- [ ] End-to-end backward executes and returns complete required gradients.
- [ ] Densification-consumed gradient contract works (`viewspace_points.grad` split logic compatible).
- [ ] Backward parity report documented with tolerance and residual gaps.

## Notes
- This task explicitly disallows transitional implementations for final merged path.
- Temporary debug instrumentation is allowed only if removable and does not alter math semantics.
