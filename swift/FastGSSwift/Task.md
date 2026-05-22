# FastGSSwift Task Plan

## Goal

Build a new SwiftPM-based FastGS port under this repository. The Swift port will use
`mlx-swift` and migrate the current Metal kernels to `MLXFast.metalKernel`, rather
than calling the existing C++ MLX primitives or loading the current `fastgs_core.metallib`.

The first milestone is a macOS Swift app that can run and visualize the FastGS
forward renderer. The iPhone app and real-time camera pipeline will follow after
the macOS path is stable.

## Architecture Direction

- Add a new SwiftPM package under `swift/FastGSSwift/`.
- Add an Xcode project or workspace for a macOS app first, then an iOS app target.
- Depend on `mlx-swift`, preferably through the local `submodules/mlx-swift` checkout during development.
- Use `MLXFast.metalKernel` as the main compute path.
- Keep the existing C++/Metal implementation as a reference implementation only.
- Do not make the Swift package depend on the current Python/nanobind/C++ primitive runtime.

## Public Swift API Target

- `FastGSModel`
  - Holds Gaussian parameters as `MLXArray`.
  - Includes means, scales, rotations, opacity, SH/colors, and any derived buffers needed by forward rendering.
- `FastGSCamera`
  - Holds camera matrices, projection data, field-of-view, viewport, and camera position.
- `FastGSRenderer`
  - Main forward rendering entry point.
  - Target API:

    ```swift
    let frame = renderer.forward(
        model: model,
        camera: camera,
        viewport: viewport
    )
    ```

- `FastGSFrame`
  - Holds rendered output as `MLXArray`.
  - First version returns MLX float RGB output.
  - Presentation conversion to BGRA/texture/IOSurface is separate from core rendering.

## MLXFastKernel Migration Strategy

Detailed implementation notes are tracked in `MigrationNotes.md`.

- Convert each `.metal` file into Swift-managed kernel source:
  - `header`: helper functions, constants, inline math, shared utility code.
  - `source`: kernel body compatible with `MLXFast.metalKernel`.
- Remove explicit Metal kernel signatures such as:

  ```metal
  kernel void name(... [[buffer(n)]], uint2 gid [[thread_position_in_grid]])
  ```

- Use `MLXFastKernel` provided names and variables instead:
  - `thread_position_in_grid`
  - `thread_position_in_threadgroup`
  - `threadgroup_position_in_grid`
  - `threads_per_threadgroup`
  - `threadgroups_per_grid`
- Map old `device` buffer parameters to `inputNames` and `outputNames`.
- Replace C++ `Params struct` usage with:
  - template args for compile-time constants,
  - scalar `MLXArray` inputs for runtime values,
  - shape-derived values from MLXFast-generated shape metadata when possible.
- Keep kernel launch configuration in Swift:
  - grid size
  - threadgroup size
  - output shapes
  - output dtypes
  - output init value where atomic accumulation is used.

## Forward Pipeline Tasks

### 1. SwiftPM Skeleton

- Create `Package.swift` for `FastGSSwift`.
- Add products:
  - `FastGSSwift`
  - optional `FastGSSwiftAppSupport`
- Add dependencies:
  - `MLX`
  - optionally `MLXFast`, `MLXNN`, or `MLXOptimizers` if needed later.
- Add basic unit test target.
- Verify a minimal `MLXFast.metalKernel` runs in Swift.

### 2. Kernel Source Organization

- [x] Add an initial kernel source module for FastGS.
- Use one Swift file per migrated stage:
  - `FastGSPreprocess.swift` currently contains the first preprocess source while the migration shape settles.
  - `PreprocessKernelSource.swift`
  - `BinningKernelSource.swift`
  - `TilePrepKernelSource.swift`
  - `RasterizeKernelSource.swift`
- Keep common Metal helper code in a shared Swift string or builder.
- [x] Add a small debug option to run kernels with `verbose: true`.

### 3. Preprocess Forward

- [x] Port an initial `fastgs_preprocess.metal` forward path through `MLXFast.metalKernel`.
- Preserve the existing math as closely as possible:
  - [x] SH color evaluation through degree 3.
  - [x] frustum check
  - [x] 3D covariance calculation
  - [x] 2D covariance projection
  - [x] conic/opacity calculation
  - [x] tile coverage
- [x] Return the Swift equivalent of current preprocess outputs:
  - [x] radii
  - [x] means2d
  - [x] depths
  - [x] cov3d
  - [x] conic_opacity
  - [x] colors
  - [x] tiles_touched
  - [x] clamped
  - [x] viewspace_points
- [x] Compare all outputs with the existing Python/C++ implementation on the first precomputed-color fixture.
- [x] Add SH degree 3 fixture and compare against existing Python/C++ implementation.
- [x] Add edge parity fixtures against Python/C++ for:
  - [x] near-plane culling and zero-initialized outputs
  - [x] precomputed 3D covariance path
  - [x] SH color clamp flags

### 4. Tile Prep and Binning

- [x] Add initial Swift binning API and output structure.
- [x] Port `fastgs_duplicate_with_keys_kernel` through `MLXFast.metalKernel`.
- [x] Use MLX built-in ops for prefix-sum, argsort, and take.
- [x] Port tile range preparation and bucket count through `MLXFast.metalKernel`.
- [x] Add first binning parity fixture from the precomputed-color preprocess output.
- [x] Add binning edge fixtures for partial culling, all-culled zero-rendered output, and varied depth sorting.
- Prefer MLX built-in ops for prefix-sum/sort if they can replace custom parallel code cleanly.
- Keep custom `MLXFast.metalKernel` only where the existing algorithm depends on bespoke Metal behavior.
- Validate:
  - [x] duplicated keys
  - [x] sorted tile ranges
  - [x] bucket offsets
  - [x] point list ordering for the first fixture
  - [x] culling/zero-rendered binning path
  - [x] varied depth ordering
  - [x] larger tile coverage with non-square spans

### 5. Rasterize Forward

- [x] Add first `fastgs_rasterize.metal` forward smoke port through `MLXFast.metalKernel`.
- [x] Add `FastGSRasterize` Swift API and output structure.
- [x] Add a 1 tile / 1 Gaussian rasterize smoke fixture.
- [x] Validate rasterize smoke output under Xcode.
- [x] Connect the first preprocess -> binning -> rasterize fixture.
- [x] Compare E2E rasterize output summaries and sampled pixels against the existing Python/C++ implementation.
- [x] Add larger 80x48, 5 Gaussian, 5x3 tile E2E fixture with non-square tile spans.
- [x] Add Swift output image export from `FastGSRasterizeOutput.outColor`.
- [x] Validate Swift-generated PNG export under Xcode.
- Port full `fastgs_rasterize.metal` parity fixtures after the smoke path is stable.
- Preserve:
  - [x] per-tile traversal in the smoke path
  - [x] transmittance accumulation in the smoke path
  - [x] color accumulation in the smoke path
  - [x] sampled intermediates needed by future backward support
  - [x] `n_contrib`
  - [x] `max_contrib`
  - [ ] optional metric count path
- [x] First output format is MLX float color array.
- [x] Add a first presentation conversion step for RGBA8/PNG export.
- [x] Add a first Metal texture presentation bridge from `outColor`.
- Remaining rasterize expansion:
  - [ ] full-image exact parity fixture instead of summary/sample checks
  - [ ] metric count path
  - [ ] multi-scene and larger Gaussian fixtures
  - [ ] CVPixelBuffer / IOSurface presentation bridge

### 6. macOS App Preview

- [x] Add a macOS app target through Xcode.
- [x] Load the large static Gaussian fixture.
- [x] Render through the SwiftPM package.
- [x] Display output using a SwiftUI-backed preview.
- [x] Display output through an `MTKView`-backed Metal texture preview.
- [x] Add a basic reload control.
- Do not prioritize live camera controls yet; recorded scanner data forward
  parity is the next validation step.

## Recorded Scanner Data Forward Plan

This is the next phase after the static fixture forward path is stable. Most
FastGS flows render and train from recorded captures, so Swift should first
prove it can consume real recorded data before attempting real-time capture.

Reference data path currently available:

- `/Users/yangdunfu/Downloads/2026_05_04_16_51_29`

Relevant existing Python/C++ targets:

- `make test-scanner`, with the data path adjusted to
  `/Users/yangdunfu/Downloads/2026_05_04_16_51_29`
- `make train-scanner-fastgs2-smoke`
- `make train-scanner-fastgs2`

Planned Swift task order:

- [x] Inspect the recorded scanner dataset format and identify the minimal
  frame/camera/point-cloud files needed for forward-only rendering.
- [x] Generate a Python/C++ recorded reference in `/private/tmp` for a reduced
  160x120, 4096-point forward case.
- [x] Store the recorded reference generator under
  `swift/FastGSSwiftTools/generate_recorded_reference.py`.
- [x] Store large recorded arrays as binary float32 buffers referenced from the
  manifest instead of inline JSON lists.
- [x] Add a small Swift loader for recorded scanner metadata and one selected
  frame, keeping it separate from live camera capture.
- [x] Feed the same recorded frame inputs into Swift `FastGSPreprocess ->
  FastGSBinning -> FastGSRasterize`.
- [x] Compare Swift output against the recorded-data Python/C++ reference using
  summary values and sampled pixels first.
- [x] Render a recorded-data Swift preview image through the existing
  `MTLTexture` / PNG export path.
- [x] Move the recorded-data loader out of the test helper if the macOS app
  should display the recorded forward case interactively.
- [x] Expand from reduced 4096-point reference to larger point counts after the
  Swift path remains stable.
- [x] Tighten 16384-point recorded parity after investigating the current
  larger-scene channel-sum divergence.
- [x] Add recorded stage summaries to localize the 16384-point divergence.
- [x] Fix the 16384-point preprocess color divergence in the SH/DC color path.
- [x] Add a repeatable `make test-swift-recorded-forward` flow that regenerates
  recorded references, runs the Xcode recorded tests, and compares Python/C++
  vs Swift stage summaries.
- [x] Display recorded scanner target and Swift render side by side in the
  macOS app.

## Native Dataset Loading Plan

The current Swift training preview still depends on Python-generated reference
manifests in `/private/tmp`. The next migration step is to let Swift read the
scanner dataset directly, following the same assumptions as the Python scanner
training code.

Initial fixed test dataset:

- `/Users/yangdunfu/Downloads/2026_05_04_16_51_29`

Planned task order:

- [x] Inspect the exact scanner dataset files used by the Python training path:
  camera/frame metadata, target images, point cloud files, and any color fields.
  - Python uses `plyfile.PlyData.read(...)` in
    `scripts/train_scanner_fixed.py` / `scripts/train_scanner_fastgs2.py`.
  - The fixed dataset uses `/Users/yangdunfu/Downloads/2026_05_04_16_51_29/points.ply`.
  - `points.ply` is ASCII PLY (`ply.text == True`) with one `vertex` element
    containing 793602 vertices.
  - Actual vertex properties are:
    `x: float`, `y: float`, `z: float`, `red: uchar`, `green: uchar`,
    `blue: uchar`, `nx: float`, `ny: float`, `nz: float`,
    `curvature: float`.
  - Python loader currently consumes only `x/y/z` and optional
    `red/green/blue`; colors are normalized from `0...255` to `0...1` and
    clipped.
  - The fixed dataset has 159 `frame_*.jpg` files and 958 `frame_*.json`
    files; Python pairs matching frame indices.
  - First image sample is `frame_00000.jpg`, RGB, 1920x1440.
  - Frame JSON uses `intrinsics` length 9 and `cameraPoseARFrame` length 16.
  - Python resizes targets to the requested training size, currently 512x512
    for the Swift app path.
- [x] Decide whether to use a SwiftPM PLY package or write a minimal in-repo
  PLY reader.
  - Use an in-repo minimal reader because the current dataset uses a narrow
    ASCII PLY subset.
  - Required first support is ASCII `vertex` PLY with `x/y/z` float and
    optional `red/green/blue` uchar properties. Ignore normals/curvature for
    the first Swift loader.
  - Implemented `FastGSPLYReader` and `FastGSPointCloud` in the Swift package.
    The first version reads positions as flat float triples and optional RGB
    colors normalized to `0...1`.
  - Added unit tests for a synthetic ASCII PLY and the fixed scanner
    `points.ply` sample values.
- [ ] Add a Swift dataset loader that can read the fixed test directory and
  produce the same first-frame training inputs currently represented by the
  generated manifest.
- [ ] Add unit/Xcode tests comparing the native Swift loader output against the
  existing Python-generated manifest for point count, camera parameters, image
  size, target image samples, and point/color samples.
- [ ] Update the macOS app to choose a dataset directory with an
  `NSOpenPanel`, while keeping the fixed directory as the test/default path.
- [ ] Retire the Python-generated manifest dependency from the main macOS
  training path after native loading parity is stable. Keep the generator only
  as a parity/reference tool.

## IOSurface and Real-Time Camera Plan

This is a later phase after recorded-data forward rendering is working. Keep
the current mock `CVPixelBuffer` bridge as infrastructure, but do not build live
capture yet.

- Camera input usually arrives as `CVPixelBuffer`.
- `CVPixelBuffer` can expose an `IOSurface`.
- [x] Add first `CVPixelBuffer` bridge prototype for 32-bit BGRA frames.
- [x] Validate mock IOSurface-compatible BGRA `CVPixelBuffer` under Xcode.
- [x] Add copy-based `CVPixelBuffer` BGRA -> RGBA `MTLTexture` bridge.
- [x] Validate camera-frame texture bytes under Xcode.
- [x] Add macOS app mock `CVPixelBuffer` preview through the `MTKView` texture path.
- `mlx-swift` supports creating an `MLXArray` from a raw pointer, so the planned camera bridge is:

  ```text
  CVPixelBuffer
    -> IOSurface/baseAddress
    -> MLXArray(rawPointer:shape:dtype:finalizer:)
    -> FastGS forward/update path
  ```

- Core FastGS rendering should still output `MLXArray`.
- Do not require `MLXFastKernel` to write directly into an IOSurface in the first version.
- Presentation path:

  ```text
  FastGSFrame.color MLXArray
    -> presentation conversion
    -> RGBA8 / MTLTexture / CVPixelBuffer / IOSurface
    -> display
  ```

- If zero-copy display becomes necessary, add a dedicated presentation Metal kernel later.
- Treat that presentation kernel separately from the FastGS math migration.
- Next bridge step: decide the ownership model for wrapping a locked
  `CVPixelBuffer` base address as an `MLXArray` without unlocking too early.
- Future app step: feed a live camera `CVPixelBuffer` into the macOS or iOS
  `MTKView` preview path after recorded-data forward parity is useful.

## Backward and Training Plan

- Backward support starts after the recorded forward path is stable.
- Current Swift forward status: the existing `FastGSPreprocess ->
  FastGSBinning -> FastGSRasterize` path does not call `stopGradient` yet.
  That is acceptable for preview/parity, but a training path should not let
  gradients flow through binning, sorting, tile ranges, bucket offsets, radii,
  metric buffers, or other discrete orchestration arrays.
- Keep the preview/inference entry point simple, and add a separate
  training/autograd entry point before exposing backward:

  ```swift
  public enum FastGSForwardGraphMode {
      case inference
      case training
  }
  ```

- Match the existing C++/Python binding's gradient boundaries:
  - stop `pointOffsets = cumsum(tilesTouched)`
  - stop binning inputs used only for discrete scheduling:
    `xys`, `depths`, `conicOpacity`, and `tilesTouched`
  - stop duplicated keys/lists, sorted keys/lists, tile ranges, bucket counts,
    and bucket offsets
  - stop raster-only scheduling buffers such as `radii` and `metricMap`
  - do not stop gradient-critical render values passed into rasterize:
    `means2D`, `colors`, `conicOpacity`, and `viewspacePoints`
  - do not stop original trainable Gaussian parameters before preprocess:
    means, DC/SH colors, opacity, scale, rotation, or precomputed covariance
- Port backward kernels behind explicit Swift APIs first:
  - [x] `FastGSRasterizeBackward.forward(...)` API and MLXFast dispatch skeleton
  - [x] full `fastgs_render_backward_kernel` math port smoke-tested under Xcode
  - [x] `FastGSPreprocessBackward.forward(...)` API and MLXFast dispatch skeleton
    - Current skeleton returns the gradient-critical smoke outputs
      `dL_dmeans3D`, `dL_dDC`, `dL_dopacities`, and `dL_dviewspacePoints`;
      the remaining primal gradients are zero-filled on the Swift side.
    - Xcode dispatch is stable when the MLXFast preprocess backward kernel is
      kept to 4 outputs. A larger single-kernel shape with 7 outputs repeatedly
      restarted the Xcode test host during the early skeleton phase. Do not
      treat that skeleton workaround as the final design: the full math port
      should still follow the original Metal kernel structure inside
      `MLXFast.metalKernel(...)`.
- Rasterize backward remaining work:
  - [x] compare `dL_dmeans2d`, `dL_dcolors`, `dL_dconicOpacity`, and
    `dL_dviewspacePoints` against Python/C++ reference
    - current parity checks compare gradient `sum`, `absSum`, `maxAbs`, and
      fixed samples from `fastgs_rasterize_backward_ref.py`
  - [x] wire rasterize backward output into preprocess backward smoke path
  - [x] port full preprocess backward math into `MLXFast.metalKernel(...)`
    while preserving the original Metal code structure as much as possible
    - Current `fastgs_preprocess_backward_swift_full_v1` is a full-port
      candidate. It is dispatch-stable under Xcode, but should remain marked
      as candidate-quality until gradient parity is broader than the first
      synthetic fixture.
  - [x] add a Python/C++ reference generator for preprocess backward using
    `mx.value_and_grad` over `preprocess_forward`
    - `swift/FastGSSwiftTools/fastgs_preprocess_backward_ref.py` writes
      `/private/tmp/fastgs_preprocess_backward_ref.json`.
    - The first fixtures match the Swift precomputed-color and SH degree-3
      preprocess fixtures and compare loss plus gradient `sum`, `absSum`,
      `maxAbs`, and fixed samples.
  - [ ] expand preprocess backward parity beyond the first synthetic fixture:
    - [x] SH color path
    - [x] precomputed covariance path skipped for now because the normal
      training path uses SH degree inputs rather than precomputed covariance
    - [x] clamping path skipped for now; revisit only if SH training shows
      clamp-related color-gradient issues
    - [ ] recorded scene subsets
- [x] Before continuing the full preprocess backward Metal math port, add an
  end-to-end autograd plumbing test. Single backward-kernel tests are useful,
  but they can miss argument ordering, closure, `CustomFunction`, and
  `valueAndGrad` integration failures.
- [x] Add a Swift training smoke path that renders an image, computes a scalar loss,
  and calls MLX Swift `valueAndGrad` / `buildValueAndGradient` over the
  trainable Gaussian parameters.
  - Use the current recorded-scene forward path as the first input fixture.
  - Use a simple image loss first, preferably MSE or L1 between rendered
    `outColor` and a target image.
  - Assert that the rendered image shape is correct, loss is finite, and all
    returned gradients have the expected shapes.
  - For the first version, the custom backward may return zero gradients for
    every trainable parameter. This is intentional: the goal is to prove the
    full forward -> loss -> backward plumbing before trusting any Metal
    gradient math.
- [x] Add an initial whole-render Swift `CustomFunction` smoke wrapper before
  the full preprocess backward math port. This proved that `valueAndGrad` can
  enter a custom VJP, but it should not become the final training graph shape.

  ```swift
  let render = CustomFunction {
      Forward { inputs in
          // call MLXFast.metalKernel-based preprocess/binning/rasterize
      }
      VJP { primals, cotangents in
          // first return zero gradients with the same shapes as primals
          // later call MLXFast.metalKernel-based rasterize/preprocess backward
      }
  }
  ```

- Replace the whole-render `CustomFunction` with stage-level `CustomFunction`
  wrappers, matching the C++ primitive design where each forward primitive owns
  its VJP:
  - [x] `FastGSRasterizeCustomFunction`
    - Forward returns all rasterize outputs, not only `outColor`:
      `bucketToTile`, `sampledT`, `sampledAr`, `finalT`, `nContrib`,
      `maxContrib`, `pixelColors`, `outColor`, and `metricCount`.
    - The loss should only consume `outColor`, but the VJP can receive the
      forward outputs through the custom function contract rather than
      recomputing rasterize in backward.
    - VJP calls `FastGSRasterizeBackward.forward(...)` and returns gradients for
      rasterize inputs. These become the intermediate gradients for preprocess
      outputs such as `xy`, `rgb`, `conicOpacity`, and `viewspacePoints`.
  - [x] `FastGSPreprocessCustomFunction`
    - Forward returns all preprocess outputs:
      `radii`, `xy`, `depths`, `cov3D`, `rgb`, `conicOpacity`,
      `tilesTouched`, `clamped`, and `viewspacePoints`.
    - VJP consumes cotangents from rasterize backward and calls
      `FastGSPreprocessBackward.forward(...)` to produce gradients for
      trainable Gaussian parameters.
  - Binning/tile scheduling remains a forward-only/discrete bridge with
    `stopGradient` boundaries around scheduling arrays such as offsets, sorted
    lists, ranges, bucket counts, and bucket offsets.
- [x] Add a full stage-level training graph smoke test:
  `FastGSPreprocessCustomFunction -> FastGSBinning.forward with stopGradient
  scheduling -> FastGSRasterizeCustomFunction -> MSE loss -> valueAndGrad`.
  This confirms the Swift graph can pass through rasterize VJP and preprocess
  VJP and return the six recorded-path trainable gradients.
- After stage-level autograd plumbing is stable, replace zero gradients stage by
  stage and validate gradients against the existing implementation before
  attempting real training from Swift.
- `FastGSTrainingBackwardMode` may be added as a temporary migration scaffold:
  - `.zero` keeps the current shape-correct zero-gradient VJP.
  - [x] `.rasterizeOnly` routes `cotangents[0]` through
    `FastGSRasterizeBackward.forward(...)` to prove loss cotangents reach the
    rasterize backward Metal path, while trainable parameter gradients can
    remain zero.
  - `.full` routes rasterize backward output into preprocess backward.
  This mode enum should not become the long-term public API; remove or hide it
  after the full backward path is stable.
- Do not split the full preprocess backward algorithm into smaller kernels as
  the preferred solution. The intended final implementation is one
  `MLXFast.metalKernel(...)` port that mirrors the original
  `fastgs_preprocess_backward.metal` structure, with any MLXFast-specific
  wrapper adjustments kept as small and explicit as possible.

### Primitive-Style Context and Cache Plan

The current Swift stage-level `CustomFunction` path passes most values through
primals, cotangents, and forward outputs. That works for smoke tests, but it is
not the right long-term shape. Like the C++ `mx::Primitive` implementation,
Swift needs explicit structs for non-primal state used by forward and backward.

The goal is to avoid hiding important backward dependencies in ad hoc closures
or recomputing them from unrelated values.

- [ ] Add a `FastGSRenderContext` / `FastGSTrainingContext` struct for stable
  non-primal state:
  - camera matrices
  - camera position
  - image width/height
  - tile bounds
  - background
  - SH degree and max coefficients
  - scale modifier / multiplier
  - dataset/frame identity where useful for debugging
- [ ] Add stage cache structs for forward outputs and scheduling intermediates
  needed by VJP:
  - preprocess outputs used by rasterize and preprocess backward
  - binning outputs, tile ranges, bucket offsets, and stopped scheduling arrays
  - rasterize forward outputs consumed by rasterize backward
- [ ] Update `FastGSPreprocessCustomFunction` and
  `FastGSRasterizeCustomFunction` to use the context/cache structs rather than
  relying on scattered captured values.
- [ ] Clearly document which values are trainable primals, which are
  cotangents, which are stopped scheduling arrays, and which are primitive-style
  context/cache values.
- [ ] Keep the Swift API close to the C++ primitive mental model so future
  backward and native dataset loading changes do not need to reshape the graph
  again.

### Optimizer Plan

- Optimizer state should live on the Swift side, matching the Python MLX model
  where the optimizer updates `MLXArray` parameters after gradients are
  computed.
- Use `submodules/mlx-swift/Source/MLXOptimizers/Optimizers.swift` as the
  primary Swift reference for:
  - per-parameter optimizer state
  - tree-shaped parameter updates
  - SGD/Adam-style update formulas
  - `innerState()` / `eval()` behavior for optimizer state arrays
- Do not require FastGSSwift to use `MLXOptimizers` directly if its `Module`
  and `ModuleParameters` shape is awkward for Gaussian splat data. It is
  acceptable to write a FastGS-specific optimizer wrapper that keeps typed
  Gaussian parameter groups and optimizer state explicitly.
- First optimizer target:
  - [x] add a small typed parameter container for trainable Gaussian arrays
  - [x] add an Adam-style update step for means, colors/SH, opacity, scale, and
    rotation
  - [x] support per-field learning rates because FastGS training usually does
    not update all Gaussian fields with the same schedule
  - [x] keep optimizer state as `MLXArray` buffers so updates remain on device
  - [x] test one synthetic gradient step before connecting to real backward
- Later optimizer work:
  - [ ] checkpoint parameter arrays and optimizer state
  - [ ] support densify/prune state migration when Gaussian count changes
  - [x] add recorded-data loss and one-step training smoke loop after backward parity
    exists
    - `make test-swift-recorded-training-smoke` regenerates the small recorded
      reference, runs Xcode, computes a nonzero image loss, calls
      `valueAndGrad`, and verifies Adam updates at least one trainable array.
    - `make test-swift-recorded-training-loop` runs a 3-step synthetic target
      loop and verifies losses stay finite while the final loss does not exceed
      the first loss by more than 1%.
    - `make test-swift-recorded-training-preview` runs 200 steps against the
      full-point 512x512 recorded target tensor and writes target/render
      side-by-side PNGs every 20 steps to
      `/private/tmp/fastgs_swift_training_preview`.
    - The preview path now regenerates a full-point 512x512 recorded reference
      under `/private/tmp/fastgs_recorded_reference_full_512` instead of using
      the 4096/16384-point reduced fixtures or the earlier small full-point
      preview.
    - The preview path sets MLX Swift `Memory.cacheLimit` to 4 GB before the
      training loop, matching the intended `mlx_set_cache_limit` behavior.
    - The preview path writes `debug_summary.json` and `debug_summary.csv`
      beside the PNGs. These logs track loss, MLX memory snapshot, each
      trainable field's gradient `sum`/`absSum`/`maxAbs`/nonzero count, the
      per-step update magnitude, and the accumulated delta from the initial
      parameters.
- Formal Swift training runner direction:
  - [x] Keep the first real runner Mac-App oriented rather than adding a
    command-line executable. A CLI runner can remain a later convenience idea.
  - [x] Start with fixed-point training only: no densify, prune, opacity reset,
    or optimizer-state migration until the rendered previews and debug logs
    look trustworthy.
  - [x] Represent runner configuration as Swift structs so the macOS app can
    own and mutate training parameters cleanly.
    - `FastGSRecordedTrainingRunConfig` owns total steps, cache limit, learning
      rates, and the current recorded reference set.
    - `FastGSRecordedTrainingReferenceSet` scans per-camera
      `recorded_manifest.json` files and falls back to the root manifest when
      per-camera references have not been generated yet.
  - [x] Add a first macOS app training button that runs one 200-step
    fixed-point training pass and displays the completed target/render result
    in the app.
  - [x] Generate and consume a full-point 512x512 recorded reference from
    `/private/tmp/fastgs_recorded_reference_full_512`. The reference generator
    rebuilds camera metadata for the requested size, so Swift receives the
    matching `tanFovX`/`tanFovY`, image size, target tensor, and manifest.
  - [x] Show current training progress as `Step current / total` in the macOS
    app toolbar.
  - [x] Add previous/next camera buttons in the macOS app toolbar. These switch
    the selected recorded camera target/render pair for the fixed 512x512
    training preview.
  - [x] Add `make swift-recorded-full-512-camera-references` to generate
    per-camera 512x512 full-point manifests under
    `/private/tmp/fastgs_recorded_reference_full_512/camera_000...`.
  - [x] Clean up the macOS app so it only exposes training-related controls:
    dataset selection, camera/frame navigation, training start, progress, and
    target/render display.
    - [x] Remove static fixture forward, mock camera frame, and early reload/test
      forward buttons from the primary app UI.
  - [ ] After fixed-point training is stable, add FastGS-style after-train
    features such as densify and prune as explicit later stages.

## Test Plan

### Kernel Tests

- Add tiny fixtures with known output.
- Test each migrated kernel independently.
- Compare Swift MLXFastKernel output against existing Python/C++ output.
- Include different image sizes and Gaussian counts.

### Pipeline Tests

- Run full forward pipeline on a small fixed Gaussian scene.
- Compare:
  - radii
  - means2d
  - tile ranges
  - point lists
  - final color
  - contribution counts.

### App Tests

- macOS app can load a fixture and render a still frame.
- macOS app can repeatedly render while camera parameters change.
- iOS app can run the same fixture after macOS is stable.

### IOSurface Tests

- Create a mock `CVPixelBuffer`.
- Wrap it as an `MLXArray`.
- Verify lifetime/finalizer behavior.
- Verify format assumptions:
  - width
  - height
  - bytes per pixel
  - row stride
  - channel order.

### Performance Tests

- Measure per-stage latency:
  - preprocess
  - binning
  - tile prep
  - rasterize
  - presentation conversion.
- Compare with the existing C++ primitive path where practical.

## Milestones

### Milestone 1: SwiftPM Proof

- [x] SwiftPM package exists.
- [x] Depends on `mlx-swift`.
- [x] Minimal `MLXFast.metalKernel` smoke wrapper exists.
- [x] `swift test` passes for package loading.
- [x] Run the Metal smoke test in an Xcode/metallib-ready environment.

### Milestone 1.5: Xcode macOS Harness

- [x] Xcode project exists.
- [x] macOS app target exists.
- [x] Xcode test target exists.
- [x] Xcode scheme runs the MLXFast Metal smoke test.
- [x] `xcodebuild test -project FastGSSwift.xcodeproj -scheme FastGSSwiftMac -destination 'platform=macOS'` passes.

### Milestone 2: Preprocess Port

- [x] Preprocess forward runs in Swift.
- [x] Xcode test runs the preprocess `MLXFast.metalKernel` on a small fixture.
- [x] Output matches existing implementation on the first precomputed-color fixture.

### Milestone 3: Full Forward Pipeline

- Preprocess, tile/binning, and rasterize forward run in Swift.
- Static fixture renders an image.

### Milestone 4: macOS App Preview

- macOS app renders from the SwiftPM package.
- Basic camera movement works.
- Frame latency is measured.

### Milestone 5: iOS Fixture Preview

- iOS app target runs the same static fixture.
- Rendering works on device or simulator where Metal/MLX support allows it.

### Milestone 6: Camera / IOSurface Input

- `CVPixelBuffer` or `IOSurface` camera frames can be wrapped into `MLXArray`.
- Real-time update path is prototyped.

### Milestone 7: Backward / Training

- Swift `CustomFunction` wraps forward and VJP.
- Backward kernels are ported and gradient parity is tested.

## Assumptions

- The Swift migration intentionally uses `MLXFast.metalKernel`.
- The current C++ primitive implementation remains the correctness reference.
- The first production-like target is macOS Apple Silicon.
- iPhone support follows after macOS forward rendering works.
- First render output can be MLX float color; direct IOSurface output is not required for the first milestone.
