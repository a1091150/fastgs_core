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
- Remaining rasterize expansion:
  - [ ] full-image exact parity fixture instead of summary/sample checks
  - [ ] metric count path
  - [ ] multi-scene and larger Gaussian fixtures
  - [ ] Metal texture / CVPixelBuffer presentation bridge

### 6. macOS App Preview

- [x] Add a macOS app target through Xcode.
- [x] Load the large static Gaussian fixture.
- [x] Render through the SwiftPM package.
- [x] Display output using a SwiftUI-backed preview.
- [x] Add a basic reload control.
- Add basic camera controls after the static render path is stable.

## IOSurface and Real-Time Camera Plan

This is a later phase after macOS forward rendering is working.

- Camera input usually arrives as `CVPixelBuffer`.
- `CVPixelBuffer` can expose an `IOSurface`.
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
    -> BGRA / CVPixelBuffer / MTLTexture / IOSurface
    -> display
  ```

- If zero-copy display becomes necessary, add a dedicated presentation Metal kernel later.
- Treat that presentation kernel separately from the FastGS math migration.

## Backward and Training Plan

- Backward support is not part of the first milestone.
- After forward parity is stable, add Swift `CustomFunction` wrappers:

  ```swift
  let render = CustomFunction {
      Forward { inputs in
          ...
      }
      VJP { primals, cotangents in
          ...
      }
  }
  ```

- Port backward kernels in this order:
  - rasterize backward
  - preprocess backward
- Keep forward intermediates available as `MLXArray` outputs so VJP can consume them.
- Validate gradients against the existing implementation before attempting training from Swift.

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
