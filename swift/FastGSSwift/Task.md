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

- Add a kernel source module for FastGS.
- Use one Swift file per migrated stage:
  - `PreprocessKernelSource.swift`
  - `BinningKernelSource.swift`
  - `TilePrepKernelSource.swift`
  - `RasterizeKernelSource.swift`
- Keep common Metal helper code in a shared Swift string or builder.
- Add a small debug option to run kernels with `verbose: true`.

### 3. Preprocess Forward

- Port `fastgs_preprocess.metal` first.
- Preserve the existing math as closely as possible:
  - SH color evaluation
  - frustum check
  - 3D covariance calculation
  - 2D covariance projection
  - conic/opacity calculation
  - tile coverage
- Return the Swift equivalent of current preprocess outputs:
  - radii
  - means2d
  - depths
  - cov3d
  - conic_opacity
  - colors
  - tiles_touched
  - clamped
  - viewspace_points
- Compare all outputs with the existing Python/C++ implementation on small fixtures.

### 4. Tile Prep and Binning

- Port tile range preparation and binning after preprocess is stable.
- Prefer MLX built-in ops for prefix-sum/sort if they can replace custom parallel code cleanly.
- Keep custom `MLXFast.metalKernel` only where the existing algorithm depends on bespoke Metal behavior.
- Validate:
  - duplicated keys
  - sorted tile ranges
  - bucket offsets
  - point list ordering.

### 5. Rasterize Forward

- Port `fastgs_rasterize.metal`.
- Preserve:
  - per-tile traversal
  - transmittance accumulation
  - color accumulation
  - sampled intermediates needed by future backward support
  - `n_contrib`
  - `max_contrib`
  - optional metric count path.
- First output format is MLX float color array.
- Add a separate presentation conversion step later.

### 6. macOS App Preview

- Add a macOS app target through Xcode.
- Load a small static Gaussian fixture.
- Render through the SwiftPM package.
- Display output using a simple MetalKit or SwiftUI-backed preview.
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

- Preprocess forward runs in Swift.
- Output matches existing implementation on small fixtures.

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
