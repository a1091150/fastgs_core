# FastGSSwift Migration Notes

This file records practical notes found while migrating the existing FastGS
Metal kernels to `mlx-swift` `MLXFast.metalKernel`.

## MLXFast Kernel Shape

`MLXFast.metalKernel` accepts a Metal function body, not a full `kernel void`
function. Keep helper functions and constants in `header`, and put only the
thread body in `source`.

Do not keep explicit buffer attributes from the C++ primitive version:

```metal
kernel void fastgs_preprocess_forward_kernel(
    device const float* means3d [[buffer(2)]],
    device float* radii [[buffer(14)]],
    uint tid [[thread_position_in_grid]])
```

Use the names passed to `inputNames` and `outputNames` directly:

```metal
uint tid = thread_position_in_grid.x;
radii[tid] = 0;
float3 p = read_packed_float3(means3d, tid);
```

MLXFast also generates shape metadata variables. For an input named `means3d`,
`means3d_shape[0]` is available in the kernel body.

## Input Address Spaces

The most important difference from the current C++ primitive kernels is address
space. The handwritten C++ Metal kernels use signatures such as
`device const float*`, but `MLXFast.metalKernel` can generate a mix of
`constant` and `device` address spaces for inputs.

During the first preprocess port, these mismatches appeared:

- `means3d` was accepted as `constant float*`.
- In the larger 5-Gaussian fixture, `means3d` appeared as `device float*`.
- `viewmatrix`, `projmatrix`, and some later inputs appeared as `device float*`.
- `shs` appeared as `device float*` when the SH color path was exercised.
- Empty SH buffers can appear as `constant float*`, even when `means3d` and
  `dc` appear as `device float*`, so SH helpers need mixed overloads too.
- Output arrays are `device` address space.

For reusable helpers, prefer address-space overloads:

```metal
inline float3 read_packed_float3(const constant float* arr, uint idx) {
  return float3(arr[3 * idx], arr[3 * idx + 1], arr[3 * idx + 2]);
}

inline float3 read_packed_float3(const device float* arr, uint idx) {
  return float3(arr[3 * idx], arr[3 * idx + 1], arr[3 * idx + 2]);
}
```

The same pattern is needed for matrix helpers and any helper that accepts input
arrays. For helpers that combine multiple inputs, such as `in_frustum` and
`compute_color_from_sh`, add overloads for the input combinations that Xcode
Metal validation exposes. SwiftPM compilation alone will not catch these.

Do not pass zero-size arrays into `MLXFast.metalKernel` for named inputs, even
when the current runtime flags mean that input is unused. With Metal API
validation enabled, MLX may skip binding the zero-size buffer while the generated
custom kernel still declares the argument, causing errors such as:

```text
missing Buffer binding at index 4 for shs[0]
```

The preprocess Swift wrapper now replaces empty optional inputs (`dc`, `sh`,
`colorsPrecomputed`, `scales`, `rotations`, and `cov3DPrecomputed`) with small
non-empty dummy buffers before dispatch. The real input validation still applies
for whichever path is active.

## Runtime Params

The C++ primitive preprocess kernel used a packed `PreprocessKernelParams`
struct passed through `set_bytes`. `MLXFast.metalKernel` does not expose that
same launch API from Swift.

The current Swift port passes runtime params as a small float32 `MLXArray`:

```swift
MLXArray([
    Float(degree),
    Float(maxSHCoefficients),
    scaleModifier,
    multiplier,
    tanFovX,
    tanFovY,
    focalX,
    focalY,
    Float(imageWidth),
    Float(imageHeight),
    Float(tileBounds.x),
    Float(tileBounds.y),
    Float(tileBounds.z),
    prefiltered ? 1.0 : 0.0,
    useCov3DPrecomputed ? 1.0 : 0.0,
    useColorsPrecomputed ? 1.0 : 0.0,
], [16])
```

Inside the Metal body, cast back to the expected type:

```metal
int degree = int(params[0]);
uint image_width = uint(params[8]);
bool use_colors_precomp = params[15] != 0.0f;
```

This is less elegant than a struct, but it keeps the Swift API simple and works
with `MLXFast.metalKernel`. Later, if a value is truly compile-time stable, move
it to `template` args.

## Output Initialization

The C++ primitive manually allocated and zeroed outputs before dispatch. In
MLXFast, use `initValue` when the kernel depends on zero-initialized output
buffers:

```swift
kernel(
    inputs,
    grid: (count, 1, 1),
    threadGroup: (threadGroupSize, 1, 1),
    outputShapes: shapes,
    outputDTypes: dtypes,
    initValue: 0
)
```

This matters for kernels with early returns such as preprocess. Without
initialization, culled gaussians can leave undefined output values.

## Testing Rules

`swift test` is useful for compiling the Swift package and checking non-Metal
API shape, but it is not enough for these kernels. In this repository, CLI
SwiftPM tests intentionally skip `MLXFast.metalKernel` execution unless
`FASTGS_RUN_METAL_TESTS=1`.

Use the Xcode project for real Metal validation:

```bash
cd swift/FastGSSwiftApps
xcodebuild test \
  -project FastGSSwift.xcodeproj \
  -scheme FastGSSwiftMac \
  -destination 'platform=macOS'
```

This has already caught MLXFast-specific address-space errors that `swift test`
could not catch.

## Preprocess Port Status

The first Swift preprocess port is in:

- `Sources/FastGSSwift/FastGSPreprocess.swift`

Current coverage:

- frustum check
- 3D covariance calculation
- precomputed 3D covariance path
- 2D covariance projection
- conic/opacity calculation
- tile coverage
- precomputed color path
- degree 0/1/2/3 SH path
- SH color clamp flags
- early-return culling with zero-initialized outputs
- all current preprocess output buffers
- parity against Python/C++ for:
  - precomputed color fixture
  - SH degree 3 fixture
  - near-plane culling fixture
  - precomputed 3D covariance fixture
  - SH clamp fixture

Known remaining work:

- Add broader fixtures for varied camera matrices, non-identity transforms,
  different image/tile sizes, and larger Gaussian counts.
- Split common Metal helpers into a shared source module once a second kernel
  needs them.
- Revisit the params representation if `template` args produce better compiled
  kernels for stable options.

## Binning and Tile Prep Status

The first Swift binning port is in:

- `Sources/FastGSSwift/FastGSBinning.swift`

Current shape:

- `FastGSBinning.forward` accepts preprocess-style `xy`, `depths`,
  `conicOpacity`, and `tilesTouched`.
- `tilesTouched` is prefix-summed with MLX `cumsum`.
- `fastgs_duplicate_with_keys_kernel` is ported through `MLXFast.metalKernel`.
- `argSort` and `take` are handled by MLX built-in ops, matching the C++
  binding pipeline.
- Tile range identification and per-tile bucket counts are ported as small
  `MLXFast.metalKernel` stages.
- `bucketOffsets` is prefix-summed with MLX `cumsum`.

Current coverage:

- duplicated unsorted keys and point list
- sorted keys
- tile ranges
- bucket count and bucket offsets
- parity for the first precomputed-color preprocess fixture
- partial culling fixture where only one gaussian emits duplicated keys
- all-culled fixture where `numRendered == 0` returns empty point lists and
  zeroed tile ranges/buckets
- varied depth fixture proving sorted point list order follows the packed
  `Float` depth bits

Porting notes:

- The duplicated key packs `tile_id << 32` with the `Float` depth bit pattern,
  matching the original Metal `as_type<uint>(depth)`. Swift fixtures must use
  `Float(1).bitPattern`, not `1.0.bitPattern`, which is a `Double`.
- The first fixture traverses tiles in column-major order before sorting because
  the original kernel flips into the `is_y` path when the y span is smaller
  than the x span.
- `conicOpacity` appeared as a `device float*` input under MLXFast, so the
  packed float helpers need both `constant` and `device` overloads here too.

Known remaining work:

- Add a larger fixture with non-square tile coverage before rasterize parity.

## Rasterize Forward Status

The first Swift rasterize port is in:

- `Sources/FastGSSwift/FastGSRasterize.swift`

Current shape:

- `FastGSRasterize.forward` accepts tile ranges, sorted point list, bucket
  offsets, means2D, colors, conic opacity, background, radii, and metric buffers.
- The first pass supports 16x16 tiles, 3 color channels, and metric counting
  disabled by default.
- Outputs mirror the C++ primitive order:
  - `bucketToTile`
  - `sampledT`
  - `sampledAr`
  - `finalT`
  - `nContrib`
  - `maxContrib`
  - `pixelColors`
  - `outColor`
  - `metricCount`

Current coverage:

- 1 tile / 1 Gaussian smoke fixture
- first preprocess -> binning -> rasterize fixture using the same
  precomputed-color scene as the Python/C++ reference path
- larger 80x48 fixture with 5 Gaussians, 5x3 tiles, and non-square tile spans
- Swift-side RGBA8/PNG export from `FastGSRasterizeOutput.outColor`
- output shapes and dtypes through the Swift API
- transmittance/color accumulation against hand-computed expected values
- sampled intermediates for the first bucket
- `nContrib`, `maxContrib`, `bucketToTile`, and `metricCount`
- E2E parity checks for:
  - per-channel `outColor` sums
  - per-channel `pixelColors` sums
  - sampled `outColor`, `pixelColors`, `finalT`, and `nContrib` at selected pixels
  - `bucketToTile`, `sampledT`, `sampledAr`, and per-tile `maxContrib`
  - preprocess radii/xy/depth/tile-count values and binning ranges/buckets for
    the larger fixture

Reference generation note:

- The E2E constants were generated by running the Python/C++ extension through
  `preprocess_forward -> cumsum -> binning_forward -> argsort/take ->
  tile_prep_forward -> cumsum -> rasterize_forward`, then recording summary
  values and a few stable sample pixels. This keeps the Swift fixture small
  while still exercising the full first forward path.
- The larger fixture also writes a Python/C++ reference preview image to
  `/private/tmp/fastgs_large_rasterize_ref.png` for visual inspection.
- The Swift export path writes the same large fixture's Swift MLXFast output to
  `/private/tmp/fastgs_swift_large_rasterize.png`. It converts channel-major
  float RGB `[3, H * W]` to interleaved RGBA8 before encoding PNG with ImageIO.
- `FastGSImageExport.texture(...)` converts the same RGBA8 bytes into an
  `.rgba8Unorm` `MTLTexture` for app presentation.

Known remaining work:

- Add a full-image exact rasterize parity fixture when the output surface gets
  larger and more varied.
- Enable and test metric count path.
- Add recorded scanner dataset forward parity before attempting live camera
  capture.
- Later, extend the CVPixelBuffer bridge from copy-based texture creation to a
  lifetime-safe `MLXArray` path if real-time camera-driven forward rendering
  becomes necessary.

## Recorded Scanner Data Plan

Live camera capture is intentionally deferred. The immediate migration target is
forward rendering from recorded scanner data, matching the existing Python/C++
workflow where data is captured first and then rendered or trained.

The currently available recorded dataset is:

- `/Users/yangdunfu/Downloads/2026_05_04_16_51_29`

Useful reference commands from the root `Makefile`:

- `make test-scanner`, with the data path adjusted to
  `/Users/yangdunfu/Downloads/2026_05_04_16_51_29`
- `make train-scanner-fastgs2-smoke`
- `make train-scanner-fastgs2`

A first reduced recorded-data reference is generated by
`swift/FastGSSwiftTools/generate_recorded_reference.py`. It uses the available
scanner dataset, renders a 160x120 frame with at most 4096 points through the
existing Python/C++ extension, and writes:

- `/private/tmp/fastgs_recorded_reference/recorded_manifest.json`
- `/private/tmp/fastgs_recorded_reference/recorded_means3d.f32`
- `/private/tmp/fastgs_recorded_reference/recorded_colors.f32`
- `/private/tmp/fastgs_recorded_reference/recorded_pred.png`
- `/private/tmp/fastgs_recorded_reference/recorded_target.png`
- `/private/tmp/fastgs_recorded_reference/recorded_sbs.png`

The manifest keeps small metadata and reference summaries in JSON, while large
point arrays are binary little-endian float32 buffers. This keeps the reference
format practical for larger recorded point counts. The current buffer entries
are:

```json
"means3dBuffer": {"path": "recorded_means3d.f32", "dtype": "float32", "shape": [4096, 3]},
"colorsBuffer": {"path": "recorded_colors.f32", "dtype": "float32", "shape": [4096, 3]}
```

The Xcode test `testRecordedScannerForwardRunsUnderXcode` loads that manifest,
runs the same inputs through Swift `preprocess -> binning -> rasterize`, compares
channel sums and sampled pixels against the Python/C++ reference, and writes:

- `/private/tmp/fastgs_recorded_reference/recorded_swift.png`

The manifest reader and forward conversion now live in the Swift package:

- `Sources/FastGSSwift/FastGSRecordedForward.swift`

`FastGSRecordedForwardScene` decodes `recorded_manifest.json`, converts
`means3d` and `colors` into `MLXArray` inputs, and runs the same forward path the
tests use. The macOS preview app uses this API for its recorded scanner frame
button, so recorded-data rendering is no longer test-only.

`generate_recorded_reference.py` is parameterized for larger references. For
example:

```bash
conda run -n fastgs_core python swift/FastGSSwiftTools/generate_recorded_reference.py \
  --max-points 16384 \
  --out-dir /private/tmp/fastgs_recorded_reference_16384
```

The Xcode suite includes `testRecordedScannerLargeForwardRunsUnderXcode` for the
16384-point reference. It currently verifies that the larger Swift path runs and
exports an image, but uses a wider channel-sum threshold because this case
exposes a larger Python/C++ vs Swift rasterize difference:

- Python/C++ sums: `[2842.2002, 2406.896, 1899.2173]`
- Swift sums observed: `[2858.0645, 2423.1213, 2056.7493]`

This should be tightened after isolating whether the divergence starts in
preprocess coverage, binning order, rasterize traversal, or accumulated
transmittance.

Stage summaries are now emitted by the Python reference generator under
`stageSummaries` in `recorded_manifest.json`, and the Xcode recorded tests write
the Swift side to:

- `/private/tmp/fastgs_recorded_reference/recorded_swift_stage_summary.json`
- `/private/tmp/fastgs_recorded_reference_16384/recorded_swift_stage_summary.json`

The 16384-point summary previously localized a divergence to preprocess color,
not geometry or binning:

- preprocess geometry matches: `visibleCount`, `radiiSum`, `tilesTouchedSum`
- binning matches: `numRendered`, point-list checksums, bucket sums
- rasterize traversal matches: `nContribSum`, `maxContribSum`, and nearly
  identical `finalTSum`
- preprocess color differs:
  - Python/C++ `rgbSums`: `[2159.5215, 1825.9961, 1445.8823]`
  - Swift `rgbSums`: `[2169.1870, 1835.5978, 1626.2614]`

The root cause was the recorded-data SH placeholder shape. The MLXFast Metal
kernel reads SH rest coefficients with a full `max_sh_coeffs` row stride, so the
recorded loader must provide `[pointCount, maxSHCoefficients, 3]` even when the
buffer is all zero. Providing only `maxSHCoefficients - 1` coefficients per
point made large scenes read across rows and changed preprocess RGB. After
padding the placeholder to the full stride, the 16384-point Swift stage summary
matches the Python/C++ reference within normal float accumulation noise:

- Python/C++ `rgbSums`: `[2159.5215, 1825.9961, 1445.8823]`
- Swift `rgbSums`: `[2159.5210, 1825.9977, 1445.8890]`
- Python/C++ output channel sums: `[2842.2002, 2406.8960, 1899.2173]`
- Swift output channel sums: `[2842.1997, 2406.8967, 1899.2119]`

The large recorded Xcode parity test now uses `channelSumAccuracy: 2e-2`.

The current mock `CVPixelBuffer` path remains useful for presentation testing,
but it is not the next critical path.

## Stop Gradient and Backward Plan

The current Swift forward path is preview/parity oriented and does not yet call
`stopGradient`. The existing C++ binding does use `mx::stop_gradient` around
non-differentiable scheduling work, so the Swift training path should add the
same graph boundaries before backward is exposed.

Keep two mental models separate:

- `MLXFast.metalKernel` is the GPU implementation mechanism.
- `CustomFunction` is the autograd contract that decides what tensors are saved
  from forward and how VJP calls the backward kernels.

The first Swift training wrapper should stop gradients for orchestration arrays:

- `pointOffsets = cumsum(tilesTouched)`
- binning-only views of `xys`, `depths`, `conicOpacity`, and `tilesTouched`
- duplicated point-list keys and indices
- sorted point-list keys and indices
- tile ranges, bucket counts, and bucket offsets
- raster scheduling inputs such as `radii` and `metricMap`

Do not stop the values that rasterize backward needs in order to propagate
gradients:

- `means2D` / `xys`
- `colors` / `rgbs`
- `conicOpacity`
- `viewspacePoints`
- original trainable Gaussian inputs before preprocess, such as means, colors,
  opacity, scale, rotation, and precomputed covariance

The safer implementation order is:

1. Add explicit Swift APIs for the backward kernels using `MLXFast.metalKernel`.
   Start with rasterize backward, then preprocess backward.
2. Test those APIs with fixed upstream gradients and compare against the current
   Python/C++ implementation.
3. Add a Swift `CustomFunction` wrapper whose forward closure calls the existing
   MLXFast forward kernels and whose VJP closure calls the explicit backward
   APIs.

This keeps the Metal port debuggable: a wrong gradient can be isolated to the
backward kernel before the autograd wrapper is involved.

## Optimizer Notes

`submodules/mlx-swift/Source/MLXOptimizers/Optimizers.swift` is the closest
Swift-side reference for training updates. Its design mirrors Python MLX: the
optimizer owns state, receives gradients, and returns updated `MLXArray`
parameters. `OptimizerBase` stores per-parameter state in a tree matching the
model parameters, while concrete optimizers such as SGD and Adam implement the
single-parameter update formula.

FastGSSwift does not need to depend on that exact API if Gaussian splat
parameters are easier to manage as typed arrays instead of `MLXNN.Module`
parameters. A FastGS-specific optimizer can still copy the important parts:

- keep first/second moment state as `MLXArray`
- update each trainable field on device
- expose `innerState()` or equivalent evaluation hooks for optimizer buffers
- support per-field learning rates for means, colors/SH, opacity, scale, and
  rotation
- plan for state migration when Gaussian densify/prune changes the point count

The first optimizer smoke test should not wait for full training. Use synthetic
gradients to prove one Adam-style update changes typed Gaussian parameter arrays
and keeps optimizer state shape-compatible.

Swift migration reference generators are kept in `swift/FastGSSwiftTools/`:

- `fastgs_preprocess_edge_refs.py`
- `fastgs_e2e_rasterize_ref.py`
- `fastgs_large_rasterize_ref.py`
- `generate_recorded_reference.py`
- `compare_recorded_stage_summary.py`

The recorded Swift forward parity flow can now be run from the repository root:

```bash
make test-swift-recorded-forward
```

This regenerates the 4096-point and 16384-point Python/C++ recorded references,
runs the macOS Xcode tests that write `recorded_swift.png` and
`recorded_swift_stage_summary.json`, then compares each Swift stage summary
against the manifest `stageSummaries`. Integer prefixes and checksums must match
exactly; float summaries use a `5e-2` absolute tolerance plus relative tolerance
for large accumulated sums. The output also prints the Python reference PNG,
side-by-side PNG, and Swift PNG paths for quick visual inspection. The Xcode
step uses an isolated DerivedData folder at `/private/tmp/fastgs_swift_xcode_derived`
so stale default Xcode build products do not affect the automation.

## CVPixelBuffer Bridge Status

`FastGSCameraFrameBridge.lockBGRAFrame(_:)` is the first camera-frame bridge
prototype:

- accepts `kCVPixelFormatType_32BGRA`
- locks the `CVPixelBuffer` for read-only access
- preserves `width`, `height`, and `bytesPerRow`
- converts BGRA camera bytes to the RGBA byte order already used by
  `FastGSImageExport.texture(rgbaBytes:width:height:device:)`
- reports whether the pixel buffer exposes an `IOSurface`

`FastGSCameraFrameBridge.texture(fromBGRA:device:usage:)` is the first
presentation bridge for camera frames. It uses the copy path:

```text
CVPixelBuffer BGRA
  -> locked, stride-aware RGBA bytes
  -> FastGSImageExport.texture(rgbaBytes:width:height:device:)
  -> MTLTexture
```

The first Xcode test builds a mock IOSurface-compatible BGRA pixel buffer,
writes deterministic pixels, and verifies the channel reorder plus stride-aware
readback. A second Xcode test creates an `MTLTexture` from that same buffer and
reads the texture bytes back to verify the presentation path. The current bridge
intentionally copies into `[UInt8]`; the next step is choosing a safe lifetime
model before exposing the locked base address to
`MLXArray(rawPointer:shape:dtype:finalizer:)`.

The macOS preview app now also has a mock camera-frame path. The toolbar camera
button creates an IOSurface-compatible BGRA `CVPixelBuffer`, converts it through
`FastGSCameraFrameBridge.texture(fromBGRA:device:)`, and displays the resulting
texture in the same `MTKView` used by the FastGS render preview. This keeps the
camera presentation path testable without committing to live capture yet.

## macOS Preview Status

The Xcode macOS app now renders the larger static fixture through the SwiftPM
package and displays the result through an `MTKView`-backed Metal texture
preview:

- app entry: `swift/FastGSSwiftApps/Apps/FastGSSwiftMac/FastGSSwiftMacApp.swift`
- render source: `FastGSPreprocessParityFixture.rasterizeLargeE2EOutput()`
- primary presentation bridge:
  `FastGSImageExport.texture(rasterizeOutput:width:height:device:)`
- debug fallback:
  `FastGSImageExport.cgImage(rasterizeOutput:width:height:)`

This is intentionally still a static preview. It proves the Swift package,
MLXFast kernels, channel-major `outColor`, and Metal presentation path can be
connected inside an app target before adding camera input, IOSurface wrapping,
or interactive camera controls.

The recorded scanner preview uses the same app surface as an inspection view:
left pane loads the manifest `targetPng`, and right pane renders the same
manifest through Swift `FastGSRecordedForwardScene`. Numeric parity remains the
job of `make test-swift-recorded-forward`; the app view is for quickly spotting
camera, orientation, and color differences.

## Suggested Porting Checklist

For each existing `.metal` stage:

1. Move constants and inline helpers into `header`.
2. Move only the kernel body into `source`.
3. Replace explicit Metal buffers with `inputNames` and `outputNames`.
4. Replace `Params` structs with params `MLXArray` or `template` args.
5. Add address-space overloads for helpers that read input buffers.
6. Set `initValue` when the old primitive zeroed outputs before dispatch.
7. Add a SwiftPM-gated test and an Xcode test that actually runs the kernel.
8. After it runs, add parity tests against the existing implementation.
