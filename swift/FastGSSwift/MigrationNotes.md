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
- `viewmatrix`, `projmatrix`, and some later inputs appeared as `device float*`.
- `shs` appeared as `device float*` when the SH color path was exercised.
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
arrays.

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
