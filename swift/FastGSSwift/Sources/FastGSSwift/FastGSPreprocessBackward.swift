import MLX

public struct FastGSPreprocessCotangents {
    public var radii: MLXArray
    public var xy: MLXArray
    public var depths: MLXArray
    public var cov3D: MLXArray
    public var rgb: MLXArray
    public var conicOpacity: MLXArray
    public var tilesTouched: MLXArray
    public var clamped: MLXArray
    public var viewspacePoints: MLXArray

    public init(
        radii: MLXArray,
        xy: MLXArray,
        depths: MLXArray,
        cov3D: MLXArray,
        rgb: MLXArray,
        conicOpacity: MLXArray,
        tilesTouched: MLXArray,
        clamped: MLXArray,
        viewspacePoints: MLXArray
    ) {
        self.radii = radii
        self.xy = xy
        self.depths = depths
        self.cov3D = cov3D
        self.rgb = rgb
        self.conicOpacity = conicOpacity
        self.tilesTouched = tilesTouched
        self.clamped = clamped
        self.viewspacePoints = viewspacePoints
    }

    public static func fromRasterizeBackward(
        _ rasterizeBackward: FastGSRasterizeBackwardOutput,
        like forwardOutput: FastGSPreprocessOutput,
        stream: StreamOrDevice = .default
    ) -> FastGSPreprocessCotangents {
        FastGSPreprocessCotangents(
            radii: MLXArray.zeros(forwardOutput.radii.shape, dtype: forwardOutput.radii.dtype, stream: stream),
            xy: rasterizeBackward.means2D,
            depths: MLXArray.zeros(forwardOutput.depths.shape, dtype: forwardOutput.depths.dtype, stream: stream),
            cov3D: MLXArray.zeros(forwardOutput.cov3D.shape, dtype: forwardOutput.cov3D.dtype, stream: stream),
            rgb: rasterizeBackward.colors,
            conicOpacity: rasterizeBackward.conicOpacity,
            tilesTouched: MLXArray.zeros(forwardOutput.tilesTouched.shape, dtype: forwardOutput.tilesTouched.dtype, stream: stream),
            clamped: MLXArray.zeros(forwardOutput.clamped.shape, dtype: forwardOutput.clamped.dtype, stream: stream),
            viewspacePoints: rasterizeBackward.viewspacePoints
        )
    }
}

public struct FastGSPreprocessBackwardOutput {
    public var means3D: MLXArray
    public var dc: MLXArray
    public var sh: MLXArray
    public var colorsPrecomputed: MLXArray
    public var opacities: MLXArray
    public var scales: MLXArray
    public var rotations: MLXArray
    public var cov3DPrecomputed: MLXArray
    public var viewMatrix: MLXArray
    public var projectionMatrix: MLXArray
    public var cameraPosition: MLXArray
    public var viewspacePoints: MLXArray
}

public enum FastGSPreprocessBackward {
    private static let kernel = MLXFast.metalKernel(
        name: "fastgs_preprocess_backward_swift_skeleton_v3",
        inputNames: [
            "params",
            "means3d",
            "dL_dxys",
            "dL_ddepths",
            "dL_dcov3d",
            "dL_drgb",
            "dL_dconic_opacity",
            "dL_dviewspace_out",
        ],
        outputNames: [
            "dL_dmeans3d",
            "dL_ddc",
            "dL_dopacities",
            "dL_dviewspace_in",
        ],
        source: FastGSPreprocessBackwardKernelSource.body,
        header: FastGSPreprocessBackwardKernelSource.header
    )

    public static func forward(
        input: FastGSPreprocessInput,
        cotangents: FastGSPreprocessCotangents,
        forwardOutput: FastGSPreprocessOutput,
        params: FastGSPreprocessParams,
        verbose: Bool = false,
        stream: StreamOrDevice = .default
    ) -> FastGSPreprocessBackwardOutput {
        validate(input: input, cotangents: cotangents, forwardOutput: forwardOutput, params: params)

        let count = input.means3D.shape[0]
        let emptyOutput = FastGSPreprocessBackwardOutput(
            means3D: MLXArray.zeros(input.means3D.shape, dtype: input.means3D.dtype, stream: stream),
            dc: MLXArray.zeros(input.dc.shape, dtype: input.dc.dtype, stream: stream),
            sh: MLXArray.zeros(input.sh.shape, dtype: input.sh.dtype, stream: stream),
            colorsPrecomputed: MLXArray.zeros(input.colorsPrecomputed.shape, dtype: input.colorsPrecomputed.dtype, stream: stream),
            opacities: MLXArray.zeros(input.opacities.shape, dtype: input.opacities.dtype, stream: stream),
            scales: MLXArray.zeros(input.scales.shape, dtype: input.scales.dtype, stream: stream),
            rotations: MLXArray.zeros(input.rotations.shape, dtype: input.rotations.dtype, stream: stream),
            cov3DPrecomputed: MLXArray.zeros(input.cov3DPrecomputed.shape, dtype: input.cov3DPrecomputed.dtype, stream: stream),
            viewMatrix: MLXArray.zeros(input.viewMatrix.shape, dtype: input.viewMatrix.dtype, stream: stream),
            projectionMatrix: MLXArray.zeros(input.projectionMatrix.shape, dtype: input.projectionMatrix.dtype, stream: stream),
            cameraPosition: MLXArray.zeros(input.cameraPosition.shape, dtype: input.cameraPosition.dtype, stream: stream),
            viewspacePoints: MLXArray.zeros(input.viewspacePoints.shape, dtype: input.viewspacePoints.dtype, stream: stream)
        )

        if count == 0 {
            return emptyOutput
        }

        let outputs = kernel(
            [
                params.backwardKernelArray(count: count),
                input.means3D,
                cotangents.xy,
                cotangents.depths,
                cotangents.cov3D,
                cotangents.rgb,
                cotangents.conicOpacity,
                cotangents.viewspacePoints,
            ],
            grid: (count, 1, 1),
            threadGroup: (max(1, min(256, count)), 1, 1),
            outputShapes: [
                input.means3D.shape,
                input.dc.shape,
                input.opacities.shape,
                input.viewspacePoints.shape,
            ],
            outputDTypes: [
                input.means3D.dtype,
                input.dc.dtype,
                input.opacities.dtype,
                input.viewspacePoints.dtype,
            ],
            initValue: 0,
            verbose: verbose,
            stream: stream
        )

        return FastGSPreprocessBackwardOutput(
            means3D: outputs[0],
            dc: outputs[1],
            sh: emptyOutput.sh,
            colorsPrecomputed: emptyOutput.colorsPrecomputed,
            opacities: outputs[2],
            scales: emptyOutput.scales,
            rotations: emptyOutput.rotations,
            cov3DPrecomputed: emptyOutput.cov3DPrecomputed,
            viewMatrix: emptyOutput.viewMatrix,
            projectionMatrix: emptyOutput.projectionMatrix,
            cameraPosition: emptyOutput.cameraPosition,
            viewspacePoints: outputs[3]
        )
    }

    public static func forward(
        input: FastGSPreprocessInput,
        preprocessOutput: FastGSPreprocessOutput,
        rasterizeBackwardOutput: FastGSRasterizeBackwardOutput,
        params: FastGSPreprocessParams,
        verbose: Bool = false,
        stream: StreamOrDevice = .default
    ) -> FastGSPreprocessBackwardOutput {
        rasterizeBackwardOutput.arrays.forEach { $0.eval() }
        return forward(
            input: input,
            cotangents: .fromRasterizeBackward(rasterizeBackwardOutput, like: preprocessOutput, stream: stream),
            forwardOutput: preprocessOutput,
            params: params,
            verbose: verbose,
            stream: stream
        )
    }

    private static func validate(
        input: FastGSPreprocessInput,
        cotangents: FastGSPreprocessCotangents,
        forwardOutput: FastGSPreprocessOutput,
        params: FastGSPreprocessParams
    ) {
        let count = input.means3D.shape[0]
        precondition(input.means3D.shape == [count, 3], "means3D must have shape [N, 3].")
        precondition(input.opacities.shape == [count], "opacities must have shape [N].")
        precondition(input.viewMatrix.shape == [4, 4], "viewMatrix must have shape [4, 4].")
        precondition(input.projectionMatrix.shape == [4, 4], "projectionMatrix must have shape [4, 4].")
        precondition(input.cameraPosition.shape == [3], "cameraPosition must have shape [3].")
        precondition(input.viewspacePoints.shape == [count, 4], "viewspacePoints must have shape [N, 4].")
        precondition(input.means3D.dtype == .float32, "FastGSPreprocessBackward currently expects float32 inputs.")
        precondition(input.viewspacePoints.dtype == .float32, "FastGSPreprocessBackward currently expects float32 viewspace points.")
        if params.useColorsPrecomputed {
            precondition(input.colorsPrecomputed.shape == [count, 3], "colorsPrecomputed must have shape [N, 3].")
        } else {
            precondition(input.dc.shape == [count, 3], "dc must have shape [N, 3].")
        }
        if params.useCov3DPrecomputed {
            precondition(input.cov3DPrecomputed.shape == [count, 6], "cov3DPrecomputed must have shape [N, 6].")
        } else {
            precondition(input.scales.shape == [count, 3], "scales must have shape [N, 3].")
            precondition(input.rotations.shape == [count, 4], "rotations must have shape [N, 4].")
        }
        precondition(cotangents.xy.shape == forwardOutput.xy.shape, "xy cotangent shape mismatch.")
        precondition(cotangents.depths.shape == forwardOutput.depths.shape, "depths cotangent shape mismatch.")
        precondition(cotangents.cov3D.shape == forwardOutput.cov3D.shape, "cov3D cotangent shape mismatch.")
        precondition(cotangents.rgb.shape == forwardOutput.rgb.shape, "rgb cotangent shape mismatch.")
        precondition(cotangents.conicOpacity.shape == forwardOutput.conicOpacity.shape, "conicOpacity cotangent shape mismatch.")
        precondition(cotangents.viewspacePoints.shape == forwardOutput.viewspacePoints.shape, "viewspace cotangent shape mismatch.")
        precondition(forwardOutput.radii.dtype == .int32, "radii forward output must be int32.")
        precondition(forwardOutput.clamped.dtype == .bool, "clamped forward output must be bool.")
    }
}

private extension FastGSPreprocessParams {
    func backwardKernelArray(count: Int) -> MLXArray {
        MLXArray([
            Float(count),
            Float(imageWidth),
            Float(imageHeight),
            useCov3DPrecomputed ? 1.0 : 0.0,
            useColorsPrecomputed ? 1.0 : 0.0,
            scaleModifier,
            tanFovX,
            tanFovY,
            0.5 * Float(imageWidth) / max(tanFovX, 1e-6),
            0.5 * Float(imageHeight) / max(tanFovY, 1e-6),
            Float(degree),
            Float(maxSHCoefficients),
        ], [12])
    }
}

private enum FastGSPreprocessBackwardKernelSource {
    static let header = """
        constant float SH_C0 = 0.28209479177387814f;

        inline float3 read_packed_float3(const device float* arr, uint idx) {
          return float3(arr[3 * idx], arr[3 * idx + 1], arr[3 * idx + 2]);
        }

        inline float4 read_packed_float4(const device float* arr, uint idx) {
          return float4(arr[4 * idx], arr[4 * idx + 1], arr[4 * idx + 2], arr[4 * idx + 3]);
        }

        inline void write_packed_float3(device float* arr, uint idx, float3 val) {
          arr[3 * idx] = val.x;
          arr[3 * idx + 1] = val.y;
          arr[3 * idx + 2] = val.z;
        }

        inline void write_packed_float4(device float* arr, uint idx, float4 val) {
          arr[4 * idx] = val.x;
          arr[4 * idx + 1] = val.y;
          arr[4 * idx + 2] = val.z;
          arr[4 * idx + 3] = val.w;
        }
        """

    static let body = """
        const uint n = uint(params[0]);
        const bool use_colors_precomp = params[4] != 0.0f;
        const uint tid = thread_position_in_grid.x;
        if (tid >= n) {
          return;
        }

        const float2 dxy = float2(dL_dxys[2 * tid], dL_dxys[2 * tid + 1]);
        const float depth_grad = dL_ddepths[tid];
        const float4 dview = read_packed_float4(dL_dviewspace_out, tid);
        write_packed_float3(
            dL_dmeans3d,
            tid,
            float3(dxy.x + dview.x, dxy.y + dview.y, depth_grad + dview.z));
        write_packed_float4(dL_dviewspace_in, tid, dview);

        dL_dopacities[tid] = dL_dconic_opacity[4 * tid + 3];
        if (use_colors_precomp) {
          write_packed_float3(dL_ddc, tid, read_packed_float3(dL_drgb, tid));
        } else {
          const float3 rgb_grad = read_packed_float3(dL_drgb, tid);
          write_packed_float3(dL_ddc, tid, SH_C0 * rgb_grad);
        }

        """
}
