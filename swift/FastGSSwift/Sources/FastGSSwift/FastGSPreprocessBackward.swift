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
        name: "fastgs_preprocess_backward_swift_full_v1",
        inputNames: [
            "params",
            "means3d",
            "scales",
            "quats",
            "viewmat",
            "projmat",
            "cam_pos",
            "sh",
            "cov3d_precomp",
            "cov3d_fwd",
            "dL_dcov3d",
            "dL_drgb",
            "dL_dxys",
            "dL_ddepths",
            "dL_dconic_opacity",
            "dL_dviewspace_out",
            "radii",
            "clamped",
        ],
        outputNames: [
            "dL_dmeans3d",
            "dL_ddc",
            "dL_dsh",
            "dL_dcolors_precomp",
            "dL_dopacities",
            "dL_dscales",
            "dL_dquats",
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

        let colorsPrecomputedInput = nonEmpty(input.colorsPrecomputed, fallbackShape: [count, 3], stream: stream)
        let cov3DPrecomputedInput = nonEmpty(input.cov3DPrecomputed, fallbackShape: [count, 6], stream: stream)
        let colorsPrecomputedOutputShape = params.useColorsPrecomputed ? input.colorsPrecomputed.shape : [count, 3]

        let outputs = kernel(
            [
                params.backwardKernelArray(count: count),
                input.means3D,
                input.scales,
                input.rotations,
                input.viewMatrix,
                input.projectionMatrix,
                input.cameraPosition,
                input.sh,
                cov3DPrecomputedInput,
                forwardOutput.cov3D,
                cotangents.cov3D,
                cotangents.rgb,
                cotangents.xy,
                cotangents.depths,
                cotangents.conicOpacity,
                cotangents.viewspacePoints,
                forwardOutput.radii,
                forwardOutput.clamped,
            ],
            grid: (count, 1, 1),
            threadGroup: (max(1, min(256, count)), 1, 1),
            outputShapes: [
                input.means3D.shape,
                input.dc.shape,
                input.sh.shape,
                colorsPrecomputedOutputShape,
                input.opacities.shape,
                input.scales.shape,
                input.rotations.shape,
                input.viewspacePoints.shape,
            ],
            outputDTypes: [
                input.means3D.dtype,
                input.dc.dtype,
                input.sh.dtype,
                colorsPrecomputedInput.dtype,
                input.opacities.dtype,
                input.scales.dtype,
                input.rotations.dtype,
                input.viewspacePoints.dtype,
            ],
            initValue: 0,
            verbose: verbose,
            stream: stream
        )

        return FastGSPreprocessBackwardOutput(
            means3D: outputs[0],
            dc: outputs[1],
            sh: outputs[2],
            colorsPrecomputed: params.useColorsPrecomputed ? outputs[3] : emptyOutput.colorsPrecomputed,
            opacities: outputs[4],
            scales: outputs[5],
            rotations: outputs[6],
            cov3DPrecomputed: emptyOutput.cov3DPrecomputed,
            viewMatrix: emptyOutput.viewMatrix,
            projectionMatrix: emptyOutput.projectionMatrix,
            cameraPosition: emptyOutput.cameraPosition,
            viewspacePoints: outputs[7]
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

    private static func nonEmpty(_ array: MLXArray, fallbackShape: [Int], stream: StreamOrDevice) -> MLXArray {
        array.size == 0 ? MLXArray.zeros(fallbackShape, dtype: array.dtype, stream: stream) : array
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
        inline float3 read_packed_float3(const device float* arr, uint idx) {
          return float3(arr[3 * idx], arr[3 * idx + 1], arr[3 * idx + 2]);
        }
        
        inline float4 mul_mat4_vec3_h(const device float* m, float3 p) {
          return float4(
              m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12],
              m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13],
              m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14],
              m[3] * p.x + m[7] * p.y + m[11] * p.z + m[15]);
        }
        
        inline float3 read_packed_float3(const device float* arr, uint idx, uint stride) {
          return float3(arr[stride * idx], arr[stride * idx + 1], arr[stride * idx + 2]);
        }
        
        inline float4 read_packed_float4(const device float* arr, uint idx, uint stride) {
          return float4(arr[stride * idx], arr[stride * idx + 1], arr[stride * idx + 2], arr[stride * idx + 3]);
        }
        
        inline float3 transform_point_4x3(const float3 p, const device float* matrix) {
          return float3(
              matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
              matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
              matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14]);
        }
        
        inline float3 transform_vec4x3_transpose(const float3 v, const device float* matrix) {
          return float3(
              matrix[0] * v.x + matrix[1] * v.y + matrix[2] * v.z,
              matrix[4] * v.x + matrix[5] * v.y + matrix[6] * v.z,
              matrix[8] * v.x + matrix[9] * v.y + matrix[10] * v.z);
        }
        
        inline float3 dnormvdv(float3 v, float3 g) {
          float len = max(length(v), 1e-6f);
          float3 n = v / len;
          return (g - n * dot(n, g)) / len;
        }
        
        
        """

    static let body = """
        uint tid = thread_position_in_grid.x;
        if (tid >= uint(params[0])) {
            return;
          }
          if (!(radii[tid] > 0)) {
            return;
          }
        
          const float3 p = read_packed_float3(means3d, tid);
          const float4 ph = mul_mat4_vec3_h(projmat, p);
          const uint cov_base = 6u * tid;
        
          const float gx = dL_dxys[2 * tid + 0];
          const float gy = dL_dxys[2 * tid + 1];
        
          // CUDA parity path: conic gradients feed cov3d + mean3d through cov2D chain.
          const bool use_cov3d_precomp = uint(params[3]) != 0u;
          const float cov3D0 = use_cov3d_precomp ? cov3d_precomp[cov_base + 0] : cov3d_fwd[cov_base + 0];
          const float cov3D1 = use_cov3d_precomp ? cov3d_precomp[cov_base + 1] : cov3d_fwd[cov_base + 1];
          const float cov3D2 = use_cov3d_precomp ? cov3d_precomp[cov_base + 2] : cov3d_fwd[cov_base + 2];
          const float cov3D3 = use_cov3d_precomp ? cov3d_precomp[cov_base + 3] : cov3d_fwd[cov_base + 3];
          const float cov3D4 = use_cov3d_precomp ? cov3d_precomp[cov_base + 4] : cov3d_fwd[cov_base + 4];
          const float cov3D5 = use_cov3d_precomp ? cov3d_precomp[cov_base + 5] : cov3d_fwd[cov_base + 5];
          const float3 cov_grad = float3(
              dL_dconic_opacity[4 * tid + 0],
              dL_dconic_opacity[4 * tid + 1],
              dL_dconic_opacity[4 * tid + 2]);
          float dcov_extra[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
          float3 dmean_cov = float3(0.0f);
          {
            const float3 mean = p;
            float3 t = transform_point_4x3(mean, viewmat);
            const float limx = 1.3f * params[6];
            const float limy = 1.3f * params[7];
            const float txtz = t.x / t.z;
            const float tytz = t.y / t.z;
            t.x = min(limx, max(-limx, txtz)) * t.z;
            t.y = min(limy, max(-limy, tytz)) * t.z;
            const float x_grad_mul = (txtz < -limx || txtz > limx) ? 0.0f : 1.0f;
            const float y_grad_mul = (tytz < -limy || tytz > limy) ? 0.0f : 1.0f;
        
            const float3x3 J = float3x3(
                params[8] / t.z, 0.0f, -(params[8] * t.x) / (t.z * t.z),
                0.0f, params[9] / t.z, -(params[9] * t.y) / (t.z * t.z),
                0.0f, 0.0f, 0.0f);
            const float3x3 W = float3x3(
                viewmat[0], viewmat[4], viewmat[8],
                viewmat[1], viewmat[5], viewmat[9],
                viewmat[2], viewmat[6], viewmat[10]);
            const float3x3 Vrk = float3x3(
                cov3D0, cov3D1, cov3D2,
                cov3D1, cov3D3, cov3D4,
                cov3D2, cov3D4, cov3D5);
            const float3x3 Tm = W * J;
            float3x3 cov2D = transpose(Tm) * transpose(Vrk) * Tm;
            float a = cov2D[0][0] + 0.3f;
            float b = cov2D[0][1];
            float c = cov2D[1][1] + 0.3f;
            float denom = a * c - b * b;
            float denom2inv = 1.0f / ((denom * denom) + 1e-7f);
            float dL_da = 0.0f;
            float dL_db = 0.0f;
            float dL_dc = 0.0f;
            if (denom2inv != 0.0f) {
              dL_da = denom2inv * (-c * c * cov_grad.x + 2.0f * b * c * cov_grad.y + (denom - a * c) * cov_grad.z);
              dL_dc = denom2inv * (-a * a * cov_grad.z + 2.0f * a * b * cov_grad.y + (denom - a * c) * cov_grad.x);
              dL_db = denom2inv * 2.0f * (b * c * cov_grad.x - (denom + 2.0f * b * b) * cov_grad.y + a * b * cov_grad.z);
        
              dcov_extra[0] = (Tm[0][0] * Tm[0][0] * dL_da + Tm[0][0] * Tm[1][0] * dL_db + Tm[1][0] * Tm[1][0] * dL_dc);
              dcov_extra[3] = (Tm[0][1] * Tm[0][1] * dL_da + Tm[0][1] * Tm[1][1] * dL_db + Tm[1][1] * Tm[1][1] * dL_dc);
              dcov_extra[5] = (Tm[0][2] * Tm[0][2] * dL_da + Tm[0][2] * Tm[1][2] * dL_db + Tm[1][2] * Tm[1][2] * dL_dc);
              dcov_extra[1] = 2.0f * Tm[0][0] * Tm[0][1] * dL_da + (Tm[0][0] * Tm[1][1] + Tm[0][1] * Tm[1][0]) * dL_db + 2.0f * Tm[1][0] * Tm[1][1] * dL_dc;
              dcov_extra[2] = 2.0f * Tm[0][0] * Tm[0][2] * dL_da + (Tm[0][0] * Tm[1][2] + Tm[0][2] * Tm[1][0]) * dL_db + 2.0f * Tm[1][0] * Tm[1][2] * dL_dc;
              dcov_extra[4] = 2.0f * Tm[0][2] * Tm[0][1] * dL_da + (Tm[0][1] * Tm[1][2] + Tm[0][2] * Tm[1][1]) * dL_db + 2.0f * Tm[1][1] * Tm[1][2] * dL_dc;
            }
        
            float dL_dT00 = 2.0f * (Tm[0][0] * Vrk[0][0] + Tm[0][1] * Vrk[0][1] + Tm[0][2] * Vrk[0][2]) * dL_da +
                            (Tm[1][0] * Vrk[0][0] + Tm[1][1] * Vrk[0][1] + Tm[1][2] * Vrk[0][2]) * dL_db;
            float dL_dT01 = 2.0f * (Tm[0][0] * Vrk[1][0] + Tm[0][1] * Vrk[1][1] + Tm[0][2] * Vrk[1][2]) * dL_da +
                            (Tm[1][0] * Vrk[1][0] + Tm[1][1] * Vrk[1][1] + Tm[1][2] * Vrk[1][2]) * dL_db;
            float dL_dT02 = 2.0f * (Tm[0][0] * Vrk[2][0] + Tm[0][1] * Vrk[2][1] + Tm[0][2] * Vrk[2][2]) * dL_da +
                            (Tm[1][0] * Vrk[2][0] + Tm[1][1] * Vrk[2][1] + Tm[1][2] * Vrk[2][2]) * dL_db;
            float dL_dT10 = 2.0f * (Tm[1][0] * Vrk[0][0] + Tm[1][1] * Vrk[0][1] + Tm[1][2] * Vrk[0][2]) * dL_dc +
                            (Tm[0][0] * Vrk[0][0] + Tm[0][1] * Vrk[0][1] + Tm[0][2] * Vrk[0][2]) * dL_db;
            float dL_dT11 = 2.0f * (Tm[1][0] * Vrk[1][0] + Tm[1][1] * Vrk[1][1] + Tm[1][2] * Vrk[1][2]) * dL_dc +
                            (Tm[0][0] * Vrk[1][0] + Tm[0][1] * Vrk[1][1] + Tm[0][2] * Vrk[1][2]) * dL_db;
            float dL_dT12 = 2.0f * (Tm[1][0] * Vrk[2][0] + Tm[1][1] * Vrk[2][1] + Tm[1][2] * Vrk[2][2]) * dL_dc +
                            (Tm[0][0] * Vrk[2][0] + Tm[0][1] * Vrk[2][1] + Tm[0][2] * Vrk[2][2]) * dL_db;
        
            float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
            float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
            float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
            float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;
        
            const float tz = 1.0f / t.z;
            const float tz2 = tz * tz;
            const float tz3 = tz2 * tz;
            const float dL_dtx = x_grad_mul * -params[8] * tz2 * dL_dJ02;
            const float dL_dty = y_grad_mul * -params[9] * tz2 * dL_dJ12;
            const float dL_dtz = -params[8] * tz2 * dL_dJ00 - params[9] * tz2 * dL_dJ11 +
                                 (2.0f * params[8] * t.x) * tz3 * dL_dJ02 +
                                 (2.0f * params[9] * t.y) * tz3 * dL_dJ12;
            dmean_cov = transform_vec4x3_transpose(float3(dL_dtx, dL_dty, dL_dtz), viewmat);
          }
        
          // CUDA preprocessCUDA parity: dL_dmean2D.xy -> dL_dmeans3D through projective Jacobian.
          const float m_w = 1.0f / (ph.w + 1.0e-7f);
          const float mul1 =
              (projmat[0] * p.x + projmat[4] * p.y + projmat[8] * p.z + projmat[12]) *
              m_w * m_w;
          const float mul2 =
              (projmat[1] * p.x + projmat[5] * p.y + projmat[9] * p.z + projmat[13]) *
              m_w * m_w;
          const float dmx = (projmat[0] * m_w - projmat[3] * mul1) * gx +
                            (projmat[1] * m_w - projmat[3] * mul2) * gy;
          const float dmy = (projmat[4] * m_w - projmat[7] * mul1) * gx +
                            (projmat[5] * m_w - projmat[7] * mul2) * gy;
          const float dmz = (projmat[8] * m_w - projmat[11] * mul1) * gx +
                            (projmat[9] * m_w - projmat[11] * mul2) * gy;
        
          dL_dmeans3d[3 * tid + 0] = dmx + dmean_cov.x;
          dL_dmeans3d[3 * tid + 1] = dmy + dmean_cov.y;
          dL_dmeans3d[3 * tid + 2] = dmz + dmean_cov.z;
          dL_dopacities[tid] = dL_dconic_opacity[4 * tid + 3];
        
          if (uint(params[4]) != 0u) {
            dL_dcolors_precomp[3 * tid + 0] = dL_drgb[3 * tid + 0];
            dL_dcolors_precomp[3 * tid + 1] = dL_drgb[3 * tid + 1];
            dL_dcolors_precomp[3 * tid + 2] = dL_drgb[3 * tid + 2];
          } else {
            float3 dL_dRGB = float3(
                clamped[3 * tid + 0] ? 0.0f : dL_drgb[3 * tid + 0],
                clamped[3 * tid + 1] ? 0.0f : dL_drgb[3 * tid + 1],
                clamped[3 * tid + 2] ? 0.0f : dL_drgb[3 * tid + 2]);
        
            // degree-0 dc term
            constexpr float SH_C0 = 0.28209479177387814f;
            dL_ddc[3 * tid + 0] = SH_C0 * dL_dRGB[0];
            dL_ddc[3 * tid + 1] = SH_C0 * dL_dRGB[1];
            dL_ddc[3 * tid + 2] = SH_C0 * dL_dRGB[2];
            dL_dcolors_precomp[3 * tid + 0] = 0.0f;
            dL_dcolors_precomp[3 * tid + 1] = 0.0f;
            dL_dcolors_precomp[3 * tid + 2] = 0.0f;
        
            if (int(params[10]) > 0 && int(params[11]) >= 3) {
              constexpr float SH_C1 = 0.4886025119029199f;
              constexpr float SH_C2[5] = {
                  1.0925484305920792f,
                  -1.0925484305920792f,
                  0.31539156525252005f,
                  -1.0925484305920792f,
                  0.5462742152960396f,
              };
              constexpr float SH_C3[7] = {
                  -0.5900435899266435f,
                  2.890611442640554f,
                  -0.4570457994644658f,
                  0.3731763325901154f,
                  -0.4570457994644658f,
                  1.445305721320277f,
                  -0.5900435899266435f,
              };
              float3 pos = read_packed_float3(means3d, tid, 3u);
              float3 campos = float3(cam_pos[0], cam_pos[1], cam_pos[2]);
              float3 dir_orig = pos - campos;
              float3 dir = normalize(dir_orig);
              float x = dir.x;
              float y = dir.y;
              float z = dir.z;
        
              uint base = (tid * uint(int(params[11]))) * 3u;
              float3 sh0 = float3(sh[base + 0], sh[base + 1], sh[base + 2]);
              float3 sh1 = float3(sh[base + 3], sh[base + 4], sh[base + 5]);
              float3 sh2 = float3(sh[base + 6], sh[base + 7], sh[base + 8]);
        
              float3 g0 = (-SH_C1 * y) * dL_dRGB;
              float3 g1 = ( SH_C1 * z) * dL_dRGB;
              float3 g2 = (-SH_C1 * x) * dL_dRGB;
              dL_dsh[base + 0] = g0.x; dL_dsh[base + 1] = g0.y; dL_dsh[base + 2] = g0.z;
              dL_dsh[base + 3] = g1.x; dL_dsh[base + 4] = g1.y; dL_dsh[base + 5] = g1.z;
              dL_dsh[base + 6] = g2.x; dL_dsh[base + 7] = g2.y; dL_dsh[base + 8] = g2.z;
        
              float3 dRGBdx = -SH_C1 * sh2;
              float3 dRGBdy = -SH_C1 * sh0;
              float3 dRGBdz =  SH_C1 * sh1;
        
              if (int(params[10]) > 1 && int(params[11]) >= 8) {
                float xx = x * x, yy = y * y, zz = z * z;
                float xy = x * y, yz = y * z, xz = x * z;
                uint b3 = base + 9u;   // coeff 3
                uint b4 = base + 12u;  // coeff 4
                uint b5 = base + 15u;  // coeff 5
                uint b6 = base + 18u;  // coeff 6
                uint b7 = base + 21u;  // coeff 7
        
                float3 sh3 = float3(sh[b3 + 0], sh[b3 + 1], sh[b3 + 2]);
                float3 sh4 = float3(sh[b4 + 0], sh[b4 + 1], sh[b4 + 2]);
                float3 sh5 = float3(sh[b5 + 0], sh[b5 + 1], sh[b5 + 2]);
                float3 sh6 = float3(sh[b6 + 0], sh[b6 + 1], sh[b6 + 2]);
                float3 sh7 = float3(sh[b7 + 0], sh[b7 + 1], sh[b7 + 2]);
        
                float3 g3 = (SH_C2[0] * xy) * dL_dRGB;
                float3 g4 = (SH_C2[1] * yz) * dL_dRGB;
                float3 g5 = (SH_C2[2] * (2.0f * zz - xx - yy)) * dL_dRGB;
                float3 g6 = (SH_C2[3] * xz) * dL_dRGB;
                float3 g7 = (SH_C2[4] * (xx - yy)) * dL_dRGB;
                dL_dsh[b3 + 0] = g3.x; dL_dsh[b3 + 1] = g3.y; dL_dsh[b3 + 2] = g3.z;
                dL_dsh[b4 + 0] = g4.x; dL_dsh[b4 + 1] = g4.y; dL_dsh[b4 + 2] = g4.z;
                dL_dsh[b5 + 0] = g5.x; dL_dsh[b5 + 1] = g5.y; dL_dsh[b5 + 2] = g5.z;
                dL_dsh[b6 + 0] = g6.x; dL_dsh[b6 + 1] = g6.y; dL_dsh[b6 + 2] = g6.z;
                dL_dsh[b7 + 0] = g7.x; dL_dsh[b7 + 1] = g7.y; dL_dsh[b7 + 2] = g7.z;
        
                dRGBdx += SH_C2[0] * y * sh3 + SH_C2[2] * 2.f * -x * sh5 + SH_C2[3] * z * sh6 + SH_C2[4] * 2.f * x * sh7;
                dRGBdy += SH_C2[0] * x * sh3 + SH_C2[1] * z * sh4 + SH_C2[2] * 2.f * -y * sh5 + SH_C2[4] * 2.f * -y * sh7;
                dRGBdz += SH_C2[1] * y * sh4 + SH_C2[2] * 4.f * z * sh5 + SH_C2[3] * x * sh6;
        
                if (int(params[10]) > 2 && int(params[11]) >= 15) {
                  uint b8 = base + 24u;
                  uint b9 = base + 27u;
                  uint b10 = base + 30u;
                  uint b11 = base + 33u;
                  uint b12 = base + 36u;
                  uint b13 = base + 39u;
                  uint b14 = base + 42u;
        
                  float3 sh8 = float3(sh[b8 + 0], sh[b8 + 1], sh[b8 + 2]);
                  float3 sh9 = float3(sh[b9 + 0], sh[b9 + 1], sh[b9 + 2]);
                  float3 sh10 = float3(sh[b10 + 0], sh[b10 + 1], sh[b10 + 2]);
                  float3 sh11 = float3(sh[b11 + 0], sh[b11 + 1], sh[b11 + 2]);
                  float3 sh12 = float3(sh[b12 + 0], sh[b12 + 1], sh[b12 + 2]);
                  float3 sh13 = float3(sh[b13 + 0], sh[b13 + 1], sh[b13 + 2]);
                  float3 sh14 = float3(sh[b14 + 0], sh[b14 + 1], sh[b14 + 2]);
        
                  float3 g8 = (SH_C3[0] * y * (3.f * xx - yy)) * dL_dRGB;
                  float3 g9 = (SH_C3[1] * xy * z) * dL_dRGB;
                  float3 g10 = (SH_C3[2] * y * (4.f * zz - xx - yy)) * dL_dRGB;
                  float3 g11 = (SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy)) * dL_dRGB;
                  float3 g12 = (SH_C3[4] * x * (4.f * zz - xx - yy)) * dL_dRGB;
                  float3 g13 = (SH_C3[5] * z * (xx - yy)) * dL_dRGB;
                  float3 g14 = (SH_C3[6] * x * (xx - 3.f * yy)) * dL_dRGB;
                  dL_dsh[b8 + 0] = g8.x; dL_dsh[b8 + 1] = g8.y; dL_dsh[b8 + 2] = g8.z;
                  dL_dsh[b9 + 0] = g9.x; dL_dsh[b9 + 1] = g9.y; dL_dsh[b9 + 2] = g9.z;
                  dL_dsh[b10 + 0] = g10.x; dL_dsh[b10 + 1] = g10.y; dL_dsh[b10 + 2] = g10.z;
                  dL_dsh[b11 + 0] = g11.x; dL_dsh[b11 + 1] = g11.y; dL_dsh[b11 + 2] = g11.z;
                  dL_dsh[b12 + 0] = g12.x; dL_dsh[b12 + 1] = g12.y; dL_dsh[b12 + 2] = g12.z;
                  dL_dsh[b13 + 0] = g13.x; dL_dsh[b13 + 1] = g13.y; dL_dsh[b13 + 2] = g13.z;
                  dL_dsh[b14 + 0] = g14.x; dL_dsh[b14 + 1] = g14.y; dL_dsh[b14 + 2] = g14.z;
        
                  dRGBdx += (
                      SH_C3[0] * sh8 * 6.f * xy +
                      SH_C3[1] * sh9 * yz +
                      SH_C3[2] * sh10 * -2.f * xy +
                      SH_C3[3] * sh11 * -6.f * xz +
                      SH_C3[4] * sh12 * (-3.f * xx + 4.f * zz - yy) +
                      SH_C3[5] * sh13 * 2.f * xz +
                      SH_C3[6] * sh14 * 3.f * (xx - yy));
        
                  dRGBdy += (
                      SH_C3[0] * sh8 * 3.f * (xx - yy) +
                      SH_C3[1] * sh9 * xz +
                      SH_C3[2] * sh10 * (-3.f * yy + 4.f * zz - xx) +
                      SH_C3[3] * sh11 * -6.f * yz +
                      SH_C3[4] * sh12 * -2.f * xy +
                      SH_C3[5] * sh13 * -2.f * yz +
                      SH_C3[6] * sh14 * -6.f * xy);
        
                  dRGBdz += (
                      SH_C3[1] * sh9 * xy +
                      SH_C3[2] * sh10 * 8.f * yz +
                      SH_C3[3] * sh11 * 3.f * (2.f * zz - xx - yy) +
                      SH_C3[4] * sh12 * 8.f * xz +
                      SH_C3[5] * sh13 * (xx - yy));
                }
              }
              float3 dL_ddir = float3(dot(dRGBdx, dL_dRGB), dot(dRGBdy, dL_dRGB), dot(dRGBdz, dL_dRGB));
              float3 dL_dmean_sh = dnormvdv(dir_orig, dL_ddir);
              dL_dmeans3d[3 * tid + 0] += dL_dmean_sh.x;
              dL_dmeans3d[3 * tid + 1] += dL_dmean_sh.y;
              dL_dmeans3d[3 * tid + 2] += dL_dmean_sh.z;
            }
          }
        
          // Backprop cov3d( scale, quat ) -> d_scales, d_quats
          if (uint(params[3]) == 0u) {
            const float3 scale = read_packed_float3(scales, tid, 3u);
            const float4 q = read_packed_float4(quats, tid, 4u);
            const float r = q.x;
            const float x = q.y;
            const float y = q.z;
            const float z = q.w;
        
            float3x3 R = float3x3(
                1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
                2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
                2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));
        
            float3 s = params[5] * scale;
            float3x3 S = float3x3(1.0f);
            S[0][0] = s.x;
            S[1][1] = s.y;
            S[2][2] = s.z;
        
            float3x3 M = S * R;
        
            const float c0 = dL_dcov3d[cov_base + 0] + dcov_extra[0];
            const float c1 = dL_dcov3d[cov_base + 1] + dcov_extra[1];
            const float c2 = dL_dcov3d[cov_base + 2] + dcov_extra[2];
            const float c3 = dL_dcov3d[cov_base + 3] + dcov_extra[3];
            const float c4 = dL_dcov3d[cov_base + 4] + dcov_extra[4];
            const float c5 = dL_dcov3d[cov_base + 5] + dcov_extra[5];
        
            float3x3 dSigma = float3x3(
                c0, 0.5f * c1, 0.5f * c2,
                0.5f * c1, c3, 0.5f * c4,
                0.5f * c2, 0.5f * c4, c5);
        
            float3x3 dM = 2.0f * M * dSigma;
            float3x3 Rt = transpose(R);
            float3x3 dMt = transpose(dM);
        
            float dsx = dot(Rt[0], dMt[0]);
            float dsy = dot(Rt[1], dMt[1]);
            float dsz = dot(Rt[2], dMt[2]);
            dL_dscales[3 * tid + 0] = dsx;
            dL_dscales[3 * tid + 1] = dsy;
            dL_dscales[3 * tid + 2] = dsz;
        
            dMt[0] *= s.x;
            dMt[1] *= s.y;
            dMt[2] *= s.z;
        
            float4 d_q;
            d_q.x = 2 * z * (dMt[0][1] - dMt[1][0]) + 2 * y * (dMt[2][0] - dMt[0][2]) + 2 * x * (dMt[1][2] - dMt[2][1]);
            d_q.y = 2 * y * (dMt[1][0] + dMt[0][1]) + 2 * z * (dMt[2][0] + dMt[0][2]) + 2 * r * (dMt[1][2] - dMt[2][1]) - 4 * x * (dMt[2][2] + dMt[1][1]);
            d_q.z = 2 * x * (dMt[1][0] + dMt[0][1]) + 2 * r * (dMt[2][0] - dMt[0][2]) + 2 * z * (dMt[1][2] + dMt[2][1]) - 4 * y * (dMt[2][2] + dMt[0][0]);
            d_q.w = 2 * r * (dMt[0][1] - dMt[1][0]) + 2 * x * (dMt[2][0] + dMt[0][2]) + 2 * y * (dMt[1][2] + dMt[2][1]) - 4 * z * (dMt[1][1] + dMt[0][0]);
        
            dL_dquats[4 * tid + 0] = d_q.x;
            dL_dquats[4 * tid + 1] = d_q.y;
            dL_dquats[4 * tid + 2] = d_q.z;
            dL_dquats[4 * tid + 3] = d_q.w;
          }
        
          // preprocess forward passes viewspace_points through; backward is identity.
          dL_dviewspace_in[4 * tid + 0] = dL_dviewspace_out[4 * tid + 0];
          dL_dviewspace_in[4 * tid + 1] = dL_dviewspace_out[4 * tid + 1];
          dL_dviewspace_in[4 * tid + 2] = dL_dviewspace_out[4 * tid + 2];
          dL_dviewspace_in[4 * tid + 3] = dL_dviewspace_out[4 * tid + 3];
        """
}
