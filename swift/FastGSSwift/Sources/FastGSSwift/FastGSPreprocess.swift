import MLX

public struct FastGSPreprocessParams: Sendable {
    public var degree: Int
    public var maxSHCoefficients: Int
    public var scaleModifier: Float
    public var tanFovX: Float
    public var tanFovY: Float
    public var imageHeight: Int
    public var imageWidth: Int
    public var tileBounds: (x: Int, y: Int, z: Int)
    public var multiplier: Float
    public var prefiltered: Bool
    public var useCov3DPrecomputed: Bool
    public var useColorsPrecomputed: Bool

    public init(
        degree: Int,
        maxSHCoefficients: Int,
        scaleModifier: Float,
        tanFovX: Float,
        tanFovY: Float,
        imageHeight: Int,
        imageWidth: Int,
        tileBounds: (x: Int, y: Int, z: Int),
        multiplier: Float,
        prefiltered: Bool = false,
        useCov3DPrecomputed: Bool = false,
        useColorsPrecomputed: Bool = false
    ) {
        self.degree = degree
        self.maxSHCoefficients = maxSHCoefficients
        self.scaleModifier = scaleModifier
        self.tanFovX = tanFovX
        self.tanFovY = tanFovY
        self.imageHeight = imageHeight
        self.imageWidth = imageWidth
        self.tileBounds = tileBounds
        self.multiplier = multiplier
        self.prefiltered = prefiltered
        self.useCov3DPrecomputed = useCov3DPrecomputed
        self.useColorsPrecomputed = useColorsPrecomputed
    }

    fileprivate var kernelArray: MLXArray {
        MLXArray([
            Float(degree),
            Float(maxSHCoefficients),
            scaleModifier,
            multiplier,
            tanFovX,
            tanFovY,
            Float(imageWidth) / (2.0 * tanFovX),
            Float(imageHeight) / (2.0 * tanFovY),
            Float(imageWidth),
            Float(imageHeight),
            Float(tileBounds.x),
            Float(tileBounds.y),
            Float(tileBounds.z),
            prefiltered ? 1.0 : 0.0,
            useCov3DPrecomputed ? 1.0 : 0.0,
            useColorsPrecomputed ? 1.0 : 0.0,
        ], [16])
    }
}

public struct FastGSPreprocessInput {
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

    public init(
        means3D: MLXArray,
        dc: MLXArray,
        sh: MLXArray,
        colorsPrecomputed: MLXArray,
        opacities: MLXArray,
        scales: MLXArray,
        rotations: MLXArray,
        cov3DPrecomputed: MLXArray,
        viewMatrix: MLXArray,
        projectionMatrix: MLXArray,
        cameraPosition: MLXArray,
        viewspacePoints: MLXArray
    ) {
        self.means3D = means3D
        self.dc = dc
        self.sh = sh
        self.colorsPrecomputed = colorsPrecomputed
        self.opacities = opacities
        self.scales = scales
        self.rotations = rotations
        self.cov3DPrecomputed = cov3DPrecomputed
        self.viewMatrix = viewMatrix
        self.projectionMatrix = projectionMatrix
        self.cameraPosition = cameraPosition
        self.viewspacePoints = viewspacePoints
    }
}

public struct FastGSPreprocessOutput {
    public var radii: MLXArray
    public var xy: MLXArray
    public var depths: MLXArray
    public var cov3D: MLXArray
    public var rgb: MLXArray
    public var conicOpacity: MLXArray
    public var tilesTouched: MLXArray
    public var clamped: MLXArray
    public var viewspacePoints: MLXArray
}

public enum FastGSPreprocess {
    private static let kernel = MLXFast.metalKernel(
        name: "fastgs_preprocess_forward",
        inputNames: [
            "params",
            "means3d",
            "dc",
            "shs",
            "colors_precomp",
            "opacities",
            "scales",
            "rotations",
            "cov3d_precomp",
            "viewmatrix",
            "projmatrix",
            "cam_pos",
            "viewspace_points_in",
        ],
        outputNames: [
            "radii",
            "points_xy_image",
            "depths",
            "cov3ds",
            "rgb",
            "conic_opacity",
            "tiles_touched",
            "clamped",
            "viewspace_points_out",
        ],
        source: FastGSPreprocessKernelSource.body,
        header: FastGSPreprocessKernelSource.header
    )

    public static func forward(
        _ input: FastGSPreprocessInput,
        params: FastGSPreprocessParams,
        verbose: Bool = false,
        stream: StreamOrDevice = .default
    ) -> FastGSPreprocessOutput {
        validate(input, params: params)

        let count = input.means3D.shape[0]
        let threadGroupSize = max(1, min(256, count))
        let outputs = kernel(
            [
                params.kernelArray,
                input.means3D,
                input.dc,
                input.sh,
                input.colorsPrecomputed,
                input.opacities,
                input.scales,
                input.rotations,
                input.cov3DPrecomputed,
                input.viewMatrix,
                input.projectionMatrix,
                input.cameraPosition,
                input.viewspacePoints,
            ],
            grid: (count, 1, 1),
            threadGroup: (threadGroupSize, 1, 1),
            outputShapes: [
                [count],
                [count, 2],
                [count],
                [count, 6],
                [count, 3],
                [count, 4],
                [count],
                [count, 3],
                [count, 4],
            ],
            outputDTypes: [
                .int32,
                .float32,
                .float32,
                .float32,
                .float32,
                .float32,
                .uint32,
                .bool,
                input.viewspacePoints.dtype,
            ],
            initValue: 0,
            verbose: verbose,
            stream: stream
        )

        return FastGSPreprocessOutput(
            radii: outputs[0],
            xy: outputs[1],
            depths: outputs[2],
            cov3D: outputs[3],
            rgb: outputs[4],
            conicOpacity: outputs[5],
            tilesTouched: outputs[6],
            clamped: outputs[7],
            viewspacePoints: outputs[8]
        )
    }

    private static func validate(_ input: FastGSPreprocessInput, params: FastGSPreprocessParams) {
        let count = input.means3D.shape[0]
        precondition(input.means3D.shape == [count, 3], "means3D must have shape [N, 3].")
        precondition(input.opacities.shape == [count], "opacities must have shape [N].")
        precondition(input.viewMatrix.shape == [4, 4], "viewMatrix must have shape [4, 4].")
        precondition(input.projectionMatrix.shape == [4, 4], "projectionMatrix must have shape [4, 4].")
        precondition(input.cameraPosition.shape == [3], "cameraPosition must have shape [3].")
        precondition(input.viewspacePoints.shape == [count, 4], "viewspacePoints must have shape [N, 4].")
        precondition(input.means3D.dtype == .float32, "FastGSPreprocess currently expects float32 inputs.")
        precondition(input.viewspacePoints.dtype == .float32, "FastGSPreprocess currently expects float32 viewspace points.")
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
    }
}

private enum FastGSPreprocessKernelSource {
    static let header = """
        #define BLOCK_X 16
        #define BLOCK_Y 16

        constant float SH_C0 = 0.28209479177387814f;
        constant float SH_C1 = 0.4886025119029199f;
        constant float SH_C2[] = {
            1.0925484305920792f,
            -1.0925484305920792f,
            0.31539156525252005f,
            -1.0925484305920792f,
            0.5462742152960396f,
        };
        constant float SH_C3[] = {
            -0.5900435899266435f,
            2.890611442640554f,
            -0.4570457994644658f,
            0.3731763325901154f,
            -0.4570457994644658f,
            1.445305721320277f,
            -0.5900435899266435f,
        };

        inline float ndc2pix(float v, int s) {
          return ((v + 1.0f) * s - 1.0f) * 0.5f;
        }

        inline float3 read_packed_float3(const constant float* arr, uint idx) {
          return float3(arr[3 * idx], arr[3 * idx + 1], arr[3 * idx + 2]);
        }

        inline float3 read_packed_float3(const device float* arr, uint idx) {
          return float3(arr[3 * idx], arr[3 * idx + 1], arr[3 * idx + 2]);
        }

        inline float4 read_packed_float4(const constant float* arr, uint idx) {
          return float4(arr[4 * idx], arr[4 * idx + 1], arr[4 * idx + 2], arr[4 * idx + 3]);
        }

        inline float4 read_packed_float4(const device float* arr, uint idx) {
          return float4(arr[4 * idx], arr[4 * idx + 1], arr[4 * idx + 2], arr[4 * idx + 3]);
        }

        inline float3 read_sh_coeff(const constant float* shs, uint idx, int max_coeffs, uint coeff_idx) {
          uint off = idx * uint(max_coeffs) * 3u + coeff_idx * 3u;
          return float3(shs[off], shs[off + 1], shs[off + 2]);
        }

        inline float3 read_sh_coeff(const device float* shs, uint idx, int max_coeffs, uint coeff_idx) {
          uint off = idx * uint(max_coeffs) * 3u + coeff_idx * 3u;
          return float3(shs[off], shs[off + 1], shs[off + 2]);
        }

        inline void write_packed_float2(device float* arr, uint idx, float2 val) {
          arr[2 * idx] = val.x;
          arr[2 * idx + 1] = val.y;
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

        inline float3 transform_point_4x3(const float3 p, const constant float* matrix) {
          return float3(
              matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
              matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
              matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14]);
        }

        inline float3 transform_point_4x3(const float3 p, const device float* matrix) {
          return float3(
              matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
              matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
              matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14]);
        }

        inline float4 transform_point_4x4(const float3 p, const constant float* matrix) {
          return float4(
              matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
              matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
              matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
              matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]);
        }

        inline float4 transform_point_4x4(const float3 p, const device float* matrix) {
          return float4(
              matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
              matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
              matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
              matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]);
        }

        inline bool in_frustum(uint idx,
                               const constant float* orig_points,
                               const constant float* viewmatrix,
                               const constant float* projmatrix,
                               bool prefiltered,
                               thread float3& p_view) {
          float3 p_orig = read_packed_float3(orig_points, idx);
          float4 p_hom = transform_point_4x4(p_orig, projmatrix);
          float p_w = 1.0f / (p_hom.w + 1.0e-7f);
          float3 p_proj = float3(p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w);
          p_view = transform_point_4x3(p_orig, viewmatrix);
          if (p_view.z <= 0.2f) {
            if (prefiltered) {
              return false;
            }
            return false;
          }
          (void)p_proj;
          return true;
        }

        inline bool in_frustum(uint idx,
                               const constant float* orig_points,
                               const device float* viewmatrix,
                               const device float* projmatrix,
                               bool prefiltered,
                               thread float3& p_view) {
          float3 p_orig = read_packed_float3(orig_points, idx);
          float4 p_hom = transform_point_4x4(p_orig, projmatrix);
          float p_w = 1.0f / (p_hom.w + 1.0e-7f);
          float3 p_proj = float3(p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w);
          p_view = transform_point_4x3(p_orig, viewmatrix);
          if (p_view.z <= 0.2f) {
            if (prefiltered) {
              return false;
            }
            return false;
          }
          (void)p_proj;
          return true;
        }

        inline float3 compute_color_from_sh(uint idx,
                                            int deg,
                                            int max_coeffs,
                                            const constant float* means,
                                            float3 campos,
                                            const constant float* dc,
                                            const constant float* shs,
                                            device bool* clamped) {
          float3 pos = read_packed_float3(means, idx);
          float3 dir = normalize(pos - campos);

          uint base = 3 * idx;
          float3 result = SH_C0 * float3(dc[base], dc[base + 1], dc[base + 2]);

          if (deg > 0) {
            float x = dir.x;
            float y = dir.y;
            float z = dir.z;
            result = result - SH_C1 * y * read_sh_coeff(shs, idx, max_coeffs, 0u) +
                     SH_C1 * z * read_sh_coeff(shs, idx, max_coeffs, 1u) -
                     SH_C1 * x * read_sh_coeff(shs, idx, max_coeffs, 2u);

            if (deg > 1) {
              float xx = x * x;
              float yy = y * y;
              float zz = z * z;
              float xy = x * y;
              float yz = y * z;
              float xz = x * z;
              result += SH_C2[0] * xy * read_sh_coeff(shs, idx, max_coeffs, 3u) +
                        SH_C2[1] * yz * read_sh_coeff(shs, idx, max_coeffs, 4u) +
                        SH_C2[2] * (2.0f * zz - xx - yy) * read_sh_coeff(shs, idx, max_coeffs, 5u) +
                        SH_C2[3] * xz * read_sh_coeff(shs, idx, max_coeffs, 6u) +
                        SH_C2[4] * (xx - yy) * read_sh_coeff(shs, idx, max_coeffs, 7u);

              if (deg > 2) {
                result += SH_C3[0] * y * (3.0f * xx - yy) * read_sh_coeff(shs, idx, max_coeffs, 8u) +
                          SH_C3[1] * xy * z * read_sh_coeff(shs, idx, max_coeffs, 9u) +
                          SH_C3[2] * y * (4.0f * zz - xx - yy) * read_sh_coeff(shs, idx, max_coeffs, 10u) +
                          SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * read_sh_coeff(shs, idx, max_coeffs, 11u) +
                          SH_C3[4] * x * (4.0f * zz - xx - yy) * read_sh_coeff(shs, idx, max_coeffs, 12u) +
                          SH_C3[5] * z * (xx - yy) * read_sh_coeff(shs, idx, max_coeffs, 13u) +
                          SH_C3[6] * x * (xx - 3.0f * yy) * read_sh_coeff(shs, idx, max_coeffs, 14u);
              }
            }
          }

          result += 0.5f;
          clamped[3 * idx + 0] = result.x < 0.0f;
          clamped[3 * idx + 1] = result.y < 0.0f;
          clamped[3 * idx + 2] = result.z < 0.0f;
          return max(result, float3(0.0f));
        }

        inline float3 compute_color_from_sh(uint idx,
                                            int deg,
                                            int max_coeffs,
                                            const constant float* means,
                                            float3 campos,
                                            const constant float* dc,
                                            const device float* shs,
                                            device bool* clamped) {
          float3 pos = read_packed_float3(means, idx);
          float3 dir = normalize(pos - campos);

          uint base = 3 * idx;
          float3 result = SH_C0 * float3(dc[base], dc[base + 1], dc[base + 2]);

          if (deg > 0) {
            float x = dir.x;
            float y = dir.y;
            float z = dir.z;
            result = result - SH_C1 * y * read_sh_coeff(shs, idx, max_coeffs, 0u) +
                     SH_C1 * z * read_sh_coeff(shs, idx, max_coeffs, 1u) -
                     SH_C1 * x * read_sh_coeff(shs, idx, max_coeffs, 2u);

            if (deg > 1) {
              float xx = x * x;
              float yy = y * y;
              float zz = z * z;
              float xy = x * y;
              float yz = y * z;
              float xz = x * z;
              result += SH_C2[0] * xy * read_sh_coeff(shs, idx, max_coeffs, 3u) +
                        SH_C2[1] * yz * read_sh_coeff(shs, idx, max_coeffs, 4u) +
                        SH_C2[2] * (2.0f * zz - xx - yy) * read_sh_coeff(shs, idx, max_coeffs, 5u) +
                        SH_C2[3] * xz * read_sh_coeff(shs, idx, max_coeffs, 6u) +
                        SH_C2[4] * (xx - yy) * read_sh_coeff(shs, idx, max_coeffs, 7u);

              if (deg > 2) {
                result += SH_C3[0] * y * (3.0f * xx - yy) * read_sh_coeff(shs, idx, max_coeffs, 8u) +
                          SH_C3[1] * xy * z * read_sh_coeff(shs, idx, max_coeffs, 9u) +
                          SH_C3[2] * y * (4.0f * zz - xx - yy) * read_sh_coeff(shs, idx, max_coeffs, 10u) +
                          SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * read_sh_coeff(shs, idx, max_coeffs, 11u) +
                          SH_C3[4] * x * (4.0f * zz - xx - yy) * read_sh_coeff(shs, idx, max_coeffs, 12u) +
                          SH_C3[5] * z * (xx - yy) * read_sh_coeff(shs, idx, max_coeffs, 13u) +
                          SH_C3[6] * x * (xx - 3.0f * yy) * read_sh_coeff(shs, idx, max_coeffs, 14u);
              }
            }
          }

          result += 0.5f;
          clamped[3 * idx + 0] = result.x < 0.0f;
          clamped[3 * idx + 1] = result.y < 0.0f;
          clamped[3 * idx + 2] = result.z < 0.0f;
          return max(result, float3(0.0f));
        }

        inline float3 compute_cov2d(const float3 mean,
                                    float focal_x,
                                    float focal_y,
                                    float tan_fovx,
                                    float tan_fovy,
                                    const thread float* cov3d,
                                    const constant float* viewmatrix) {
          float3 t = transform_point_4x3(mean, viewmatrix);
          float limx = 1.3f * tan_fovx;
          float limy = 1.3f * tan_fovy;
          float txtz = t.x / t.z;
          float tytz = t.y / t.z;
          t.x = min(limx, max(-limx, txtz)) * t.z;
          t.y = min(limy, max(-limy, tytz)) * t.z;

          float3x3 j = float3x3(
              focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
              0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
              0.0f, 0.0f, 0.0f);

          float3x3 w = float3x3(
              viewmatrix[0], viewmatrix[4], viewmatrix[8],
              viewmatrix[1], viewmatrix[5], viewmatrix[9],
              viewmatrix[2], viewmatrix[6], viewmatrix[10]);

          float3x3 t_mat = w * j;
          float3x3 vrk = float3x3(
              cov3d[0], cov3d[1], cov3d[2],
              cov3d[1], cov3d[3], cov3d[4],
              cov3d[2], cov3d[4], cov3d[5]);
          float3x3 cov = transpose(t_mat) * transpose(vrk) * t_mat;
          cov[0][0] += 0.3f;
          cov[1][1] += 0.3f;
          return float3(cov[0][0], cov[0][1], cov[1][1]);
        }

        inline float3 compute_cov2d(const float3 mean,
                                    float focal_x,
                                    float focal_y,
                                    float tan_fovx,
                                    float tan_fovy,
                                    const thread float* cov3d,
                                    const device float* viewmatrix) {
          float3 t = transform_point_4x3(mean, viewmatrix);
          float limx = 1.3f * tan_fovx;
          float limy = 1.3f * tan_fovy;
          float txtz = t.x / t.z;
          float tytz = t.y / t.z;
          t.x = min(limx, max(-limx, txtz)) * t.z;
          t.y = min(limy, max(-limy, tytz)) * t.z;

          float3x3 j = float3x3(
              focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
              0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
              0.0f, 0.0f, 0.0f);

          float3x3 w = float3x3(
              viewmatrix[0], viewmatrix[4], viewmatrix[8],
              viewmatrix[1], viewmatrix[5], viewmatrix[9],
              viewmatrix[2], viewmatrix[6], viewmatrix[10]);

          float3x3 t_mat = w * j;
          float3x3 vrk = float3x3(
              cov3d[0], cov3d[1], cov3d[2],
              cov3d[1], cov3d[3], cov3d[4],
              cov3d[2], cov3d[4], cov3d[5]);
          float3x3 cov = transpose(t_mat) * transpose(vrk) * t_mat;
          cov[0][0] += 0.3f;
          cov[1][1] += 0.3f;
          return float3(cov[0][0], cov[0][1], cov[1][1]);
        }

        inline void compute_cov3d(float3 scale, float mod, float4 rot, device float* cov3d) {
          float3x3 s = float3x3(1.0f);
          s[0][0] = mod * scale.x;
          s[1][1] = mod * scale.y;
          s[2][2] = mod * scale.z;

          float r = rot.x;
          float x = rot.y;
          float y = rot.z;
          float z = rot.w;

          float3x3 rmat = float3x3(
              1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
              2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
              2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));

          float3x3 m = s * rmat;
          float3x3 sigma = transpose(m) * m;
          cov3d[0] = sigma[0][0];
          cov3d[1] = sigma[0][1];
          cov3d[2] = sigma[0][2];
          cov3d[3] = sigma[1][1];
          cov3d[4] = sigma[1][2];
          cov3d[5] = sigma[2][2];
        }

        inline float evaluate_opacity_factor(float dx, float dy, float4 co) {
          return 0.5f * (co.x * dx * dx + co.z * dy * dy) + co.y * dx * dy;
        }

        inline float2 compute_ellipse_intersection(float4 con_o,
                                                   float disc,
                                                   float t,
                                                   float2 p,
                                                   bool is_y,
                                                   float coord) {
          float p_u = is_y ? p.y : p.x;
          float p_v = is_y ? p.x : p.y;
          float coeff = is_y ? con_o.x : con_o.z;
          float h = coord - p_u;
          float sqrt_term = sqrt(disc * h * h + t * coeff);
          return float2(
              (-con_o.y * h - sqrt_term) / coeff + p_v,
              (-con_o.y * h + sqrt_term) / coeff + p_v);
        }

        inline uint process_tiles(float4 con_o,
                                  float disc,
                                  float t,
                                  float2 p,
                                  float2 bbox_min,
                                  float2 bbox_max,
                                  float2 bbox_argmin,
                                  float2 bbox_argmax,
                                  int2 rect_min,
                                  int2 rect_max,
                                  uint3 grid,
                                  bool is_y) {
          float block_u = is_y ? BLOCK_Y : BLOCK_X;
          float block_v = is_y ? BLOCK_X : BLOCK_Y;

          if (is_y) {
            rect_min = int2(rect_min.y, rect_min.x);
            rect_max = int2(rect_max.y, rect_max.x);
            bbox_min = float2(bbox_min.y, bbox_min.x);
            bbox_max = float2(bbox_max.y, bbox_max.x);
            bbox_argmin = float2(bbox_argmin.y, bbox_argmin.x);
            bbox_argmax = float2(bbox_argmax.y, bbox_argmax.x);
          }

          uint tiles_count = 0;
          float2 intersect_min_line;
          float2 intersect_max_line = float2(bbox_max.y, bbox_min.y);
          float ellipse_min;
          float ellipse_max;
          float min_line = rect_min.x * block_u;

          if (bbox_min.x <= min_line) {
            intersect_min_line =
                compute_ellipse_intersection(con_o, disc, t, p, is_y, rect_min.x * block_u);
          } else {
            intersect_min_line = intersect_max_line;
          }

          for (int u = rect_min.x; u < rect_max.x; ++u) {
            float max_line = min_line + block_u;
            if (max_line <= bbox_max.x) {
              intersect_max_line =
                  compute_ellipse_intersection(con_o, disc, t, p, is_y, max_line);
            }

            if (min_line <= bbox_argmin.y && bbox_argmin.y < max_line) {
              ellipse_min = bbox_min.y;
            } else {
              ellipse_min = min(intersect_min_line.x, intersect_max_line.x);
            }

            if (min_line <= bbox_argmax.y && bbox_argmax.y < max_line) {
              ellipse_max = bbox_max.y;
            } else {
              ellipse_max = max(intersect_min_line.y, intersect_max_line.y);
            }

            int min_tile_v = max(rect_min.y, min(rect_max.y, (int)(ellipse_min / block_v)));
            int max_tile_v =
                min(rect_max.y, max(rect_min.y, (int)(ellipse_max / block_v + 1)));
            tiles_count += uint(max_tile_v - min_tile_v);

            intersect_min_line = intersect_max_line;
            min_line = max_line;
          }
          return tiles_count;
        }

        inline uint duplicate_to_tiles_touched(float2 p, float4 con_o, uint3 grid, float mult) {
          float disc = con_o.y * con_o.y - con_o.x * con_o.z;
          if (con_o.x <= 0.0f || con_o.z <= 0.0f || disc >= 0.0f) {
            return 0u;
          }

          float t = 2.0f * log(con_o.w * 255.0f);
          t = mult * t;

          float x_term = sqrt(-(con_o.y * con_o.y * t) / (disc * con_o.x));
          x_term = (con_o.y < 0.0f) ? x_term : -x_term;
          float y_term = sqrt(-(con_o.y * con_o.y * t) / (disc * con_o.z));
          y_term = (con_o.y < 0.0f) ? y_term : -y_term;

          float2 bbox_argmin = float2(p.y - y_term, p.x - x_term);
          float2 bbox_argmax = float2(p.y + y_term, p.x + x_term);

          float2 bbox_min = float2(
              compute_ellipse_intersection(con_o, disc, t, p, true, bbox_argmin.x).x,
              compute_ellipse_intersection(con_o, disc, t, p, false, bbox_argmin.y).x);
          float2 bbox_max = float2(
              compute_ellipse_intersection(con_o, disc, t, p, true, bbox_argmax.x).y,
              compute_ellipse_intersection(con_o, disc, t, p, false, bbox_argmax.y).y);

          int2 rect_min = int2(
              max(0, min((int)grid.x, (int)(bbox_min.x / BLOCK_X))),
              max(0, min((int)grid.y, (int)(bbox_min.y / BLOCK_Y))));
          int2 rect_max = int2(
              max(0, min((int)grid.x, (int)(bbox_max.x / BLOCK_X + 1))),
              max(0, min((int)grid.y, (int)(bbox_max.y / BLOCK_Y + 1))));

          int y_span = rect_max.y - rect_min.y;
          int x_span = rect_max.x - rect_min.x;
          if (y_span * x_span == 0) {
            return 0u;
          }

          bool is_y = y_span < x_span;
          return process_tiles(
              con_o,
              disc,
              t,
              p,
              bbox_min,
              bbox_max,
              bbox_argmin,
              bbox_argmax,
              rect_min,
              rect_max,
              grid,
              is_y);
        }
        """

    static let body = """
        uint tid = thread_position_in_grid.x;
        uint n = uint(means3d_shape[0]);
        if (tid >= n) {
          return;
        }

        int degree = int(params[0]);
        int max_sh_coeffs = int(params[1]);
        float scale_modifier = params[2];
        float mult = params[3];
        float tan_fovx = params[4];
        float tan_fovy = params[5];
        float focal_x = params[6];
        float focal_y = params[7];
        uint image_width = uint(params[8]);
        uint image_height = uint(params[9]);
        uint3 grid = uint3(uint(params[10]), uint(params[11]), uint(params[12]));
        bool prefiltered = params[13] != 0.0f;
        bool use_cov3d_precomp = params[14] != 0.0f;
        bool use_colors_precomp = params[15] != 0.0f;

        radii[tid] = 0;
        tiles_touched[tid] = 0u;

        float3 p_view;
        if (!in_frustum(tid, means3d, viewmatrix, projmatrix, prefiltered, p_view)) {
          return;
        }

        float3 p_orig = read_packed_float3(means3d, tid);
        float4 p_hom = transform_point_4x4(p_orig, projmatrix);
        float p_w = 1.0f / (p_hom.w + 1.0e-7f);
        float3 p_proj = float3(p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w);

        thread float local_cov3d[6];
        if (use_cov3d_precomp) {
          for (uint i = 0; i < 6; ++i) {
            local_cov3d[i] = cov3d_precomp[tid * 6 + i];
            cov3ds[tid * 6 + i] = local_cov3d[i];
          }
        } else {
          compute_cov3d(read_packed_float3(scales, tid), scale_modifier, read_packed_float4(rotations, tid),
                        cov3ds + tid * 6);
          for (uint i = 0; i < 6; ++i) {
            local_cov3d[i] = cov3ds[tid * 6 + i];
          }
        }

        float3 cov = compute_cov2d(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, local_cov3d, viewmatrix);

        float det = cov.x * cov.z - cov.y * cov.y;
        if (det == 0.0f) {
          return;
        }
        float det_inv = 1.f / det;
        float3 conic = float3(cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv);

        float mid = 0.5f * (cov.x + cov.z);
        float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
        float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
        float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
        float2 point_image = float2(ndc2pix(p_proj.x, int(image_width)),
                                    ndc2pix(p_proj.y, int(image_height)));

        float4 con_o = float4(conic.x, conic.y, conic.z, opacities[tid]);
        uint tiles_count = duplicate_to_tiles_touched(point_image, con_o, grid, mult);
        if (tiles_count == 0u) {
          return;
        }

        if (use_colors_precomp) {
          write_packed_float3(rgb, tid, read_packed_float3(colors_precomp, tid));
          clamped[3 * tid + 0] = false;
          clamped[3 * tid + 1] = false;
          clamped[3 * tid + 2] = false;
        } else {
          float3 result = compute_color_from_sh(
              tid, degree, max_sh_coeffs, means3d, read_packed_float3(cam_pos, 0), dc, shs, clamped);
          write_packed_float3(rgb, tid, result);
        }

        depths[tid] = p_view.z;
        radii[tid] = int(my_radius);
        write_packed_float2(points_xy_image, tid, point_image);
        write_packed_float4(conic_opacity, tid, con_o);
        tiles_touched[tid] = tiles_count;

        write_packed_float4(
            viewspace_points_out,
            tid,
            float4(p_view.x, p_view.y, p_view.z, viewspace_points_in[4 * tid + 3]));
        """
}
