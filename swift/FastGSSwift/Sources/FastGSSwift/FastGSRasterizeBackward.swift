import MLX

public struct FastGSRasterizeCotangents {
    public var bucketToTile: MLXArray
    public var sampledT: MLXArray
    public var sampledAr: MLXArray
    public var finalT: MLXArray
    public var nContrib: MLXArray
    public var maxContrib: MLXArray
    public var pixelColors: MLXArray
    public var outColor: MLXArray
    public var metricCount: MLXArray

    public init(
        bucketToTile: MLXArray,
        sampledT: MLXArray,
        sampledAr: MLXArray,
        finalT: MLXArray,
        nContrib: MLXArray,
        maxContrib: MLXArray,
        pixelColors: MLXArray,
        outColor: MLXArray,
        metricCount: MLXArray
    ) {
        self.bucketToTile = bucketToTile
        self.sampledT = sampledT
        self.sampledAr = sampledAr
        self.finalT = finalT
        self.nContrib = nContrib
        self.maxContrib = maxContrib
        self.pixelColors = pixelColors
        self.outColor = outColor
        self.metricCount = metricCount
    }

    public static func outColorOnes(
        like forwardOutput: FastGSRasterizeOutput,
        stream: StreamOrDevice = .default
    ) -> FastGSRasterizeCotangents {
        FastGSRasterizeCotangents(
            bucketToTile: MLXArray.zeros(forwardOutput.bucketToTile.shape, dtype: forwardOutput.bucketToTile.dtype, stream: stream),
            sampledT: MLXArray.zeros(forwardOutput.sampledT.shape, dtype: forwardOutput.sampledT.dtype, stream: stream),
            sampledAr: MLXArray.zeros(forwardOutput.sampledAr.shape, dtype: forwardOutput.sampledAr.dtype, stream: stream),
            finalT: MLXArray.zeros(forwardOutput.finalT.shape, dtype: forwardOutput.finalT.dtype, stream: stream),
            nContrib: MLXArray.zeros(forwardOutput.nContrib.shape, dtype: forwardOutput.nContrib.dtype, stream: stream),
            maxContrib: MLXArray.zeros(forwardOutput.maxContrib.shape, dtype: forwardOutput.maxContrib.dtype, stream: stream),
            pixelColors: MLXArray.zeros(forwardOutput.pixelColors.shape, dtype: forwardOutput.pixelColors.dtype, stream: stream),
            outColor: MLXArray.ones(forwardOutput.outColor.shape, dtype: forwardOutput.outColor.dtype, stream: stream),
            metricCount: MLXArray.zeros(forwardOutput.metricCount.shape, dtype: forwardOutput.metricCount.dtype, stream: stream)
        )
    }
}

public struct FastGSRasterizeBackwardOutput {
    public var ranges: MLXArray
    public var pointList: MLXArray
    public var bucketOffsets: MLXArray
    public var means2D: MLXArray
    public var colors: MLXArray
    public var conicOpacity: MLXArray
    public var background: MLXArray
    public var radii: MLXArray
    public var metricMap: MLXArray
    public var viewspacePoints: MLXArray

    public var arrays: [MLXArray] {
        [
            ranges,
            pointList,
            bucketOffsets,
            means2D,
            colors,
            conicOpacity,
            background,
            radii,
            metricMap,
            viewspacePoints,
        ]
    }
}

public enum FastGSRasterizeBackward {
    private static let kernel = MLXFast.metalKernel(
        name: "fastgs_render_backward_swift_skeleton",
        inputNames: [
            "params",
            "ranges",
            "point_list",
            "per_tile_bucket_offset",
            "means2d",
            "colors",
            "conic_opacity",
            "background",
            "dL_dout_color",
            "bucket_to_tile",
            "sampled_t",
            "sampled_ar",
            "final_t",
            "n_contrib",
            "max_contrib",
            "pixel_colors",
        ],
        outputNames: [
            "dL_dmeans2d",
            "dL_dcolors",
            "dL_dconic_opacity",
            "dL_dviewspace_points",
        ],
        source: FastGSRasterizeBackwardKernelSource.body,
        header: FastGSRasterizeBackwardKernelSource.header
    )

    public static func forward(
        input: FastGSRasterizeInput,
        cotangents: FastGSRasterizeCotangents,
        forwardOutput: FastGSRasterizeOutput,
        params: FastGSRasterizeParams,
        verbose: Bool = false,
        stream: StreamOrDevice = .default
    ) -> FastGSRasterizeBackwardOutput {
        validate(input: input, cotangents: cotangents, forwardOutput: forwardOutput, params: params)

        let bucketSum = Int(input.bucketOffsets.asArray(UInt32.self).last ?? 0)
        let emptyOutput = FastGSRasterizeBackwardOutput(
            ranges: MLXArray.zeros(input.ranges.shape, dtype: input.ranges.dtype, stream: stream),
            pointList: MLXArray.zeros(input.pointList.shape, dtype: input.pointList.dtype, stream: stream),
            bucketOffsets: MLXArray.zeros(input.bucketOffsets.shape, dtype: input.bucketOffsets.dtype, stream: stream),
            means2D: MLXArray.zeros(input.means2D.shape, dtype: input.means2D.dtype, stream: stream),
            colors: MLXArray.zeros(input.colors.shape, dtype: input.colors.dtype, stream: stream),
            conicOpacity: MLXArray.zeros(input.conicOpacity.shape, dtype: input.conicOpacity.dtype, stream: stream),
            background: MLXArray.zeros(input.background.shape, dtype: input.background.dtype, stream: stream),
            radii: MLXArray.zeros(input.radii.shape, dtype: input.radii.dtype, stream: stream),
            metricMap: MLXArray.zeros(input.metricMap.shape, dtype: input.metricMap.dtype, stream: stream),
            viewspacePoints: MLXArray.zeros(input.means2D.shape[0] == 0 ? [0, 4] : [input.means2D.shape[0], 4], dtype: .float32, stream: stream)
        )

        if bucketSum == 0 {
            return emptyOutput
        }

        let outputs = kernel(
            [
                params.backwardKernelArray(bucketSum: bucketSum),
                input.ranges,
                input.pointList,
                input.bucketOffsets,
                input.means2D,
                input.colors,
                input.conicOpacity,
                input.background,
                cotangents.outColor,
                forwardOutput.bucketToTile,
                forwardOutput.sampledT,
                forwardOutput.sampledAr,
                forwardOutput.finalT,
                forwardOutput.nContrib,
                forwardOutput.maxContrib,
                forwardOutput.pixelColors,
            ],
            grid: (bucketSum * 32, 1, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [
                input.means2D.shape,
                input.colors.shape,
                input.conicOpacity.shape,
                emptyOutput.viewspacePoints.shape,
            ],
            outputDTypes: [
                input.means2D.dtype,
                input.colors.dtype,
                input.conicOpacity.dtype,
                .float32,
            ],
            initValue: 0,
            verbose: verbose,
            stream: stream
        )

        return FastGSRasterizeBackwardOutput(
            ranges: emptyOutput.ranges,
            pointList: emptyOutput.pointList,
            bucketOffsets: emptyOutput.bucketOffsets,
            means2D: outputs[0],
            colors: outputs[1],
            conicOpacity: outputs[2],
            background: emptyOutput.background,
            radii: emptyOutput.radii,
            metricMap: emptyOutput.metricMap,
            viewspacePoints: outputs[3]
        )
    }

    public static func forward(
        preprocessOutput: FastGSPreprocessOutput,
        binningOutput: FastGSBinningOutput,
        rasterizeOutput: FastGSRasterizeOutput,
        cotangents: FastGSRasterizeCotangents,
        background: MLXArray,
        params: FastGSRasterizeParams,
        metricMap: MLXArray? = nil,
        metricCount: MLXArray? = nil,
        verbose: Bool = false,
        stream: StreamOrDevice = .default
    ) -> FastGSRasterizeBackwardOutput {
        let count = preprocessOutput.xy.shape[0]
        let numPixels = params.imageWidth * params.imageHeight
        return forward(
            input: FastGSRasterizeInput(
                ranges: binningOutput.ranges,
                pointList: binningOutput.pointList,
                bucketOffsets: binningOutput.bucketOffsets,
                means2D: preprocessOutput.xy,
                colors: preprocessOutput.rgb,
                conicOpacity: preprocessOutput.conicOpacity,
                background: background,
                radii: preprocessOutput.radii,
                metricMap: metricMap ?? MLXArray.zeros([numPixels], dtype: .int32, stream: stream),
                metricCount: metricCount ?? MLXArray.zeros([count], dtype: .int32, stream: stream)
            ),
            cotangents: cotangents,
            forwardOutput: rasterizeOutput,
            params: params,
            verbose: verbose,
            stream: stream
        )
    }

    private static func validate(
        input: FastGSRasterizeInput,
        cotangents: FastGSRasterizeCotangents,
        forwardOutput: FastGSRasterizeOutput,
        params: FastGSRasterizeParams
    ) {
        precondition(cotangents.outColor.shape == forwardOutput.outColor.shape, "outColor cotangent shape mismatch.")
        precondition(cotangents.outColor.dtype == .float32, "outColor cotangent must be float32.")
        precondition(forwardOutput.bucketToTile.dtype == .uint32, "bucketToTile must be uint32.")
        precondition(forwardOutput.sampledT.dtype == .float32, "sampledT must be float32.")
        precondition(forwardOutput.sampledAr.dtype == .float32, "sampledAr must be float32.")
        precondition(forwardOutput.finalT.dtype == .float32, "finalT must be float32.")
        precondition(forwardOutput.nContrib.dtype == .uint32, "nContrib must be uint32.")
        precondition(forwardOutput.maxContrib.dtype == .uint32, "maxContrib must be uint32.")
        precondition(forwardOutput.pixelColors.dtype == .float32, "pixelColors must be float32.")
        precondition(forwardOutput.outColor.shape == [params.numChannels, params.imageWidth * params.imageHeight])
        precondition(input.means2D.dtype == .float32, "means2D must be float32.")
        precondition(input.colors.dtype == .float32, "colors must be float32.")
        precondition(input.conicOpacity.dtype == .float32, "conicOpacity must be float32.")
        precondition(input.background.dtype == .float32, "background must be float32.")
    }
}

private extension FastGSRasterizeParams {
    func backwardKernelArray(bucketSum: Int) -> MLXArray {
        MLXArray([
            UInt32(imageWidth),
            UInt32(imageHeight),
            UInt32(blockX),
            UInt32(blockY),
            UInt32(numChannels),
            UInt32(numTiles),
            UInt32(bucketSum),
            UInt32(blockX * blockY),
        ], [8])
    }
}

private enum FastGSRasterizeBackwardKernelSource {
    static let header = """
        inline float2 read_packed_float2(const constant float* arr, uint idx) {
          return float2(arr[2 * idx], arr[2 * idx + 1]);
        }

        inline float2 read_packed_float2(const device float* arr, uint idx) {
          return float2(arr[2 * idx], arr[2 * idx + 1]);
        }

        inline float4 read_packed_float4(const constant float* arr, uint idx) {
          return float4(arr[4 * idx], arr[4 * idx + 1], arr[4 * idx + 2], arr[4 * idx + 3]);
        }

        inline float4 read_packed_float4(const device float* arr, uint idx) {
          return float4(arr[4 * idx], arr[4 * idx + 1], arr[4 * idx + 2], arr[4 * idx + 3]);
        }

        inline void atomic_add_float(device float* ptr, uint index, float value) {
          atomic_fetch_add_explicit(
              (device atomic_float*)&ptr[index], value, memory_order_relaxed);
        }
        """

    static let body = """
        const uint image_width = params[0];
        const uint image_height = params[1];
        const uint block_x = params[2];
        const uint block_y = params[3];
        const uint num_channels = params[4];
        const uint num_tiles = params[5];
        const uint bucket_sum = params[6];
        const uint block_size = params[7];
        const uint C = min(num_channels, 3u);

        const uint lane = thread_position_in_threadgroup.x;
        const uint tid_in_tg = thread_position_in_threadgroup.x;
        const uint bucket = threadgroup_position_in_grid.x;
        if (bucket >= bucket_sum) {
          return;
        }

        const uint tile_id = bucket_to_tile[bucket];
        if (tile_id >= num_tiles) {
          return;
        }

        const uint2 range = uint2(ranges[2 * tile_id], ranges[2 * tile_id + 1]);
        const int num_splats_in_tile = int(range.y - range.x);
        const uint bbm = (tile_id == 0u) ? 0u : per_tile_bucket_offset[tile_id - 1u];
        const int bucket_idx_in_tile = int(bucket - bbm);
        const int splat_idx_in_tile = bucket_idx_in_tile * 32 + int(lane);
        const int splat_idx_global = int(range.x) + splat_idx_in_tile;
        const bool valid_splat = (splat_idx_in_tile < num_splats_in_tile);

        if (bucket_idx_in_tile * 32 >= int(max_contrib[tile_id])) {
          return;
        }

        int gaussian_idx = 0;
        float2 xy = float2(0.0f);
        float4 con_o = float4(0.0f);
        float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f;
        if (valid_splat) {
          gaussian_idx = int(point_list[splat_idx_global]);
          xy = read_packed_float2(means2d, uint(gaussian_idx));
          con_o = read_packed_float4(conic_opacity, uint(gaussian_idx));
          if (C > 0u) c0 = colors[uint(gaussian_idx) * num_channels + 0u];
          if (C > 1u) c1 = colors[uint(gaussian_idx) * num_channels + 1u];
          if (C > 2u) c2 = colors[uint(gaussian_idx) * num_channels + 2u];
        }

        float reg_dmean_x = 0.0f;
        float reg_dmean_y = 0.0f;
        float reg_dmean_z = 0.0f;
        float reg_dmean_w = 0.0f;
        float reg_dconic_x = 0.0f;
        float reg_dconic_y = 0.0f;
        float reg_dconic_w = 0.0f;
        float reg_dopacity = 0.0f;
        float reg_dcolor0 = 0.0f, reg_dcolor1 = 0.0f, reg_dcolor2 = 0.0f;

        const uint horizontal_blocks = (image_width + block_x - 1u) / block_x;
        const uint2 tile = uint2(tile_id % horizontal_blocks, tile_id / horizontal_blocks);
        const uint2 pix_min = uint2(tile.x * block_x, tile.y * block_y);
        const float ddelx_dx = 0.5f * float(image_width);
        const float ddely_dy = 0.5f * float(image_height);

        float T = 0.0f;
        float T_final = 0.0f;
        float last_contributor = 0.0f;
        float ar0 = 0.0f, ar1 = 0.0f, ar2 = 0.0f;
        float dL_dpixel0 = 0.0f, dL_dpixel1 = 0.0f, dL_dpixel2 = 0.0f;

        threadgroup float shared_sampled_ar[32 * 3 + 1];
        threadgroup float shared_pixels[32 * 3];
        const device float* sampled_ar_bucket = sampled_ar + bucket * block_size * num_channels;

        for (int i = 0; i < int(block_size) + 31; ++i) {
          if ((i % 32) == 0) {
            if (C > 0u) {
              int shift0 = i + int(tid_in_tg);
              shared_sampled_ar[0 * 32 + tid_in_tg] = sampled_ar_bucket[shift0];
            }
            if (C > 1u) {
              int shift1 = int(block_size) + i + int(tid_in_tg);
              shared_sampled_ar[1 * 32 + tid_in_tg] = sampled_ar_bucket[shift1];
            }
            if (C > 2u) {
              int shift2 = 2 * int(block_size) + i + int(tid_in_tg);
              shared_sampled_ar[2 * 32 + tid_in_tg] = sampled_ar_bucket[shift2];
            }

            const uint local_id = uint(i) + tid_in_tg;
            const uint2 pix = uint2(
                pix_min.x + (local_id % block_x),
                pix_min.y + (local_id / block_x));
            const bool pix_valid = pix.x < image_width && pix.y < image_height;
            const uint id = pix_valid ? (image_width * pix.y + pix.x) : 0u;

            if (C > 0u) {
              shared_pixels[0 * 32 + tid_in_tg] =
                  pix_valid ? pixel_colors[0u * image_height * image_width + id] : 0.0f;
            }
            if (C > 1u) {
              shared_pixels[1 * 32 + tid_in_tg] =
                  pix_valid ? pixel_colors[1u * image_height * image_width + id] : 0.0f;
            }
            if (C > 2u) {
              shared_pixels[2 * 32 + tid_in_tg] =
                  pix_valid ? pixel_colors[2u * image_height * image_width + id] : 0.0f;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
          }

          T = simd_shuffle_up(T, 1);
          last_contributor = simd_shuffle_up(last_contributor, 1);
          T_final = simd_shuffle_up(T_final, 1);
          ar0 = simd_shuffle_up(ar0, 1);
          ar1 = simd_shuffle_up(ar1, 1);
          ar2 = simd_shuffle_up(ar2, 1);
          dL_dpixel0 = simd_shuffle_up(dL_dpixel0, 1);
          dL_dpixel1 = simd_shuffle_up(dL_dpixel1, 1);
          dL_dpixel2 = simd_shuffle_up(dL_dpixel2, 1);

          const int idx = i - int(lane);
          const uint2 pix = uint2(
              pix_min.x + (uint(idx) % block_x),
              pix_min.y + (uint(idx) / block_x));
          const bool valid_pixel = pix.x < image_width && pix.y < image_height;
          const uint pix_id = valid_pixel ? (image_width * pix.y + pix.x) : 0u;
          const float2 pixf = float2(float(pix.x), float(pix.y));

          if (valid_splat && valid_pixel && lane == 0u && idx < int(block_size)) {
            T = sampled_t[bucket * block_size + uint(idx)];
            const int ii = i % 32;
            if (C > 0u) ar0 = -shared_pixels[0 * 32 + ii] + shared_sampled_ar[0 * 32 + ii];
            if (C > 1u) ar1 = -shared_pixels[1 * 32 + ii] + shared_sampled_ar[1 * 32 + ii];
            if (C > 2u) ar2 = -shared_pixels[2 * 32 + ii] + shared_sampled_ar[2 * 32 + ii];
            T_final = final_t[pix_id];
            last_contributor = float(n_contrib[pix_id]);
            if (C > 0u) dL_dpixel0 = dL_dout_color[0u * image_height * image_width + pix_id];
            if (C > 1u) dL_dpixel1 = dL_dout_color[1u * image_height * image_width + pix_id];
            if (C > 2u) dL_dpixel2 = dL_dout_color[2u * image_height * image_width + pix_id];
          }

          if (valid_splat && valid_pixel && 0 <= idx && idx < int(block_size)) {
            if (splat_idx_in_tile >= int(last_contributor)) {
              continue;
            }

            const float2 d = float2(xy.x - pixf.x, xy.y - pixf.y);
            const float power =
                -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
            if (power > 0.0f) {
              continue;
            }
            const float G = exp(power);
            const float alpha = min(0.99f, con_o.w * G);
            if (alpha < (1.0f / 255.0f)) {
              continue;
            }

            const float dchannel_dcolor = alpha * T;
            const float one_minus_alpha_reci = 1.0f / (1.0f - alpha);
            float dL_dalpha = 0.0f;

            if (C > 0u) {
              ar0 += dchannel_dcolor * c0;
              reg_dcolor0 += dchannel_dcolor * dL_dpixel0;
              dL_dalpha += (c0 * T + one_minus_alpha_reci * ar0) * dL_dpixel0;
            }
            if (C > 1u) {
              ar1 += dchannel_dcolor * c1;
              reg_dcolor1 += dchannel_dcolor * dL_dpixel1;
              dL_dalpha += (c1 * T + one_minus_alpha_reci * ar1) * dL_dpixel1;
            }
            if (C > 2u) {
              ar2 += dchannel_dcolor * c2;
              reg_dcolor2 += dchannel_dcolor * dL_dpixel2;
              dL_dalpha += (c2 * T + one_minus_alpha_reci * ar2) * dL_dpixel2;
            }

            float bg_dot_dpixel = 0.0f;
            if (C > 0u) bg_dot_dpixel += background[0] * dL_dpixel0;
            if (C > 1u) bg_dot_dpixel += background[1] * dL_dpixel1;
            if (C > 2u) bg_dot_dpixel += background[2] * dL_dpixel2;
            dL_dalpha += (-T_final * one_minus_alpha_reci) * bg_dot_dpixel;
            T *= (1.0f - alpha);

            const float dL_dG = con_o.w * dL_dalpha;
            const float gdx = G * d.x;
            const float gdy = G * d.y;
            const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
            const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

            if (con_o.w * G > 0.99f) {
              continue;
            }

            const float tmp_x = dL_dG * dG_ddelx * ddelx_dx;
            const float tmp_y = dL_dG * dG_ddely * ddely_dy;
            reg_dmean_x += tmp_x;
            reg_dmean_y += tmp_y;
            reg_dmean_z += fabs(tmp_x);
            reg_dmean_w += fabs(tmp_y);

            reg_dconic_x += -0.5f * gdx * d.x * dL_dG;
            reg_dconic_y += -0.5f * gdx * d.y * dL_dG;
            reg_dconic_w += -0.5f * gdy * d.y * dL_dG;
            reg_dopacity += G * dL_dalpha;
          }
        }

        if (valid_splat) {
          const uint g = uint(gaussian_idx);
          atomic_add_float(dL_dmeans2d, 2 * g + 0, reg_dmean_x);
          atomic_add_float(dL_dmeans2d, 2 * g + 1, reg_dmean_y);
          atomic_add_float(dL_dviewspace_points, 4 * g + 0, reg_dmean_x);
          atomic_add_float(dL_dviewspace_points, 4 * g + 1, reg_dmean_y);
          atomic_add_float(dL_dviewspace_points, 4 * g + 2, reg_dmean_z);
          atomic_add_float(dL_dviewspace_points, 4 * g + 3, reg_dmean_w);

          atomic_add_float(dL_dconic_opacity, 4 * g + 0, reg_dconic_x);
          atomic_add_float(dL_dconic_opacity, 4 * g + 1, reg_dconic_y);
          atomic_add_float(dL_dconic_opacity, 4 * g + 2, reg_dconic_w);
          atomic_add_float(dL_dconic_opacity, 4 * g + 3, reg_dopacity);

          if (C > 0u) {
            atomic_add_float(dL_dcolors, g * num_channels + 0u, reg_dcolor0);
          }
          if (C > 1u) {
            atomic_add_float(dL_dcolors, g * num_channels + 1u, reg_dcolor1);
          }
          if (C > 2u) {
            atomic_add_float(dL_dcolors, g * num_channels + 2u, reg_dcolor2);
          }
        }
        """
}
