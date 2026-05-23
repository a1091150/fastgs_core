import MLX

public struct FastGSRasterizeParams: Sendable {
    public var imageWidth: Int
    public var imageHeight: Int
    public var blockX: Int
    public var blockY: Int
    public var numChannels: Int
    public var numTiles: Int
    public var getMetricCount: Bool

    public init(
        imageWidth: Int,
        imageHeight: Int,
        blockX: Int = 16,
        blockY: Int = 16,
        numChannels: Int = 3,
        numTiles: Int,
        getMetricCount: Bool = false
    ) {
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.blockX = blockX
        self.blockY = blockY
        self.numChannels = numChannels
        self.numTiles = numTiles
        self.getMetricCount = getMetricCount
    }

    fileprivate func kernelArray(bucketSum: Int) -> MLXArray {
        MLXArray([
            UInt32(imageWidth),
            UInt32(imageHeight),
            UInt32(blockX),
            UInt32(blockY),
            UInt32(numChannels),
            UInt32(numTiles),
            UInt32(bucketSum),
            getMetricCount ? UInt32(1) : UInt32(0),
        ], [8])
    }
}

public struct FastGSRasterizeInput {
    public var ranges: MLXArray
    public var pointList: MLXArray
    public var bucketOffsets: MLXArray
    public var means2D: MLXArray
    public var colors: MLXArray
    public var conicOpacity: MLXArray
    public var background: MLXArray
    public var radii: MLXArray
    public var metricMap: MLXArray
    public var metricCount: MLXArray

    public init(
        ranges: MLXArray,
        pointList: MLXArray,
        bucketOffsets: MLXArray,
        means2D: MLXArray,
        colors: MLXArray,
        conicOpacity: MLXArray,
        background: MLXArray,
        radii: MLXArray,
        metricMap: MLXArray,
        metricCount: MLXArray
    ) {
        self.ranges = ranges
        self.pointList = pointList
        self.bucketOffsets = bucketOffsets
        self.means2D = means2D
        self.colors = colors
        self.conicOpacity = conicOpacity
        self.background = background
        self.radii = radii
        self.metricMap = metricMap
        self.metricCount = metricCount
    }
}

public struct FastGSRasterizeOutput {
    public var bucketToTile: MLXArray
    public var sampledT: MLXArray
    public var sampledAr: MLXArray
    public var finalT: MLXArray
    public var nContrib: MLXArray
    public var maxContrib: MLXArray
    public var pixelColors: MLXArray
    public var outColor: MLXArray
    public var metricCount: MLXArray
}

public enum FastGSRasterize {
    private static let kernel = MLXFast.metalKernel(
        name: "fastgs_render_forward_swift",
        inputNames: [
            "params",
            "ranges",
            "point_list",
            "per_tile_bucket_offset",
            "means2d",
            "colors",
            "conic_opacity",
            "background",
            "radii",
            "metric_map",
        ],
        outputNames: [
            "bucket_to_tile",
            "sampled_t",
            "sampled_ar",
            "final_t",
            "n_contrib",
            "max_contrib",
            "pixel_colors",
            "out_color",
            "metric_count",
        ],
        source: FastGSRasterizeKernelSource.body,
        header: FastGSRasterizeKernelSource.header
    )

    private static let previewKernel = MLXFast.metalKernel(
        name: "fastgs_render_preview_swift",
        inputNames: [
            "params",
            "ranges",
            "point_list",
            "means2d",
            "colors",
            "conic_opacity",
            "background",
        ],
        outputNames: [
            "out_color",
        ],
        source: FastGSRasterizeKernelSource.previewBody,
        header: FastGSRasterizeKernelSource.header
    )

    public static func forward(
        _ input: FastGSRasterizeInput,
        params: FastGSRasterizeParams,
        verbose: Bool = false,
        stream: StreamOrDevice = .default
    ) -> FastGSRasterizeOutput {
        validate(input, params: params)

        let bucketSum = Int(input.bucketOffsets.asArray(UInt32.self).last ?? 0)
        let numPixels = params.imageWidth * params.imageHeight
        let sampleSize = bucketSum * params.blockX * params.blockY
        let sampledArSize = params.numChannels * sampleSize

        if bucketSum == 0 {
            return FastGSRasterizeOutput(
                bucketToTile: MLXArray.zeros([0], dtype: .uint32, stream: stream),
                sampledT: MLXArray.zeros([0], dtype: .float32, stream: stream),
                sampledAr: MLXArray.zeros([0], dtype: .float32, stream: stream),
                finalT: MLXArray.zeros([numPixels], dtype: .float32, stream: stream),
                nContrib: MLXArray.zeros([numPixels], dtype: .uint32, stream: stream),
                maxContrib: MLXArray.zeros([params.numTiles], dtype: .uint32, stream: stream),
                pixelColors: MLXArray.zeros([params.numChannels, numPixels], dtype: .float32, stream: stream),
                outColor: MLXArray.zeros([params.numChannels, numPixels], dtype: .float32, stream: stream),
                metricCount: MLXArray.zeros(input.metricCount.shape, dtype: input.metricCount.dtype, stream: stream)
            )
        }

        let tilesX = (params.imageWidth + params.blockX - 1) / params.blockX
        let tilesY = (params.imageHeight + params.blockY - 1) / params.blockY
        let outputs = kernel(
            [
                params.kernelArray(bucketSum: bucketSum),
                input.ranges,
                input.pointList,
                input.bucketOffsets,
                input.means2D,
                input.colors,
                input.conicOpacity,
                input.background,
                input.radii,
                input.metricMap,
            ],
            grid: (tilesX * params.blockX, tilesY * params.blockY, 1),
            threadGroup: (params.blockX, params.blockY, 1),
            outputShapes: [
                [sampleSize],
                [sampleSize],
                [sampledArSize],
                [numPixels],
                [numPixels],
                [params.numTiles],
                [params.numChannels, numPixels],
                [params.numChannels, numPixels],
                input.metricCount.shape,
            ],
            outputDTypes: [
                .uint32,
                .float32,
                .float32,
                .float32,
                .uint32,
                .uint32,
                .float32,
                .float32,
                input.metricCount.dtype,
            ],
            initValue: 0,
            verbose: verbose,
            stream: stream
        )

        return FastGSRasterizeOutput(
            bucketToTile: outputs[0],
            sampledT: outputs[1],
            sampledAr: outputs[2],
            finalT: outputs[3],
            nContrib: outputs[4],
            maxContrib: outputs[5],
            pixelColors: outputs[6],
            outColor: outputs[7],
            metricCount: outputs[8]
        )
    }

    public static func forward(
        preprocessOutput: FastGSPreprocessOutput,
        binningOutput: FastGSBinningOutput,
        background: MLXArray,
        params: FastGSRasterizeParams,
        metricMap: MLXArray? = nil,
        metricCount: MLXArray? = nil,
        verbose: Bool = false,
        stream: StreamOrDevice = .default
    ) -> FastGSRasterizeOutput {
        let count = preprocessOutput.xy.shape[0]
        let numPixels = params.imageWidth * params.imageHeight
        return forward(
            FastGSRasterizeInput(
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
            params: params,
            verbose: verbose,
            stream: stream
        )
    }

    public static func previewOutColor(
        _ input: FastGSRasterizeInput,
        params: FastGSRasterizeParams,
        verbose: Bool = false,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        validate(input, params: params)

        let numPixels = params.imageWidth * params.imageHeight
        let bucketSum = Int(input.bucketOffsets.asArray(UInt32.self).last ?? 0)
        if bucketSum == 0 {
            return MLXArray.zeros([params.numChannels, numPixels], dtype: .float32, stream: stream)
        }

        let tilesX = (params.imageWidth + params.blockX - 1) / params.blockX
        let tilesY = (params.imageHeight + params.blockY - 1) / params.blockY
        return previewKernel(
            [
                params.kernelArray(bucketSum: bucketSum),
                input.ranges,
                input.pointList,
                input.means2D,
                input.colors,
                input.conicOpacity,
                input.background,
            ],
            grid: (tilesX * params.blockX, tilesY * params.blockY, 1),
            threadGroup: (params.blockX, params.blockY, 1),
            outputShapes: [[params.numChannels, numPixels]],
            outputDTypes: [.float32],
            initValue: 0,
            verbose: verbose,
            stream: stream
        )[0]
    }

    public static func previewOutColor(
        preprocessOutput: FastGSPreprocessOutput,
        binningOutput: FastGSBinningOutput,
        background: MLXArray,
        params: FastGSRasterizeParams,
        verbose: Bool = false,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        let count = preprocessOutput.xy.shape[0]
        let numPixels = params.imageWidth * params.imageHeight
        return previewOutColor(
            FastGSRasterizeInput(
                ranges: binningOutput.ranges,
                pointList: binningOutput.pointList,
                bucketOffsets: binningOutput.bucketOffsets,
                means2D: preprocessOutput.xy,
                colors: preprocessOutput.rgb,
                conicOpacity: preprocessOutput.conicOpacity,
                background: background,
                radii: preprocessOutput.radii,
                metricMap: MLXArray.zeros([numPixels], dtype: .int32, stream: stream),
                metricCount: MLXArray.zeros([count], dtype: .int32, stream: stream)
            ),
            params: params,
            verbose: verbose,
            stream: stream
        )
    }

    private static func validate(_ input: FastGSRasterizeInput, params: FastGSRasterizeParams) {
        let count = input.means2D.shape[0]
        let numPixels = params.imageWidth * params.imageHeight
        precondition(params.blockX == 16 && params.blockY == 16, "FastGSRasterize currently supports 16x16 blocks.")
        precondition(params.numChannels == 3, "FastGSRasterize currently supports 3 color channels.")
        precondition(input.ranges.shape == [params.numTiles, 2], "ranges must have shape [numTiles, 2].")
        precondition(input.bucketOffsets.shape == [params.numTiles], "bucketOffsets must have shape [numTiles].")
        precondition(input.means2D.shape == [count, 2], "means2D must have shape [N, 2].")
        precondition(input.colors.shape == [count, params.numChannels], "colors must have shape [N, C].")
        precondition(input.conicOpacity.shape == [count, 4], "conicOpacity must have shape [N, 4].")
        precondition(input.background.shape == [params.numChannels], "background must have shape [C].")
        precondition(input.radii.shape == [count], "radii must have shape [N].")
        precondition(input.metricMap.shape == [numPixels], "metricMap must have shape [H * W].")
        precondition(input.metricCount.shape == [count], "metricCount must have shape [N].")
        precondition(input.pointList.dtype == .uint32, "pointList must be uint32.")
        precondition(input.ranges.dtype == .uint32, "ranges must be uint32.")
        precondition(input.bucketOffsets.dtype == .uint32, "bucketOffsets must be uint32.")
        precondition(input.means2D.dtype == .float32, "means2D must be float32.")
        precondition(input.colors.dtype == .float32, "colors must be float32.")
        precondition(input.conicOpacity.dtype == .float32, "conicOpacity must be float32.")
        precondition(input.background.dtype == .float32, "background must be float32.")
        precondition(input.radii.dtype == .int32, "radii must be int32.")
        precondition(input.metricMap.dtype == .int32, "metricMap must be int32.")
        precondition(input.metricCount.dtype == .int32, "metricCount must be int32.")
    }
}

private enum FastGSRasterizeKernelSource {
    static let header = """
        #define BLOCK_X 16
        #define BLOCK_Y 16

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
        """

    static let body = """
        threadgroup uint shared_last_contributor[BLOCK_X * BLOCK_Y];
        uint2 tid = uint2(thread_position_in_threadgroup.x, thread_position_in_threadgroup.y);
        uint2 gid = uint2(thread_position_in_grid.x, thread_position_in_grid.y);
        uint2 tgp = uint2(threadgroup_position_in_grid.x, threadgroup_position_in_grid.y);

        uint image_width = params[0];
        uint image_height = params[1];
        uint block_x = params[2];
        uint block_y = params[3];
        uint num_channels = params[4];
        uint get_flag = params[7];

        uint pix_x = gid.x;
        uint pix_y = gid.y;
        uint tile_x = tgp.x;
        uint tile_y = tgp.y;
        uint tiles_x = (image_width + block_x - 1u) / block_x;
        uint tiles_y = (image_height + block_y - 1u) / block_y;

        if (tile_x >= tiles_x || tile_y >= tiles_y) {
          return;
        }

        uint tile_id = tile_y * tiles_x + tile_x;
        uint2 range = uint2(ranges[2 * tile_id], ranges[2 * tile_id + 1]);
        int to_do = int(range.y - range.x);
        int num_buckets = (to_do + 31) / 32;
        uint bbm = (tile_id == 0u) ? 0u : per_tile_bucket_offset[tile_id - 1u];

        uint local_rank = tid.y * block_x + tid.x;
        for (int i = 0;
             i < (num_buckets + int(block_x * block_y) - 1) / int(block_x * block_y);
             ++i) {
          int bucket_idx = i * int(block_x * block_y) + int(local_rank);
          if (bucket_idx < num_buckets) {
            bucket_to_tile[bbm + uint(bucket_idx)] = tile_id;
          }
        }

        bool inside = pix_x < image_width && pix_y < image_height;
        bool done = !inside;
        uint pix_id = inside ? pix_y * image_width + pix_x : 0u;
        float2 pixf = float2(float(pix_x), float(pix_y));
        float t_val = 1.0f;
        uint contributor = 0u;
        uint last_contributor = 0u;
        float c_accum[3] = {0.0f, 0.0f, 0.0f};

        for (uint idx = range.x; !done && idx < range.y; ++idx) {
          if (((idx - range.x) % 32u) == 0u) {
            sampled_t[bbm * (block_x * block_y) + local_rank] = t_val;
            for (uint ch = 0; ch < min(num_channels, 3u); ++ch) {
              sampled_ar[(bbm * num_channels * (block_x * block_y)) +
                         ch * (block_x * block_y) + local_rank] = c_accum[ch];
            }
            bbm++;
          }

          contributor++;
          uint coll_id = point_list[idx];
          float2 xy = read_packed_float2(means2d, coll_id);
          float2 d = xy - pixf;
          float4 con_o = read_packed_float4(conic_opacity, coll_id);
          float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) -
                        con_o.y * d.x * d.y;
          if (power > 0.0f) {
            continue;
          }

          float alpha = min(0.99f, con_o.w * exp(power));
          if (alpha < 1.0f / 255.0f) {
            continue;
          }

          float test_t = t_val * (1.0f - alpha);
          if (test_t < 0.0001f) {
            done = true;
            break;
          }

          for (uint ch = 0; ch < min(num_channels, 3u); ++ch) {
            c_accum[ch] += colors[coll_id * num_channels + ch] * alpha * t_val;
          }

          if (get_flag != 0u && metric_map[pix_id] == 1) {
            atomic_fetch_add_explicit(
                (device atomic_int*)&metric_count[coll_id], 1, memory_order_relaxed);
          }

          t_val = test_t;
          last_contributor = contributor;
        }

        if (inside) {
          final_t[pix_id] = t_val;
          n_contrib[pix_id] = last_contributor;
          for (uint ch = 0; ch < min(num_channels, 3u); ++ch) {
            pixel_colors[ch * image_height * image_width + pix_id] = c_accum[ch];
            out_color[ch * image_height * image_width + pix_id] =
                c_accum[ch] + t_val * background[ch];
          }
        }

        shared_last_contributor[local_rank] = last_contributor;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = (block_x * block_y) / 2u; stride > 0u; stride >>= 1u) {
          if (local_rank < stride) {
            shared_last_contributor[local_rank] = max(
                shared_last_contributor[local_rank],
                shared_last_contributor[local_rank + stride]);
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (local_rank == 0u) {
          max_contrib[tile_id] = shared_last_contributor[0];
        }

        (void)radii;
        """

    static let previewBody = """
        threadgroup float3 shared_xy_opacity[BLOCK_X * BLOCK_Y];
        threadgroup float3 shared_conic[BLOCK_X * BLOCK_Y];
        threadgroup float3 shared_colors[BLOCK_X * BLOCK_Y];

        uint2 tid = uint2(thread_position_in_threadgroup.x, thread_position_in_threadgroup.y);
        uint2 gid = uint2(thread_position_in_grid.x, thread_position_in_grid.y);
        uint2 tgp = uint2(threadgroup_position_in_grid.x, threadgroup_position_in_grid.y);

        uint image_width = params[0];
        uint image_height = params[1];
        uint block_x = params[2];
        uint block_y = params[3];
        uint num_channels = params[4];

        uint pix_x = gid.x;
        uint pix_y = gid.y;
        uint tile_x = tgp.x;
        uint tile_y = tgp.y;
        uint tiles_x = (image_width + block_x - 1u) / block_x;
        uint tiles_y = (image_height + block_y - 1u) / block_y;

        if (tile_x >= tiles_x || tile_y >= tiles_y) {
          return;
        }

        uint tile_id = tile_y * tiles_x + tile_x;
        uint2 range = uint2(ranges[2 * tile_id], ranges[2 * tile_id + 1]);
        bool inside = pix_x < image_width && pix_y < image_height;
        uint pix_id = inside ? pix_y * image_width + pix_x : 0u;
        float2 pixf = float2(float(pix_x), float(pix_y));
        float t_val = 1.0f;
        bool done = !inside;
        float c_accum[3] = {0.0f, 0.0f, 0.0f};
        uint local_rank = tid.y * block_x + tid.x;
        uint block_size = block_x * block_y;
        uint num_batches = (range.y - range.x + block_size - 1u) / block_size;

        for (uint batch = 0u; batch < num_batches; ++batch) {
          uint batch_start = range.x + batch * block_size;
          uint load_idx = batch_start + local_rank;
          if (load_idx < range.y) {
            uint coll_id = point_list[load_idx];
            float2 xy = read_packed_float2(means2d, coll_id);
            float4 con_o = read_packed_float4(conic_opacity, coll_id);
            shared_xy_opacity[local_rank] = float3(xy.x, xy.y, con_o.w);
            shared_conic[local_rank] = float3(con_o.x, con_o.y, con_o.z);
            shared_colors[local_rank] = float3(
                colors[coll_id * num_channels + 0u],
                colors[coll_id * num_channels + 1u],
                colors[coll_id * num_channels + 2u]);
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          uint batch_size = min(block_size, range.y - batch_start);
          for (uint local_idx = 0u; !done && local_idx < batch_size; ++local_idx) {
            float3 xy_opacity = shared_xy_opacity[local_idx];
            float3 conic = shared_conic[local_idx];
            float2 d = float2(xy_opacity.x, xy_opacity.y) - pixf;
            float power = -0.5f * (conic.x * d.x * d.x + conic.z * d.y * d.y) -
                          conic.y * d.x * d.y;
            if (power > 0.0f) {
              continue;
            }

            float alpha = min(0.99f, xy_opacity.z * exp(power));
            if (alpha < 1.0f / 255.0f) {
              continue;
            }

            float test_t = t_val * (1.0f - alpha);
            if (test_t < 0.0001f) {
              done = true;
              break;
            }

            float3 color = shared_colors[local_idx];
            c_accum[0] += color.x * alpha * t_val;
            c_accum[1] += color.y * alpha * t_val;
            c_accum[2] += color.z * alpha * t_val;
            t_val = test_t;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (inside) {
          for (uint ch = 0; ch < min(num_channels, 3u); ++ch) {
            out_color[ch * image_height * image_width + pix_id] =
                c_accum[ch] + t_val * background[ch];
          }
        }
        """
}
