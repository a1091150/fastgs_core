import MLX

public struct FastGSBinningParams: Sendable {
    public var multiplier: Float
    public var tileBounds: (x: Int, y: Int, z: Int)

    public init(multiplier: Float, tileBounds: (x: Int, y: Int, z: Int)) {
        self.multiplier = multiplier
        self.tileBounds = tileBounds
    }

    fileprivate var kernelArray: MLXArray {
        MLXArray([
            multiplier,
            Float(tileBounds.x),
            Float(tileBounds.y),
            Float(tileBounds.z),
        ], [4])
    }
}

public struct FastGSBinningInput {
    public var xy: MLXArray
    public var depths: MLXArray
    public var conicOpacity: MLXArray
    public var tilesTouched: MLXArray

    public init(
        xy: MLXArray,
        depths: MLXArray,
        conicOpacity: MLXArray,
        tilesTouched: MLXArray
    ) {
        self.xy = xy
        self.depths = depths
        self.conicOpacity = conicOpacity
        self.tilesTouched = tilesTouched
    }
}

public struct FastGSBinningOutput {
    public var pointOffsets: MLXArray
    public var pointListKeysUnsorted: MLXArray
    public var pointListUnsorted: MLXArray
    public var pointListKeys: MLXArray
    public var pointList: MLXArray
    public var ranges: MLXArray
    public var bucketCount: MLXArray
    public var bucketOffsets: MLXArray
}

public enum FastGSBinning {
    private static let duplicateKernel = MLXFast.metalKernel(
        name: "fastgs_duplicate_with_keys_swift",
        inputNames: [
            "params",
            "points_xy",
            "depths",
            "point_offsets",
            "conic_opacity",
            "tiles_touched",
        ],
        outputNames: [
            "gaussian_keys_unsorted",
            "gaussian_values_unsorted",
        ],
        source: FastGSBinningKernelSource.duplicateBody,
        header: FastGSBinningKernelSource.duplicateHeader
    )

    private static let identifyRangesKernel = MLXFast.metalKernel(
        name: "fastgs_identify_tile_ranges_swift",
        inputNames: ["params", "point_list_keys"],
        outputNames: ["ranges"],
        source: FastGSBinningKernelSource.identifyRangesBody
    )

    private static let bucketCountKernel = MLXFast.metalKernel(
        name: "fastgs_per_tile_bucket_count_swift",
        inputNames: ["params", "ranges"],
        outputNames: ["bucket_count"],
        source: FastGSBinningKernelSource.bucketCountBody
    )

    public static func forward(
        _ input: FastGSBinningInput,
        params: FastGSBinningParams,
        verbose: Bool = false,
        stream: StreamOrDevice = .default
    ) -> FastGSBinningOutput {
        validate(input)

        let count = input.xy.shape[0]
        let pointOffsets = cumsum(input.tilesTouched, stream: stream)
        let numRendered = Int(pointOffsets.asArray(UInt32.self).last ?? 0)
        let numTiles = params.tileBounds.x * params.tileBounds.y

        if numRendered == 0 {
            return FastGSBinningOutput(
                pointOffsets: pointOffsets,
                pointListKeysUnsorted: MLXArray.zeros([0], dtype: .uint64, stream: stream),
                pointListUnsorted: MLXArray.zeros([0], dtype: .uint32, stream: stream),
                pointListKeys: MLXArray.zeros([0], dtype: .uint64, stream: stream),
                pointList: MLXArray.zeros([0], dtype: .uint32, stream: stream),
                ranges: MLXArray.zeros([numTiles, 2], dtype: .uint32, stream: stream),
                bucketCount: MLXArray.zeros([numTiles], dtype: .uint32, stream: stream),
                bucketOffsets: MLXArray.zeros([numTiles], dtype: .uint32, stream: stream)
            )
        }

        let threadGroupSize = max(1, min(256, count))
        let duplicateOutputs = duplicateKernel(
            [
                params.kernelArray,
                input.xy,
                input.depths,
                pointOffsets,
                input.conicOpacity,
                input.tilesTouched,
            ],
            grid: (count, 1, 1),
            threadGroup: (threadGroupSize, 1, 1),
            outputShapes: [
                [numRendered],
                [numRendered],
            ],
            outputDTypes: [
                .uint64,
                .uint32,
            ],
            initValue: 0,
            verbose: verbose,
            stream: stream
        )

        let pointListKeysUnsorted = duplicateOutputs[0]
        let pointListUnsorted = duplicateOutputs[1]
        let sortedIndices = argSort(pointListKeysUnsorted, stream: stream)
        let pointListKeys = take(pointListKeysUnsorted, sortedIndices, stream: stream)
        let pointList = take(pointListUnsorted, sortedIndices, stream: stream)

        let tilePrepParams = MLXArray([UInt32(numRendered), UInt32(numTiles)], [2])
        let tileThreadGroupSize = max(1, min(256, numRendered))
        let ranges = identifyRangesKernel(
            [tilePrepParams, pointListKeys],
            grid: (numRendered, 1, 1),
            threadGroup: (tileThreadGroupSize, 1, 1),
            outputShapes: [[numTiles, 2]],
            outputDTypes: [.uint32],
            initValue: 0,
            verbose: verbose,
            stream: stream
        )[0]

        let bucketThreadGroupSize = max(1, min(256, numTiles))
        let bucketCount = bucketCountKernel(
            [tilePrepParams, ranges],
            grid: (numTiles, 1, 1),
            threadGroup: (bucketThreadGroupSize, 1, 1),
            outputShapes: [[numTiles]],
            outputDTypes: [.uint32],
            initValue: 0,
            verbose: verbose,
            stream: stream
        )[0]
        let bucketOffsets = cumsum(bucketCount, stream: stream)

        return FastGSBinningOutput(
            pointOffsets: pointOffsets,
            pointListKeysUnsorted: pointListKeysUnsorted,
            pointListUnsorted: pointListUnsorted,
            pointListKeys: pointListKeys,
            pointList: pointList,
            ranges: ranges,
            bucketCount: bucketCount,
            bucketOffsets: bucketOffsets
        )
    }

    public static func forward(
        preprocessOutput: FastGSPreprocessOutput,
        params: FastGSBinningParams,
        verbose: Bool = false,
        stream: StreamOrDevice = .default
    ) -> FastGSBinningOutput {
        forward(
            FastGSBinningInput(
                xy: preprocessOutput.xy,
                depths: preprocessOutput.depths,
                conicOpacity: preprocessOutput.conicOpacity,
                tilesTouched: preprocessOutput.tilesTouched
            ),
            params: params,
            verbose: verbose,
            stream: stream
        )
    }

    private static func validate(_ input: FastGSBinningInput) {
        let count = input.xy.shape[0]
        precondition(input.xy.shape == [count, 2], "xy must have shape [N, 2].")
        precondition(input.depths.shape == [count], "depths must have shape [N].")
        precondition(input.conicOpacity.shape == [count, 4], "conicOpacity must have shape [N, 4].")
        precondition(input.tilesTouched.shape == [count], "tilesTouched must have shape [N].")
        precondition(input.xy.dtype == .float32, "FastGSBinning currently expects float32 xy.")
        precondition(input.depths.dtype == .float32, "FastGSBinning currently expects float32 depths.")
        precondition(input.conicOpacity.dtype == .float32, "FastGSBinning currently expects float32 conicOpacity.")
        precondition(input.tilesTouched.dtype == .uint32, "FastGSBinning expects uint32 tilesTouched.")
    }
}

private enum FastGSBinningKernelSource {
    static let duplicateHeader = """
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

        inline void process_tiles(float4 con_o,
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
                                  bool is_y,
                                  uint idx,
                                  uint off,
                                  float depth,
                                  device ulong* gaussian_keys_unsorted,
                                  device uint* gaussian_values_unsorted) {
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

            int min_tile_v = max(rect_min.y, min(rect_max.y, int(ellipse_min / block_v)));
            int max_tile_v = min(rect_max.y, max(rect_min.y, int(ellipse_max / block_v + 1)));

            for (int v = min_tile_v; v < max_tile_v; ++v) {
              ulong key = is_y ? ulong(u * int(grid.x) + v) : ulong(v * int(grid.x) + u);
              key <<= 32;
              key |= ulong(as_type<uint>(depth));
              gaussian_keys_unsorted[off] = key;
              gaussian_values_unsorted[off] = idx;
              off++;
            }

            intersect_min_line = intersect_max_line;
            min_line = max_line;
          }
        }

        inline void duplicate_with_keys(float2 p,
                                        float4 con_o,
                                        uint3 grid,
                                        float mult,
                                        uint idx,
                                        uint off,
                                        float depth,
                                        device ulong* gaussian_keys_unsorted,
                                        device uint* gaussian_values_unsorted) {
          float disc = con_o.y * con_o.y - con_o.x * con_o.z;
          if (con_o.x <= 0.0f || con_o.z <= 0.0f || disc >= 0.0f) {
            return;
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
              max(0, min(int(grid.x), int(bbox_min.x / BLOCK_X))),
              max(0, min(int(grid.y), int(bbox_min.y / BLOCK_Y))));
          int2 rect_max = int2(
              max(0, min(int(grid.x), int(bbox_max.x / BLOCK_X + 1))),
              max(0, min(int(grid.y), int(bbox_max.y / BLOCK_Y + 1))));

          int y_span = rect_max.y - rect_min.y;
          int x_span = rect_max.x - rect_min.x;
          if (y_span * x_span == 0) {
            return;
          }

          bool is_y = y_span < x_span;
          process_tiles(
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
              is_y,
              idx,
              off,
              depth,
              gaussian_keys_unsorted,
              gaussian_values_unsorted);
        }
        """

    static let duplicateBody = """
        uint idx = thread_position_in_grid.x;
        if (idx >= uint(points_xy_shape[0])) {
          return;
        }

        if (tiles_touched[idx] == 0u) {
          return;
        }

        uint off = (idx == 0u) ? 0u : point_offsets[idx - 1];
        duplicate_with_keys(
            read_packed_float2(points_xy, idx),
            read_packed_float4(conic_opacity, idx),
            uint3(uint(params[1]), uint(params[2]), uint(params[3])),
            params[0],
            idx,
            off,
            depths[idx],
            gaussian_keys_unsorted,
            gaussian_values_unsorted);
        """

    static let identifyRangesBody = """
        uint gid = thread_position_in_grid.x;
        uint num_rendered = params[0];
        if (gid >= num_rendered) {
          return;
        }

        ulong key = point_list_keys[gid];
        uint currtile = uint(key >> 32);

        if (gid == 0) {
          ranges[2 * currtile] = 0u;
        } else {
          uint prevtile = uint(point_list_keys[gid - 1] >> 32);
          if (currtile != prevtile) {
            ranges[2 * prevtile + 1] = gid;
            ranges[2 * currtile] = gid;
          }
        }
        if (gid == num_rendered - 1) {
          ranges[2 * currtile + 1] = num_rendered;
        }
        """

    static let bucketCountBody = """
        uint gid = thread_position_in_grid.x;
        uint num_tiles = params[1];
        if (gid >= num_tiles) {
          return;
        }

        uint start = ranges[2 * gid];
        uint end = ranges[2 * gid + 1];
        int num_splats = int(end - start);
        int num_buckets = (num_splats + 31) / 32;
        bucket_count[gid] = uint(num_buckets);
        """
}
