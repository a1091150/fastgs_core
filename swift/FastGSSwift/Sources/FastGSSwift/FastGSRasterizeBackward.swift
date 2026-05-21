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
    public var metricCount: MLXArray

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
            metricCount,
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
            "dL_dbackground",
        ],
        source: FastGSRasterizeBackwardKernelSource.body
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
            metricCount: MLXArray.zeros(input.metricCount.shape, dtype: input.metricCount.dtype, stream: stream)
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
                input.background.shape,
            ],
            outputDTypes: [
                input.means2D.dtype,
                input.colors.dtype,
                input.conicOpacity.dtype,
                input.background.dtype,
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
            background: outputs[3],
            radii: emptyOutput.radii,
            metricMap: emptyOutput.metricMap,
            metricCount: emptyOutput.metricCount
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
    static let body = """
        uint tid = thread_position_in_grid.x;
        uint bucket = tid / 32u;
        if (bucket >= params[6]) {
          return;
        }

        (void)ranges;
        (void)point_list;
        (void)per_tile_bucket_offset;
        (void)means2d;
        (void)colors;
        (void)conic_opacity;
        (void)background;
        (void)dL_dout_color;
        (void)bucket_to_tile;
        (void)sampled_t;
        (void)sampled_ar;
        (void)final_t;
        (void)n_contrib;
        (void)max_contrib;
        (void)pixel_colors;
        (void)dL_dmeans2d;
        (void)dL_dcolors;
        (void)dL_dconic_opacity;
        (void)dL_dbackground;
        """
}
