import Foundation
import MLX

public enum FastGSRasterizeCustomFunction {
    public static func call(
        _ input: FastGSRasterizeInput,
        params: FastGSRasterizeParams,
        stream: StreamOrDevice = .default
    ) -> FastGSRasterizeOutput {
        let function = make(params: params, stream: stream)
        return output(from: function(arrays(from: input)))
    }

    public static func make(
        params: FastGSRasterizeParams,
        stream: StreamOrDevice = .default
    ) -> ([MLXArray]) -> [MLXArray] {
        let cache = FastGSRasterizeCustomFunctionCache()
        return CustomFunction {
            Forward { primals in
                let output = FastGSRasterize.forward(input(from: primals), params: params, stream: stream)
                let arrays = arrays(from: output)
                cache.store(output: output)
                return arrays
            }
            VJP { primals, cotangents in
                let input = input(from: primals)
                let output = cache.output ?? output(from: cotangents.map {
                    MLXArray.zeros($0.shape, dtype: $0.dtype, stream: stream)
                })
                let gradients = FastGSRasterizeBackward.forward(
                    input: input,
                    cotangents: cotangentsFromRasterizeOutputs(cotangents),
                    forwardOutput: output,
                    params: params,
                    stream: stream
                )
                return [
                    MLXArray.zeros(input.ranges.shape, dtype: input.ranges.dtype, stream: stream),
                    MLXArray.zeros(input.pointList.shape, dtype: input.pointList.dtype, stream: stream),
                    MLXArray.zeros(input.bucketOffsets.shape, dtype: input.bucketOffsets.dtype, stream: stream),
                    gradients.means2D,
                    gradients.colors,
                    gradients.conicOpacity,
                    MLXArray.zeros(input.background.shape, dtype: input.background.dtype, stream: stream),
                    MLXArray.zeros(input.radii.shape, dtype: input.radii.dtype, stream: stream),
                    MLXArray.zeros(input.metricMap.shape, dtype: input.metricMap.dtype, stream: stream),
                    MLXArray.zeros(input.metricCount.shape, dtype: input.metricCount.dtype, stream: stream),
                ]
            }
        }
    }

    public static func arrays(from input: FastGSRasterizeInput) -> [MLXArray] {
        [
            input.ranges,
            input.pointList,
            input.bucketOffsets,
            input.means2D,
            input.colors,
            input.conicOpacity,
            input.background,
            input.radii,
            input.metricMap,
            input.metricCount,
        ]
    }

    public static func input(from arrays: [MLXArray]) -> FastGSRasterizeInput {
        precondition(arrays.count == 10, "FastGSRasterizeCustomFunction expects 10 primals.")
        return FastGSRasterizeInput(
            ranges: arrays[0],
            pointList: arrays[1],
            bucketOffsets: arrays[2],
            means2D: arrays[3],
            colors: arrays[4],
            conicOpacity: arrays[5],
            background: arrays[6],
            radii: arrays[7],
            metricMap: arrays[8],
            metricCount: arrays[9]
        )
    }

    public static func arrays(from output: FastGSRasterizeOutput) -> [MLXArray] {
        [
            output.bucketToTile,
            output.sampledT,
            output.sampledAr,
            output.finalT,
            output.nContrib,
            output.maxContrib,
            output.pixelColors,
            output.outColor,
            output.metricCount,
        ]
    }

    public static func output(from arrays: [MLXArray]) -> FastGSRasterizeOutput {
        precondition(arrays.count == 9, "FastGSRasterizeCustomFunction expects 9 outputs.")
        return FastGSRasterizeOutput(
            bucketToTile: arrays[0],
            sampledT: arrays[1],
            sampledAr: arrays[2],
            finalT: arrays[3],
            nContrib: arrays[4],
            maxContrib: arrays[5],
            pixelColors: arrays[6],
            outColor: arrays[7],
            metricCount: arrays[8]
        )
    }

    private static func cotangentsFromRasterizeOutputs(_ arrays: [MLXArray]) -> FastGSRasterizeCotangents {
        precondition(arrays.count == 9, "FastGSRasterizeCustomFunction expects 9 output cotangents.")
        return FastGSRasterizeCotangents(
            bucketToTile: arrays[0],
            sampledT: arrays[1],
            sampledAr: arrays[2],
            finalT: arrays[3],
            nContrib: arrays[4],
            maxContrib: arrays[5],
            pixelColors: arrays[6],
            outColor: arrays[7],
            metricCount: arrays[8]
        )
    }
}

private final class FastGSRasterizeCustomFunctionCache: @unchecked Sendable {
    private let lock = NSLock()
    private var cachedOutput: FastGSRasterizeOutput?

    var output: FastGSRasterizeOutput? {
        lock.withLock { cachedOutput }
    }

    func store(output: FastGSRasterizeOutput) {
        lock.withLock {
            cachedOutput = output
        }
    }
}
