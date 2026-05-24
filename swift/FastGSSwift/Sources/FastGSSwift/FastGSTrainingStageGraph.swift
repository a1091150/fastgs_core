import Foundation
import MLX

public struct FastGSTrainingDensificationStats {
    public var radii: MLXArray
    public var viewspaceGradients: MLXArray

    public init(radii: MLXArray, viewspaceGradients: MLXArray) {
        self.radii = radii
        self.viewspaceGradients = viewspaceGradients
    }
}

public struct FastGSTrainingStageGraphResult {
    public var loss: MLXArray
    public var gradients: [MLXArray]
    public var densificationStats: FastGSTrainingDensificationStats

    public init(
        loss: MLXArray,
        gradients: [MLXArray],
        densificationStats: FastGSTrainingDensificationStats
    ) {
        self.loss = loss
        self.gradients = gradients
        self.densificationStats = densificationStats
    }
}

public enum FastGSTrainingStageGraph {
    public static func valueAndGrad(
        scene: FastGSRecordedForwardScene,
        parameters: FastGSTrainableParameters,
        target: MLXArray,
        stream: StreamOrDevice = .default
    ) -> FastGSTrainingSmokeResult {
        let primals = parameters.arrays
        let context = FastGSTrainingRenderContext(scene: scene, stream: stream)
        let lossFunction: ([MLXArray]) -> [MLXArray] = { arrays in
            let parameters = FastGSTrainableParameters(
                means3D: arrays[0],
                dc: arrays[1],
                sh: arrays[2],
                opacityLogits: arrays[3],
                scales: arrays[4],
                rotations: arrays[5]
            )
            let outColor = render(context: context, parameters: parameters, stream: stream)
            return [mean(square(outColor - target), stream: stream)]
        }
        let valueAndGradient = MLX.valueAndGrad(
            lossFunction,
            argumentNumbers: Array(0..<primals.count)
        )
        let (values, gradients) = valueAndGradient(primals)
        return FastGSTrainingSmokeResult(loss: values[0], gradients: gradients)
    }

    public static func valueAndGradWithDensificationStats(
        scene: FastGSRecordedForwardScene,
        parameters: FastGSTrainableParameters,
        target: MLXArray,
        stream: StreamOrDevice = .default
    ) -> FastGSTrainingStageGraphResult {
        let primals = parameters.arrays
        let context = FastGSTrainingRenderContext(scene: scene, stream: stream)
        let statsCapture = FastGSTrainingDensificationStatsCapture()
        let lossFunction: ([MLXArray]) -> [MLXArray] = { arrays in
            let parameters = FastGSTrainableParameters(
                means3D: arrays[0],
                dc: arrays[1],
                sh: arrays[2],
                opacityLogits: arrays[3],
                scales: arrays[4],
                rotations: arrays[5]
            )
            let outColor = render(
                context: context,
                parameters: parameters,
                statsCapture: statsCapture,
                stream: stream
            )
            return [mean(square(outColor - target), stream: stream)]
        }
        let valueAndGradient = MLX.valueAndGrad(
            lossFunction,
            argumentNumbers: Array(0..<primals.count)
        )
        let (values, gradients) = valueAndGradient(primals)
        return FastGSTrainingStageGraphResult(
            loss: values[0],
            gradients: gradients,
            densificationStats: statsCapture.stats(stream: stream)
        )
    }

    public static func render(
        scene: FastGSRecordedForwardScene,
        parameters: FastGSTrainableParameters,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        render(context: FastGSTrainingRenderContext(scene: scene, stream: stream), parameters: parameters, stream: stream)
    }

    public static func renderDefault(
        scene: FastGSRecordedForwardScene,
        parameters: FastGSTrainableParameters
    ) -> MLXArray {
        render(scene: scene, parameters: parameters, stream: .default)
    }

    public static func render(
        context: FastGSTrainingRenderContext,
        parameters: FastGSTrainableParameters,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        render(
            context: context,
            parameters: parameters,
            viewspacePoints: MLXArray.zeros([parameters.gaussianCount, 4], dtype: .float32, stream: stream),
            stream: stream
        )
    }

    public static func render(
        context: FastGSTrainingRenderContext,
        parameters: FastGSTrainableParameters,
        viewspacePoints: MLXArray,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        render(
            context: context,
            parameters: parameters,
            viewspacePoints: viewspacePoints,
            statsCapture: nil,
            stream: stream
        )
    }

    private static func render(
        context: FastGSTrainingRenderContext,
        parameters: FastGSTrainableParameters,
        viewspacePoints: MLXArray? = nil,
        statsCapture: FastGSTrainingDensificationStatsCapture?,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        let resolvedViewspacePoints = viewspacePoints ?? MLXArray.zeros([parameters.gaussianCount, 4], dtype: .float32, stream: stream)
        let preprocessInput = context.preprocessInput(parameters: parameters, viewspacePoints: resolvedViewspacePoints, stream: stream)
        let preprocess = FastGSPreprocessCustomFunction.call(
            preprocessInput,
            params: context.preprocessParams,
            stream: stream
        )
        statsCapture?.store(radii: preprocess.radii)
        let stoppedPreprocess = FastGSPreprocessOutput(
            radii: stopGradient(preprocess.radii, stream: stream),
            xy: preprocess.xy,
            depths: stopGradient(preprocess.depths, stream: stream),
            cov3D: stopGradient(preprocess.cov3D, stream: stream),
            rgb: preprocess.rgb,
            conicOpacity: preprocess.conicOpacity,
            tilesTouched: stopGradient(preprocess.tilesTouched, stream: stream),
            clamped: stopGradient(preprocess.clamped, stream: stream),
            viewspacePoints: preprocess.viewspacePoints
        )
        let binning = FastGSBinning.forward(
            preprocessOutput: stoppedPreprocess,
            params: FastGSBinningParams(multiplier: 1, tileBounds: context.tileBounds),
            stream: stream
        )
        let rasterizeInput = context.rasterizeInput(
            preprocess: stoppedPreprocess,
            binning: binning,
            stream: stream
        )
        return FastGSRasterizeCustomFunction.call(
            rasterizeInput,
            params: context.rasterizeParams,
            backwardCapture: statsCapture?.rasterizeBackwardCapture,
            stream: stream
        ).outColor
    }
}

private final class FastGSTrainingDensificationStatsCapture: @unchecked Sendable {
    let rasterizeBackwardCapture = FastGSRasterizeBackwardCapture()
    private let lock = NSLock()
    private var capturedRadii: MLXArray?

    func store(radii: MLXArray) {
        lock.withLock {
            capturedRadii = radii
        }
    }

    func stats(stream: StreamOrDevice) -> FastGSTrainingDensificationStats {
        let radii = lock.withLock { capturedRadii }
        guard let radii else {
            preconditionFailure("FastGSTrainingStageGraph missing captured preprocess radii.")
        }
        guard let viewspaceGradients = rasterizeBackwardCapture.viewspacePoints else {
            preconditionFailure("FastGSTrainingStageGraph missing captured rasterize viewspace gradients.")
        }
        return FastGSTrainingDensificationStats(
            radii: radii,
            viewspaceGradients: viewspaceGradients
        )
    }
}
