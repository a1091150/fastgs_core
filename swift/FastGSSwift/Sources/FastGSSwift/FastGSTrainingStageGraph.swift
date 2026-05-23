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
        let trainingResult = valueAndGrad(
            scene: scene,
            parameters: parameters,
            target: target,
            stream: stream
        )
        let stats = densificationStats(
            scene: scene,
            parameters: parameters,
            target: target,
            stream: stream
        )
        return FastGSTrainingStageGraphResult(
            loss: trainingResult.loss,
            gradients: trainingResult.gradients,
            densificationStats: stats
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
        let preprocessInput = context.preprocessInput(parameters: parameters, viewspacePoints: viewspacePoints, stream: stream)
        let preprocess = FastGSPreprocessCustomFunction.call(
            preprocessInput,
            params: context.preprocessParams,
            stream: stream
        )
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
            stream: stream
        ).outColor
    }

    private static func densificationStats(
        scene: FastGSRecordedForwardScene,
        parameters: FastGSTrainableParameters,
        target: MLXArray,
        stream: StreamOrDevice
    ) -> FastGSTrainingDensificationStats {
        let context = FastGSTrainingRenderContext(scene: scene, stream: stream)
        let viewspacePoints = MLXArray.zeros([parameters.gaussianCount, 4], dtype: .float32, stream: stream)
        let preprocessInput = context.preprocessInput(
            parameters: parameters,
            viewspacePoints: viewspacePoints,
            stream: stream
        )
        let preprocess = FastGSPreprocess.forward(
            preprocessInput,
            params: context.preprocessParams,
            stream: stream
        )
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
        let rasterizeOutput = FastGSRasterize.forward(
            rasterizeInput,
            params: context.rasterizeParams,
            stream: stream
        )
        let outColorCotangentScale = Float(2) / Float(rasterizeOutput.outColor.shape.reduce(1, *))
        let rasterizeCotangents = FastGSRasterizeCotangents(
            bucketToTile: MLXArray.zeros(rasterizeOutput.bucketToTile.shape, dtype: rasterizeOutput.bucketToTile.dtype, stream: stream),
            sampledT: MLXArray.zeros(rasterizeOutput.sampledT.shape, dtype: rasterizeOutput.sampledT.dtype, stream: stream),
            sampledAr: MLXArray.zeros(rasterizeOutput.sampledAr.shape, dtype: rasterizeOutput.sampledAr.dtype, stream: stream),
            finalT: MLXArray.zeros(rasterizeOutput.finalT.shape, dtype: rasterizeOutput.finalT.dtype, stream: stream),
            nContrib: MLXArray.zeros(rasterizeOutput.nContrib.shape, dtype: rasterizeOutput.nContrib.dtype, stream: stream),
            maxContrib: MLXArray.zeros(rasterizeOutput.maxContrib.shape, dtype: rasterizeOutput.maxContrib.dtype, stream: stream),
            pixelColors: MLXArray.zeros(rasterizeOutput.pixelColors.shape, dtype: rasterizeOutput.pixelColors.dtype, stream: stream),
            outColor: outColorCotangentScale * (rasterizeOutput.outColor - target),
            metricCount: MLXArray.zeros(rasterizeOutput.metricCount.shape, dtype: rasterizeOutput.metricCount.dtype, stream: stream)
        )
        let rasterizeBackward = FastGSRasterizeBackward.forward(
            input: rasterizeInput,
            cotangents: rasterizeCotangents,
            forwardOutput: rasterizeOutput,
            params: context.rasterizeParams,
            stream: stream
        )
        return FastGSTrainingDensificationStats(
            radii: preprocess.radii,
            viewspaceGradients: rasterizeBackward.viewspacePoints
        )
    }
}
