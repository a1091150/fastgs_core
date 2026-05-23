import Foundation
import MLX

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
                opacities: arrays[3],
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
        let preprocessInput = context.preprocessInput(parameters: parameters)
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
}
