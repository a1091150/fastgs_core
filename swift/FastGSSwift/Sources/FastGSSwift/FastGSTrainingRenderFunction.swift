import Foundation
import MLX

public struct FastGSTrainingSmokeResult {
    public var loss: MLXArray
    public var gradients: [MLXArray]

    public init(loss: MLXArray, gradients: [MLXArray]) {
        self.loss = loss
        self.gradients = gradients
    }
}

public enum FastGSTrainingBackwardMode {
    case zero
    case rasterizeOnly
}

public enum FastGSTrainingRenderFunction {
    public static func renderer(
        scene: FastGSRecordedForwardScene,
        backwardMode: FastGSTrainingBackwardMode = .zero,
        stream: StreamOrDevice = .default
    ) -> ([MLXArray]) -> [MLXArray] {
        CustomFunction {
            Forward { primals in
                let parameters = trainableParameters(from: primals)
                do {
                    return [try scene.render(parameters: parameters).outColor]
                } catch {
                    preconditionFailure("FastGSTrainingRenderFunction forward failed: \(error)")
                }
            }
            VJP { primals, cotangents in
                if backwardMode == .rasterizeOnly {
                    runRasterizeBackwardSmoke(scene: scene, primals: primals, cotangents: cotangents, stream: stream)
                }
                return primals.map { MLXArray.zeros($0.shape, dtype: $0.dtype, stream: stream) }
            }
        }
    }

    public static func zeroVJPRenderer(
        scene: FastGSRecordedForwardScene,
        stream: StreamOrDevice = .default
    ) -> ([MLXArray]) -> [MLXArray] {
        renderer(scene: scene, backwardMode: .zero, stream: stream)
    }

    public static func mseLoss(
        renderer: @escaping ([MLXArray]) -> [MLXArray],
        target: MLXArray,
        stream: StreamOrDevice = .default
    ) -> ([MLXArray]) -> [MLXArray] {
        { primals in
            let rendered = renderer(primals)[0]
            let diff = rendered - target
            return [mean(square(diff), stream: stream)]
        }
    }

    public static func valueAndZeroGrad(
        scene: FastGSRecordedForwardScene,
        parameters: FastGSTrainableParameters,
        target: MLXArray,
        backwardMode: FastGSTrainingBackwardMode = .zero,
        stream: StreamOrDevice = .default
    ) -> FastGSTrainingSmokeResult {
        let primals = parameters.arrays
        let renderer = renderer(scene: scene, backwardMode: backwardMode, stream: stream)
        let lossFunction = mseLoss(renderer: renderer, target: target, stream: stream)
        let valueAndGradient = valueAndGrad(
            lossFunction,
            argumentNumbers: Array(0..<primals.count)
        )
        let (values, gradients) = valueAndGradient(primals)
        return FastGSTrainingSmokeResult(loss: values[0], gradients: gradients)
    }

    private static func trainableParameters(from arrays: [MLXArray]) -> FastGSTrainableParameters {
        precondition(arrays.count == 6, "Expected means3D, dc, sh, opacities, scales, and rotations.")
        return FastGSTrainableParameters(
            means3D: arrays[0],
            dc: arrays[1],
            sh: arrays[2],
            opacities: arrays[3],
            scales: arrays[4],
            rotations: arrays[5]
        )
    }

    private static func runRasterizeBackwardSmoke(
        scene: FastGSRecordedForwardScene,
        primals: [MLXArray],
        cotangents: [MLXArray],
        stream: StreamOrDevice
    ) {
        precondition(cotangents.count == 1, "Render VJP expects one outColor cotangent.")
        let parameters = trainableParameters(from: primals)
        do {
            let stages = try scene.renderStages(parameters: parameters)
            let rasterizeOutput = stages.rasterize
            let rasterizeCotangents = FastGSRasterizeCotangents(
                bucketToTile: MLXArray.zeros(rasterizeOutput.bucketToTile.shape, dtype: rasterizeOutput.bucketToTile.dtype, stream: stream),
                sampledT: MLXArray.zeros(rasterizeOutput.sampledT.shape, dtype: rasterizeOutput.sampledT.dtype, stream: stream),
                sampledAr: MLXArray.zeros(rasterizeOutput.sampledAr.shape, dtype: rasterizeOutput.sampledAr.dtype, stream: stream),
                finalT: MLXArray.zeros(rasterizeOutput.finalT.shape, dtype: rasterizeOutput.finalT.dtype, stream: stream),
                nContrib: MLXArray.zeros(rasterizeOutput.nContrib.shape, dtype: rasterizeOutput.nContrib.dtype, stream: stream),
                maxContrib: MLXArray.zeros(rasterizeOutput.maxContrib.shape, dtype: rasterizeOutput.maxContrib.dtype, stream: stream),
                pixelColors: MLXArray.zeros(rasterizeOutput.pixelColors.shape, dtype: rasterizeOutput.pixelColors.dtype, stream: stream),
                outColor: cotangents[0],
                metricCount: MLXArray.zeros(rasterizeOutput.metricCount.shape, dtype: rasterizeOutput.metricCount.dtype, stream: stream)
            )
            let params = FastGSRasterizeParams(
                imageWidth: scene.manifest.width,
                imageHeight: scene.manifest.height,
                numTiles: ((scene.manifest.width + 15) / 16) * ((scene.manifest.height + 15) / 16)
            )
            let gradients = FastGSRasterizeBackward.forward(
                preprocessOutput: stages.preprocess,
                binningOutput: stages.binning,
                rasterizeOutput: rasterizeOutput,
                cotangents: rasterizeCotangents,
                background: MLXArray(scene.manifest.background.map(Float.init), [3]),
                params: params,
                stream: stream
            )
            gradients.arrays.forEach { $0.eval() }
        } catch {
            preconditionFailure("FastGSTrainingRenderFunction rasterize-only VJP failed: \(error)")
        }
    }
}
