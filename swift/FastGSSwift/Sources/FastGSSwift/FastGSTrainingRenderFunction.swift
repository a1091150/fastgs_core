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

public enum FastGSTrainingRenderFunction {
    public static func zeroVJPRenderer(
        scene: FastGSRecordedForwardScene,
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
            VJP { primals, _ in
                primals.map { MLXArray.zeros($0.shape, dtype: $0.dtype, stream: stream) }
            }
        }
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
        stream: StreamOrDevice = .default
    ) -> FastGSTrainingSmokeResult {
        let primals = parameters.arrays
        let renderer = zeroVJPRenderer(scene: scene, stream: stream)
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
}
