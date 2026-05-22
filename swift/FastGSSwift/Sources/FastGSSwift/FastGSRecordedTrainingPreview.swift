import Foundation
import MLX

public struct FastGSRecordedTrainingPreviewConfig {
    public var totalSteps: Int
    public var cacheLimitBytes: Int
    public var learningRates: FastGSAdamLearningRates

    public init(
        totalSteps: Int = 200,
        cacheLimitBytes: Int = 4 * 1024 * 1024 * 1024,
        learningRates: FastGSAdamLearningRates = FastGSAdamLearningRates(
            means3D: 5e-5,
            dc: 5e-4,
            sh: 5e-4,
            opacities: 5e-4,
            scales: 5e-5,
            rotations: 5e-5
        )
    ) {
        self.totalSteps = totalSteps
        self.cacheLimitBytes = cacheLimitBytes
        self.learningRates = learningRates
    }
}

public struct FastGSRecordedTrainingPreviewResult {
    public var targetRGBA: [UInt8]
    public var renderRGBA: [UInt8]
    public var width: Int
    public var height: Int
    public var pointCount: Int

    public init(
        targetRGBA: [UInt8],
        renderRGBA: [UInt8],
        width: Int,
        height: Int,
        pointCount: Int
    ) {
        self.targetRGBA = targetRGBA
        self.renderRGBA = renderRGBA
        self.width = width
        self.height = height
        self.pointCount = pointCount
    }
}

public enum FastGSRecordedTrainingPreview {
    public static func run(
        manifestURL: URL,
        config: FastGSRecordedTrainingPreviewConfig = FastGSRecordedTrainingPreviewConfig(),
        progress: ((Int) -> Void)? = nil
    ) throws -> FastGSRecordedTrainingPreviewResult {
        Memory.cacheLimit = config.cacheLimitBytes

        let scene = try FastGSRecordedForwardScene(manifestURL: manifestURL)
        let target = try scene.targetOutColor()
        var parameters = try scene.initialTrainableParameters()
        var optimizer = FastGSAdamOptimizer(learningRates: config.learningRates)

        for step in 1...config.totalSteps {
            let result = FastGSTrainingStageGraph.valueAndGrad(
                scene: scene,
                parameters: parameters,
                target: target
            )
            parameters = optimizer.update(
                parameters: parameters,
                gradients: trainableGradients(from: result.gradients)
            )
            eval(parameters: parameters, optimizer: optimizer)
            progress?(step)
        }

        let render = FastGSTrainingStageGraph.render(scene: scene, parameters: parameters)
        return FastGSRecordedTrainingPreviewResult(
            targetRGBA: FastGSImageExport.rgbaBytes(
                outColor: target,
                width: scene.manifest.width,
                height: scene.manifest.height
            ),
            renderRGBA: FastGSImageExport.rgbaBytes(
                outColor: render,
                width: scene.manifest.width,
                height: scene.manifest.height
            ),
            width: scene.manifest.width,
            height: scene.manifest.height,
            pointCount: scene.manifest.pointCount
        )
    }

    private static func trainableGradients(from gradients: [MLXArray]) -> FastGSTrainableGradients {
        precondition(gradients.count == 6, "recorded training preview expects six trainable gradients")
        return FastGSTrainableGradients(
            means3D: gradients[0],
            dc: gradients[1],
            sh: gradients[2],
            opacities: gradients[3],
            scales: gradients[4],
            rotations: gradients[5]
        )
    }

    private static func eval(parameters: FastGSTrainableParameters, optimizer: FastGSAdamOptimizer) {
        for array in parameters.arrays {
            array.eval()
        }
        for array in optimizer.stateArrays() {
            array.eval()
        }
    }
}
