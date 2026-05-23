import Foundation
import MLX

public struct FastGSGaussianScoringResult {
    public var importanceScores: [Float]?
    public var pruningScores: [Float]
    public var metricCounts: [Float]?
    public var metricScore: [Float]
    public var sampledFrameCount: Int

    public init(
        importanceScores: [Float]?,
        pruningScores: [Float],
        metricCounts: [Float]?,
        metricScore: [Float],
        sampledFrameCount: Int
    ) {
        self.importanceScores = importanceScores
        self.pruningScores = pruningScores
        self.metricCounts = metricCounts
        self.metricScore = metricScore
        self.sampledFrameCount = sampledFrameCount
    }
}

public enum FastGSGaussianScoring {
    public static func compute(
        scenes: [FastGSRecordedForwardScene],
        parameters: FastGSTrainableParameters,
        sceneIndices: [Int],
        targets: [MLXArray]? = nil,
        lossThreshold: Float,
        densify: Bool,
        stream: StreamOrDevice = .default
    ) throws -> FastGSGaussianScoringResult {
        parameters.validateTopology()
        precondition(lossThreshold >= 0, "lossThreshold must be non-negative")
        precondition(sceneIndices.allSatisfy { $0 >= 0 && $0 < scenes.count }, "scene index out of bounds")
        if let targets {
            precondition(targets.count == scenes.count, "target count mismatch")
        }

        let count = parameters.gaussianCount
        guard !sceneIndices.isEmpty else {
            let zeros = [Float](repeating: 0, count: count)
            return FastGSGaussianScoringResult(
                importanceScores: densify ? zeros : nil,
                pruningScores: zeros,
                metricCounts: densify ? zeros : nil,
                metricScore: zeros,
                sampledFrameCount: 0
            )
        }

        var fullMetricCounts = densify ? [Float](repeating: 0, count: count) : nil
        var fullMetricScore = [Float](repeating: 0, count: count)

        for sceneIndex in sceneIndices {
            let scene = scenes[sceneIndex]
            let target: MLXArray
            if let targets {
                target = targets[sceneIndex]
            } else {
                target = try scene.targetOutColor()
            }
            precondition(target.shape == [3, scene.manifest.width * scene.manifest.height], "target shape mismatch")

            let stages = try scene.renderStages(parameters: parameters)
            let predValues = stages.rasterize.outColor.asArray(Float.self)
            let targetValues = target.asArray(Float.self)
            let photometricLoss = meanAbsoluteDifference(predValues, targetValues)
            let metricMap = metricMapFromL1(
                predValues: predValues,
                targetValues: targetValues,
                pixelCount: scene.manifest.width * scene.manifest.height,
                lossThreshold: lossThreshold
            )
            let metricCount = metricCountForScene(
                scene: scene,
                stages: stages,
                parameters: parameters,
                metricMap: metricMap,
                stream: stream
            )
            precondition(metricCount.count == count, "metric count mismatch")

            if densify {
                for index in 0..<count {
                    fullMetricCounts?[index] += metricCount[index]
                }
            }
            for index in 0..<count {
                fullMetricScore[index] += photometricLoss * metricCount[index]
            }
        }

        let pruningScores = normalizedScores(fullMetricScore)
        let importanceScores = fullMetricCounts.map { counts in
            counts.map { Foundation.floor($0 / Float(max(sceneIndices.count, 1))) }
        }
        return FastGSGaussianScoringResult(
            importanceScores: importanceScores,
            pruningScores: pruningScores,
            metricCounts: fullMetricCounts,
            metricScore: fullMetricScore,
            sampledFrameCount: sceneIndices.count
        )
    }

    public static func evenlySpacedSceneIndices(sceneCount: Int, sampleCount: Int) -> [Int] {
        precondition(sceneCount >= 0, "sceneCount must be non-negative")
        precondition(sampleCount >= 0, "sampleCount must be non-negative")
        guard sceneCount > 0 && sampleCount > 0 else { return [] }
        let count = min(sceneCount, sampleCount)
        return (0..<count).map { index in
            Int((Double(index) * Double(sceneCount) / Double(count)).rounded(.down))
        }
    }
}

private func metricCountForScene(
    scene: FastGSRecordedForwardScene,
    stages: FastGSRecordedForwardStages,
    parameters: FastGSTrainableParameters,
    metricMap: [Int32],
    stream: StreamOrDevice
) -> [Float] {
    let tileBounds = (
        x: (scene.manifest.width + 15) / 16,
        y: (scene.manifest.height + 15) / 16,
        z: 1
    )
    let output = FastGSRasterize.forward(
        preprocessOutput: stages.preprocess,
        binningOutput: stages.binning,
        background: MLXArray(scene.manifest.background.map(Float.init), [3]),
        params: FastGSRasterizeParams(
            imageWidth: scene.manifest.width,
            imageHeight: scene.manifest.height,
            numTiles: tileBounds.x * tileBounds.y,
            getMetricCount: true
        ),
        metricMap: MLXArray(metricMap, [scene.manifest.width * scene.manifest.height]),
        metricCount: MLXArray.zeros([parameters.gaussianCount], dtype: .int32, stream: stream),
        stream: stream
    )
    return output.metricCount.asArray(Int32.self).map(Float.init)
}

private func metricMapFromL1(
    predValues: [Float],
    targetValues: [Float],
    pixelCount: Int,
    lossThreshold: Float
) -> [Int32] {
    precondition(predValues.count == targetValues.count, "prediction/target count mismatch")
    precondition(predValues.count == pixelCount * 3, "outColor count mismatch")
    var l1Map = [Float](repeating: 0, count: pixelCount)
    for pixel in 0..<pixelCount {
        let r = abs(predValues[pixel] - targetValues[pixel])
        let g = abs(predValues[pixelCount + pixel] - targetValues[pixelCount + pixel])
        let b = abs(predValues[2 * pixelCount + pixel] - targetValues[2 * pixelCount + pixel])
        l1Map[pixel] = (r + g + b) / 3
    }
    let minValue = l1Map.min() ?? 0
    let maxValue = l1Map.max() ?? 0
    let denominator = max(maxValue - minValue, 1.0e-6)
    return l1Map.map { (($0 - minValue) / denominator) > lossThreshold ? 1 : 0 }
}

private func meanAbsoluteDifference(_ lhs: [Float], _ rhs: [Float]) -> Float {
    precondition(lhs.count == rhs.count, "mean absolute difference count mismatch")
    guard !lhs.isEmpty else { return 0 }
    let sum = zip(lhs, rhs).reduce(Float(0)) { partial, pair in
        partial + abs(pair.0 - pair.1)
    }
    return sum / Float(lhs.count)
}

private func normalizedScores(_ values: [Float]) -> [Float] {
    guard let minValue = values.min(), let maxValue = values.max() else {
        return []
    }
    let denominator = max(maxValue - minValue, 1.0e-6)
    return values.map { ($0 - minValue) / denominator }
}
