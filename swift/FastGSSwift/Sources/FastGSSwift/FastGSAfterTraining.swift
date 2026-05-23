import Foundation
import MLX

public struct FastGSOpacityAfterTrainResult {
    public var parameters: FastGSTrainableParameters
    public var optimizerState: FastGSAdamState?

    public init(parameters: FastGSTrainableParameters, optimizerState: FastGSAdamState?) {
        self.parameters = parameters
        self.optimizerState = optimizerState
    }
}

public struct FastGSPruneOnlyResult {
    public var parameters: FastGSTrainableParameters
    public var optimizerState: FastGSAdamState?
    public var densificationState: FastGSDensificationState?
    public var pruneMask: [Bool]
    public var opacityHits: Int
    public var screenSizeHits: Int
    public var worldScaleHits: Int
    public var prunedCount: Int
    public var keptCount: Int

    public init(
        parameters: FastGSTrainableParameters,
        optimizerState: FastGSAdamState?,
        densificationState: FastGSDensificationState?,
        pruneMask: [Bool],
        opacityHits: Int,
        screenSizeHits: Int,
        worldScaleHits: Int
    ) {
        self.parameters = parameters
        self.optimizerState = optimizerState
        self.densificationState = densificationState
        self.pruneMask = pruneMask
        self.opacityHits = opacityHits
        self.screenSizeHits = screenSizeHits
        self.worldScaleHits = worldScaleHits
        self.prunedCount = pruneMask.filter(\.self).count
        self.keptCount = pruneMask.count - prunedCount
    }
}

public struct FastGSCloneResult {
    public var parameters: FastGSTrainableParameters
    public var optimizerState: FastGSAdamState?
    public var densificationState: FastGSDensificationState
    public var cloneMask: [Bool]
    public var clonedCount: Int

    public init(
        parameters: FastGSTrainableParameters,
        optimizerState: FastGSAdamState?,
        densificationState: FastGSDensificationState,
        cloneMask: [Bool]
    ) {
        self.parameters = parameters
        self.optimizerState = optimizerState
        self.densificationState = densificationState
        self.cloneMask = cloneMask
        self.clonedCount = cloneMask.filter(\.self).count
    }
}

public enum FastGSAfterTraining {
    public static func capOpacity(
        parameters: FastGSTrainableParameters,
        optimizerState: FastGSAdamState? = nil,
        maxOpacity: Float,
        stream: StreamOrDevice = .default
    ) -> FastGSOpacityAfterTrainResult {
        precondition(maxOpacity >= 0 && maxOpacity <= 1, "maxOpacity must be in [0, 1]")
        parameters.validateTopology()
        optimizerState?.validateTopology(parameters: parameters)

        var updated = parameters
        let cappedProbabilities = minimum(
            parameters.opacityProbabilities(stream: stream),
            maxOpacity,
            stream: stream
        )
        updated.opacityLogits = FastGSOpacity.logits(fromProbabilities: cappedProbabilities, stream: stream)
        let updatedState = optimizerState?.resettingOpacityLogitState(like: updated, stream: stream)
        return FastGSOpacityAfterTrainResult(parameters: updated, optimizerState: updatedState)
    }

    public static func resetOpacity(
        parameters: FastGSTrainableParameters,
        optimizerState: FastGSAdamState? = nil,
        resetValue: Float,
        stream: StreamOrDevice = .default
    ) -> FastGSOpacityAfterTrainResult {
        capOpacity(
            parameters: parameters,
            optimizerState: optimizerState,
            maxOpacity: resetValue,
            stream: stream
        )
    }

    public static func pruneOnly(
        parameters: FastGSTrainableParameters,
        optimizerState: FastGSAdamState? = nil,
        densificationState: FastGSDensificationState? = nil,
        minOpacity: Float,
        maxScreenSize: Float? = nil,
        maxWorldScaleFactor: Float? = nil,
        sceneExtent: Float? = nil,
        minGaussians: Int = 1,
        stream: StreamOrDevice = .default
    ) -> FastGSPruneOnlyResult {
        precondition(minOpacity >= 0 && minOpacity <= 1, "minOpacity must be in [0, 1]")
        precondition(minGaussians >= 0, "minGaussians must be non-negative")
        if let maxScreenSize {
            precondition(maxScreenSize >= 0, "maxScreenSize must be non-negative")
        }
        if let maxWorldScaleFactor {
            precondition(maxWorldScaleFactor >= 0, "maxWorldScaleFactor must be non-negative")
        }
        parameters.validateTopology()
        optimizerState?.validateTopology(parameters: parameters)
        densificationState?.validate(count: parameters.gaussianCount)

        let count = parameters.gaussianCount
        guard count > 0 else {
            return FastGSPruneOnlyResult(
                parameters: parameters,
                optimizerState: optimizerState,
                densificationState: densificationState,
                pruneMask: [],
                opacityHits: 0,
                screenSizeHits: 0,
                worldScaleHits: 0
            )
        }

        let opacities = parameters.opacityProbabilities(stream: stream).asArray(Float.self)
        precondition(opacities.count == count, "opacity count mismatch")

        let opacityMask = opacities.map { $0 < minOpacity }
        let screenMask = makeScreenSizeMask(
            densificationState: densificationState,
            maxScreenSize: maxScreenSize,
            count: count
        )
        let worldMask = makeWorldScaleMask(
            parameters: parameters,
            sceneExtent: sceneExtent ?? densificationState?.sceneExtent,
            maxWorldScaleFactor: maxWorldScaleFactor,
            count: count
        )

        let opacityHits = opacityMask.filter(\.self).count
        let screenSizeHits = screenMask.filter(\.self).count
        let worldScaleHits = worldMask.filter(\.self).count
        var pruneMask = (0..<count).map { index in
            opacityMask[index] || screenMask[index] || worldMask[index]
        }
        enforceMinimumGaussians(mask: &pruneMask, opacities: opacities, minGaussians: min(minGaussians, count))

        let prunedParameters = parameters.prune(mask: pruneMask, stream: stream)
        let prunedOptimizerState = optimizerState?.prune(mask: pruneMask, stream: stream)
        let prunedDensificationState = densificationState?.pruned(mask: pruneMask)
        prunedOptimizerState?.validateTopology(parameters: prunedParameters)
        prunedDensificationState?.validate(count: prunedParameters.gaussianCount)

        return FastGSPruneOnlyResult(
            parameters: prunedParameters,
            optimizerState: prunedOptimizerState,
            densificationState: prunedDensificationState,
            pruneMask: pruneMask,
            opacityHits: opacityHits,
            screenSizeHits: screenSizeHits,
            worldScaleHits: worldScaleHits
        )
    }

    public static func clone(
        parameters: FastGSTrainableParameters,
        optimizerState: FastGSAdamState? = nil,
        densificationState: FastGSDensificationState,
        gradThreshold: Float,
        dense: Float,
        sceneExtent: Float? = nil,
        importanceScores: [Float]? = nil,
        importanceScoreThreshold: Float = -.infinity,
        stream: StreamOrDevice = .default
    ) -> FastGSCloneResult {
        precondition(gradThreshold >= 0, "gradThreshold must be non-negative")
        precondition(dense >= 0, "dense must be non-negative")
        parameters.validateTopology()
        optimizerState?.validateTopology(parameters: parameters)
        densificationState.validate(count: parameters.gaussianCount)

        let count = parameters.gaussianCount
        if let importanceScores {
            precondition(importanceScores.count == count, "importance score count mismatch")
        }
        guard count > 0 else {
            return FastGSCloneResult(
                parameters: parameters,
                optimizerState: optimizerState,
                densificationState: densificationState,
                cloneMask: []
            )
        }

        let resolvedSceneExtent = sceneExtent ?? densificationState.sceneExtent
        precondition(resolvedSceneExtent >= 0, "sceneExtent must be non-negative")
        let averages = densificationState.averageGradients()
        let maxScales = gaussianMaxScales(parameters: parameters, count: count)
        let scaleThreshold = dense * resolvedSceneExtent
        let cloneMask = (0..<count).map { index in
            let gradientPass = averages.gradient[index] >= gradThreshold
            let scalePass = maxScales[index] <= scaleThreshold
            let scorePass = importanceScores.map { $0[index] > importanceScoreThreshold } ?? true
            return gradientPass && scalePass && scorePass
        }
        let cloneIndices = cloneMask.enumerated().compactMap { index, shouldClone in
            shouldClone ? index : nil
        }
        guard !cloneIndices.isEmpty else {
            return FastGSCloneResult(
                parameters: parameters,
                optimizerState: optimizerState,
                densificationState: densificationState,
                cloneMask: cloneMask
            )
        }

        let clonedTail = parameters.take(indices: cloneIndices, stream: stream)
        let clonedParameters = parameters.appending(clonedTail, stream: stream)
        let clonedOptimizerState = optimizerState?.appendingZeroRows(like: clonedTail, stream: stream)
        let clonedDensificationState = densificationState.appendingResetRows(count: cloneIndices.count)
        clonedOptimizerState?.validateTopology(parameters: clonedParameters)
        clonedDensificationState.validate(count: clonedParameters.gaussianCount)

        return FastGSCloneResult(
            parameters: clonedParameters,
            optimizerState: clonedOptimizerState,
            densificationState: clonedDensificationState,
            cloneMask: cloneMask
        )
    }
}

private func makeScreenSizeMask(
    densificationState: FastGSDensificationState?,
    maxScreenSize: Float?,
    count: Int
) -> [Bool] {
    guard let densificationState, let maxScreenSize, maxScreenSize > 0 else {
        return [Bool](repeating: false, count: count)
    }
    densificationState.validate(count: count)
    return densificationState.maxRadii2D.map { $0 > maxScreenSize }
}

private func makeWorldScaleMask(
    parameters: FastGSTrainableParameters,
    sceneExtent: Float?,
    maxWorldScaleFactor: Float?,
    count: Int
) -> [Bool] {
    guard let sceneExtent, let maxWorldScaleFactor, sceneExtent > 0, maxWorldScaleFactor > 0 else {
        return [Bool](repeating: false, count: count)
    }
    let maxScales = gaussianMaxScales(parameters: parameters, count: count)
    let threshold = sceneExtent * maxWorldScaleFactor
    return maxScales.map { $0 > threshold }
}

private func gaussianMaxScales(parameters: FastGSTrainableParameters, count: Int) -> [Float] {
    let scales = parameters.scales.asArray(Float.self)
    precondition(scales.count % count == 0, "scale count mismatch")
    let width = scales.count / count
    precondition(width > 0, "scale width mismatch")
    return (0..<count).map { index in
        let base = index * width
        return scales[base..<(base + width)].map { Foundation.exp($0) }.max() ?? 0
    }
}

private func enforceMinimumGaussians(mask: inout [Bool], opacities: [Float], minGaussians: Int) {
    guard minGaussians > 0 else { return }
    precondition(mask.count == opacities.count, "minimum Gaussian guard count mismatch")
    var keptCount = mask.filter { !$0 }.count
    guard keptCount < minGaussians else { return }

    let indicesByOpacity = opacities.indices.sorted { lhs, rhs in
        if opacities[lhs] == opacities[rhs] {
            return lhs < rhs
        }
        return opacities[lhs] > opacities[rhs]
    }
    for index in indicesByOpacity where mask[index] {
        mask[index] = false
        keptCount += 1
        if keptCount >= minGaussians {
            break
        }
    }
}
