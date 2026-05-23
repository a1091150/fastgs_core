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

public struct FastGSSplitResult {
    public var parameters: FastGSTrainableParameters
    public var optimizerState: FastGSAdamState?
    public var densificationState: FastGSDensificationState
    public var splitMask: [Bool]
    public var sourceCount: Int
    public var childCount: Int

    public init(
        parameters: FastGSTrainableParameters,
        optimizerState: FastGSAdamState?,
        densificationState: FastGSDensificationState,
        splitMask: [Bool],
        childCount: Int
    ) {
        self.parameters = parameters
        self.optimizerState = optimizerState
        self.densificationState = densificationState
        self.splitMask = splitMask
        self.sourceCount = splitMask.filter(\.self).count
        self.childCount = childCount
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

    public static func split(
        parameters: FastGSTrainableParameters,
        optimizerState: FastGSAdamState? = nil,
        densificationState: FastGSDensificationState,
        gradAbsThreshold: Float,
        dense: Float,
        splitFactor: Int,
        sceneExtent: Float? = nil,
        importanceScores: [Float]? = nil,
        importanceScoreThreshold: Float = -.infinity,
        standardNormalSamples: [Float]? = nil,
        stream: StreamOrDevice = .default
    ) -> FastGSSplitResult {
        precondition(gradAbsThreshold >= 0, "gradAbsThreshold must be non-negative")
        precondition(dense >= 0, "dense must be non-negative")
        precondition(splitFactor > 0, "splitFactor must be positive")
        parameters.validateTopology()
        optimizerState?.validateTopology(parameters: parameters)
        densificationState.validate(count: parameters.gaussianCount)

        let count = parameters.gaussianCount
        if let importanceScores {
            precondition(importanceScores.count == count, "importance score count mismatch")
        }
        guard count > 0 else {
            return FastGSSplitResult(
                parameters: parameters,
                optimizerState: optimizerState,
                densificationState: densificationState,
                splitMask: [],
                childCount: 0
            )
        }

        let resolvedSceneExtent = sceneExtent ?? densificationState.sceneExtent
        precondition(resolvedSceneExtent >= 0, "sceneExtent must be non-negative")
        let averages = densificationState.averageGradients()
        let maxScales = gaussianMaxScales(parameters: parameters, count: count)
        let scaleThreshold = dense * resolvedSceneExtent
        let splitMask = (0..<count).map { index in
            let gradientPass = averages.gradientAbs[index] >= gradAbsThreshold
            let scalePass = maxScales[index] > scaleThreshold
            let scorePass = importanceScores.map { $0[index] > importanceScoreThreshold } ?? true
            return gradientPass && scalePass && scorePass
        }
        let splitIndices = splitMask.enumerated().compactMap { index, shouldSplit in
            shouldSplit ? index : nil
        }
        guard !splitIndices.isEmpty else {
            return FastGSSplitResult(
                parameters: parameters,
                optimizerState: optimizerState,
                densificationState: densificationState,
                splitMask: splitMask,
                childCount: 0
            )
        }

        let childCount = splitIndices.count * splitFactor
        let samples = standardNormalSamples ?? MLXRandom.normal([childCount, 3], stream: stream).asArray(Float.self)
        precondition(samples.count == childCount * 3, "standardNormalSamples must have shape [sourceCount * splitFactor, 3]")

        let childParameters = splitChildren(
            parameters: parameters,
            splitIndices: splitIndices,
            splitFactor: splitFactor,
            standardNormalSamples: samples
        )
        let appendedParameters = parameters.appending(childParameters, stream: stream)
        let appendedOptimizerState = optimizerState?.appendingZeroRows(like: childParameters, stream: stream)
        let appendedDensificationState = densificationState.appendingResetRows(count: childCount)
        let pruneMask = splitMask + [Bool](repeating: false, count: childCount)
        let splitParameters = appendedParameters.prune(mask: pruneMask, stream: stream)
        let splitOptimizerState = appendedOptimizerState?.prune(mask: pruneMask, stream: stream)
        let splitDensificationState = appendedDensificationState.pruned(mask: pruneMask)

        splitOptimizerState?.validateTopology(parameters: splitParameters)
        splitDensificationState.validate(count: splitParameters.gaussianCount)
        return FastGSSplitResult(
            parameters: splitParameters,
            optimizerState: splitOptimizerState,
            densificationState: splitDensificationState,
            splitMask: splitMask,
            childCount: childCount
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

private func splitChildren(
    parameters: FastGSTrainableParameters,
    splitIndices: [Int],
    splitFactor: Int,
    standardNormalSamples: [Float]
) -> FastGSTrainableParameters {
    let count = parameters.gaussianCount
    let means = parameters.means3D.asArray(Float.self)
    let dc = parameters.dc.asArray(Float.self)
    let sh = parameters.sh.asArray(Float.self)
    let opacityLogits = parameters.opacityLogits.asArray(Float.self)
    let scales = parameters.scales.asArray(Float.self)
    let rotations = parameters.rotations.asArray(Float.self)
    let cov3D = parameters.cov3DPrecomputed?.asArray(Float.self)

    let meansWidth = gaussianFieldWidth(parameters.means3D, count: count, name: "means3D")
    let dcWidth = gaussianFieldWidth(parameters.dc, count: count, name: "dc")
    let shWidth = gaussianFieldWidth(parameters.sh, count: count, name: "sh")
    let opacityWidth = gaussianFieldWidth(parameters.opacityLogits, count: count, name: "opacityLogits")
    let scalesWidth = gaussianFieldWidth(parameters.scales, count: count, name: "scales")
    let rotationsWidth = gaussianFieldWidth(parameters.rotations, count: count, name: "rotations")
    let cov3DWidth = parameters.cov3DPrecomputed.map { gaussianFieldWidth($0, count: count, name: "cov3DPrecomputed") }
    precondition(meansWidth == 3, "means3D split currently expects xyz rows")
    precondition(scalesWidth == 3, "scales split currently expects xyz log-scale rows")
    precondition(rotationsWidth == 4, "rotations split currently expects wxyz quaternion rows")
    precondition(opacityWidth == 1, "opacityLogits split currently expects scalar rows")

    let childCount = splitIndices.count * splitFactor
    var childMeans = [Float]()
    var childDC = [Float]()
    var childSH = [Float]()
    var childOpacityLogits = [Float]()
    var childScales = [Float]()
    var childRotations = [Float]()
    var childCov3D = cov3D.map { _ in [Float]() }
    childMeans.reserveCapacity(childCount * meansWidth)
    childDC.reserveCapacity(childCount * dcWidth)
    childSH.reserveCapacity(childCount * shWidth)
    childOpacityLogits.reserveCapacity(childCount * opacityWidth)
    childScales.reserveCapacity(childCount * scalesWidth)
    childRotations.reserveCapacity(childCount * rotationsWidth)
    if let cov3DWidth {
        childCov3D?.reserveCapacity(childCount * cov3DWidth)
    }

    let shrink = Foundation.log(0.8 * Float(splitFactor))
    var sampleIndex = 0
    for sourceIndex in splitIndices {
        let sourceMean = Array(row(sourceIndex, width: meansWidth, values: means))
        let sourceScales = Array(row(sourceIndex, width: scalesWidth, values: scales))
        let sourceRotation = Array(row(sourceIndex, width: rotationsWidth, values: rotations))
        let sourcePhysicalScales = sourceScales.map(Foundation.exp)
        let normalizedRotation = normalizedQuaternion(sourceRotation)
        for _ in 0..<splitFactor {
            let local = [
                standardNormalSamples[sampleIndex] * sourcePhysicalScales[0],
                standardNormalSamples[sampleIndex + 1] * sourcePhysicalScales[1],
                standardNormalSamples[sampleIndex + 2] * sourcePhysicalScales[2],
            ]
            sampleIndex += 3
            let offset = rotate(local, byWXYZQuaternion: normalizedRotation)
            childMeans.append(sourceMean[0] + offset[0])
            childMeans.append(sourceMean[1] + offset[1])
            childMeans.append(sourceMean[2] + offset[2])
            childDC.append(contentsOf: row(sourceIndex, width: dcWidth, values: dc))
            childSH.append(contentsOf: row(sourceIndex, width: shWidth, values: sh))
            childOpacityLogits.append(contentsOf: row(sourceIndex, width: opacityWidth, values: opacityLogits))
            childScales.append(contentsOf: sourceScales.map { $0 - shrink })
            childRotations.append(contentsOf: row(sourceIndex, width: rotationsWidth, values: rotations))
            if let cov3D, let cov3DWidth {
                childCov3D?.append(contentsOf: row(sourceIndex, width: cov3DWidth, values: cov3D))
            }
        }
    }

    return FastGSTrainableParameters(
        means3D: MLXArray(childMeans, [childCount] + Array(parameters.means3D.shape.dropFirst())),
        dc: MLXArray(childDC, [childCount] + Array(parameters.dc.shape.dropFirst())),
        sh: MLXArray(childSH, [childCount] + Array(parameters.sh.shape.dropFirst())),
        opacityLogits: MLXArray(childOpacityLogits, [childCount] + Array(parameters.opacityLogits.shape.dropFirst())),
        scales: MLXArray(childScales, [childCount] + Array(parameters.scales.shape.dropFirst())),
        rotations: MLXArray(childRotations, [childCount] + Array(parameters.rotations.shape.dropFirst())),
        cov3DPrecomputed: childCov3D.map { MLXArray($0, [childCount] + Array(parameters.cov3DPrecomputed?.shape.dropFirst() ?? [])) }
    )
}

private func gaussianFieldWidth(_ array: MLXArray, count: Int, name: String) -> Int {
    precondition(!array.shape.isEmpty && array.shape[0] == count, "\(name) Gaussian count mismatch")
    let elementCount = array.shape.reduce(1, *)
    precondition(count == 0 || elementCount % count == 0, "\(name) row width mismatch")
    return count == 0 ? 0 : elementCount / count
}

private func row(_ index: Int, width: Int, values: [Float]) -> ArraySlice<Float> {
    let start = index * width
    return values[start..<(start + width)]
}

private func normalizedQuaternion(_ quaternion: [Float]) -> [Float] {
    precondition(quaternion.count == 4, "quaternion must have four components")
    let norm = max(1.0e-8, (quaternion[0] * quaternion[0]
        + quaternion[1] * quaternion[1]
        + quaternion[2] * quaternion[2]
        + quaternion[3] * quaternion[3]).squareRoot())
    return quaternion.map { $0 / norm }
}

private func rotate(_ vector: [Float], byWXYZQuaternion quaternion: [Float]) -> [Float] {
    precondition(vector.count == 3, "vector must have three components")
    let w = quaternion[0]
    let x = quaternion[1]
    let y = quaternion[2]
    let z = quaternion[3]
    let vx = vector[0]
    let vy = vector[1]
    let vz = vector[2]
    return [
        (1 - 2 * y * y - 2 * z * z) * vx + (2 * x * y - 2 * z * w) * vy + (2 * x * z + 2 * y * w) * vz,
        (2 * x * y + 2 * z * w) * vx + (1 - 2 * x * x - 2 * z * z) * vy + (2 * y * z - 2 * x * w) * vz,
        (2 * x * z - 2 * y * w) * vx + (2 * y * z + 2 * x * w) * vy + (1 - 2 * x * x - 2 * y * y) * vz,
    ]
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
