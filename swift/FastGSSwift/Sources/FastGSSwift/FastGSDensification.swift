import Foundation

public struct FastGSDensificationConfig: Codable, Equatable, Sendable {
    public var densifyFromStep: Int
    public var densifyUntilStep: Int
    public var densificationInterval: Int
    public var opacityResetInterval: Int
    public var opacityResetValue: Float
    public var opacityCapAfterDensify: Float
    public var gradThreshold: Float
    public var gradAbsThreshold: Float
    public var dense: Float
    public var lossThreshold: Float
    public var importanceScoreThreshold: Float
    public var densifyCameraSampleCount: Int
    public var splitFactor: Int
    public var minOpacity: Float
    public var finalPruneMinOpacity: Float
    public var finalPruneStartStep: Int
    public var finalPruneEndStep: Int
    public var finalPruneInterval: Int
    public var finalPruneScoreThreshold: Float
    public var finalPruneMinGaussians: Int
    public var maxScreenSize: Float
    public var maxWorldScaleFactor: Float
    public var pruneBudgetFactor: Float
    public var pruneGaussians: Bool

    public init(
        densifyFromStep: Int = 500,
        densifyUntilStep: Int = 15_000,
        densificationInterval: Int = 500,
        opacityResetInterval: Int = 3_000,
        opacityResetValue: Float = 0.82,
        opacityCapAfterDensify: Float = 0.82,
        gradThreshold: Float = 2.0e-4,
        gradAbsThreshold: Float = 1.2e-3,
        dense: Float = 0.01,
        lossThreshold: Float = 0.06,
        importanceScoreThreshold: Float = 5.0,
        densifyCameraSampleCount: Int = 10,
        splitFactor: Int = 2,
        minOpacity: Float = 0.005,
        finalPruneMinOpacity: Float = 0.1,
        finalPruneStartStep: Int = 15_000,
        finalPruneEndStep: Int = 30_000,
        finalPruneInterval: Int = 3_000,
        finalPruneScoreThreshold: Float = 0.9,
        finalPruneMinGaussians: Int = 64,
        maxScreenSize: Float = 20,
        maxWorldScaleFactor: Float = 0.1,
        pruneBudgetFactor: Float = 0.5,
        pruneGaussians: Bool = true
    ) {
        self.densifyFromStep = densifyFromStep
        self.densifyUntilStep = densifyUntilStep
        self.densificationInterval = densificationInterval
        self.opacityResetInterval = opacityResetInterval
        self.opacityResetValue = opacityResetValue
        self.opacityCapAfterDensify = opacityCapAfterDensify
        self.gradThreshold = gradThreshold
        self.gradAbsThreshold = gradAbsThreshold
        self.dense = dense
        self.lossThreshold = lossThreshold
        self.importanceScoreThreshold = importanceScoreThreshold
        self.densifyCameraSampleCount = densifyCameraSampleCount
        self.splitFactor = splitFactor
        self.minOpacity = minOpacity
        self.finalPruneMinOpacity = finalPruneMinOpacity
        self.finalPruneStartStep = finalPruneStartStep
        self.finalPruneEndStep = finalPruneEndStep
        self.finalPruneInterval = finalPruneInterval
        self.finalPruneScoreThreshold = finalPruneScoreThreshold
        self.finalPruneMinGaussians = finalPruneMinGaussians
        self.maxScreenSize = maxScreenSize
        self.maxWorldScaleFactor = maxWorldScaleFactor
        self.pruneBudgetFactor = pruneBudgetFactor
        self.pruneGaussians = pruneGaussians
    }

    public func shouldAccumulateStats(step: Int) -> Bool {
        step < densifyUntilStep
    }

    public func shouldDensifyAndPrune(step: Int) -> Bool {
        densificationInterval > 0
            && step > densifyFromStep
            && step < densifyUntilStep
            && step % densificationInterval == 0
    }

    public func shouldResetOpacity(step: Int) -> Bool {
        opacityResetInterval > 0
            && step < densifyUntilStep
            && step % opacityResetInterval == 0
    }

    public func shouldFinalPrune(step: Int) -> Bool {
        finalPruneInterval > 0
            && step > finalPruneStartStep
            && step < finalPruneEndStep
            && step % finalPruneInterval == 0
    }
}

public struct FastGSDensificationState: Codable, Equatable, Sendable {
    public var maxRadii2D: [Float]
    public var xyzGradAccum: [Float]
    public var xyzGradAccumAbs: [Float]
    public var denom: [Float]
    public var tmpRadii: [Float]?
    public var sceneExtent: Float

    public init(count: Int, sceneExtent: Float = 1) {
        precondition(count >= 0, "count must be non-negative")
        self.maxRadii2D = [Float](repeating: 0, count: count)
        self.xyzGradAccum = [Float](repeating: 0, count: count)
        self.xyzGradAccumAbs = [Float](repeating: 0, count: count)
        self.denom = [Float](repeating: 0, count: count)
        self.tmpRadii = nil
        self.sceneExtent = sceneExtent
    }

    public var count: Int {
        maxRadii2D.count
    }

    public mutating func reset(count: Int, sceneExtent: Float? = nil) {
        precondition(count >= 0, "count must be non-negative")
        maxRadii2D = [Float](repeating: 0, count: count)
        xyzGradAccum = [Float](repeating: 0, count: count)
        xyzGradAccumAbs = [Float](repeating: 0, count: count)
        denom = [Float](repeating: 0, count: count)
        tmpRadii = nil
        if let sceneExtent {
            self.sceneExtent = sceneExtent
        }
    }

    public func validate(count expectedCount: Int? = nil) {
        if let expectedCount {
            precondition(count == expectedCount, "densification state count mismatch")
        }
        precondition(xyzGradAccum.count == count, "xyzGradAccum count mismatch")
        precondition(xyzGradAccumAbs.count == count, "xyzGradAccumAbs count mismatch")
        precondition(denom.count == count, "denom count mismatch")
        if let tmpRadii {
            precondition(tmpRadii.count == count, "tmpRadii count mismatch")
        }
    }

    public mutating func update(
        radii: [Float],
        viewspaceGradients: [Float],
        viewspaceGradientWidth: Int = 4
    ) {
        validate(count: radii.count)
        precondition(viewspaceGradientWidth >= 4, "viewspaceGradientWidth must include x, y, z, w gradients")
        precondition(
            viewspaceGradients.count == radii.count * viewspaceGradientWidth,
            "viewspace gradient count mismatch"
        )

        tmpRadii = radii
        for index in radii.indices where radii[index] > 0 {
            let base = index * viewspaceGradientWidth
            let gx = viewspaceGradients[base]
            let gy = viewspaceGradients[base + 1]
            let gz = viewspaceGradients[base + 2]
            let gw = viewspaceGradients[base + 3]

            maxRadii2D[index] = max(maxRadii2D[index], radii[index])
            xyzGradAccum[index] += (gx * gx + gy * gy).squareRoot()
            xyzGradAccumAbs[index] += (gz * gz + gw * gw).squareRoot()
            denom[index] += 1
        }
    }

    public func averageGradients() -> (gradient: [Float], gradientAbs: [Float]) {
        validate()
        var gradient = [Float](repeating: 0, count: count)
        var gradientAbs = [Float](repeating: 0, count: count)
        for index in 0..<count where denom[index] > 0 {
            gradient[index] = xyzGradAccum[index] / denom[index]
            gradientAbs[index] = xyzGradAccumAbs[index] / denom[index]
        }
        return (gradient, gradientAbs)
    }

    public func pruned(mask: [Bool]) -> FastGSDensificationState {
        validate(count: mask.count)
        return FastGSDensificationState(
            maxRadii2D: pruneValues(maxRadii2D, mask: mask),
            xyzGradAccum: pruneValues(xyzGradAccum, mask: mask),
            xyzGradAccumAbs: pruneValues(xyzGradAccumAbs, mask: mask),
            denom: pruneValues(denom, mask: mask),
            tmpRadii: tmpRadii.map { pruneValues($0, mask: mask) },
            sceneExtent: sceneExtent
        )
    }

    public mutating func prune(mask: [Bool]) {
        self = pruned(mask: mask)
    }
}

private extension FastGSDensificationState {
    init(
        maxRadii2D: [Float],
        xyzGradAccum: [Float],
        xyzGradAccumAbs: [Float],
        denom: [Float],
        tmpRadii: [Float]?,
        sceneExtent: Float
    ) {
        self.maxRadii2D = maxRadii2D
        self.xyzGradAccum = xyzGradAccum
        self.xyzGradAccumAbs = xyzGradAccumAbs
        self.denom = denom
        self.tmpRadii = tmpRadii
        self.sceneExtent = sceneExtent
        validate()
    }
}

private func pruneValues<T>(_ values: [T], mask: [Bool]) -> [T] {
    precondition(values.count == mask.count, "prune mask count mismatch")
    return values.enumerated().compactMap { index, value in
        mask[index] ? nil : value
    }
}
