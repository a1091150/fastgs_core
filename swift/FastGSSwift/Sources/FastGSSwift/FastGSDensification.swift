import Foundation
import MLX

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
    public var scoringSeed: UInt64
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

    enum CodingKeys: String, CodingKey {
        case densifyFromStep
        case densifyUntilStep
        case densificationInterval
        case opacityResetInterval
        case opacityResetValue
        case opacityCapAfterDensify
        case gradThreshold
        case gradAbsThreshold
        case dense
        case lossThreshold
        case importanceScoreThreshold
        case densifyCameraSampleCount
        case scoringSeed
        case splitFactor
        case minOpacity
        case finalPruneMinOpacity
        case finalPruneStartStep
        case finalPruneEndStep
        case finalPruneInterval
        case finalPruneScoreThreshold
        case finalPruneMinGaussians
        case maxScreenSize
        case maxWorldScaleFactor
        case pruneBudgetFactor
        case pruneGaussians
    }

    public init(
        densifyFromStep: Int = 500,
        densifyUntilStep: Int = 15_000,
        densificationInterval: Int = 500,
        opacityResetInterval: Int = 3_000,
        opacityResetValue: Float = 0.82,
        opacityCapAfterDensify: Float = 0.82,
        gradThreshold: Float = 2.0e-4,
        gradAbsThreshold: Float = 1.2e-3,
        dense: Float = 0.001,
        lossThreshold: Float = 0.1,
        importanceScoreThreshold: Float = 5.0,
        densifyCameraSampleCount: Int = 10,
        scoringSeed: UInt64 = 42,
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
        self.scoringSeed = scoringSeed
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

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            densifyFromStep: try container.decodeIfPresent(Int.self, forKey: .densifyFromStep) ?? 500,
            densifyUntilStep: try container.decodeIfPresent(Int.self, forKey: .densifyUntilStep) ?? 15_000,
            densificationInterval: try container.decodeIfPresent(Int.self, forKey: .densificationInterval) ?? 500,
            opacityResetInterval: try container.decodeIfPresent(Int.self, forKey: .opacityResetInterval) ?? 3_000,
            opacityResetValue: try container.decodeIfPresent(Float.self, forKey: .opacityResetValue) ?? 0.82,
            opacityCapAfterDensify: try container.decodeIfPresent(Float.self, forKey: .opacityCapAfterDensify) ?? 0.82,
            gradThreshold: try container.decodeIfPresent(Float.self, forKey: .gradThreshold) ?? 2.0e-4,
            gradAbsThreshold: try container.decodeIfPresent(Float.self, forKey: .gradAbsThreshold) ?? 1.2e-3,
            dense: try container.decodeIfPresent(Float.self, forKey: .dense) ?? 0.001,
            lossThreshold: try container.decodeIfPresent(Float.self, forKey: .lossThreshold) ?? 0.1,
            importanceScoreThreshold: try container.decodeIfPresent(Float.self, forKey: .importanceScoreThreshold) ?? 5.0,
            densifyCameraSampleCount: try container.decodeIfPresent(Int.self, forKey: .densifyCameraSampleCount) ?? 10,
            scoringSeed: try container.decodeIfPresent(UInt64.self, forKey: .scoringSeed) ?? 42,
            splitFactor: try container.decodeIfPresent(Int.self, forKey: .splitFactor) ?? 2,
            minOpacity: try container.decodeIfPresent(Float.self, forKey: .minOpacity) ?? 0.005,
            finalPruneMinOpacity: try container.decodeIfPresent(Float.self, forKey: .finalPruneMinOpacity) ?? 0.1,
            finalPruneStartStep: try container.decodeIfPresent(Int.self, forKey: .finalPruneStartStep) ?? 15_000,
            finalPruneEndStep: try container.decodeIfPresent(Int.self, forKey: .finalPruneEndStep) ?? 30_000,
            finalPruneInterval: try container.decodeIfPresent(Int.self, forKey: .finalPruneInterval) ?? 3_000,
            finalPruneScoreThreshold: try container.decodeIfPresent(Float.self, forKey: .finalPruneScoreThreshold) ?? 0.9,
            finalPruneMinGaussians: try container.decodeIfPresent(Int.self, forKey: .finalPruneMinGaussians) ?? 64,
            maxScreenSize: try container.decodeIfPresent(Float.self, forKey: .maxScreenSize) ?? 20,
            maxWorldScaleFactor: try container.decodeIfPresent(Float.self, forKey: .maxWorldScaleFactor) ?? 0.1,
            pruneBudgetFactor: try container.decodeIfPresent(Float.self, forKey: .pruneBudgetFactor) ?? 0.5,
            pruneGaussians: try container.decodeIfPresent(Bool.self, forKey: .pruneGaussians) ?? true
        )
    }

    public static func scannerFastGS2Base(scheduleScale: Float = 1) -> FastGSDensificationConfig {
        precondition(scheduleScale > 0, "scheduleScale must be positive")
        func scaled(_ value: Int) -> Int {
            max(1, Int((Float(value) * scheduleScale).rounded()))
        }
        return FastGSDensificationConfig(
            densifyFromStep: scaled(500),
            densifyUntilStep: scaled(15_000),
            densificationInterval: scaled(500),
            opacityResetInterval: scaled(3_000),
            opacityResetValue: 0.82,
            opacityCapAfterDensify: 0.82,
            gradThreshold: 2.0e-4,
            gradAbsThreshold: 1.2e-3,
            dense: 0.001,
            lossThreshold: 0.1,
            importanceScoreThreshold: 5.0,
            densifyCameraSampleCount: 10,
            scoringSeed: 42,
            splitFactor: 2,
            minOpacity: 0.005,
            finalPruneMinOpacity: 0.1,
            finalPruneStartStep: scaled(15_000),
            finalPruneEndStep: scaled(30_000),
            finalPruneInterval: scaled(3_000),
            finalPruneScoreThreshold: 0.9,
            finalPruneMinGaussians: 64,
            maxScreenSize: 20,
            maxWorldScaleFactor: 0.1,
            pruneBudgetFactor: 0.5,
            pruneGaussians: true
        )
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

    public mutating func update(
        radii: MLXArray,
        viewspaceGradients: MLXArray
    ) {
        let radiiValues = radii.asArray(Int32.self).map(Float.init)
        update(
            radii: radiiValues,
            viewspaceGradients: viewspaceGradients.asArray(Float.self),
            viewspaceGradientWidth: viewspaceGradients.shape.count > 1 ? viewspaceGradients.shape[1] : 4
        )
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

    public func appendingResetRows(count appendedCount: Int) -> FastGSDensificationState {
        precondition(appendedCount >= 0, "appendedCount must be non-negative")
        var result = self
        result.reset(count: count + appendedCount)
        return result
    }

    public mutating func appendResetRows(count appendedCount: Int) {
        self = appendingResetRows(count: appendedCount)
    }

    public func appendingZeroRows(count appendedCount: Int) -> FastGSDensificationState {
        precondition(appendedCount >= 0, "appendedCount must be non-negative")
        validate()
        return FastGSDensificationState(
            maxRadii2D: maxRadii2D + [Float](repeating: 0, count: appendedCount),
            xyzGradAccum: xyzGradAccum + [Float](repeating: 0, count: appendedCount),
            xyzGradAccumAbs: xyzGradAccumAbs + [Float](repeating: 0, count: appendedCount),
            denom: denom + [Float](repeating: 0, count: appendedCount),
            tmpRadii: tmpRadii.map { $0 + [Float](repeating: 0, count: appendedCount) },
            sceneExtent: sceneExtent
        )
    }

    public mutating func appendZeroRows(count appendedCount: Int) {
        self = appendingZeroRows(count: appendedCount)
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
