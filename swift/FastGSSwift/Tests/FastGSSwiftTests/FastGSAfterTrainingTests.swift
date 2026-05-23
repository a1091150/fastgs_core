import FastGSSwift
import Foundation
import MLX
import XCTest

final class FastGSAfterTrainingTests: XCTestCase {
    func testOpacityCapClampsProbabilitiesAndStoresLogits() throws {
        guard ProcessInfo.processInfo.environment["FASTGS_RUN_METAL_TESTS"] == "1" else {
            throw XCTSkip("MLX after-training tests require an Xcode/metallib-ready environment.")
        }

        let parameters = makeAfterTrainingParameters()
        let result = FastGSAfterTraining.capOpacity(
            parameters: parameters,
            maxOpacity: 0.82
        )

        XCTAssertNil(result.optimizerState)
        assertAfterTrainingClose(result.parameters.opacityProbabilities().asArray(Float.self), [0.2, 0.82, 0.82])
        assertAfterTrainingClose(result.parameters.means3D.asArray(Float.self), parameters.means3D.asArray(Float.self))
        assertAfterTrainingClose(result.parameters.scales.asArray(Float.self), parameters.scales.asArray(Float.self))
    }

    func testOpacityResetClearsOnlyOpacityOptimizerState() throws {
        guard ProcessInfo.processInfo.environment["FASTGS_RUN_METAL_TESTS"] == "1" else {
            throw XCTSkip("MLX after-training tests require an Xcode/metallib-ready environment.")
        }

        let parameters = makeAfterTrainingParameters()
        let state = makeAfterTrainingAdamState()
        let result = FastGSAfterTraining.resetOpacity(
            parameters: parameters,
            optimizerState: state,
            resetValue: 0.5
        )

        let updatedState = try XCTUnwrap(result.optimizerState)
        XCTAssertEqual(updatedState.step, 7)
        assertAfterTrainingClose(result.parameters.opacityProbabilities().asArray(Float.self), [0.2, 0.5, 0.5])
        assertAfterTrainingClose(updatedState.opacityLogits.firstMoment.asArray(Float.self), [0, 0, 0])
        assertAfterTrainingClose(updatedState.opacityLogits.secondMoment.asArray(Float.self), [0, 0, 0])
        assertAfterTrainingClose(updatedState.means3D.firstMoment.asArray(Float.self), state.means3D.firstMoment.asArray(Float.self))
        assertAfterTrainingClose(updatedState.rotations.secondMoment.asArray(Float.self), state.rotations.secondMoment.asArray(Float.self))
    }

    func testPruneOnlyAppliesOpacityScreenAndWorldMasks() throws {
        guard ProcessInfo.processInfo.environment["FASTGS_RUN_METAL_TESTS"] == "1" else {
            throw XCTSkip("MLX after-training tests require an Xcode/metallib-ready environment.")
        }

        let parameters = makePruneOnlyParameters(opacityProbabilities: [0.2, 0.001, 0.8, 0.004])
        let state = FastGSAdamState(step: 11, parameters: parameters)
        var densificationState = FastGSDensificationState(count: 4, sceneExtent: 10)
        densificationState.maxRadii2D = [2, 3, 99, 4]
        densificationState.xyzGradAccum = [10, 20, 30, 40]
        densificationState.xyzGradAccumAbs = [11, 21, 31, 41]
        densificationState.denom = [1, 2, 3, 4]
        densificationState.tmpRadii = [5, 6, 7, 8]

        let result = FastGSAfterTraining.pruneOnly(
            parameters: parameters,
            optimizerState: state,
            densificationState: densificationState,
            minOpacity: 0.005,
            maxScreenSize: 20,
            maxWorldScaleFactor: 0.1,
            minGaussians: 1
        )

        XCTAssertEqual(result.pruneMask, [false, true, true, true])
        XCTAssertEqual(result.opacityHits, 2)
        XCTAssertEqual(result.screenSizeHits, 1)
        XCTAssertEqual(result.worldScaleHits, 1)
        XCTAssertEqual(result.prunedCount, 3)
        XCTAssertEqual(result.keptCount, 1)
        XCTAssertEqual(result.parameters.gaussianCount, 1)
        XCTAssertEqual(result.optimizerState?.step, 11)
        XCTAssertEqual(result.optimizerState?.means3D.firstMoment.shape, [1, 3])
        XCTAssertEqual(result.densificationState?.maxRadii2D, [2])
        XCTAssertEqual(result.densificationState?.xyzGradAccum, [10])
        XCTAssertEqual(result.densificationState?.xyzGradAccumAbs, [11])
        XCTAssertEqual(result.densificationState?.denom, [1])
        XCTAssertEqual(result.densificationState?.tmpRadii, [5])
        assertAfterTrainingClose(result.parameters.opacityProbabilities().asArray(Float.self), [0.2])
    }

    func testPruneOnlyMinimumGaussianGuardKeepsHighestOpacityRows() throws {
        guard ProcessInfo.processInfo.environment["FASTGS_RUN_METAL_TESTS"] == "1" else {
            throw XCTSkip("MLX after-training tests require an Xcode/metallib-ready environment.")
        }

        let parameters = makePruneOnlyParameters(opacityProbabilities: [0.1, 0.2, 0.3])
        let result = FastGSAfterTraining.pruneOnly(
            parameters: parameters,
            minOpacity: 0.9,
            minGaussians: 2
        )

        XCTAssertEqual(result.pruneMask, [true, false, false])
        XCTAssertEqual(result.prunedCount, 1)
        XCTAssertEqual(result.keptCount, 2)
        XCTAssertEqual(result.parameters.gaussianCount, 2)
        assertAfterTrainingClose(result.parameters.opacityProbabilities().asArray(Float.self), [0.2, 0.3])
    }

    func testFinalPruneUsesScoresAndKeepsHighestScoredRows() throws {
        guard ProcessInfo.processInfo.environment["FASTGS_RUN_METAL_TESTS"] == "1" else {
            throw XCTSkip("MLX after-training tests require an Xcode/metallib-ready environment.")
        }

        let parameters = makePruneOnlyParameters(opacityProbabilities: [0.2, 0.3, 0.4, 0.5])
        let state = FastGSAdamState(step: 19, parameters: parameters)
        var densificationState = FastGSDensificationState(count: 4, sceneExtent: 10)
        densificationState.maxRadii2D = [1, 1, 1, 1]
        densificationState.xyzGradAccum = [10, 20, 30, 40]
        densificationState.xyzGradAccumAbs = [11, 21, 31, 41]
        densificationState.denom = [1, 2, 3, 4]

        let result = FastGSAfterTraining.finalPrune(
            parameters: parameters,
            optimizerState: state,
            densificationState: densificationState,
            pruningScores: [0.1, 0.95, 0.2, 0.8],
            scoreThreshold: 0.9,
            minOpacity: 0,
            minGaussians: 2
        )

        XCTAssertEqual(result.pruneMask, [true, false, true, false])
        XCTAssertEqual(result.scoreHits, 3)
        XCTAssertEqual(result.prunedCount, 2)
        XCTAssertEqual(result.keptCount, 2)
        XCTAssertEqual(result.parameters.gaussianCount, 2)
        XCTAssertEqual(result.optimizerState?.step, 19)
        XCTAssertEqual(result.densificationState?.xyzGradAccum, [20, 40])
        assertAfterTrainingClose(result.parameters.opacityProbabilities().asArray(Float.self), [0.3, 0.5])
    }

    func testCloneAppendsSmallHighGradientRowsAndResetsState() throws {
        guard ProcessInfo.processInfo.environment["FASTGS_RUN_METAL_TESTS"] == "1" else {
            throw XCTSkip("MLX after-training tests require an Xcode/metallib-ready environment.")
        }

        let parameters = makePruneOnlyParameters(opacityProbabilities: [0.2, 0.4, 0.6, 0.8])
        let optimizerState = FastGSAdamState(step: 13, parameters: parameters)
        var densificationState = FastGSDensificationState(count: 4, sceneExtent: 10)
        densificationState.xyzGradAccum = [0.01, 0.03, 0.50, 0.60]
        densificationState.xyzGradAccumAbs = [1, 2, 3, 4]
        densificationState.denom = [100, 100, 100, 100]
        densificationState.maxRadii2D = [4, 5, 6, 7]
        densificationState.tmpRadii = [8, 9, 10, 11]

        let result = FastGSAfterTraining.clone(
            parameters: parameters,
            optimizerState: optimizerState,
            densificationState: densificationState,
            gradThreshold: 2.0e-4,
            dense: 0.02,
            importanceScores: [9, 11, 12, 13],
            importanceScoreThreshold: 10
        )

        XCTAssertEqual(result.cloneMask, [false, true, false, false])
        XCTAssertEqual(result.clonedCount, 1)
        XCTAssertEqual(result.parameters.gaussianCount, 5)
        XCTAssertEqual(result.optimizerState?.step, 13)
        XCTAssertEqual(result.densificationState.count, 5)
        XCTAssertEqual(result.densificationState.maxRadii2D, [0, 0, 0, 0, 0])
        XCTAssertEqual(result.densificationState.xyzGradAccum, [0, 0, 0, 0, 0])
        XCTAssertNil(result.densificationState.tmpRadii)

        assertAfterTrainingClose(result.parameters.means3D.asArray(Float.self), [
            0, 1, 2,
            3, 4, 5,
            6, 7, 8,
            9, 10, 11,
            3, 4, 5
        ])
        assertAfterTrainingClose(result.parameters.opacityProbabilities().asArray(Float.self), [0.2, 0.4, 0.6, 0.8, 0.4])
        assertAfterTrainingClose(result.optimizerState?.means3D.firstMoment.asArray(Float.self) ?? [], [
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0
        ])
    }

    func testSplitAppendsChildrenPrunesSourcesAndResetsState() throws {
        guard ProcessInfo.processInfo.environment["FASTGS_RUN_METAL_TESTS"] == "1" else {
            throw XCTSkip("MLX after-training tests require an Xcode/metallib-ready environment.")
        }

        let parameters = makePruneOnlyParameters(opacityProbabilities: [0.2, 0.4, 0.6, 0.8])
        let optimizerState = FastGSAdamState(step: 17, parameters: parameters)
        var densificationState = FastGSDensificationState(count: 4, sceneExtent: 10)
        densificationState.xyzGradAccum = [1, 2, 3, 4]
        densificationState.xyzGradAccumAbs = [0.01, 0.02, 0.50, 0.03]
        densificationState.denom = [100, 100, 100, 100]
        densificationState.maxRadii2D = [4, 5, 6, 7]
        densificationState.tmpRadii = [8, 9, 10, 11]

        let result = FastGSAfterTraining.split(
            parameters: parameters,
            optimizerState: optimizerState,
            densificationState: densificationState,
            gradAbsThreshold: 1.2e-3,
            dense: 0.02,
            splitFactor: 2,
            importanceScores: [9, 11, 12, 13],
            importanceScoreThreshold: 10,
            standardNormalSamples: [
                1, 0, 0,
                0, 1, 0,
            ]
        )

        XCTAssertEqual(result.splitMask, [false, false, true, false])
        XCTAssertEqual(result.sourceCount, 1)
        XCTAssertEqual(result.childCount, 2)
        XCTAssertEqual(result.parameters.gaussianCount, 5)
        XCTAssertEqual(result.optimizerState?.step, 17)
        XCTAssertEqual(result.densificationState.count, 5)
        XCTAssertEqual(result.densificationState.maxRadii2D, [0, 0, 0, 0, 0])
        XCTAssertNil(result.densificationState.tmpRadii)
        assertAfterTrainingClose(result.parameters.means3D.asArray(Float.self), [
            0, 1, 2,
            3, 4, 5,
            9, 10, 11,
            9, 7, 8,
            6, 10, 8
        ])
        assertAfterTrainingClose(result.parameters.opacityProbabilities().asArray(Float.self), [0.2, 0.4, 0.8, 0.6, 0.6])
        assertAfterTrainingClose(result.parameters.scales.asArray(Float.self).suffix(6).map { $0 }, [
            Foundation.log(Float(3.0 / 1.6)), Foundation.log(Float(3.0 / 1.6)), Foundation.log(Float(3.0 / 1.6)),
            Foundation.log(Float(3.0 / 1.6)), Foundation.log(Float(3.0 / 1.6)), Foundation.log(Float(3.0 / 1.6)),
        ])
    }
}

private func makeAfterTrainingParameters() -> FastGSTrainableParameters {
    FastGSTrainableParameters(
        means3D: MLXArray((0..<9).map { Float($0) }, [3, 3]),
        dc: MLXArray([Float](repeating: 0.1, count: 9), [3, 1, 3]),
        sh: MLXArray([Float](repeating: 0.2, count: 18), [3, 2, 3]),
        opacityLogits: FastGSOpacity.logits(fromProbabilities: MLXArray([Float(0.2), 0.9, 0.82], [3])),
        scales: MLXArray([Float](repeating: 0.3, count: 9), [3, 3]),
        rotations: MLXArray([
            Float(1), 0, 0, 0,
            1, 0, 0, 0,
            1, 0, 0, 0
        ], [3, 4]),
        cov3DPrecomputed: MLXArray([Float](repeating: 0.4, count: 18), [3, 6])
    )
}

private func makeAfterTrainingAdamState() -> FastGSAdamState {
    FastGSAdamState(
        step: 7,
        means3D: makeAfterTrainingAdamField(shape: [3, 3], start: 10),
        dc: makeAfterTrainingAdamField(shape: [3, 1, 3], start: 20),
        sh: makeAfterTrainingAdamField(shape: [3, 2, 3], start: 30),
        opacityLogits: makeAfterTrainingAdamField(shape: [3], start: 40),
        scales: makeAfterTrainingAdamField(shape: [3, 3], start: 50),
        rotations: makeAfterTrainingAdamField(shape: [3, 4], start: 60),
        cov3DPrecomputed: makeAfterTrainingAdamField(shape: [3, 6], start: 70)
    )
}

private func makeAfterTrainingAdamField(shape: [Int], start: Float) -> FastGSAdamFieldState {
    let count = shape.reduce(1, *)
    let first = (0..<count).map { start + Float($0) }
    let second = (0..<count).map { start + 1_000 + Float($0) }
    return FastGSAdamFieldState(
        firstMoment: MLXArray(first, shape),
        secondMoment: MLXArray(second, shape)
    )
}

private func makePruneOnlyParameters(opacityProbabilities: [Float]) -> FastGSTrainableParameters {
    let count = opacityProbabilities.count
    let scaleRows: [[Float]] = [
        [log(0.1), log(0.1), log(0.1)],
        [log(0.2), log(0.2), log(0.2)],
        [log(3.0), log(3.0), log(3.0)],
        [log(0.3), log(0.3), log(0.3)]
    ]
    let scales = (0..<count).flatMap { index in
        scaleRows[index % scaleRows.count]
    }
    return FastGSTrainableParameters(
        means3D: MLXArray((0..<(count * 3)).map { Float($0) }, [count, 3]),
        dc: MLXArray([Float](repeating: 0.1, count: count * 3), [count, 1, 3]),
        sh: MLXArray([Float](repeating: 0.2, count: count * 6), [count, 2, 3]),
        opacityLogits: FastGSOpacity.logits(fromProbabilities: MLXArray(opacityProbabilities, [count])),
        scales: MLXArray(scales, [count, 3]),
        rotations: MLXArray([Float](repeating: 0, count: count * 4), [count, 4]),
        cov3DPrecomputed: MLXArray([Float](repeating: 0.4, count: count * 6), [count, 6])
    )
}

private func assertAfterTrainingClose(
    _ actual: [Float],
    _ expected: [Float],
    accuracy: Float = 1e-5,
    file: StaticString = #filePath,
    line: UInt = #line
) {
    XCTAssertEqual(actual.count, expected.count, file: file, line: line)
    for (actualValue, expectedValue) in zip(actual, expected) {
        XCTAssertEqual(actualValue, expectedValue, accuracy: accuracy, file: file, line: line)
    }
}
