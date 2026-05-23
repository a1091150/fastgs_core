import FastGSSwift
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
