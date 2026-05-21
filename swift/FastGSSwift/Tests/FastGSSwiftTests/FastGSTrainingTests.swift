import FastGSSwift
import MLX
import XCTest

final class FastGSTrainingTests: XCTestCase {
    func testAdamOptimizerAppliesSyntheticGradientStep() throws {
        guard ProcessInfo.processInfo.environment["FASTGS_RUN_METAL_TESTS"] == "1" else {
            throw XCTSkip("MLX array optimizer tests require an Xcode/metallib-ready environment.")
        }

        let parameters = FastGSTrainableParameters(
            means3D: MLXArray([Float(1), 2, 3, 4, 5, 6], [2, 3]),
            dc: MLXArray([Float(0.2), 0.4, 0.6, 0.8, 1.0, 1.2], [2, 1, 3]),
            sh: MLXArray([Float](repeating: 0.1, count: 12), [2, 2, 3]),
            opacities: MLXArray([Float(0.5), 0.7], [2]),
            scales: MLXArray([Float](repeating: 0.3, count: 6), [2, 3]),
            rotations: MLXArray([Float(1), 0, 0, 0, 1, 0, 0, 0], [2, 4]),
            cov3DPrecomputed: MLXArray([Float](repeating: 0.05, count: 12), [2, 6])
        )
        let gradients = FastGSTrainableGradients(
            means3D: MLXArray([Float](repeating: 1, count: 6), [2, 3]),
            dc: MLXArray([Float](repeating: -1, count: 6), [2, 1, 3]),
            sh: MLXArray([Float](repeating: 0.5, count: 12), [2, 2, 3]),
            opacities: MLXArray([Float](repeating: 1, count: 2), [2]),
            scales: MLXArray([Float](repeating: -1, count: 6), [2, 3]),
            rotations: MLXArray([Float](repeating: 1, count: 8), [2, 4]),
            cov3DPrecomputed: MLXArray([Float](repeating: -1, count: 12), [2, 6])
        )

        var optimizer = FastGSAdamOptimizer(
            learningRates: FastGSAdamLearningRates(
                means3D: 0.01,
                dc: 0.02,
                sh: 0.03,
                opacities: 0.04,
                scales: 0.05,
                rotations: 0.06,
                cov3DPrecomputed: 0.07
            )
        )

        let updated = optimizer.update(parameters: parameters, gradients: gradients)

        XCTAssertEqual(updated.means3D.shape, parameters.means3D.shape)
        XCTAssertEqual(updated.dc.shape, parameters.dc.shape)
        XCTAssertEqual(updated.sh.shape, parameters.sh.shape)
        XCTAssertEqual(updated.opacities.shape, parameters.opacities.shape)
        XCTAssertEqual(updated.scales.shape, parameters.scales.shape)
        XCTAssertEqual(updated.rotations.shape, parameters.rotations.shape)
        XCTAssertEqual(updated.cov3DPrecomputed?.shape, parameters.cov3DPrecomputed?.shape)

        assertTrainingClose(updated.means3D.asArray(Float.self), [0.99, 1.99, 2.99, 3.99, 4.99, 5.99])
        assertTrainingClose(updated.dc.asArray(Float.self), [0.22, 0.42, 0.62, 0.82, 1.02, 1.22])
        assertTrainingClose(updated.sh.asArray(Float.self), [Float](repeating: 0.07, count: 12))
        assertTrainingClose(updated.opacities.asArray(Float.self), [0.46, 0.66])
        assertTrainingClose(updated.scales.asArray(Float.self), [Float](repeating: 0.35, count: 6))
        assertTrainingClose(updated.rotations.asArray(Float.self), [0.94, -0.06, -0.06, -0.06, 0.94, -0.06, -0.06, -0.06])
        assertTrainingClose(updated.cov3DPrecomputed?.asArray(Float.self) ?? [], [Float](repeating: 0.12, count: 12))

        XCTAssertEqual(optimizer.state?.step, 1)
        XCTAssertEqual(optimizer.stateArrays().count, 14)
    }

    func testAdamOptimizerKeepsOptionalCovarianceAbsent() throws {
        guard ProcessInfo.processInfo.environment["FASTGS_RUN_METAL_TESTS"] == "1" else {
            throw XCTSkip("MLX array optimizer tests require an Xcode/metallib-ready environment.")
        }

        let parameters = FastGSTrainableParameters(
            means3D: MLXArray([Float](repeating: 1, count: 3), [1, 3]),
            dc: MLXArray([Float](repeating: 1, count: 3), [1, 1, 3]),
            sh: MLXArray([Float](repeating: 1, count: 3), [1, 1, 3]),
            opacities: MLXArray([Float](repeating: 1, count: 1), [1]),
            scales: MLXArray([Float](repeating: 1, count: 3), [1, 3]),
            rotations: MLXArray([Float](repeating: 1, count: 4), [1, 4])
        )
        let gradients = FastGSTrainableGradients(
            means3D: MLXArray([Float](repeating: 1, count: 3), [1, 3]),
            dc: MLXArray([Float](repeating: 1, count: 3), [1, 1, 3]),
            sh: MLXArray([Float](repeating: 1, count: 3), [1, 1, 3]),
            opacities: MLXArray([Float](repeating: 1, count: 1), [1]),
            scales: MLXArray([Float](repeating: 1, count: 3), [1, 3]),
            rotations: MLXArray([Float](repeating: 1, count: 4), [1, 4])
        )

        var optimizer = FastGSAdamOptimizer()
        let updated = optimizer.update(parameters: parameters, gradients: gradients)

        XCTAssertNil(updated.cov3DPrecomputed)
        XCTAssertNil(optimizer.state?.cov3DPrecomputed)
        XCTAssertEqual(optimizer.stateArrays().count, 12)
    }
}

private func assertTrainingClose(
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
