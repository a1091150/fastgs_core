import FastGSSwift
import MLX
import XCTest

final class FastGSTrainingTopologyTests: XCTestCase {
    func testTrainableParametersTakeRows() throws {
        guard ProcessInfo.processInfo.environment["FASTGS_RUN_METAL_TESTS"] == "1" else {
            throw XCTSkip("MLX topology tests require an Xcode/metallib-ready environment.")
        }

        let parameters = makeTopologyParameters(count: 4)
        let taken = parameters.take(indices: [2, 0])

        XCTAssertEqual(taken.gaussianCount, 2)
        XCTAssertEqual(taken.means3D.shape, [2, 3])
        XCTAssertEqual(taken.dc.shape, [2, 1, 3])
        XCTAssertEqual(taken.sh.shape, [2, 2, 3])
        XCTAssertEqual(taken.opacityLogits.shape, [2])
        XCTAssertEqual(taken.scales.shape, [2, 3])
        XCTAssertEqual(taken.rotations.shape, [2, 4])
        XCTAssertEqual(taken.cov3DPrecomputed?.shape, [2, 6])
        assertTopologyClose(taken.means3D.asArray(Float.self), [20, 21, 22, 0, 1, 2])
        assertTopologyClose(taken.opacityLogits.asArray(Float.self), [102, 100])
        assertTopologyClose(taken.cov3DPrecomputed?.asArray(Float.self) ?? [], [
            500 + 12, 500 + 13, 500 + 14, 500 + 15, 500 + 16, 500 + 17,
            500, 501, 502, 503, 504, 505
        ])
    }

    func testTrainableParametersPruneRows() throws {
        guard ProcessInfo.processInfo.environment["FASTGS_RUN_METAL_TESTS"] == "1" else {
            throw XCTSkip("MLX topology tests require an Xcode/metallib-ready environment.")
        }

        let parameters = makeTopologyParameters(count: 4)
        let pruned = parameters.prune(mask: [false, true, false, true])

        XCTAssertEqual(pruned.gaussianCount, 2)
        assertTopologyClose(pruned.means3D.asArray(Float.self), [0, 1, 2, 20, 21, 22])
        assertTopologyClose(pruned.opacityLogits.asArray(Float.self), [100, 102])
    }

    func testTrainableParametersAppendRows() throws {
        guard ProcessInfo.processInfo.environment["FASTGS_RUN_METAL_TESTS"] == "1" else {
            throw XCTSkip("MLX topology tests require an Xcode/metallib-ready environment.")
        }

        let head = makeTopologyParameters(count: 2, offset: 0)
        let tail = makeTopologyParameters(count: 1, offset: 1_000)
        let appended = head.appending(tail)

        XCTAssertEqual(appended.gaussianCount, 3)
        XCTAssertEqual(appended.means3D.shape, [3, 3])
        XCTAssertEqual(appended.dc.shape, [3, 1, 3])
        XCTAssertEqual(appended.sh.shape, [3, 2, 3])
        XCTAssertEqual(appended.cov3DPrecomputed?.shape, [3, 6])
        assertTopologyClose(appended.means3D.asArray(Float.self), [
            0, 1, 2,
            10, 11, 12,
            1_000, 1_001, 1_002
        ])
        assertTopologyClose(appended.opacityLogits.asArray(Float.self), [100, 101, 1_100])
    }

    func testAdamStatePruneRows() throws {
        guard ProcessInfo.processInfo.environment["FASTGS_RUN_METAL_TESTS"] == "1" else {
            throw XCTSkip("MLX topology tests require an Xcode/metallib-ready environment.")
        }

        let state = makeAdamTopologyState(count: 4, step: 12)
        let pruned = state.prune(mask: [false, true, false, true])

        XCTAssertEqual(pruned.step, 12)
        XCTAssertEqual(pruned.means3D.firstMoment.shape, [2, 3])
        XCTAssertEqual(pruned.opacityLogits.firstMoment.shape, [2])
        XCTAssertEqual(pruned.cov3DPrecomputed?.firstMoment.shape, [2, 6])
        assertTopologyClose(pruned.means3D.firstMoment.asArray(Float.self), [0, 1, 2, 20, 21, 22])
        assertTopologyClose(pruned.means3D.secondMoment.asArray(Float.self), [1_000, 1_001, 1_002, 1_020, 1_021, 1_022])
        assertTopologyClose(pruned.opacityLogits.firstMoment.asArray(Float.self), [100, 102])
    }

    func testAdamStateAppendsZeroRows() throws {
        guard ProcessInfo.processInfo.environment["FASTGS_RUN_METAL_TESTS"] == "1" else {
            throw XCTSkip("MLX topology tests require an Xcode/metallib-ready environment.")
        }

        let state = makeAdamTopologyState(count: 2, step: 18)
        let tailParameters = makeTopologyParameters(count: 1, offset: 1_000)
        let appended = state.appendingZeroRows(like: tailParameters)

        XCTAssertEqual(appended.step, 18)
        XCTAssertEqual(appended.means3D.firstMoment.shape, [3, 3])
        XCTAssertEqual(appended.cov3DPrecomputed?.firstMoment.shape, [3, 6])
        assertTopologyClose(appended.means3D.firstMoment.asArray(Float.self), [
            0, 1, 2,
            10, 11, 12,
            0, 0, 0
        ])
        assertTopologyClose(appended.means3D.secondMoment.asArray(Float.self), [
            1_000, 1_001, 1_002,
            1_010, 1_011, 1_012,
            0, 0, 0
        ])
        assertTopologyClose(appended.opacityLogits.firstMoment.asArray(Float.self), [100, 101, 0])
    }

    func testAdamStateResettingOpacityStateOnlyClearsOpacityMoments() throws {
        guard ProcessInfo.processInfo.environment["FASTGS_RUN_METAL_TESTS"] == "1" else {
            throw XCTSkip("MLX topology tests require an Xcode/metallib-ready environment.")
        }

        let parameters = makeTopologyParameters(count: 2)
        let state = makeAdamTopologyState(count: 2, step: 21)
        let reset = state.resettingOpacityLogitState(like: parameters)

        XCTAssertEqual(reset.step, 21)
        assertTopologyClose(reset.means3D.firstMoment.asArray(Float.self), [0, 1, 2, 10, 11, 12])
        assertTopologyClose(reset.opacityLogits.firstMoment.asArray(Float.self), [0, 0])
        assertTopologyClose(reset.opacityLogits.secondMoment.asArray(Float.self), [0, 0])
        assertTopologyClose(reset.scales.secondMoment.asArray(Float.self), [
            1_300, 1_301, 1_302,
            1_310, 1_311, 1_312
        ])
    }
}

private func makeTopologyParameters(count: Int, offset: Float = 0) -> FastGSTrainableParameters {
    FastGSTrainableParameters(
        means3D: MLXArray(makeTopologyRows(count: count, width: 3, offset: offset), [count, 3]),
        dc: MLXArray(makeTopologyRows(count: count, width: 3, offset: offset + 100), [count, 1, 3]),
        sh: MLXArray(makeTopologyRows(count: count, width: 6, offset: offset + 200), [count, 2, 3]),
        opacityLogits: MLXArray((0..<count).map { offset + 100 + Float($0) }, [count]),
        scales: MLXArray(makeTopologyRows(count: count, width: 3, offset: offset + 300), [count, 3]),
        rotations: MLXArray(makeTopologyRows(count: count, width: 4, offset: offset + 400), [count, 4]),
        cov3DPrecomputed: MLXArray(makeTopologyRows(count: count, width: 6, offset: offset + 500), [count, 6])
    )
}

private func makeTopologyRows(count: Int, width: Int, offset: Float) -> [Float] {
    (0..<count).flatMap { row in
        (0..<width).map { column in
            offset + Float(row * 10 + column)
        }
    }
}

private func makeAdamTopologyState(count: Int, step: Int) -> FastGSAdamState {
    FastGSAdamState(
        step: step,
        means3D: makeAdamField(count: count, shape: [count, 3], width: 3, offset: 0),
        dc: makeAdamField(count: count, shape: [count, 1, 3], width: 3, offset: 100),
        sh: makeAdamField(count: count, shape: [count, 2, 3], width: 6, offset: 200),
        opacityLogits: makeAdamField(count: count, shape: [count], width: 1, offset: 100),
        scales: makeAdamField(count: count, shape: [count, 3], width: 3, offset: 300),
        rotations: makeAdamField(count: count, shape: [count, 4], width: 4, offset: 400),
        cov3DPrecomputed: makeAdamField(count: count, shape: [count, 6], width: 6, offset: 500)
    )
}

private func makeAdamField(count: Int, shape: [Int], width: Int, offset: Float) -> FastGSAdamFieldState {
    FastGSAdamFieldState(
        firstMoment: MLXArray(makeTopologyRows(count: count, width: width, offset: offset), shape),
        secondMoment: MLXArray(makeTopologyRows(count: count, width: width, offset: offset + 1_000), shape)
    )
}

private func assertTopologyClose(
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
