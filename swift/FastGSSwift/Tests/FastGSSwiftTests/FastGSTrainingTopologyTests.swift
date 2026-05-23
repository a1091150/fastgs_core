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
        XCTAssertEqual(taken.opacities.shape, [2])
        XCTAssertEqual(taken.scales.shape, [2, 3])
        XCTAssertEqual(taken.rotations.shape, [2, 4])
        XCTAssertEqual(taken.cov3DPrecomputed?.shape, [2, 6])
        assertTopologyClose(taken.means3D.asArray(Float.self), [20, 21, 22, 0, 1, 2])
        assertTopologyClose(taken.opacities.asArray(Float.self), [102, 100])
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
        assertTopologyClose(pruned.opacities.asArray(Float.self), [100, 102])
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
        assertTopologyClose(appended.opacities.asArray(Float.self), [100, 101, 1_100])
    }
}

private func makeTopologyParameters(count: Int, offset: Float = 0) -> FastGSTrainableParameters {
    FastGSTrainableParameters(
        means3D: MLXArray(makeTopologyRows(count: count, width: 3, offset: offset), [count, 3]),
        dc: MLXArray(makeTopologyRows(count: count, width: 3, offset: offset + 100), [count, 1, 3]),
        sh: MLXArray(makeTopologyRows(count: count, width: 6, offset: offset + 200), [count, 2, 3]),
        opacities: MLXArray((0..<count).map { offset + 100 + Float($0) }, [count]),
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
