import FastGSSwift
import MLX
import XCTest

final class FastGSSmokeKernelTests: XCTestCase {
    func testPackageLoads() {
        XCTAssertTrue(true)
    }

    func testDoubleKernel() throws {
        guard ProcessInfo.processInfo.environment["FASTGS_RUN_METAL_TESTS"] == "1" else {
            throw XCTSkip(
                "Set FASTGS_RUN_METAL_TESTS=1 when running from an Xcode/metallib-ready environment."
            )
        }

        let input = MLXArray([Float(1), Float(2), Float(3), Float(4)], [4])
        let output = FastGSSmokeKernel.double(input)

        XCTAssertEqual(output.shape, [4])
        XCTAssertEqual(output.dtype, .float32)
        XCTAssertEqual(output.asArray(Float.self), [2, 4, 6, 8])
    }

    func testPreprocessKernel() throws {
        guard ProcessInfo.processInfo.environment["FASTGS_RUN_METAL_TESTS"] == "1" else {
            throw XCTSkip("MLXFast Metal tests require an Xcode/metallib-ready environment.")
        }

        let output = FastGSPreprocessParityFixture.precomputedColorOutput()

        XCTAssertEqual(output.radii.shape, [2])
        XCTAssertEqual(output.xy.shape, [2, 2])
        XCTAssertEqual(output.depths.shape, [2])
        XCTAssertEqual(output.cov3D.shape, [2, 6])
        XCTAssertEqual(output.rgb.shape, [2, 3])
        XCTAssertEqual(output.conicOpacity.shape, [2, 4])
        XCTAssertEqual(output.tilesTouched.shape, [2])
        XCTAssertEqual(output.clamped.shape, [2, 3])
        XCTAssertEqual(output.viewspacePoints.shape, [2, 4])

        XCTAssertEqual(output.radii.asArray(Int32.self), FastGSPreprocessParityFixture.expectedRadii)
        XCTAssertEqual(output.tilesTouched.asArray(UInt32.self), FastGSPreprocessParityFixture.expectedTilesTouched)
        XCTAssertEqual(output.clamped.asArray(Bool.self), FastGSPreprocessParityFixture.expectedClamped)
        assertClose(output.xy.asArray(Float.self), FastGSPreprocessParityFixture.expectedXY)
        assertClose(output.depths.asArray(Float.self), FastGSPreprocessParityFixture.expectedDepths)
        assertClose(output.cov3D.asArray(Float.self), FastGSPreprocessParityFixture.expectedCov3D)
        assertClose(output.rgb.asArray(Float.self), FastGSPreprocessParityFixture.expectedRGB)
        assertClose(output.conicOpacity.asArray(Float.self), FastGSPreprocessParityFixture.expectedConicOpacity)
        assertClose(output.viewspacePoints.asArray(Float.self), FastGSPreprocessParityFixture.expectedViewspacePoints)
    }

    func testPreprocessSHDegree3Kernel() throws {
        guard ProcessInfo.processInfo.environment["FASTGS_RUN_METAL_TESTS"] == "1" else {
            throw XCTSkip("MLXFast Metal tests require an Xcode/metallib-ready environment.")
        }

        let output = FastGSPreprocessParityFixture.shDegree3Output()

        XCTAssertEqual(output.radii.asArray(Int32.self), FastGSPreprocessParityFixture.expectedRadii)
        XCTAssertEqual(output.tilesTouched.asArray(UInt32.self), FastGSPreprocessParityFixture.expectedTilesTouched)
        XCTAssertEqual(output.clamped.asArray(Bool.self), FastGSPreprocessParityFixture.expectedClamped)
        assertClose(output.xy.asArray(Float.self), FastGSPreprocessParityFixture.expectedXY)
        assertClose(output.depths.asArray(Float.self), FastGSPreprocessParityFixture.expectedDepths)
        assertClose(output.cov3D.asArray(Float.self), FastGSPreprocessParityFixture.expectedCov3D)
        assertClose(output.rgb.asArray(Float.self), FastGSPreprocessParityFixture.expectedSHDegree3RGB)
        assertClose(output.conicOpacity.asArray(Float.self), FastGSPreprocessParityFixture.expectedConicOpacity)
        assertClose(output.viewspacePoints.asArray(Float.self), FastGSPreprocessParityFixture.expectedViewspacePoints)
    }

    func testPreprocessCullingKernel() throws {
        guard ProcessInfo.processInfo.environment["FASTGS_RUN_METAL_TESTS"] == "1" else {
            throw XCTSkip("MLXFast Metal tests require an Xcode/metallib-ready environment.")
        }

        let output = FastGSPreprocessParityFixture.cullingOutput()

        XCTAssertEqual(output.radii.asArray(Int32.self), FastGSPreprocessParityFixture.expectedCullingRadii)
        XCTAssertEqual(output.tilesTouched.asArray(UInt32.self), FastGSPreprocessParityFixture.expectedCullingTilesTouched)
        XCTAssertEqual(output.clamped.asArray(Bool.self), FastGSPreprocessParityFixture.expectedClamped)
        assertClose(output.xy.asArray(Float.self), FastGSPreprocessParityFixture.expectedCullingXY)
        assertClose(output.depths.asArray(Float.self), FastGSPreprocessParityFixture.expectedCullingDepths)
        assertClose(output.cov3D.asArray(Float.self), FastGSPreprocessParityFixture.expectedCullingCov3D)
        assertClose(output.rgb.asArray(Float.self), FastGSPreprocessParityFixture.expectedCullingRGB)
        assertClose(output.conicOpacity.asArray(Float.self), FastGSPreprocessParityFixture.expectedCullingConicOpacity)
        assertClose(output.viewspacePoints.asArray(Float.self), FastGSPreprocessParityFixture.expectedCullingViewspacePoints)
    }

    func testPreprocessCov3DPrecomputedKernel() throws {
        guard ProcessInfo.processInfo.environment["FASTGS_RUN_METAL_TESTS"] == "1" else {
            throw XCTSkip("MLXFast Metal tests require an Xcode/metallib-ready environment.")
        }

        let output = FastGSPreprocessParityFixture.cov3DPrecomputedOutput()

        XCTAssertEqual(output.radii.asArray(Int32.self), FastGSPreprocessParityFixture.expectedCov3DPrecomputedRadii)
        XCTAssertEqual(output.tilesTouched.asArray(UInt32.self), FastGSPreprocessParityFixture.expectedTilesTouched)
        XCTAssertEqual(output.clamped.asArray(Bool.self), FastGSPreprocessParityFixture.expectedClamped)
        assertClose(output.xy.asArray(Float.self), FastGSPreprocessParityFixture.expectedXY)
        assertClose(output.depths.asArray(Float.self), FastGSPreprocessParityFixture.expectedDepths)
        assertClose(output.cov3D.asArray(Float.self), FastGSPreprocessParityFixture.expectedCov3DPrecomputedCov3D)
        assertClose(output.rgb.asArray(Float.self), FastGSPreprocessParityFixture.expectedRGB)
        assertClose(output.conicOpacity.asArray(Float.self), FastGSPreprocessParityFixture.expectedCov3DPrecomputedConicOpacity)
        assertClose(output.viewspacePoints.asArray(Float.self), FastGSPreprocessParityFixture.expectedViewspacePoints)
    }

    func testPreprocessSHClampKernel() throws {
        guard ProcessInfo.processInfo.environment["FASTGS_RUN_METAL_TESTS"] == "1" else {
            throw XCTSkip("MLXFast Metal tests require an Xcode/metallib-ready environment.")
        }

        let output = FastGSPreprocessParityFixture.shClampOutput()

        XCTAssertEqual(output.radii.asArray(Int32.self), FastGSPreprocessParityFixture.expectedRadii)
        XCTAssertEqual(output.tilesTouched.asArray(UInt32.self), FastGSPreprocessParityFixture.expectedTilesTouched)
        XCTAssertEqual(output.clamped.asArray(Bool.self), FastGSPreprocessParityFixture.expectedSHClampClamped)
        assertClose(output.xy.asArray(Float.self), FastGSPreprocessParityFixture.expectedXY)
        assertClose(output.depths.asArray(Float.self), FastGSPreprocessParityFixture.expectedDepths)
        assertClose(output.cov3D.asArray(Float.self), FastGSPreprocessParityFixture.expectedCov3D)
        assertClose(output.rgb.asArray(Float.self), FastGSPreprocessParityFixture.expectedSHClampRGB)
        assertClose(output.conicOpacity.asArray(Float.self), FastGSPreprocessParityFixture.expectedConicOpacity)
        assertClose(output.viewspacePoints.asArray(Float.self), FastGSPreprocessParityFixture.expectedViewspacePoints)
    }
}

private func assertClose(
    _ actual: [Float],
    _ expected: [Float],
    accuracy: Float = 1e-5,
    file: StaticString = #filePath,
    line: UInt = #line
) {
    XCTAssertEqual(actual.count, expected.count, file: file, line: line)
    for (index, pair) in zip(actual, expected).enumerated() {
        XCTAssertEqual(pair.0, pair.1, accuracy: accuracy, "Mismatch at index \(index)", file: file, line: line)
    }
}
