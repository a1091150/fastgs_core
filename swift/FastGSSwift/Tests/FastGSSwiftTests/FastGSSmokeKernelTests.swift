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

        let output = makePreprocessFixture()

        XCTAssertEqual(output.radii.shape, [2])
        XCTAssertEqual(output.xy.shape, [2, 2])
        XCTAssertEqual(output.depths.shape, [2])
        XCTAssertEqual(output.cov3D.shape, [2, 6])
        XCTAssertEqual(output.rgb.shape, [2, 3])
        XCTAssertEqual(output.conicOpacity.shape, [2, 4])
        XCTAssertEqual(output.tilesTouched.shape, [2])
        XCTAssertEqual(output.clamped.shape, [2, 3])
        XCTAssertEqual(output.viewspacePoints.shape, [2, 4])

        XCTAssertEqual(output.radii.asArray(Int32.self), [97, 102])
        XCTAssertEqual(output.depths.asArray(Float.self), [1, 1])
        XCTAssertEqual(output.rgb.asArray(Float.self), [1, 0, 0, 0, 1, 0])
        XCTAssertEqual(output.viewspacePoints.asArray(Float.self), [0, 0, 1, 7, 0.25, -0.25, 1, 9])
    }
}

func makePreprocessFixture() -> FastGSPreprocessOutput {
    let input = FastGSPreprocessInput(
        means3D: MLXArray([Float(0), 0, 1, 0.25, -0.25, 1], [2, 3]),
        dc: MLXArray([Float](repeating: 0, count: 6), [2, 3]),
        sh: MLXArray([Float](), [2, 0, 3]),
        colorsPrecomputed: MLXArray([Float(1), 0, 0, 0, 1, 0], [2, 3]),
        opacities: MLXArray([Float(1), 1], [2]),
        scales: MLXArray([Float(1), 1, 1, 1, 1, 1], [2, 3]),
        rotations: MLXArray([Float(1), 0, 0, 0, 1, 0, 0, 0], [2, 4]),
        cov3DPrecomputed: MLXArray([Float](), [0]),
        viewMatrix: MLXArray([
            Float(1), 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ], [4, 4]),
        projectionMatrix: MLXArray([
            Float(1), 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ], [4, 4]),
        cameraPosition: MLXArray([Float(0), 0, 0], [3]),
        viewspacePoints: MLXArray([Float(0), 0, 0, 7, 0, 0, 0, 9], [2, 4])
    )

    let params = FastGSPreprocessParams(
        degree: 0,
        maxSHCoefficients: 0,
        scaleModifier: 1,
        tanFovX: 1,
        tanFovY: 1,
        imageHeight: 64,
        imageWidth: 64,
        tileBounds: (x: 4, y: 4, z: 1),
        multiplier: 1,
        useColorsPrecomputed: true
    )

    return FastGSPreprocess.forward(input, params: params)
}
