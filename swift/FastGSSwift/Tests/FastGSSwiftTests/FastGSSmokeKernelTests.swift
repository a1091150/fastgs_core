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
}
