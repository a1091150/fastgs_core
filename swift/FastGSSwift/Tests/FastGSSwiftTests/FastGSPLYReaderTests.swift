import FastGSSwift
import XCTest

final class FastGSPLYReaderTests: XCTestCase {
    func testReadsASCIIPositionsAndColors() throws {
        let text = """
        ply
        format ascii 1.0
        element vertex 2
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        property float nx
        end_header
        1.0 2.0 3.0 255 128 0 0.1
        -1.5 0.25 4.5 10 20 30 0.2
        """

        let pointCloud = try FastGSPLYReader.readPointCloud(text: text)

        XCTAssertEqual(pointCloud.count, 2)
        XCTAssertEqual(pointCloud.points, [1, 2, 3, -1.5, 0.25, 4.5])
        assertClose(pointCloud.colors ?? [], [1, 128.0 / 255.0, 0, 10.0 / 255.0, 20.0 / 255.0, 30.0 / 255.0])
    }

    func testReadsFixedScannerPLYHeaderAndSamples() throws {
        let url = URL(fileURLWithPath: "/Users/yangdunfu/Downloads/2026_05_04_16_51_29/points.ply")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("Fixed scanner dataset is not available at \(url.path).")
        }

        let pointCloud = try FastGSPLYReader.readPointCloud(url: url)

        XCTAssertEqual(pointCloud.count, 793602)
        assertClose(Array(pointCloud.points.prefix(9)), [
            0.051589999,
            -0.277200013,
            -1.207640052,
            0.077950001,
            -0.286089987,
            -1.237779975,
            0.104570001,
            -0.293579996,
            -1.262359977,
        ], accuracy: 1e-6)
        assertClose(Array((pointCloud.colors ?? []).prefix(9)), [
            208.0 / 255.0,
            203.0 / 255.0,
            180.0 / 255.0,
            212.0 / 255.0,
            207.0 / 255.0,
            183.0 / 255.0,
            215.0 / 255.0,
            210.0 / 255.0,
            185.0 / 255.0,
        ], accuracy: 1e-6)
    }
}

private func assertClose(
    _ actual: [Float],
    _ expected: [Float],
    accuracy: Float = 1e-6,
    file: StaticString = #filePath,
    line: UInt = #line
) {
    XCTAssertEqual(actual.count, expected.count, file: file, line: line)
    for (actualValue, expectedValue) in zip(actual, expected) {
        XCTAssertEqual(actualValue, expectedValue, accuracy: accuracy, file: file, line: line)
    }
}
