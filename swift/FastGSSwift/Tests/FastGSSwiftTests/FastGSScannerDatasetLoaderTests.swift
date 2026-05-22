import FastGSSwift
import XCTest

final class FastGSScannerDatasetLoaderTests: XCTestCase {
    func testLoadsFixedScannerDatasetFirstFrame() throws {
        let datasetURL = URL(fileURLWithPath: "/Users/yangdunfu/Downloads/2026_05_04_16_51_29", isDirectory: true)
        guard FileManager.default.fileExists(atPath: datasetURL.appendingPathComponent("points.ply").path) else {
            throw XCTSkip("Fixed scanner dataset is not available at \(datasetURL.path).")
        }

        let dataset = try FastGSScannerDatasetLoader.load(
            directory: datasetURL,
            options: FastGSScannerDatasetOptions(width: 512, height: 512, maxFrames: 8)
        )

        XCTAssertEqual(dataset.pointCloud.count, 793602)
        XCTAssertEqual(dataset.basePointCount, 793602)
        XCTAssertEqual(dataset.frames.count, 8)
        XCTAssertEqual(dataset.frames[0].index, 0)
        XCTAssertEqual(dataset.frames[0].camera.imageWidth, 512)
        XCTAssertEqual(dataset.frames[0].camera.imageHeight, 512)
        XCTAssertEqual(dataset.frames[0].targetCHW.count, 3 * 512 * 512)
        XCTAssertTrue((dataset.frames[0].targetCHW.min() ?? -1) >= 0)
        XCTAssertTrue((dataset.frames[0].targetCHW.max() ?? 2) <= 1)

        assertClose(Array(dataset.pointCloud.points.prefix(9)), [
            0.233708397,
            -6.333032131,
            2.322546005,
            0.372316390,
            -6.491516113,
            2.369292021,
            0.512291491,
            -6.620764256,
            2.408676386,
        ], accuracy: 1e-5)
        assertClose(Array((dataset.pointCloud.colors ?? []).prefix(9)), [
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

    func testFixedScannerCameraMatchesRecordedManifest() throws {
        let datasetURL = URL(fileURLWithPath: "/Users/yangdunfu/Downloads/2026_05_04_16_51_29", isDirectory: true)
        let manifestURL = URL(fileURLWithPath: "/private/tmp/fastgs_recorded_reference_full_512/camera_000/recorded_manifest.json")
        guard FileManager.default.fileExists(atPath: datasetURL.appendingPathComponent("points.ply").path) else {
            throw XCTSkip("Fixed scanner dataset is not available at \(datasetURL.path).")
        }
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw XCTSkip("Recorded reference manifest is not available at \(manifestURL.path).")
        }

        let dataset = try FastGSScannerDatasetLoader.load(
            directory: datasetURL,
            options: FastGSScannerDatasetOptions(width: 512, height: 512, maxFrames: 8)
        )
        let manifest = try JSONDecoder().decode(
            FastGSRecordedForwardManifest.self,
            from: Data(contentsOf: manifestURL)
        )
        let camera = dataset.frames[0].camera

        XCTAssertEqual(camera.imageWidth, manifest.width)
        XCTAssertEqual(camera.imageHeight, manifest.height)
        XCTAssertEqual(dataset.pointCloud.count, manifest.pointCount)
        XCTAssertEqual(manifest.shDegree, 3)
        XCTAssertEqual(camera.tanFovX, Float(manifest.tanFovX), accuracy: 1e-6)
        XCTAssertEqual(camera.tanFovY, Float(manifest.tanFovY), accuracy: 1e-6)
        assertClose(camera.campos, manifest.campos.map(Float.init), accuracy: 1e-5)
        assertClose(camera.viewmatrix, manifest.viewmatrix.map(Float.init), accuracy: 1e-5)
        assertClose(camera.projmatrix, manifest.projmatrix.map(Float.init), accuracy: 1e-5)
    }

    func testFixedScannerCameraSwitchPerformanceReport() throws {
        guard ProcessInfo.processInfo.environment["FASTGS_RUN_PERF_REPORT"] == "1" else {
            throw XCTSkip("Set FASTGS_RUN_PERF_REPORT=1 to print scanner switch timing diagnostics.")
        }

        let datasetURL = URL(fileURLWithPath: "/Users/yangdunfu/Downloads/2026_05_04_16_51_29", isDirectory: true)
        guard FileManager.default.fileExists(atPath: datasetURL.appendingPathComponent("points.ply").path) else {
            throw XCTSkip("Fixed scanner dataset is not available at \(datasetURL.path).")
        }

        var report = FastGSScannerTimingReport()

        let contents = try report.measure("directory scan") {
            try FileManager.default.contentsOfDirectory(at: datasetURL, includingPropertiesForKeys: nil)
        }
        let framePairCount = Set(contents.compactMap(scannerFrameIndex)).count

        let pointCloud = try report.measure("points.ply read") {
            try FastGSPLYReader.readPointCloud(url: datasetURL.appendingPathComponent("points.ply"))
        }

        let dataset = try report.measure("loader maxFrames=1") {
            try FastGSScannerDatasetLoader.load(
                directory: datasetURL,
                options: FastGSScannerDatasetOptions(
                    width: 512,
                    height: 512,
                    maxFrames: 1,
                    startIndex: 0,
                    normalizeWithAllFramePairs: true
                )
            )
        }
        let cache = try report.measure("cache load once") {
            try FastGSScannerDatasetLoader.loadCache(
                directory: datasetURL,
                options: FastGSScannerDatasetOptions(width: 512, height: 512, normalizeWithAllFramePairs: true)
            )
        }
        _ = try report.measure("cached frame load") {
            try FastGSScannerDatasetLoader.loadDataset(
                cache: cache,
                frameIndex: cache.frameDescriptors[min(1, cache.frameDescriptors.count - 1)].index,
                width: 512,
                height: 512
            )
        }

        let scene = report.measure("scene init") {
            FastGSRecordedForwardScene(scannerDataset: dataset, frameIndex: 0)
        }

        if ProcessInfo.processInfo.environment["FASTGS_RUN_METAL_TESTS"] == "1" {
            let parameters = try report.measure("initial parameters") {
                try scene.initialTrainableParameters()
            }
            _ = try report.measure("swift render") {
                try scene.render(parameters: parameters).outColor.eval()
            }
        }

        print(
            """

            FastGS scanner switch timing report
            dataset: \(datasetURL.path)
            directory entries: \(contents.count)
            frame-like entries: \(framePairCount)
            point count: \(pointCloud.count)
            loaded frames: \(dataset.frames.count)
            \(report.description)
            """
        )
    }
}

private struct FastGSScannerTimingReport {
    private var rows = [(String, TimeInterval)]()

    mutating func measure<T>(_ name: String, _ work: () throws -> T) rethrows -> T {
        let start = Date()
        let value = try work()
        rows.append((name, Date().timeIntervalSince(start)))
        return value
    }

    var description: String {
        rows
            .map { name, seconds in
                "\(name.padding(toLength: 22, withPad: " ", startingAt: 0)) \(String(format: "%.3f", seconds)) s"
            }
            .joined(separator: "\n")
    }
}

private func scannerFrameIndex(_ url: URL) -> Int? {
    let stem = url.deletingPathExtension().lastPathComponent
    guard stem.hasPrefix("frame_") else {
        return nil
    }
    return Int(stem.dropFirst("frame_".count))
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
