import FastGSSwift
import CoreVideo
import Metal
import MLX
import XCTest

final class FastGSSmokeXcodeTests: XCTestCase {
    private let recordedManifestURL = URL(fileURLWithPath: "/private/tmp/fastgs_recorded_reference/recorded_manifest.json")
    private let recordedLargeManifestURL = URL(fileURLWithPath: "/private/tmp/fastgs_recorded_reference_16384/recorded_manifest.json")
    private let recordedFullManifestURL = URL(fileURLWithPath: "/private/tmp/fastgs_recorded_reference_full_512/recorded_manifest.json")
    private let rasterizeBackwardReferenceURL = URL(fileURLWithPath: "/private/tmp/fastgs_rasterize_backward_ref.json")
    private let preprocessBackwardReferenceURL = URL(fileURLWithPath: "/private/tmp/fastgs_preprocess_backward_ref.json")

    func testAdamOptimizerAppliesSyntheticGradientStepUnderXcode() {
        let parameters = FastGSTrainableParameters(
            means3D: MLXArray([Float(1), 2, 3, 4, 5, 6], [2, 3]),
            dc: MLXArray([Float(0.2), 0.4, 0.6, 0.8, 1.0, 1.2], [2, 1, 3]),
            sh: MLXArray([Float](repeating: 0.1, count: 12), [2, 2, 3]),
            opacityLogits: MLXArray([Float(0.5), 0.7], [2]),
            scales: MLXArray([Float](repeating: 0.3, count: 6), [2, 3]),
            rotations: MLXArray([Float(1), 0, 0, 0, 1, 0, 0, 0], [2, 4]),
            cov3DPrecomputed: MLXArray([Float](repeating: 0.05, count: 12), [2, 6])
        )
        let gradients = FastGSTrainableGradients(
            means3D: MLXArray([Float](repeating: 1, count: 6), [2, 3]),
            dc: MLXArray([Float](repeating: -1, count: 6), [2, 1, 3]),
            sh: MLXArray([Float](repeating: 0.5, count: 12), [2, 2, 3]),
            opacityLogits: MLXArray([Float](repeating: 1, count: 2), [2]),
            scales: MLXArray([Float](repeating: -1, count: 6), [2, 3]),
            rotations: MLXArray([Float](repeating: 1, count: 8), [2, 4]),
            cov3DPrecomputed: MLXArray([Float](repeating: -1, count: 12), [2, 6])
        )

        var optimizer = FastGSAdamOptimizer(
            learningRates: FastGSAdamLearningRates(
                means3D: 0.01,
                dc: 0.02,
                sh: 0.03,
                opacityLogits: 0.04,
                scales: 0.05,
                rotations: 0.06,
                cov3DPrecomputed: 0.07
            )
        )

        let updated = optimizer.update(parameters: parameters, gradients: gradients)

        XCTAssertEqual(updated.means3D.shape, parameters.means3D.shape)
        XCTAssertEqual(updated.dc.shape, parameters.dc.shape)
        XCTAssertEqual(updated.sh.shape, parameters.sh.shape)
        XCTAssertEqual(updated.opacityLogits.shape, parameters.opacityLogits.shape)
        XCTAssertEqual(updated.scales.shape, parameters.scales.shape)
        XCTAssertEqual(updated.rotations.shape, parameters.rotations.shape)
        XCTAssertEqual(updated.cov3DPrecomputed?.shape, parameters.cov3DPrecomputed?.shape)

        assertClose(updated.means3D.asArray(Float.self), [0.99, 1.99, 2.99, 3.99, 4.99, 5.99])
        assertClose(updated.dc.asArray(Float.self), [0.22, 0.42, 0.62, 0.82, 1.02, 1.22])
        assertClose(updated.sh.asArray(Float.self), [Float](repeating: 0.07, count: 12))
        assertClose(updated.opacityLogits.asArray(Float.self), [0.46, 0.66])
        assertClose(updated.scales.asArray(Float.self), [Float](repeating: 0.35, count: 6))
        assertClose(updated.rotations.asArray(Float.self), [0.94, -0.06, -0.06, -0.06, 0.94, -0.06, -0.06, -0.06])
        assertClose(updated.cov3DPrecomputed?.asArray(Float.self) ?? [], [Float](repeating: 0.12, count: 12))

        XCTAssertEqual(optimizer.state?.step, 1)
        XCTAssertEqual(optimizer.stateArrays().count, 14)
    }

    func testFinalPruneUsesScoresAndKeepsHighestScoredRowsUnderXcode() {
        let parameters = FastGSTrainableParameters(
            means3D: MLXArray((0..<12).map { Float($0) }, [4, 3]),
            dc: MLXArray([Float](repeating: 0.1, count: 12), [4, 1, 3]),
            sh: MLXArray([Float](repeating: 0.2, count: 24), [4, 2, 3]),
            opacityLogits: FastGSOpacity.logits(fromProbabilities: MLXArray([Float(0.2), 0.3, 0.4, 0.5], [4])),
            scales: MLXArray([Float](repeating: log(0.1), count: 12), [4, 3]),
            rotations: MLXArray([
                Float(1), 0, 0, 0,
                1, 0, 0, 0,
                1, 0, 0, 0,
                1, 0, 0, 0,
            ], [4, 4]),
            cov3DPrecomputed: MLXArray([Float](repeating: 0.4, count: 24), [4, 6])
        )
        let optimizerState = FastGSAdamState(step: 19, parameters: parameters)
        var densificationState = FastGSDensificationState(count: 4, sceneExtent: 10)
        densificationState.xyzGradAccum = [10, 20, 30, 40]
        densificationState.xyzGradAccumAbs = [11, 21, 31, 41]
        densificationState.denom = [1, 2, 3, 4]

        let result = FastGSAfterTraining.finalPrune(
            parameters: parameters,
            optimizerState: optimizerState,
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
        assertClose(result.parameters.opacityProbabilities().asArray(Float.self), [0.3, 0.5])
    }

    func testCheckpointRoundTripsParametersAndInfoUnderXcode() throws {
        let checkpointDirectory = URL(
            fileURLWithPath: "/private/tmp/fastgs_swift_checkpoint_roundtrip",
            isDirectory: true
        )
        try? FileManager.default.removeItem(at: checkpointDirectory)

        let parameters = FastGSTrainableParameters(
            means3D: MLXArray([Float(1), 2, 3, 4, 5, 6], [2, 3]),
            dc: MLXArray([Float(0.2), 0.4, 0.6, 0.8, 1.0, 1.2], [2, 1, 3]),
            sh: MLXArray([Float](repeating: 0.1, count: 12), [2, 2, 3]),
            opacityLogits: MLXArray([Float(0.5), 0.7], [2]),
            scales: MLXArray([Float](repeating: 0.3, count: 6), [2, 3]),
            rotations: MLXArray([Float(1), 0, 0, 0, 1, 0, 0, 0], [2, 4]),
            cov3DPrecomputed: MLXArray([Float](repeating: 0.05, count: 12), [2, 6])
        )
        let info = FastGSTrainingCheckpointInfo(
            createdAt: "2026-05-23T00:00:00Z",
            datasetDirectory: "/tmp/fastgs_dataset",
            outputDirectory: checkpointDirectory.deletingLastPathComponent().path,
            imageWidth: 512,
            imageHeight: 512,
            maxFrames: 9999,
            trainingSteps: 200,
            completedStep: 200,
            frameCount: 42,
            pointCount: 2
        )

        try FastGSCheckpoint.save(parameters: parameters, info: info, directory: checkpointDirectory)
        let loaded = try FastGSCheckpoint.load(directory: checkpointDirectory)

        XCTAssertTrue(FileManager.default.fileExists(atPath: FastGSCheckpoint.parameterURL(in: checkpointDirectory).path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: FastGSCheckpoint.infoURL(in: checkpointDirectory).path))
        XCTAssertEqual(loaded.info, info)
        XCTAssertEqual(loaded.parameters.means3D.shape, parameters.means3D.shape)
        XCTAssertEqual(loaded.parameters.dc.shape, parameters.dc.shape)
        XCTAssertEqual(loaded.parameters.sh.shape, parameters.sh.shape)
        XCTAssertEqual(loaded.parameters.opacityLogits.shape, parameters.opacityLogits.shape)
        XCTAssertEqual(loaded.parameters.scales.shape, parameters.scales.shape)
        XCTAssertEqual(loaded.parameters.rotations.shape, parameters.rotations.shape)
        XCTAssertEqual(loaded.parameters.cov3DPrecomputed?.shape, parameters.cov3DPrecomputed?.shape)
        assertClose(loaded.parameters.means3D.asArray(Float.self), parameters.means3D.asArray(Float.self))
        assertClose(loaded.parameters.dc.asArray(Float.self), parameters.dc.asArray(Float.self))
        assertClose(loaded.parameters.sh.asArray(Float.self), parameters.sh.asArray(Float.self))
        assertClose(loaded.parameters.opacityLogits.asArray(Float.self), parameters.opacityLogits.asArray(Float.self))
        assertClose(loaded.parameters.scales.asArray(Float.self), parameters.scales.asArray(Float.self))
        assertClose(loaded.parameters.rotations.asArray(Float.self), parameters.rotations.asArray(Float.self))
        assertClose(
            loaded.parameters.cov3DPrecomputed?.asArray(Float.self) ?? [],
            parameters.cov3DPrecomputed?.asArray(Float.self) ?? []
        )
    }

    func testCloneAfterTrainingAppendsRowsUnderXcode() {
        let opacityLogits = FastGSOpacity.logits(fromProbabilities: MLXArray([Float(0.2), 0.4, 0.6, 0.8], [4]))
        let parameters = FastGSTrainableParameters(
            means3D: MLXArray((0..<12).map { Float($0) }, [4, 3]),
            dc: MLXArray([Float](repeating: 0.1, count: 12), [4, 1, 3]),
            sh: MLXArray([Float](repeating: 0.2, count: 24), [4, 2, 3]),
            opacityLogits: opacityLogits,
            scales: MLXArray([
                Foundation.log(Float(0.1)), Foundation.log(Float(0.1)), Foundation.log(Float(0.1)),
                Foundation.log(Float(0.2)), Foundation.log(Float(0.2)), Foundation.log(Float(0.2)),
                Foundation.log(Float(3.0)), Foundation.log(Float(3.0)), Foundation.log(Float(3.0)),
                Foundation.log(Float(0.3)), Foundation.log(Float(0.3)), Foundation.log(Float(0.3)),
            ], [4, 3]),
            rotations: MLXArray([Float](repeating: 0, count: 16), [4, 4]),
            cov3DPrecomputed: MLXArray([Float](repeating: 0.4, count: 24), [4, 6])
        )
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
        XCTAssertNil(result.densificationState.tmpRadii)
        assertClose(result.parameters.means3D.asArray(Float.self), [
            0, 1, 2,
            3, 4, 5,
            6, 7, 8,
            9, 10, 11,
            3, 4, 5,
        ])
        assertClose(result.parameters.opacityProbabilities().asArray(Float.self), [0.2, 0.4, 0.6, 0.8, 0.4])
        assertClose(result.optimizerState?.means3D.firstMoment.asArray(Float.self) ?? [], [Float](repeating: 0, count: 15))
    }

    func testSplitAfterTrainingAppendsChildrenAndPrunesSourcesUnderXcode() {
        let opacityLogits = FastGSOpacity.logits(fromProbabilities: MLXArray([Float(0.2), 0.4, 0.6, 0.8], [4]))
        let parameters = FastGSTrainableParameters(
            means3D: MLXArray((0..<12).map { Float($0) }, [4, 3]),
            dc: MLXArray([Float](repeating: 0.1, count: 12), [4, 1, 3]),
            sh: MLXArray([Float](repeating: 0.2, count: 24), [4, 2, 3]),
            opacityLogits: opacityLogits,
            scales: MLXArray([
                Foundation.log(Float(0.1)), Foundation.log(Float(0.1)), Foundation.log(Float(0.1)),
                Foundation.log(Float(0.2)), Foundation.log(Float(0.2)), Foundation.log(Float(0.2)),
                Foundation.log(Float(3.0)), Foundation.log(Float(3.0)), Foundation.log(Float(3.0)),
                Foundation.log(Float(0.3)), Foundation.log(Float(0.3)), Foundation.log(Float(0.3)),
            ], [4, 3]),
            rotations: MLXArray([
                Float(1), 0, 0, 0,
                1, 0, 0, 0,
                1, 0, 0, 0,
                1, 0, 0, 0,
            ], [4, 4]),
            cov3DPrecomputed: MLXArray([Float](repeating: 0.4, count: 24), [4, 6])
        )
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
        assertClose(result.parameters.means3D.asArray(Float.self), [
            0, 1, 2,
            3, 4, 5,
            9, 10, 11,
            9, 7, 8,
            6, 10, 8,
        ])
        assertClose(result.parameters.opacityProbabilities().asArray(Float.self), [0.2, 0.4, 0.8, 0.6, 0.6])
        assertClose(result.parameters.scales.asArray(Float.self).suffix(6).map { $0 }, [
            Foundation.log(Float(3.0 / 1.6)), Foundation.log(Float(3.0 / 1.6)), Foundation.log(Float(3.0 / 1.6)),
            Foundation.log(Float(3.0 / 1.6)), Foundation.log(Float(3.0 / 1.6)), Foundation.log(Float(3.0 / 1.6)),
        ])
    }

    func testImageExportRGBABytes() {
        let outColor = MLXArray([
            Float(0), 0.5,
            1, -0.25,
            0.25, 2,
        ], [3, 2])

        XCTAssertEqual(
            FastGSImageExport.rgbaBytes(outColor: outColor, width: 2, height: 1),
            [
                0, 255, 64, 255,
                128, 0, 255, 255,
            ]
        )
    }

    func testImageExportTextureRunsUnderXcode() throws {
        let outColor = MLXArray([
            Float(0), 0.5,
            1, -0.25,
            0.25, 2,
        ], [3, 2])
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let texture = try XCTUnwrap(FastGSImageExport.texture(outColor: outColor, width: 2, height: 1, device: device))

        XCTAssertEqual(texture.width, 2)
        XCTAssertEqual(texture.height, 1)
        XCTAssertEqual(texture.pixelFormat, .rgba8Unorm)

        var bytes = [UInt8](repeating: 0, count: 8)
        texture.getBytes(
            &bytes,
            bytesPerRow: 2 * 4,
            from: MTLRegionMake2D(0, 0, 2, 1),
            mipmapLevel: 0
        )
        XCTAssertEqual(bytes, [
            0, 255, 64, 255,
            128, 0, 255, 255,
        ])
    }

    func testRecordedScannerForwardRunsUnderXcode() throws {
        try assertRecordedForward(
            manifestURL: recordedManifestURL,
            outputPNGURL: URL(fileURLWithPath: "/private/tmp/fastgs_recorded_reference/recorded_swift.png"),
            summaryURL: URL(fileURLWithPath: "/private/tmp/fastgs_recorded_reference/recorded_swift_stage_summary.json")
        )
    }

    func testRecordedScannerLargeForwardRunsUnderXcode() throws {
        try assertRecordedForward(
            manifestURL: recordedLargeManifestURL,
            outputPNGURL: URL(fileURLWithPath: "/private/tmp/fastgs_recorded_reference_16384/recorded_swift.png"),
            summaryURL: URL(fileURLWithPath: "/private/tmp/fastgs_recorded_reference_16384/recorded_swift_stage_summary.json"),
            channelSumAccuracy: 2e-2
        )
    }

    func testRecordedScannerRenderFPSBenchmarkUnderXcode() throws {
        let benchmarkMarkerURL = URL(fileURLWithPath: "/private/tmp/fastgs_run_render_benchmark")
        guard ProcessInfo.processInfo.environment["FASTGS_RUN_RENDER_BENCHMARK"] == "1"
            || FileManager.default.fileExists(atPath: benchmarkMarkerURL.path)
        else {
            throw XCTSkip("Set FASTGS_RUN_RENDER_BENCHMARK=1 or create \(benchmarkMarkerURL.path) to run the recorded scanner render FPS benchmark.")
        }

        let datasetURL = URL(fileURLWithPath: "/Users/yangdunfu/Downloads/2026_05_04_16_51_29", isDirectory: true)
        guard FileManager.default.fileExists(atPath: datasetURL.appendingPathComponent("points.ply").path) else {
            throw XCTSkip("Fixed scanner dataset is not available at \(datasetURL.path).")
        }

        let environment = ProcessInfo.processInfo.environment
        let markerConfig = FastGSBenchmarkMarkerConfig(url: benchmarkMarkerURL)
        let width = markerConfig.intValue("width") ?? environment.intValue("FASTGS_RENDER_BENCH_WIDTH", default: 512)
        let height = markerConfig.intValue("height") ?? environment.intValue("FASTGS_RENDER_BENCH_HEIGHT", default: 512)
        let rounds = markerConfig.intValue("rounds") ?? environment.intValue("FASTGS_RENDER_BENCH_ROUNDS", default: 5)
        let warmupFrames = markerConfig.intValue("warmup") ?? environment.intValue("FASTGS_RENDER_BENCH_WARMUP", default: 3)
        let secondsPerRound = markerConfig.doubleValue("seconds") ?? environment.doubleValue("FASTGS_RENDER_BENCH_SECONDS", default: 1.0)

        let cache = try FastGSScannerDatasetLoader.loadCache(
            directory: datasetURL,
            options: FastGSScannerDatasetOptions(width: width, height: height, normalizeWithAllFramePairs: true)
        )
        let dataset = try FastGSScannerDatasetLoader.loadDataset(
            cache: cache,
            frameIndex: cache.frameDescriptors[0].index,
            width: width,
            height: height
        )
        let scene = FastGSRecordedForwardScene(scannerDataset: dataset, frameIndex: 0)
        let parameters = try scene.initialTrainableParameters()

        if warmupFrames > 0 {
            for _ in 0..<warmupFrames {
                try scene.render(parameters: parameters).outColor.eval()
            }
        }

        var roundFPS = [Double]()
        var frameLatencies = [Double]()
        var frameCount = 0
        for _ in 0..<max(1, rounds) {
            let roundStart = Date()
            var roundFrameCount = 0
            repeat {
                let frameStart = Date()
                try scene.render(parameters: parameters).outColor.eval()
                let latency = Date().timeIntervalSince(frameStart)
                frameLatencies.append(latency)
                frameCount += 1
                roundFrameCount += 1
            } while Date().timeIntervalSince(roundStart) < secondsPerRound || roundFrameCount == 0

            let roundSeconds = Date().timeIntervalSince(roundStart)
            roundFPS.append(Double(roundFrameCount) / roundSeconds)
        }

        let report = FastGSRenderBenchmarkReport(
            width: width,
            height: height,
            pointCount: scene.manifest.pointCount,
            warmupFrames: warmupFrames,
            secondsPerRound: secondsPerRound,
            roundFPS: roundFPS,
            frameLatencies: frameLatencies
        )
        print("\n\(report.description)\n")

        XCTAssertGreaterThan(frameCount, 0)
        XCTAssertTrue(report.fpsMean.isFinite)
        XCTAssertTrue(report.latencyMeanMilliseconds.isFinite)
    }

    func testRecordedScannerRenderTextureFPSBenchmarkUnderXcode() throws {
        let benchmarkMarkerURL = URL(fileURLWithPath: "/private/tmp/fastgs_run_render_texture_benchmark")
        guard ProcessInfo.processInfo.environment["FASTGS_RUN_RENDER_TEXTURE_BENCHMARK"] == "1"
            || FileManager.default.fileExists(atPath: benchmarkMarkerURL.path)
        else {
            throw XCTSkip("Set FASTGS_RUN_RENDER_TEXTURE_BENCHMARK=1 or create \(benchmarkMarkerURL.path) to run the recorded scanner render texture FPS benchmark.")
        }

        let datasetURL = URL(fileURLWithPath: "/Users/yangdunfu/Downloads/2026_05_04_16_51_29", isDirectory: true)
        guard FileManager.default.fileExists(atPath: datasetURL.appendingPathComponent("points.ply").path) else {
            throw XCTSkip("Fixed scanner dataset is not available at \(datasetURL.path).")
        }
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())

        let environment = ProcessInfo.processInfo.environment
        let markerConfig = FastGSBenchmarkMarkerConfig(url: benchmarkMarkerURL)
        let width = markerConfig.intValue("width") ?? environment.intValue("FASTGS_RENDER_TEXTURE_BENCH_WIDTH", default: 512)
        let height = markerConfig.intValue("height") ?? environment.intValue("FASTGS_RENDER_TEXTURE_BENCH_HEIGHT", default: 512)
        let rounds = markerConfig.intValue("rounds") ?? environment.intValue("FASTGS_RENDER_TEXTURE_BENCH_ROUNDS", default: 5)
        let warmupFrames = markerConfig.intValue("warmup") ?? environment.intValue("FASTGS_RENDER_TEXTURE_BENCH_WARMUP", default: 3)
        let secondsPerRound = markerConfig.doubleValue("seconds") ?? environment.doubleValue("FASTGS_RENDER_TEXTURE_BENCH_SECONDS", default: 1.0)
        let previewOnly = markerConfig.boolValue("preview") ?? environment.boolValue("FASTGS_RENDER_TEXTURE_BENCH_PREVIEW", default: false)

        let cache = try FastGSScannerDatasetLoader.loadCache(
            directory: datasetURL,
            options: FastGSScannerDatasetOptions(width: width, height: height, normalizeWithAllFramePairs: true)
        )
        let dataset = try FastGSScannerDatasetLoader.loadDataset(
            cache: cache,
            frameIndex: cache.frameDescriptors[0].index,
            width: width,
            height: height
        )
        let scene = FastGSRecordedForwardScene(scannerDataset: dataset, frameIndex: 0)
        let parameters = try scene.initialTrainableParameters()

        if warmupFrames > 0 {
            for _ in 0..<warmupFrames {
                let outColor = previewOnly
                    ? try scene.renderPreviewOutColor(parameters: parameters)
                    : try scene.render(parameters: parameters).outColor
                XCTAssertNotNil(FastGSImageExport.texture(outColor: outColor, width: width, height: height, device: device))
            }
        }

        var roundFPS = [Double]()
        var frameLatencies = [Double]()
        var frameCount = 0
        for _ in 0..<max(1, rounds) {
            let roundStart = Date()
            var roundFrameCount = 0
            repeat {
                let frameStart = Date()
                let outColor = previewOnly
                    ? try scene.renderPreviewOutColor(parameters: parameters)
                    : try scene.render(parameters: parameters).outColor
                let texture = FastGSImageExport.texture(outColor: outColor, width: width, height: height, device: device)
                let latency = Date().timeIntervalSince(frameStart)
                XCTAssertNotNil(texture)
                frameLatencies.append(latency)
                frameCount += 1
                roundFrameCount += 1
            } while Date().timeIntervalSince(roundStart) < secondsPerRound || roundFrameCount == 0

            let roundSeconds = Date().timeIntervalSince(roundStart)
            roundFPS.append(Double(roundFrameCount) / roundSeconds)
        }

        let report = FastGSRenderBenchmarkReport(
            title: previewOnly ? "FastGS recorded preview render texture benchmark" : "FastGS recorded render texture benchmark",
            width: width,
            height: height,
            pointCount: scene.manifest.pointCount,
            warmupFrames: warmupFrames,
            secondsPerRound: secondsPerRound,
            roundFPS: roundFPS,
            frameLatencies: frameLatencies
        )
        print("\n\(report.description)\n")

        XCTAssertGreaterThan(frameCount, 0)
        XCTAssertTrue(report.fpsMean.isFinite)
        XCTAssertTrue(report.latencyMeanMilliseconds.isFinite)
    }

    func testRecordedScannerForwardStageTimingUnderXcode() throws {
        let timingMarkerURL = URL(fileURLWithPath: "/private/tmp/fastgs_run_stage_timing")
        guard ProcessInfo.processInfo.environment["FASTGS_RUN_STAGE_TIMING"] == "1"
            || FileManager.default.fileExists(atPath: timingMarkerURL.path)
        else {
            throw XCTSkip("Set FASTGS_RUN_STAGE_TIMING=1 or create \(timingMarkerURL.path) to run the recorded scanner stage timing report.")
        }

        let datasetURL = URL(fileURLWithPath: "/Users/yangdunfu/Downloads/2026_05_04_16_51_29", isDirectory: true)
        guard FileManager.default.fileExists(atPath: datasetURL.appendingPathComponent("points.ply").path) else {
            throw XCTSkip("Fixed scanner dataset is not available at \(datasetURL.path).")
        }

        let environment = ProcessInfo.processInfo.environment
        let markerConfig = FastGSBenchmarkMarkerConfig(url: timingMarkerURL)
        let width = markerConfig.intValue("width") ?? environment.intValue("FASTGS_STAGE_TIMING_WIDTH", default: 512)
        let height = markerConfig.intValue("height") ?? environment.intValue("FASTGS_STAGE_TIMING_HEIGHT", default: 512)
        let rounds = markerConfig.intValue("rounds") ?? environment.intValue("FASTGS_STAGE_TIMING_ROUNDS", default: 5)
        let warmupRounds = markerConfig.intValue("warmup") ?? environment.intValue("FASTGS_STAGE_TIMING_WARMUP", default: 1)
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())

        let cache = try FastGSScannerDatasetLoader.loadCache(
            directory: datasetURL,
            options: FastGSScannerDatasetOptions(width: width, height: height, normalizeWithAllFramePairs: true)
        )
        let dataset = try FastGSScannerDatasetLoader.loadDataset(
            cache: cache,
            frameIndex: cache.frameDescriptors[0].index,
            width: width,
            height: height
        )
        let scene = FastGSRecordedForwardScene(scannerDataset: dataset, frameIndex: 0)
        let parameters = try scene.initialTrainableParameters()

        if warmupRounds > 0 {
            for _ in 0..<warmupRounds {
                _ = try scene.timedRenderStages(parameters: parameters)
            }
        }

        var timings = [FastGSRecordedForwardTimingReport]()
        var imageReadbackMilliseconds = [Double]()
        var textureUploadMilliseconds = [Double]()

        for _ in 0..<max(1, rounds) {
            let timed = try scene.timedRenderStages(parameters: parameters)
            timings.append(timed.timing)

            var started = Date()
            let rgba = FastGSImageExport.rgbaBytes(
                outColor: timed.stages.rasterize.outColor,
                width: width,
                height: height
            )
            imageReadbackMilliseconds.append(Date().timeIntervalSince(started) * 1000)

            started = Date()
            XCTAssertNotNil(FastGSImageExport.texture(rgbaBytes: rgba, width: width, height: height, device: device))
            textureUploadMilliseconds.append(Date().timeIntervalSince(started) * 1000)
        }

        let report = FastGSForwardStageTimingSummary(
            width: width,
            height: height,
            pointCount: scene.manifest.pointCount,
            rounds: max(1, rounds),
            warmupRounds: warmupRounds,
            timings: timings,
            imageReadbackMilliseconds: imageReadbackMilliseconds,
            textureUploadMilliseconds: textureUploadMilliseconds
        )
        print("\n\(report.description)\n")

        XCTAssertEqual(timings.count, max(1, rounds))
        XCTAssertTrue(report.forwardMeanMilliseconds.isFinite)
    }

    private func assertRecordedForward(
        manifestURL: URL,
        outputPNGURL: URL,
        summaryURL: URL,
        channelSumAccuracy: Float = 1.0,
        sampleAccuracy: Float = 2e-2
    ) throws {
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw XCTSkip("Generate \(manifestURL.path) first.")
        }

        let scene = try FastGSRecordedForwardScene(manifestURL: manifestURL)
        let manifest = scene.manifest
        let stages = try scene.renderStages()
        let output = stages.rasterize
        let outColor = output.outColor.asArray(Float.self)

        assertClose(
            channelSums(outColor, channels: 3),
            manifest.predChannelSums.map(Float.init),
            accuracy: channelSumAccuracy
        )
        assertClose(
            samples(outColor, ids: manifest.samplePixelIds, channels: 3),
            manifest.predSamples.map(Float.init),
            accuracy: sampleAccuracy
        )

        try FastGSImageExport.writePNG(
            outColor: output.outColor,
            width: manifest.width,
            height: manifest.height,
            to: outputPNGURL
        )

        let summary = recordedStageSummary(stages, sampleIDs: manifest.samplePixelIds)
        let data = try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: summaryURL)
    }

    func testCameraFrameBridgeReadsBGRAUnderXcode() throws {
        let pixelBuffer = try makeBGRA32PixelBuffer(width: 2, height: 1)
        let frame = try FastGSCameraFrameBridge.lockBGRAFrame(pixelBuffer)

        XCTAssertEqual(frame.width, 2)
        XCTAssertEqual(frame.height, 1)
        XCTAssertGreaterThanOrEqual(frame.bytesPerRow, 8)
        XCTAssertEqual(frame.rgbaBytes, [
            30, 20, 10, 255,
            70, 60, 50, 128,
        ])
    }

    func testCameraFrameBridgeTextureRunsUnderXcode() throws {
        let pixelBuffer = try makeBGRA32PixelBuffer(width: 2, height: 1)
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let texture = try XCTUnwrap(FastGSCameraFrameBridge.texture(fromBGRA: pixelBuffer, device: device))

        XCTAssertEqual(texture.width, 2)
        XCTAssertEqual(texture.height, 1)
        XCTAssertEqual(texture.pixelFormat, .rgba8Unorm)

        var bytes = [UInt8](repeating: 0, count: 8)
        texture.getBytes(
            &bytes,
            bytesPerRow: 2 * 4,
            from: MTLRegionMake2D(0, 0, 2, 1),
            mipmapLevel: 0
        )
        XCTAssertEqual(bytes, [
            30, 20, 10, 255,
            70, 60, 50, 128,
        ])
    }

    func testMLXFastMetalKernelRunsUnderXcode() {
        let input = MLXArray([Float(1), Float(2), Float(3), Float(4)], [4])
        let output = FastGSSmokeKernel.double(input)

        XCTAssertEqual(output.shape, [4])
        XCTAssertEqual(output.dtype, .float32)
        XCTAssertEqual(output.asArray(Float.self), [2, 4, 6, 8])
    }

    func testPreprocessKernelRunsUnderXcode() {
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

    func testPreprocessSHDegree3KernelRunsUnderXcode() {
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

    func testPreprocessCullingKernelRunsUnderXcode() {
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

    func testPreprocessCov3DPrecomputedKernelRunsUnderXcode() {
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

    func testPreprocessSHClampKernelRunsUnderXcode() {
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

    func testBinningKernelRunsUnderXcode() {
        let output = FastGSPreprocessParityFixture.binningOutput()

        assertBinning(
            output,
            pointOffsets: FastGSPreprocessParityFixture.expectedBinningPointOffsets,
            pointListKeysUnsorted: FastGSPreprocessParityFixture.expectedBinningPointListKeysUnsorted,
            pointListUnsorted: FastGSPreprocessParityFixture.expectedBinningPointListUnsorted,
            pointListKeys: FastGSPreprocessParityFixture.expectedBinningPointListKeys,
            pointList: nil,
            ranges: FastGSPreprocessParityFixture.expectedBinningRanges,
            bucketCount: FastGSPreprocessParityFixture.expectedBinningBucketCount,
            bucketOffsets: FastGSPreprocessParityFixture.expectedBinningBucketOffsets
        )
    }

    func testBinningCullingKernelRunsUnderXcode() {
        let output = FastGSPreprocessParityFixture.cullingBinningOutput()

        assertBinning(
            output,
            pointOffsets: FastGSPreprocessParityFixture.expectedCullingBinningPointOffsets,
            pointListKeysUnsorted: FastGSPreprocessParityFixture.expectedCullingBinningPointListKeysUnsorted,
            pointListUnsorted: FastGSPreprocessParityFixture.expectedCullingBinningPointListUnsorted,
            pointListKeys: FastGSPreprocessParityFixture.expectedCullingBinningPointListKeys,
            pointList: FastGSPreprocessParityFixture.expectedCullingBinningPointListUnsorted,
            ranges: FastGSPreprocessParityFixture.expectedCullingBinningRanges,
            bucketCount: FastGSPreprocessParityFixture.expectedBinningBucketCount,
            bucketOffsets: FastGSPreprocessParityFixture.expectedBinningBucketOffsets
        )
    }

    func testBinningAllCulledKernelRunsUnderXcode() {
        let output = FastGSPreprocessParityFixture.allCulledBinningOutput()

        assertBinning(
            output,
            pointOffsets: FastGSPreprocessParityFixture.expectedAllCulledBinningPointOffsets,
            pointListKeysUnsorted: [],
            pointListUnsorted: [],
            pointListKeys: [],
            pointList: [],
            ranges: FastGSPreprocessParityFixture.expectedAllCulledBinningRanges,
            bucketCount: FastGSPreprocessParityFixture.expectedAllCulledBinningBucketCount,
            bucketOffsets: FastGSPreprocessParityFixture.expectedAllCulledBinningBucketOffsets
        )
    }

    func testBinningVariedDepthKernelRunsUnderXcode() {
        let output = FastGSPreprocessParityFixture.variedDepthBinningOutput()

        assertBinning(
            output,
            pointOffsets: FastGSPreprocessParityFixture.expectedVariedDepthBinningPointOffsets,
            pointListKeysUnsorted: FastGSPreprocessParityFixture.expectedVariedDepthBinningPointListKeysUnsorted,
            pointListUnsorted: FastGSPreprocessParityFixture.expectedVariedDepthBinningPointListUnsorted,
            pointListKeys: FastGSPreprocessParityFixture.expectedVariedDepthBinningPointListKeys,
            pointList: FastGSPreprocessParityFixture.expectedVariedDepthBinningPointList,
            ranges: FastGSPreprocessParityFixture.expectedBinningRanges,
            bucketCount: FastGSPreprocessParityFixture.expectedBinningBucketCount,
            bucketOffsets: FastGSPreprocessParityFixture.expectedBinningBucketOffsets
        )
    }

    func testRasterizeSmokeKernelRunsUnderXcode() {
        let output = FastGSPreprocessParityFixture.rasterizeSmokeOutput()

        XCTAssertEqual(output.bucketToTile.shape, [256])
        XCTAssertEqual(output.sampledT.shape, [256])
        XCTAssertEqual(output.sampledAr.shape, [3 * 256])
        XCTAssertEqual(output.finalT.shape, [1])
        XCTAssertEqual(output.nContrib.shape, [1])
        XCTAssertEqual(output.maxContrib.shape, [1])
        XCTAssertEqual(output.pixelColors.shape, [3, 1])
        XCTAssertEqual(output.outColor.shape, [3, 1])
        XCTAssertEqual(output.metricCount.shape, [1])

        XCTAssertEqual(output.bucketToTile.asArray(UInt32.self), Array(repeating: UInt32(0), count: 256))
        assertClose(output.sampledT.asArray(Float.self), FastGSPreprocessParityFixture.expectedRasterizeSmokeSampledT)
        assertClose(output.sampledAr.asArray(Float.self), FastGSPreprocessParityFixture.expectedRasterizeSmokeSampledAr)
        assertClose(output.finalT.asArray(Float.self), FastGSPreprocessParityFixture.expectedRasterizeSmokeFinalT)
        XCTAssertEqual(output.nContrib.asArray(UInt32.self), FastGSPreprocessParityFixture.expectedRasterizeSmokeNContrib)
        XCTAssertEqual(output.maxContrib.asArray(UInt32.self), FastGSPreprocessParityFixture.expectedRasterizeSmokeMaxContrib)
        assertClose(output.pixelColors.asArray(Float.self), FastGSPreprocessParityFixture.expectedRasterizeSmokePixelColors)
        assertClose(output.outColor.asArray(Float.self), FastGSPreprocessParityFixture.expectedRasterizeSmokeOutColor)
        XCTAssertEqual(output.metricCount.asArray(Int32.self), FastGSPreprocessParityFixture.expectedRasterizeSmokeMetricCount)
    }

    func testRasterizeE2EKernelRunsUnderXcode() {
        let output = FastGSPreprocessParityFixture.rasterizeE2EOutput()

        assertRasterizeE2E(output)
    }

    func testRasterizeLargeE2EKernelRunsUnderXcode() {
        let preprocess = FastGSPreprocessParityFixture.rasterizeLargeE2EPreprocessOutput()
        let binning = FastGSPreprocessParityFixture.rasterizeLargeE2EBinningOutput()
        let output = FastGSPreprocessParityFixture.rasterizeLargeE2EOutput()

        assertRasterizeLargeE2E(preprocess: preprocess, binning: binning, output: output)
    }

    func testRasterizeLargeE2EPNGExportRunsUnderXcode() throws {
        let output = FastGSPreprocessParityFixture.rasterizeLargeE2EOutput()
        let url = URL(fileURLWithPath: "/private/tmp/fastgs_swift_large_rasterize.png")
        try? FileManager.default.removeItem(at: url)

        try FastGSImageExport.writePNG(rasterizeOutput: output, width: 80, height: 48, to: url)

        let data = try Data(contentsOf: url)
        XCTAssertGreaterThan(data.count, 100)
        XCTAssertEqual(Array(data.prefix(8)), [137, 80, 78, 71, 13, 10, 26, 10])

        let bytes = FastGSImageExport.rgbaBytes(rasterizeOutput: output, width: 80, height: 48)
        XCTAssertTrue(bytes.contains { $0 != 0 })
        XCTAssertEqual(bytes[3], 255)
        XCTAssertEqual(bytes[4 * (80 * 48 - 1) + 3], 255)
    }

    func testRasterizeBackwardKernelRunsUnderXcode() {
        let preprocess = FastGSPreprocessParityFixture.rasterizeLargeE2EPreprocessOutput()
        let binning = FastGSPreprocessParityFixture.rasterizeLargeE2EBinningOutput()
        let rasterize = FastGSPreprocessParityFixture.rasterizeLargeE2EOutput()
        let cotangents = FastGSRasterizeCotangents.outColorOnes(like: rasterize)
        let background = MLXArray([Float(0.025), 0.03, 0.04], [3])
        let output = FastGSRasterizeBackward.forward(
            preprocessOutput: preprocess,
            binningOutput: binning,
            rasterizeOutput: rasterize,
            cotangents: cotangents,
            background: background,
            params: FastGSPreprocessParityFixture.rasterizeLargeE2EParams
        )

        assertRasterizeBackwardSkeleton(preprocess: preprocess, binning: binning, background: background, output: output)
    }

    func testRasterizeBackwardMatchesReferenceSummaryUnderXcode() throws {
        guard FileManager.default.fileExists(atPath: rasterizeBackwardReferenceURL.path) else {
            throw XCTSkip("Generate \(rasterizeBackwardReferenceURL.path) first.")
        }

        let reference = try JSONDecoder().decode(
            RasterizeBackwardReference.self,
            from: Data(contentsOf: rasterizeBackwardReferenceURL)
        )
        let preprocess = FastGSPreprocessParityFixture.rasterizeLargeE2EPreprocessOutput()
        let binning = FastGSPreprocessParityFixture.rasterizeLargeE2EBinningOutput()
        let rasterize = FastGSPreprocessParityFixture.rasterizeLargeE2EOutput()
        let cotangents = FastGSRasterizeCotangents.outColorOnes(like: rasterize)
        let output = FastGSRasterizeBackward.forward(
            preprocessOutput: preprocess,
            binningOutput: binning,
            rasterizeOutput: rasterize,
            cotangents: cotangents,
            background: MLXArray([Float(0.025), 0.03, 0.04], [3]),
            params: FastGSPreprocessParityFixture.rasterizeLargeE2EParams
        )

        assertGradientSummary(output.means2D, reference.gradients.means2D, accuracy: 1e-3)
        assertGradientSummary(output.colors, reference.gradients.colors, accuracy: 1e-3)
        assertGradientSummary(output.conicOpacity, reference.gradients.conicOpacity, accuracy: 1e-2)
        assertGradientSummary(output.viewspacePoints, reference.gradients.viewspacePoints, accuracy: 1e-2)
    }

    func testPreprocessBackwardKernelRunsUnderXcode() {
        let input = FastGSPreprocessParityFixture.rasterizeLargeE2EInput()
        let preprocess = preprocessBackwardSmokeForwardOutput(count: input.means3D.shape[0])
        let output = FastGSPreprocessBackward.forward(
            input: input,
            cotangents: preprocessBackwardSmokeCotangents(like: preprocess),
            forwardOutput: preprocess,
            params: FastGSPreprocessParityFixture.rasterizeLargeE2EPreprocessParams
        )

        assertPreprocessBackwardSkeleton(input: input, output: output)
    }

    func testPreprocessBackwardMatchesReferenceSummaryUnderXcode() throws {
        guard ProcessInfo.processInfo.environment["FASTGS_RUN_SLOW_PARITY"] == "1" else {
            throw XCTSkip("Set FASTGS_RUN_SLOW_PARITY=1 to run the full preprocess backward parity check.")
        }
        guard FileManager.default.fileExists(atPath: preprocessBackwardReferenceURL.path) else {
            throw XCTSkip("Generate \(preprocessBackwardReferenceURL.path) first.")
        }

        let reference = try JSONDecoder().decode(
            PreprocessBackwardReference.self,
            from: Data(contentsOf: preprocessBackwardReferenceURL)
        )
        let input = FastGSPreprocessParityFixture.precomputedColorInput()
        let params = FastGSPreprocessParityFixture.precomputedColorParams
        let forwardOutput = FastGSPreprocessParityFixture.precomputedColorOutput()
        let loss = forwardOutput.xy.sum()
            + 0.1 * forwardOutput.depths.sum()
            + 0.2 * forwardOutput.cov3D.sum()
            + 0.3 * forwardOutput.rgb.sum()
            + 0.4 * forwardOutput.conicOpacity.sum()
            + 0.5 * forwardOutput.viewspacePoints.sum()
        let cotangents = FastGSPreprocessCotangents(
            radii: MLXArray.zeros(forwardOutput.radii.shape, dtype: forwardOutput.radii.dtype),
            xy: MLXArray.ones(forwardOutput.xy.shape, dtype: forwardOutput.xy.dtype),
            depths: 0.1 * MLXArray.ones(forwardOutput.depths.shape, dtype: forwardOutput.depths.dtype),
            cov3D: 0.2 * MLXArray.ones(forwardOutput.cov3D.shape, dtype: forwardOutput.cov3D.dtype),
            rgb: 0.3 * MLXArray.ones(forwardOutput.rgb.shape, dtype: forwardOutput.rgb.dtype),
            conicOpacity: 0.4 * MLXArray.ones(forwardOutput.conicOpacity.shape, dtype: forwardOutput.conicOpacity.dtype),
            tilesTouched: MLXArray.zeros(forwardOutput.tilesTouched.shape, dtype: forwardOutput.tilesTouched.dtype),
            clamped: MLXArray.zeros(forwardOutput.clamped.shape, dtype: forwardOutput.clamped.dtype),
            viewspacePoints: 0.5 * MLXArray.ones(forwardOutput.viewspacePoints.shape, dtype: forwardOutput.viewspacePoints.dtype)
        )
        let gradients = FastGSPreprocessBackward.forward(
            input: input,
            cotangents: cotangents,
            forwardOutput: forwardOutput,
            params: params
        )

        XCTAssertEqual(loss.item(Float.self), Float(reference.loss), accuracy: 1e-3)
        assertGradientSummary(gradients.means3D, reference.gradients.means3D, accuracy: 2e-3)
        assertGradientSummary(gradients.colorsPrecomputed, reference.gradients.colorsPrecomputed, accuracy: 1e-5)
        assertGradientSummary(gradients.opacities, reference.gradients.opacities, accuracy: 1e-5)
        assertGradientSummary(gradients.scales, reference.gradients.scales, accuracy: 2e-3)
        assertGradientSummary(gradients.rotations, reference.gradients.rotations, accuracy: 1e-5)
        assertGradientSummary(gradients.viewspacePoints, reference.gradients.viewspacePoints, accuracy: 1e-5)
    }

    func testPreprocessBackwardSHDegree3MatchesReferenceSummaryUnderXcode() throws {
        guard ProcessInfo.processInfo.environment["FASTGS_RUN_SLOW_PARITY"] == "1" else {
            throw XCTSkip("Set FASTGS_RUN_SLOW_PARITY=1 to run the full preprocess backward parity check.")
        }
        guard FileManager.default.fileExists(atPath: preprocessBackwardReferenceURL.path) else {
            throw XCTSkip("Generate \(preprocessBackwardReferenceURL.path) first.")
        }

        let reference = try JSONDecoder().decode(
            PreprocessBackwardReference.self,
            from: Data(contentsOf: preprocessBackwardReferenceURL)
        ).fixtures.shDegree3
        let input = FastGSPreprocessParityFixture.shDegree3Input()
        let params = FastGSPreprocessParityFixture.shDegree3Params
        let forwardOutput = FastGSPreprocessParityFixture.shDegree3Output()
        let loss = forwardOutput.xy.sum()
            + 0.1 * forwardOutput.depths.sum()
            + 0.2 * forwardOutput.cov3D.sum()
            + 0.3 * forwardOutput.rgb.sum()
            + 0.4 * forwardOutput.conicOpacity.sum()
            + 0.5 * forwardOutput.viewspacePoints.sum()
        let cotangents = FastGSPreprocessCotangents(
            radii: MLXArray.zeros(forwardOutput.radii.shape, dtype: forwardOutput.radii.dtype),
            xy: MLXArray.ones(forwardOutput.xy.shape, dtype: forwardOutput.xy.dtype),
            depths: 0.1 * MLXArray.ones(forwardOutput.depths.shape, dtype: forwardOutput.depths.dtype),
            cov3D: 0.2 * MLXArray.ones(forwardOutput.cov3D.shape, dtype: forwardOutput.cov3D.dtype),
            rgb: 0.3 * MLXArray.ones(forwardOutput.rgb.shape, dtype: forwardOutput.rgb.dtype),
            conicOpacity: 0.4 * MLXArray.ones(forwardOutput.conicOpacity.shape, dtype: forwardOutput.conicOpacity.dtype),
            tilesTouched: MLXArray.zeros(forwardOutput.tilesTouched.shape, dtype: forwardOutput.tilesTouched.dtype),
            clamped: MLXArray.zeros(forwardOutput.clamped.shape, dtype: forwardOutput.clamped.dtype),
            viewspacePoints: 0.5 * MLXArray.ones(forwardOutput.viewspacePoints.shape, dtype: forwardOutput.viewspacePoints.dtype)
        )
        let gradients = FastGSPreprocessBackward.forward(
            input: input,
            cotangents: cotangents,
            forwardOutput: forwardOutput,
            params: params
        )

        XCTAssertEqual(loss.item(Float.self), Float(reference.loss), accuracy: 1e-3)
        assertGradientSummary(gradients.means3D, reference.gradients.means3D, accuracy: 2e-3)
        assertGradientSummary(gradients.dc, reference.gradients.dc, accuracy: 1e-5)
        assertGradientSummary(gradients.sh, reference.gradients.sh, accuracy: 1e-5)
        assertGradientSummary(gradients.opacities, reference.gradients.opacities, accuracy: 1e-5)
        assertGradientSummary(gradients.scales, reference.gradients.scales, accuracy: 2e-3)
        assertGradientSummary(gradients.rotations, reference.gradients.rotations, accuracy: 1e-5)
        assertGradientSummary(gradients.viewspacePoints, reference.gradients.viewspacePoints, accuracy: 1e-5)
    }

    func testRecordedTrainingZeroVJPValueAndGradRunsUnderXcode() throws {
        guard FileManager.default.fileExists(atPath: recordedManifestURL.path) else {
            throw XCTSkip("Generate \(recordedManifestURL.path) first.")
        }

        let scene = try FastGSRecordedForwardScene(manifestURL: recordedManifestURL)
        let parameters = try scene.initialTrainableParameters()
        let target = try scene.render(parameters: parameters).outColor
        let result = FastGSTrainingRenderFunction.valueAndZeroGrad(
            scene: scene,
            parameters: parameters,
            target: target
        )

        XCTAssertEqual(result.loss.shape, [])
        XCTAssertTrue(result.loss.item(Float.self).isFinite)
        XCTAssertEqual(result.loss.item(Float.self), 0, accuracy: 1e-7)
        XCTAssertEqual(result.gradients.count, parameters.arrays.count)

        for (gradient, parameter) in zip(result.gradients, parameters.arrays) {
            XCTAssertEqual(gradient.shape, parameter.shape)
            XCTAssertEqual(gradient.dtype, parameter.dtype)
            XCTAssertTrue(gradient.asArray(Float.self).allSatisfy { $0 == 0 })
        }
    }

    func testRecordedTrainingRasterizeOnlyVJPValueAndGradRunsUnderXcode() throws {
        guard FileManager.default.fileExists(atPath: recordedManifestURL.path) else {
            throw XCTSkip("Generate \(recordedManifestURL.path) first.")
        }

        let scene = try FastGSRecordedForwardScene(manifestURL: recordedManifestURL)
        let parameters = try scene.initialTrainableParameters()
        let target = try scene.render(parameters: parameters).outColor
        let result = FastGSTrainingRenderFunction.valueAndZeroGrad(
            scene: scene,
            parameters: parameters,
            target: target,
            backwardMode: .rasterizeOnly
        )

        XCTAssertEqual(result.loss.shape, [])
        XCTAssertTrue(result.loss.item(Float.self).isFinite)
        XCTAssertEqual(result.gradients.count, 6)
        XCTAssertEqual(result.gradients.count, parameters.arrays.count)

        for (gradient, parameter) in zip(result.gradients, parameters.arrays) {
            XCTAssertEqual(gradient.shape, parameter.shape)
            XCTAssertEqual(gradient.dtype, parameter.dtype)
            XCTAssertTrue(gradient.asArray(Float.self).allSatisfy { $0 == 0 })
        }
    }

    func testRasterizeCustomFunctionValueAndGradRunsUnderXcode() throws {
        guard FileManager.default.fileExists(atPath: recordedManifestURL.path) else {
            throw XCTSkip("Generate \(recordedManifestURL.path) first.")
        }

        let scene = try FastGSRecordedForwardScene(manifestURL: recordedManifestURL)
        let stages = try scene.renderStages()
        let params = FastGSRasterizeParams(
            imageWidth: scene.manifest.width,
            imageHeight: scene.manifest.height,
            numTiles: ((scene.manifest.width + 15) / 16) * ((scene.manifest.height + 15) / 16)
        )
        let input = FastGSRasterizeInput(
            ranges: stages.binning.ranges,
            pointList: stages.binning.pointList,
            bucketOffsets: stages.binning.bucketOffsets,
            means2D: stages.preprocess.xy,
            colors: stages.preprocess.rgb,
            conicOpacity: stages.preprocess.conicOpacity,
            background: MLXArray(scene.manifest.background.map(Float.init), [3]),
            radii: stages.preprocess.radii,
            metricMap: MLXArray.zeros([scene.manifest.width * scene.manifest.height], dtype: .int32),
            metricCount: MLXArray.zeros([scene.manifest.pointCount], dtype: .int32)
        )
        let function = FastGSRasterizeCustomFunction.make(params: params)
        let primals = FastGSRasterizeCustomFunction.arrays(from: input)
        let lossFunction: ([MLXArray]) -> [MLXArray] = { arrays in
            let outputs = function(arrays)
            return [mean(square(outputs[7]))]
        }

        let valueAndGradient = valueAndGrad(lossFunction, argumentNumbers: [3, 4, 5])
        let (values, gradients) = valueAndGradient(primals)

        XCTAssertEqual(values[0].shape, [])
        XCTAssertTrue(values[0].item(Float.self).isFinite)
        XCTAssertEqual(gradients.count, 3)
        XCTAssertEqual(gradients[0].shape, input.means2D.shape)
        XCTAssertEqual(gradients[1].shape, input.colors.shape)
        XCTAssertEqual(gradients[2].shape, input.conicOpacity.shape)
        XCTAssertTrue(gradients[0].asArray(Float.self).contains { abs($0) > 1e-7 })
        XCTAssertTrue(gradients[1].asArray(Float.self).contains { abs($0) > 1e-7 })
        XCTAssertTrue(gradients[2].asArray(Float.self).contains { abs($0) > 1e-7 })
    }

    func testPreprocessCustomFunctionValueAndGradRunsUnderXcode() throws {
        guard FileManager.default.fileExists(atPath: recordedManifestURL.path) else {
            throw XCTSkip("Generate \(recordedManifestURL.path) first.")
        }

        let scene = try FastGSRecordedForwardScene(manifestURL: recordedManifestURL)
        let parameters = try scene.initialTrainableParameters()
        let input = FastGSPreprocessInput(
            means3D: parameters.means3D,
            dc: parameters.dc,
            sh: parameters.sh,
            colorsPrecomputed: MLXArray.zeros([0, 3], dtype: .float32),
            opacities: parameters.opacityProbabilities(),
            scales: parameters.scales,
            rotations: parameters.rotations,
            cov3DPrecomputed: MLXArray.zeros([0, 6], dtype: .float32),
            viewMatrix: MLXArray(scene.manifest.viewmatrix.map(Float.init), [4, 4]),
            projectionMatrix: MLXArray(scene.manifest.projmatrix.map(Float.init), [4, 4]),
            cameraPosition: MLXArray(Array(scene.manifest.campos.prefix(3)).map(Float.init), [3]),
            viewspacePoints: MLXArray.zeros([scene.manifest.pointCount, 4], dtype: .float32)
        )
        let params = FastGSPreprocessParams(
            degree: scene.manifest.shDegree,
            maxSHCoefficients: (scene.manifest.shDegree + 1) * (scene.manifest.shDegree + 1),
            scaleModifier: 1,
            tanFovX: Float(scene.manifest.tanFovX),
            tanFovY: Float(scene.manifest.tanFovY),
            imageHeight: scene.manifest.height,
            imageWidth: scene.manifest.width,
            tileBounds: ((scene.manifest.width + 15) / 16, (scene.manifest.height + 15) / 16, 1),
            multiplier: 1
        )
        let function = FastGSPreprocessCustomFunction.make(params: params)
        let primals = FastGSPreprocessCustomFunction.arrays(from: input)
        let lossFunction: ([MLXArray]) -> [MLXArray] = { arrays in
            let outputs = function(arrays)
            return [mean(square(outputs[1]))]
        }

        let valueAndGradient = valueAndGrad(lossFunction, argumentNumbers: [0, 1, 2, 4, 5, 6])
        let (values, gradients) = valueAndGradient(primals)

        XCTAssertEqual(values[0].shape, [])
        XCTAssertTrue(values[0].item(Float.self).isFinite)
        XCTAssertEqual(gradients.count, 6)
        XCTAssertEqual(gradients[0].shape, input.means3D.shape)
        XCTAssertEqual(gradients[1].shape, input.dc.shape)
        XCTAssertEqual(gradients[2].shape, input.sh.shape)
        XCTAssertEqual(gradients[3].shape, input.opacities.shape)
        XCTAssertEqual(gradients[4].shape, input.scales.shape)
        XCTAssertEqual(gradients[5].shape, input.rotations.shape)
        XCTAssertTrue(gradients[0].asArray(Float.self).contains { abs($0) > 1e-7 })
    }

    func testTrainingStageGraphValueAndGradRunsUnderXcode() throws {
        guard FileManager.default.fileExists(atPath: recordedManifestURL.path) else {
            throw XCTSkip("Generate \(recordedManifestURL.path) first.")
        }

        let scene = try FastGSRecordedForwardScene(manifestURL: recordedManifestURL)
        let parameters = try scene.initialTrainableParameters()
        let target = FastGSTrainingStageGraph.render(scene: scene, parameters: parameters)
        let result = FastGSTrainingStageGraph.valueAndGrad(
            scene: scene,
            parameters: parameters,
            target: target
        )

        XCTAssertEqual(result.loss.shape, [])
        XCTAssertTrue(result.loss.item(Float.self).isFinite)
        XCTAssertEqual(result.loss.item(Float.self), 0, accuracy: 1e-7)
        XCTAssertEqual(result.gradients.count, 6)
        XCTAssertEqual(result.gradients[0].shape, parameters.means3D.shape)
        XCTAssertEqual(result.gradients[1].shape, parameters.dc.shape)
        XCTAssertEqual(result.gradients[2].shape, parameters.sh.shape)
        XCTAssertEqual(result.gradients[3].shape, parameters.opacityLogits.shape)
        XCTAssertEqual(result.gradients[4].shape, parameters.scales.shape)
        XCTAssertEqual(result.gradients[5].shape, parameters.rotations.shape)
    }

    func testTrainingStageGraphDensificationStatsUnderXcode() throws {
        guard FileManager.default.fileExists(atPath: recordedManifestURL.path) else {
            throw XCTSkip("Generate \(recordedManifestURL.path) first.")
        }

        let scene = try FastGSRecordedForwardScene(manifestURL: recordedManifestURL)
        let parameters = try scene.initialTrainableParameters()
        let target = 0.9 * FastGSTrainingStageGraph.render(scene: scene, parameters: parameters)
        let result = FastGSTrainingStageGraph.valueAndGradWithDensificationStats(
            scene: scene,
            parameters: parameters,
            target: target
        )

        XCTAssertEqual(result.loss.shape, [])
        XCTAssertTrue(result.loss.item(Float.self).isFinite)
        XCTAssertEqual(result.gradients.count, 6)
        XCTAssertEqual(result.densificationStats.radii.shape, [parameters.gaussianCount])
        XCTAssertEqual(result.densificationStats.viewspaceGradients.shape, [parameters.gaussianCount, 4])
        XCTAssertTrue(result.densificationStats.radii.asArray(Int32.self).contains { $0 > 0 })
        XCTAssertTrue(result.densificationStats.viewspaceGradients.asArray(Float.self).contains { abs($0) > 1e-7 })
    }

    func testGaussianScoringProducesImportanceAndPruningScoresUnderXcode() throws {
        guard FileManager.default.fileExists(atPath: recordedManifestURL.path) else {
            throw XCTSkip("Generate \(recordedManifestURL.path) first.")
        }

        let scene = try FastGSRecordedForwardScene(manifestURL: recordedManifestURL)
        let parameters = try scene.initialTrainableParameters()
        let target = 0.9 * FastGSTrainingStageGraph.render(scene: scene, parameters: parameters)
        let result = try FastGSGaussianScoring.compute(
            scenes: [scene],
            parameters: parameters,
            sceneIndices: [0],
            targets: [target],
            lossThreshold: 0,
            densify: true
        )

        XCTAssertEqual(result.sampledFrameCount, 1)
        XCTAssertEqual(result.importanceScores?.count, parameters.gaussianCount)
        XCTAssertEqual(result.metricCounts?.count, parameters.gaussianCount)
        XCTAssertEqual(result.pruningScores.count, parameters.gaussianCount)
        XCTAssertEqual(result.metricScore.count, parameters.gaussianCount)
        XCTAssertTrue(result.metricCounts?.contains { $0 > 0 } == true)
        XCTAssertTrue(result.metricScore.contains { $0 > 0 })
        XCTAssertTrue(result.pruningScores.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 })
    }

    func testRecordedSmallTrainingStepUpdatesParametersUnderXcode() throws {
        guard FileManager.default.fileExists(atPath: recordedManifestURL.path) else {
            throw XCTSkip("Generate \(recordedManifestURL.path) first.")
        }

        let scene = try FastGSRecordedForwardScene(manifestURL: recordedManifestURL)
        let parameters = try scene.initialTrainableParameters()
        let initialRender = FastGSTrainingStageGraph.render(scene: scene, parameters: parameters)
        let target = 0.9 * initialRender
        let result = FastGSTrainingStageGraph.valueAndGrad(
            scene: scene,
            parameters: parameters,
            target: target
        )

        XCTAssertEqual(result.loss.shape, [])
        XCTAssertTrue(result.loss.item(Float.self).isFinite)
        XCTAssertGreaterThan(result.loss.item(Float.self), 0)
        XCTAssertEqual(result.gradients.count, 6)
        XCTAssertEqual(result.gradients[0].shape, parameters.means3D.shape)
        XCTAssertEqual(result.gradients[1].shape, parameters.dc.shape)
        XCTAssertEqual(result.gradients[2].shape, parameters.sh.shape)
        XCTAssertEqual(result.gradients[3].shape, parameters.opacityLogits.shape)
        XCTAssertEqual(result.gradients[4].shape, parameters.scales.shape)
        XCTAssertEqual(result.gradients[5].shape, parameters.rotations.shape)
        XCTAssertTrue(result.gradients.contains { hasFiniteNonZeroValues($0) })

        var optimizer = FastGSAdamOptimizer(
            learningRates: recordedTrainingSmokeLearningRates()
        )
        let gradientStruct = trainableGradients(from: result.gradients)
        let updated = optimizer.update(parameters: parameters, gradients: gradientStruct)

        XCTAssertEqual(optimizer.state?.step, 1)
        XCTAssertFalse(optimizer.stateArrays().isEmpty)
        XCTAssertTrue(
            maxAbsDiff(updated.means3D, parameters.means3D) > 0
                || maxAbsDiff(updated.dc, parameters.dc) > 0
                || maxAbsDiff(updated.sh, parameters.sh) > 0
                || maxAbsDiff(updated.opacityLogits, parameters.opacityLogits) > 0
                || maxAbsDiff(updated.scales, parameters.scales) > 0
                || maxAbsDiff(updated.rotations, parameters.rotations) > 0
        )
    }

    func testRecordedSmallTrainingLoopReducesSyntheticLossUnderXcode() throws {
        guard FileManager.default.fileExists(atPath: recordedManifestURL.path) else {
            throw XCTSkip("Generate \(recordedManifestURL.path) first.")
        }

        let scene = try FastGSRecordedForwardScene(manifestURL: recordedManifestURL)
        var parameters = try scene.initialTrainableParameters()
        let initialParameters = parameters
        let target = 0.9 * FastGSTrainingStageGraph.render(scene: scene, parameters: parameters)
        var optimizer = FastGSAdamOptimizer(
            learningRates: recordedTrainingSmokeLearningRates()
        )
        var losses = [Float]()

        for step in 0..<3 {
            let result = FastGSTrainingStageGraph.valueAndGrad(
                scene: scene,
                parameters: parameters,
                target: target
            )
            let loss = result.loss.item(Float.self)
            XCTAssertTrue(loss.isFinite)
            XCTAssertGreaterThan(loss, 0)
            XCTAssertEqual(result.gradients.count, 6)
            XCTAssertTrue(result.gradients.contains { hasFiniteNonZeroValues($0) })

            losses.append(loss)
            parameters = optimizer.update(
                parameters: parameters,
                gradients: trainableGradients(from: result.gradients)
            )
            XCTAssertEqual(optimizer.state?.step, step + 1)
            XCTAssertFalse(optimizer.stateArrays().isEmpty)
        }

        XCTAssertEqual(losses.count, 3)
        XCTAssertLessThanOrEqual(losses.last ?? .infinity, (losses.first ?? 0) * 1.01)
        XCTAssertTrue(
            maxAbsDiff(parameters.means3D, initialParameters.means3D) > 0
                || maxAbsDiff(parameters.dc, initialParameters.dc) > 0
                || maxAbsDiff(parameters.sh, initialParameters.sh) > 0
                || maxAbsDiff(parameters.opacityLogits, initialParameters.opacityLogits) > 0
                || maxAbsDiff(parameters.scales, initialParameters.scales) > 0
                || maxAbsDiff(parameters.rotations, initialParameters.rotations) > 0
        )
    }

    func testRecordedTrainingLoopAppliesPruneOnlyUnderXcode() throws {
        guard FileManager.default.fileExists(atPath: recordedManifestURL.path) else {
            throw XCTSkip("Generate \(recordedManifestURL.path) first.")
        }

        let scene = try FastGSRecordedForwardScene(manifestURL: recordedManifestURL)
        var summaries = [FastGSRecordedTrainingPruneSummary]()
        let config = FastGSRecordedTrainingRunConfig(
            totalSteps: 1,
            previewInterval: 0,
            learningRates: recordedTrainingSmokeLearningRates(),
            densification: FastGSDensificationConfig(
                densifyFromStep: 0,
                densifyUntilStep: 10,
                densificationInterval: 1,
                gradThreshold: Float.greatestFiniteMagnitude,
                gradAbsThreshold: Float.greatestFiniteMagnitude,
                minOpacity: 0.99,
                maxScreenSize: 0,
                maxWorldScaleFactor: 0,
                pruneGaussians: true
            )
        )

        let result = try FastGSRecordedTrainingPreview.run(
            scene: scene,
            config: config,
            pruneSummary: { summaries.append($0) }
        )

        let summary = try XCTUnwrap(summaries.first)
        XCTAssertEqual(summary.step, 1)
        XCTAssertEqual(summary.reason, "densify_prune")
        XCTAssertEqual(summary.beforeCount, scene.manifest.pointCount)
        XCTAssertEqual(summary.afterCount, 1)
        XCTAssertEqual(summary.prunedCount, scene.manifest.pointCount - 1)
        XCTAssertEqual(summary.keptCount, 1)
        XCTAssertEqual(summary.clonedCount, 0)
        XCTAssertEqual(summary.splitSourceCount, 0)
        XCTAssertEqual(summary.splitChildCount, 0)
        XCTAssertEqual(summary.scoringSampleCount, 1)
        XCTAssertEqual(result.pointCount, 1)
        XCTAssertEqual(result.parameters?.gaussianCount, 1)
        XCTAssertEqual(result.renderRGBA.count, scene.manifest.width * scene.manifest.height * 4)
    }

    func testRecordedTrainingPreview200StepsWritesSideBySidePNGsUnderXcode() throws {
        guard FileManager.default.fileExists(atPath: recordedFullManifestURL.path) else {
            throw XCTSkip("Generate \(recordedFullManifestURL.path) first.")
        }

        Memory.cacheLimit = 4 * 1024 * 1024 * 1024

        let scene = try FastGSRecordedForwardScene(manifestURL: recordedFullManifestURL)
        var parameters = try scene.initialTrainableParameters()
        let initialParameters = parameters
        let target = try scene.targetOutColor()
        var optimizer = FastGSAdamOptimizer(
            learningRates: FastGSAdamLearningRates(
                means3D: 5e-5,
                dc: 5e-4,
                sh: 5e-4,
                opacityLogits: 5e-4,
                scales: 5e-5,
                rotations: 5e-5
            )
        )
        let outputDirectory = URL(fileURLWithPath: "/private/tmp/fastgs_swift_training_preview", isDirectory: true)
        try? FileManager.default.removeItem(at: outputDirectory)
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)

        var firstLoss: Float?
        var lastLoss: Float = .infinity
        var debugRows = [[String: Any]]()
        for step in 1...200 {
            let previousParameters = parameters
            let result = FastGSTrainingStageGraph.valueAndGrad(
                scene: scene,
                parameters: parameters,
                target: target
            )
            let loss = result.loss.item(Float.self)
            XCTAssertTrue(loss.isFinite)
            XCTAssertGreaterThanOrEqual(loss, 0)
            XCTAssertEqual(result.gradients.count, 6)
            XCTAssertTrue(result.gradients.contains { hasFiniteNonZeroValues($0) })
            firstLoss = firstLoss ?? loss
            lastLoss = loss

            parameters = optimizer.update(
                parameters: parameters,
                gradients: trainableGradients(from: result.gradients)
            )

            if step == 1 || step % 20 == 0 {
                debugRows.append(
                    trainingDebugRow(
                        step: step,
                        loss: loss,
                        gradients: result.gradients,
                        parameters: parameters,
                        previousParameters: previousParameters,
                        initialParameters: initialParameters,
                        memory: Memory.snapshot()
                    )
                )
            }

            if step % 20 == 0 {
                let render = FastGSTrainingStageGraph.render(scene: scene, parameters: parameters)
                let url = outputDirectory.appendingPathComponent(String(format: "step_%03d_sbs.png", step))
                try writeSideBySidePNG(
                    target: target,
                    render: render,
                    width: scene.manifest.width,
                    height: scene.manifest.height,
                    to: url
                )
            }
        }

        try writeTrainingDebugSummary(debugRows, to: outputDirectory)
        XCTAssertEqual(optimizer.state?.step, 200)
        XCTAssertLessThanOrEqual(lastLoss, (firstLoss ?? lastLoss) * 10)
        XCTAssertTrue(FileManager.default.fileExists(atPath: outputDirectory.appendingPathComponent("step_200_sbs.png").path))
    }
}

private func preprocessBackwardSmokeForwardOutput(count: Int) -> FastGSPreprocessOutput {
    FastGSPreprocessOutput(
        radii: MLXArray.ones([count], dtype: .int32),
        xy: MLXArray.zeros([count, 2], dtype: .float32),
        depths: MLXArray.zeros([count], dtype: .float32),
        cov3D: MLXArray.zeros([count, 6], dtype: .float32),
        rgb: MLXArray.zeros([count, 3], dtype: .float32),
        conicOpacity: MLXArray.zeros([count, 4], dtype: .float32),
        tilesTouched: MLXArray.zeros([count], dtype: .uint32),
        clamped: MLXArray.zeros([count, 3], dtype: .bool),
        viewspacePoints: MLXArray.zeros([count, 4], dtype: .float32)
    )
}

private func preprocessBackwardSmokeCotangents(like output: FastGSPreprocessOutput) -> FastGSPreprocessCotangents {
    let cotangents = FastGSPreprocessCotangents(
        radii: MLXArray.zeros(output.radii.shape, dtype: output.radii.dtype),
        xy: MLXArray.ones(output.xy.shape, dtype: output.xy.dtype),
        depths: MLXArray.ones(output.depths.shape, dtype: output.depths.dtype),
        cov3D: MLXArray.ones(output.cov3D.shape, dtype: output.cov3D.dtype),
        rgb: MLXArray.ones(output.rgb.shape, dtype: output.rgb.dtype),
        conicOpacity: MLXArray.ones(output.conicOpacity.shape, dtype: output.conicOpacity.dtype),
        tilesTouched: MLXArray.zeros(output.tilesTouched.shape, dtype: output.tilesTouched.dtype),
        clamped: MLXArray.zeros(output.clamped.shape, dtype: output.clamped.dtype),
        viewspacePoints: MLXArray.ones(output.viewspacePoints.shape, dtype: output.viewspacePoints.dtype)
    )
    eval([
        cotangents.radii,
        cotangents.xy,
        cotangents.depths,
        cotangents.cov3D,
        cotangents.rgb,
        cotangents.conicOpacity,
        cotangents.tilesTouched,
        cotangents.clamped,
        cotangents.viewspacePoints,
    ])
    return cotangents
}

private struct RasterizeBackwardReference: Decodable {
    var gradients: RasterizeBackwardGradients
}

private struct RasterizeBackwardGradients: Decodable {
    var means2D: GradientReference
    var colors: GradientReference
    var conicOpacity: GradientReference
    var viewspacePoints: GradientReference
}

private struct PreprocessBackwardReference: Decodable {
    var loss: Double
    var gradients: PreprocessBackwardGradients
    var fixtures: PreprocessBackwardFixtures
}

private struct PreprocessBackwardFixtures: Decodable {
    var precomputedColor: PreprocessBackwardFixtureReference
    var shDegree3: PreprocessBackwardSHFixtureReference
}

private struct PreprocessBackwardFixtureReference: Decodable {
    var loss: Double
    var gradients: PreprocessBackwardGradients
}

private struct PreprocessBackwardSHFixtureReference: Decodable {
    var loss: Double
    var gradients: PreprocessBackwardSHGradients
}

private struct PreprocessBackwardGradients: Decodable {
    var means3D: GradientReference
    var colorsPrecomputed: GradientReference
    var opacities: GradientReference
    var scales: GradientReference
    var rotations: GradientReference
    var viewspacePoints: GradientReference
}

private struct PreprocessBackwardSHGradients: Decodable {
    var means3D: GradientReference
    var dc: GradientReference
    var sh: GradientReference
    var opacities: GradientReference
    var scales: GradientReference
    var rotations: GradientReference
    var viewspacePoints: GradientReference
}

private struct GradientReference: Decodable {
    var shape: [Int]
    var sum: Double
    var absSum: Double
    var maxAbs: Double
    var samples: [Double]
    var sampleIds: [Int]
}

private func assertGradientSummary(
    _ array: MLXArray,
    _ reference: GradientReference,
    accuracy: Double,
    file: StaticString = #filePath,
    line: UInt = #line
) {
    let values = array.asArray(Float.self).map(Double.init)
    XCTAssertEqual(array.shape, reference.shape, file: file, line: line)
    XCTAssertEqual(values.reduce(0, +), reference.sum, accuracy: accuracy, file: file, line: line)
    XCTAssertEqual(values.map(abs).reduce(0, +), reference.absSum, accuracy: accuracy, file: file, line: line)
    XCTAssertEqual(values.map(abs).max() ?? 0, reference.maxAbs, accuracy: accuracy, file: file, line: line)

    let samples = reference.sampleIds.map { values[$0] }
    XCTAssertEqual(samples.count, reference.samples.count, file: file, line: line)
    for (actual, expected) in zip(samples, reference.samples) {
        XCTAssertEqual(actual, expected, accuracy: accuracy, file: file, line: line)
    }
}

private func assertRasterizeBackwardSkeleton(
    preprocess: FastGSPreprocessOutput,
    binning: FastGSBinningOutput,
    background: MLXArray,
    output: FastGSRasterizeBackwardOutput,
    file: StaticString = #filePath,
    line: UInt = #line
) {
    let params = FastGSPreprocessParityFixture.rasterizeLargeE2EParams
    XCTAssertEqual(output.ranges.shape, binning.ranges.shape, file: file, line: line)
    XCTAssertEqual(output.pointList.shape, binning.pointList.shape, file: file, line: line)
    XCTAssertEqual(output.bucketOffsets.shape, binning.bucketOffsets.shape, file: file, line: line)
    XCTAssertEqual(output.means2D.shape, preprocess.xy.shape, file: file, line: line)
    XCTAssertEqual(output.colors.shape, preprocess.rgb.shape, file: file, line: line)
    XCTAssertEqual(output.conicOpacity.shape, preprocess.conicOpacity.shape, file: file, line: line)
    XCTAssertEqual(output.background.shape, background.shape, file: file, line: line)
    XCTAssertEqual(output.radii.shape, preprocess.radii.shape, file: file, line: line)
    XCTAssertEqual(output.metricMap.shape, [params.imageWidth * params.imageHeight], file: file, line: line)
    XCTAssertEqual(output.viewspacePoints.shape, preprocess.viewspacePoints.shape, file: file, line: line)
    XCTAssertEqual(output.means2D.dtype, .float32, file: file, line: line)
    XCTAssertEqual(output.colors.dtype, .float32, file: file, line: line)
    XCTAssertEqual(output.conicOpacity.dtype, .float32, file: file, line: line)
    XCTAssertEqual(output.background.dtype, .float32, file: file, line: line)
    XCTAssertEqual(output.viewspacePoints.dtype, .float32, file: file, line: line)
    XCTAssertTrue(output.colors.asArray(Float.self).contains { abs($0) > 1e-7 }, file: file, line: line)
    XCTAssertTrue(output.conicOpacity.asArray(Float.self).contains { abs($0) > 1e-7 }, file: file, line: line)
    XCTAssertTrue(output.viewspacePoints.asArray(Float.self).contains { abs($0) > 1e-7 }, file: file, line: line)
}

private func assertPreprocessBackwardSkeleton(
    input: FastGSPreprocessInput,
    output: FastGSPreprocessBackwardOutput,
    file: StaticString = #filePath,
    line: UInt = #line
) {
    XCTAssertEqual(output.means3D.shape, input.means3D.shape, file: file, line: line)
    XCTAssertEqual(output.dc.shape, input.dc.shape, file: file, line: line)
    XCTAssertEqual(output.sh.shape, input.sh.shape, file: file, line: line)
    XCTAssertEqual(output.colorsPrecomputed.shape, input.colorsPrecomputed.shape, file: file, line: line)
    XCTAssertEqual(output.opacities.shape, input.opacities.shape, file: file, line: line)
    XCTAssertEqual(output.scales.shape, input.scales.shape, file: file, line: line)
    XCTAssertEqual(output.rotations.shape, input.rotations.shape, file: file, line: line)
    XCTAssertEqual(output.cov3DPrecomputed.shape, input.cov3DPrecomputed.shape, file: file, line: line)
    XCTAssertEqual(output.viewMatrix.shape, input.viewMatrix.shape, file: file, line: line)
    XCTAssertEqual(output.projectionMatrix.shape, input.projectionMatrix.shape, file: file, line: line)
    XCTAssertEqual(output.cameraPosition.shape, input.cameraPosition.shape, file: file, line: line)
    XCTAssertEqual(output.viewspacePoints.shape, input.viewspacePoints.shape, file: file, line: line)
    XCTAssertTrue(output.means3D.asArray(Float.self).contains { abs($0) > 1e-7 }, file: file, line: line)
    XCTAssertTrue(output.opacities.asArray(Float.self).contains { abs($0) > 1e-7 }, file: file, line: line)
}

private func assertBinning(
    _ output: FastGSBinningOutput,
    pointOffsets: [UInt32],
    pointListKeysUnsorted: [UInt64],
    pointListUnsorted: [UInt32],
    pointListKeys: [UInt64],
    pointList: [UInt32]?,
    ranges: [UInt32],
    bucketCount: [UInt32],
    bucketOffsets: [UInt32],
    file: StaticString = #filePath,
    line: UInt = #line
) {
    XCTAssertEqual(output.pointOffsets.asArray(UInt32.self), pointOffsets, file: file, line: line)
    XCTAssertEqual(output.pointListKeysUnsorted.asArray(UInt64.self), pointListKeysUnsorted, file: file, line: line)
    XCTAssertEqual(output.pointListUnsorted.asArray(UInt32.self), pointListUnsorted, file: file, line: line)
    XCTAssertEqual(output.pointListKeys.asArray(UInt64.self), pointListKeys, file: file, line: line)
    if let pointList {
        XCTAssertEqual(output.pointList.asArray(UInt32.self), pointList, file: file, line: line)
    }
    XCTAssertEqual(output.ranges.asArray(UInt32.self), ranges, file: file, line: line)
    XCTAssertEqual(output.bucketCount.asArray(UInt32.self), bucketCount, file: file, line: line)
    XCTAssertEqual(output.bucketOffsets.asArray(UInt32.self), bucketOffsets, file: file, line: line)
}

private func assertRasterizeE2E(
    _ output: FastGSRasterizeOutput,
    file: StaticString = #filePath,
    line: UInt = #line
) {
    XCTAssertEqual(output.bucketToTile.shape, [16 * 256], file: file, line: line)
    XCTAssertEqual(output.sampledT.shape, [16 * 256], file: file, line: line)
    XCTAssertEqual(output.sampledAr.shape, [3 * 16 * 256], file: file, line: line)
    XCTAssertEqual(output.finalT.shape, [64 * 64], file: file, line: line)
    XCTAssertEqual(output.nContrib.shape, [64 * 64], file: file, line: line)
    XCTAssertEqual(output.maxContrib.shape, [16], file: file, line: line)
    XCTAssertEqual(output.pixelColors.shape, [3, 64 * 64], file: file, line: line)
    XCTAssertEqual(output.outColor.shape, [3, 64 * 64], file: file, line: line)

    let bucketToTile = output.bucketToTile.asArray(UInt32.self)
    XCTAssertEqual(
        Array(bucketToTile.prefix(16)),
        FastGSPreprocessParityFixture.expectedRasterizeE2EBucketToTilePrefix,
        file: file,
        line: line
    )
    XCTAssertEqual(
        Array(bucketToTile.dropFirst(16)),
        Array(repeating: UInt32(0), count: bucketToTile.count - 16),
        file: file,
        line: line
    )
    assertClose(
        Array(output.sampledT.asArray(Float.self).prefix(32)),
        FastGSPreprocessParityFixture.expectedRasterizeE2ESampledTPrefix,
        file: file,
        line: line
    )
    assertClose(
        Array(output.sampledAr.asArray(Float.self).prefix(12)),
        FastGSPreprocessParityFixture.expectedRasterizeE2ESampledArPrefix,
        file: file,
        line: line
    )

    let outColor = output.outColor.asArray(Float.self)
    let pixelColors = output.pixelColors.asArray(Float.self)
    assertClose(channelSums(outColor, channels: 3), FastGSPreprocessParityFixture.expectedRasterizeE2EOutColorSums, accuracy: 1e-3, file: file, line: line)
    assertClose(channelSums(pixelColors, channels: 3), FastGSPreprocessParityFixture.expectedRasterizeE2EPixelColorSums, accuracy: 1e-3, file: file, line: line)
    XCTAssertEqual(output.nContrib.asArray(UInt32.self).reduce(0, +), FastGSPreprocessParityFixture.expectedRasterizeE2ENContribSum, file: file, line: line)
    XCTAssertEqual(output.maxContrib.asArray(UInt32.self), FastGSPreprocessParityFixture.expectedRasterizeE2EMaxContrib, file: file, line: line)
    XCTAssertEqual(output.metricCount.asArray(Int32.self), [0, 0], file: file, line: line)

    let finalT = output.finalT.asArray(Float.self)
    XCTAssertEqual(finalT.reduce(0, +), FastGSPreprocessParityFixture.expectedRasterizeE2EFinalTSum, accuracy: 1e-3, file: file, line: line)
    assertClose(samples(outColor, ids: FastGSPreprocessParityFixture.expectedRasterizeE2ESampleIDs, channels: 3), FastGSPreprocessParityFixture.expectedRasterizeE2EOutColorSamples, accuracy: 1e-5, file: file, line: line)
    assertClose(samples(pixelColors, ids: FastGSPreprocessParityFixture.expectedRasterizeE2ESampleIDs, channels: 3), FastGSPreprocessParityFixture.expectedRasterizeE2EPixelColorSamples, accuracy: 1e-5, file: file, line: line)
    assertClose(FastGSPreprocessParityFixture.expectedRasterizeE2ESampleIDs.map { finalT[$0] }, FastGSPreprocessParityFixture.expectedRasterizeE2EFinalTSamples, accuracy: 1e-5, file: file, line: line)
    XCTAssertEqual(
        FastGSPreprocessParityFixture.expectedRasterizeE2ESampleIDs.map { output.nContrib.asArray(UInt32.self)[$0] },
        FastGSPreprocessParityFixture.expectedRasterizeE2ENContribSamples,
        file: file,
        line: line
    )
}

private func assertRasterizeLargeE2E(
    preprocess: FastGSPreprocessOutput,
    binning: FastGSBinningOutput,
    output: FastGSRasterizeOutput,
    file: StaticString = #filePath,
    line: UInt = #line
) {
    XCTAssertEqual(preprocess.radii.asArray(Int32.self), FastGSPreprocessParityFixture.expectedRasterizeLargeE2ERadii, file: file, line: line)
    XCTAssertEqual(preprocess.tilesTouched.asArray(UInt32.self), FastGSPreprocessParityFixture.expectedRasterizeLargeE2ETilesTouched, file: file, line: line)
    assertClose(preprocess.xy.asArray(Float.self), FastGSPreprocessParityFixture.expectedRasterizeLargeE2EXY, accuracy: 1e-5, file: file, line: line)
    assertClose(preprocess.depths.asArray(Float.self), FastGSPreprocessParityFixture.expectedRasterizeLargeE2EDepths, accuracy: 1e-5, file: file, line: line)

    XCTAssertEqual(binning.pointOffsets.asArray(UInt32.self), FastGSPreprocessParityFixture.expectedRasterizeLargeE2EPointOffsets, file: file, line: line)
    XCTAssertEqual(binning.ranges.asArray(UInt32.self), FastGSPreprocessParityFixture.expectedRasterizeLargeE2ERanges, file: file, line: line)
    XCTAssertEqual(binning.bucketCount.asArray(UInt32.self), FastGSPreprocessParityFixture.expectedRasterizeLargeE2EBucketCount, file: file, line: line)
    XCTAssertEqual(binning.bucketOffsets.asArray(UInt32.self), FastGSPreprocessParityFixture.expectedRasterizeLargeE2EBucketOffsets, file: file, line: line)

    XCTAssertEqual(output.bucketToTile.shape, [15 * 256], file: file, line: line)
    XCTAssertEqual(output.sampledT.shape, [15 * 256], file: file, line: line)
    XCTAssertEqual(output.sampledAr.shape, [3 * 15 * 256], file: file, line: line)
    XCTAssertEqual(output.finalT.shape, [80 * 48], file: file, line: line)
    XCTAssertEqual(output.nContrib.shape, [80 * 48], file: file, line: line)
    XCTAssertEqual(output.maxContrib.shape, [15], file: file, line: line)
    XCTAssertEqual(output.pixelColors.shape, [3, 80 * 48], file: file, line: line)
    XCTAssertEqual(output.outColor.shape, [3, 80 * 48], file: file, line: line)

    let bucketToTile = output.bucketToTile.asArray(UInt32.self)
    XCTAssertEqual(Array(bucketToTile.prefix(15)), FastGSPreprocessParityFixture.expectedRasterizeLargeE2EBucketToTilePrefix, file: file, line: line)
    XCTAssertEqual(Array(bucketToTile.dropFirst(15)), Array(repeating: UInt32(0), count: bucketToTile.count - 15), file: file, line: line)
    assertClose(Array(output.sampledT.asArray(Float.self).prefix(48)), FastGSPreprocessParityFixture.expectedRasterizeLargeE2ESampledTPrefix, file: file, line: line)
    assertClose(Array(output.sampledAr.asArray(Float.self).prefix(24)), FastGSPreprocessParityFixture.expectedRasterizeLargeE2ESampledArPrefix, file: file, line: line)

    let outColor = output.outColor.asArray(Float.self)
    let pixelColors = output.pixelColors.asArray(Float.self)
    assertClose(channelSums(outColor, channels: 3), FastGSPreprocessParityFixture.expectedRasterizeLargeE2EOutColorSums, accuracy: 2e-2, file: file, line: line)
    assertClose(channelSums(pixelColors, channels: 3), FastGSPreprocessParityFixture.expectedRasterizeLargeE2EPixelColorSums, accuracy: 2e-2, file: file, line: line)
    XCTAssertEqual(output.nContrib.asArray(UInt32.self).reduce(0, +), FastGSPreprocessParityFixture.expectedRasterizeLargeE2ENContribSum, file: file, line: line)
    XCTAssertEqual(output.maxContrib.asArray(UInt32.self), FastGSPreprocessParityFixture.expectedRasterizeLargeE2EMaxContrib, file: file, line: line)
    XCTAssertEqual(output.metricCount.asArray(Int32.self), [0, 0, 0, 0, 0], file: file, line: line)

    let finalT = output.finalT.asArray(Float.self)
    XCTAssertEqual(finalT.reduce(0, +), FastGSPreprocessParityFixture.expectedRasterizeLargeE2EFinalTSum, accuracy: 2e-3, file: file, line: line)
    assertClose(samples(outColor, ids: FastGSPreprocessParityFixture.expectedRasterizeLargeE2ESampleIDs, channels: 3), FastGSPreprocessParityFixture.expectedRasterizeLargeE2EOutColorSamples, accuracy: 1e-5, file: file, line: line)
    assertClose(samples(pixelColors, ids: FastGSPreprocessParityFixture.expectedRasterizeLargeE2ESampleIDs, channels: 3), FastGSPreprocessParityFixture.expectedRasterizeLargeE2EPixelColorSamples, accuracy: 1e-5, file: file, line: line)
    assertClose(FastGSPreprocessParityFixture.expectedRasterizeLargeE2ESampleIDs.map { finalT[$0] }, FastGSPreprocessParityFixture.expectedRasterizeLargeE2EFinalTSamples, accuracy: 1e-5, file: file, line: line)
    XCTAssertEqual(
        FastGSPreprocessParityFixture.expectedRasterizeLargeE2ESampleIDs.map { output.nContrib.asArray(UInt32.self)[$0] },
        FastGSPreprocessParityFixture.expectedRasterizeLargeE2ENContribSamples,
        file: file,
        line: line
    )
}

private func channelSums(_ values: [Float], channels: Int) -> [Float] {
    let count = values.count / channels
    return (0..<channels).map { channel in
        values[(channel * count)..<((channel + 1) * count)].reduce(0, +)
    }
}

private func samples(_ values: [Float], ids: [Int], channels: Int) -> [Float] {
    let count = values.count / channels
    return (0..<channels).flatMap { channel in
        ids.map { values[channel * count + $0] }
    }
}

private func hasFiniteNonZeroValues(_ array: MLXArray) -> Bool {
    array.asArray(Float.self).contains { $0.isFinite && abs($0) > 1e-7 }
}

private func maxAbsDiff(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
    let left = lhs.asArray(Float.self)
    let right = rhs.asArray(Float.self)
    return zip(left, right).map { abs($0 - $1) }.max() ?? 0
}

private func recordedTrainingSmokeLearningRates() -> FastGSAdamLearningRates {
    FastGSAdamLearningRates(
        means3D: 1e-4,
        dc: 1e-3,
        sh: 1e-3,
        opacityLogits: 1e-3,
        scales: 1e-4,
        rotations: 1e-4
    )
}

private func trainableGradients(from gradients: [MLXArray]) -> FastGSTrainableGradients {
    precondition(gradients.count == 6, "recorded training smoke expects six trainable gradients")
    return FastGSTrainableGradients(
        means3D: gradients[0],
        dc: gradients[1],
        sh: gradients[2],
        opacityLogits: gradients[3],
        scales: gradients[4],
        rotations: gradients[5]
    )
}

private func writeSideBySidePNG(
    target: MLXArray,
    render: MLXArray,
    width: Int,
    height: Int,
    to url: URL
) throws {
    let left = FastGSImageExport.rgbaBytes(outColor: target, width: width, height: height)
    let right = FastGSImageExport.rgbaBytes(outColor: render, width: width, height: height)
    var combined = [UInt8](repeating: 0, count: width * 2 * height * 4)
    for row in 0..<height {
        let sourceStart = row * width * 4
        let targetStart = row * width * 2 * 4
        combined.replaceSubrange(targetStart..<(targetStart + width * 4), with: left[sourceStart..<(sourceStart + width * 4)])
        combined.replaceSubrange((targetStart + width * 4)..<(targetStart + width * 2 * 4), with: right[sourceStart..<(sourceStart + width * 4)])
    }
    try FastGSImageExport.writePNG(rgbaBytes: combined, width: width * 2, height: height, to: url)
}

private func trainingDebugRow(
    step: Int,
    loss: Float,
    gradients: [MLXArray],
    parameters: FastGSTrainableParameters,
    previousParameters: FastGSTrainableParameters,
    initialParameters: FastGSTrainableParameters,
    memory: Memory.Snapshot
) -> [String: Any] {
    let names = ["means3D", "dc", "sh", "opacityLogits", "scales", "rotations"]
    let currentArrays = parameters.arrays
    let previousArrays = previousParameters.arrays
    let initialArrays = initialParameters.arrays
    var fields = [String: Any]()
    for index in names.indices {
        fields[names[index]] = [
            "gradient": numericSummary(gradients[index]),
            "updateMaxAbs": maxAbsDiff(currentArrays[index], previousArrays[index]),
            "deltaFromInitialMaxAbs": maxAbsDiff(currentArrays[index], initialArrays[index]),
        ]
    }
    return [
        "step": step,
        "loss": loss,
        "memory": [
            "active": memory.activeMemory,
            "cache": memory.cacheMemory,
            "peak": memory.peakMemory,
        ],
        "fields": fields,
    ]
}

private func numericSummary(_ array: MLXArray) -> [String: Any] {
    let values = array.asArray(Float.self)
    let finiteValues = values.filter(\.isFinite)
    return [
        "shape": array.shape,
        "sum": finiteValues.reduce(Float(0), +),
        "absSum": finiteValues.reduce(Float(0)) { $0 + abs($1) },
        "maxAbs": finiteValues.map(abs).max() ?? 0,
        "nonZeroCount": finiteValues.filter { abs($0) > 1e-7 }.count,
    ]
}

private func writeTrainingDebugSummary(_ rows: [[String: Any]], to directory: URL) throws {
    let jsonURL = directory.appendingPathComponent("debug_summary.json")
    let jsonData = try JSONSerialization.data(withJSONObject: rows, options: [.prettyPrinted, .sortedKeys])
    try jsonData.write(to: jsonURL)

    let csvURL = directory.appendingPathComponent("debug_summary.csv")
    let fieldNames = ["means3D", "dc", "sh", "opacityLogits", "scales", "rotations"]
    var lines = [
        "step,loss,field,grad_sum,grad_abs_sum,grad_max_abs,grad_nonzero_count,update_max_abs,delta_from_initial_max_abs,memory_active,memory_cache,memory_peak"
    ]
    for row in rows {
        let step = row["step"] as? Int ?? 0
        let loss = row["loss"] as? Float ?? .nan
        let memory = row["memory"] as? [String: Int] ?? [:]
        let fields = row["fields"] as? [String: Any] ?? [:]
        for fieldName in fieldNames {
            let field = fields[fieldName] as? [String: Any] ?? [:]
            let gradient = field["gradient"] as? [String: Any] ?? [:]
            lines.append([
                "\(step)",
                "\(loss)",
                fieldName,
                "\(gradient["sum"] as? Float ?? .nan)",
                "\(gradient["absSum"] as? Float ?? .nan)",
                "\(gradient["maxAbs"] as? Float ?? .nan)",
                "\(gradient["nonZeroCount"] as? Int ?? 0)",
                "\(field["updateMaxAbs"] as? Float ?? .nan)",
                "\(field["deltaFromInitialMaxAbs"] as? Float ?? .nan)",
                "\(memory["active"] ?? 0)",
                "\(memory["cache"] ?? 0)",
                "\(memory["peak"] ?? 0)",
            ].joined(separator: ","))
        }
    }
    try lines.joined(separator: "\n").write(to: csvURL, atomically: true, encoding: .utf8)
}

private func recordedStageSummary(_ stages: FastGSRecordedForwardStages, sampleIDs: [Int]) -> [String: Any] {
    let radii = stages.preprocess.radii.asArray(Int32.self)
    let xy = stages.preprocess.xy.asArray(Float.self)
    let depths = stages.preprocess.depths.asArray(Float.self)
    let rgb = stages.preprocess.rgb.asArray(Float.self)
    let conic = stages.preprocess.conicOpacity.asArray(Float.self)
    let tilesTouched = stages.preprocess.tilesTouched.asArray(UInt32.self)
    let visibleIndices = radii.indices.filter { radii[$0] > 0 }
    let pointListKeys = stages.binning.pointListKeys.asArray(UInt64.self)
    let pointList = stages.binning.pointList.asArray(UInt32.self)
    let bucketCount = stages.binning.bucketCount.asArray(UInt32.self)
    let bucketOffsets = stages.binning.bucketOffsets.asArray(UInt32.self)
    let ranges = stages.binning.ranges.asArray(UInt32.self)
    let outColor = stages.rasterize.outColor.asArray(Float.self)
    let pixelColors = stages.rasterize.pixelColors.asArray(Float.self)
    let finalT = stages.rasterize.finalT.asArray(Float.self)
    let nContrib = stages.rasterize.nContrib.asArray(UInt32.self)
    let maxContrib = stages.rasterize.maxContrib.asArray(UInt32.self)

    return [
        "preprocess": [
            "visibleCount": visibleIndices.count,
            "radiiSum": radii.reduce(0, +),
            "tilesTouchedSum": tilesTouched.reduce(UInt32(0), +),
            "depthSumVisible": visibleIndices.reduce(Float(0)) { $0 + depths[$1] },
            "xysSumVisible": [
                visibleIndices.reduce(Float(0)) { $0 + xy[$1 * 2] },
                visibleIndices.reduce(Float(0)) { $0 + xy[$1 * 2 + 1] },
            ],
            "rgbSums": rowMajorColumnSums(rgb, columns: 3),
            "conicOpacitySums": rowMajorColumnSums(conic, columns: 4),
            "radiiPrefix": Array(radii.prefix(16)),
            "tilesTouchedPrefix": Array(tilesTouched.prefix(16)),
        ],
        "binning": [
            "numRendered": Int(stages.binning.pointOffsets.asArray(UInt32.self).last ?? 0),
            "pointListKeyPrefix": Array(pointListKeys.prefix(16)).map(String.init),
            "pointListPrefix": Array(pointList.prefix(16)),
            "pointListKeyChecksum": String(pointListKeys.prefix(4096).reduce(UInt64(0)) { $0 &+ $1 }),
            "pointListChecksum": pointList.prefix(4096).reduce(UInt64(0)) { $0 + UInt64($1) },
        ],
        "tile": [
            "bucketSum": Int(bucketOffsets.last ?? 0),
            "bucketCountSum": bucketCount.reduce(UInt32(0), +),
            "bucketCountPrefix": Array(bucketCount.prefix(16)),
            "bucketOffsetPrefix": Array(bucketOffsets.prefix(16)),
            "rangesPrefix": Array(ranges.prefix(32)),
        ],
        "rasterize": [
            "outColorSums": channelSums(outColor, channels: 3),
            "pixelColorSums": channelSums(pixelColors, channels: 3),
            "finalTSum": finalT.reduce(0, +),
            "nContribSum": nContrib.reduce(UInt32(0), +),
            "maxContribSum": maxContrib.reduce(UInt32(0), +),
            "outSamples": samples(outColor, ids: sampleIDs, channels: 3),
            "finalTSamples": sampleIDs.map { finalT[$0] },
            "nContribSamples": sampleIDs.map { nContrib[$0] },
        ],
    ]
}

private func rowMajorColumnSums(_ values: [Float], columns: Int) -> [Float] {
    let rows = values.count / columns
    return (0..<columns).map { column in
        (0..<rows).reduce(Float(0)) { $0 + values[$1 * columns + column] }
    }
}

private func makeBGRA32PixelBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
    let attributes: [CFString: Any] = [
        kCVPixelBufferIOSurfacePropertiesKey: [:],
    ]
    var pixelBuffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(
        kCFAllocatorDefault,
        width,
        height,
        kCVPixelFormatType_32BGRA,
        attributes as CFDictionary,
        &pixelBuffer
    )
    XCTAssertEqual(status, kCVReturnSuccess)

    let buffer = try XCTUnwrap(pixelBuffer)
    CVPixelBufferLockBaseAddress(buffer, [])
    defer {
        CVPixelBufferUnlockBaseAddress(buffer, [])
    }

    let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
    let baseAddress = try XCTUnwrap(CVPixelBufferGetBaseAddress(buffer))
    let bytes = baseAddress.assumingMemoryBound(to: UInt8.self)
    let pixels: [UInt8] = [
        10, 20, 30, 255,
        50, 60, 70, 128,
    ]

    for row in 0..<height {
        let rowStart = row * bytesPerRow
        for column in 0..<width {
            let source = (row * width + column) * 4
            let destination = rowStart + column * 4
            bytes[destination + 0] = pixels[source + 0]
            bytes[destination + 1] = pixels[source + 1]
            bytes[destination + 2] = pixels[source + 2]
            bytes[destination + 3] = pixels[source + 3]
        }
    }

    return buffer
}

private struct FastGSRenderBenchmarkReport {
    var title = "FastGS recorded render benchmark"
    var width: Int
    var height: Int
    var pointCount: Int
    var warmupFrames: Int
    var secondsPerRound: Double
    var roundFPS: [Double]
    var frameLatencies: [Double]

    var fpsMean: Double { mean(roundFPS) }
    var fpsMedian: Double { percentile(roundFPS, fraction: 0.5) }
    var fpsMin: Double { roundFPS.min() ?? .nan }
    var fpsMax: Double { roundFPS.max() ?? .nan }
    var latencyMeanMilliseconds: Double { mean(frameLatencies) * 1000 }
    var latencyMedianMilliseconds: Double { percentile(frameLatencies, fraction: 0.5) * 1000 }
    var latencyP95Milliseconds: Double { percentile(frameLatencies, fraction: 0.95) * 1000 }
    var latencyMinMilliseconds: Double { (frameLatencies.min() ?? .nan) * 1000 }
    var latencyMaxMilliseconds: Double { (frameLatencies.max() ?? .nan) * 1000 }

    var description: String {
        """
        \(title)
        image: \(width)x\(height)
        points: \(pointCount)
        warmup frames: \(warmupFrames)
        seconds per round: \(format(secondsPerRound))
        measured frames: \(frameLatencies.count)
        round FPS: \(roundFPS.map(format).joined(separator: ", "))
        FPS mean/median/min/max: \(format(fpsMean)) / \(format(fpsMedian)) / \(format(fpsMin)) / \(format(fpsMax))
        latency ms mean/median/p95/min/max: \(format(latencyMeanMilliseconds)) / \(format(latencyMedianMilliseconds)) / \(format(latencyP95Milliseconds)) / \(format(latencyMinMilliseconds)) / \(format(latencyMaxMilliseconds))
        """
    }
}

private struct FastGSForwardStageTimingSummary {
    var width: Int
    var height: Int
    var pointCount: Int
    var rounds: Int
    var warmupRounds: Int
    var timings: [FastGSRecordedForwardTimingReport]
    var imageReadbackMilliseconds: [Double]
    var textureUploadMilliseconds: [Double]

    var forwardMeanMilliseconds: Double {
        mean(timings.map(\.totalWithoutImageReadbackMilliseconds))
    }

    var description: String {
        let binning = timings.map(\.binning)
        return """
        FastGS recorded forward stage timing
        image: \(width)x\(height)
        points: \(pointCount)
        warmup rounds: \(warmupRounds)
        measured rounds: \(rounds)
        numRendered mean: \(format(mean(binning.map { Double($0.numRendered) })))
        numTiles: \(binning.first?.numTiles ?? 0)

        stage ms mean / median / p95:
        preprocess: \(summary(timings.map(\.preprocessMilliseconds)))
        binning.total: \(summary(binning.map(\.totalMilliseconds)))
          binning.cumsum: \(summary(binning.map(\.cumsumMilliseconds)))
          binning.numRenderedReadback: \(summary(binning.map(\.numRenderedReadbackMilliseconds)))
          binning.duplicate: \(summary(binning.map(\.duplicateMilliseconds)))
          binning.sortAndTake: \(summary(binning.map(\.sortAndTakeMilliseconds)))
          binning.identifyRanges: \(summary(binning.map(\.identifyRangesMilliseconds)))
          binning.bucketCount: \(summary(binning.map(\.bucketCountMilliseconds)))
          binning.bucketOffsets: \(summary(binning.map(\.bucketOffsetsMilliseconds)))
        rasterize: \(summary(timings.map(\.rasterizeMilliseconds)))
        forward total without image readback: \(summary(timings.map(\.totalWithoutImageReadbackMilliseconds)))
        image readback rgbaBytes/asArray: \(summary(imageReadbackMilliseconds))
        texture upload from CPU rgba: \(summary(textureUploadMilliseconds))
        total with current CPU presentation path: \(summary(zip(timings, imageReadbackMilliseconds).map { $0.totalWithoutImageReadbackMilliseconds + $1 } ))
        """
    }

    private func summary(_ values: [Double]) -> String {
        "\(format(mean(values))) / \(format(percentile(values, fraction: 0.5))) / \(format(percentile(values, fraction: 0.95)))"
    }
}

private struct FastGSBenchmarkMarkerConfig {
    private var values = [String: String]()

    init(url: URL) {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else {
            return
        }
        for rawLine in text.split(whereSeparator: \.isNewline) {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !line.isEmpty, !line.hasPrefix("#"), let equals = line.firstIndex(of: "=") else {
                continue
            }
            let key = line[..<equals].trimmingCharacters(in: .whitespacesAndNewlines)
            let value = line[line.index(after: equals)...].trimmingCharacters(in: .whitespacesAndNewlines)
            values[key] = value
        }
    }

    func intValue(_ key: String) -> Int? {
        values[key].flatMap(Int.init)
    }

    func doubleValue(_ key: String) -> Double? {
        values[key].flatMap(Double.init)
    }

    func boolValue(_ key: String) -> Bool? {
        values[key].flatMap(parseBool)
    }
}

private extension Dictionary where Key == String, Value == String {
    func intValue(_ key: String, default defaultValue: Int) -> Int {
        self[key].flatMap(Int.init) ?? defaultValue
    }

    func doubleValue(_ key: String, default defaultValue: Double) -> Double {
        self[key].flatMap(Double.init) ?? defaultValue
    }

    func boolValue(_ key: String, default defaultValue: Bool) -> Bool {
        self[key].flatMap(parseBool) ?? defaultValue
    }
}

private func parseBool(_ value: String) -> Bool? {
    switch value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
    case "1", "true", "yes", "y", "on":
        return true
    case "0", "false", "no", "n", "off":
        return false
    default:
        return nil
    }
}

private func mean(_ values: [Double]) -> Double {
    guard !values.isEmpty else {
        return .nan
    }
    return values.reduce(0, +) / Double(values.count)
}

private func percentile(_ values: [Double], fraction: Double) -> Double {
    guard !values.isEmpty else {
        return .nan
    }
    let sorted = values.sorted()
    let clampedFraction = min(max(fraction, 0), 1)
    let index = Int((Double(sorted.count - 1) * clampedFraction).rounded())
    return sorted[index]
}

private func format(_ value: Double) -> String {
    String(format: "%.3f", value)
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
