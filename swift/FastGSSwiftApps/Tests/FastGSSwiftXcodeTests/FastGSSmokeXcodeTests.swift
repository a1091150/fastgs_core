import FastGSSwift
import CoreVideo
import Metal
import MLX
import XCTest

final class FastGSSmokeXcodeTests: XCTestCase {
    private let recordedManifestURL = URL(fileURLWithPath: "/private/tmp/fastgs_recorded_reference/recorded_manifest.json")
    private let recordedLargeManifestURL = URL(fileURLWithPath: "/private/tmp/fastgs_recorded_reference_16384/recorded_manifest.json")
    private let rasterizeBackwardReferenceURL = URL(fileURLWithPath: "/private/tmp/fastgs_rasterize_backward_ref.json")

    func testAdamOptimizerAppliesSyntheticGradientStepUnderXcode() {
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

        assertClose(updated.means3D.asArray(Float.self), [0.99, 1.99, 2.99, 3.99, 4.99, 5.99])
        assertClose(updated.dc.asArray(Float.self), [0.22, 0.42, 0.62, 0.82, 1.02, 1.22])
        assertClose(updated.sh.asArray(Float.self), [Float](repeating: 0.07, count: 12))
        assertClose(updated.opacities.asArray(Float.self), [0.46, 0.66])
        assertClose(updated.scales.asArray(Float.self), [Float](repeating: 0.35, count: 6))
        assertClose(updated.rotations.asArray(Float.self), [0.94, -0.06, -0.06, -0.06, 0.94, -0.06, -0.06, -0.06])
        assertClose(updated.cov3DPrecomputed?.asArray(Float.self) ?? [], [Float](repeating: 0.12, count: 12))

        XCTAssertEqual(optimizer.state?.step, 1)
        XCTAssertEqual(optimizer.stateArrays().count, 14)
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
            opacities: parameters.opacities,
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
