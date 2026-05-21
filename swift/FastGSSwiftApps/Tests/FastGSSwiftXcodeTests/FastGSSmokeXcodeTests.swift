import FastGSSwift
import CoreVideo
import Metal
import MLX
import XCTest

final class FastGSSmokeXcodeTests: XCTestCase {
    private let recordedManifestURL = URL(fileURLWithPath: "/private/tmp/fastgs_recorded_reference/recorded_manifest.json")

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
        guard FileManager.default.fileExists(atPath: recordedManifestURL.path) else {
            throw XCTSkip("Generate /private/tmp/fastgs_recorded_reference/recorded_manifest.json first.")
        }

        let manifest = try JSONDecoder().decode(
            RecordedForwardManifest.self,
            from: Data(contentsOf: recordedManifestURL)
        )
        let output = recordedForwardOutput(manifest)
        let outColor = output.outColor.asArray(Float.self)

        assertClose(
            channelSums(outColor, channels: 3),
            manifest.predChannelSums.map(Float.init),
            accuracy: 1.0
        )
        assertClose(
            samples(outColor, ids: manifest.samplePixelIds, channels: 3),
            manifest.predSamples.map(Float.init),
            accuracy: 2e-2
        )

        try FastGSImageExport.writePNG(
            outColor: output.outColor,
            width: manifest.width,
            height: manifest.height,
            to: URL(fileURLWithPath: "/private/tmp/fastgs_recorded_reference/recorded_swift.png")
        )
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

private struct RecordedForwardManifest: Decodable {
    var width: Int
    var height: Int
    var pointCount: Int
    var shDegree: Int
    var scale: Double
    var opacity: Double
    var tanFovX: Double
    var tanFovY: Double
    var background: [Double]
    var viewmatrix: [Double]
    var projmatrix: [Double]
    var campos: [Double]
    var means3d: [Double]
    var colors: [Double]
    var predChannelSums: [Double]
    var samplePixelIds: [Int]
    var predSamples: [Double]
}

private func recordedForwardOutput(_ manifest: RecordedForwardManifest) -> FastGSRasterizeOutput {
    let count = manifest.pointCount
    let tileBounds = (
        x: (manifest.width + 15) / 16,
        y: (manifest.height + 15) / 16,
        z: 1
    )
    let maxSHCoefficients = (manifest.shDegree + 1) * (manifest.shDegree + 1)
    let means = MLXArray(manifest.means3d.map(Float.init), [count, 3])
    let colors = manifest.colors.map(Float.init)
    let shC0 = Float(0.28209479177387814)
    let dc = MLXArray(colors.map { ($0 - 0.5) / shC0 }, [count, 3])
    let sh = MLXArray.zeros([count, maxSHCoefficients - 1, 3], dtype: .float32)
    let opacities = MLXArray(Array(repeating: Float(manifest.opacity), count: count), [count])
    let scales = MLXArray(Array(repeating: Float(manifest.scale), count: count * 3), [count, 3])
    var rotations = [Float](repeating: 0, count: count * 4)
    for index in 0..<count {
        rotations[index * 4] = 1
    }
    let preprocess = FastGSPreprocess.forward(
        FastGSPreprocessInput(
            means3D: means,
            dc: dc,
            sh: sh,
            colorsPrecomputed: MLXArray.zeros([0, 3], dtype: .float32),
            opacities: opacities,
            scales: scales,
            rotations: MLXArray(rotations, [count, 4]),
            cov3DPrecomputed: MLXArray.zeros([0, 6], dtype: .float32),
            viewMatrix: MLXArray(manifest.viewmatrix.map(Float.init), [4, 4]),
            projectionMatrix: MLXArray(manifest.projmatrix.map(Float.init), [4, 4]),
            cameraPosition: MLXArray(Array(manifest.campos.prefix(3)).map(Float.init), [3]),
            viewspacePoints: MLXArray.zeros([count, 4], dtype: .float32)
        ),
        params: FastGSPreprocessParams(
            degree: manifest.shDegree,
            maxSHCoefficients: maxSHCoefficients,
            scaleModifier: 1,
            tanFovX: Float(manifest.tanFovX),
            tanFovY: Float(manifest.tanFovY),
            imageHeight: manifest.height,
            imageWidth: manifest.width,
            tileBounds: tileBounds,
            multiplier: 1
        )
    )
    let binning = FastGSBinning.forward(
        preprocessOutput: preprocess,
        params: FastGSBinningParams(multiplier: 1, tileBounds: tileBounds)
    )
    return FastGSRasterize.forward(
        preprocessOutput: preprocess,
        binningOutput: binning,
        background: MLXArray(manifest.background.map(Float.init), [3]),
        params: FastGSRasterizeParams(
            imageWidth: manifest.width,
            imageHeight: manifest.height,
            numTiles: tileBounds.x * tileBounds.y
        )
    )
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
