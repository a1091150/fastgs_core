import MLX

public enum FastGSPreprocessParityFixture {
    public static let precomputedColorParams = FastGSPreprocessParams(
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

    public static let shDegree3Params = FastGSPreprocessParams(
        degree: 3,
        maxSHCoefficients: 15,
        scaleModifier: 1,
        tanFovX: 1,
        tanFovY: 1,
        imageHeight: 64,
        imageWidth: 64,
        tileBounds: (x: 4, y: 4, z: 1),
        multiplier: 1,
        useColorsPrecomputed: false
    )

    public static let cov3DPrecomputedParams = FastGSPreprocessParams(
        degree: 0,
        maxSHCoefficients: 0,
        scaleModifier: 1,
        tanFovX: 1,
        tanFovY: 1,
        imageHeight: 64,
        imageWidth: 64,
        tileBounds: (x: 4, y: 4, z: 1),
        multiplier: 1,
        useCov3DPrecomputed: true,
        useColorsPrecomputed: true
    )

    public static let shClampParams = FastGSPreprocessParams(
        degree: 0,
        maxSHCoefficients: 0,
        scaleModifier: 1,
        tanFovX: 1,
        tanFovY: 1,
        imageHeight: 64,
        imageWidth: 64,
        tileBounds: (x: 4, y: 4, z: 1),
        multiplier: 1,
        useColorsPrecomputed: false
    )

    public static let binningParams = FastGSBinningParams(
        multiplier: 1,
        tileBounds: (x: 4, y: 4, z: 1)
    )

    public static let rasterizeSmokeParams = FastGSRasterizeParams(
        imageWidth: 1,
        imageHeight: 1,
        numTiles: 1
    )

    public static let rasterizeE2EParams = FastGSRasterizeParams(
        imageWidth: 64,
        imageHeight: 64,
        numTiles: 16
    )

    public static func precomputedColorInput() -> FastGSPreprocessInput {
        FastGSPreprocessInput(
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
    }

    public static func shDegree3Input() -> FastGSPreprocessInput {
        FastGSPreprocessInput(
            means3D: MLXArray([Float(0), 0, 1, 0.25, -0.25, 1], [2, 3]),
            dc: MLXArray([Float(0.2), -0.1, 0.05, -0.05, 0.15, 0.1], [2, 3]),
            sh: MLXArray([
                Float(0.05), -0.02, 0.03,
                -0.01, 0.04, -0.02,
                0.02, 0.01, 0.05,
                0.03, -0.02, 0.01,
                -0.04, 0.03, 0.02,
                0.02, -0.01, 0.04,
                -0.03, 0.02, -0.01,
                0.01, 0.03, -0.04,
                0.02, 0.02, 0.01,
                -0.01, 0.01, 0.03,
                0.04, -0.03, 0.02,
                -0.02, 0.04, -0.01,
                0.03, 0.01, -0.02,
                -0.04, -0.02, 0.03,
                0.01, -0.03, 0.04,
                -0.03, 0.02, 0.01,
                0.04, -0.01, 0.02,
                -0.02, 0.03, -0.04,
                0.01, 0.02, 0.03,
                0.03, -0.04, 0.01,
                -0.01, 0.05, -0.02,
                0.02, -0.03, 0.04,
                -0.04, 0.01, 0.02,
                0.03, 0.02, -0.01,
                0.01, -0.02, 0.04,
                -0.03, 0.04, 0.02,
                0.02, 0.01, -0.03,
                -0.01, 0.03, 0.05,
                0.04, -0.02, 0.01,
                -0.02, 0.02, -0.04,
            ], [2, 15, 3]),
            colorsPrecomputed: MLXArray([Float](), [0]),
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
    }

    public static func cullingInput() -> FastGSPreprocessInput {
        var input = precomputedColorInput()
        input.means3D = MLXArray([Float(0), 0, 0.1, 0.25, -0.25, 1], [2, 3])
        return input
    }

    public static func allCulledInput() -> FastGSPreprocessInput {
        var input = precomputedColorInput()
        input.means3D = MLXArray([Float(0), 0, 0.1, 0.25, -0.25, 0.1], [2, 3])
        return input
    }

    public static func cov3DPrecomputedInput() -> FastGSPreprocessInput {
        FastGSPreprocessInput(
            means3D: MLXArray([Float(0), 0, 1, 0.25, -0.25, 1], [2, 3]),
            dc: MLXArray([Float](), [0]),
            sh: MLXArray([Float](), [0]),
            colorsPrecomputed: MLXArray([Float(1), 0, 0, 0, 1, 0], [2, 3]),
            opacities: MLXArray([Float(1), 1], [2]),
            scales: MLXArray([Float](), [0]),
            rotations: MLXArray([Float](), [0]),
            cov3DPrecomputed: MLXArray([
                Float(0.5), 0.1, 0, 0.75, 0.05, 1.25,
                1.5, -0.2, 0.1, 1.1, 0, 0.8,
            ], [2, 6]),
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
    }

    public static func shClampInput() -> FastGSPreprocessInput {
        FastGSPreprocessInput(
            means3D: MLXArray([Float(0), 0, 1, 0.25, -0.25, 1], [2, 3]),
            dc: MLXArray([Float(-3), -2, -1, 0.2, 0.1, -4], [2, 3]),
            sh: MLXArray([Float](), [2, 0, 3]),
            colorsPrecomputed: MLXArray([Float](), [0]),
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
    }

    public static func precomputedColorOutput() -> FastGSPreprocessOutput {
        FastGSPreprocess.forward(precomputedColorInput(), params: precomputedColorParams)
    }

    public static func shDegree3Output() -> FastGSPreprocessOutput {
        FastGSPreprocess.forward(shDegree3Input(), params: shDegree3Params)
    }

    public static func cullingOutput() -> FastGSPreprocessOutput {
        FastGSPreprocess.forward(cullingInput(), params: precomputedColorParams)
    }

    public static func allCulledOutput() -> FastGSPreprocessOutput {
        FastGSPreprocess.forward(allCulledInput(), params: precomputedColorParams)
    }

    public static func cov3DPrecomputedOutput() -> FastGSPreprocessOutput {
        FastGSPreprocess.forward(cov3DPrecomputedInput(), params: cov3DPrecomputedParams)
    }

    public static func shClampOutput() -> FastGSPreprocessOutput {
        FastGSPreprocess.forward(shClampInput(), params: shClampParams)
    }

    public static func binningOutput() -> FastGSBinningOutput {
        FastGSBinning.forward(preprocessOutput: precomputedColorOutput(), params: binningParams)
    }

    public static func cullingBinningOutput() -> FastGSBinningOutput {
        FastGSBinning.forward(preprocessOutput: cullingOutput(), params: binningParams)
    }

    public static func allCulledBinningOutput() -> FastGSBinningOutput {
        FastGSBinning.forward(preprocessOutput: allCulledOutput(), params: binningParams)
    }

    public static func variedDepthBinningOutput() -> FastGSBinningOutput {
        FastGSBinning.forward(
            FastGSBinningInput(
                xy: MLXArray([Float(31.5), 31.5, 31.5, 31.5], [2, 2]),
                depths: MLXArray([Float(1.1), 0.9], [2]),
                conicOpacity: MLXArray([
                    Float(0.0009762764093466103), 0, 0.0009762764093466103, 1,
                    0.0009762764093466103, 0, 0.0009762764093466103, 1,
                ], [2, 4]),
                tilesTouched: MLXArray([UInt32(16), 16], [2])
            ),
            params: binningParams
        )
    }

    public static func rasterizeSmokeOutput() -> FastGSRasterizeOutput {
        FastGSRasterize.forward(
            FastGSRasterizeInput(
                ranges: MLXArray([UInt32(0), 1], [1, 2]),
                pointList: MLXArray([UInt32(0)], [1]),
                bucketOffsets: MLXArray([UInt32(1)], [1]),
                means2D: MLXArray([Float(0), 0], [1, 2]),
                colors: MLXArray([Float(0.25), 0.5, 0.75], [1, 3]),
                conicOpacity: MLXArray([Float(1), 0, 1, 0.5], [1, 4]),
                background: MLXArray([Float(0.1), 0.2, 0.3], [3]),
                radii: MLXArray([Int32(1)], [1]),
                metricMap: MLXArray([Int32(0)], [1]),
                metricCount: MLXArray([Int32(0)], [1])
            ),
            params: rasterizeSmokeParams
        )
    }

    public static func rasterizeE2EOutput() -> FastGSRasterizeOutput {
        FastGSRasterize.forward(
            preprocessOutput: precomputedColorOutput(),
            binningOutput: binningOutput(),
            background: MLXArray([Float(0.1), 0.2, 0.3], [3]),
            params: rasterizeE2EParams
        )
    }

    public static let expectedRadii: [Int32] = [97, 102]
    public static let expectedXY: [Float] = [
        31.5, 31.5,
        39.5, 23.5,
    ]
    public static let expectedDepths: [Float] = [1, 1]
    public static let expectedCov3D: [Float] = [
        1, 0, 0, 1, 0, 1,
        1, 0, 0, 1, 0, 1,
    ]
    public static let expectedRGB: [Float] = [
        1, 0, 0,
        0, 1, 0,
    ]
    public static let expectedSHDegree3RGB: [Float] = [
        0.5492215752601624, 0.514880895614624, 0.5221005082130432,
        0.5011177659034729, 0.5716356635093689, 0.49035653471946716,
    ]
    public static let expectedConicOpacity: [Float] = [
        0.0009762764093466103, -0, 0.0009762764093466103, 1,
        0.0009220529464073479, 0.000054223455663304776, 0.0009220529464073479, 1,
    ]
    public static let expectedTilesTouched: [UInt32] = [16, 16]
    public static let expectedClamped: [Bool] = [
        false, false, false,
        false, false, false,
    ]
    public static let expectedViewspacePoints: [Float] = [
        0, 0, 1, 7,
        0.25, -0.25, 1, 9,
    ]
    public static let expectedCullingRadii: [Int32] = [0, 102]
    public static let expectedCullingXY: [Float] = [
        0, 0,
        39.5, 23.5,
    ]
    public static let expectedCullingDepths: [Float] = [0, 1]
    public static let expectedCullingCov3D: [Float] = [
        0, 0, 0, 0, 0, 0,
        1, 0, 0, 1, 0, 1,
    ]
    public static let expectedCullingRGB: [Float] = [
        0, 0, 0,
        0, 1, 0,
    ]
    public static let expectedCullingConicOpacity: [Float] = [
        0, 0, 0, 0,
        0.0009220529464073479, 0.000054223455663304776, 0.0009220529464073479, 1,
    ]
    public static let expectedCullingTilesTouched: [UInt32] = [0, 16]
    public static let expectedCullingViewspacePoints: [Float] = [
        0, 0, 0, 0,
        0.25, -0.25, 1, 9,
    ]

    public static let expectedCov3DPrecomputedRadii: [Int32] = [86, 122]
    public static let expectedCov3DPrecomputedCov3D: [Float] = [
        0.5, 0.1, 0, 0.75, 0.05, 1.25,
        1.5, -0.2, 0.1, 1.1, 0, 0.8,
    ]
    public static let expectedCov3DPrecomputedConicOpacity: [Float] = [
        0.002005406655371189, -0.0002672831469681114, 0.0013371987733989954, 1,
        0.0006705858977511525, 0.00013116816990077496, 0.0008746252860873938, 1,
    ]

    public static let expectedSHClampRGB: [Float] = [
        0, 0, 0.217905193567276,
        0.5564189553260803, 0.5282095074653625, 0,
    ]
    public static let expectedSHClampClamped: [Bool] = [
        true, true, false,
        false, false, true,
    ]

    public static let expectedBinningPointOffsets: [UInt32] = [16, 32]
    private static let expectedBinningDepthBits = UInt64(Float(1).bitPattern)
    private static let expectedBinningUnsortedTileOrder = [
        0, 4, 8, 12,
        1, 5, 9, 13,
        2, 6, 10, 14,
        3, 7, 11, 15,
    ]
    public static let expectedBinningPointListKeysUnsorted: [UInt64] = (0..<2).flatMap { _ in
        expectedBinningUnsortedTileOrder.map { tile in
            (UInt64(tile) << 32) | expectedBinningDepthBits
        }
    }
    public static let expectedBinningPointListUnsorted: [UInt32] =
        Array(repeating: UInt32(0), count: 16) + Array(repeating: UInt32(1), count: 16)
    public static let expectedBinningPointListKeys: [UInt64] = (0..<16).flatMap { tile in
        Array(repeating: (UInt64(tile) << 32) | expectedBinningDepthBits, count: 2)
    }
    public static let expectedBinningRanges: [UInt32] = (0..<16).flatMap { tile in
        [UInt32(tile * 2), UInt32(tile * 2 + 2)]
    }
    public static let expectedBinningBucketCount: [UInt32] = Array(repeating: 1, count: 16)
    public static let expectedBinningBucketOffsets: [UInt32] = (1...16).map(UInt32.init)

    public static let expectedCullingBinningPointOffsets: [UInt32] = [0, 16]
    public static let expectedCullingBinningPointListKeysUnsorted: [UInt64] =
        expectedBinningUnsortedTileOrder.map { tile in
            (UInt64(tile) << 32) | expectedBinningDepthBits
        }
    public static let expectedCullingBinningPointListUnsorted: [UInt32] = Array(repeating: 1, count: 16)
    public static let expectedCullingBinningPointListKeys: [UInt64] = (0..<16).map { tile in
        (UInt64(tile) << 32) | expectedBinningDepthBits
    }
    public static let expectedCullingBinningRanges: [UInt32] = (0..<16).flatMap { tile in
        [UInt32(tile), UInt32(tile + 1)]
    }

    public static let expectedAllCulledBinningPointOffsets: [UInt32] = [0, 0]
    public static let expectedAllCulledBinningRanges: [UInt32] = Array(repeating: 0, count: 32)
    public static let expectedAllCulledBinningBucketCount: [UInt32] = Array(repeating: 0, count: 16)
    public static let expectedAllCulledBinningBucketOffsets: [UInt32] = Array(repeating: 0, count: 16)

    private static let expectedVariedDepthFirstBits = UInt64(Float(1.1).bitPattern)
    private static let expectedVariedDepthSecondBits = UInt64(Float(0.9).bitPattern)
    public static let expectedVariedDepthBinningPointOffsets: [UInt32] = [16, 32]
    public static let expectedVariedDepthBinningPointListKeysUnsorted: [UInt64] =
        expectedBinningUnsortedTileOrder.map { tile in
            (UInt64(tile) << 32) | expectedVariedDepthFirstBits
        } + expectedBinningUnsortedTileOrder.map { tile in
            (UInt64(tile) << 32) | expectedVariedDepthSecondBits
        }
    public static let expectedVariedDepthBinningPointListUnsorted: [UInt32] =
        Array(repeating: UInt32(0), count: 16) + Array(repeating: UInt32(1), count: 16)
    public static let expectedVariedDepthBinningPointListKeys: [UInt64] = (0..<16).flatMap { tile in
        [
            (UInt64(tile) << 32) | expectedVariedDepthSecondBits,
            (UInt64(tile) << 32) | expectedVariedDepthFirstBits,
        ]
    }
    public static let expectedVariedDepthBinningPointList: [UInt32] = (0..<16).flatMap { _ in
        [UInt32(1), UInt32(0)]
    }

    public static let expectedRasterizeSmokeBucketToTile: [UInt32] = [0]
    public static let expectedRasterizeSmokeSampledT: [Float] = [1] + Array(repeating: 0, count: 255)
    public static let expectedRasterizeSmokeSampledAr: [Float] = Array(repeating: 0, count: 3 * 256)
    public static let expectedRasterizeSmokeFinalT: [Float] = [0.5]
    public static let expectedRasterizeSmokeNContrib: [UInt32] = [1]
    public static let expectedRasterizeSmokeMaxContrib: [UInt32] = [1]
    public static let expectedRasterizeSmokePixelColors: [Float] = [0.125, 0.25, 0.375]
    public static let expectedRasterizeSmokeOutColor: [Float] = [0.175, 0.35, 0.525]
    public static let expectedRasterizeSmokeMetricCount: [Int32] = [0]

    public static let expectedRasterizeE2EBucketToTilePrefix: [UInt32] = (0..<16).map(UInt32.init)
    public static let expectedRasterizeE2ESampledTPrefix: [Float] = Array(repeating: 1, count: 32)
    public static let expectedRasterizeE2ESampledArPrefix: [Float] = Array(repeating: 0, count: 12)
    public static let expectedRasterizeE2EOutColorSums: [Float] = [
        3037.8857421875,
        784.2421875,
        117.37377166748047,
    ]
    public static let expectedRasterizeE2EPixelColorSums: [Float] = [
        2998.760986328125,
        705.9930419921875,
        0,
    ]
    public static let expectedRasterizeE2EFinalTSum: Float = 391.24591064453125
    public static let expectedRasterizeE2ENContribSum: UInt32 = 8192
    public static let expectedRasterizeE2EMaxContrib: [UInt32] = Array(repeating: 2, count: 16)
    public static let expectedRasterizeE2ESampleIDs: [Int] = [
        0,
        31 + 31 * 64,
        32 + 32 * 64,
        39 + 23 * 64,
        63 + 63 * 64,
    ]
    public static let expectedRasterizeE2EOutColorSamples: [Float] = [
        0.419337660074234, 0.9900542497634888, 0.9900542497634888, 0.939261794090271, 0.419337660074234,
        0.302304744720459, 0.0095659289509058, 0.0095659289509058, 0.06031261011958122, 0.302304744720459,
        0.1192961260676384, 0.0001627731107873842, 0.0001627731107873842, 0.00018239683413412422, 0.1192961260676384,
    ]
    public static let expectedRasterizeE2EPixelColorSamples: [Float] = [
        0.3795722723007202, 0.9900000095367432, 0.9900000095367432, 0.9392009973526001, 0.3795722723007202,
        0.2227739840745926, 0.009457413107156754, 0.009457413107156754, 0.06019101291894913, 0.2227739840745926,
        0, 0, 0, 0, 0,
    ]
    public static let expectedRasterizeE2EFinalTSamples: [Float] = [
        0.397653728723526,
        0.0005425770068541169,
        0.0005425770068541169,
        0.0006079894374124706,
        0.397653728723526,
    ]
    public static let expectedRasterizeE2ENContribSamples: [UInt32] = [2, 2, 2, 2, 2]
}
