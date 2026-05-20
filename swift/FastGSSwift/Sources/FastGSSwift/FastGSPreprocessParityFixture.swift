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

    public static func precomputedColorOutput() -> FastGSPreprocessOutput {
        FastGSPreprocess.forward(precomputedColorInput(), params: precomputedColorParams)
    }

    public static func shDegree3Output() -> FastGSPreprocessOutput {
        FastGSPreprocess.forward(shDegree3Input(), params: shDegree3Params)
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
}
