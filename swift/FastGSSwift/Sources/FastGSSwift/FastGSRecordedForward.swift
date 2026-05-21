import Foundation
import MLX

public struct FastGSRecordedForwardManifest: Decodable {
    public var width: Int
    public var height: Int
    public var pointCount: Int
    public var shDegree: Int
    public var scale: Double
    public var opacity: Double
    public var tanFovX: Double
    public var tanFovY: Double
    public var background: [Double]
    public var viewmatrix: [Double]
    public var projmatrix: [Double]
    public var campos: [Double]
    public var means3d: [Double]
    public var colors: [Double]
    public var predChannelSums: [Double]
    public var samplePixelIds: [Int]
    public var predSamples: [Double]

    public init(
        width: Int,
        height: Int,
        pointCount: Int,
        shDegree: Int,
        scale: Double,
        opacity: Double,
        tanFovX: Double,
        tanFovY: Double,
        background: [Double],
        viewmatrix: [Double],
        projmatrix: [Double],
        campos: [Double],
        means3d: [Double],
        colors: [Double],
        predChannelSums: [Double],
        samplePixelIds: [Int],
        predSamples: [Double]
    ) {
        self.width = width
        self.height = height
        self.pointCount = pointCount
        self.shDegree = shDegree
        self.scale = scale
        self.opacity = opacity
        self.tanFovX = tanFovX
        self.tanFovY = tanFovY
        self.background = background
        self.viewmatrix = viewmatrix
        self.projmatrix = projmatrix
        self.campos = campos
        self.means3d = means3d
        self.colors = colors
        self.predChannelSums = predChannelSums
        self.samplePixelIds = samplePixelIds
        self.predSamples = predSamples
    }
}

public enum FastGSRecordedForwardError: Error {
    case invalidPointCount(expected: Int, meansCount: Int, colorsCount: Int)
    case invalidCameraData(viewmatrixCount: Int, projmatrixCount: Int, camposCount: Int)
    case invalidBackground(count: Int)
}

public struct FastGSRecordedForwardScene {
    public var manifest: FastGSRecordedForwardManifest

    public init(manifest: FastGSRecordedForwardManifest) {
        self.manifest = manifest
    }

    public init(manifestURL: URL) throws {
        self.manifest = try JSONDecoder().decode(
            FastGSRecordedForwardManifest.self,
            from: Data(contentsOf: manifestURL)
        )
    }

    public func render(verbose: Bool = false) throws -> FastGSRasterizeOutput {
        try validate()

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
            ),
            verbose: verbose
        )
        let binning = FastGSBinning.forward(
            preprocessOutput: preprocess,
            params: FastGSBinningParams(multiplier: 1, tileBounds: tileBounds),
            verbose: verbose
        )
        return FastGSRasterize.forward(
            preprocessOutput: preprocess,
            binningOutput: binning,
            background: MLXArray(manifest.background.map(Float.init), [3]),
            params: FastGSRasterizeParams(
                imageWidth: manifest.width,
                imageHeight: manifest.height,
                numTiles: tileBounds.x * tileBounds.y
            ),
            verbose: verbose
        )
    }

    private func validate() throws {
        let count = manifest.pointCount
        guard manifest.means3d.count == count * 3, manifest.colors.count == count * 3 else {
            throw FastGSRecordedForwardError.invalidPointCount(
                expected: count,
                meansCount: manifest.means3d.count,
                colorsCount: manifest.colors.count
            )
        }
        guard manifest.viewmatrix.count == 16, manifest.projmatrix.count == 16, manifest.campos.count >= 3 else {
            throw FastGSRecordedForwardError.invalidCameraData(
                viewmatrixCount: manifest.viewmatrix.count,
                projmatrixCount: manifest.projmatrix.count,
                camposCount: manifest.campos.count
            )
        }
        guard manifest.background.count == 3 else {
            throw FastGSRecordedForwardError.invalidBackground(count: manifest.background.count)
        }
    }
}
