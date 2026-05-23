import Foundation
import MLX

public struct FastGSRecordedForwardManifest: Decodable {
    public struct Float32Buffer: Decodable {
        public var path: String
        public var dtype: String
        public var shape: [Int]
    }

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
    public var means3dBuffer: Float32Buffer?
    public var colorsBuffer: Float32Buffer?
    public var targetBuffer: Float32Buffer?
    public var predChannelSums: [Double]
    public var samplePixelIds: [Int]
    public var predSamples: [Double]
    public var targetPng: String?

    private enum CodingKeys: String, CodingKey {
        case width
        case height
        case pointCount
        case shDegree
        case scale
        case opacity
        case tanFovX
        case tanFovY
        case background
        case viewmatrix
        case projmatrix
        case campos
        case means3d
        case colors
        case means3dBuffer
        case colorsBuffer
        case targetBuffer
        case predChannelSums
        case samplePixelIds
        case predSamples
        case targetPng
    }

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
        means3dBuffer: Float32Buffer? = nil,
        colorsBuffer: Float32Buffer? = nil,
        targetBuffer: Float32Buffer? = nil,
        predChannelSums: [Double],
        samplePixelIds: [Int],
        predSamples: [Double],
        targetPng: String? = nil
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
        self.means3dBuffer = means3dBuffer
        self.colorsBuffer = colorsBuffer
        self.targetBuffer = targetBuffer
        self.predChannelSums = predChannelSums
        self.samplePixelIds = samplePixelIds
        self.predSamples = predSamples
        self.targetPng = targetPng
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        width = try container.decode(Int.self, forKey: .width)
        height = try container.decode(Int.self, forKey: .height)
        pointCount = try container.decode(Int.self, forKey: .pointCount)
        shDegree = try container.decode(Int.self, forKey: .shDegree)
        scale = try container.decode(Double.self, forKey: .scale)
        opacity = try container.decode(Double.self, forKey: .opacity)
        tanFovX = try container.decode(Double.self, forKey: .tanFovX)
        tanFovY = try container.decode(Double.self, forKey: .tanFovY)
        background = try container.decode([Double].self, forKey: .background)
        viewmatrix = try container.decode([Double].self, forKey: .viewmatrix)
        projmatrix = try container.decode([Double].self, forKey: .projmatrix)
        campos = try container.decode([Double].self, forKey: .campos)
        means3d = try container.decodeIfPresent([Double].self, forKey: .means3d) ?? []
        colors = try container.decodeIfPresent([Double].self, forKey: .colors) ?? []
        means3dBuffer = try container.decodeIfPresent(Float32Buffer.self, forKey: .means3dBuffer)
        colorsBuffer = try container.decodeIfPresent(Float32Buffer.self, forKey: .colorsBuffer)
        targetBuffer = try container.decodeIfPresent(Float32Buffer.self, forKey: .targetBuffer)
        predChannelSums = try container.decode([Double].self, forKey: .predChannelSums)
        samplePixelIds = try container.decode([Int].self, forKey: .samplePixelIds)
        predSamples = try container.decode([Double].self, forKey: .predSamples)
        targetPng = try container.decodeIfPresent(String.self, forKey: .targetPng)
    }
}

public struct FastGSRecordedForwardStages {
    public var preprocess: FastGSPreprocessOutput
    public var binning: FastGSBinningOutput
    public var rasterize: FastGSRasterizeOutput
}

public struct FastGSRecordedForwardTimingReport: Sendable {
    public var preprocessMilliseconds: Double
    public var binning: FastGSBinningTimingReport
    public var rasterizeMilliseconds: Double
    public var imageReadbackMilliseconds: Double?

    public var totalWithoutImageReadbackMilliseconds: Double {
        preprocessMilliseconds + binning.totalMilliseconds + rasterizeMilliseconds
    }

    public var totalWithImageReadbackMilliseconds: Double {
        totalWithoutImageReadbackMilliseconds + (imageReadbackMilliseconds ?? 0)
    }
}

public struct FastGSTimedRecordedForwardStages {
    public var stages: FastGSRecordedForwardStages
    public var timing: FastGSRecordedForwardTimingReport
}

public enum FastGSRecordedForwardError: Error {
    case invalidPointCount(expected: Int, meansCount: Int, colorsCount: Int)
    case invalidCameraData(viewmatrixCount: Int, projmatrixCount: Int, camposCount: Int)
    case invalidBackground(count: Int)
    case invalidBuffer(name: String, dtype: String, shape: [Int], expectedShape: [Int])
    case invalidBufferByteCount(name: String, expected: Int, actual: Int)
}

public struct FastGSRecordedForwardScene {
    public var manifest: FastGSRecordedForwardManifest
    public var manifestDirectory: URL?
    public var directMeans3D: [Float]?
    public var directColors: [Float]?
    public var directTarget: [Float]?

    public init(manifest: FastGSRecordedForwardManifest) {
        self.manifest = manifest
        self.manifestDirectory = nil
        self.directMeans3D = nil
        self.directColors = nil
        self.directTarget = nil
    }

    public init(manifestURL: URL) throws {
        self.manifest = try JSONDecoder().decode(
            FastGSRecordedForwardManifest.self,
            from: Data(contentsOf: manifestURL)
        )
        self.manifestDirectory = manifestURL.deletingLastPathComponent()
        self.directMeans3D = nil
        self.directColors = nil
        self.directTarget = nil
    }

    public init(scannerDataset: FastGSScannerDataset, frameIndex: Int, shDegree: Int = 3, scale: Double = 0.02, opacity: Double = 0.82) {
        let frame = scannerDataset.frames[min(max(frameIndex, 0), scannerDataset.frames.count - 1)]
        self.manifest = FastGSRecordedForwardManifest(
            width: frame.camera.imageWidth,
            height: frame.camera.imageHeight,
            pointCount: scannerDataset.pointCloud.count,
            shDegree: shDegree,
            scale: scale,
            opacity: opacity,
            tanFovX: Double(frame.camera.tanFovX),
            tanFovY: Double(frame.camera.tanFovY),
            background: [0, 0, 0],
            viewmatrix: frame.camera.viewmatrix.map(Double.init),
            projmatrix: frame.camera.projmatrix.map(Double.init),
            campos: frame.camera.campos.map(Double.init),
            means3d: [],
            colors: [],
            predChannelSums: [],
            samplePixelIds: [],
            predSamples: []
        )
        self.manifestDirectory = nil
        self.directMeans3D = scannerDataset.pointCloud.points
        self.directColors = scannerDataset.pointCloud.colors
        self.directTarget = frame.targetCHW
    }

    public func render(verbose: Bool = false) throws -> FastGSRasterizeOutput {
        return try renderStages(verbose: verbose).rasterize
    }

    public func render(
        parameters: FastGSTrainableParameters,
        verbose: Bool = false
    ) throws -> FastGSRasterizeOutput {
        return try renderStages(parameters: parameters, verbose: verbose).rasterize
    }

    public func renderStages(verbose: Bool = false) throws -> FastGSRecordedForwardStages {
        return try renderStages(parameters: initialTrainableParameters(), verbose: verbose)
    }

    public func renderStages(
        parameters: FastGSTrainableParameters,
        verbose: Bool = false
    ) throws -> FastGSRecordedForwardStages {
        try validate()

        let tileBounds = tileBounds()
        let maxSHCoefficients = maxSHCoefficients()
        let preprocess = FastGSPreprocess.forward(
            try preprocessInput(parameters: parameters),
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
        let rasterize = FastGSRasterize.forward(
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
        return FastGSRecordedForwardStages(preprocess: preprocess, binning: binning, rasterize: rasterize)
    }

    public func timedRenderStages(
        parameters: FastGSTrainableParameters,
        includeImageReadback: Bool = false,
        verbose: Bool = false
    ) throws -> FastGSTimedRecordedForwardStages {
        let tileBounds = tileBounds()
        let maxSHCoefficients = maxSHCoefficients()

        var started = CFAbsoluteTimeGetCurrent()
        let preprocess = FastGSPreprocess.forward(
            try preprocessInput(parameters: parameters),
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
        [
            preprocess.radii,
            preprocess.xy,
            preprocess.depths,
            preprocess.cov3D,
            preprocess.rgb,
            preprocess.conicOpacity,
            preprocess.tilesTouched,
            preprocess.clamped,
            preprocess.viewspacePoints,
        ].forEach { $0.eval() }
        let preprocessMilliseconds = elapsedMilliseconds(since: started)

        let timedBinning = FastGSBinning.timedForward(
            preprocessOutput: preprocess,
            params: FastGSBinningParams(multiplier: 1, tileBounds: tileBounds),
            verbose: verbose
        )

        started = CFAbsoluteTimeGetCurrent()
        let rasterize = FastGSRasterize.forward(
            preprocessOutput: preprocess,
            binningOutput: timedBinning.output,
            background: MLXArray(manifest.background.map(Float.init), [3]),
            params: FastGSRasterizeParams(
                imageWidth: manifest.width,
                imageHeight: manifest.height,
                numTiles: tileBounds.x * tileBounds.y
            ),
            verbose: verbose
        )
        [
            rasterize.bucketToTile,
            rasterize.sampledT,
            rasterize.sampledAr,
            rasterize.finalT,
            rasterize.nContrib,
            rasterize.maxContrib,
            rasterize.pixelColors,
            rasterize.outColor,
            rasterize.metricCount,
        ].forEach { $0.eval() }
        let rasterizeMilliseconds = elapsedMilliseconds(since: started)

        let imageReadbackMilliseconds: Double?
        if includeImageReadback {
            started = CFAbsoluteTimeGetCurrent()
            _ = FastGSImageExport.rgbaBytes(
                outColor: rasterize.outColor,
                width: manifest.width,
                height: manifest.height
            )
            imageReadbackMilliseconds = elapsedMilliseconds(since: started)
        } else {
            imageReadbackMilliseconds = nil
        }

        return FastGSTimedRecordedForwardStages(
            stages: FastGSRecordedForwardStages(
                preprocess: preprocess,
                binning: timedBinning.output,
                rasterize: rasterize
            ),
            timing: FastGSRecordedForwardTimingReport(
                preprocessMilliseconds: preprocessMilliseconds,
                binning: timedBinning.timing,
                rasterizeMilliseconds: rasterizeMilliseconds,
                imageReadbackMilliseconds: imageReadbackMilliseconds
            )
        )
    }

    public func initialTrainableParameters() throws -> FastGSTrainableParameters {
        try validate()

        let count = manifest.pointCount
        let maxSHCoefficients = maxSHCoefficients()
        let means = MLXArray(try floatBuffer(name: "means3d", descriptor: manifest.means3dBuffer, fallback: manifest.means3d, expectedShape: [count, 3]), [count, 3])
        let colors = try floatBuffer(name: "colors", descriptor: manifest.colorsBuffer, fallback: manifest.colors, expectedShape: [count, 3])
        let shC0 = Float(0.28209479177387814)
        let dc = MLXArray(colors.map { ($0 - 0.5) / shC0 }, [count, 3])
        let sh = MLXArray.zeros([count, maxSHCoefficients, 3], dtype: .float32)
        let opacities = MLXArray(Array(repeating: Float(manifest.opacity), count: count), [count])
        let scales = MLXArray(Array(repeating: Float(manifest.scale), count: count * 3), [count, 3])
        var rotations = [Float](repeating: 0, count: count * 4)
        for index in 0..<count {
            rotations[index * 4] = 1
        }

        return FastGSTrainableParameters(
            means3D: means,
            dc: dc,
            sh: sh,
            opacities: opacities,
            scales: scales,
            rotations: MLXArray(rotations, [count, 4])
        )
    }

    public func targetOutColor() throws -> MLXArray {
        let shape = [3, manifest.width * manifest.height]
        return MLXArray(try floatBuffer(name: "target", descriptor: manifest.targetBuffer, fallback: [], expectedShape: shape), shape)
    }

    private func preprocessInput(parameters: FastGSTrainableParameters) throws -> FastGSPreprocessInput {
        try validate(parameters: parameters)

        let count = manifest.pointCount
        return FastGSPreprocessInput(
            means3D: parameters.means3D,
            dc: parameters.dc,
            sh: parameters.sh,
            colorsPrecomputed: MLXArray.zeros([0, 3], dtype: .float32),
            opacities: parameters.opacities,
            scales: parameters.scales,
            rotations: parameters.rotations,
            cov3DPrecomputed: MLXArray.zeros([0, 6], dtype: .float32),
            viewMatrix: MLXArray(manifest.viewmatrix.map(Float.init), [4, 4]),
            projectionMatrix: MLXArray(manifest.projmatrix.map(Float.init), [4, 4]),
            cameraPosition: MLXArray(Array(manifest.campos.prefix(3)).map(Float.init), [3]),
            viewspacePoints: MLXArray.zeros([count, 4], dtype: .float32)
        )
    }

    private func tileBounds() -> (x: Int, y: Int, z: Int) {
        (
            x: (manifest.width + 15) / 16,
            y: (manifest.height + 15) / 16,
            z: 1
        )
    }

    private func maxSHCoefficients() -> Int {
        (manifest.shDegree + 1) * (manifest.shDegree + 1)
    }

    private func elapsedMilliseconds(since start: CFAbsoluteTime) -> Double {
        (CFAbsoluteTimeGetCurrent() - start) * 1000
    }

    private func validate() throws {
        let count = manifest.pointCount
        let meansCount = directMeans3D?.count ?? (manifest.means3dBuffer == nil ? manifest.means3d.count : manifest.means3dBuffer?.shape.reduce(1, *) ?? 0)
        let colorsCount = directColors?.count ?? (manifest.colorsBuffer == nil ? manifest.colors.count : manifest.colorsBuffer?.shape.reduce(1, *) ?? 0)
        guard meansCount == count * 3, colorsCount == count * 3 else {
            throw FastGSRecordedForwardError.invalidPointCount(
                expected: count,
                meansCount: meansCount,
                colorsCount: colorsCount
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

    private func validate(parameters: FastGSTrainableParameters) throws {
        let count = manifest.pointCount
        let maxSHCoefficients = maxSHCoefficients()
        precondition(parameters.means3D.shape == [count, 3], "means3D must have shape [N, 3].")
        precondition(parameters.dc.shape == [count, 3], "dc must have shape [N, 3].")
        precondition(parameters.sh.shape == [count, maxSHCoefficients, 3], "sh must have shape [N, C, 3].")
        precondition(parameters.opacities.shape == [count], "opacities must have shape [N].")
        precondition(parameters.scales.shape == [count, 3], "scales must have shape [N, 3].")
        precondition(parameters.rotations.shape == [count, 4], "rotations must have shape [N, 4].")
        precondition(parameters.cov3DPrecomputed == nil, "recorded training smoke path currently uses scale/rotation covariance.")
    }

    private func floatBuffer(
        name: String,
        descriptor: FastGSRecordedForwardManifest.Float32Buffer?,
        fallback: [Double],
        expectedShape: [Int]
    ) throws -> [Float] {
        if let direct = directFloatBuffer(name: name) {
            let expectedCount = expectedShape.reduce(1, *)
            guard direct.count == expectedCount else {
                throw FastGSRecordedForwardError.invalidBufferByteCount(
                    name: name,
                    expected: expectedCount * MemoryLayout<Float>.stride,
                    actual: direct.count * MemoryLayout<Float>.stride
                )
            }
            return direct
        }
        guard let descriptor else {
            return fallback.map(Float.init)
        }
        guard descriptor.dtype == "float32", descriptor.shape == expectedShape else {
            throw FastGSRecordedForwardError.invalidBuffer(
                name: name,
                dtype: descriptor.dtype,
                shape: descriptor.shape,
                expectedShape: expectedShape
            )
        }
        let url = descriptor.path.hasPrefix("/")
            ? URL(fileURLWithPath: descriptor.path)
            : (manifestDirectory ?? URL(fileURLWithPath: ".")).appendingPathComponent(descriptor.path)
        let data = try Data(contentsOf: url)
        let expectedCount = expectedShape.reduce(1, *)
        let expectedByteCount = expectedCount * MemoryLayout<Float>.stride
        guard data.count == expectedByteCount else {
            throw FastGSRecordedForwardError.invalidBufferByteCount(name: name, expected: expectedByteCount, actual: data.count)
        }
        return data.withUnsafeBytes { rawBuffer in
            Array(rawBuffer.bindMemory(to: Float.self))
        }
    }

    private func directFloatBuffer(name: String) -> [Float]? {
        switch name {
        case "means3d":
            directMeans3D
        case "colors":
            directColors
        case "target":
            directTarget
        default:
            nil
        }
    }
}
