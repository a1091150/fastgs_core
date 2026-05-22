import Foundation
import MLX

public struct FastGSTrainingRenderContext {
    public var pointCount: Int
    public var width: Int
    public var height: Int
    public var tileBounds: (x: Int, y: Int, z: Int)
    public var preprocessParams: FastGSPreprocessParams
    public var rasterizeParams: FastGSRasterizeParams
    public var viewMatrix: MLXArray
    public var projectionMatrix: MLXArray
    public var cameraPosition: MLXArray
    public var background: MLXArray
    public var emptyColorsPrecomputed: MLXArray
    public var emptyCov3DPrecomputed: MLXArray
    public var viewspacePoints: MLXArray
    public var metricMap: MLXArray
    public var metricCount: MLXArray

    public init(scene: FastGSRecordedForwardScene, stream: StreamOrDevice = .default) {
        let width = scene.manifest.width
        let height = scene.manifest.height
        let pointCount = scene.manifest.pointCount
        let tileBounds = (
            x: (width + 15) / 16,
            y: (height + 15) / 16,
            z: 1
        )
        let maxSHCoefficients = (scene.manifest.shDegree + 1) * (scene.manifest.shDegree + 1)

        self.pointCount = pointCount
        self.width = width
        self.height = height
        self.tileBounds = tileBounds
        self.preprocessParams = FastGSPreprocessParams(
            degree: scene.manifest.shDegree,
            maxSHCoefficients: maxSHCoefficients,
            scaleModifier: 1,
            tanFovX: Float(scene.manifest.tanFovX),
            tanFovY: Float(scene.manifest.tanFovY),
            imageHeight: height,
            imageWidth: width,
            tileBounds: tileBounds,
            multiplier: 1
        )
        self.rasterizeParams = FastGSRasterizeParams(
            imageWidth: width,
            imageHeight: height,
            numTiles: tileBounds.x * tileBounds.y
        )
        self.viewMatrix = MLXArray(scene.manifest.viewmatrix.map(Float.init), [4, 4])
        self.projectionMatrix = MLXArray(scene.manifest.projmatrix.map(Float.init), [4, 4])
        self.cameraPosition = MLXArray(Array(scene.manifest.campos.prefix(3)).map(Float.init), [3])
        self.background = MLXArray(scene.manifest.background.map(Float.init), [3])
        self.emptyColorsPrecomputed = MLXArray.zeros([0, 3], dtype: .float32, stream: stream)
        self.emptyCov3DPrecomputed = MLXArray.zeros([0, 6], dtype: .float32, stream: stream)
        self.viewspacePoints = MLXArray.zeros([pointCount, 4], dtype: .float32, stream: stream)
        self.metricMap = MLXArray.zeros([width * height], dtype: .int32, stream: stream)
        self.metricCount = MLXArray.zeros([pointCount], dtype: .int32, stream: stream)
    }

    public func preprocessInput(parameters: FastGSTrainableParameters) -> FastGSPreprocessInput {
        FastGSPreprocessInput(
            means3D: parameters.means3D,
            dc: parameters.dc,
            sh: parameters.sh,
            colorsPrecomputed: emptyColorsPrecomputed,
            opacities: parameters.opacities,
            scales: parameters.scales,
            rotations: parameters.rotations,
            cov3DPrecomputed: emptyCov3DPrecomputed,
            viewMatrix: viewMatrix,
            projectionMatrix: projectionMatrix,
            cameraPosition: cameraPosition,
            viewspacePoints: viewspacePoints
        )
    }

    public func rasterizeInput(
        preprocess: FastGSPreprocessOutput,
        binning: FastGSBinningOutput,
        stream: StreamOrDevice = .default
    ) -> FastGSRasterizeInput {
        FastGSRasterizeInput(
            ranges: stopGradient(binning.ranges, stream: stream),
            pointList: stopGradient(binning.pointList, stream: stream),
            bucketOffsets: stopGradient(binning.bucketOffsets, stream: stream),
            means2D: preprocess.xy,
            colors: preprocess.rgb,
            conicOpacity: preprocess.conicOpacity,
            background: background,
            radii: stopGradient(preprocess.radii, stream: stream),
            metricMap: metricMap,
            metricCount: metricCount
        )
    }
}
