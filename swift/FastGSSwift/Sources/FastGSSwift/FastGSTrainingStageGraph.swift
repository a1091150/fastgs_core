import Foundation
import MLX

public enum FastGSTrainingStageGraph {
    public static func valueAndGrad(
        scene: FastGSRecordedForwardScene,
        parameters: FastGSTrainableParameters,
        target: MLXArray,
        stream: StreamOrDevice = .default
    ) -> FastGSTrainingSmokeResult {
        let primals = parameters.arrays
        let lossFunction: ([MLXArray]) -> [MLXArray] = { arrays in
            let parameters = FastGSTrainableParameters(
                means3D: arrays[0],
                dc: arrays[1],
                sh: arrays[2],
                opacities: arrays[3],
                scales: arrays[4],
                rotations: arrays[5]
            )
            let outColor = render(scene: scene, parameters: parameters, stream: stream)
            return [mean(square(outColor - target), stream: stream)]
        }
        let valueAndGradient = MLX.valueAndGrad(
            lossFunction,
            argumentNumbers: Array(0..<primals.count)
        )
        let (values, gradients) = valueAndGradient(primals)
        return FastGSTrainingSmokeResult(loss: values[0], gradients: gradients)
    }

    public static func render(
        scene: FastGSRecordedForwardScene,
        parameters: FastGSTrainableParameters,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        let preprocessParams = preprocessParams(for: scene)
        let preprocessInput = FastGSPreprocessInput(
            means3D: parameters.means3D,
            dc: parameters.dc,
            sh: parameters.sh,
            colorsPrecomputed: MLXArray.zeros([0, 3], dtype: .float32, stream: stream),
            opacities: parameters.opacities,
            scales: parameters.scales,
            rotations: parameters.rotations,
            cov3DPrecomputed: MLXArray.zeros([0, 6], dtype: .float32, stream: stream),
            viewMatrix: MLXArray(scene.manifest.viewmatrix.map(Float.init), [4, 4]),
            projectionMatrix: MLXArray(scene.manifest.projmatrix.map(Float.init), [4, 4]),
            cameraPosition: MLXArray(Array(scene.manifest.campos.prefix(3)).map(Float.init), [3]),
            viewspacePoints: MLXArray.zeros([scene.manifest.pointCount, 4], dtype: .float32, stream: stream)
        )
        let preprocess = FastGSPreprocessCustomFunction.call(
            preprocessInput,
            params: preprocessParams,
            stream: stream
        )
        let stoppedPreprocess = FastGSPreprocessOutput(
            radii: stopGradient(preprocess.radii, stream: stream),
            xy: preprocess.xy,
            depths: stopGradient(preprocess.depths, stream: stream),
            cov3D: stopGradient(preprocess.cov3D, stream: stream),
            rgb: preprocess.rgb,
            conicOpacity: preprocess.conicOpacity,
            tilesTouched: stopGradient(preprocess.tilesTouched, stream: stream),
            clamped: stopGradient(preprocess.clamped, stream: stream),
            viewspacePoints: preprocess.viewspacePoints
        )
        let tileBounds = tileBounds(for: scene)
        let binning = FastGSBinning.forward(
            preprocessOutput: stoppedPreprocess,
            params: FastGSBinningParams(multiplier: 1, tileBounds: tileBounds),
            stream: stream
        )
        let rasterizeParams = FastGSRasterizeParams(
            imageWidth: scene.manifest.width,
            imageHeight: scene.manifest.height,
            numTiles: tileBounds.x * tileBounds.y
        )
        let numPixels = scene.manifest.width * scene.manifest.height
        let rasterizeInput = FastGSRasterizeInput(
            ranges: stopGradient(binning.ranges, stream: stream),
            pointList: stopGradient(binning.pointList, stream: stream),
            bucketOffsets: stopGradient(binning.bucketOffsets, stream: stream),
            means2D: stoppedPreprocess.xy,
            colors: stoppedPreprocess.rgb,
            conicOpacity: stoppedPreprocess.conicOpacity,
            background: MLXArray(scene.manifest.background.map(Float.init), [3]),
            radii: stopGradient(stoppedPreprocess.radii, stream: stream),
            metricMap: MLXArray.zeros([numPixels], dtype: .int32, stream: stream),
            metricCount: MLXArray.zeros([scene.manifest.pointCount], dtype: .int32, stream: stream)
        )
        return FastGSRasterizeCustomFunction.call(
            rasterizeInput,
            params: rasterizeParams,
            stream: stream
        ).outColor
    }

    private static func preprocessParams(for scene: FastGSRecordedForwardScene) -> FastGSPreprocessParams {
        let tileBounds = tileBounds(for: scene)
        let maxSHCoefficients = (scene.manifest.shDegree + 1) * (scene.manifest.shDegree + 1)
        return FastGSPreprocessParams(
            degree: scene.manifest.shDegree,
            maxSHCoefficients: maxSHCoefficients,
            scaleModifier: 1,
            tanFovX: Float(scene.manifest.tanFovX),
            tanFovY: Float(scene.manifest.tanFovY),
            imageHeight: scene.manifest.height,
            imageWidth: scene.manifest.width,
            tileBounds: tileBounds,
            multiplier: 1
        )
    }

    private static func tileBounds(for scene: FastGSRecordedForwardScene) -> (x: Int, y: Int, z: Int) {
        (
            x: (scene.manifest.width + 15) / 16,
            y: (scene.manifest.height + 15) / 16,
            z: 1
        )
    }
}
