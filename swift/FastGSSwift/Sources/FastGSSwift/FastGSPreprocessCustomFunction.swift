import Foundation
import MLX

public enum FastGSPreprocessCustomFunction {
    public static func call(
        _ input: FastGSPreprocessInput,
        params: FastGSPreprocessParams,
        stream: StreamOrDevice = .default
    ) -> FastGSPreprocessOutput {
        let function = make(params: params, stream: stream)
        return output(from: function(arrays(from: input)))
    }

    public static func make(
        params: FastGSPreprocessParams,
        stream: StreamOrDevice = .default
    ) -> ([MLXArray]) -> [MLXArray] {
        let cache = FastGSPreprocessCustomFunctionCache()
        return CustomFunction {
            Forward { primals in
                let output = FastGSPreprocess.forward(input(from: primals), params: params, stream: stream)
                cache.store(output: output)
                return arrays(from: output)
            }
            VJP { primals, cotangents in
                let input = input(from: primals)
                guard let output = cache.output else {
                    preconditionFailure("FastGSPreprocessCustomFunction missing cached forward output.")
                }
                let gradients = FastGSPreprocessBackward.forward(
                    input: input,
                    cotangents: cotangentsFromPreprocessOutputs(cotangents),
                    forwardOutput: output,
                    params: params,
                    stream: stream
                )
                return arrays(from: gradients)
            }
        }
    }

    public static func arrays(from input: FastGSPreprocessInput) -> [MLXArray] {
        [
            input.means3D,
            input.dc,
            input.sh,
            input.colorsPrecomputed,
            input.opacities,
            input.scales,
            input.rotations,
            input.cov3DPrecomputed,
            input.viewMatrix,
            input.projectionMatrix,
            input.cameraPosition,
            input.viewspacePoints,
        ]
    }

    public static func input(from arrays: [MLXArray]) -> FastGSPreprocessInput {
        precondition(arrays.count == 12, "FastGSPreprocessCustomFunction expects 12 primals.")
        return FastGSPreprocessInput(
            means3D: arrays[0],
            dc: arrays[1],
            sh: arrays[2],
            colorsPrecomputed: arrays[3],
            opacities: arrays[4],
            scales: arrays[5],
            rotations: arrays[6],
            cov3DPrecomputed: arrays[7],
            viewMatrix: arrays[8],
            projectionMatrix: arrays[9],
            cameraPosition: arrays[10],
            viewspacePoints: arrays[11]
        )
    }

    public static func arrays(from output: FastGSPreprocessOutput) -> [MLXArray] {
        [
            output.radii,
            output.xy,
            output.depths,
            output.cov3D,
            output.rgb,
            output.conicOpacity,
            output.tilesTouched,
            output.clamped,
            output.viewspacePoints,
        ]
    }

    public static func output(from arrays: [MLXArray]) -> FastGSPreprocessOutput {
        precondition(arrays.count == 9, "FastGSPreprocessCustomFunction expects 9 outputs.")
        return FastGSPreprocessOutput(
            radii: arrays[0],
            xy: arrays[1],
            depths: arrays[2],
            cov3D: arrays[3],
            rgb: arrays[4],
            conicOpacity: arrays[5],
            tilesTouched: arrays[6],
            clamped: arrays[7],
            viewspacePoints: arrays[8]
        )
    }

    public static func arrays(from output: FastGSPreprocessBackwardOutput) -> [MLXArray] {
        [
            output.means3D,
            output.dc,
            output.sh,
            output.colorsPrecomputed,
            output.opacities,
            output.scales,
            output.rotations,
            output.cov3DPrecomputed,
            output.viewMatrix,
            output.projectionMatrix,
            output.cameraPosition,
            output.viewspacePoints,
        ]
    }

    private static func cotangentsFromPreprocessOutputs(_ arrays: [MLXArray]) -> FastGSPreprocessCotangents {
        precondition(arrays.count == 9, "FastGSPreprocessCustomFunction expects 9 output cotangents.")
        return FastGSPreprocessCotangents(
            radii: arrays[0],
            xy: arrays[1],
            depths: arrays[2],
            cov3D: arrays[3],
            rgb: arrays[4],
            conicOpacity: arrays[5],
            tilesTouched: arrays[6],
            clamped: arrays[7],
            viewspacePoints: arrays[8]
        )
    }
}

private final class FastGSPreprocessCustomFunctionCache: @unchecked Sendable {
    private let lock = NSLock()
    private var cachedOutput: FastGSPreprocessOutput?

    var output: FastGSPreprocessOutput? {
        lock.withLock { cachedOutput }
    }

    func store(output: FastGSPreprocessOutput) {
        lock.withLock {
            cachedOutput = output
        }
    }
}
