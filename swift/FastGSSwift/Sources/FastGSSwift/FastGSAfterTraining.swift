import MLX

public struct FastGSOpacityAfterTrainResult {
    public var parameters: FastGSTrainableParameters
    public var optimizerState: FastGSAdamState?

    public init(parameters: FastGSTrainableParameters, optimizerState: FastGSAdamState?) {
        self.parameters = parameters
        self.optimizerState = optimizerState
    }
}

public enum FastGSAfterTraining {
    public static func capOpacity(
        parameters: FastGSTrainableParameters,
        optimizerState: FastGSAdamState? = nil,
        maxOpacity: Float,
        stream: StreamOrDevice = .default
    ) -> FastGSOpacityAfterTrainResult {
        precondition(maxOpacity >= 0 && maxOpacity <= 1, "maxOpacity must be in [0, 1]")
        parameters.validateTopology()
        optimizerState?.validateTopology(parameters: parameters)

        var updated = parameters
        updated.opacities = minimum(parameters.opacities, maxOpacity, stream: stream)
        let updatedState = optimizerState?.resettingOpacityState(like: updated, stream: stream)
        return FastGSOpacityAfterTrainResult(parameters: updated, optimizerState: updatedState)
    }

    public static func resetOpacity(
        parameters: FastGSTrainableParameters,
        optimizerState: FastGSAdamState? = nil,
        resetValue: Float,
        stream: StreamOrDevice = .default
    ) -> FastGSOpacityAfterTrainResult {
        capOpacity(
            parameters: parameters,
            optimizerState: optimizerState,
            maxOpacity: resetValue,
            stream: stream
        )
    }
}
