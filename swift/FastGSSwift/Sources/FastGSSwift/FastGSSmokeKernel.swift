import MLX

public enum FastGSSmokeKernel {
    private static let kernel = MLXFast.metalKernel(
        name: "fastgs_smoke_double",
        inputNames: ["x"],
        outputNames: ["out"],
        source: """
            uint elem = thread_position_in_grid.x;
            out[elem] = x[elem] * 2.0f;
        """
    )

    public static func double(_ input: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
        precondition(input.dtype == .float32, "FastGSSmokeKernel.double expects float32 input.")

        let count = input.size
        let threadGroupSize = max(1, min(256, count))

        return kernel(
            [input],
            grid: (count, 1, 1),
            threadGroup: (threadGroupSize, 1, 1),
            outputShapes: [input.shape],
            outputDTypes: [input.dtype],
            stream: stream
        )[0]
    }
}
