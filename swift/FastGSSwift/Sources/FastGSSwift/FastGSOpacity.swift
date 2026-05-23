import MLX

public enum FastGSOpacity {
    public static func probabilities(
        fromLogits logits: MLXArray,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        sigmoid(logits, stream: stream)
    }

    public static func logits(
        fromProbabilities probabilities: MLXArray,
        epsilon: Float = 1e-6,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        let clipped = clip(probabilities, min: epsilon, max: 1 - epsilon, stream: stream)
        return log(clipped / (1 - clipped), stream: stream)
    }
}
