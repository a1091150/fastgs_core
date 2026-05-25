import Foundation
import MLX

public enum FastGSLoss {
    public static let defaultLambdaDSSIM: Float = 0.2

    public static func fastGSCUDALoss(
        predictionCHW: MLXArray,
        targetCHW: MLXArray,
        width: Int,
        height: Int,
        lambdaDSSIM: Float = defaultLambdaDSSIM,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        let l1 = mean(abs(predictionCHW - targetCHW), stream: stream)
        let ssimValue = ssimCHW(
            predictionCHW,
            targetCHW,
            width: width,
            height: height,
            stream: stream
        )
        return (1.0 - lambdaDSSIM) * l1 + lambdaDSSIM * (1.0 - ssimValue)
    }

    public static func ssimCHW(
        _ lhsCHW: MLXArray,
        _ rhsCHW: MLXArray,
        width: Int,
        height: Int,
        windowSize: Int = 11,
        sigma: Float = 1.5,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        precondition(width > 0 && height > 0, "SSIM image dimensions must be positive.")
        precondition(windowSize % 2 == 1, "SSIM window size must be odd.")

        let lhs = reshaped(lhsCHW, 3, height, width, stream: stream)
        let rhs = reshaped(rhsCHW, 3, height, width, stream: stream)
        let kernel = gaussianKernel2D(windowSize: windowSize, sigma: sigma)
        let weight = MLXArray(kernel, [1, windowSize, windowSize, 1])
        let padding = IntOrPair(integerLiteral: windowSize / 2)

        let lhsChannels = lhs.split(parts: 3, axis: 0, stream: stream)
        let rhsChannels = rhs.split(parts: 3, axis: 0, stream: stream)
        var maps = [MLXArray]()
        maps.reserveCapacity(3)

        for channel in 0..<3 {
            let lhsChannel = nhwcSingleChannel(lhsChannels[channel], stream: stream)
            let rhsChannel = nhwcSingleChannel(rhsChannels[channel], stream: stream)
            let muLhs = conv2d(lhsChannel, weight, padding: padding, stream: stream)
            let muRhs = conv2d(rhsChannel, weight, padding: padding, stream: stream)
            let muLhsSquared = muLhs * muLhs
            let muRhsSquared = muRhs * muRhs
            let muLhsMuRhs = muLhs * muRhs
            let sigmaLhsSquared = conv2d(lhsChannel * lhsChannel, weight, padding: padding, stream: stream) - muLhsSquared
            let sigmaRhsSquared = conv2d(rhsChannel * rhsChannel, weight, padding: padding, stream: stream) - muRhsSquared
            let sigmaLhsRhs = conv2d(lhsChannel * rhsChannel, weight, padding: padding, stream: stream) - muLhsMuRhs

            let c1 = MLXArray(Float(0.01 * 0.01))
            let c2 = MLXArray(Float(0.03 * 0.03))
            let numerator = (2.0 * muLhsMuRhs + c1) * (2.0 * sigmaLhsRhs + c2)
            let denominator = (muLhsSquared + muRhsSquared + c1) * (sigmaLhsSquared + sigmaRhsSquared + c2)
            maps.append(numerator / denominator)
        }

        return mean(concatenated(maps, axis: 3, stream: stream), stream: stream)
    }

    private static func nhwcSingleChannel(_ chwChannel: MLXArray, stream: StreamOrDevice) -> MLXArray {
        expandedDimensions(chwChannel.transposed(1, 2, 0, stream: stream), axis: 0, stream: stream)
    }

    private static func gaussianKernel2D(windowSize: Int, sigma: Float) -> [Float] {
        let center = windowSize / 2
        var oneD = [Float]()
        oneD.reserveCapacity(windowSize)
        for index in 0..<windowSize {
            let x = Float(index - center)
            oneD.append(expf(-(x * x) / (2.0 * sigma * sigma)))
        }
        let sum = oneD.reduce(Float(0), +)
        let normalized = oneD.map { $0 / sum }
        var kernel = [Float]()
        kernel.reserveCapacity(windowSize * windowSize)
        for y in 0..<windowSize {
            for x in 0..<windowSize {
                kernel.append(normalized[y] * normalized[x])
            }
        }
        return kernel
    }
}
