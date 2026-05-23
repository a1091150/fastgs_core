import Foundation
import MLX

public struct FastGSTrainableParameters {
    public var means3D: MLXArray
    public var dc: MLXArray
    public var sh: MLXArray
    public var opacities: MLXArray
    public var scales: MLXArray
    public var rotations: MLXArray
    public var cov3DPrecomputed: MLXArray?

    public init(
        means3D: MLXArray,
        dc: MLXArray,
        sh: MLXArray,
        opacities: MLXArray,
        scales: MLXArray,
        rotations: MLXArray,
        cov3DPrecomputed: MLXArray? = nil
    ) {
        self.means3D = means3D
        self.dc = dc
        self.sh = sh
        self.opacities = opacities
        self.scales = scales
        self.rotations = rotations
        self.cov3DPrecomputed = cov3DPrecomputed
    }

    public var arrays: [MLXArray] {
        var result = [means3D, dc, sh, opacities, scales, rotations]
        if let cov3DPrecomputed {
            result.append(cov3DPrecomputed)
        }
        return result
    }
}

public struct FastGSTrainableGradients {
    public var means3D: MLXArray
    public var dc: MLXArray
    public var sh: MLXArray
    public var opacities: MLXArray
    public var scales: MLXArray
    public var rotations: MLXArray
    public var cov3DPrecomputed: MLXArray?

    public init(
        means3D: MLXArray,
        dc: MLXArray,
        sh: MLXArray,
        opacities: MLXArray,
        scales: MLXArray,
        rotations: MLXArray,
        cov3DPrecomputed: MLXArray? = nil
    ) {
        self.means3D = means3D
        self.dc = dc
        self.sh = sh
        self.opacities = opacities
        self.scales = scales
        self.rotations = rotations
        self.cov3DPrecomputed = cov3DPrecomputed
    }
}

public struct FastGSAdamLearningRates {
    public var means3D: Float
    public var dc: Float
    public var sh: Float
    public var opacities: Float
    public var scales: Float
    public var rotations: Float
    public var cov3DPrecomputed: Float

    public init(
        means3D: Float = 1e-3,
        dc: Float = 1e-3,
        sh: Float = 1e-3,
        opacities: Float = 1e-3,
        scales: Float = 1e-3,
        rotations: Float = 1e-3,
        cov3DPrecomputed: Float = 1e-3
    ) {
        self.means3D = means3D
        self.dc = dc
        self.sh = sh
        self.opacities = opacities
        self.scales = scales
        self.rotations = rotations
        self.cov3DPrecomputed = cov3DPrecomputed
    }
}

public struct FastGSAdamFieldState {
    public var firstMoment: MLXArray
    public var secondMoment: MLXArray

    public init(firstMoment: MLXArray, secondMoment: MLXArray) {
        self.firstMoment = firstMoment
        self.secondMoment = secondMoment
    }

    public init(zerosLike parameter: MLXArray) {
        self.firstMoment = MLXArray.zeros(like: parameter)
        self.secondMoment = MLXArray.zeros(like: parameter)
    }

    public var arrays: [MLXArray] {
        [firstMoment, secondMoment]
    }
}

public struct FastGSAdamState {
    public var step: Int
    public var means3D: FastGSAdamFieldState
    public var dc: FastGSAdamFieldState
    public var sh: FastGSAdamFieldState
    public var opacities: FastGSAdamFieldState
    public var scales: FastGSAdamFieldState
    public var rotations: FastGSAdamFieldState
    public var cov3DPrecomputed: FastGSAdamFieldState?

    public init(
        step: Int = 0,
        means3D: FastGSAdamFieldState,
        dc: FastGSAdamFieldState,
        sh: FastGSAdamFieldState,
        opacities: FastGSAdamFieldState,
        scales: FastGSAdamFieldState,
        rotations: FastGSAdamFieldState,
        cov3DPrecomputed: FastGSAdamFieldState? = nil
    ) {
        self.step = step
        self.means3D = means3D
        self.dc = dc
        self.sh = sh
        self.opacities = opacities
        self.scales = scales
        self.rotations = rotations
        self.cov3DPrecomputed = cov3DPrecomputed
    }

    public init(step: Int = 0, parameters: FastGSTrainableParameters) {
        self.step = step
        self.means3D = FastGSAdamFieldState(zerosLike: parameters.means3D)
        self.dc = FastGSAdamFieldState(zerosLike: parameters.dc)
        self.sh = FastGSAdamFieldState(zerosLike: parameters.sh)
        self.opacities = FastGSAdamFieldState(zerosLike: parameters.opacities)
        self.scales = FastGSAdamFieldState(zerosLike: parameters.scales)
        self.rotations = FastGSAdamFieldState(zerosLike: parameters.rotations)
        self.cov3DPrecomputed = parameters.cov3DPrecomputed.map {
            FastGSAdamFieldState(zerosLike: $0)
        }
    }

    public var arrays: [MLXArray] {
        var result = [MLXArray]()
        result.append(contentsOf: means3D.arrays)
        result.append(contentsOf: dc.arrays)
        result.append(contentsOf: sh.arrays)
        result.append(contentsOf: opacities.arrays)
        result.append(contentsOf: scales.arrays)
        result.append(contentsOf: rotations.arrays)
        if let cov3DPrecomputed {
            result.append(contentsOf: cov3DPrecomputed.arrays)
        }
        return result
    }
}

public struct FastGSAdamOptimizer {
    public var learningRates: FastGSAdamLearningRates
    public var beta1: Float
    public var beta2: Float
    public var epsilon: Float
    public var biasCorrection: Bool
    public private(set) var state: FastGSAdamState?

    public init(
        learningRates: FastGSAdamLearningRates = FastGSAdamLearningRates(),
        beta1: Float = 0.9,
        beta2: Float = 0.999,
        epsilon: Float = 1e-8,
        biasCorrection: Bool = true
    ) {
        precondition(beta1 >= 0 && beta1 < 1, "beta1 must be in [0, 1)")
        precondition(beta2 >= 0 && beta2 < 1, "beta2 must be in [0, 1)")
        precondition(epsilon >= 0, "epsilon must be non-negative")
        self.learningRates = learningRates
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.biasCorrection = biasCorrection
    }

    public mutating func update(
        parameters: FastGSTrainableParameters,
        gradients: FastGSTrainableGradients,
        stream: StreamOrDevice = .default
    ) -> FastGSTrainableParameters {
        validate(parameters: parameters, gradients: gradients)

        var currentState = state ?? FastGSAdamState(parameters: parameters)
        currentState.step += 1

        let means3D = updateField(
            parameter: parameters.means3D,
            gradient: gradients.means3D,
            learningRate: learningRates.means3D,
            step: currentState.step,
            state: &currentState.means3D,
            stream: stream
        )
        let dc = updateField(
            parameter: parameters.dc,
            gradient: gradients.dc,
            learningRate: learningRates.dc,
            step: currentState.step,
            state: &currentState.dc,
            stream: stream
        )
        let sh = updateField(
            parameter: parameters.sh,
            gradient: gradients.sh,
            learningRate: learningRates.sh,
            step: currentState.step,
            state: &currentState.sh,
            stream: stream
        )
        let opacities = updateField(
            parameter: parameters.opacities,
            gradient: gradients.opacities,
            learningRate: learningRates.opacities,
            step: currentState.step,
            state: &currentState.opacities,
            stream: stream
        )
        let scales = updateField(
            parameter: parameters.scales,
            gradient: gradients.scales,
            learningRate: learningRates.scales,
            step: currentState.step,
            state: &currentState.scales,
            stream: stream
        )
        let rotations = updateField(
            parameter: parameters.rotations,
            gradient: gradients.rotations,
            learningRate: learningRates.rotations,
            step: currentState.step,
            state: &currentState.rotations,
            stream: stream
        )

        let cov3DPrecomputed: MLXArray?
        if let parameter = parameters.cov3DPrecomputed,
           let gradient = gradients.cov3DPrecomputed {
            var fieldState = currentState.cov3DPrecomputed ?? FastGSAdamFieldState(zerosLike: parameter)
            cov3DPrecomputed = updateField(
                parameter: parameter,
                gradient: gradient,
                learningRate: learningRates.cov3DPrecomputed,
                step: currentState.step,
                state: &fieldState,
                stream: stream
            )
            currentState.cov3DPrecomputed = fieldState
        } else {
            cov3DPrecomputed = parameters.cov3DPrecomputed
            currentState.cov3DPrecomputed = nil
        }

        state = currentState

        return FastGSTrainableParameters(
            means3D: means3D,
            dc: dc,
            sh: sh,
            opacities: opacities,
            scales: scales,
            rotations: rotations,
            cov3DPrecomputed: cov3DPrecomputed
        )
    }

    public func stateArrays() -> [MLXArray] {
        state?.arrays ?? []
    }

    private mutating func updateField(
        parameter: MLXArray,
        gradient: MLXArray,
        learningRate: Float,
        step: Int,
        state: inout FastGSAdamFieldState,
        stream: StreamOrDevice
    ) -> MLXArray {
        let firstMoment = beta1 * state.firstMoment + (1 - beta1) * gradient
        let secondMoment = beta2 * state.secondMoment + (1 - beta2) * square(gradient)

        let update: MLXArray
        if biasCorrection {
            let firstScale = Float(1 - pow(Double(beta1), Double(step)))
            let secondScale = Float(1 - pow(Double(beta2), Double(step)))
            let firstMomentHat = firstMoment / firstScale
            let secondMomentHat = secondMoment / secondScale
            update = learningRate * firstMomentHat / (sqrt(secondMomentHat, stream: stream) + epsilon)
        } else {
            update = learningRate * firstMoment / (sqrt(secondMoment, stream: stream) + epsilon)
        }

        state = FastGSAdamFieldState(firstMoment: firstMoment, secondMoment: secondMoment)
        return parameter - update
    }

    private func validate(
        parameters: FastGSTrainableParameters,
        gradients: FastGSTrainableGradients
    ) {
        precondition(parameters.means3D.shape == gradients.means3D.shape, "means3D gradient shape mismatch")
        precondition(parameters.dc.shape == gradients.dc.shape, "dc gradient shape mismatch")
        precondition(parameters.sh.shape == gradients.sh.shape, "sh gradient shape mismatch")
        precondition(parameters.opacities.shape == gradients.opacities.shape, "opacities gradient shape mismatch")
        precondition(parameters.scales.shape == gradients.scales.shape, "scales gradient shape mismatch")
        precondition(parameters.rotations.shape == gradients.rotations.shape, "rotations gradient shape mismatch")
        precondition(
            parameters.cov3DPrecomputed?.shape == gradients.cov3DPrecomputed?.shape,
            "cov3DPrecomputed gradient shape mismatch"
        )
    }
}
