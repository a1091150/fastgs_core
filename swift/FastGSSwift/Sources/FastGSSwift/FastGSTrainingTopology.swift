import MLX

public extension FastGSTrainableParameters {
    var gaussianCount: Int {
        means3D.shape.first ?? 0
    }

    func validateTopology() {
        let count = gaussianCount
        validateGaussianField(means3D, name: "means3D", count: count)
        validateGaussianField(dc, name: "dc", count: count)
        validateGaussianField(sh, name: "sh", count: count)
        validateGaussianField(opacityLogits, name: "opacityLogits", count: count)
        validateGaussianField(scales, name: "scales", count: count)
        validateGaussianField(rotations, name: "rotations", count: count)
        if let cov3DPrecomputed {
            validateGaussianField(cov3DPrecomputed, name: "cov3DPrecomputed", count: count)
        }
    }

    func take(indices: [Int], stream: StreamOrDevice = .default) -> FastGSTrainableParameters {
        validateTopology()
        let count = gaussianCount
        precondition(indices.allSatisfy { $0 >= 0 && $0 < count }, "take indices out of bounds")

        let indexArray = MLXArray(indices.map(Int32.init), [indices.count])
        return FastGSTrainableParameters(
            means3D: means3D.take(indexArray, axis: 0, stream: stream),
            dc: dc.take(indexArray, axis: 0, stream: stream),
            sh: sh.take(indexArray, axis: 0, stream: stream),
            opacityLogits: opacityLogits.take(indexArray, axis: 0, stream: stream),
            scales: scales.take(indexArray, axis: 0, stream: stream),
            rotations: rotations.take(indexArray, axis: 0, stream: stream),
            cov3DPrecomputed: cov3DPrecomputed?.take(indexArray, axis: 0, stream: stream)
        )
    }

    func prune(mask: [Bool], stream: StreamOrDevice = .default) -> FastGSTrainableParameters {
        validateTopology()
        precondition(mask.count == gaussianCount, "prune mask count mismatch")
        let keptIndices = mask.enumerated().compactMap { index, shouldPrune in
            shouldPrune ? nil : index
        }
        return take(indices: keptIndices, stream: stream)
    }

    func appending(_ tail: FastGSTrainableParameters, stream: StreamOrDevice = .default) -> FastGSTrainableParameters {
        validateTopology()
        tail.validateTopology()
        validateAppendShape(means3D, tail.means3D, name: "means3D")
        validateAppendShape(dc, tail.dc, name: "dc")
        validateAppendShape(sh, tail.sh, name: "sh")
        validateAppendShape(opacityLogits, tail.opacityLogits, name: "opacityLogits")
        validateAppendShape(scales, tail.scales, name: "scales")
        validateAppendShape(rotations, tail.rotations, name: "rotations")

        let cov3DPrecomputed: MLXArray?
        switch (self.cov3DPrecomputed, tail.cov3DPrecomputed) {
        case let (.some(lhs), .some(rhs)):
            validateAppendShape(lhs, rhs, name: "cov3DPrecomputed")
            cov3DPrecomputed = concatenated([lhs, rhs], axis: 0, stream: stream)
        case (.none, .none):
            cov3DPrecomputed = nil
        default:
            preconditionFailure("cov3DPrecomputed append shape mismatch")
        }

        return FastGSTrainableParameters(
            means3D: concatenated([means3D, tail.means3D], axis: 0, stream: stream),
            dc: concatenated([dc, tail.dc], axis: 0, stream: stream),
            sh: concatenated([sh, tail.sh], axis: 0, stream: stream),
            opacityLogits: concatenated([opacityLogits, tail.opacityLogits], axis: 0, stream: stream),
            scales: concatenated([scales, tail.scales], axis: 0, stream: stream),
            rotations: concatenated([rotations, tail.rotations], axis: 0, stream: stream),
            cov3DPrecomputed: cov3DPrecomputed
        )
    }
}

public extension FastGSAdamFieldState {
    func validateTopology(name: String) {
        validateGaussianField(firstMoment, name: "\(name).firstMoment", count: gaussianCount)
        validateGaussianField(secondMoment, name: "\(name).secondMoment", count: gaussianCount)
        precondition(firstMoment.shape == secondMoment.shape, "\(name) moment shape mismatch")
        precondition(firstMoment.dtype == secondMoment.dtype, "\(name) moment dtype mismatch")
    }

    var gaussianCount: Int {
        firstMoment.shape.first ?? 0
    }

    func take(indices: [Int], stream: StreamOrDevice = .default) -> FastGSAdamFieldState {
        validateTopology(name: "field")
        let count = gaussianCount
        precondition(indices.allSatisfy { $0 >= 0 && $0 < count }, "take indices out of bounds")
        let indexArray = MLXArray(indices.map(Int32.init), [indices.count])
        return FastGSAdamFieldState(
            firstMoment: firstMoment.take(indexArray, axis: 0, stream: stream),
            secondMoment: secondMoment.take(indexArray, axis: 0, stream: stream)
        )
    }

    func prune(mask: [Bool], stream: StreamOrDevice = .default) -> FastGSAdamFieldState {
        validateTopology(name: "field")
        precondition(mask.count == gaussianCount, "prune mask count mismatch")
        let keptIndices = mask.enumerated().compactMap { index, shouldPrune in
            shouldPrune ? nil : index
        }
        return take(indices: keptIndices, stream: stream)
    }

    func appendingZeroRows(like tailParameter: MLXArray, stream: StreamOrDevice = .default) -> FastGSAdamFieldState {
        validateTopology(name: "field")
        validateAppendShape(firstMoment, tailParameter, name: "field.firstMoment")
        validateAppendShape(secondMoment, tailParameter, name: "field.secondMoment")
        let firstZeros = MLXArray.zeros(tailParameter.shape, dtype: firstMoment.dtype, stream: stream)
        let secondZeros = MLXArray.zeros(tailParameter.shape, dtype: secondMoment.dtype, stream: stream)
        return FastGSAdamFieldState(
            firstMoment: concatenated([firstMoment, firstZeros], axis: 0, stream: stream),
            secondMoment: concatenated([secondMoment, secondZeros], axis: 0, stream: stream)
        )
    }

    func reset(toZerosLike parameter: MLXArray, stream: StreamOrDevice = .default) -> FastGSAdamFieldState {
        validateAppendShape(firstMoment, parameter, name: "field.firstMoment")
        validateAppendShape(secondMoment, parameter, name: "field.secondMoment")
        return FastGSAdamFieldState(
            firstMoment: MLXArray.zeros(parameter.shape, dtype: firstMoment.dtype, stream: stream),
            secondMoment: MLXArray.zeros(parameter.shape, dtype: secondMoment.dtype, stream: stream)
        )
    }
}

public extension FastGSAdamState {
    func validateTopology(parameters: FastGSTrainableParameters? = nil) {
        let count = means3D.gaussianCount
        means3D.validateTopology(name: "means3D")
        dc.validateTopology(name: "dc")
        sh.validateTopology(name: "sh")
        opacityLogits.validateTopology(name: "opacityLogits")
        scales.validateTopology(name: "scales")
        rotations.validateTopology(name: "rotations")
        cov3DPrecomputed?.validateTopology(name: "cov3DPrecomputed")
        precondition(dc.gaussianCount == count, "dc state Gaussian count mismatch")
        precondition(sh.gaussianCount == count, "sh state Gaussian count mismatch")
        precondition(opacityLogits.gaussianCount == count, "opacityLogits state Gaussian count mismatch")
        precondition(scales.gaussianCount == count, "scales state Gaussian count mismatch")
        precondition(rotations.gaussianCount == count, "rotations state Gaussian count mismatch")
        if let cov3DPrecomputed {
            precondition(cov3DPrecomputed.gaussianCount == count, "cov3DPrecomputed state Gaussian count mismatch")
        }

        if let parameters {
            parameters.validateTopology()
            precondition(means3D.firstMoment.shape == parameters.means3D.shape, "means3D state shape mismatch")
            precondition(dc.firstMoment.shape == parameters.dc.shape, "dc state shape mismatch")
            precondition(sh.firstMoment.shape == parameters.sh.shape, "sh state shape mismatch")
            precondition(opacityLogits.firstMoment.shape == parameters.opacityLogits.shape, "opacityLogits state shape mismatch")
            precondition(scales.firstMoment.shape == parameters.scales.shape, "scales state shape mismatch")
            precondition(rotations.firstMoment.shape == parameters.rotations.shape, "rotations state shape mismatch")
            precondition(
                cov3DPrecomputed?.firstMoment.shape == parameters.cov3DPrecomputed?.shape,
                "cov3DPrecomputed state shape mismatch"
            )
        }
    }

    func prune(mask: [Bool], stream: StreamOrDevice = .default) -> FastGSAdamState {
        validateTopology()
        precondition(mask.count == means3D.gaussianCount, "prune mask count mismatch")
        return FastGSAdamState(
            step: step,
            means3D: means3D.prune(mask: mask, stream: stream),
            dc: dc.prune(mask: mask, stream: stream),
            sh: sh.prune(mask: mask, stream: stream),
            opacityLogits: opacityLogits.prune(mask: mask, stream: stream),
            scales: scales.prune(mask: mask, stream: stream),
            rotations: rotations.prune(mask: mask, stream: stream),
            cov3DPrecomputed: cov3DPrecomputed?.prune(mask: mask, stream: stream)
        )
    }

    func appendingZeroRows(
        like tailParameters: FastGSTrainableParameters,
        stream: StreamOrDevice = .default
    ) -> FastGSAdamState {
        validateTopology()
        tailParameters.validateTopology()

        let cov3DPrecomputed: FastGSAdamFieldState?
        switch (self.cov3DPrecomputed, tailParameters.cov3DPrecomputed) {
        case let (.some(state), .some(parameter)):
            cov3DPrecomputed = state.appendingZeroRows(like: parameter, stream: stream)
        case (.none, .none):
            cov3DPrecomputed = nil
        default:
            preconditionFailure("cov3DPrecomputed append state mismatch")
        }

        return FastGSAdamState(
            step: step,
            means3D: means3D.appendingZeroRows(like: tailParameters.means3D, stream: stream),
            dc: dc.appendingZeroRows(like: tailParameters.dc, stream: stream),
            sh: sh.appendingZeroRows(like: tailParameters.sh, stream: stream),
            opacityLogits: opacityLogits.appendingZeroRows(like: tailParameters.opacityLogits, stream: stream),
            scales: scales.appendingZeroRows(like: tailParameters.scales, stream: stream),
            rotations: rotations.appendingZeroRows(like: tailParameters.rotations, stream: stream),
            cov3DPrecomputed: cov3DPrecomputed
        )
    }

    func resettingOpacityLogitState(like parameters: FastGSTrainableParameters, stream: StreamOrDevice = .default)
        -> FastGSAdamState
    {
        validateTopology(parameters: parameters)
        return FastGSAdamState(
            step: step,
            means3D: means3D,
            dc: dc,
            sh: sh,
            opacityLogits: opacityLogits.reset(toZerosLike: parameters.opacityLogits, stream: stream),
            scales: scales,
            rotations: rotations,
            cov3DPrecomputed: cov3DPrecomputed
        )
    }
}

private func validateGaussianField(_ array: MLXArray, name: String, count: Int) {
    precondition(!array.shape.isEmpty, "\(name) must include a Gaussian axis")
    precondition(array.shape[0] == count, "\(name) Gaussian count mismatch")
}

private func validateAppendShape(_ lhs: MLXArray, _ rhs: MLXArray, name: String) {
    precondition(!lhs.shape.isEmpty && !rhs.shape.isEmpty, "\(name) must include a Gaussian axis")
    precondition(lhs.shape.dropFirst().elementsEqual(rhs.shape.dropFirst()), "\(name) append suffix shape mismatch")
    precondition(lhs.dtype == rhs.dtype, "\(name) append dtype mismatch")
}
