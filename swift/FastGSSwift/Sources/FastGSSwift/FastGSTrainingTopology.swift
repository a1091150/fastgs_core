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
        validateGaussianField(opacities, name: "opacities", count: count)
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
            opacities: opacities.take(indexArray, axis: 0, stream: stream),
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
        validateAppendShape(opacities, tail.opacities, name: "opacities")
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
            opacities: concatenated([opacities, tail.opacities], axis: 0, stream: stream),
            scales: concatenated([scales, tail.scales], axis: 0, stream: stream),
            rotations: concatenated([rotations, tail.rotations], axis: 0, stream: stream),
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
