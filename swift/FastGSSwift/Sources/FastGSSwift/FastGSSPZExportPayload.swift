import Foundation
import MLX

public struct FastGSSPZExportPayload: Equatable {
    public var numPoints: Int
    public var shDegree: Int32
    public var positions: [Float]
    public var scales: [Float]
    public var rotationsXYZW: [Float]
    public var alphas: [Float]
    public var colors: [Float]
    public var sh: [Float]

    public init(
        numPoints: Int,
        shDegree: Int32,
        positions: [Float],
        scales: [Float],
        rotationsXYZW: [Float],
        alphas: [Float],
        colors: [Float],
        sh: [Float]
    ) {
        self.numPoints = numPoints
        self.shDegree = shDegree
        self.positions = positions
        self.scales = scales
        self.rotationsXYZW = rotationsXYZW
        self.alphas = alphas
        self.colors = colors
        self.sh = sh
    }
}

public enum FastGSSPZExportPayloadError: Error, Equatable {
    case unsupportedSHShape([Int])
}

public extension FastGSTrainableParameters {
    func spzExportPayload() throws -> FastGSSPZExportPayload {
        validateTopology()
        let count = gaussianCount
        let sh = try FastGSSPZExportPayload.shPayload(from: self.sh, gaussianCount: count)
        return FastGSSPZExportPayload(
            numPoints: count,
            shDegree: sh.degree,
            positions: FastGSSPZExportPayload.positionsForPLYRDFViewerRotate180(from: means3D.asArray(Float.self)),
            scales: scales.asArray(Float.self),
            rotationsXYZW: FastGSSPZExportPayload.rotationsXYZWForPLYRDFViewerRotate180(fromWXYZ: rotations.asArray(Float.self)),
            alphas: opacityLogits.asArray(Float.self),
            colors: dc.asArray(Float.self),
            sh: FastGSSPZExportPayload.shForPLYRDFViewerRotate180(
                sh.values,
                gaussianCount: count,
                shDegree: sh.degree
            )
        )
    }

    func spzExportPayload(
        scannerNormalizationTranslation translation: [Float],
        scannerNormalizationScale scale: Float
    ) throws -> FastGSSPZExportPayload {
        validateTopology()
        precondition(translation.count >= 3, "scanner normalization translation must contain xyz.")
        precondition(scale != 0, "scanner normalization scale must be non-zero.")

        let count = gaussianCount
        let sh = try FastGSSPZExportPayload.shPayload(from: self.sh, gaussianCount: count)
        return FastGSSPZExportPayload(
            numPoints: count,
            shDegree: sh.degree,
            positions: FastGSSPZExportPayload.positionsForScannerSPZRUBViewerRotate180(
                fromTrainingPositions: means3D.asArray(Float.self),
                translation: translation,
                scale: scale
            ),
            scales: FastGSSPZExportPayload.scalesForScannerPLYRDFViewer(
                fromTrainingLogScales: scales.asArray(Float.self),
                normalizationScale: scale
            ),
            rotationsXYZW: FastGSSPZExportPayload.rotationsXYZWForScannerSPZRUBViewerRotate180(
                fromTrainingWXYZ: rotations.asArray(Float.self)
            ),
            alphas: opacityLogits.asArray(Float.self),
            colors: dc.asArray(Float.self),
            sh: FastGSSPZExportPayload.shForPLYRDFViewerRotate180(
                sh.values,
                gaussianCount: count,
                shDegree: sh.degree
            )
        )
    }
}

private extension FastGSSPZExportPayload {
    static func shPayload(from array: MLXArray, gaussianCount: Int) throws -> (degree: Int32, values: [Float]) {
        let values = array.asArray(Float.self)
        guard gaussianCount > 0 else {
            return (0, [])
        }
        let channelCount = gaussianCount * 3
        guard values.count % channelCount == 0 else {
            throw FastGSSPZExportPayloadError.unsupportedSHShape(array.shape)
        }
        let coefficientsPerPoint = values.count / channelCount
        switch coefficientsPerPoint {
        case 0:
            return (0, [])
        case 1:
            return (0, [])
        case 3:
            return (1, values)
        case 4:
            return (1, dropFirstCoefficientPerPoint(values, gaussianCount: gaussianCount, coefficientsPerPoint: coefficientsPerPoint))
        case 8:
            return (2, values)
        case 9:
            return (2, dropFirstCoefficientPerPoint(values, gaussianCount: gaussianCount, coefficientsPerPoint: coefficientsPerPoint))
        case 15:
            return (3, values)
        case 16:
            return (3, dropFirstCoefficientPerPoint(values, gaussianCount: gaussianCount, coefficientsPerPoint: coefficientsPerPoint))
        default:
            throw FastGSSPZExportPayloadError.unsupportedSHShape(array.shape)
        }
    }

    static func dropFirstCoefficientPerPoint(
        _ values: [Float],
        gaussianCount: Int,
        coefficientsPerPoint: Int
    ) -> [Float] {
        var result = [Float]()
        result.reserveCapacity(gaussianCount * max(coefficientsPerPoint - 1, 0) * 3)
        for gaussian in 0..<gaussianCount {
            let pointStart = gaussian * coefficientsPerPoint * 3
            let restStart = pointStart + 3
            let pointEnd = pointStart + coefficientsPerPoint * 3
            if restStart < pointEnd {
                result.append(contentsOf: values[restStart..<pointEnd])
            }
        }
        return result
    }

    static func positionsForPLYRDFViewerRotate180(from positions: [Float]) -> [Float] {
        precondition(positions.count % 3 == 0, "FastGS positions must be [N, 3].")
        var result = [Float]()
        result.reserveCapacity(positions.count)
        for offset in stride(from: 0, to: positions.count, by: 3) {
            result.append(-positions[offset + 0])
            result.append(-positions[offset + 1])
            result.append(positions[offset + 2])
        }
        return result
    }

    static func positionsForScannerSPZRUBViewerRotate180(
        fromTrainingPositions positions: [Float],
        translation: [Float],
        scale: Float
    ) -> [Float] {
        precondition(positions.count % 3 == 0, "FastGS positions must be [N, 3].")
        var result = [Float]()
        result.reserveCapacity(positions.count)
        for offset in stride(from: 0, to: positions.count, by: 3) {
            let axisX = positions[offset + 0] / scale + translation[0]
            let axisY = positions[offset + 1] / scale + translation[1]
            let axisZ = positions[offset + 2] / scale + translation[2]

            let rawX = axisX
            let rawY = -axisZ
            let rawZ = axisY

            result.append(-rawX)
            result.append(rawY)
            result.append(-rawZ)
        }
        return result
    }

    static func scalesForScannerPLYRDFViewer(
        fromTrainingLogScales scales: [Float],
        normalizationScale: Float
    ) -> [Float] {
        let logScale = logf(normalizationScale)
        return scales.map { $0 - logScale }
    }

    static func rotationsXYZWForPLYRDFViewerRotate180(fromWXYZ rotations: [Float]) -> [Float] {
        precondition(rotations.count % 4 == 0, "FastGS rotations must be [N, 4] WXYZ quaternions.")
        var result = [Float]()
        result.reserveCapacity(rotations.count)
        for offset in stride(from: 0, to: rotations.count, by: 4) {
            let quaternion = quaternionWXYZForRotate180AroundZ(
                w: rotations[offset + 0],
                x: rotations[offset + 1],
                y: rotations[offset + 2],
                z: rotations[offset + 3]
            )
            result.append(quaternion.x)
            result.append(quaternion.y)
            result.append(quaternion.z)
            result.append(quaternion.w)
        }
        return result
    }

    static func rotationsXYZWForScannerSPZRUBViewerRotate180(fromTrainingWXYZ rotations: [Float]) -> [Float] {
        precondition(rotations.count % 4 == 0, "FastGS rotations must be [N, 4] WXYZ quaternions.")
        var result = [Float]()
        result.reserveCapacity(rotations.count)
        let half = Float(Foundation.sqrt(0.5))
        // Scanner training coordinates are raw RDF with y/z permuted by [x, z, -y].
        // SPZ stores RUB internally, so this conversion directly maps training
        // coordinates to the same RUB basis used by positions above. The resulting
        // 180 degree rotation has axis (0, 1, -1) / sqrt(2).
        let spzRUBFromTraining = (w: Float(0), x: Float(0), y: half, z: -half)
        for offset in stride(from: 0, to: rotations.count, by: 4) {
            let training = (
                w: rotations[offset + 0],
                x: rotations[offset + 1],
                y: rotations[offset + 2],
                z: rotations[offset + 3]
            )
            let quaternion = normalizedWXYZ(multiplyWXYZ(spzRUBFromTraining, training))
            result.append(quaternion.x)
            result.append(quaternion.y)
            result.append(quaternion.z)
            result.append(quaternion.w)
        }
        return result
    }

    static func shForPLYRDFViewerRotate180(_ values: [Float], gaussianCount: Int, shDegree: Int32) -> [Float] {
        guard gaussianCount > 0, shDegree > 0 else {
            return values
        }
        let signs = shAxisFlipSigns(flipX: -1, flipY: -1, flipZ: 1)
        let coefficientsPerPoint = Int(shCoefficientsWithoutDC(for: shDegree))
        guard coefficientsPerPoint > 0, values.count == gaussianCount * coefficientsPerPoint * 3 else {
            return values
        }
        var result = values
        for gaussian in 0..<gaussianCount {
            let pointStart = gaussian * coefficientsPerPoint * 3
            for coefficient in 0..<coefficientsPerPoint {
                let sign = signs[coefficient]
                let offset = pointStart + coefficient * 3
                result[offset + 0] *= sign
                result[offset + 1] *= sign
                result[offset + 2] *= sign
            }
        }
        return result
    }

    static func shCoefficientsWithoutDC(for degree: Int32) -> Int32 {
        switch degree {
        case 0:
            return 0
        case 1:
            return 3
        case 2:
            return 8
        default:
            return 15
        }
    }

    static func shAxisFlipSigns(flipX x: Float, flipY y: Float, flipZ z: Float) -> [Float] {
        [
            y,
            z,
            x,
            x * y,
            y * z,
            1,
            x * z,
            1,
            y,
            x * y * z,
            y,
            z,
            x,
            z,
            x,
        ]
    }

    static func quaternionWXYZForRotate180AroundZ(
        w: Float,
        x: Float,
        y: Float,
        z: Float
    ) -> (w: Float, x: Float, y: Float, z: Float) {
        // Left-multiply by the 180 degree Z rotation quaternion (0, 0, 0, 1).
        let quaternion = (
            w: -z,
            x: -y,
            y: x,
            z: w
        )
        let normSquared = quaternion.w * quaternion.w +
            quaternion.x * quaternion.x +
            quaternion.y * quaternion.y +
            quaternion.z * quaternion.z
        let norm = max(sqrtf(normSquared), 1.0e-8)
        return (
            w: quaternion.w / norm,
            x: quaternion.x / norm,
            y: quaternion.y / norm,
            z: quaternion.z / norm
        )
    }

    static func multiplyWXYZ(
        _ lhs: (w: Float, x: Float, y: Float, z: Float),
        _ rhs: (w: Float, x: Float, y: Float, z: Float)
    ) -> (w: Float, x: Float, y: Float, z: Float) {
        (
            w: lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z,
            x: lhs.w * rhs.x + lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y,
            y: lhs.w * rhs.y - lhs.x * rhs.z + lhs.y * rhs.w + lhs.z * rhs.x,
            z: lhs.w * rhs.z + lhs.x * rhs.y - lhs.y * rhs.x + lhs.z * rhs.w
        )
    }

    static func normalizedWXYZ(
        _ quaternion: (w: Float, x: Float, y: Float, z: Float)
    ) -> (w: Float, x: Float, y: Float, z: Float) {
        let normSquared = quaternion.w * quaternion.w +
            quaternion.x * quaternion.x +
            quaternion.y * quaternion.y +
            quaternion.z * quaternion.z
        let norm = max(sqrtf(normSquared), 1.0e-8)
        return (
            w: quaternion.w / norm,
            x: quaternion.x / norm,
            y: quaternion.y / norm,
            z: quaternion.z / norm
        )
    }

    static func rotationMatrixFromWXYZ(w: Float, x: Float, y: Float, z: Float) -> [Float] {
        let normSquared: Float = w * w + x * x + y * y + z * z
        let norm = max(sqrtf(normSquared), 1.0e-8)
        let w = w / norm
        let x = x / norm
        let y = y / norm
        let z = z / norm
        let xx = x * x
        let yy = y * y
        let zz = z * z
        let xy = x * y
        let xz = x * z
        let yz = y * z
        let wx = w * x
        let wy = w * y
        let wz = w * z
        return [
            1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy),
            2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),
            2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy),
        ]
    }

    static func scaniverseAxisTransform(_ matrix: [Float]) -> [Float] {
        [
            matrix[0], -matrix[2], matrix[1],
            -matrix[6], matrix[8], -matrix[7],
            matrix[3], -matrix[5], matrix[4],
        ]
    }

    static func quaternionWXYZ(fromRotationMatrix matrix: [Float]) -> (w: Float, x: Float, y: Float, z: Float) {
        let trace = matrix[0] + matrix[4] + matrix[8]
        let quaternion: (w: Float, x: Float, y: Float, z: Float)
        if trace > 0 {
            let s = sqrtf(trace + 1.0) * 2.0
            quaternion = (
                w: 0.25 * s,
                x: (matrix[7] - matrix[5]) / s,
                y: (matrix[2] - matrix[6]) / s,
                z: (matrix[3] - matrix[1]) / s
            )
        } else if matrix[0] > matrix[4], matrix[0] > matrix[8] {
            let s = sqrtf(1.0 + matrix[0] - matrix[4] - matrix[8]) * 2.0
            quaternion = (
                w: (matrix[7] - matrix[5]) / s,
                x: 0.25 * s,
                y: (matrix[1] + matrix[3]) / s,
                z: (matrix[2] + matrix[6]) / s
            )
        } else if matrix[4] > matrix[8] {
            let s = sqrtf(1.0 + matrix[4] - matrix[0] - matrix[8]) * 2.0
            quaternion = (
                w: (matrix[2] - matrix[6]) / s,
                x: (matrix[1] + matrix[3]) / s,
                y: 0.25 * s,
                z: (matrix[5] + matrix[7]) / s
            )
        } else {
            let s = sqrtf(1.0 + matrix[8] - matrix[0] - matrix[4]) * 2.0
            quaternion = (
                w: (matrix[3] - matrix[1]) / s,
                x: (matrix[2] + matrix[6]) / s,
                y: (matrix[5] + matrix[7]) / s,
                z: 0.25 * s
            )
        }
        let normSquared = quaternion.w * quaternion.w +
            quaternion.x * quaternion.x +
            quaternion.y * quaternion.y +
            quaternion.z * quaternion.z
        let norm = max(sqrtf(normSquared), 1.0e-8)
        return (
            w: quaternion.w / norm,
            x: quaternion.x / norm,
            y: quaternion.y / norm,
            z: quaternion.z / norm
        )
    }
}
