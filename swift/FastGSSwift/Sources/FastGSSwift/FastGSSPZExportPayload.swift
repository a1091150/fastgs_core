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
            positions: FastGSSPZExportPayload.positionsForScaniversePreview(from: means3D.asArray(Float.self)),
            scales: scales.asArray(Float.self),
            rotationsXYZW: FastGSSPZExportPayload.rotationsXYZWForScaniversePreview(fromWXYZ: rotations.asArray(Float.self)),
            alphas: opacityLogits.asArray(Float.self),
            colors: dc.asArray(Float.self),
            sh: sh.values
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

    static func positionsForScaniversePreview(from positions: [Float]) -> [Float] {
        precondition(positions.count % 3 == 0, "FastGS positions must be [N, 3].")
        var result = [Float]()
        result.reserveCapacity(positions.count)
        for offset in stride(from: 0, to: positions.count, by: 3) {
            result.append(positions[offset + 0])
            result.append(-positions[offset + 2])
            result.append(positions[offset + 1])
        }
        return result
    }

    static func rotationsXYZWForScaniversePreview(fromWXYZ rotations: [Float]) -> [Float] {
        precondition(rotations.count % 4 == 0, "FastGS rotations must be [N, 4] WXYZ quaternions.")
        var result = [Float]()
        result.reserveCapacity(rotations.count)
        for offset in stride(from: 0, to: rotations.count, by: 4) {
            let matrix = rotationMatrixFromWXYZ(
                w: rotations[offset + 0],
                x: rotations[offset + 1],
                y: rotations[offset + 2],
                z: rotations[offset + 3]
            )
            let transformed = scaniverseAxisTransform(matrix)
            let quaternion = quaternionWXYZ(fromRotationMatrix: transformed)
            result.append(quaternion.x)
            result.append(quaternion.y)
            result.append(quaternion.z)
            result.append(quaternion.w)
        }
        return result
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
