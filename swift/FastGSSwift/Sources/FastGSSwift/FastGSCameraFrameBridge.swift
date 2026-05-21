import Foundation

#if canImport(CoreVideo)
import CoreVideo

#if canImport(Metal)
import Metal
#endif

public enum FastGSCameraFrameBridge {
    public enum BridgeError: Error, Equatable {
        case unsupportedPixelFormat(OSType)
        case missingBaseAddress
    }

    public struct LockedBGRAFrame {
        public let width: Int
        public let height: Int
        public let bytesPerRow: Int
        public let rgbaBytes: [UInt8]
        public let hasIOSurface: Bool
    }

    public static func lockBGRAFrame(_ pixelBuffer: CVPixelBuffer) throws -> LockedBGRAFrame {
        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        guard pixelFormat == kCVPixelFormatType_32BGRA else {
            throw BridgeError.unsupportedPixelFormat(pixelFormat)
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
        }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw BridgeError.missingBaseAddress
        }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        var rgbaBytes = [UInt8](repeating: 0, count: width * height * 4)
        let source = baseAddress.assumingMemoryBound(to: UInt8.self)

        for row in 0..<height {
            let sourceRow = source.advanced(by: row * bytesPerRow)
            let destinationRow = row * width * 4

            for column in 0..<width {
                let sourcePixel = sourceRow.advanced(by: column * 4)
                let destinationPixel = destinationRow + column * 4
                rgbaBytes[destinationPixel + 0] = sourcePixel[2]
                rgbaBytes[destinationPixel + 1] = sourcePixel[1]
                rgbaBytes[destinationPixel + 2] = sourcePixel[0]
                rgbaBytes[destinationPixel + 3] = sourcePixel[3]
            }
        }

        return LockedBGRAFrame(
            width: width,
            height: height,
            bytesPerRow: bytesPerRow,
            rgbaBytes: rgbaBytes,
            hasIOSurface: CVPixelBufferGetIOSurface(pixelBuffer) != nil
        )
    }

    #if canImport(Metal)
    public static func texture(
        fromBGRA pixelBuffer: CVPixelBuffer,
        device: MTLDevice,
        usage: MTLTextureUsage = [.shaderRead]
    ) throws -> MTLTexture? {
        let frame = try lockBGRAFrame(pixelBuffer)
        return FastGSImageExport.texture(
            rgbaBytes: frame.rgbaBytes,
            width: frame.width,
            height: frame.height,
            device: device,
            usage: usage
        )
    }
    #endif
}
#endif
