import Foundation
import MLX

#if canImport(CoreGraphics) && canImport(ImageIO) && canImport(UniformTypeIdentifiers)
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
#endif

public enum FastGSImageExport {
    public static func rgbaBytes(
        outColor: MLXArray,
        width: Int,
        height: Int,
        alpha: UInt8 = 255
    ) -> [UInt8] {
        precondition(width > 0 && height > 0, "width and height must be positive.")
        precondition(outColor.shape == [3, width * height], "outColor must have shape [3, width * height].")
        precondition(outColor.dtype == .float32, "outColor must be float32.")

        let values = outColor.asArray(Float.self)
        let pixelCount = width * height
        var rgba = [UInt8](repeating: 0, count: pixelCount * 4)

        for pixel in 0..<pixelCount {
            rgba[4 * pixel + 0] = byte(values[pixel])
            rgba[4 * pixel + 1] = byte(values[pixelCount + pixel])
            rgba[4 * pixel + 2] = byte(values[2 * pixelCount + pixel])
            rgba[4 * pixel + 3] = alpha
        }

        return rgba
    }

    public static func rgbaBytes(
        rasterizeOutput: FastGSRasterizeOutput,
        width: Int,
        height: Int,
        alpha: UInt8 = 255
    ) -> [UInt8] {
        rgbaBytes(outColor: rasterizeOutput.outColor, width: width, height: height, alpha: alpha)
    }

    @discardableResult
    public static func writePNG(
        outColor: MLXArray,
        width: Int,
        height: Int,
        to url: URL
    ) throws -> URL {
        let rgba = rgbaBytes(outColor: outColor, width: width, height: height)
        try writePNG(rgbaBytes: rgba, width: width, height: height, to: url)
        return url
    }

    @discardableResult
    public static func writePNG(
        rasterizeOutput: FastGSRasterizeOutput,
        width: Int,
        height: Int,
        to url: URL
    ) throws -> URL {
        try writePNG(outColor: rasterizeOutput.outColor, width: width, height: height, to: url)
    }

    @discardableResult
    public static func writePNG(
        rgbaBytes: [UInt8],
        width: Int,
        height: Int,
        to url: URL
    ) throws -> URL {
        precondition(width > 0 && height > 0, "width and height must be positive.")
        precondition(rgbaBytes.count == width * height * 4, "rgbaBytes must contain width * height * 4 bytes.")

        #if canImport(CoreGraphics) && canImport(ImageIO) && canImport(UniformTypeIdentifiers)
        let data = Data(rgbaBytes)
        guard let provider = CGDataProvider(data: data as CFData) else {
            throw FastGSImageExportError.cannotCreateDataProvider
        }
        guard let image = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else {
            throw FastGSImageExportError.cannotCreateImage
        }
        guard let destination = CGImageDestinationCreateWithURL(
            url as CFURL,
            UTType.png.identifier as CFString,
            1,
            nil
        ) else {
            throw FastGSImageExportError.cannotCreateDestination
        }

        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw FastGSImageExportError.cannotFinalizeDestination
        }
        return url
        #else
        throw FastGSImageExportError.pngUnavailable
        #endif
    }

    private static func byte(_ value: Float) -> UInt8 {
        let clamped = min(max(value, 0), 1)
        return UInt8((clamped * 255).rounded())
    }
}

public enum FastGSImageExportError: Error, Equatable {
    case cannotCreateDataProvider
    case cannotCreateImage
    case cannotCreateDestination
    case cannotFinalizeDestination
    case pngUnavailable
}
