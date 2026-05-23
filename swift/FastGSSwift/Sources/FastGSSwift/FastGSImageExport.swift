import Foundation
import MLX

#if canImport(Metal)
import Metal
#endif

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

    #if canImport(Metal)
    public static func texture(
        outColor: MLXArray,
        width: Int,
        height: Int,
        device: MTLDevice,
        usage: MTLTextureUsage = [.shaderRead]
    ) -> MTLTexture? {
        if let texture = FastGSOutColorTextureRenderer.shared.texture(
            outColor: outColor,
            width: width,
            height: height,
            device: device,
            usage: usage
        ) {
            return texture
        }
        return texture(rgbaBytes: rgbaBytes(outColor: outColor, width: width, height: height), width: width, height: height, device: device, usage: usage)
    }

    public static func texture(
        rasterizeOutput: FastGSRasterizeOutput,
        width: Int,
        height: Int,
        device: MTLDevice,
        usage: MTLTextureUsage = [.shaderRead]
    ) -> MTLTexture? {
        texture(outColor: rasterizeOutput.outColor, width: width, height: height, device: device, usage: usage)
    }

    public static func texture(
        rgbaBytes: [UInt8],
        width: Int,
        height: Int,
        device: MTLDevice,
        usage: MTLTextureUsage = [.shaderRead]
    ) -> MTLTexture? {
        precondition(width > 0 && height > 0, "width and height must be positive.")
        precondition(rgbaBytes.count == width * height * 4, "rgbaBytes must contain width * height * 4 bytes.")

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = usage
        descriptor.storageMode = .managed

        guard let texture = device.makeTexture(descriptor: descriptor) else {
            return nil
        }

        texture.replace(
            region: MTLRegionMake2D(0, 0, width, height),
            mipmapLevel: 0,
            withBytes: rgbaBytes,
            bytesPerRow: width * 4
        )
        return texture
    }
    #endif

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
        let image = try cgImage(rgbaBytes: rgbaBytes, width: width, height: height)
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

    #if canImport(CoreGraphics)
    public static func cgImage(
        outColor: MLXArray,
        width: Int,
        height: Int
    ) throws -> CGImage {
        try cgImage(rgbaBytes: rgbaBytes(outColor: outColor, width: width, height: height), width: width, height: height)
    }

    public static func cgImage(
        rasterizeOutput: FastGSRasterizeOutput,
        width: Int,
        height: Int
    ) throws -> CGImage {
        try cgImage(outColor: rasterizeOutput.outColor, width: width, height: height)
    }

    public static func cgImage(
        rgbaBytes: [UInt8],
        width: Int,
        height: Int
    ) throws -> CGImage {
        precondition(width > 0 && height > 0, "width and height must be positive.")
        precondition(rgbaBytes.count == width * height * 4, "rgbaBytes must contain width * height * 4 bytes.")

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
        return image
    }
    #endif

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

#if canImport(Metal)
public enum FastGSOutColorTextureRendererError: Error, Equatable {
    case invalidShape([Int], expected: [Int])
    case invalidDType(DType)
    case cannotCreateCommandQueue
    case cannotCreateLibrary
    case cannotCreateFunction
    case cannotCreatePipeline
    case cannotCreateSourceBuffer
    case cannotCreateTexture
    case cannotCreateCommandBuffer
    case cannotCreateCommandEncoder
}

public final class FastGSOutColorTextureRenderer {
    public static let shared = FastGSOutColorTextureRenderer()

    private let lock = NSLock()
    private var pipelineCache: [ObjectIdentifier: MTLComputePipelineState] = [:]
    private var commandQueueCache: [ObjectIdentifier: MTLCommandQueue] = [:]

    public init() {}

    public func texture(
        outColor: MLXArray,
        width: Int,
        height: Int,
        device: MTLDevice,
        usage: MTLTextureUsage = [.shaderRead]
    ) -> MTLTexture? {
        try? makeTexture(outColor: outColor, width: width, height: height, device: device, usage: usage)
    }

    public func makeTexture(
        outColor: MLXArray,
        width: Int,
        height: Int,
        device: MTLDevice,
        usage: MTLTextureUsage = [.shaderRead]
    ) throws -> MTLTexture {
        precondition(width > 0 && height > 0, "width and height must be positive.")
        let expectedShape = [3, width * height]
        guard outColor.shape == expectedShape else {
            throw FastGSOutColorTextureRendererError.invalidShape(outColor.shape, expected: expectedShape)
        }
        guard outColor.dtype == .float32 else {
            throw FastGSOutColorTextureRendererError.invalidDType(outColor.dtype)
        }
        guard let sourceBuffer = outColor.asMTLBuffer(device: device, noCopy: true) else {
            throw FastGSOutColorTextureRendererError.cannotCreateSourceBuffer
        }

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = usage.union([.shaderWrite])
        descriptor.storageMode = .shared
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw FastGSOutColorTextureRendererError.cannotCreateTexture
        }

        try encodeCopy(outColorBuffer: sourceBuffer, width: width, height: height, texture: texture, device: device)
        return texture
    }

    public func copy(
        outColor: MLXArray,
        width: Int,
        height: Int,
        into texture: MTLTexture,
        device: MTLDevice
    ) throws {
        let expectedShape = [3, width * height]
        guard outColor.shape == expectedShape else {
            throw FastGSOutColorTextureRendererError.invalidShape(outColor.shape, expected: expectedShape)
        }
        guard outColor.dtype == .float32 else {
            throw FastGSOutColorTextureRendererError.invalidDType(outColor.dtype)
        }
        guard let sourceBuffer = outColor.asMTLBuffer(device: device, noCopy: true) else {
            throw FastGSOutColorTextureRendererError.cannotCreateSourceBuffer
        }
        try encodeCopy(outColorBuffer: sourceBuffer, width: width, height: height, texture: texture, device: device)
    }

    private func encodeCopy(
        outColorBuffer: MTLBuffer,
        width: Int,
        height: Int,
        texture: MTLTexture,
        device: MTLDevice
    ) throws {
        guard let commandQueue = try commandQueue(device: device) else {
            throw FastGSOutColorTextureRendererError.cannotCreateCommandQueue
        }
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw FastGSOutColorTextureRendererError.cannotCreateCommandBuffer
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw FastGSOutColorTextureRendererError.cannotCreateCommandEncoder
        }

        let pipeline = try pipeline(device: device)
        var constants = FastGSOutColorTextureConstants(width: UInt32(width), height: UInt32(height))
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(outColorBuffer, offset: 0, index: 0)
        encoder.setBytes(&constants, length: MemoryLayout<FastGSOutColorTextureConstants>.stride, index: 1)
        encoder.setTexture(texture, index: 0)

        let threadsPerThreadgroup = MTLSize(width: min(pipeline.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1)
        let grid = MTLSize(width: width * height, height: 1, depth: 1)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    private func commandQueue(device: MTLDevice) throws -> MTLCommandQueue? {
        let key = ObjectIdentifier(device)
        return lock.withLock {
            if let cached = commandQueueCache[key] {
                return cached
            }
            let queue = device.makeCommandQueue()
            commandQueueCache[key] = queue
            return queue
        }
    }

    private func pipeline(device: MTLDevice) throws -> MTLComputePipelineState {
        let key = ObjectIdentifier(device)
        return try lock.withLock {
            if let cached = pipelineCache[key] {
                return cached
            }
            guard let library = try? device.makeLibrary(source: Self.kernelSource, options: nil) else {
                throw FastGSOutColorTextureRendererError.cannotCreateLibrary
            }
            guard let function = library.makeFunction(name: "fastgs_out_color_to_texture") else {
                throw FastGSOutColorTextureRendererError.cannotCreateFunction
            }
            guard let pipeline = try? device.makeComputePipelineState(function: function) else {
                throw FastGSOutColorTextureRendererError.cannotCreatePipeline
            }
            pipelineCache[key] = pipeline
            return pipeline
        }
    }

    private struct FastGSOutColorTextureConstants {
        var width: UInt32
        var height: UInt32
    }

    private static let kernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct FastGSOutColorTextureConstants {
        uint width;
        uint height;
    };

    kernel void fastgs_out_color_to_texture(
        device const float* outColor [[buffer(0)]],
        constant FastGSOutColorTextureConstants& constants [[buffer(1)]],
        texture2d<float, access::write> output [[texture(0)]],
        uint pixel [[thread_position_in_grid]]
    ) {
        const uint pixelCount = constants.width * constants.height;
        if (pixel >= pixelCount) {
            return;
        }
        const uint x = pixel % constants.width;
        const uint y = pixel / constants.width;
        const float r = clamp(outColor[pixel], 0.0f, 1.0f);
        const float g = clamp(outColor[pixelCount + pixel], 0.0f, 1.0f);
        const float b = clamp(outColor[pixelCount * 2 + pixel], 0.0f, 1.0f);
        output.write(float4(r, g, b, 1.0f), uint2(x, y));
    }
    """
}
#endif
