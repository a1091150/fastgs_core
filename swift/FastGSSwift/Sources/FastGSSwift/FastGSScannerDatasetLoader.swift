import CoreGraphics
import Foundation
import ImageIO

public struct FastGSScannerDatasetOptions {
    public var width: Int
    public var height: Int
    public var maxFrames: Int
    public var frameStep: Int
    public var startIndex: Int
    public var normalizeWithAllFramePairs: Bool

    public init(
        width: Int = 512,
        height: Int = 512,
        maxFrames: Int = 0,
        frameStep: Int = 1,
        startIndex: Int = 0,
        normalizeWithAllFramePairs: Bool = false
    ) {
        self.width = width
        self.height = height
        self.maxFrames = maxFrames
        self.frameStep = max(1, frameStep)
        self.startIndex = startIndex
        self.normalizeWithAllFramePairs = normalizeWithAllFramePairs
    }
}

public struct FastGSScannerFrame {
    public var index: Int
    public var imageURL: URL
    public var jsonURL: URL
    public var camera: FastGSScannerCamera
    public var targetCHW: [Float]

    public init(
        index: Int,
        imageURL: URL,
        jsonURL: URL,
        camera: FastGSScannerCamera,
        targetCHW: [Float]
    ) {
        self.index = index
        self.imageURL = imageURL
        self.jsonURL = jsonURL
        self.camera = camera
        self.targetCHW = targetCHW
    }
}

public struct FastGSScannerCamera {
    public var viewmatrix: [Float]
    public var projmatrix: [Float]
    public var campos: [Float]
    public var imageWidth: Int
    public var imageHeight: Int
    public var tanFovX: Float
    public var tanFovY: Float

    public init(
        viewmatrix: [Float],
        projmatrix: [Float],
        campos: [Float],
        imageWidth: Int,
        imageHeight: Int,
        tanFovX: Float,
        tanFovY: Float
    ) {
        self.viewmatrix = viewmatrix
        self.projmatrix = projmatrix
        self.campos = campos
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.tanFovX = tanFovX
        self.tanFovY = tanFovY
    }
}

public struct FastGSScannerDataset {
    public var directory: URL
    public var pointCloud: FastGSPointCloud
    public var basePointCount: Int
    public var frames: [FastGSScannerFrame]
    public var normalizationTranslation: [Float]
    public var normalizationScale: Float

    public init(
        directory: URL,
        pointCloud: FastGSPointCloud,
        basePointCount: Int,
        frames: [FastGSScannerFrame],
        normalizationTranslation: [Float],
        normalizationScale: Float
    ) {
        self.directory = directory
        self.pointCloud = pointCloud
        self.basePointCount = basePointCount
        self.frames = frames
        self.normalizationTranslation = normalizationTranslation
        self.normalizationScale = normalizationScale
    }
}

public struct FastGSScannerFrameDescriptor {
    public var index: Int
    public var imageURL: URL
    public var jsonURL: URL

    public init(index: Int, imageURL: URL, jsonURL: URL) {
        self.index = index
        self.imageURL = imageURL
        self.jsonURL = jsonURL
    }
}

public struct FastGSScannerDatasetCache {
    public var directory: URL
    public var pointCloud: FastGSPointCloud
    public var basePointCount: Int
    public var frameDescriptors: [FastGSScannerFrameDescriptor]
    public var normalizationTranslation: [Float]
    public var normalizationScale: Float

    public init(
        directory: URL,
        pointCloud: FastGSPointCloud,
        basePointCount: Int,
        frameDescriptors: [FastGSScannerFrameDescriptor],
        normalizationTranslation: [Float],
        normalizationScale: Float
    ) {
        self.directory = directory
        self.pointCloud = pointCloud
        self.basePointCount = basePointCount
        self.frameDescriptors = frameDescriptors
        self.normalizationTranslation = normalizationTranslation
        self.normalizationScale = normalizationScale
    }
}

public enum FastGSScannerDatasetLoaderError: Error, Equatable {
    case missingPointCloud(URL)
    case noFramePairs(URL)
    case invalidFrameJSON(URL)
    case invalidIntrinsics(URL, count: Int)
    case invalidCameraPose(URL, count: Int)
    case invalidImage(URL)
    case cannotCreateImageContext(width: Int, height: Int)
    case invalidRenderedImageBuffer(width: Int, height: Int, actual: Int)
}

public enum FastGSScannerDatasetLoader {
    public static func loadCache(
        directory: URL,
        options: FastGSScannerDatasetOptions = FastGSScannerDatasetOptions()
    ) throws -> FastGSScannerDatasetCache {
        let pointsURL = directory.appendingPathComponent("points.ply")
        guard FileManager.default.fileExists(atPath: pointsURL.path) else {
            throw FastGSScannerDatasetLoaderError.missingPointCloud(pointsURL)
        }

        let framePairs = try collectFramePairs(
            directory: directory,
            options: FastGSScannerDatasetOptions(
                width: options.width,
                height: options.height,
                maxFrames: 0,
                frameStep: options.frameStep,
                startIndex: 0,
                normalizeWithAllFramePairs: false
            )
        )
        let rawPointCloud = try FastGSPLYReader.readPointCloud(url: pointsURL)
        let axis = axisTransform3x3()
        let normalization = try pointCloudNormalization(framePairs: framePairs, axis: axis)
        let pointCloud = normalizedPointCloud(
            rawPointCloud: rawPointCloud,
            axis: axis,
            normalization: normalization
        )

        return FastGSScannerDatasetCache(
            directory: directory,
            pointCloud: pointCloud,
            basePointCount: rawPointCloud.count,
            frameDescriptors: framePairs.map {
                FastGSScannerFrameDescriptor(index: $0.index, imageURL: $0.imageURL, jsonURL: $0.jsonURL)
            },
            normalizationTranslation: normalization.translation,
            normalizationScale: normalization.scale
        )
    }

    public static func loadDataset(
        cache: FastGSScannerDatasetCache,
        frameIndex: Int,
        width: Int,
        height: Int
    ) throws -> FastGSScannerDataset {
        let frame = try loadFrame(cache: cache, frameIndex: frameIndex, width: width, height: height)
        return FastGSScannerDataset(
            directory: cache.directory,
            pointCloud: cache.pointCloud,
            basePointCount: cache.basePointCount,
            frames: [frame],
            normalizationTranslation: cache.normalizationTranslation,
            normalizationScale: cache.normalizationScale
        )
    }

    public static func loadFrame(
        cache: FastGSScannerDatasetCache,
        frameIndex: Int,
        width: Int,
        height: Int
    ) throws -> FastGSScannerFrame {
        let descriptor = cache.frameDescriptors.first { $0.index == frameIndex }
            ?? cache.frameDescriptors[min(max(frameIndex, 0), max(cache.frameDescriptors.count - 1, 0))]
        let raw = try decodeScannerFrameMetadata(url: descriptor.jsonURL)
        guard raw.intrinsics.count == 9 else {
            throw FastGSScannerDatasetLoaderError.invalidIntrinsics(descriptor.jsonURL, count: raw.intrinsics.count)
        }
        guard raw.cameraPoseARFrame.count == 16 else {
            throw FastGSScannerDatasetLoaderError.invalidCameraPose(descriptor.jsonURL, count: raw.cameraPoseARFrame.count)
        }

        let axis = axisTransform3x3()
        let imageSize = try imageDimensions(url: descriptor.imageURL)
        var normalizedC2W = applyAxisTransform(axis, to4x4: raw.cameraPoseARFrame.map(Float.init))
        normalizedC2W[3] = (normalizedC2W[3] - cache.normalizationTranslation[0]) * cache.normalizationScale
        normalizedC2W[7] = (normalizedC2W[7] - cache.normalizationTranslation[1]) * cache.normalizationScale
        normalizedC2W[11] = (normalizedC2W[11] - cache.normalizationTranslation[2]) * cache.normalizationScale

        let camera = buildCamera(
            intrinsics: raw.intrinsics.map(Float.init),
            c2w: normalizedC2W,
            rawWidth: imageSize.width,
            rawHeight: imageSize.height,
            width: width,
            height: height
        )
        return FastGSScannerFrame(
            index: descriptor.index,
            imageURL: descriptor.imageURL,
            jsonURL: descriptor.jsonURL,
            camera: camera,
            targetCHW: try loadTargetImageCHW(url: descriptor.imageURL, width: width, height: height)
        )
    }

    public static func load(
        directory: URL,
        options: FastGSScannerDatasetOptions = FastGSScannerDatasetOptions()
    ) throws -> FastGSScannerDataset {
        let pointsURL = directory.appendingPathComponent("points.ply")
        guard FileManager.default.fileExists(atPath: pointsURL.path) else {
            throw FastGSScannerDatasetLoaderError.missingPointCloud(pointsURL)
        }

        let rawPointCloud = try FastGSPLYReader.readPointCloud(url: pointsURL)
        let framePairs = try collectFramePairs(directory: directory, options: options)
        let normalizationFramePairs = options.normalizeWithAllFramePairs
            ? try collectFramePairs(
                directory: directory,
                options: FastGSScannerDatasetOptions(
                    width: options.width,
                    height: options.height,
                    maxFrames: 0,
                    frameStep: options.frameStep,
                    startIndex: options.startIndex,
                    normalizeWithAllFramePairs: false
                )
            )
            : framePairs
        let axis = axisTransform3x3()
        let normalization = try pointCloudNormalization(framePairs: normalizationFramePairs, axis: axis)
        let pointCloud = normalizedPointCloud(rawPointCloud: rawPointCloud, axis: axis, normalization: normalization)

        let frames = try framePairs.map { pair in
            let raw = try decodeScannerFrameMetadata(url: pair.jsonURL)
            guard raw.intrinsics.count == 9 else {
                throw FastGSScannerDatasetLoaderError.invalidIntrinsics(pair.jsonURL, count: raw.intrinsics.count)
            }
            guard raw.cameraPoseARFrame.count == 16 else {
                throw FastGSScannerDatasetLoaderError.invalidCameraPose(pair.jsonURL, count: raw.cameraPoseARFrame.count)
            }
            let imageSize = try imageDimensions(url: pair.imageURL)
            var normalizedC2W = applyAxisTransform(axis, to4x4: raw.cameraPoseARFrame.map(Float.init))
            normalizedC2W[3] = (normalizedC2W[3] - normalization.translation[0]) * normalization.scale
            normalizedC2W[7] = (normalizedC2W[7] - normalization.translation[1]) * normalization.scale
            normalizedC2W[11] = (normalizedC2W[11] - normalization.translation[2]) * normalization.scale

            let camera = buildCamera(
                intrinsics: raw.intrinsics.map(Float.init),
                c2w: normalizedC2W,
                rawWidth: imageSize.width,
                rawHeight: imageSize.height,
                width: options.width,
                height: options.height
            )
            return FastGSScannerFrame(
                index: pair.index,
                imageURL: pair.imageURL,
                jsonURL: pair.jsonURL,
                camera: camera,
                targetCHW: try loadTargetImageCHW(url: pair.imageURL, width: options.width, height: options.height)
            )
        }

        return FastGSScannerDataset(
            directory: directory,
            pointCloud: pointCloud,
            basePointCount: rawPointCloud.count,
            frames: frames,
            normalizationTranslation: normalization.translation,
            normalizationScale: normalization.scale
        )
    }

    private struct FramePair {
        var index: Int
        var imageURL: URL
        var jsonURL: URL
    }

    private struct ScannerFrameMetadata: Decodable {
        var intrinsics: [Double]
        var cameraPoseARFrame: [Double]
    }

    private static func pointCloudNormalization(
        framePairs: [FramePair],
        axis: [Float]
    ) throws -> (translation: [Float], scale: Float) {
        var cameraPositions = [[Float]]()
        for pair in framePairs {
            let raw = try decodeScannerFrameMetadata(url: pair.jsonURL)
            guard raw.cameraPoseARFrame.count == 16 else {
                throw FastGSScannerDatasetLoaderError.invalidCameraPose(pair.jsonURL, count: raw.cameraPoseARFrame.count)
            }
            let c2wSource = raw.cameraPoseARFrame.map(Float.init)
            let c2w = applyAxisTransform(axis, to4x4: c2wSource)
            cameraPositions.append([c2w[3], c2w[7], c2w[11]])
        }
        return computeNormalization(cameraPositions: cameraPositions)
    }

    private static func normalizedPointCloud(
        rawPointCloud: FastGSPointCloud,
        axis: [Float],
        normalization: (translation: [Float], scale: Float)
    ) -> FastGSPointCloud {
        var transformedPoints = [Float]()
        transformedPoints.reserveCapacity(rawPointCloud.points.count)
        for index in stride(from: 0, to: rawPointCloud.points.count, by: 3) {
            let point = [
                rawPointCloud.points[index],
                rawPointCloud.points[index + 1],
                rawPointCloud.points[index + 2],
            ]
            transformedPoints.append(contentsOf: multiply(axis, point))
        }

        for index in stride(from: 0, to: transformedPoints.count, by: 3) {
            transformedPoints[index] = (transformedPoints[index] - normalization.translation[0]) * normalization.scale
            transformedPoints[index + 1] = (transformedPoints[index + 1] - normalization.translation[1]) * normalization.scale
            transformedPoints[index + 2] = (transformedPoints[index + 2] - normalization.translation[2]) * normalization.scale
        }

        return FastGSPointCloud(
            points: transformedPoints,
            colors: rawPointCloud.colors ?? Array(repeating: 0.5, count: rawPointCloud.count * 3),
            count: rawPointCloud.count
        )
    }

    private static func collectFramePairs(
        directory: URL,
        options: FastGSScannerDatasetOptions
    ) throws -> [FramePair] {
        let contents = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        var images = [Int: URL]()
        var jsons = [Int: URL]()
        for url in contents {
            guard let index = frameIndex(url) else {
                continue
            }
            if url.pathExtension.lowercased() == "jpg" {
                images[index] = url
            } else if url.pathExtension.lowercased() == "json" {
                jsons[index] = url
            }
        }

        var indices = Array(Set(images.keys).intersection(jsons.keys))
            .filter { $0 >= options.startIndex }
            .sorted()
        if options.frameStep > 1 {
            indices = indices.enumerated().compactMap { offset, index in
                offset % options.frameStep == 0 ? index : nil
            }
        }
        if options.maxFrames > 0 {
            indices = Array(indices.prefix(options.maxFrames))
        }

        let pairs = indices.compactMap { index -> FramePair? in
            guard let imageURL = images[index], let jsonURL = jsons[index] else {
                return nil
            }
            return FramePair(index: index, imageURL: imageURL, jsonURL: jsonURL)
        }
        guard !pairs.isEmpty else {
            throw FastGSScannerDatasetLoaderError.noFramePairs(directory)
        }
        return pairs
    }

    private static func frameIndex(_ url: URL) -> Int? {
        let stem = url.deletingPathExtension().lastPathComponent
        guard stem.hasPrefix("frame_") else {
            return nil
        }
        return Int(stem.dropFirst("frame_".count))
    }

    private static func decodeScannerFrameMetadata(url: URL) throws -> ScannerFrameMetadata {
        do {
            return try JSONDecoder().decode(ScannerFrameMetadata.self, from: Data(contentsOf: url))
        } catch {
            throw FastGSScannerDatasetLoaderError.invalidFrameJSON(url)
        }
    }

    private static func axisTransform3x3() -> [Float] {
        [
            1, 0, 0,
            0, 0, 1,
            0, -1, 0,
        ]
    }

    private static func applyAxisTransform(_ axis: [Float], to4x4 matrix: [Float]) -> [Float] {
        var result = matrix
        for row in 0..<3 {
            for col in 0..<4 {
                result[row * 4 + col] =
                    axis[row * 3] * matrix[col] +
                    axis[row * 3 + 1] * matrix[4 + col] +
                    axis[row * 3 + 2] * matrix[8 + col]
            }
        }
        return result
    }

    private static func computeNormalization(cameraPositions: [[Float]]) -> (translation: [Float], scale: Float) {
        var translation = [Float](repeating: 0, count: 3)
        for position in cameraPositions {
            translation[0] += position[0]
            translation[1] += position[1]
            translation[2] += position[2]
        }
        let invCount = 1 / Float(cameraPositions.count)
        translation[0] *= invCount
        translation[1] *= invCount
        translation[2] *= invCount

        var maxAbs: Float = 0
        for position in cameraPositions {
            maxAbs = max(maxAbs, abs(position[0] - translation[0]))
            maxAbs = max(maxAbs, abs(position[1] - translation[1]))
            maxAbs = max(maxAbs, abs(position[2] - translation[2]))
        }
        return (translation, maxAbs > 0 ? 1 / maxAbs : 1)
    }

    private static func buildCamera(
        intrinsics: [Float],
        c2w: [Float],
        rawWidth: Int,
        rawHeight: Int,
        width: Int,
        height: Int,
        znear: Float = 0.001,
        zfar: Float = 1000
    ) -> FastGSScannerCamera {
        let sx = Float(width) / Float(rawWidth)
        let sy = Float(height) / Float(rawHeight)
        let fx = intrinsics[0] * sx
        let fy = intrinsics[4] * sy

        let r = [
            c2w[0], -c2w[1], -c2w[2],
            c2w[4], -c2w[5], -c2w[6],
            c2w[8], -c2w[9], -c2w[10],
        ]
        let t = [c2w[3], c2w[7], c2w[11]]
        let rinv = transpose3x3(r)
        let tinv = multiply(rinv, t).map { -$0 }

        var view = identity4x4()
        for row in 0..<3 {
            for col in 0..<3 {
                view[row * 4 + col] = rinv[row * 3 + col]
            }
            view[row * 4 + 3] = tinv[row]
        }

        let fovX = 2 * atan(Float(width) / (2 * fx))
        let fovY = 2 * atan(Float(height) / (2 * fy))
        let top = znear * tan(0.5 * fovY)
        let bottom = -top
        let right = znear * tan(0.5 * fovX)
        let left = -right

        let projection = [
            2 * znear / (right - left), 0, (right + left) / (right - left), 0,
            0, 2 * znear / (top - bottom), (top + bottom) / (top - bottom), 0,
            0, 0, (zfar + znear) / (zfar - znear), -(zfar * znear) / (zfar - znear),
            0, 0, 1, 0,
        ]
        let fullProjection = multiply4x4(projection, view)

        return FastGSScannerCamera(
            viewmatrix: transpose4x4(view),
            projmatrix: transpose4x4(fullProjection),
            campos: t,
            imageWidth: width,
            imageHeight: height,
            tanFovX: tan(0.5 * fovX),
            tanFovY: tan(0.5 * fovY)
        )
    }

    private static func imageDimensions(url: URL) throws -> (width: Int, height: Int) {
        guard
            let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
            let width = properties[kCGImagePropertyPixelWidth] as? Int,
            let height = properties[kCGImagePropertyPixelHeight] as? Int
        else {
            throw FastGSScannerDatasetLoaderError.invalidImage(url)
        }
        return (width, height)
    }

    private static func loadTargetImageCHW(url: URL, width: Int, height: Int) throws -> [Float] {
        guard
            let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw FastGSScannerDatasetLoaderError.invalidImage(url)
        }

        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var rgba = [UInt8](repeating: 0, count: height * bytesPerRow)
        guard
            let context = CGContext(
                data: &rgba,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            )
        else {
            throw FastGSScannerDatasetLoaderError.cannotCreateImageContext(width: width, height: height)
        }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        guard rgba.count == height * width * 4 else {
            throw FastGSScannerDatasetLoaderError.invalidRenderedImageBuffer(
                width: width,
                height: height,
                actual: rgba.count
            )
        }

        var chw = [Float](repeating: 0, count: 3 * width * height)
        for pixel in 0..<(width * height) {
            let base = pixel * 4
            let alpha = Float(rgba[base + 3]) / 255
            let red = Float(rgba[base]) / 255
            let green = Float(rgba[base + 1]) / 255
            let blue = Float(rgba[base + 2]) / 255
            chw[pixel] = red * alpha + (1 - alpha)
            chw[width * height + pixel] = green * alpha + (1 - alpha)
            chw[2 * width * height + pixel] = blue * alpha + (1 - alpha)
        }
        return chw
    }

    private static func multiply(_ matrix: [Float], _ vector: [Float]) -> [Float] {
        [
            matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
            matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2],
            matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2],
        ]
    }

    private static func transpose3x3(_ matrix: [Float]) -> [Float] {
        [
            matrix[0], matrix[3], matrix[6],
            matrix[1], matrix[4], matrix[7],
            matrix[2], matrix[5], matrix[8],
        ]
    }

    private static func identity4x4() -> [Float] {
        [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ]
    }

    private static func transpose4x4(_ matrix: [Float]) -> [Float] {
        [
            matrix[0], matrix[4], matrix[8], matrix[12],
            matrix[1], matrix[5], matrix[9], matrix[13],
            matrix[2], matrix[6], matrix[10], matrix[14],
            matrix[3], matrix[7], matrix[11], matrix[15],
        ]
    }

    private static func multiply4x4(_ lhs: [Float], _ rhs: [Float]) -> [Float] {
        var result = [Float](repeating: 0, count: 16)
        for row in 0..<4 {
            for col in 0..<4 {
                var value: Float = 0
                for k in 0..<4 {
                    value += lhs[row * 4 + k] * rhs[k * 4 + col]
                }
                result[row * 4 + col] = value
            }
        }
        return result
    }
}
