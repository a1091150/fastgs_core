import Foundation
import MLX

public struct FastGSRecordedTrainingReferenceSet {
    public var referenceDirectory: URL
    public var manifests: [URL]

    public init(referenceDirectory: URL, manifests: [URL]) {
        self.referenceDirectory = referenceDirectory
        self.manifests = manifests
    }

    public init(referenceDirectory: URL) {
        self.referenceDirectory = referenceDirectory

        let fileManager = FileManager.default
        let contents = (try? fileManager.contentsOfDirectory(
            at: referenceDirectory,
            includingPropertiesForKeys: nil
        )) ?? []
        let perCameraURLs = contents
            .filter { $0.lastPathComponent.hasPrefix("camera_") }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            .map { $0.appendingPathComponent("recorded_manifest.json") }
            .filter { fileManager.fileExists(atPath: $0.path) }

        if !perCameraURLs.isEmpty {
            self.manifests = perCameraURLs
        } else {
            let rootManifest = referenceDirectory.appendingPathComponent("recorded_manifest.json")
            self.manifests = fileManager.fileExists(atPath: rootManifest.path) ? [rootManifest] : []
        }
    }

    public var count: Int {
        manifests.count
    }

    public var isEmpty: Bool {
        manifests.isEmpty
    }

    public func clampedIndex(_ index: Int) -> Int {
        guard !manifests.isEmpty else {
            return 0
        }
        return min(max(index, 0), manifests.count - 1)
    }

    public func wrappingIndex(_ index: Int) -> Int {
        guard !manifests.isEmpty else {
            return 0
        }
        return (index % manifests.count + manifests.count) % manifests.count
    }

    public func manifestURL(at index: Int) -> URL? {
        guard !manifests.isEmpty else {
            return nil
        }
        return manifests[clampedIndex(index)]
    }
}

public struct FastGSRecordedTrainingRunConfig {
    public var referenceSet: FastGSRecordedTrainingReferenceSet
    public var totalSteps: Int
    public var cacheLimitBytes: Int
    public var learningRates: FastGSAdamLearningRates

    public init(
        referenceSet: FastGSRecordedTrainingReferenceSet = FastGSRecordedTrainingReferenceSet(
            referenceDirectory: URL(fileURLWithPath: "/private/tmp/fastgs_recorded_reference_full_512", isDirectory: true)
        ),
        totalSteps: Int = 200,
        cacheLimitBytes: Int = 4 * 1024 * 1024 * 1024,
        learningRates: FastGSAdamLearningRates = FastGSAdamLearningRates(
            means3D: 5e-5,
            dc: 5e-4,
            sh: 5e-4,
            opacities: 5e-4,
            scales: 5e-5,
            rotations: 5e-5
        )
    ) {
        self.referenceSet = referenceSet
        self.totalSteps = totalSteps
        self.cacheLimitBytes = cacheLimitBytes
        self.learningRates = learningRates
    }
}

public typealias FastGSRecordedTrainingPreviewConfig = FastGSRecordedTrainingRunConfig

public struct FastGSRecordedTrainingPreviewResult {
    public var targetRGBA: [UInt8]
    public var renderRGBA: [UInt8]
    public var width: Int
    public var height: Int
    public var pointCount: Int

    public init(
        targetRGBA: [UInt8],
        renderRGBA: [UInt8],
        width: Int,
        height: Int,
        pointCount: Int
    ) {
        self.targetRGBA = targetRGBA
        self.renderRGBA = renderRGBA
        self.width = width
        self.height = height
        self.pointCount = pointCount
    }
}

public enum FastGSRecordedTrainingPreview {
    public static func run(
        config: FastGSRecordedTrainingRunConfig = FastGSRecordedTrainingRunConfig(),
        cameraIndex: Int,
        progress: ((Int) -> Void)? = nil
    ) throws -> FastGSRecordedTrainingPreviewResult {
        guard let manifestURL = config.referenceSet.manifestURL(at: cameraIndex) else {
            throw FastGSRecordedTrainingPreviewError.noRecordedReference(config.referenceSet.referenceDirectory)
        }
        return try run(manifestURL: manifestURL, config: config, progress: progress)
    }

    public static func run(
        manifestURL: URL,
        config: FastGSRecordedTrainingRunConfig = FastGSRecordedTrainingRunConfig(),
        progress: ((Int) -> Void)? = nil
    ) throws -> FastGSRecordedTrainingPreviewResult {
        Memory.cacheLimit = config.cacheLimitBytes

        let scene = try FastGSRecordedForwardScene(manifestURL: manifestURL)
        let target = try scene.targetOutColor()
        var parameters = try scene.initialTrainableParameters()
        var optimizer = FastGSAdamOptimizer(learningRates: config.learningRates)

        for step in 1...config.totalSteps {
            let result = FastGSTrainingStageGraph.valueAndGrad(
                scene: scene,
                parameters: parameters,
                target: target
            )
            parameters = optimizer.update(
                parameters: parameters,
                gradients: trainableGradients(from: result.gradients)
            )
            eval(parameters: parameters, optimizer: optimizer)
            progress?(step)
        }

        let render = FastGSTrainingStageGraph.render(scene: scene, parameters: parameters)
        return FastGSRecordedTrainingPreviewResult(
            targetRGBA: FastGSImageExport.rgbaBytes(
                outColor: target,
                width: scene.manifest.width,
                height: scene.manifest.height
            ),
            renderRGBA: FastGSImageExport.rgbaBytes(
                outColor: render,
                width: scene.manifest.width,
                height: scene.manifest.height
            ),
            width: scene.manifest.width,
            height: scene.manifest.height,
            pointCount: scene.manifest.pointCount
        )
    }

    private static func trainableGradients(from gradients: [MLXArray]) -> FastGSTrainableGradients {
        precondition(gradients.count == 6, "recorded training preview expects six trainable gradients")
        return FastGSTrainableGradients(
            means3D: gradients[0],
            dc: gradients[1],
            sh: gradients[2],
            opacities: gradients[3],
            scales: gradients[4],
            rotations: gradients[5]
        )
    }

    private static func eval(parameters: FastGSTrainableParameters, optimizer: FastGSAdamOptimizer) {
        for array in parameters.arrays {
            array.eval()
        }
        for array in optimizer.stateArrays() {
            array.eval()
        }
    }
}

public enum FastGSRecordedTrainingPreviewError: Error, Equatable {
    case noRecordedReference(URL)
}
