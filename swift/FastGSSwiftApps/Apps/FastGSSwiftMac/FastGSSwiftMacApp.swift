import FastGSSwift
import AppKit
import CoreImage
import Metal
import MetalKit
import MLX
import SwiftUI

@main
struct FastGSSwiftMacApp: App {
    var body: some Scene {
        WindowGroup {
            RenderPreviewView()
                .frame(minWidth: 720, minHeight: 520)
        }
    }
}

@MainActor
private final class RenderPreviewModel: ObservableObject {
    private var trainingConfig = FastGSRecordedTrainingRunConfig()
    private var datasetDirectory = URL(fileURLWithPath: "/Users/yangdunfu/Downloads/2026_05_04_16_51_29", isDirectory: true)
    private var outputDirectory = URL(fileURLWithPath: "/private/tmp/fastgs_swift_mac_training", isDirectory: true)
    @Published var texture: MTLTexture?
    @Published var fallbackImage: CGImage?
    @Published var targetImage: CGImage?
    @Published var status = "Ready"
    @Published var trainingStep = 0
    @Published var totalTrainingSteps = 0
    @Published var cameraIndex = 0
    @Published var renderSize = "512 x 512"
    @Published var previewAspectRatio = 1.0
    @Published var previewMode: RenderPreviewMode = .single
    @Published var isTraining = false
    @Published var datasetLabel = "2026_05_04_16_51_29"
    @Published var cameraCount = 1
    @Published var outputLabel = "/private/tmp/fastgs_swift_mac_training"
    @Published var trainingWidth = 512
    @Published var trainingHeight = 512
    @Published var maxFrames = 9999
    @Published var trainingSteps = 200
    @Published var isDatasetLoaded = false
    @Published var isLoadingDataset = false
    @Published var isRenderingPreview = false
    @Published var saveTrainingArtifacts = false
    private var frameIndices = [Int]()
    private let previewScheduler = FastGSRenderPreviewScheduler(maximumFramesPerSecond: 60)
    private var scannerCache: FastGSScannerDatasetCache?
    private let trainingPreviewRequests = FastGSMacTrainingPreviewRequests()
    private var trainedParameters: FastGSTrainableParameters?
    private var trainedOptimizerState: FastGSAdamState?
    private var trainedDensificationState: FastGSDensificationState?
    let device = MTLCreateSystemDefaultDevice()

    init() {
        refreshOutputLabel()
        refreshTrainingConfig()
        totalTrainingSteps = trainingConfig.totalSteps
        cameraCount = 0
        status = "Press Load to read scanner frames"
    }

    var cameraLabel: String {
        let count = max(cameraCount, 1)
        if frameIndices.indices.contains(cameraIndex) {
            return "Camera \(cameraIndex + 1) / \(count)  frame_\(String(format: "%05d", frameIndices[cameraIndex]))"
        }
        return "Camera \(min(cameraIndex + 1, count)) / \(count)"
    }

    func trainRecordedFrame(mode: FastGSMacTrainingStartMode) {
        guard !isTraining else {
            return
        }
        guard isDatasetLoaded else {
            status = "Press Load before training"
            return
        }

        isTraining = true
        trainingStep = 0
        refreshTrainingConfig()
        totalTrainingSteps = trainingConfig.totalSteps
        guard cameraCount > 0 else {
            status = "Training failed: choose a folder with frame_*.jpg/json and points.ply"
            isTraining = false
            return
        }
        guard let scannerCache else {
            status = "Training failed: press Load to read scanner frames"
            isTraining = false
            return
        }
        let config = trainingConfig
        let trainingWidth = max(16, trainingWidth)
        let trainingHeight = max(16, trainingHeight)
        let maxFrames = max(1, maxFrames)
        let selectedFrameIndex = frameIndices.indices.contains(cameraIndex) ? frameIndices[cameraIndex] : cameraIndex
        let trainingFrameCount = min(maxFrames, scannerCache.frameDescriptors.count)
        let datasetDirectoryPath = datasetDirectory.path
        let shouldSaveTrainingArtifacts = saveTrainingArtifacts
        let runDirectory = shouldSaveTrainingArtifacts ? outputDirectory.appendingPathComponent(timestampedTrainingRunName(), isDirectory: true) : nil
        status = "\(mode.statusVerb) \(trainingFrameCount) frames..."

        Task {
            do {
                guard let device else {
                    status = "Training failed: no Metal device"
                    isTraining = false
                    return
                }
                let outputDirectory = outputDirectory
                let runDirectory = runDirectory
                let trainingWidth = trainingWidth
                let trainingHeight = trainingHeight
                let maxFrames = maxFrames
                let selectedFrameIndex = selectedFrameIndex
                let previewScheduler = previewScheduler
                let scannerCache = scannerCache
                let trainingPreviewRequests = trainingPreviewRequests
                let shouldSaveTrainingArtifacts = shouldSaveTrainingArtifacts
                let inMemoryInitialParameters = mode.usesCurrentParameters ? trainedParameters : nil
                let inMemoryInitialOptimizerState = mode.usesCurrentParameters ? trainedOptimizerState : nil
                let inMemoryInitialDensificationState = mode.usesCurrentParameters ? trainedDensificationState : nil

                let trained = try await Task.detached(priority: .userInitiated) {
                    try FastGSMacMLXRuntime.run {
                        let initialParameters: FastGSTrainableParameters?
                        let initialOptimizerState: FastGSAdamState?
                        let initialDensificationState: FastGSDensificationState?
                        if mode.usesCurrentParameters {
                            if let inMemoryInitialParameters {
                                initialParameters = inMemoryInitialParameters
                                initialOptimizerState = inMemoryInitialOptimizerState
                                initialDensificationState = inMemoryInitialDensificationState
                            } else if let checkpointDirectory = latestCheckpointDirectory(in: outputDirectory) {
                                let checkpoint = try FastGSCheckpoint.loadTrainingState(directory: checkpointDirectory)
                                initialParameters = checkpoint.parameters
                                initialOptimizerState = checkpoint.optimizerState
                                initialDensificationState = checkpoint.densificationState
                            } else {
                                throw RenderPreviewError.missingCheckpoint(outputDirectory)
                            }
                        } else {
                            initialParameters = nil
                            initialOptimizerState = nil
                            initialDensificationState = nil
                        }
                        if let runDirectory {
                            try FileManager.default.createDirectory(at: runDirectory, withIntermediateDirectories: true)
                        }
                        let progress: (Int) -> Void = { step in
                            Task { @MainActor in
                                self.trainingStep = step
                                self.status = "\(mode.statusVerb) \(trainingFrameCount) frames..."
                            }
                        }
                        let pruneSummary: (FastGSRecordedTrainingPruneSummary) -> Void = { summary in
                            Task { @MainActor in
                                self.status = "\(summary.reason) step \(summary.step): \(summary.beforeCount) -> \(summary.afterCount), clone \(summary.clonedCount), split \(summary.splitChildCount), score \(summary.scoreHits), prune \(summary.prunedCount)"
                            }
                        }
                        let preview: (FastGSRecordedTrainingPreviewResult) throws -> Void = { preview in
                            guard shouldSaveTrainingArtifacts, let runDirectory else {
                                return
                            }
                            let frameIndex = preview.frameIndex ?? selectedFrameIndex
                            try writeSideBySidePNG(
                                targetRGBA: preview.targetRGBA,
                                renderRGBA: preview.renderRGBA,
                                width: preview.width,
                                height: preview.height,
                                to: trainingOutputURL(
                                    outputDirectory: runDirectory,
                                    cameraIndex: frameIndex,
                                    step: preview.step
                                )
                            )
                        }
                        let scheduledPreview: (Int, FastGSTrainableParameters) throws -> FastGSRecordedTrainingPreviewResult? = { step, parameters in
                            guard
                                let frameIndex = trainingPreviewRequests.consumeFrameIndex()
                            else {
                                return nil
                            }
                            let dataset = try FastGSScannerDatasetLoader.loadDataset(
                                cache: scannerCache,
                                frameIndex: frameIndex,
                                width: trainingWidth,
                                height: trainingHeight
                            )
                            let result = try trainingPreviewResult(
                                dataset: dataset,
                                step: step,
                                parameters: parameters,
                                width: trainingWidth,
                                height: trainingHeight
                            )
                            let targetImage = try FastGSImageExport.cgImage(
                                rgbaBytes: result.targetRGBA,
                                width: result.width,
                                height: result.height
                            )
                            let renderImage = try FastGSImageExport.cgImage(
                                rgbaBytes: result.renderRGBA,
                                width: result.width,
                                height: result.height
                            )
                            Task { @MainActor in
                                self.targetImage = targetImage
                                self.texture = nil
                                self.fallbackImage = renderImage
                                self.renderSize = "\(result.width) x \(result.height)"
                                self.previewAspectRatio = Double(result.width) / Double(result.height)
                                self.previewMode = .recordedSideBySide
                                self.status = "Rendered preview \(self.cameraLabel) at step \(step)"
                            }
                            return nil
                        }
                        let scenes = try trainingScenes(
                            cache: scannerCache,
                            width: trainingWidth,
                            height: trainingHeight,
                            maxFrames: maxFrames
                        )
                        let result = try FastGSRecordedTrainingPreview.run(
                            scenes: scenes,
                            config: config,
                            initialParameters: initialParameters,
                            initialOptimizerState: initialOptimizerState,
                            initialDensificationState: initialDensificationState,
                            progress: progress,
                            pruneSummary: pruneSummary,
                            previewScheduler: previewScheduler,
                            scheduledPreview: scheduledPreview,
                            preview: preview
                        )

                        guard let texture = FastGSImageExport.texture(
                            rgbaBytes: result.renderRGBA,
                            width: result.width,
                            height: result.height,
                            device: device
                        ) else {
                            throw RenderPreviewError.textureCreationFailed
                        }
                        let image = try FastGSImageExport.cgImage(
                            rgbaBytes: result.renderRGBA,
                            width: result.width,
                            height: result.height
                        )
                        let targetImage = try FastGSImageExport.cgImage(
                            rgbaBytes: result.targetRGBA,
                            width: result.width,
                            height: result.height
                        )
                        let checkpointDirectory = runDirectory?.appendingPathComponent("checkpoint", isDirectory: true)
                        if let parameters = result.parameters {
                            if shouldSaveTrainingArtifacts, let checkpointDirectory, let runDirectory {
                                let info = FastGSTrainingCheckpointInfo(
                                    datasetDirectory: datasetDirectoryPath,
                                    outputDirectory: runDirectory.path,
                                    imageWidth: result.width,
                                    imageHeight: result.height,
                                    maxFrames: maxFrames,
                                    trainingSteps: config.totalSteps,
                                    completedStep: result.step,
                                    frameCount: trainingFrameCount,
                                    pointCount: result.pointCount,
                                    gaussianCount: result.parameters?.gaussianCount ?? result.pointCount,
                                    afterTrainingConfig: config.densification,
                                    note: mode.metadataName
                                )
                                try FastGSCheckpoint.save(
                                    parameters: parameters,
                                    info: info,
                                    optimizerState: result.optimizerState,
                                    densificationState: result.densificationState,
                                    directory: checkpointDirectory
                                )
                                try writeTrainingRunMetadata(
                                    mode: mode,
                                    info: info,
                                    runDirectory: runDirectory,
                                    checkpointDirectory: checkpointDirectory
                                )
                            }
                        }
                        return (
                            texture,
                            image,
                            targetImage,
                            result.width,
                            result.height,
                            result.pointCount,
                            result.parameters,
                            result.optimizerState,
                            result.densificationState,
                            runDirectory
                        )
                    }
                }.value

                texture = trained.0
                fallbackImage = trained.1
                targetImage = trained.2
                renderSize = "\(trained.3) x \(trained.4)"
                previewAspectRatio = Double(trained.3) / Double(trained.4)
                previewMode = .recordedSideBySide
                trainedParameters = trained.6
                trainedOptimizerState = trained.7
                trainedDensificationState = trained.8
                if let runDirectory = trained.9 {
                    status = "\(mode.completedVerb) on \(trainingFrameCount) frames, wrote artifacts to \(runDirectory.path)"
                } else {
                    status = "\(mode.completedVerb) on \(trainingFrameCount) frames"
                }
            } catch {
                status = "Training failed: \(error)"
            }
            isTraining = false
        }
    }

    func selectCamera(delta: Int) {
        guard isDatasetLoaded else {
            status = "Press Load before switching cameras"
            return
        }
        guard !isRenderingPreview else {
            status = "Rendering \(cameraLabel)..."
            return
        }
        let count = cameraCount
        guard count > 0 else {
            status = "Choose a scanner dataset folder"
            return
        }
        cameraIndex = (cameraIndex + delta + count) % count
        trainingStep = 0
        totalTrainingSteps = trainingConfig.totalSteps
        status = "Selected \(cameraLabel)"
        if isTraining {
            let selectedFrameIndex = frameIndices.indices.contains(cameraIndex) ? frameIndices[cameraIndex] : cameraIndex
            trainingPreviewRequests.requestFrameIndex(selectedFrameIndex)
            previewScheduler.requestRender()
            status = "Queued preview \(cameraLabel)"
        } else {
            loadSelectedTargetPreview()
        }
    }

    func chooseDatasetDirectory() {
        guard !isTraining else {
            return
        }

        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.directoryURL = datasetDirectory
        panel.message = "Choose a scanner dataset folder containing points.ply and frame_*.jpg/json"

        if panel.runModal() == .OK, let url = panel.url {
            datasetDirectory = url
            cameraIndex = 0
            resetLoadedDataset()
            datasetLabel = datasetDirectory.lastPathComponent
            status = "Press Load to read \(datasetLabel)"
        }
    }

    func chooseOutputDirectory() {
        guard !isTraining else {
            return
        }

        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.directoryURL = outputDirectory
        panel.message = "Choose where training preview PNGs should be written"

        if panel.runModal() == .OK, let url = panel.url {
            outputDirectory = url
            refreshOutputLabel()
            status = "Selected output \(outputLabel)"
        }
    }

    func applySettings() {
        trainingWidth = max(16, trainingWidth)
        trainingHeight = max(16, trainingHeight)
        maxFrames = max(1, maxFrames)
        trainingSteps = max(1, trainingSteps)
        refreshTrainingConfig()
        trainedParameters = nil
        trainedOptimizerState = nil
        trainedDensificationState = nil
        totalTrainingSteps = trainingConfig.totalSteps
        renderSize = "\(trainingWidth) x \(trainingHeight)"
        if isDatasetLoaded {
            status = "Updated training settings"
            loadSelectedTargetPreview()
        } else {
            status = "Updated settings. Press Load to read scanner frames"
        }
    }

    func loadDataset() {
        guard !isTraining, !isLoadingDataset else {
            return
        }

        resetLoadedDataset(clearStatus: false)
        trainedParameters = nil
        trainedOptimizerState = nil
        trainedDensificationState = nil
        isLoadingDataset = true
        datasetLabel = datasetDirectory.lastPathComponent
        status = "Loading scanner frames and initial render..."
        let datasetDirectory = datasetDirectory
        let width = max(16, trainingWidth)
        let height = max(16, trainingHeight)
        let maxFrames = max(1, maxFrames)
        let device = device

        Task {
            do {
                let loaded = try await Task.detached(priority: .userInitiated) {
                    try FastGSMacMLXRuntime.run {
                        let cache = try FastGSScannerDatasetLoader.loadCache(
                            directory: datasetDirectory,
                            options: FastGSScannerDatasetOptions(
                                width: width,
                                height: height,
                                maxFrames: maxFrames,
                                normalizeWithAllFramePairs: true
                            )
                        )
                        let indices = cache.frameDescriptors.map(\.index)
                        guard !indices.isEmpty else {
                            throw RenderPreviewError.noFramePairs
                        }
                        let firstIndex = indices[0]
                        let preview = try initialRenderPreview(
                            cache: cache,
                            frameIndex: firstIndex,
                            width: width,
                            height: height,
                            maxFrames: maxFrames,
                            device: device
                        )
                        // Uncomment while profiling if MLX cache growth should be reclaimed after each preview.
                        // Memory.clearCache()
                        return (cache, indices, preview)
                    }
                }.value

                scannerCache = loaded.0
                frameIndices = loaded.1
                cameraCount = loaded.1.count
                cameraIndex = 0
                targetImage = loaded.2.target
                texture = loaded.2.renderTexture
                fallbackImage = loaded.2.render
                renderSize = "\(width) x \(height)"
                previewAspectRatio = Double(width) / Double(height)
                previewMode = .recordedSideBySide
                isDatasetLoaded = true
                status = "Loaded \(cameraCount) scanner frames with initial render"
            } catch {
                resetLoadedDataset(clearStatus: false)
                status = "Load failed: \(error)"
            }
            isLoadingDataset = false
        }
    }

    private func refreshDataset() {
        datasetLabel = datasetDirectory.lastPathComponent
        frameIndices = scannerFramePairIndices(directory: datasetDirectory)
        cameraCount = frameIndices.count
        if cameraCount > 0 {
            cameraIndex = min(max(cameraIndex, 0), cameraCount - 1)
        } else {
            cameraIndex = 0
        }
    }

    private func refreshOutputLabel() {
        outputLabel = outputDirectory.path
    }

    private func resetLoadedDataset(clearStatus: Bool = true) {
        isDatasetLoaded = false
        frameIndices = []
        cameraCount = 0
        cameraIndex = 0
        targetImage = nil
        texture = nil
        fallbackImage = nil
        scannerCache = nil
        trainedParameters = nil
        trainedOptimizerState = nil
        trainedDensificationState = nil
        previewMode = .single
        if clearStatus {
            status = "Press Load to read scanner frames"
        }
    }

    private func refreshTrainingConfig() {
        trainingConfig.totalSteps = max(1, trainingSteps)
    }

    private func loadSelectedTargetPreview() {
        guard !isTraining, isDatasetLoaded, cameraCount > 0 else {
            return
        }
        guard !isRenderingPreview else {
            status = "Rendering \(cameraLabel)..."
            return
        }
        let selectedFrameIndex = frameIndices.indices.contains(cameraIndex) ? frameIndices[cameraIndex] : cameraIndex
        guard let scannerCache else {
            status = "Press Load to read scanner frames"
            return
        }
        let width = max(16, trainingWidth)
        let height = max(16, trainingHeight)
        let maxFrames = max(1, maxFrames)
        let trainedParameters = trainedParameters
        let device = device
        isRenderingPreview = true
        status = "Rendering \(cameraLabel)..."

        Task {
            do {
                let preview = try await Task.detached(priority: .userInitiated) {
                    try FastGSMacMLXRuntime.run {
                        let preview = try initialRenderPreview(
                            cache: scannerCache,
                            frameIndex: selectedFrameIndex,
                            width: width,
                            height: height,
                            maxFrames: maxFrames,
                            parameters: trainedParameters,
                            device: device
                        )
                        // Uncomment while profiling if MLX cache growth should be reclaimed after each preview.
                        // Memory.clearCache()
                        return preview
                    }
                }.value

                targetImage = preview.target
                texture = preview.renderTexture
                fallbackImage = preview.render
                renderSize = "\(width) x \(height)"
                previewAspectRatio = Double(width) / Double(height)
                previewMode = .recordedSideBySide
                status = "Rendered \(cameraLabel)"
            } catch {
                status = "Render preview failed: \(error)"
            }
            isRenderingPreview = false
        }
    }
}

private enum RenderPreviewMode {
    case single
    case recordedSideBySide
}

private enum FastGSMacTrainingStartMode: String, Sendable {
    case fromInitial
    case continueTraining

    var usesCurrentParameters: Bool {
        self == .continueTraining
    }

    var statusVerb: String {
        switch self {
        case .fromInitial:
            "Training"
        case .continueTraining:
            "Continuing training"
        }
    }

    var completedVerb: String {
        switch self {
        case .fromInitial:
            "Training completed"
        case .continueTraining:
            "Continued training completed"
        }
    }

    var metadataName: String {
        switch self {
        case .fromInitial:
            "train_from_initial"
        case .continueTraining:
            "continue_training"
        }
    }
}

private enum RenderPreviewError: Error {
    case textureCreationFailed
    case missingPointCloud
    case noFramePairs
    case missingFrameImage(Int)
    case missingCheckpoint(URL)
}

private enum FastGSMacMLXRuntime {
    private static let lock = NSLock()

    static func run<T>(_ work: () throws -> T) rethrows -> T {
        try lock.withLock {
            try work()
        }
    }
}

private final class FastGSMacTrainingPreviewRequests: @unchecked Sendable {
    private let lock = NSLock()
    private var pendingFrameIndex: Int?

    func requestFrameIndex(_ frameIndex: Int) {
        lock.withLock {
            pendingFrameIndex = frameIndex
        }
    }

    func consumeFrameIndex() -> Int? {
        lock.withLock {
            let frameIndex = pendingFrameIndex
            pendingFrameIndex = nil
            return frameIndex
        }
    }
}

private func trainingOutputURL(outputDirectory: URL, cameraIndex: Int, step: Int) -> URL {
    outputDirectory.appendingPathComponent(
        String(format: "camera_%03d_step_%03d_sbs.png", cameraIndex, step)
    )
}

private func timestampedTrainingRunName(date: Date = Date()) -> String {
    let formatter = DateFormatter()
    formatter.locale = Locale(identifier: "en_US_POSIX")
    formatter.dateFormat = "yyyyMMdd_HHmm"
    return formatter.string(from: date)
}

private func latestCheckpointDirectory(in outputDirectory: URL) -> URL? {
    let fileManager = FileManager.default
    let directCheckpoint = outputDirectory.appendingPathComponent("checkpoint", isDirectory: true)
    if fileManager.fileExists(atPath: FastGSCheckpoint.parameterURL(in: directCheckpoint).path) {
        return directCheckpoint
    }

    guard let contents = try? fileManager.contentsOfDirectory(
        at: outputDirectory,
        includingPropertiesForKeys: [.isDirectoryKey],
        options: [.skipsHiddenFiles]
    ) else {
        return nil
    }

    return contents
        .filter { url in
            guard let values = try? url.resourceValues(forKeys: [.isDirectoryKey]), values.isDirectory == true else {
                return false
            }
            let checkpoint = url.appendingPathComponent("checkpoint", isDirectory: true)
            return fileManager.fileExists(atPath: FastGSCheckpoint.parameterURL(in: checkpoint).path)
        }
        .sorted { $0.lastPathComponent > $1.lastPathComponent }
        .first?
        .appendingPathComponent("checkpoint", isDirectory: true)
}

private func writeTrainingRunMetadata(
    mode: FastGSMacTrainingStartMode,
    info: FastGSTrainingCheckpointInfo,
    runDirectory: URL,
    checkpointDirectory: URL
) throws {
    let metadata: [String: String] = [
        "mode": mode.metadataName,
        "createdAt": info.createdAt,
        "datasetDirectory": info.datasetDirectory,
        "outputDirectory": info.outputDirectory ?? runDirectory.path,
        "checkpointDirectory": checkpointDirectory.path,
        "parameterFile": FastGSCheckpoint.parameterURL(in: checkpointDirectory).path,
        "optimizerFile": info.optimizerFile.map { checkpointDirectory.appendingPathComponent($0).path } ?? "",
        "densificationStateFile": info.densificationStateFile.map { checkpointDirectory.appendingPathComponent($0).path } ?? "",
    ]
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(metadata).write(
        to: runDirectory.appendingPathComponent("training_run.json", isDirectory: false),
        options: .atomic
    )
}

private func writeSideBySidePNG(
    targetRGBA: [UInt8],
    renderRGBA: [UInt8],
    width: Int,
    height: Int,
    to url: URL
) throws {
    try FileManager.default.createDirectory(
        at: url.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )

    var combined = [UInt8](repeating: 0, count: width * 2 * height * 4)
    for row in 0..<height {
        let sourceStart = row * width * 4
        let sourceEnd = sourceStart + width * 4
        let targetStart = row * width * 2 * 4
        combined.replaceSubrange(targetStart..<(targetStart + width * 4), with: targetRGBA[sourceStart..<sourceEnd])
        combined.replaceSubrange((targetStart + width * 4)..<(targetStart + width * 2 * 4), with: renderRGBA[sourceStart..<sourceEnd])
    }
    try FastGSImageExport.writePNG(rgbaBytes: combined, width: width * 2, height: height, to: url)
}

private func rgbaBytes(chw: [Float], width: Int, height: Int) -> [UInt8] {
    let pixels = width * height
    var rgba = [UInt8](repeating: 255, count: pixels * 4)
    guard chw.count >= pixels * 3 else {
        return rgba
    }
    for pixel in 0..<pixels {
        rgba[pixel * 4] = UInt8(min(max(chw[pixel], 0), 1) * 255)
        rgba[pixel * 4 + 1] = UInt8(min(max(chw[pixels + pixel], 0), 1) * 255)
        rgba[pixel * 4 + 2] = UInt8(min(max(chw[pixels * 2 + pixel], 0), 1) * 255)
        rgba[pixel * 4 + 3] = 255
    }
    return rgba
}

private func targetPreviewCGImage(directory: URL, frameIndex: Int, width: Int, height: Int) throws -> CGImage {
    guard let imageURL = scannerFrameImageURL(directory: directory, frameIndex: frameIndex) else {
        throw RenderPreviewError.missingFrameImage(frameIndex)
    }
    guard
        let source = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
        let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else {
        throw RenderPreviewError.missingFrameImage(frameIndex)
    }

    let bytesPerPixel = 4
    let bytesPerRow = width * bytesPerPixel
    var rgba = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
    guard let context = CGContext(
        data: &rgba,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: bytesPerRow,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else {
        throw RenderPreviewError.missingFrameImage(frameIndex)
    }
    context.interpolationQuality = .high
    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
    return try FastGSImageExport.cgImage(rgbaBytes: rgba, width: width, height: height)
}

private func initialRenderPreview(
    directory: URL,
    frameIndex: Int,
    width: Int,
    height: Int,
    maxFrames: Int,
    parameters: FastGSTrainableParameters? = nil,
    device: MTLDevice? = nil
) throws -> (target: CGImage, render: CGImage?, renderTexture: MTLTexture?) {
    let dataset = try FastGSScannerDatasetLoader.load(
        directory: directory,
        options: FastGSScannerDatasetOptions(
            width: width,
            height: height,
            maxFrames: maxFrames,
            startIndex: frameIndex,
            normalizeWithAllFramePairs: true
        )
    )
    let scene = FastGSRecordedForwardScene(scannerDataset: dataset, frameIndex: 0)
    let render = try previewOutColor(scene: scene, parameters: parameters)
    let targetRGBA = FastGSImageExport.rgbaBytes(
        outColor: try scene.targetOutColor(),
        width: width,
        height: height
    )
    let renderTexture = deviceTexture(outColor: render, width: width, height: height, device: device)
    let renderImage = try renderTexture == nil ? FastGSImageExport.cgImage(outColor: render, width: width, height: height) : nil
    return (
        target: try FastGSImageExport.cgImage(rgbaBytes: targetRGBA, width: width, height: height),
        render: renderImage,
        renderTexture: renderTexture
    )
}

private func initialRenderPreview(
    cache: FastGSScannerDatasetCache,
    frameIndex: Int,
    width: Int,
    height: Int,
    maxFrames: Int,
    parameters: FastGSTrainableParameters? = nil,
    device: MTLDevice? = nil
) throws -> (target: CGImage, render: CGImage?, renderTexture: MTLTexture?) {
    let dataset = try FastGSScannerDatasetLoader.loadDataset(
        cache: cache,
        frameIndex: frameIndex,
        width: width,
        height: height
    )
    return try renderPreview(dataset: dataset, width: width, height: height, parameters: parameters, device: device)
}

private func renderPreview(
    dataset: FastGSScannerDataset,
    width: Int,
    height: Int,
    parameters: FastGSTrainableParameters? = nil,
    device: MTLDevice? = nil
) throws -> (target: CGImage, render: CGImage?, renderTexture: MTLTexture?) {
    let scene = FastGSRecordedForwardScene(scannerDataset: dataset, frameIndex: 0)
    let render = try previewOutColor(scene: scene, parameters: parameters)
    let targetRGBA = FastGSImageExport.rgbaBytes(
        outColor: try scene.targetOutColor(),
        width: width,
        height: height
    )
    let renderTexture = deviceTexture(outColor: render, width: width, height: height, device: device)
    let renderImage = try renderTexture == nil ? FastGSImageExport.cgImage(outColor: render, width: width, height: height) : nil
    return (
        target: try FastGSImageExport.cgImage(rgbaBytes: targetRGBA, width: width, height: height),
        render: renderImage,
        renderTexture: renderTexture
    )
}

private func deviceTexture(outColor: MLXArray, width: Int, height: Int, device: MTLDevice?) -> MTLTexture? {
    guard let device else {
        return nil
    }
    return FastGSImageExport.texture(outColor: outColor, width: width, height: height, device: device)
}

private func previewOutColor(
    scene: FastGSRecordedForwardScene,
    parameters: FastGSTrainableParameters?
) throws -> MLXArray {
    if let parameters {
        return try scene.renderPreviewOutColor(parameters: parameters)
    }
    return try scene.renderPreviewOutColor(parameters: scene.initialTrainableParameters())
}

private func trainingPreviewResult(
    dataset: FastGSScannerDataset,
    step: Int,
    parameters: FastGSTrainableParameters,
    width: Int,
    height: Int
) throws -> FastGSRecordedTrainingPreviewResult {
    let scene = FastGSRecordedForwardScene(scannerDataset: dataset, frameIndex: 0)
    let targetRGBA = FastGSImageExport.rgbaBytes(
        outColor: try scene.targetOutColor(),
        width: width,
        height: height
    )
    let renderRGBA = FastGSImageExport.rgbaBytes(
        outColor: try scene.renderPreviewOutColor(parameters: parameters),
        width: width,
        height: height
    )
    return FastGSRecordedTrainingPreviewResult(
        step: step,
        targetRGBA: targetRGBA,
        renderRGBA: renderRGBA,
        width: width,
        height: height,
        pointCount: scene.manifest.pointCount,
        parameters: parameters,
        frameIndex: scene.scannerFrameIndex
    )
}

private func trainingScenes(
    cache: FastGSScannerDatasetCache,
    width: Int,
    height: Int,
    maxFrames: Int
) throws -> [FastGSRecordedForwardScene] {
    let frameDescriptors = Array(cache.frameDescriptors.prefix(max(1, maxFrames)))
    return try frameDescriptors.map { descriptor in
        let dataset = try FastGSScannerDatasetLoader.loadDataset(
            cache: cache,
            frameIndex: descriptor.index,
            width: width,
            height: height
        )
        return FastGSRecordedForwardScene(scannerDataset: dataset, frameIndex: 0)
    }
}

private struct RenderPreviewView: View {
    @StateObject private var model = RenderPreviewModel()
    @State private var showingSettings = false

    var body: some View {
        VStack(spacing: 0) {
            toolbar
            Divider()
            preview
        }
        .sheet(isPresented: $showingSettings) {
            TrainingSettingsView(model: model)
        }
    }

    private var toolbar: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("FastGSSwift")
                        .font(.headline)
                    Text("\(model.renderSize)  \(model.status)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    if model.totalTrainingSteps > 0 {
                        Text("Step \(model.trainingStep) / \(model.totalTrainingSteps)")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(model.isTraining ? .primary : .secondary)
                    }
                    Text(model.cameraLabel)
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }

                Spacer()

                Button {
                    showingSettings = true
                } label: {
                    Label("Settings", systemImage: "slider.horizontal.3")
                }
                .disabled(model.isTraining)
                .help("Training settings")

                Button {
                    model.loadDataset()
                } label: {
                    Label("Load", systemImage: "tray.and.arrow.down")
                }
                .disabled(model.isTraining || model.isLoadingDataset)
                .help("Load scanner frame list and target preview")

                Button {
                    model.selectCamera(delta: -1)
                } label: {
                    Image(systemName: "chevron.left")
                }
                .disabled(!model.isDatasetLoaded || model.isRenderingPreview)
                .help("Previous recorded camera")

                Button {
                    model.selectCamera(delta: 1)
                } label: {
                    Image(systemName: "chevron.right")
                }
                .disabled(!model.isDatasetLoaded || model.isRenderingPreview)
                .help("Next recorded camera")

                Button {
                    model.trainRecordedFrame(mode: .fromInitial)
                } label: {
                    Label("Train From Initial", systemImage: "play.circle")
                }
                .disabled(model.isTraining || !model.isDatasetLoaded)
                .help("Train native scanner dataset frames from initial Gaussian parameters")

                Button {
                    model.trainRecordedFrame(mode: .continueTraining)
                } label: {
                    Label("Continue Training", systemImage: "forward.circle")
                }
                .disabled(model.isTraining || !model.isDatasetLoaded)
                .help("Continue from current trained parameters or the latest checkpoint in the output directory")
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private var preview: some View {
        ZStack {
            Color(nsColor: .textBackgroundColor)

            if model.previewMode == .recordedSideBySide {
                recordedSideBySidePreview
                    .padding(24)
            } else if let texture = model.texture, let device = model.device {
                MetalTexturePreview(texture: texture, device: device)
                    .aspectRatio(model.previewAspectRatio, contentMode: .fit)
                    .padding(24)
            } else if let image = model.fallbackImage {
                Image(decorative: image, scale: 1)
                    .interpolation(.none)
                    .resizable()
                    .scaledToFit()
                    .padding(24)
            } else {
                VStack(spacing: 8) {
                    Text(model.cameraLabel)
                        .font(.headline)
                    Text(model.status)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private var recordedSideBySidePreview: some View {
        HStack(spacing: 16) {
            previewPane(title: "Target") {
                if let image = model.targetImage {
                    Image(decorative: image, scale: 1)
                        .interpolation(.none)
                        .resizable()
                        .scaledToFit()
                } else {
                    ProgressView()
                }
            }

            previewPane(title: "Swift Render") {
                if let texture = model.texture, let device = model.device {
                    MetalTexturePreview(texture: texture, device: device)
                        .aspectRatio(model.previewAspectRatio, contentMode: .fit)
                } else if let image = model.fallbackImage {
                    Image(decorative: image, scale: 1)
                        .interpolation(.none)
                        .resizable()
                        .scaledToFit()
                } else {
                    Text("Render not trained")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private func previewPane<Content: View>(title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            ZStack {
                Color(nsColor: .windowBackgroundColor)
                content()
                    .aspectRatio(model.previewAspectRatio, contentMode: .fit)
                    .padding(12)
            }
        }
    }
}

private struct TrainingSettingsView: View {
    @ObservedObject var model: RenderPreviewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Training Settings")
                    .font(.headline)
                Spacer()
                Button {
                    model.applySettings()
                    dismiss()
                } label: {
                    Label("Done", systemImage: "checkmark")
                }
                .keyboardShortcut(.defaultAction)
            }

            folderControl(
                title: "Dataset",
                value: model.datasetLabel,
                systemImage: "folder",
                action: model.chooseDatasetDirectory
            )

            folderControl(
                title: "Output",
                value: model.outputLabel,
                systemImage: "square.and.arrow.down",
                action: model.chooseOutputDirectory
            )

            HStack(spacing: 12) {
                numericField(title: "Width", value: $model.trainingWidth, range: 16...4096)
                numericField(title: "Height", value: $model.trainingHeight, range: 16...4096)
                numericField(title: "Max Frames", value: $model.maxFrames, range: 1...9999)
                numericField(title: "Training Steps", value: $model.trainingSteps, range: 1...100000)
            }

            Toggle("Save training images and parameters", isOn: $model.saveTrainingArtifacts)
                .disabled(model.isTraining)
        }
        .padding(20)
        .frame(width: 620)
    }

    private func folderControl(
        title: String,
        value: String,
        systemImage: String,
        action: @escaping () -> Void
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            HStack(spacing: 8) {
                Text(value)
                    .font(.caption.monospaced())
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .frame(maxWidth: .infinity, alignment: .leading)
                Button(action: action) {
                    Image(systemName: systemImage)
                }
                .disabled(model.isTraining)
            }
        }
    }

    private func numericField(title: String, value: Binding<Int>, range: ClosedRange<Int>) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            TextField(title, value: clamped(value, range: range), format: .number)
                .textFieldStyle(.roundedBorder)
                .frame(width: 116)
                .disabled(model.isTraining)
        }
    }

    private func clamped(_ value: Binding<Int>, range: ClosedRange<Int>) -> Binding<Int> {
        Binding(
            get: { value.wrappedValue },
            set: { value.wrappedValue = min(max($0, range.lowerBound), range.upperBound) }
        )
    }
}

private func scannerFramePairIndices(directory: URL) -> [Int] {
    guard FileManager.default.fileExists(atPath: directory.appendingPathComponent("points.ply").path) else {
        return []
    }
    guard let contents = try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) else {
        return []
    }

    var images = Set<Int>()
    var jsons = Set<Int>()
    for url in contents {
        guard let index = scannerFrameIndex(url) else {
            continue
        }
        if url.pathExtension.lowercased() == "jpg" {
            images.insert(index)
        } else if url.pathExtension.lowercased() == "json" {
            jsons.insert(index)
        }
    }
    return images.intersection(jsons).sorted()
}

private func scannerFrameImageURL(directory: URL, frameIndex: Int) -> URL? {
    let candidates = [
        String(format: "frame_%05d.jpg", frameIndex),
        String(format: "frame_%05d.jpeg", frameIndex),
        "frame_\(frameIndex).jpg",
        "frame_\(frameIndex).jpeg",
    ]
    for candidate in candidates {
        let url = directory.appendingPathComponent(candidate)
        if FileManager.default.fileExists(atPath: url.path) {
            return url
        }
    }
    return nil
}

private func scannerFrameIndex(_ url: URL) -> Int? {
    let stem = url.deletingPathExtension().lastPathComponent
    guard stem.hasPrefix("frame_") else {
        return nil
    }
    return Int(stem.dropFirst("frame_".count))
}

private struct MetalTexturePreview: NSViewRepresentable {
    var texture: MTLTexture
    var device: MTLDevice

    func makeCoordinator() -> Coordinator {
        Coordinator(device: device)
    }

    func makeNSView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: device)
        view.colorPixelFormat = .bgra8Unorm
        view.framebufferOnly = false
        view.enableSetNeedsDisplay = true
        view.isPaused = true
        view.clearColor = MTLClearColor(red: 0.03, green: 0.03, blue: 0.035, alpha: 1)
        view.delegate = context.coordinator
        return view
    }

    func updateNSView(_ view: MTKView, context: Context) {
        context.coordinator.texture = texture
        view.setNeedsDisplay(view.bounds)
    }

    final class Coordinator: NSObject, MTKViewDelegate {
        var texture: MTLTexture?
        private let commandQueue: MTLCommandQueue?
        private let ciContext: CIContext

        init(device: MTLDevice) {
            commandQueue = device.makeCommandQueue()
            ciContext = CIContext(mtlDevice: device)
        }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

        func draw(in view: MTKView) {
            guard
                let texture,
                let drawable = view.currentDrawable,
                let commandBuffer = commandQueue?.makeCommandBuffer(),
                let image = CIImage(mtlTexture: texture, options: [.colorSpace: CGColorSpaceCreateDeviceRGB()])
            else {
                return
            }

            let drawableBounds = CGRect(origin: .zero, size: view.drawableSize)
            let scale = min(
                drawableBounds.width / CGFloat(texture.width),
                drawableBounds.height / CGFloat(texture.height)
            )
            let width = CGFloat(texture.width) * scale
            let height = CGFloat(texture.height) * scale
            let x = (drawableBounds.width - width) * 0.5
            let y = (drawableBounds.height - height) * 0.5
            let transform = CGAffineTransform(
                a: scale,
                b: 0,
                c: 0,
                d: -scale,
                tx: x,
                ty: y + height
            )

            ciContext.render(
                image.transformed(by: transform),
                to: drawable.texture,
                commandBuffer: commandBuffer,
                bounds: drawableBounds,
                colorSpace: CGColorSpaceCreateDeviceRGB()
            )
            commandBuffer.present(drawable)
            commandBuffer.commit()
        }
    }
}
