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
    public var previewInterval: Int
    public var cacheLimitBytes: Int
    public var learningRates: FastGSAdamLearningRates
    public var densification: FastGSDensificationConfig

    public init(
        referenceSet: FastGSRecordedTrainingReferenceSet = FastGSRecordedTrainingReferenceSet(
            referenceDirectory: URL(fileURLWithPath: "/private/tmp/fastgs_recorded_reference_full_512", isDirectory: true)
        ),
        totalSteps: Int = 200,
        previewInterval: Int = 20,
        cacheLimitBytes: Int = 4 * 1024 * 1024 * 1024,
        learningRates: FastGSAdamLearningRates = FastGSAdamLearningRates(
            means3D: 5e-5,
            dc: 5e-4,
            sh: 5e-4,
            opacityLogits: 5e-4,
            scales: 5e-5,
            rotations: 5e-5
        ),
        densification: FastGSDensificationConfig = FastGSDensificationConfig()
    ) {
        self.referenceSet = referenceSet
        self.totalSteps = totalSteps
        self.previewInterval = previewInterval
        self.cacheLimitBytes = cacheLimitBytes
        self.learningRates = learningRates
        self.densification = densification
    }
}

public typealias FastGSRecordedTrainingPreviewConfig = FastGSRecordedTrainingRunConfig

public struct FastGSRecordedTrainingPreviewResult {
    public var step: Int
    public var targetRGBA: [UInt8]
    public var renderRGBA: [UInt8]
    public var width: Int
    public var height: Int
    public var pointCount: Int
    public var parameters: FastGSTrainableParameters?
    public var optimizerState: FastGSAdamState?
    public var densificationState: FastGSDensificationState?
    public var frameIndex: Int?

    public init(
        step: Int,
        targetRGBA: [UInt8],
        renderRGBA: [UInt8],
        width: Int,
        height: Int,
        pointCount: Int,
        parameters: FastGSTrainableParameters? = nil,
        optimizerState: FastGSAdamState? = nil,
        densificationState: FastGSDensificationState? = nil,
        frameIndex: Int? = nil
    ) {
        self.step = step
        self.targetRGBA = targetRGBA
        self.renderRGBA = renderRGBA
        self.width = width
        self.height = height
        self.pointCount = pointCount
        self.parameters = parameters
        self.optimizerState = optimizerState
        self.densificationState = densificationState
        self.frameIndex = frameIndex
    }
}

public struct FastGSRecordedTrainingPruneSummary: Sendable {
    public var step: Int
    public var reason: String
    public var beforeCount: Int
    public var afterCount: Int
    public var opacityHits: Int
    public var screenSizeHits: Int
    public var worldScaleHits: Int
    public var scoreHits: Int
    public var prunedCount: Int
    public var keptCount: Int
    public var clonedCount: Int
    public var splitSourceCount: Int
    public var splitChildCount: Int
    public var scoringSampleCount: Int
    public var opacityCapped: Bool
    public var opacityReset: Bool
    public var densificationDenomNonzeroCount: Int
    public var maxRadii2DMax: Float
    public var avgGradMean: Float
    public var avgGradMax: Float
    public var avgGradAbsMean: Float
    public var avgGradAbsMax: Float
    public var physicalScaleMin: Float
    public var physicalScaleMean: Float
    public var physicalScaleMax: Float
    public var opacityMin: Float
    public var opacityMean: Float
    public var opacityMax: Float
    public var scaleThreshold: Float
    public var cloneGradientPassCount: Int
    public var cloneScalePassCount: Int
    public var cloneScorePassCount: Int
    public var cloneCandidateCount: Int
    public var splitGradientPassCount: Int
    public var splitScalePassCount: Int
    public var splitScorePassCount: Int
    public var splitCandidateCount: Int
    public var importanceScoreMin: Float
    public var importanceScoreMean: Float
    public var importanceScoreMax: Float
    public var pruningScoreMin: Float
    public var pruningScoreMean: Float
    public var pruningScoreMax: Float

    public init(
        step: Int,
        reason: String,
        beforeCount: Int,
        afterCount: Int,
        opacityHits: Int,
        screenSizeHits: Int,
        worldScaleHits: Int,
        scoreHits: Int = 0,
        prunedCount: Int,
        keptCount: Int,
        clonedCount: Int = 0,
        splitSourceCount: Int = 0,
        splitChildCount: Int = 0,
        scoringSampleCount: Int = 0,
        opacityCapped: Bool = false,
        opacityReset: Bool = false,
        densificationDenomNonzeroCount: Int = 0,
        maxRadii2DMax: Float = 0,
        avgGradMean: Float = 0,
        avgGradMax: Float = 0,
        avgGradAbsMean: Float = 0,
        avgGradAbsMax: Float = 0,
        physicalScaleMin: Float = 0,
        physicalScaleMean: Float = 0,
        physicalScaleMax: Float = 0,
        opacityMin: Float = 0,
        opacityMean: Float = 0,
        opacityMax: Float = 0,
        scaleThreshold: Float = 0,
        cloneGradientPassCount: Int = 0,
        cloneScalePassCount: Int = 0,
        cloneScorePassCount: Int = 0,
        cloneCandidateCount: Int = 0,
        splitGradientPassCount: Int = 0,
        splitScalePassCount: Int = 0,
        splitScorePassCount: Int = 0,
        splitCandidateCount: Int = 0,
        importanceScoreMin: Float = 0,
        importanceScoreMean: Float = 0,
        importanceScoreMax: Float = 0,
        pruningScoreMin: Float = 0,
        pruningScoreMean: Float = 0,
        pruningScoreMax: Float = 0
    ) {
        self.step = step
        self.reason = reason
        self.beforeCount = beforeCount
        self.afterCount = afterCount
        self.opacityHits = opacityHits
        self.screenSizeHits = screenSizeHits
        self.worldScaleHits = worldScaleHits
        self.scoreHits = scoreHits
        self.prunedCount = prunedCount
        self.keptCount = keptCount
        self.clonedCount = clonedCount
        self.splitSourceCount = splitSourceCount
        self.splitChildCount = splitChildCount
        self.scoringSampleCount = scoringSampleCount
        self.opacityCapped = opacityCapped
        self.opacityReset = opacityReset
        self.densificationDenomNonzeroCount = densificationDenomNonzeroCount
        self.maxRadii2DMax = maxRadii2DMax
        self.avgGradMean = avgGradMean
        self.avgGradMax = avgGradMax
        self.avgGradAbsMean = avgGradAbsMean
        self.avgGradAbsMax = avgGradAbsMax
        self.physicalScaleMin = physicalScaleMin
        self.physicalScaleMean = physicalScaleMean
        self.physicalScaleMax = physicalScaleMax
        self.opacityMin = opacityMin
        self.opacityMean = opacityMean
        self.opacityMax = opacityMax
        self.scaleThreshold = scaleThreshold
        self.cloneGradientPassCount = cloneGradientPassCount
        self.cloneScalePassCount = cloneScalePassCount
        self.cloneScorePassCount = cloneScorePassCount
        self.cloneCandidateCount = cloneCandidateCount
        self.splitGradientPassCount = splitGradientPassCount
        self.splitScalePassCount = splitScalePassCount
        self.splitScorePassCount = splitScorePassCount
        self.splitCandidateCount = splitCandidateCount
        self.importanceScoreMin = importanceScoreMin
        self.importanceScoreMean = importanceScoreMean
        self.importanceScoreMax = importanceScoreMax
        self.pruningScoreMin = pruningScoreMin
        self.pruningScoreMean = pruningScoreMean
        self.pruningScoreMax = pruningScoreMax
    }
}

public enum FastGSRecordedTrainingPreview {
    public static func run(
        config: FastGSRecordedTrainingRunConfig = FastGSRecordedTrainingRunConfig(),
        cameraIndex: Int,
        progress: ((Int) -> Void)? = nil,
        pruneSummary: ((FastGSRecordedTrainingPruneSummary) -> Void)? = nil,
        previewScheduler: FastGSRenderPreviewScheduler? = nil,
        scheduledPreview: ((Int, FastGSTrainableParameters) throws -> FastGSRecordedTrainingPreviewResult?)? = nil,
        preview: ((FastGSRecordedTrainingPreviewResult) throws -> Void)? = nil
    ) throws -> FastGSRecordedTrainingPreviewResult {
        guard let manifestURL = config.referenceSet.manifestURL(at: cameraIndex) else {
            throw FastGSRecordedTrainingPreviewError.noRecordedReference(config.referenceSet.referenceDirectory)
        }
        return try run(manifestURL: manifestURL, config: config, progress: progress, pruneSummary: pruneSummary, previewScheduler: previewScheduler, scheduledPreview: scheduledPreview, preview: preview)
    }

    public static func run(
        manifestURL: URL,
        config: FastGSRecordedTrainingRunConfig = FastGSRecordedTrainingRunConfig(),
        progress: ((Int) -> Void)? = nil,
        pruneSummary: ((FastGSRecordedTrainingPruneSummary) -> Void)? = nil,
        previewScheduler: FastGSRenderPreviewScheduler? = nil,
        scheduledPreview: ((Int, FastGSTrainableParameters) throws -> FastGSRecordedTrainingPreviewResult?)? = nil,
        preview: ((FastGSRecordedTrainingPreviewResult) throws -> Void)? = nil
    ) throws -> FastGSRecordedTrainingPreviewResult {
        Memory.cacheLimit = config.cacheLimitBytes

        let scene = try FastGSRecordedForwardScene(manifestURL: manifestURL)
        return try run(scene: scene, config: config, progress: progress, pruneSummary: pruneSummary, previewScheduler: previewScheduler, scheduledPreview: scheduledPreview, preview: preview)
    }

    public static func run(
        scannerDatasetDirectory: URL,
        cameraIndex: Int,
        width: Int = 512,
        height: Int = 512,
        maxFrames: Int = 1,
        config: FastGSRecordedTrainingRunConfig = FastGSRecordedTrainingRunConfig(),
        progress: ((Int) -> Void)? = nil,
        pruneSummary: ((FastGSRecordedTrainingPruneSummary) -> Void)? = nil,
        previewScheduler: FastGSRenderPreviewScheduler? = nil,
        scheduledPreview: ((Int, FastGSTrainableParameters) throws -> FastGSRecordedTrainingPreviewResult?)? = nil,
        preview: ((FastGSRecordedTrainingPreviewResult) throws -> Void)? = nil
    ) throws -> FastGSRecordedTrainingPreviewResult {
        Memory.cacheLimit = config.cacheLimitBytes

        let dataset = try FastGSScannerDatasetLoader.load(
            directory: scannerDatasetDirectory,
            options: FastGSScannerDatasetOptions(
                width: width,
                height: height,
                maxFrames: maxFrames,
                startIndex: cameraIndex,
                normalizeWithAllFramePairs: true
            )
        )
        let scenes = dataset.frames.indices.map {
            FastGSRecordedForwardScene(scannerDataset: dataset, frameIndex: $0)
        }
        return try run(scenes: scenes, config: config, progress: progress, pruneSummary: pruneSummary, previewScheduler: previewScheduler, scheduledPreview: scheduledPreview, preview: preview)
    }

    public static func run(
        scene: FastGSRecordedForwardScene,
        config: FastGSRecordedTrainingRunConfig = FastGSRecordedTrainingRunConfig(),
        progress: ((Int) -> Void)? = nil,
        pruneSummary: ((FastGSRecordedTrainingPruneSummary) -> Void)? = nil,
        previewScheduler: FastGSRenderPreviewScheduler? = nil,
        scheduledPreview: ((Int, FastGSTrainableParameters) throws -> FastGSRecordedTrainingPreviewResult?)? = nil,
        preview: ((FastGSRecordedTrainingPreviewResult) throws -> Void)? = nil
    ) throws -> FastGSRecordedTrainingPreviewResult {
        try run(scenes: [scene], config: config, progress: progress, pruneSummary: pruneSummary, previewScheduler: previewScheduler, scheduledPreview: scheduledPreview, preview: preview)
    }

    public static func run(
        scenes: [FastGSRecordedForwardScene],
        config: FastGSRecordedTrainingRunConfig = FastGSRecordedTrainingRunConfig(),
        initialParameters: FastGSTrainableParameters? = nil,
        initialOptimizerState: FastGSAdamState? = nil,
        initialDensificationState: FastGSDensificationState? = nil,
        progress: ((Int) -> Void)? = nil,
        pruneSummary: ((FastGSRecordedTrainingPruneSummary) -> Void)? = nil,
        previewScheduler: FastGSRenderPreviewScheduler? = nil,
        scheduledPreview: ((Int, FastGSTrainableParameters) throws -> FastGSRecordedTrainingPreviewResult?)? = nil,
        preview: ((FastGSRecordedTrainingPreviewResult) throws -> Void)? = nil
    ) throws -> FastGSRecordedTrainingPreviewResult {
        Memory.cacheLimit = config.cacheLimitBytes
        precondition(!scenes.isEmpty, "recorded training preview expects at least one scene")

        let targets = try scenes.map { try $0.targetOutColor() }
        var parameters = try initialParameters ?? scenes[0].initialTrainableParameters()
        var optimizer = FastGSAdamOptimizer(learningRates: config.learningRates)
        if let initialOptimizerState {
            initialOptimizerState.validateTopology(parameters: parameters)
            optimizer.replaceState(initialOptimizerState)
        }
        var densificationState = initialDensificationState ?? FastGSDensificationState(
            count: parameters.gaussianCount,
            sceneExtent: estimatedSceneExtent(parameters: parameters)
        )
        densificationState.validate(count: parameters.gaussianCount)
        var lastScene = scenes[0]
        var lastTarget = targets[0]

        for step in 1...config.totalSteps {
            let sceneIndex = (step - 1) % scenes.count
            let scene = scenes[sceneIndex]
            let target = targets[sceneIndex]
            lastScene = scene
            lastTarget = target

            let shouldAccumulateStats = config.densification.shouldAccumulateStats(step: step)
            let result: FastGSTrainingSmokeResult
            if shouldAccumulateStats {
                let resultWithStats = FastGSTrainingStageGraph.valueAndGradWithDensificationStats(
                    scene: scene,
                    parameters: parameters,
                    target: target
                )
                densificationState.update(
                    radii: resultWithStats.densificationStats.radii,
                    viewspaceGradients: resultWithStats.densificationStats.viewspaceGradients
                )
                result = FastGSTrainingSmokeResult(loss: resultWithStats.loss, gradients: resultWithStats.gradients)
            } else {
                result = FastGSTrainingStageGraph.valueAndGrad(
                    scene: scene,
                    parameters: parameters,
                    target: target
                )
            }
            parameters = optimizer.update(
                parameters: parameters,
                gradients: trainableGradients(from: result.gradients)
            )
            eval(parameters: parameters, optimizer: optimizer)

            if let summary = try applyAfterTrainingIfNeeded(
                step: step,
                config: config.densification,
                scenes: scenes,
                targets: targets,
                parameters: &parameters,
                optimizer: &optimizer,
                densificationState: &densificationState
            ) {
                eval(parameters: parameters, optimizer: optimizer)
                pruneSummary?(summary)
            }
            progress?(step)

            let shouldRenderScheduledPreview = previewScheduler?.consumeRenderRequest() == true
            if shouldRenderScheduledPreview, let scheduledPreviewResult = try scheduledPreview?(step, parameters) {
                try preview?(scheduledPreviewResult)
            }

            if shouldWritePreview(step: step, config: config) || (shouldRenderScheduledPreview && scheduledPreview == nil) {
                let render = FastGSTrainingStageGraph.render(scene: scene, parameters: parameters)
                let targetRGBA = FastGSImageExport.rgbaBytes(
                    outColor: target,
                    width: scene.manifest.width,
                    height: scene.manifest.height
                )
                try preview?(
                    FastGSRecordedTrainingPreviewResult(
                        step: step,
                        targetRGBA: targetRGBA,
                        renderRGBA: FastGSImageExport.rgbaBytes(
                            outColor: render,
                            width: scene.manifest.width,
                            height: scene.manifest.height
                        ),
                        width: scene.manifest.width,
                        height: scene.manifest.height,
                        pointCount: parameters.gaussianCount,
                        parameters: parameters,
                        optimizerState: optimizer.state,
                        densificationState: densificationState,
                        frameIndex: scene.scannerFrameIndex
                    )
                )
            }
        }

        let render = FastGSTrainingStageGraph.render(scene: lastScene, parameters: parameters)
        return FastGSRecordedTrainingPreviewResult(
            step: config.totalSteps,
            targetRGBA: FastGSImageExport.rgbaBytes(
                outColor: lastTarget,
                width: lastScene.manifest.width,
                height: lastScene.manifest.height
            ),
            renderRGBA: FastGSImageExport.rgbaBytes(
                outColor: render,
                width: lastScene.manifest.width,
                height: lastScene.manifest.height
            ),
            width: lastScene.manifest.width,
            height: lastScene.manifest.height,
            pointCount: parameters.gaussianCount,
            parameters: parameters,
            optimizerState: optimizer.state,
            densificationState: densificationState,
            frameIndex: lastScene.scannerFrameIndex
        )
    }

    private static func trainableGradients(from gradients: [MLXArray]) -> FastGSTrainableGradients {
        precondition(gradients.count == 6, "recorded training preview expects six trainable gradients")
        return FastGSTrainableGradients(
            means3D: gradients[0],
            dc: gradients[1],
            sh: gradients[2],
            opacityLogits: gradients[3],
            scales: gradients[4],
            rotations: gradients[5]
        )
    }

    private static func shouldWritePreview(step: Int, config: FastGSRecordedTrainingRunConfig) -> Bool {
        guard config.previewInterval > 0 else {
            return false
        }
        return step % config.previewInterval == 0
    }

    private static func eval(parameters: FastGSTrainableParameters, optimizer: FastGSAdamOptimizer) {
        for array in parameters.arrays {
            array.eval()
        }
        for array in optimizer.stateArrays() {
            array.eval()
        }
    }

    private static func applyAfterTrainingIfNeeded(
        step: Int,
        config: FastGSDensificationConfig,
        scenes: [FastGSRecordedForwardScene],
        targets: [MLXArray],
        parameters: inout FastGSTrainableParameters,
        optimizer: inout FastGSAdamOptimizer,
        densificationState: inout FastGSDensificationState
    ) throws -> FastGSRecordedTrainingPruneSummary? {
        let beforeCount = parameters.gaussianCount
        let sceneExtent = densificationState.sceneExtent

        if config.shouldDensifyAndPrune(step: step) {
            let beforeDiagnostics = makeAfterTrainingDiagnostics(
                parameters: parameters,
                densificationState: densificationState,
                dense: config.dense,
                sceneExtent: sceneExtent,
                gradThreshold: config.gradThreshold,
                gradAbsThreshold: config.gradAbsThreshold
            )
            let sampleIndices = FastGSGaussianScoring.evenlySpacedSceneIndices(
                sceneCount: scenes.count,
                sampleCount: config.densifyCameraSampleCount
            )
            let scoring = try FastGSGaussianScoring.compute(
                scenes: scenes,
                parameters: parameters,
                sceneIndices: sampleIndices,
                targets: targets,
                lossThreshold: config.lossThreshold,
                densify: true
            )
            let scoredBeforeDiagnostics = beforeDiagnostics.withScores(
                importanceScores: scoring.importanceScores,
                pruningScores: scoring.pruningScores,
                importanceScoreThreshold: config.importanceScoreThreshold
            )
            let cloneResult = FastGSAfterTraining.clone(
                parameters: parameters,
                optimizerState: optimizer.state,
                densificationState: densificationState,
                gradThreshold: config.gradThreshold,
                dense: config.dense,
                sceneExtent: sceneExtent,
                importanceScores: scoring.importanceScores,
                importanceScoreThreshold: config.importanceScoreThreshold,
                resetDensificationState: false
            )
            parameters = cloneResult.parameters
            optimizer.replaceState(cloneResult.optimizerState)
            densificationState = cloneResult.densificationState

            let paddedImportance = scoring.importanceScores.map {
                paddedScores($0, count: parameters.gaussianCount)
            }
            let splitDiagnostics = makeAfterTrainingDiagnostics(
                parameters: parameters,
                densificationState: densificationState,
                dense: config.dense,
                sceneExtent: sceneExtent,
                gradThreshold: config.gradThreshold,
                gradAbsThreshold: config.gradAbsThreshold
            ).withScores(
                importanceScores: paddedImportance,
                pruningScores: nil,
                importanceScoreThreshold: config.importanceScoreThreshold
            )
            let splitResult = FastGSAfterTraining.split(
                parameters: parameters,
                optimizerState: optimizer.state,
                densificationState: densificationState,
                gradAbsThreshold: config.gradAbsThreshold,
                dense: config.dense,
                splitFactor: config.splitFactor,
                sceneExtent: sceneExtent,
                importanceScores: paddedImportance,
                importanceScoreThreshold: config.importanceScoreThreshold
            )
            parameters = splitResult.parameters
            optimizer.replaceState(splitResult.optimizerState)
            densificationState = splitResult.densificationState

            var opacityHits = 0
            var screenSizeHits = 0
            var worldScaleHits = 0
            var prunedCount = 0
            var keptCount = parameters.gaussianCount
            if config.pruneGaussians {
                let paddedPruningScores = paddedScores(scoring.pruningScores, count: parameters.gaussianCount)
                let maxScreenSize = step > config.opacityResetInterval ? config.maxScreenSize : 0
                let pruneResult = FastGSAfterTraining.pruneOnly(
                    parameters: parameters,
                    optimizerState: optimizer.state,
                    densificationState: densificationState,
                    minOpacity: config.minOpacity,
                    maxScreenSize: maxScreenSize,
                    maxWorldScaleFactor: config.maxWorldScaleFactor,
                    sceneExtent: sceneExtent,
                    pruningScores: paddedPruningScores,
                    pruneBudgetFactor: config.pruneBudgetFactor,
                    minGaussians: 1
                )
                parameters = pruneResult.parameters
                optimizer.replaceState(pruneResult.optimizerState)
                if let prunedDensificationState = pruneResult.densificationState {
                    densificationState = prunedDensificationState
                } else {
                    densificationState.reset(count: parameters.gaussianCount, sceneExtent: sceneExtent)
                }
                opacityHits = pruneResult.opacityHits
                screenSizeHits = pruneResult.screenSizeHits
                worldScaleHits = pruneResult.worldScaleHits
                prunedCount = pruneResult.prunedCount
                keptCount = pruneResult.keptCount
            }

            let capped = FastGSAfterTraining.capOpacity(
                parameters: parameters,
                optimizerState: optimizer.state,
                maxOpacity: config.opacityCapAfterDensify
            )
            parameters = capped.parameters
            optimizer.replaceState(capped.optimizerState)
            let shouldResetOpacity = config.shouldResetOpacity(step: step)
            if shouldResetOpacity {
                let reset = FastGSAfterTraining.resetOpacity(
                    parameters: parameters,
                    optimizerState: optimizer.state,
                    resetValue: config.opacityResetValue
                )
                parameters = reset.parameters
                optimizer.replaceState(reset.optimizerState)
            }
            densificationState.reset(count: parameters.gaussianCount, sceneExtent: sceneExtent)

            return FastGSRecordedTrainingPruneSummary(
                step: step,
                reason: "densify_prune",
                beforeCount: beforeCount,
                afterCount: parameters.gaussianCount,
                opacityHits: opacityHits,
                screenSizeHits: screenSizeHits,
                worldScaleHits: worldScaleHits,
                prunedCount: prunedCount,
                keptCount: keptCount,
                clonedCount: cloneResult.clonedCount,
                splitSourceCount: splitResult.sourceCount,
                splitChildCount: splitResult.childCount,
                scoringSampleCount: scoring.sampledFrameCount,
                opacityCapped: true,
                opacityReset: shouldResetOpacity,
                densificationDenomNonzeroCount: scoredBeforeDiagnostics.denomNonzeroCount,
                maxRadii2DMax: scoredBeforeDiagnostics.maxRadii2DMax,
                avgGradMean: scoredBeforeDiagnostics.avgGradMean,
                avgGradMax: scoredBeforeDiagnostics.avgGradMax,
                avgGradAbsMean: scoredBeforeDiagnostics.avgGradAbsMean,
                avgGradAbsMax: scoredBeforeDiagnostics.avgGradAbsMax,
                physicalScaleMin: scoredBeforeDiagnostics.physicalScaleMin,
                physicalScaleMean: scoredBeforeDiagnostics.physicalScaleMean,
                physicalScaleMax: scoredBeforeDiagnostics.physicalScaleMax,
                opacityMin: scoredBeforeDiagnostics.opacityMin,
                opacityMean: scoredBeforeDiagnostics.opacityMean,
                opacityMax: scoredBeforeDiagnostics.opacityMax,
                scaleThreshold: scoredBeforeDiagnostics.scaleThreshold,
                cloneGradientPassCount: scoredBeforeDiagnostics.cloneGradientPassCount,
                cloneScalePassCount: scoredBeforeDiagnostics.cloneScalePassCount,
                cloneScorePassCount: scoredBeforeDiagnostics.cloneScorePassCount,
                cloneCandidateCount: scoredBeforeDiagnostics.cloneCandidateCount,
                splitGradientPassCount: splitDiagnostics.splitGradientPassCount,
                splitScalePassCount: splitDiagnostics.splitScalePassCount,
                splitScorePassCount: splitDiagnostics.splitScorePassCount,
                splitCandidateCount: splitDiagnostics.splitCandidateCount,
                importanceScoreMin: scoredBeforeDiagnostics.importanceScoreMin,
                importanceScoreMean: scoredBeforeDiagnostics.importanceScoreMean,
                importanceScoreMax: scoredBeforeDiagnostics.importanceScoreMax,
                pruningScoreMin: scoredBeforeDiagnostics.pruningScoreMin,
                pruningScoreMean: scoredBeforeDiagnostics.pruningScoreMean,
                pruningScoreMax: scoredBeforeDiagnostics.pruningScoreMax
            )
        }

        if config.shouldFinalPrune(step: step), config.pruneGaussians {
            let beforeDiagnostics = makeAfterTrainingDiagnostics(
                parameters: parameters,
                densificationState: densificationState,
                dense: config.dense,
                sceneExtent: sceneExtent,
                gradThreshold: config.gradThreshold,
                gradAbsThreshold: config.gradAbsThreshold
            )
            let sampleIndices = FastGSGaussianScoring.evenlySpacedSceneIndices(
                sceneCount: scenes.count,
                sampleCount: config.densifyCameraSampleCount
            )
            let scoring = try FastGSGaussianScoring.compute(
                scenes: scenes,
                parameters: parameters,
                sceneIndices: sampleIndices,
                targets: targets,
                lossThreshold: config.lossThreshold,
                densify: false
            )
            let scoredDiagnostics = beforeDiagnostics.withScores(
                importanceScores: scoring.importanceScores,
                pruningScores: scoring.pruningScores,
                importanceScoreThreshold: config.importanceScoreThreshold
            )
            let result = FastGSAfterTraining.finalPrune(
                parameters: parameters,
                optimizerState: optimizer.state,
                densificationState: densificationState,
                pruningScores: scoring.pruningScores,
                scoreThreshold: config.finalPruneScoreThreshold,
                minOpacity: config.finalPruneMinOpacity,
                maxScreenSize: config.maxScreenSize,
                maxWorldScaleFactor: config.maxWorldScaleFactor,
                sceneExtent: sceneExtent,
                minGaussians: config.finalPruneMinGaussians
            )
            parameters = result.parameters
            optimizer.replaceState(result.optimizerState)
            if let prunedDensificationState = result.densificationState {
                densificationState = prunedDensificationState
            } else {
                densificationState.reset(count: parameters.gaussianCount, sceneExtent: sceneExtent)
            }

            return FastGSRecordedTrainingPruneSummary(
                step: step,
                reason: "final_prune",
                beforeCount: beforeCount,
                afterCount: parameters.gaussianCount,
                opacityHits: result.opacityHits,
                screenSizeHits: result.screenSizeHits,
                worldScaleHits: result.worldScaleHits,
                scoreHits: result.scoreHits,
                prunedCount: result.prunedCount,
                keptCount: result.keptCount,
                scoringSampleCount: scoring.sampledFrameCount,
                densificationDenomNonzeroCount: scoredDiagnostics.denomNonzeroCount,
                maxRadii2DMax: scoredDiagnostics.maxRadii2DMax,
                avgGradMean: scoredDiagnostics.avgGradMean,
                avgGradMax: scoredDiagnostics.avgGradMax,
                avgGradAbsMean: scoredDiagnostics.avgGradAbsMean,
                avgGradAbsMax: scoredDiagnostics.avgGradAbsMax,
                physicalScaleMin: scoredDiagnostics.physicalScaleMin,
                physicalScaleMean: scoredDiagnostics.physicalScaleMean,
                physicalScaleMax: scoredDiagnostics.physicalScaleMax,
                opacityMin: scoredDiagnostics.opacityMin,
                opacityMean: scoredDiagnostics.opacityMean,
                opacityMax: scoredDiagnostics.opacityMax,
                scaleThreshold: scoredDiagnostics.scaleThreshold,
                cloneGradientPassCount: scoredDiagnostics.cloneGradientPassCount,
                cloneScalePassCount: scoredDiagnostics.cloneScalePassCount,
                cloneScorePassCount: scoredDiagnostics.cloneScorePassCount,
                cloneCandidateCount: scoredDiagnostics.cloneCandidateCount,
                splitGradientPassCount: scoredDiagnostics.splitGradientPassCount,
                splitScalePassCount: scoredDiagnostics.splitScalePassCount,
                splitScorePassCount: scoredDiagnostics.splitScorePassCount,
                splitCandidateCount: scoredDiagnostics.splitCandidateCount,
                importanceScoreMin: scoredDiagnostics.importanceScoreMin,
                importanceScoreMean: scoredDiagnostics.importanceScoreMean,
                importanceScoreMax: scoredDiagnostics.importanceScoreMax,
                pruningScoreMin: scoredDiagnostics.pruningScoreMin,
                pruningScoreMean: scoredDiagnostics.pruningScoreMean,
                pruningScoreMax: scoredDiagnostics.pruningScoreMax
            )
        }

        if config.shouldResetOpacity(step: step) {
            let reset = FastGSAfterTraining.resetOpacity(
                parameters: parameters,
                optimizerState: optimizer.state,
                resetValue: config.opacityResetValue
            )
            parameters = reset.parameters
            optimizer.replaceState(reset.optimizerState)
            return FastGSRecordedTrainingPruneSummary(
                step: step,
                reason: "opacity_reset",
                beforeCount: beforeCount,
                afterCount: parameters.gaussianCount,
                opacityHits: 0,
                screenSizeHits: 0,
                worldScaleHits: 0,
                prunedCount: 0,
                keptCount: parameters.gaussianCount,
                opacityReset: true
            )
        }

        return nil
    }

    private static func paddedScores(_ scores: [Float], count: Int) -> [Float] {
        precondition(scores.count <= count, "scores cannot be longer than gaussian count")
        return scores + [Float](repeating: 0, count: count - scores.count)
    }

    private static func estimatedSceneExtent(parameters: FastGSTrainableParameters) -> Float {
        let means = parameters.means3D.asArray(Float.self)
        guard means.count >= 3 else {
            return 1
        }
        var maxRadius: Float = 0
        for index in stride(from: 0, to: means.count - 2, by: 3) {
            let x = means[index]
            let y = means[index + 1]
            let z = means[index + 2]
            maxRadius = max(maxRadius, (x * x + y * y + z * z).squareRoot())
        }
        return max(1, maxRadius)
    }
}

private struct FastGSAfterTrainingDiagnostics {
    var denomNonzeroCount: Int
    var maxRadii2DMax: Float
    var avgGradients: [Float]
    var avgGradAbs: [Float]
    var maxScales: [Float]
    var opacities: [Float]
    var scaleThreshold: Float
    var gradThreshold: Float
    var gradAbsThreshold: Float
    var avgGradMean: Float
    var avgGradMax: Float
    var avgGradAbsMean: Float
    var avgGradAbsMax: Float
    var physicalScaleMin: Float
    var physicalScaleMean: Float
    var physicalScaleMax: Float
    var opacityMin: Float
    var opacityMean: Float
    var opacityMax: Float
    var cloneGradientPassCount: Int
    var cloneScalePassCount: Int
    var cloneScorePassCount: Int
    var cloneCandidateCount: Int
    var splitGradientPassCount: Int
    var splitScalePassCount: Int
    var splitScorePassCount: Int
    var splitCandidateCount: Int
    var importanceScoreMin: Float
    var importanceScoreMean: Float
    var importanceScoreMax: Float
    var pruningScoreMin: Float
    var pruningScoreMean: Float
    var pruningScoreMax: Float

    func withScores(
        importanceScores: [Float]?,
        pruningScores: [Float]?,
        importanceScoreThreshold: Float
    ) -> FastGSAfterTrainingDiagnostics {
        var copy = self
        let importanceScores = importanceScores ?? [Float](repeating: 0, count: avgGradients.count)
        let pruningScores = pruningScores ?? [Float](repeating: 0, count: avgGradients.count)
        let scorePass = importanceScores.map { $0 > importanceScoreThreshold }
        copy.cloneScorePassCount = scorePass.filter(\.self).count
        copy.splitScorePassCount = copy.cloneScorePassCount
        copy.cloneCandidateCount = avgGradients.indices.filter {
            avgGradients[$0] >= gradThreshold && maxScales[$0] <= scaleThreshold && scorePass[$0]
        }.count
        copy.splitCandidateCount = avgGradAbs.indices.filter {
            avgGradAbs[$0] >= gradAbsThreshold && maxScales[$0] > scaleThreshold && scorePass[$0]
        }.count
        let importanceStats = fastGSScalarStats(importanceScores)
        copy.importanceScoreMin = importanceStats.min
        copy.importanceScoreMean = importanceStats.mean
        copy.importanceScoreMax = importanceStats.max
        let pruningStats = fastGSScalarStats(pruningScores)
        copy.pruningScoreMin = pruningStats.min
        copy.pruningScoreMean = pruningStats.mean
        copy.pruningScoreMax = pruningStats.max
        return copy
    }
}

private func makeAfterTrainingDiagnostics(
    parameters: FastGSTrainableParameters,
    densificationState: FastGSDensificationState,
    dense: Float,
    sceneExtent: Float,
    gradThreshold: Float,
    gradAbsThreshold: Float
) -> FastGSAfterTrainingDiagnostics {
    let averages = densificationState.averageGradients()
    let maxScales = fastGSPhysicalMaxScales(parameters: parameters)
    let opacities = parameters.opacityProbabilities().asArray(Float.self)
    let scaleThreshold = dense * sceneExtent
    let avgGradStats = fastGSScalarStats(averages.gradient)
    let avgGradAbsStats = fastGSScalarStats(averages.gradientAbs)
    let scaleStats = fastGSScalarStats(maxScales)
    let opacityStats = fastGSScalarStats(opacities)
    return FastGSAfterTrainingDiagnostics(
        denomNonzeroCount: densificationState.denom.filter { $0 > 0 }.count,
        maxRadii2DMax: densificationState.maxRadii2D.max() ?? 0,
        avgGradients: averages.gradient,
        avgGradAbs: averages.gradientAbs,
        maxScales: maxScales,
        opacities: opacities,
        scaleThreshold: scaleThreshold,
        gradThreshold: gradThreshold,
        gradAbsThreshold: gradAbsThreshold,
        avgGradMean: avgGradStats.mean,
        avgGradMax: avgGradStats.max,
        avgGradAbsMean: avgGradAbsStats.mean,
        avgGradAbsMax: avgGradAbsStats.max,
        physicalScaleMin: scaleStats.min,
        physicalScaleMean: scaleStats.mean,
        physicalScaleMax: scaleStats.max,
        opacityMin: opacityStats.min,
        opacityMean: opacityStats.mean,
        opacityMax: opacityStats.max,
        cloneGradientPassCount: averages.gradient.filter { $0 >= gradThreshold }.count,
        cloneScalePassCount: maxScales.filter { $0 <= scaleThreshold }.count,
        cloneScorePassCount: 0,
        cloneCandidateCount: 0,
        splitGradientPassCount: averages.gradientAbs.filter { $0 >= gradAbsThreshold }.count,
        splitScalePassCount: maxScales.filter { $0 > scaleThreshold }.count,
        splitScorePassCount: 0,
        splitCandidateCount: 0,
        importanceScoreMin: 0,
        importanceScoreMean: 0,
        importanceScoreMax: 0,
        pruningScoreMin: 0,
        pruningScoreMean: 0,
        pruningScoreMax: 0
    )
}

private func fastGSPhysicalMaxScales(parameters: FastGSTrainableParameters) -> [Float] {
    let scales = parameters.scales.asArray(Float.self)
    let count = parameters.gaussianCount
    guard count > 0 else {
        return []
    }
    precondition(scales.count % count == 0, "scale count mismatch")
    let width = scales.count / count
    return (0..<count).map { index in
        let base = index * width
        return scales[base..<(base + width)].map { Foundation.exp($0) }.max() ?? 0
    }
}

private func fastGSScalarStats(_ values: [Float]) -> (min: Float, mean: Float, max: Float) {
    guard !values.isEmpty else {
        return (0, 0, 0)
    }
    var minValue = Float.infinity
    var maxValue = -Float.infinity
    var sum: Double = 0
    for value in values {
        minValue = min(minValue, value)
        maxValue = max(maxValue, value)
        sum += Double(value)
    }
    return (minValue, Float(sum / Double(values.count)), maxValue)
}

public enum FastGSRecordedTrainingPreviewError: Error, Equatable {
    case noRecordedReference(URL)
}
