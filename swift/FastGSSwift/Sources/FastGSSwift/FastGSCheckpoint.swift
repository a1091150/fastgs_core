import Foundation
import MLX

public struct FastGSTrainingCheckpointInfo: Codable, Equatable {
    public var formatVersion: Int
    public var createdAt: String
    public var datasetDirectory: String
    public var outputDirectory: String?
    public var imageWidth: Int
    public var imageHeight: Int
    public var maxFrames: Int
    public var trainingSteps: Int
    public var completedStep: Int
    public var frameCount: Int?
    public var pointCount: Int?
    public var gaussianCount: Int?
    public var afterTrainingConfig: FastGSDensificationConfig?
    public var optimizerStep: Int?
    public var parameterFile: String
    public var optimizerFile: String?
    public var densificationStateFile: String?
    public var note: String?

    public init(
        formatVersion: Int = 2,
        createdAt: String = ISO8601DateFormatter().string(from: Date()),
        datasetDirectory: String,
        outputDirectory: String? = nil,
        imageWidth: Int,
        imageHeight: Int,
        maxFrames: Int,
        trainingSteps: Int,
        completedStep: Int,
        frameCount: Int? = nil,
        pointCount: Int? = nil,
        gaussianCount: Int? = nil,
        afterTrainingConfig: FastGSDensificationConfig? = nil,
        optimizerStep: Int? = nil,
        parameterFile: String = FastGSCheckpoint.parameterFileName,
        optimizerFile: String? = nil,
        densificationStateFile: String? = nil,
        note: String? = nil
    ) {
        self.formatVersion = formatVersion
        self.createdAt = createdAt
        self.datasetDirectory = datasetDirectory
        self.outputDirectory = outputDirectory
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.maxFrames = maxFrames
        self.trainingSteps = trainingSteps
        self.completedStep = completedStep
        self.frameCount = frameCount
        self.pointCount = pointCount
        self.gaussianCount = gaussianCount
        self.afterTrainingConfig = afterTrainingConfig
        self.optimizerStep = optimizerStep
        self.parameterFile = parameterFile
        self.optimizerFile = optimizerFile
        self.densificationStateFile = densificationStateFile
        self.note = note
    }
}

public enum FastGSCheckpointError: Error, LocalizedError {
    case missingParameter(String)
    case missingOptimizerState(String)
    case gaussianCountMismatch(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .missingParameter(let name):
            return "FastGS checkpoint is missing parameter '\(name)'"
        case .missingOptimizerState(let name):
            return "FastGS checkpoint is missing optimizer state '\(name)'"
        case .gaussianCountMismatch(let expected, let actual):
            return "FastGS checkpoint Gaussian count mismatch: expected \(expected), got \(actual)"
        }
    }
}

public enum FastGSCheckpoint {
    public static let parameterFileName = "parameters.safetensors"
    public static let optimizerFileName = "optimizer.safetensors"
    public static let densificationStateFileName = "densification_state.json"
    public static let infoFileName = "training_info.json"

    public static func save(
        parameters: FastGSTrainableParameters,
        info: FastGSTrainingCheckpointInfo,
        optimizerState: FastGSAdamState? = nil,
        densificationState: FastGSDensificationState? = nil,
        directory: URL
    ) throws {
        try save(
            parameters: parameters,
            info: info,
            optimizerState: optimizerState,
            densificationState: densificationState,
            directory: directory,
            stream: .default
        )
    }

    public static func save(
        parameters: FastGSTrainableParameters,
        info: FastGSTrainingCheckpointInfo,
        optimizerState: FastGSAdamState? = nil,
        densificationState: FastGSDensificationState? = nil,
        directory: URL,
        stream: StreamOrDevice
    ) throws {
        precondition(directory.isFileURL)
        parameters.validateTopology()
        optimizerState?.validateTopology(parameters: parameters)
        densificationState?.validate(count: parameters.gaussianCount)

        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )

        try MLX.save(
            arrays: namedArrays(parameters),
            metadata: [
                "format": "fastgs-swift",
                "formatVersion": "\(info.formatVersion)",
            ],
            url: parameterURL(in: directory),
            stream: stream
        )

        if let optimizerState {
            try MLX.save(
                arrays: namedOptimizerArrays(optimizerState),
                metadata: [
                    "format": "fastgs-swift-optimizer",
                    "formatVersion": "\(info.formatVersion)",
                    "step": "\(optimizerState.step)",
                ],
                url: optimizerURL(in: directory),
                stream: stream
            )
        }

        if let densificationState {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(densificationState).write(to: densificationStateURL(in: directory), options: .atomic)
        }

        var infoToWrite = info
        infoToWrite.parameterFile = parameterFileName
        infoToWrite.optimizerFile = optimizerState == nil ? nil : optimizerFileName
        infoToWrite.densificationStateFile = densificationState == nil ? nil : densificationStateFileName
        infoToWrite.gaussianCount = parameters.gaussianCount
        infoToWrite.optimizerStep = optimizerState?.step
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(infoToWrite).write(to: infoURL(in: directory), options: .atomic)
    }

    public static func loadParameters(
        directory: URL
    ) throws -> FastGSTrainableParameters {
        try loadParameters(directory: directory, stream: .cpu)
    }

    public static func loadParameters(
        directory: URL,
        stream: StreamOrDevice
    ) throws -> FastGSTrainableParameters {
        let arrays = try MLX.loadArrays(url: parameterURL(in: directory), stream: stream)
        let parameters = try parameters(from: arrays)
        if let expectedCount = try? loadInfo(directory: directory).gaussianCount {
            try validateGaussianCount(expected: expectedCount, actual: parameters.gaussianCount)
        }
        return parameters
    }

    public static func loadInfo(directory: URL) throws -> FastGSTrainingCheckpointInfo {
        let data = try Data(contentsOf: infoURL(in: directory))
        return try JSONDecoder().decode(FastGSTrainingCheckpointInfo.self, from: data)
    }

    public static func load(
        directory: URL
    ) throws -> (parameters: FastGSTrainableParameters, info: FastGSTrainingCheckpointInfo) {
        try load(directory: directory, stream: .cpu)
    }

    public static func load(
        directory: URL,
        stream: StreamOrDevice
    ) throws -> (parameters: FastGSTrainableParameters, info: FastGSTrainingCheckpointInfo) {
        (try loadParameters(directory: directory, stream: stream), try loadInfo(directory: directory))
    }

    public static func loadTrainingState(
        directory: URL
    ) throws -> (
        parameters: FastGSTrainableParameters,
        info: FastGSTrainingCheckpointInfo,
        optimizerState: FastGSAdamState?,
        densificationState: FastGSDensificationState?
    ) {
        try loadTrainingState(directory: directory, stream: .cpu)
    }

    public static func loadTrainingState(
        directory: URL,
        stream: StreamOrDevice
    ) throws -> (
        parameters: FastGSTrainableParameters,
        info: FastGSTrainingCheckpointInfo,
        optimizerState: FastGSAdamState?,
        densificationState: FastGSDensificationState?
    ) {
        let info = try loadInfo(directory: directory)
        let parameters = try loadParameters(directory: directory, stream: stream)
        let optimizerState = try loadOptimizerStateIfPresent(directory: directory, info: info, parameters: parameters, stream: stream)
        let densificationState = try loadDensificationStateIfPresent(directory: directory, info: info, parameters: parameters)
        return (parameters, info, optimizerState, densificationState)
    }

    public static func parameterURL(in directory: URL) -> URL {
        directory.appendingPathComponent(parameterFileName, isDirectory: false)
    }

    public static func optimizerURL(in directory: URL) -> URL {
        directory.appendingPathComponent(optimizerFileName, isDirectory: false)
    }

    public static func densificationStateURL(in directory: URL) -> URL {
        directory.appendingPathComponent(densificationStateFileName, isDirectory: false)
    }

    public static func infoURL(in directory: URL) -> URL {
        directory.appendingPathComponent(infoFileName, isDirectory: false)
    }

    public static func namedArrays(_ parameters: FastGSTrainableParameters) -> [String: MLXArray] {
        var arrays = [
            "means3D": parameters.means3D,
            "dc": parameters.dc,
            "sh": parameters.sh,
            "opacityLogits": parameters.opacityLogits,
            "scales": parameters.scales,
            "rotations": parameters.rotations,
        ]
        if let cov3DPrecomputed = parameters.cov3DPrecomputed {
            arrays["cov3DPrecomputed"] = cov3DPrecomputed
        }
        return arrays
    }

    public static func namedOptimizerArrays(_ state: FastGSAdamState) -> [String: MLXArray] {
        var arrays = [
            "means3D.firstMoment": state.means3D.firstMoment,
            "means3D.secondMoment": state.means3D.secondMoment,
            "dc.firstMoment": state.dc.firstMoment,
            "dc.secondMoment": state.dc.secondMoment,
            "sh.firstMoment": state.sh.firstMoment,
            "sh.secondMoment": state.sh.secondMoment,
            "opacityLogits.firstMoment": state.opacityLogits.firstMoment,
            "opacityLogits.secondMoment": state.opacityLogits.secondMoment,
            "scales.firstMoment": state.scales.firstMoment,
            "scales.secondMoment": state.scales.secondMoment,
            "rotations.firstMoment": state.rotations.firstMoment,
            "rotations.secondMoment": state.rotations.secondMoment,
        ]
        if let cov3DPrecomputed = state.cov3DPrecomputed {
            arrays["cov3DPrecomputed.firstMoment"] = cov3DPrecomputed.firstMoment
            arrays["cov3DPrecomputed.secondMoment"] = cov3DPrecomputed.secondMoment
        }
        return arrays
    }

    public static func parameters(from arrays: [String: MLXArray]) throws -> FastGSTrainableParameters {
        FastGSTrainableParameters(
            means3D: try requiredArray("means3D", in: arrays),
            dc: try requiredArray("dc", in: arrays),
            sh: try requiredArray("sh", in: arrays),
            opacityLogits: try opacityLogitsArray(in: arrays),
            scales: try requiredArray("scales", in: arrays),
            rotations: try requiredArray("rotations", in: arrays),
            cov3DPrecomputed: arrays["cov3DPrecomputed"]
        )
    }

    private static func requiredArray(_ name: String, in arrays: [String: MLXArray]) throws -> MLXArray {
        guard let array = arrays[name] else {
            throw FastGSCheckpointError.missingParameter(name)
        }
        return array
    }

    private static func opacityLogitsArray(in arrays: [String: MLXArray]) throws -> MLXArray {
        if let opacityLogits = arrays["opacityLogits"] {
            return opacityLogits
        }
        if let legacyOpacities = arrays["opacities"] {
            return FastGSOpacity.logits(fromProbabilities: legacyOpacities, stream: .cpu)
        }
        throw FastGSCheckpointError.missingParameter("opacityLogits")
    }

    private static func loadOptimizerStateIfPresent(
        directory: URL,
        info: FastGSTrainingCheckpointInfo,
        parameters: FastGSTrainableParameters,
        stream: StreamOrDevice
    ) throws -> FastGSAdamState? {
        guard info.optimizerFile != nil else {
            return nil
        }
        let (arrays, metadata) = try MLX.loadArraysAndMetadata(url: optimizerURL(in: directory), stream: stream)
        let step = metadata["step"].flatMap(Int.init) ?? info.optimizerStep ?? info.completedStep
        let state = FastGSAdamState(
            step: step,
            means3D: try requiredOptimizerField("means3D", in: arrays),
            dc: try requiredOptimizerField("dc", in: arrays),
            sh: try requiredOptimizerField("sh", in: arrays),
            opacityLogits: try requiredOptimizerField("opacityLogits", in: arrays),
            scales: try requiredOptimizerField("scales", in: arrays),
            rotations: try requiredOptimizerField("rotations", in: arrays),
            cov3DPrecomputed: arrays["cov3DPrecomputed.firstMoment"] == nil
                ? nil
                : try requiredOptimizerField("cov3DPrecomputed", in: arrays)
        )
        state.validateTopology(parameters: parameters)
        return state
    }

    private static func loadDensificationStateIfPresent(
        directory: URL,
        info: FastGSTrainingCheckpointInfo,
        parameters: FastGSTrainableParameters
    ) throws -> FastGSDensificationState? {
        guard info.densificationStateFile != nil else {
            return nil
        }
        let state = try JSONDecoder().decode(
            FastGSDensificationState.self,
            from: Data(contentsOf: densificationStateURL(in: directory))
        )
        state.validate(count: parameters.gaussianCount)
        return state
    }

    private static func requiredOptimizerField(
        _ name: String,
        in arrays: [String: MLXArray]
    ) throws -> FastGSAdamFieldState {
        guard let firstMoment = arrays["\(name).firstMoment"] else {
            throw FastGSCheckpointError.missingOptimizerState("\(name).firstMoment")
        }
        guard let secondMoment = arrays["\(name).secondMoment"] else {
            throw FastGSCheckpointError.missingOptimizerState("\(name).secondMoment")
        }
        return FastGSAdamFieldState(firstMoment: firstMoment, secondMoment: secondMoment)
    }

    private static func validateGaussianCount(expected: Int?, actual: Int) throws {
        guard let expected else { return }
        if expected != actual {
            throw FastGSCheckpointError.gaussianCountMismatch(expected: expected, actual: actual)
        }
    }
}
