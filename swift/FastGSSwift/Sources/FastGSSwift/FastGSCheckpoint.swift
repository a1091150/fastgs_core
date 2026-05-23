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
    public var parameterFile: String
    public var note: String?

    public init(
        formatVersion: Int = 1,
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
        parameterFile: String = FastGSCheckpoint.parameterFileName,
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
        self.parameterFile = parameterFile
        self.note = note
    }
}

public enum FastGSCheckpointError: Error, LocalizedError {
    case missingParameter(String)

    public var errorDescription: String? {
        switch self {
        case .missingParameter(let name):
            return "FastGS checkpoint is missing parameter '\(name)'"
        }
    }
}

public enum FastGSCheckpoint {
    public static let parameterFileName = "parameters.safetensors"
    public static let infoFileName = "training_info.json"

    public static func save(
        parameters: FastGSTrainableParameters,
        info: FastGSTrainingCheckpointInfo,
        directory: URL
    ) throws {
        try save(parameters: parameters, info: info, directory: directory, stream: .default)
    }

    public static func save(
        parameters: FastGSTrainableParameters,
        info: FastGSTrainingCheckpointInfo,
        directory: URL,
        stream: StreamOrDevice
    ) throws {
        precondition(directory.isFileURL)

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

        var infoToWrite = info
        infoToWrite.parameterFile = parameterFileName
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
        return try parameters(from: arrays)
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

    public static func parameterURL(in directory: URL) -> URL {
        directory.appendingPathComponent(parameterFileName, isDirectory: false)
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
}
