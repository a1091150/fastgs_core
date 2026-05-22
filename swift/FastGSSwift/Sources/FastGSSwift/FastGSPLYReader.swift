import Foundation

public struct FastGSPointCloud {
    public var points: [Float]
    public var colors: [Float]?
    public var count: Int

    public init(points: [Float], colors: [Float]?, count: Int) {
        self.points = points
        self.colors = colors
        self.count = count
    }
}

public enum FastGSPLYReaderError: Error, Equatable {
    case invalidUTF8(URL)
    case missingHeader
    case unsupportedFormat(String)
    case missingVertexElement
    case missingRequiredProperty(String)
    case invalidVertexLine(index: Int)
    case invalidFloat(property: String, value: String, index: Int)
    case invalidColor(property: String, value: String, index: Int)
    case vertexCountMismatch(expected: Int, actual: Int)
}

public enum FastGSPLYReader {
    public static func readPointCloud(url: URL) throws -> FastGSPointCloud {
        let data = try Data(contentsOf: url)
        guard let text = String(data: data, encoding: .utf8) else {
            throw FastGSPLYReaderError.invalidUTF8(url)
        }
        return try readPointCloud(text: text)
    }

    public static func readPointCloud(text: String) throws -> FastGSPointCloud {
        var lines = text.split(whereSeparator: \.isNewline).map(String.init)
        guard !lines.isEmpty, lines.removeFirst() == "ply" else {
            throw FastGSPLYReaderError.missingHeader
        }

        var vertexCount: Int?
        var vertexProperties = [String]()
        var readingVertexProperties = false
        var dataStart = 0

        for (offset, line) in lines.enumerated() {
            let parts = line.split(separator: " ").map(String.init)
            guard !parts.isEmpty else {
                continue
            }

            switch parts[0] {
            case "format":
                guard parts.count >= 2, parts[1] == "ascii" else {
                    throw FastGSPLYReaderError.unsupportedFormat(parts.dropFirst().joined(separator: " "))
                }
            case "element":
                readingVertexProperties = parts.count >= 3 && parts[1] == "vertex"
                if readingVertexProperties {
                    vertexCount = Int(parts[2])
                }
            case "property":
                if readingVertexProperties, let name = parts.last {
                    vertexProperties.append(name)
                }
            case "end_header":
                dataStart = offset + 1
                readingVertexProperties = false
                break
            default:
                continue
            }

            if parts[0] == "end_header" {
                break
            }
        }

        guard let vertexCount else {
            throw FastGSPLYReaderError.missingVertexElement
        }

        let xIndex = try propertyIndex("x", in: vertexProperties)
        let yIndex = try propertyIndex("y", in: vertexProperties)
        let zIndex = try propertyIndex("z", in: vertexProperties)
        let redIndex = vertexProperties.firstIndex(of: "red")
        let greenIndex = vertexProperties.firstIndex(of: "green")
        let blueIndex = vertexProperties.firstIndex(of: "blue")
        let hasColors = redIndex != nil && greenIndex != nil && blueIndex != nil

        var points = [Float]()
        points.reserveCapacity(vertexCount * 3)
        var colors = hasColors ? [Float]() : nil
        colors?.reserveCapacity(vertexCount * 3)

        var parsedVertices = 0
        for line in lines.dropFirst(dataStart) {
            if parsedVertices >= vertexCount {
                break
            }
            let values = line.split(separator: " ").map(String.init)
            guard values.count >= vertexProperties.count else {
                throw FastGSPLYReaderError.invalidVertexLine(index: parsedVertices)
            }

            points.append(try floatValue(values[xIndex], property: "x", index: parsedVertices))
            points.append(try floatValue(values[yIndex], property: "y", index: parsedVertices))
            points.append(try floatValue(values[zIndex], property: "z", index: parsedVertices))

            if let redIndex, let greenIndex, let blueIndex {
                colors?.append(try colorValue(values[redIndex], property: "red", index: parsedVertices))
                colors?.append(try colorValue(values[greenIndex], property: "green", index: parsedVertices))
                colors?.append(try colorValue(values[blueIndex], property: "blue", index: parsedVertices))
            }

            parsedVertices += 1
        }

        guard parsedVertices == vertexCount else {
            throw FastGSPLYReaderError.vertexCountMismatch(expected: vertexCount, actual: parsedVertices)
        }

        return FastGSPointCloud(points: points, colors: colors, count: vertexCount)
    }

    private static func propertyIndex(_ property: String, in properties: [String]) throws -> Int {
        guard let index = properties.firstIndex(of: property) else {
            throw FastGSPLYReaderError.missingRequiredProperty(property)
        }
        return index
    }

    private static func floatValue(_ value: String, property: String, index: Int) throws -> Float {
        guard let result = Float(value) else {
            throw FastGSPLYReaderError.invalidFloat(property: property, value: value, index: index)
        }
        return result
    }

    private static func colorValue(_ value: String, property: String, index: Int) throws -> Float {
        guard let raw = Float(value) else {
            throw FastGSPLYReaderError.invalidColor(property: property, value: value, index: index)
        }
        return min(max(raw / 255, 0), 1)
    }
}
