import FastGSSwift
import AppKit
import CoreImage
import Metal
import MetalKit
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
    private let outputDirectory = URL(fileURLWithPath: "/private/tmp/fastgs_swift_mac_training", isDirectory: true)
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
    let device = MTLCreateSystemDefaultDevice()

    init() {
        refreshDataset()
        totalTrainingSteps = trainingConfig.totalSteps
        status = cameraCount > 0
            ? "Ready for native dataset training"
            : "Choose a scanner dataset folder"
    }

    var cameraLabel: String {
        let count = max(cameraCount, 1)
        return "Camera \(min(cameraIndex + 1, count)) / \(count)"
    }

    func trainRecordedFrame() {
        guard !isTraining else {
            return
        }

        isTraining = true
        trainingStep = 0
        refreshDataset()
        totalTrainingSteps = trainingConfig.totalSteps
        guard cameraCount > 0 else {
            status = "Training failed: choose a folder with frame_*.jpg/json and points.ply"
            isTraining = false
            return
        }
        let config = trainingConfig
        status = "Training \(cameraLabel)..."

        Task {
            do {
                guard let device else {
                    status = "Training failed: no Metal device"
                    isTraining = false
                    return
                }
                let cameraIndex = cameraIndex
                let outputDirectory = outputDirectory
                let datasetDirectory = datasetDirectory

                let trained = try await Task.detached(priority: .userInitiated) {
                    let result = try FastGSRecordedTrainingPreview.run(
                        scannerDatasetDirectory: datasetDirectory,
                        cameraIndex: cameraIndex,
                        config: config
                    ) { step in
                        Task { @MainActor in
                            self.trainingStep = step
                            self.status = "Training \(self.cameraLabel)..."
                        }
                    } preview: { preview in
                        try writeSideBySidePNG(
                            targetRGBA: preview.targetRGBA,
                            renderRGBA: preview.renderRGBA,
                            width: preview.width,
                            height: preview.height,
                            to: trainingOutputURL(
                                outputDirectory: outputDirectory,
                                cameraIndex: cameraIndex,
                                step: preview.step
                            )
                        )
                    }

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
                    return (texture, image, targetImage, result.width, result.height, result.pointCount)
                }.value

                texture = trained.0
                fallbackImage = trained.1
                targetImage = trained.2
                renderSize = "\(trained.3) x \(trained.4)"
                previewAspectRatio = Double(trained.3) / Double(trained.4)
                previewMode = .recordedSideBySide
                status = "Training completed, wrote previews to \(outputDirectory.path)"
            } catch {
                status = "Training failed: \(error)"
            }
            isTraining = false
        }
    }

    func selectCamera(delta: Int) {
        guard !isTraining else {
            return
        }
        refreshDataset()
        let count = cameraCount
        guard count > 0 else {
            status = "Choose a scanner dataset folder"
            return
        }
        cameraIndex = (cameraIndex + delta + count) % count
        trainingStep = 0
        totalTrainingSteps = trainingConfig.totalSteps
        status = "Selected \(cameraLabel)"
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
            refreshDataset()
            status = cameraCount > 0
                ? "Selected dataset \(datasetLabel)"
                : "Folder is missing scanner frame pairs or points.ply"
        }
    }

    private func refreshDataset() {
        datasetLabel = datasetDirectory.lastPathComponent
        cameraCount = scannerFramePairCount(directory: datasetDirectory)
        if cameraCount > 0 {
            cameraIndex = min(max(cameraIndex, 0), cameraCount - 1)
        } else {
            cameraIndex = 0
        }
    }
}

private enum RenderPreviewMode {
    case single
    case recordedSideBySide
}

private enum RenderPreviewError: Error {
    case textureCreationFailed
}

private func trainingOutputURL(outputDirectory: URL, cameraIndex: Int, step: Int) -> URL {
    outputDirectory.appendingPathComponent(
        String(format: "camera_%03d_step_%03d_sbs.png", cameraIndex, step)
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

private struct RenderPreviewView: View {
    @StateObject private var model = RenderPreviewModel()

    var body: some View {
        VStack(spacing: 0) {
            toolbar
            Divider()
            preview
        }
    }

    private var toolbar: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text("FastGSSwift")
                    .font(.headline)
                Text(model.datasetLabel)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
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
                model.chooseDatasetDirectory()
            } label: {
                Image(systemName: "folder")
            }
            .disabled(model.isTraining)
            .help("Choose scanner dataset folder")

            Button {
                model.selectCamera(delta: -1)
            } label: {
                Image(systemName: "chevron.left")
            }
            .disabled(model.isTraining)
            .help("Previous recorded camera")

            Button {
                model.selectCamera(delta: 1)
            } label: {
                Image(systemName: "chevron.right")
            }
            .disabled(model.isTraining)
            .help("Next recorded camera")

            Button {
                model.trainRecordedFrame()
            } label: {
                Image(systemName: "play.circle")
            }
            .disabled(model.isTraining)
            .help("Train native scanner dataset 512 x 512 frame")
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
                    ProgressView()
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

private func scannerFramePairCount(directory: URL) -> Int {
    guard FileManager.default.fileExists(atPath: directory.appendingPathComponent("points.ply").path) else {
        return 0
    }
    guard let contents = try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) else {
        return 0
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
    return images.intersection(jsons).count
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
