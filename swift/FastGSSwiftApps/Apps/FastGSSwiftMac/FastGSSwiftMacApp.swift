import FastGSSwift
import AppKit
import CoreImage
import CoreVideo
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
    private let recordedManifestURL = URL(fileURLWithPath: "/private/tmp/fastgs_recorded_reference/recorded_manifest.json")
    private let trainingManifestURL = URL(fileURLWithPath: "/private/tmp/fastgs_recorded_reference_full_512/recorded_manifest.json")
    private let trainingTotalSteps = 200
    @Published var texture: MTLTexture?
    @Published var fallbackImage: CGImage?
    @Published var targetImage: CGImage?
    @Published var status = "Ready"
    @Published var trainingStep = 0
    @Published var totalTrainingSteps = 0
    @Published var renderSize = "80 x 48"
    @Published var previewAspectRatio = 80.0 / 48.0
    @Published var previewMode: RenderPreviewMode = .single
    @Published var isRendering = false
    @Published var isTraining = false
    let device = MTLCreateSystemDefaultDevice()

    func render() {
        guard !isRendering, !isTraining else {
            return
        }

        isRendering = true
        status = "Rendering..."

        Task {
            do {
                guard let device else {
                    status = "Render failed: no Metal device"
                    isRendering = false
                    return
                }

                let rendered = try await Task.detached(priority: .userInitiated) {
                    let output = FastGSPreprocessParityFixture.rasterizeLargeE2EOutput()
                    guard let texture = FastGSImageExport.texture(
                        rasterizeOutput: output,
                        width: 80,
                        height: 48,
                        device: device
                    ) else {
                        throw RenderPreviewError.textureCreationFailed
                    }
                    let image = try FastGSImageExport.cgImage(rasterizeOutput: output, width: 80, height: 48)
                    return (texture, image)
                }.value

                texture = rendered.0
                fallbackImage = rendered.1
                targetImage = nil
                renderSize = "80 x 48"
                previewAspectRatio = 80.0 / 48.0
                previewMode = .single
                status = "Rendered with Swift MLXFast texture"
            } catch {
                status = "Render failed: \(error)"
            }
            isRendering = false
        }
    }

    func renderMockCameraFrame() {
        guard !isRendering, !isTraining else {
            return
        }

        isRendering = true
        status = "Rendering mock camera frame..."

        Task {
            do {
                guard let device else {
                    status = "Camera frame failed: no Metal device"
                    isRendering = false
                    return
                }

                let rendered = try await Task.detached(priority: .userInitiated) {
                    let width = 160
                    let height = 96
                    let pixelBuffer = try makeMockCameraPixelBuffer(width: width, height: height)
                    guard let texture = try FastGSCameraFrameBridge.texture(fromBGRA: pixelBuffer, device: device) else {
                        throw RenderPreviewError.textureCreationFailed
                    }
                    return (texture, width, height)
                }.value

                texture = rendered.0
                fallbackImage = nil
                targetImage = nil
                renderSize = "\(rendered.1) x \(rendered.2)"
                previewAspectRatio = Double(rendered.1) / Double(rendered.2)
                previewMode = .single
                status = "Rendered mock CVPixelBuffer texture"
            } catch {
                status = "Camera frame failed: \(error)"
            }
            isRendering = false
        }
    }

    func renderRecordedFrame() {
        guard !isRendering, !isTraining else {
            return
        }

        isRendering = true
        let manifestURL = recordedManifestURL
        status = "Rendering recorded scanner frame..."

        Task {
            do {
                guard let device else {
                    status = "Recorded render failed: no Metal device"
                    isRendering = false
                    return
                }

                let rendered = try await Task.detached(priority: .userInitiated) {
                    let scene = try FastGSRecordedForwardScene(manifestURL: manifestURL)
                    let target = try loadRecordedTargetImage(scene: scene)
                    let output = try scene.render()
                    guard let texture = FastGSImageExport.texture(
                        rasterizeOutput: output,
                        width: scene.manifest.width,
                        height: scene.manifest.height,
                        device: device
                    ) else {
                        throw RenderPreviewError.textureCreationFailed
                    }
                    let image = try FastGSImageExport.cgImage(
                        rasterizeOutput: output,
                        width: scene.manifest.width,
                        height: scene.manifest.height
                    )
                    return (texture, image, target, scene.manifest.width, scene.manifest.height, scene.manifest.pointCount)
                }.value

                texture = rendered.0
                fallbackImage = rendered.1
                targetImage = rendered.2
                renderSize = "\(rendered.3) x \(rendered.4)"
                previewAspectRatio = Double(rendered.3) / Double(rendered.4)
                previewMode = .recordedSideBySide
                status = "Rendered recorded scanner frame, \(rendered.5) points"
            } catch {
                status = "Recorded render failed: \(error)"
            }
            isRendering = false
        }
    }

    func trainRecordedFrame() {
        guard !isRendering, !isTraining else {
            return
        }

        isTraining = true
        trainingStep = 0
        totalTrainingSteps = trainingTotalSteps
        let manifestURL = trainingManifestURL
        let totalSteps = trainingTotalSteps
        status = "Training recorded 512 x 512 frame..."

        Task {
            do {
                guard let device else {
                    status = "Training failed: no Metal device"
                    isTraining = false
                    return
                }

                guard FileManager.default.fileExists(atPath: manifestURL.path) else {
                    status = "Training failed: run make swift-recorded-full-512-reference first"
                    isTraining = false
                    return
                }

                let trained = try await Task.detached(priority: .userInitiated) {
                    Memory.cacheLimit = 4 * 1024 * 1024 * 1024

                    let scene = try FastGSRecordedForwardScene(manifestURL: manifestURL)
                    let target = try scene.targetOutColor()
                    var parameters = try scene.initialTrainableParameters()
                    var optimizer = FastGSAdamOptimizer(
                        learningRates: FastGSAdamLearningRates(
                            means3D: 5e-5,
                            dc: 5e-4,
                            sh: 5e-4,
                            opacities: 5e-4,
                            scales: 5e-5,
                            rotations: 5e-5
                        )
                    )

                    for step in 1...totalSteps {
                        let result = FastGSTrainingStageGraph.valueAndGrad(
                            scene: scene,
                            parameters: parameters,
                            target: target
                        )
                        parameters = optimizer.update(
                            parameters: parameters,
                            gradients: fastGSMacTrainableGradients(from: result.gradients)
                        )
                        fastGSEvalTrainingState(parameters: parameters, optimizer: optimizer)
                        await MainActor.run {
                            self.trainingStep = step
                            self.status = "Training recorded 512 x 512 frame..."
                        }
                    }

                    let render = FastGSTrainingStageGraph.render(scene: scene, parameters: parameters)
                    guard let texture = FastGSImageExport.texture(
                        outColor: render,
                        width: scene.manifest.width,
                        height: scene.manifest.height,
                        device: device
                    ) else {
                        throw RenderPreviewError.textureCreationFailed
                    }
                    let image = try FastGSImageExport.cgImage(
                        outColor: render,
                        width: scene.manifest.width,
                        height: scene.manifest.height
                    )
                    let targetImage = try FastGSImageExport.cgImage(
                        outColor: target,
                        width: scene.manifest.width,
                        height: scene.manifest.height
                    )
                    return (texture, image, targetImage, scene.manifest.width, scene.manifest.height, scene.manifest.pointCount)
                }.value

                texture = trained.0
                fallbackImage = trained.1
                targetImage = trained.2
                renderSize = "\(trained.3) x \(trained.4)"
                previewAspectRatio = Double(trained.3) / Double(trained.4)
                previewMode = .recordedSideBySide
                status = "Training completed, \(trained.5) points"
            } catch {
                status = "Training failed: \(error)"
            }
            isTraining = false
        }
    }
}

private enum RenderPreviewMode {
    case single
    case recordedSideBySide
}

private enum RenderPreviewError: Error {
    case textureCreationFailed
    case pixelBufferCreationFailed(CVReturn)
    case missingPixelBufferBaseAddress
    case missingRecordedTargetImage
    case cannotLoadRecordedTargetImage(URL)
}

private struct RenderPreviewView: View {
    @StateObject private var model = RenderPreviewModel()

    var body: some View {
        VStack(spacing: 0) {
            toolbar
            Divider()
            preview
        }
        .task {
            model.render()
        }
    }

    private var toolbar: some View {
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
            }

            Spacer()

            Button {
                model.renderMockCameraFrame()
            } label: {
                Image(systemName: "camera.viewfinder")
            }
            .disabled(model.isRendering || model.isTraining)
            .help("Render mock camera frame")

            Button {
                model.renderRecordedFrame()
            } label: {
                Image(systemName: "photo.on.rectangle")
            }
            .disabled(model.isRendering || model.isTraining)
            .help("Render recorded scanner frame")

            Button {
                model.trainRecordedFrame()
            } label: {
                Image(systemName: "play.circle")
            }
            .disabled(model.isRendering || model.isTraining)
            .help("Train recorded 512 x 512 frame")

            Button {
                model.render()
            } label: {
                Image(systemName: "arrow.clockwise")
            }
            .disabled(model.isRendering || model.isTraining)
            .help("Reload render")
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
                ProgressView()
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

private func fastGSMacTrainableGradients(from gradients: [MLXArray]) -> FastGSTrainableGradients {
    precondition(gradients.count == 6, "macOS recorded training expects six trainable gradients")
    return FastGSTrainableGradients(
        means3D: gradients[0],
        dc: gradients[1],
        sh: gradients[2],
        opacities: gradients[3],
        scales: gradients[4],
        rotations: gradients[5]
    )
}

private func fastGSEvalTrainingState(parameters: FastGSTrainableParameters, optimizer: FastGSAdamOptimizer) {
    for array in parameters.arrays {
        array.eval()
    }
    for array in optimizer.stateArrays() {
        array.eval()
    }
}

private func loadRecordedTargetImage(scene: FastGSRecordedForwardScene) throws -> CGImage {
    guard let targetPng = scene.manifest.targetPng else {
        throw RenderPreviewError.missingRecordedTargetImage
    }
    let url = targetPng.hasPrefix("/")
        ? URL(fileURLWithPath: targetPng)
        : (scene.manifestDirectory ?? URL(fileURLWithPath: ".")).appendingPathComponent(targetPng)
    guard let image = NSImage(contentsOf: url), let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        throw RenderPreviewError.cannotLoadRecordedTargetImage(url)
    }
    return cgImage
}

private func makeMockCameraPixelBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
    let attributes: [CFString: Any] = [
        kCVPixelBufferIOSurfacePropertiesKey: [:],
    ]
    var pixelBuffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(
        kCFAllocatorDefault,
        width,
        height,
        kCVPixelFormatType_32BGRA,
        attributes as CFDictionary,
        &pixelBuffer
    )
    guard status == kCVReturnSuccess, let pixelBuffer else {
        throw RenderPreviewError.pixelBufferCreationFailed(status)
    }

    CVPixelBufferLockBaseAddress(pixelBuffer, [])
    defer {
        CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
    }

    guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
        throw RenderPreviewError.missingPixelBufferBaseAddress
    }

    let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
    let bytes = baseAddress.assumingMemoryBound(to: UInt8.self)
    for y in 0..<height {
        for x in 0..<width {
            let offset = y * bytesPerRow + x * 4
            let checker = ((x / 16) + (y / 16)).isMultiple(of: 2)
            let horizontal = UInt8((x * 255) / max(width - 1, 1))
            let vertical = UInt8((y * 255) / max(height - 1, 1))
            bytes[offset + 0] = checker ? 48 : 180
            bytes[offset + 1] = vertical
            bytes[offset + 2] = horizontal
            bytes[offset + 3] = 255
        }
    }

    return pixelBuffer
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
