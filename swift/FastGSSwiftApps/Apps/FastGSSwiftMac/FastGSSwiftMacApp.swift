import FastGSSwift
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
    @Published var texture: MTLTexture?
    @Published var fallbackImage: CGImage?
    @Published var status = "Ready"
    @Published var renderSize = "80 x 48"
    @Published var isRendering = false
    let device = MTLCreateSystemDefaultDevice()

    func render() {
        guard !isRendering else {
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
                status = "Rendered with Swift MLXFast texture"
            } catch {
                status = "Render failed: \(error)"
            }
            isRendering = false
        }
    }
}

private enum RenderPreviewError: Error {
    case textureCreationFailed
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
            }

            Spacer()

            Button {
                model.render()
            } label: {
                Image(systemName: "arrow.clockwise")
            }
            .disabled(model.isRendering)
            .help("Reload render")
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private var preview: some View {
        ZStack {
            Color(nsColor: .textBackgroundColor)

            if let texture = model.texture, let device = model.device {
                MetalTexturePreview(texture: texture, device: device)
                    .aspectRatio(80.0 / 48.0, contentMode: .fit)
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
