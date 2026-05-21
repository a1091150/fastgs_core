import FastGSSwift
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
    @Published var image: CGImage?
    @Published var status = "Ready"
    @Published var renderSize = "80 x 48"
    @Published var isRendering = false

    func render() {
        guard !isRendering else {
            return
        }

        isRendering = true
        status = "Rendering..."

        Task {
            do {
                let rendered = try await Task.detached(priority: .userInitiated) {
                    let output = FastGSPreprocessParityFixture.rasterizeLargeE2EOutput()
                    let image = try FastGSImageExport.cgImage(rasterizeOutput: output, width: 80, height: 48)
                    return image
                }.value

                image = rendered
                status = "Rendered with Swift MLXFast"
            } catch {
                status = "Render failed: \(error)"
            }
            isRendering = false
        }
    }
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

            if let image = model.image {
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
