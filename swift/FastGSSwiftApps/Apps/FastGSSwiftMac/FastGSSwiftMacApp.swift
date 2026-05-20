import FastGSSwift
import SwiftUI

@main
struct FastGSSwiftMacApp: App {
    var body: some Scene {
        WindowGroup {
            VStack(spacing: 12) {
                Text("FastGSSwift")
                    .font(.title)
                Text("MLXFastKernel macOS harness")
                    .foregroundStyle(.secondary)
            }
            .frame(minWidth: 360, minHeight: 220)
        }
    }
}
