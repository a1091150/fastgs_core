import Foundation

public final class FastGSRenderPreviewScheduler: @unchecked Sendable {
    private let lock = NSLock()
    private var pendingRequest = false
    private var lastRenderTime = Date.distantPast
    private var minimumInterval: TimeInterval

    public init(maximumFramesPerSecond: Double = 60) {
        self.minimumInterval = maximumFramesPerSecond > 0 ? 1 / maximumFramesPerSecond : 0
    }

    public func setMaximumFramesPerSecond(_ maximumFramesPerSecond: Double) {
        lock.withLock {
            minimumInterval = maximumFramesPerSecond > 0 ? 1 / maximumFramesPerSecond : 0
        }
    }

    public func requestRender() {
        lock.withLock {
            pendingRequest = true
        }
    }

    public func consumeRenderRequest(now: Date = Date()) -> Bool {
        lock.withLock {
            guard pendingRequest else {
                return false
            }
            guard now.timeIntervalSince(lastRenderTime) >= minimumInterval else {
                return false
            }
            pendingRequest = false
            lastRenderTime = now
            return true
        }
    }
}
