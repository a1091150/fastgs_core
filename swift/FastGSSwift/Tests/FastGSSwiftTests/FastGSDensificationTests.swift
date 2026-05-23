import FastGSSwift
import XCTest

final class FastGSDensificationTests: XCTestCase {
    func testDensificationConfigDefaultSchedule() {
        let config = FastGSDensificationConfig()

        XCTAssertTrue(config.shouldAccumulateStats(step: 0))
        XCTAssertTrue(config.shouldAccumulateStats(step: 14_999))
        XCTAssertFalse(config.shouldAccumulateStats(step: 15_000))

        XCTAssertFalse(config.shouldDensifyAndPrune(step: 500))
        XCTAssertTrue(config.shouldDensifyAndPrune(step: 1_000))
        XCTAssertFalse(config.shouldDensifyAndPrune(step: 15_000))

        XCTAssertTrue(config.shouldResetOpacity(step: 3_000))
        XCTAssertFalse(config.shouldResetOpacity(step: 15_000))

        XCTAssertFalse(config.shouldFinalPrune(step: 15_000))
        XCTAssertTrue(config.shouldFinalPrune(step: 18_000))
        XCTAssertFalse(config.shouldFinalPrune(step: 30_000))
    }

    func testDensificationStateInitializesAndResets() {
        var state = FastGSDensificationState(count: 3, sceneExtent: 2.5)

        XCTAssertEqual(state.count, 3)
        XCTAssertEqual(state.sceneExtent, 2.5)
        XCTAssertEqual(state.maxRadii2D, [0, 0, 0])
        XCTAssertEqual(state.xyzGradAccum, [0, 0, 0])
        XCTAssertEqual(state.xyzGradAccumAbs, [0, 0, 0])
        XCTAssertEqual(state.denom, [0, 0, 0])
        XCTAssertNil(state.tmpRadii)

        state.reset(count: 2, sceneExtent: 4)

        XCTAssertEqual(state.count, 2)
        XCTAssertEqual(state.sceneExtent, 4)
        XCTAssertEqual(state.maxRadii2D, [0, 0])
        XCTAssertEqual(state.xyzGradAccum, [0, 0])
        XCTAssertEqual(state.xyzGradAccumAbs, [0, 0])
        XCTAssertEqual(state.denom, [0, 0])
        XCTAssertNil(state.tmpRadii)
    }

    func testDensificationStateAccumulatesVisibleRadiiAndGradients() {
        var state = FastGSDensificationState(count: 3, sceneExtent: 1)

        state.update(
            radii: [2, 0, 5],
            viewspaceGradients: [
                3, 4, 0, 0,
                9, 9, 9, 9,
                0, 0, 6, 8
            ]
        )

        XCTAssertEqual(state.tmpRadii, [2, 0, 5])
        XCTAssertEqual(state.maxRadii2D, [2, 0, 5])
        XCTAssertEqual(state.xyzGradAccum, [5, 0, 0])
        XCTAssertEqual(state.xyzGradAccumAbs, [0, 0, 10])
        XCTAssertEqual(state.denom, [1, 0, 1])

        state.update(
            radii: [1, 0, 6],
            viewspaceGradients: [
                0, 0, 1, 1,
                2, 2, 2, 2,
                8, 6, 0, 0
            ]
        )

        let averages = state.averageGradients()
        XCTAssertEqual(state.tmpRadii, [1, 0, 6])
        XCTAssertEqual(state.maxRadii2D, [2, 0, 6])
        XCTAssertEqual(state.xyzGradAccum, [5, 0, 10])
        XCTAssertEqual(state.xyzGradAccumAbs[0], Float(2).squareRoot(), accuracy: 1e-6)
        XCTAssertEqual(state.xyzGradAccumAbs[1], 0)
        XCTAssertEqual(state.xyzGradAccumAbs[2], 10)
        XCTAssertEqual(state.denom, [2, 0, 2])
        XCTAssertEqual(averages.gradient, [2.5, 0, 5])
        XCTAssertEqual(averages.gradientAbs[0], Float(2).squareRoot() / 2, accuracy: 1e-6)
        XCTAssertEqual(averages.gradientAbs[1], 0)
        XCTAssertEqual(averages.gradientAbs[2], 5)
    }

    func testDensificationStatePrunesRows() {
        var state = FastGSDensificationState(count: 4, sceneExtent: 3)
        state.maxRadii2D = [1, 2, 3, 4]
        state.xyzGradAccum = [10, 20, 30, 40]
        state.xyzGradAccumAbs = [11, 21, 31, 41]
        state.denom = [5, 6, 7, 8]
        state.tmpRadii = [9, 10, 11, 12]

        let pruned = state.pruned(mask: [false, true, false, true])

        XCTAssertEqual(pruned.count, 2)
        XCTAssertEqual(pruned.sceneExtent, 3)
        XCTAssertEqual(pruned.maxRadii2D, [1, 3])
        XCTAssertEqual(pruned.xyzGradAccum, [10, 30])
        XCTAssertEqual(pruned.xyzGradAccumAbs, [11, 31])
        XCTAssertEqual(pruned.denom, [5, 7])
        XCTAssertEqual(pruned.tmpRadii, [9, 11])
    }
}
