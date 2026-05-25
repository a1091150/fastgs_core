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

    func testScannerFastGS2BaseScheduleScalesToTenThousandSteps() {
        let config = FastGSDensificationConfig.scannerFastGS2Base(scheduleScale: Float(10_000) / 30_000)

        XCTAssertEqual(config.densifyFromStep, 167)
        XCTAssertEqual(config.densifyUntilStep, 5_000)
        XCTAssertEqual(config.densificationInterval, 167)
        XCTAssertEqual(config.opacityResetInterval, 1_000)
        XCTAssertEqual(config.finalPruneStartStep, 5_000)
        XCTAssertEqual(config.finalPruneEndStep, 10_000)
        XCTAssertEqual(config.finalPruneInterval, 1_000)
        XCTAssertEqual(config.dense, 0.001)
        XCTAssertEqual(config.lossThreshold, 0.1)

        XCTAssertFalse(config.shouldDensifyAndPrune(step: 167))
        XCTAssertTrue(config.shouldDensifyAndPrune(step: 334))
        XCTAssertFalse(config.shouldDensifyAndPrune(step: 5_000))

        XCTAssertTrue(config.shouldResetOpacity(step: 1_000))
        XCTAssertFalse(config.shouldResetOpacity(step: 5_000))

        XCTAssertFalse(config.shouldFinalPrune(step: 5_000))
        XCTAssertTrue(config.shouldFinalPrune(step: 6_000))
        XCTAssertFalse(config.shouldFinalPrune(step: 10_000))
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

    func testDensificationStateAppendResetRows() {
        var state = FastGSDensificationState(count: 2, sceneExtent: 5)
        state.maxRadii2D = [3, 4]
        state.xyzGradAccum = [5, 6]
        state.xyzGradAccumAbs = [7, 8]
        state.denom = [1, 2]
        state.tmpRadii = [9, 10]

        let appended = state.appendingResetRows(count: 3)

        XCTAssertEqual(appended.count, 5)
        XCTAssertEqual(appended.sceneExtent, 5)
        XCTAssertEqual(appended.maxRadii2D, [0, 0, 0, 0, 0])
        XCTAssertEqual(appended.xyzGradAccum, [0, 0, 0, 0, 0])
        XCTAssertEqual(appended.xyzGradAccumAbs, [0, 0, 0, 0, 0])
        XCTAssertEqual(appended.denom, [0, 0, 0, 0, 0])
        XCTAssertNil(appended.tmpRadii)
    }

    func testDensificationStateAppendZeroRowsPreservesAccumulatedStats() {
        var state = FastGSDensificationState(count: 2, sceneExtent: 5)
        state.maxRadii2D = [3, 4]
        state.xyzGradAccum = [5, 6]
        state.xyzGradAccumAbs = [7, 8]
        state.denom = [1, 2]
        state.tmpRadii = [9, 10]

        let appended = state.appendingZeroRows(count: 2)

        XCTAssertEqual(appended.count, 4)
        XCTAssertEqual(appended.sceneExtent, 5)
        XCTAssertEqual(appended.maxRadii2D, [3, 4, 0, 0])
        XCTAssertEqual(appended.xyzGradAccum, [5, 6, 0, 0])
        XCTAssertEqual(appended.xyzGradAccumAbs, [7, 8, 0, 0])
        XCTAssertEqual(appended.denom, [1, 2, 0, 0])
        XCTAssertEqual(appended.tmpRadii, [9, 10, 0, 0])
    }
}
