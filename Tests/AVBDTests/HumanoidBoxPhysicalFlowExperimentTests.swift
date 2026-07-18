import XCTest
@testable import AVBDLearn

final class HumanoidBoxPhysicalFlowExperimentTests: XCTestCase {
    func testPhysicalFlowStageRoundTripsWithoutLosingLineage() throws {
        let stages = [
            HumanoidBoxPhysicalFlowStage(
                trajectory: [0.1, -0.2, 0.3, -0.4], controlSteps: 16,
                trajectoryDurationSteps: 64),
            HumanoidBoxPhysicalFlowStage(
                trajectory: [0, 0, 0, 0], controlSteps: 3),
        ]
        let data = try JSONEncoder().encode(stages)
        let decoded = try JSONDecoder().decode(
            [HumanoidBoxPhysicalFlowStage].self, from: data)
        XCTAssertEqual(decoded.count, 2)
        XCTAssertEqual(decoded[0].trajectory, stages[0].trajectory)
        XCTAssertEqual(decoded[0].controlSteps, 16)
        XCTAssertEqual(decoded[0].trajectoryDurationSteps, 64)
        XCTAssertEqual(decoded[1].trajectory, stages[1].trajectory)
        XCTAssertEqual(decoded[1].controlSteps, 3)
        XCTAssertNil(decoded[1].trajectoryDurationSteps)
    }

    func testTargetDiscoveryRequiresBothPopulationAndGenerations() {
        var configuration = HumanoidBoxPhysicalFlowConfiguration()
        configuration.targetDiscoveryPopulationSize = 8
        XCTAssertThrowsError(try configuration.validate())

        configuration.targetDiscoveryGenerations = 1
        XCTAssertNoThrow(try configuration.validate())

        configuration.targetDiscoveryPopulationSize = 0
        XCTAssertThrowsError(try configuration.validate())
    }

    func testBaseLegCompositionOverrideIsBounded() {
        var configuration = HumanoidBoxPhysicalFlowConfiguration()
        configuration.carryBaseLegActionFractionOverride = 1
        XCTAssertNoThrow(try configuration.validate())
        configuration.carryBaseLegActionFractionOverride = 1.001
        XCTAssertThrowsError(try configuration.validate())
        configuration.carryBaseLegActionFractionOverride = -0.001
        XCTAssertThrowsError(try configuration.validate())
    }

    func testLegBlendSplineStartsContinuouslyAndReachesLastKnot() {
        let parameters = [Float](repeating: 0, count: 20)
            + [0.2, 0.6, 1.0]
        XCTAssertEqual(HumanoidBoxPhysicalFlowExperiment.legBlendFraction(
            parameters, progress: 0, armParameterCount: 20,
            knotCount: 3), 0, accuracy: 1e-6)
        XCTAssertEqual(HumanoidBoxPhysicalFlowExperiment.legBlendFraction(
            parameters, progress: 1.0 / 6.0, armParameterCount: 20,
            knotCount: 3), 0.1, accuracy: 1e-6)
        XCTAssertEqual(HumanoidBoxPhysicalFlowExperiment.legBlendFraction(
            parameters, progress: 1, armParameterCount: 20,
            knotCount: 3), 1, accuracy: 1e-6)

        var configuration = HumanoidBoxPhysicalFlowConfiguration()
        configuration.legBlendKnotCount = -1
        XCTAssertThrowsError(try configuration.validate())
    }
}
