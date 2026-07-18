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
            HumanoidBoxPhysicalFlowStage(
                trajectory: [], controlSteps: 44, policyOnly: true),
        ]
        let data = try JSONEncoder().encode(stages)
        let decoded = try JSONDecoder().decode(
            [HumanoidBoxPhysicalFlowStage].self, from: data)
        XCTAssertEqual(decoded.count, 3)
        XCTAssertEqual(decoded[0].trajectory, stages[0].trajectory)
        XCTAssertEqual(decoded[0].controlSteps, 16)
        XCTAssertEqual(decoded[0].trajectoryDurationSteps, 64)
        XCTAssertEqual(decoded[1].trajectory, stages[1].trajectory)
        XCTAssertEqual(decoded[1].controlSteps, 3)
        XCTAssertNil(decoded[1].trajectoryDurationSteps)
        XCTAssertEqual(decoded[2].trajectory, [])
        XCTAssertEqual(decoded[2].controlSteps, 44)
        XCTAssertEqual(decoded[2].policyOnly, true)
    }

    func testTargetDiscoveryRequiresBothPopulationAndGenerations() {
        var configuration = HumanoidBoxPhysicalFlowConfiguration()
        configuration.targetDiscoveryPopulationSize = 8
        XCTAssertThrowsError(try configuration.validate())

        configuration.targetDiscoveryGenerations = 1
        XCTAssertNoThrow(try configuration.validate())

        configuration.targetDiscoveryPopulationSize = 0
        XCTAssertThrowsError(try configuration.validate())

        configuration.targetDiscoveryPopulationSize = 8
        configuration.minimumTargetCarryDistanceMeters = 0.4
        configuration.targetDiscoveryObjectiveCarryDistanceMeters = 0.39
        XCTAssertThrowsError(try configuration.validate())
        configuration.targetDiscoveryObjectiveCarryDistanceMeters = 0.45
        XCTAssertNoThrow(try configuration.validate())
        configuration.targetGenerationSteps = 120
        configuration.targetExecutionSteps = 121
        XCTAssertThrowsError(try configuration.validate())
        configuration.targetExecutionSteps = 60
        XCTAssertNoThrow(try configuration.validate())
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

    func testLegResidualSplineIsBoundedAndStartsContinuously() {
        var parameters = [Float](repeating: 0, count: 20 + 3 + 20)
        parameters[20 + 3 + 2] = 0.5
        parameters[20 + 3 + 10 + 2] = -1
        XCTAssertEqual(HumanoidBoxPhysicalFlowExperiment.legResidualAction(
            parameters, action: 2, progress: 0,
            armParameterCount: 20, blendKnotCount: 3,
            residualKnotCount: 2, maximumAction: 0.2), 0,
            accuracy: 1e-6)
        XCTAssertEqual(HumanoidBoxPhysicalFlowExperiment.legResidualAction(
            parameters, action: 2, progress: 0.5,
            armParameterCount: 20, blendKnotCount: 3,
            residualKnotCount: 2, maximumAction: 0.2), 0.1,
            accuracy: 1e-6)
        XCTAssertEqual(HumanoidBoxPhysicalFlowExperiment.legResidualAction(
            parameters, action: 2, progress: 1,
            armParameterCount: 20, blendKnotCount: 3,
            residualKnotCount: 2, maximumAction: 0.2), -0.2,
            accuracy: 1e-6)

        var configuration = HumanoidBoxPhysicalFlowConfiguration()
        configuration.legResidualKnotCount = -1
        XCTAssertThrowsError(try configuration.validate())
        configuration.legResidualKnotCount = 2
        configuration.maximumLegResidualAction = 1.01
        XCTAssertThrowsError(try configuration.validate())
    }

    func testFlowDistillationConfigurationRequiresValidDAggerSchedule() {
        var configuration = HumanoidBoxFlowDistillationConfiguration(
            epochs: 4, aggregationRounds: 4)
        XCTAssertNoThrow(try configuration.validate())

        configuration.aggregationRounds = 5
        XCTAssertThrowsError(try configuration.validate())
        configuration.aggregationRounds = 4
        configuration.finalTeacherMix = 0.8
        configuration.initialTeacherMix = 0.5
        XCTAssertThrowsError(try configuration.validate())
        configuration.initialTeacherMix = 0.8
        configuration.stateAlignmentLookahead = 0
        XCTAssertThrowsError(try configuration.validate())
        configuration.stateAlignmentLookahead = 1
        configuration.policySourceRowWeight = 0.99
        XCTAssertThrowsError(try configuration.validate())
    }

    func testStateAlignmentIsMeasuredMonotonicAndLookaheadBounded() {
        let dimension = 105
        var references = [[Float]](
            repeating: [Float](repeating: 0, count: dimension), count: 4)
        for phase in references.indices {
            references[phase][69] = Float(phase)
            references[phase][90] = Float(phase) / 3
        }
        var observation = ContiguousArray(
            repeating: Float(0), count: 2 * dimension)
        observation[dimension + 69] = 2
        observation[dimension + 90] = 2.0 / 3.0

        let aligned = HumanoidBoxFlowDistillation.alignedTargetPhase(
            observation: observation, environment: 1,
            observationDimension: dimension, references: references,
            previousPhase: 1, lookahead: 1)
        XCTAssertEqual(aligned.phase, 2)
        XCTAssertEqual(aligned.distance, 0, accuracy: 1e-6)

        observation[dimension + 69] = 0
        observation[dimension + 90] = 0
        let monotonic = HumanoidBoxFlowDistillation.alignedTargetPhase(
            observation: observation, environment: 1,
            observationDimension: dimension, references: references,
            previousPhase: 2, lookahead: 1)
        XCTAssertEqual(monotonic.phase, 2)
        XCTAssertGreaterThan(monotonic.distance, 0)
    }

    func testStableCarryVerifierLatchesFailureAndRejectsFalseProgress() {
        let stable = HumanoidBoxStableCarrySample(
            terminated: false, truncated: false,
            leftContact: 1, rightContact: 1,
            sourceSupportContact: 0, lifted: true,
            rootUprightAlignment: 0.98,
            boxUprightAlignment: 0.97,
            clearanceMeters: 0.02,
            carryDistanceMeters: 0.36)
        XCTAssertTrue(HumanoidBoxStableCarryVerifier.isStable(
            previouslyFailed: false, sample: stable))
        XCTAssertTrue(HumanoidBoxStableCarryVerifier.isFinalSuccess(
            previouslyFailed: false, sample: stable,
            maximumStableCarryDistanceMeters: 0.36,
            requiredCarryDistanceMeters: 0.35))

        var failed = stable
        failed.terminated = true
        XCTAssertFalse(HumanoidBoxStableCarryVerifier.isStable(
            previouslyFailed: false, sample: failed))
        XCTAssertTrue(HumanoidBoxStableCarryVerifier.failed(
            previouslyFailed: false, sample: failed))
        XCTAssertFalse(HumanoidBoxStableCarryVerifier.isFinalSuccess(
            previouslyFailed: true, sample: stable,
            maximumStableCarryDistanceMeters: 0.50,
            requiredCarryDistanceMeters: 0.35))

        var flyingBox = stable
        flyingBox.leftContact = 0
        flyingBox.carryDistanceMeters = 1.0
        XCTAssertFalse(HumanoidBoxStableCarryVerifier.isStable(
            previouslyFailed: false, sample: flyingBox))
    }
}
