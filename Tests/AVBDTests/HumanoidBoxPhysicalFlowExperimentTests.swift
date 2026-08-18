import XCTest
@testable import SimCore
@testable import PhysicsAVBD
@testable import Robotics
@testable import RL
@testable import MLXRL

final class HumanoidBoxPhysicalFlowExperimentTests: XCTestCase {
    func testRecoveryCommitRequiresEveryConfiguredPhysicalGate() {
        func feasible(
            stable: Bool = true, carry: Float = 0.36,
            destination: Float = 0.01, clearance: Float = 0.02,
            grasp: Float = 0.94
        ) -> Bool {
            HumanoidBoxPhysicalFlowExperiment.recoveryControlFeasible(
                physicalStable: stable,
                carryDistance: carry, minimumCarryDistance: 0.35,
                destinationProgress: destination,
                minimumDestinationProgress: 0.005,
                clearance: clearance, minimumClearance: 0.01,
                graspQuality: grasp, minimumGraspQuality: 0.9)
        }

        XCTAssertTrue(feasible())
        XCTAssertFalse(feasible(stable: false))
        XCTAssertFalse(feasible(carry: 0.349))
        XCTAssertFalse(feasible(destination: 0.004))
        XCTAssertFalse(feasible(clearance: 0.009))
        XCTAssertFalse(feasible(grasp: 0.899))

        // Destination progress is a horizon goal, not a reason to apply an
        // unsafe control. The separately tested retention gate remains hard
        // while the scheduled terminal goal ranks/selects whole horizons.
        XCTAssertTrue(HumanoidBoxPhysicalFlowExperiment
            .recoveryRetentionFeasible(
                physicalStable: true,
                carryDistance: 0.36, minimumCarryDistance: 0.35,
                clearance: 0.02, minimumClearance: 0.01,
                graspQuality: 0.94, minimumGraspQuality: 0.9))
        XCTAssertFalse(HumanoidBoxPhysicalFlowExperiment
            .recoveryRetentionFeasible(
                physicalStable: true,
                carryDistance: 0.349, minimumCarryDistance: 0.35,
                clearance: 0.02, minimumClearance: 0.01,
                graspQuality: 0.94, minimumGraspQuality: 0.9))
        XCTAssertTrue(HumanoidBoxPhysicalFlowExperiment
            .terminalRecoveryViable(
                clearance: 0.02, minimumClearance: 0.02,
                boxVerticalVelocity: -0.15,
                maximumDownwardVelocity: 0.15))
        XCTAssertFalse(HumanoidBoxPhysicalFlowExperiment
            .terminalRecoveryViable(
                clearance: 0.019, minimumClearance: 0.02,
                boxVerticalVelocity: 0,
                maximumDownwardVelocity: 0.15))
        XCTAssertFalse(HumanoidBoxPhysicalFlowExperiment
            .terminalRecoveryViable(
                clearance: 0.03, minimumClearance: 0.02,
                boxVerticalVelocity: -0.151,
                maximumDownwardVelocity: 0.15))
        XCTAssertTrue(HumanoidBoxPhysicalFlowExperiment
            .terminalRecoveryViable(
                clearance: 0.03, minimumClearance: 0.02,
                boxVerticalVelocity: 0,
                maximumDownwardVelocity: 0.15,
                footUnloadingFraction: 0.91,
                minimumFootUnloading: 0.9))
        XCTAssertFalse(HumanoidBoxPhysicalFlowExperiment
            .terminalRecoveryViable(
                clearance: 0.03, minimumClearance: 0.02,
                boxVerticalVelocity: 0,
                maximumDownwardVelocity: 0.15,
                footUnloadingFraction: 0.89,
                minimumFootUnloading: 0.9))
        XCTAssertTrue(HumanoidBoxPhysicalFlowExperiment
            .pathDownwardVelocityFeasible(
                boxVerticalVelocity: -0.35,
                maximumDownwardVelocity: 0.35))
        XCTAssertFalse(HumanoidBoxPhysicalFlowExperiment
            .pathDownwardVelocityFeasible(
                boxVerticalVelocity: -0.351,
                maximumDownwardVelocity: 0.35))
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment
                .pathDownwardVelocityPenalty(
                    boxVerticalVelocity: -0.35,
                    maximumDownwardVelocity: 0.35),
            0)
        XCTAssertGreaterThan(
            HumanoidBoxPhysicalFlowExperiment
                .pathDownwardVelocityPenalty(
                    boxVerticalVelocity: -0.351,
                    maximumDownwardVelocity: 0.35),
            4)
    }

    func testRecedingProgressConstraintRampsToTheFinalGoal() {
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment.pathCarryMinimum(
                terminalMinimum: 0.34, maximumRegression: 0.02),
            0.32, accuracy: 1e-7)
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment.scheduledProgressMinimum(
                finalMinimum: 0.008, absoluteStep: 1,
                executionSteps: 8),
            0.001, accuracy: 1e-7)
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment.scheduledProgressMinimum(
                finalMinimum: 0.008, absoluteStep: 4,
                executionSteps: 8),
            0.004, accuracy: 1e-7)
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment.scheduledProgressMinimum(
                finalMinimum: 0.008, absoluteStep: 12,
                executionSteps: 8),
            0.008, accuracy: 1e-7)
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment.scheduledAbsoluteMinimum(
                initial: 0.72, finalMinimum: 0.80,
                absoluteStep: 2, executionSteps: 8),
            0.74, accuracy: 1e-7)
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment.scheduledAbsoluteMinimum(
                initial: 0.72, finalMinimum: 0.80,
                absoluteStep: 8, executionSteps: 8),
            0.80, accuracy: 1e-7)
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment.scheduledIntegerMinimum(
                finalMinimum: 2, absoluteStep: 1, executionSteps: 8), 0)
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment.scheduledIntegerMinimum(
                finalMinimum: 2, absoluteStep: 4, executionSteps: 8), 1)
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment.scheduledIntegerMinimum(
                finalMinimum: 2, absoluteStep: 8, executionSteps: 8), 2)

        var invalidRegression = HumanoidBoxPhysicalFlowConfiguration()
        invalidRegression.minimumTargetCarryDistanceMeters = 0.01
        invalidRegression.maximumTargetPathCarryRegressionMeters = 0.02
        XCTAssertThrowsError(try invalidRegression.validate())
    }

    func testPhysicalTouchdownMinimumIsAHardPlannerBarrier() {
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment.hardIntegerMinimumPenalty(
                value: 2, minimum: 2), 0)
        XCTAssertGreaterThan(
            HumanoidBoxPhysicalFlowExperiment.hardIntegerMinimumPenalty(
                value: 1, minimum: 2), 1)
        XCTAssertGreaterThan(
            HumanoidBoxPhysicalFlowExperiment.hardIntegerMinimumPenalty(
                value: 0, minimum: 2),
            HumanoidBoxPhysicalFlowExperiment.hardIntegerMinimumPenalty(
                value: 1, minimum: 2))
    }

    func testLegPolicyProjectionMasksOnlyUnownedWholeBodyChannels() {
        let carryDimension = HumanoidBoxCarryTask.observationDimension
        var observations = ContiguousArray<Float>(
            (0..<(2 * carryDimension)).map(Float.init))
        observations[9] = 0.5
        observations[carryDimension + 9] = 0.75
        let projected = HumanoidBoxPhysicalFlowExperiment
            .isolatedLegPolicyObservations(
                observations, numEnvironments: 2)
        XCTAssertEqual(projected.count,
                       2 * HumanoidIsaacVelocityTask.observationDimension)
        for environment in 0..<2 {
            let sourceBase = environment
                * HumanoidIsaacVelocityTask.observationDimension
            let carryBase = environment * carryDimension
            XCTAssertEqual(projected[sourceBase + 9],
                           observations[carryBase + 9])
            for joint in 0..<10 {
                XCTAssertEqual(projected[sourceBase + 12 + joint],
                               observations[carryBase + 12 + joint])
                XCTAssertEqual(projected[sourceBase + 31 + joint],
                               observations[carryBase + 31 + joint])
                XCTAssertEqual(projected[sourceBase + 50 + joint],
                               observations[carryBase + 50 + joint])
            }
            for joint in 10..<19 {
                XCTAssertEqual(projected[sourceBase + 12 + joint], 0)
                XCTAssertEqual(projected[sourceBase + 31 + joint], 0)
                XCTAssertEqual(projected[sourceBase + 50 + joint], 0)
            }
        }
    }

    func testHolonomicOverrideProjectsPointGoalIntoPlanarCommand() {
        let dimension = HumanoidBoxCarryTask.observationDimension
        var observations = ContiguousArray<Float>(
            repeating: 0, count: 2 * dimension)
        observations[103] = 3
        observations[104] = 4
        observations[dimension + 103] = -4
        observations[dimension + 104] = 3
        HumanoidBoxPhysicalFlowExperiment.overrideLocomotionCommand(
            &observations, numEnvironments: 2,
            forwardOnly: false, holonomic: true, speed: 0.5)
        XCTAssertEqual(observations[9], 0.3, accuracy: 1e-6)
        XCTAssertEqual(observations[10], 0.4, accuracy: 1e-6)
        XCTAssertEqual(observations[11], 0, accuracy: 1e-6)
        XCTAssertEqual(observations[dimension + 9], -0.4,
                       accuracy: 1e-6)
        XCTAssertEqual(observations[dimension + 10], 0.3,
                       accuracy: 1e-6)
        XCTAssertEqual(observations[dimension + 11], 0, accuracy: 1e-6)
    }

    func testPhysicalFlowStageRoundTripsWithoutLosingLineage() throws {
        let stages = [
            HumanoidBoxPhysicalFlowStage(
                trajectory: [0.1, -0.2, 0.3, -0.4], controlSteps: 16,
                trajectoryDurationSteps: 64),
            HumanoidBoxPhysicalFlowStage(
                trajectory: [0, 0, 0, 0], controlSteps: 3),
            HumanoidBoxPhysicalFlowStage(
                trajectory: [], controlSteps: 44, policyOnly: true),
            HumanoidBoxPhysicalFlowStage(
                trajectory: [0.2, 0.1, 0, -0.1], controlSteps: 7,
                minimumCarryDistanceMeters: 0.375,
                minimumClearanceMeters: 0.04,
                minimumGraspQuality: 0.67,
                certificationDwellSteps: 4),
            HumanoidBoxPhysicalFlowStage(
                trajectory: [], controlSteps: 2,
                trajectorySequence: [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.2, 0.3, 0.4, 0.5],
                ],
                trajectorySequencePhaseSteps: [0, 2],
                trajectorySequenceStepDenominator: 4,
                forwardOnlyBaseCommand: true,
                locomotionCheckpointDirectory: "/tmp/h1-flat-checkpoint",
                locomotionCommandSpeed: 0.5,
                canonicalizeReplicasBeforeExecution: true,
                continueFromPreviousTrajectoryTerminal: true,
                graspAnchorFeedbackBlend: 0.8,
                graspAnchorFeedbackVelocityHorizonSeconds: 0.08,
                graspAnchorFeedbackMaximumActionCorrection: 0.9,
                graspAnchorFeedbackInwardPreloadMeters: 0.005,
                leftGraspAnchorBoxLocalMeters: [0.09, 0.17, 0.05],
                rightGraspAnchorBoxLocalMeters: [0.04, -0.15, 0.15],
                graspAnchorBoxHeightMeters: 0.92,
                minimumCarryDistanceMeters: 0.39,
                minimumDestinationProgressMeters: 0.08,
                minimumRootDestinationProgressMeters: 0.06,
                minimumTouchdowns: 1,
                minimumAlternatingSteps: 2,
                minimumSwingFootLiftMeters: 0.015,
                minimumFootAirTimeSeconds: 0.12,
                minimumFootUnloadingFraction: 0.9,
                minimumTerminalFootUnloadingFraction: 0.85,
                minimumClearanceMeters: 0.04,
                maximumPathDownwardBoxVelocityMPS: 0.35,
                certificationDwellSteps: 2),
            HumanoidBoxPhysicalFlowStage(
                trajectory: [0.1, 0.2, 0.3, 0.4], controlSteps: 1,
                holonomicBaseCommand: true,
                locomotionCheckpointDirectory: "/tmp/h1-flat-checkpoint",
                locomotionCommandSpeed: 0.3),
        ]
        let data = try JSONEncoder().encode(stages)
        let decoded = try JSONDecoder().decode(
            [HumanoidBoxPhysicalFlowStage].self, from: data)
        XCTAssertEqual(decoded.count, 6)
        XCTAssertEqual(decoded[0].trajectory, stages[0].trajectory)
        XCTAssertEqual(decoded[0].controlSteps, 16)
        XCTAssertEqual(decoded[0].trajectoryDurationSteps, 64)
        XCTAssertEqual(decoded[1].trajectory, stages[1].trajectory)
        XCTAssertEqual(decoded[1].controlSteps, 3)
        XCTAssertNil(decoded[1].trajectoryDurationSteps)
        XCTAssertEqual(decoded[2].trajectory, [])
        XCTAssertEqual(decoded[2].controlSteps, 44)
        XCTAssertEqual(decoded[2].policyOnly, true)
        XCTAssertEqual(decoded[3].minimumCarryDistanceMeters, 0.375)
        XCTAssertEqual(decoded[3].minimumClearanceMeters, 0.04)
        XCTAssertEqual(decoded[3].minimumGraspQuality, 0.67)
        XCTAssertEqual(decoded[3].certificationDwellSteps, 4)
        XCTAssertEqual(decoded[4].trajectorySequence,
                       stages[4].trajectorySequence)
        XCTAssertEqual(decoded[4].trajectorySequencePhaseSteps, [0, 2])
        XCTAssertEqual(decoded[4].trajectorySequenceStepDenominator, 4)
        XCTAssertEqual(decoded[4].forwardOnlyBaseCommand, true)
        XCTAssertEqual(decoded[4].locomotionCheckpointDirectory,
                       "/tmp/h1-flat-checkpoint")
        XCTAssertEqual(decoded[4].locomotionCommandSpeed, 0.5)
        XCTAssertEqual(
            decoded[4].canonicalizeReplicasBeforeExecution, true)
        XCTAssertEqual(
            decoded[4].continueFromPreviousTrajectoryTerminal, true)
        XCTAssertEqual(decoded[4].graspAnchorFeedbackBlend, 0.8)
        XCTAssertEqual(
            decoded[4].graspAnchorFeedbackVelocityHorizonSeconds, 0.08)
        XCTAssertEqual(
            decoded[4].graspAnchorFeedbackMaximumActionCorrection, 0.9)
        XCTAssertEqual(
            decoded[4].graspAnchorFeedbackInwardPreloadMeters, 0.005)
        XCTAssertEqual(
            decoded[4].leftGraspAnchorBoxLocalMeters,
            [0.09, 0.17, 0.05])
        XCTAssertEqual(
            decoded[4].rightGraspAnchorBoxLocalMeters,
            [0.04, -0.15, 0.15])
        XCTAssertEqual(decoded[4].graspAnchorBoxHeightMeters, 0.92)
        XCTAssertEqual(decoded[4].minimumDestinationProgressMeters, 0.08)
        XCTAssertEqual(
            decoded[4].minimumRootDestinationProgressMeters, 0.06)
        XCTAssertEqual(decoded[4].minimumTouchdowns, 1)
        XCTAssertEqual(decoded[4].minimumAlternatingSteps, 2)
        XCTAssertEqual(decoded[4].minimumSwingFootLiftMeters, 0.015)
        XCTAssertEqual(decoded[4].minimumFootAirTimeSeconds, 0.12)
        XCTAssertEqual(decoded[4].minimumFootUnloadingFraction, 0.9)
        XCTAssertEqual(
            decoded[4].minimumTerminalFootUnloadingFraction, 0.85)
        XCTAssertEqual(
            decoded[4].maximumPathDownwardBoxVelocityMPS, 0.35)
        XCTAssertEqual(decoded[5].holonomicBaseCommand, true)
        XCTAssertEqual(decoded[5].forwardOnlyBaseCommand, nil)
    }

    func testTargetFailureRoundTripPreservesExactReplayLineageAndNearMiss()
        throws
    {
        let warmupActions = [
            [Float](repeating: 0.125, count: 19),
            [Float](repeating: -0.25, count: 19),
        ]
        let sourceActions = [
            [Float](repeating: 0.375, count: 19),
            [Float](repeating: -0.5, count: 19),
        ]
        let nearMissParameters: [Float] = [0.2, -0.3, 0.4]
        let boundary = HumanoidBoxPhysicalFlowTrajectoryBoundary(
            armDelta: [0.1, -0.2], legBlend: 0.65,
            legResidual: [0.03, -0.04], torsoResidual: 0.05,
            armAsymmetry: [0.06, -0.07])
        let object: [String: Any] = [
            "experiment": "humanoid-box-target-discovery-failure-v0",
            "seed": 91,
            "targetGeneratingTrajectory": [Float(0), Float(0.1)],
            "targetGenerationSteps": 8,
            "sourceStages": [],
            "sourceWarmupAppliedActions": warmupActions,
            "sourceAppliedActions": sourceActions,
            "nearMissParameters": nearMissParameters,
            "nearMissPhaseStep": 3,
            "nearMissTrajectoryStart": [
                "armDelta": boundary.armDelta,
                "legBlend": boundary.legBlend,
                "legResidual": boundary.legResidual,
                "torsoResidual": boundary.torsoResidual,
                "armAsymmetry": boundary.armAsymmetry,
            ],
            "legBlendKnotCount": 1,
            "legResidualKnotCount": 1,
            "torsoResidualKnotCount": 1,
            "maximumClearanceMeters": Float(0.02),
            "maximumCarryDistanceMeters": Float(0.3),
            "maximumStableCarryDistanceMeters": Float(0.29),
            "maximumFeasibilityDwellSteps": 2,
            "requiredCarryDistanceMeters": Float(0.35),
            "requiredClearanceMeters": Float(0.01),
            "requiredFeasibilityDwellSteps": 4,
            "targetPlanningGatePassed": false,
            "goGatePassed": false,
        ]
        let encodedFixture = try JSONSerialization.data(
            withJSONObject: object)
        let decodedFixture = try JSONDecoder().decode(
            HumanoidBoxPhysicalFlowTargetFailure.self,
            from: encodedFixture)
        let roundTripped = try JSONDecoder().decode(
            HumanoidBoxPhysicalFlowTargetFailure.self,
            from: JSONEncoder().encode(decodedFixture))

        XCTAssertEqual(roundTripped.sourceWarmupAppliedActions,
                       warmupActions)
        XCTAssertEqual(roundTripped.sourceAppliedActions, sourceActions)
        XCTAssertEqual(roundTripped.nearMissParameters,
                       nearMissParameters)
        XCTAssertEqual(roundTripped.nearMissPhaseStep, 3)
        XCTAssertEqual(roundTripped.nearMissTrajectoryStart, boundary)
    }

    func testTargetDiscoveryRequiresBothPopulationAndGenerations() {
        var configuration = HumanoidBoxPhysicalFlowConfiguration()
        configuration.targetDiscoveryPopulationSize = 8
        XCTAssertThrowsError(try configuration.validate())

        configuration.targetDiscoveryGenerations = 1
        XCTAssertNoThrow(try configuration.validate())
        configuration.targetDiscoveryArmOnly = true
        XCTAssertNoThrow(try configuration.validate())
        configuration.targetDiscoveryPopulationSize = 0
        XCTAssertThrowsError(try configuration.validate())
        configuration.targetDiscoveryPopulationSize = 8
        configuration.targetDiscoveryArmOnly = false
        configuration.targetDiscoveryLowerBodyOnly = true
        XCTAssertNoThrow(try configuration.validate())
        configuration.targetDiscoveryArmOnly = true
        XCTAssertThrowsError(try configuration.validate())
        configuration.targetDiscoveryArmOnly = false
        configuration.targetDiscoveryLowerBodyOnly = false

        configuration.recedingHorizonSteps = 4
        configuration.recedingTerminalHoldSteps = 2
        XCTAssertNoThrow(try configuration.validate())
        configuration.recedingControlHorizonSteps = 4
        XCTAssertNoThrow(try configuration.validate())
        configuration.recedingControlHorizonSteps = 5
        XCTAssertThrowsError(try configuration.validate())
        configuration.recedingControlHorizonSteps = 1
        configuration.recedingValidationReplicaCount = 9
        XCTAssertThrowsError(try configuration.validate())
        configuration.recedingValidationReplicaCount = 1
        XCTAssertNoThrow(try configuration.validate())
        configuration.recedingValidationReplicaCount = nil
        configuration.recedingValidationCandidateCount = 9
        XCTAssertThrowsError(try configuration.validate())
        configuration.recedingValidationCandidateCount = 4
        configuration.recedingValidationMinimumSuccessFraction = 0.49
        XCTAssertThrowsError(try configuration.validate())
        configuration.recedingValidationMinimumSuccessFraction = 1.01
        XCTAssertThrowsError(try configuration.validate())
        configuration.recedingValidationMinimumSuccessFraction = 0.8
        XCTAssertNoThrow(try configuration.validate())
        configuration.reconstructionValidationCandidateCount = 129
        XCTAssertThrowsError(try configuration.validate())
        configuration.reconstructionValidationCandidateCount = 1
        XCTAssertThrowsError(try configuration.validate())
        configuration.reconstructionValidationCandidateCount = 4
        configuration.reconstructionValidationMinimumSuccessFraction = 0.49
        XCTAssertThrowsError(try configuration.validate())
        configuration.reconstructionValidationMinimumSuccessFraction = 0.8
        XCTAssertNoThrow(try configuration.validate())
        configuration.reconstructionOneEnvironmentSearch = true
        XCTAssertNoThrow(try configuration.validate())
        configuration.reconstructionOneEnvironmentSearch = false
        configuration.minimumTargetTerminalFootUnloadingFraction = 1.01
        XCTAssertThrowsError(try configuration.validate())
        configuration.minimumTargetTerminalFootUnloadingFraction = 0.9
        XCTAssertNoThrow(try configuration.validate())
        configuration.minimumTargetTerminalFootUnloadingFraction = nil
        configuration.legBlendKnotCount = 1
        configuration.recedingLocomotionCheckpointDirectory = "/tmp/h1-flat"
        XCTAssertNoThrow(try configuration.validate())
        configuration.recedingLocomotionCommandSpeed = 0.5
        XCTAssertThrowsError(try configuration.validate())
        configuration.recedingForwardOnlyBaseCommand = true
        XCTAssertNoThrow(try configuration.validate())
        configuration.recedingHolonomicBaseCommand = true
        XCTAssertThrowsError(try configuration.validate())
        configuration.recedingForwardOnlyBaseCommand = false
        XCTAssertNoThrow(try configuration.validate())
        configuration.recedingHolonomicBaseCommand = false
        configuration.recedingForwardOnlyBaseCommand = true
        configuration.recedingHorizonSteps = 0
        XCTAssertThrowsError(try configuration.validate())
        configuration.recedingTerminalHoldSteps = 0
        configuration.recedingLocomotionCheckpointDirectory = nil
        configuration.recedingLocomotionCommandSpeed = nil
        configuration.recedingForwardOnlyBaseCommand = false
        configuration.recedingHolonomicBaseCommand = false
        configuration.legBlendKnotCount = 0

        configuration.targetDiscoveryPopulationSize = 0
        XCTAssertThrowsError(try configuration.validate())

        configuration.targetDiscoveryPopulationSize = 8
        configuration.minimumTargetCarryDistanceMeters = 0.4
        configuration.targetDiscoveryObjectiveCarryDistanceMeters = 0.39
        XCTAssertThrowsError(try configuration.validate())
        configuration.targetDiscoveryObjectiveCarryDistanceMeters = 0.45
        XCTAssertNoThrow(try configuration.validate())
        configuration.minimumTargetDestinationProgressMeters = 0.2
        configuration.targetDiscoveryObjectiveDestinationProgressMeters = 0.19
        XCTAssertThrowsError(try configuration.validate())
        configuration.targetDiscoveryObjectiveDestinationProgressMeters = 0.25
        XCTAssertNoThrow(try configuration.validate())
        configuration.minimumTargetRootDestinationProgressMeters = 0.1
        configuration.targetDiscoveryObjectiveRootDestinationProgressMeters =
            0.09
        XCTAssertThrowsError(try configuration.validate())
        configuration.targetDiscoveryObjectiveRootDestinationProgressMeters =
            0.15
        XCTAssertNoThrow(try configuration.validate())
        configuration.minimumTargetAlternatingSteps = -1
        XCTAssertThrowsError(try configuration.validate())
        configuration.minimumTargetAlternatingSteps = 0
        configuration.minimumTargetTouchdowns = -1
        XCTAssertThrowsError(try configuration.validate())
        configuration.minimumTargetTouchdowns = 1
        configuration.minimumTargetAlternatingSteps = 1
        configuration.targetDiscoveryObjectiveSwingFootLiftMeters = 0
        XCTAssertThrowsError(try configuration.validate())
        configuration.targetDiscoveryObjectiveSwingFootLiftMeters = 0.04
        XCTAssertNoThrow(try configuration.validate())
        configuration.minimumTargetSwingFootLiftMeters = 0.05
        XCTAssertThrowsError(try configuration.validate())
        configuration.minimumTargetSwingFootLiftMeters = 0.015
        XCTAssertNoThrow(try configuration.validate())
        configuration.minimumTargetFootAirTimeSeconds = 0.12
        configuration.targetDiscoveryObjectiveFootAirTimeSeconds = 0.10
        XCTAssertThrowsError(try configuration.validate())
        configuration.targetDiscoveryObjectiveFootAirTimeSeconds = 0.16
        XCTAssertNoThrow(try configuration.validate())
        configuration.minimumTargetFootAirTimeSeconds = -0.01
        XCTAssertThrowsError(try configuration.validate())
        configuration.minimumTargetFootAirTimeSeconds = 0.12
        configuration.minimumTargetFootUnloadingFraction = 0.8
        configuration.targetDiscoveryObjectiveFootUnloadingFraction = 0.7
        XCTAssertThrowsError(try configuration.validate())
        configuration.targetDiscoveryObjectiveFootUnloadingFraction = 0.95
        XCTAssertNoThrow(try configuration.validate())
        configuration.minimumTargetFootUnloadingFraction = 1.01
        XCTAssertThrowsError(try configuration.validate())
        configuration.minimumTargetFootUnloadingFraction = 0.8
        configuration.minimumTargetClearanceMeters = 0.04
        configuration.targetDiscoveryObjectiveClearanceMeters = 0.03
        XCTAssertThrowsError(try configuration.validate())
        configuration.targetDiscoveryObjectiveClearanceMeters = 0.08
        XCTAssertNoThrow(try configuration.validate())
        configuration.graspAnchorFeedbackBlend = 1.01
        XCTAssertThrowsError(try configuration.validate())
        configuration.graspAnchorFeedbackBlend = 0.8
        XCTAssertNoThrow(try configuration.validate())
        configuration.graspAnchorFeedbackVelocityHorizonSeconds = 0.201
        XCTAssertThrowsError(try configuration.validate())
        configuration.graspAnchorFeedbackVelocityHorizonSeconds = 0.08
        XCTAssertNoThrow(try configuration.validate())
        configuration.graspAnchorFeedbackMaximumActionCorrection = 1.001
        XCTAssertThrowsError(try configuration.validate())
        configuration.graspAnchorFeedbackMaximumActionCorrection = 1
        XCTAssertNoThrow(try configuration.validate())
        configuration.graspAnchorFeedbackInwardPreloadMeters = 0.031
        XCTAssertThrowsError(try configuration.validate())
        configuration.graspAnchorFeedbackInwardPreloadMeters = 0.005
        XCTAssertNoThrow(try configuration.validate())
        configuration.maximumTargetPathDownwardBoxVelocityMPS = -0.01
        XCTAssertThrowsError(try configuration.validate())
        configuration.maximumTargetPathDownwardBoxVelocityMPS = 0.35
        XCTAssertNoThrow(try configuration.validate())
        configuration.minimumTargetTerminalClearanceMeters = 0.03
        XCTAssertThrowsError(try configuration.validate())
        configuration.minimumTargetTerminalClearanceMeters = 0.05
        configuration.maximumTargetTerminalDownwardBoxVelocityMPS = -0.01
        XCTAssertThrowsError(try configuration.validate())
        configuration.maximumTargetTerminalDownwardBoxVelocityMPS = 0.05
        XCTAssertNoThrow(try configuration.validate())
        configuration.minimumTargetGraspQuality = -0.01
        XCTAssertThrowsError(try configuration.validate())
        configuration.minimumTargetGraspQuality = 0.67
        configuration.targetDiscoveryObjectiveGraspQuality = 0.66
        XCTAssertThrowsError(try configuration.validate())
        configuration.targetDiscoveryObjectiveGraspQuality = 0.8
        XCTAssertNoThrow(try configuration.validate())
        configuration.targetFeasibilityDwellSteps = 0
        XCTAssertThrowsError(try configuration.validate())
        configuration.targetFeasibilityDwellSteps = 8
        XCTAssertNoThrow(try configuration.validate())
        configuration.recedingHorizonSteps = 4
        XCTAssertThrowsError(try configuration.validate())
        configuration.targetFeasibilityDwellSteps = 4
        XCTAssertNoThrow(try configuration.validate())
        configuration.recedingHorizonSteps = 0
        configuration.targetFeasibilityDwellSteps = 8
        configuration.targetGenerationSteps = 120
        configuration.targetExecutionSteps = 121
        XCTAssertThrowsError(try configuration.validate())
        configuration.targetExecutionSteps = 60
        XCTAssertNoThrow(try configuration.validate())
        configuration.targetSelectionStep = 61
        XCTAssertThrowsError(try configuration.validate())
        configuration.targetSelectionStep = 60
        XCTAssertNoThrow(try configuration.validate())
        configuration.candidateTrajectoryDurationSteps = 0
        XCTAssertThrowsError(try configuration.validate())
        configuration.candidateTrajectoryDurationSteps = 30
        XCTAssertNoThrow(try configuration.validate())
        configuration.robustActionNoiseStandardDeviation = -0.001
        XCTAssertThrowsError(try configuration.validate())
        configuration.robustActionNoiseStandardDeviation = 0.001
        XCTAssertNoThrow(try configuration.validate())
        configuration.optimizationActionNoiseStandardDeviation = -0.001
        XCTAssertThrowsError(try configuration.validate())
        configuration.optimizationActionNoiseStandardDeviation = 0.002
        configuration.optimizationActionNoiseReplicaCount = 0
        XCTAssertThrowsError(try configuration.validate())
        configuration.optimizationActionNoiseReplicaCount = 4
        XCTAssertNoThrow(try configuration.validate())
        configuration.armAsymmetryKnotCount = -1
        XCTAssertThrowsError(try configuration.validate())
        configuration.armAsymmetryKnotCount = 4
        configuration.maximumArmAsymmetryAction = 0
        XCTAssertThrowsError(try configuration.validate())
        configuration.maximumArmAsymmetryAction = 0.25
        XCTAssertNoThrow(try configuration.validate())
        configuration.recedingHorizonSteps = -1
        XCTAssertThrowsError(try configuration.validate())
        configuration.targetFeasibilityDwellSteps = 4
        configuration.recedingHorizonSteps = 4
        XCTAssertNoThrow(try configuration.validate())
        configuration.recedingLocomotionBlendProposal = 0.5
        XCTAssertThrowsError(try configuration.validate())
        configuration.legBlendKnotCount = 3
        XCTAssertNoThrow(try configuration.validate())
        configuration.recedingLocomotionBlendProposal = 1.1
        XCTAssertThrowsError(try configuration.validate())
    }

    func testRobustValidationUsesCeilingCountAndExactReplayBranch() {
        let accepted =
            HumanoidBoxPhysicalFlowExperiment.robustValidationSummary(
                replicaPasses: [true, true, true, true, false],
                minimumSuccessFraction: 0.8)
        XCTAssertEqual(accepted.requiredSuccessCount, 4)
        XCTAssertEqual(accepted.successFraction, 0.8, accuracy: 1e-6)
        XCTAssertEqual(accepted.lowerQuantileIndex, 1)
        XCTAssertEqual(accepted.upperQuantileIndex, 3)
        XCTAssertTrue(accepted.replicaAgreementPassed)
        XCTAssertTrue(accepted.exactBranchPassed)
        XCTAssertTrue(accepted.passed)

        let hiddenExactFailure =
            HumanoidBoxPhysicalFlowExperiment.robustValidationSummary(
                replicaPasses: [false, true, true, true, true],
                minimumSuccessFraction: 0.8)
        XCTAssertTrue(hiddenExactFailure.replicaAgreementPassed)
        XCTAssertFalse(hiddenExactFailure.exactBranchPassed)
        XCTAssertFalse(hiddenExactFailure.passed)

        let insufficient =
            HumanoidBoxPhysicalFlowExperiment.robustValidationSummary(
                replicaPasses: [true, true, true, false, false],
                minimumSuccessFraction: 0.8)
        XCTAssertFalse(insufficient.replicaAgreementPassed)
        XCTAssertFalse(insufficient.passed)

        let batch64 =
            HumanoidBoxPhysicalFlowExperiment.robustValidationSummary(
                replicaPasses: [Bool](repeating: true, count: 52)
                    + [Bool](repeating: false, count: 12),
                minimumSuccessFraction: 0.8)
        XCTAssertEqual(batch64.requiredSuccessCount, 52)
        XCTAssertEqual(batch64.lowerQuantileIndex, 12)
        XCTAssertEqual(batch64.upperQuantileIndex, 51)
        XCTAssertTrue(batch64.passed)
    }

    func testReconstructionEliteCountUsesTheActuallyValidatedSet() {
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment.reconstructionEliteCount(
                populationSize: 128,
                eliteFraction: 0.05,
                validatedCandidateCount: 4),
            4)
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment.reconstructionEliteCount(
                populationSize: 128,
                eliteFraction: 0.05,
                validatedCandidateCount: 8),
            6)
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment.reconstructionEliteCount(
                populationSize: 32,
                eliteFraction: 0.05,
                validatedCandidateCount: 4),
            2)
    }

    func testPhysicalFlowStagePrefixPreservesExecutedControlSchema() {
        let sequence = [
            [Float](repeating: 0.1, count: 4),
            [Float](repeating: 0.2, count: 4),
            [Float](repeating: 0.3, count: 4),
        ]
        let stage = HumanoidBoxPhysicalFlowStage(
            trajectory: [],
            controlSteps: 3,
            trajectorySequence: sequence,
            trajectorySequencePhaseSteps: [0, 2, 3],
            trajectorySequenceStepDenominator: 8,
            appliedNormalizedActions: (0..<3).map {
                [Float](repeating: Float($0), count: 19)
            },
            canonicalizeReplicasBeforeExecution: true,
            continueFromPreviousTrajectoryTerminal: true,
            minimumFootUnloadingFraction: 0.9,
            minimumTerminalFootUnloadingFraction: 0.88,
            certificationDwellSteps: 2)
        let prefix = stage.prefix(controlSteps: 2)
        XCTAssertEqual(prefix?.controlSteps, 2)
        XCTAssertEqual(prefix?.trajectorySequence, Array(sequence.prefix(2)))
        XCTAssertEqual(prefix?.trajectorySequencePhaseSteps, [0, 2])
        XCTAssertEqual(prefix?.appliedNormalizedActions?.count, 2)
        XCTAssertEqual(
            prefix?.appliedNormalizedActions?.last?.first, 1)
        XCTAssertEqual(prefix?.trajectorySequenceStepDenominator, 8)
        XCTAssertEqual(
            prefix?.canonicalizeReplicasBeforeExecution, true)
        XCTAssertEqual(
            prefix?.continueFromPreviousTrajectoryTerminal, true)
        XCTAssertEqual(prefix?.trajectoryEvaluationStep(at: 0), 0)
        XCTAssertEqual(prefix?.trajectoryEvaluationStep(at: 1), 2)
        XCTAssertEqual(prefix?.minimumFootUnloadingFraction, 0.9)
        XCTAssertEqual(
            prefix?.minimumTerminalFootUnloadingFraction, 0.88)
        XCTAssertEqual(prefix?.certificationDwellSteps, 2)
        XCTAssertNil(stage.prefix(controlSteps: 0))
        XCTAssertNil(stage.prefix(controlSteps: 4))

        let spline = HumanoidBoxPhysicalFlowStage(
            trajectory: [0.1, 0.2, 0.3, 0.4],
            controlSteps: 10,
            trajectoryDurationSteps: 20)
        let splinePrefix = spline.prefix(controlSteps: 6)
        XCTAssertEqual(splinePrefix?.controlSteps, 6)
        XCTAssertEqual(splinePrefix?.trajectory, spline.trajectory)
        XCTAssertEqual(splinePrefix?.trajectoryDurationSteps, 20)
    }

    func testComposedSplineRetainsItsCertifiedTerminalBoundary() {
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment.continuedTrajectoryValue(
                zeroStartedValue: 0.25,
                initialValue: 1,
                progress: 0.1,
                knotCount: 2),
            1.05,
            accuracy: 1e-6)
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment.continuedTrajectoryValue(
                zeroStartedValue: 0.25,
                initialValue: 1,
                progress: 0.5,
                knotCount: 2),
            0.25,
            accuracy: 1e-6)

        let parameters: [Float] = [
            0.1, 0.2, -0.3, 0.4, // symmetric arm
            0.6, // leg blend
            0.1, 0.2, 0.3, 0.4, 0.5,
            -0.1, -0.2, -0.3, -0.4, -0.5, // leg residuals
            0.7, // torso
            0.2, -0.2, 0.4, -0.4, // arm asymmetry
        ]
        let boundary = HumanoidBoxPhysicalFlowExperiment
            .structuredTrajectoryBoundary(
                parameters,
                progress: 1,
                armKnotCount: 1,
                blendKnotCount: 1,
                legResidualKnotCount: 1,
                maximumLegResidualAction: 0.5,
                torsoResidualKnotCount: 1,
                maximumTorsoResidualAction: 0.2,
                armAsymmetryKnotCount: 1,
                maximumArmAsymmetryAction: 0.25)
        XCTAssertEqual(
            boundary.armDelta,
            [0.1, 0.2, -0.3, 0.4, 0.1, -0.2, 0.3, 0.4])
        XCTAssertEqual(boundary.legBlend, 0.6, accuracy: 1e-6)
        XCTAssertEqual(boundary.legResidual[4], 0.25, accuracy: 1e-6)
        XCTAssertEqual(boundary.legResidual[9], -0.25, accuracy: 1e-6)
        XCTAssertEqual(boundary.torsoResidual, 0.14, accuracy: 1e-6)
        XCTAssertEqual(
            boundary.armAsymmetry,
            [0.05, -0.05, 0.1, -0.1])

        let continuationParameters: [Float] = [
            // Two symmetric-arm knots that must be replaced.
            -0.9, -0.8, -0.7, -0.6,
            0.9, 0.8, 0.7, 0.6,
            // Two leg-blend knots that must remain searchable.
            0.2, 0.8,
            // One ten-action leg-residual knot that must remain searchable.
            0.01, 0.02, 0.03, 0.04, 0.05,
            -0.01, -0.02, -0.03, -0.04, -0.05,
            // One torso-residual knot that must remain searchable.
            -0.7,
            // Two arm-asymmetry knots that must be replaced.
            -0.9, -0.8, -0.7, -0.6,
            0.9, 0.8, 0.7, 0.6,
        ]
        let held = HumanoidBoxPhysicalFlowExperiment
            .trajectoryHoldingSourceTerminalUpperBody(
                continuationParameters,
                boundary: boundary,
                armKnotCount: 2,
                blendKnotCount: 2,
                legResidualKnotCount: 1,
                torsoResidualKnotCount: 1,
                armAsymmetryKnotCount: 2,
                maximumArmAsymmetryAction: 0.25)
        XCTAssertEqual(
            Array(held[0..<8]),
            [0.1, 0.2, -0.3, 0.4, 0.1, 0.2, -0.3, 0.4])
        XCTAssertEqual(
            Array(held[8..<21]),
            Array(continuationParameters[8..<21]))
        XCTAssertEqual(
            Array(held[21..<29]),
            [0.2, -0.2, 0.4, -0.4, 0.2, -0.2, 0.4, -0.4])

        let tied = HumanoidBoxPhysicalFlowExperiment
            .trajectoryTyingArmKnots(
                continuationParameters,
                armKnotCount: 2,
                blendKnotCount: 2,
                legResidualKnotCount: 1,
                torsoResidualKnotCount: 1,
                armAsymmetryKnotCount: 2)
        XCTAssertEqual(
            Array(tied[0..<8]),
            [-0.9, -0.8, -0.7, -0.6,
             -0.9, -0.8, -0.7, -0.6])
        XCTAssertEqual(
            Array(tied[8..<21]),
            Array(continuationParameters[8..<21]))
        XCTAssertEqual(
            Array(tied[21..<29]),
            [-0.9, -0.8, -0.7, -0.6,
             -0.9, -0.8, -0.7, -0.6])

        let cleanLocomotion = HumanoidBoxPhysicalFlowExperiment
            .trajectoryWithLocomotionProposal(
                continuationParameters,
                blend: 0.75,
                armKnotCount: 2,
                blendKnotCount: 2,
                legResidualKnotCount: 1,
                torsoResidualKnotCount: 1,
                armAsymmetryKnotCount: 2,
                zeroResiduals: true)
        XCTAssertEqual(
            Array(cleanLocomotion[0..<8]),
            Array(continuationParameters[0..<8]))
        XCTAssertEqual(
            Array(cleanLocomotion[8..<10]), [0.75, 0.75])
        XCTAssertEqual(
            Array(cleanLocomotion[10..<21]),
            [Float](repeating: 0, count: 11))
        XCTAssertEqual(
            Array(cleanLocomotion[21..<29]),
            Array(continuationParameters[21..<29]))
    }

    func testPhasedRecedingHorizonRetainsExactlyConfiguredHold() {
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment.recedingEvaluationSteps(
                horizon: 16, phaseStep: 0, terminalHoldSteps: 8),
            24)
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment.recedingEvaluationSteps(
                horizon: 16, phaseStep: 2, terminalHoldSteps: 8),
            22)
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment.recedingEvaluationSteps(
                horizon: 16, phaseStep: 15, terminalHoldSteps: 8),
            9)
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment.recedingEvaluationSteps(
                horizon: 16, phaseStep: 14, terminalHoldSteps: 8),
            10)
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment
                .recedingControlHorizonRemainder(
                    configuredSteps: 4, horizon: 16,
                    selectedPhaseStep: 0),
            3)
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment
                .recedingControlHorizonRemainder(
                    configuredSteps: 4, horizon: 16,
                    selectedPhaseStep: 14),
            1)
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment
                .recedingRequiredSafePrefixSteps(
                    activePlanSteps: 16, controlHorizonSteps: 2,
                    safetyLookaheadSteps: nil),
            16)
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment
                .recedingRequiredSafePrefixSteps(
                    activePlanSteps: 16, controlHorizonSteps: 2,
                    safetyLookaheadSteps: 2),
            4)
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment
                .recedingRequiredSafePrefixSteps(
                    activePlanSteps: 2, controlHorizonSteps: 2,
                    safetyLookaheadSteps: 2),
            2)

        var invalidInitialPhase =
            HumanoidBoxPhysicalFlowConfiguration()
        invalidInitialPhase.recedingInitialPhaseStep = 1
        XCTAssertThrowsError(try invalidInitialPhase.validate())
        var invalidLookahead =
            HumanoidBoxPhysicalFlowConfiguration()
        invalidLookahead.recedingSafetyLookaheadSteps = 0
        XCTAssertThrowsError(try invalidLookahead.validate())
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment
                .recedingControlHorizonRemainder(
                    configuredSteps: 1, horizon: 16,
                    selectedPhaseStep: 0),
            0)
    }

    func testRecedingPlanBoundaryUsesThePreviouslyExecutedControl() {
        let initial = HumanoidBoxPhysicalFlowExperiment
            .StructuredTrajectoryBoundary(
                armDelta: [Float](repeating: 0.4, count: 8),
                legBlend: 0.6,
                legResidual: [Float](repeating: -0.2, count: 10),
                torsoResidual: 0.1,
                armAsymmetry: [Float](repeating: 0.08, count: 4))
        let parameters = [Float](repeating: 0, count: 80)
        let blended = HumanoidBoxPhysicalFlowExperiment
            .continuedStructuredTrajectoryBoundary(
                parameters,
                progress: 0.125,
                initial: initial,
                armKnotCount: 4,
                blendKnotCount: 4,
                legResidualKnotCount: 4,
                maximumLegResidualAction: 0.5,
                torsoResidualKnotCount: 4,
                maximumTorsoResidualAction: 0.2,
                armAsymmetryKnotCount: 4,
                maximumArmAsymmetryAction: 0.5)
        XCTAssertEqual(blended.armDelta[0], 0.2, accuracy: 1e-6)
        XCTAssertEqual(blended.legBlend, 0.3, accuracy: 1e-6)
        XCTAssertEqual(blended.legResidual[0], -0.1, accuracy: 1e-6)
        XCTAssertEqual(blended.torsoResidual, 0.05, accuracy: 1e-6)
        XCTAssertEqual(blended.armAsymmetry[0], 0.04, accuracy: 1e-6)

        let faded = HumanoidBoxPhysicalFlowExperiment
            .continuedStructuredTrajectoryBoundary(
                parameters,
                progress: 0.25,
                initial: initial,
                armKnotCount: 4,
                blendKnotCount: 4,
                legResidualKnotCount: 4,
                maximumLegResidualAction: 0.5,
                torsoResidualKnotCount: 4,
                maximumTorsoResidualAction: 0.2,
                armAsymmetryKnotCount: 4,
                maximumArmAsymmetryAction: 0.5)
        XCTAssertEqual(faded.armDelta[0], 0, accuracy: 1e-6)
        XCTAssertEqual(faded.legBlend, 0, accuracy: 1e-6)
        XCTAssertEqual(faded.legResidual[0], 0, accuracy: 1e-6)
        XCTAssertEqual(faded.torsoResidual, 0, accuracy: 1e-6)
        XCTAssertEqual(faded.armAsymmetry[0], 0, accuracy: 1e-6)
    }

    func testSerializedRecedingSequenceReplaysPlanBoundaryEvolution() {
        let initial = HumanoidBoxPhysicalFlowExperiment
            .StructuredTrajectoryBoundary(
                armDelta: [Float](repeating: 0.4, count: 8),
                legBlend: 0,
                legResidual: [Float](repeating: 0, count: 10),
                torsoResidual: 0,
                armAsymmetry: [Float](repeating: 0, count: 4))
        let firstPlan = [Float](repeating: 0, count: 4)
        let secondPlan = [Float](repeating: 0.5, count: 4)
        let replay = HumanoidBoxPhysicalFlowExperiment
            .structuredTrajectorySequenceBoundaryReplay(
                [firstPlan, firstPlan, secondPlan],
                phaseSteps: [0, 1, 0],
                denominator: 4,
                initial: initial,
                armKnotCount: 1,
                blendKnotCount: 0,
                legResidualKnotCount: 0,
                maximumLegResidualAction: 0.5,
                torsoResidualKnotCount: 0,
                maximumTorsoResidualAction: 0.2,
                armAsymmetryKnotCount: 0,
                maximumArmAsymmetryAction: 0.5)

        XCTAssertEqual(replay.starts.count, 3)
        XCTAssertEqual(replay.starts[0], initial)
        XCTAssertEqual(replay.starts[1], initial)
        XCTAssertEqual(
            replay.starts[2]?.armDelta[0] ?? .nan,
            0.2, accuracy: 1e-6)
        XCTAssertEqual(
            replay.terminal?.armDelta[0] ?? .nan,
            0.275, accuracy: 1e-6)
    }

    func testRecedingValidationReservesSlotsForConstraintFrontiers() {
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment
                .prioritizedValidationParameters(
                    lossRanked: [[0], [1], [2], [3]],
                    frontiers: [[8], [9], [8]],
                    limit: 4),
            [[8], [9], [0], [1]])
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment
                .prioritizedValidationParameters(
                    lossRanked: [[0], [1]],
                    frontiers: [[0]],
                    limit: 2),
            [[0], [1]])
    }

    func testRecedingTerminalFootUnloadingSamplesPlanEndpoint() {
        XCTAssertFalse(
            HumanoidBoxPhysicalFlowExperiment.isRecedingPlanEndpoint(
                predictedStep: 14, activePlanSteps: 16))
        XCTAssertTrue(
            HumanoidBoxPhysicalFlowExperiment.isRecedingPlanEndpoint(
                predictedStep: 15, activePlanSteps: 16))
        XCTAssertTrue(
            HumanoidBoxPhysicalFlowExperiment.isRecedingPlanEndpoint(
                predictedStep: 1, activePlanSteps: 2))
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment
                .recedingTerminalAbsoluteStep(
                    committedStep: 0, horizon: 16, phaseStep: 0),
            16)
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment
                .recedingTerminalAbsoluteStep(
                    committedStep: 2, horizon: 16, phaseStep: 2),
            16)
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment
                .recedingTerminalAbsoluteStep(
                    committedStep: 4, horizon: 16, phaseStep: 0),
            20)
    }

    func testSwingFrontierPreservesRareDynamicModesForRepair() {
        typealias Score = HumanoidBoxPhysicalFlowExperiment
            .TargetDiscoverySwingFrontierScore
        let incumbent = Score(
            firstControlSafe: true,
            swingMilestonePassed: true,
            predictedPathSafe: false,
            terminalGoalFeasible: false,
            maximumFootAirTimeSeconds: 0.04,
            maximumSwingFootLiftMeters: 0.01,
            firstStablePathViolationStep: 8,
            terminalRecoveryViable: false,
            terminalRecoveryMargin: 0.8,
            graspQualityDwellSteps: 7,
            loss: 10)
        XCTAssertFalse(HumanoidBoxPhysicalFlowExperiment
            .swingFrontierIsBetter(
                Score(
                    firstControlSafe: false,
                    swingMilestonePassed: true,
                    predictedPathSafe: true,
                    terminalGoalFeasible: true,
                    maximumFootAirTimeSeconds: 0.2,
                    maximumSwingFootLiftMeters: 0.04,
                    firstStablePathViolationStep: nil,
                    terminalRecoveryViable: true,
                    terminalRecoveryMargin: 1,
                    graspQualityDwellSteps: 16,
                    loss: 1),
                than: incumbent))
        XCTAssertTrue(HumanoidBoxPhysicalFlowExperiment
            .swingFrontierIsBetter(
                Score(
                    firstControlSafe: true,
                    swingMilestonePassed: true,
                    predictedPathSafe: false,
                    terminalGoalFeasible: false,
                    maximumFootAirTimeSeconds: 0.04,
                    maximumSwingFootLiftMeters: 0.008,
                    firstStablePathViolationStep: 8,
                    terminalRecoveryViable: true,
                    terminalRecoveryMargin: 1,
                    graspQualityDwellSteps: 7,
                    loss: 20),
                than: incumbent))
        XCTAssertTrue(HumanoidBoxPhysicalFlowExperiment
            .swingFrontierIsBetter(
                Score(
                    firstControlSafe: true,
                    swingMilestonePassed: true,
                    predictedPathSafe: false,
                    terminalGoalFeasible: false,
                    maximumFootAirTimeSeconds: 0.08,
                    maximumSwingFootLiftMeters: 0.01,
                    firstStablePathViolationStep: 8,
                    terminalRecoveryViable: false,
                    terminalRecoveryMargin: 0.8,
                    graspQualityDwellSteps: 7,
                    loss: 20),
                than: incumbent))
        XCTAssertTrue(HumanoidBoxPhysicalFlowExperiment
            .swingFrontierIsBetter(
                Score(
                    firstControlSafe: true,
                    swingMilestonePassed: true,
                    predictedPathSafe: true,
                    terminalGoalFeasible: true,
                    maximumFootAirTimeSeconds: 0.04,
                    maximumSwingFootLiftMeters: 0.008,
                    firstStablePathViolationStep: nil,
                    terminalRecoveryViable: true,
                    terminalRecoveryMargin: 1,
                    graspQualityDwellSteps: 16,
                    loss: 30),
                than: incumbent))
        XCTAssertFalse(HumanoidBoxPhysicalFlowExperiment
            .swingFrontierIsBetter(
                Score(
                    firstControlSafe: true,
                    swingMilestonePassed: false,
                    predictedPathSafe: true,
                    terminalGoalFeasible: true,
                    maximumFootAirTimeSeconds: 0,
                    maximumSwingFootLiftMeters: 0,
                    firstStablePathViolationStep: nil,
                    terminalRecoveryViable: true,
                    terminalRecoveryMargin: 1,
                    graspQualityDwellSteps: 16,
                    loss: 1),
                than: incumbent))
        XCTAssertTrue(HumanoidBoxPhysicalFlowExperiment
            .swingFrontierIsBetter(
                Score(
                    firstControlSafe: true,
                    swingMilestonePassed: true,
                    predictedPathSafe: false,
                    terminalGoalFeasible: false,
                    maximumFootAirTimeSeconds: 0.04,
                    maximumSwingFootLiftMeters: 0.01,
                    firstStablePathViolationStep: 12,
                    terminalRecoveryViable: false,
                    terminalRecoveryMargin: 0.5,
                    graspQualityDwellSteps: 4,
                    loss: 20),
                than: incumbent))
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

    func testFeasibilityFrontierPreservesLongestCertifiedWindow() {
        typealias Score = HumanoidBoxPhysicalFlowExperiment
            .TargetDiscoveryFeasibilityFrontierScore
        let oneControl = Score(
            firstControlSafe: true,
            predictedPathSafe: true,
            maximumFeasibilityDwellSteps: 1,
            terminalGoalFeasible: false,
            terminalRecoveryViable: false,
            terminalRecoveryMargin: -2,
            loss: 20)
        XCTAssertTrue(HumanoidBoxPhysicalFlowExperiment
            .feasibilityFrontierIsBetter(
                oneControl,
                than: Score(
                    firstControlSafe: true,
                    predictedPathSafe: true,
                    maximumFeasibilityDwellSteps: 0,
                    terminalGoalFeasible: true,
                    terminalRecoveryViable: true,
                    terminalRecoveryMargin: 1,
                    loss: 1)))
        XCTAssertFalse(HumanoidBoxPhysicalFlowExperiment
            .feasibilityFrontierIsBetter(
                Score(
                    firstControlSafe: false,
                    predictedPathSafe: true,
                    maximumFeasibilityDwellSteps: 2,
                    terminalGoalFeasible: true,
                    terminalRecoveryViable: true,
                    terminalRecoveryMargin: 1,
                    loss: 1),
                than: oneControl))
        XCTAssertTrue(HumanoidBoxPhysicalFlowExperiment
            .feasibilityFrontierIsBetter(
                Score(
                    firstControlSafe: true,
                    predictedPathSafe: true,
                    maximumFeasibilityDwellSteps: 1,
                    terminalGoalFeasible: false,
                    terminalRecoveryViable: false,
                    terminalRecoveryMargin: -1,
                    loss: 30),
                than: oneControl))
    }

    func testRecedingPathRankingPrioritizesSafetyAndRecoveryBeforeGoalOrLoss() {
        typealias Score = HumanoidBoxPhysicalFlowExperiment
            .TargetDiscoveryPathScore
        let unsafeGoal = Score(
            firstControlSafe: true,
            commitPathSafe: true,
            terminalGoalFeasible: true,
            predictedPathSafe: false,
            firstStablePathViolationStep: 7,
            maximumFeasibilityDwellSteps: 12,
            terminalRecoveryViable: false,
            loss: 1)
        let pathSafeNoGoal = Score(
            firstControlSafe: true,
            commitPathSafe: true,
            terminalGoalFeasible: false,
            predictedPathSafe: true,
            firstStablePathViolationStep: nil,
            maximumFeasibilityDwellSteps: 1,
            terminalRecoveryViable: false,
            loss: 500)
        XCTAssertTrue(HumanoidBoxPhysicalFlowExperiment
            .pathScoreIsBetter(pathSafeNoGoal, than: unsafeGoal))
        XCTAssertFalse(HumanoidBoxPhysicalFlowExperiment
            .pathScoreIsBetter(unsafeGoal, than: pathSafeNoGoal))

        var recoverableNoGoal = pathSafeNoGoal
        recoverableNoGoal.terminalRecoveryViable = true
        var unrecoverableGoal = pathSafeNoGoal
        unrecoverableGoal.terminalGoalFeasible = true
        unrecoverableGoal.loss = 0
        XCTAssertTrue(HumanoidBoxPhysicalFlowExperiment.pathScoreIsBetter(
            recoverableNoGoal, than: unrecoverableGoal))
        XCTAssertFalse(HumanoidBoxPhysicalFlowExperiment.pathScoreIsBetter(
            unrecoverableGoal, than: recoverableNoGoal))

        var laterViolation = unsafeGoal
        laterViolation.terminalGoalFeasible = false
        var earlierViolation = laterViolation
        earlierViolation.firstStablePathViolationStep = 5
        earlierViolation.loss = 0
        XCTAssertTrue(HumanoidBoxPhysicalFlowExperiment.pathScoreIsBetter(
            laterViolation, than: earlierViolation))

        var unsafeFirstControl = laterViolation
        unsafeFirstControl.firstControlSafe = false
        unsafeFirstControl.predictedPathSafe = true
        unsafeFirstControl.firstStablePathViolationStep = nil
        XCTAssertFalse(HumanoidBoxPhysicalFlowExperiment.pathScoreIsBetter(
            unsafeFirstControl, than: laterViolation))

        var unsafeCommit = pathSafeNoGoal
        unsafeCommit.commitPathSafe = false
        unsafeCommit.terminalRecoveryViable = true
        unsafeCommit.terminalGoalFeasible = true
        unsafeCommit.loss = 0
        XCTAssertFalse(HumanoidBoxPhysicalFlowExperiment.pathScoreIsBetter(
            unsafeCommit, than: pathSafeNoGoal))
    }

    func testPreservedUpperBodySeedRequiresExplicitContinuationBoundary() {
        var configuration = HumanoidBoxPhysicalFlowConfiguration()
        configuration.preserveProvidedContinuationSeedUpperBody = true
        XCTAssertThrowsError(try configuration.validate())
        configuration.continueTrajectoryFromSourceTerminal = true
        XCTAssertNoThrow(try configuration.validate())
    }

    func testHardMinimumPenaltyCannotHideANearMiss() {
        XCTAssertTrue(HumanoidBoxPhysicalFlowExperiment.isUnsupported(
            pedestalContact: 0, destinationContact: 0, groundContact: 0))
        XCTAssertFalse(HumanoidBoxPhysicalFlowExperiment.isUnsupported(
            pedestalContact: 1, destinationContact: 0, groundContact: 0))
        XCTAssertFalse(HumanoidBoxPhysicalFlowExperiment.isUnsupported(
            pedestalContact: 0, destinationContact: 1, groundContact: 0))
        XCTAssertFalse(HumanoidBoxPhysicalFlowExperiment.isUnsupported(
            pedestalContact: 0, destinationContact: 0, groundContact: 1))
        XCTAssertTrue(HumanoidBoxPhysicalFlowExperiment
            .minimumDestinationProgressPassed(-0.2, minimum: 0))
        XCTAssertFalse(HumanoidBoxPhysicalFlowExperiment
            .minimumDestinationProgressPassed(0.01, minimum: 0.02))
        XCTAssertTrue(HumanoidBoxPhysicalFlowExperiment
            .minimumDestinationProgressPassed(0.02, minimum: 0.02))
        XCTAssertEqual(HumanoidBoxPhysicalFlowExperiment.hardMinimumPenalty(
            value: 0.375, minimum: 0.375, scale: 0.01), 0)
        XCTAssertEqual(HumanoidBoxPhysicalFlowExperiment.hardMinimumPenalty(
            value: 0.374, minimum: 0.375, scale: 0.01), 4.1,
            accuracy: 1e-5)
        XCTAssertGreaterThan(
            HumanoidBoxPhysicalFlowExperiment.hardMinimumPenalty(
                value: 0.35, minimum: 0.375, scale: 0.01), 4.1)
        XCTAssertEqual(HumanoidBoxPhysicalFlowExperiment.hardDwellPenalty(
            achieved: 8, required: 8), 0)
        XCTAssertEqual(HumanoidBoxPhysicalFlowExperiment.hardDwellPenalty(
            achieved: 7, required: 8), 1.125, accuracy: 1e-6)
        XCTAssertGreaterThan(
            HumanoidBoxPhysicalFlowExperiment.hardDwellPenalty(
                achieved: 7, required: 8), 1)
        XCTAssertGreaterThan(
            HumanoidBoxPhysicalFlowExperiment.hardDwellPenalty(
                achieved: 0, required: 8),
            HumanoidBoxPhysicalFlowExperiment.hardDwellPenalty(
                achieved: 7, required: 8))
        XCTAssertEqual(HumanoidBoxPhysicalFlowExperiment
            .hardViolationPenalty(violations: 0, totalSteps: 74), 0)
        XCTAssertEqual(HumanoidBoxPhysicalFlowExperiment
            .hardViolationPenalty(violations: 1, totalSteps: 74),
            4 + 1.0 / 74.0, accuracy: 1e-6)
        XCTAssertEqual(HumanoidBoxPhysicalFlowExperiment
            .feasibilityWindowPenalty([0, 0, 0], required: 8),
            4.625, accuracy: 1e-6)
        XCTAssertEqual(HumanoidBoxPhysicalFlowExperiment
            .feasibilityWindowPenalty(
                [0, 0, 0, 0, 0, 0, 0, 0], required: 8), 0)
        XCTAssertEqual(HumanoidBoxPhysicalFlowExperiment
            .feasibilityWindowPenalty(
                [0, 0, 0, 4.2, 0, 0, 0, 0], required: 8), 4.2,
            accuracy: 1e-6)

        var tail: [Float] = []
        for penalty: Float in [9, 7, 0.4, 0.2] {
            HumanoidBoxPhysicalFlowExperiment.appendFeasibilityPenalty(
                penalty, to: &tail, required: 2)
        }
        XCTAssertEqual(tail, [0.4, 0.2])
        XCTAssertEqual(HumanoidBoxPhysicalFlowExperiment
            .feasibilityWindowPenalty(tail, required: 2), 0.4,
            accuracy: 1e-6)
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

    func testRecedingWarmStartAdvancesEveryStructuredActionHead() {
        let armKnots = 4
        let blendKnots = 4
        let residualKnots = 4
        let torsoKnots = 4
        let asymmetryKnots = 4
        let armCount = 4 * armKnots
        var parameters = (0..<armCount).map {
            Float($0 + 1) / Float(2 * armCount)
        }
        parameters += [0.2, 0.4, 0.6, 0.8]
        parameters += (0..<(10 * residualKnots)).map {
            -0.4 + 0.8 * Float($0) / Float(10 * residualKnots - 1)
        }
        parameters += [-0.3, -0.1, 0.2, 0.4]
        parameters += (0..<(4 * asymmetryKnots)).map {
            -0.25 + 0.5 * Float($0) / Float(4 * asymmetryKnots - 1)
        }
        let shifted = HumanoidBoxPhysicalFlowExperiment
            .shiftedRecedingTrajectory(
                parameters, horizon: 4,
                armKnotCount: armKnots,
                blendKnotCount: blendKnots,
                legResidualKnotCount: residualKnots,
                maximumLegResidualAction: 0.5,
                torsoResidualKnotCount: torsoKnots,
                maximumTorsoResidualAction: 0.2,
                armAsymmetryKnotCount: asymmetryKnots,
                maximumArmAsymmetryAction: 0.25)
        XCTAssertEqual(shifted.count, parameters.count)
        XCTAssertTrue(shifted.allSatisfy { $0 >= -1 && $0 <= 1 })

        let oldArm = HumanoidBoxCarryActuationProbe.trajectoryArmDelta(
            Array(parameters.prefix(armCount)), knotCount: armKnots,
            progress: 0.5)
        let newArm = HumanoidBoxCarryActuationProbe.trajectoryArmDelta(
            Array(shifted.prefix(armCount)), knotCount: armKnots,
            progress: 0.25)
        for component in 0..<8 {
            XCTAssertEqual(newArm[component], oldArm[component],
                           accuracy: 1e-6)
        }
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment.legBlendFraction(
                shifted, progress: 0.25,
                armParameterCount: armCount, knotCount: blendKnots),
            HumanoidBoxPhysicalFlowExperiment.legBlendFraction(
                parameters, progress: 0.5,
                armParameterCount: armCount, knotCount: blendKnots),
            accuracy: 1e-6)
        for action in 0..<10 {
            XCTAssertEqual(
                HumanoidBoxPhysicalFlowExperiment.legResidualAction(
                    shifted, action: action, progress: 0.25,
                    armParameterCount: armCount,
                    blendKnotCount: blendKnots,
                    residualKnotCount: residualKnots,
                    maximumAction: 0.5),
                HumanoidBoxPhysicalFlowExperiment.legResidualAction(
                    parameters, action: action, progress: 0.5,
                    armParameterCount: armCount,
                    blendKnotCount: blendKnots,
                    residualKnotCount: residualKnots,
                    maximumAction: 0.5),
                accuracy: 1e-6)
        }
        XCTAssertEqual(
            HumanoidBoxPhysicalFlowExperiment.torsoResidualAction(
                shifted, progress: 0.25,
                armParameterCount: armCount,
                blendKnotCount: blendKnots,
                legResidualKnotCount: residualKnots,
                torsoResidualKnotCount: torsoKnots,
                maximumAction: 0.2),
            HumanoidBoxPhysicalFlowExperiment.torsoResidualAction(
                parameters, progress: 0.5,
                armParameterCount: armCount,
                blendKnotCount: blendKnots,
                legResidualKnotCount: residualKnots,
                torsoResidualKnotCount: torsoKnots,
                maximumAction: 0.2),
            accuracy: 1e-6)
        for action in 0..<4 {
            XCTAssertEqual(
                HumanoidBoxPhysicalFlowExperiment.armAsymmetryAction(
                    shifted, action: action, progress: 0.25,
                    armParameterCount: armCount,
                    blendKnotCount: blendKnots,
                    legResidualKnotCount: residualKnots,
                    torsoResidualKnotCount: torsoKnots,
                    asymmetryKnotCount: asymmetryKnots,
                    maximumAction: 0.25),
                HumanoidBoxPhysicalFlowExperiment.armAsymmetryAction(
                    parameters, action: action, progress: 0.5,
                    armParameterCount: armCount,
                    blendKnotCount: blendKnots,
                    legResidualKnotCount: residualKnots,
                    torsoResidualKnotCount: torsoKnots,
                    asymmetryKnotCount: asymmetryKnots,
                    maximumAction: 0.25),
                accuracy: 1e-6)
        }
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

    func testLegResidualTrajectoryResamplingPreservesSignal() {
        var parameters = [Float](repeating: 0, count: 20 + 3 + 20)
        parameters[20 + 3 + 2] = 0.6
        parameters[20 + 3 + 10 + 2] = -0.4
        let resampled = HumanoidBoxPhysicalFlowExperiment
            .resampledLegResidualTrajectory(
                parameters, armParameterCount: 20,
                blendKnotCount: 3, sourceResidualKnotCount: 2,
                targetResidualKnotCount: 4)
        XCTAssertEqual(resampled.count, 20 + 3 + 40)
        for knot in 0..<4 {
            let progress = Float(knot + 1) / 4
            let original = HumanoidBoxPhysicalFlowExperiment
                .legResidualAction(
                    parameters, action: 2, progress: progress,
                    armParameterCount: 20, blendKnotCount: 3,
                    residualKnotCount: 2, maximumAction: 0.25)
            let converted = HumanoidBoxPhysicalFlowExperiment
                .legResidualAction(
                    resampled, action: 2, progress: progress,
                    armParameterCount: 20, blendKnotCount: 3,
                    residualKnotCount: 4, maximumAction: 0.25)
            XCTAssertEqual(converted, original, accuracy: 1e-6)
        }
    }

    func testResidualRangeResamplingPreservesPhysicalAction() {
        let source: [Float] = [
            0, 0, 0, 0, // one arm knot
            0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, // one leg knot
        ]
        let converted = HumanoidBoxPhysicalFlowExperiment
            .resampledLegResidualTrajectory(
                source,
                armParameterCount: 4,
                blendKnotCount: 0,
                sourceResidualKnotCount: 1,
                targetResidualKnotCount: 1,
                sourceResidualMaximumAction: 0.25,
                targetResidualMaximumAction: 0.5)
        XCTAssertEqual(converted[4], 0.4, accuracy: 1e-6)
        let before = HumanoidBoxPhysicalFlowExperiment.legResidualAction(
            source, action: 0, progress: 1,
            armParameterCount: 4, blendKnotCount: 0,
            residualKnotCount: 1, maximumAction: 0.25)
        let after = HumanoidBoxPhysicalFlowExperiment.legResidualAction(
            converted, action: 0, progress: 1,
            armParameterCount: 4, blendKnotCount: 0,
            residualKnotCount: 1, maximumAction: 0.5)
        XCTAssertEqual(after, before, accuracy: 1e-6)
    }

    func testArmAsymmetryRangeResamplingPreservesPhysicalAction() {
        let source: [Float] = [
            0, 0, 0, 0, // one symmetric arm knot
            0.8, 0, 0, 0, // one antisymmetric arm knot
        ]
        let converted = HumanoidBoxPhysicalFlowExperiment
            .resampledLegResidualTrajectory(
                source,
                armParameterCount: 4,
                blendKnotCount: 0,
                sourceResidualKnotCount: 0,
                targetResidualKnotCount: 0,
                sourceArmAsymmetryKnotCount: 1,
                targetArmAsymmetryKnotCount: 1,
                sourceArmAsymmetryMaximumAction: 0.25,
                targetArmAsymmetryMaximumAction: 0.5)
        XCTAssertEqual(converted[4], 0.4, accuracy: 1e-6)
        let before = HumanoidBoxPhysicalFlowExperiment.armAsymmetryAction(
            source, action: 0, progress: 1,
            armParameterCount: 4, blendKnotCount: 0,
            legResidualKnotCount: 0, torsoResidualKnotCount: 0,
            asymmetryKnotCount: 1, maximumAction: 0.25)
        let after = HumanoidBoxPhysicalFlowExperiment.armAsymmetryAction(
            converted, action: 0, progress: 1,
            armParameterCount: 4, blendKnotCount: 0,
            legResidualKnotCount: 0, torsoResidualKnotCount: 0,
            asymmetryKnotCount: 1, maximumAction: 0.5)
        XCTAssertEqual(after, before, accuracy: 1e-6)
    }

    func testTorsoResidualSplineIsBoundedAndBackwardsCompatible() {
        let legacy = [Float](repeating: 0, count: 20 + 3 + 40)
        var upgraded = HumanoidBoxPhysicalFlowExperiment
            .resampledLegResidualTrajectory(
                legacy, armParameterCount: 20, blendKnotCount: 3,
                sourceResidualKnotCount: 4,
                targetResidualKnotCount: 4,
                sourceTorsoResidualKnotCount: 0,
                targetTorsoResidualKnotCount: 3)
        XCTAssertEqual(upgraded.count, legacy.count + 3)
        XCTAssertEqual(Array(upgraded.suffix(3)), [0, 0, 0])
        upgraded[upgraded.count - 3] = 0.2
        upgraded[upgraded.count - 2] = -0.4
        upgraded[upgraded.count - 1] = 1
        XCTAssertEqual(HumanoidBoxPhysicalFlowExperiment.torsoResidualAction(
            upgraded, progress: 0, armParameterCount: 20,
            blendKnotCount: 3, legResidualKnotCount: 4,
            torsoResidualKnotCount: 3, maximumAction: 0.25),
            0, accuracy: 1e-6)
        XCTAssertEqual(HumanoidBoxPhysicalFlowExperiment.torsoResidualAction(
            upgraded, progress: 1, armParameterCount: 20,
            blendKnotCount: 3, legResidualKnotCount: 4,
            torsoResidualKnotCount: 3, maximumAction: 0.25),
            0.25, accuracy: 1e-6)

        var configuration = HumanoidBoxPhysicalFlowConfiguration()
        configuration.torsoResidualKnotCount = -1
        XCTAssertThrowsError(try configuration.validate())
        configuration.torsoResidualKnotCount = 3
        configuration.maximumTorsoResidualAction = 1.001
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

    func testFlowDistillationUsesBehaviorPreservingOutputHeadUpdateByDefault()
        throws
    {
        let configuration = HumanoidBoxFlowDistillationConfiguration()
        XCTAssertTrue(configuration.trainOutputHeadsOnly)
        try configuration.validate()
    }

    func testActionChunkConfigurationRejectsInvalidSafetyBounds() throws {
        var configuration = HumanoidBoxFlowActionChunkConfiguration()
        XCTAssertNoThrow(try configuration.validate())
        configuration.horizon = 0
        XCTAssertThrowsError(try configuration.validate())
        configuration.horizon = 8
        configuration.maximumResidualAction = 2.001
        XCTAssertThrowsError(try configuration.validate())
        configuration.maximumResidualAction = 2
        configuration.trainingReplayCount = 0
        XCTAssertThrowsError(try configuration.validate())
        configuration.trainingReplayCount = 2
        configuration.trainingActionNoiseStandardDeviation = -0.001
        XCTAssertThrowsError(try configuration.validate())
        configuration.trainingActionNoiseStandardDeviation = 0.001
        configuration.exactReplayRowWeight = 0.99
        XCTAssertThrowsError(try configuration.validate())
        configuration.exactReplayRowWeight = 1
        configuration.exactFineTuneEpochs = -1
        XCTAssertThrowsError(try configuration.validate())
        configuration.exactFineTuneEpochs = 1
        configuration.exactFineTuneLearningRate = 0
        XCTAssertThrowsError(try configuration.validate())
        configuration.exactFineTuneLearningRate = 1e-5
        configuration.exactActionBiasWeight = -1
        XCTAssertThrowsError(try configuration.validate())
    }

    func testActionChunkMetadataBindsItsExactTeacherAndTrainingContract()
        throws
    {
        let sourceFingerprint = String(repeating: "a", count: 64)
        let flowFingerprint = String(repeating: "b", count: 64)
        let configuration = HumanoidBoxFlowActionChunkConfiguration(seed: 77)
        var metadata = HumanoidBoxFlowActionChunkMetadata(
            schemaVersion: 2,
            observationDimension: 3,
            actionDimension: 19,
            horizon: configuration.horizon,
            hiddenDimension: configuration.hiddenDimension,
            maximumResidualAction: configuration.maximumResidualAction,
            normalizeObservations: true,
            sourceCheckpoint: "/tmp/source-checkpoint",
            flowReport: "/tmp/teacher-flow.json",
            sourceCheckpointFingerprint: sourceFingerprint,
            flowReportFingerprint: flowFingerprint,
            trainingConfiguration: configuration,
            normalizer: RunningNormalizerSnapshot(
                count: 1, mean: [0, 0, 0], variance: [1, 1, 1]))
        XCTAssertNoThrow(try metadata.validateForLoading())
        XCTAssertTrue(metadata.isBound(
            toSourceCheckpoint: "/tmp/renamed-source",
            sourceFingerprint: sourceFingerprint,
            flowReportPath: "/tmp/renamed-flow.json",
            flowFingerprint: flowFingerprint))
        XCTAssertFalse(metadata.isBound(
            toSourceCheckpoint: metadata.sourceCheckpoint,
            sourceFingerprint: String(repeating: "c", count: 64),
            flowReportPath: metadata.flowReport,
            flowFingerprint: flowFingerprint))

        metadata.flowReportFingerprint = nil
        XCTAssertThrowsError(try metadata.validateForLoading())
        metadata.flowReportFingerprint = flowFingerprint
        var mismatchedConfiguration = configuration
        mismatchedConfiguration.horizon += 1
        metadata.trainingConfiguration = mismatchedConfiguration
        XCTAssertThrowsError(try metadata.validateForLoading())
    }

    func testLegacyActionChunkMetadataRequiresItsRecordedDependencyPaths()
        throws
    {
        let metadata = HumanoidBoxFlowActionChunkMetadata(
            schemaVersion: 1,
            observationDimension: 2,
            actionDimension: 19,
            horizon: 4,
            hiddenDimension: 32,
            maximumResidualAction: 0.5,
            normalizeObservations: true,
            sourceCheckpoint: "/tmp/legacy-source",
            flowReport: "/tmp/legacy-flow.json",
            normalizer: RunningNormalizerSnapshot(
                count: 0, mean: [0, 0], variance: [1, 1]))
        XCTAssertNoThrow(try metadata.validateForLoading())
        XCTAssertTrue(metadata.isBound(
            toSourceCheckpoint: "/tmp/legacy-source",
            sourceFingerprint: "ignored",
            flowReportPath: "/tmp/legacy-flow.json",
            flowFingerprint: "ignored"))
        XCTAssertFalse(metadata.isBound(
            toSourceCheckpoint: "/tmp/substituted-source",
            sourceFingerprint: "ignored",
            flowReportPath: "/tmp/legacy-flow.json",
            flowFingerprint: "ignored"))
    }

    func testBalanceArtifactUsesGenerationStepsAsFeedbackHorizon() throws {
        let targetSteps = 32
        var object: [String: Any] = [
            "physicalBalanceGatePassed": true,
            "targetExecutionSteps": targetSteps,
            "targetGenerationSteps": 8,
            "targetGeneratingTrajectory": [Float(0)],
            "targetGeneratingTrajectorySequence": Array(
                repeating: [Float(0)], count: targetSteps),
            "targetCommittedTrace": (0...targetSteps).map { step in
                step == 0
                    ? ["step": step]
                    : [
                        "step": step,
                        "appliedNormalizedActions": Array(
                            repeating: Float(step), count: 19),
                    ]
            },
            "recedingLocomotionCheckpointDirectory":
                "/tmp/h1-flat-checkpoint",
            "recedingLocomotionCommandSpeed": Float(0.2),
            "recedingForwardOnlyBaseCommand": true,
            "legBlendKnotCount": 1,
            "finalCarryDistanceMeters": Float(0.38),
            "sourceStages": [[
                "controlSteps": 1,
                "policyOnly": true,
            ]],
            "sourceAppliedActions": [[Float](repeating: 0, count: 19)],
            "sourceWarmupAppliedActions": [
                [Float](repeating: 0.1, count: 19),
                [Float](repeating: 0.2, count: 19),
            ],
        ]
        let data = try JSONSerialization.data(withJSONObject: object)
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("json")
        try data.write(to: path, options: .atomic)
        defer { try? FileManager.default.removeItem(at: path) }

        let artifact = try HumanoidBoxFlowDistillation.loadArtifact(
            path.path, allowPhysicalBalance: true)
        XCTAssertTrue(artifact.physicalBalanceOnly)
        XCTAssertFalse(artifact.reusableFrontier)
        XCTAssertEqual(artifact.targetSteps, targetSteps)
        XCTAssertEqual(artifact.targetDuration, 8)
        XCTAssertEqual(artifact.targetTrajectorySequence?.count, targetSteps)
        XCTAssertEqual(artifact.targetAppliedActions?.count, targetSteps)
        XCTAssertEqual(artifact.targetAppliedActions?.first?.count, 19)
        XCTAssertEqual(artifact.targetAppliedActions?.last?.first,
                       Float(targetSteps))
        XCTAssertEqual(artifact.sourceAppliedActions?.count, 1)
        XCTAssertEqual(artifact.sourceAppliedActions?.first?.count, 19)
        XCTAssertEqual(artifact.sourceWarmupAppliedActions?.count, 2)
        XCTAssertEqual(
            artifact.sourceWarmupAppliedActions?.last?.first, 0.2)

        // A legacy balance flag must not bypass a newly declared physical
        // grasp requirement. Only a full measured dwell may make the artifact
        // eligible for teacher distillation.
        object["requiredGraspQuality"] = Float(0.67)
        object["maximumGraspQualityThresholdDwellSteps"] = 0
        try JSONSerialization.data(withJSONObject: object)
            .write(to: path, options: .atomic)
        XCTAssertThrowsError(try HumanoidBoxFlowDistillation.loadArtifact(
            path.path, allowPhysicalBalance: true))
        object["maximumGraspQualityThresholdDwellSteps"] = targetSteps
        try JSONSerialization.data(withJSONObject: object)
            .write(to: path, options: .atomic)
        XCTAssertNoThrow(try HumanoidBoxFlowDistillation.loadArtifact(
            path.path, allowPhysicalBalance: true))
        object["physicalBalanceGatePassed"] = false
        object["targetFinitePrefixGatePassed"] = true
        object["recedingHorizonSteps"] = 8
        object["targetPredictedRecoveryPathSafe"] = false
        try JSONSerialization.data(withJSONObject: object)
            .write(to: path, options: .atomic)
        let finitePrefix = try HumanoidBoxFlowDistillation.loadArtifact(
            path.path, allowPhysicalBalance: true)
        XCTAssertFalse(finitePrefix.reusableFrontier)

        object["targetPredictedRecoveryPathSafe"] = true
        try JSONSerialization.data(withJSONObject: object)
            .write(to: path, options: .atomic)
        let recoverySafe = try HumanoidBoxFlowDistillation.loadArtifact(
            path.path, allowPhysicalBalance: true)
        XCTAssertTrue(recoverySafe.reusableFrontier)

        object["targetReusableFrontierGatePassed"] = false
        try JSONSerialization.data(withJSONObject: object)
            .write(to: path, options: .atomic)
        let explicitlyNonReusable = try HumanoidBoxFlowDistillation
            .loadArtifact(path.path, allowPhysicalBalance: true)
        XCTAssertFalse(explicitlyNonReusable.reusableFrontier)
        XCTAssertEqual(
            artifact.targetLocomotionCheckpointDirectory,
            "/tmp/h1-flat-checkpoint")
        XCTAssertEqual(artifact.targetLocomotionCommandSpeed, 0.2)
        XCTAssertTrue(artifact.targetForwardOnlyBaseCommand)
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

    func testGraspFeedbackDampedLeastSquaresTracksCartesianDelta() {
        let delta = F3(0.02, -0.01, 0.03)
        let damping: Float = 0.01
        let jointDelta =
            HumanoidBoxPhysicalFlowExperiment
                .dampedLeastSquaresJointDelta(
                    jacobian: [
                        F3(1, 0, 0),
                        F3(0, 1, 0),
                        F3(0, 0, 1),
                        F3(0, 0, 0),
                    ],
                    taskDelta: delta,
                    damping: damping)
        let scale = 1 / (1 + damping * damping)
        XCTAssertEqual(jointDelta.count, 4)
        XCTAssertEqual(jointDelta[0], scale * delta.x, accuracy: 1e-6)
        XCTAssertEqual(jointDelta[1], scale * delta.y, accuracy: 1e-6)
        XCTAssertEqual(jointDelta[2], scale * delta.z, accuracy: 1e-6)
        XCTAssertEqual(jointDelta[3], 0, accuracy: 1e-6)
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
        XCTAssertTrue(HumanoidBoxStableCarryVerifier.isFinalSuccess(
            previouslyFailed: false, sample: stable,
            maximumStableCarryDistanceMeters: 0.36,
            requiredCarryDistanceMeters: 0.35,
            maximumStableDestinationProgressMeters: -0.02,
            requiredDestinationProgressMeters: 0))
        XCTAssertFalse(HumanoidBoxStableCarryVerifier.isFinalSuccess(
            previouslyFailed: false, sample: stable,
            maximumStableCarryDistanceMeters: 0.36,
            requiredCarryDistanceMeters: 0.35,
            maximumStableDestinationProgressMeters: 0.019,
            requiredDestinationProgressMeters: 0.02))
        XCTAssertTrue(HumanoidBoxStableCarryVerifier.isFinalSuccess(
            previouslyFailed: false, sample: stable,
            maximumStableCarryDistanceMeters: 0.36,
            requiredCarryDistanceMeters: 0.35,
            maximumStableDestinationProgressMeters: 0.02,
            requiredDestinationProgressMeters: 0.02))
        var slippingGrasp = stable
        slippingGrasp.graspQuality = 0.6
        XCTAssertFalse(HumanoidBoxStableCarryVerifier.isStable(
            previouslyFailed: false, sample: slippingGrasp,
            requiredGraspQuality: 0.67))
        XCTAssertTrue(HumanoidBoxStableCarryVerifier.isStable(
            previouslyFailed: false, sample: stable,
            requiredGraspQuality: 0.67))

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
