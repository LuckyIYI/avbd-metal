import XCTest
import simd
@testable import AVBDCore

final class HumanoidBoxCarryTests: XCTestCase {
    func testImitationMilestoneIsIndependentFromTaskSuccess() throws {
        let task = try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: 2, seed: 60, observationNoise: false))
        var result = RLStepBatch(spec: task.spec)
        result.imitationMilestones[0] = true
        XCTAssertTrue(result.imitationMilestones[0])
        XCTAssertFalse(result.successes[0])
        try result.validate(for: task.spec)
        result.clearSignals()
        XCTAssertEqual(result.imitationMilestones, [false, false])
    }

    func testTerminalHandContactComesFromPhysicalManifold() throws {
        let env = try HumanoidWalkEnv(
            numEnvironments: 1, seed: 61, includeProjectile: true,
            projectileDimensions: F3(repeating: 0.10),
            projectileMass: 0.2, projectileFriction: 1,
            controlProfile: .isaacLab)
        let hand = env.manipulationStates()[0].leftHand.position
        env.placeCarryBoxes(environmentIDs: [0], positions: [hand])
        env.step(
            normalizedActions: ContiguousArray(repeating: 0, count: 19),
            decimation: 1, clampActions: false,
            clampTargetsToLimits: false)
        let contacts = env.boxHandContacts()
        XCTAssertTrue(contacts.left[0])
        XCTAssertFalse(contacts.right[0])
    }

    func testBoxSupportContactComesFromFinitePhysicalGeometry() throws {
        let pedestalSize = F3(0.36, 0.50, 0.66)
        let env = try HumanoidWalkEnv(
            numEnvironments: 1, seed: 610, includeProjectile: true,
            projectileDimensions: F3(0.22, 0.30, 0.24),
            projectileMass: 2, projectileFriction: 1.2,
            carryPedestalSize: pedestalSize,
            carryPedestalCenter: F3(0.55, 0, 0.33),
            controlProfile: .isaacLab)
        let actions = ContiguousArray(repeating: Float(0), count: 19)

        env.placeCarryBoxes(
            environmentIDs: [0], positions: [F3(0.55, 0, 0.78)])
        env.step(normalizedActions: actions, decimation: 2,
                 clampActions: false, clampTargetsToLimits: false)
        XCTAssertTrue(env.boxSupportContacts().pedestal[0])

        // This is the same vertical position, but beyond the finite
        // pedestal.  A height plane would report a false drop; the physical
        // manifold correctly reports no support.
        env.placeCarryBoxes(
            environmentIDs: [0], positions: [F3(1.20, 0, 0.78)])
        env.step(normalizedActions: actions, decimation: 1,
                 clampActions: false, clampTargetsToLimits: false)
        let away = env.boxSupportContacts()
        XCTAssertFalse(away.pedestal[0])
        XCTAssertFalse(away.ground[0])
    }

    func testSourceAndDestinationTableContactsRemainDistinct() throws {
        let top = HumanoidBoxCarryTask.pedestalDimensions
        let env = try HumanoidWalkEnv(
            numEnvironments: 1, seed: 612, includeProjectile: true,
            projectileDimensions: HumanoidBoxCarryTask.boxDimensions,
            projectileMass: 2, projectileFriction: 1.2,
            carryPedestalSize: top,
            carryPedestalCenter: F3(0.68, 0, 0.63),
            carryPedestalLegs: true,
            carryDestinationPedestalSize: top,
            carryDestinationPedestalCenter: F3(1.43, 0, 0.63),
            carryDestinationPedestalLegs: true,
            controlProfile: .isaacLab)
        let actions = ContiguousArray(repeating: Float(0), count: 19)

        env.placeCarryBoxes(
            environmentIDs: [0], positions: [F3(0.55, 0, 0.78)])
        env.step(normalizedActions: actions, decimation: 2,
                 clampActions: false, clampTargetsToLimits: false)
        var contacts = env.boxCarrySupportContacts()
        XCTAssertTrue(contacts.source[0])
        XCTAssertFalse(contacts.destination[0])

        env.placeCarryBoxes(
            environmentIDs: [0], positions: [F3(1.30, 0, 0.78)])
        env.step(normalizedActions: actions, decimation: 2,
                 clampActions: false, clampTargetsToLimits: false)
        contacts = env.boxCarrySupportContacts()
        XCTAssertFalse(contacts.source[0])
        XCTAssertTrue(contacts.destination[0])
    }

    func testFinalPlacementRequiresDestinationSupportAndPhysicalRelease() {
        func qualifies(
            destinationContact: Bool = true,
            leftHandContact: Bool = false,
            linearSpeed: Float = 0
        ) -> Bool {
            HumanoidBoxCarryTask.isStablePlacement(
                lifted: true, destinationContact: destinationContact,
                planarDistance: 0.02, heightError: 0.01,
                uprightAlignment: 0.99,
                leftHandContact: leftHandContact, rightHandContact: false,
                linearSpeed: linearSpeed, angularSpeed: 0)
        }

        XCTAssertTrue(qualifies())
        XCTAssertFalse(qualifies(destinationContact: false))
        XCTAssertFalse(qualifies(leftHandContact: true))
        XCTAssertFalse(qualifies(
            linearSpeed: HumanoidBoxCarryTask.placementMaximumLinearSpeed
                + 0.01))
    }

    func testCarryTableLeavesCentralLegApproachLane() throws {
        let task = try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: 1, seed: 611, observationNoise: false,
            minimumTrainingStationDistance: 0.55,
            evaluationStationDistance: 0.55,
            stationDistanceCurriculumControlSteps: 0))
        _ = try task.reset(seed: 611)
        let ref = task.environment.refs[0]
        let top = try XCTUnwrap(ref.carryPedestal)
        let destinationTop = try XCTUnwrap(ref.carryDestinationPedestal)
        XCTAssertEqual(ref.carryPedestalParts.count, 4)
        XCTAssertEqual(ref.carryPedestalPartOffsets.count, 4)
        XCTAssertEqual(ref.carryDestinationPedestalParts.count, 4)
        XCTAssertEqual(
            task.environment.scene.bodies[top].size,
            HumanoidBoxCarryTask.pedestalDimensions)

        let topState = task.environment.solver.bodyStates([top])[0]
        XCTAssertEqual(topState.position.x, 0.68, accuracy: 1e-5)
        XCTAssertEqual(
            topState.position.z,
            HumanoidBoxCarryTask.tabletopCenterHeight,
            accuracy: 1e-5)
        let nearTableEdge = topState.position.x
            - 0.5 * HumanoidBoxCarryTask.pedestalDimensions.x
        let pregraspX = 0.55 - HumanoidBoxCarryTask.pregraspOffset
        XCTAssertGreaterThan(nearTableEdge - pregraspX, 0.15)
        let destinationState = task.environment.solver.bodyStates(
            [destinationTop])[0]
        XCTAssertEqual(destinationState.position.x, 0.68, accuracy: 1e-5)
        XCTAssertEqual(destinationState.position.y, 0.75, accuracy: 1e-5)
        XCTAssertGreaterThan(
            destinationState.position.y
                - 0.5 * HumanoidBoxCarryTask.pedestalDimensions.y
                - (topState.position.y
                    + 0.5 * HumanoidBoxCarryTask.pedestalDimensions.y),
            0.10)
        XCTAssertEqual(
            task.currentPlacementTarget(environment: 0),
            F3(0.55, 0.75, 0.78))

        let legStates = task.environment.solver.bodyStates(
            ref.carryPedestalParts)
        XCTAssertTrue(legStates.allSatisfy { abs($0.position.y) > 0.24 })
        for (part, state) in zip(ref.carryPedestalParts, legStates) {
            XCTAssertEqual(
                state.position.z
                    - 0.5 * task.environment.scene.bodies[part].size.z,
                0, accuracy: 1e-5)
        }
    }

    func testCarryTaskPreservesFlatObservationPrefixForTransfer() throws {
        let task = try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: 2, seed: 62, observationNoise: false))
        let mapping = try XCTUnwrap(
            task.initializationObservationSourceIndices(sourceDimension: 69))
        XCTAssertEqual(mapping.count, HumanoidBoxCarryTask.observationDimension)
        XCTAssertEqual(Array(mapping.prefix(69)), (0..<69).map(Optional.some))
        XCTAssertTrue(mapping.dropFirst(69).allSatisfy { $0 == nil })

        let observation = try task.reset(seed: 900)
        XCTAssertEqual(
            observation.policy.count,
            2 * HumanoidBoxCarryTask.observationDimension)
        let boxes = task.environment.boxStates()
        XCTAssertEqual(boxes[0].position.x, 1.40, accuracy: 1e-5)
        XCTAssertEqual(boxes[0].position.z, 0.78, accuracy: 1e-5)
    }

    func testCarryTaskTransfersGoalLocomotionToExplicitNavigationInputs()
        throws
    {
        let task = try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: 1, seed: 621, observationNoise: false))
        let mapping = try XCTUnwrap(
            task.initializationObservationSourceIndices(
                sourceDimension: HumanoidIsaacVelocityTask
                    .goalObservationDimension))

        XCTAssertEqual(mapping.count, HumanoidBoxCarryTask.observationDimension)
        XCTAssertEqual(
            Array(mapping.prefix(HumanoidIsaacVelocityTask.observationDimension)),
            (0..<HumanoidIsaacVelocityTask.observationDimension)
                .map(Optional.some))
        XCTAssertEqual(mapping[103], 69)
        XCTAssertEqual(mapping[104], 70)
        for index in HumanoidIsaacVelocityTask.observationDimension..<103 {
            XCTAssertNil(mapping[index])
        }

        let observation = try task.reset(seed: 622)
        XCTAssertEqual(observation.policy[103], 0, accuracy: 1e-6)
        XCTAssertEqual(observation.policy[104], 0, accuracy: 1e-6)
    }

    func testCarryCurriculumExpandsStationWithoutChangingPhysicsTask() throws {
        let environments = 2
        let task = try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: environments, seed: 63,
            observationNoise: false,
            minimumTrainingStationDistance: 0.55,
            evaluationStationDistance: 1.40,
            stationDistanceCurriculumControlSteps: 100))
        task.setTrainingMode(true)
        task.setTrainingProgress(environmentSteps: 0)
        _ = try task.reset(seed: 901)
        let near = task.environment.boxStates().map(\.position.x)
        XCTAssertTrue(near.allSatisfy { $0 >= 0.55 && $0 <= 0.61 })

        task.setTrainingProgress(environmentSteps: 100 * environments)
        _ = try task.reset(seed: 901)
        let far = task.environment.boxStates().map(\.position.x)
        XCTAssertTrue(far.allSatisfy { $0 >= 1.34 && $0 <= 1.40 })
    }

    func testNearCarryCurriculumStillRequiresARealApproach() throws {
        let task = try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: 2, seed: 631,
            observationNoise: false,
            minimumTrainingStationDistance: 0.55,
            evaluationStationDistance: 0.55,
            stationDistanceCurriculumControlSteps: 0))
        task.setTrainingMode(true)
        let observation = try task.reset(seed: 902)

        for environment in 0..<2 {
            let base = environment * HumanoidBoxCarryTask.observationDimension
            XCTAssertEqual(observation.policy[base + 86], 1,
                "near-box reset must remain in locomotion mode")
            XCTAssertEqual(observation.policy[base + 87], 0)
        }
        XCTAssertEqual(task.policyExpertGates(observation.policy), [0, 0])
        let states = task.environment.states()
        let boxes = task.environment.boxStates()
        for environment in 0..<2 {
            XCTAssertGreaterThan(
                boxes[environment].position.x - states[environment].root.position.x,
                HumanoidBoxCarryTask.pregraspOffset
                    + HumanoidBoxCarryTask.manipulationEntryDistance)
        }
    }

    func testCarryCurriculumExpandsDistanceAndRetentionToFinalEvaluation() throws {
        let environments = 2
        let task = try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: environments, seed: 65,
            observationNoise: false,
            minimumTrainingLiftClearance: 0.001,
            liftClearance: 0.04,
            liftClearanceCurriculumControlSteps: 100,
            minimumTrainingCarryDistance: 0.05,
            carryDistance: 0.75,
            carryDistanceCurriculumControlSteps: 100,
            minimumTrainingSuccessDwellSteps: 1,
            successDwellSteps: 10,
            successDwellCurriculumControlSteps: 100))
        task.setTrainingMode(true)
        task.setTrainingProgress(environmentSteps: 0)
        XCTAssertEqual(task.currentLiftClearance, 0.001, accuracy: 1e-6)
        XCTAssertEqual(task.currentCarryDistance, 0.05, accuracy: 1e-6)
        XCTAssertEqual(task.currentSuccessDwellSteps, 1)

        task.setTrainingProgress(environmentSteps: 50 * environments)
        XCTAssertEqual(task.currentLiftClearance, 0.0205, accuracy: 1e-6)
        XCTAssertEqual(task.currentCarryDistance, 0.40, accuracy: 1e-6)
        XCTAssertEqual(task.currentSuccessDwellSteps, 5)

        task.setTrainingProgress(environmentSteps: 100 * environments)
        XCTAssertEqual(task.currentLiftClearance, 0.04, accuracy: 1e-6)
        XCTAssertEqual(task.currentCarryDistance, 0.75, accuracy: 1e-6)
        XCTAssertEqual(task.currentSuccessDwellSteps, 10)

        task.setTrainingMode(false)
        task.setTrainingProgress(environmentSteps: 0)
        XCTAssertEqual(task.currentLiftClearance, 0.04, accuracy: 1e-6)
        XCTAssertEqual(task.currentCarryDistance, 0.75, accuracy: 1e-6)
        XCTAssertEqual(task.currentSuccessDwellSteps, 10)
    }

    func testManipulationAndCarryUseDisjointPolicyExperts() throws {
        let task = try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: 2, seed: 66, observationNoise: false))
        var observations = ContiguousArray(
            repeating: Float(0),
            count: 2 * HumanoidBoxCarryTask.observationDimension)
        observations[87] = 1
        observations[102] = 1
        observations[HumanoidBoxCarryTask.observationDimension + 87] = 1
        observations[HumanoidBoxCarryTask.observationDimension + 88] = 1
        observations[HumanoidBoxCarryTask.observationDimension + 89] = 1
        observations[HumanoidBoxCarryTask.observationDimension + 90] = 1
        observations[HumanoidBoxCarryTask.observationDimension + 102] = 1

        XCTAssertEqual(task.policyExpertGates(observations), [1, 0])
        XCTAssertEqual(task.policyStandExpertGates(observations), [0, 1])
        XCTAssertTrue(task.freezesBasePolicyExpert)
        XCTAssertTrue(task.freezesLowSpeedPolicyExpert)
        XCTAssertEqual(
            task.policyStandExpertActionMask,
            ContiguousArray(
                [Float](repeating: 0.75, count: 11)
                    + [Float](repeating: 1, count: 8)))
        XCTAssertTrue(
            task.initializesPolicyStandExpertFromPolicyExpertOnTransfer)
        XCTAssertEqual(
            task.spec.configurationValues["manipulationHandoffSteps"], 24)
        XCTAssertEqual(task.spec.configurationValues["carryHandoffSteps"], 12)
        XCTAssertEqual(
            task.spec.configurationValues["carryCommandRampSteps"], 12)
    }

    func testCarryHandoffBlendsManipulationAndCarryExperts() throws {
        let task = try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: 1, seed: 661, observationNoise: false))
        var observations = ContiguousArray(
            repeating: Float(0),
            count: HumanoidBoxCarryTask.observationDimension)
        observations[87] = 1
        observations[89] = 1
        observations[90] = 0.25
        observations[102] = 1

        XCTAssertEqual(task.policyExpertGates(observations), [0.75])
        XCTAssertEqual(task.policyStandExpertGates(observations), [0.25])
    }

    func testCompositionalCarryKeepsFrozenArmsAndLoadedLegsDisjoint() throws {
        let task = try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: 1, seed: 662, observationNoise: false,
            carryBaseLegActionFraction: 0,
            initializeCarryExpertFromManipulationExpertOnTransfer: false,
            compositionalCarryController: true,
            initializeCarryExpertFromBaseOnTransfer: true))
        var observations = ContiguousArray(
            repeating: Float(0),
            count: HumanoidBoxCarryTask.observationDimension)
        observations[87] = 1
        observations[89] = 1
        observations[90] = 1
        observations[102] = 1

        XCTAssertEqual(task.policyExpertGates(observations), [1])
        XCTAssertEqual(task.policyStandExpertGates(observations), [1])
        XCTAssertEqual(
            task.policyExpertActionMask,
            ContiguousArray(
                [Float](repeating: 0, count: 10)
                    + [Float](repeating: 1, count: 9)))
        XCTAssertEqual(
            task.policyStandExpertActionMask,
            ContiguousArray(
                [Float](repeating: 1, count: 10)
                    + [Float](repeating: 0, count: 9)))
        XCTAssertTrue(task.initializesPolicyStandExpertFromBaseOnTransfer)
        XCTAssertFalse(
            task.initializesPolicyStandExpertFromPolicyExpertOnTransfer)
        let referenceWeights = task
            .policyReferenceActionRegularizationWeights(
                observations, actionDimension: 19)
        XCTAssertEqual(
            Array(referenceWeights[0..<10]),
            [Float](repeating: 0, count: 10))
        XCTAssertEqual(
            Array(referenceWeights[10..<19]),
            [Float](repeating: 1, count: 9))
        XCTAssertEqual(
            task.spec.configurationValues["compositionalCarryController"], 1)
        XCTAssertEqual(
            task.spec.configurationValues[
                "initializeCarryExpertFromBaseOnTransfer"], 1)
    }

    func testUpperBodyCarryHandsWholeBodyPickupToSplitCarry() throws {
        let task = try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: 1, seed: 663, observationNoise: false,
            carryBaseLegActionFraction: 1,
            upperBodyCarryController: true,
            carryArmReferenceWeight: 0))
        var observations = ContiguousArray(
            repeating: Float(0),
            count: HumanoidBoxCarryTask.observationDimension)
        observations[87] = 1
        observations[89] = 1
        observations[90] = 0.25
        observations[102] = 1

        let upperBodyMask = ContiguousArray(
            [Float](repeating: 0, count: 10)
                + [Float](repeating: 1, count: 9))
        XCTAssertEqual(task.policyExpertGates(observations), [0.75])
        XCTAssertEqual(task.policyStandExpertGates(observations), [0.25])
        XCTAssertEqual(task.policyAuxiliaryExpertGates(observations), [0])
        XCTAssertNil(task.policyExpertActionMask)
        XCTAssertEqual(task.policyStandExpertActionMask, upperBodyMask)
        XCTAssertEqual(
            task.policyAuxiliaryExpertActionMask,
            ContiguousArray(
                [Float](repeating: 1, count: 10)
                    + [Float](repeating: 0, count: 9)))
        XCTAssertTrue(task.freezesStandPolicyExpert)
        let referenceWeights = task
            .policyReferenceActionRegularizationWeights(
                observations, actionDimension: 19)
        XCTAssertEqual(
            Array(referenceWeights[0..<10]),
            [Float](repeating: 0, count: 10))
        XCTAssertEqual(
            Array(referenceWeights[10..<19]),
            [Float](repeating: 0, count: 9))
        XCTAssertEqual(
            task.spec.configurationValues["upperBodyCarryController"], 1)

        let blended = try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: 1, seed: 664, observationNoise: false,
            carryBaseLegActionFraction: 0.5,
            upperBodyCarryController: true))
        XCTAssertEqual(
            blended.policyStandExpertActionMask,
            upperBodyMask)
        XCTAssertEqual(
            blended.policyAuxiliaryExpertGates(observations), [0.125])
    }

    func testPlacementRefinementCanTrainTorsoAndCarryArms() throws {
        let task = try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: 1, seed: 665, observationNoise: false,
            carryHolonomicCommand: true,
            carryBaseLegActionFraction: 0,
            freezeCarryPolicyExpert: false,
            upperBodyCarryController: true,
            carryLocomotionControlsTorso: true,
            carryStartReplayProbability: 0.75,
            advanceReplaySnapshotAtDestinationContact: true,
            carryArmReferenceWeight: 0.2))
        var observations = ContiguousArray(
            repeating: Float(0),
            count: HumanoidBoxCarryTask.observationDimension)
        observations[89] = 1
        observations[90] = 1

        XCTAssertEqual(
            task.policyStandExpertActionMask,
            ContiguousArray(
                [Float](repeating: 0, count: 11)
                    + [Float](repeating: 1, count: 8)))
        XCTAssertEqual(
            task.policyAuxiliaryExpertActionMask,
            ContiguousArray(
                [Float](repeating: 1, count: 11)
                    + [Float](repeating: 0, count: 8)))
        XCTAssertFalse(task.freezesStandPolicyExpert)

        let referenceWeights = task
            .policyReferenceActionRegularizationWeights(
                observations, actionDimension: 19)
        XCTAssertEqual(
            Array(referenceWeights[0..<11]),
            [Float](repeating: 0, count: 11))
        XCTAssertEqual(
            Array(referenceWeights[11..<19]),
            [Float](repeating: 0.2, count: 8))

        XCTAssertEqual(
            task.spec.configurationValues["carryHolonomicCommand"], 1)
        XCTAssertEqual(
            task.spec.configurationValues["freezeCarryPolicyExpert"], 0)
        XCTAssertEqual(
            task.spec.configurationValues["carryLocomotionControlsTorso"], 1)
        XCTAssertEqual(
            task.spec.configurationValues[
                "advanceReplaySnapshotAtDestinationContact"], 1)
    }

    func testDestinationReplayRequiresCarryResetProbability() {
        XCTAssertThrowsError(try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: 1, seed: 666, observationNoise: false,
            advanceReplaySnapshotAtDestinationContact: true)))
    }

    func testCarryRevisionAppendsObservedHandoffWithoutChangingOldInputs()
        throws
    {
        let task = try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: 1, seed: 67, observationNoise: false))
        let mapping = try XCTUnwrap(
            task.initializationObservationSourceIndices(sourceDimension: 90))
        XCTAssertEqual(mapping.count, HumanoidBoxCarryTask.observationDimension)
        XCTAssertEqual(Array(mapping.prefix(90)), (0..<90).map(Optional.some))
        XCTAssertNil(mapping[90])
        XCTAssertNil(mapping[91])
        XCTAssertNil(mapping[92])
    }

    func testCarryActorUpdatesBeginAtSettledManipulationPhase() throws {
        let task = try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: 3, seed: 672, observationNoise: false))
        var observations = ContiguousArray(
            repeating: Float(0),
            count: 3 * HumanoidBoxCarryTask.observationDimension)
        let second = HumanoidBoxCarryTask.observationDimension
        let third = 2 * HumanoidBoxCarryTask.observationDimension
        observations[second + 87] = 1
        observations[third + 87] = 1
        observations[third + 89] = 1

        XCTAssertEqual(
            task.policyActorTrainingWeights(observations), [0, 1, 1])
        XCTAssertEqual(
            task.policyReferenceRegularizationWeights(observations), [1, 1, 1])
        let actionWeights = task.policyReferenceActionRegularizationWeights(
            observations, actionDimension: 19)
        XCTAssertEqual(
            Array(actionWeights[0..<38]),
            [Float](repeating: 1, count: 38))
        XCTAssertEqual(
            Array(actionWeights[38..<(38 + 11)]),
            [Float](repeating: 0, count: 11))
        XCTAssertEqual(
            Array(actionWeights[(38 + 11)..<57]),
            [Float](repeating: 1, count: 8))
    }

    func testCarryReferenceStateInitializationIsExplicitAndTrainingOnly()
        throws
    {
        let task = try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: 2, seed: 673, maxEpisodeSteps: 1,
            observationNoise: false, carryStartReplayProbability: 0.8))
        XCTAssertEqual(
            try XCTUnwrap(task.spec.configurationValues[
                "carryStartReplayProbability"]),
            0.8, accuracy: 1e-6)

        // No reference exists before a real unsupported lift. Even with a
        // terminal reset and the feature configured, evaluation cannot
        // synthesize or enter a carry state.
        var result = RLStepBatch(spec: task.spec)
        let actions = try RLActionBatch(
            numEnvironments: 2, actionDimension: 19,
            values: ContiguousArray(repeating: 0, count: 2 * 19))
        try task.step(actions: actions, into: &result)
        XCTAssertEqual(
            result.metrics["task/reference_state_reset"],
            ContiguousArray(repeating: Float(0), count: 2))
    }

    func testCarryRejectsInvalidReferenceStateProbability() {
        XCTAssertThrowsError(try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: 1, carryStartReplayProbability: 1.01)))
    }

    func testCarryExpertCanAdaptArmsWithoutChangingFrozenPickupExpert()
        throws
    {
        let task = try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: 1, observationNoise: false,
            freezeManipulationPolicyExpert: true,
            carryArmReferenceWeight: 0))
        var observations = ContiguousArray(
            repeating: Float(0),
            count: HumanoidBoxCarryTask.observationDimension)
        observations[87] = 1
        observations[89] = 1
        observations[90] = 1
        let weights = task.policyReferenceActionRegularizationWeights(
            observations, actionDimension: 19)
        XCTAssertEqual(weights, ContiguousArray(repeating: 0, count: 19))
        XCTAssertTrue(task.freezesLowSpeedPolicyExpert)
        XCTAssertEqual(
            task.spec.configurationValues["carryArmReferenceWeight"], 0)
    }

    func testCarryHoldClearanceRewardTargetIsExplicit() throws {
        let task = try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: 1, liftClearance: 0.04,
            carryHoldClearanceMultiplier: 2,
            carryProgressRewardWeight: 300))
        XCTAssertEqual(
            task.spec.configurationValues["carryHoldClearanceMultiplier"], 2)
        XCTAssertEqual(
            task.spec.configurationValues["carryProgressRewardWeight"], 300)
        XCTAssertThrowsError(try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: 1, carryHoldClearanceMultiplier: 0.9)))
    }

    func testCarryLocomotionRewardMultiplierIsExplicitAndBounded() throws {
        let task = try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: 1, seed: 665, observationNoise: false,
            carryLocomotionRewardMultiplier: 10,
            carryTrackingVariance: 0.025))
        XCTAssertEqual(
            task.spec.configurationValues[
                "carryLocomotionRewardMultiplier"], 10)
        XCTAssertEqual(
            task.spec.configurationValues["carryTrackingVariance"], 0.025)
        XCTAssertThrowsError(try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: 1, seed: 666, observationNoise: false,
            carryLocomotionRewardMultiplier: 0)))
        XCTAssertThrowsError(try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: 1, seed: 667, observationNoise: false,
            carryTrackingVariance: 0.001)))
    }

    func testDestinationBearingCurriculumStartsForwardAndEndsAtLeftTable()
        throws
    {
        let task = try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: 1, seed: 665, observationNoise: false,
            minimumTrainingStationDistance: 0.55,
            evaluationStationDistance: 0.55,
            destinationBearingCurriculumControlSteps: 1_000))
        task.setTrainingMode(true)
        task.setTrainingProgress(environmentSteps: 0)
        _ = try task.reset(seed: 665)
        let placement = task.currentPlacementTarget(environment: 0)
        XCTAssertEqual(placement.x, 0.55, accuracy: 1e-5)
        XCTAssertEqual(placement.y, 0.75, accuracy: 1e-5)
        let forward = task.currentCarryNavigationTarget(environment: 0)
        XCTAssertEqual(forward.x, 1.30, accuracy: 1e-5)
        XCTAssertEqual(forward.y, 0, accuracy: 1e-5)

        task.setTrainingProgress(environmentSteps: 1_000)
        _ = try task.reset(seed: 666)
        let left = task.currentCarryNavigationTarget(environment: 0)
        XCTAssertEqual(left.x, 0.55, accuracy: 1e-5)
        XCTAssertEqual(left.y, 0.75, accuracy: 1e-5)
        XCTAssertEqual(
            task.spec.configurationValues[
                "destinationBearingCurriculumControlSteps"], 1_000)

        task.setTrainingMode(false)
        _ = try task.reset(seed: 667)
        let evaluation = task.currentCarryNavigationTarget(environment: 0)
        XCTAssertEqual(evaluation.x, 0.55, accuracy: 1e-5)
        XCTAssertEqual(evaluation.y, 0.75, accuracy: 1e-5)
    }

    func testBatchedCarryStepIsFiniteAndReplicaIsolated() throws {
        let task = try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: 4, seed: 64, maxEpisodeSteps: 20,
            observationNoise: false))
        var result = RLStepBatch(spec: task.spec)
        let actions = try RLActionBatch(
            numEnvironments: 4, actionDimension: 19,
            values: ContiguousArray(repeating: 0, count: 4 * 19))
        try task.step(actions: actions, into: &result)
        XCTAssertTrue(result.rewards.allSatisfy(\.isFinite))
        XCTAssertTrue(result.observations.policy.allSatisfy(\.isFinite))
        XCTAssertEqual(
            result.metrics["reward/grip_retention"],
            ContiguousArray(repeating: Float(0), count: 4))
        XCTAssertEqual(
            result.metrics["penalty/box_descent"],
            ContiguousArray(repeating: Float(0), count: 4))
        XCTAssertEqual(task.environment.boxStates().count, 4)
        XCTAssertTrue(task.environment.refs.enumerated().allSatisfy {
            environment, ref in
            ref.projectile != nil && ref.carryPedestal != nil
                && ref.carryDestinationPedestal != nil
                && task.environment.scene.colliders.contains {
                    $0.body == ref.projectile
                        && $0.collisionGroup == UInt32(environment + 1)
                }
        })
    }
}
