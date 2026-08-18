import XCTest
import simd
@testable import AVBDCore

final class HumanoidBoxCarryTests: XCTestCase {
    func testEnvironmentZeroPhysicsIsInvariantToBatchSize() throws {
        func makeTask(_ count: Int) throws -> HumanoidBoxCarryTask {
            try HumanoidBoxCarryTask(configuration: .init(
                numEnvironments: count, seed: 58, maxEpisodeSteps: 200,
                autoReset: false, observationNoise: false,
                minimumTrainingStationDistance: 0.55,
                evaluationStationDistance: 0.55,
                stationDistanceCurriculumControlSteps: 0))
        }
        let single = try makeTask(1)
        let batched = try makeTask(8)
        _ = try single.reset(seed: 58)
        _ = try batched.reset(seed: 58)
        var singleResult = RLStepBatch(spec: single.spec)
        var batchedResult = RLStepBatch(spec: batched.spec)

        for step in 0..<160 {
            var singleActions = RLActionBatch(spec: single.spec)
            var batchedActions = RLActionBatch(spec: batched.spec)
            for environment in 0..<batched.spec.numEnvironments {
                for joint in 0..<batched.spec.action.elementCount {
                    let action = 0.08 * sin(Float(step * 23 + joint))
                    batchedActions[environment, joint] = action
                    if environment == 0 {
                        singleActions[0, joint] = action
                    }
                }
            }
            try single.step(actions: singleActions, into: &singleResult)
            try batched.step(actions: batchedActions, into: &batchedResult)

            let singleRoot = single.environment.states()[0].root
            let batchedRoot = batched.environment.states()[0].root
            let singleBox = single.environment.boxStates()[0]
            let batchedBox = batched.environment.boxStates()[0]
            for (actual, expected) in [
                (singleRoot.position, batchedRoot.position),
                (singleRoot.linearVelocity, batchedRoot.linearVelocity),
                (singleBox.position, batchedBox.position),
                (singleBox.linearVelocity, batchedBox.linearVelocity),
            ] {
                XCTAssertEqual(
                    actual.x, expected.x, accuracy: 0,
                    "batch-size divergence at control \(step + 1)")
                XCTAssertEqual(
                    actual.y, expected.y, accuracy: 0,
                    "batch-size divergence at control \(step + 1)")
                XCTAssertEqual(
                    actual.z, expected.z, accuracy: 0,
                    "batch-size divergence at control \(step + 1)")
            }
        }
    }

    func testOpposingFaceGraspAcceptsStableFiniteFaceContacts() {
        let measuredLeft = F3(0.10171407, 0.17255615, 0.09225361)
        let measuredRight = F3(0.033942565, -0.16913806, 0.13562159)
        let quality = HumanoidBoxCarryTask.opposingFaceGraspQuality(
            localLeftHand: measuredLeft, localRightHand: measuredRight)
        XCTAssertGreaterThan(quality, 0.95)
        XCTAssertTrue(HumanoidBoxCarryTask.isOpposingFaceGrasp(
            localLeftHand: measuredLeft, localRightHand: measuredRight,
            bilateralContact: true, boxUprightAlignment: 0.93))

        XCTAssertEqual(HumanoidBoxCarryTask.opposingFaceGraspQuality(
            localLeftHand: measuredLeft,
            localRightHand: F3(0.03, 0.17, 0.13)), 0)
        XCTAssertFalse(HumanoidBoxCarryTask.isOpposingFaceGrasp(
            localLeftHand: F3(0.30, 0.175, 0),
            localRightHand: measuredRight,
            bilateralContact: true, boxUprightAlignment: 0.93))
        XCTAssertFalse(HumanoidBoxCarryTask.isOpposingFaceGrasp(
            localLeftHand: measuredLeft, localRightHand: measuredRight,
            bilateralContact: false, boxUprightAlignment: 0.93))
    }

    func testLoadBearingGraspRequiresMeasuredFrictionCapacity() {
        XCTAssertFalse(HumanoidBoxCarryTask.frictionGraspSupportsWeight(
            leftNormalLoad: 0, rightNormalLoad: 50,
            boxMass: 2, friction: 1.2))
        XCTAssertFalse(HumanoidBoxCarryTask.frictionGraspSupportsWeight(
            leftNormalLoad: 4, rightNormalLoad: 4,
            boxMass: 2, friction: 1.2))
        XCTAssertTrue(HumanoidBoxCarryTask.frictionGraspSupportsWeight(
            leftNormalLoad: 10, rightNormalLoad: 10,
            boxMass: 2, friction: 1.2))
        XCTAssertEqual(
            HumanoidBoxCarryTask.frictionGraspSupportFraction(
                leftNormalLoad: 8.175, rightNormalLoad: 8.175,
                boxMass: 2, friction: 1.2),
            1, accuracy: 1e-5)
    }

    func testCarryStepReportsFiniteActuatorAuthorityDiagnostics() throws {
        let task = try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: 1, seed: 581, maxEpisodeSteps: 20,
            autoReset: false, observationNoise: false))
        _ = try task.reset(seed: 581)
        var result = RLStepBatch(spec: task.spec)
        try task.step(
            actions: RLActionBatch(spec: task.spec), into: &result)

        for key in [
            "state/maximum_actuator_torque_ratio",
            "state/maximum_arm_actuator_torque_ratio",
            "state/saturated_actuator_count",
            "state/saturated_arm_actuator_count",
            "state/minimum_joint_limit_margin_rad",
            "state/maximum_requested_target_clamp_rad",
        ] {
            let values = try XCTUnwrap(result.metrics[key], key)
            XCTAssertEqual(values.count, 1, key)
            XCTAssertTrue(values[0].isFinite, key)
        }
        XCTAssertGreaterThanOrEqual(
            result.metrics["state/maximum_actuator_torque_ratio"]![0], 0)
        XCTAssertGreaterThanOrEqual(
            result.metrics["state/maximum_requested_target_clamp_rad"]![0], 0)
        XCTAssertLessThanOrEqual(
            result.metrics["state/saturated_arm_actuator_count"]![0], 8)
        XCTAssertLessThanOrEqual(
            result.metrics["state/saturated_actuator_count"]![0], 19)
    }

    func testSpeculationSnapshotReplaysNextDynamicsExactly() throws {
        let task = try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: 2, seed: 59, maxEpisodeSteps: 200,
            autoReset: false, observationNoise: false,
            minimumTrainingStationDistance: 0.55,
            evaluationStationDistance: 0.55,
            stationDistanceCurriculumControlSteps: 0))
        _ = try task.reset(seed: 59)
        var result = RLStepBatch(spec: task.spec)

        func actions(_ phase: Int) -> RLActionBatch {
            var batch = RLActionBatch(spec: task.spec)
            for environment in 0..<task.spec.numEnvironments {
                for joint in 0..<task.spec.action.elementCount {
                    batch[environment, joint] = 0.08 * sin(
                        Float(phase * 23 + environment * 7 + joint))
                }
            }
            return batch
        }

        for step in 0..<5 {
            try task.step(actions: actions(step), into: &result)
        }
        let snapshot = task.captureSpeculationSnapshot()

        for step in 5..<9 {
            try task.step(actions: actions(step), into: &result)
        }
        let expectedHumanoids = task.environment.states()
        let expectedObjects = task.environment.boxStates()
        let expectedObservations = result.observations.policy
        let expectedCarry = result.metrics["state/carry_distance_m"]!
        let expectedRootDestination = result.metrics[
            "state/root_destination_distance_m"]!
        let expectedAlternatingSteps = result.metrics[
            "state/loaded_alternating_steps"]!
        let expectedTouchdowns = result.metrics[
            "state/loaded_touchdowns"]!
        let expectedLeftFootAirTime = result.metrics[
            "state/left_loaded_foot_air_time_s"]!
        let expectedRightFootAirTime = result.metrics[
            "state/right_loaded_foot_air_time_s"]!
        let expectedMaximumFootAirTime = result.metrics[
            "state/maximum_loaded_foot_air_time_s"]!
        let expectedLeftFootNormalLoad = result.metrics[
            "state/left_foot_normal_load"]!
        let expectedRightFootNormalLoad = result.metrics[
            "state/right_foot_normal_load"]!
        let expectedFootUnloadingFraction = result.metrics[
            "state/foot_unloading_fraction"]!
        let expectedSwingClearance = result.metrics[
            "state/maximum_loaded_swing_clearance_m"]!
        let expectedLeftLoadBearingContact = result.metrics[
            "state/left_load_bearing_foot_contact"]!
        let expectedRightLoadBearingContact = result.metrics[
            "state/right_load_bearing_foot_contact"]!
        let expectedLeftFootContact = result.metrics[
            "state/left_foot_contact"]!
        let expectedRightFootContact = result.metrics[
            "state/right_foot_contact"]!

        task.restoreSpeculationSnapshot(snapshot)
        for step in 5..<9 {
            try task.step(actions: actions(step), into: &result)
        }
        let replayHumanoids = task.environment.states()
        let replayObjects = task.environment.boxStates()

        func assertEqual(_ lhs: F3, _ rhs: F3, file: StaticString = #filePath,
                         line: UInt = #line) {
            XCTAssertEqual(lhs.x, rhs.x, accuracy: 1e-6, file: file, line: line)
            XCTAssertEqual(lhs.y, rhs.y, accuracy: 1e-6, file: file, line: line)
            XCTAssertEqual(lhs.z, rhs.z, accuracy: 1e-6, file: file, line: line)
        }
        for environment in expectedHumanoids.indices {
            assertEqual(
                expectedHumanoids[environment].root.position,
                replayHumanoids[environment].root.position)
            assertEqual(
                expectedHumanoids[environment].root.linearVelocity,
                replayHumanoids[environment].root.linearVelocity)
            assertEqual(
                expectedObjects[environment].position,
                replayObjects[environment].position)
            assertEqual(
                expectedObjects[environment].linearVelocity,
                replayObjects[environment].linearVelocity)
        }
        XCTAssertEqual(result.observations.policy.count,
                       expectedObservations.count)
        for (actual, expected) in zip(
            result.observations.policy, expectedObservations) {
            XCTAssertEqual(actual, expected, accuracy: 1e-6)
        }
        for (actual, expected) in zip(
            result.metrics["state/carry_distance_m"]!, expectedCarry) {
            XCTAssertEqual(actual, expected, accuracy: 1e-6)
        }
        XCTAssertEqual(
            Array(result.metrics["state/root_destination_distance_m"]!),
            Array(expectedRootDestination))
        XCTAssertEqual(
            Array(result.metrics["state/loaded_alternating_steps"]!),
            Array(expectedAlternatingSteps))
        XCTAssertEqual(
            Array(result.metrics["state/loaded_touchdowns"]!),
            Array(expectedTouchdowns))
        XCTAssertEqual(
            Array(result.metrics["state/left_loaded_foot_air_time_s"]!),
            Array(expectedLeftFootAirTime))
        XCTAssertEqual(
            Array(result.metrics["state/right_loaded_foot_air_time_s"]!),
            Array(expectedRightFootAirTime))
        XCTAssertEqual(
            Array(result.metrics["state/maximum_loaded_foot_air_time_s"]!),
            Array(expectedMaximumFootAirTime))
        XCTAssertEqual(
            Array(result.metrics["state/left_foot_normal_load"]!),
            Array(expectedLeftFootNormalLoad))
        XCTAssertEqual(
            Array(result.metrics["state/right_foot_normal_load"]!),
            Array(expectedRightFootNormalLoad))
        XCTAssertEqual(
            Array(result.metrics["state/foot_unloading_fraction"]!),
            Array(expectedFootUnloadingFraction))
        XCTAssertEqual(
            Array(result.metrics["state/maximum_loaded_swing_clearance_m"]!),
            Array(expectedSwingClearance))
        XCTAssertEqual(
            Array(result.metrics["state/left_load_bearing_foot_contact"]!),
            Array(expectedLeftLoadBearingContact))
        XCTAssertEqual(
            Array(result.metrics["state/right_load_bearing_foot_contact"]!),
            Array(expectedRightLoadBearingContact))
        XCTAssertEqual(
            Array(result.metrics["state/left_foot_contact"]!),
            Array(expectedLeftFootContact))
        XCTAssertEqual(
            Array(result.metrics["state/right_foot_contact"]!),
            Array(expectedRightFootContact))
    }

    func testPortableSpeculationBoundaryReplaysAcrossBatchShapes() throws {
        func makeTask(_ count: Int) throws -> HumanoidBoxCarryTask {
            try HumanoidBoxCarryTask(configuration: .init(
                numEnvironments: count, seed: 593, maxEpisodeSteps: 200,
                autoReset: false, observationNoise: false,
                minimumTrainingStationDistance: 0.55,
                evaluationStationDistance: 0.55,
                stationDistanceCurriculumControlSteps: 0))
        }
        let source = try makeTask(2)
        let single = try makeTask(1)
        _ = try source.reset(seed: 593)
        _ = try single.reset(seed: 593)
        var sourceResult = RLStepBatch(spec: source.spec)
        for step in 0..<6 {
            var actions = RLActionBatch(spec: source.spec)
            for environment in 0..<2 {
                for joint in 0..<source.spec.action.elementCount {
                    actions[environment, joint] =
                        0.04 * sin(Float(step * 19 + joint))
                }
            }
            try source.step(actions: actions, into: &sourceResult)
        }
        source.canonicalizeSpeculationReplicas(sourceEnvironment: 0)
        let portable = source.capturePortableSpeculationState(
            sourceEnvironment: 0)
        single.restorePortableSpeculationState(portable)

        func assertStateEqual(
            _ lhs: HumanoidState, _ rhs: HumanoidState,
            file: StaticString = #filePath, line: UInt = #line
        ) {
            for (actual, expected) in [
                (lhs.root.position, rhs.root.position),
                (lhs.root.linearVelocity, rhs.root.linearVelocity),
                (lhs.leftFoot.position, rhs.leftFoot.position),
                (lhs.rightFoot.position, rhs.rightFoot.position),
            ] {
                XCTAssertEqual(actual.x, expected.x, accuracy: 0,
                               file: file, line: line)
                XCTAssertEqual(actual.y, expected.y, accuracy: 0,
                               file: file, line: line)
                XCTAssertEqual(actual.z, expected.z, accuracy: 0,
                               file: file, line: line)
            }
        }
        assertStateEqual(
            source.environment.states()[0], single.environment.states()[0])
        let sourceBox = source.environment.boxStates()[0]
        let singleBox = single.environment.boxStates()[0]
        XCTAssertEqual(sourceBox.position.x, singleBox.position.x, accuracy: 0)
        XCTAssertEqual(sourceBox.position.y, singleBox.position.y, accuracy: 0)
        XCTAssertEqual(sourceBox.position.z, singleBox.position.z, accuracy: 0)

        var sourceAction = RLActionBatch(spec: source.spec)
        var singleAction = RLActionBatch(spec: single.spec)
        for environment in 0..<2 {
            for joint in 0..<source.spec.action.elementCount {
                let action = 0.04 * cos(Float(131 + joint))
                sourceAction[environment, joint] = action
                if environment == 0 { singleAction[0, joint] = action }
            }
        }
        var singleResult = RLStepBatch(spec: single.spec)
        try source.step(actions: sourceAction, into: &sourceResult)
        try single.step(actions: singleAction, into: &singleResult)
        assertStateEqual(
            source.environment.states()[0], single.environment.states()[0])
    }

    func testLoadedFootContactRequiresSolverNormalLoadShare() {
        XCTAssertEqual(HumanoidBoxCarryTask.loadBearingFootContacts(
            manifoldContacts: [true, true],
            normalLoads: [500, 500]), [true, true])
        XCTAssertEqual(HumanoidBoxCarryTask.loadBearingFootContacts(
            manifoldContacts: [true, true],
            normalLoads: [999, 1]), [true, false])
        XCTAssertEqual(HumanoidBoxCarryTask.loadBearingFootContacts(
            manifoldContacts: [true, true],
            normalLoads: [0, 0]), [false, false])
        XCTAssertEqual(HumanoidBoxCarryTask.loadBearingFootContacts(
            manifoldContacts: [false, true],
            normalLoads: [500, 500]), [false, true])
    }

    func testSpeculationCanonicalizationStartsReplicasFromOneBranch()
        throws
    {
        let task = try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: 2, seed: 591, maxEpisodeSteps: 200,
            autoReset: false, observationNoise: false,
            minimumTrainingStationDistance: 0.55,
            evaluationStationDistance: 0.55,
            stationDistanceCurriculumControlSteps: 0))
        var observation = try task.reset(seed: 591)
        var result = RLStepBatch(spec: task.spec)
        for step in 0..<4 {
            var actions = RLActionBatch(spec: task.spec)
            for joint in 0..<task.spec.action.elementCount {
                actions[0, joint] = 0.1 * sin(Float(step + joint))
                actions[1, joint] = -0.1 * cos(Float(step + joint))
            }
            try task.step(actions: actions, into: &result)
            observation = result.observations
        }

        task.canonicalizeSpeculationReplicas(
            observation: &observation, result: &result,
            sourceEnvironment: 0)
        let canonical = task.environment.states()
        let boxes = task.environment.boxStates()
        XCTAssertEqual(
            canonical[0].root.position.x,
            canonical[1].root.position.x, accuracy: 1e-7)
        XCTAssertEqual(
            canonical[0].root.position.y,
            canonical[1].root.position.y, accuracy: 1e-7)
        XCTAssertEqual(
            canonical[0].root.position.z,
            canonical[1].root.position.z, accuracy: 1e-7)
        XCTAssertEqual(
            canonical[0].root.linearVelocity.x,
            canonical[1].root.linearVelocity.x, accuracy: 1e-7)
        XCTAssertEqual(
            boxes[0].position.x, boxes[1].position.x, accuracy: 1e-7)
        XCTAssertEqual(
            boxes[0].position.z, boxes[1].position.z, accuracy: 1e-7)
        let observationWidth = task.spec.observation.elementCount
        XCTAssertEqual(
            Array(observation.policy[0..<observationWidth]),
            Array(observation.policy[
                observationWidth..<(2 * observationWidth)]))
        for values in result.metrics.values {
            XCTAssertEqual(values[0], values[1])
        }

        var identical = RLActionBatch(spec: task.spec)
        for environment in 0..<2 {
            for joint in 0..<task.spec.action.elementCount {
                identical[environment, joint] = 0.05 * sin(Float(joint))
            }
        }
        try task.step(actions: identical, into: &result)
        let advanced = task.environment.states()
        XCTAssertEqual(
            advanced[0].root.position.x,
            advanced[1].root.position.x, accuracy: 1e-5)
        XCTAssertEqual(
            advanced[0].root.position.y,
            advanced[1].root.position.y, accuracy: 1e-5)
        XCTAssertEqual(
            advanced[0].root.position.z,
            advanced[1].root.position.z, accuracy: 1e-5)
    }

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
        let normalLoads = env.boxHandContactNormalLoads()
        XCTAssertTrue(contacts.left[0])
        XCTAssertFalse(contacts.right[0])
        XCTAssertGreaterThan(normalLoads.left[0], 0)
        XCTAssertEqual(normalLoads.right[0], 0)
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
            initializeCarryLocomotionExpertFromBaseOnTransfer: true,
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
        XCTAssertTrue(
            task.initializesPolicyAuxiliaryExpertFromBaseOnTransfer)
        XCTAssertEqual(
            task.policyAuxiliaryExpertZeroedObservationIndicesOnTransfer,
            Array(23...30) + Array(42...49) + Array(61...68))
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
        XCTAssertEqual(
            task.spec.configurationValues[
                "initializeCarryLocomotionExpertFromBaseOnTransfer"], 1)
        XCTAssertThrowsError(try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: 1,
            initializeCarryLocomotionExpertFromBaseOnTransfer: true)))

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
            carryTrackingVariance: 0.025,
            coupledCarryCommandTracking: true,
            carryRootProgressRewardWeight: 400,
            carryAlternatingStepRewardWeight: 8,
            minimumLoadedAlternatingSteps: 2))
        XCTAssertEqual(
            task.spec.configurationValues[
                "carryLocomotionRewardMultiplier"], 10)
        XCTAssertEqual(
            task.spec.configurationValues["carryTrackingVariance"], 0.025)
        XCTAssertEqual(
            task.spec.configurationValues["coupledCarryCommandTracking"], 1)
        XCTAssertEqual(
            task.spec.configurationValues["carryRootProgressRewardWeight"], 400)
        XCTAssertEqual(
            task.spec.configurationValues[
                "carryAlternatingStepRewardWeight"], 8)
        XCTAssertEqual(
            task.spec.configurationValues["minimumLoadedAlternatingSteps"], 2)
        XCTAssertThrowsError(try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: 1, seed: 666, observationNoise: false,
            carryLocomotionRewardMultiplier: 0)))
        XCTAssertThrowsError(try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: 1, seed: 667, observationNoise: false,
            carryTrackingVariance: 0.001)))
        XCTAssertThrowsError(try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: 1, carryRootProgressRewardWeight: -1)))
        XCTAssertThrowsError(try HumanoidBoxCarryTask(configuration: .init(
            numEnvironments: 1, minimumLoadedAlternatingSteps: 101)))
    }

    func testCoupledCarryTrackingClosesZeroYawStandingLoophole() {
        let variance: Float = 0.005
        let stoppedPlanarError: Float = 0.15 * 0.15
        let historical = HumanoidBoxCarryTask.commandTrackingReward(
            planarErrorSquared: stoppedPlanarError, yawErrorSquared: 0,
            variance: variance, coupled: false)
        let coupled = HumanoidBoxCarryTask.commandTrackingReward(
            planarErrorSquared: stoppedPlanarError, yawErrorSquared: 0,
            variance: variance, coupled: true)
        let perfect = HumanoidBoxCarryTask.commandTrackingReward(
            planarErrorSquared: 0, yawErrorSquared: 0,
            variance: variance, coupled: true)
        XCTAssertGreaterThan(historical, 1)
        XCTAssertLessThan(coupled, 0.03)
        XCTAssertEqual(perfect, 2, accuracy: 1e-6)
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

    func testH1ShoulderMotorCoordinateMatchesMeasuredHandJacobian() throws {
        let environment = try HumanoidWalkEnv(
            numEnvironments: 2, includeProjectile: true)
        let action = 11
        let reference = environment.refs[1]
        let joint = environment.scene.joints[reference.motors[action]]
        let initialBodies = environment.solver.bodyStates([
            joint.bodyB, reference.leftHand,
        ])
        let initialTorso = environment.states()[1].torso
        let axisWorld = initialBodies[0].rotation.act(joint.hingeAxis!)
        let anchorWorld = initialBodies[0].position
            + initialBodies[0].rotation.act(joint.rB)
        let geometricColumnWorld = cross(
            axisWorld, initialBodies[1].position - anchorWorld)
        let geometricColumnTorso = initialTorso.rotation.conjugate.act(
            geometricColumnWorld)

        var actions = ContiguousArray(
            repeating: Float(0), count: 2 * 19)
        actions[19 + action] = 0.2
        for _ in 0..<12 {
            environment.step(
                normalizedActions: actions, decimation: 4)
        }

        let humanoids = environment.states()
        let hands = environment.manipulationStates()
        func torsoLocalHand(_ environment: Int) -> F3 {
            humanoids[environment].torso.rotation.conjugate.act(
                hands[environment].leftHand.position
                    - humanoids[environment].torso.position)
        }
        let measured = torsoLocalHand(1) - torsoLocalHand(0)
        let angleDelta = humanoids[1].jointAngles[action]
            - humanoids[0].jointAngles[action]
        XCTAssertGreaterThan(abs(angleDelta), 0.01)
        XCTAssertGreaterThan(
            dot(measured, geometricColumnTorso * angleDelta), 0,
            "positive motor motion must follow the measured geometric "
                + "child-link Jacobian")
    }
}
