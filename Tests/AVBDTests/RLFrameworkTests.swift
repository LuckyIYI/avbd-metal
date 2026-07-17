import XCTest
import Foundation
import simd
import MLX
@testable import AVBDCore
@testable import AVBDLearn

private final class TestVectorPolicyInference: VectorPolicyInferencing {
    let observationDimension: Int
    let actionDimension: Int
    let controlPeriodSeconds: Double
    let checkpointFingerprint: String
    var result: ContiguousArray<Float> = [0.25, -0.5]
    var error: Error?

    init(observationDimension: Int = 3, actionDimension: Int = 2,
         controlPeriodSeconds: Double = 0.02,
         checkpointFingerprint: String = "qualified-test-policy") {
        self.observationDimension = observationDimension
        self.actionDimension = actionDimension
        self.controlPeriodSeconds = controlPeriodSeconds
        self.checkpointFingerprint = checkpointFingerprint
    }

    func actions(for observation: ContiguousArray<Float>) throws
        -> ContiguousArray<Float> {
        if let error { throw error }
        return result
    }
}

final class RLFrameworkTests: XCTestCase {
    func testArachneClassicalKinematicsAndWavePartition() {
        XCTAssertEqual(
            Arachne15ClassicalController.defaultWaveGroups.flatMap { $0 }
                .sorted(), Array(0..<8))
        for leg in 0..<8 {
            let hip: Float = leg.isMultiple(of: 2) ? 0.21 : -0.18
            let knee: Float = leg.isMultiple(of: 3) ? -0.22 : 0.16
            let foot = Arachne15ClassicalController.forwardKinematics(
                leg: leg, hip: hip, knee: knee)
            let recovered = Arachne15ClassicalController.inverseKinematics(
                leg: leg, footInBody: foot)
            XCTAssertEqual(recovered.hip, hip, accuracy: 1e-5)
            XCTAssertEqual(recovered.knee, knee, accuracy: 1e-5)
            XCTAssertFalse(recovered.wasConstrained)
        }
    }

    func testArachneRevealPlannerUsesReserveTravelAndReturnsHome() {
        let reveal = Arachne15RevealController()
        XCTAssertEqual(reveal.totalSteps, 482)
        XCTAssertEqual(reveal.phase, .folding)
        XCTAssertEqual(
            Arachne15RevealController.foldOrder.flatMap { $0 }.sorted(),
            Array(0..<8))
        XCTAssertTrue(
            Arachne15RevealController.foldOrder.allSatisfy { $0.count == 4 })
        XCTAssertEqual(
            Arachne15RevealController.unfoldOrder.flatMap { $0 }.sorted(),
            Array(0..<8))
        XCTAssertTrue(
            Arachne15RevealController.unfoldOrder.allSatisfy { $0.count == 4 })
        XCTAssertTrue(Arachne15RevealController.targetsRespectMechanicalLimits(
            Arachne15RevealController.compactJointTargets))
        XCTAssertGreaterThan(
            Arachne15RevealController.compactJointTargets[1],
            Arachne15PolicyContract.actionScales[1],
            "reveal should use reserved mechanical travel unavailable to RL")

        let compact = reveal.jointTargets(at: reveal.foldingSteps)
        XCTAssertEqual(
            compact, Arachne15RevealController.compactJointTargets)
        let final = reveal.jointTargets(at: reveal.totalSteps)
        XCTAssertEqual(final, Arachne15RevealController.deployedJointTargets)

        var previous = reveal.nextJointTargets()
        var maximumPerTickChange: Float = 0
        while !reveal.isComplete {
            let current = reveal.nextJointTargets()
            for index in current.indices {
                maximumPerTickChange = max(
                    maximumPerTickChange,
                    abs(current[index] - previous[index]))
            }
            previous = current
        }
        XCTAssertLessThan(maximumPerTickChange, 0.08)
        XCTAssertEqual(Array(previous),
                       Arachne15RevealController.deployedJointTargets)
    }

    func testArachneRevealIsExecutedByMotorsAndHandsBackPhysicalState()
        throws {
        let task = try Arachne15LocomotionTask(configuration: .init(
            numEnvironments: 1, seed: 42_050,
            maxEpisodeSteps: 500,
            standingCommandProbability: 1,
            initialRollPitchRange: 0, initialYawRange: 0,
            observationNoise: false, maximumActionLatencySteps: 0,
            domainRandomization: .init(), autoReset: false))
        var observation = try task.reset(seed: 42_051)
        let initial = task.environment.states()[0]
        let reveal = Arachne15RevealController(
            initialJointTargets: initial.jointAngles)
        task.environment.setCommissioningTorqueScale(
            Arachne15RevealController.commissioningTorqueScale)
        var minimumUpright: Float = 1
        var maximumMinimumCompactKnee: Float = -.infinity
        while !reveal.isComplete {
            let targets = reveal.nextJointTargets()
            task.environment.stepJointTargets(
                targets, decimation: task.configuration.controlDecimation)
            let state = task.environment.states()[0]
            minimumUpright = min(
                minimumUpright,
                state.root.rotation.conjugate.act(F3(0, 0, 1)).z)
            if reveal.phase == .compactHold {
                var minimumKneeThisTick = Float.infinity
                for knee in stride(from: 1, to: 16, by: 2) {
                    minimumKneeThisTick = min(
                        minimumKneeThisTick, state.jointAngles[knee])
                }
                maximumMinimumCompactKnee = max(
                    maximumMinimumCompactKnee, minimumKneeThisTick)
            }
        }
        task.environment.setCommissioningTorqueScale(1)
        let final = task.environment.states()[0]
        let planarDisplacement = simd_length(SIMD2<Float>(
            final.root.position.x - initial.root.position.x,
            final.root.position.y - initial.root.position.y))
        XCTAssertGreaterThan(
            maximumMinimumCompactKnee, 0.70,
            "every knee must physically reach the under-body compact pose")
        XCTAssertGreaterThan(minimumUpright, 0.85)
        XCTAssertGreaterThan(final.root.position.z, 0.06)
        XCTAssertLessThan(planarDisplacement, 0.04)
        XCTAssertLessThan(
            final.jointAngles.map(abs).max() ?? .infinity, 0.12,
            "final joints: \(final.jointAngles); root: \(final.root.position)")

        try task.resumeAfterCommissioning(into: &observation)
        XCTAssertTrue(observation.policy.allSatisfy(\.isFinite))
        var result = RLStepBatch(spec: task.spec)
        try task.step(
            actions: RLActionBatch(spec: task.spec), into: &result)
        XCTAssertFalse(result.terminated[0])
    }

    func testArachneClassicalControllerPhysicallyReachesFrontGoal() throws {
        let task = try Arachne15LocomotionTask(
            configuration: .init(
                numEnvironments: 1, seed: 42_100,
                maxEpisodeSteps: 800, initialRollPitchRange: 0,
                initialYawRange: 0, observationNoise: false,
                maximumActionLatencySteps: 0,
                domainRandomization: .init(), pointGoal: true,
                minimumGoalDistance: 0.6, maximumGoalDistance: 0.6,
                maximumGoalDirectionAngle: .pi,
                autoReset: false),
            taskID: "arachne15-goal-v0")
        try task.setGoal(
            environment: 0, direction: F3(1, 0, 0), distance: 0.6)
        _ = try task.reset(seed: 42_101)
        let controller = Arachne15ClassicalController()
        controller.reset(states: task.environment.states())
        var result = RLStepBatch(spec: task.spec)
        for _ in 0..<task.spec.maxEpisodeSteps {
            let actions = controller.actions(
                states: task.environment.states(),
                commands: [task.currentCommand(environment: 0)],
                spec: task.spec)
            XCTAssertTrue(actions.values.allSatisfy { (-1...1).contains($0) })
            try task.step(actions: actions, into: &result)
            if result.terminated[0] || result.truncated[0] { break }
        }
        XCTAssertTrue(result.successes[0])
        XCTAssertEqual(result.metrics["episode/survived"]?[0], 1)
        XCTAssertLessThanOrEqual(
            result.metrics["episode/final_goal_distance_m"]?[0] ?? .infinity,
            task.configuration.goalRadius)
        XCTAssertLessThan(
            result.metrics["episode/foot_collider_penetration_rmse_m"]?[0]
                ?? .infinity,
            0.0005)
    }

    func testRegisteredTasksExposeNormalizedBatchedSpaces() throws {
        XCTAssertEqual(BuiltInRLTasks.registry.taskIDs,
                       ["arachne15-goal-v0", "arachne15-velocity-v0",
                        "arm-pusht-v0",
                        "humanoid-goal-v0",
                        "humanoid-isaac-flat-v0", "humanoid-isaac-goal-v0",
                        "humanoid-velocity-v0", "humanoid-walk-v0",
                        "maniskill-pusht-v1", "pusht-state-v0"])
        XCTAssertEqual(VectorRLAlgorithmRegistry.builtIn.algorithmIDs, ["ppo"])

        let task = try BuiltInRLTasks.registry.make(
            "pusht-state-v0",
            configuration: RLTaskConfiguration(numEnvironments: 4, seed: 7))
        XCTAssertEqual(task.spec.numEnvironments, 4)
        XCTAssertEqual(task.spec.observation.shape, [12])
        XCTAssertEqual(task.spec.action.shape, [2])
        XCTAssertEqual(task.spec.action.lowerBound, [-1, -1])
        XCTAssertEqual(task.spec.action.upperBound, [1, 1])
    }

    func testArachneRegisteredTaskIsBatchedAndLoadBearing() throws {
        XCTAssertEqual(Arachne15LocomotionTask.creditedCommandProgress(
            measured: 0.02, commandSpeed: 0.15, controlStep: 0.02),
            0.003, accuracy: 1e-7)
        XCTAssertEqual(Arachne15LocomotionTask.creditedCommandProgress(
            measured: -0.01, commandSpeed: 0.15, controlStep: 0.02),
            -0.01, accuracy: 1e-7)
        let registered = try BuiltInRLTasks.registry.make(
            "arachne15-velocity-v0",
            configuration: RLTaskConfiguration(
                numEnvironments: 8, seed: 91,
                options: [
                    "standingCommandProbability": 1,
                    "initialRollPitchRange": 0,
                    "initialYawRange": 0,
                    "observationNoise": 0,
                    "maximumActionLatencySteps": 0,
                    "domainRandomization": 0,
                ]))
        let task = try XCTUnwrap(registered as? Arachne15LocomotionTask)
        XCTAssertEqual(task.spec.revision, 6)
        XCTAssertEqual(task.spec.observation.shape, [60])
        XCTAssertEqual(task.spec.action.shape, [16])
        XCTAssertEqual(task.configuration.goalCommandSpeed, 0.15)
        XCTAssertEqual(task.configuration.goalBoundaryCommandSpeed, 0.02)
        XCTAssertEqual(task.configuration.commandProgressRewardWeight, 20)
        XCTAssertEqual(task.configuration.velocityErrorPenaltyWeight, 5)
        XCTAssertEqual(task.configuration.yawErrorPenaltyWeight, 5)
        XCTAssertEqual(task.spec.simulationStep, 0.002, accuracy: 1e-7)
        XCTAssertEqual(task.spec.controlStep, 0.02, accuracy: 1e-7)
        XCTAssertEqual(task.environment.scene.settings.iterations, 20)
        XCTAssertEqual(task.spec.configurationValues["solverIterations"], 20)
        XCTAssertEqual(task.environment.scene.settings.collisionMargin,
                       0.00025, accuracy: 1e-8)
        for key in [
            "massScaleLower", "massScaleUpper",
            "inertiaScaleLower", "inertiaScaleUpper",
            "frictionScaleLower", "frictionScaleUpper",
            "motorTorqueScaleLower", "motorTorqueScaleUpper",
            "motorStiffnessScaleLower", "motorStiffnessScaleUpper",
            "motorDampingScaleLower", "motorDampingScaleUpper",
            "armatureScaleLower", "armatureScaleUpper",
        ] {
            XCTAssertEqual(task.spec.configurationValues[key], 1, key)
        }
        XCTAssertEqual(task.environment.refs.count, 8)
        XCTAssertEqual(task.environment.scene.rigidMeshes.count, 0,
                       "large headless batches must not replicate CAD meshes")
        XCTAssertEqual(task.environment.scene.colliders
            .filter(\.collisionEnabled).count, 1 + 8 * 39)
        for e in 0..<8 {
            XCTAssertEqual(task.currentCommand(environment: e), .zero)
            let groups = Set(task.environment.scene.colliders
                .filter { $0.body != task.environment.groundBody
                    && task.environment.refs[e].bodies.contains($0.body) }
                .map(\.collisionGroup))
            XCTAssertEqual(groups, [UInt32(e + 1)])
        }

        let observation = try task.reset(seed: 92)
        XCTAssertTrue(observation.policy.allSatisfy(\.isFinite))
        let actions = RLActionBatch(spec: task.spec)
        var result = RLStepBatch(spec: task.spec)
        for _ in 0..<25 {
            try task.step(actions: actions, into: &result)
            XCTAssertFalse(result.terminated.contains(true))
            XCTAssertTrue(result.observations.policy.allSatisfy(\.isFinite))
            XCTAssertTrue(result.metrics["state/root_height_m"]!
                .allSatisfy { $0 > 0.035 })
            let clearances = try XCTUnwrap(
                result.metrics["state/minimum_foot_collider_clearance_m"])
            XCTAssertTrue(clearances.allSatisfy(\.isFinite))
            XCTAssertTrue(clearances.allSatisfy { $0 > -0.001 },
                          "the 8 mm Arachne feet must not settle through the floor")
        }
    }

    func testArachneGoalSamplesDeterministicVisibleNonCollidingTargets() throws {
        let options: [String: Float] = [
            "minimumGoalDistance": 0.8,
            "maximumGoalDistance": 1.2,
            "maximumGoalDirectionAngle": 3.1415927,
            "initialRollPitchRange": 0,
            "initialYawRange": 0,
            "observationNoise": 0,
            "maximumActionLatencySteps": 0,
            "domainRandomization": 0,
        ]
        let registered = try BuiltInRLTasks.registry.make(
            "arachne15-goal-v0",
            configuration: RLTaskConfiguration(
                numEnvironments: 4, seed: 501, autoReset: false,
                options: options))
        let task = try XCTUnwrap(registered as? Arachne15LocomotionTask)
        XCTAssertTrue(task.usesPointGoal)
        XCTAssertEqual(task.spec.id, "arachne15-goal-v0")
        XCTAssertEqual(task.spec.revision, 6)
        XCTAssertEqual(task.spec.observation.shape, [60])
        XCTAssertEqual(task.spec.action.shape, [16])
        XCTAssertEqual(task.configuration.maximumGoalDirectionAngle, .pi)
        XCTAssertEqual(task.evaluationCriteria.minimumTaskMetrics[
            "episode/minimum_foot_collider_clearance_m"], -0.003)
        XCTAssertEqual(task.evaluationCriteria.maximumTaskMetrics[
            "episode/foot_collider_penetration_rmse_m"], 0.0005)
        XCTAssertEqual(task.evaluationCriteria.maximumTaskMetrics[
            "episode/foot_collider_penetration_over_1mm_fraction"], 0.025)
        XCTAssertEqual(task.initializationObservationVarianceFloors,
                       [9: 0.01, 10: 0.01, 11: 0.16])
        let actionsToMirror = ContiguousArray((0..<16).map(Float.init))
        let mirroredActions = task.mirrorPolicyActions(actionsToMirror)
        XCTAssertEqual(mirroredActions[0], -actionsToMirror[8])
        XCTAssertEqual(mirroredActions[1], actionsToMirror[9])
        XCTAssertEqual(task.mirrorPolicyActions(mirroredActions),
                       actionsToMirror)
        let observationsToMirror = ContiguousArray((0..<60).map(Float.init))
        let mirroredObservations = task.mirrorPolicyObservations(
            observationsToMirror)
        XCTAssertEqual(mirroredObservations[9], observationsToMirror[9])
        XCTAssertEqual(mirroredObservations[10], -observationsToMirror[10])
        XCTAssertEqual(mirroredObservations[11], -observationsToMirror[11])
        XCTAssertEqual(task.mirrorPolicyObservations(mirroredObservations),
                       observationsToMirror)
        XCTAssertEqual(task.environment.scene.colliders
            .filter(\.collisionEnabled).count, 1 + 4 * 39,
            "visual goals must not change physical collision topology")
        for ref in task.environment.refs {
            let start = try XCTUnwrap(ref.startMarker)
            let goal = try XCTUnwrap(ref.goalMarker)
            let markerColliders = task.environment.scene.colliders.filter {
                $0.body == start || $0.body == goal
            }
            XCTAssertEqual(markerColliders.count, 3)
            XCTAssertTrue(markerColliders.allSatisfy {
                !$0.collisionEnabled && $0.isRendered
            })
            XCTAssertFalse(ref.bodies.contains(start))
            XCTAssertFalse(ref.bodies.contains(goal))
        }

        _ = try task.reset(seed: 777)
        let firstGoals = (0..<4).map {
            task.currentGoalPosition(environment: $0)
        }
        let firstRoots = task.environment.states().map(\.root.position)
        for e in 0..<4 {
            let delta = firstGoals[e] - firstRoots[e]
            let distance = sqrt(delta.x * delta.x + delta.y * delta.y)
            XCTAssertTrue((0.8...1.2).contains(distance))
            let command = task.currentCommand(environment: e)
            XCTAssertLessThanOrEqual(
                sqrt(command.x * command.x + command.y * command.y),
                0.15 + 1e-6)
            XCTAssertLessThanOrEqual(abs(command.z), 0.8 + 1e-6)
        }
        _ = try task.reset(seed: 777)
        XCTAssertEqual((0..<4).map {
            task.currentGoalPosition(environment: $0)
        }, firstGoals)

        try task.setGoal(
            environment: 0, direction: F3(0, 2, 0), distance: 1.1)
        let northObservation = try task.reset(seed: 778)
        let state = task.environment.states()[0]
        let goal = task.currentGoalPosition(environment: 0)
        XCTAssertEqual(goal.x, state.root.position.x, accuracy: 1e-5)
        XCTAssertEqual(goal.y - state.root.position.y, 1.1, accuracy: 1e-5)
        let marker = try XCTUnwrap(task.environment.refs[0].goalMarker)
        let markerState = task.environment.solver.bodyStates([marker])[0]
        XCTAssertEqual(markerState.position.x, goal.x, accuracy: 1e-5)
        XCTAssertEqual(markerState.position.y, goal.y, accuracy: 1e-5)

        try task.setGoal(
            environment: 0, direction: F3(2, 0, 0), distance: 1.1)
        let eastObservation = try task.reset(seed: 778)
        for i in 0..<Arachne15LocomotionTask.observationDimension
            where !(9..<12).contains(i) {
            XCTAssertEqual(
                northObservation.policy[i], eastObservation.policy[i],
                accuracy: 1e-6,
                "goal marker/coordinates leaked into observation index \(i)")
        }
        XCTAssertNotEqual(
            northObservation.policy[9], eastObservation.policy[9],
            "only the documented local command should expose target intent")

        let actions = RLActionBatch(spec: task.spec)
        var result = RLStepBatch(spec: task.spec)
        try task.step(actions: actions, into: &result)
        XCTAssertTrue(result.observations.policy.allSatisfy(\.isFinite))
        XCTAssertNotNil(result.metrics["state/goal_distance_m"])
        XCTAssertNotNil(result.metrics["reward/goal_progress"])
        XCTAssertNotNil(result.metrics["reward/command_progress"])
        XCTAssertNotNil(result.metrics["state/command_progress_m"])
        XCTAssertNotNil(result.metrics[
            "episode/foot_collider_penetration_rmse_m"])
        XCTAssertNotNil(result.metrics[
            "episode/foot_collider_penetration_over_1mm_fraction"])

        let replayTask = try BuiltInRLTasks.registry.make(
            "arachne15-goal-v0",
            configuration: RLTaskConfiguration(
                numEnvironments: 1, seed: 501, autoReset: false,
                options: task.spec.configurationValues))
        XCTAssertEqual(
            replayTask.spec.configurationValues,
            task.spec.configurationValues,
            "serialized training configuration must reconstruct exact replay")
    }

    func testArachneHardwarePolicyContractMatchesSimulatorEncoding() throws {
        XCTAssertEqual(Arachne15PolicyContract.jointNames.count, 16)
        XCTAssertEqual(Arachne15PolicyContract.jointNames.first,
                       "right_rear_hip")
        XCTAssertEqual(Arachne15PolicyContract.jointNames.last,
                       "left_front_knee")
        XCTAssertEqual(Arachne15PolicyContract.actionScales,
                       Arachne15Env.actionScales)
        let task = try Arachne15LocomotionTask(configuration: .init(
            numEnvironments: 1, seed: 710,
            standingCommandProbability: 1,
            initialRollPitchRange: 0, initialYawRange: 0,
            observationNoise: false, maximumActionLatencySteps: 0,
            domainRandomization: .init(), autoReset: false))
        let actual = try task.reset(seed: 711)
        let state = task.environment.states()[0]
        let inverse = state.root.rotation.conjugate
        let expected = try Arachne15PolicyContract.encode(.init(
            bodyLinearVelocity: inverse.act(state.root.linearVelocity),
            bodyAngularVelocity: inverse.act(state.root.angularVelocity),
            projectedGravity: inverse.act(F3(0, 0, 1)),
            commandedBodyTwist: .zero,
            jointPositions: state.jointAngles,
            jointVelocities: state.jointVelocities,
            previousActions: [Float](repeating: 0, count: 16)))
        XCTAssertEqual(actual.policy, expected)

        let forward = Arachne15PolicyContract.pointGoalCommand(
            worldGoal: F3(1, 0, 0), rootPosition: .zero,
            rootRotation: Quat(ix: 0, iy: 0, iz: 0, r: 1),
            goalRadius: 0.12, slowdownDistance: 0.5,
            cruiseSpeed: 0.15, boundarySpeed: 0.02,
            maximumYawRate: 0.8)
        XCTAssertEqual(forward, F3(0.15, 0, 0))
        let targets = try Arachne15PolicyContract.relativeJointTargets(
            for: ContiguousArray((0..<16).map {
                $0.isMultiple(of: 2) ? 1 : -1
            }))
        for j in targets.indices {
            XCTAssertEqual(targets[j], j.isMultiple(of: 2) ? 0.35 : -0.45)
        }
    }

    func testArachneHardwareCalibrationFailsClosedAndMapsActuators() throws {
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let templateURL = packageRoot.appendingPathComponent(
            "Robots/Arachne15/iphone/hardware-calibration.template.json")
        let template = try JSONDecoder().decode(
            Arachne15HardwareCalibration.self,
            from: Data(contentsOf: templateURL))
        let fingerprint = template.policyCheckpointFingerprint
        XCTAssertThrowsError(try template.validate(
            expectedPolicyFingerprint: fingerprint))
        XCTAssertTrue(template.validationFailures(
            expectedPolicyFingerprint: fingerprint).contains(
                "calibration is not commissioned"))

        let lower: [Float] = (0..<16).map {
            $0.isMultiple(of: 2) ? -0.55 : -0.7
        }
        let upper: [Float] = (0..<16).map {
            $0.isMultiple(of: 2) ? 0.55 : 0.7
        }
        let calibration = Arachne15HardwareCalibration(
            robotSerial: "AR15-001", commissioned: true,
            measuredAtUTC: "2026-07-16T00:00:00Z",
            policyCheckpointFingerprint: fingerprint,
            servoIDs: Array(1...16),
            servoZeroRadians: [Float](repeating: 2.5, count: 16),
            servoDirectionSigns: (0..<16).map {
                $0 < 8 ? Float(1) : Float(-1)
            },
            jointLowerRadians: lower, jointUpperRadians: upper,
            currentLimitsMilliamps: [Int](repeating: 300, count: 16),
            maximumServoTemperatureCelsius: 55,
            measuredMaximumRoundTripLatencySeconds: 0.018)
        try calibration.validate(expectedPolicyFingerprint: fingerprint)
        let actions = ContiguousArray((0..<16).map {
            $0.isMultiple(of: 2) ? Float(0.5) : Float(-0.5)
        })
        let servo = try calibration.servoPositionRadians(
            for: actions, expectedPolicyFingerprint: fingerprint)
        let recovered = try calibration.policyJointPositions(
            servoPositionRadians: servo,
            expectedPolicyFingerprint: fingerprint)
        let expected = try Arachne15PolicyContract.relativeJointTargets(
            for: actions)
        for j in 0..<16 {
            XCTAssertEqual(recovered[j], expected[j], accuracy: 1e-6)
        }
        let velocity = try calibration.policyJointVelocities(
            servoVelocityRadiansPerSecond: [Float](repeating: 0.2, count: 16),
            expectedPolicyFingerprint: fingerprint)
        XCTAssertEqual(velocity[0], 0.2)
        XCTAssertEqual(velocity[8], -0.2)
        XCTAssertThrowsError(try calibration.validate(
            expectedPolicyFingerprint: "another-policy"))
    }

    func testArachneGoalReportsDirectionalAcceptanceCohorts() throws {
        let registered = try BuiltInRLTasks.registry.make(
            "arachne15-goal-v0",
            configuration: RLTaskConfiguration(
                numEnvironments: 8, seed: 612, autoReset: false,
                options: [
                    "maxEpisodeSteps": 1,
                    "minimumGoalDistance": 1,
                    "maximumGoalDistance": 1,
                    "maximumGoalDirectionAngle": 0.1,
                    "initialRollPitchRange": 0,
                    "initialYawRange": 0,
                    "observationNoise": 0,
                    "maximumActionLatencySteps": 0,
                    "domainRandomization": 0,
                ]))
        let task = try XCTUnwrap(registered as? Arachne15LocomotionTask)
        _ = try task.reset(seed: 613)
        var result = RLStepBatch(spec: task.spec)
        try task.step(actions: RLActionBatch(spec: task.spec), into: &result)
        XCTAssertTrue(result.truncated.allSatisfy { $0 })
        for e in 0..<task.spec.numEnvironments {
            let directionalBins = [
                "episode/goal_front_bin", "episode/goal_left_bin",
                "episode/goal_rear_bin", "episode/goal_right_bin",
            ].reduce(Float(0)) { partial, name in
                partial + (result.metrics[name]?[e] ?? 0)
            }
            XCTAssertEqual(directionalBins, 1)
            XCTAssertEqual(result.metrics["episode/goal_front_bin"]?[e], 1)
            XCTAssertEqual(result.metrics["episode/goal_near_bin"]?[e], 1)
            XCTAssertEqual(result.metrics["episode/goal_far_bin"]?[e], 0)
            XCTAssertTrue(result.metrics[
                "episode/minimum_foot_collider_clearance_m"]![e].isFinite)
        }
    }

    func testBuiltInTasksRejectUnknownOptionsInsteadOfUsingDefaults() {
        XCTAssertThrowsError(try BuiltInRLTasks.registry.make(
            "humanoid-goal-v0",
            configuration: RLTaskConfiguration(
                numEnvironments: 1,
                options: ["goalSlowdownDistanceMeters": 5]))) { error in
            let message = String(describing: error)
            XCTAssertTrue(message.contains("goalSlowdownDistanceMeters"))
            XCTAssertTrue(message.contains("goalSlowdownDistance"))
        }
        XCTAssertThrowsError(try BuiltInRLTasks.registry.make(
            "arachne15-goal-v0",
            configuration: RLTaskConfiguration(
                numEnvironments: 1,
                options: [
                    "massScaleLower": 1.2,
                    "massScaleUpper": 0.8,
                ]))) { error in
            XCTAssertTrue(String(describing: error).contains("massScale"))
        }
    }

    func testHumanoidGoalAcceptsSerializedZeroStandingProbabilityOnly() throws {
        let task = try BuiltInRLTasks.registry.make(
            "humanoid-goal-v0",
            configuration: RLTaskConfiguration(
                numEnvironments: 1, seed: 9,
                options: [
                    "commandGatedActor": 1,
                    "standingCommandProbability": 0,
                ]))
        XCTAssertEqual(
            task.spec.configurationValues["standingCommandProbability"], 0)

        XCTAssertThrowsError(try BuiltInRLTasks.registry.make(
            "humanoid-goal-v0",
            configuration: RLTaskConfiguration(
                numEnvironments: 1, seed: 9,
                options: ["standingCommandProbability": 0.1]))) { error in
            XCTAssertTrue(String(describing: error).contains(
                "requires standingCommandProbability=0"))
        }
    }

    func testRegisteredHumanoidUsesLocomotionCommandRange() throws {
        let registered = try BuiltInRLTasks.registry.make(
            "humanoid-walk-v0",
            configuration: RLTaskConfiguration(numEnvironments: 2, seed: 41))
        let task = try XCTUnwrap(registered as? HumanoidWalkTask)
        XCTAssertEqual(task.spec.revision,
                       RLPhysicsContract.fixedGainActuatorV2(35))
        XCTAssertEqual(task.configuration.minimumCommandSpeed, 0.45)
        XCTAssertEqual(task.configuration.maximumCommandSpeed, 0.65)
        XCTAssertEqual(task.configuration.standingCommandProbability, 0)
        XCTAssertEqual(task.configuration.standStillVelocityPenaltyWeight, 2)
        XCTAssertEqual(task.configuration.standStillJointDeviationPenaltyWeight, 1)
        XCTAssertEqual(task.configuration.actionTargetResponse, 1)
        XCTAssertEqual(task.configuration.alternatingTouchdownRewardWeight, 2)
        XCTAssertEqual(task.configuration.lateralPenaltyWarmupControlSteps, 0)
        XCTAssertEqual(task.configuration.lateralPenaltyRampControlSteps, 0)
        XCTAssertEqual(task.configuration.laneTrackingStandardDeviation, 0.30)
        XCTAssertEqual(task.configuration.flightPenaltyWeight, 1)
        XCTAssertEqual(task.configuration.initialRollPitchRange, 0.015)
        XCTAssertEqual(task.configuration.initialYawRange, 0.05)
        XCTAssertEqual(task.spec.configurationValues[
            "alternatingTouchdownRewardWeight"], 2)
        XCTAssertEqual(task.spec.configurationValues[
            "lateralPenaltyWarmupControlSteps"], 0)
        _ = try task.reset(seed: 43)
        for environment in 0..<2 {
            XCTAssertTrue((0.45...0.65).contains(
                task.currentCommandSpeed(environment: environment)))
        }
        task.setTrainingMode(true)
        _ = try task.reset(seed: 44)
        XCTAssertEqual(task.trainingCurriculumProgress, 1)
        for environment in 0..<2 {
            XCTAssertTrue((0.45...0.65).contains(
                task.currentCommandSpeed(environment: environment)))
        }
    }

    func testRegisteredHumanoidVelocityTaskMatchesUpstreamJoystickInterface()
        throws {
        let registered = try BuiltInRLTasks.registry.make(
            "humanoid-velocity-v0",
            configuration: RLTaskConfiguration(
                numEnvironments: 4, seed: 44,
                options: ["standingCommandProbability": 1]))
        let task = try XCTUnwrap(registered as? HumanoidVelocityTask)
        XCTAssertEqual(task.spec.revision,
                       RLPhysicsContract.fixedGainActuatorV2(2))
        XCTAssertEqual(task.spec.observation.shape, [675])
        XCTAssertEqual(task.spec.action.shape, [19])
        XCTAssertEqual(task.spec.action.lowerBound, [Float](
            repeating: -1, count: 19))
        XCTAssertEqual(task.spec.action.upperBound, [Float](
            repeating: 1, count: 19))
        XCTAssertEqual(task.spec.simulationStep, 0.004, accuracy: 1e-7)
        XCTAssertEqual(task.spec.controlStep, 0.02, accuracy: 1e-7)
        XCTAssertEqual(task.configuration.controlDecimation, 5)
        XCTAssertEqual(task.environment.controlProfile, .mujocoPlayground)
        XCTAssertEqual(task.configuration.commandResamplingSteps, 500)
        XCTAssertEqual(task.configuration.minimumForwardVelocity, -0.6)
        XCTAssertEqual(task.configuration.maximumForwardVelocity, 1.5)
        XCTAssertEqual(task.configuration.minimumLateralVelocity, -0.8)
        XCTAssertEqual(task.configuration.maximumLateralVelocity, 0.8)
        XCTAssertEqual(task.configuration.minimumYawRate, -0.7)
        XCTAssertEqual(task.configuration.maximumYawRate, 0.7)
        XCTAssertEqual(task.initializationObservationVarianceFloors.count, 45)
        for history in 0..<HumanoidVelocityTask.observationHistorySteps {
            let base = history
                * HumanoidVelocityTask.observationFrameDimension
            XCTAssertEqual(
                task.initializationObservationVarianceFloors[base + 4], 0.25)
            XCTAssertEqual(
                task.initializationObservationVarianceFloors[base + 5], 0.25)
            XCTAssertEqual(
                task.initializationObservationVarianceFloors[base + 6], 0.25)
        }

        let observation = try task.reset(seed: 45)
        for environment in 0..<task.spec.numEnvironments {
            XCTAssertEqual(task.currentCommand(environment: environment), .zero)
            let row = environment * task.spec.observation.elementCount
            for history in 0..<HumanoidVelocityTask.observationHistorySteps {
                let frame = row + history
                    * HumanoidVelocityTask.observationFrameDimension
                XCTAssertEqual(observation.policy[frame + 4], 0)
                XCTAssertEqual(observation.policy[frame + 5], 0)
                XCTAssertEqual(observation.policy[frame + 6], 0)
            }
        }
        XCTAssertTrue(observation.policy.allSatisfy(\.isFinite))

        for (environment, state) in task.environment.states().enumerated() {
            XCTAssertEqual(state.root.position.z, 0.97, accuracy: 1e-5)
            XCTAssertTrue(state.jointAngles.allSatisfy { abs($0) < 1e-4 })
            let ref = task.environment.refs[environment]
            for motor in ref.motors {
                let joint = task.environment.scene.joints[motor]
                XCTAssertEqual(joint.motorStiffness,
                               motor == ref.motors[10] ? 400
                                   : motor >= ref.motors[11] ? 400 : 800)
            }
            XCTAssertEqual(
                task.environment.scene.joints[ref.motors[0]].motorDamping,
                26.398067, accuracy: 1e-5)
            XCTAssertEqual(
                task.environment.scene.joints[ref.motors[4]].motorDamping,
                19.341247, accuracy: 1e-5)
        }
        let enabledRobotColliders = task.environment.scene.colliders.filter {
            $0.collisionEnabled && $0.body != task.environment.groundBody
        }
        XCTAssertEqual(enabledRobotColliders.count, 4 * 6)
        XCTAssertTrue(enabledRobotColliders.allSatisfy { collider in
            task.environment.refs.contains { ref in
                collider.body == ref.leftFoot || collider.body == ref.rightFoot
            }
        })
        XCTAssertTrue(task.environment.scene.colliders.contains {
            !$0.collisionEnabled && $0.isRendered
        })

        XCTAssertThrowsError(try HumanoidVelocityTask(configuration: .init(
            numEnvironments: 1, maxEpisodeSteps: 500,
            commandResamplingSteps: 500)))
    }

    func testHumanoidVelocityPolicySymmetryIsAnExactInvolution() throws {
        let task = try HumanoidVelocityTask(configuration: .init(
            numEnvironments: 2, seed: 46))
        let actions = ContiguousArray((0..<38).map { Float($0) - 8.5 })
        let mirroredActions = task.mirrorPolicyActions(actions)
        XCTAssertEqual(task.mirrorPolicyActions(mirroredActions), actions)

        let count = 2 * HumanoidVelocityTask.observationDimension
        let observation = ContiguousArray((0..<count).map {
            Float($0) * 0.01 + 0.125
        })
        let mirrored = task.mirrorPolicyObservations(observation)
        XCTAssertEqual(task.mirrorPolicyObservations(mirrored), observation)
        XCTAssertEqual(mirrored[0], -observation[0])
        XCTAssertEqual(mirrored[1], observation[1])
        XCTAssertEqual(mirrored[2], -observation[2])
        XCTAssertEqual(mirrored[3], observation[3])
        XCTAssertEqual(mirrored[4], observation[4])
        XCTAssertEqual(mirrored[5], -observation[5])
        XCTAssertEqual(mirrored[6], -observation[6])
        XCTAssertEqual(mirrored[7], -observation[12])
        XCTAssertEqual(mirrored[9], observation[14])
        XCTAssertEqual(mirrored[26], -observation[31])
    }

    func testHumanoidVelocityFirstResetIsLoadBearing() throws {
        let task = try HumanoidVelocityTask(configuration: .init(
            numEnvironments: 8, seed: 24_002,
            minimumForwardVelocity: 1,
            maximumForwardVelocity: 1,
            minimumLateralVelocity: 0,
            maximumLateralVelocity: 0,
            minimumYawRate: 0,
            maximumYawRate: 0,
            standingCommandProbability: 0,
            initialRollPitchRange: 0,
            initialYawRange: 0))
        _ = try task.reset(seed: 24_002)
        let actions = RLActionBatch(spec: task.spec)
        var result = RLStepBatch(spec: task.spec)
        for _ in 0..<25 {
            try task.step(actions: actions, into: &result)
            XCTAssertFalse(result.terminated.contains(true))
            XCTAssertTrue(
                result.metrics["state/torso_height_m"]!.allSatisfy {
                    $0 >= 0.92
                })
            XCTAssertTrue(
                result.metrics["state/feet_in_contact"]!.allSatisfy {
                    $0 == 2
                })
        }
    }

    func testRegisteredIsaacH1FlatTaskMatchesPublicInterface() throws {
        let registered = try BuiltInRLTasks.registry.make(
            "humanoid-isaac-flat-v0",
            configuration: RLTaskConfiguration(
                numEnvironments: 4, seed: 24_100,
                options: ["observationNoise": 0]))
        let task = try XCTUnwrap(
            registered as? HumanoidIsaacVelocityTask)
        XCTAssertEqual(task.spec.revision,
                       RLPhysicsContract.fixedGainActuatorV2(10))
        XCTAssertEqual(task.spec.observation.shape, [69])
        XCTAssertEqual(task.spec.action.shape, [19])
        XCTAssertEqual(task.spec.simulationStep, 0.005, accuracy: 1e-7)
        XCTAssertEqual(task.spec.controlStep, 0.02, accuracy: 1e-7)
        XCTAssertEqual(task.environment.scene.settings.gravity,
                       -9.81, accuracy: 1e-7)
        XCTAssertEqual(task.environment.scene.settings.frictionCombineMode,
                       .multiply)
        let rootBody = task.environment.scene.bodies[
            task.environment.refs[0].root]
        XCTAssertEqual(rootBody.rotation.real, 1, accuracy: 1e-6,
                       "Isaac USD omits principalAxes, so inertia is link-aligned")
        XCTAssertLessThan(simd_length(rootBody.rotation.imag), 1e-6)
        XCTAssertNil(task.spec.action.lowerBound)
        XCTAssertNil(task.spec.action.upperBound)
        XCTAssertEqual(task.environment.controlProfile, .isaacLab)
        XCTAssertEqual(task.spec.configurationValues[
            "standingCommandProbability"], 0.02)
        for ref in task.environment.refs {
            for (index, motor) in ref.motors.enumerated() {
                XCTAssertEqual(task.environment.scene.joints[motor].motorTorque,
                               index == 4 || index == 9 ? 100 : 300)
                XCTAssertEqual(task.environment.scene.joints[motor].armature,
                               0.1, accuracy: 1e-6)
            }
        }
        let observation = try task.reset(seed: 24_101)
        XCTAssertTrue(observation.policy.allSatisfy(\.isFinite))
        for environment in 0..<task.spec.numEnvironments {
            let base = environment * HumanoidIsaacVelocityTask.observationDimension
            XCTAssertEqual(observation.policy[base + 6], 0, accuracy: 1e-5)
            XCTAssertEqual(observation.policy[base + 7], 0, accuracy: 1e-5)
            XCTAssertEqual(observation.policy[base + 8], -1, accuracy: 1e-5)
        }
        XCTAssertTrue(task.environment.states().allSatisfy {
            abs($0.root.position.z - 1.05) < 1e-5
                && $0.jointVelocities.allSatisfy { abs($0) < 1e-5 }
        })
        for (environment, state) in task.environment.states().enumerated() {
            let command = task.currentCommand(environment: environment)
            let direction = task.currentCommandDirection(
                environment: environment)
            let projection = task.currentCommandProjection(
                environment: environment)
            let origin = F3(
                state.root.position.x, state.root.position.y,
                task.environment.refs[environment].center.z)
            XCTAssertEqual(simd_length(direction), 1, accuracy: 1e-5)
            XCTAssertEqual(
                simd_length(projection - origin),
                command.x * Float(task.configuration.commandResamplingSteps)
                    * task.spec.controlStep,
                accuracy: 1e-4)
            let left = HumanoidIsaacVelocityTask.footHullGroundClearance(
                state.leftFoot, vertices: IsaacH1CollisionHulls.leftAnkle)
            let right = HumanoidIsaacVelocityTask.footHullGroundClearance(
                state.rightFoot, vertices: IsaacH1CollisionHulls.rightAnkle)
            XCTAssertTrue(left.isFinite && right.isFinite)
            XCTAssertGreaterThan(left, -0.02,
                                 "left cooked foot hull starts too deep: \(left)")
            XCTAssertGreaterThan(right, -0.02,
                                  "right cooked foot hull starts too deep: \(right)")
        }

        let enabledRobotColliders = task.environment.scene.colliders.filter {
            $0.collisionEnabled && $0.body != task.environment.groundBody
        }
        XCTAssertEqual(enabledRobotColliders.count, 3 * task.spec.numEnvironments)
        XCTAssertTrue(enabledRobotColliders.allSatisfy {
            $0.convexHullVertices.count == 34 && !$0.isRendered
        })
        XCTAssertTrue(enabledRobotColliders.allSatisfy {
            $0.friction == 0.8 && $0.dynamicFriction == 0.6
        })
        XCTAssertTrue(enabledRobotColliders.allSatisfy { collider in
            task.environment.refs.contains { ref in
                collider.body == ref.leftFoot || collider.body == ref.rightFoot
                    || collider.body == ref.torso
            }
        })
        for ref in task.environment.refs {
            XCTAssertTrue(enabledRobotColliders.contains {
                $0.body == ref.leftFoot
            })
            XCTAssertTrue(enabledRobotColliders.contains {
                $0.body == ref.rightFoot
            })
            XCTAssertTrue(enabledRobotColliders.contains {
                $0.body == ref.torso
            })
        }

        let actions = ContiguousArray((0..<38).map { Float($0) - 8.5 })
        XCTAssertEqual(
            task.mirrorPolicyActions(task.mirrorPolicyActions(actions)),
            actions)
        let policy = ContiguousArray((0..<(4 * 69)).map {
            Float($0) * 0.01 - 0.5
        })
        XCTAssertEqual(
            task.mirrorPolicyObservations(
                task.mirrorPolicyObservations(policy)),
            policy)
    }

    func testExternalUnitreeH1PolicyRetainsVerifiedTenSecondTransfer() throws {
        let repository = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let policyDirectory = repository.appendingPathComponent(
            "checkpoints/external/unitree-h1").path
        guard FileManager.default.fileExists(
            atPath: policyDirectory + "/policy.safetensors"),
              FileManager.default.fileExists(
                atPath: policyDirectory + "/manifest.json") else {
            throw XCTSkip("external Unitree H1 checkpoint is not installed")
        }

        let session = try UnitreeH1Sim2SimSession(
            policyDirectory: policyDirectory,
            command: SIMD3<Float>(0.5, 0, 0))
        let report = try session.run(controlSteps: 500)

        XCTAssertEqual(report.checkpointSHA256,
            "44a0fbceb81f3877833ae9a398d039bea1759cb0d3c8188181013885f70589eb")
        XCTAssertTrue(report.policyVerification.passed)
        XCTAssertTrue(report.finite)
        XCTAssertFalse(report.fell)
        XCTAssertTrue((3.9...4.15).contains(report.forwardDistanceMeters))
        XCTAssertGreaterThan(report.minimumPelvisHeightMeters, 0.96)
        XCTAssertGreaterThan(report.minimumUprightAlignment, 0.995)
        XCTAssertLessThan(abs(report.lateralDistanceMeters), 0.25)
    }

    func testRegisteredIsaacH1GoalTransfersFlatPolicyAndOwnsArrival()
        throws {
        let registered = try BuiltInRLTasks.registry.make(
            "humanoid-isaac-goal-v0",
            configuration: RLTaskConfiguration(
                numEnvironments: 2, seed: 24_200, autoReset: false,
                options: [
                    "observationNoise": 0,
                    "initialYawRange": 0,
                    "minimumGoalDistance": 4,
                    "maximumGoalDistance": 4,
                    "goalDwellSteps": 2,
                ]))
        let task = try XCTUnwrap(
            registered as? HumanoidIsaacVelocityTask)
        XCTAssertEqual(task.spec.id, "humanoid-isaac-goal-v0")
        XCTAssertEqual(task.spec.revision,
                       RLPhysicsContract.fixedGainActuatorV2(2))
        XCTAssertTrue(task.usesPointGoal)
        XCTAssertEqual(task.spec.observation.shape, [71])
        XCTAssertEqual(task.spec.action.shape, [19])
        XCTAssertEqual(task.spec.configurationValues["pointGoal"], 1)
        let replayConfigured = try BuiltInRLTasks.registry.make(
            "humanoid-isaac-goal-v0",
            configuration: RLTaskConfiguration(
                numEnvironments: 1, seed: 24_200, autoReset: false,
                options: task.spec.configurationValues))
        XCTAssertEqual(
            replayConfigured.spec.configurationValues,
            task.spec.configurationValues,
            "serialized task configuration must reconstruct in replay")
        let mapping = try XCTUnwrap(task
            .initializationObservationSourceIndices(sourceDimension: 69))
        XCTAssertEqual(mapping.count, 71)
        XCTAssertEqual(mapping[0], 0)
        XCTAssertEqual(mapping[68], 68)
        XCTAssertNil(mapping[69])
        XCTAssertNil(mapping[70])

        let observation = try task.reset(seed: 24_201)
        XCTAssertEqual(observation.policy.count, 2 * 71)
        for environment in 0..<2 {
            let root = task.environment.states()[environment].root.position
            let goal = task.currentGoalPosition(environment: environment)
            XCTAssertEqual(
                simd_length(SIMD2(goal.x - root.x, goal.y - root.y)),
                4, accuracy: 1e-5)
            XCTAssertEqual(task.currentCommand(environment: environment).x,
                           0.55, accuracy: 1e-6)
        }
        XCTAssertEqual(task.mirrorPolicyObservations(
            task.mirrorPolicyObservations(observation.policy)),
            observation.policy)

        try task.setGoal(environment: 0, direction: F3(1, 0, 0),
                         distance: 0.1)
        XCTAssertEqual(task.currentCommand(environment: 0), .zero)
        var result = RLStepBatch(spec: task.spec)
        let zero = RLActionBatch(spec: task.spec)
        try task.step(actions: zero, into: &result)
        try task.step(actions: zero, into: &result)
        XCTAssertTrue(result.terminated[0])
        XCTAssertTrue(result.successes[0])
        XCTAssertEqual(result.metrics["episode/goal_reached"]?[0], 1)
        XCTAssertLessThanOrEqual(
            result.metrics["episode/final_goal_distance_m"]?[0] ?? .infinity,
            task.configuration.goalRadius)
    }

    func testIsaacH1GoalLaunchesBatchedPhysicalProjectiles() throws {
        let registered = try BuiltInRLTasks.registry.make(
            "humanoid-isaac-goal-v0",
            configuration: RLTaskConfiguration(
                numEnvironments: 2, seed: 24_300, autoReset: false,
                options: [
                    "maxEpisodeSteps": 2,
                    "commandResamplingSteps": 1,
                    "observationNoise": 0,
                    "initialYawRange": 0,
                    "minimumGoalDistance": 4,
                    "maximumGoalDistance": 4,
                    "projectileProbability": 1,
                    "projectileSize": 0.25,
                    "minimumProjectileSpeed": 4,
                    "maximumProjectileSpeed": 4,
                    "minimumProjectileLaunchStep": 0,
                    "maximumProjectileLaunchStep": 0,
                ]))
        let task = try XCTUnwrap(
            registered as? HumanoidIsaacVelocityTask)
        XCTAssertEqual(task.spec.revision,
                       RLPhysicsContract.fixedGainActuatorV2(4))
        XCTAssertEqual(task.trainingProjectileProbability, 1)
        for e in 0..<2 {
            XCTAssertTrue(task.hasProjectile(environment: e))
            let projectile = try XCTUnwrap(task.environment.refs[e].projectile)
            XCTAssertEqual(task.environment.solver.bodyMass(projectile), 8,
                           accuracy: 1e-5)
            let groups = task.environment.scene.colliders
                .filter { $0.body == projectile }.map(\.collisionGroup)
            XCTAssertEqual(groups, [UInt32(e + 1)])
            let ref = task.environment.refs[e]
            let authoredRobotPrimitives = task.environment.scene.colliders
                .filter {
                    ref.bodies.contains($0.body)
                        && $0.convexHullVertices.isEmpty
                }
            XCTAssertFalse(authoredRobotPrimitives.isEmpty)
            XCTAssertTrue(authoredRobotPrimitives
                .allSatisfy(\.collisionEnabled),
                "every imported H1 primitive, including hidden protective "
                    + "colliders, must receive external contacts")
            let renderedRobotColliders = task.environment.scene.colliders
                .filter { ref.bodies.contains($0.body) && $0.isRendered }
            XCTAssertFalse(renderedRobotColliders.isEmpty)
            XCTAssertTrue(renderedRobotColliders.allSatisfy(\.collisionEnabled),
                          "every rendered H1 collision primitive must be physical")
            for hand in [ref.leftHand, ref.rightHand] {
                let handColliders = renderedRobotColliders.filter {
                    $0.body == hand
                }
                XCTAssertEqual(handColliders.count, 2,
                               "forearm capsule and terminal hand sphere")
                XCTAssertTrue(handColliders.allSatisfy(\.collisionEnabled))
            }
        }

        _ = try task.reset(seed: 24_301)
        var result = RLStepBatch(spec: task.spec)
        let zero = RLActionBatch(spec: task.spec)
        try task.step(actions: zero, into: &result)
        for e in 0..<2 {
            let projectile = try XCTUnwrap(task.environment.refs[e].projectile)
            let state = task.environment.solver.bodyStates([projectile])[0]
            XCTAssertGreaterThan(simd_length(state.linearVelocity), 3)
            XCTAssertGreaterThan(state.position.z, 0.5)
        }
        try task.step(actions: zero, into: &result)
        for e in 0..<2 {
            XCTAssertTrue(result.terminated[e] || result.truncated[e])
            XCTAssertEqual(result.metrics["episode/disturbed_bin"]?[e], 1)
            XCTAssertEqual(result.metrics["episode/projectile_launched"]?[e], 1)
        }
        for metric in [
            "episode/projectile_left_bin",
            "episode/projectile_right_bin",
            "episode/projectile_early_bin",
            "episode/projectile_late_bin",
            "episode/impact_to_terminal_steps",
            "episode/fell_after_impact",
            "episode/minimum_post_impact_upright_cosine",
            "episode/minimum_post_impact_torso_height_m",
        ] {
            XCTAssertEqual(result.metrics[metric]?.count, 2,
                           "missing projectile diagnostic \(metric)")
        }

        task.environment.reset([0, 1], seeds: [24_302, 24_303])
        let ref = task.environment.refs[0]
        let hand = task.environment.solver.bodyStates([ref.leftHand])[0]
        task.environment.throwProjectiles(
            environmentIDs: [0],
            positions: [hand.position + F3(0, 0.50, 0)],
            velocities: [F3(0, -6, 0)],
            angularVelocities: [.zero])
        let actions = ContiguousArray(
            repeating: Float(0), count: 2 * task.spec.action.elementCount)
        var contacted = false
        var handLateralVelocityAtContact: Float = 0
        var projectileLateralVelocityAtContact: Float = -6
        for _ in 0..<40 where !contacted {
            task.environment.step(
                normalizedActions: actions, decimation: 1,
                clampActions: false, clampTargetsToLimits: false)
            contacted = task.environment.projectileRobotContacts()[0]
            if contacted {
                handLateralVelocityAtContact = task.environment.solver
                    .bodyVelocity(ref.leftHand).y
                let projectile = try XCTUnwrap(ref.projectile)
                projectileLateralVelocityAtContact = task.environment.solver
                    .bodyVelocity(projectile).y
            }
        }
        XCTAssertTrue(contacted, "projectile must form a hand contact manifold")
        XCTAssertLessThan(handLateralVelocityAtContact, -0.02,
                          "the physical hand must receive projectile momentum")
        XCTAssertGreaterThan(projectileLateralVelocityAtContact, -5.5,
                             "the projectile must lose momentum to the robot")
    }

    func testStandingCommandCohortUsesLowSpeedTaskRevision() throws {
        let registered = try BuiltInRLTasks.registry.make(
            "humanoid-walk-v0",
            configuration: RLTaskConfiguration(
                numEnvironments: 2, seed: 45,
                options: [
                    "standingCommandProbability": 0.35,
                    "standingCommandCurriculumControlSteps": 100,
                ]))
        let task = try XCTUnwrap(registered as? HumanoidWalkTask)
        XCTAssertEqual(task.spec.revision,
                       RLPhysicsContract.fixedGainActuatorV2(42))
        XCTAssertEqual(task.configuration.standingCommandProbability, 0.35)
        XCTAssertEqual(task.trainingStandingCommandProbability, 0.35)
        task.setTrainingMode(true)
        XCTAssertEqual(task.trainingStandingCommandProbability, 0)
        task.setTrainingProgress(environmentSteps: 2 * 50)
        XCTAssertEqual(task.trainingStandingCommandProbability, 0.175,
                       accuracy: 1e-6)
        task.setTrainingMode(false)
        XCTAssertEqual(task.trainingStandingCommandProbability, 0.35)
    }

    func testCommandGatedActorRoutesStandingAndPreservesMovingRevision() throws {
        let registered = try BuiltInRLTasks.registry.make(
            "humanoid-walk-v0",
            configuration: RLTaskConfiguration(
                numEnvironments: 2, seed: 46,
                options: [
                    "standingCommandProbability": 1,
                    "commandGatedActor": 1,
                    "expertGateCommandSpeed": 0.45,
                ]))
        let task = try XCTUnwrap(registered as? HumanoidWalkTask)
        XCTAssertEqual(task.spec.revision,
                       RLPhysicsContract.fixedGainActuatorV2(44))
        XCTAssertTrue(task.usesPolicyExpertGate)
        XCTAssertTrue(task.freezesBasePolicyExpert)
        XCTAssertTrue(task.initializesPolicyExpertFromBaseOnTransfer)
        var observation = try task.reset(seed: 47)
        observation.policy[50] = 0.44
        observation.policy[task.spec.observation.elementCount + 50] = 0.45
        XCTAssertEqual(task.policyExpertGates(observation.policy), [1, 0])
    }

    func testRegisteredHumanoidGoalVariesDirectionAndOwnsPhysicalProjectiles()
        throws {
        let registered = try BuiltInRLTasks.registry.make(
            "humanoid-goal-v0",
            configuration: RLTaskConfiguration(numEnvironments: 16, seed: 47,
                                                options: [
                "maxEpisodeSteps": 8,
                "goalDirectionCurriculumControlSteps": 0,
                "projectileProbability": 1,
                "minimumProjectileLaunchStep": 0,
                "maximumProjectileLaunchStep": 0,
                "minimumProjectileSpeed": 4,
                "maximumProjectileSpeed": 4,
            ]))
        let task = try XCTUnwrap(registered as? HumanoidWalkTask)
        XCTAssertEqual(task.spec.id, "humanoid-goal-v0")
        XCTAssertEqual(task.spec.revision,
                       RLPhysicsContract.fixedGainActuatorV2(10))
        XCTAssertTrue(task.usesPointGoal)
        XCTAssertEqual(task.spec.observation.shape, [630])
        let firstObservation = try task.reset(seed: 49)
        let firstDirections = (0..<16).map {
            task.currentGoalDirection(environment: $0)
        }
        let repeatedObservation = try task.reset(seed: 49)
        let repeatedDirections = (0..<16).map {
            task.currentGoalDirection(environment: $0)
        }
        XCTAssertEqual(firstObservation.policy, repeatedObservation.policy)
        for environment in 0..<16 {
            XCTAssertLessThan(simd_distance(
                firstDirections[environment], repeatedDirections[environment]), 1e-7)
            XCTAssertNotNil(task.environment.refs[environment].projectile)
            let projectile = try XCTUnwrap(
                task.environment.refs[environment].projectile)
            let groups = task.environment.scene.colliders
                .filter { $0.body == projectile }.map(\.collisionGroup)
            XCTAssertEqual(groups, [UInt32(environment + 1)])
        }
        let angles = firstDirections.map { atan2($0.y, $0.x) }
        XCTAssertGreaterThan(angles.max()! - angles.min()!, 2)

        let actions = try RLActionBatch(
            numEnvironments: 16,
            actionDimension: HumanoidWalkEnv.jointRanges.count,
            values: ContiguousArray(repeating: 0,
                count: 16 * HumanoidWalkEnv.jointRanges.count))
        var result = RLStepBatch(spec: task.spec)
        try task.step(actions: actions, into: &result)
        XCTAssertTrue(result.observations.policy.allSatisfy(\.isFinite))
        for ref in task.environment.refs {
            let projectile = try XCTUnwrap(ref.projectile)
            let state = task.environment.solver.bodyStates([projectile])[0]
            XCTAssertGreaterThan(state.position.z, 0.5)
            XCTAssertGreaterThan(simd_length(state.linearVelocity), 1)
        }
    }

    func testHumanoidGoalCanRouteApproachAndArrivalToLowSpeedExpert() throws {
        let registered = try BuiltInRLTasks.registry.make(
            "humanoid-goal-v0",
            configuration: RLTaskConfiguration(
                numEnvironments: 2, seed: 48,
                options: [
                    "commandGatedActor": 1,
                    "expertGateCommandSpeed": 0.45,
                    "projectileProbability": 0,
                ]))
        let task = try XCTUnwrap(registered as? HumanoidWalkTask)
        XCTAssertEqual(task.spec.revision,
                       RLPhysicsContract.fixedGainActuatorV2(11))
        XCTAssertTrue(task.usesPolicyExpertGate)
        XCTAssertFalse(task.freezesBasePolicyExpert)
        var observation = try task.reset(seed: 49)
        observation.policy[50] = 0.44
        observation.policy[task.spec.observation.elementCount + 50] = 0.45
        XCTAssertEqual(task.policyExpertGates(observation.policy), [1, 0])
    }

    func testHumanoidGoalCanFreezeVerifiedCruiseExpertForArrivalStage() throws {
        let registered = try BuiltInRLTasks.registry.make(
            "humanoid-goal-v0",
            configuration: RLTaskConfiguration(
                numEnvironments: 1, seed: 50,
                options: [
                    "commandGatedActor": 1,
                    "freezeBasePolicyExpert": 1,
                    "standStillFallPenalty": 20,
                    "projectileProbability": 0,
                ]))
        let task = try XCTUnwrap(registered as? HumanoidWalkTask)
        XCTAssertEqual(task.spec.revision,
                       RLPhysicsContract.fixedGainActuatorV2(14))
        XCTAssertTrue(task.usesPolicyExpertGate)
        XCTAssertTrue(task.freezesBasePolicyExpert)
        XCTAssertEqual(task.spec.configurationValues["freezeBasePolicyExpert"], 1)
    }

    func testHumanoidMixedCommandsCanTrainMovingAndStandingActorsSeparately() throws {
        let registered = try BuiltInRLTasks.registry.make(
            "humanoid-walk-v0",
            configuration: RLTaskConfiguration(
                numEnvironments: 2, seed: 50,
                options: [
                    "commandGatedActor": 1,
                    "expertGateCommandSpeed": 0.05,
                    "trainBasePolicyExpert": 1,
                    "standingCommandProbability": 0.2,
                ]))
        let task = try XCTUnwrap(registered as? HumanoidWalkTask)
        XCTAssertTrue(task.usesPolicyExpertGate)
        XCTAssertFalse(task.freezesBasePolicyExpert)
        XCTAssertEqual(task.spec.configurationValues["trainBasePolicyExpert"], 1)
        var observation = try task.reset(seed: 51)
        let dimension = task.spec.observation.elementCount
        observation.policy[50] = 0
        observation.policy[dimension + 50] = 0.2
        XCTAssertEqual(task.policyExpertGates(observation.policy), [1, 0])
    }

    func testHumanoidThreeModeActorRoutesCruiseCreepAndStandExclusively() throws {
        let registered = try BuiltInRLTasks.registry.make(
            "humanoid-walk-v0",
            configuration: RLTaskConfiguration(
                numEnvironments: 3, seed: 52,
                options: [
                    "standingCommandProbability": 0.25,
                    "commandGatedActor": 1,
                    "threeModeActor": 1,
                    "expertGateCommandSpeed": 0.4,
                ]))
        let task = try XCTUnwrap(registered as? HumanoidWalkTask)
        XCTAssertEqual(task.spec.revision,
                       RLPhysicsContract.fixedGainActuatorV2(444))
        XCTAssertTrue(task.usesPolicyExpertGate)
        XCTAssertTrue(task.usesPolicyStandExpertGate)
        XCTAssertTrue(task.freezesBasePolicyExpert)
        XCTAssertEqual(task.spec.configurationValues["threeModeActor"], 1)
        var observation = try task.reset(seed: 53)
        let dimension = task.spec.observation.elementCount
        for (row, command) in [Float(0), 0.3, 0.5].enumerated() {
            observation.policy[row * dimension + 50] = command
        }
        XCTAssertEqual(task.policyExpertGates(observation.policy), [0, 1, 0])
        XCTAssertEqual(
            task.policyStandExpertGates(observation.policy), [1, 0, 0])
        let blended = try HumanoidWalkTask(configuration: .init(
            numEnvironments: 3, commandGatedActor: true,
            threeModeActor: true, expertGateCommandSpeed: 0.4,
            expertGateBlendWidth: 0.2))
        var blendedObservation = try blended.reset(seed: 54)
        let blendedDimension = blended.spec.observation.elementCount
        for (row, command) in [Float(0), 0.3, 0.5].enumerated() {
            blendedObservation.policy[row * blendedDimension + 50] = command
        }
        let blendedGates = blended.policyExpertGates(blendedObservation.policy)
        XCTAssertEqual(blendedGates[0], 0, accuracy: 1e-6)
        XCTAssertEqual(blendedGates[1], 0.5, accuracy: 1e-6)
        XCTAssertEqual(blendedGates[2], 0, accuracy: 1e-6)
        XCTAssertEqual(
            blended.policyStandExpertGates(blendedObservation.policy), [1, 0, 0])
    }

    func testPointGoalBlendsBrakingIntoStandingByMeasuredSpeed() throws {
        let task = try HumanoidWalkTask(configuration: .init(
            numEnvironments: 4, commandGatedActor: true,
            threeModeActor: true, expertGateCommandSpeed: 0.4,
            expertGateBlendWidth: 0.2,
            standExpertBlendStartSpeed: 0.25,
            standExpertBlendWidth: 0.15), taskID: "humanoid-goal-v0")
        var observation = try task.reset(seed: 55)
        let dimension = task.spec.observation.elementCount
        for (row, speed) in [Float(0.30), 0.175, 0.10, -0.30].enumerated() {
            observation.policy[row * dimension + 1] = speed
            observation.policy[row * dimension + 50] = 0
        }
        let braking = task.policyExpertGates(observation.policy)
        let standing = task.policyStandExpertGates(observation.policy)
        XCTAssertEqual(braking[0], 1, accuracy: 1e-6)
        XCTAssertEqual(standing[0], 0, accuracy: 1e-6)
        XCTAssertEqual(braking[1], 0.5, accuracy: 1e-6)
        XCTAssertEqual(standing[1], 0.5, accuracy: 1e-6)
        XCTAssertEqual(braking[2], 0, accuracy: 1e-6)
        XCTAssertEqual(standing[2], 1, accuracy: 1e-6)
        XCTAssertEqual(braking[3], 1, accuracy: 1e-6)
        XCTAssertEqual(standing[3], 0, accuracy: 1e-6)
        for row in 0..<4 {
            XCTAssertEqual(braking[row] + standing[row], 1, accuracy: 1e-6)
        }

        XCTAssertThrowsError(try HumanoidWalkTask(configuration: .init(
            numEnvironments: 1, commandGatedActor: true,
            threeModeActor: true, standExpertBlendStartSpeed: 0.25,
            standExpertBlendWidth: 0.15)))
        XCTAssertThrowsError(try HumanoidWalkTask(configuration: .init(
            numEnvironments: 1, commandGatedActor: true,
            threeModeActor: true, standExpertBlendStartSpeed: 0.10,
            standExpertBlendWidth: 0.15), taskID: "humanoid-goal-v0"))

        XCTAssertEqual(HumanoidWalkTask.standExpertBlendWeight(
            measuredSpeed: 0.10, start: 0.25, width: 0.15,
            requiresDoubleSupport: true, bothFeetInContact: false), 0)
        XCTAssertEqual(HumanoidWalkTask.standExpertBlendWeight(
            measuredSpeed: 0.10, start: 0.25, width: 0.15,
            requiresDoubleSupport: true, bothFeetInContact: true), 1,
            accuracy: 1e-6)
        XCTAssertEqual(HumanoidWalkTask.standExpertMeasuredSpeed(
            projectedSpeed: 0.05,
            measuredRootVelocity: F3(0.05, 0.30, 0),
            usesPlanarSpeed: false), 0.05, accuracy: 1e-6)
        XCTAssertEqual(HumanoidWalkTask.standExpertMeasuredSpeed(
            projectedSpeed: 0.05,
            measuredRootVelocity: F3(0.05, 0.30, 0),
            usesPlanarSpeed: true), sqrt(0.05 * 0.05 + 0.30 * 0.30),
            accuracy: 1e-6)
    }

    func testLegacyExpertMigrationPreservesSharedGoalBrakingPolicy() {
        XCTAssertTrue(VectorActorCritic.legacyExpertIsStandOnly(
            taskConfiguration: ["standingCommandProbability": 1]))
        XCTAssertTrue(VectorActorCritic.legacyExpertIsStandOnly(
            taskConfiguration: ["trainBasePolicyExpert": 1]))
        XCTAssertFalse(VectorActorCritic.legacyExpertIsStandOnly(
            taskConfiguration: ["freezeBasePolicyExpert": 1]))
    }

    func testStandExpertCompositionReparameterizesItsInputNormalizer() throws {
        let sourceMean = [Double](repeating: 0, count: 2)
        let sourceVariance = [Double](repeating: 1, count: 2)
        let destinationMean = [Double](arrayLiteral: 2, -3)
        let destinationVariance = [Double](arrayLiteral: 4, 9)
        var destination: [String: MLXArray] = [
            "standActor1.weight": MLXArray.zeros([1, 2]),
            "standActor1.bias": MLXArray.zeros([1]),
        ]
        var source: [String: MLXArray] = [
            "standActor1.weight": MLXArray([Float(3), 5]).reshaped([1, 2]),
            "standActor1.bias": MLXArray([Float(7)]),
        ]
        for layer in ["standActor2", "standActor3", "standActorOutput"] {
            destination["\(layer).weight"] = MLXArray.zeros([1, 1])
            destination["\(layer).bias"] = MLXArray.zeros([1])
            source["\(layer).weight"] = MLXArray.ones([1, 1])
            source["\(layer).bias"] = MLXArray.ones([1])
        }
        let composed = try VectorActorCritic.initializingStandExpert(
            destination, from: source,
            sourceNormalizer: .init(count: 10, mean: sourceMean,
                variance: sourceVariance),
            destinationNormalizer: .init(count: 20,
                mean: destinationMean, variance: destinationVariance))
        XCTAssertEqual(composed["standActor1.weight"]!
            .asArray(Float.self), [6, 15])
        // 7 + 3*2 + 5*(-3) = -2.
        XCTAssertEqual(composed["standActor1.bias"]!
            .asArray(Float.self), [-2])
        XCTAssertEqual(composed["standActor2.weight"]!
            .asArray(Float.self), [1])

        var configuration = VectorPPOConfig()
        configuration.minibatchSize = 24
        configuration.standExpertInitializationCheckpoint = "/stand"
        XCTAssertThrowsError(try configuration.validate(batchSize: 24))
        configuration.initializationCheckpoint = "/base"
        XCTAssertNoThrow(try configuration.validate(batchSize: 24))
    }

    func testPolicyExpertCompositionUsesSourceBaseAndReparameterizesInput()
        throws {
        let sourceMean = [Double](arrayLiteral: 0, 0)
        let sourceVariance = [Double](arrayLiteral: 1, 1)
        let destinationMean = [Double](arrayLiteral: 2, -3)
        let destinationVariance = [Double](arrayLiteral: 4, 9)
        let destination: [String: MLXArray] = [
            "expertActor1.weight": MLXArray.zeros([1, 2]),
            "expertActor1.bias": MLXArray.zeros([1]),
        ]
        var source: [String: MLXArray] = [
            "actor1.weight": MLXArray([Float(3), 5]).reshaped([1, 2]),
            "actor1.bias": MLXArray([Float(7)]),
        ]
        for layer in ["actor2", "actor3", "actorOutput"] {
            source["\(layer).weight"] = MLXArray.ones([1, 1])
            source["\(layer).bias"] = MLXArray.ones([1])
        }
        let composed = try VectorActorCritic.initializingPolicyExpert(
            destination, from: source,
            sourceNormalizer: .init(count: 10, mean: sourceMean,
                variance: sourceVariance),
            destinationNormalizer: .init(count: 20,
                mean: destinationMean, variance: destinationVariance))
        XCTAssertEqual(composed["expertActor1.weight"]!
            .asArray(Float.self), [6, 15])
        XCTAssertEqual(composed["expertActor1.bias"]!
            .asArray(Float.self), [-2])
        XCTAssertEqual(composed["expertActor2.weight"]!
            .asArray(Float.self), [1])

        var configuration = VectorPPOConfig()
        configuration.minibatchSize = 24
        configuration.policyExpertInitializationCheckpoint = "/precision"
        XCTAssertThrowsError(try configuration.validate(batchSize: 24))
        configuration.initializationCheckpoint = "/base"
        XCTAssertNoThrow(try configuration.validate(batchSize: 24))
    }

    func testObservationSchemaTransferPreservesOldPolicyAndAddsZeroColumns()
        throws {
        let sourceValues: [Float] = [1, 2, 3, 4, 5, 6]
        var weights = [String: MLXArray]()
        for name in ["actor1.weight", "expertActor1.weight",
                     "standActor1.weight", "critic1.weight"] {
            weights[name] = MLXArray(sourceValues).reshaped([2, 3])
        }
        weights["actor1.bias"] = MLXArray([Float(9), 10])
        let remapped = try VectorActorCritic.remappingObservationInputs(
            weights, sourceIndices: [0, nil, 2, 1])
        let expected: [Float] = [1, 0, 3, 2, 4, 0, 6, 5]
        for name in ["actor1.weight", "expertActor1.weight",
                     "standActor1.weight", "critic1.weight"] {
            XCTAssertEqual(remapped[name]?.shape, [2, 4])
            XCTAssertEqual(remapped[name]?.asArray(Float.self), expected)
        }
        XCTAssertEqual(remapped["actor1.bias"]?.asArray(Float.self), [9, 10])
        XCTAssertThrowsError(try VectorActorCritic.remappingObservationInputs(
            weights, sourceIndices: [3]))
    }

    func testThreeModeTaskCanFreezeVerifiedLowSpeedExpert() throws {
        let registered = try BuiltInRLTasks.registry.make(
            "humanoid-goal-v0",
            configuration: RLTaskConfiguration(
                numEnvironments: 1, seed: 55,
                options: [
                    "commandGatedActor": 1,
                    "threeModeActor": 1,
                    "freezeBasePolicyExpert": 1,
                    "freezeLowSpeedPolicyExpert": 1,
                    "projectileProbability": 0,
                ]))
        let task = try XCTUnwrap(registered as? HumanoidWalkTask)
        XCTAssertTrue(task.freezesBasePolicyExpert)
        XCTAssertTrue(task.freezesLowSpeedPolicyExpert)
        XCTAssertEqual(
            task.spec.configurationValues["freezeLowSpeedPolicyExpert"], 1)
        XCTAssertThrowsError(try HumanoidWalkTask(configuration: .init(
            numEnvironments: 1, commandGatedActor: true,
            freezeLowSpeedPolicyExpert: true)))
    }

    func testPointGoalCanSampleReachableMetricRouteRange() throws {
        let registered = try BuiltInRLTasks.registry.make(
            "humanoid-goal-v0",
            configuration: RLTaskConfiguration(
                numEnvironments: 32, seed: 56,
                options: [
                    "minimumGoalDistanceMeters": 7,
                    "maximumGoalDistanceMeters": 9,
                    "projectileProbability": 0,
                ]))
        let task = try XCTUnwrap(registered as? HumanoidWalkTask)
        _ = try task.reset(seed: 57)
        XCTAssertEqual(task.spec.configurationValues[
            "minimumGoalDistanceMeters"], 7)
        XCTAssertEqual(task.spec.configurationValues[
            "maximumGoalDistanceMeters"], 9)
        XCTAssertEqual(task.evaluationCriteria.minimumTaskMetrics[
            "episode/alternating_steps"], 10.5)
        let states = task.environment.states()
        for e in 0..<task.spec.numEnvironments {
            let offset = task.currentGoalPosition(environment: e)
                - states[e].root.position
            let distance = simd_length(F3(offset.x, offset.y, 0))
            XCTAssertGreaterThanOrEqual(distance, 7 - 1e-4)
            XCTAssertLessThanOrEqual(distance, 9 + 1e-4)
        }
        XCTAssertThrowsError(try HumanoidWalkTask(
            configuration: .init(
                numEnvironments: 1, goalRadius: 1.5,
                minimumGoalDistanceMeters: 1,
                maximumGoalDistanceMeters: 9),
            taskID: "humanoid-goal-v0"))
    }

    func testHumanoidGoalCanBlendCruiseAndArrivalExpertsSmoothly() throws {
        let registered = try BuiltInRLTasks.registry.make(
            "humanoid-goal-v0",
            configuration: RLTaskConfiguration(
                numEnvironments: 4, seed: 51,
                options: [
                    "commandGatedActor": 1,
                    "expertGateCommandSpeed": 0.4,
                    "expertGateBlendWidth": 0.2,
                    "freezeBasePolicyExpert": 1,
                    "standStillFallPenalty": 20,
                    "projectileProbability": 0,
                ]))
        let task = try XCTUnwrap(registered as? HumanoidWalkTask)
        XCTAssertEqual(task.spec.revision,
                       RLPhysicsContract.fixedGainActuatorV2(18))
        XCTAssertEqual(task.spec.configurationValues["expertGateBlendWidth"], 0.2)
        var observation = try task.reset(seed: 53)
        let dimension = task.spec.observation.elementCount
        for (row, command) in [Float(0.4), 0.3, 0.2, 0.1].enumerated() {
            observation.policy[row * dimension + 50] = command
        }
        let gates = task.policyExpertGates(observation.policy)
        XCTAssertEqual(gates[0], 0, accuracy: 1e-6)
        XCTAssertEqual(gates[1], 0.5, accuracy: 1e-6)
        XCTAssertEqual(gates[2], 1, accuracy: 1e-6)
        XCTAssertEqual(gates[3], 1, accuracy: 1e-6)
        XCTAssertThrowsError(try HumanoidWalkTask(configuration: .init(
            numEnvironments: 1, commandGatedActor: true,
            expertGateCommandSpeed: 0.2, expertGateBlendWidth: 0.3)))
    }

    func testHumanoidTrainingCanDesynchronizeInitialTimeoutsWithoutChangingEvaluation() throws {
        let task = try HumanoidWalkTask(configuration: .init(
            numEnvironments: 128, seed: 52, maxEpisodeSteps: 20,
            controlDecimation: 1, trainingInitialEpisodeAgeFraction: 1))
        XCTAssertEqual(task.spec.configurationValues[
            "trainingInitialEpisodeAgeFraction"], 1)

        task.setTrainingMode(true)
        var trainingObservation = try task.reset(seed: 53)
        let actions = try RLActionBatch(
            numEnvironments: 128,
            actionDimension: HumanoidWalkEnv.jointRanges.count,
            values: ContiguousArray(repeating: 0,
                count: 128 * HumanoidWalkEnv.jointRanges.count))
        var trainingStep = RLStepBatch(spec: task.spec)
        try task.step(actions: actions, into: &trainingStep)
        XCTAssertTrue(trainingStep.truncated.contains(true))
        XCTAssertFalse(trainingStep.truncated.allSatisfy { $0 })

        task.setTrainingMode(false)
        trainingObservation = try task.reset(seed: 53)
        XCTAssertTrue(trainingObservation.policy.allSatisfy(\.isFinite))
        var evaluationStep = RLStepBatch(spec: task.spec)
        try task.step(actions: actions, into: &evaluationStep)
        XCTAssertFalse(evaluationStep.truncated.contains(true))
    }

    func testHumanoidGoalCurriculumCanExpandFromAcceptedSteeringRange() throws {
        let task = try HumanoidWalkTask(configuration: .init(
            numEnvironments: 64, seed: 53, maxEpisodeSteps: 8,
            controlDecimation: 1, maximumGoalDirectionAngle: .pi,
            initialGoalDirectionAngle: .pi / 2,
            goalDirectionCurriculumControlSteps: 1_000,
            initialGoalDistanceScale: 0.25,
            goalDistanceCurriculumControlSteps: 1_000),
            taskID: "humanoid-goal-v0", taskRevision: 4)
        task.setTrainingMode(true)
        _ = try task.reset(seed: 59)
        XCTAssertEqual(task.trainingGoalDirectionRange, .pi / 2,
                       accuracy: 1e-6)
        XCTAssertEqual(task.trainingGoalDistanceScale, 0.25, accuracy: 1e-6)
        task.setTrainingProgress(environmentSteps: 64 * 500)
        XCTAssertEqual(task.trainingGoalDirectionRange, 3 * .pi / 4,
                       accuracy: 1e-6)
        XCTAssertEqual(task.trainingGoalDistanceScale, 0.625, accuracy: 1e-6)
        task.setTrainingMode(false)
        XCTAssertEqual(task.trainingGoalDirectionRange, .pi, accuracy: 1e-6)
        XCTAssertEqual(task.trainingGoalDistanceScale, 1, accuracy: 1e-6)
        XCTAssertThrowsError(try HumanoidWalkTask(configuration: .init(
            numEnvironments: 1, maximumGoalDirectionAngle: .pi / 2,
            initialGoalDirectionAngle: .pi),
            taskID: "humanoid-goal-v0", taskRevision: 4))
    }

    func testHumanoidGoalOverrideIsNormalizedAndAppliedThroughReset() throws {
        let task = try HumanoidWalkTask(configuration: .init(
            numEnvironments: 1, seed: 61, maxEpisodeSteps: 16,
            controlDecimation: 1, maximumGoalDirectionAngle: .pi,
            projectileProbability: 1, minimumProjectileLaunchStep: 0,
            maximumProjectileLaunchStep: 15),
            taskID: "humanoid-goal-v0", taskRevision: 10)
        try task.setGoalOverride(
            environment: 0, direction: F3(0, 3, 7), distance: 5)
        _ = try task.reset(seed: 67)
        XCTAssertLessThan(simd_distance(
            task.currentGoalDirection(environment: 0), F3(0, 1, 0)), 1e-6)
        let root = task.environment.states()[0].root.position
        XCTAssertLessThan(simd_distance(
            task.currentGoalPosition(environment: 0), root + F3(0, 5, 0)), 1e-5)
        XCTAssertTrue(task.hasProjectile(environment: 0))

        task.clearGoalOverride(environment: 0)
        _ = try task.reset(seed: 67)
        XCTAssertGreaterThan(abs(task.currentGoalDirection(environment: 0).x), 1e-3)
        XCTAssertThrowsError(try HumanoidWalkTask(configuration: .init(
            numEnvironments: 1), taskID: "humanoid-walk-v0")
            .setGoalOverride(environment: 0, direction: F3(1, 0, 0), distance: 5))
    }

    func testHumanoidResetSeedProducesDeterministicJointConsistentPoseDiversity()
        throws {
        let task = try HumanoidWalkTask(configuration: .init(
            numEnvironments: 4, seed: 101, maxEpisodeSteps: 8,
            controlDecimation: 1))
        _ = try task.reset(seed: 103)
        let first = task.environment.states()
        _ = try task.reset(seed: 103)
        let repeated = task.environment.states()
        for environment in 0..<4 {
            XCTAssertLessThan(simd_distance(
                first[environment].root.position,
                repeated[environment].root.position), 1e-6)
            XCTAssertLessThan(simd_distance(
                first[environment].root.rotation.imag,
                repeated[environment].root.rotation.imag), 1e-6)
            XCTAssertEqual(first[environment].root.rotation.real,
                           repeated[environment].root.rotation.real,
                           accuracy: 1e-6)
            XCTAssertTrue(first[environment].jointAngles.allSatisfy {
                abs($0) < 1e-3
            }, "whole-body reset perturbation must preserve joint angles")
            let up = first[environment].root.rotation.act(F3(0, 0, 1))
            XCTAssertGreaterThan(up.z, 0.9995)
        }
        let headings = first.map {
            $0.root.rotation.act(F3(1, 0, 0)).y
        }
        XCTAssertGreaterThan(headings.max()! - headings.min()!, 1e-3,
                             "distinct reset seeds must create pose diversity")
    }

    func testHumanoidBatchedReplicasRemainTranslationInvariant() throws {
        let environment = try HumanoidWalkEnv(numEnvironments: 512, seed: 107)
        let ids = Array(0..<512)
        environment.reset(ids, seeds: [UInt64](repeating: 109, count: 512))
        let actionCount = 512 * HumanoidWalkEnv.jointRanges.count
        let actions = ContiguousArray<Float>(repeating: 0, count: actionCount)
        for _ in 0..<24 {
            environment.step(normalizedActions: actions, decimation: 4)
        }
        let states = environment.states()
        let reference = states[0]
        let referenceCenter = environment.refs[0].center
        for replica in [1, 127, 255, 511] {
            let state = states[replica]
            let relative = state.root.position - environment.refs[replica].center
            let referenceRelative = reference.root.position - referenceCenter
            XCTAssertLessThan(simd_distance(relative, referenceRelative), 2e-4,
                              "replica \(replica) root translation diverged")
            XCTAssertLessThan(simd_distance(state.root.rotation.imag,
                                             reference.root.rotation.imag), 2e-4)
            XCTAssertEqual(state.root.rotation.real, reference.root.rotation.real,
                           accuracy: 2e-4)
            for joint in state.jointAngles.indices {
                XCTAssertEqual(state.jointAngles[joint],
                               reference.jointAngles[joint], accuracy: 2e-4,
                               "replica \(replica) joint \(joint) diverged")
            }
        }
    }

    func testHumanoidPolicySymmetryIsAnExactInvolution() throws {
        let task = try HumanoidWalkTask(configuration: HumanoidWalkTaskConfig(
            numEnvironments: 2, seed: 42))
        let action = ContiguousArray((0..<38).map { Float($0) - 11.5 })
        let mirroredAction = task.mirrorPolicyActions(action)
        XCTAssertEqual(task.mirrorPolicyActions(mirroredAction), action)
        for row in 0..<2 {
            let base = row * task.spec.action.elementCount
            for joint in 0..<task.spec.action.elementCount {
                XCTAssertEqual(
                    mirroredAction[base + joint],
                    task.policyActionMirrorSigns[joint]
                        * action[base + task.policyActionMirrorSourceIndices[joint]])
            }
        }
        XCTAssertEqual(mirroredAction[0], -action[5])
        XCTAssertEqual(mirroredAction[2], action[7])
        XCTAssertEqual(mirroredAction[10], -action[10])
        XCTAssertEqual(mirroredAction[11], action[15])
        XCTAssertEqual(mirroredAction[12], -action[16])

        let count = 2 * task.spec.observation.elementCount
        let observation = ContiguousArray((0..<count).map { Float($0) + 0.25 })
        let mirroredObservation = task.mirrorPolicyObservations(observation)
        XCTAssertEqual(task.mirrorPolicyObservations(mirroredObservation), observation)
        XCTAssertEqual(mirroredObservation[2], -observation[2])
        XCTAssertEqual(mirroredObservation[5], -observation[5])
        XCTAssertEqual(mirroredObservation[9], -observation[9])
        XCTAssertEqual(mirroredObservation[10], observation[10])
        XCTAssertEqual(mirroredObservation[12], -observation[17])
        XCTAssertEqual(mirroredObservation[14], observation[19])
        XCTAssertEqual(mirroredObservation[31], -observation[36])
        XCTAssertEqual(mirroredObservation[51], -observation[56])
    }

    func testPointGoalLateralVelocityObservationIsVersionedAndMirrored()
        throws {
        XCTAssertEqual(HumanoidWalkTask.pointGoalAuxiliaryObservation(
            proximity: 0.75, lateralVelocity: -0.18,
            usesLateralVelocity: false), 0.75)
        XCTAssertEqual(HumanoidWalkTask.pointGoalAuxiliaryObservation(
            proximity: 0.75, lateralVelocity: -0.18,
            usesLateralVelocity: true), -0.18)

        let legacy = try BuiltInRLTasks.registry.make(
            "humanoid-goal-v0",
            configuration: RLTaskConfiguration(
                numEnvironments: 1, seed: 43,
                options: ["projectileProbability": 0]))
        let revised = try BuiltInRLTasks.registry.make(
            "humanoid-goal-v0",
            configuration: RLTaskConfiguration(
                numEnvironments: 1, seed: 43,
                options: [
                    "projectileProbability": 0,
                    "goalObservationUsesLateralVelocity": 1,
                ]))
        XCTAssertEqual(revised.spec.revision - legacy.spec.revision, 102_400)
        let task = try XCTUnwrap(revised as? HumanoidWalkTask)
        XCTAssertTrue(task.configuration.goalObservationUsesLateralVelocity)
        XCTAssertEqual(task.spec.configurationValues[
            "goalObservationUsesLateralVelocity"], 1)

        var observation = ContiguousArray(
            repeating: Float(0), count: task.spec.observation.elementCount)
        observation[2] = 0.18
        let mirrored = task.mirrorPolicyObservations(observation)
        XCTAssertEqual(mirrored[2], -0.18)
        XCTAssertEqual(task.mirrorPolicyObservations(mirrored), observation)

        let appended = try BuiltInRLTasks.registry.make(
            "humanoid-goal-v0",
            configuration: RLTaskConfiguration(
                numEnvironments: 1, seed: 43,
                options: [
                    "projectileProbability": 0,
                    "goalObservationIncludesLateralVelocity": 1,
                ]))
        XCTAssertEqual(appended.spec.revision - legacy.spec.revision, 204_800)
        let appendedTask = try XCTUnwrap(appended as? HumanoidWalkTask)
        XCTAssertEqual(appended.spec.observation.elementCount,
                       legacy.spec.observation.elementCount + 9)
        let sourceIndices = try XCTUnwrap(appendedTask
            .initializationObservationSourceIndices(
                sourceDimension: legacy.spec.observation.elementCount))
        XCTAssertEqual(sourceIndices.count, appended.spec.observation.elementCount)
        XCTAssertEqual(sourceIndices[0], 0)
        XCTAssertEqual(sourceIndices[629], 629)
        XCTAssertNil(sourceIndices[630])
        XCTAssertNil(sourceIndices[638])
        var appendedObservation = ContiguousArray(
            repeating: Float(0),
            count: appended.spec.observation.elementCount)
        appendedObservation[2] = 0.75
        appendedObservation[630] = 0.18
        appendedObservation[638] = -0.12
        let appendedMirror = appendedTask.mirrorPolicyObservations(
            appendedObservation)
        XCTAssertEqual(appendedMirror[2], 0.75)
        XCTAssertEqual(appendedMirror[630], -0.18)
        XCTAssertEqual(appendedMirror[638], 0.12)
        XCTAssertEqual(appendedTask.mirrorPolicyObservations(appendedMirror),
                       appendedObservation)

        XCTAssertThrowsError(try HumanoidWalkTask(configuration: .init(
            numEnvironments: 1,
            goalObservationIncludesLateralVelocity: true)))
        XCTAssertThrowsError(try HumanoidWalkTask(configuration: .init(
            numEnvironments: 1,
            goalObservationUsesLateralVelocity: true,
            goalObservationIncludesLateralVelocity: true),
            taskID: "humanoid-goal-v0"))
    }

    func testHumanoidHomePoseStartsAboveUprightTerminationHeight() throws {
        let environment = try HumanoidWalkEnv(numEnvironments: 4, seed: 45)
        let initialStates = environment.states()
        let heights = initialStates.map { $0.root.position.z }
        XCTAssertTrue(heights.allSatisfy {
            $0 > HumanoidLocomotionObjective.minimumPelvisHeight + 0.10
                && $0 < 1.15
        }, "initial H1 pelvis heights \(heights), joint angles "
            + "\(initialStates.first?.jointAngles ?? []) do not leave a safe "
            + "termination margin")
        for state in initialStates {
            XCTAssertTrue(state.jointAngles.allSatisfy { abs($0) < 1e-4 },
                          "rebased H1 home pose must read as zero policy offsets")
        }
    }

    func testHumanoidIsaacHomePosePreservesAuthoredContactTransient() throws {
        let environment = try HumanoidWalkEnv(numEnvironments: 2, seed: 46)
        for state in environment.states() {
            let left = HumanoidIsaacVelocityTask.footHullGroundClearance(
                state.leftFoot, vertices: IsaacH1CollisionHulls.leftAnkle)
            let right = HumanoidIsaacVelocityTask.footHullGroundClearance(
                state.rightFoot, vertices: IsaacH1CollisionHulls.rightAnkle)
            // Isaac authors the crouched pose at a 1.05 m root height; it
            // does not lower the free root until a foot just touches. Keep
            // that initial contact transient explicit instead of silently
            // changing the dynamics under transferred/trained checkpoints.
            XCTAssertTrue((0.04...0.10).contains(left),
                          "left authored foot clearance is \(left) m")
            XCTAssertTrue((0.04...0.10).contains(right),
                          "right authored foot clearance is \(right) m")
        }
    }

    func testHumanoidCalfProtectionSpheresRemainPhysicalButAreNotRendered()
        throws {
        let task = try HumanoidWalkTask(configuration: .init(
            numEnvironments: 1, seed: 7, maxEpisodeSteps: 8,
            controlDecimation: 1))
        let hidden = task.environment.scene.colliders.filter { !$0.isRendered }
        let calfSpheres = hidden.filter {
            $0.shape == .sphere && $0.convexHullVertices.isEmpty
        }
        XCTAssertEqual(calfSpheres.count, 2)
        XCTAssertTrue(calfSpheres.allSatisfy { $0.size.x == 0.1 })
        let physicalUSDHulls = hidden.filter {
            !$0.convexHullVertices.isEmpty
        }
        XCTAssertEqual(physicalUSDHulls.count, 3,
                       "two ankle hulls and one torso hull stay physical")
        XCTAssertEqual(hidden.count, 5)
        XCTAssertEqual(
            task.environment.solver.renderRigidBodyCount,
            task.environment.scene.colliders.filter(\.isRendered).count)
    }

    func testHumanoidTrainingCurriculumDoesNotLeakIntoEvaluation() throws {
        let task = try HumanoidWalkTask(configuration: HumanoidWalkTaskConfig(
            numEnvironments: 4, seed: 47,
            minimumCommandSpeed: 0.45, maximumCommandSpeed: 0.65,
            commandCurriculumControlSteps: 20_000,
            lateralPenaltyWarmupControlSteps: 10_000,
            lateralPenaltyRampControlSteps: 10_000))
        task.setTrainingMode(true)
        _ = try task.reset(seed: 49)
        XCTAssertEqual(task.trainingCurriculumProgress, 0)
        XCTAssertEqual(task.trainingLateralPenaltyScale, 0)
        for environment in 0..<4 {
            XCTAssertTrue((0.20...0.35).contains(
                task.currentCommandSpeed(environment: environment)))
        }
        task.setTrainingProgress(environmentSteps: 4 * 10_000)
        _ = try task.reset(seed: 51)
        XCTAssertEqual(task.trainingCurriculumProgress, 0.5, accuracy: 1e-6)
        XCTAssertEqual(task.trainingLateralPenaltyScale, 0, accuracy: 1e-6)
        for environment in 0..<4 {
            XCTAssertTrue((0.325...0.50).contains(
                task.currentCommandSpeed(environment: environment)))
        }
        task.setTrainingProgress(environmentSteps: 4 * 15_000)
        XCTAssertEqual(task.trainingLateralPenaltyScale, 0.5, accuracy: 1e-6)
        task.setTrainingMode(false)
        _ = try task.reset(seed: 53)
        XCTAssertEqual(task.trainingCurriculumProgress, 1)
        XCTAssertEqual(task.trainingLateralPenaltyScale, 1)
        for environment in 0..<4 {
            XCTAssertTrue((0.45...0.65).contains(
                task.currentCommandSpeed(environment: environment)))
        }
    }

    func testHumanoidStandingCommandCohortIsSeededAndExplicit() throws {
        let task = try HumanoidWalkTask(configuration: HumanoidWalkTaskConfig(
            numEnvironments: 8, seed: 59, minimumCommandSpeed: 0.45,
            maximumCommandSpeed: 0.65, standingCommandProbability: 1,
            standStillVelocityPenaltyWeight: 10,
            standStillJointDeviationPenaltyWeight: 2,
            standStillDoubleSupportRewardWeight: 3))
        _ = try task.reset(seed: 61)
        XCTAssertEqual(task.spec.configurationValues[
            "standingCommandProbability"], 1)
        XCTAssertEqual(task.spec.configurationValues[
            "standStillVelocityPenaltyWeight"], 10)
        XCTAssertEqual(task.spec.configurationValues[
            "standStillJointDeviationPenaltyWeight"], 2)
        XCTAssertEqual(task.spec.configurationValues[
            "standStillDoubleSupportRewardWeight"], 3)
        for environment in 0..<8 {
            XCTAssertEqual(task.currentCommandSpeed(environment: environment), 0)
        }
    }

    func testPushTTaskStepAndAutoResetContract() throws {
        let task = try PushTTask(configuration: PushTTaskConfig(
            numEnvironments: 4, seed: 9, maxEpisodeSteps: 2,
            controlDecimation: 1, autoReset: true))
        var observation = try task.reset(seed: 11)
        XCTAssertEqual(observation.policy.count, 4 * 12)
        XCTAssertTrue(observation.policy.allSatisfy(\.isFinite))

        let action = RLActionBatch(spec: task.spec)
        var result = RLStepBatch(spec: task.spec)
        try task.step(actions: action, into: &result)
        XCTAssertFalse(result.truncated.contains(true))
        try task.step(actions: action, into: &result)
        XCTAssertTrue(result.truncated.allSatisfy { $0 })
        XCTAssertTrue(result.hasFinalObservation.allSatisfy { $0 })
        XCTAssertTrue(result.observations.policy.allSatisfy(\.isFinite))
        XCTAssertEqual(result.metrics["episode/return"]?.count, 4)
        observation = result.observations
        XCTAssertEqual(observation.policy.count, 48)
    }

    func testHumanoidTaskProducesFiniteBatchedTransitions() throws {
        let task = try HumanoidWalkTask(configuration: HumanoidWalkTaskConfig(
            numEnvironments: 2, seed: 3, maxEpisodeSteps: 8,
            controlDecimation: 1))
        let initial = try task.reset(seed: 5)
        XCTAssertEqual(initial.policy.count, 2 * task.spec.observation.elementCount)
        XCTAssertEqual(task.spec.observation.shape, [630])
        XCTAssertGreaterThan(initial.policy[7], 0.998)
        XCTAssertLessThan(abs(initial.policy[8]), 0.06)
        let action = RLActionBatch(spec: task.spec)
        var result = RLStepBatch(spec: task.spec)
        for _ in 0..<4 {
            try task.step(actions: action, into: &result)
            XCTAssertTrue(result.observations.policy.allSatisfy(\.isFinite))
            XCTAssertTrue(result.rewards.allSatisfy(\.isFinite))
        }
        XCTAssertEqual(result.metrics["reward/velocity_tracking"]?.count, 2)
        XCTAssertEqual(task.environment.refs.count, 2)
        XCTAssertEqual(task.environment.refs[0].motors.count,
                       task.spec.action.elementCount)
    }

    func testHumanoidDirectTargetsAreDefaultAndFilteringIsExplicit() throws {
        let direct = try HumanoidWalkTask(configuration: HumanoidWalkTaskConfig(
            numEnvironments: 1, seed: 13, maxEpisodeSteps: 8,
            controlDecimation: 1))
        let filtered = try HumanoidWalkTask(configuration: HumanoidWalkTaskConfig(
            numEnvironments: 1, seed: 13, maxEpisodeSteps: 8,
            controlDecimation: 1, actionTargetResponse: 0.25))
        _ = try direct.reset(seed: 17)
        _ = try filtered.reset(seed: 17)
        var action = RLActionBatch(spec: direct.spec)
        action[0, 0] = 1
        var directResult = RLStepBatch(spec: direct.spec)
        var filteredResult = RLStepBatch(spec: filtered.spec)
        try direct.step(actions: action, into: &directResult)
        try filtered.step(actions: action, into: &filteredResult)

        // One H1 observation frame is [root 12, joints 19, velocities 19,
        // command 1, previous applied action 19]. Derive it from the task so
        // future imported actuator revisions cannot silently stale this test.
        let firstActionIndex = 12 + 2 * direct.spec.action.elementCount + 1
        XCTAssertEqual(directResult.observations.policy[firstActionIndex],
                       1, accuracy: 1e-6)
        XCTAssertEqual(filteredResult.observations.policy[firstActionIndex],
                       0.25, accuracy: 1e-6)
        XCTAssertThrowsError(try HumanoidWalkTask(
            configuration: HumanoidWalkTaskConfig(
                numEnvironments: 1, actionTargetResponse: 0)))
    }

    func testHumanoid64EnvironmentDefaultStanceSurvivesOnePPORollout() throws {
        let task = try HumanoidWalkTask(configuration: HumanoidWalkTaskConfig(
            numEnvironments: 64, seed: 31, maxEpisodeSteps: 600,
            controlDecimation: 4, autoReset: false))
        _ = try task.reset(seed: 37)
        let action = RLActionBatch(spec: task.spec)
        var result = RLStepBatch(spec: task.spec)
        // A floating-base H1 needs active balance. Its deterministic crouched
        // reset must nevertheless stay recoverable for a complete default
        // 24-step PPO rollout; requiring passive standing would make
        // immobility an artificial task attractor.
        for step in 0..<24 {
            try task.step(actions: action, into: &result)
            if result.terminated.contains(true) {
                let state = task.environment.states()[0]
                let up = state.torso.rotation.act(F3(0, 0, 1))
                XCTFail("zero-action humanoid fell at control step \(step + 1); "
                    + "root=\(state.root.position), up=\(up), "
                    + "joints=\(state.jointAngles)")
                return
            }
        }
        XCTAssertFalse(result.truncated.contains(true))
        XCTAssertTrue(result.observations.policy.allSatisfy(\.isFinite))
    }

    func testHumanoidMotorAnchorsMatchAtNeutralSpawn() throws {
        let environment = try HumanoidWalkEnv(numEnvironments: 2, seed: 39)
        for reference in environment.refs {
            for motor in reference.motors {
                let joint = environment.scene.joints[motor]
                let bodyA = environment.scene.bodies[joint.bodyA]
                let bodyB = environment.scene.bodies[joint.bodyB]
                let anchorA = bodyA.position + bodyA.rotation.act(joint.rA)
                let anchorB = bodyB.position + bodyB.rotation.act(joint.rB)
                XCTAssertLessThan(simd_length(anchorA - anchorB), 1e-5,
                                  "motor joint \(motor) has a reset anchor impulse")
            }
        }
    }

    func testHumanoidResetJointAnglesMatchNeutralMotorTargets() throws {
        let task = try HumanoidWalkTask(configuration: HumanoidWalkTaskConfig(
            numEnvironments: 1, seed: 211, maxEpisodeSteps: 8,
            controlDecimation: 1))
        _ = try task.reset(seed: 223)
        let angles = task.environment.states()[0].jointAngles
        XCTAssertEqual(angles.count, HumanoidWalkEnv.defaultJointPositions.count)
        for (joint, pair) in zip(angles,
                               HumanoidWalkEnv.defaultJointPositions).enumerated() {
            XCTAssertEqual(pair.0, pair.1, accuracy: 0.03,
                           "joint \(joint) reset angle must match its neutral target")
        }
    }

    func testHumanoidLegActuationCanCreateStableSingleSupport() throws {
        // A locomotion task is not learnable if its normalized action range
        // cannot unload either foot while the opposite leg supports the body.
        // Sweep staged weight-transfer and left-leg targets as a plant-level
        // diagnostic; these are never exposed to PPO as a controller or
        // demonstration.
        var patterns = [(shiftRoll: Float,
                         pitch: Float, knee: Float, ankle: Float)]()
        for shiftRoll in [Float(-1), -0.5, 0, 0.5, 1] {
            for pitch in [Float(-1), 0, 1] {
                for knee in [Float(0.5), 1] {
                    for ankle in [Float(-1), 0, 1] {
                        patterns.append((shiftRoll, pitch, knee, ankle))
                    }
                }
            }
        }
        let environment = try HumanoidWalkEnv(
            numEnvironments: patterns.count, seed: 59)
        var actions = ContiguousArray(
            repeating: Float(0),
            count: patterns.count * HumanoidWalkEnv.jointRanges.count)
        var stableSingleSupportSteps = [Int](repeating: 0,
                                              count: patterns.count)
        var anySingleSupportSteps = [Int](repeating: 0,
                                           count: patterns.count)
        var maximumSwingClearance = [Float](repeating: 0,
                                             count: patterns.count)
        for step in 0..<90 {
            // First move the pelvis over the right foot, then flex the left
            // leg. This proves controllability; it is neither a policy input
            // nor a training trajectory.
            let shiftResponse = min(Float(step + 1) / 20, 1)
            let swingResponse = simd_clamp(Float(step - 29) / 15, 0, 1)
            for (environmentID, pattern) in patterns.enumerated() {
                let offset = environmentID * HumanoidWalkEnv.jointRanges.count
                actions[offset + 1] = pattern.shiftRoll * shiftResponse
                actions[offset + 6] = pattern.shiftRoll * shiftResponse
                actions[offset + 2] = pattern.pitch * swingResponse
                actions[offset + 3] = pattern.knee * swingResponse
                actions[offset + 4] = pattern.ankle * swingResponse
            }
            environment.step(normalizedActions: actions, decimation: 4)
            for (environmentID, state) in environment.states().enumerated() {
                let up = state.torso.rotation.act(F3(0, 0, 1))
                let leftContact = HumanoidWalkTask.footInContact(state.leftFoot)
                let rightContact = HumanoidWalkTask.footInContact(state.rightFoot)
                maximumSwingClearance[environmentID] = max(
                    maximumSwingClearance[environmentID],
                    HumanoidWalkTask.footGroundClearance(state.leftFoot))
                if !leftContact, rightContact {
                    anySingleSupportSteps[environmentID] += 1
                }
                if state.root.position.z >= 0.42, up.z >= 0.35,
                   !leftContact, rightContact {
                    stableSingleSupportSteps[environmentID] += 1
                }
            }
        }
        let best = stableSingleSupportSteps.indices.max {
            stableSingleSupportSteps[$0] < stableSingleSupportSteps[$1]
        }!
        let bestLift = maximumSwingClearance.indices.max {
            maximumSwingClearance[$0] < maximumSwingClearance[$1]
        }!
        XCTAssertGreaterThanOrEqual(
            stableSingleSupportSteps[best], 3,
            "normalized leg targets never created stable single support; "
                + "best pattern=\(patterns[best]), steps="
                + "\(stableSingleSupportSteps[best]), clearance="
                + "\(maximumSwingClearance[best]); best lift pattern="
                + "\(patterns[bestLift]), any support="
                + "\(anySingleSupportSteps[bestLift]), clearance="
                + "\(maximumSwingClearance[bestLift])")
        XCTAssertGreaterThan(maximumSwingClearance[best], 0.04)
    }

    func testHumanoidObjectiveRewardsSustainedCommandedTravelNotSprintAndFall() {
        let dt: Float = 1 / 50
        let command: Float = 0.3
        let standingStep = HumanoidLocomotionObjective.reward(
            controlStep: dt, commandedSpeed: command,
            measuredVelocity: .zero, forwardDisplacement: 0,
            tiltSquared: 0, angularVelocityXYSquared: 0,
            yawAngularVelocitySquared: 0, headingErrorSquared: 0,
            actionRateSquared: 0, feetAirTime: 0,
            feetSlideSpeed: 0, jointDeviation: 0, fallen: false)
        let walkingStep = HumanoidLocomotionObjective.reward(
            controlStep: dt, commandedSpeed: command,
            measuredVelocity: F3(command, 0, 0),
            forwardDisplacement: command * dt,
            tiltSquared: 0, angularVelocityXYSquared: 0,
            yawAngularVelocitySquared: 0, headingErrorSquared: 0,
            actionRateSquared: 0, feetAirTime: 0,
            feetSlideSpeed: 0, jointDeviation: 0, fallen: false)
        XCTAssertGreaterThan(walkingStep, standingStep)
        let locomotionCommand: Float = 0.55
        let fullSpeedStanding = HumanoidLocomotionObjective.reward(
            controlStep: dt, commandedSpeed: locomotionCommand,
            measuredVelocity: .zero, forwardDisplacement: 0,
            tiltSquared: 0, angularVelocityXYSquared: 0,
            yawAngularVelocitySquared: 0, headingErrorSquared: 0,
            actionRateSquared: 0, feetAirTime: 0,
            feetSlideSpeed: 0, jointDeviation: 0, fallen: false)
        let fullSpeedWalking = HumanoidLocomotionObjective.reward(
            controlStep: dt, commandedSpeed: locomotionCommand,
            measuredVelocity: F3(locomotionCommand, 0, 0),
            forwardDisplacement: locomotionCommand * dt,
            tiltSquared: 0, angularVelocityXYSquared: 0,
            yawAngularVelocitySquared: 0, headingErrorSquared: 0,
            actionRateSquared: 0, feetAirTime: 0,
            feetSlideSpeed: 0, jointDeviation: 0, fallen: false)
        XCTAssertGreaterThan(fullSpeedWalking, fullSpeedStanding)
        XCTAssertEqual(HumanoidLocomotionObjective.laneTracking(
            lateralDisplacementSquared: 0,
            standardDeviation: 0.30), 1, accuracy: 1e-6)
        XCTAssertEqual(HumanoidLocomotionObjective.laneTracking(
            lateralDisplacementSquared: 0.09,
            standardDeviation: 0.30), exp(-1), accuracy: 1e-6)
        let alternatingTouchdown = HumanoidLocomotionObjective.reward(
            controlStep: dt, commandedSpeed: locomotionCommand,
            measuredVelocity: F3(locomotionCommand, 0, 0),
            forwardDisplacement: locomotionCommand * dt,
            tiltSquared: 0, angularVelocityXYSquared: 0,
            yawAngularVelocitySquared: 0, headingErrorSquared: 0,
            actionRateSquared: 0, feetAirTime: 0,
            alternatingTouchdown: 1, alternatingTouchdownWeight: 0.25,
            feetSlideSpeed: 0, jointDeviation: 0, fallen: false)
        XCTAssertEqual(alternatingTouchdown - fullSpeedWalking, 0.25,
                       accuracy: 1e-6)
        let bothFeetAirborne = HumanoidLocomotionObjective.reward(
            controlStep: dt, commandedSpeed: locomotionCommand,
            measuredVelocity: F3(locomotionCommand, 0, 0),
            forwardDisplacement: locomotionCommand * dt,
            tiltSquared: 0, angularVelocityXYSquared: 0,
            yawAngularVelocitySquared: 0, headingErrorSquared: 0,
            actionRateSquared: 0, feetAirTime: 0,
            feetFlight: 1, feetFlightPenaltyWeight: 1,
            feetSlideSpeed: 0, jointDeviation: 0, fallen: false)
        XCTAssertEqual(fullSpeedWalking - bothFeetAirborne, dt,
                       accuracy: 1e-6,
                       "a phase-free airborne interval must be less valuable "
                           + "than otherwise identical grounded travel")
        let plantedGlide = HumanoidLocomotionObjective.reward(
            controlStep: dt, commandedSpeed: locomotionCommand,
            measuredVelocity: F3(0.30, 0, 0),
            forwardDisplacement: 0.30 * dt,
            tiltSquared: 0, angularVelocityXYSquared: 0,
            yawAngularVelocitySquared: 0, headingErrorSquared: 0,
            actionRateSquared: 0, feetAirTime: 0,
            // Both support feet translating with the base at 0.30 m/s.
            feetSlideSpeed: 2 * 0.30,
            jointDeviation: 0, fallen: false)
        let frictionlessGlide = HumanoidLocomotionObjective.reward(
            controlStep: dt, commandedSpeed: locomotionCommand,
            measuredVelocity: F3(0.30, 0, 0),
            forwardDisplacement: 0.30 * dt,
            tiltSquared: 0, angularVelocityXYSquared: 0,
            yawAngularVelocitySquared: 0, headingErrorSquared: 0,
            actionRateSquared: 0, feetAirTime: 0,
            feetSlideSpeed: 0, jointDeviation: 0, fallen: false)
        XCTAssertLessThan(
            plantedGlide, frictionlessGlide,
            "contact-foot translation must pay the published H1 slip cost")
        let exploratoryWalking = HumanoidLocomotionObjective.reward(
            controlStep: dt, commandedSpeed: locomotionCommand,
            measuredVelocity: F3(locomotionCommand, 0, 0),
            forwardDisplacement: locomotionCommand * dt,
            tiltSquared: 0, angularVelocityXYSquared: 0,
            yawAngularVelocitySquared: 0, headingErrorSquared: 0,
            actionRateSquared: 10, feetAirTime: 0,
            feetSlideSpeed: 0, jointDeviation: 0, fallen: false)
        XCTAssertGreaterThan(
            exploratoryWalking, 0,
            "ordinary exploratory action changes must not erase all dense locomotion reward")
        let lateralDrift = HumanoidLocomotionObjective.reward(
            controlStep: dt, commandedSpeed: locomotionCommand,
            measuredVelocity: F3(locomotionCommand, 0, 0),
            forwardDisplacement: locomotionCommand * dt,
            tiltSquared: 0, angularVelocityXYSquared: 0,
            yawAngularVelocitySquared: 0, headingErrorSquared: 0,
            actionRateSquared: 0, feetAirTime: 0,
            feetSlideSpeed: 0, jointDeviation: 0, fallen: false,
            lateralDisplacementSquared: 0.25)
        XCTAssertLessThan(lateralDrift, fullSpeedWalking)
        let sprintStep = HumanoidLocomotionObjective.reward(
            controlStep: dt, commandedSpeed: command,
            measuredVelocity: F3(0.9, 0, 0),
            forwardDisplacement: 0.9 * dt,
            tiltSquared: 0, angularVelocityXYSquared: 0,
            yawAngularVelocitySquared: 0, headingErrorSquared: 0,
            actionRateSquared: 0, feetAirTime: 0,
            feetSlideSpeed: 0, jointDeviation: 0, fallen: false)
        XCTAssertGreaterThan(walkingStep, sprintStep)
        let backwardFacingStep = HumanoidLocomotionObjective.reward(
            controlStep: dt, commandedSpeed: command,
            measuredVelocity: F3(command, 0, 0),
            forwardDisplacement: command * dt,
            tiltSquared: 0, angularVelocityXYSquared: 0,
            yawAngularVelocitySquared: 0, headingErrorSquared: 4,
            actionRateSquared: 0, feetAirTime: 0,
            feetSlideSpeed: 0, jointDeviation: 0, fallen: false)
        XCTAssertLessThan(backwardFacingStep, walkingStep)

        let heavilyPenalizedAlive = HumanoidLocomotionObjective.reward(
            controlStep: dt, commandedSpeed: command,
            measuredVelocity: .zero, forwardDisplacement: 0,
            tiltSquared: 4, angularVelocityXYSquared: 10,
            yawAngularVelocitySquared: 10, headingErrorSquared: 4,
            actionRateSquared: 100, feetAirTime: 0,
            feetSlideSpeed: 10, jointDeviation: 10, fallen: false)
        let heavilyPenalizedFall = HumanoidLocomotionObjective.reward(
            controlStep: dt, commandedSpeed: command,
            measuredVelocity: .zero, forwardDisplacement: 0,
            tiltSquared: 4, angularVelocityXYSquared: 10,
            yawAngularVelocitySquared: 10, headingErrorSquared: 4,
            actionRateSquared: 100, feetAirTime: 0,
            feetSlideSpeed: 10, jointDeviation: 10, fallen: true)
        XCTAssertLessThan(heavilyPenalizedAlive, 0)
        XCTAssertEqual(heavilyPenalizedFall - heavilyPenalizedAlive,
                       -HumanoidLocomotionObjective.terminationPenalty,
                       accuracy: 1e-6)

        let fullWalk = 1_000 * walkingStep
        let fullStand = 1_000 * standingStep
        let sprintThenFall = 149 * sprintStep
            + HumanoidLocomotionObjective.reward(
                controlStep: dt, commandedSpeed: command,
                measuredVelocity: F3(0.9, 0, 0),
                forwardDisplacement: 0.9 * dt,
                tiltSquared: 0, angularVelocityXYSquared: 0,
                yawAngularVelocitySquared: 0, headingErrorSquared: 0,
                actionRateSquared: 0, feetAirTime: 0,
                feetSlideSpeed: 0, jointDeviation: 0, fallen: true)
        XCTAssertGreaterThan(fullWalk, fullStand)
        XCTAssertGreaterThan(fullStand, sprintThenFall)
    }

    func testHumanoidFootClearanceCostIsPhaseFreeAndRewardsSwingClearance() {
        let speedSquared: Float = 0.6 * 0.6
        let draggingCost = HumanoidLocomotionObjective.footClearanceCost(
            clearance: 0, horizontalSpeedSquared: speedSquared)
        let swingCost = HumanoidLocomotionObjective.footClearanceCost(
            clearance: HumanoidLocomotionObjective.targetFootClearance,
            horizontalSpeedSquared: speedSquared)
        let stationaryCost = HumanoidLocomotionObjective.footClearanceCost(
            clearance: 0, horizontalSpeedSquared: 0)

        XCTAssertGreaterThan(draggingCost, 0)
        XCTAssertEqual(swingCost, 0, accuracy: 1e-7)
        XCTAssertEqual(stationaryCost, 0, accuracy: 1e-7)

        let dt: Float = 1 / 30
        let commonVelocity = F3(0.55, 0, 0)
        let draggingReward = HumanoidLocomotionObjective.reward(
            controlStep: dt, commandedSpeed: 0.55,
            measuredVelocity: commonVelocity, forwardDisplacement: 0.55 * dt,
            tiltSquared: 0, angularVelocityXYSquared: 0,
            yawAngularVelocitySquared: 0, headingErrorSquared: 0,
            actionRateSquared: 0, feetAirTime: 0,
            feetSlideSpeed: 0, jointDeviation: 0, fallen: false,
            feetClearanceCost: draggingCost)
        let swingReward = HumanoidLocomotionObjective.reward(
            controlStep: dt, commandedSpeed: 0.55,
            measuredVelocity: commonVelocity, forwardDisplacement: 0.55 * dt,
            tiltSquared: 0, angularVelocityXYSquared: 0,
            yawAngularVelocitySquared: 0, headingErrorSquared: 0,
            actionRateSquared: 0, feetAirTime: 0,
            feetSlideSpeed: 0, jointDeviation: 0, fallen: false,
            feetClearanceCost: swingCost)
        XCTAssertGreaterThan(swingReward, draggingReward)
    }

    func testHumanoidPositiveBipedAirTimeIsDenseAndPhaseFree() {
        let singleStance = HumanoidLocomotionObjective.positiveBipedAirTime(
            leftInContact: true, rightInContact: false,
            leftAirTime: 0, rightAirTime: 0.25,
            leftContactTime: 0.40, rightContactTime: 0)
        let mirrored = HumanoidLocomotionObjective.positiveBipedAirTime(
            leftInContact: false, rightInContact: true,
            leftAirTime: 0.25, rightAirTime: 0,
            leftContactTime: 0, rightContactTime: 0.40)
        let doubleSupport = HumanoidLocomotionObjective.positiveBipedAirTime(
            leftInContact: true, rightInContact: true,
            leftAirTime: 0, rightAirTime: 0,
            leftContactTime: 1, rightContactTime: 1)
        let flight = HumanoidLocomotionObjective.positiveBipedAirTime(
            leftInContact: false, rightInContact: false,
            leftAirTime: 0.3, rightAirTime: 0.3,
            leftContactTime: 0, rightContactTime: 0)

        XCTAssertEqual(singleStance, 0.25, accuracy: 1e-6)
        XCTAssertEqual(mirrored, singleStance, accuracy: 1e-6)
        XCTAssertEqual(doubleSupport, 0)
        XCTAssertEqual(flight, 0)
    }

    func testHumanoidGaitRewardsFadeOutForStandCommands() {
        XCTAssertLessThan(HumanoidLocomotionObjective.velocityTracking(
            measured: 0.6, commanded: 0.3, standardDeviation: 0.2),
            HumanoidLocomotionObjective.velocityTracking(
                measured: 0.6, commanded: 0.3))
        XCTAssertEqual(HumanoidLocomotionObjective.movementCommandWeight(
            commandedSpeed: 0), 0, accuracy: 1e-6)
        XCTAssertEqual(HumanoidLocomotionObjective.movementCommandWeight(
            commandedSpeed: 0.1), 0.5, accuracy: 1e-6)
        XCTAssertEqual(HumanoidLocomotionObjective.movementCommandWeight(
            commandedSpeed: 0.2), 1, accuracy: 1e-6)
        XCTAssertEqual(HumanoidLocomotionObjective.movementCommandWeight(
            commandedSpeed: 0.65), 1, accuracy: 1e-6)

        let velocity = F3(0.3, 0.4, 2)
        XCTAssertEqual(HumanoidLocomotionObjective.standStillVelocityCost(
            commandedSpeed: 0, measuredVelocity: velocity), 0.25,
            accuracy: 1e-6)
        XCTAssertEqual(HumanoidLocomotionObjective.standStillVelocityCost(
            commandedSpeed: 0.1, measuredVelocity: velocity), 0.125,
            accuracy: 1e-6)
        XCTAssertEqual(HumanoidLocomotionObjective.standStillVelocityCost(
            commandedSpeed: 0.2, measuredVelocity: velocity), 0,
            accuracy: 1e-6)
        XCTAssertEqual(HumanoidLocomotionObjective.standStillVelocityCost(
            commandedSpeed: 0.65, measuredVelocity: velocity), 0,
            accuracy: 1e-6)
        XCTAssertEqual(HumanoidLocomotionObjective.standStillJointDeviationCost(
            commandedSpeed: 0, jointDeviationAbsolute: 0.75), 0.75,
            accuracy: 1e-6)
        XCTAssertEqual(HumanoidLocomotionObjective.standStillJointDeviationCost(
            commandedSpeed: 0.1, jointDeviationAbsolute: 0.75), 0.375,
            accuracy: 1e-6)
        XCTAssertEqual(HumanoidLocomotionObjective.standStillJointDeviationCost(
            commandedSpeed: 0.2, jointDeviationAbsolute: 0.75), 0,
            accuracy: 1e-6)
        XCTAssertEqual(HumanoidLocomotionObjective
            .standStillDoubleSupportReward(
                commandedSpeed: 0, bothFeetInContact: true), 1,
            accuracy: 1e-6)
        XCTAssertEqual(HumanoidLocomotionObjective
            .standStillDoubleSupportReward(
                commandedSpeed: 0.1, bothFeetInContact: true), 0.5,
            accuracy: 1e-6)
        XCTAssertEqual(HumanoidLocomotionObjective
            .standStillDoubleSupportReward(
                commandedSpeed: 0, bothFeetInContact: false), 0,
            accuracy: 1e-6)
        XCTAssertEqual(HumanoidLocomotionObjective
            .standStillDoubleSupportReward(
                commandedSpeed: 0.2, bothFeetInContact: true), 0,
            accuracy: 1e-6)
        XCTAssertEqual(HumanoidLocomotionObjective.standStillFallCost(
            commandedSpeed: 0, fallen: true, penalty: 20), 20,
            accuracy: 1e-6)
        XCTAssertEqual(HumanoidLocomotionObjective.standStillFallCost(
            commandedSpeed: 0.1, fallen: true, penalty: 20), 10,
            accuracy: 1e-6)
        XCTAssertEqual(HumanoidLocomotionObjective.standStillFallCost(
            commandedSpeed: 0.2, fallen: true, penalty: 20), 0,
            accuracy: 1e-6)
        XCTAssertEqual(HumanoidLocomotionObjective.standStillFallCost(
            commandedSpeed: 0, fallen: false, penalty: 20), 0,
            accuracy: 1e-6)
    }

    func testStandingFallPenaltyHasDistinctSerializedTaskRevision() throws {
        let registered = try BuiltInRLTasks.registry.make(
            "humanoid-walk-v0",
            configuration: RLTaskConfiguration(
                numEnvironments: 1, seed: 50,
                options: [
                    "standingCommandProbability": 0.5,
                    "commandGatedActor": 1,
                    "standStillFallPenalty": 20,
                ]))
        let task = try XCTUnwrap(registered as? HumanoidWalkTask)
        XCTAssertEqual(task.spec.revision,
                       RLPhysicsContract.fixedGainActuatorV2(45))
        XCTAssertEqual(task.configuration.standStillFallPenalty, 20)
        XCTAssertEqual(task.spec.configurationValues["standStillFallPenalty"], 20)
    }

    func testLowSpeedTrackingWidthIsSerializedAndRevisioned() throws {
        let registered = try BuiltInRLTasks.registry.make(
            "humanoid-walk-v0",
            configuration: RLTaskConfiguration(
                numEnvironments: 1, seed: 51,
                options: [
                    "velocityTrackingStandardDeviation": 0.2,
                    "velocityTrackingErrorPenaltyWeight": 10,
                ]))
        let task = try XCTUnwrap(registered as? HumanoidWalkTask)
        XCTAssertEqual(task.spec.revision,
                       RLPhysicsContract.fixedGainActuatorV2(2_435))
        XCTAssertEqual(task.spec.configurationValues[
            "velocityTrackingStandardDeviation"], 0.2)
        XCTAssertEqual(task.spec.configurationValues[
            "velocityTrackingErrorPenaltyWeight"], 10)
        XCTAssertThrowsError(try HumanoidWalkTask(configuration: .init(
            numEnvironments: 1, velocityTrackingStandardDeviation: 0)))
        XCTAssertThrowsError(try HumanoidWalkTask(configuration: .init(
            numEnvironments: 1, velocityTrackingErrorPenaltyWeight: -1)))
    }

    func testHumanoidStepExchangeRequiresLandingFootToPassOtherFoot() {
        let heading = F3(1, 0, 0)
        XCTAssertTrue(HumanoidLocomotionObjective.isLeadingTouchdown(
            touchdownFootPosition: F3(0.16, 0.08, 0),
            otherFootPosition: F3(0, -0.08, 0), heading: heading))
        XCTAssertTrue(HumanoidLocomotionObjective.isLeadingTouchdown(
            touchdownFootPosition: F3(0.16, -0.08, 0),
            otherFootPosition: F3(0, 0.08, 0), heading: heading),
            "either leg may lead without a prescribed gait phase")
        XCTAssertFalse(HumanoidLocomotionObjective.isLeadingTouchdown(
            touchdownFootPosition: F3(-0.16, 0.08, 0),
            otherFootPosition: F3(0, -0.08, 0), heading: heading),
            "a rear split-stance touchdown must not count as a step")
        XCTAssertFalse(HumanoidLocomotionObjective.isLeadingTouchdown(
            touchdownFootPosition: F3(0.04, 0.08, 0),
            otherFootPosition: F3(0, -0.08, 0), heading: heading),
            "contact chatter without a real foot exchange must not count")
    }

    func testPointGoalYawCommandTurnsAlongShortestBoundedDirection() {
        let fortyFiveDegreesLeft = F3(cos(.pi / 4), -sin(.pi / 4), 0)
        let fortyFiveDegreesRight = F3(cos(.pi / 4), sin(.pi / 4), 0)
        XCTAssertEqual(HumanoidLocomotionObjective.pointGoalYawRate(
            relativeHeading: F3(1, 0, 0)), 0, accuracy: 1e-6)
        XCTAssertEqual(HumanoidLocomotionObjective.pointGoalYawRate(
            relativeHeading: fortyFiveDegreesLeft), 1.2, accuracy: 1e-6)
        XCTAssertEqual(HumanoidLocomotionObjective.pointGoalYawRate(
            relativeHeading: fortyFiveDegreesRight), -1.2, accuracy: 1e-6)
        XCTAssertEqual(HumanoidLocomotionObjective.pointGoalYawRate(
            relativeHeading: F3(-1, 0, 0)), -1.2, accuracy: 1e-6)
    }

    func testPointGoalCommandSlowsToAStableDwell() throws {
        XCTAssertEqual(HumanoidLocomotionObjective.pointGoalCommandSpeed(
            remainingDistance: 4, cruiseSpeed: 0.6,
            goalRadius: 1.5, slowdownDistance: 3), 0.6,
            accuracy: 1e-6)
        XCTAssertEqual(HumanoidLocomotionObjective.pointGoalCommandSpeed(
            remainingDistance: 2.25, cruiseSpeed: 0.6,
            goalRadius: 1.5, slowdownDistance: 3), 0.3,
            accuracy: 1e-6)
        XCTAssertEqual(HumanoidLocomotionObjective.pointGoalCommandSpeed(
            remainingDistance: 1.5, cruiseSpeed: 0.6,
            goalRadius: 1.5, slowdownDistance: 3), 0,
            accuracy: 1e-6)
        XCTAssertEqual(HumanoidLocomotionObjective.pointGoalCommandSpeed(
            remainingDistance: 2.25, cruiseSpeed: 0.6,
            goalRadius: 1.5, slowdownDistance: 3,
            boundaryCommandSpeed: 0.2), 0.4, accuracy: 1e-6)
        XCTAssertEqual(HumanoidLocomotionObjective.pointGoalCommandSpeed(
            remainingDistance: 1.500_001, cruiseSpeed: 0.6,
            goalRadius: 1.5, slowdownDistance: 3,
            boundaryCommandSpeed: 0.2), 0.2, accuracy: 1e-5)
        XCTAssertEqual(HumanoidLocomotionObjective.pointGoalCommandSpeed(
            remainingDistance: 1.5, cruiseSpeed: 0.6,
            goalRadius: 1.5, slowdownDistance: 3,
            boundaryCommandSpeed: 0.2), 0, accuracy: 1e-6)
        XCTAssertEqual(HumanoidLocomotionObjective.pointGoalProximity(
            remainingDistance: 4, goalRadius: 1.5,
            slowdownDistance: 3), 0, accuracy: 1e-6)
        XCTAssertEqual(HumanoidLocomotionObjective.pointGoalProximity(
            remainingDistance: 1.5, goalRadius: 1.5,
            slowdownDistance: 3), 1, accuracy: 1e-6)
        XCTAssertEqual(HumanoidLocomotionObjective.pointGoalArrivalQuality(
            proximity: 1, planarSpeed: 0, upright: 1,
            arrivalSpeed: 0.25), 1, accuracy: 1e-6)
        XCTAssertLessThan(HumanoidLocomotionObjective.pointGoalArrivalQuality(
            proximity: 1, planarSpeed: 0.5, upright: 1,
            arrivalSpeed: 0.25), 0.02)
        XCTAssertEqual(HumanoidLocomotionObjective.pointGoalArrivalQuality(
            proximity: 0, planarSpeed: 0, upright: 1,
            arrivalSpeed: 0.25), 0, accuracy: 1e-6)

        let task = try HumanoidWalkTask(configuration: .init(
            numEnvironments: 1, seed: 751, maxEpisodeSteps: 8,
            controlDecimation: 1, minimumCommandSpeed: 0,
            maximumCommandSpeed: 0, initialRollPitchRange: 0,
            initialYawRange: 0, maximumGoalDirectionAngle: 0,
            goalDwellSteps: 2, goalStableDwellRewardWeight: 10,
            autoReset: false),
            taskID: "humanoid-goal-v0", taskRevision: 10)
        XCTAssertEqual(
            task.spec.configurationValues["goalStableDwellRewardWeight"], 10)
        _ = try task.reset(seed: 753)
        let actions = try RLActionBatch(
            numEnvironments: 1,
            actionDimension: HumanoidWalkEnv.jointRanges.count,
            values: ContiguousArray(repeating: 0,
                count: HumanoidWalkEnv.jointRanges.count))
        var result = RLStepBatch(spec: task.spec)
        try task.step(actions: actions, into: &result)
        XCTAssertFalse(result.successes[0])
        try task.step(actions: actions, into: &result)
        XCTAssertTrue(result.successes[0])
        XCTAssertGreaterThan(
            result.metrics["reward/goal_stable_dwell"]?[0] ?? 0, 0)
        XCTAssertTrue(result.terminated[0])
        XCTAssertFalse(result.truncated[0])
        XCTAssertEqual(result.metrics["episode/goal_dwell_steps"]?[0], 2)
        XCTAssertEqual(result.metrics["episode/goal_radius_entered_bin"]?[0], 1)
        XCTAssertEqual(
            result.metrics["episode/goal_radius_entry_speed_mps"]?[0] ?? .nan, 0,
            accuracy: 1e-6)
        XCTAssertEqual(result.metrics[
            "episode/goal_radius_entry_forward_speed_mps"]?[0] ?? .nan, 0,
            accuracy: 1e-6)
        XCTAssertEqual(result.metrics[
            "episode/goal_radius_entry_lateral_speed_mps"]?[0] ?? .nan, 0,
            accuracy: 1e-6)
        XCTAssertEqual(result.metrics[
            "episode/goal_radius_exited_after_entry"]?[0], 0)
        XCTAssertEqual(
            result.metrics["episode/goal_radius_minimum_speed_mps"]?[0] ?? .nan, 0,
            accuracy: 1e-6)
        XCTAssertEqual(result.metrics["episode/goal_radius_inside_steps"]?[0], 2)
        XCTAssertEqual(result.metrics["episode/goal_fell_after_entry"]?[0], 0)
        XCTAssertEqual(result.metrics[
            "episode/goal_failure_fall_before_entry"]?[0], 0)
        XCTAssertEqual(result.metrics[
            "episode/goal_failure_timeout_after_entry"]?[0], 0)
        XCTAssertEqual(result.metrics[
            "episode/goal_failure_timeout_without_entry"]?[0], 0)
    }

    func testHumanoidSuccessRequiresFullHorizonCommandTracking() {
        XCTAssertFalse(HumanoidLocomotionObjective.isSuccessful(
            timedOut: false, fallen: false, forwardDistance: 6,
            lateralDistance: 0, headingAlignment: 1,
            alternatingSteps: 24,
            commandedSpeed: 0.3, elapsed: 20))
        XCTAssertFalse(HumanoidLocomotionObjective.isSuccessful(
            timedOut: true, fallen: true, forwardDistance: 6,
            lateralDistance: 0, headingAlignment: 1,
            alternatingSteps: 24,
            commandedSpeed: 0.3, elapsed: 20))
        XCTAssertFalse(HumanoidLocomotionObjective.isSuccessful(
            timedOut: true, fallen: false, forwardDistance: 3.5,
            lateralDistance: 0, headingAlignment: 1,
            alternatingSteps: 24,
            commandedSpeed: 0.3, elapsed: 20))
        XCTAssertTrue(HumanoidLocomotionObjective.isSuccessful(
            timedOut: true, fallen: false, forwardDistance: 6,
            lateralDistance: 0, headingAlignment: 1,
            alternatingSteps: 24,
            commandedSpeed: 0.3, elapsed: 20))
        XCTAssertFalse(HumanoidLocomotionObjective.isSuccessful(
            timedOut: true, fallen: false, forwardDistance: 8.5,
            lateralDistance: 0, headingAlignment: 1,
            alternatingSteps: 24,
            commandedSpeed: 0.3, elapsed: 20))
        XCTAssertFalse(HumanoidLocomotionObjective.isSuccessful(
            timedOut: true, fallen: false, forwardDistance: 6,
            lateralDistance: 0.31, headingAlignment: 1,
            alternatingSteps: 24,
            commandedSpeed: 0.3, elapsed: 20))
        XCTAssertFalse(HumanoidLocomotionObjective.isSuccessful(
            timedOut: true, fallen: false, forwardDistance: 6,
            lateralDistance: 0, headingAlignment: 0.74,
            alternatingSteps: 24,
            commandedSpeed: 0.3, elapsed: 20))
        XCTAssertFalse(HumanoidLocomotionObjective.isSuccessful(
            timedOut: true, fallen: false, forwardDistance: 6,
            lateralDistance: 0, headingAlignment: 1,
            alternatingSteps: 19,
            commandedSpeed: 0.3, elapsed: 20))
    }

    func testHumanoidStandingSuccessRequiresSurvivalAndLowDriftNotGait() {
        XCTAssertTrue(HumanoidLocomotionObjective.isSuccessful(
            timedOut: true, fallen: false, forwardDistance: 1.9,
            lateralDistance: 0.2, headingAlignment: 0.9,
            alternatingSteps: 0, commandedSpeed: 0, elapsed: 20))
        XCTAssertFalse(HumanoidLocomotionObjective.isSuccessful(
            timedOut: true, fallen: false, forwardDistance: 2.1,
            lateralDistance: 0.2, headingAlignment: 0.9,
            alternatingSteps: 30, commandedSpeed: 0, elapsed: 20))
        XCTAssertFalse(HumanoidLocomotionObjective.isSuccessful(
            timedOut: false, fallen: true, forwardDistance: 0,
            lateralDistance: 0, headingAlignment: 1,
            alternatingSteps: 0, commandedSpeed: 0, elapsed: 10))
    }

    func testMixedCommandAcceptanceRequiresStandingAndMovingCohorts() throws {
        let task = try HumanoidWalkTask(configuration: HumanoidWalkTaskConfig(
            numEnvironments: 1, seed: 5, standingCommandProbability: 0.35))
        let failures = task.evaluationCriteria.failures(
            successRate: 0.90, meanEpisodeLength: 950,
            maxEpisodeSteps: task.spec.maxEpisodeSteps,
            taskMetrics: [
                "episode/standing_success_rate": 0.79,
                "episode/moving_success_rate": 0.95,
                "episode/speed_error_mps": 0.08,
                "episode/heading_alignment": 0.9,
                "episode/lateral_distance_m": 0.2,
            ])
        XCTAssertTrue(failures.contains {
            $0.contains("episode/standing_success_rate")
        })
    }

    func testHumanoidAcceptanceGateRejectsStandingMetrics() throws {
        let task = try HumanoidWalkTask(configuration: HumanoidWalkTaskConfig(
            numEnvironments: 1, seed: 3))
        let failures = task.evaluationCriteria.failures(
            successRate: 0, meanEpisodeLength: 600,
            maxEpisodeSteps: task.spec.maxEpisodeSteps,
            taskMetrics: [
                "episode/forward_distance_m": 0,
                "episode/forward_speed_mps": 0,
                "episode/speed_error_mps": 0.3,
                "episode/heading_alignment": 1,
                "episode/lateral_distance_m": 0,
                "episode/alternating_steps": 0,
            ])
        XCTAssertFalse(failures.isEmpty)
    }

    func testArmPushTUsesArticulatedActionsAndFiniteState() throws {
        let task = try ArmPushTTask(configuration: ArmPushTTaskConfig(
            numEnvironments: 3, seed: 13, maxEpisodeSteps: 8,
            controlDecimation: 1))
        XCTAssertEqual(task.spec.revision,
                       RLPhysicsContract.fixedGainActuatorV2(114))
        XCTAssertEqual(task.spec.action.name, "joint_delta_position")
        XCTAssertEqual(task.configuration.jointDeltaActionScale, 0.1)
        XCTAssertEqual(task.configuration.endEffectorDeltaActionScale, 0)
        XCTAssertEqual(task.configuration.linkLength1, 0.42)
        XCTAssertEqual(task.configuration.linkLength2, 0.40)
        XCTAssertEqual(task.environment.configuredLinkLengths,
                       ArmPushTEnv.linkLengths)
        XCTAssertEqual(task.spec.configurationValues["linkLength1"], 0.42)
        XCTAssertEqual(task.spec.configurationValues["linkLength2"], 0.40)
        XCTAssertEqual(task.configuration.linkMass, 2.7)
        XCTAssertEqual(task.configuration.tipMass, 0.75)
        XCTAssertEqual(task.configuration.motorTorque, 100)
        XCTAssertEqual(task.configuration.motorStiffness, 1_000)
        XCTAssertEqual(task.configuration.motorDamping, 100)
        XCTAssertEqual(task.configuration.motorArmature, 0.1)
        XCTAssertEqual(task.configuration.blockMass, 0.8)
        XCTAssertEqual(task.configuration.blockStaticFriction, 3)
        XCTAssertEqual(task.configuration.blockDynamicFriction, 3)
        XCTAssertEqual(task.configuration.coverageProgressWeight, 0)
        XCTAssertEqual(task.configuration.coverageRewardWeight, 0)
        XCTAssertEqual(task.configuration.poseProgressRewardWeight, 0)
        XCTAssertEqual(task.configuration.poseRewardWeight, 1)
        XCTAssertEqual(task.configuration.poseRewardDistanceScale, 5)
        XCTAssertEqual(task.configuration.reachingRewardWeight, 0.05)
        XCTAssertEqual(task.configuration.actionMagnitudePenaltyWeight, 0)
        XCTAssertFalse(task.configuration.precisionGatedActor)
        XCTAssertEqual(task.configuration.precisionExpertGateCoverage, 0.75)
        XCTAssertEqual(task.configuration.precisionExpertReleaseCoverage, 0.75)
        XCTAssertFalse(task.configuration.freezeBasePolicyExpert)
        XCTAssertEqual(task.configuration.goalProgressWeight, 0)
        XCTAssertEqual(task.configuration.reachProgressWeight, 0)
        XCTAssertEqual(task.configuration.yawProgressWeight, 0)
        XCTAssertEqual(task.configuration.actionRatePenaltyWeight, 0)
        XCTAssertEqual(task.configuration.pushContactProgressWeight, 0)
        XCTAssertEqual(task.configuration.reachDistancePenaltyWeight, 0)
        XCTAssertEqual(task.configuration.goalDistancePenaltyWeight, 0)
        XCTAssertEqual(task.configuration.yawErrorPenaltyWeight, 0)
        XCTAssertEqual(task.configuration.blockSpawnGoalBlend, 0)
        XCTAssertEqual(task.configuration.blockSpawnRadius, 0.07)
        XCTAssertEqual(task.configuration.blockSpawnYawRange, 0.35)
        XCTAssertEqual(task.configuration.successCoverage,
                       ArmPushTEnv.successCoverage)
        XCTAssertEqual(task.configuration.successYawTolerance, .pi)
        XCTAssertEqual(task.configuration.reachCurriculumSuccessDistance, 0)
        XCTAssertEqual(task.configuration.pushContactCurriculumSuccessDistance, 0)
        XCTAssertEqual(task.configuration.pushContactDistancePenaltyWeight, 0)
        XCTAssertEqual(
            task.configuration.pushContactCurriculumMaximumGoalRegression, 0)
        XCTAssertEqual(
            task.configuration.pushContactCurriculumMinimumGoalProgress, 0)
        XCTAssertEqual(
            task.configuration.pushContactCurriculumMaximumGoalDistance, 0)
        XCTAssertEqual(task.configuration.successRewardOverride, 3)
        let initial = try task.reset(seed: 17)
        XCTAssertEqual(initial.policy.count, 3 * 19)
        XCTAssertEqual(task.spec.action.shape, [2])
        XCTAssertEqual(task.environment.refs[0].motors.count, 2)
        let refs = task.environment.refs[0]
        let blockMass = [refs.blockBar, refs.blockStem].reduce(Float.zero) {
            total, bodyIndex in
            let body = task.environment.scene.bodies[bodyIndex]
            return total + body.density * body.size.x * body.size.y * body.size.z
        }
        XCTAssertEqual(blockMass, ArmPushTEnv.defaultBlockMass, accuracy: 1e-5)
        XCTAssertEqual(task.environment.scene.bodies[refs.blockBar].friction,
                       ArmPushTEnv.defaultBlockFriction)
        XCTAssertEqual(
            task.environment.scene.bodies[refs.blockBar].dynamicFriction,
            ArmPushTEnv.defaultBlockFriction)
        XCTAssertEqual(task.environment.scene.bodies[refs.blockBar].size,
                       ArmPushTEnv.blockBarSize)
        XCTAssertEqual(task.environment.scene.bodies[refs.blockStem].size,
                       ArmPushTEnv.blockStemSize)
        for link in refs.bodies.prefix(2) {
            let body = task.environment.scene.bodies[link]
            XCTAssertEqual(body.density * body.size.x * body.size.y
                * body.size.z, task.configuration.linkMass, accuracy: 1e-5)
        }
        for motor in refs.motors {
            let joint = task.environment.scene.joints[motor]
            XCTAssertEqual(joint.motorTorque, 100)
            XCTAssertEqual(joint.motorStiffness, 1_000)
            XCTAssertEqual(joint.motorDamping, 100)
            XCTAssertEqual(joint.armature, 0.1)
        }
        let action = RLActionBatch(spec: task.spec)
        var result = RLStepBatch(spec: task.spec)
        for _ in 0..<4 { try task.step(actions: action, into: &result) }
        XCTAssertTrue(result.observations.policy.allSatisfy(\.isFinite))
        XCTAssertTrue(result.rewards.allSatisfy(\.isFinite))
        XCTAssertTrue(try XCTUnwrap(
            result.metrics["reward/push_contact_progress"])
            .allSatisfy(\.isFinite))
    }

    func testArmPushTScoreIsCanonicalGeometricCoverage() {
        let goal = ArmPushTEnv.goalPosition
        XCTAssertEqual(ArmPushTEnv.coverage(
            blockPosition: goal, blockYaw: 0,
            goalPosition: goal, goalYaw: 0), 1, accuracy: 1e-5)
        XCTAssertEqual(ArmPushTEnv.coverage(
            blockPosition: SIMD2(-1, -1), blockYaw: 0,
            goalPosition: goal, goalYaw: 0), 0, accuracy: 1e-6)

        let translated = ArmPushTEnv.coverage(
            blockPosition: goal + SIMD2<Float>(0.06, 0), blockYaw: 0,
            goalPosition: goal, goalYaw: 0)
        let rotated = ArmPushTEnv.coverage(
            blockPosition: goal, blockYaw: .pi / 2,
            goalPosition: goal, goalYaw: 0)
        XCTAssertLessThan(translated, ArmPushTEnv.successCoverage,
                          "the former loose pose tolerance must not count as solved")
        XCTAssertLessThan(rotated, ArmPushTEnv.successCoverage)
        XCTAssertGreaterThan(translated, 0)
        XCTAssertGreaterThan(rotated, 0)
    }

    func testArmPushTPrecisionRewardSeparatesPoseErrors() {
        let solved = ArmPushTTask.precisionPoseReward(
            goalDistance: 0, yawError: 0)
        let translated = ArmPushTTask.precisionPoseReward(
            goalDistance: 0.5, yawError: 0)
        let rotated = ArmPushTTask.precisionPoseReward(
            goalDistance: 0, yawError: .pi / 2)
        let farAndReversed = ArmPushTTask.precisionPoseReward(
            goalDistance: 20, yawError: .pi)

        XCTAssertEqual(solved, 1, accuracy: 1e-6)
        XCTAssertLessThan(translated, solved)
        XCTAssertLessThan(rotated, solved)
        XCTAssertGreaterThan(translated, 0.5,
                             "translation must not erase rotation feedback")
        XCTAssertGreaterThan(rotated, 0.5,
                             "rotation must not erase translation feedback")
        XCTAssertEqual(farAndReversed, 0, accuracy: 1e-5)
        XCTAssertEqual(ArmPushTTask.reachingKernel(distance: 0), 1,
                       accuracy: 1e-6)
        XCTAssertLessThan(ArmPushTTask.reachingKernel(distance: 2),
                          ArmPushTTask.reachingKernel(distance: 0.5))
    }

    func testArmPushTExpertIKUsesPolicyJointActionInterface() {
        let target = ArmPushTEnv.goalPosition
        let action = ArmPushTEnv.normalizedJointTargets(tipTarget: target)
        let q1 = ArmPushTEnv.defaultJointPositions[0]
            + action.x * ArmPushTEnv.actionScales[0]
        let q2 = ArmPushTEnv.defaultJointPositions[1]
            + action.y * ArmPushTEnv.actionScales[1]
        let lengths = ArmPushTEnv.linkLengths
        let reconstructed = ArmPushTEnv.basePosition
            + lengths.x * SIMD2(cos(q1), sin(q1))
            + lengths.y * SIMD2(cos(q1 + q2), sin(q1 + q2))
        XCTAssertEqual(reconstructed.x, target.x, accuracy: 1e-4)
        XCTAssertEqual(reconstructed.y, target.y, accuracy: 1e-4)
        XCTAssertTrue((-1...1).contains(action.x))
        XCTAssertTrue((-1...1).contains(action.y))
    }

    func testArmPushTWorkspaceReachesEveryGoalFaceWithMargin() {
        let localVertices: [SIMD2<Float>] = [
            SIMD2(-0.10, 0), SIMD2(0.10, 0),
            SIMD2(-0.10, 0.05), SIMD2(0.10, 0.05),
            SIMD2(-0.025, -0.15), SIMD2(0.025, -0.15),
        ]
        let toolRadius: Float = 0.016
        let requiredReach = localVertices.map {
            simd_length(ArmPushTEnv.goalPosition + $0
                - ArmPushTEnv.basePosition) + toolRadius
        }.max()!
        let availableReach = ArmPushTEnv.linkLengths.x
            + ArmPushTEnv.linkLengths.y

        XCTAssertGreaterThan(requiredReach, 0.64,
            "the old arm could not reach the far goal face")
        XCTAssertGreaterThanOrEqual(availableReach - requiredReach, 0.05,
            "canonical goal-face contacts need recovery margin")
    }

    func testArmPushTJointDeltaControllerUsesMeasuredPosition() {
        let current = SIMD2<Float>(0.6, 1.2)
        let target = ArmPushTEnv.normalizedJointTargets(
            currentAngles: current, deltaActions: SIMD2(1, -1),
            deltaScale: 0.1)
        let reconstructed = SIMD2(
            ArmPushTEnv.defaultJointPositions[0]
                + target.x * ArmPushTEnv.actionScales[0],
            ArmPushTEnv.defaultJointPositions[1]
                + target.y * ArmPushTEnv.actionScales[1])

        XCTAssertEqual(reconstructed.x, current.x + 0.1, accuracy: 1e-6)
        XCTAssertEqual(reconstructed.y, current.y - 0.1, accuracy: 1e-6)
        XCTAssertLessThan(target.x, 1,
                          "a unit delta action must not command the joint limit")
    }

    func testArmPushTEndEffectorDeltaControllerIsExplicitAndBounded() throws {
        let target = ArmPushTTask.endEffectorDeltaTarget(
            currentPosition: SIMD2(0.1, -0.2),
            deltaActions: SIMD2(2, -0.5), deltaScale: 0.04)
        XCTAssertEqual(target.x, 0.14, accuracy: 1e-6)
        XCTAssertEqual(target.y, -0.22, accuracy: 1e-6)

        let task = try ArmPushTTask(configuration: ArmPushTTaskConfig(
            numEnvironments: 1, maxEpisodeSteps: 32, controlDecimation: 1,
            endEffectorDeltaActionScale: 0.04))
        XCTAssertEqual(task.spec.revision,
                       RLPhysicsContract.fixedGainActuatorV2(116))
        XCTAssertEqual(task.spec.action.name, "end_effector_delta_position")
        XCTAssertEqual(task.spec.configurationValues[
            "endEffectorDeltaActionScale"], 0.04)

        _ = try task.reset(seed: 31)
        let initial = task.environment.states()[0].tipPosition
        var result = RLStepBatch(spec: task.spec)
        let action = try RLActionBatch(
            numEnvironments: 1, actionDimension: 2,
            values: ContiguousArray([0, 1]))
        for _ in 0..<12 {
            try task.step(actions: action, into: &result)
        }
        let moved = task.environment.states()[0].tipPosition
        XCTAssertGreaterThan(moved.y, initial.y + 1e-3)
        XCTAssertTrue(result.observations.policy.allSatisfy(\.isFinite))
    }

    func testArmPushTCanContinueAfterSuccessWithoutRewardingAvoidance() throws {
        let legacy = try ArmPushTTask(configuration: ArmPushTTaskConfig(
            numEnvironments: 1, maxEpisodeSteps: 2, controlDecimation: 1,
            autoReset: false, blockSpawnGoalBlend: 1,
            blockSpawnRadius: 0, blockSpawnYawRange: 0))
        var legacyResult = RLStepBatch(spec: legacy.spec)
        try legacy.step(actions: RLActionBatch(spec: legacy.spec),
                        into: &legacyResult)
        XCTAssertTrue(legacyResult.terminated[0])
        XCTAssertTrue(legacyResult.successes[0])

        let continuing = try ArmPushTTask(configuration: ArmPushTTaskConfig(
            numEnvironments: 1, maxEpisodeSteps: 2, controlDecimation: 1,
            autoReset: false, endEffectorDeltaActionScale: 0.04,
            blockSpawnGoalBlend: 1, blockSpawnRadius: 0,
            blockSpawnYawRange: 0, continueAfterSuccess: true))
        XCTAssertEqual(continuing.spec.revision,
                       RLPhysicsContract.fixedGainActuatorV2(117))
        XCTAssertEqual(continuing.spec.configurationValues[
            "continueAfterSuccess"], 1)
        var result = RLStepBatch(spec: continuing.spec)
        let zero = RLActionBatch(spec: continuing.spec)
        try continuing.step(actions: zero, into: &result)
        XCTAssertFalse(result.terminated[0])
        XCTAssertFalse(result.truncated[0])
        XCTAssertTrue(result.successes[0],
                      "the transition signal remains instantaneous")
        XCTAssertEqual(result.rewards[0], 3, accuracy: 1e-5)

        try continuing.step(actions: zero, into: &result)
        XCTAssertFalse(result.terminated[0])
        XCTAssertTrue(result.truncated[0])
        XCTAssertTrue(result.successes[0])
        XCTAssertEqual(result.metrics["episode/success_once"]?[0], 1)
        XCTAssertEqual(result.metrics["episode/success_at_end"]?[0], 1)
        XCTAssertEqual(result.metrics["episode/length"]?[0], 2)
    }

    func testArmPushTPoseProgressRewardIsExplicitAndRevisioned() throws {
        let task = try ArmPushTTask(configuration: ArmPushTTaskConfig(
            numEnvironments: 2, maxEpisodeSteps: 4, controlDecimation: 1,
            autoReset: false, poseProgressRewardWeight: 100,
            poseRewardWeight: 0, reachingRewardWeight: 0,
            continueAfterSuccess: true))
        XCTAssertEqual(task.spec.revision,
                       RLPhysicsContract.fixedGainActuatorV2(118))
        XCTAssertEqual(task.spec.configurationValues[
            "poseProgressRewardWeight"], 100)
        var result = RLStepBatch(spec: task.spec)
        try task.step(actions: RLActionBatch(spec: task.spec), into: &result)
        XCTAssertTrue(try XCTUnwrap(result.metrics[
            "reward/pose_progress"]).allSatisfy(\.isFinite))
        XCTAssertTrue(try XCTUnwrap(result.metrics[
            "reward/pose_precision"]).allSatisfy { $0 == 0 })
    }

    func testArmPushTActionMagnitudeCostRewardsStoppingDeltaControl() throws {
        let task = try ArmPushTTask(configuration: ArmPushTTaskConfig(
            numEnvironments: 1, maxEpisodeSteps: 4, controlDecimation: 1,
            autoReset: false, actionMagnitudePenaltyWeight: 0.05))
        XCTAssertEqual(task.spec.revision,
                       RLPhysicsContract.fixedGainActuatorV2(119))
        XCTAssertEqual(task.spec.configurationValues[
            "actionMagnitudePenaltyWeight"], 0.05)
        var result = RLStepBatch(spec: task.spec)
        var action = RLActionBatch(spec: task.spec)
        action.values[0] = 2
        action.values[1] = -2
        try task.step(actions: action, into: &result)
        XCTAssertEqual(result.metrics["penalty/action_magnitude"]?[0]
            ?? .nan, -0.1, accuracy: 1e-6)
    }

    func testArmPushTPrecisionExpertGateIsGeometricAndRevisioned() throws {
        let task = try ArmPushTTask(configuration: ArmPushTTaskConfig(
            numEnvironments: 2, precisionGatedActor: true,
            precisionExpertGateCoverage: 0.75,
            freezeBasePolicyExpert: true))
        XCTAssertEqual(task.spec.revision,
                       RLPhysicsContract.fixedGainActuatorV2(120))
        XCTAssertTrue(task.usesPolicyExpertGate)
        XCTAssertTrue(task.freezesBasePolicyExpert)
        XCTAssertEqual(task.spec.configurationValues[
            "precisionGatedActor"], 1)
        XCTAssertEqual(task.spec.configurationValues[
            "precisionExpertGateCoverage"], 0.75)
        XCTAssertEqual(task.spec.configurationValues[
            "freezeBasePolicyExpert"], 1)

        var observations = ContiguousArray(
            repeating: Float(0), count: 2 * task.spec.observation.elementCount)
        // Row zero reconstructs an exactly aligned block and goal pose.
        observations[9] = 1
        // Row one has the same orientation but a distant goal delta.
        let second = task.spec.observation.elementCount
        observations[second + 9] = 1
        observations[second + 11] = 0.4 / 1.3
        XCTAssertEqual(task.policyExpertGates(observations), [1, 0])

        XCTAssertThrowsError(try ArmPushTTask(configuration: .init(
            numEnvironments: 1, freezeBasePolicyExpert: true)))
    }

    func testArmPushTPrecisionExpertGateUsesExplicitHysteresis() throws {
        let task = try ArmPushTTask(configuration: ArmPushTTaskConfig(
            numEnvironments: 1, precisionGatedActor: true,
            precisionExpertGateCoverage: 0.75,
            precisionExpertReleaseCoverage: 0.5,
            freezeBasePolicyExpert: true))
        XCTAssertEqual(task.spec.revision,
                       RLPhysicsContract.fixedGainActuatorV2(121))
        XCTAssertEqual(task.spec.configurationValues[
            "precisionExpertReleaseCoverage"], 0.5)

        let middleOffset = try XCTUnwrap((1...200).lazy
            .map { Float($0) * 0.001 }
            .first { offset in
                let coverage = ArmPushTEnv.coverage(
                    blockPosition: .zero, blockYaw: 0,
                    goalPosition: SIMD2(offset, 0), goalYaw: 0)
                return coverage >= 0.5 && coverage < 0.75
            })
        func observation(goalOffset: Float) -> ContiguousArray<Float> {
            var value = ContiguousArray(
                repeating: Float(0),
                count: task.spec.observation.elementCount)
            value[9] = 1
            value[11] = goalOffset / 1.3
            return value
        }
        XCTAssertEqual(task.policyExpertGates(
            observation(goalOffset: middleOffset)), [0])
        XCTAssertEqual(task.policyExpertGates(
            observation(goalOffset: 0)), [1])
        XCTAssertEqual(task.policyExpertGates(
            observation(goalOffset: middleOffset)), [1])
        XCTAssertEqual(task.policyExpertGates(
            observation(goalOffset: 1)), [0])
    }

    func testArmPushTDefaultMatchesTwentyHertzBenchmarkControl() throws {
        let task = try ArmPushTTask(configuration: ArmPushTTaskConfig(
            numEnvironments: 1, seed: 19))
        XCTAssertEqual(task.configuration.controlDecimation, 6)
        XCTAssertEqual(task.spec.simulationStep, 1 / 120, accuracy: 1e-7)
        XCTAssertEqual(
            task.spec.simulationStep * Float(task.spec.controlDecimation),
            1 / 20, accuracy: 1e-7)
        XCTAssertEqual(
            task.spec.simulationStep * Float(task.spec.controlDecimation)
                * Float(task.spec.maxEpisodeSteps),
            5, accuracy: 1e-6)
    }

    func testArmPushTFixedPDDeltaActionMovesThenBrakes() throws {
        let task = try ArmPushTTask(configuration: ArmPushTTaskConfig(
            numEnvironments: 1, seed: 23, maxEpisodeSteps: 100,
            controlDecimation: 4, autoReset: false,
            blockSpawnRadius: 0))
        _ = try task.reset(seed: 29)
        let initial = task.environment.states()[0].jointAngles[0]
        var result = RLStepBatch(spec: task.spec)
        var action = RLActionBatch(spec: task.spec)
        action.values[0] = 1
        try task.step(actions: action, into: &result)
        let driven = task.environment.states()[0].jointAngles[0]

        action.values[0] = 0
        try task.step(actions: action, into: &result)
        var previous = task.environment.states()[0].jointAngles[0]
        let firstCoastDelta = abs(previous - driven)
        var lastCoastDelta = firstCoastDelta
        for _ in 0..<12 {
            try task.step(actions: action, into: &result)
            let angle = task.environment.states()[0].jointAngles[0]
            lastCoastDelta = abs(angle - previous)
            previous = angle
        }

        XCTAssertGreaterThan(abs(driven - initial), 1e-4,
                             "a bounded joint-delta command must move the arm")
        XCTAssertLessThan(lastCoastDelta, max(firstCoastDelta * 0.25, 1e-5),
                          "fixed PD damping must arrest rather than coast")
        XCTAssertTrue(result.observations.policy.allSatisfy(\.isFinite))
    }

    func testArmPushTContactCurriculumRejectsMovingObjectAway() {
        XCTAssertTrue(ArmPushTTask.pushContactCurriculumSucceeded(
            pushContactDistance: 0.1, successDistance: 0.25,
            goalDistance: 1.0, initialGoalDistance: 1.0,
            maximumGoalRegression: 0))
        XCTAssertFalse(ArmPushTTask.pushContactCurriculumSucceeded(
            pushContactDistance: 0.1, successDistance: 0.25,
            goalDistance: 1.01, initialGoalDistance: 1.0,
            maximumGoalRegression: 0))
        XCTAssertTrue(ArmPushTTask.pushContactCurriculumSucceeded(
            pushContactDistance: 0.1, successDistance: 0.25,
            goalDistance: 1.01, initialGoalDistance: 1.0,
            maximumGoalRegression: 0.02))
        XCTAssertFalse(ArmPushTTask.pushContactCurriculumSucceeded(
            pushContactDistance: 0.1, successDistance: 0.25,
            goalDistance: 0.85, initialGoalDistance: 1.0,
            maximumGoalRegression: 0, minimumGoalProgress: 0.2))
        XCTAssertTrue(ArmPushTTask.pushContactCurriculumSucceeded(
            pushContactDistance: 0.1, successDistance: 0.25,
            goalDistance: 0.8, initialGoalDistance: 1.0,
            maximumGoalRegression: 0, minimumGoalProgress: 0.2))
        XCTAssertTrue(ArmPushTTask.pushContactCurriculumSucceeded(
            pushContactDistance: 0.1, successDistance: 0.25,
            goalDistance: 0.8, initialGoalDistance: 1.0,
            maximumGoalRegression: 0, minimumGoalProgress: 0.2,
            yawError: 0.19, maximumYawError: 0.2))
        XCTAssertFalse(ArmPushTTask.pushContactCurriculumSucceeded(
            pushContactDistance: 0.1, successDistance: 0.25,
            goalDistance: 0.8, initialGoalDistance: 1.0,
            maximumGoalRegression: 0, minimumGoalProgress: 0.2,
            yawError: 0.21, maximumYawError: 0.2))
        XCTAssertTrue(ArmPushTTask.pushContactCurriculumSucceeded(
            pushContactDistance: 0.1, successDistance: 0.25,
            goalDistance: 0.039, initialGoalDistance: 0.18,
            maximumGoalRegression: 0,
            maximumGoalDistance: 0.04))
        XCTAssertTrue(ArmPushTTask.pushContactCurriculumSucceeded(
            pushContactDistance: 0.1, successDistance: 0.25,
            goalDistance: 0.039, initialGoalDistance: 0.03,
            maximumGoalRegression: 0.01,
            maximumGoalDistance: 0.04))
        XCTAssertFalse(ArmPushTTask.pushContactCurriculumSucceeded(
            pushContactDistance: 0.1, successDistance: 0.25,
            goalDistance: 0.039, initialGoalDistance: 0.03,
            maximumGoalRegression: 0,
            maximumGoalDistance: 0.04))
        XCTAssertFalse(ArmPushTTask.pushContactCurriculumSucceeded(
            pushContactDistance: 0.1, successDistance: 0.25,
            goalDistance: 0.041, initialGoalDistance: 0.18,
            maximumGoalRegression: 0,
            maximumGoalDistance: 0.04))

        let absoluteTask = try? ArmPushTTask(configuration: ArmPushTTaskConfig(
            numEnvironments: 1,
            pushContactCurriculumMaximumGoalDistance: 0.04))
        XCTAssertEqual(absoluteTask?.spec.revision,
                       RLPhysicsContract.fixedGainActuatorV2(115))
        XCTAssertEqual(absoluteTask?.spec.configurationValues[
            "pushContactCurriculumMaximumGoalDistance"], 0.04)
    }

    func testSuccessImitationMarksOnlySuccessfulEpisodeSegment() {
        var mask = [Float](repeating: 0, count: 4 * 3)
        VectorPPOTrainer.markSuccessfulEpisodeSegment(
            mask: &mask, environment: 1, numEnvironments: 3,
            startStep: 1, endStep: 2)

        XCTAssertEqual(mask, [
            0, 0, 0,
            0, 1, 0,
            0, 1, 0,
            0, 0, 0,
        ])
    }

    func testPPORecoversTaskObservationSignedPermutation() throws {
        let permutation = try VectorPPOTrainer.signedPermutation(
            dimension: 3
        ) { values in
            var output = values
            for row in 0..<(values.count / 3) {
                let base = row * 3
                output[base] = -values[base + 1]
                output[base + 1] = values[base]
                output[base + 2] = values[base + 2]
            }
            return output
        }
        XCTAssertEqual(permutation.sources, [1, 0, 2])
        XCTAssertEqual(permutation.signs, [-1, 1, 1])
    }

    func testPPOMirroredExpertInitializationIsExactWithNormalization() throws {
        let source: [String: MLXArray] = [
            "actor1.weight": MLXArray([Float](
                [1, 2, 3, -2, 0.5, 4])).reshaped([2, 3]),
            "actor1.bias": MLXArray([Float](arrayLiteral: 0.25, -0.75)),
            "actor2.weight": MLXArray.eye(2),
            "actor2.bias": MLXArray.zeros([2]),
            "actor3.weight": MLXArray.eye(2),
            "actor3.bias": MLXArray.zeros([2]),
            "actorOutput.weight": MLXArray([Float](
                [1.5, -2, 0.75, 3])).reshaped([2, 2]),
            "actorOutput.bias": MLXArray([Float](arrayLiteral: 0.5, -1)),
        ]
        let normalizer = RunningNormalizerSnapshot(
            count: 100, mean: [1, -2, 0.5], variance: [4, 9, 16])
        let observationSources = [1, 0, 2]
        let observationSigns: [Float] = [-1, 1, 1]
        let actionSources = [1, 0]
        let actionSigns: [Float] = [-1, 1]
        let mirrored = try VectorActorCritic
            .initializingPolicyExpertAsMirroredBase(
                source, observationSources: observationSources,
                observationSigns: observationSigns,
                actionSources: actionSources, actionSigns: actionSigns,
                normalizer: normalizer, normalizesObservations: true)

        let raw: [Float] = [2.5, -0.5, -1]
        let mean = normalizer.mean.map(Float.init)
        let sigma = normalizer.variance.map { sqrt(Float($0)) }
        let normalized = zip(zip(raw, mean), sigma).map {
            ($0.0.0 - $0.0.1) / $0.1
        }
        let mirroredRaw = observationSources.indices.map {
            observationSigns[$0] * raw[observationSources[$0]]
        }
        let mirroredNormalized = zip(zip(mirroredRaw, mean), sigma).map {
            ($0.0.0 - $0.0.1) / $0.1
        }
        func affine(_ weight: [Float], _ bias: [Float], _ input: [Float],
                    rows: Int, columns: Int) -> [Float] {
            var output = bias
            for row in 0..<rows {
                for column in 0..<columns {
                    output[row] += weight[row * columns + column]
                        * input[column]
                }
            }
            return output
        }
        let baseFirst = affine(
            source["actor1.weight"]!.asArray(Float.self),
            source["actor1.bias"]!.asArray(Float.self),
            mirroredNormalized, rows: 2, columns: 3)
        let expertFirst = affine(
            mirrored["expertActor1.weight"]!.asArray(Float.self),
            mirrored["expertActor1.bias"]!.asArray(Float.self),
            normalized, rows: 2, columns: 3)
        for (actual, expected) in zip(expertFirst, baseFirst) {
            XCTAssertEqual(actual, expected, accuracy: 1e-5)
        }

        let hidden: [Float] = [0.2, -0.4]
        let baseOutput = affine(
            source["actorOutput.weight"]!.asArray(Float.self),
            source["actorOutput.bias"]!.asArray(Float.self),
            hidden, rows: 2, columns: 2)
        let expertOutput = affine(
            mirrored["expertActorOutput.weight"]!.asArray(Float.self),
            mirrored["expertActorOutput.bias"]!.asArray(Float.self),
            hidden, rows: 2, columns: 2)
        let expectedOutput = actionSources.indices.map {
            actionSigns[$0] * baseOutput[actionSources[$0]]
        }
        for (actual, expected) in zip(expertOutput, expectedOutput) {
            XCTAssertEqual(actual, expected, accuracy: 1e-6)
        }
    }

    func testArmPushTRandomizesTaskStateWithoutControllerFeatures() throws {
        let task = try ArmPushTTask(configuration: ArmPushTTaskConfig(
            numEnvironments: 1, seed: 13, maxEpisodeSteps: 8,
            controlDecimation: 1))
        _ = try task.reset(seed: 17)
        let first = task.environment.states()[0]
        _ = try task.reset(seed: 18)
        let second = task.environment.states()[0]
        XCTAssertNotEqual(first.blockPosition, second.blockPosition)
        XCTAssertNotEqual(first.blockYaw, second.blockYaw)
        XCTAssertEqual(task.spec.observation.shape, [19])
        XCTAssertFalse(task.evaluationCriteria.failures(
            successRate: 0, meanEpisodeLength: 400,
            maxEpisodeSteps: task.spec.maxEpisodeSteps,
            taskMetrics: [
                "episode/normalized_score": 0,
                "episode/goal_distance_m": 1,
                "episode/yaw_error_rad": 1,
            ]).isEmpty)
    }

    func testArmPushTPrecisionCurriculumResetIsExplicitAndReproducible() throws {
        let task = try ArmPushTTask(configuration: ArmPushTTaskConfig(
            numEnvironments: 1, seed: 13, maxEpisodeSteps: 8,
            controlDecimation: 1, blockSpawnGoalBlend: 1,
            blockSpawnRadius: 0, blockSpawnYawRange: 0))
        _ = try task.reset(seed: 17)
        let state = task.environment.states()[0]
        XCTAssertEqual(state.blockPosition.x,
                       task.environment.refs[0].goalPosition.x,
                       accuracy: 2e-4)
        XCTAssertEqual(state.blockPosition.y,
                       task.environment.refs[0].goalPosition.y,
                       accuracy: 2e-4)
        XCTAssertEqual(state.blockYaw, task.environment.refs[0].goalYaw,
                       accuracy: 2e-4)
        XCTAssertEqual(task.spec.configurationValues["blockSpawnGoalBlend"], 1)
        XCTAssertEqual(task.spec.configurationValues["blockSpawnRadius"], 0)
        XCTAssertEqual(task.spec.configurationValues["blockSpawnYawRange"], 0)
    }

    func testArmPushTLateralCurriculumSamplesRequestedHalfPlane() throws {
        let nominal = simd_normalize(ArmPushTEnv.goalPosition)
        let lateral = SIMD2<Float>(-nominal.y, nominal.x)

        for bias: Float in [1, -1] {
            let task = try ArmPushTTask(configuration: ArmPushTTaskConfig(
                numEnvironments: 32, seed: 23, maxEpisodeSteps: 8,
                controlDecimation: 1, blockSpawnLateralBias: bias))
            let states = task.environment.states()
            for environmentIndex in states.indices {
                let reference = task.environment.refs[environmentIndex]
                let center = reference.goalPosition - ArmPushTEnv.goalPosition
                let spawnOffset = states[environmentIndex].blockPosition - center
                XCTAssertGreaterThanOrEqual(
                    bias * simd_dot(spawnOffset, lateral), -1e-5)
            }
            XCTAssertEqual(task.spec.revision,
                           RLPhysicsContract.fixedGainActuatorV2(122))
            XCTAssertEqual(
                task.spec.configurationValues["blockSpawnLateralBias"], bias)
        }

        let canonical = try ArmPushTTask(configuration: ArmPushTTaskConfig(
            numEnvironments: 1, seed: 23, maxEpisodeSteps: 8,
            controlDecimation: 1))
        XCTAssertNil(canonical.spec.configurationValues["blockSpawnLateralBias"])
        XCTAssertNotEqual(canonical.spec.revision,
                          RLPhysicsContract.fixedGainActuatorV2(122))
    }

    func testArmPushTRepeatedResetReplaysIdenticalTrajectory() throws {
        let task = try ArmPushTTask(configuration: ArmPushTTaskConfig(
            numEnvironments: 1, seed: 13, maxEpisodeSteps: 64,
            controlDecimation: 4, autoReset: false))

        func rollout() throws -> [Float] {
            _ = try task.reset(seed: 19)
            var trace = [Float]()
            trace.reserveCapacity(24 * 22)
            var result = RLStepBatch(spec: task.spec)
            for step in 0..<24 {
                let phase = Float(step) * 0.17
                let action = try RLActionBatch(
                    numEnvironments: 1, actionDimension: 2,
                    values: ContiguousArray([0.35 * sin(phase),
                                             0.45 * cos(phase)]))
                try task.step(actions: action, into: &result)
                trace.append(contentsOf: result.observations.policy)
                trace.append(result.rewards[0])
                trace.append(result.terminated[0] ? 1 : 0)
                trace.append(result.truncated[0] ? 1 : 0)
            }
            return trace
        }

        let first = try rollout()
        let repeated = try rollout()
        XCTAssertEqual(first.count, repeated.count)
        for index in first.indices {
            guard abs(first[index] - repeated[index]) <= 2e-5 else {
                XCTFail("post-reset arm trajectory diverged at \(index): "
                    + "\(first[index]) != \(repeated[index])")
                break
            }
        }
    }

    func testArmPushTCanTransferReferenceFreeRevision3ObservationSchema()
        throws
    {
        let registered = try BuiltInRLTasks.registry.make(
            "arm-pusht-v0",
            configuration: RLTaskConfiguration(
                numEnvironments: 1, seed: 8))
        let task = try XCTUnwrap(registered as? ArmPushTTask)
        let mapping = try XCTUnwrap(task
            .initializationObservationSourceIndices(sourceDimension: 16))
        XCTAssertEqual(mapping.count, 19)
        XCTAssertEqual(Array(mapping[0..<6]), [0, 1, 2, 3, 4, 5])
        XCTAssertNil(mapping[6])
        XCTAssertNil(mapping[7])
        XCTAssertEqual(mapping[8], 6)
        XCTAssertEqual(mapping[9], 7)
        XCTAssertNil(mapping[10])
        XCTAssertEqual(Array(mapping[11..<19]),
                       [8, 9, 10, 11, 12, 13, 14, 15])
        XCTAssertNil(task.initializationObservationSourceIndices(
            sourceDimension: 21),
            "controller-derived legacy observations must remain rejected")
    }

    func testGAEStopsAcrossEpisodeBoundaries() {
        let uninterrupted = GeneralizedAdvantageEstimator.compute(
            rewards: [1, 1], values: [0, 0], dones: [false, false],
            lastValues: [0], numEnvironments: 1, horizon: 2,
            gamma: 1, lambda: 1)
        XCTAssertEqual(uninterrupted.advantages[0], 2, accuracy: 1e-6)
        XCTAssertEqual(uninterrupted.advantages[1], 1, accuracy: 1e-6)

        let terminated = GeneralizedAdvantageEstimator.compute(
            rewards: [1, 1], values: [0, 0], dones: [true, false],
            lastValues: [0], numEnvironments: 1, horizon: 2,
            gamma: 1, lambda: 1)
        XCTAssertEqual(terminated.advantages, [1, 1])

        let timeoutCorrected = GeneralizedAdvantageEstimator.compute(
            rewards: [1 + 0.9 * 5], values: [0], dones: [true],
            lastValues: [0], numEnvironments: 1, horizon: 1,
            gamma: 0.9, lambda: 0.95)
        XCTAssertEqual(timeoutCorrected.returns[0], 5.5, accuracy: 1e-6)
    }

    func testPPOImportanceRatioRemainsFiniteForExtremeLogProbabilityShift() {
        let results = [Float(1_000), -1_000, 0].map {
            VectorPPOTrainer.stableImportanceRatio(
                logProbabilityDifference: $0)
        }
        let ratios = results.map(\.ratio)
        let logs = results.map(\.logRatio)
        XCTAssertTrue(ratios.allSatisfy(\.isFinite))
        XCTAssertEqual(logs, [20, -20, 0])
        XCTAssertEqual(ratios[0], exp(20), accuracy: exp(20) * 1e-6)
        XCTAssertEqual(ratios[1], exp(-20), accuracy: 1e-12)
        XCTAssertEqual(ratios[2], 1, accuracy: 1e-6)
    }

    func testPPOSymmetryRowsCannotCollapseKLController() {
        let originalDifference: Float = 0.1
        let expected = exp(originalDifference) - 1 - originalDifference
        let measured = VectorPPOTrainer.weightedApproximateKL(
            logProbabilityDifferences: [originalDifference, 20],
            weights: [1, 0])
        XCTAssertEqual(measured, expected, accuracy: 1e-6)
        XCTAssertEqual(VectorPPOTrainer.weightedApproximateKL(
            logProbabilityDifferences: [20], weights: [0]), 0)
    }

    func testPPOKLStrategiesMatchReferenceSemantics() {
        XCTAssertTrue(VectorPPOTrainer.shouldStopForKL(
            minibatchKL: 0.100_001, targetKL: 0.1,
            schedule: .earlyStop))
        XCTAssertFalse(VectorPPOTrainer.shouldStopForKL(
            minibatchKL: 0.100_001, targetKL: 0.1,
            schedule: .adaptive))
        XCTAssertTrue(VectorPPOTrainer.shouldStopForKL(
            minibatchKL: 0.400_001, targetKL: 0.1,
            schedule: .adaptive))
        XCTAssertFalse(VectorPPOTrainer.shouldStopForKL(
            minibatchKL: 1, targetKL: 0.1, schedule: .none))
        XCTAssertFalse(VectorPPOTrainer.shouldStopForKL(
            minibatchKL: 1, targetKL: 0, schedule: .earlyStop))

        var historical = VectorPPOConfig()
        XCTAssertEqual(historical.resolvedKLSchedule, .adaptive)
        historical.klSchedule = .earlyStop
        XCTAssertEqual(historical.resolvedKLSchedule, .earlyStop)
    }

    func testPPOFrozenActorRowsCannotDiluteRoutedExpertUpdates() {
        let weights = VectorPPOTrainer.actorTrainingWeights(
            expertGates: [0, 1, 0, 1],
            standExpertGates: [0, 0, 1, 0],
            freezesBaseActor: true, freezesExpertActor: false)
        XCTAssertEqual(weights, [0, 1, 1, 1])
        XCTAssertEqual(VectorPPOTrainer.actorTrainingWeights(
            expertGates: [0, 1, 0], standExpertGates: [0, 0, 1],
            freezesBaseActor: true, freezesExpertActor: true), [0, 0, 1])

        let normalized = VectorPPOTrainer.weightedNormalizedAdvantages(
            [1, 3, 100], weights: [1, 1, 0])
        XCTAssertEqual(normalized.mean, 2, accuracy: 1e-6)
        XCTAssertEqual(normalized.variance, 1, accuracy: 1e-6)
        XCTAssertEqual(normalized.values[0], -1, accuracy: 1e-6)
        XCTAssertEqual(normalized.values[1], 1, accuracy: 1e-6)
        let noTrainableRows = VectorPPOTrainer.weightedNormalizedAdvantages(
            [1, 2], weights: [0, 0])
        XCTAssertEqual(noTrainableRows.values, [0, 0])
        XCTAssertEqual(noTrainableRows.scale, 0)
    }

    func testPPOExplorationBoundsRejectInvalidSchedules() throws {
        var fixed = VectorPPOConfig(updates: 1, rolloutSteps: 4,
                                    minibatchSize: 4,
                                    minimumActionStd: 0.5,
                                    maximumActionStd: 0.5)
        XCTAssertEqual(fixed.resolvedRewardScale, 1)
        try fixed.validate(batchSize: 4)
        fixed.rewardScale = 1 / 3
        XCTAssertEqual(fixed.resolvedRewardScale, 1 / 3, accuracy: 1e-6)
        try fixed.validate(batchSize: 4)
        fixed.minimumActionStd = 0.6
        XCTAssertThrowsError(try fixed.validate(batchSize: 4))
        fixed.minimumActionStd = nil
        fixed.maximumActionStd = 0
        XCTAssertThrowsError(try fixed.validate(batchSize: 4))
        fixed.maximumActionStd = nil
        fixed.rewardScale = 0
        XCTAssertThrowsError(try fixed.validate(batchSize: 4))
    }

    func testPPOAcceptsExactIsaacFlatNetworkAndGaussianPolicy() throws {
        var configuration = VectorPPOConfig(
            updates: 1, rolloutSteps: 4, minibatchSize: 4,
            hiddenDimensions: [128, 128, 128],
            actionDistribution: .gaussian)
        try configuration.validate(batchSize: 4)
        XCTAssertEqual(configuration.resolvedHiddenDimensions, [128, 128, 128])
        XCTAssertEqual(configuration.resolvedActionDistribution, .gaussian)

        configuration.hiddenDimensions = [128, 128]
        XCTAssertThrowsError(try configuration.validate(batchSize: 4))
        configuration.hiddenDimensions = [128, 0, 128]
        XCTAssertThrowsError(try configuration.validate(batchSize: 4))
    }

    func testPPOMirrorLossRejectsInvalidCoefficients() throws {
        var configuration = VectorPPOConfig(
            updates: 1, rolloutSteps: 4, minibatchSize: 4,
            symmetryMirrorLossCoefficient: 0.01)
        try configuration.validate(batchSize: 4)
        configuration.symmetryMirrorLossCoefficient = -0.01
        XCTAssertThrowsError(try configuration.validate(batchSize: 4))
        configuration.symmetryMirrorLossCoefficient = .infinity
        XCTAssertThrowsError(try configuration.validate(batchSize: 4))

        configuration.symmetryMirrorLossCoefficient = 0.01
        configuration.referencePolicyCoefficient = -0.01
        XCTAssertThrowsError(try configuration.validate(batchSize: 4))
        configuration.referencePolicyCoefficient = .infinity
        XCTAssertThrowsError(try configuration.validate(batchSize: 4))
    }

    func testPPOResumeAllowsExtensionButRejectsOptimizerMutation() {
        let checkpoint = VectorPPOConfig(
            updates: 50, rolloutSteps: 24, updateEpochs: 5,
            minibatchSize: 3_072, learningRate: 1e-5,
            entropyCoefficient: 0.001, targetKL: 0.01,
            initialActionStd: 0.5,
            initializationCheckpoint: "/an/earlier/policy",
            useTaskSymmetryAugmentation: false,
            symmetryMirrorLossCoefficient: 0.01,
            checkpointInterval: 50, seed: 3_102)
        var extensionOnly = checkpoint
        extensionOnly.updates = 200
        extensionOnly.checkpointInterval = 25
        extensionOnly.initializationCheckpoint = nil
        XCTAssertTrue(extensionOnly.resumeIncompatibilities(
            with: checkpoint).isEmpty)

        extensionOnly.learningRate = 5e-6
        extensionOnly.useTaskSymmetryAugmentation = true
        XCTAssertEqual(Set(extensionOnly.resumeIncompatibilities(
            with: checkpoint)), ["learningRate", "useTaskSymmetryAugmentation"])

        var changedMirrorLoss = checkpoint
        changedMirrorLoss.symmetryMirrorLossCoefficient = 0.02
        XCTAssertEqual(changedMirrorLoss.resumeIncompatibilities(
            with: checkpoint), ["symmetryMirrorLossCoefficient"])

        var changedReferenceRetention = checkpoint
        changedReferenceRetention.referencePolicyCoefficient = 0.5
        XCTAssertEqual(changedReferenceRetention.resumeIncompatibilities(
            with: checkpoint), ["referencePolicyCoefficient"])

        var frozenNormalizer = checkpoint
        frozenNormalizer.updateObservationNormalizer = false
        XCTAssertEqual(frozenNormalizer.resumeIncompatibilities(
            with: checkpoint), ["updateObservationNormalizer"])

        var rescaledRewards = checkpoint
        rescaledRewards.rewardScale = 1 / 3
        XCTAssertEqual(rescaledRewards.resumeIncompatibilities(
            with: checkpoint), ["rewardScale"])

        XCTAssertEqual(checkpoint.resolvedOptimizerEpsilon, 1e-8)
        var changedAdam = checkpoint
        changedAdam.optimizerEpsilon = 1e-5
        XCTAssertEqual(changedAdam.resumeIncompatibilities(
            with: checkpoint), ["optimizerEpsilon"])

        var changedArchitecture = checkpoint
        changedArchitecture.hiddenDimensions = [128, 128, 128]
        changedArchitecture.actionDistribution = .gaussian
        XCTAssertEqual(Set(changedArchitecture.resumeIncompatibilities(
            with: checkpoint)), ["hiddenDimensions", "actionDistribution"])

        var changedKLSemantics = checkpoint
        changedKLSemantics.klSchedule = .earlyStop
        XCTAssertEqual(changedKLSemantics.resumeIncompatibilities(
            with: checkpoint), ["klSchedule"])
    }

    func testPPOResumePreservesInitializationProvenance() {
        let checkpoint = VectorPPOConfig(
            updates: 50,
            initializationCheckpoint: "/policies/locomotion-parent",
            policyExpertInitializationCheckpoint: "/policies/recovery-parent",
            standExpertInitializationCheckpoint: "/policies/stand-parent",
            checkpointInterval: 10)
        var resumedCommand = checkpoint
        resumedCommand.updates = 100
        resumedCommand.initializationCheckpoint = nil
        resumedCommand.policyExpertInitializationCheckpoint = nil
        resumedCommand.standExpertInitializationCheckpoint = nil

        let persisted = resumedCommand
            .preservingInitializationProvenance(from: checkpoint)
        XCTAssertEqual(persisted.updates, 100)
        XCTAssertEqual(persisted.initializationCheckpoint,
                       "/policies/locomotion-parent")
        XCTAssertEqual(persisted.policyExpertInitializationCheckpoint,
                       "/policies/recovery-parent")
        XCTAssertEqual(persisted.standExpertInitializationCheckpoint,
                       "/policies/stand-parent")
    }

    func testH1ReferenceRetentionRelaxesOnlyAfterPhysicalImpact() throws {
        let registered = try BuiltInRLTasks.registry.make(
            "humanoid-isaac-goal-v0",
            configuration: RLTaskConfiguration(
                numEnvironments: 1, seed: 24_350, autoReset: false,
                options: [
                    "maxEpisodeSteps": 40,
                    "commandResamplingSteps": 1,
                    "observationNoise": 0,
                    "initialYawRange": 0,
                    "minimumGoalDistance": 4,
                    "maximumGoalDistance": 4,
                    "projectileProbability": 1,
                    "recoveryGatedActor": 1,
                    "freezeBasePolicyExpert": 1,
                    "recoveryContextObservations": 1,
                    "recoveryContextDuration": 2,
                    "recoveryExpertSide": -1,
                    "postImpactUprightRewardWeight": 2,
                    "postImpactAngularVelocityPenaltyWeight": 0.2,
                    "postImpactFallPenalty": 1_000,
                    "minimumProjectileSpeed": 4,
                    "maximumProjectileSpeed": 4,
                    "projectileLeftProbability": 0.8,
                    "minimumProjectileLaunchStep": 39,
                    "maximumProjectileLaunchStep": 39,
                ]))
        let task = try XCTUnwrap(registered as? HumanoidIsaacVelocityTask)
        XCTAssertEqual(task.spec.revision,
                       RLPhysicsContract.fixedGainActuatorV2(7))
        XCTAssertEqual(task.spec.observation.elementCount, 73)
        XCTAssertEqual(task.spec.configurationValues[
            "postImpactUprightRewardWeight"], 2)
        XCTAssertEqual(task.spec.configurationValues[
            "postImpactAngularVelocityPenaltyWeight"], 0.2)
        XCTAssertEqual(task.spec.configurationValues["postImpactFallPenalty"],
                       1_000)
        XCTAssertEqual(task.spec.configurationValues[
            "projectileLeftProbability"], 0.8)
        XCTAssertEqual(task.spec.configurationValues["recoveryExpertSide"], -1)
        XCTAssertTrue(task.usesPolicyExpertGate)
        XCTAssertTrue(task.freezesBasePolicyExpert)
        XCTAssertTrue(task.initializesPolicyExpertFromBaseOnTransfer)
        var preserveExpertConfiguration = task.configuration
        preserveExpertConfiguration.initializeRecoveryExpertFromBaseOnTransfer =
            false
        let preserveExpertTask = try HumanoidIsaacVelocityTask(
            configuration: preserveExpertConfiguration,
            taskID: "humanoid-isaac-goal-v0")
        XCTAssertFalse(
            preserveExpertTask.initializesPolicyExpertFromBaseOnTransfer)
        XCTAssertEqual(preserveExpertTask.spec.configurationValues[
            "initializeRecoveryExpertFromBaseOnTransfer"], 0)
        var mirroredExpertConfiguration = task.configuration
        mirroredExpertConfiguration.initializeRecoveryExpertFromBaseOnTransfer =
            false
        mirroredExpertConfiguration
            .initializeRecoveryExpertFromMirroredBaseOnTransfer = true
        let mirroredExpertTask = try HumanoidIsaacVelocityTask(
            configuration: mirroredExpertConfiguration,
            taskID: "humanoid-isaac-goal-v0")
        XCTAssertTrue(mirroredExpertTask
            .initializesPolicyExpertFromMirroredBaseOnTransfer)
        XCTAssertEqual(mirroredExpertTask.spec.configurationValues[
            "initializeRecoveryExpertFromMirroredBaseOnTransfer"], 1)
        var blendedExpertConfiguration = mirroredExpertConfiguration
        blendedExpertConfiguration.recoveryExpertGatePeak = 0.25
        blendedExpertConfiguration.recoveryExpertGateDecay = true
        let blendedExpertTask = try HumanoidIsaacVelocityTask(
            configuration: blendedExpertConfiguration,
            taskID: "humanoid-isaac-goal-v0")
        XCTAssertEqual(blendedExpertTask.spec.configurationValues[
            "recoveryExpertGatePeak"], 0.25)
        XCTAssertEqual(blendedExpertTask.spec.configurationValues[
            "recoveryExpertGateDecay"], 1)
        var blendedObservation = try blendedExpertTask.reset(seed: 24_350)
        blendedObservation.policy[71] = -1
        blendedObservation.policy[72] = 0.4
        XCTAssertEqual(blendedExpertTask.policyExpertGates(
            blendedObservation.policy)[0], 0.15, accuracy: 1e-6)
        var observation = try task.reset(seed: 24_351)
        XCTAssertEqual(observation.policy.count, 73)
        XCTAssertEqual(observation.policy[71], 0)
        XCTAssertEqual(observation.policy[72], 0)
        let mapping = try XCTUnwrap(task
            .initializationObservationSourceIndices(sourceDimension: 71))
        XCTAssertEqual(mapping.count, 73)
        XCTAssertEqual(mapping[70], 70)
        XCTAssertNil(mapping[71])
        XCTAssertNil(mapping[72])
        XCTAssertEqual(task.policyReferenceRegularizationWeights(
            observation.policy), [1])
        XCTAssertEqual(task.policyExpertGates(observation.policy), [0])

        let ref = task.environment.refs[0]
        let hand = task.environment.solver.bodyStates([ref.leftHand])[0]
        task.environment.throwProjectiles(
            environmentIDs: [0],
            positions: [hand.position + F3(0, 0.50, 0)],
            velocities: [F3(0, -6, 0)], angularVelocities: [.zero])
        let zero = RLActionBatch(spec: task.spec)
        var result = RLStepBatch(spec: task.spec)
        var impacted = false
        for _ in 0..<12 where !impacted {
            try task.step(actions: zero, into: &result)
            impacted = result.metrics["state/projectile_robot_contact"]?[0] == 1
            observation = result.observations
        }
        XCTAssertTrue(impacted)
        XCTAssertEqual(task.policyReferenceRegularizationWeights(
            observation.policy), [0])
        let expectedExpertGate: Float = observation.policy[71] < 0 ? 1 : 0
        XCTAssertEqual(task.policyExpertGates(observation.policy),
                       [expectedExpertGate])
        XCTAssertNotEqual(result.metrics["reward/post_impact_recovery"]?[0], 0)
        XCTAssertEqual(abs(observation.policy[71]), 1)
        XCTAssertGreaterThan(observation.policy[72], 0)
        let mirrored = task.mirrorPolicyObservations(observation.policy)
        XCTAssertEqual(mirrored[71], -observation.policy[71])
        XCTAssertEqual(mirrored[72], observation.policy[72])
        XCTAssertEqual(task.policyExpertGates(mirrored),
                       [1 - expectedExpertGate])
        XCTAssertEqual(task.mirrorPolicyObservations(mirrored),
                       observation.policy)

        observation = try task.reset(seed: 24_352)
        XCTAssertEqual(observation.policy[71], 0)
        XCTAssertEqual(observation.policy[72], 0)
        XCTAssertEqual(task.policyReferenceRegularizationWeights(
            observation.policy), [1])
        XCTAssertEqual(task.policyExpertGates(observation.policy), [0])
    }

    func testH1FullPolicySerializesPostImpactRewardContract() throws {
        let registered = try BuiltInRLTasks.registry.make(
            "humanoid-isaac-goal-v0",
            configuration: RLTaskConfiguration(
                numEnvironments: 1, seed: 24_352, autoReset: false,
                options: [
                    "maxEpisodeSteps": 40,
                    "commandResamplingSteps": 1,
                    "observationNoise": 0,
                    "initialYawRange": 0,
                    "minimumGoalDistance": 4,
                    "maximumGoalDistance": 4,
                    "projectileProbability": 1,
                    "projectileLeftProbability": 0.75,
                    "minimumProjectileSpeed": 4,
                    "maximumProjectileSpeed": 4,
                    "minimumProjectileLaunchStep": 20,
                    "maximumProjectileLaunchStep": 20,
                    "postImpactUprightRewardWeight": 0.5,
                    "postImpactAngularVelocityPenaltyWeight": 0.05,
                    "postImpactFallPenalty": 50,
                ]))
        let task = try XCTUnwrap(registered as? HumanoidIsaacVelocityTask)
        XCTAssertFalse(task.usesPolicyExpertGate)
        XCTAssertEqual(task.spec.configurationValues[
            "projectileLeftProbability"], 0.75)
        XCTAssertEqual(task.spec.configurationValues[
            "postImpactUprightRewardWeight"], 0.5)
        XCTAssertEqual(task.spec.configurationValues[
            "postImpactAngularVelocityPenaltyWeight"], 0.05)
        XCTAssertEqual(task.spec.configurationValues[
            "postImpactFallPenalty"], 50)
    }

    func testLiveCheckpointDiscoveryIgnoresPartialAndIncompatibleSnapshots()
        throws {
        let manager = FileManager.default
        let run = manager.temporaryDirectory.appendingPathComponent(
            "avbd-live-checkpoints-\(UUID().uuidString)", isDirectory: true)
        defer { try? manager.removeItem(at: run) }

        func write(update: Int, task: String = "humanoid-walk-v0",
                   revision: Int = 31, completedUpdates: Int? = nil,
                   complete: Bool = true) throws {
            let directory = run.appendingPathComponent(
                String(format: "checkpoints/update-%06d", update),
                isDirectory: true)
            try manager.createDirectory(at: directory,
                                        withIntermediateDirectories: true)
            try Data([1]).write(to: directory
                .appendingPathComponent("policy.safetensors"))
            try Data([2]).write(to: directory
                .appendingPathComponent("optimizer.safetensors"))
            let metadata = "{\"task\":\"\(task)\",\"taskRevision\":\(revision)}"
            try Data(metadata.utf8).write(to: directory
                .appendingPathComponent("metadata.json"))
            if complete {
                let state = "{\"completedUpdates\":\(completedUpdates ?? update),"
                    + "\"environmentSteps\":\(update * 1000)}"
                try Data(state.utf8).write(to: directory
                    .appendingPathComponent("training-state.json"))
            }
        }

        try write(update: 50)
        try write(update: 100, complete: false)
        try write(update: 150, task: "arm-pusht-v0")
        try write(update: 200, completedUpdates: 199)
        let first = try XCTUnwrap(
            VectorPolicyCheckpointDiscovery.latestCompleteCheckpoint(
            inRunDirectory: run.path, task: "humanoid-walk-v0",
            taskRevision: 31))
        XCTAssertEqual(URL(fileURLWithPath: first.directory).lastPathComponent,
                       "update-000050")
        XCTAssertEqual(first.completedUpdates, 50)
        XCTAssertEqual(first.environmentSteps, 50_000)

        try write(update: 250)
        XCTAssertEqual(VectorPolicyCheckpointDiscovery.latestCompleteCheckpoint(
            inRunDirectory: run.path, task: "humanoid-walk-v0",
            taskRevision: 31)?.completedUpdates, 250)
    }

    func testResumeReconcilesMetricsLogToDurableCheckpoint() throws {
        let manager = FileManager.default
        let directory = manager.temporaryDirectory.appendingPathComponent(
            "avbd-metrics-resume-\(UUID().uuidString)", isDirectory: true)
        defer { try? manager.removeItem(at: directory) }
        try manager.createDirectory(at: directory,
                                    withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("metrics.jsonl")
        let original = [
            "{\"update\":0,\"marker\":\"durable-a\"}",
            "{\"update\":1,\"marker\":\"durable-b\"}",
            "{\"update\":2,\"marker\":\"uncommitted\"}",
            "not-a-complete-json-row",
            "{\"update\":1,\"marker\":\"duplicate\"}",
        ].joined(separator: "\n")
        try Data(original.utf8).write(to: url)

        try VectorPPOTrainer.reconcileMetricsLog(
            at: url, completedUpdates: 2)

        let lines = try String(contentsOf: url, encoding: .utf8)
            .split(separator: "\n").map(String.init)
        XCTAssertEqual(lines, [
            "{\"update\":0,\"marker\":\"durable-a\"}",
            "{\"update\":1,\"marker\":\"durable-b\"}",
        ])

        let missing = directory.appendingPathComponent("missing.jsonl")
        try VectorPPOTrainer.reconcileMetricsLog(
            at: missing, completedUpdates: 0)
        XCTAssertEqual(try Data(contentsOf: missing), Data())
    }

    func testRunningObservationNormalization() {
        var normalizer = RunningObservationNormalizer(dimension: 2)
        normalizer.update(ContiguousArray([1, 2, 3, 4]), rows: 2)
        let normalized = normalizer.normalize(ContiguousArray([1, 2, 3, 4]))
        XCTAssertEqual(normalizer.snapshot.mean, [2, 3])
        XCTAssertEqual(normalized[0], -0.70710677, accuracy: 1e-5)
        XCTAssertEqual(normalized[2], 0.70710677, accuracy: 1e-5)
    }

    func testTransferredNormalizerCanUseFinitePriorStrength() throws {
        let snapshot = RunningNormalizerSnapshot(
            count: 29_626_368, mean: [0.99, -0.04],
            variance: [0.001, 0.018])
        let limited = snapshot.limitingPriorCount(to: 12_288)
        XCTAssertEqual(limited.count, 12_288)
        XCTAssertEqual(limited.mean, snapshot.mean)
        XCTAssertEqual(limited.variance, snapshot.variance)
        XCTAssertEqual(snapshot.limitingPriorCount(to: nil).count,
                       snapshot.count)
        let widened = try snapshot.applyingVarianceFloors([0: 0.25])
        XCTAssertEqual(widened.variance, [0.25, 0.018])
        XCTAssertEqual(widened.mean, snapshot.mean)
        XCTAssertThrowsError(try snapshot.applyingVarianceFloors([2: 0.25]))
        let remapped = try snapshot.remappingObservationChannels(
            sourceIndices: [1, nil, 0])
        XCTAssertEqual(remapped.count, snapshot.count)
        XCTAssertEqual(remapped.mean, [-0.04, 0, 0.99])
        XCTAssertEqual(remapped.variance, [0.018, 1, 0.001])
        XCTAssertThrowsError(try snapshot.remappingObservationChannels(
            sourceIndices: [2]))

        var invalid = VectorPPOConfig(updates: 1, rolloutSteps: 4,
                                      minibatchSize: 4)
        invalid.initializationNormalizerPriorCount = 1
        XCTAssertThrowsError(try invalid.validate(batchSize: 4))
    }

    func testGoalTransferWidensNavigationAndZeroSpeedHistoryChannels() throws {
        let task = try HumanoidWalkTask(configuration: .init(
            numEnvironments: 1, maximumGoalDirectionAngle: .pi / 2),
            taskID: "humanoid-goal-v0", taskRevision: 4)
        let floors = task.initializationObservationVarianceFloors
        XCTAssertEqual(floors.count, 36)
        XCTAssertEqual(floors[2], 0.25)
        XCTAssertEqual(floors[7], 0.25)
        XCTAssertEqual(floors[8], 0.25)
        XCTAssertEqual(floors[72], 0.25)
        XCTAssertEqual(floors[77], 0.25)
        XCTAssertEqual(floors[78], 0.25)
        XCTAssertEqual(floors[50], 0.09)
        XCTAssertEqual(floors[120], 0.09)
        XCTAssertNil(floors[6])
        XCTAssertNil(floors[9])

        let inheritedRangeTask = try HumanoidWalkTask(configuration: .init(
            numEnvironments: 1, maximumGoalDirectionAngle: .pi / 4),
            taskID: "humanoid-goal-v0", taskRevision: 4)
        XCTAssertEqual(inheritedRangeTask
            .initializationObservationVarianceFloors.count, 36)

        let wideSpeedTask = try HumanoidWalkTask(configuration: .init(
            numEnvironments: 1, minimumCommandSpeed: 0,
            maximumCommandSpeed: 0.65))
        XCTAssertEqual(wideSpeedTask
            .initializationObservationVarianceFloors.count, 9)
        XCTAssertEqual(wideSpeedTask
            .initializationObservationVarianceFloors[50], 0.09)

        let appendedTask = try HumanoidWalkTask(configuration: .init(
            numEnvironments: 1, maximumGoalDirectionAngle: .pi / 2,
            goalObservationIncludesLateralVelocity: true),
            taskID: "humanoid-goal-v0", taskRevision: 4)
        let appendedFloors = appendedTask
            .initializationObservationVarianceFloors
        XCTAssertEqual(appendedFloors.count, 45)
        XCTAssertEqual(appendedFloors[630], 0.04)
        XCTAssertEqual(appendedFloors[638], 0.04)
    }

    func testEvaluationEpisodeQuotasDoNotFavorEarlyTerminations() throws {
        XCTAssertEqual(VectorPPOTrainer.evaluationEpisodeQuotas(
            requestedEpisodes: 10, numEnvironments: 4), [3, 3, 2, 2])
        XCTAssertEqual(VectorPPOTrainer.evaluationEpisodeQuotas(
            requestedEpisodes: 4, numEnvironments: 4), [1, 1, 1, 1])
        XCTAssertEqual(VectorPPOTrainer.evaluationEpisodeQuotas(
            requestedEpisodes: 2, numEnvironments: 4), [1, 1, 0, 0])
        XCTAssertEqual(VectorPPOTrainer.evaluationEpisodeSeed(
            base: 123, episodeIndex: 0), 123)
        XCTAssertNotEqual(VectorPPOTrainer.evaluationEpisodeSeed(
            base: 123, episodeIndex: 1), 123)

        let task = try BuiltInRLTasks.registry.make(
            "humanoid-walk-v0",
            configuration: RLTaskConfiguration(
                numEnvironments: 2, seed: 5, autoReset: false))
        XCTAssertFalse(task.spec.autoReset)
    }

    func testActionBatchRejectsWrongShape() {
        XCTAssertThrowsError(try RLActionBatch(
            numEnvironments: 4, actionDimension: 2,
            values: ContiguousArray(repeating: 0, count: 7)))
    }

    func testActionBatchRejectsNonFinitePolicyOutput() throws {
        let task = try BuiltInRLTasks.registry.make(
            "pusht-state-v0",
            configuration: RLTaskConfiguration(numEnvironments: 2, seed: 1))
        var action = RLActionBatch(spec: task.spec)
        action.values[1] = .nan
        var result = RLStepBatch(spec: task.spec)
        XCTAssertThrowsError(try task.step(actions: action, into: &result))
    }

    func testEvaluationAggregateUsesDistinctSeedsAndSeedLevelQuartiles() throws {
        let rates: [Float] = [0.80, 0.85, 0.90, 0.95, 1.00]
        var reports = [PPOEvaluationMetrics]()
        for index in 0..<5 {
            let report = PPOEvaluationMetrics(
                provenanceVersion: 2,
                task: "humanoid-walk-v0", taskRevision: 33,
                checkpointDirectory: "run-\(index)",
                checkpointFingerprint: "fingerprint-\(index)",
                trainingSeed: UInt64(index + 1),
                evaluationSeed: UInt64(10_001 + index),
                evaluationEnvironments: 512,
                trainingUpdates: 3_000,
                trainingEnvironmentSteps: 100_000_000 + index,
                episodes: 512,
                successes: 410 + index * 10,
                successRate: rates[index],
                meanReturn: Float(index), meanEpisodeLength: 600,
                taskMetrics: ["episode/forward_distance_m": Float(index + 4)],
                acceptance: PPOEvaluationAcceptance(passed: true, failures: []))
            reports.append(report)
        }
        let aggregate = try PPOEvaluationAggregate.make(reports)
        XCTAssertEqual(aggregate.totalEpisodes, 2_560)
        XCTAssertEqual(aggregate.acceptedRuns, 5)
        XCTAssertTrue(aggregate.allRunsPassed)
        XCTAssertTrue(aggregate.hasRequiredRunCount)
        XCTAssertTrue(aggregate.allRunsHaveRequiredEpisodes)
        XCTAssertTrue(aggregate.provenanceComplete)
        XCTAssertTrue(aggregate.allRunsFromScratch)
        XCTAssertTrue(aggregate.publishable)
        XCTAssertEqual(aggregate.successRate.median, 0.90, accuracy: 1e-6)
        XCTAssertEqual(aggregate.successRate.firstQuartile, 0.85, accuracy: 1e-6)
        XCTAssertEqual(aggregate.successRate.thirdQuartile, 0.95, accuracy: 1e-6)
        XCTAssertEqual(aggregate.trainingUpdates.median, 3_000, accuracy: 1e-6)
        XCTAssertGreaterThanOrEqual(
            aggregate.trainingEnvironmentSteps.minimum, 100_000_000)
        XCTAssertEqual(
            aggregate.taskMetrics["episode/forward_distance_m"]!.median,
            6, accuracy: 1e-6)

        var duplicate = reports
        duplicate[4].trainingSeed = duplicate[0].trainingSeed
        XCTAssertThrowsError(try PPOEvaluationAggregate.make(duplicate))
        let incomplete = try PPOEvaluationAggregate.make(Array(reports.prefix(4)))
        XCTAssertFalse(incomplete.hasRequiredRunCount)
        XCTAssertFalse(incomplete.publishable)
        var transferred = reports
        transferred[0].initializationCheckpoint = "shared-parent"
        let transferAggregate = try PPOEvaluationAggregate.make(transferred)
        XCTAssertFalse(transferAggregate.allRunsFromScratch)
        XCTAssertFalse(transferAggregate.publishable)
    }

    func testCheckpointAggregateSeparatesEvaluationFromTrainingSeeds() throws {
        var reports = [PPOEvaluationMetrics]()
        for index in 0..<4 {
            reports.append(PPOEvaluationMetrics(
                provenanceVersion: 2,
                task: "humanoid-walk-v0", taskRevision: 33,
                checkpointDirectory: "run/checkpoint",
                checkpointFingerprint: "immutable-policy-a",
                trainingSeed: 77, evaluationSeed: UInt64(200 + index),
                evaluationEnvironments: 512,
                trainingUpdates: 50, trainingEnvironmentSteps: 614_400,
                episodes: 512, successes: 460 + index,
                successRate: Float(460 + index) / 512,
                meanReturn: 48 + Float(index), meanEpisodeLength: 980,
                taskMetrics: ["episode/forward_distance_m": 12 + Float(index)],
                acceptance: PPOEvaluationAcceptance(passed: true, failures: [])))
        }
        let aggregate = try PPOCheckpointEvaluationAggregate.make(reports)
        XCTAssertEqual(aggregate.scope,
                       "single_checkpoint_across_evaluation_seeds")
        XCTAssertEqual(aggregate.totalEpisodes, 2_048)
        XCTAssertEqual(aggregate.totalSuccesses, 1_846)
        XCTAssertEqual(aggregate.pooledSuccessRate, Float(1_846) / 2_048,
                       accuracy: 1e-6)
        XCTAssertTrue(aggregate.robustAcrossEvaluationSeeds)

        var duplicateEvaluationSeed = reports
        duplicateEvaluationSeed[3].evaluationSeed = reports[0].evaluationSeed
        XCTAssertThrowsError(try PPOCheckpointEvaluationAggregate.make(
            duplicateEvaluationSeed))
        var differentCheckpoint = reports
        differentCheckpoint[3].checkpointDirectory = "another/checkpoint"
        XCTAssertThrowsError(try PPOCheckpointEvaluationAggregate.make(
            differentCheckpoint))
        var differentTaskContract = reports
        differentTaskContract[3].evaluationTaskConfiguration = [
            "validationCollisionProfile": 1,
        ]
        differentTaskContract[3].taskConfigurationTransferred = true
        XCTAssertThrowsError(try PPOCheckpointEvaluationAggregate.make(
            differentTaskContract))
        XCTAssertThrowsError(try PPOEvaluationAggregate.make(reports))
    }

    func testCheckpointSelectionIsDeterministicAndRejectsTestSeedLeakage() throws {
        func report(update: Int, successes: Int, meanReturn: Float,
                    passed: Bool = true) -> PPOEvaluationMetrics {
            PPOEvaluationMetrics(
                provenanceVersion: 2,
                task: "humanoid-walk-v0", taskRevision: 33,
                checkpointDirectory: "run/checkpoints/update-\(update)",
                checkpointFingerprint: "fingerprint-\(update)",
                initializationCheckpoint: "parent/checkpoint",
                trainingSeed: 501, evaluationSeed: 25_000,
                evaluationEnvironments: 128,
                trainingUpdates: update,
                trainingEnvironmentSteps: update * 12_288,
                episodes: 128, successes: successes,
                successRate: Float(successes) / 128,
                meanReturn: meanReturn, meanEpisodeLength: 980,
                taskMetrics: [:],
                acceptance: PPOEvaluationAcceptance(
                    passed: passed, failures: passed ? [] : ["failed gate"]))
        }
        let selection = try PPOCheckpointSelection.make([
            report(update: 20, successes: 119, meanReturn: 52),
            report(update: 40, successes: 119, meanReturn: 47),
            report(update: 10, successes: 120, meanReturn: 60, passed: false),
        ])
        XCTAssertEqual(selection.selectedTrainingUpdates, 20)
        XCTAssertEqual(selection.selectedCheckpointFingerprint, "fingerprint-20")
        XCTAssertEqual(selection.schemaVersion, 3)
        XCTAssertEqual(selection.validationSeeds, [25_000])
        XCTAssertEqual(selection.candidates.first?.validationRuns, 1)

        var leakedTest = report(update: 20, successes: 120, meanReturn: 53)
        XCTAssertThrowsError(try selection.validateTestReport(leakedTest))
        leakedTest.evaluationSeed = 26_000
        try selection.validateTestReport(leakedTest)
        leakedTest.checkpointFingerprint = "mutated-policy"
        XCTAssertThrowsError(try selection.validateTestReport(leakedTest))
    }

    func testCheckpointSelectionUsesWorstOfMultipleValidationSeeds() throws {
        func report(update: Int, seed: UInt64, successes: Int,
                    meanReturn: Float = 50) -> PPOEvaluationMetrics {
            PPOEvaluationMetrics(
                provenanceVersion: 2,
                task: "humanoid-walk-v0", taskRevision: 33,
                checkpointDirectory: "run/checkpoints/update-\(update)",
                checkpointFingerprint: "fingerprint-\(update)",
                initializationCheckpoint: "parent/checkpoint",
                trainingSeed: 501, evaluationSeed: seed,
                evaluationEnvironments: 100,
                trainingUpdates: update,
                trainingEnvironmentSteps: update * 12_288,
                episodes: 100, successes: successes,
                successRate: Float(successes) / 100,
                meanReturn: meanReturn, meanEpisodeLength: 980,
                taskMetrics: [:],
                acceptance: PPOEvaluationAcceptance(
                    passed: successes >= 90,
                    failures: successes >= 90 ? [] : ["success gate"]))
        }
        // Update 20 has the better pooled score (190 vs 184), but one lucky
        // and one weak stream. Update 40 is selected for its stronger floor.
        let selection = try PPOCheckpointSelection.make([
            report(update: 20, seed: 25_000, successes: 100),
            report(update: 20, seed: 25_001, successes: 90),
            report(update: 40, seed: 25_000, successes: 92),
            report(update: 40, seed: 25_001, successes: 92),
        ])
        XCTAssertEqual(selection.selectedTrainingUpdates, 40)
        XCTAssertEqual(selection.validationSeeds, [25_000, 25_001])
        XCTAssertEqual(selection.validationEpisodesPerCandidate, 200)
        XCTAssertEqual(selection.validationEpisodesPerSeed, 100)
        XCTAssertEqual(selection.candidates.first?.totalEpisodes, 200)

        var test = report(update: 40, seed: 25_001, successes: 95)
        XCTAssertThrowsError(try selection.validateTestReport(test))
        test.evaluationSeed = 26_000
        try selection.validateTestReport(test)
    }

    func testCheckpointFingerprintCoversReplaySemanticFiles() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "avbd-checkpoint-fingerprint-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        for (name, byte) in [("metadata.json", UInt8(1)),
                             ("policy.safetensors", UInt8(2)),
                             ("training-state.json", UInt8(3))] {
            try Data([byte]).write(to: directory.appendingPathComponent(name))
        }
        let first = try VectorPPOTrainer.checkpointFingerprint(
            directory: directory.path)
        XCTAssertEqual(first.count, 64)
        try Data([4]).write(to: directory.appendingPathComponent("policy.safetensors"))
        let changed = try VectorPPOTrainer.checkpointFingerprint(
            directory: directory.path)
        XCTAssertNotEqual(first, changed)
    }

    func testDeploymentBundlePreservesEvaluatedCheckpointIdentity() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "avbd-policy-export-\(UUID().uuidString)", isDirectory: true)
        let checkpoint = root.appendingPathComponent("checkpoint", isDirectory: true)
        let output = root.appendingPathComponent("field-bundle", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(
            at: checkpoint, withIntermediateDirectories: true)
        let metadata = VectorPolicyMetadata(
            architectureVersion: VectorActorCritic.architectureVersion,
            task: "arachne15-goal-v0", taskRevision: 5,
            taskConfiguration: ["pointGoal": 1],
            observationDimension: 60, actionDimension: 16,
            simulationStep: 0.002, controlDecimation: 10,
            maxEpisodeSteps: 1_000,
            ppo: VectorPPOConfig(normalizeObservations: true),
            normalizer: RunningNormalizerSnapshot(
                count: 2, mean: [Double](repeating: 0, count: 60),
                variance: [Double](repeating: 1, count: 60)))
        let trainingState = VectorPPOTrainingState(
            completedUpdates: 42, environmentSteps: 1_376_256)
        try JSONEncoder().encode(metadata).write(
            to: checkpoint.appendingPathComponent("metadata.json"))
        try JSONEncoder().encode(trainingState).write(
            to: checkpoint.appendingPathComponent("training-state.json"))
        try Data([1, 2, 3, 4]).write(
            to: checkpoint.appendingPathComponent("policy.safetensors"))

        let sourceFingerprint = try VectorPPOTrainer.checkpointFingerprint(
            directory: checkpoint.path)
        let manifest = try VectorPolicyDeploymentBundle.export(
            checkpointDirectory: checkpoint.path,
            outputDirectory: output.path)
        XCTAssertEqual(manifest.checkpointFingerprint, sourceFingerprint)
        XCTAssertEqual(manifest.controlFrequencyHz, 50, accuracy: 1e-4)
        XCTAssertEqual(manifest.trainingUpdates, 42)
        XCTAssertEqual(try VectorPPOTrainer.checkpointFingerprint(
            directory: output.path), sourceFingerprint)
        XCTAssertTrue(FileManager.default.fileExists(atPath: output
            .appendingPathComponent(
                VectorPolicyDeploymentBundle.manifestFileName).path))
        XCTAssertThrowsError(try VectorPolicyDeploymentBundle.export(
            checkpointDirectory: checkpoint.path,
            outputDirectory: output.path))
    }

    func testTrackedArachneDeploymentRuntimeHasExactInferenceParity() throws {
        guard ProcessInfo.processInfo.environment[
            "AVBD_MLX_INTEGRATION_TESTS"] == "1" else {
            throw XCTSkip(
                "requires an Xcode-packaged MLX default.metallib")
        }
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let bundle = packageRoot.appendingPathComponent(
            "Robots/Arachne15/policies/"
                + "arachne15-goal-r6-update-000020", isDirectory: true)
        let fingerprint =
            "30c125b7f01b73bdd1524bc96cf8deb5e8a09897593a49e87aa6ce96f16d3027"
        let runtime = try VectorPolicyDeploymentRuntime(
            bundleDirectory: bundle.path,
            expectedTask: "arachne15-goal-v0",
            expectedTaskRevision: 6,
            expectedCheckpointFingerprint: fingerprint)
        XCTAssertEqual(runtime.observationDimension, 60)
        XCTAssertEqual(runtime.actionDimension, 16)
        XCTAssertEqual(runtime.controlPeriodSeconds, 0.02, accuracy: 1e-8)

        let observation = ContiguousArray((0..<60).map {
            Float(($0 % 9) - 4) * 0.01
        })
        let deployedActions = try runtime.actions(for: observation)
        let checkpointRunner = try VectorPolicyRunner(
            checkpointDirectory: bundle.path)
        let checkpointActions = try checkpointRunner.actions(for: observation)
        XCTAssertEqual(deployedActions.count, 16)
        for (deployed, checkpoint) in zip(deployedActions, checkpointActions) {
            XCTAssertEqual(deployed, checkpoint, accuracy: 1e-7)
            XCTAssertTrue(deployed.isFinite)
            XCTAssertLessThanOrEqual(abs(deployed), 1.0001)
        }
        XCTAssertThrowsError(try runtime.actions(
            for: ContiguousArray(repeating: 0, count: 59)))
        var nonFinite = observation
        nonFinite[17] = .nan
        XCTAssertThrowsError(try runtime.actions(for: nonFinite))
    }

    func testDeploymentRuntimeRejectsTamperedPolicy() throws {
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = packageRoot.appendingPathComponent(
            "Robots/Arachne15/policies/"
                + "arachne15-goal-r6-update-000020", isDirectory: true)
        let copy = FileManager.default.temporaryDirectory.appendingPathComponent(
            "avbd-tampered-deployment-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: copy) }
        try FileManager.default.copyItem(at: source, to: copy)
        let policyURL = copy.appendingPathComponent("policy.safetensors")
        var bytes = try Data(contentsOf: policyURL)
        bytes[bytes.index(before: bytes.endIndex)] ^= 0xff
        try bytes.write(to: policyURL, options: .atomic)
        XCTAssertThrowsError(try VectorPolicyDeploymentRuntime(
            bundleDirectory: copy.path)) { error in
            XCTAssertTrue(String(describing: error).contains("SHA-256"))
        }
    }

    func testGuardedPolicyControllerFailsClosedAndRequiresRearm() throws {
        let inference = TestVectorPolicyInference()
        let controller = GuardedPolicyController(inference: inference)
        let values = ContiguousArray<Float>([1, 2, 3])
        var command = controller.command(for: .init(
            sequence: 10, timestampSeconds: 99.99, values: values),
            nowSeconds: 100)
        XCTAssertEqual(command.mode, .safeStop)
        XCTAssertEqual(command.fault, .notArmed)

        controller.arm()
        command = controller.command(for: .init(
            sequence: 10, timestampSeconds: 99.99, values: values),
            nowSeconds: 100)
        XCTAssertEqual(command.mode, .active)
        XCTAssertNil(command.fault)
        XCTAssertEqual(command.values, [0.25, -0.5])
        XCTAssertEqual(command.policyFingerprint, "qualified-test-policy")
        XCTAssertEqual(command.validUntilSeconds - command.createdAtSeconds,
                       0.02, accuracy: 1e-8)

        command = controller.command(for: .init(
            sequence: 10, timestampSeconds: 100, values: values),
            nowSeconds: 100.01)
        XCTAssertEqual(command.mode, .safeStop)
        XCTAssertEqual(command.fault, .outOfOrderObservation)
        XCTAssertFalse(controller.isArmed)

        controller.arm()
        command = controller.command(for: .init(
            sequence: 11, timestampSeconds: 99, values: values),
            nowSeconds: 100)
        XCTAssertEqual(command.fault, .staleObservation)
        XCTAssertFalse(controller.isArmed)

        controller.arm()
        inference.result = [.nan, 0]
        command = controller.command(for: .init(
            sequence: 12, timestampSeconds: 100, values: values),
            nowSeconds: 100.01)
        XCTAssertEqual(command.fault, .invalidAction)
        XCTAssertEqual(command.values, [0, 0])
        XCTAssertFalse(controller.isArmed)
    }

    func testArachneDeploymentControllerEmitsCalibratedDeadlineFrame() throws {
        let inference = TestVectorPolicyInference(
            observationDimension: 60, actionDimension: 16,
            checkpointFingerprint: "arachne-qualified")
        inference.result = ContiguousArray((0..<16).map {
            $0.isMultiple(of: 2) ? Float(0.5) : Float(-0.5)
        })
        let calibration = Arachne15HardwareCalibration(
            robotSerial: "AR15-TEST", commissioned: true,
            measuredAtUTC: "2026-07-16T00:00:00Z",
            policyCheckpointFingerprint: "arachne-qualified",
            servoIDs: Array(1...16),
            servoZeroRadians: [Float](repeating: 2.5, count: 16),
            servoDirectionSigns: [Float](repeating: 1, count: 16),
            jointLowerRadians: (0..<16).map {
                $0.isMultiple(of: 2) ? -0.55 : -0.7
            },
            jointUpperRadians: (0..<16).map {
                $0.isMultiple(of: 2) ? 0.55 : 0.7
            },
            currentLimitsMilliamps: [Int](repeating: 300, count: 16),
            maximumServoTemperatureCelsius: 55,
            measuredMaximumRoundTripLatencySeconds: 0.018)
        let deployment = try Arachne15DeploymentController(
            inference: inference, calibration: calibration)
        let input = Arachne15PolicyInput(
            bodyLinearVelocity: .zero, bodyAngularVelocity: .zero,
            projectedGravity: F3(0, 0, 1), commandedBodyTwist: .zero,
            jointPositions: [Float](repeating: 0, count: 16),
            jointVelocities: [Float](repeating: 0, count: 16),
            previousActions: [Float](repeating: 0, count: 16))
        var frame = deployment.command(
            for: input, sequence: 1, sensorTimestampSeconds: 10,
            nowSeconds: 10.01)
        XCTAssertEqual(frame.mode, .safeStop)
        XCTAssertEqual(frame.servoPositionRadians, [])

        deployment.arm()
        frame = deployment.command(
            for: input, sequence: 2, sensorTimestampSeconds: 10,
            nowSeconds: 10.01)
        XCTAssertEqual(frame.mode, .active)
        XCTAssertEqual(frame.servoIDs, Array(1...16))
        XCTAssertEqual(frame.policyFingerprint, "arachne-qualified")
        XCTAssertEqual(frame.servoPositionRadians.count, 16)
        XCTAssertEqual(frame.servoPositionRadians[0], 2.675, accuracy: 1e-6)
        XCTAssertEqual(frame.servoPositionRadians[1], 2.275, accuracy: 1e-6)
        XCTAssertEqual(try JSONDecoder().decode(
            Arachne15ServoCommandFrame.self,
            from: JSONEncoder().encode(frame)), frame)

        var corrupt = input
        corrupt.projectedGravity = .zero
        frame = deployment.command(
            for: corrupt, sequence: 3, sensorTimestampSeconds: 10.02,
            nowSeconds: 10.03)
        XCTAssertEqual(frame.mode, .safeStop)
        XCTAssertEqual(frame.fault, .invalidObservation)
        XCTAssertTrue(frame.servoPositionRadians.isEmpty)
        XCTAssertFalse(deployment.supervisor.isArmed)
    }
}
