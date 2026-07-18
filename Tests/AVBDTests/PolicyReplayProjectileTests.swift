import XCTest
import simd
@testable import AVBDCore

final class PolicyReplayProjectileTests: XCTestCase {
    func testUnitreeH1ReusableBoxIsPhysicalAndTransfersMomentum() throws {
        let environment = try UnitreeH1Sim2SimEnv(
            projectileSize: 0.25, projectileMass: 8)
        let reference = environment.refs
        XCTAssertFalse(reference.bodies.contains(reference.projectile))
        XCTAssertEqual(
            environment.solver.bodyMass(reference.projectile), 8,
            accuracy: 1e-5)
        let projectileColliders = environment.scene.colliders.filter {
            $0.body == reference.projectile
        }
        XCTAssertEqual(projectileColliders.count, 1)
        XCTAssertEqual(projectileColliders[0].shape, .box)
        XCTAssertEqual(projectileColliders[0].size, F3(repeating: 0.25))
        XCTAssertEqual(projectileColliders[0].collisionGroup, 1)
        XCTAssertTrue(projectileColliders[0].collisionEnabled)
        let robotColliders = environment.scene.colliders.filter {
            reference.bodies.contains($0.body)
        }
        XCTAssertFalse(robotColliders.isEmpty)
        XCTAssertTrue(robotColliders.allSatisfy {
            $0.collisionEnabled && $0.collisionGroup == 1
        })

        var projectileState = environment.solver.bodyStates([
            reference.projectile,
        ])[0]
        XCTAssertEqual(projectileState.position.z, -4, accuracy: 1e-6)
        XCTAssertEqual(simd_length(projectileState.linearVelocity), 0,
                       accuracy: 1e-6)

        environment.throwBoxes(
            environmentIDs: [0], sideSigns: [1],
            launchDistance: 1.2, speed: 6)
        projectileState = environment.solver.bodyStates([
            reference.projectile,
        ])[0]
        let target = environment.state().torso.position
        XCTAssertGreaterThan(
            simd_dot(projectileState.linearVelocity,
                     target - projectileState.position),
            0, "ballistic box must initially travel toward the measured torso")
        let incomingDirection = simd_normalize(projectileState.linearVelocity)
        let incomingSpeed = simd_dot(
            projectileState.linearVelocity, incomingDirection)
        let rootVelocityBefore = environment.state().root.linearVelocity
        let neutralTargets: ContiguousArray<Float> = [
            0, 0, -0.1, 0.3, -0.2,
            0, 0, -0.1, 0.3, -0.2,
        ]
        var contacted = false
        var speedAtContact = Float.greatestFiniteMagnitude
        var rootVelocityDeltaAtContact: Float = 0
        for _ in 0..<180 where !contacted {
            environment.step(
                jointPositionTargets: neutralTargets, decimation: 1)
            contacted = environment.boxRobotContacts()[0]
            if contacted {
                speedAtContact = simd_dot(
                    environment.solver.bodyVelocity(reference.projectile),
                    incomingDirection)
                rootVelocityDeltaAtContact = simd_dot(
                    environment.state().root.linearVelocity
                        - rootVelocityBefore,
                    incomingDirection)
            }
        }
        XCTAssertTrue(contacted,
                      "box must form a physical contact with the imported H1")
        XCTAssertLessThan(speedAtContact, incomingSpeed - 0.1,
                          "contact must remove incoming projectile momentum")
        XCTAssertGreaterThan(rootVelocityDeltaAtContact, 0.005,
                             "H1 must receive momentum from the box")

        environment.hideBoxes(environmentIDs: [0])
        projectileState = environment.solver.bodyStates([
            reference.projectile,
        ])[0]
        XCTAssertEqual(projectileState.position.z, -4, accuracy: 1e-6)
        XCTAssertEqual(simd_length(projectileState.linearVelocity), 0,
                       accuracy: 1e-6)
        XCTAssertEqual(simd_length(projectileState.angularVelocity), 0,
                       accuracy: 1e-6)
    }

    func testIsaacFlatReplayProbePreservesNominalPlantUntilImpact() throws {
        let options: [String: Float] = [
            "standingCommandProbability": 1,
            "initialYawRange": 0,
            "observationNoise": 0,
        ]
        let nominal = try XCTUnwrap(
            BuiltInRLTasks.registry.make(
                "humanoid-isaac-flat-v0",
                configuration: .init(
                    numEnvironments: 1, seed: 91, autoReset: false,
                    options: options)) as? HumanoidIsaacVelocityTask)
        let replay = try XCTUnwrap(
            BuiltInRLTasks.registry.make(
                "humanoid-isaac-flat-v0",
                configuration: .init(
                    numEnvironments: 1, seed: 91, autoReset: false,
                    includeInteractiveRobustnessProbes: true,
                    options: options)) as? HumanoidIsaacVelocityTask)

        // The runtime-only scene feature cannot alter checkpoint semantics.
        XCTAssertEqual(replay.spec, nominal.spec)
        XCTAssertFalse(nominal.hasProjectile(environment: 0))
        XCTAssertTrue(replay.hasProjectile(environment: 0))
        let reference = replay.environment.refs[0]
        let projectile = try XCTUnwrap(reference.projectile)
        XCTAssertEqual(replay.environment.solver.bodyMass(projectile), 8,
                       accuracy: 1e-5)

        // The three exact H1_MINIMAL hulls still own terrain contact. Extra
        // arm/hand/body primitives are same-replica only, so they can receive
        // box impacts without changing flat-ground locomotion before launch.
        let enabled = replay.environment.scene.colliders.filter(
            \.collisionEnabled)
        let minimalHulls = enabled.filter {
            !$0.convexHullVertices.isEmpty
                && reference.bodies.contains($0.body)
        }
        XCTAssertEqual(minimalHulls.count, 3)
        XCTAssertTrue(minimalHulls.allSatisfy(
            \.collidesWithSharedGeometry))
        let handBodies = Set([reference.leftHand, reference.rightHand])
        let handColliders = enabled.filter { handBodies.contains($0.body) }
        XCTAssertFalse(handColliders.isEmpty)
        XCTAssertTrue(handColliders.allSatisfy {
            !$0.collidesWithSharedGeometry && $0.collisionGroup == 1
        })

        let nominalInitial = try nominal.reset(seed: 92)
        let replayInitial = try replay.reset(seed: 92)
        XCTAssertEqual(nominalInitial.policy, replayInitial.policy)
        let action = RLActionBatch(spec: nominal.spec, repeating: 0)
        var nominalResult = RLStepBatch(spec: nominal.spec)
        var replayResult = RLStepBatch(spec: replay.spec)
        for _ in 0..<10 {
            try nominal.step(actions: action, into: &nominalResult)
            try replay.step(actions: action, into: &replayResult)
            for (expected, actual) in zip(
                nominalResult.observations.policy,
                replayResult.observations.policy) {
                XCTAssertEqual(actual, expected, accuracy: 1e-6)
            }
        }

        replay.throwRobustnessBoxes(
            environmentIDs: [0], sideSigns: [1],
            launchDistance: 1.2, speed: 6)
        let launch = replay.environment.solver.bodyStates([projectile])[0]
        let incomingDirection = simd_normalize(launch.linearVelocity)
        var contacted = false
        var transferredVelocity: Float = 0
        for _ in 0..<30 where !contacted {
            try nominal.step(actions: action, into: &nominalResult)
            try replay.step(actions: action, into: &replayResult)
            contacted = replay.environment.boxRobotContacts()[0]
            if contacted {
                let nominalRoot = nominal.environment.states()[0]
                    .root.linearVelocity
                let replayRoot = replay.environment.states()[0]
                    .root.linearVelocity
                transferredVelocity = simd_dot(
                    replayRoot - nominalRoot, incomingDirection)
            }
        }
        XCTAssertTrue(contacted,
                      "manual box must contact the full replay collision plant")
        XCTAssertGreaterThan(
            transferredVelocity, 0.005,
            "the replay-only contact must transfer momentum to H1")
    }

    func testArachneTrainingDefaultHasNoProjectileTopology() throws {
        let environment = try Arachne15Env(
            numEnvironments: 8,
            domainRandomization: .init(), solverIterations: 20)
        XCTAssertTrue(environment.refs.allSatisfy { $0.projectile == nil })
        XCTAssertEqual(environment.scene.colliders
            .filter(\.collisionEnabled).count, 1 + 8 * 39,
            "training must not pay for replay-only bodies or colliders")
    }

    func testArachneReusableBoxesAreIsolatedResettableAndPhysical() throws {
        let task = try Arachne15LocomotionTask(configuration: .init(
            numEnvironments: 2, seed: 81,
            standingCommandProbability: 1,
            initialRollPitchRange: 0, initialYawRange: 0,
            observationNoise: false, maximumActionLatencySteps: 0,
            domainRandomization: .init(),
            autoReset: false),
            includeInteractiveRobustnessProbe: true,
            projectileSize: 0.05, projectileMass: 0.10)
        let environment = task.environment
        XCTAssertTrue(environment.hasProjectile(environment: 0))
        let projectiles = try environment.refs.map {
            try XCTUnwrap($0.projectile)
        }
        for replica in environment.refs.indices {
            let projectile = projectiles[replica]
            XCTAssertFalse(environment.refs[replica].bodies.contains(projectile))
            XCTAssertEqual(environment.solver.bodyMass(projectile), 0.10,
                           accuracy: 1e-6)
            let colliders = environment.scene.colliders.filter {
                $0.body == projectile
            }
            XCTAssertEqual(colliders.count, 1)
            XCTAssertEqual(colliders[0].shape, .box)
            XCTAssertEqual(colliders[0].size, F3(repeating: 0.05))
            XCTAssertEqual(colliders[0].collisionGroup, UInt32(replica + 1))
            XCTAssertTrue(colliders[0].collisionEnabled)
            let robotColliders = environment.scene.colliders.filter {
                $0.collisionEnabled
                    && environment.refs[replica].bodies.contains($0.body)
            }
            XCTAssertEqual(robotColliders.count, 39)
            XCTAssertTrue(robotColliders.allSatisfy {
                $0.collisionGroup == UInt32(replica + 1)
            })
        }

        environment.throwBoxes(
            environmentIDs: [0, 1], sideSigns: [1, -1],
            launchDistance: 0.35, speed: 2.5)
        var projectileStates = environment.solver.bodyStates(projectiles)
        let robotStates = environment.states()
        for replica in projectiles.indices {
            XCTAssertGreaterThan(projectileStates[replica].position.z, 0.04)
            XCTAssertGreaterThan(
                simd_dot(
                    projectileStates[replica].linearVelocity,
                    robotStates[replica].root.position
                        - projectileStates[replica].position),
                0, "box must travel toward its replica's measured chassis")
        }

        environment.hideBoxes(environmentIDs: [0])
        projectileStates = environment.solver.bodyStates(projectiles)
        XCTAssertEqual(projectileStates[0].position.z, -4, accuracy: 1e-6)
        XCTAssertEqual(simd_length(projectileStates[0].linearVelocity), 0,
                       accuracy: 1e-6)
        XCTAssertGreaterThan(projectileStates[1].position.z, 0.04,
                             "hiding one replica must not reset another")
        environment.reset([1], seeds: [82])
        projectileStates = environment.solver.bodyStates(projectiles)
        XCTAssertEqual(projectileStates[1].position.z, -4, accuracy: 1e-6)
        XCTAssertEqual(simd_length(projectileStates[1].linearVelocity), 0,
                       accuracy: 1e-6)
        XCTAssertEqual(simd_length(projectileStates[1].angularVelocity), 0,
                       accuracy: 1e-6)

        environment.throwBoxes(
            environmentIDs: [0], sideSigns: [1],
            launchDistance: 0.35, speed: 2.5)
        let launched = environment.solver.bodyStates([projectiles[0]])[0]
        let incomingDirection = simd_normalize(launched.linearVelocity)
        let incomingSpeed = simd_dot(
            launched.linearVelocity, incomingDirection)
        let neutralActions = ContiguousArray<Float>(
            repeating: 0,
            count: environment.numEnvironments * Arachne15Env.actionDimension)
        var contacted = false
        var speedAtContact = Float.greatestFiniteMagnitude
        var chassisVelocityDeltaAtContact: Float = 0
        for _ in 0..<160 where !contacted {
            environment.step(actions: neutralActions, decimation: 1)
            let contacts = environment.boxRobotContacts()
            contacted = contacts[0]
            XCTAssertFalse(contacts[1],
                           "collision groups must isolate the control replica")
            if contacted {
                speedAtContact = simd_dot(
                    environment.solver.bodyVelocity(projectiles[0]),
                    incomingDirection)
                let states = environment.states()
                chassisVelocityDeltaAtContact = simd_dot(
                    states[0].root.linearVelocity
                        - states[1].root.linearVelocity,
                    incomingDirection)
            }
        }
        XCTAssertTrue(contacted,
                      "box must form a physical Arachne chassis contact")
        XCTAssertLessThan(speedAtContact, incomingSpeed - 0.05,
                          "contact must remove incoming projectile momentum")
        XCTAssertGreaterThan(chassisVelocityDeltaAtContact, 0.005,
                             "impacted Arachne must receive box momentum")
    }
}
