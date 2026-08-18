import XCTest
import simd
import CryptoKit
@testable import AVBDCore

final class ManiSkillPushTTests: XCTestCase {
    func testBundledPandaPusherMatchesPinnedMenagerieJointContract() throws {
        let asset = try MJCFAsset.bundledPandaPusher()
        XCTAssertEqual(asset.name, "avbd_panda_pusher")
        XCTAssertEqual(asset.bodyNames,
                       (0...7).map { "link\($0)" })
        XCTAssertEqual(asset.jointNames,
                       (1...7).map { "joint\($0)" })
        XCTAssertEqual(asset.actuatorNames,
                       (1...7).map { "actuator\($0)" })
    }

    func testPackagedPandaAndManiSkillProvenanceIsHashBound() throws {
        func url(_ resource: String, _ fileExtension: String?,
                 _ subdirectory: String) throws -> URL {
            try MJCFAsset.bundledResourceURL(
                resource: resource, withExtension: fileExtension,
                subdirectory: subdirectory)
        }
        func digest(_ url: URL) throws -> String {
            SHA256.hash(data: try Data(contentsOf: url)).map {
                String(format: "%02x", $0)
            }.joined()
        }
        func json(_ url: URL) throws -> [String: Any] {
            try XCTUnwrap(JSONSerialization.jsonObject(
                with: Data(contentsOf: url)) as? [String: Any])
        }

        let pandaDirectory = "Assets/panda_pusher"
        let pandaManifest = try json(url(
            "PROVENANCE", "json", pandaDirectory))
        let pandaAsset = try XCTUnwrap(
            pandaManifest["asset"] as? [String: Any])
        XCTAssertEqual(
            try digest(url("panda_pusher", "xml", pandaDirectory)),
            pandaAsset["sha256"] as? String)
        let pandaLicense = try XCTUnwrap(
            pandaManifest["redistributedLicense"] as? [String: Any])
        XCTAssertEqual(
            try digest(url("LICENSE", nil, pandaDirectory)),
            pandaLicense["sha256"] as? String)
        let pandaUpstream = try XCTUnwrap(
            pandaManifest["upstream"] as? [String: Any])
        XCTAssertEqual(pandaUpstream["revision"] as? String,
                       "da76818e269b82289eba39808e2fb91d679d6994")
        XCTAssertEqual(pandaUpstream["license"] as? String, "Apache-2.0")

        let taskDirectory = "Assets/maniskill_pusht"
        let taskManifest = try json(url(
            "PROVENANCE", "json", taskDirectory))
        let redistributed = try XCTUnwrap(
            taskManifest["redistributedFiles"] as? [[String: Any]])
        let expected = Dictionary(uniqueKeysWithValues: try redistributed.map {
            entry -> (String, String) in
            let path = try XCTUnwrap(entry["path"] as? String)
            return (URL(fileURLWithPath: path).lastPathComponent,
                    try XCTUnwrap(entry["sha256"] as? String))
        })
        XCTAssertEqual(try digest(url("LICENSE", nil, taskDirectory)),
                       expected["LICENSE"])
        XCTAssertEqual(try digest(url("NOTICE", nil, taskDirectory)),
                       expected["NOTICE"])
        let taskUpstream = try XCTUnwrap(
            taskManifest["upstream"] as? [String: Any])
        XCTAssertEqual(taskUpstream["revision"] as? String,
                       "62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3")
        XCTAssertEqual(taskUpstream["license"] as? String, "Apache-2.0")

        XCTAssertThrowsError(try url(
            "panda_stick", "xml", "Assets/panda_stick"))
    }

    func testPublishedInitialPoseResolvesToPublishedTCP() throws {
        let task = try ManiSkillPushTTask(configuration: .init(
            numEnvironments: 1, seed: 9351,
            autoReset: false, robotInitialJointNoise: 0))
        XCTAssertEqual(task.spec.revision,
                       RLPhysicsContract.deterministicColorSolveV1(9))
        var observations = RLObservationBatch(spec: task.spec)
        try task.reset(environments: nil, seed: 9351,
                       into: &observations)
        let state = try XCTUnwrap(task.environment.states().first)
        // ManiSkill records [-0.321, 0.284, 0.024]. AVBD places the work
        // surface at z=0 rather than SAPIEN's small table/contact offset, so
        // XY must agree tightly while Z differs by less than one centimetre.
        XCTAssertEqual(state.tcpPosition.x, -0.321, accuracy: 0.002)
        XCTAssertEqual(state.tcpPosition.y, 0.284, accuracy: 0.002)
        XCTAssertEqual(state.tcpPosition.z, 0.024, accuracy: 0.010)
        XCTAssertEqual(state.jointPositions.count, 7)
        for (actual, expected) in zip(
            state.jointPositions,
            PandaStickPushTEnv.defaultJointPositions) {
            XCTAssertEqual(actual, expected, accuracy: 2e-4)
        }
    }

    func testCartesianIKReachesPushTContactWorkspaceWithVerticalTool() throws {
        let task = try ManiSkillPushTTask(configuration: .init(
            numEnvironments: 1, seed: 9351,
            autoReset: false, robotInitialJointNoise: 0))
        let home = try task.environment.kinematicTCPPose(
            jointPositions: PandaStickPushTEnv.defaultJointPositions)
        let targets = [
            F3(-0.24, -0.16, 0.024),
            F3(-0.08, -0.16, 0.024),
            F3(-0.08, -0.04, 0.024),
        ]
        var seed = PandaStickPushTEnv.defaultJointPositions
        for target in targets {
            seed = try task.environment.inverseKinematics(
                targetPosition: target, targetRotation: home.rotation,
                initialJointPositions: seed)
            let reached = try task.environment.kinematicTCPPose(
                jointPositions: seed)
            XCTAssertLessThan(length(reached.position - target), 0.002)
            let relative = home.rotation * reached.rotation.conjugate
            let angle = 2 * atan2(length(relative.imag), abs(relative.real))
            XCTAssertLessThan(angle, 0.03)
            for (joint, range) in zip(
                seed, PandaStickPushTEnv.jointRanges) {
                XCTAssertGreaterThanOrEqual(joint, range.0)
                XCTAssertLessThanOrEqual(joint, range.1)
            }
        }
    }

    func testArticulatedOffCenterSweepCanRotateT() throws {
        let task = try ManiSkillPushTTask(configuration: .init(
            numEnvironments: 1, seed: 9351, autoReset: false,
            robotInitialJointNoise: 0))
        var observations = RLObservationBatch(spec: task.spec)
        try task.reset(environments: nil, seed: 9351, into: &observations)
        let env = task.environment
        let ref = env.refs[0]
        let blockCenter = F3(ref.goalPosition.x, ref.goalPosition.y, 0.021)
        env.solver.setBodyStates([.init(
            body: ref.block, position: ref.center + blockCenter,
            rotation: Quat(angle: 0, axis: F3(0, 0, 1)))])

        let homePose = try env.kinematicTCPPose(
            jointPositions: PandaStickPushTEnv.defaultJointPositions)
        var ikSeed = PandaStickPushTEnv.defaultJointPositions
        var maximumTrackingError: Float = 0
        func follow(_ start: F3, _ end: F3, steps: Int) throws {
            for step in 1...steps {
                let alpha = Float(step) / Float(steps)
                let target = simd_mix(start, end, F3(repeating: alpha))
                ikSeed = try env.inverseKinematics(
                    targetPosition: target, targetRotation: homePose.rotation,
                    initialJointPositions: ikSeed)
                let state = env.states()[0]
                var action = ContiguousArray(repeating: Float(0), count: 7)
                for joint in 0..<7 {
                    action[joint] = simd_clamp(
                        (ikSeed[joint] - state.jointPositions[joint]) / 0.1,
                        -1, 1)
                }
                env.step(actions: action, decimation: 5, deltaScale: 0.1)
                let after = env.states()[0]
                maximumTrackingError = max(
                    maximumTrackingError,
                    length(after.tcpPosition - target))
            }
        }

        let outsideLow = F3(blockCenter.x + 0.080,
                            blockCenter.y - 0.105, 0.024)
        let outsideHigh = F3(outsideLow.x, outsideLow.y, 0.12)
        try follow(homePose.position, outsideHigh, steps: 30)
        try follow(outsideHigh, outsideLow, steps: 20)
        try follow(outsideLow, outsideLow, steps: 40)
        let before = env.states()[0]
        XCTAssertLessThan(length(before.tcpPosition - outsideLow), 0.002,
                          "physical TCP must agree with its IK/contact pose")
        try follow(outsideLow,
                   F3(outsideLow.x, blockCenter.y + 0.010, outsideLow.z),
                   steps: 50)
        let after = env.states()[0]
        let beforeForward = before.blockRotation.act(F3(1, 0, 0))
        let afterForward = after.blockRotation.act(F3(1, 0, 0))
        let yawDelta = atan2(afterForward.y, afterForward.x)
            - atan2(beforeForward.y, beforeForward.x)
        XCTAssertGreaterThan(yawDelta, 0.10,
                             "off-center +Y sweep must impart positive yaw")
        XCTAssertGreaterThan(length(after.blockPosition - before.blockPosition),
                             0.005)
        XCTAssertLessThan(maximumTrackingError, 0.05)
    }

    func testClockwiseSweepEscapesLearnedPolicyTerminalPose() throws {
        let task = try ManiSkillPushTTask(configuration: .init(
            numEnvironments: 1, seed: 4796, autoReset: false,
            robotInitialJointNoise: 0))
        var observations = RLObservationBatch(spec: task.spec)
        try task.reset(environments: nil, seed: 4796, into: &observations)
        let env = task.environment
        let ref = env.refs[0]

        // Deterministic state observed at step 54 of the held-out update-742
        // policy. The T is translated onto the goal but remains 0.254 rad
        // under-rotated while the stick is in continuous physical contact.
        var ikSeed: [Float] = [
            0.21925, 0.48949, -0.41799, -2.37415,
            -0.25840, 2.83594, 2.54970,
        ]
        try env.setRobotJointPositions(ikSeed)
        env.solver.setBodyStates([.init(
            body: ref.block,
            position: ref.center + F3(-0.15569, -0.10734, 0.019),
            rotation: Quat(angle: -0.79329, axis: F3(0, 0, 1)))])

        var zero = ContiguousArray(repeating: Float(0), count: 7)
        for _ in 0..<4 {
            env.step(actions: zero, decimation: 5, deltaScale: 0.1)
        }
        let before = env.states()[0]
        let relative = SIMD2(
            before.tcpPosition.x - before.blockPosition.x,
            before.tcpPosition.y - before.blockPosition.y)
        let beforeForward = before.blockRotation.act(F3(1, 0, 0))
        let beforeYaw = atan2(beforeForward.y, beforeForward.x)
        let c0 = cos(beforeYaw), s0 = sin(beforeYaw)
        let startLocal = SIMD2(
            c0 * relative.x + s0 * relative.y,
            -s0 * relative.x + c0 * relative.y)
        let targetRotation = before.tcpRotation
        var maximumTrackingError: Float = 0
        var sawRobotContact = false

        for step in 1...80 {
            let local: SIMD2<Float>
            if step <= 30 {
                local = simd_mix(
                    startLocal, SIMD2(0.085, -0.004),
                    SIMD2(repeating: Float(step) / 30))
            } else {
                local = simd_mix(
                    SIMD2(0.085, -0.004), SIMD2(0.085, -0.024),
                    SIMD2(repeating: min(Float(step - 30) / 20, 1)))
            }
            let currentBlock = env.states()[0]
            let currentForward = currentBlock.blockRotation.act(F3(1, 0, 0))
            let yaw = atan2(currentForward.y, currentForward.x)
            let c = cos(yaw), s = sin(yaw)
            let target = F3(
                currentBlock.blockPosition.x + c * local.x - s * local.y,
                currentBlock.blockPosition.y + s * local.x + c * local.y,
                before.tcpPosition.z)
            ikSeed = try env.inverseKinematics(
                targetPosition: target, targetRotation: targetRotation,
                initialJointPositions: ikSeed)
            let state = env.states()[0]
            for joint in 0..<7 {
                zero[joint] = simd_clamp(
                    (ikSeed[joint] - state.jointPositions[joint]) / 0.1,
                    -1, 1)
            }
            env.step(actions: zero, decimation: 5, deltaScale: 0.1)
            let after = env.states()[0]
            maximumTrackingError = max(
                maximumTrackingError, length(after.tcpPosition - target))
            let robotBodies = Set(ref.robotBodies)
            sawRobotContact = sawRobotContact || env.solver
                .activeRigidContactPairs().contains { pair in
                    (pair.0 == ref.block && robotBodies.contains(pair.1))
                        || (pair.1 == ref.block && robotBodies.contains(pair.0))
                }
        }

        let after = env.states()[0]
        let afterForward = after.blockRotation.act(F3(1, 0, 0))
        var yawDelta = atan2(afterForward.y, afterForward.x)
            - atan2(beforeForward.y, beforeForward.x)
        yawDelta -= 2 * .pi * (yawDelta / (2 * .pi)).rounded()
        XCTAssertTrue(sawRobotContact)
        XCTAssertLessThan(yawDelta, -0.15,
                          "a feasible clockwise contact sweep must rotate the T")
        XCTAssertLessThan(maximumTrackingError, 0.05)
    }

    func testMountedBaseProxyMatchesSourceBoundsAndCannotFightTable() throws {
        let task = try ManiSkillPushTTask(configuration: .init(
            numEnvironments: 1, seed: 9351,
            autoReset: false, robotInitialJointNoise: 0))
        let env = task.environment
        let root = env.refs[0].robotBodies[0]
        let rootCollider = try XCTUnwrap(env.scene.colliders.first {
            $0.body == root
        })
        XCTAssertEqual(rootCollider.shape, .box)
        let body = env.scene.bodies[root]
        let rotation = (body.rotation * rootCollider.localRotation).normalized
        let center = body.position + body.rotation.act(rootCollider.localPosition)
        let half = rootCollider.size * 0.5
        let minimumZ = center.z
            - abs(rotation.act(F3(1, 0, 0)).z) * half.x
            - abs(rotation.act(F3(0, 1, 0)).z) * half.y
            - abs(rotation.act(F3(0, 0, 1)).z) * half.z
        XCTAssertGreaterThanOrEqual(minimumZ, -1e-5)
        XCTAssertLessThanOrEqual(minimumZ, 1e-4)
        XCTAssertTrue(env.scene.collisionExclusions.contains {
            ($0.bodyA == 0 && $0.bodyB == root)
                || ($0.bodyA == root && $0.bodyB == 0)
        })
    }

    func testAVBDPusherMatchesFirstPartyDesignAndTCP() throws {
        let task = try ManiSkillPushTTask(configuration: .init(
            numEnvironments: 1, seed: 9351,
            autoReset: false, robotInitialJointNoise: 0))
        let env = task.environment
        let ref = env.refs[0]
        let stick = try XCTUnwrap(env.scene.colliders.first {
            $0.body == ref.link7
                && $0.shape == .capsule
                && abs($0.size.y - 0.009) < 1e-7
        })
        XCTAssertEqual(stick.size.x, 0.092, accuracy: 1e-7)
        XCTAssertEqual(stick.size.y, 0.009, accuracy: 1e-7)
        XCTAssertEqual(stick.size.x + 2 * stick.size.y, 0.110,
                       accuracy: 1e-7)

        // Express the collider center in link7's authored frame. Its positive
        // Z cap must terminate at AVBD's TCP, 255 mm from link7.
        let centerInLink = ref.link7Frame.rotation.conjugate.act(
            stick.localPosition - ref.link7Frame.position)
        let axisInLink = ref.link7Frame.rotation.conjugate.act(
            stick.localRotation.act(F3(0, 0, 1)))
        let halfEnvelope = stick.size.x * 0.5 + stick.size.y
        let positiveTip = centerInLink + axisInLink * halfEnvelope
        let negativeTip = centerInLink - axisInLink * halfEnvelope
        let tcp = PandaStickPushTEnv.tcpInLink7
        XCTAssertLessThan(
            min(length(positiveTip - tcp), length(negativeTip - tcp)), 1e-6)
    }

    func testReducedPlantPreservesMenagerieEffortAndFrictionSemantics() throws {
        let task = try ManiSkillPushTTask(configuration: .init(
            numEnvironments: 1, seed: 9351,
            autoReset: false, robotInitialJointNoise: 0))
        let env = task.environment
        let ref = env.refs[0]
        let effortLimits = ref.motors.map {
            env.scene.joints[$0].motorTorque
        }
        XCTAssertEqual(effortLimits, [87, 87, 87, 87, 12, 12, 12])

        let robotBodies = Set(ref.robotBodies)
        let robotColliders = env.scene.colliders.filter {
            robotBodies.contains($0.body)
        }
        let pusher = try XCTUnwrap(robotColliders.first {
            $0.shape == .capsule && abs($0.size.y - 0.009) < 1e-7
        })
        XCTAssertEqual(pusher.friction, 0.3, accuracy: 1e-7)
        for proxy in robotColliders where !(
            proxy.shape == .capsule && abs(proxy.size.y - 0.009) < 1e-7
        ) {
            XCTAssertEqual(proxy.friction, 1, accuracy: 1e-7)
        }
    }

    func testCentimeterScaleContactMarginPreservesTThickness() throws {
        let task = try ManiSkillPushTTask(configuration: .init(
            numEnvironments: 1, seed: 9351,
            autoReset: false, robotInitialJointNoise: 0))
        XCTAssertEqual(task.environment.scene.settings.collisionMargin, 0.001,
                       accuracy: 1e-7)
        var observation = RLObservationBatch(spec: task.spec)
        try task.reset(environments: nil, seed: 9351, into: &observation)
        let action = RLActionBatch(spec: task.spec)
        var result = RLStepBatch(spec: task.spec)
        for _ in 0..<20 { try task.step(actions: action, into: &result) }
        let settled = task.environment.states()[0]
        // Full T thickness is 4 cm. With 1 mm contact slop its COM should
        // settle at roughly 19 mm, not the old deeply interpenetrated 10 mm.
        XCTAssertEqual(settled.blockPosition.z, 0.019, accuracy: 0.002)
    }

    func testSpawnAndGoalMatchManiSkillPushTV1() throws {
        let task = try ManiSkillPushTTask(configuration: .init(
            numEnvironments: 16, seed: 4796,
            robotInitialJointNoise: 0))
        var observations = RLObservationBatch(spec: task.spec)
        try task.reset(environments: nil, seed: 4796,
                       into: &observations)
        for (environment, state) in task.environment.states().enumerated() {
            let goal = task.environment.refs[environment].goalPosition
            XCTAssertEqual(goal.x, -0.156, accuracy: 1e-6)
            XCTAssertEqual(goal.y, -0.1, accuracy: 1e-6)
            XCTAssertGreaterThanOrEqual(state.blockPosition.x, goal.x - 0.1)
            XCTAssertLessThanOrEqual(state.blockPosition.x, goal.x + 0.1)
            XCTAssertGreaterThanOrEqual(state.blockPosition.y, goal.y - 0.1)
            XCTAssertLessThanOrEqual(state.blockPosition.y, goal.y + 0.2)
            XCTAssertEqual(state.blockPosition.z, 0.021, accuracy: 1e-4)
        }
    }

    func testCoverageAndRewardMatchPublishedEquations() {
        let coverage = PandaStickPushTEnv.coverage(
            blockPosition: PandaStickPushTEnv.goalPosition,
            blockYaw: PandaStickPushTEnv.goalYaw,
            goalPosition: PandaStickPushTEnv.goalPosition,
            goalYaw: PandaStickPushTEnv.goalYaw)
        XCTAssertEqual(coverage, 1, accuracy: 1e-5)
        let pseudoCoverage = PandaStickPushTEnv.officialPseudoRenderCoverage(
            blockPosition: PandaStickPushTEnv.goalPosition,
            blockYaw: PandaStickPushTEnv.goalYaw,
            goalPosition: PandaStickPushTEnv.goalPosition,
            goalYaw: PandaStickPushTEnv.goalYaw)
        // The upstream UV grid has an intentional one-pixel Y asymmetry, so
        // even an exact geometric pose renders 782/825 overlapping pixels.
        XCTAssertEqual(pseudoCoverage, 0.9478788, accuracy: 1e-6)
        func translatedInGoalFrame(_ offset: SIMD2<Float>) -> SIMD2<Float> {
            let yaw = PandaStickPushTEnv.goalYaw
            return PandaStickPushTEnv.goalPosition + SIMD2(
                cos(yaw) * offset.x - sin(yaw) * offset.y,
                sin(yaw) * offset.x + cos(yaw) * offset.y)
        }
        XCTAssertEqual(PandaStickPushTEnv.officialPseudoRenderCoverage(
            blockPosition: translatedInGoalFrame(SIMD2(0.01, 0)),
            blockYaw: PandaStickPushTEnv.goalYaw,
            goalPosition: PandaStickPushTEnv.goalPosition,
            goalYaw: PandaStickPushTEnv.goalYaw), 0.8484849, accuracy: 1e-6)
        XCTAssertEqual(PandaStickPushTEnv.officialPseudoRenderCoverage(
            blockPosition: translatedInGoalFrame(SIMD2(0.005, 0.005)),
            blockYaw: PandaStickPushTEnv.goalYaw + 0.05,
            goalPosition: PandaStickPushTEnv.goalPosition,
            goalYaw: PandaStickPushTEnv.goalYaw), 0.8606061, accuracy: 1e-6)
        XCTAssertEqual(PandaStickPushTEnv.officialPseudoRenderCoverage(
            blockPosition: PandaStickPushTEnv.goalPosition + SIMD2(0.3, 0),
            blockYaw: PandaStickPushTEnv.goalYaw,
            goalPosition: PandaStickPushTEnv.goalPosition,
            goalYaw: PandaStickPushTEnv.goalYaw), 0, accuracy: 1e-6)
        XCTAssertEqual(ManiSkillPushTTask.denseReward(
            goalDistance: 0, yawError: 0,
            tcpDistance: 0, success: false), 1.05, accuracy: 1e-6)
        XCTAssertEqual(ManiSkillPushTTask.denseReward(
            goalDistance: 1, yawError: .pi,
            tcpDistance: 1, success: false), 0, accuracy: 0.001)
        XCTAssertEqual(ManiSkillPushTTask.denseReward(
            goalDistance: 1, yawError: .pi,
            tcpDistance: 1, success: true), 3, accuracy: 1e-6)
    }

    func testTUsesOneCompoundRigidActorLikeSAPIEN() throws {
        let task = try ManiSkillPushTTask(configuration: .init(
            numEnvironments: 1, seed: 9351,
            autoReset: false, robotInitialJointNoise: 0))
        let block = task.environment.refs[0].block
        XCTAssertEqual(task.environment.scene.settings.frictionCombineMode,
                       .average)
        XCTAssertEqual(task.environment.scene.settings.frictionCombineMode
            .combine(PandaStickPushTEnv.blockFriction, 0.3), 1.65,
            accuracy: 1e-6)
        let body = task.environment.scene.bodies[block]
        XCTAssertEqual(body.mass, PandaStickPushTEnv.blockMass)
        let inertia = try XCTUnwrap(body.diagonalInertia)
        XCTAssertEqual(inertia.x,
                       PandaStickPushTEnv.blockDiagonalInertia.x,
                       accuracy: 1e-8)
        XCTAssertEqual(inertia.y,
                       PandaStickPushTEnv.blockDiagonalInertia.y,
                       accuracy: 1e-8)
        XCTAssertEqual(inertia.z,
                       PandaStickPushTEnv.blockDiagonalInertia.z,
                       accuracy: 1e-8)
        let colliders = task.environment.scene.colliders.filter {
            $0.body == block
        }
        XCTAssertEqual(colliders.count, 2)
        XCTAssertFalse(task.environment.scene.joints.contains {
            $0.bodyA == block || $0.bodyB == block
        })
    }

    func testSevenAxisDeltaActionMovesPhysicalArticulation() throws {
        let task = try ManiSkillPushTTask(configuration: .init(
            numEnvironments: 1, seed: 1788,
            robotInitialJointNoise: 0))
        var observations = RLObservationBatch(spec: task.spec)
        try task.reset(environments: nil, seed: 1788,
                       into: &observations)
        let before = task.environment.states()[0]
        var action = RLActionBatch(spec: task.spec)
        action.values[0] = 1
        var result = RLStepBatch(spec: task.spec)
        for _ in 0..<4 { try task.step(actions: action, into: &result) }
        let after = task.environment.states()[0]
        XCTAssertGreaterThan(after.jointPositions[0], before.jointPositions[0])
        XCTAssertGreaterThan(length(after.tcpPosition - before.tcpPosition), 1e-4)
        XCTAssertTrue(result.observations.policy.allSatisfy(\.isFinite))
        XCTAssertTrue(result.rewards.allSatisfy(\.isFinite))
    }

    func testTerminalMetricsExposePoseFailureInsteadOfOnlyProxyScore() throws {
        let task = try ManiSkillPushTTask(configuration: .init(
            numEnvironments: 1, seed: 4796, maxEpisodeSteps: 1,
            autoReset: false, robotInitialJointNoise: 0))
        var observations = RLObservationBatch(spec: task.spec)
        try task.reset(environments: nil, seed: 4796, into: &observations)
        var result = RLStepBatch(spec: task.spec)
        try task.step(actions: RLActionBatch(spec: task.spec), into: &result)

        XCTAssertTrue(result.truncated[0])
        for name in [
            "episode/official_coverage_terminal",
            "episode/continuous_coverage_terminal",
            "episode/goal_distance_m_terminal",
            "episode/yaw_error_rad_terminal",
            "episode/tcp_distance_m_terminal",
        ] {
            let metric = try XCTUnwrap(result.metrics[name])
            XCTAssertEqual(metric.count, 1)
            XCTAssertTrue(metric[0].isFinite, "missing finite \(name)")
        }
        XCTAssertGreaterThan(
            try XCTUnwrap(result.metrics["episode/goal_distance_m_terminal"])[0],
            0)
        XCTAssertGreaterThan(
            try XCTUnwrap(result.metrics["episode/yaw_error_rad_terminal"])[0],
            0)
    }

    func testZeroDeltaDoesNotRachetGravitySagIntoPandaTarget() throws {
        let task = try ManiSkillPushTTask(configuration: .init(
            numEnvironments: 1, seed: 9351,
            robotInitialJointNoise: 0))
        var observations = RLObservationBatch(spec: task.spec)
        try task.reset(environments: nil, seed: 9351, into: &observations)
        let before = task.environment.states()[0]
        let action = RLActionBatch(spec: task.spec)
        var result = RLStepBatch(spec: task.spec)
        for _ in 0..<task.spec.maxEpisodeSteps {
            try task.step(actions: action, into: &result)
        }
        let after = task.environment.states()[0]
        XCTAssertLessThan(length(after.tcpPosition - before.tcpPosition), 0.002)
        for (actual, expected) in zip(
            after.jointPositions, before.jointPositions) {
            XCTAssertEqual(actual, expected, accuracy: 0.002)
        }
    }

    func testOfficialScaleBatchRetainsArticulationPrecision() throws {
        let task = try ManiSkillPushTTask(configuration: .init(
            numEnvironments: 1_024, seed: 9351,
            autoReset: false, robotInitialJointNoise: 0))
        var observations = RLObservationBatch(spec: task.spec)
        try task.reset(environments: nil, seed: 9351, into: &observations)
        let action = RLActionBatch(spec: task.spec)
        var result = RLStepBatch(spec: task.spec)
        for _ in 0..<task.spec.maxEpisodeSteps {
            try task.step(actions: action, into: &result)
        }
        for (environment, state) in task.environment.states().enumerated() {
            let goal = task.environment.refs[environment].goalPosition
            let blockXY = SIMD2(state.blockPosition.x, state.blockPosition.y)
            XCTAssertLessThan(length(blockXY - goal), 0.5)
            XCTAssertLessThan(length(state.blockPosition - state.tcpPosition), 1)
        }
    }
}
