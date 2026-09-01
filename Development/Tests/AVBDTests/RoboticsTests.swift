import XCTest
import simd
@testable import SimCore
@testable import PhysicsAVBD
@testable import Robotics
@testable import RL

final class RoboticsTests: XCTestCase {
    func testRobotContractErrorsPreserveUserFacingDiagnostics() {
        let error = RobotContractError.invalidValueCount(
            label: "joint positions", expected: 19, actual: 18)
        XCTAssertEqual(error.localizedDescription, error.description)
    }

    func testHumanoidManipulationStateReadsPhysicalHandTipsAndBox() throws {
        let env = try HumanoidWalkEnv(
            numEnvironments: 1, seed: 40, includeProjectile: true,
            projectileDimensions: F3(0.28, 0.24, 0.50),
            projectileMass: 2, projectileFriction: 1.2,
            controlProfile: .isaacLab)
        env.placeCarryBoxes(
            environmentIDs: [0], positions: [F3(0.55, 0, 0.25)])
        let state = env.manipulationStates()[0]
        XCTAssertEqual(state.object.position.x, 0.55, accuracy: 1e-5)
        XCTAssertEqual(state.object.position.z, 0.25, accuracy: 1e-5)
        // The terminal collision spheres are bilateral and substantially
        // forward/downstream of their elbow-body COMs.
        XCTAssertGreaterThan(state.leftHand.position.y, 0)
        XCTAssertLessThan(state.rightHand.position.y, 0)
        XCTAssertGreaterThan(state.leftHand.position.x, 0.05)
        XCTAssertGreaterThan(state.rightHand.position.x, 0.05)
        XCTAssertGreaterThan(state.leftHand.position.z, 0.25)
        XCTAssertGreaterThan(state.rightHand.position.z, 0.25)
    }

    func testHumanoidResetRestoresIdenticalSolverTrajectory() throws {
        let env = try HumanoidWalkEnv(
            numEnvironments: 1, seed: 41, controlProfile: .isaacLab)
        let zero = ContiguousArray<Float>(
            repeating: 0, count: HumanoidWalkEnv.jointRanges.count)

        func rollout() -> HumanoidState {
            for _ in 0..<100 {
                env.step(normalizedActions: zero, decimation: 4,
                         clampActions: false, clampTargetsToLimits: false)
            }
            return env.states()[0]
        }

        env.reset([0], seeds: [9001], initialRollPitchRange: 0,
                  initialYawRange: 0)
        let first = rollout()
        env.reset([0], seeds: [9001], initialRollPitchRange: 0,
                  initialYawRange: 0)
        let second = rollout()

        XCTAssertLessThan(length(first.root.position - second.root.position),
                          2e-4)
        XCTAssertLessThan(length(first.root.linearVelocity
                                  - second.root.linearVelocity), 2e-3)
        XCTAssertLessThan(length(first.root.angularVelocity
                                  - second.root.angularVelocity), 2e-3)
        XCTAssertLessThan(zip(first.jointAngles, second.jointAngles).map {
            abs($0 - $1)
        }.max() ?? 0, 2e-4)
    }

    func testUnactuatedHingeDoesNotConstrainItsFreeAxis() throws {
        var scene = PhysicsScene(name: "free-hinge-axis")
        scene.settings.dt = 1 / 240
        scene.settings.gravity = 0
        scene.settings.iterations = 16
        let link = scene.addBody(
            size: F3(0.4, 0.2, 0.2), density: 1, friction: 0,
            position: .zero)
        scene.addJoint(SceneJoint(
            bodyA: -1, bodyB: link, rA: .zero, rB: .zero,
            stiffnessLin: .infinity, stiffnessAng: .infinity,
            hingeAxis: F3(0, 0, 1)))
        let gpu = try GPUSolver(scene: scene)
        gpu.setBodyStates([.init(
            body: link, position: .zero,
            rotation: Quat(real: 1, imag: .zero),
            angularVelocity: F3(0, 0, 4))])
        for _ in 0..<120 { gpu.step() }
        let state = gpu.bodyStates([link])[0]
        XCTAssertEqual(state.angularVelocity.z, 4, accuracy: 0.08,
                       "hard hinge alignment must not damp free-axis spin")
        XCTAssertLessThan(abs(state.angularVelocity.x), 0.02)
        XCTAssertLessThan(abs(state.angularVelocity.y), 0.02)
    }

    func testFixedPDServoStepResponse() throws {
        var scene = PhysicsScene(name: "fixed-pd-step")
        scene.settings.iterations = 16
        let link = scene.addBody(size: F3(1.0, 0.14, 0.14), density: 1.2,
                                 friction: 0.4, position: F3(0.5, 0, 2))
        scene.addJoint(SceneJoint(
            bodyA: -1, bodyB: link, rA: F3(0, 0, 2), rB: F3(-0.5, 0, 0),
            stiffnessLin: .infinity, stiffnessAng: .infinity,
            hingeAxis: F3(0, 1, 0), motorTarget: 0, motorTorque: 260,
            motorStiffness: 200, motorDamping: 5))
        let gpu = try GPUSolver(scene: scene)
        for _ in 0..<60 { gpu.step() }
        gpu.setMotorTarget(0, angle: 0.8)
        var maxAngle: Float = -9
        for _ in 0..<240 {
            gpu.step()
            maxAngle = max(maxAngle, gpu.motorAngle(0))
        }
        XCTAssertEqual(gpu.motorAngle(0), 0.8, accuracy: 0.04)
        XCTAssertLessThan(maxAngle - 0.8, 0.08)
    }

    func testUnsaturatedPDMatchesImplicitAndExplicitAnalyticSteps() throws {
        let initialVelocity: Float = 0.7
        let target: Float = 0.4
        let kp: Float = 20
        let kd: Float = 3

        func step(_ mode: JointMotorMode, dt: Float,
                  iterations: Int) throws -> Float {
            var scene = PhysicsScene(name: "pd-analytic-step")
            scene.settings.dt = dt
            scene.settings.gravity = 0
            scene.settings.iterations = iterations
            let link = scene.addBody(
                size: F3(repeating: 0.2), density: 0, friction: 0,
                position: .zero, mass: 1,
                diagonalInertia: F3(repeating: 1),
                collisionEnabled: false)
            scene.addJoint(SceneJoint(
                bodyA: -1, bodyB: link, rA: .zero, rB: .zero,
                stiffnessLin: .infinity, stiffnessAng: .infinity,
                hingeAxis: F3(0, 0, 1), motorTarget: target,
                motorTorque: 100, motorStiffness: kp,
                motorDamping: kd, motorMode: mode))
            let gpu = try GPUSolver(scene: scene)
            gpu.setBodyStates([.init(
                body: link, position: .zero,
                rotation: Quat(real: 1, imag: .zero),
                angularVelocity: F3(0, 0, initialVelocity))])
            gpu.step()
            return gpu.bodyStates([link])[0].angularVelocity.z
        }

        for dt: Float in [0.01, 0.002] {
            let implicitExpected = (initialVelocity + kp * dt * target)
                / (1 + kd * dt + kp * dt * dt)
            let explicitExpected = initialVelocity
                + (kp * target - kd * initialVelocity) * dt
            for iterations in [1, 12] {
                XCTAssertEqual(
                    try step(.implicitPositionPD, dt: dt,
                             iterations: iterations),
                    implicitExpected, accuracy: 4e-4)
                XCTAssertEqual(
                    try step(.explicitTorquePD, dt: dt,
                             iterations: iterations),
                    explicitExpected, accuracy: 4e-4)
            }
        }
    }

    func testPDTorqueLimitMatchesOneStepImpulseAcrossModesAndIterations() throws {
        for mode in [JointMotorMode.implicitPositionPD, .explicitTorquePD] {
            for iterations in [1, 2, 12] {
                var scene = PhysicsScene(name: "pd-effort-limit")
                scene.settings.dt = 0.005
                scene.settings.gravity = 0
                scene.settings.iterations = iterations
                let link = scene.addBody(
                    size: F3(repeating: 0.2), density: 0, friction: 0,
                    position: .zero, mass: 1,
                    diagonalInertia: F3(repeating: 1),
                    collisionEnabled: false)
                scene.addJoint(SceneJoint(
                    bodyA: -1, bodyB: link, rA: .zero, rB: .zero,
                    stiffnessLin: .infinity, stiffnessAng: .infinity,
                    hingeAxis: F3(0, 0, 1), motorTarget: 1,
                    motorTorque: 10, motorStiffness: 200,
                    motorDamping: 5, motorMode: mode))
                let gpu = try GPUSolver(scene: scene)

                gpu.step()

                // The request exceeds 10 Nm, so delta omega must be exactly
                // tau * dt / I = 0.05 rad/s. The active clamp has zero
                // derivative and therefore cannot depend on solver sweeps.
                let state = gpu.bodyStates([link])[0]
                XCTAssertEqual(
                    state.angularVelocity.z, 0.05, accuracy: 3e-4,
                    "mode=\(mode), iterations=\(iterations)")
            }
        }
    }

    func testVelocityMotorAppliesBoundedPhysicalEffort() throws {
        var scene = PhysicsScene(name: "velocity-effort-limit")
        scene.settings.dt = 0.005
        scene.settings.gravity = 0
        scene.settings.iterations = 1
        let link = scene.addBody(
            size: F3(repeating: 0.2), density: 0, friction: 0,
            position: .zero, mass: 1,
            diagonalInertia: F3(repeating: 1), collisionEnabled: false)
        scene.addJoint(SceneJoint(
            bodyA: -1, bodyB: link, rA: .zero, rB: .zero,
            stiffnessLin: .infinity, stiffnessAng: .infinity,
            hingeAxis: F3(0, 0, 1), motorTarget: 4, motorTorque: 10,
            motorDamping: 20, motorMode: .velocity))
        let gpu = try GPUSolver(scene: scene)

        gpu.step()

        XCTAssertEqual(gpu.bodyStates([link])[0].angularVelocity.z,
                       0.05, accuracy: 3e-4)
        gpu.setMotorVelocity(0, radiansPerSecond: -4)
        gpu.step()
        XCTAssertEqual(gpu.bodyStates([link])[0].angularVelocity.z,
                       0, accuracy: 6e-4)
    }

    func testVelocityMotorGainHasPhysicalUnitsAndIgnoresAngle() throws {
        let initialVelocity: Float = 0.7
        let targetVelocity: Float = 2
        let kd: Float = 3

        for dt: Float in [0.01, 0.002] {
            let expected = initialVelocity
                + kd * (targetVelocity - initialVelocity) * dt
            for initialAngle: Float in [0, 1.2] {
                for iterations in [1, 12] {
                    var scene = PhysicsScene(name: "velocity-analytic-step")
                    scene.settings.dt = dt
                    scene.settings.gravity = 0
                    scene.settings.iterations = iterations
                    let link = scene.addBody(
                        size: F3(repeating: 0.2), density: 0,
                        friction: 0, position: .zero, mass: 1,
                        diagonalInertia: F3(repeating: 1),
                        collisionEnabled: false)
                    scene.addJoint(SceneJoint(
                        bodyA: -1, bodyB: link, rA: .zero, rB: .zero,
                        stiffnessLin: .infinity,
                        stiffnessAng: .infinity,
                        hingeAxis: F3(0, 0, 1),
                        motorTarget: targetVelocity, motorTorque: 100,
                        motorDamping: kd, motorMode: .velocity))
                    let gpu = try GPUSolver(scene: scene)
                    gpu.setBodyStates([.init(
                        body: link, position: .zero,
                        rotation: Quat(
                            angle: initialAngle, axis: F3(0, 0, 1)),
                        angularVelocity: F3(0, 0, initialVelocity))])

                    gpu.step()

                    XCTAssertEqual(
                        gpu.bodyStates([link])[0].angularVelocity.z,
                        expected, accuracy: 4e-4,
                        "dt=\(dt), angle=\(initialAngle), "
                            + "iterations=\(iterations)")
                }
            }
        }
    }

    func testMotorModesReplayIdenticallyAfterInPlaceReset() throws {
        for mode in [JointMotorMode.implicitPositionPD,
                     .explicitTorquePD, .velocity] {
            var scene = PhysicsScene(name: "motor-reset-replay")
            scene.settings.dt = 0.005
            scene.settings.gravity = 0
            scene.settings.iterations = 4
            let link = scene.addBody(
                size: F3(repeating: 0.2), density: 0, friction: 0,
                position: .zero, mass: 1,
                diagonalInertia: F3(repeating: 1),
                collisionEnabled: false)
            let velocityMode = mode == .velocity
            scene.addJoint(SceneJoint(
                bodyA: -1, bodyB: link, rA: .zero, rB: .zero,
                stiffnessLin: .infinity, stiffnessAng: .infinity,
                hingeAxis: F3(0, 0, 1),
                motorTarget: velocityMode ? 1 : 0.4,
                motorTorque: 100,
                motorStiffness: velocityMode ? 0 : 20,
                motorDamping: 3, motorMode: mode))
            let gpu = try GPUSolver(scene: scene)
            let initial = GPUSolver.BodyStateUpdate(
                body: link, position: .zero,
                rotation: Quat(angle: 0.1, axis: F3(0, 0, 1)),
                angularVelocity: F3(0, 0, 0.3))

            func rollout() -> [(angle: Float, velocity: Float)] {
                (0..<30).map { _ in
                    gpu.step()
                    return gpu.motorStates([0])[0]
                }
            }

            gpu.setBodyStates([initial])
            let first = rollout()
            gpu.setBodyStates([initial])
            let second = rollout()
            for (a, b) in zip(first, second) {
                XCTAssertEqual(a.angle, b.angle, accuracy: 1e-6,
                               "mode=\(mode)")
                XCTAssertEqual(a.velocity, b.velocity, accuracy: 1e-6,
                               "mode=\(mode)")
            }
        }
    }

    func testJointLimitsHold() throws {
        var scene = PhysicsScene(name: "lim")
        scene.settings.iterations = 16
        let link = scene.addBody(size: F3(1.0, 0.14, 0.14), density: 1.2, friction: 0.4,
                                 position: F3(0.5, 0, 2))
        scene.addJoint(SceneJoint(bodyA: -1, bodyB: link,
                                  rA: F3(0, 0, 2), rB: F3(-0.5, 0, 0),
                                  stiffnessLin: .infinity, stiffnessAng: .infinity,
                                  hingeAxis: F3(0, 1, 0),
                                  motorTarget: 0, motorTorque: 500,
                                  motorStiffness: 200, motorDamping: 5,
                                  limitLo: -0.3, limitHi: 0.6))
        let gpu = try GPUSolver(scene: scene)
        gpu.setMotorTarget(0, angle: 2.0)        // beyond the limit
        for _ in 0..<300 { gpu.step() }
        XCTAssertLessThan(gpu.motorAngle(0), 0.75, "limit must hold against the motor")
    }

    func testParallelEnvsIsolatedAndLive() throws {
        let env = try PushTEnv(numEnvs: 16, seed: 3)
        let acts = (0..<16).map { _ in SIMD2<Float>(1.2, 0.5) }
        for _ in 0..<80 { env.step(actions: acts) }
        for e in 0..<16 {
            XCTAssertLessThan(length(env.tipPos(e) - SIMD2<Float>(1.2, 0.5)), 0.6)
            let (bp, _) = env.blockPose(e)
            XCTAssertLessThan(max(abs(bp.x), abs(bp.y)), 3.4)
        }
    }

    func testInPlaceResetRandomizesAndIsolates() throws {
        let env = try PushTEnv(numEnvs: 8, seed: 3)
        let acts = (0..<8).map { _ in SIMD2<Float>(1.4, 0.6) }
        for _ in 0..<60 { env.step(actions: acts) }
        let before = (0..<8).map { env.blockPose($0).0 }
        for e in 0..<4 { env.reset(e, seed: UInt64(100 + e)) }
        for _ in 0..<30 { env.step(actions: acts) }
        var changed = 0
        for e in 0..<4 where length(env.blockPose(e).0 - before[e]) > 0.3 { changed += 1 }
        XCTAssertGreaterThanOrEqual(changed, 3, "reset must re-randomize")
        for e in 4..<8 {
            XCTAssertLessThan(length(env.blockPose(e).0 - before[e]), 0.3,
                              "reset must not disturb other envs")
        }
        // KNOWN ISSUE (tracked): post-reset large yaw swings can still jam;
        // tracked as the closed-loop tip-servo work item.
    }

    func testObservationsRenderAllChannels() throws {
        let env = try PushTEnv(numEnvs: 3, seed: 5)
        for _ in 0..<20 { env.step(actions: (0..<3).map { _ in SIMD2(1.0, 0.4) }) }
        let obs = env.observations()
        let res = env.obsRes
        for e in 0..<3 {
            var red = 0, green = 0, blue = 0
            for i in 0..<(res * res) {
                let o = (e * res * res + i) * 3
                if obs[o] > 150 && obs[o + 1] < 100 { red += 1 }
                if obs[o + 1] > 100 && obs[o] < 100 && obs[o + 2] < 100 { green += 1 }
                if obs[o + 2] > 150 && obs[o] < 120 { blue += 1 }
            }
            XCTAssertGreaterThan(red, 20)
            XCTAssertGreaterThan(green, 20)
            XCTAssertGreaterThan(blue, 3)
        }
    }
}
