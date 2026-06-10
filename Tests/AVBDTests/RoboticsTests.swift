import XCTest
import simd
@testable import AVBDCore

final class RoboticsTests: XCTestCase {
    func testServoStepResponseNoOvershoot() throws {
        var scene = PhysicsScene(name: "step")
        scene.settings.iterations = 16
        let link = scene.addBody(size: F3(1.0, 0.14, 0.14), density: 1.2, friction: 0.4,
                                 position: F3(0.5, 0, 2))
        scene.addJoint(SceneJoint(bodyA: -1, bodyB: link,
                                  rA: F3(0, 0, 2), rB: F3(-0.5, 0, 0),
                                  stiffnessLin: .infinity, stiffnessAng: .infinity,
                                  hingeAxis: F3(0, 1, 0),
                                  motorTarget: 0, motorTorque: 260))
        let gpu = try GPUSolver(scene: scene)
        for _ in 0..<60 { gpu.step() }
        gpu.setMotorTarget(0, angle: 0.8)
        var maxA: Float = -9
        for _ in 0..<240 { gpu.step(); maxA = max(maxA, gpu.motorAngle(0)) }
        XCTAssertEqual(gpu.motorAngle(0), 0.8, accuracy: 0.03)
        XCTAssertLessThan(maxA - 0.8, 0.05, "damped servo must not overshoot")
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
