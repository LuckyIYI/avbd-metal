import XCTest
import simd
@testable import AVBDCore

final class RoboticsTests: XCTestCase {
    func testServoTracksTargets() throws {
        var scene = PhysicsScene(name: "motor")
        scene.settings.iterations = 20
        let link = scene.addCapsule(length: 1.4, radius: 0.08, density: 2,
                                    friction: 0.5, position: F3(0, 0, 2.0))
        scene.addJoint(SceneJoint(bodyA: -1, bodyB: link,
                                  rA: F3(0, 0, 1.2), rB: F3(0, 0, -0.8),
                                  stiffnessLin: .infinity, stiffnessAng: .infinity,
                                  hingeAxis: F3(0, 1, 0),
                                  motorTarget: 0, motorTorque: 60))
        let gpu = try GPUSolver(scene: scene)
        for target in [Float(0.8), -1.2] {
            gpu.setMotorTarget(0, angle: target)
            for _ in 0..<240 { gpu.step() }
            XCTAssertEqual(gpu.motorAngle(0), target, accuracy: 0.05)
        }
    }

    func testParallelEnvsIsolatedAndLive() throws {
        let env = try PushTEnv(numEnvs: 16, seed: 3)
        var acts = (0..<16).map { _ in SIMD2<Float>(1.2, 0.5) }
        for _ in 0..<80 { env.step(actions: acts) }
        // tips track; blocks stay inside their own env fences
        for e in 0..<16 {
            let t = env.tipPos(e)
            XCTAssertLessThan(length(t - SIMD2<Float>(1.2, 0.5)), 0.6)
            let (bp, _) = env.blockPose(e)
            XCTAssertLessThan(max(abs(bp.x), abs(bp.y)), 3.4,
                              "block must stay in its env")
        }
        _ = acts
        acts = (0..<16).map { _ in SIMD2<Float>(-1.0, -0.8) }
        for _ in 0..<80 { env.step(actions: acts) }
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
            XCTAssertGreaterThan(red, 20, "block visible")
            XCTAssertGreaterThan(green, 20, "goal visible")
            XCTAssertGreaterThan(blue, 3, "tip visible")
        }
    }
}
