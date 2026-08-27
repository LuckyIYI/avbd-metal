import XCTest
import simd
@testable import SimCore
@testable import PhysicsAVBD

/// Kinetic friction must be MEMORYLESS: an object pushed along a surface
/// stays where the push leaves it.
///
/// The bug this pins down: the contact dual update only refreshed the
/// stick flag in its within-cone branch, so a contact that stuck once and
/// then started sliding kept its original warm-start anchors for ever. The
/// tangential constraint accumulated the whole slide, the applied force
/// stayed cone-capped (so pushing still "worked"), and when the push
/// stopped, that capped force crept the body all the way back to its
/// first touchdown point - an invisible rubber band to spawn. Lifting
/// breaks contact and resets anchors, which is why every pick-and-place
/// test passed while the first pure planar-pushing task failed in minutes.
final class SlidingFrictionTests: XCTestCase {

    private func pushRetractScene() -> (PhysicsScene, box: Int, probe: Int,
                                        driver: Int, start: F3) {
        var scene = PhysicsScene(name: "slide-memoryless")
        scene.settings.dt = 1.0 / 240
        scene.settings.iterations = 20
        scene.settings.betaLin = 20000
        scene.settings.betaAng = 500
        scene.settings.lambdaMax = 1200
        scene.settings.collisionMargin = 0.003
        scene.settings.gravity = -9.81
        _ = scene.addBody(size: F3(2, 2, 0.3), density: 0, friction: 1.0,
                          position: F3(0, 0, -0.15))
        let box = scene.addBody(size: F3(0.11, 0.032, 0.030), density: 420,
                                friction: 0.30, dynamicFriction: 0.25,
                                position: F3(0, 0, 0.017),
                                rotation: Quat(real: 1, imag: .zero),
                                velocity: .zero, shape: .box)
        let start = F3(0, -0.12, 0.017)
        let probe = scene.addBody(size: F3(repeating: 0.028), density: 3000,
                                  friction: 0.2, position: start,
                                  rotation: Quat(real: 1, imag: .zero),
                                  velocity: .zero, shape: .box,
                                  mass: nil, diagonalInertia: nil,
                                  gravityScale: 0)
        let driver = scene.joints.count
        scene.addJoint(SceneJoint(bodyA: -1, bodyB: probe, rA: start,
                                  rB: .zero, stiffnessLin: 3000,
                                  stiffnessAng: 0))
        return (scene, box, probe, driver, start)
    }

    /// Drive the probe into the box, slide it ~7 cm, retract the probe, let
    /// everything settle. The box must KEEP most of its displacement; the
    /// bug pulled it back to within a millimetre of its spawn.
    func testPushedBoxStaysPut_GPU() throws {
        let (scene, box, probe, driver, start) = pushRetractScene()
        let gpu = try GPUSolver(scene: scene)
        for _ in 0..<120 { try gpu.submitStep() }
        try gpu.synchronize()
        let maxOffset: Float = 12.0 / 3000
        for tick in 0...700 {
            let t = min(Float(tick) / 420, 1)
            var goal = start + F3(0, 0.16 * t, 0)
            if tick > 460 { goal = start - F3(0, 0.15, 0) }   // retract clear
            let p = gpu.bodyPosition(probe)
            var delta = goal - p
            let l = simd_length(delta)
            if l > maxOffset { delta *= maxOffset / l }
            gpu.setJointWorldAnchor(driver, point: p + delta)
            for _ in 0..<2 { try gpu.submitStep() }
            try gpu.synchronize()
        }
        let displaced = gpu.bodyPosition(box).y
        XCTAssertGreaterThan(displaced, 0.04,
            "pushed box slid back toward its spawn anchor - kinetic friction "
            + "is acting as an elastic tether, not a dissipative force")
    }
}
