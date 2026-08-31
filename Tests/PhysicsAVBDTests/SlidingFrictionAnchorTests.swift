import XCTest
import simd
@testable import SimCore
@testable import PhysicsAVBD

/// Kinetic friction must dissipate, not store: with the stick flag only
/// refreshed in-cone, a once-stuck contact kept stale anchors through a
/// slide and crept the object back to its touchdown point after release.
final class SlidingFrictionAnchorTests: XCTestCase {

    /// Push a resting cube with a compliant probe, retract, wait: the cube
    /// must stay put. The probe must be compliant and force-capped - a
    /// pose-teleported pusher does not reproduce the bug.
    func testPushedCubeStaysWhereFrictionStoppedIt() throws {
        var scene = PhysicsScene(name: "slide-anchor")
        scene.settings.dt = 1.0 / 240
        scene.settings.iterations = 20
        scene.settings.betaLin = 20000
        scene.settings.betaAng = 500
        scene.settings.lambdaMax = 1200
        scene.settings.collisionMargin = 0.003
        scene.settings.gravity = -9.81
        _ = scene.addBody(size: F3(2, 2, 0.3), density: 0, friction: 1.0,
                          position: F3(0, 0, -0.15))

        let cubeSize = F3(0.11, 0.032, 0.030)
        let cube = scene.addBody(size: cubeSize, density: 420,
                                 friction: 0.30, dynamicFriction: 0.25,
                                 position: F3(0, 0, cubeSize.z / 2 + 0.002),
                                 rotation: Quat(real: 1, imag: .zero),
                                 velocity: .zero, shape: .box)

        // gravity-free probe on a soft world tether, ~12 N force cap
        let strike = F3(0, -0.12, 0.017)
        let probe = scene.addBody(size: F3(repeating: 0.028), density: 3000,
                                  friction: 0.2, position: strike,
                                  rotation: Quat(real: 1, imag: .zero),
                                  velocity: .zero, shape: .box,
                                  mass: nil, diagonalInertia: nil,
                                  gravityScale: 0)
        let driver = scene.joints.count
        scene.addJoint(SceneJoint(bodyA: -1, bodyB: probe,
                                  rA: strike, rB: .zero,
                                  stiffnessLin: 3000, stiffnessAng: 0))

        let solver = try GPUSolver(scene: scene)
        for _ in 0..<120 { try solver.submitStep() }
        try solver.synchronize()

        let dir = F3(0, 1, 0)
        let maxOffset: Float = 12.0 / 3000
        var pushedY: Float = 0
        for tick in 0...700 {
            let t = min(Float(tick) / 420, 1)
            var goal = strike + dir * (0.16 * t)
            if tick > 460 {
                goal = strike - dir * 0.15      // retract clear of everything
            }
            let probePos = solver.bodyPosition(probe)
            var delta = goal - probePos
            let l = simd_length(delta)
            if l > maxOffset { delta *= maxOffset / l }
            solver.setJointWorldAnchor(driver, point: probePos + delta)
            for _ in 0..<2 { try solver.submitStep() }
            try solver.synchronize()
            if tick == 460 { pushedY = solver.bodyPosition(cube).y }
        }

        let finalY = solver.bodyPosition(cube).y
        XCTAssertGreaterThan(pushedY, 0.05,
                             "the probe should have pushed the cube")
        // unfixed: creeps from ~0.067 back to ~0.001
        XCTAssertGreaterThan(finalY, pushedY - 0.015,
                             "a released object must stay where friction "
                             + "stopped it, not creep back to where it "
                             + "first touched down")
        XCTAssertLessThan(
            simd_length(solver.bodyVelocity(cube)), 0.02,
            "and it must be at rest, not still being dragged")
    }

    /// Same invariant on the CPU backend.
    func testPushedBoxStaysPutOnCPUBackend() throws {
        let solver = CPUSolver()
        solver.dt = 1.0 / 240
        solver.iterations = 20
        solver.betaLin = 20000
        solver.collisionMargin = 0.003
        solver.gravity = -9.81
        _ = solver.addBody(size: F3(2, 2, 0.3), density: 0,
                           friction: 1.0, position: F3(0, 0, -0.15))
        let cube = solver.addBody(size: F3(0.11, 0.032, 0.030), density: 420,
                                  friction: 0.30,
                                  position: F3(0, 0, 0.017))

        for _ in 0..<120 { solver.step() }
        let rest = cube.positionLin

        // shove sideways through velocity for a while, then release
        for _ in 0..<240 {
            cube.velocityLin.y = max(cube.velocityLin.y, 0.10)
            solver.step()
        }
        let pushed = cube.positionLin.y - rest.y
        for _ in 0..<480 { solver.step() }
        let finalY = cube.positionLin.y - rest.y

        XCTAssertGreaterThan(pushed, 0.02, "the shove should have moved it")
        XCTAssertGreaterThan(finalY, pushed - 0.01,
                             "no creep back toward the touchdown point")
    }
}
