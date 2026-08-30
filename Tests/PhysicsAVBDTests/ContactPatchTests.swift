import XCTest
import simd
@testable import SimCore
@testable import PhysicsAVBD

/// Contact-manifold regressions. Both were invisible in aggregate force
/// readings - the loads looked right and the resting depth was correct - and
/// only showed up as wrong *behaviour*: a grasped ball spinning freely in the
/// hand, a small capsule balanced on a single point.
final class ContactPatchTests: XCTestCase {

    // MARK: - a sphere on a face is a patch, not a point

    /// A sphere resting on a box face produces a multi-point manifold. With a
    /// single point, friction has no lever arm about the contact normal, so a
    /// two-finger grasp holds the ball on a free-spinning axle.
    func testSphereOnFaceProducesAContactPatch() throws {
        var scene = PhysicsScene(name: "patch")
        scene.settings.collisionMargin = 0.003
        scene.settings.spherePatchContacts = true
        _ = scene.addBody(size: F3(0.3, 0.3, 0.05), density: 0, friction: 0.9,
                          position: F3(0, 0, -0.025))
        let ball = scene.addSphere(diameter: 0.058, density: 900,
                                   friction: 0.9, position: F3(0, 0, 0.04))
        let solver = try GPUSolver(scene: scene)
        for _ in 0..<240 { solver.step() }

        let counts = solver.activeRigidContactCounts()
            .filter { $0.bodyA == ball || $0.bodyB == ball }
        XCTAssertEqual(counts.count, 1, "one manifold with the ground")
        XCTAssertGreaterThan(counts.first?.contacts ?? 0, 1,
                             "a resting sphere needs a contact patch, not a "
                             + "single point")
        // the patch must not change where the sphere rests
        XCTAssertEqual(solver.bodyPosition(ball).z, 0.029 - 0.003,
                       accuracy: 0.002)
    }

    /// The patch has to give friction a real moment arm: a spun ball resting
    /// on a high-friction face must be brought to a stop.
    func testContactPatchDampsSpinAboutTheContactNormal() throws {
        var scene = PhysicsScene(name: "spin")
        scene.settings.collisionMargin = 0.003
        scene.settings.spherePatchContacts = true
        _ = scene.addBody(size: F3(0.3, 0.3, 0.05), density: 0, friction: 0.9,
                          position: F3(0, 0, -0.025))
        let ball = scene.addSphere(diameter: 0.058, density: 900,
                                   friction: 0.9, position: F3(0, 0, 0.04))
        let solver = try GPUSolver(scene: scene)
        for _ in 0..<180 { solver.step() }

        let p = solver.bodyPosition(ball)
        solver.setBodyStates([.init(body: ball, position: p,
                                    rotation: solver.bodyRotation(ball),
                                    linearVelocity: .zero,
                                    angularVelocity: F3(0, 0, 20))])
        for _ in 0..<240 { solver.step() }
        // measured: 1e-7 rad/s with the patch, 7.2 rad/s without it
        XCTAssertLessThan(abs(solver.bodyAngularVelocity(ball).z), 1.0,
                          "surface friction over the patch must remove spin "
                          + "about the contact normal")
    }

    /// The patch is opt-in: a scene that does not ask for it keeps the
    /// single geometric contact point, so no existing rig changes.
    func testSpherePatchIsOffByDefault() throws {
        var scene = PhysicsScene(name: "default")
        scene.settings.collisionMargin = 0.003
        _ = scene.addBody(size: F3(0.3, 0.3, 0.05), density: 0, friction: 0.9,
                          position: F3(0, 0, -0.025))
        let ball = scene.addSphere(diameter: 0.058, density: 900,
                                   friction: 0.9, position: F3(0, 0, 0.04))
        let solver = try GPUSolver(scene: scene)
        for _ in 0..<240 { solver.step() }
        let counts = solver.activeRigidContactCounts()
            .filter { $0.bodyA == ball || $0.bodyB == ball }
        XCTAssertEqual(counts.first?.contacts, 1)
    }

    // MARK: - capsule contact dedup must scale with the capsule

    /// A short capsule lying on a box keeps more than one contact. The dedup
    /// radius was a flat 5 cm, which is longer than a small capsule, so every
    /// seeded contact collapsed onto one point and the capsule balanced on a
    /// single-point axle.
    func testShortCapsuleOnBoxKeepsMoreThanOneContact() throws {
        var scene = PhysicsScene(name: "capsule")
        scene.settings.collisionMargin = 0.003
        _ = scene.addBody(size: F3(0.4, 0.4, 0.05), density: 0, friction: 0.8,
                          position: F3(0, 0, -0.025))
        // 22 mm long, 5 mm radius: a cable link, lying along +x
        let link = scene.addBody(
            size: F3(0.022, 0.005, 0), density: 300, friction: 0.8,
            position: F3(0, 0, 0.02),
            rotation: Quat(angle: .pi / 2, axis: F3(0, 1, 0)),
            velocity: .zero, shape: .capsule)
        let solver = try GPUSolver(scene: scene)
        for _ in 0..<240 { solver.step() }

        let counts = solver.activeRigidContactCounts()
            .filter { $0.bodyA == link || $0.bodyB == link }
        XCTAssertGreaterThan(counts.first?.contacts ?? 0, 1,
                             "a capsule lying on a face rests on a line, so "
                             + "its manifold needs more than one point")
    }
}
