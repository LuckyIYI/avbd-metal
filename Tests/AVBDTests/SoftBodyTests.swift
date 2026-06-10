import XCTest
import simd
@testable import AVBDCore

/// Soft body physics per the paper: 3-DOF particles, hard rod elements,
/// Neo-Hookean tets, and two-way rigid coupling.
final class SoftBodyTests: XCTestCase {

    func testParticleRodChainInextensible() throws {
        var scene = PhysicsScene(name: "pc")
        scene.settings.iterations = 20
        var prev = -1
        for k in 0..<12 {
            let p = scene.addParticle(radius: 0.05, mass: 0.02,
                                      position: F3(Float(k) * 0.25, 0, 4))
            if k == 0 {
                scene.addJoint(SceneJoint(bodyA: -1, bodyB: p, rA: F3(0, 0, 4), rB: .zero))
            } else {
                scene.addSpring(SceneSpring(bodyA: prev, bodyB: p, rA: .zero, rB: .zero,
                                            stiffness: 1e5, hard: true))
            }
            prev = p
        }
        let gpu = try GPUSolver(scene: scene)
        for _ in 0..<300 { gpu.step() }
        for k in 1..<12 {
            let d = distance(gpu.bodyPosition(k - 1), gpu.bodyPosition(k))
            XCTAssertEqual(d, 0.25, accuracy: 0.01, "hard rods must stay inextensible")
        }
        XCTAssertEqual(gpu.bodyRotation(5).real, 1.0, accuracy: 1e-4,
                       "particles carry no rotational state")
    }

    func testTetVolumePreservation() throws {
        var scene = PhysicsScene(name: "tet1")
        scene.settings.iterations = 20
        let p0 = scene.addParticle(radius: 0.05, mass: 0.1, position: F3(0, 0, 3))
        let p1 = scene.addParticle(radius: 0.05, mass: 0.1, position: F3(0.5, 0, 3))
        let p2 = scene.addParticle(radius: 0.05, mass: 0.1, position: F3(0, 0.5, 3))
        let p3 = scene.addParticle(radius: 0.05, mass: 0.1, position: F3(0, 0, 3.5))
        scene.addJoint(SceneJoint(bodyA: -1, bodyB: p0, rA: F3(0, 0, 3), rB: .zero))
        scene.addTet(SceneTet(ids: (p0, p1, p2, p3), mu: 2000, lambda: 20000))
        let gpu = try GPUSolver(scene: scene)
        for _ in 0..<300 { gpu.step() }
        let a = gpu.bodyPosition(p0), b = gpu.bodyPosition(p1)
        let c = gpu.bodyPosition(p2), d = gpu.bodyPosition(p3)
        let v = abs(dot(b - a, cross(c - a, d - a))) / 6
        XCTAssertEqual(v * 6 / (0.5 * 0.5 * 0.5), 1.0, accuracy: 0.05,
                       "Neo-Hookean tet must preserve volume")
    }

    func testRigidRestsOnSoftBlock() throws {
        let scene = Demos.softbody(count: 3)
        var ball = -1
        for (i, b) in scene.bodies.enumerated()
            where b.shape == .sphere && b.size.x > 1.0 { ball = i }
        let gpu = try GPUSolver(scene: scene)
        for _ in 0..<500 { gpu.step() }
        XCTAssertGreaterThan(gpu.bodyPosition(ball).z, 0.8,
                             "heavy rigid ball must rest ON the soft block (two-way coupling)")
    }

    func testFlagStaysAttachedToPole() throws {
        let scene = Demos.cloth(res: 14, ball: false)
        var attach = -1
        for j in scene.joints
            where j.bodyA >= 0 && scene.bodies[j.bodyB].isParticle
            && !scene.bodies[j.bodyA].isParticle && j.stiffnessLin.isInfinite {
            attach = j.bodyB
            break
        }
        let gpu = try GPUSolver(scene: scene)
        for _ in 0..<500 { gpu.step() }
        XCTAssertGreaterThan(gpu.bodyPosition(attach).z, 3.5,
                             "flag must stay hard-attached high on the deformable pole")
    }
}
