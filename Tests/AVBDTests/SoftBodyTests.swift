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
        var scene = PhysicsScene(name: "ballonblock")
        scene.settings.iterations = 20
        scene.settings.betaLin = 20000
        scene.settings.lambdaMax = 800
        Demos.addGround(&scene)
        _ = Demos.addSoftBlock(&scene, center: F3(0, 0, 1.2), nx: 4, ny: 4, nz: 4,
                               spacing: 0.34, mu: 4000, lambda: 40000)
        let ball = scene.addSphere(diameter: 1.1, density: 3, friction: 0.6,
                                   position: F3(0, 0, 4.2))
        let gpu = try GPUSolver(scene: scene)
        for _ in 0..<500 { gpu.step() }
        XCTAssertGreaterThan(gpu.bodyPosition(ball).z, 0.8,
                             "heavy rigid ball must rest ON the soft block (two-way coupling)")
    }

    func testBallRestsOnSoftBunny() throws {
        let scene = Demos.softbody(res: 10)
        var ball = -1
        for (i, b) in scene.bodies.enumerated()
            where b.shape == .sphere && b.size.x > 0.4 { ball = i }
        XCTAssertGreaterThanOrEqual(ball, 0)
        let gpu = try GPUSolver(scene: scene)
        for _ in 0..<500 { gpu.step() }
        XCTAssertGreaterThan(gpu.bodyPosition(ball).z, 0.55,
                             "ball must rest on the bunny's back, not pass to the floor")
    }

    func testSkinnedSoftMeshUploadsAndSteps() throws {
        let p: [F3] = [
            F3(-0.5, -0.5, 0), F3(0.5, -0.5, 0), F3(0.5, 0.5, 0), F3(-0.5, 0.5, 0),
            F3(-0.5, -0.5, 1), F3(0.5, -0.5, 1), F3(0.5, 0.5, 1), F3(-0.5, 0.5, 1)
        ]
        let tris = [
            (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
            (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)
        ]
        let mesh = SurfaceMesh(vertices: p, normals: [], triangles: tris)
        var scene = PhysicsScene(name: "skin-cube")
        Demos.addGround(&scene)
        _ = Demos.addSkinnedSoftMesh(&scene, mesh: mesh, center: F3(0, 0, 0.08),
                                     height: 1.0, res: 3, visualVertexLimit: 64,
                                     minTetElements: 1,
                                     mu: 1800)
        XCTAssertEqual(scene.skinnedMeshes.count, 1)
        XCTAssertEqual(scene.skinnedMeshes[0].triangles.count, tris.count)
        XCTAssertGreaterThan(scene.tets.count, 0)

        let gpu = try GPUSolver(scene: scene)
        XCTAssertNotNil(gpu.renderSkinnedSurface)
        for _ in 0..<5 { gpu.step() }
        XCTAssertTrue(gpu.bodyPosition(1).z.isFinite)
    }

    func testMeshSoftBodiesDropOnPinnedClothBuilds() throws {
        let scene = Demos.meshclothdrop(res: 10, scale: 1)
        XCTAssertGreaterThanOrEqual(scene.skinnedMeshes.count, 2)
        XCTAssertGreaterThan(scene.tris.count, 0)
        for mesh in scene.skinnedMeshes {
            let bodies = Set(mesh.bodyIDs)
            let tets = scene.tets.filter {
                bodies.contains($0.ids.0) && bodies.contains($0.ids.1)
                    && bodies.contains($0.ids.2) && bodies.contains($0.ids.3)
            }
            XCTAssertGreaterThanOrEqual(tets.count, 600)
        }
        let pinned = scene.joints.filter {
            $0.bodyA == -1 && scene.bodies[$0.bodyB].isParticle
                && $0.stiffnessLin.isInfinite
        }
        XCTAssertGreaterThanOrEqual(pinned.count, 4)
        XCTAssertTrue(scene.bodies.contains { !$0.isParticle && $0.density > 0 })

        let gpu = try GPUSolver(scene: scene)
        XCTAssertNotNil(gpu.renderSkinnedSurface)
        for _ in 0..<8 { gpu.step() }
        XCTAssertTrue(gpu.maxConstraintError().isFinite)
    }

    func testSkinnedBunnyDropsIntoRigidBox() throws {
        let scene = Demos.skinnedbunny(res: 8, count: 2)
        XCTAssertEqual(scene.skinnedMeshes.count, 2)
        let staticBoxes = scene.bodies.filter { !$0.isParticle && $0.density == 0 && $0.shape == .box }
        XCTAssertGreaterThanOrEqual(staticBoxes.count, 6)
        for mesh in scene.skinnedMeshes {
            let bodies = Set(mesh.bodyIDs)
            let tets = scene.tets.filter {
                bodies.contains($0.ids.0) && bodies.contains($0.ids.1)
                    && bodies.contains($0.ids.2) && bodies.contains($0.ids.3)
            }
            XCTAssertGreaterThanOrEqual(tets.count, 1500)
        }
        let gpu = try GPUSolver(scene: scene)
        XCTAssertNotNil(gpu.renderSkinnedSurface)
        for _ in 0..<5 { gpu.step() }
        XCTAssertTrue(gpu.maxConstraintError().isFinite)
    }

    func testSoftBlocksDontInterpenetrate() throws {
        // Soft-soft contact is element-based (soft V-V sphere pairs are
        // banned at broadphase): without tet BOUNDARY faces as collision
        // triangles the two blocks pass straight through each other.
        var scene = PhysicsScene(name: "sbsb")
        scene.settings.iterations = 20
        Demos.addGround(&scene)
        _ = Demos.addSoftBlock(&scene, center: F3(0, 0, 0.7), nx: 4, ny: 4, nz: 4,
                               spacing: 0.34, mu: 4000, lambda: 40000,
                               massPerNode: 0.05)
        let top = Demos.addSoftBlock(&scene, center: F3(0.15, 0.1, 2.3),
                                     nx: 3, ny: 3, nz: 3, spacing: 0.34,
                                     mu: 4000, lambda: 40000, massPerNode: 0.05)
        let gpu = try GPUSolver(scene: scene)
        for _ in 0..<400 { gpu.step() }
        var topMinZ: Float = .greatestFiniteMagnitude
        var com = F3(0, 0, 0)
        for i in top {
            let p = gpu.bodyPosition(i)
            topMinZ = min(topMinZ, p.z)
            com += p
        }
        com /= Float(top.count)
        XCTAssertGreaterThan(topMinZ, 0.95,
                             "top soft block must rest ON the bottom block, not sink into it")
        XCTAssertGreaterThan(com.z, 1.15,
                             "top soft block must stay stacked, not slide through to the floor")
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
