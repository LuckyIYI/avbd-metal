import XCTest
import CryptoKit
import simd
@testable import SimCore
@testable import PhysicsAVBD

/// Contact-manifold regressions. Both were invisible in aggregate force
/// readings - the loads looked right and the resting depth was correct - and
/// only showed up as wrong *behaviour*: a grasped ball spinning freely in the
/// hand, a small capsule balanced on a single point.
final class ContactPatchTests: XCTestCase {

    private func sphereBoxScene(patchContacts: Bool) -> (PhysicsScene, Int) {
        var scene = PhysicsScene(name: "sphere-box-patch")
        scene.settings.collisionMargin = 0.003
        scene.settings.spherePatchContacts = patchContacts
        _ = scene.addBody(size: F3(0.3, 0.3, 0.05), density: 0, friction: 0.9,
                          position: F3(0, 0, -0.025))
        let ball = scene.addSphere(diameter: 0.058, density: 900,
                                   friction: 0.9, position: F3(0, 0, 0.04))
        return (scene, ball)
    }

    private func convexPadScene(
        patchContacts: Bool = true,
        sphereFirst: Bool = false,
        spherePosition: F3 = F3(0, 0, 0.026)
    ) -> (PhysicsScene, Int, Int) {
        var scene = PhysicsScene(name: "convex-pad-patch")
        scene.settings.gravity = 0
        scene.settings.collisionMargin = 0.003
        scene.settings.spherePatchContacts = patchContacts
        let half = F3(0.15, 0.15, 0.025)
        let vertices = [
            F3(-half.x, -half.y, -half.z), F3(half.x, -half.y, -half.z),
            F3(-half.x, half.y, -half.z), F3(half.x, half.y, -half.z),
            F3(-half.x, -half.y, half.z), F3(half.x, -half.y, half.z),
            F3(-half.x, half.y, half.z), F3(half.x, half.y, half.z),
        ]
        func addPad() -> Int {
            let pad = scene.addBody(
                size: F3(0.3, 0.3, 0.05), density: 0, friction: 0.9,
                position: F3(0, 0, -0.025), collisionEnabled: false)
            _ = scene.addConvexCollider(
                body: pad, vertices: vertices, friction: 0.9)
            return pad
        }
        let pad: Int
        let ball: Int
        if sphereFirst {
            ball = scene.addSphere(
                diameter: 0.058, density: 900, friction: 0.9,
                position: spherePosition)
            pad = addPad()
        } else {
            pad = addPad()
            ball = scene.addSphere(
                diameter: 0.058, density: 900, friction: 0.9,
                position: spherePosition)
        }
        return (scene, pad, ball)
    }

    private func cpuManifold(
        _ solver: CPUSolver, bodies: Set<Int>
    ) -> CPUManifold? {
        solver.forces.compactMap { $0 as? CPUManifold }.first {
            guard let a = $0.bodyA?.index, let b = $0.bodyB?.index else {
                return false
            }
            return Set([a, b]) == bodies
        }
    }

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
        func residualSpin(patchContacts: Bool) throws -> Float {
            let (scene, ball) = sphereBoxScene(patchContacts: patchContacts)
            let solver = try GPUSolver(scene: scene)
            for _ in 0..<180 { solver.step() }

            let p = solver.bodyPosition(ball)
            solver.setBodyStates([.init(body: ball, position: p,
                                        rotation: solver.bodyRotation(ball),
                                        linearVelocity: .zero,
                                        angularVelocity: F3(0, 0, 20))])
            for _ in 0..<240 { solver.step() }
            return abs(solver.bodyAngularVelocity(ball).z)
        }

        let pointResidual = try residualSpin(patchContacts: false)
        let patchResidual = try residualSpin(patchContacts: true)
        XCTAssertGreaterThan(pointResidual, 5,
                             "the control must expose the missing torsional "
                             + "friction of a point contact")
        XCTAssertLessThan(patchResidual, 1.0,
                          "surface friction over the patch must remove spin "
                          + "about the contact normal")
        XCTAssertLessThan(patchResidual, pointResidual * 0.2)
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

    /// PhysicsScene conversion authors explicit CPUCollider records. The public
    /// setting must reach that ordinary path, not only the legacy implicit-body
    /// fallback used by direct CPUSolver callers.
    func testSceneCPUBackendHonorsSpherePatchSetting() throws {
        let (scene, ball) = sphereBoxScene(patchContacts: true)
        let solver = try scene.makeCPUSolverChecked()
        for _ in 0..<240 { solver.step() }
        let manifold = cpuManifold(solver, bodies: Set([0, ball]))
        XCTAssertNotNil(manifold)
        XCTAssertGreaterThan(manifold?.numContacts ?? 0, 1,
                             "the authored-collider CPU path dropped the public "
                             + "spherePatchContacts setting")
    }

    /// A cooked convex face is still a flat pad. The public setting says
    /// sphere-face, so CPU and GPU must not silently narrow that to boxes.
    func testSpherePatchWorksAgainstConvexPadOnCPUAndGPU() throws {
        for sphereFirst in [false, true] {
            let (scene, pad, ball) = convexPadScene(sphereFirst: sphereFirst)

            let cpu = try scene.makeCPUSolverChecked()
            cpu.step()
            XCTAssertGreaterThan(
                cpuManifold(cpu, bodies: Set([pad, ball]))?.numContacts ?? 0, 1,
                "CPU convex-pad contact remained a single support witness")

            let gpu = try GPUSolver(scene: scene)
            try gpu.submitStep()
            try gpu.synchronize()
            let counts = gpu.activeRigidContactCounts().filter {
                Set([$0.bodyA, $0.bodyB]) == Set([pad, ball])
            }
            XCTAssertEqual(counts.count, 1)
            XCTAssertGreaterThan(
                counts.first?.contacts ?? 0, 1,
                "GPU convex-pad contact remained a single support witness")
        }
    }

    /// Opt-in patching must not turn every sphere-hull witness into an
    /// artificial footprint: disabled patches and true edge contacts retain
    /// one exact support point on both backends.
    func testConvexPatchIsOptInAndLimitedToBroadFaces() throws {
        for scene in [
            convexPadScene(patchContacts: false).0,
            convexPadScene(spherePosition: F3(0.164, 0, 0.026)).0,
        ] {
            let cpu = try scene.makeCPUSolverChecked()
            cpu.step()
            XCTAssertEqual(
                cpu.forces.compactMap { $0 as? CPUManifold }
                    .map(\.numContacts).max(),
                1)

            let gpu = try GPUSolver(scene: scene)
            try gpu.submitStep()
            try gpu.synchronize()
            XCTAssertEqual(gpu.activeRigidContactCounts().first?.contacts, 1)
        }
    }

    /// The generalized convex-face path must produce the behavior that
    /// justified it, not merely a larger contact count: its lever arms must
    /// actually transmit a torsional friction impulse on both backends.
    func testConvexFacePatchDampsSpinOnCPUAndGPU() throws {
        func cpuResidual(patchContacts: Bool) throws -> Float {
            var (scene, _, ball) = convexPadScene(
                patchContacts: patchContacts,
                spherePosition: F3(0, 0, 0.04))
            scene.settings.gravity = -10
            let solver = try scene.makeCPUSolverChecked()
            for _ in 0..<180 { solver.step() }
            solver.bodies[ball].velocityAng = F3(0, 0, 20)
            for _ in 0..<240 { solver.step() }
            return abs(solver.bodies[ball].velocityAng.z)
        }
        func gpuResidual(patchContacts: Bool) throws -> Float {
            var (scene, _, ball) = convexPadScene(
                patchContacts: patchContacts,
                spherePosition: F3(0, 0, 0.04))
            scene.settings.gravity = -10
            let solver = try GPUSolver(scene: scene)
            for _ in 0..<180 { solver.step() }
            solver.setBodyStates([.init(
                body: ball, position: solver.bodyPosition(ball),
                rotation: solver.bodyRotation(ball), linearVelocity: .zero,
                angularVelocity: F3(0, 0, 20))])
            for _ in 0..<240 { solver.step() }
            return abs(solver.bodyAngularVelocity(ball).z)
        }

        let cpuPoint = try cpuResidual(patchContacts: false)
        let cpuPatch = try cpuResidual(patchContacts: true)
        let gpuPoint = try gpuResidual(patchContacts: false)
        let gpuPatch = try gpuResidual(patchContacts: true)
        XCTAssertGreaterThan(cpuPoint, 5)
        XCTAssertLessThan(cpuPatch, 1)
        XCTAssertLessThan(cpuPatch, cpuPoint * 0.2)
        XCTAssertGreaterThan(gpuPoint, 5)
        XCTAssertLessThan(gpuPatch, 1)
        XCTAssertLessThan(gpuPatch, gpuPoint * 0.2)
    }

    /// The compatibility kernel exists because even unreachable Metal branches
    /// changed fast-math trajectories. Its bytes are a deliberate regression
    /// boundary; enhanced analytic semantics must dispatch another PSO.
    func testAnalyticCompatibilityKernelRemainsFrozen() throws {
        let resources = GPUSolver.physicsResourceBundle
        let urls = (resources.urls(
            forResourcesWithExtension: "metal", subdirectory: nil) ?? [])
            + (resources.urls(
                forResourcesWithExtension: "metal", subdirectory: "Shaders") ?? [])
        let url = try XCTUnwrap(urls.first {
            $0.lastPathComponent == "35_analytic_compat.metal"
        })
        let digest = SHA256.hash(data: try Data(contentsOf: url))
            .map { String(format: "%02x", $0) }.joined()
        XCTAssertEqual(
            digest,
            "09fe536edbfd1540b100c2d5ed1d12fdcd3c7b09e01a9f646f21b45127b42b2c")
    }

    /// Patch-enabled sphere-box pairs and the scale-aware capsule-box fix must
    /// use the enhanced analytic kernel, while untouched scenes keep the frozen
    /// compatibility path. Runtime setting changes must take effect too.
    func testEnhancedAnalyticFeaturesUseDisjointCompatibilityOverlay() throws {
        let (plainScene, _) = sphereBoxScene(patchContacts: false)
        let plain = try GPUSolver(scene: plainScene)
        XCTAssertTrue(plain.usesHullFreeAnalyticCompatibilityKernelForTesting)
        XCTAssertFalse(plain.usesEnhancedAnalyticNarrowPhaseForTesting)
        plain.settings.spherePatchContacts = true
        XCTAssertTrue(plain.usesHullFreeAnalyticCompatibilityKernelForTesting)
        XCTAssertTrue(plain.usesEnhancedAnalyticNarrowPhaseForTesting,
                      "enabling the public setting after initialization must "
                      + "add the enhanced analytic overlay without replacing "
                      + "the frozen compatibility PSO")

        var capsuleScene = PhysicsScene(name: "enhanced-capsule")
        _ = capsuleScene.addBody(
            size: F3(0.4, 0.4, 0.05), density: 0, friction: 0.8,
            position: F3(0, 0, -0.025))
        _ = capsuleScene.addBody(
            size: F3(0.022, 0.005, 0), density: 300, friction: 0.8,
            position: F3(0, 0, 0.02),
            rotation: Quat(angle: .pi / 2, axis: F3(0, 1, 0)),
            shape: .capsule)
        let capsule = try GPUSolver(scene: capsuleScene)
        XCTAssertTrue(capsule.usesHullFreeAnalyticCompatibilityKernelForTesting)
        XCTAssertTrue(capsule.usesEnhancedAnalyticNarrowPhaseForTesting,
                      "scale-aware capsule dedup must use a disjoint overlay, "
                      + "not mutate or replace the frozen PSO")
    }
}
