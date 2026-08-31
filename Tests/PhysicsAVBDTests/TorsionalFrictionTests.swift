import XCTest
import CryptoKit
import Metal
import simd
@testable import SimCore
@testable import PhysicsAVBD

/// Contact-material regressions. Smooth geometry keeps its exact witnesses;
/// resistance to rotation about a one-point contact normal is a bounded
/// material torque, not a ring of invented geometric contacts.
final class TorsionalFrictionTests: XCTestCase {

    private let torsionalRadius: Float = 0.006

    private func sphereBoxScene(
        torsionalFriction: Float
    ) -> (PhysicsScene, Int) {
        var scene = PhysicsScene(name: "sphere-box-torsion")
        scene.settings.collisionMargin = 0.003
        _ = scene.addBody(
            size: F3(0.3, 0.3, 0.05), density: 0, friction: 0.9,
            torsionalFriction: torsionalFriction,
            position: F3(0, 0, -0.025))
        let ball = scene.addSphere(
            diameter: 0.058, density: 900, friction: 0.9,
            torsionalFriction: torsionalFriction,
            position: F3(0, 0, 0.04))
        return (scene, ball)
    }

    private func tippedConvexScene(
        torsionalFriction: Float
    ) -> (PhysicsScene, Int) {
        var scene = PhysicsScene(name: "convex-vertex-torsion")
        scene.settings.collisionMargin = 0.003
        _ = scene.addBody(
            size: F3(0.3, 0.3, 0.05), density: 0, friction: 0.9,
            torsionalFriction: torsionalFriction,
            position: F3(0, 0, -0.025))
        let body = scene.addBody(
            size: F3(repeating: 0.06), density: 0, friction: 0.9,
            torsionalFriction: torsionalFriction,
            position: F3(0, 0, 0.03), mass: 1,
            diagonalInertia: F3(repeating: 0.01), collisionEnabled: false)
        _ = scene.addConvexCollider(
            body: body,
            vertices: [
                F3(0, 0, -0.03),
                F3(0.03, 0.02, 0.03),
                F3(-0.03, 0.02, 0.03),
                F3(0, -0.03, 0.03),
            ],
            friction: 0.9, torsionalFriction: torsionalFriction)
        scene.addJoint(SceneJoint(
            bodyA: -1, bodyB: body, rA: .zero, rB: .zero,
            stiffnessLin: 0, stiffnessAng: .infinity,
            hingeAxis: F3(0, 0, 1)))
        return (scene, body)
    }

    private func boxFaceScene(
        torsionalFriction: Float
    ) -> (PhysicsScene, Int) {
        var scene = PhysicsScene(name: "box-face-torsion")
        scene.settings.collisionMargin = 0.003
        _ = scene.addBody(
            size: F3(0.3, 0.3, 0.05), density: 0, friction: 0,
            torsionalFriction: torsionalFriction,
            position: F3(0, 0, -0.025))
        let box = scene.addBody(
            size: F3(repeating: 0.06), density: 900, friction: 0,
            torsionalFriction: torsionalFriction,
            position: F3(0, 0, 0.04))
        return (scene, box)
    }

    private func convexPadScene(
        torsionalFriction: Float,
        sphereFirst: Bool = false
    ) -> (PhysicsScene, Int, Int) {
        var scene = PhysicsScene(name: "convex-pad-torsion")
        scene.settings.gravity = 0
        scene.settings.collisionMargin = 0.003
        let h = F3(0.15, 0.15, 0.025)
        let vertices = [
            F3(-h.x, -h.y, -h.z), F3(h.x, -h.y, -h.z),
            F3(-h.x, h.y, -h.z), F3(h.x, h.y, -h.z),
            F3(-h.x, -h.y, h.z), F3(h.x, -h.y, h.z),
            F3(-h.x, h.y, h.z), F3(h.x, h.y, h.z),
        ]
        func addPad(_ scene: inout PhysicsScene) -> Int {
            let pad = scene.addBody(
                size: F3(0.3, 0.3, 0.05), density: 0, friction: 0.9,
                torsionalFriction: torsionalFriction,
                position: F3(0, 0, -0.025), collisionEnabled: false)
            _ = scene.addConvexCollider(
                body: pad, vertices: vertices, friction: 0.9,
                torsionalFriction: torsionalFriction)
            return pad
        }
        let pad: Int
        let sphere: Int
        if sphereFirst {
            sphere = scene.addSphere(
                diameter: 0.058, density: 900, friction: 0.9,
                torsionalFriction: torsionalFriction,
                position: F3(0, 0, 0.026))
            pad = addPad(&scene)
        } else {
            pad = addPad(&scene)
            sphere = scene.addSphere(
                diameter: 0.058, density: 900, friction: 0.9,
                torsionalFriction: torsionalFriction,
                position: F3(0, 0, 0.026))
        }
        return (scene, pad, sphere)
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

    private func gpuContactCount(
        _ solver: GPUSolver, bodies: Set<Int>
    ) -> Int? {
        solver.activeRigidContactCounts().first {
            Set([$0.bodyA, $0.bodyB]) == bodies
        }?.contacts
    }

    func testSphereFaceRemainsOneExactContactOnCPUAndGPU() throws {
        try requireMetal()
        let (scene, ball) = sphereBoxScene(
            torsionalFriction: torsionalRadius)

        let cpu = try scene.makeCPUSolverChecked()
        for _ in 0..<240 { cpu.step() }
        XCTAssertEqual(cpuManifold(cpu, bodies: Set([0, ball]))?.numContacts, 1)

        let gpu = try GPUSolver(scene: scene)
        for _ in 0..<240 { gpu.step() }
        XCTAssertEqual(gpuContactCount(gpu, bodies: Set([0, ball])), 1)
        XCTAssertEqual(gpu.bodyPosition(ball).z, 0.029 - 0.003,
                       accuracy: 0.002)
    }

    func testConvexFaceRemainsOneExactContactInBothOperandOrders() throws {
        try requireMetal()
        for sphereFirst in [false, true] {
            let (scene, pad, sphere) = convexPadScene(
                torsionalFriction: torsionalRadius,
                sphereFirst: sphereFirst)
            let pair = Set([pad, sphere])

            let cpu = try scene.makeCPUSolverChecked()
            cpu.step()
            XCTAssertEqual(cpuManifold(cpu, bodies: pair)?.numContacts, 1)

            let gpu = try GPUSolver(scene: scene)
            try gpu.submitStep()
            try gpu.synchronize()
            XCTAssertEqual(gpuContactCount(gpu, bodies: pair), 1)
        }
    }

    func testTorsionalMaterialDampsSphereSpinOnCPUAndGPU() throws {
        try requireMetal()
        func cpuResidual(_ coefficient: Float) throws -> Float {
            let (scene, ball) = sphereBoxScene(torsionalFriction: coefficient)
            let solver = try scene.makeCPUSolverChecked()
            for _ in 0..<5 { solver.step() }
            solver.bodies[ball].velocityAng = F3(0, 0, 20)
            for _ in 0..<240 { solver.step() }
            return abs(solver.bodies[ball].velocityAng.z)
        }
        func gpuResidual(_ coefficient: Float) throws -> Float {
            let (scene, ball) = sphereBoxScene(torsionalFriction: coefficient)
            let solver = try GPUSolver(scene: scene)
            for _ in 0..<180 { solver.step() }
            solver.setBodyStates([.init(
                body: ball, position: solver.bodyPosition(ball),
                rotation: solver.bodyRotation(ball), linearVelocity: .zero,
                angularVelocity: F3(0, 0, 20))])
            for _ in 0..<240 { solver.step() }
            XCTAssertEqual(gpuContactCount(solver, bodies: Set([0, ball])), 1)
            return abs(solver.bodyAngularVelocity(ball).z)
        }

        let cpuControl = try cpuResidual(0)
        let cpuTorsion = try cpuResidual(torsionalRadius)
        let gpuControl = try gpuResidual(0)
        let gpuTorsion = try gpuResidual(torsionalRadius)
        XCTAssertGreaterThan(cpuControl, 5)
        XCTAssertLessThan(cpuTorsion, 1)
        XCTAssertLessThan(cpuTorsion, cpuControl * 0.2)
        XCTAssertGreaterThan(gpuControl, 5)
        XCTAssertLessThan(gpuTorsion, 1)
        XCTAssertLessThan(gpuTorsion, gpuControl * 0.2)
    }

    func testOneAuthoredSurfaceContributesThroughSymmetricAverage() throws {
        try requireMetal()
        var scene = PhysicsScene(name: "one-sided-torsion")
        scene.settings.collisionMargin = 0.003
        _ = scene.addBody(
            size: F3(0.3, 0.3, 0.05), density: 0, friction: 0.9,
            torsionalFriction: 0.012, position: F3(0, 0, -0.025))
        let ball = scene.addSphere(
            diameter: 0.058, density: 900, friction: 0.9,
            position: F3(0, 0, 0.04))

        let cpu = try scene.makeCPUSolverChecked()
        for _ in 0..<180 { cpu.step() }
        let manifold = try XCTUnwrap(
            cpuManifold(cpu, bodies: Set([0, ball])))
        XCTAssertEqual(manifold.torsionalFriction, 0.006, accuracy: 1.0e-7)
        cpu.bodies[ball].velocityAng = F3(0, 0, 20)
        for _ in 0..<240 { cpu.step() }
        XCTAssertLessThan(abs(cpu.bodies[ball].velocityAng.z), 1)

        let gpu = try GPUSolver(scene: scene)
        for _ in 0..<180 { gpu.step() }
        gpu.setBodyStates([.init(
            body: ball, position: gpu.bodyPosition(ball),
            rotation: gpu.bodyRotation(ball), linearVelocity: .zero,
            angularVelocity: F3(0, 0, 20))])
        for _ in 0..<240 { gpu.step() }
        XCTAssertLessThan(abs(gpu.bodyAngularVelocity(ball).z), 1)
        XCTAssertEqual(gpuContactCount(gpu, bodies: Set([0, ball])), 1)
    }

    /// This is deliberately not a sphere test: the solver term applies to any
    /// true one-point manifold and is independent of narrow-phase shape code.
    func testTorsionalMaterialAlsoDampsConvexVertexSpin() throws {
        try requireMetal()
        func cpuResidual(_ coefficient: Float) throws -> (Float, Int) {
            let (scene, body) = tippedConvexScene(
                torsionalFriction: coefficient)
            let solver = try scene.makeCPUSolverChecked()
            for _ in 0..<30 { solver.step() }
            solver.bodies[body].velocityAng = F3(0, 0, 20)
            for _ in 0..<120 { solver.step() }
            let manifold = cpuManifold(solver, bodies: Set([0, body]))
            return (abs(solver.bodies[body].velocityAng.z),
                    manifold?.numContacts ?? 0)
        }
        func gpuResidual(_ coefficient: Float) throws -> (Float, Int) {
            let (scene, body) = tippedConvexScene(
                torsionalFriction: coefficient)
            let solver = try GPUSolver(scene: scene)
            for _ in 0..<30 { solver.step() }
            solver.setBodyStates([.init(
                body: body, position: solver.bodyPosition(body),
                rotation: solver.bodyRotation(body), linearVelocity: .zero,
                angularVelocity: F3(0, 0, 20))])
            for _ in 0..<120 { solver.step() }
            return (abs(solver.bodyAngularVelocity(body).z),
                    gpuContactCount(solver, bodies: Set([0, body])) ?? 0)
        }

        let cpuControl = try cpuResidual(0)
        let cpuTorsion = try cpuResidual(torsionalRadius)
        let gpuControl = try gpuResidual(0)
        let gpuTorsion = try gpuResidual(torsionalRadius)
        XCTAssertEqual(cpuTorsion.1, 1)
        XCTAssertEqual(gpuTorsion.1, 1)
        XCTAssertGreaterThan(cpuControl.0, 5)
        XCTAssertLessThan(cpuTorsion.0, 9)
        XCTAssertLessThan(cpuTorsion.0, cpuControl.0 * 0.75)
        XCTAssertGreaterThan(gpuControl.0, 5)
        XCTAssertLessThan(gpuTorsion.0, 9)
        XCTAssertLessThan(gpuTorsion.0, gpuControl.0 * 0.75)
    }

    /// A face manifold has several real witnesses, but torsion remains one
    /// aggregate mode. Ordinary friction is zero so only the new material
    /// term can remove rotation about the contact normal.
    func testMultipointFaceUsesOneAggregateTorsionalMode() throws {
        try requireMetal()
        func cpuResidual(_ coefficient: Float) throws -> (Float, Int) {
            let (scene, box) = boxFaceScene(torsionalFriction: coefficient)
            let solver = try scene.makeCPUSolverChecked()
            for _ in 0..<180 { solver.step() }
            solver.bodies[box].velocityAng = F3(0, 0, 20)
            for _ in 0..<120 { solver.step() }
            return (abs(solver.bodies[box].velocityAng.z),
                    cpuManifold(solver, bodies: Set([0, box]))?.numContacts ?? 0)
        }
        func gpuResidual(_ coefficient: Float) throws -> (Float, Int) {
            let (scene, box) = boxFaceScene(torsionalFriction: coefficient)
            let solver = try GPUSolver(scene: scene)
            for _ in 0..<180 { solver.step() }
            solver.setBodyStates([.init(
                body: box, position: solver.bodyPosition(box),
                rotation: solver.bodyRotation(box), linearVelocity: .zero,
                angularVelocity: F3(0, 0, 20))])
            for _ in 0..<120 { solver.step() }
            return (abs(solver.bodyAngularVelocity(box).z),
                    gpuContactCount(solver, bodies: Set([0, box])) ?? 0)
        }

        let cpuControl = try cpuResidual(0)
        let cpuTorsion = try cpuResidual(torsionalRadius)
        let gpuControl = try gpuResidual(0)
        let gpuTorsion = try gpuResidual(torsionalRadius)
        XCTAssertGreaterThan(cpuTorsion.1, 1)
        XCTAssertGreaterThan(gpuTorsion.1, 1)
        XCTAssertGreaterThan(cpuControl.0, 5)
        XCTAssertLessThan(cpuTorsion.0, 1)
        XCTAssertGreaterThan(gpuControl.0, 5)
        XCTAssertLessThan(gpuTorsion.0, 1)
    }

    /// A capsule lying on a face has a real line-like manifold. Preserve the
    /// scale-aware dedup correction; torsional friction does not synthesize it.
    func testShortCapsuleOnBoxKeepsMoreThanOneContact() throws {
        try requireMetal()
        var scene = PhysicsScene(name: "short-capsule")
        scene.settings.collisionMargin = 0.003
        _ = scene.addBody(
            size: F3(0.4, 0.4, 0.05), density: 0, friction: 0.8,
            position: F3(0, 0, -0.025))
        let link = scene.addBody(
            size: F3(0.022, 0.005, 0), density: 300, friction: 0.8,
            position: F3(0, 0, 0.02),
            rotation: Quat(angle: .pi / 2, axis: F3(0, 1, 0)),
            shape: .capsule)
        let solver = try GPUSolver(scene: scene)
        for _ in 0..<240 { solver.step() }
        XCTAssertGreaterThan(
            gpuContactCount(solver, bodies: Set([0, link])) ?? 0, 1)
    }

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

    func testOnlyCapsuleDedupUsesEnhancedAnalyticOverlay() throws {
        try requireMetal()
        let (sphereScene, _) = sphereBoxScene(
            torsionalFriction: torsionalRadius)
        let sphere = try GPUSolver(scene: sphereScene)
        XCTAssertTrue(sphere.usesHullFreeAnalyticCompatibilityKernelForTesting)
        XCTAssertFalse(sphere.usesEnhancedAnalyticNarrowPhaseForTesting,
                       "torsional friction belongs to the solver, not an "
                       + "alternate sphere narrow phase")

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
        XCTAssertTrue(capsule.usesEnhancedAnalyticNarrowPhaseForTesting)
    }

    func testTorsionalStateStorageIsAllocatedOnlyWhenReachable() throws {
        try requireMetal()
        let (ordinaryScene, _) = sphereBoxScene(torsionalFriction: 0)
        let ordinary = try GPUSolver(scene: ordinaryScene)
        XCTAssertEqual(ordinary.torsionState.length, 16)
        XCTAssertEqual(ordinary.prevTorsionState.length, 16)

        let (torsionalScene, _) = sphereBoxScene(
            torsionalFriction: torsionalRadius)
        let torsional = try GPUSolver(scene: torsionalScene)
        XCTAssertGreaterThan(torsional.torsionState.length, 16)
        XCTAssertEqual(
            torsional.torsionState.length,
            torsional.prevTorsionState.length)
    }

    func testTorsionDoesNotAddPerColorOrPerIterationDispatches() throws {
        try requireMetal()

        func dispatches(_ coefficient: Float)
            throws -> [GPUSolver.SolveDispatchForTesting] {
            var (scene, _) = sphereBoxScene(
                torsionalFriction: coefficient)
            scene.settings.iterations = 4
            let solver = try GPUSolver(scene: scene)
            var result: [GPUSolver.SolveDispatchForTesting] = []
            solver.solveDispatchObserverForTesting = { result.append($0) }
            try solver.submitStep()
            try solver.synchronize()
            return result
        }

        let ordinary = try dispatches(0)
        let torsional = try dispatches(torsionalRadius)
        XCTAssertFalse(ordinary.isEmpty)
        XCTAssertEqual(
            torsional.count, ordinary.count,
            "torsion must specialize existing solve passes, not append "
                + "one dispatch per color and iteration")
        XCTAssertTrue(torsional.contains(.primal(torsion: true)))
        XCTAssertTrue(torsional.contains(.dual(torsion: true)))
        XCTAssertFalse(torsional.contains(.primal(torsion: false)))
        XCTAssertFalse(torsional.contains(.dual(torsion: false)))
    }

    func testPersistentSmallSceneKeepsSingleDispatchWithTorsion() throws {
        try requireMetal()
        var (scene, ball) = sphereBoxScene(
            torsionalFriction: torsionalRadius)
        scene.settings.iterations = 20
        let solver = try GPUSolver(scene: scene)
        solver.persistentSolveForTesting = true
        var dispatches: [GPUSolver.SolveDispatchForTesting] = []
        solver.solveDispatchObserverForTesting = { dispatches.append($0) }

        for _ in 0..<180 { try solver.submitStep() }
        try solver.synchronize()
        solver.setBodyStates([.init(
            body: ball, position: solver.bodyPosition(ball),
            rotation: solver.bodyRotation(ball), linearVelocity: .zero,
            angularVelocity: F3(0, 0, 20))])
        dispatches.removeAll(keepingCapacity: true)
        for _ in 0..<120 { try solver.submitStep() }
        try solver.synchronize()

        XCTAssertEqual(dispatches.count, 120)
        XCTAssertTrue(dispatches.allSatisfy {
            $0 == .persistent(torsion: true)
        })
        XCTAssertLessThan(abs(solver.bodyAngularVelocity(ball).z), 1)
        XCTAssertNil(solver.runtimeFailure)
    }

    func testLegacyCPUImplicitColliderTracksTorsionalMaterialMutation() throws {
        let solver = CPUSolver()
        let ground = solver.addBody(
            size: F3(0.3, 0.3, 0.05), density: 0, friction: 0.9,
            position: F3(0, 0, -0.025))
        let ball = solver.addBody(
            size: F3(repeating: 0.058), density: 900, friction: 0.9,
            position: F3(0, 0, 0.04), shape: .sphere)
        ground.torsionalFriction = 0.012
        for _ in 0..<30 { solver.step() }
        let manifold = try XCTUnwrap(cpuManifold(
            solver, bodies: Set([ground.index, ball.index])))
        XCTAssertEqual(
            manifold.torsionalFriction,
            0.006, accuracy: 1.0e-7)
    }

    private func requireMetal() throws {
        if MTLCreateSystemDefaultDevice() == nil {
            throw XCTSkip("Metal is unavailable")
        }
    }
}
