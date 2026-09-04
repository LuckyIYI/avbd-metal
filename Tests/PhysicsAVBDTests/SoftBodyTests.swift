import XCTest
import simd
@testable import SimCore
@testable import PhysicsAVBD
@testable import GPUSimDemos

/// Soft body physics per the paper: 3-DOF particles, hard rod elements,
/// Neo-Hookean tets, and two-way rigid coupling.
final class SoftBodyTests: XCTestCase {
    func testLinearVelocityImpulsePreservesAccelerationHistory() throws {
        var scene = PhysicsScene(name: "impulse-history")
        scene.settings.gravity = -9.81
        let body = scene.addBody(
            size: F3(repeating: 0.2), density: 1, friction: 0.5,
            position: F3(0, 0, 1), shape: .box)
        let solver = try GPUSolver(scene: scene)
        solver.setBodyStates([.init(
            body: body, position: F3(0, 0, 1), rotation: Quat(),
            linearVelocity: F3(0.25, -0.5, 0.75),
            angularVelocity: .zero)])

        let beforeVelocity = solver.velLin.contents().bindMemory(
            to: SIMD4<Float>.self, capacity: solver.numBodies)[body]
        let beforePrevious = solver.prevVelLin.contents().bindMemory(
            to: SIMD4<Float>.self, capacity: solver.numBodies)[body]
        let accelerationHistory = beforeVelocity - beforePrevious
        let delta = F3(-0.4, 0.6, -0.8)

        solver.applyLinearVelocityImpulses([.init(
            body: body, deltaVelocity: delta)])

        let afterVelocity = solver.velLin.contents().bindMemory(
            to: SIMD4<Float>.self, capacity: solver.numBodies)[body]
        let afterPrevious = solver.prevVelLin.contents().bindMemory(
            to: SIMD4<Float>.self, capacity: solver.numBodies)[body]
        XCTAssertEqual(afterVelocity.x, beforeVelocity.x + delta.x, accuracy: 1e-7)
        XCTAssertEqual(afterVelocity.y, beforeVelocity.y + delta.y, accuracy: 1e-7)
        XCTAssertEqual(afterVelocity.z, beforeVelocity.z + delta.z, accuracy: 1e-7)
        XCTAssertEqual(afterVelocity.x - afterPrevious.x,
                       accelerationHistory.x, accuracy: 1e-7)
        XCTAssertEqual(afterVelocity.y - afterPrevious.y,
                       accelerationHistory.y, accuracy: 1e-7)
        XCTAssertEqual(afterVelocity.z - afterPrevious.z,
                       accelerationHistory.z, accuracy: 1e-7)
    }

    func testSnapshotsAreBoundToTheirOriginatingSolver() throws {
        var scene = PhysicsScene(name: "snapshot-owner")
        _ = scene.addBody(
            size: F3(repeating: 0.2), density: 1, friction: 0.5,
            position: F3(0, 0, 1), shape: .box)
        let first = try GPUSolver(scene: scene)
        let second = try GPUSolver(scene: scene)

        let rigid = first.captureRigidSpeculationSnapshot()
        let full = first.captureSimulationSnapshot()
        XCTAssertTrue(first.owns(rigid))
        XCTAssertTrue(first.owns(full))
        XCTAssertFalse(second.owns(rigid),
            "equal-sized solvers must not accept each other's topology-opaque snapshots")
        XCTAssertFalse(second.owns(full),
            "full simulation snapshots require the same ownership guard")
    }

    private func particleBounds(_ scene: PhysicsScene,
                                bodyIDs: [Int]) -> (F3, F3) {
        var mn = F3(repeating: Float.greatestFiniteMagnitude)
        var mx = F3(repeating: -Float.greatestFiniteMagnitude)
        for id in bodyIDs {
            let p = scene.bodies[id].position
            mn = min(mn, p)
            mx = max(mx, p)
        }
        return (mn, mx)
    }

    private func separated(_ a: (F3, F3), _ b: (F3, F3),
                           clearance: Float = 0.01) -> Bool {
        a.1.x + clearance < b.0.x || b.1.x + clearance < a.0.x
            || a.1.y + clearance < b.0.y || b.1.y + clearance < a.0.y
            || a.1.z + clearance < b.0.z || b.1.z + clearance < a.0.z
    }

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

    func testSimulationSnapshotReplaysSoftContactBranchAfterImpulse() throws {
        var scene = PhysicsScene(name: "soft-snapshot-contact")
        scene.settings.dt = 1 / 120
        scene.settings.iterations = 8
        scene.settings.gravity = 0
        let p0 = scene.addParticle(
            radius: 0.02, mass: 1, friction: 1,
            position: F3(-1, -1, 0.5))
        let p1 = scene.addParticle(
            radius: 0.02, mass: 1, friction: 1,
            position: F3(1, -1, 0.5))
        let p2 = scene.addParticle(
            radius: 0.02, mass: 1, friction: 1,
            position: F3(0, 1, 0.5))
        let particles = [p0, p1, p2]
        scene.addTri(SceneTri(ids: (p0, p1, p2)))
        _ = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 1,
            position: .zero, shape: .box)
        let solver = try GPUSolver(scene: scene)
        try solver.submitStep()
        try solver.synchronize()
        XCTAssertGreaterThan(solver.lastNumSoft, 0)

        let checkpoint = solver.captureSimulationSnapshot()
        let checkpointStates = solver.bodyStates(particles)
        let impulse = particles.map {
            GPUSolver.BodyLinearVelocityImpulse(
                body: $0, deltaVelocity: F3(0.15, 0, 0))
        }
        solver.applyLinearVelocityImpulses(impulse)
        for _ in 0..<12 { try solver.submitStep() }
        try solver.synchronize()
        let firstBranch = solver.bodyStates(particles)

        solver.restoreSimulationSnapshot(checkpoint)
        let restored = solver.bodyStates(particles)
        for (expected, actual) in zip(checkpointStates, restored) {
            XCTAssertEqual(actual.position.x, expected.position.x, accuracy: 1e-7)
            XCTAssertEqual(actual.position.y, expected.position.y, accuracy: 1e-7)
            XCTAssertEqual(actual.position.z, expected.position.z, accuracy: 1e-7)
            XCTAssertEqual(actual.linearVelocity.x,
                           expected.linearVelocity.x, accuracy: 1e-7)
            XCTAssertEqual(actual.linearVelocity.y,
                           expected.linearVelocity.y, accuracy: 1e-7)
            XCTAssertEqual(actual.linearVelocity.z,
                           expected.linearVelocity.z, accuracy: 1e-7)
        }
        XCTAssertGreaterThan(solver.lastNumSoft, 0,
            "restoring a soft checkpoint must not cold-start its contacts")

        solver.applyLinearVelocityImpulses(impulse)
        for _ in 0..<12 { try solver.submitStep() }
        try solver.synchronize()
        let replayedBranch = solver.bodyStates(particles)
        for (expected, actual) in zip(firstBranch, replayedBranch) {
            XCTAssertEqual(actual.position.x, expected.position.x, accuracy: 2e-5)
            XCTAssertEqual(actual.position.y, expected.position.y, accuracy: 2e-5)
            XCTAssertEqual(actual.position.z, expected.position.z, accuracy: 2e-5)
            XCTAssertEqual(actual.linearVelocity.x,
                           expected.linearVelocity.x, accuracy: 2e-4)
            XCTAssertEqual(actual.linearVelocity.y,
                           expected.linearVelocity.y, accuracy: 2e-4)
            XCTAssertEqual(actual.linearVelocity.z,
                           expected.linearVelocity.z, accuracy: 2e-4)
        }
        XCTAssertNil(solver.runtimeFailure)
    }

    func testPortalRetryCapIsAnExplicitPrecisionOptIn() throws {
        func cap(configure: (inout SimSettings) -> Void) throws -> Float {
            var scene = PhysicsScene(name: "portal-retry-cap")
            scene.settings.dt = 1 / 120
            scene.settings.gravity = 0
            configure(&scene.settings)
            let p0 = scene.addParticle(
                radius: 0.01, mass: 1, friction: 1, position: F3(-1, -1, 0.5))
            let p1 = scene.addParticle(
                radius: 0.01, mass: 1, friction: 1, position: F3(1, -1, 0.5))
            let p2 = scene.addParticle(
                radius: 0.01, mass: 1, friction: 1, position: F3(0, 1, 0.5))
            scene.addTri(SceneTri(ids: (p0, p1, p2)))
            _ = scene.addBody(
                size: F3(repeating: 1), density: 0, friction: 1,
                position: .zero, shape: .box)
            let solver = try GPUSolver(scene: scene)
            try solver.submitStep()
            try solver.synchronize()
            return solver.params.deformablePortalRetryCap
        }
        // The default heuristic lands below the rigid margin here (5 mm vs
        // 10 mm), which must not enable the widened retry on its own.
        XCTAssertEqual(try cap { _ in }, 0)
        XCTAssertEqual(try cap { $0.deformableCollisionMargin = 0.02 }, 0,
            "an explicit margin no tighter than the rigid one stays legacy")
        XCTAssertEqual(try cap { $0.deformableCollisionMargin = 0.001 }, 0.02,
            "only an explicit tighter deformable margin opts in")
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

    /// A plush-stiffness block (E ~ 1.5 kPa) under a heavy plate flattens to
    /// a sheet with the bare stable Neo-Hookean model; compaction stiffening
    /// must let it keep a meaningful thickness while leaving the unloaded
    /// block's rest height unchanged.
    func testTetCompactionStiffeningKeepsCrushedPlushThick() throws {
        func crushedThickness(onset: Float, gain: Float) throws -> Float {
            var scene = PhysicsScene(name: "plush-compaction")
            scene.settings.iterations = 20
            scene.settings.betaLin = 20000
            scene.settings.lambdaMax = 800
            scene.settings.tetCompactionOnset = onset
            scene.settings.tetCompactionGain = gain
            Demos.addGround(&scene)
            let ids = Demos.addSoftBlock(&scene, center: F3(0, 0, 0.6), nx: 5, ny: 5, nz: 4,
                                         spacing: 0.3, mu: 600, lambda: 600,
                                         massPerNode: 0.02)
            _ = scene.addBody(size: F3(2.0, 2.0, 0.2), density: 400,
                              friction: 0.6, position: F3(0, 0, 1.8))
            let gpu = try GPUSolver(scene: scene)
            for _ in 0..<400 { gpu.step() }
            gpu.sync()
            let states = gpu.bodyStates(ids)
            XCTAssertNil(gpu.runtimeFailure)
            let zs = states.map(\.position.z)
            return zs.max()! - zs.min()!
        }
        let bare = try crushedThickness(onset: 0, gain: 0)
        let stiffened = try crushedThickness(onset: 0.6, gain: 30)
        print("compaction test: bare \(bare) stiffened \(stiffened) (rest 0.9)")
        XCTAssertLessThan(bare, 0.35, "the bare model must flatten under this plate")
        XCTAssertGreaterThan(stiffened, bare * 1.6,
            "compaction stiffening must resist the crush well beyond the bare model")
        XCTAssertLessThan(stiffened, 0.9,
            "stiffening below the onset must not prevent ordinary compression")
    }

    func testTetCompactionSettingsSyncAtRuntime() throws {
        var scene = PhysicsScene(name: "runtime-compaction-settings")
        scene.settings.gravity = 0
        let p0 = scene.addParticle(radius: 0.02, mass: 1,
                                   position: F3(0, 0, 0))
        let p1 = scene.addParticle(radius: 0.02, mass: 1,
                                   position: F3(1, 0, 0))
        let p2 = scene.addParticle(radius: 0.02, mass: 1,
                                   position: F3(0, 1, 0))
        let p3 = scene.addParticle(radius: 0.02, mass: 1,
                                   position: F3(0, 0, 1))
        scene.addTet(SceneTet(ids: (p0, p1, p2, p3), mu: 100, lambda: 200))
        let solver = try GPUSolver(scene: scene)

        solver.settings.tetCompactionOnset = 0.57
        solver.settings.tetCompactionGain = 13
        try solver.submitStep()
        try solver.synchronize()
        XCTAssertEqual(solver.params.tetCompactionOnset, 0.57, accuracy: 1e-7)
        XCTAssertEqual(solver.params.tetCompactionGain, 13, accuracy: 1e-7)

        solver.settings.tetCompactionOnset = -1
        solver.settings.tetCompactionGain = -2
        try solver.submitStep()
        try solver.synchronize()
        XCTAssertEqual(solver.params.tetCompactionOnset, 0)
        XCTAssertEqual(solver.params.tetCompactionGain, 0)
    }

    func testZeroTetCompactionOnsetDisablesPenaltyForInvertedTet() throws {
        func run(gain: Float) throws -> [GPUSolver.RigidBodyState] {
            var scene = PhysicsScene(name: "zero-onset-inverted-tet")
            scene.settings.dt = 1 / 120
            scene.settings.gravity = 0
            scene.settings.iterations = 1
            scene.settings.tetCompactionOnset = 0
            scene.settings.tetCompactionGain = gain
            let rest = [
                F3(0, 0, 0), F3(1, 0, 0),
                F3(0, 1, 0), F3(0, 0, 1),
            ]
            let ids = rest.map {
                scene.addParticle(radius: 0.01, mass: 1, position: $0)
            }
            scene.addTet(SceneTet(
                ids: (ids[0], ids[1], ids[2], ids[3]),
                mu: 500, lambda: 1_000))
            let solver = try GPUSolver(scene: scene)
            // Swap two vertices after construction so the current signed
            // volume is negative relative to the authored rest tetrahedron.
            solver.setBodyStates([
                .init(body: ids[1], position: rest[2], rotation: Quat(),
                      linearVelocity: .zero, angularVelocity: .zero),
                .init(body: ids[2], position: rest[1], rotation: Quat(),
                      linearVelocity: .zero, angularVelocity: .zero),
            ])
            try solver.submitStep()
            try solver.synchronize()
            XCTAssertNil(solver.runtimeFailure)
            return solver.bodyStates(ids)
        }

        let disabledByGain = try run(gain: 0)
        let disabledByOnset = try run(gain: 50)
        for (expected, actual) in zip(disabledByGain, disabledByOnset) {
            XCTAssertEqual(actual.position.x.bitPattern,
                           expected.position.x.bitPattern)
            XCTAssertEqual(actual.position.y.bitPattern,
                           expected.position.y.bitPattern)
            XCTAssertEqual(actual.position.z.bitPattern,
                           expected.position.z.bitPattern)
            XCTAssertEqual(actual.linearVelocity.x.bitPattern,
                           expected.linearVelocity.x.bitPattern)
            XCTAssertEqual(actual.linearVelocity.y.bitPattern,
                           expected.linearVelocity.y.bitPattern)
            XCTAssertEqual(actual.linearVelocity.z.bitPattern,
                           expected.linearVelocity.z.bitPattern)
        }
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
        XCTAssertGreaterThan(scene.settings.clothRenderScale, 0.5,
                             "the rendered sheet should match its contact surface")
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
        let renderSkin = try XCTUnwrap(gpu.renderSkinnedSurface)
        XCTAssertEqual(renderSkin.triCount,
                       scene.skinnedMeshes.reduce(0) { $0 + $1.triangles.count })
        XCTAssertGreaterThan(gpu.numEdges, 0,
                             "pinned cloth must keep EE contact edges")

        // The bundled mesh is a smooth visual skin over the voxel FEM proxy.
        // Rendering the proxy itself produces only axis-aligned faces.
        let mesh = try XCTUnwrap(scene.skinnedMeshes.first)
        // ModelIO welds the source OBJ's duplicate positions on import; no
        // scene-level simplification is applied to the resulting 34,834
        // unique vertices or 69,451 faces.
        XCTAssertEqual(mesh.vertices.count, 34_834,
                       "the demo must retain the complete imported Stanford Bunny")
        XCTAssertEqual(mesh.triangles.count, 69_451)
        func restPosition(_ vertex: SceneSkinnedVertex) -> F3 {
            let ids = [vertex.ids.0, vertex.ids.1, vertex.ids.2, vertex.ids.3]
            return zip(ids, [vertex.weights.x, vertex.weights.y,
                             vertex.weights.z, vertex.weights.w]).reduce(.zero) {
                $0 + scene.bodies[$1.0].position * $1.1
            }
        }
        let positions = mesh.vertices.map(restPosition)
        var roundedFaces = 0
        for tri in mesh.triangles {
            let n = normalize(cross(positions[tri.1] - positions[tri.0],
                                    positions[tri.2] - positions[tri.0]))
            if max(abs(n.x), max(abs(n.y), abs(n.z))) < 0.995 {
                roundedFaces += 1
            }
        }
        XCTAssertGreaterThan(roundedFaces, mesh.triangles.count / 3,
                             "the bundled skin must not expose the voxel/tet proxy")
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
        XCTAssertEqual(gpu.numEdges, 0,
                       "closed tet solids should use VT/RT contact, not cloth EE edges")
        let nonParticles = scene.bodies.filter { !$0.isParticle }.count
        XCTAssertEqual(gpu.renderRigidBodyCount, nonParticles,
                       "skinned soft particles should not be rendered as hidden rigid instances")
        XCTAssertLessThan(gpu.renderRigidBodyCount, scene.bodies.count / 20)
        for _ in 0..<5 { gpu.step() }
        XCTAssertTrue(gpu.maxConstraintError().isFinite)
    }

    func testSkinnedBunniesStartSeparatedInBox() throws {
        let scene = Demos.skinnedbunny(res: 8, count: 36)
        let bounds = scene.skinnedMeshes.map { particleBounds(scene, bodyIDs: $0.bodyIDs) }
        for i in 0..<bounds.count {
            for j in (i + 1)..<bounds.count {
                XCTAssertTrue(separated(bounds[i], bounds[j]),
                              "skinned bunnies \(i) and \(j) must not overlap at init")
            }
        }
    }

    func testSkinnedBunnyCageContainsSkinAndPreservesVolume() throws {
        let scene = try XCTUnwrap(Demos.make("skinnedbunny", scale: 1))
        let skin = try XCTUnwrap(scene.skinnedMeshes.first)
        let bodySet = Set(skin.bodyIDs)
        let tets = scene.tets.filter {
            bodySet.contains($0.ids.0) && bodySet.contains($0.ids.1)
                && bodySet.contains($0.ids.2) && bodySet.contains($0.ids.3)
        }
        XCTAssertFalse(tets.isEmpty)
        XCTAssertEqual(tets.count % 5, 0,
                       "the voxel cage uses five tetrahedra per occupied cell")
        func volume(_ tet: SceneTet, _ position: (Int) -> F3) -> Float {
            let a = position(tet.ids.0)
            return abs(dot(position(tet.ids.1) - a,
                           cross(position(tet.ids.2) - a,
                                 position(tet.ids.3) - a))) / 6
        }
        let restVolume = tets.reduce(Float.zero) {
            $0 + volume($1) { scene.bodies[$0].position }
        }
        XCTAssertGreaterThan(restVolume, 0)
        let firstCellBodies = Set(tets.prefix(5).flatMap {
            [$0.ids.0, $0.ids.1, $0.ids.2, $0.ids.3]
        })
        let firstCellPositions = firstCellBodies.map { scene.bodies[$0].position }
        let firstCellMin = firstCellPositions.reduce(F3(repeating: .greatestFiniteMagnitude), min)
        let firstCellMax = firstCellPositions.reduce(F3(repeating: -.greatestFiniteMagnitude), max)
        let firstCellExtent = firstCellMax - firstCellMin
        let voxelSpacing = max(firstCellExtent.x,
                               max(firstCellExtent.y, firstCellExtent.z))
        let cellMins = stride(from: 0, to: tets.count, by: 5).map { start -> F3 in
            tets[start..<min(start + 5, tets.count)].flatMap {
                [$0.ids.0, $0.ids.1, $0.ids.2, $0.ids.3]
            }.reduce(F3(repeating: .greatestFiniteMagnitude)) {
                min($0, scene.bodies[$1].position)
            }
        }
        let cellOrigin = cellMins.reduce(F3(repeating: .greatestFiniteMagnitude), min)
        let occupiedCells = Set(cellMins.map { p in
            let q = (p - cellOrigin) / voxelSpacing
            return SIMD3(Int(q.x.rounded()), Int(q.y.rounded()), Int(q.z.rounded()))
        })
        XCTAssertEqual(occupiedCells.count, tets.count / 5)
        let maxCell = occupiedCells.reduce(SIMD3<Int>(repeating: 0)) {
            SIMD3(Swift.max($0.x, $1.x), Swift.max($0.y, $1.y),
                  Swift.max($0.z, $1.z))
        }
        var outside = Set<SIMD3<Int>>()
        var queue = [SIMD3(-1, -1, -1)]
        outside.insert(queue[0])
        var head = 0
        while head < queue.count {
            let c = queue[head]
            head += 1
            for d in [SIMD3(1, 0, 0), SIMD3(-1, 0, 0), SIMD3(0, 1, 0),
                      SIMD3(0, -1, 0), SIMD3(0, 0, 1), SIMD3(0, 0, -1)] {
                let n = c &+ d
                guard n.x >= -1, n.y >= -1, n.z >= -1,
                      n.x <= maxCell.x + 1, n.y <= maxCell.y + 1,
                      n.z <= maxCell.z + 1,
                      !occupiedCells.contains(n), outside.insert(n).inserted else { continue }
                queue.append(n)
            }
        }
        var cavities = 0
        for x in 0...maxCell.x {
            for y in 0...maxCell.y {
                for z in 0...maxCell.z {
                    let c = SIMD3(x, y, z)
                    if !occupiedCells.contains(c) && !outside.contains(c) { cavities += 1 }
                }
            }
        }
        XCTAssertEqual(cavities, 0,
                       "a volumetric bunny cage must not be a hollow tet shell")
        var minimumWeight: Float = 1
        var maximumWeight: Float = 0
        var maximumWeightSumError: Float = 0
        for vertex in skin.vertices {
            let w = vertex.weights
            minimumWeight = min(minimumWeight,
                                min(min(w.x, w.y), min(w.z, w.w)))
            maximumWeight = max(maximumWeight,
                                max(max(w.x, w.y), max(w.z, w.w)))
            maximumWeightSumError = max(
                maximumWeightSumError, abs(w.x + w.y + w.z + w.w - 1))
        }
        XCTAssertGreaterThanOrEqual(
            minimumWeight, -1e-4,
            "the rendered skin must be contained by its tetrahedral cage")
        XCTAssertLessThanOrEqual(
            maximumWeight, 1.0001,
            "contained skin bindings must not extrapolate outside a tet")
        XCTAssertLessThanOrEqual(maximumWeightSumError, 1e-4)
        func surfaceVolume(_ positions: [F3]) -> Float {
            let center = positions.reduce(F3.zero, +) / Float(positions.count)
            return abs(skin.triangles.reduce(Float.zero) {
                $0 + dot(positions[$1.0] - center,
                         cross(positions[$1.1] - center,
                               positions[$1.2] - center)) / 6
            })
        }
        let restSkinPositions = skin.vertices.map { vertex -> F3 in
            let ids = [vertex.ids.0, vertex.ids.1, vertex.ids.2, vertex.ids.3]
            let weights = [vertex.weights.x, vertex.weights.y,
                           vertex.weights.z, vertex.weights.w]
            return zip(ids, weights).reduce(.zero) {
                $0 + scene.bodies[$1.0].position * $1.1
            }
        }
        let restSkinVolume = surfaceVolume(restSkinPositions)
        XCTAssertGreaterThan(restSkinVolume, 0)

        let gpu = try GPUSolver(scene: scene)
        for _ in 0..<300 { gpu.step() }
        try gpu.synchronize()
        let currentVolume = tets.reduce(Float.zero) {
            $0 + volume($1) { gpu.bodyPosition($0) }
        }
        let currentSkinPositions = skin.vertices.map { vertex -> F3 in
            let ids = [vertex.ids.0, vertex.ids.1, vertex.ids.2, vertex.ids.3]
            let weights = [vertex.weights.x, vertex.weights.y,
                           vertex.weights.z, vertex.weights.w]
            return zip(ids, weights).reduce(.zero) {
                $0 + gpu.bodyPosition($1.0) * $1.1
            }
        }
        let currentSkinVolume = surfaceVolume(currentSkinPositions)
        XCTAssertGreaterThan(currentVolume / restVolume, 0.98,
                             "the bunny's FEM cage must preserve tet volume")
        XCTAssertGreaterThan(currentSkinVolume / restSkinVolume, 0.95,
                             "the visible bunny must follow the volumetric cage, not collapse like a shell")
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
