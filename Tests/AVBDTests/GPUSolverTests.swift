import XCTest
import simd
@testable import AVBDCore

final class GPUSolverTests: XCTestCase {
    func testPerBodyGravityScaleMatchesCPUAndGPU() throws {
        var scene = PhysicsScene(name: "gravity-scale")
        scene.settings.dt = 1 / 100
        scene.settings.iterations = 4
        _ = scene.addBody(
            size: F3(repeating: 0.1), density: 1_000, friction: 0,
            position: F3(0, 0, 2), shape: .sphere, gravityScale: 0)
        _ = scene.addBody(
            size: F3(repeating: 0.1), density: 1_000, friction: 0,
            position: F3(2, 0, 2), shape: .sphere, gravityScale: 1)

        let cpu = scene.makeCPUSolver()
        let gpu = try GPUSolver(scene: scene)
        for _ in 0..<10 {
            cpu.step()
            gpu.step()
        }
        let gpuStates = gpu.bodyStates([0, 1])
        XCTAssertEqual(cpu.bodies[0].positionLin.z, 2, accuracy: 1e-5)
        XCTAssertEqual(gpuStates[0].position.z, 2, accuracy: 1e-5)
        XCTAssertLessThan(cpu.bodies[1].positionLin.z, 1.95)
        XCTAssertEqual(gpuStates[1].position.z,
                       cpu.bodies[1].positionLin.z, accuracy: 2e-3)
    }

    func testRigidLinearDampingMatchesCPUAndGPU() throws {
        var scene = PhysicsScene(name: "rigid-damping")
        scene.settings.gravity = 0
        scene.settings.dt = 1 / 120
        scene.settings.rigidLinearDamping = 2
        let body = scene.addBody(
            size: F3(repeating: 0.2), density: 1_000, friction: 0,
            position: .zero, velocity: F3(1, 0, 0), shape: .sphere)

        let cpu = scene.makeCPUSolver()
        let gpu = try makeGPU(scene)
        for _ in 0..<120 {
            cpu.step()
            gpu.step()
        }

        let expected = exp(Float(-2))
        XCTAssertEqual(cpu.bodies[body].velocityLin.x, expected, accuracy: 1e-4)
        XCTAssertEqual(gpu.bodyVelocity(body).x, expected, accuracy: 1e-4)
        XCTAssertEqual(gpu.bodyVelocity(body).x,
                       cpu.bodies[body].velocityLin.x, accuracy: 1e-5)
    }

    func testRigidAngularDampingMatchesCPUAndGPU() throws {
        var scene = PhysicsScene(name: "rigid-angular-damping")
        scene.settings.gravity = 0
        scene.settings.dt = 1 / 120
        scene.settings.rigidAngularDamping = 2
        let body = scene.addBody(
            size: F3(0.3, 0.2, 0.1), density: 1_000, friction: 0,
            position: .zero)

        let cpu = scene.makeCPUSolver()
        let gpu = try makeGPU(scene)
        let initialAngularVelocity = F3(0, 0, 1)
        cpu.bodies[body].velocityAng = initialAngularVelocity
        gpu.setBodyStates([.init(
            body: body, position: .zero,
            rotation: Quat(real: 1, imag: .zero),
            angularVelocity: initialAngularVelocity)])
        for _ in 0..<120 {
            cpu.step()
            gpu.step()
        }

        let expected = exp(Float(-2))
        let gpuAngular = gpu.bodyStates([body])[0].angularVelocity.z
        XCTAssertEqual(cpu.bodies[body].velocityAng.z,
                       expected, accuracy: 2e-3)
        XCTAssertEqual(gpuAngular, expected, accuracy: 2e-3)
        XCTAssertEqual(gpuAngular, cpu.bodies[body].velocityAng.z,
                       accuracy: 2e-3)
    }

    func testRigidDampingDoesNotDampParticles() throws {
        var scene = PhysicsScene(name: "rigid-damping-particle-isolation")
        scene.settings.gravity = 0
        scene.settings.dt = 1 / 120
        scene.settings.rigidLinearDamping = 2
        scene.settings.rigidAngularDamping = 2
        let particle = scene.addParticle(
            radius: 0.04, mass: 0.01, friction: 0,
            position: .zero, velocity: F3(1, 0, 0))

        let cpu = scene.makeCPUSolver()
        let gpu = try makeGPU(scene)
        for _ in 0..<120 {
            cpu.step()
            gpu.step()
        }

        XCTAssertEqual(cpu.bodies[particle].velocityLin.x, 1,
                       accuracy: 1e-4)
        XCTAssertEqual(gpu.bodyVelocity(particle).x, 1, accuracy: 1e-4)
    }

    func testWorldSpaceInertiaRotatesPrincipalAxes() {
        let principal = F3(1, 2, 3)
        let rotation = Quat(angle: .pi / 2, axis: F3(0, 0, 1))
        let inertia = worldInertiaRows(rotation, principal)
        XCTAssertEqual(inertia.r0.x, 2, accuracy: 1e-5)
        XCTAssertEqual(inertia.r1.y, 1, accuracy: 1e-5)
        XCTAssertEqual(inertia.r2.z, 3, accuracy: 1e-5)
        XCTAssertEqual(inertia.r0.y, 0, accuracy: 1e-5)
        XCTAssertEqual(inertia.r0.z, 0, accuracy: 1e-5)
        XCTAssertEqual(inertia.r1.z, 0, accuracy: 1e-5)
    }

    func makeGPU(_ scene: PhysicsScene) throws -> GPUSolver {
        do {
            return try GPUSolver(scene: scene)
        } catch GPUSolver.AVBDError.shaderCompile(let msg) {
            XCTFail("shader compile failed:\n\(msg)")
            throw GPUSolver.AVBDError.shaderCompile(msg)
        }
    }

    func testShadersCompile() throws {
        let solver = try makeGPU(Demos.ground())
        XCTAssertNotNil(solver.device)
        XCTAssertFalse(solver.pso.isEmpty)
    }

    func testCollisionGroupsIsolateOverlappingSimulationReplicas() throws {
        var scene = PhysicsScene(name: "overlapping-collision-groups")
        scene.settings.dt = 1 / 120
        scene.settings.iterations = 16
        _ = scene.addBody(
            size: F3(20, 20, 0.2), density: 0, friction: 0.9,
            position: F3(0, 0, -0.1))
        let first = scene.addBody(
            size: F3(repeating: 1), density: 1, friction: 0.8,
            position: F3(0, 0, 1.5), collisionGroup: 1)
        let second = scene.addBody(
            size: F3(repeating: 1), density: 1, friction: 0.8,
            position: F3(0, 0, 1.5), collisionGroup: 2)

        let solver = try makeGPU(scene)
        for _ in 0..<240 { solver.step() }
        let a = solver.bodyStates([first])[0]
        let b = solver.bodyStates([second])[0]

        // Both replicas contact shared group-zero ground, but their perfectly
        // overlapping bodies never see one another and therefore evolve
        // identically rather than being pushed apart.
        XCTAssertEqual(a.position.x.bitPattern, b.position.x.bitPattern)
        XCTAssertEqual(a.position.y.bitPattern, b.position.y.bitPattern)
        XCTAssertEqual(a.position.z.bitPattern, b.position.z.bitPattern)
        XCTAssertEqual(a.rotation.imag.x.bitPattern, b.rotation.imag.x.bitPattern)
        XCTAssertEqual(a.rotation.imag.y.bitPattern, b.rotation.imag.y.bitPattern)
        XCTAssertEqual(a.rotation.imag.z.bitPattern, b.rotation.imag.z.bitPattern)
        XCTAssertEqual(a.rotation.real.bitPattern, b.rotation.real.bitPattern)
        XCTAssertEqual(a.position.z, 0.5, accuracy: 0.06)
        XCTAssertEqual(solver.lastNumPairs, 2)
    }

    func testFreeFallMatchesAnalytic() throws {
        var scene = PhysicsScene(name: "fall")
        _ = scene.addBody(size: F3(1, 1, 1), density: 1, friction: 0.5, position: F3(0, 0, 100))
        let solver = try makeGPU(scene)

        let steps = 30
        for _ in 0..<steps { solver.step() }

        var v: Float = 0, z: Float = 100
        for _ in 0..<steps {
            v += solver.settings.gravity * solver.settings.dt
            z += v * solver.settings.dt
        }
        XCTAssertEqual(solver.bodyPosition(0).z, z, accuracy: 0.01)
    }

    func testExplicitImportedInertiaOverridesCollisionPrimitiveDensity() throws {
        var scene = PhysicsScene(name: "explicit-inertia")
        let expectedInertia = F3(0.0490211, 0.0445821, 0.00824619)
        let body = scene.addBody(
            size: F3(0.3, 0.2, 0.1), density: 0, friction: 0.5,
            position: F3(0, 0, 10), mass: 5.39,
            diagonalInertia: expectedInertia)

        let gpu = try makeGPU(scene)
        XCTAssertEqual(gpu.bodyMass(body), 5.39, accuracy: 1e-6)
        let gpuInertia = gpu.bodyDiagonalInertia(body)
        XCTAssertEqual(gpuInertia.x, expectedInertia.x, accuracy: 1e-7)
        XCTAssertEqual(gpuInertia.y, expectedInertia.y, accuracy: 1e-7)
        XCTAssertEqual(gpuInertia.z, expectedInertia.z, accuracy: 1e-7)

        let cpu = scene.makeCPUSolver()
        XCTAssertEqual(cpu.bodies[body].mass, 5.39, accuracy: 1e-6)
        XCTAssertEqual(cpu.bodies[body].moment.x, expectedInertia.x, accuracy: 1e-7)
        XCTAssertEqual(cpu.bodies[body].moment.y, expectedInertia.y, accuracy: 1e-7)
        XCTAssertEqual(cpu.bodies[body].moment.z, expectedInertia.z, accuracy: 1e-7)
    }

    func testOffsetCompoundCollidersSupportOneRigidBody() throws {
        var scene = PhysicsScene(name: "compound-collider-support")
        scene.settings.dt = 1 / 120
        scene.settings.iterations = 20
        _ = scene.addBody(size: F3(20, 20, 0.1), density: 0,
                          friction: 1, position: F3(0, 0, -0.05))
        let body = scene.addBody(
            size: F3(0.4, 0.4, 0.4), density: 0, friction: 0.9,
            position: F3(0, 0, 1.5), mass: 2,
            diagonalInertia: F3(0.25, 0.25, 0.25),
            collisionEnabled: false)
        _ = scene.addCollider(body: body, size: F3(0.30, 0.30, 0.20),
                              localPosition: F3(-0.35, 0, -0.80))
        _ = scene.addCollider(body: body, size: F3(0.30, 0.30, 0.20),
                              localPosition: F3(0.35, 0, -0.80))

        let solver = try makeGPU(scene)
        for _ in 0..<360 { solver.step() }
        let state = solver.bodyStates([body])[0]

        // Ground top is z=0 and each foot is 0.1 m thick, so an offset-aware
        // body rests with its COM near 0.9 m. Treating the colliders as
        // centered would instead put the COM near 0.1 m.
        XCTAssertEqual(state.position.z, 0.90, accuracy: 0.08)
        XCTAssertLessThan(abs(state.rotation.imag.x), 0.08)
        XCTAssertLessThan(abs(state.rotation.imag.y), 0.08)
        XCTAssertGreaterThanOrEqual(solver.lastNumPairs, 2)
    }

    func testConvexHullColliderRestsOnBoxTerrain() throws {
        var scene = PhysicsScene(name: "convex-hull-box-contact")
        scene.settings.dt = 1 / 120
        scene.settings.gravity = -9.81
        scene.settings.iterations = 20
        let ground = scene.addBody(
            size: F3(20, 20, 0.2), density: 0, friction: 1,
            position: F3(0, 0, -0.1))
        let body = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 1,
            position: F3(0, 0, 2), mass: 1,
            diagonalInertia: F3(repeating: 1 / 6),
            collisionEnabled: false)
        let hull = [
            F3(-0.5, -0.5, -0.5), F3(0.5, -0.5, -0.5),
            F3(-0.5, 0.5, -0.5), F3(0.5, 0.5, -0.5),
            F3(-0.5, -0.5, 0.5), F3(0.5, -0.5, 0.5),
            F3(-0.5, 0.5, 0.5), F3(0.5, 0.5, 0.5),
        ]
        let collider = scene.addConvexCollider(
            body: body, vertices: hull, friction: 1)
        XCTAssertEqual(scene.colliders[collider].convexHullVertices.count, 8)
        XCTAssertEqual(scene.colliders[collider].size, F3(repeating: 1))

        let solver = try makeGPU(scene)
        for _ in 0..<360 { solver.step() }
        let state = solver.bodyStates([body])[0]
        XCTAssertEqual(state.position.z, 0.5, accuracy: 0.06)
        XCTAssertLessThan(abs(state.rotation.imag.x), 0.05)
        XCTAssertLessThan(abs(state.rotation.imag.y), 0.05)
        XCTAssertTrue(solver.activeRigidContactPairs().contains {
            ($0.0 == ground && $0.1 == body)
                || ($0.0 == body && $0.1 == ground)
        })
    }

    /// Imported robot feet are compound capsule colliders. Their material
    /// contact anchors must persist while sticking; rebuilding those anchors
    /// every frame turns static friction into bounded per-frame creep and lets
    /// a locomotion policy translate in permanent double support.
    func testCompoundCapsuleFootHoldsUnderSubCoulombActuatorLoad() throws {
        var scene = PhysicsScene(name: "compound-capsule-static-friction")
        scene.settings.dt = 1 / 120
        scene.settings.iterations = 20
        _ = scene.addBody(size: F3(20, 8, 0.20), density: 0,
                          friction: 1.1, position: F3(0, 0, -0.10))

        let foot = scene.addBody(
            size: F3(0.35, 0.20, 0.20), density: 0, friction: 1.1,
            position: F3(0, 0, 0.36),
            mass: 5, diagonalInertia: F3(0.08, 0.12, 0.08),
            collisionEnabled: false)
        let capsuleAlongX = Quat(angle: .pi / 2, axis: F3(0, 1, 0))
        for y in [-0.07 as Float, 0.07] {
            _ = scene.addCollider(
                body: foot, size: F3(0.24, 0.035, 0), friction: 1.1,
                localPosition: F3(0, y, -0.31),
                localRotation: capsuleAlongX, shape: .capsule)
        }
        _ = scene.addCollider(
            body: foot, size: F3(0.14, 0.035, 0), friction: 1.1,
            localPosition: F3(0.14, 0, -0.31),
            localRotation: Quat(angle: .pi / 2, axis: F3(1, 0, 0)),
            shape: .capsule)
        // 500 N/m * 0.04 m = 20 N. The available Coulomb force is about
        // 1.1 * 5 kg * 10 m/s² = 55 N, so a correct static contact must hold.
        scene.addJoint(SceneJoint(
            bodyA: -1, bodyB: foot, rA: F3(0, 0, 0.36), rB: .zero,
            stiffnessLin: 500, stiffnessAng: 0))

        let solver = try makeGPU(scene)
        for _ in 0..<600 { solver.step() }
        let settled = solver.bodyPosition(foot)
        for _ in 0..<1_200 {
            let p = solver.bodyPosition(foot)
            solver.setJointWorldAnchors([.init(
                joint: 0, point: F3(p.x + 0.04, p.y, p.z))])
            solver.step()
        }
        let state = solver.bodyStates([foot])[0]

        XCTAssertLessThan(abs(state.position.x - settled.x), 0.001,
                          "compound capsule foot crept under a sub-Coulomb actuator load")
        XCTAssertLessThan(length(state.linearVelocity), 0.03)
    }

    func testSoftParticlesFreeFallLikeRigidBodiesWithDamping() throws {
        var scene = PhysicsScene(name: "soft-rigid-freefall")
        scene.settings.particleDamping = 20
        let rigid = scene.addBody(size: F3(0.4, 0.4, 0.4), density: 1,
                                  friction: 0.5, position: F3(-2, 0, 20))
        let particle = scene.addParticle(radius: 0.08, mass: 0.02,
                                         friction: 0.5, position: F3(0, 0, 20))
        let p0 = scene.addParticle(radius: 0.05, mass: 0.02,
                                   friction: 0.5, position: F3(2, 0, 19.75))
        let p1 = scene.addParticle(radius: 0.05, mass: 0.02,
                                   friction: 0.5, position: F3(2.4, 0, 19.75))
        let p2 = scene.addParticle(radius: 0.05, mass: 0.02,
                                   friction: 0.5, position: F3(2, 0.4, 19.75))
        let p3 = scene.addParticle(radius: 0.05, mass: 0.02,
                                   friction: 0.5, position: F3(2, 0, 20.75))
        scene.addTet(SceneTet(ids: (p0, p1, p2, p3), mu: 2000, lambda: 20000))
        let solver = try makeGPU(scene)

        let steps = 40
        for _ in 0..<steps { solver.step() }

        var v: Float = 0, z: Float = 20
        for _ in 0..<steps {
            v += solver.settings.gravity * solver.settings.dt
            z += v * solver.settings.dt
        }
        let tetZ = (solver.bodyPosition(p0).z + solver.bodyPosition(p1).z
                    + solver.bodyPosition(p2).z + solver.bodyPosition(p3).z) * 0.25
        XCTAssertEqual(solver.bodyPosition(rigid).z, z, accuracy: 0.02)
        XCTAssertEqual(solver.bodyPosition(particle).z, z, accuracy: 0.02)
        XCTAssertEqual(tetZ, z, accuracy: 0.02)
        XCTAssertEqual(solver.bodyVelocity(particle).z,
                       solver.bodyVelocity(rigid).z, accuracy: 0.02)
    }

    func testBoxRestsOnGround() throws {
        let solver = try makeGPU(Demos.ground())
        for _ in 0..<180 { solver.step() }
        let z = solver.bodyPosition(1).z
        XCTAssertEqual(z, 0.5, accuracy: 0.03, "box should rest at z=0.5 (got \(z))")
        XCTAssertLessThan(length(solver.bodyVelocity(1)), 0.05)
    }

    func testFastBoxDoesNotStartBelowFloor() throws {
        var scene = PhysicsScene(name: "fast-floor-box")
        _ = scene.addBody(size: F3(200, 200, 2), density: 0, friction: 0.7,
                          position: F3(0, 0, -1))
        let box = scene.addBody(size: F3(1, 1, 1), density: 1, friction: 0.5,
                                position: F3(0, 0, 0.72), velocity: F3(0, 0, -18))
        let solver = try makeGPU(scene)

        var minZ = Float.greatestFiniteMagnitude
        for _ in 0..<20 {
            solver.step()
            minZ = min(minZ, solver.bodyPosition(box).z)
        }
        XCTAssertGreaterThan(minZ, 0.45, "fast box center dipped below floor support: \(minZ)")
    }

    func testFastBoxFloorImpactDoesNotAddSpuriousSpin() throws {
        var scene = PhysicsScene(name: "fast-floor-box-no-spin")
        _ = scene.addBody(size: F3(200, 200, 2), density: 0, friction: 0.7,
                          position: F3(0, 0, -1))
        let box = scene.addBody(size: F3(1, 1, 1), density: 1, friction: 0.5,
                                position: F3(0, 0, 0.72), velocity: F3(0, 0, -18))
        let solver = try makeGPU(scene)

        for _ in 0..<20 { solver.step() }
        let spin = length(solver.bodyAngularVelocity(box))
        XCTAssertLessThan(spin, 0.05, "symmetric floor impact added spin: \(spin)")
    }

    func testFastBoxDoesNotTunnelThroughWall() throws {
        var scene = PhysicsScene(name: "fast-wall-box")
        scene.settings.gravity = 0
        _ = scene.addBody(size: F3(0.2, 20, 20), density: 0, friction: 0.7,
                          position: F3(0, 0, 0))
        let box = scene.addBody(size: F3(1, 1, 1), density: 1, friction: 0.5,
                                position: F3(-0.72, 0, 0), velocity: F3(18, 0, 0))
        let solver = try makeGPU(scene)

        var maxX = -Float.greatestFiniteMagnitude
        for _ in 0..<20 {
            solver.step()
            maxX = max(maxX, solver.bodyPosition(box).x)
        }
        XCTAssertLessThan(maxX, -0.55, "fast box center crossed wall support: \(maxX)")
    }

    func testFastParticleDoesNotStartBelowFloor() throws {
        var scene = PhysicsScene(name: "fast-floor-particle")
        _ = scene.addBody(size: F3(200, 200, 2), density: 0, friction: 0.7,
                          position: F3(0, 0, -1))
        let p = scene.addParticle(radius: 0.05, mass: 0.02, friction: 0.5,
                                  position: F3(0, 0, 0.16), velocity: F3(0, 0, -8))
        let solver = try makeGPU(scene)

        var minZ = Float.greatestFiniteMagnitude
        for _ in 0..<20 {
            solver.step()
            minZ = min(minZ, solver.bodyPosition(p).z)
        }
        XCTAssertGreaterThan(minZ, 0.035, "fast particle center dipped below floor support: \(minZ)")
    }

    func testPendulumJointHolds() throws {
        var scene = PhysicsScene(name: "pend")
        let bob = scene.addBody(size: F3(0.5, 0.5, 0.5), density: 1, friction: 0.5,
                                position: F3(1, 0, 0))
        scene.addJoint(SceneJoint(bodyA: -1, bodyB: bob, rA: F3(0, 0, 0), rB: F3(-1, 0, 0)))
        let solver = try makeGPU(scene)
        for _ in 0..<120 { solver.step() }
        XCTAssertLessThan(solver.maxConstraintError(), 0.01)
    }

    func testStackStability() throws {
        let solver = try makeGPU(Demos.stack(height: 5))
        for _ in 0..<300 { solver.step() }
        for i in 0..<5 {
            let p = solver.bodyPosition(1 + i)
            XCTAssertEqual(p.z, 0.5 + Float(i), accuracy: 0.08, "box \(i) z")
            XCTAssertLessThan(length(F3(p.x, p.y, 0)), 0.15, "box \(i) lateral drift")
        }
    }

    func testContactRichTrajectoryIsBitwiseDeterministic() throws {
        var scene = PhysicsScene(name: "deterministic-contact-grid")
        scene.settings.dt = 1 / 120
        scene.settings.iterations = 12
        _ = scene.addBody(size: F3(80, 20, 0.2), density: 0,
                          friction: 0.9, position: F3(0, 0, -0.1))
        var dynamicBodies = [Int]()
        for lane in 0..<16 {
            let x = Float(lane) * 2 - 15
            for level in 0..<4 {
                dynamicBodies.append(scene.addBody(
                    size: F3(0.8, 0.8, 0.8), density: 1,
                    friction: 0.8,
                    position: F3(x, 0, 0.4 + Float(level) * 0.81)))
            }
        }

        let first = try makeGPU(scene)
        let second = try makeGPU(scene)
        for _ in 0..<240 {
            first.step()
            second.step()
        }
        first.sync()
        second.sync()
        let firstStates = first.bodyStates(dynamicBodies)
        let secondStates = second.bodyStates(dynamicBodies)
        XCTAssertEqual(first.lastNumPairs, second.lastNumPairs)
        XCTAssertEqual(firstStates.count, secondStates.count)
        for i in firstStates.indices {
            let a = firstStates[i]
            let b = secondStates[i]
            XCTAssertEqual(a.position.x.bitPattern, b.position.x.bitPattern)
            XCTAssertEqual(a.position.y.bitPattern, b.position.y.bitPattern)
            XCTAssertEqual(a.position.z.bitPattern, b.position.z.bitPattern)
            XCTAssertEqual(a.rotation.imag.x.bitPattern,
                           b.rotation.imag.x.bitPattern)
            XCTAssertEqual(a.rotation.imag.y.bitPattern,
                           b.rotation.imag.y.bitPattern)
            XCTAssertEqual(a.rotation.imag.z.bitPattern,
                           b.rotation.imag.z.bitPattern)
            XCTAssertEqual(a.rotation.real.bitPattern,
                           b.rotation.real.bitPattern)
            XCTAssertEqual(a.linearVelocity.x.bitPattern,
                           b.linearVelocity.x.bitPattern)
            XCTAssertEqual(a.linearVelocity.y.bitPattern,
                           b.linearVelocity.y.bitPattern)
            XCTAssertEqual(a.linearVelocity.z.bitPattern,
                           b.linearVelocity.z.bitPattern)
        }
    }

    /// GPU trajectory should track the CPU reference closely for a couple of
    /// seconds on a contact-rich scene (identical algorithm, GS-order differs).
    func testCPUGPUParity() throws {
        let scene = Demos.stack(height: 3)
        let cpu = scene.makeCPUSolver()
        let gpu = try makeGPU(scene)

        for _ in 0..<120 {
            cpu.step()
            gpu.step()
        }
        for i in 0..<scene.bodies.count {
            let pc = cpu.bodies[i].positionLin
            let pg = gpu.bodyPosition(i)
            XCTAssertLessThan(length(pc - pg), 0.05,
                              "body \(i): cpu \(pc) vs gpu \(pg)")
        }
    }
}
