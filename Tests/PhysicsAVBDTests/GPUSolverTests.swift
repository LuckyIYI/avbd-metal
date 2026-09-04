import XCTest
import simd
@testable import SimCore
@testable import PhysicsAVBD
@testable import GPUSimDemos

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

    func testCheckedSubmissionFailureIsStickyAndDoesNotAdvanceFrame() throws {
        let solver = try makeGPU(Demos.ground())
        solver.commandBufferFactoryForTesting = { nil }

        XCTAssertThrowsError(try solver.submitStep()) { error in
            XCTAssertEqual(
                error as? GPUSolver.RuntimeFailure,
                .commandBufferCreation(operation: "physics", frame: 1))
        }
        XCTAssertEqual(solver.frameIndex, 0)
        XCTAssertEqual(
            solver.runtimeFailure,
            .commandBufferCreation(operation: "physics", frame: 1))
        XCTAssertThrowsError(try solver.submitStep()) { error in
            XCTAssertEqual(error as? GPUSolver.RuntimeFailure,
                           solver.runtimeFailure)
        }
    }

    func testLegacyRateMotorAdvancesWrappedServoTarget() throws {
        var scene = PhysicsScene(name: "legacy-rate-motor")
        scene.settings.dt = 0.25
        scene.settings.gravity = 0
        let body = scene.addBody(
            size: F3(repeating: 0.2), density: 1, friction: 0,
            position: .zero)
        scene.addJoint(SceneJoint(
            bodyA: -1, bodyB: body, rA: .zero, rB: .zero,
            stiffnessLin: .infinity, stiffnessAng: .infinity,
            hingeAxis: F3(0, 0, 1), motorTorque: 10,
            motorRate: 2))
        let solver = try makeGPU(scene)

        XCTAssertEqual(solver.motorTargetForTesting(0), 0)
        try solver.submitStep()
        try solver.synchronize()

        XCTAssertEqual(solver.motorTargetForTesting(0), 0.5,
                       accuracy: 1e-6)
    }

    func testSynchronousSecondFrameFailureDrainsPriorSubmission() throws {
        let solver = try makeGPU(Demos.ground())
        try solver.submitStep()
        XCTAssertEqual(solver.inflightCountForTesting, 1)
        solver.commandBufferFactoryForTesting = { nil }

        XCTAssertThrowsError(try solver.submitStep()) { error in
            XCTAssertEqual(
                error as? GPUSolver.RuntimeFailure,
                .commandBufferCreation(operation: "physics", frame: 2))
        }

        XCTAssertEqual(solver.inflightCountForTesting, 0)
        XCTAssertGreaterThan(solver.lastPairCandidates, 0,
                             "the older submitted frame was not retired")
        XCTAssertEqual(solver.frameIndex, 1)
    }

    func testCheckedEncoderFailureNamesStageAndDoesNotCommitFrame() throws {
        let solver = try makeGPU(Demos.ground())
        solver.deniedEncoderStageForTesting = "narrowphase"

        XCTAssertThrowsError(try solver.submitStep()) { error in
            XCTAssertEqual(
                error as? GPUSolver.RuntimeFailure,
                .commandEncoderCreation(
                    operation: "physics", stage: "narrowphase", frame: 1))
        }
        XCTAssertEqual(solver.frameIndex, 0)
    }

    func testCheckedDiagnosticsCannotReportFalsePerfectOnMetalFailure() throws {
        let solver = try makeGPU(Demos.ground())
        solver.commandBufferFactoryForTesting = { nil }

        XCTAssertThrowsError(try solver.maxConstraintErrorChecked()) { error in
            XCTAssertEqual(
                error as? GPUSolver.RuntimeFailure,
                .commandBufferCreation(
                    operation: "constraint diagnostics", frame: 0))
        }
    }

    func testAsynchronousExecutionFailureIsStickyAndDrainsAllFrames() throws {
        let solver = try makeGPU(Demos.ground())
        try solver.submitStep()
        try solver.submitStep()
        let injected = GPUSolver.RuntimeFailure.commandExecution(
            operation: "physics", frame: 1, status: -1,
            domain: "AVBDTests", code: 7, message: "synthetic GPU fault")
        solver.completionFailureForTesting = { operation, frame in
            operation == "physics" && frame == 1 ? injected : nil
        }

        XCTAssertThrowsError(try solver.synchronize()) { error in
            XCTAssertEqual(error as? GPUSolver.RuntimeFailure, injected)
        }
        XCTAssertEqual(solver.inflightCountForTesting, 0)
        XCTAssertEqual(solver.runtimeFailure, injected)
        XCTAssertThrowsError(try solver.submitStep()) { error in
            XCTAssertEqual(error as? GPUSolver.RuntimeFailure, injected)
        }
    }

    func testProfilingToggleRetiresPriorReadbackOwner() throws {
        let solver = try makeGPU(Demos.ground())
        try solver.submitStep()
        solver.profiling = true

        try solver.submitStep()
        XCTAssertEqual(solver.profileFrames, 1)
        XCTAssertGreaterThan(solver.lastPairCandidates, 0)
        XCTAssertEqual(solver.inflightCountForTesting, 0)

        solver.profiling = false
        try solver.submitStep()
        try solver.synchronize()
        XCTAssertNil(solver.runtimeFailure)
        XCTAssertEqual(solver.frameIndex, 3)
    }

    private func rigidCapacityScene(sharedStatics: Int) -> PhysicsScene {
        var scene = PhysicsScene(name: "rigid-capacity-\(sharedStatics)")
        scene.settings.gravity = 0
        scene.settings.iterations = 1
        let size = F3(repeating: 0.1)
        for _ in 0..<sharedStatics {
            _ = scene.addBody(
                size: size, density: 0, friction: 0, position: .zero,
                shape: .sphere)
        }
        // Nonzero groups isolate dynamic bodies from each other while shared
        // group-zero statics collide with every replica. Demand is therefore
        // exactly 64 * sharedStatics with no high-chromatic dynamic clique.
        for group in 1...64 {
            _ = scene.addBody(
                size: size, density: 1, friction: 0, position: .zero,
                shape: .sphere, collisionGroup: UInt32(group))
        }
        return scene
    }

    func testExactRigidPairCapacityIsValid() throws {
        let solver = try GPUSolver(
            scene: rigidCapacityScene(sharedStatics: 64),
            maxPairsPerBody: 0)
        XCTAssertEqual(solver.maxPairs, 4_096)

        try solver.submitStep()
        try solver.synchronize()

        XCTAssertEqual(solver.lastPairCandidates, 4_096)
        XCTAssertEqual(solver.lastNumPairs, 4_096)
        XCTAssertNil(solver.runtimeFailure)
    }

    func testRigidPairOverflowPoisonsInsteadOfDroppingContacts() throws {
        let solver = try GPUSolver(
            scene: rigidCapacityScene(sharedStatics: 65),
            maxPairsPerBody: 0)
        try solver.submitStep()

        XCTAssertThrowsError(try solver.synchronize()) { error in
            XCTAssertEqual(
                error as? GPUSolver.RuntimeFailure,
                .rigidPairCapacity(
                    frame: 1, required: 4_160, capacity: 4_096))
        }
        XCTAssertEqual(solver.lastPairCandidates, 4_160)
        XCTAssertEqual(solver.lastNumPairs, 4_096)
        XCTAssertEqual(solver.runtimeFailure,
                       .rigidPairCapacity(
                           frame: 1, required: 4_160, capacity: 4_096))
    }

    func testUnresolvedDynamicColoringPoisonsTheSolve() throws {
        var scene = PhysicsScene(name: "color-capacity")
        scene.settings.gravity = 0
        scene.settings.iterations = 1
        for _ in 0..<65 {
            _ = scene.addBody(
                size: F3(repeating: 0.1), density: 1, friction: 0,
                position: .zero, shape: .sphere)
        }
        let solver = try makeGPU(scene)
        try solver.submitStep()

        XCTAssertThrowsError(try solver.synchronize()) { error in
            guard let failure = error as? GPUSolver.RuntimeFailure,
                  case .unresolvedColoring(let frame, let conflicts)
                    = failure else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(frame, 1)
            XCTAssertGreaterThan(conflicts, 0)
        }
    }

    func testLargeUIStackRepairsDynamicColoringAcrossContactChanges() throws {
        var scene = try XCTUnwrap(Demos.make("stack", scale: 4))
        _ = scene.addDragSlot()
        let solver = try makeGPU(scene)

        for _ in 0..<120 {
            try solver.submitStep()
            try solver.synchronize()
        }

        XCTAssertNil(solver.runtimeFailure)
        XCTAssertLessThan(solver.lastMaxColorUsed, AVBD_MAX_COLORS)
        let colorChanges = solver.changedFlag.contents().bindMemory(
            to: UInt32.self, capacity: 24)
        XCTAssertEqual(
            colorChanges[19], 0,
            "a settled dense stack must not run the serial coloring fallback")
    }

    func testLargeBoxOfBoxesRunsCompactRigidSolve() throws {
        var scene = Demos.boxOfBoxes(scale: 4)
        scene.settings.iterations = 2
        let solver = try makeGPU(scene)

        XCTAssertGreaterThan(solver.bodyCount, 1_024)
        try solver.submitStep()
        try solver.synchronize()

        XCTAssertNil(solver.runtimeFailure)
        XCTAssertGreaterThan(solver.lastNumPairs, 0)
    }

    func testStaticColorPaletteExhaustionFailsDuringInitialization() throws {
        var scene = PhysicsScene(name: "static-color-capacity")
        scene.settings.gravity = 0
        let particles = (0..<65).map { index in
            scene.addParticle(
                radius: 0.01, mass: 0.01, friction: 0,
                position: F3(Float(index), 0, 1))
        }
        scene.addTri(SceneTri(ids: (particles[0], particles[1], particles[2])))
        for a in particles.indices {
            for b in (a + 1)..<particles.count {
                scene.addJoint(SceneJoint(
                    bodyA: particles[a], bodyB: particles[b],
                    rA: .zero, rB: .zero, stiffnessLin: 1))
            }
        }

        XCTAssertThrowsError(try GPUSolver(scene: scene)) { error in
            XCTAssertEqual(
                error as? GPUSolver.RuntimeFailure,
                .staticColorCapacity(body: 64, required: 65, capacity: 64))
        }
    }

    /// A skin vertex outside its tet must keep its rest offset from the tet
    /// under compression (carried by the tet's rotation) instead of being
    /// scaled with the tet's strain, while inside vertices stay barycentric.
    func testSkinOverhangIsCarriedRigidlyNotAffinely() throws {
        var scene = Demos.ground()
        // Rest tet: unit-ish corners. Current pose: the same tet compressed
        // to half size, authored directly as static particles.
        let rest: [F3] = [F3(0, 0, 0), F3(0.1, 0, 0), F3(0, 0.1, 0), F3(0, 0, 0.1)]
        // Current pose: anisotropic compression followed by a rotation, so
        // the polar factor is a non-trivial rotation.
        let rot = simd_quatf(angle: 0.7, axis: normalize(F3(1, 2, 3)))
        // Strongly anisotropic: an unscaled polar iteration is nowhere near
        // orthogonal after a handful of steps at this conditioning.
        let stretch = simd_float3x3(diagonal: F3(0.002, 0.8, 0.6))
        let current = rest.map { rot.act(stretch * $0) + F3(0, 0, 0.3) }
        var ids: [Int] = []
        for p in current {
            ids.append(scene.addParticle(radius: 0.005, mass: 0, position: p))
        }
        let dm = simd_float3x3(columns: (rest[1] - rest[0], rest[2] - rest[0],
                                         rest[3] - rest[0]))
        let inv = dm.inverse
        func row(_ r: Int) -> F3 {
            F3(inv.columns.0[r], inv.columns.1[r], inv.columns.2[r])
        }
        let inside = SIMD4<Float>(0.25, 0.25, 0.25, 0.25)
        let outside = SIMD4<Float>(-2, 1, 1, 1)   // three tet lengths out
        let tuple = (ids[0], ids[1], ids[2], ids[3])
        let mesh = SceneSkinnedMesh(
            vertices: [
                SceneSkinnedVertex(ids: tuple, weights: inside,
                                   restNormal: F3(0, 0, 1),
                                   restInv0: row(0), restInv1: row(1),
                                   restInv2: row(2)),
                SceneSkinnedVertex(ids: tuple, weights: outside,
                                   restNormal: F3(0, 0, 1),
                                   restInv0: row(0), restInv1: row(1),
                                   restInv2: row(2)),
            ],
            triangles: [(0, 1, 1)], bodyIDs: ids)
        scene.addSkinnedMesh(mesh)
        let solver = try makeGPU(scene)
        solver.step()
        solver.sync()
        guard let queue = solver.metalDevice.makeCommandQueue(),
              let command = queue.makeCommandBuffer(),
              let instances = solver.metalDevice.makeBuffer(length: 4096)
        else { return XCTFail("could not allocate render test resources") }
        try solver.encodeBuildInstancesChecked(command, instances: instances)
        command.commit()
        command.waitUntilCompleted()
        let surface = try XCTUnwrap(solver.renderSkinnedSurface)
        let out = surface.vertices.contents()
            .bindMemory(to: SkinVertexGPU.self, capacity: 2)
        func bary(_ w: SIMD4<Float>, _ x: [F3]) -> F3 {
            x[0] * w.x + x[1] * w.y + x[2] * w.z + x[3] * w.w
        }
        let drawnInside = F3(out[0].position.x, out[0].position.y, out[0].position.z)
        XCTAssertLessThan(simd_length(drawnInside - bary(inside, current)), 1e-6)
        // Affine extrapolation would place the overhang at bary(outside,
        // current): half the rest offset. Rigid carriage keeps the full
        // rest offset, turned by the rotation part of F.
        let anchor = bary(SIMD4<Float>(0, 1, 1, 1) / 3, current)
        let restOffset = bary(outside, rest) - bary(SIMD4<Float>(0, 1, 1, 1) / 3, rest)
        let expected = anchor + rot.act(restOffset)
        let drawnOutside = F3(out[1].position.x, out[1].position.y, out[1].position.z)
        XCTAssertLessThan(simd_length(drawnOutside - expected), 1e-5,
            "overhang \(drawnOutside) expected \(expected), affine would be "
                + "\(bary(outside, current))")
        XCTAssertGreaterThan(
            simd_length(drawnOutside - bary(outside, current)), 0.01,
            "the overhang must not follow the affine extrapolation")
    }

    /// An inverted tet has no rotation to carry an overhang with: the
    /// vertex must fall back to the affine map instead of shooting away.
    func testSkinOverhangFallsBackToAffineOnInvertedTet() throws {
        var scene = Demos.ground()
        let rest: [F3] = [F3(0, 0, 0), F3(0.1, 0, 0), F3(0, 0.1, 0), F3(0, 0, 0.1)]
        let mirror = simd_float3x3(diagonal: F3(-0.5, 1, 1))     // det < 0
        let current = rest.map { mirror * $0 + F3(0, 0, 0.3) }
        var ids: [Int] = []
        for p in current { ids.append(scene.addParticle(radius: 0.005, mass: 0, position: p)) }
        let dm = simd_float3x3(columns: (rest[1] - rest[0], rest[2] - rest[0], rest[3] - rest[0]))
        let inv = dm.inverse
        func row(_ r: Int) -> F3 { F3(inv.columns.0[r], inv.columns.1[r], inv.columns.2[r]) }
        let outside = SIMD4<Float>(-2, 1, 1, 1)
        let tuple = (ids[0], ids[1], ids[2], ids[3])
        scene.addSkinnedMesh(SceneSkinnedMesh(
            vertices: [SceneSkinnedVertex(ids: tuple, weights: outside, restNormal: F3(0, 0, 1),
                                          restInv0: row(0), restInv1: row(1), restInv2: row(2))],
            triangles: [(0, 0, 0)], bodyIDs: ids))
        let solver = try makeGPU(scene)
        solver.step()
        solver.sync()
        guard let queue = solver.metalDevice.makeCommandQueue(),
              let command = queue.makeCommandBuffer(),
              let instances = solver.metalDevice.makeBuffer(length: 4096)
        else { return XCTFail("could not allocate render test resources") }
        try solver.encodeBuildInstancesChecked(command, instances: instances)
        command.commit()
        command.waitUntilCompleted()
        let surface = try XCTUnwrap(solver.renderSkinnedSurface)
        let out = surface.vertices.contents().bindMemory(to: SkinVertexGPU.self, capacity: 1)
        let drawn = F3(out[0].position.x, out[0].position.y, out[0].position.z)
        let affine = current[0] * outside.x + current[1] * outside.y
            + current[2] * outside.z + current[3] * outside.w
        XCTAssertLessThan(simd_length(drawn - affine), 1e-5,
                          "inverted tet: drawn \(drawn) should equal the affine map \(affine)")
    }

    func testCheckedRenderInstanceEncoderFailureIsNotSilentOrPhysicsFatal() throws {
        let solver = try makeGPU(Demos.ground())
        guard let queue = solver.metalDevice.makeCommandQueue(),
              let command = queue.makeCommandBuffer(),
              let instances = solver.metalDevice.makeBuffer(length: 256)
        else { return XCTFail("could not allocate render test resources") }
        solver.deniedEncoderStageForTesting = "render-instances"

        XCTAssertThrowsError(try solver.encodeBuildInstancesChecked(
            command, instances: instances)) { error in
            XCTAssertEqual(
                error as? GPUSolver.RuntimeFailure,
                .commandEncoderCreation(
                    operation: "render instances", stage: "build-instances",
                    frame: 0))
        }
        XCTAssertNil(solver.runtimeFailure,
                     "a visual-frame failure must not poison physics")
    }

    func testFracturedJointInspectionAndSelectiveRepair() throws {
        var scene = PhysicsScene(name: "joint-repair")
        scene.settings.gravity = 0
        let a = scene.addBody(
            size: F3(repeating: 0.2), density: 1_000, friction: 0,
            position: .zero)
        let b = scene.addBody(
            size: F3(repeating: 0.2), density: 1_000, friction: 0,
            position: F3(0.3, 0, 0))
        scene.addJoint(SceneJoint(
            bodyA: a, bodyB: b, rA: .zero, rB: .zero,
            stiffnessLin: .infinity, stiffnessAng: .infinity,
            fracture: 10))
        scene.addJoint(SceneJoint(
            bodyA: a, bodyB: b, rA: .zero, rB: .zero,
            stiffnessLin: 250, stiffnessAng: 125, fracture: 10))
        let drag = scene.addDragSlot()
        let solver = try makeGPU(scene)

        let joints = solver.joints.contents().bindMemory(
            to: JointGPU.self, capacity: scene.joints.count)
        let initialPenalty0 = joints[0].penaltyLin
        let initialPenalty1 = joints[1].penaltyAng
        for index in [0, 1, drag] {
            joints[index].header.z = 1
            joints[index].lambdaLin = SIMD4(repeating: 7)
            joints[index].lambdaAng = SIMD4(repeating: 9)
            joints[index].penaltyLin = SIMD4(repeating: 99)
            joints[index].penaltyAng = SIMD4(repeating: 101)
            joints[index].motor.z = 4
            joints[index].dynamics.y = 5
            joints[index].dynamics.z = 6
        }

        XCTAssertEqual(solver.brokenJointIndices(), [0, 1])
        XCTAssertEqual(solver.debugBrokenJoints(), 2)

        solver.repairJoints([1, 1, drag])
        XCTAssertEqual(solver.brokenJointIndices(), [0])
        XCTAssertEqual(joints[1].header.z, 0)
        XCTAssertEqual(joints[1].lambdaLin, .zero)
        XCTAssertEqual(joints[1].lambdaAng, .zero)
        XCTAssertEqual(joints[1].penaltyAng, initialPenalty1)
        XCTAssertEqual(joints[1].motor.z, 0)
        XCTAssertEqual(joints[1].dynamics.y, 0)
        XCTAssertEqual(joints[1].dynamics.z, 0)
        XCTAssertEqual(joints[drag].header.z, 1,
                       "repair must not activate a disabled drag slot")

        solver.repairJoints()
        XCTAssertEqual(solver.brokenJointIndices(), [])
        XCTAssertEqual(joints[0].header.z, 0)
        XCTAssertEqual(joints[0].penaltyLin, initialPenalty0)
    }

    func testDisconnectedSoftMeshUsesStructuralContactCapacity() throws {
        var scene = PhysicsScene(name: "disconnected-soft-capacity")
        for triangle in 0..<50 {
            let x = Float(triangle) * 2
            let ids = (0..<3).map { corner in
                scene.addParticle(
                    radius: 0.01, mass: 0.01, friction: 0,
                    position: F3(x, Float(corner), 1))
            }
            scene.addTri(SceneTri(ids: (ids[0], ids[1], ids[2])))
        }
        let solver = try makeGPU(scene)

        // Full Planar emission budgets 8 records per surface vertex/edge,
        // plus four rigid-triangle records per triangle. Raw demand is still
        // checked exactly and fails terminally rather than clipping silently.
        XCTAssertEqual(solver.maxSoft, 2_600)
    }

    func testShadersCompile() throws {
        let solver = try makeGPU(Demos.ground())
        XCTAssertNotNil(solver.device)
        XCTAssertTrue(GPUSolver.requiredKernelNames.isSubset(of: solver.pso.keys))
    }

    func testRequiredKernelValidationFailsClosedDeterministically() throws {
        try GPUSolver.validateRequiredKernelNames(GPUSolver.requiredKernelNames)

        var incomplete = GPUSolver.requiredKernelNames
        incomplete.remove("build_instances")
        incomplete.remove("adj_clear_degrees")
        XCTAssertThrowsError(
            try GPUSolver.validateRequiredKernelNames(incomplete)
        ) { error in
            XCTAssertEqual(
                error as? GPUSolver.AVBDError,
                .kernelMissing("adj_clear_degrees"))
        }
    }

    func testMissingShaderResourcesFailWithTypedError() throws {
        XCTAssertThrowsError(
            try GPUSolver.validateShaderResourceURLs([])
        ) { error in
            XCTAssertEqual(
                error as? GPUSolver.AVBDError,
                .shaderCompile(
                    "no .metal resources were found in the PhysicsAVBD bundle"))
        }
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

    func testFastOpposingDynamicBoxesDoNotTunnelThroughEachOther() throws {
        var scene = PhysicsScene(name: "fast-opposing-boxes")
        scene.settings.gravity = 0
        scene.settings.iterations = 20
        let left = scene.addBody(
            size: F3(1, 1, 1), density: 1, friction: 0,
            position: F3(-0.6, 0, 0), velocity: F3(42, 0, 0))
        let right = scene.addBody(
            size: F3(1, 1, 1), density: 1, friction: 0,
            position: F3(0.6, 0, 0), velocity: F3(-42, 0, 0))
        let solver = try makeGPU(scene)

        var minimumOrder = Float.greatestFiniteMagnitude
        for _ in 0..<20 {
            solver.step()
            minimumOrder = min(
                minimumOrder,
                solver.bodyPosition(right).x - solver.bodyPosition(left).x)
        }
        let finalOrder = solver.bodyPosition(right).x
            - solver.bodyPosition(left).x
        XCTAssertGreaterThan(
            minimumOrder, 0,
            "opposing dynamic boxes crossed through each other: \(minimumOrder)")
        XCTAssertGreaterThan(
            finalOrder, 0.99,
            "opposing boxes did not resolve their overlap: \(finalOrder)")
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

    /// Deformable contacts (V-T, E-E, rigid-triangle, Planar-DAT) must be as
    /// reproducible as rigid manifolds. Their emission is parallel, so the
    /// solver has to assign contact indices and element candidate order by
    /// content rather than by which thread won an atomic append.
    /// The bounded colour loop plus the colour-tail dispatch must reproduce
    /// the fully dispatched loop bit for bit: run one solver with a margin
    /// that never engages the tail and one that pushes every colour into it.
    private func assertBitwiseEqualStates(
        _ a: GPUSolver, _ b: GPUSolver, bodies: [Int], _ label: String,
        file: StaticString = #filePath, line: UInt = #line) {
        let sa = a.bodyStates(bodies)
        let sb = b.bodyStates(bodies)
        var mismatches = 0
        for i in sa.indices where !bitwiseEqual(sa[i], sb[i]) {
            mismatches += 1
        }
        XCTAssertEqual(mismatches, 0,
            "\(label): \(mismatches) of \(bodies.count) bodies differ",
            file: file, line: line)
    }

    private func bitwiseEqual(
        _ a: GPUSolver.RigidBodyState, _ b: GPUSolver.RigidBodyState
    ) -> Bool {
        a.position.x.bitPattern == b.position.x.bitPattern
            && a.position.y.bitPattern == b.position.y.bitPattern
            && a.position.z.bitPattern == b.position.z.bitPattern
            && a.rotation.imag.x.bitPattern == b.rotation.imag.x.bitPattern
            && a.rotation.imag.y.bitPattern == b.rotation.imag.y.bitPattern
            && a.rotation.imag.z.bitPattern == b.rotation.imag.z.bitPattern
            && a.rotation.real.bitPattern == b.rotation.real.bitPattern
            && a.linearVelocity.x.bitPattern == b.linearVelocity.x.bitPattern
            && a.linearVelocity.y.bitPattern == b.linearVelocity.y.bitPattern
            && a.linearVelocity.z.bitPattern == b.linearVelocity.z.bitPattern
            && a.angularVelocity.x.bitPattern == b.angularVelocity.x.bitPattern
            && a.angularVelocity.y.bitPattern == b.angularVelocity.y.bitPattern
            && a.angularVelocity.z.bitPattern == b.angularVelocity.z.bitPattern
    }

    func testDispatchMetricCountsEachEncodedDispatchOnce() throws {
        let solver = try makeGPU(Demos.stack(height: 2))
        try solver.submitStep()
        try solver.synchronize()
        XCTAssertGreaterThan(solver.dispatchesLastFrame, 0)
        XCTAssertEqual(solver.dispatchesLastFrame,
                       solver.dispatchNamesLastFrame["raw"],
            "raw is a category for every encoded dispatch, not a second dispatch")
    }

    /// The parallel rank sorts (bp_rank_cells, adj_rank) replace serial
    /// per-bucket insertion sorts; the sorted streams must be identical, so
    /// the trajectories must match bit for bit.
    func testParallelRankSortsMatchSerialSorts() throws {
        var cloth = Demos.cloth(res: 12, ball: true)
        cloth.settings.deterministic = true
        var pile = Demos.boxpile(count: 60)
        pile.settings.deterministic = true
        for (label, scene) in [("cloth", cloth), ("boxpile", pile)] {
            let bodies = Array(0..<scene.bodies.count)
            let parallel = try makeGPU(scene)
            let serial = try makeGPU(scene)
            serial.legacySerialSortsForTesting = true
            for _ in 0..<120 {
                parallel.step()
                serial.step()
            }
            parallel.sync()
            serial.sync()
            XCTAssertNil(parallel.runtimeFailure, label)
            assertBitwiseEqualStates(parallel, serial, bodies: bodies, label)
        }
    }

    /// A single tet solid without cloth vertices never emits a V-T or E-E
    /// contact (the kernels drop every same-solid candidate), so skipping the
    /// element grid and both emitters must leave the trajectory untouched.
    func testSurfaceEmissionSkipIsExactForSingleSolid() throws {
        var scene = Demos.softbody()
        scene.settings.deterministic = true
        let bodies = Array(0..<scene.bodies.count)
        let skipping = try makeGPU(scene)
        let emitting = try makeGPU(scene)
        emitting.forceSurfaceEmissionForTesting = true
        for s in [skipping, emitting] { s.surfaceTruncationMode = .isotropicDAT }
        XCTAssertGreaterThan(skipping.numTris, 0, "fixture needs boundary faces")
        for _ in 0..<120 {
            skipping.step()
            emitting.step()
        }
        skipping.sync()
        emitting.sync()
        XCTAssertNil(skipping.runtimeFailure)
        XCTAssertLessThan(skipping.dispatchesLastFrame,
                          emitting.dispatchesLastFrame,
                          "the skip must remove dispatches")
        assertBitwiseEqualStates(skipping, emitting, bodies: bodies, "softbody")
    }

    func testDynamicColorTailIsBitwiseExact() throws {
        for (label, torsion) in [("plain", false), ("torsion", true)] {
            var scene = Demos.cloth(res: 12, ball: true)
            scene.settings.deterministic = true
            if torsion {
                for i in scene.colliders.indices {
                    scene.colliders[i].torsionalFriction = 0.02
                }
            }
            let bodies = Array(0..<scene.bodies.count)
            let loop = try makeGPU(scene)
            let tail = try makeGPU(scene)
            loop.surfaceTruncationMode = .isotropicDAT
            tail.surfaceTruncationMode = .isotropicDAT
            loop.colorBoundMarginForTesting = AVBD_MAX_COLORS
            tail.colorBoundMarginForTesting = -AVBD_MAX_COLORS
            var tailDispatches = 0
            tail.solveDispatchObserverForTesting = {
                if case .primalTail = $0 { tailDispatches += 1 }
            }
            for _ in 0..<120 {
                loop.step()
                tail.step()
            }
            loop.sync()
            tail.sync()
            XCTAssertNil(tail.runtimeFailure, label)
            XCTAssertEqual(loop.lastColorTailColors, 0, label)
            XCTAssertEqual(tail.lastColorTailColors,
                           tail.lastMaxColorUsed + 1, label)
            XCTAssertGreaterThan(tailDispatches, 0, label)
            XCTAssertGreaterThan(tail.lastNumSoft, 0, label)
            assertBitwiseEqualStates(
                loop, tail, bodies: bodies,
                "\(label) dispatched loop vs colour tail "
                    + "(colours \(tail.lastMaxColorUsed + 1))")
        }
    }

    func testDeformableContactTrajectoryIsBitwiseDeterministic() throws {
        // res 12 stays under the 1024-body ordered-coloring limit; res 34
        // exceeds it and exercises the compact multi-slot neighbor stream.
        for (mode, res) in [
            (GPUSolver.SurfaceTruncationMode.isotropicDAT, 12),
            (.planarDAT, 12), (.isotropicDAT, 34),
        ] {
            var scene = Demos.cloth(res: res, ball: true)
            scene.settings.deterministic = true
            let bodies = Array(0..<scene.bodies.count)
            let first = try makeGPU(scene)
            let second = try makeGPU(scene)
            first.surfaceTruncationMode = mode
            second.surfaceTruncationMode = mode
            for _ in 0..<150 {
                first.step()
                second.step()
            }
            first.sync()
            second.sync()
            XCTAssertNil(first.runtimeFailure)
            XCTAssertGreaterThan(first.lastNumSoft, 0,
                "\(mode): the fixture must exercise soft contacts")
            XCTAssertGreaterThan(res * res + 20, 0)
            print("deterministic \(mode) res \(res): bodies \(bodies.count) "
                + "colors \(first.lastMaxColorUsed + 1) soft \(first.lastNumSoft)")
            XCTAssertEqual(first.lastNumSoft, second.lastNumSoft, "\(mode)")
            let a = first.bodyStates(bodies)
            let b = second.bodyStates(bodies)
            var mismatches = 0
            for i in a.indices where a[i].position.x.bitPattern
                != b[i].position.x.bitPattern
                || a[i].position.y.bitPattern != b[i].position.y.bitPattern
                || a[i].position.z.bitPattern != b[i].position.z.bitPattern
                || a[i].linearVelocity.x.bitPattern
                    != b[i].linearVelocity.x.bitPattern
                || a[i].rotation.real.bitPattern
                    != b[i].rotation.real.bitPattern {
                mismatches += 1
            }
            XCTAssertEqual(mismatches, 0,
                "\(mode): \(mismatches) of \(bodies.count) bodies diverged "
                    + "between two identical solvers")
        }
    }

    func testDeformableDeterministicColoringTracksRuntimeSetting() throws {
        var scene = Demos.cloth(res: 8, ball: true)
        scene.settings.deterministic = false
        let solver = try makeGPU(scene)
        XCTAssertFalse(solver.usesDynamicColoring)

        solver.settings.deterministic = true
        XCTAssertTrue(solver.usesDynamicColoring)
        try solver.submitStep()
        try solver.synchronize()
        XCTAssertNil(solver.runtimeFailure)

        solver.settings.deterministic = false
        XCTAssertFalse(solver.usesDynamicColoring)
        try solver.submitStep()
        try solver.synchronize()
        XCTAssertNil(solver.runtimeFailure,
            "switching back must use the preserved static topology palette")
    }

    func testRuntimeDeterministicColoringLazilyAllocatesLargeNeighborStream() throws {
        var scene = Demos.cloth(res: 34, ball: false)
        scene.settings.deterministic = false
        let solver = try makeGPU(scene)
        XCTAssertEqual(solver.adjNeighbor.length, 16,
            "ordinary large cloth should keep the construction-time placeholder")

        solver.settings.deterministic = true
        try solver.submitStep()
        try solver.synchronize()
        XCTAssertGreaterThan(solver.adjNeighbor.length, 16)
        XCTAssertTrue(solver.usesDynamicColoring)
        XCTAssertNil(solver.runtimeFailure)
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
