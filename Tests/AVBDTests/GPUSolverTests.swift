import XCTest
import simd
@testable import AVBDCore

final class GPUSolverTests: XCTestCase {
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

    func testBoxRestsOnGround() throws {
        let solver = try makeGPU(Demos.ground())
        for _ in 0..<180 { solver.step() }
        let z = solver.bodyPosition(1).z
        XCTAssertEqual(z, 0.5, accuracy: 0.03, "box should rest at z=0.5 (got \(z))")
        XCTAssertLessThan(length(solver.bodyVelocity(1)), 0.05)
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
