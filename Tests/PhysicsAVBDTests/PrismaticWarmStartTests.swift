import XCTest
import simd
@testable import SimCore
@testable import PhysicsAVBD

final class PrismaticWarmStartTests: XCTestCase {
    private func rail() -> PhysicsScene {
        var scene = PhysicsScene(name: "fast rail stop crossing")
        scene.settings.gravity = 0
        scene.settings.dt = 1 / 120
        scene.settings.iterations = 1
        let a = scene.addBody(size: F3(repeating: 0.1), density: 0,
                              friction: 0, position: .zero, collisionEnabled: false)
        let b = scene.addBody(size: F3(repeating: 0.1), density: 1000,
                              friction: 0, position: .zero, collisionEnabled: false)
        var joint = SceneJoint(bodyA: a, bodyB: b, rA: .zero, rB: .zero,
                               stiffnessLin: .infinity, stiffnessAng: .infinity)
        joint.prismaticAxis = F3(1, 0, 0)
        joint.translationLimits = -0.01...0.01
        scene.addJoint(joint)
        return scene
    }

    func testCPUOppositeStopsDiscardAxialForceBeforePrimalAndDual() throws {
        let cpu = try rail().makeCPUSolverChecked()
        let joint = try XCTUnwrap(cpu.forces.compactMap { $0 as? CPUJoint }.first)
        let body = cpu.bodies[1]
        func force() -> F3 {
            var lin = Mat3Rows(), ang = Mat3Rows(), cross = Mat3Rows()
            var rhs = F3.zero, torque = F3.zero
            joint.updatePrimal(body, 0, &lin, &ang, &cross, &rhs, &torque)
            return rhs
        }
        for oldStop: Float in [-1, 1] {
            body.positionLin = F3(oldStop * 0.02, 0, 0)
            joint.lambdaLin = .zero
            _ = joint.initialize()
            joint.penaltyLin = F3(repeating: 1000)
            let carried = F3(-oldStop * 50, 3, -4)
            joint.lambdaLin = carried
            let warm = cpu.alpha * cpu.gamma
            _ = joint.initialize()
            XCTAssertEqual(joint.lambdaLin.x, carried.x * warm, accuracy: 1e-5,
                           "keep a multiplier that still belongs to the same stop")

            // Jump directly from one violated bound to the other, with no
            // intervening evaluation inside the free interval.
            body.positionLin.x = -oldStop * 0.02
            joint.lambdaLin = F3(0, 3, -4)
            let expectedForce = force()
            joint.lambdaLin = carried
            XCTAssertLessThan(length(force() - expectedForce), 1e-5,
                              "the first primal solve must not apply the opposite stop's force")
            let expectedDual = joint.penaltyLin * joint.currentCLin() + F3(0, 3, -4)
            joint.updateDual(0)
            XCTAssertLessThan(length(joint.lambdaLin - expectedDual), 1e-5,
                              "dual updates must rebuild the new axial multiplier from zero")

            body.positionLin.x = oldStop * 0.02
            joint.lambdaLin = -carried
            _ = joint.initialize()
            XCTAssertEqual(joint.lambdaLin.x, 0, accuracy: 1e-5,
                           "also discard stale axial force when the switch occurs between steps")
            XCTAssertEqual(joint.lambdaLin.y, -3 * warm, accuracy: 1e-5)
            XCTAssertEqual(joint.lambdaLin.z, 4 * warm, accuracy: 1e-5)
        }
    }

    func testGPUOppositeStopsMatchACleanAxialWarmStart() throws {
        func step(previous: Float, current: Float, axial: Float, predictedCrossing: Bool = false) throws -> F3 {
            var scene = rail()
            scene.bodies[1].position.x = (predictedCrossing ? previous : current) * 0.02
            if predictedCrossing {
                scene.bodies[1].velocity.x = (current - previous) * 0.02 / scene.settings.dt
            }
            let gpu = try GPUSolver(scene: scene)
            let joint = gpu.joints.contents().bindMemory(to: JointGPU.self, capacity: 1)
            // This models the persisted state after a loaded stop, followed
            // by a narrow-rail crossing before the next constraint evaluation.
            joint.pointee.translationLimits.w = previous
            joint.pointee.lambdaLin = SIMD4(axial, 3, -4, 0)
            joint.pointee.penaltyLin = SIMD4(1000, 1000, 1000, 0)
            try gpu.submitStep()
            try gpu.synchronize()
            return gpu.bodyPosition(1)
        }
        for previous: Float in [-1, 1] {
            let stale = -previous * 50
            let crossed = try step(previous: previous, current: -previous, axial: stale)
            let clean = try step(previous: previous, current: -previous, axial: 0)
            XCTAssertLessThan(length(crossed - clean), 1e-6,
                              "a stop switch must retain only the transverse warm start")
            let predicted = try step(previous: previous, current: -previous, axial: stale, predictedCrossing: true)
            let predictedClean = try step(previous: previous, current: -previous, axial: 0, predictedCrossing: true)
            XCTAssertLessThan(length(predicted - predictedClean), 1e-6,
                              "also reject stale force when prediction crosses after joint initialization")
            let sameStop = try step(previous: previous, current: previous, axial: stale)
            let noWarmStart = try step(previous: previous, current: previous, axial: 0)
            XCTAssertGreaterThan(length(sameStop - noWarmStart), 1e-5,
                                 "do not disable useful warm starting at an unchanged stop")
        }
    }
}
