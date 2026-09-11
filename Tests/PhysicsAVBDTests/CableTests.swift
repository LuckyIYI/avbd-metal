import XCTest
import simd
@testable import SimCore
@testable import PhysicsAVBD
import GPUSimDemos

final class CableTests: XCTestCase {
    private let material = CableMaterial(stretchRigidity: 2000, shearRigidity: 800,
        bendRigidity: 0.3, twistRigidity: 0.1, dampingTime: 0)

    func testCapsuleBoxContactNormalInBothOrders() {
        let cap = CPURigid(index: 0, size: F3(0.2, 0.03, 0), density: 1000,
            friction: 0.5, position: F3(0, 0, 0.025),
            rotation: Quat(angle: .pi / 2, axis: F3(0, 1, 0)), shape: .capsule)
        let floor = CPURigid(index: 1, size: F3(2, 2, 0.2), density: 0,
                             friction: 0.5, position: F3(0, 0, -0.1))
        for isA in [true, false] {
            var contacts: [ContactPoint] = []
            var basis = (F3.zero, F3.zero, F3.zero)
            let count = CPUManifold.collideCapsuleBox(cap, floor, capsuleIsA: isA,
                margin: 0, &contacts, &basis)
            XCTAssertGreaterThan(count, 0)
            XCTAssertEqual(basis.0.z, isA ? 1 : -1, accuracy: 1e-6)
            for contact in contacts {
                let capOffset = isA ? contact.rA : contact.rB
                let boxOffset = isA ? contact.rB : contact.rA
                let gap = cap.positionLin + capOffset
                    - transform(floor.positionLin, floor.positionAng, boxOffset)
                XCTAssertEqual(gap.z, -0.005, accuracy: 1e-6)
            }
        }
    }

    func testFullForceGradientMatchesEnergyOnBothBodies() throws {
        var scene = PhysicsScene(name: "cable derivative")
        scene.settings.gravity = 0
        try scene.addCable(points: [.zero, F3(0.1, 0, 0.3), F3(0.1, 0.2, 0.5)],
            radius: 0.03, density: 1000, material: material, collisionEnabled: false)
        let solver = try scene.makeCPUSolverChecked()
        let force = try XCTUnwrap(solver.forces.first as? CPUCable)
        let a = solver.bodies[0], b = solver.bodies[1]
        b.positionLin += F3(0.007, -0.011, 0.013)
        b.positionAng = Quat(angle: 0.6, axis: normalize(F3(1, -2, 3))) * b.positionAng
        func energy() -> Float {
            let c = a.positionAng.inverse.act(
                transform(b.positionLin, b.positionAng, force.rB) - a.positionLin) - force.rA
            let e = cableRotationLog((a.positionAng * force.restRel).inverse * b.positionAng).value
            return 0.5 * (dot(c, force.material.linearStiffness * c)
                          + dot(e, force.material.angularStiffness * e))
        }
        for body in [a, b] {
            var l = Mat3Rows(), r = Mat3Rows(), c = Mat3Rows(), f = F3.zero, t = F3.zero
            force.updatePrimal(body, 0, &l, &r, &c, &f, &t)
            for axis in 0..<3 {
                var unit = F3.zero; unit[axis] = 1
                let epsilon: Float = 0.00005
                let p = body.positionLin, q = body.positionAng
                body.positionLin = p + unit * epsilon; let ePlus = energy()
                body.positionLin = p - unit * epsilon; let eMinus = energy()
                body.positionLin = p
                XCTAssertEqual(f[axis], (ePlus - eMinus) / (2 * epsilon),
                    accuracy: max(0.02, abs(f[axis]) * 0.003))
                body.positionAng = Quat(angle: epsilon, axis: unit) * q; let aPlus = energy()
                body.positionAng = Quat(angle: -epsilon, axis: unit) * q; let aMinus = energy()
                body.positionAng = q
                XCTAssertEqual(t[axis], (aPlus - aMinus) / (2 * epsilon),
                    accuracy: max(0.02, abs(t[axis]) * 0.003))
            }
            // The complete GN block must be symmetric positive semidefinite.
            for v in [F3(1, 2, 3), F3(-2, 0.3, 1), F3(0, 0, 1)] {
                let w = cross(v, F3(0.2, -0.1, 0.4))
                XCTAssertGreaterThanOrEqual(dot(v, l.mul(v)) + 2 * dot(w, c.mul(v))
                    + dot(w, r.mul(w)), -0.001)
            }
        }
    }

    func testCPUAndMetalSmallDeformationParityAndReset() throws {
        var scene = PhysicsScene(name: "cable parity")
        scene.settings.dt = 1 / 240; scene.settings.iterations = 32
        scene.settings.gravity = -9.81
        try scene.addCable(points: [F3(0, 0, 1), F3(0.2, 0, 1), F3(0.4, 0, 1)],
            radius: 0.03, density: 1000, material: material, fixedSegments: [0],
            collisionEnabled: false)
        let cpu = try scene.makeCPUSolverChecked(), gpu = try GPUSolver(scene: scene)
        for _ in 0..<30 { try cpu.stepChecked(); try gpu.submitStep() }
        try gpu.synchronize()
        XCTAssertLessThan(length(cpu.bodies[1].positionLin - gpu.bodyPosition(1)), 0.003)
        XCTAssertLessThan(length(quatSub(cpu.bodies[1].positionAng, gpu.bodyRotation(1))), 0.01)
        gpu.setBodyStates(scene.bodies.enumerated().map { index, body in
            .init(body: index, position: body.position, rotation: body.rotation,
                  linearVelocity: .zero, angularVelocity: .zero)
        })
        let fresh = try GPUSolver(scene: scene)
        for _ in 0..<10 { try fresh.submitStep(); try gpu.submitStep() }
        try fresh.synchronize(); try gpu.synchronize()
        XCTAssertLessThan(length(fresh.bodyPosition(1) - gpu.bodyPosition(1)), 0.0001)
    }

    func testDemoMetadataAndJointABI() {
        let scene = Demos.cables(count: 3, segments: 12)
        XCTAssertEqual(scene.cables.count, 3)
        XCTAssertEqual(scene.cables[0].jointIDs.count, 11)
        XCTAssertEqual(JointGPU.cableFlag, 128)
        XCTAssertEqual(JointGPU.cableFlag & (4 | 64), 0, "break-load flags must not select cables")
        // Cable storage remains in the existing prefix; other joint features
        // (e.g. PR #36's responses) may append fields to the overall record.
        XCTAssertEqual(MemoryLayout<JointGPU>.offset(of: \.motor), 208)
        XCTAssertEqual(MemoryLayout<JointGPU>.offset(of: \.limits), 224)
        XCTAssertGreaterThanOrEqual(MemoryLayout<JointGPU>.stride, 256)
        XCTAssertEqual(MemoryLayout<JointGPU>.stride % 16, 0)
    }
}
