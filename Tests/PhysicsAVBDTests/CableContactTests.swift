import XCTest
import simd
import SimCore
@testable import PhysicsAVBD

final class CableContactTests: XCTestCase {
    func testMaterialSpinCannotEraseNormalContactForce() throws {
        for shape in [BodyShape.capsule, .box, .sphere, .torus] {
            for cableFirst in [false, true] {
                var scene = PhysicsScene(name: "spinning material contact")
                scene.settings.gravity = 0; scene.settings.collisionMargin = 0.001
                let material = CableMaterial(stretchRigidity: 1000, shearRigidity: 1000,
                                             bendRigidity: 0.1, twistRigidity: 0.1)
                func addCable() throws -> Int {
                    try scene.addCable(points: [F3(-0.1,0,0.059), F3(0.1,0,0.059)],
                        radius: 0.03, density: 1000, material: material, friction: 0).bodyIDs[0]
                }
                func addSupport() throws {
                    if shape == .capsule {
                        try scene.addCable(points: [F3(-0.1,0,0), F3(0.1,0,0)],
                            radius: 0.03, density: 1000, material: material,
                            friction: 0, fixedSegments: [0])
                    } else {
                        let size: F3 = shape == .box ? F3(0.6,0.6,0.2)
                            : (shape == .sphere ? F3(repeating: 0.06) : F3(0.08,0.03,0))
                        _ = scene.addBody(size: size, density: 0, friction: 0,
                            position: shape == .box ? F3(0,0,-0.07) : .zero, shape: shape)
                    }
                }
                let cable: Int
                if cableFirst { cable = try addCable(); try addSupport() }
                else { try addSupport(); cable = try addCable() }
                let solver = try scene.makeCPUSolverChecked()
                try solver.stepChecked()
                let contact = try XCTUnwrap(solver.forces.compactMap { $0 as? CPUManifold }.first,
                                            "shape=\(shape), cableFirst=\(cableFirst)")
                for body in solver.bodies {
                    body.initialLin = body.positionLin; body.initialAng = body.positionAng
                }
                for i in contact.contacts.indices {
                    contact.contacts[i].C0 = F3(-0.001,0,0)
                    contact.contacts[i].penalty = F3(1000,0,0)
                    contact.contacts[i].lambda = .zero
                }
                let body = solver.bodies[cable]
                func normalForce() -> F3 {
                    var l = Mat3Rows(), r = Mat3Rows(), c = Mat3Rows(), f = F3.zero, t = F3.zero
                    contact.updatePrimal(body, 0, &l, &r, &c, &f, &t)
                    return f
                }
                let reference = normalForce()
                XCTAssertGreaterThan(length(reference), 0.9)
                body.positionAng = Quat(angle: 0.8, axis: body.initialAng.act(F3(0,0,1))) * body.initialAng
                XCTAssertLessThan(length(normalForce() - reference), 0.001,
                                  "axial spin must not change normal contact: \(shape)")
            }
        }
    }

    func testShortCrossedCapsulesUseInteriorWitnessesAtEveryScale() {
        for length: Float in [0.2, 0.02, 0.002] {
            for angle: Float in [.pi / 2, .pi / 6, 0.01] {
                let radius = length * 0.03
                let a = CPURigid(index: 0, size: F3(length, radius, 0), density: 1000,
                    friction: 0.5, position: .zero,
                    rotation: Quat(from: F3(0,0,1), to: F3(1,0,0)), shape: .capsule)
                let b = CPURigid(index: 1, size: F3(length, radius, 0), density: 1000,
                    friction: 0.5, position: F3(0,0,1.5 * radius),
                    rotation: Quat(from: F3(0,0,1), to: F3(cos(angle),sin(angle),0)), shape: .capsule)
                var contacts: [ContactPoint] = []
                var basis = (F3.zero, F3.zero, F3.zero)
                XCTAssertEqual(CPUManifold.collideCapsuleCapsule(a,b,margin: 0, &contacts,&basis), 1,
                               "length=\(length), angle=\(angle)")
                guard let contact = contacts.first else { continue }
                let gap = dot(basis.0, a.positionLin + contact.rA - b.positionLin - contact.rB)
                XCTAssertEqual(gap, -0.5 * radius, accuracy: radius * 0.002)
                XCTAssertEqual(abs(basis.0.z), 1, accuracy: 0.002)
            }
        }
    }
}
