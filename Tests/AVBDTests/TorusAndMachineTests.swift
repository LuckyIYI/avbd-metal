import XCTest
import simd
@testable import AVBDCore

final class TorusAndMachineTests: XCTestCase {
    func testTorusRestsFlat() throws {
        var scene = PhysicsScene(name: "t")
        _ = scene.addBody(size: F3(100, 100, 2), density: 0, friction: 0.5, position: F3(0, 0, -1))
        _ = scene.addTorus(major: 0.45, minor: 0.12, density: 1, friction: 0.5, position: F3(0, 0, 2))
        let solver = try GPUSolver(scene: scene)
        for _ in 0..<400 { solver.step() }
        XCTAssertEqual(solver.bodyPosition(1).z, 0.12, accuracy: 0.1)  // slow alpha recovery from impact
        XCTAssertGreaterThan(abs(solver.bodyRotation(1).act(F3(0, 0, 1)).z), 0.95, "should lie flat")
    }

    func testInterlockedToriHoldWeight() throws {
        var scene = PhysicsScene(name: "link")
        let t1 = scene.addTorus(major: 0.45, minor: 0.12, density: 1, friction: 0.4,
                                position: F3(0, 0, 6), rotation: Quat(angle: .pi/2, axis: F3(1,0,0)))
        let t2 = scene.addTorus(major: 0.45, minor: 0.12, density: 1, friction: 0.4,
                                position: F3(0, 0, 5.34),
                                rotation: (Quat(angle: .pi/2, axis: F3(0,0,1)) * Quat(angle: .pi/2, axis: F3(1,0,0))).normalized)
        scene.addJoint(SceneJoint(bodyA: -1, bodyB: t1, rA: F3(0, 0, 6.4), rB: F3(0.45, 0, 0)))
        let w = scene.addBody(size: F3(0.5, 0.5, 0.5), density: 8, friction: 0.5, position: F3(0, 0, 4.4))
        scene.addJoint(SceneJoint(bodyA: t2, bodyB: w, rA: F3(0.45, 0, 0), rB: F3(0, 0, 0.45)))
        let solver = try GPUSolver(scene: scene)
        for _ in 0..<420 { solver.step() }
        XCTAssertGreaterThan(solver.bodyPosition(w).z, 3.0, "linked tori must carry 1kg")
    }

    func testSphereCaughtInRing() throws {
        var scene = PhysicsScene(name: "h")
        _ = scene.addBody(size: F3(100, 100, 2), density: 0, friction: 0.5, position: F3(0, 0, -1))
        _ = scene.addTorus(major: 0.6, minor: 0.15, density: 0, friction: 0.5,
                           position: F3(0, 0, 1.5), rotation: Quat(angle: .pi/2, axis: F3(1,0,0)))
        _ = scene.addSphere(diameter: 0.7, density: 1, friction: 0.5, position: F3(0, 0, 1.52))
        let solver = try GPUSolver(scene: scene)
        for _ in 0..<240 { solver.step() }
        XCTAssertGreaterThan(solver.bodyPosition(2).z, 0.8, "ball should rest inside the ring")
    }

    /// Mail hangs purely mechanically: rings rest on post pegs, no world
    /// constraints on the cloth at all.
    func testChainmailHangsOnPegs() throws {
        let scene = Demos.chainmail(rings: 5, drops: 3)
        var ringIdx: [Int] = []
        for (i, b) in scene.bodies.enumerated() where b.shape == .torus { ringIdx.append(i) }
        let solver = try GPUSolver(scene: scene)
        for _ in 0..<700 { solver.step() }
        for i in ringIdx {
            XCTAssertGreaterThan(solver.bodyPosition(i).z, 4.5, "ring \(i) fell out of the sheet")
        }
        var caught = 0
        for i in (scene.bodies.count - 3)..<scene.bodies.count where solver.bodyPosition(i).z > 4 { caught += 1 }
        XCTAssertGreaterThanOrEqual(caught, 2, "mail should catch the dropped objects")
    }

    /// Kinematic spinner friction conveys (spin-aware contact targets).
    func testSpinnerFrictionConveys() throws {
        var scene = PhysicsScene(name: "roll")
        let roller = scene.addBody(size: F3(2, 6, 2), density: 0, friction: 1.0,
                                   position: F3(0, 0, 1))
        scene.addSpinner(SceneSpinner(body: roller, axis: F3(0, 1, 0), omega: 0.5))
        _ = scene.addBody(size: F3(0.6, 0.6, 0.6), density: 1, friction: 1.0,
                          position: F3(0, 0, 2.4))
        let solver = try GPUSolver(scene: scene)
        for _ in 0..<60 { solver.step() }
        XCTAssertGreaterThan(solver.bodyVelocity(1).x, 0.3, "box must ride the roller")
    }

    /// Roller conveyor: motorized spinning rollers carry boxes downstream.
    func testRollerConveyor() throws {
        let scene = Demos.treadmill(boxes: 5)
        let solver = try GPUSolver(scene: scene)
        let boxStart = scene.bodies.count - 5
        var x0: Float = 0
        for f in 0..<360 {
            solver.step()
            if f == 89 { x0 = (0..<5).map { solver.bodyPosition(boxStart + $0).x }.reduce(0,+) / 5 }
        }
        let x1 = (0..<5).map { solver.bodyPosition(boxStart + $0).x }.reduce(0,+) / 5
        XCTAssertGreaterThan(x1 - x0, 1.0, "rollers must convey boxes (+\(x1 - x0))")
    }
}
