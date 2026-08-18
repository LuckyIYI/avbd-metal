import XCTest
import simd
@testable import SimCore
@testable import PhysicsAVBD

final class SphereTests: XCTestCase {
    func testSphereRestsOnGround() throws {
        var scene = PhysicsScene(name: "s")
        _ = scene.addBody(size: F3(100, 100, 2), density: 0, friction: 0.5, position: F3(0, 0, -1))
        _ = scene.addSphere(diameter: 1, density: 1, friction: 0.5, position: F3(0, 0, 2))
        let solver = try GPUSolver(scene: scene)
        for _ in 0..<240 { solver.step() }
        XCTAssertEqual(solver.bodyPosition(1).z, 0.5, accuracy: 0.04)
        XCTAssertLessThan(length(solver.bodyVelocity(1)), 0.05)
    }

    /// Rolling without slip: v = omega * r on an incline; acceleration must
    /// be near the analytic 5/7 g sin(theta) of a solid sphere.
    func testSphereRollsWithoutSlip() throws {
        var scene = PhysicsScene(name: "r")
        let theta: Float = 0.25
        _ = scene.addBody(size: F3(40, 10, 1), density: 0, friction: 0.6,
                          position: F3(0, 0, 0), rotation: Quat(angle: theta, axis: F3(0, 1, 0)))
        _ = scene.addSphere(diameter: 1, density: 1, friction: 0.6, position: F3(0, 0, 1.2))
        let solver = try GPUSolver(scene: scene)
        for _ in 0..<20 { solver.step() }
        let v0 = length(solver.bodyVelocity(1))
        let frames = 60
        for _ in 0..<frames { solver.step() }
        let v1 = length(solver.bodyVelocity(1))
        let w1 = length(solver.bodyAngularVelocity(1))
        XCTAssertEqual(v1 / max(w1, 0.01), 0.5, accuracy: 0.05, "rolling: v = w r")
        let accel = (v1 - v0) / (Float(frames) * solver.settings.dt)
        let analytic = 5.0 / 7.0 * 10 * sin(theta)
        XCTAssertEqual(accel, analytic, accuracy: analytic * 0.2,
                       "rolling accel \(accel) vs analytic \(analytic)")
    }

    /// A sliding (fast-spinning) sphere must not sink through the floor —
    /// regression for the body-fixed anchor swing bug.
    func testSlidingSphereStaysOnGround() throws {
        var scene = PhysicsScene(name: "sl")
        _ = scene.addBody(size: F3(200, 200, 2), density: 0, friction: 0.3, position: F3(0, 0, -1))
        _ = scene.addSphere(diameter: 0.7, density: 1, friction: 0.3,
                            position: F3(0, 0, 1), velocity: F3(8, 0, 0))
        let solver = try GPUSolver(scene: scene)
        for _ in 0..<300 { solver.step() }
        XCTAssertEqual(solver.bodyPosition(1).z, 0.35, accuracy: 0.05)
    }

    func testSphereSphereCollision() throws {
        var scene = PhysicsScene(name: "ss")
        _ = scene.addBody(size: F3(100, 100, 2), density: 0, friction: 0.5, position: F3(0, 0, -1))
        _ = scene.addSphere(diameter: 1, density: 1, friction: 0.5, position: F3(0, 0, 0.5))
        _ = scene.addSphere(diameter: 1, density: 1, friction: 0.5, position: F3(0.1, 0, 1.6))
        let solver = try GPUSolver(scene: scene)
        for _ in 0..<300 { solver.step() }
        // top ball rolls off, both end on ground, not interpenetrating
        let d = distance(solver.bodyPosition(1), solver.bodyPosition(2))
        XCTAssertGreaterThan(d, 0.93, "spheres must not interpenetrate (d=\(d))")
        XCTAssertEqual(solver.bodyPosition(1).z, 0.5, accuracy: 0.05)
        XCTAssertEqual(solver.bodyPosition(2).z, 0.5, accuracy: 0.05)
    }

    /// Slide: balls steadily descend the chute, stay inside it, and none
    /// fall through the world.
    func testSlideBallsDescend() throws {
        let scene = Demos.swirl(turns: 3, balls: 10)
        let solver = try GPUSolver(scene: scene)
        let ballStart = scene.bodies.count - 10
        let z0 = (0..<10).map { solver.bodyPosition(ballStart + $0).z }.reduce(0, +) / 10
        for _ in 0..<900 { solver.step() }
        var sum: Float = 0
        for k in 0..<10 {
            let z = solver.bodyPosition(ballStart + k).z
            XCTAssertGreaterThan(z, -1, "ball \(k) fell through the world")
            sum += z
        }
        XCTAssertLessThan(sum / 10, z0 - 3, "balls should descend the chute (avg \(sum / 10) from \(z0))")
    }
}
