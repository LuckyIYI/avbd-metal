import XCTest
import simd
@testable import AVBDCore

final class WreckingBallTests: XCTestCase {
    func testBallSwingsAndDemolishes() throws {
        let scene = Demos.wreckingball(floors: 3)
        var ball = -1, topSlab = -1
        var bestZ: Float = -1
        for (i, b) in scene.bodies.enumerated() {
            if b.shape == .sphere && b.size.x > 1.5 { ball = i }
            if b.density > 0 && b.shape == .box && b.size.x > 10 && b.position.z > bestZ {
                bestZ = b.position.z; topSlab = i
            }
        }
        let slabP0 = scene.bodies[topSlab].position
        let gpu = try GPUSolver(scene: scene)
        var reached = false
        for _ in 0..<600 {
            gpu.step()
            if gpu.bodyPosition(ball).x < 7.5 { reached = true }
        }
        XCTAssertTrue(reached, "ball should swing into the building")
        XCTAssertGreaterThan(distance(gpu.bodyPosition(topSlab), slabP0), 1.0,
                             "impact should visibly damage the structure")
    }
}
