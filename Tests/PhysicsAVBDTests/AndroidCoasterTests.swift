import XCTest
import simd
@testable import SimCore
@testable import PhysicsAVBD
@testable import GPUSimDemos

final class AndroidCoasterTests: XCTestCase {
    /// A ball set at the top must ride every stadium lap around the statue
    /// down to the catch pen — without harming the statue on the way.
    func testBallRidesTheWholeRoute() throws {
        let scene = Demos.androidCoaster(scale: 1)
        var ball = -1
        var statue: [Int] = []
        for (i, b) in scene.bodies.enumerated() {
            if b.shape == .sphere && ball < 0 { ball = i }
            if b.shape == .box && b.density > 0 { statue.append(i) }
        }
        let s0 = statue.map { scene.bodies[$0].position }
        var penWalls: [F3] = []
        for b in scene.bodies where b.density == 0 && b.shape == .box
            && b.size.z > 1.0 && b.size.z < 1.2 { penWalls.append(b.position) }
        let penC = penWalls.reduce(F3.zero, +) / Float(penWalls.count)
        let gpu = try GPUSolver(scene: scene)
        var reachedPen = false
        for frame in 0..<9000 {
            gpu.step()
            if frame % 60 == 59 {
                let p = gpu.bodyPosition(ball)
                reachedPen = p.z < 1
                    && distance(F3(p.x, p.y, 0), F3(penC.x, penC.y, 0)) < 5
                if reachedPen { break }
            }
        }
        XCTAssertTrue(
            reachedPen,
            "ball should complete the route and enter the catch pen")
        // The pen must retain the ball, not merely register a fly-through.
        for _ in 0..<300 { gpu.step() }
        let p = gpu.bodyPosition(ball)
        XCTAssertLessThan(distance(F3(p.x, p.y, 0), F3(penC.x, penC.y, 0)), 5.0,
                          "ball should finish near the catch pen")
        var fallen = 0
        for (k, bi) in statue.enumerated()
            where gpu.bodyPosition(bi).z < s0[k].z * 0.5 { fallen += 1 }
        XCTAssertEqual(fallen, 0, "the statue must survive the ride")
    }

    /// The statue is real masonry: a heavy ball dropped on it breaks bricks.
    func testStatueIsDestructible() throws {
        var scene = Demos.android(scale: 1)
        _ = scene.addSphere(diameter: 1.6, density: 8, friction: 0.5,
                            position: F3(0, 0, 14))
        var statue: [Int] = []
        for (i, b) in scene.bodies.enumerated()
            where b.shape == .box && b.density > 0 { statue.append(i) }
        let s0 = statue.map { scene.bodies[$0].position }
        let gpu = try GPUSolver(scene: scene)
        for _ in 0..<400 { gpu.step() }
        var displaced = 0
        for (k, bi) in statue.enumerated()
            where distance(gpu.bodyPosition(bi), s0[k]) > 0.5 { displaced += 1 }
        XCTAssertGreaterThan(displaced, 10,
                             "a wrecking hit must break bricks loose (got \(displaced))")
    }
}
