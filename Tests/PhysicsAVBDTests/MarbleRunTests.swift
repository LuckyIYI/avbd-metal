import XCTest
import simd
@testable import SimCore
@testable import PhysicsAVBD

final class MarbleRunTests: XCTestCase {
    /// The marble rollercoaster: a train of marbles descends the full
    /// course — helix, windmill, zigzag, hinged seesaw, final spiral — and
    /// collects in the pool. Tolerate one or two escape artists.
    func testMarblesReachThePool() throws {
        let scene = Demos.marblerun(marbles: 10)
        // pool floor = the large flat static box near the course end
        var poolC = F3.zero
        for b in scene.bodies where b.density == 0 && b.shape == .box
            && abs(b.size.x - 4.5) < 0.01 && abs(b.size.y - 4.5) < 0.01 {
            poolC = b.position
        }
        let solver = try GPUSolver(scene: scene)
        let mStart = scene.bodies.count - 10
        for _ in 0..<5400 { solver.step() }
        var inPool = 0
        for k in 0..<10 {
            let p = solver.bodyPosition(mStart + k)
            if abs(p.x - poolC.x) < 2.5 && abs(p.y - poolC.y) < 2.5
                && p.z > poolC.z && p.z < poolC.z + 1.5 {
                inPool += 1
            }
        }
        XCTAssertGreaterThanOrEqual(inPool, 4, "a solid majority of marbles should finish (got \(inPool)/10)")
    }
}
