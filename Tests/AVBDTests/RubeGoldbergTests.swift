import XCTest
import simd
@testable import AVBDCore

final class RubeGoldbergTests: XCTestCase {
    /// The whole cascade must fire: marble -> rail -> dominoes -> ball off
    /// the table -> ramp -> all monoliths felled.
    func testCascadeCompletes() throws {
        let scene = Demos.rubegoldberg()
        var monoliths: [Int] = []
        for (i, b) in scene.bodies.enumerated()
            where b.shape == .box && abs(b.size.x - 0.16) < 0.01 && b.position.x > 18 {
            monoliths.append(i)
        }
        XCTAssertEqual(monoliths.count, 4)
        let p0 = monoliths.map { scene.bodies[$0].position }
        let gpu = try GPUSolver(scene: scene)
        for _ in 0..<1200 { gpu.step() }
        var felled = 0
        for (k, m) in monoliths.enumerated()
            where distance(gpu.bodyPosition(m), p0[k]) > 0.4 { felled += 1 }
        XCTAssertGreaterThanOrEqual(felled, 3, "cascade should fell the monoliths (got \(felled)/4)")
    }
}
