import XCTest
import Metal
import SimCore
import simd
@testable import PhysicsAVBD

final class ScanCapacityTests: XCTestCase {
    func testCompoundPairScanFitsScratchAndPreservesExclusiveOffsets() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal required") }
        var scene = PhysicsScene(name: "pair-scan-exceeds-grid")
        scene.settings.gravity = 0
        scene.settings.iterations = 0
        for index in 0..<160 {
            let body = scene.addBody(size: F3(repeating: 0.2), density: 1,
                friction: 0.5, position: F3(Float(index % 20) * 0.25,
                    Float(index / 20) * 0.25, 1), collisionEnabled: false)
            for offset: Float in [-0.04, 0.04] {
                scene.addCollider(body: body, size: F3(repeating: 0.25),
                    localPosition: F3(offset, 0, 0), shape: .sphere)
            }
        }
        let solver = try GPUSolver(scene: scene)
        XCTAssertTrue(solver.usesRigidColliderHierarchy)
        let blocks = (solver.hierarchyPairCapacity + 1023) / 1024
        XCTAssertGreaterThan(blocks, (solver.gridHashSize + 1023) / 1024 + 1,
            "fixture must exceed the old grid-only allocation")
        XCTAssertGreaterThanOrEqual(solver.scanBlockSums.length / 4, blocks)
        for profiled in [false, true] {
            solver.profiling = profiled
            try solver.submitStep(); try solver.synchronize()
            let counts = solver.pairCount.contents().assumingMemoryBound(to: UInt32.self)
            let starts = solver.pairStart.contents().assumingMemoryBound(to: UInt32.self)
            var expected: UInt32 = 0
            for index in 0..<solver.hierarchyPairCapacity {
                XCTAssertEqual(starts[index], expected, "offset at \(index)")
                expected += counts[index]
            }
            XCTAssertEqual(Int(expected), solver.lastNumPairs)
            let pairs = Array(UnsafeBufferPointer(start:
                solver.pairs.contents().assumingMemoryBound(to: SIMD2<UInt32>.self),
                count: solver.lastNumPairs))
            XCTAssertFalse(pairs.isEmpty)
            XCTAssertEqual(Set(pairs).count, pairs.count)
            XCTAssertTrue(pairs.allSatisfy { $0.x < $0.y })
        }
    }
}
