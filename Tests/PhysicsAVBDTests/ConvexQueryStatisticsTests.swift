import XCTest
import Metal
import SimCore
import simd
@testable import PhysicsAVBD
@testable import GPUSimDemos

/// Convex query recovery statistics: every support-mapped pair resolves at
/// exactly one stage, the complete separating-axis search prunes edge pairs
/// with the Gauss-map test, and the pruned search still certifies the same
/// captured contacts as the unpruned one did.
final class ConvexQueryStatisticsTests: XCTestCase {
    private func requireMetal() throws {
        if MTLCreateSystemDefaultDevice() == nil { throw XCTSkip("Metal is unavailable") }
    }

    private func total(_ s: ConvexQueryStatistics) -> Int {
        s.mprAccepted + s.gjkSeparated + s.recovered + s.failures
    }

    func testRestingHullPairResolvesExactlyOnceAndReportsThroughStatistics() throws {
        try requireMetal()
        let source = Demos.convexDecomposition(scale: 1)
        let assetID = try XCTUnwrap(source.colliders[7].convexAssetID)
        let asset = source.convexAssets[assetID]
        var scene = PhysicsScene(name: "hull-on-box-statistics")
        scene.settings.gravity = -9.81
        _ = scene.addBody(size: F3(4, 4, 0.2), density: 0, friction: 0.6, position: F3(0, 0, -0.1))
        let hull = scene.addBody(size: F3(repeating: 1), density: 1, friction: 0.6,
                                 position: F3(0, 0, 0.35), collisionEnabled: false)
        _ = scene.addConvexCollider(body: hull, asset: asset, friction: 0.6)
        let solver = try GPUSolver(scene: scene)
        // Step until the falling hull rests on the floor box.
        var steps = 0
        repeat {
            try solver.submitStep(); try solver.synchronize(); steps += 1
        } while solver.activeRigidContactPairs().isEmpty && steps < 600
        XCTAssertNil(solver.runtimeFailure)
        XCTAssertFalse(solver.activeRigidContactPairs().isEmpty, "hull never reached the floor")
        let stats = solver.lastConvexQueryStatistics
        XCTAssertEqual(total(stats), solver.lastNumPairs,
                       "every hull-involving candidate pair resolves at exactly one stage: \(stats)")
        XCTAssertEqual(stats.failures, 0)
        XCTAssertGreaterThan(stats.mprAccepted + stats.recovered, 0, "the resting hull is in contact")
        let dictionary = solver.rigidContactStatistics()
        XCTAssertEqual(dictionary["convex_mpr_accepted"], stats.mprAccepted)
        XCTAssertEqual(dictionary["convex_sat_edge_pairs_pruned"], stats.satEdgePairsPruned)
        XCTAssertEqual(dictionary["convex_query_failures"], 0)
    }

    func testStatisticsSurviveSpeculationSnapshotRoundTrip() throws {
        try requireMetal()
        let source = Demos.convexDecomposition(scale: 1)
        let assetID = try XCTUnwrap(source.colliders[7].convexAssetID)
        var scene = PhysicsScene(name: "hull-statistics-snapshot")
        scene.settings.gravity = -9.81
        _ = scene.addBody(size: F3(4, 4, 0.2), density: 0, friction: 0.6, position: F3(0, 0, -0.1))
        let hull = scene.addBody(size: F3(repeating: 1), density: 1, friction: 0.6,
                                 position: F3(0, 0, 0.35), collisionEnabled: false)
        _ = scene.addConvexCollider(body: hull, asset: source.convexAssets[assetID], friction: 0.6)
        let solver = try GPUSolver(scene: scene)
        for _ in 0..<10 { try solver.submitStep(); try solver.synchronize() }
        let before = solver.lastConvexQueryStatistics
        let snapshot = solver.captureRigidSpeculationSnapshot()
        for _ in 0..<5 { try solver.submitStep(); try solver.synchronize() }
        solver.restoreRigidSpeculationSnapshot(snapshot)
        XCTAssertEqual(solver.lastConvexQueryStatistics, before)
    }

    func testCapturedPairsStillCertifyContactsWithPrunedSeparatingAxes() throws {
        try requireMetal()
        let fixtures = try JSONSerialization.jsonObject(with: Data(capturedFloorPairsJSON.utf8)) as! [[[String: Any]]]
        var searched = 0, pruned = 0, tested = 0
        for (index, pair) in fixtures.enumerated() {
            let vertexCounts = pair.map { ($0["vertices"] as! [[NSNumber]]).count }
            if vertexCounts.contains(where: { $0 > ConvexAssetLimits.maximumVerticesPerHull }) { continue }
            func f(_ s: [String: Any], _ key: String) -> [Float] { (s[key] as! [NSNumber]).map { $0.floatValue } }
            func xyz(_ a: [Float]) -> F3 { F3(a[0], a[1], a[2]) }
            let a = pair[0], b = pair[1]
            let q = f(a, "rotation"), qb = f(b, "rotation")
            var scene = PhysicsScene(name: "captured-statistics-\(index)")
            scene.settings.gravity = 0
            scene.settings.iterations = 0
            let isHull = f(b, "center_kind")[3] == 4
            let floor = scene.addBody(size: xyz(f(b, "dimensions")), density: 0, friction: 0.5,
                position: xyz(f(b, "center_kind")), rotation: Quat(vector: SIMD4(qb[0], qb[1], qb[2], qb[3])),
                collisionEnabled: !isHull)
            if isHull {
                let points = (b["vertices"] as! [[NSNumber]]).map { xyz($0.map { $0.floatValue }) }
                _ = scene.addConvexCollider(body: floor, vertices: points)
            }
            let owner = scene.addBody(size: F3(repeating: 1), density: 1, friction: 0.5,
                position: xyz(f(a, "center_kind")), rotation: Quat(vector: SIMD4(q[0], q[1], q[2], q[3])),
                collisionEnabled: false)
            _ = scene.addConvexCollider(body: owner, vertices: (a["vertices"] as! [[NSNumber]]).map { xyz($0.map { $0.floatValue }) })
            let solver = try GPUSolver(scene: scene)
            try solver.submitStep()
            try solver.synchronize()
            XCTAssertNil(solver.runtimeFailure, "captured pair \(index)")
            XCTAssertFalse(solver.activeRigidContactPairs().isEmpty, "captured pair \(index) lost its contact")
            let stats = solver.lastConvexQueryStatistics
            XCTAssertEqual(total(stats), solver.lastNumPairs, "captured pair \(index): \(stats)")
            XCTAssertEqual(stats.failures, 0, "captured pair \(index)")
            searched += stats.satQueries
            pruned += stats.satEdgePairsPruned
            tested += stats.satEdgeAxesTested
            print("captured pair \(index) statistics: \(stats.dictionary.sorted { $0.key < $1.key })")
        }
        // The captured failures are the pairs that reach the complete search.
        // With pruning, the search still certifies every contact above while
        // testing a minority of the edge/edge axes it used to test.
        if searched > 0 {
            XCTAssertGreaterThan(pruned, 0)
            XCTAssertLessThan(tested, pruned, "Gauss-map pruning should remove most edge pairs: tested \(tested), pruned \(pruned)")
        }
    }
}
