import XCTest
import Metal
import SimCore
import simd
@testable import PhysicsAVBD

/// Captured hull pairs that latched a fail-closed frame in a live run. The
/// detection band of the capture is reproduced exactly: the capturing scene
/// used a 4 mm collision margin, and the speculative cap follows from the
/// colliders' own bounding radii.
final class NarrowphaseRegressionPairTests: XCTestCase {
    private struct Outcome { var failure: Error?; var contacts: Int }

    private func requireMetal() throws {
        if MTLCreateSystemDefaultDevice() == nil { throw XCTSkip("Metal is unavailable") }
    }

    private func pairs() throws -> [[String: Any]] {
        let root = try JSONSerialization.jsonObject(with: Data(narrowphaseRegressionPairsJSON.utf8)) as! [String: Any]
        return root["pairs"] as! [[String: Any]]
    }

    /// `approach` gives shape B a velocity towards shape A. The caller's
    /// detection band is margin + min(approach * dt, cap); at rest it is the
    /// bare margin, so the band edge of a capture taken from a thrown prop
    /// can only be reproduced with an approaching body.
    private func run(_ pair: [String: Any], translateB: F3 = .zero, approach: Float = 0) throws -> Outcome {
        func floats(_ s: [String: Any], _ key: String) -> [Float] { (s[key] as! [NSNumber]).map { $0.floatValue } }
        func xyz(_ a: [Float]) -> F3 { F3(a[0], a[1], a[2]) }
        var scene = PhysicsScene(name: pair["name"] as! String)
        scene.settings.gravity = 0
        scene.settings.iterations = 0
        scene.settings.collisionMargin = 0.004
        let shapes = pair["shapes"] as! [[String: Any]]
        let centerA = xyz(floats(shapes[0], "center"))
        let centerB = xyz(floats(shapes[1], "center")) + translateB
        // The capture's certified axis from shape 0 to shape 1; the detection
        // band grows with the approach speed along the contact normal only.
        let axis = xyz(floats(pair, "normal"))
        for (index, shape) in shapes.enumerated() {
            let vertices = (shape["vertices"] as! [[NSNumber]]).map { xyz($0.map { $0.floatValue }) }
            let q = floats(shape, "rotation")
            let lo = vertices.reduce(F3(repeating: .greatestFiniteMagnitude), simd.min)
            let hi = vertices.reduce(F3(repeating: -.greatestFiniteMagnitude), simd.max)
            let velocity = index == 1 && approach > 0 ? -simd_normalize(axis) * approach : F3.zero
            let body = scene.addBody(size: simd.max(hi - lo, F3(repeating: 1e-3)), density: 1, friction: 0.5,
                                     position: index == 0 ? centerA : centerB,
                                     rotation: Quat(vector: SIMD4(q[0], q[1], q[2], q[3])),
                                     velocity: velocity, collisionEnabled: false)
            _ = scene.addConvexCollider(body: body, vertices: vertices)
        }
        let solver = try GPUSolver(scene: scene)
        try solver.submitStep()
        var failure: Error?
        do { try solver.synchronize() } catch { failure = error }
        return Outcome(failure: failure, contacts: solver.activeRigidContactPairs().count)
    }

    func testDegenerateEdgesAreSkippedInsteadOfLatchingFailure() throws {
        try requireMetal()
        for pair in try pairs() where (pair["name"] as! String).hasPrefix("degenerate_edge_abort") {
            let outcome = try run(pair)
            XCTAssertNil(outcome.failure, "\(pair["name"]!): \(String(describing: outcome.failure))")
            if pair["name"] as! String == "degenerate_edge_abort" {
                XCTAssertGreaterThan(outcome.contacts, 0, "the penetrating pair must keep its contact")
            }
        }
    }

    func testCertifiedSeparationBeyondBandIsAnAnswerAndBandEdgeIsSharp() throws {
        try requireMetal()
        for pair in try pairs() where (pair["name"] as! String).hasPrefix("certified_separation_beyond_band") {
            let outcome = try run(pair)
            XCTAssertNil(outcome.failure, "\(pair["name"]!): \(String(describing: outcome.failure))")
            XCTAssertEqual(outcome.contacts, 0, "\(pair["name"]!) is separated beyond the band")
            for variant in pair["band_variants"] as! [[String: Any]] {
                let shift = (variant["translate_shape_b"] as! [NSNumber]).map { $0.floatValue }
                // 30 m/s towards A opens the band to margin + cap, the bound
                // the capture recorded.
                let moved = try run(pair, translateB: F3(shift[0], shift[1], shift[2]), approach: 30)
                XCTAssertNil(moved.failure, "\(pair["name"]!) \(variant["label"]!)")
                if variant["label"] as! String == "just_inside" {
                    XCTAssertGreaterThan(moved.contacts, 0, "\(pair["name"]!) 200 um inside the band keeps a contact")
                } else {
                    XCTAssertEqual(moved.contacts, 0, "\(pair["name"]!) 200 um outside the band has none")
                }
            }
        }
    }

    func testSeparatedPairsInsideBandDoNotFailOnMain() throws {
        try requireMetal()
        // On the integration branch with tightened MPR admission these pairs
        // reach the complete search and its separated-case manifold witness
        // does not attain the certified distance (open item). On this base the
        // standard query answers them; keep them so a regression is visible.
        for pair in try pairs() where (pair["name"] as! String).hasPrefix("bad_manifold_witness") {
            let outcome = try run(pair, approach: 30)
            XCTAssertNil(outcome.failure, "\(pair["name"]!): \(String(describing: outcome.failure))")
        }
    }
}
