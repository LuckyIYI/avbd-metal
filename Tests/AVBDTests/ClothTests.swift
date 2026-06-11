import XCTest
import simd
@testable import AVBDCore

/// Cloth as a first-class citizen: surface-element contacts (V-T, rigid-T,
/// E-E) with the full AVBD treatment. These are the plan gates.
final class ClothTests: XCTestCase {

    /// Gate 1a: sheet folded over itself twice (3 layers) settles on the
    /// ground; layers must hold separation for 10 simulated seconds.
    func testFoldedClothNoInterpenetration() throws {
        let scene = Demos.clothfold(res: 20)
        let gpu = try GPUSolver(scene: scene)
        for _ in 0..<600 { gpu.step() }
        let (gap, stretch) = gpu.debugClothMetrics()
        // skin = rv + rt = 0.09; equilibrium sits at -margin (-0.01).
        // Half-skin is the no-tunnel red line.
        XCTAssertGreaterThan(gap, -0.045,
                             "fold layers must not interpenetrate (gap \(gap))")
        XCTAssertLessThan(stretch, 0.05, "fold should not stress the weave")
        XCTAssertGreaterThan(gpu.lastNumSoft, 100,
                             "V-T contacts must be active between layers")
        // settled: kinetic energy of the pile ~ zero
        var ke: Float = 0
        for b in 0..<scene.bodies.count where scene.bodies[b].isParticle {
            ke += 0.5 * gpu.bodyMass(b) * length_squared(gpu.bodyVelocity(b))
        }
        XCTAssertLessThan(ke, 0.01, "folded pile must settle, not jitter")
    }

    /// Gate 1b: a rigid box of 8x the cloth's total mass rests on a draped
    /// cloth by face contact — no poke-through between nodes, no dimple
    /// thrash, box stays put.
    func testHeavyBoxRestsOnDrapedCloth() throws {
        let scene = Demos.boxoncloth(res: 20, massRatio: 8)
        let box = scene.bodies.count - 1
        let gpu = try GPUSolver(scene: scene)
        for _ in 0..<480 { gpu.step() }
        let p = gpu.bodyPosition(box)
        // pedestal top at 0.8 + cloth skin + half box (0.275): resting well
        // above the bare pedestal proves the cloth carries the box
        XCTAssertGreaterThan(p.z, 1.05,
                             "box must rest ON the cloth, not poke through (z \(p.z))")
        XCTAssertLessThan(max(abs(p.x), abs(p.y)), 0.3,
                          "box must rest where it was dropped")
        XCTAssertLessThan(length(gpu.bodyVelocity(box)), 0.1, "box must be at rest")
        let (gap, _) = gpu.debugClothMetrics()
        XCTAssertGreaterThan(gap, -0.05, "cloth self-clearance under load")
        XCTAssertGreaterThan(gpu.lastNumSoft, 10,
                             "rigid-feature-vs-triangle contacts must be active")
    }

    /// The drape must come to rest (the energy-injection regression test:
    /// stale color bounds / unbounded duals made piles thrash at m/s).
    func testDrapeSettles() throws {
        let scene = Demos.drape(res: 24)
        let gpu = try GPUSolver(scene: scene)
        for _ in 0..<480 { gpu.step() }
        var ke: Float = 0
        for b in 0..<scene.bodies.count where scene.bodies[b].isParticle {
            ke += 0.5 * gpu.bodyMass(b) * length_squared(gpu.bodyVelocity(b))
        }
        XCTAssertLessThan(ke, 0.5, "draped cloth must dissipate, not pump (KE \(ke))")
        let (gap, stretch) = gpu.debugClothMetrics()
        XCTAssertGreaterThan(gap, -0.05, "no deep self-penetration in the drape")
        XCTAssertLessThan(stretch, 0.08, "drape hangs without tearing the weave")
    }
}
