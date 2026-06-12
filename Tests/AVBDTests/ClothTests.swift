import XCTest
import simd
@testable import AVBDCore

/// Cloth as a first-class citizen: surface-element contacts (V-T, rigid-T,
/// E-E) with the full AVBD treatment. These are the plan gates.
final class ClothTests: XCTestCase {

    /// Gate 1a: sheet folded over itself twice (3 layers) settles on the
    /// ground; layers must hold separation for 10 simulated seconds.
    func testFoldedClothNoInterpenetration() throws {
        let scene = Demos.clothfold(res: 36)
        let gpu = try GPUSolver(scene: scene)
        for _ in 0..<600 { gpu.step() }
        let (gap, stretch) = gpu.debugClothMetrics()
        // towel-scale fold: spacing 2.2/35, r = 0.36*spacing ~ 0.0226,
        // skin = 2r ~ 0.045; equilibrium sits at -margin. Half-skin is
        // the no-tunnel red line.
        XCTAssertGreaterThan(gap, -0.023,
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

    /// Gate 3: strip B dragged across taut strip A — their long edges rub
    /// corner-over-corner at the moving crossing. B must ride OVER A's
    /// midplane the whole way, never pass through.
    func testDraggedStripsDontPassAtCorners() throws {
        let (scene, dragJoints, aNodes, bNodes) = Demos.eecross()
        let gpu = try GPUSolver(scene: scene)
        for _ in 0..<120 { gpu.step() }             // settle
        var anchors = dragJoints.map { i -> F3 in
            scene.joints[i].rA
        }
        // Pass-through detector: a B node whose closest point on an A
        // triangle is INTERIOR and on the wrong (under) side, deeper than
        // half the skin, has gone through the strip. Hanging next to A's
        // edges resolves to boundary closest points and doesn't count.
        let aSet = Set(aNodes)
        let aTris = scene.tris.filter {
            aSet.contains($0.ids.0) && aSet.contains($0.ids.1) && aSet.contains($0.ids.2)
        }
        var violations = 0
        var sawEE = 0
        for f in 0..<300 {                          // 1.5 m drag over 5 s
            for (k, j) in dragJoints.enumerated() {
                anchors[k].y -= 0.30 / 60
                gpu.setJointWorldAnchor(j, point: anchors[k])
            }
            gpu.step()
            if f % 5 != 0 { continue }
            for b in bNodes {
                let p = gpu.bodyPosition(b)
                guard abs(p.x) < 1.0 && abs(p.y) < 0.25 && abs(p.z - 0.12) < 0.25 else { continue }
                for t in aTris {
                    let a = gpu.bodyPosition(t.ids.0)
                    let bb = gpu.bodyPosition(t.ids.1)
                    let c = gpu.bodyPosition(t.ids.2)
                    let n = normalize(cross(bb - a, c - a))    // +z winding
                    let sd = dot(p - a, n)
                    guard sd < -0.06 && sd > -0.3 else { continue }
                    // interior check via barycentric of the planar projection
                    let q = p - n * sd
                    let v0 = bb - a, v1 = c - a, v2 = q - a
                    let d00 = dot(v0, v0), d01 = dot(v0, v1), d11 = dot(v1, v1)
                    let d20 = dot(v2, v0), d21 = dot(v2, v1)
                    let den = d00 * d11 - d01 * d01
                    guard abs(den) > 1e-12 else { continue }
                    let v = (d11 * d20 - d01 * d21) / den
                    let w = (d00 * d21 - d01 * d20) / den
                    if v > 0.1 && w > 0.1 && (1 - v - w) > 0.1 {
                        violations += 1
                        print(String(format: "VIOLATION f=%d node=%d depth=%.3f at (%.2f, %.2f, %.2f)",
                                     f, b, sd, p.x, p.y, p.z))
                    }
                }
            }
            if f % 30 == 0 { sawEE += gpu.debugSoftKinds().ee }
        }
        XCTAssertEqual(violations, 0,
                       "strip B must never sit under an interior point of strip A")
        XCTAssertGreaterThan(sawEE, 0, "E-E contacts must fire during the drag")
    }

    /// Gate 2a: hammock under a rigid box — structural stretch < 2% with
    /// hard rods carrying the load.
    func testHammockInextensible() throws {
        let scene = Demos.hammock(res: 16)
        let gpu = try GPUSolver(scene: scene)
        for _ in 0..<480 { gpu.step() }
        let (gap, stretch) = gpu.debugClothMetrics()
        XCTAssertLessThan(stretch, 0.02,
                          "hammock must stay inextensible under load (stretch \(stretch))")
        XCTAssertGreaterThan(gap, -0.06, "no self-tunneling in the hammock")
        // the cargo must be cradled, not dropped
        let box = scene.bodies.count - 1
        XCTAssertGreaterThan(gpu.bodyPosition(box).z, 0.5,
                             "hammock must cradle the box above the ground")
    }

    /// Gate 2b: whipping flag, 30 s — the KE envelope must decay (hard-rod
    /// duals on rotating rods must not pump the pendulum modes).
    func testFlagWhipNoEnergyGrowth() throws {
        let scene = Demos.flagwhip(res: 14)
        let gpu = try GPUSolver(scene: scene)
        var windowMax: [Float] = []
        var cur: Float = 0
        for f in 0..<1800 {
            gpu.step()
            if f % 10 == 0 {
                var ke: Float = 0
                for b in 0..<scene.bodies.count where scene.bodies[b].isParticle {
                    ke += 0.5 * gpu.bodyMass(b) * length_squared(gpu.bodyVelocity(b))
                }
                cur = max(cur, ke)
            }
            if (f + 1) % 600 == 0 { windowMax.append(cur); cur = 0 }
        }
        XCTAssertLessThan(windowMax[2], windowMax[0] * 1.05,
                          "whip KE envelope must decay, not grow \(windowMax)")
        let (_, stretch) = gpu.debugClothMetrics()
        XCTAssertLessThan(stretch, 0.02, "rods stay inextensible while whipping")
    }

    /// Gate 4: tets + membrane cloth + rigid bodies in ONE unified solve,
    /// stable at 20 iterations. Everything must come to rest with the cloth
    /// draped over the soft block and the rigids resting on top.
    func testCombinedTetClothRigidStable() throws {
        let scene = Demos.clothcombo(res: 16)
        XCTAssertEqual(scene.settings.iterations, 20)
        let gpu = try GPUSolver(scene: scene)
        for _ in 0..<600 { gpu.step() }
        var ke: Float = 0
        for b in 0..<scene.bodies.count where scene.bodies[b].isParticle {
            ke += 0.5 * gpu.bodyMass(b) * length_squared(gpu.bodyVelocity(b))
        }
        XCTAssertLessThan(ke, 0.2, "combined scene must settle (KE \(ke))")
        let (gap, stretch) = gpu.debugClothMetrics()
        XCTAssertGreaterThan(gap, -0.06, "no cloth self-penetration in the pile")
        XCTAssertLessThan(stretch, 0.02, "membrane cloth holds its weave")
        // the dropped rigids must rest on the stack, not inside the ground
        for b in (scene.bodies.count - 2)..<scene.bodies.count {
            let p = gpu.bodyPosition(b)
            XCTAssertGreaterThan(p.z, 0.1, "rigid body \(b) rests above ground")
            XCTAssertLessThan(length(gpu.bodyVelocity(b)), 0.2, "rigid at rest")
        }
    }

    /// The drape must come to rest (the energy-injection regression test:
    /// stale color bounds / unbounded duals made piles thrash at m/s).
    func testDrapeSettles() throws {
        let scene = Demos.drape(res: 24)
        let gpu = try GPUSolver(scene: scene)
        for _ in 0..<720 { gpu.step() }
        var ke: Float = 0
        for b in 0..<scene.bodies.count where scene.bodies[b].isParticle {
            ke += 0.5 * gpu.bodyMass(b) * length_squared(gpu.bodyVelocity(b))
        }
        XCTAssertLessThan(ke, 0.8, "draped cloth must dissipate, not pump (KE \(ke))")
        let (gap, stretch) = gpu.debugClothMetrics()
        XCTAssertGreaterThan(gap, -0.05, "no deep self-penetration in the drape")
        // localized wedged-fold pockets may hold capped-lambda rods at ~10%;
        // the inextensibility-under-load gate is the hammock (<2%)
        XCTAssertLessThan(stretch, 0.15, "drape hangs without tearing the weave")
    }
}
