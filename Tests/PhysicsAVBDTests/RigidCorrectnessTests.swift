import XCTest
import simd
@testable import SimCore
@testable import PhysicsAVBD

/// Targeted rigid-body correctness: static friction thresholds and
/// mass-ratio stacking (each box larger than the one beneath).
final class RigidCorrectnessTests: XCTestCase {

    /// On the 30° slope with a mu=1 plane, pair friction is sqrt(mu_box):
    /// boxes with sqrt(mu) well below tan(30°) ≈ 0.577 must slide, boxes
    /// well above must stick. (Boxes near the threshold are left alone —
    /// that boundary is genuinely marginal.)
    func testStaticFrictionSweep() throws {
        let scene = Demos.slopefriction(count: 8)
        let gpu = try GPUSolver(scene: scene)
        let first = 2                       // 0 = ground, 1 = slope
        var initial: [F3] = []
        for k in 0..<8 { initial.append(gpu.bodyPosition(first + k)) }
        for _ in 0..<400 { gpu.step() }
        for k in [0, 1] {                   // mu 0.0, 0.14 -> slide
            let d = gpu.bodyPosition(first + k) - initial[k]
            XCTAssertGreaterThan(d.x, 0.5,
                                 "low-friction box \(k) must slide down the slope")
        }
        for k in [6, 7] {                   // mu 0.86, 1.0 -> stick
            let d = gpu.bodyPosition(first + k) - initial[k]
            XCTAssertLessThan(length(d), 0.08,
                              "high-friction box \(k) must hold static friction")
        }
    }

    /// Growing stack: a 1.45x size step per level is a ~3x mass step —
    /// the heavy top must neither sink through nor squeeze out the base.
    func testRatioStackStands() throws {
        let scene = Demos.ratiostack(levels: 5)
        let top = scene.bodies.count - 1
        let gpu = try GPUSolver(scene: scene)
        let z0 = gpu.bodyPosition(top).z
        for _ in 0..<600 { gpu.step() }
        let p = gpu.bodyPosition(top)
        XCTAssertEqual(p.z, z0, accuracy: 0.1 * z0,
                       "top of the ratio stack must keep its height")
        XCTAssertLessThan(length(F3(p.x, p.y, 0)), 0.25,
                          "ratio stack must not lean or slide sideways")
    }

    /// The rubber-tire cart must actually drive: motor torque through the
    /// soft tires into ground friction.
    func testSoftWheelDrives() throws {
        let scene = Demos.softwheel(res: 9, drive: 6)
        let gpu = try GPUSolver(scene: scene)
        let x0 = gpu.bodyPosition(1).x      // chassis
        for _ in 0..<300 { gpu.step() }
        XCTAssertGreaterThan(gpu.bodyPosition(1).x - x0, 2.0,
                             "cart must drive forward on its rubber tires")
    }

    /// The bed must be quiet: ball settles on the blanket, nothing flies.
    func testBedSettles() throws {
        let scene = Demos.bed(res: 30)
        var ball = -1
        for (i, b) in scene.bodies.enumerated()
            where b.shape == .sphere && !b.isParticle { ball = i }
        XCTAssertGreaterThanOrEqual(ball, 0)
        let gpu = try GPUSolver(scene: scene)
        for _ in 0..<500 { gpu.step() }
        let z = gpu.bodyPosition(ball).z
        XCTAssertGreaterThan(z, 0.5, "ball must rest on the bed, not the floor")
        XCTAssertLessThan(z, 1.4, "ball must have settled, not be bouncing")
    }
}
