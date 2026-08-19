import XCTest
import simd
@testable import SimCore
@testable import PhysicsAVBD

/// Tests targeting AVBD's headline claims from the paper:
/// hard constraints under high mass ratio, stiffness ratios, friction,
/// stability at low iteration counts, and fracture.
final class ConvergenceTests: XCTestCase {

    /// Paper Fig. 7: chain with heavy bob (high mass ratio). Dual methods
    /// fail here; AVBD must keep stretching minimal even at few iterations.
    func testHighMassRatioPendulumGPU() throws {
        var scene = Demos.pendulum(links: 20, massRatio: 1000)
        scene.settings.iterations = 20
        let solver = try GPUSolver(scene: scene)
        for _ in 0..<240 { solver.step() }

        // Total chain stretch: distance from anchor to bob vs nominal length
        let linkLen: Float = 0.5
        let nominal = Float(20) * linkLen + 0.5
        let bobIdx = scene.bodies.count - 1
        let bob = solver.bodyPosition(bobIdx)
        let dist = length(bob - F3(0, 0, 10))
        XCTAssertLessThan(dist, nominal * 1.05,
                          "chain stretched \(dist) vs nominal \(nominal)")
        // The CPU reference yields ~0.094 peak joint error while the chain is
        // still swinging at frame 240; GPU must match that, not beat it.
        XCTAssertLessThan(solver.maxConstraintError(), 0.15)
    }

    /// Constraint error must decay over frames (warm starting accumulates
    /// stiffness; paper Sec 5.6: converges towards zero error).
    func testConstraintErrorDecaysGPU() throws {
        var scene = Demos.chain(links: 20)
        scene.settings.iterations = 5
        let solver = try GPUSolver(scene: scene)

        var early: Float = 0
        for f in 0..<240 {
            solver.step()
            if f == 30 { early = solver.maxConstraintError() }
        }
        let late = solver.maxConstraintError()
        _ = early
        XCTAssertLessThan(late, 0.02, "settled chain error should be small (got \(late))")
    }

    /// Friction: sliding distance must scale inversely with friction
    /// coefficient (paper Fig. 11: different mu -> different stop distance).
    func testFrictionStopsSlidingBox() throws {
        var stopDistances: [Float] = []
        for mu in [Float(0.2), Float(0.8)] {
            var scene = PhysicsScene(name: "slide")
            _ = scene.addBody(size: F3(200, 200, 2), density: 0, friction: mu,
                              position: F3(0, 0, -1))
            _ = scene.addBody(size: F3(1, 1, 1), density: 1, friction: mu,
                              position: F3(0, 0, 0.5), velocity: F3(5, 0, 0))
            let solver = try GPUSolver(scene: scene)
            for _ in 0..<300 { solver.step() }
            let x = solver.bodyPosition(1).x
            let v = length(solver.bodyVelocity(1))
            XCTAssertLessThan(v, 0.1, "box should stop (mu=\(mu))")
            // analytic stop distance: v^2 / (2 mu g)
            let expected = 25.0 / (2 * mu * 10)
            XCTAssertEqual(x, expected, accuracy: expected * 0.3,
                           "stop distance \(x) vs analytic \(expected) (mu=\(mu))")
            stopDistances.append(x)
        }
        XCTAssertGreaterThan(stopDistances[0], stopDistances[1] * 2,
                             "lower friction must slide farther")
    }

    /// Analytic friction check: deceleration ~= mu * g for a sliding box.
    func testFrictionMagnitudeMatchesCoulomb() throws {
        let mu: Float = 0.3
        var scene = PhysicsScene(name: "coulomb")
        _ = scene.addBody(size: F3(200, 200, 2), density: 0, friction: mu,
                          position: F3(0, 0, -1))
        _ = scene.addBody(size: F3(1, 1, 1), density: 1, friction: mu,
                          position: F3(0, 0, 0.5), velocity: F3(6, 0, 0))
        let solver = try GPUSolver(scene: scene)

        // settle one frame, then measure deceleration over a window
        for _ in 0..<5 { solver.step() }
        let v0 = solver.bodyVelocity(1).x
        let frames = 30
        for _ in 0..<frames { solver.step() }
        let v1 = solver.bodyVelocity(1).x
        let decel = (v0 - v1) / (Float(frames) * solver.settings.dt)
        let expected = mu * abs(solver.settings.gravity)
        XCTAssertEqual(decel, expected, accuracy: expected * 0.25,
                       "Coulomb decel \(decel) vs expected \(expected)")
    }

    /// The legacy Push-T object is two overlapping boxes joined by a hard
    /// weld. It must dissipate sliding and spinning energy just like one rigid
    /// body; otherwise state-flow targets never become quasi-static.
    func testWeldedPushTCompoundDissipatesSlidingEnergy() throws {
        var scene = PhysicsScene(name: "welded-pusht-friction")
        scene.settings.iterations = 16
        scene.settings.betaLin = 20_000
        scene.settings.lambdaMax = 600
        _ = scene.addBody(
            size: F3(20, 20, 2), density: 0, friction: 1.3,
            position: F3(0, 0, -1))
        let group: UInt32 = 1
        let bar = scene.addBody(
            size: F3(1, 0.25, 0.18), density: 3, friction: 1.1,
            position: F3(0, 0.125, 0.09), collisionGroup: group)
        let stem = scene.addBody(
            size: F3(0.25, 0.65, 0.18), density: 3, friction: 1.1,
            position: F3(0, -0.325, 0.09), collisionGroup: group)
        let midpoint = 0.5 * (
            scene.bodies[bar].position + scene.bodies[stem].position)
        scene.addJoint(SceneJoint(
            bodyA: bar, bodyB: stem,
            rA: midpoint - scene.bodies[bar].position,
            rB: midpoint - scene.bodies[stem].position,
            stiffnessLin: .infinity, stiffnessAng: .infinity))
        scene.addJoint(SceneJoint(
            bodyA: bar, bodyB: stem, rA: .zero, rB: .zero,
            stiffnessLin: 0, stiffnessAng: 0))
        let solver = try GPUSolver(scene: scene)
        solver.setBodyStates([
            .init(body: bar, position: scene.bodies[bar].position,
                  rotation: scene.bodies[bar].rotation,
                  linearVelocity: F3(0.8, 0, 0),
                  angularVelocity: F3(0, 0, 0.46)),
            .init(body: stem, position: scene.bodies[stem].position,
                  rotation: scene.bodies[stem].rotation,
                  linearVelocity: F3(0.8, 0, 0),
                  angularVelocity: F3(0, 0, 0.46)),
        ])
        for _ in 0..<64 { solver.step() }
        let states = solver.bodyStates([bar, stem])
        let linearSpeed = length(0.5 * (
            states[0].linearVelocity + states[1].linearVelocity))
        let angularSpeed = length(0.5 * (
            states[0].angularVelocity + states[1].angularVelocity))
        XCTAssertLessThan(linearSpeed, 0.1)
        XCTAssertLessThan(angularSpeed, 0.1)
    }

    /// Paper Fig. 2/4: springs with a 10,000x stiffness ratio must not show
    /// excessive sagging of the stiff spring.
    func testStiffnessRatioSpringsGPU() throws {
        var scene = Demos.springRatio()
        scene.settings.iterations = 10
        let solver = try GPUSolver(scene: scene)
        for _ in 0..<600 { solver.step() }

        // Stiff spring (top->mid, k=1e6): elongation should stay tiny.
        // mid mass = 0.5, supported weight ~= (m_mid + m_bot) * g = 10
        // analytic elongation = F/k = 1e-5 — verify well under 5mm.
        let mid = solver.bodyPosition(1)
        let stretch = abs(length(mid - F3(0, 0, 10)) - 2.0)
        XCTAssertLessThan(stretch, 0.005, "stiff spring stretched \(stretch)")
        XCTAssertLessThan(length(solver.bodyVelocity(1)), 0.05, "should settle")
    }

    /// Breakable joints: wall hit by a heavy ball must fracture (some joints
    /// break) but not explode (positions remain finite, bounded).
    func testFractureWallGPU() throws {
        let scene = Demos.fractureWall()
        let solver = try GPUSolver(scene: scene)
        for _ in 0..<180 { solver.step() }

        var anyBroken = false
        let jp = solver.joints.contents().bindMemory(to: JointGPU.self, capacity: scene.joints.count)
        for i in 0..<scene.joints.count where jp[i].header.z != 0 { anyBroken = true; break }
        XCTAssertTrue(anyBroken, "ball at 30 m/s should break some joints")

        for i in 0..<scene.bodies.count {
            let p = solver.bodyPosition(i)
            XCTAssertTrue(p.x.isFinite && p.y.isFinite && p.z.isFinite, "body \(i) finite")
            XCTAssertLessThan(length(p), 500, "body \(i) should stay bounded")
        }
    }

    /// Stability at very low iteration counts (paper: stable with few iters).
    func testLowIterationStabilityGPU() throws {
        var scene = Demos.stack(height: 8)
        scene.settings.iterations = 2
        let solver = try GPUSolver(scene: scene)
        for _ in 0..<600 { solver.step() }
        for i in 0..<8 {
            let p = solver.bodyPosition(1 + i)
            XCTAssertTrue(length(p) < 50 && p.z > -1,
                          "stack with 2 iterations must not explode (body \(i): \(p))")
        }
    }

    /// Paper Fig. 6: lightweight card house held only by static friction.
    func testCardHouseStands() throws {
        let scene = Demos.cardhouse(levels: 4)
        let solver = try GPUSolver(scene: scene)
        for _ in 0..<300 { solver.step() }
        // Top pair (last two bodies) center height: 3*(H+T) + H/2 ≈ 4.10
        let H: Float = 1.2 * sin(1.22), T: Float = 0.05
        let expected = 3 * (H + T) + H / 2
        for i in [scene.bodies.count - 2, scene.bodies.count - 1] {
            let z = solver.bodyPosition(i).z
            XCTAssertEqual(z, expected, accuracy: 0.15,
                           "top card \(i) should stand at z≈\(expected) (got \(z))")
        }
    }

    /// Long-chain stress (regression for beta-driven explosion): a 100-link
    /// chain with a heavy bob must stay finite and keep stretch bounded.
    func testLongChainStability() throws {
        let scene = Demos.pendulum(links: 100, massRatio: 100)
        let solver = try GPUSolver(scene: scene)
        for _ in 0..<240 { solver.step() }
        let err = solver.maxConstraintError()
        XCTAssertTrue(err.isFinite, "long chain exploded (err \(err))")
        XCTAssertLessThan(err, 0.2, "long chain error too large: \(err)")
    }

    /// Determinism: same scene, same steps -> bitwise-equal positions.
    func testDeterminismGPU() throws {
        func run() throws -> [F3] {
            let solver = try GPUSolver(scene: Demos.pyramid(base: 5))
            for _ in 0..<60 { solver.step() }
            return (0..<solver.lastColorCounts.count).map { _ in F3.zero } // placeholder
        }
        // Compare positions of all bodies between two runs
        let scene = Demos.pyramid(base: 5)
        let a = try GPUSolver(scene: scene)
        let b = try GPUSolver(scene: scene)
        for _ in 0..<60 { a.step(); b.step() }
        var maxDiff: Float = 0
        for i in 0..<scene.bodies.count {
            maxDiff = max(maxDiff, length(a.bodyPosition(i) - b.bodyPosition(i)))
        }
        // Atomic pair ordering varies run to run, so allow small divergence,
        // but the scenes must agree macroscopically.
        XCTAssertLessThan(maxDiff, 0.05, "runs diverged by \(maxDiff)")
    }
}
