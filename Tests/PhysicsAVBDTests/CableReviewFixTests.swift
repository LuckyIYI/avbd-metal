import XCTest
import simd
@testable import SimCore
@testable import PhysicsAVBD

/// Fixes from the review of the capsule-chain cable path: capsule contact
/// anchors, CPU/GPU parity of hard-constraint seeding and dual bounds,
/// tension-only CPU rods, the capsule inertia tensor, and junction
/// exclusions for knotted cords.
final class CableReviewFixTests: XCTestCase {
    private let alongX = Quat(angle: .pi / 2, axis: F3(0, 1, 0))

    /// Minimal rotation taking local +z onto a unit direction.
    private func alignZ(to direction: F3) -> Quat {
        let z = F3(0, 0, 1)
        let cosine = min(max(dot(z, direction), -1), 1)
        if cosine > 0.99999 { return Quat(real: 1, imag: .zero) }
        if cosine < -0.99999 { return Quat(angle: .pi, axis: F3(1, 0, 0)) }
        return Quat(angle: acos(cosine), axis: normalize(cross(z, direction)))
    }

    func testCapsuleInertiaLimits() {
        let rod = capsuleInertia(mass: 1, cylinderLength: 2, radius: 1e-4)
        XCTAssertEqual(rod.x, 4.0 / 12.0, accuracy: 1e-3)
        XCTAssertEqual(rod.z, 0, accuracy: 1e-6)
        let ball = capsuleInertia(mass: 1, cylinderLength: 0, radius: 0.5)
        XCTAssertEqual(ball.x, 0.4 * 0.25, accuracy: 1e-6)
        XCTAssertEqual(ball.z, 0.4 * 0.25, accuracy: 1e-6)
        // Caps add mass far from the centre: the full tensor exceeds the
        // bare-cylinder formula applied to the same total mass.
        let m: Float = 1, L: Float = 0.3, r: Float = 0.05
        let full = capsuleInertia(mass: m, cylinderLength: L, radius: r)
        XCTAssertGreaterThan(full.x, m * (L * L / 12 + r * r / 4))
    }

    func testCapsuleLinkHoldsStaticFrictionOnSlope() throws {
        // A capsule made with addCapsule now keeps body-local contact
        // anchors, so its static-friction anchor survives across frames.
        // With the old world-space anchor the link crept downhill forever.
        var scene = PhysicsScene(name: "capsule-slope")
        scene.settings.dt = 1.0 / 120.0
        scene.settings.iterations = 20
        let tilt: Float = 15 * .pi / 180
        let slope = Quat(angle: tilt, axis: F3(0, 1, 0))
        _ = scene.addBody(size: F3(6, 6, 0.2), density: 0, friction: 0.6,
                          position: .zero, rotation: slope)
        // Axis along the incline direction so the link cannot roll.
        let downhill = slope.act(F3(1, 0, 0))
        let normal = slope.act(F3(0, 0, 1))
        let start = normal * (0.1 + 0.04)
        let link = scene.addCapsule(length: 0.4, radius: 0.04, density: 500,
                                    friction: 0.6, position: start,
                                    rotation: slope * alongX)
        XCTAssertFalse(scene.colliders.first { $0.body == link }!.usesWorldSpaceRoundAnchor)
        let gpu = try GPUSolver(scene: scene)
        for _ in 0..<240 { gpu.step() }
        let p = gpu.bodyPosition(link)
        let slid = dot(p - start, downhill)
        XCTAssertLessThan(abs(slid), 0.01,
                          "capsule crept \(slid) m along a slope it should hold on")
    }

    func testCPUHardRodIsTensionOnly() {
        let solver = CPUSolver()
        solver.gravity = 0
        solver.iterations = 10
        let a = solver.addBody(size: F3(repeating: 0.1), density: 1, friction: 0,
                               position: .zero)
        let b = solver.addBody(size: F3(repeating: 0.1), density: 1, friction: 0,
                               position: F3(0.5, 0, 0))
        let rod = solver.addSpring(a, b, rA: .zero, rB: .zero, stiffness: 1e5, rest: 1.0)
        rod.hard = true
        for _ in 0..<120 { solver.step() }
        XCTAssertEqual(distance(a.positionLin, b.positionLin), 0.5, accuracy: 1e-3,
                       "a compressed hard rod must not push its ends apart")
    }

    func testHardJointsAreLoadBearingOnFrameOneOnCPU() {
        // A heavy box hanging from a world ball joint: with the mass/dt^2
        // penalty floor the first frame sags by well under a millimetre,
        // as on the GPU, instead of the centimetres PENALTY_MIN allowed.
        var s = PhysicsScene(name: "first-frame")
        s.settings.dt = 1.0 / 60.0
        s.settings.iterations = 10
        let box = s.addBody(size: F3(repeating: 0.5), density: 8,
                            friction: 0.5, position: F3(0, 0, 1))
        s.addJoint(SceneJoint(bodyA: -1, bodyB: box,
                              rA: F3(0, 0, 1.25), rB: F3(0, 0, 0.25)))
        let cpu = s.makeCPUSolver()
        cpu.step()
        XCTAssertLessThan(cpu.maxConstraintError(), 1e-3)
    }

    func testHangingCapsuleChainCPUAndGPUAgree() throws {
        // Ball-jointed capsule chain released from a gentle sideways curve
        // and settled under drag: both backends now seed hard joints at the
        // same floor and bound their duals the same way, so they settle to
        // the same shape.
        var scene = PhysicsScene(name: "chain-parity")
        scene.settings.dt = 1.0 / 120.0
        scene.settings.iterations = 20
        scene.settings.rigidLinearDamping = 2
        scene.settings.rigidAngularDamping = 2
        let points = (0...8).map { i -> F3 in
            let t = Float(i) / 8
            return F3(0.25 * sin(t * .pi), 0, 2 - Float(i) * 0.125)
        }
        var links: [Int] = []
        var half: [Float] = []
        for i in 0..<(points.count - 1) {
            let d = points[i + 1] - points[i]
            let span = length(d)
            let link = scene.addCapsule(length: span - 0.02, radius: 0.01, density: 900,
                                        friction: 0.3,
                                        position: (points[i] + points[i + 1]) / 2,
                                        rotation: alignZ(to: d / span))
            links.append(link)
            half.append(span / 2)
        }
        scene.addJoint(SceneJoint(bodyA: -1, bodyB: links[0],
                                  rA: points[0], rB: F3(0, 0, -half[0])))
        for i in 1..<links.count {
            scene.addJoint(SceneJoint(bodyA: links[i - 1], bodyB: links[i],
                                      rA: F3(0, 0, half[i - 1]), rB: F3(0, 0, -half[i])))
        }
        let gpu = try GPUSolver(scene: scene)
        let cpu = scene.makeCPUSolver()
        for _ in 0..<600 { gpu.step(); cpu.step() }
        var worst: Float = 0
        for body in links {
            let g = gpu.bodyPosition(body)
            let c = cpu.bodies[body].positionLin
            XCTAssertTrue(g.x.isFinite && g.y.isFinite && g.z.isFinite)
            worst = max(worst, distance(g, c))
        }
        XCTAssertLessThan(worst, 0.005, "CPU and GPU chains diverged by \(worst)")
        let tip = gpu.bodyPosition(links.last!)
        XCTAssertLessThan(abs(tip.x), 0.02, "a ball chain must hang straight (tip x \(tip.x))")
        XCTAssertLessThan(gpu.maxConstraintError(), 1e-3)
        XCTAssertLessThan(cpu.maxConstraintError(), 1e-3)
    }

    func testJunctionExclusionsSkipSelfPairs() {
        var s = PhysicsScene(name: "junction")
        let a = s.addCapsule(length: 0.2, radius: 0.01, density: 1, friction: 0.3,
                             position: .zero)
        let b = s.addCapsule(length: 0.2, radius: 0.01, density: 1, friction: 0.3,
                             position: F3(1, 0, 0))
        let c = s.addCapsule(length: 0.2, radius: 0.01, density: 1, friction: 0.3,
                             position: F3(2, 0, 0))
        s.addCableJunctionExclusions([[a, b], [b, c]])
        let pairs = Set(s.collisionExclusions.map { Set([$0.bodyA, $0.bodyB]) })
        XCTAssertEqual(pairs, [Set([a, b]), Set([a, c]), Set([b, c])])
    }
}
