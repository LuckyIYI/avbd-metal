import XCTest
import simd
@testable import AVBDCore

final class GaudiFunicularTests: XCTestCase {
    private func authoredMass(_ body: SceneBody) -> Float {
        if let mass = body.mass { return mass }
        switch body.shape {
        case .box:
            return body.density * body.size.x * body.size.y * body.size.z
        case .sphere:
            let r = body.size.x / 2
            return body.density * 4 / 3 * .pi * r * r * r
        case .capsule:
            return body.density * .pi * body.size.y * body.size.y
                * (body.size.x + 4 * body.size.y / 3)
        case .torus:
            return body.density * 2 * .pi * .pi
                * body.size.x * body.size.y * body.size.y
        }
    }

    func testPublishedTempleModuleDrivesPlanAndLoads() throws {
        let scene = Demos.gaudiFunicular()
        XCTAssertEqual(scene.name, "gaudifunicular")
        XCTAssertTrue(Demos.all.contains("gaudifunicular"))
        XCTAssertEqual(GaudiFunicularBlueprint.moduleMeters, 7.5)

        // Static spherical pins include the 84 foundation-plan control points
        // and 82 tower-ring suspension pins. Their extrema must reproduce the
        // published 90 m x 60 m interior/transept envelope at model scale.
        let pins = scene.bodies.filter {
            !$0.isDynamic && $0.shape == .sphere && abs($0.position.z - 12.0) < 1e-4
        }
        XCTAssertEqual(pins.count, 166)
        let xs = pins.map(\.position.x)
        let ys = pins.map(\.position.y)
        XCTAssertEqual(xs.max()! - xs.min()!, GaudiFunicularBlueprint.interiorLength,
                       accuracy: 1e-4)
        XCTAssertEqual(ys.max()! - ys.min()!, GaudiFunicularBlueprint.transeptWidth,
                       accuracy: 1e-4)

        // Fifty-eight noncentral nave bays, four transept extensions, and all
        // eighteen towers carry canvas shot bags. The two central-crossing
        // reactions are subsumed by the Jesus tower's dedicated load.
        let bagIDs = scene.bodies.indices.filter {
            scene.bodies[$0].isDynamic && scene.bodies[$0].shape == .box
        }
        XCTAssertEqual(bagIDs.count, 80)
        let masses = bagIDs.map { id -> Float in
            let b = scene.bodies[id]
            return b.density * b.size.x * b.size.y * b.size.z
        }
        XCTAssertGreaterThan(masses.max()!, masses.min()! * 8,
                             "crossing bags must include the central tower reactions")

        // Every cord and shot bag participates in contact. The only explicit
        // exclusions are sibling cord branches tied into the same knot.
        let cordBodies = Set(scene.bodies.indices.filter {
            scene.bodies[$0].isDynamic && scene.bodies[$0].shape == .capsule
        })
        let bagBodies = Set(bagIDs)
        XCTAssertTrue(scene.colliders.filter { cordBodies.contains($0.body) }
            .allSatisfy(\.collisionEnabled))
        XCTAssertTrue(scene.colliders.filter { bagBodies.contains($0.body) }
            .allSatisfy(\.collisionEnabled))
        XCTAssertGreaterThan(scene.collisionExclusions.count, 100)
        XCTAssertTrue(scene.colliders.contains { $0.renderColor != nil })
        XCTAssertGreaterThan(scene.joints.count, 1_000)
    }

    func testDampingIsExposedAndCanBeDisabled() throws {
        let tunables = Demos.tunables("gaudifunicular")
        XCTAssertTrue(tunables.contains { $0.key == "linearDamping"
            && $0.range.contains(0) })
        XCTAssertTrue(tunables.contains { $0.key == "angularDamping"
            && $0.range.contains(0) })

        let scene = try XCTUnwrap(Demos.make(
            "gaudifunicular",
            params: ["linearDamping": 0, "angularDamping": 0]))
        XCTAssertEqual(scene.settings.rigidLinearDamping, 0)
        XCTAssertEqual(scene.settings.rigidAngularDamping, 0)
    }

    func testWeightedNetworkRemainsSuspendedUnderRealGravity() throws {
        let scene = Demos.gaudiFunicular(segmentsPerCable: 2,
                                         loadScale: 1, slack: 1.12)
        let bagIDs = scene.bodies.indices.filter {
            scene.bodies[$0].isDynamic && scene.bodies[$0].shape == .box
        }
        let initial = bagIDs.map { scene.bodies[$0].position }
        let gpu = try GPUSolver(scene: scene)
        for _ in 0..<360 { gpu.step() } // three physical seconds at 120 Hz

        var moved = false
        for (i, body) in bagIDs.enumerated() {
            let p = gpu.bodyPosition(body)
            XCTAssertTrue(p.x.isFinite && p.y.isFinite && p.z.isFinite)
            XCTAssertLessThan(p.z, 12.0, "every modeled load must hang below the plan")
            XCTAssertGreaterThan(p.z, -1, "no load may detach from the cable network")
            moved = moved || distance(p, initial[i]) > 0.01
        }
        XCTAssertTrue(moved, "gravity should physically recompute the seeded geometry")
        XCTAssertEqual(gpu.debugBrokenJoints(), 0)
        XCTAssertGreaterThan(gpu.lastNumPairs, 0,
                             "the dense cable net should exercise rigid contact")
    }

    func testRopeRopeAndBagRopeNarrowphase() throws {
        var scene = PhysicsScene(name: "gaudi-contact-contract")
        scene.settings.gravity = 0
        scene.settings.dt = 1.0 / 120.0
        scene.settings.iterations = 20
        scene.settings.collisionMargin = 0.002
        scene.settings.rigidLinearDamping = 3
        scene.settings.rigidAngularDamping = 3
        let alongX = Quat(angle: .pi / 2, axis: F3(0, 1, 0))
        let alongY = Quat(angle: .pi / 2, axis: F3(1, 0, 0))
        let ropeA = scene.addCapsule(length: 0.8, radius: 0.04, density: 1,
                                     friction: 0.3, position: .zero,
                                     rotation: alongX)
        let ropeB = scene.addCapsule(length: 0.8, radius: 0.04, density: 1,
                                     friction: 0.3, position: F3(0, 0, 0.055),
                                     rotation: alongY)
        let bag = scene.addBody(size: F3(0.12, 0.12, 0.12), density: 1,
                                friction: 0.5,
                                position: F3(0.30, 0, 0.075))
        let gpu = try GPUSolver(scene: scene)
        gpu.step()
        let contacts = gpu.activeRigidContactPairs().map {
            Set([$0.0, $0.1])
        }
        XCTAssertTrue(contacts.contains(Set([ropeA, ropeB])),
                      "crossing capsule cords must generate contact")
        XCTAssertTrue(contacts.contains(Set([ropeA, bag])),
                      "a shot bag must contact a capsule cord")

        for _ in 0..<240 { gpu.step() }
        if let contact = gpu.debugWorstRigidContactPenetration() {
            XCTAssertLessThan(contact.depth, 0.003,
                              "isolated rigid contacts must separate, not chatter")
        }
        let settled = gpu.bodyStates([ropeA, ropeB, bag])
        XCTAssertTrue(settled.allSatisfy {
            $0.position.x.isFinite && $0.position.y.isFinite && $0.position.z.isFinite
                && length($0.linearVelocity) < 0.1
        })
    }

    func testGaudiLongHorizonConvergence() throws {
        let scene = Demos.gaudiFunicular(segmentsPerCable: 5,
                                         loadScale: 1, slack: 1.12)
        let dynamic = scene.bodies.indices.filter { scene.bodies[$0].isDynamic }
        let masses = dynamic.map { authoredMass(scene.bodies[$0]) }
        let gpu = try GPUSolver(scene: scene)
        var samples: [(ke: Float, peak: Float, error: Float,
                       joint: Float, contact: Float, pairs: Int)] = []
        var sawRopeRope = false
        var sawBagRope = false
        for frame in 0..<2_400 {
            gpu.step()
            if (frame + 1) % 120 == 0 {
                let states = gpu.bodyStates(dynamic)
                let allStates = gpu.bodyStates(Array(scene.bodies.indices))
                var ke: Float = 0
                var peak: Float = 0
                for i in states.indices {
                    let speed = length(states[i].linearVelocity)
                    ke += 0.5 * masses[i] * speed * speed
                    peak = max(peak, speed)
                }
                var jointError: Float = 0
                for joint in scene.joints where joint.stiffnessLin > 0 {
                    let a = joint.bodyA < 0
                        ? joint.rA
                        : allStates[joint.bodyA].position
                            + allStates[joint.bodyA].rotation.act(joint.rA)
                    let b = allStates[joint.bodyB].position
                            + allStates[joint.bodyB].rotation.act(joint.rB)
                    jointError = max(jointError, distance(a, b))
                }
                for pair in gpu.activeRigidContactPairs() {
                    let shapes = (scene.bodies[pair.0].shape,
                                  scene.bodies[pair.1].shape)
                    if shapes.0 == .capsule && shapes.1 == .capsule {
                        sawRopeRope = true
                    }
                    if (shapes.0 == .box && shapes.1 == .capsule)
                        || (shapes.0 == .capsule && shapes.1 == .box) {
                        sawBagRope = true
                    }
                }
                let sample = (ke, peak, gpu.maxConstraintError(), jointError,
                              gpu.debugWorstRigidContactPenetration()?.depth ?? 0,
                              gpu.lastNumPairs)
                samples.append(sample)
            }
        }
        XCTAssertTrue(samples.allSatisfy {
            $0.ke.isFinite && $0.peak.isFinite && $0.error.isFinite
                && $0.joint.isFinite && $0.contact.isFinite
        })
        XCTAssertEqual(gpu.debugBrokenJoints(), 0)
        XCTAssertTrue(sawRopeRope, "the full rig must exercise cord-cord contact")
        XCTAssertTrue(sawBagRope, "the full rig must exercise bag-cord contact")
        XCTAssertTrue(samples.allSatisfy { $0.pairs > 0 })

        let late = samples.suffix(5)
        let final = try XCTUnwrap(samples.last)
        XCTAssertLessThan(final.ke, samples[0].ke * 0.01)
        XCTAssertLessThan(late.map(\.ke).max()!, 0.001)
        XCTAssertLessThan(late.map(\.peak).max()!, 0.08)
        XCTAssertLessThan(late.map(\.error).max()!, 0.004)
        XCTAssertLessThan(late.map(\.joint).max()!, 0.001)
        XCTAssertLessThan(late.map(\.contact).max()!, 0.003)
    }
}
