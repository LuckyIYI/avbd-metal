import XCTest
import Metal
import SimCore
import simd
@testable import PhysicsAVBD

final class ConvexScaleMismatchTests: XCTestCase {
    // Captured from a rigid kitchen replay: a 17 mm hull beside a 12 m floor.
    // Unmodified main latches ConvexQuery on the first step of this pair.
    func testCapturedSmallHullAboveLargeFloor() throws {
        try checkPair(gap: nil, reverse: false, offset: .zero, contact: false)
    }

    func testCapturedNearFloorPair() throws {
        try checkPair(gap: nil, reverse: false, offset: .zero, contact: false, fixture: 1)
        try checkPair(gap: nil, reverse: true, offset: .zero, contact: false, fixture: 1)
    }

    func testSeparatedPairIsStableAcrossOrderingAndTranslation() throws {
        for reverse in [false, true] {
            for offset in [F3.zero, F3(20, -10, 3)] {
                try checkPair(gap: nil, reverse: reverse, offset: offset, contact: false)
            }
        }
    }

    func testContactBandAndPenetrationAreNotCulled() throws {
        for reverse in [false, true] {
            for gap: Float in [0.0005, 0, -0.0005] {
                try checkPair(gap: gap, reverse: reverse, offset: .zero, contact: true)
            }
        }
    }

    func testApproachingPairKeepsSpeculativeContact() throws {
        try checkPair(gap: 0.01, reverse: false, offset: .zero,
            contact: true, velocity: F3(0, 0, -0.8))
    }

    private func checkPair(gap: Float?, reverse: Bool, offset: F3,
                           contact: Bool, velocity: F3 = .zero, fixture: Int = 0) throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal required") }
        var vertices = [F3(-0.0074652088806033134,0.005854279734194279,0.0010771384695544839),
            F3(-0.005832132417708635,0.00679713673889637,-0.0010771384695544839),
            F3(-0.004977664444595575,0.007290464360266924,0.0010771384695544839),
            F3(-0.003344587981700897,0.008233321830630302,-0.0010771384695544839),
            F3(0.0037226618733257055,-0.008726038038730621,0.0010771384695544839),
            F3(0.005056063178926706,-0.007392636500298977,-0.0010771384695544839),
            F3(0.005753733683377504,-0.006694965995848179,0.0010771384695544839),
            F3(0.007087135221809149,-0.005361564923077822,-0.0010771384695544839)]

        var orientation = Quat(vector: SIMD4<Float>(
            0.14539207518100739, 0.08989617228507996,
            -0.038406576961278915, 0.9845327734947205))
        var position = F3(-0.05671069025993347, -5.149870872497559, 0.028766755014657974)
        if fixture == 1 {
            vertices = [F3(-0.0034568565897643566,-0.008186818100512028,-0.0010771384695544839),
                F3(-0.001635396503843367,-0.008674876764416695,0.0010771384695544839),
                F3(-0.0012201626086607575,0.008802560158073902,-0.0010771384695544839),
                F3(-0.0006823610165156424,-0.008930242620408535,-0.0010771384695544839),
                F3(0.0006655517499893904,0.008802560158073902,0.0010771384695544839),
                F3(0.001139099127613008,-0.009418301284313202,0.0010771384695544839),
                F3(0.0016522066434845328,0.008802560158073902,-0.0010771384695544839),
                F3(0.0035379210021346807,0.008802560158073902,0.0010771384695544839)]
            orientation = Quat(vector: SIMD4<Float>(0.0071857888251543045,-0.0037543768994510174,-0.05527998507022858,0.9984378814697266))
            position = F3(0.06335224211215973,-5.131350517272949,-0.0004442156059667468)
        }
        if let gap {
            let bottom = vertices.map { orientation.act($0).z }.min()!
            position.z = -0.005 - bottom + gap
        }
        var scene = PhysicsScene(name: "small-hull-large-floor")
        scene.settings.gravity = 0
        scene.settings.collisionMargin = 0.001
        func addFloor() {
            _ = scene.addBody(size: F3(12, 11, 0.14), density: 0,
                friction: 0.5, position: F3(0, 0, -0.075) + offset)
        }
        if reverse { addFloor() }
        let hull = scene.addBody(size: F3(repeating: 0.02), density: 1,
            friction: 0.5, position: position + offset, rotation: orientation,
            collisionEnabled: false)
        _ = scene.addConvexCollider(body: hull, vertices: vertices)
        if !reverse { addFloor() }
        let solver = try GPUSolver(scene: scene)
        if velocity != .zero {
            solver.applyLinearVelocityImpulses([.init(body: hull, deltaVelocity: velocity)])
        }
        try solver.submitStep()
        try solver.synchronize()
        XCTAssertNil(solver.runtimeFailure)
        XCTAssertGreaterThan(solver.lastNumPairs, 0)
        XCTAssertEqual(!solver.activeRigidContactPairs().isEmpty, contact,
            "Support-plane recovery must preserve the contact band and actual overlap")
    }
}
