import XCTest
import simd
@testable import PhysicsAVBD
@testable import SimCore

final class ParentFrameJointTargetTests: XCTestCase {
    func testActuatorTargetStaysInRotatedParentFrame() throws {
        var scene = PhysicsScene(name: "parent-frame actuator")
        scene.settings.gravity = 0
        let parent = scene.addBody(size: F3(repeating: 0.1), density: 0, friction: 0.5,
            position: F3(1, 0, 0), rotation: Quat(angle: .pi / 2, axis: F3(0, 0, 1)),
            collisionEnabled: false)
        let child = scene.addBody(size: F3(repeating: 0.1), density: 100, friction: 0.5,
            position: F3(1, 0.1, 0), collisionEnabled: false)
        scene.addJoint(SceneJoint(bodyA: parent, bodyB: child,
            rA: F3(0.1, 0, 0), rB: .zero, stiffnessLin: 1500, stiffnessAng: 0))
        let gpu = try GPUSolver(scene: scene)
        gpu.setJointParentAnchors([.init(joint: 0, point: F3(0.2, 0, 0))])
        for _ in 0..<120 { try gpu.submitStep() }
        try gpu.synchronize()
        XCTAssertLessThan(distance(gpu.bodyPosition(child), F3(1, 0.2, 0)), 0.002)
        // Moving the parent does not require another target update.
        gpu.setBodyPose(parent, position: F3(2, 0, 0), rotation: Quat(angle: 0, axis: F3(0, 0, 1)))
        for _ in 0..<120 { try gpu.submitStep() }
        try gpu.synchronize()
        XCTAssertLessThan(distance(gpu.bodyPosition(child), F3(2.2, 0, 0)), 0.002)
    }
}
