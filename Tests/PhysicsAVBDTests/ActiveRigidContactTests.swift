import XCTest
import Metal
import SimCore
import simd
@testable import PhysicsAVBD

final class ActiveRigidContactTests: XCTestCase {
    func testSparseContactScheduleMatchesFullSchedule() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {throw XCTSkip("Metal required")}
        var scene=PhysicsScene(name:"sparse-active-contact-parity")
        scene.settings.iterations=6
        scene.addBody(size:F3(80,4,0.2),density:0,friction:0.6,position:F3(0,0,-0.1))
        for index in 0..<260 {
            scene.addBody(size:F3(repeating:0.12),density:100,friction:0.6,
                position:F3(Float(index%130)*0.5-32,Float(index/130)*0.5,0.061))
        }
        let reference=try GPUSolver(scene:scene)
        let compact=try GPUSolver(scene:scene)
        reference.activeRigidContactCompactionEnabled=false
        XCTAssertTrue(compact.usesActiveRigidContacts)
        for _ in 0..<15 {
            try reference.submitStep();try reference.synchronize()
            try compact.submitStep();try compact.synchronize()
        }
        XCTAssertEqual(reference.lastNumPairs,compact.lastNumPairs)
        let list=compact.activeRigidManifolds.contents().assumingMemoryBound(to:UInt32.self)
        XCTAssertGreaterThan(list[0],0)
        XCTAssertEqual(Set((0..<Int(list[0])).map{list[$0+1]}).count,Int(list[0]))
        for i in 0..<scene.bodies.count {
            XCTAssertLessThan(length(reference.bodyPosition(i)-compact.bodyPosition(i)),1e-5)
            XCTAssertLessThan(length(reference.bodyVelocity(i)-compact.bodyVelocity(i)),1e-4)
        }

        let before=Set(compact.activeRigidContactPairs().map { SIMD2($0.0,$0.1) })
        let checkpoint=compact.captureSimulationSnapshot()
        compact.setBodyPose(1,position:F3(100,0,1),rotation:Quat(real:1,imag:.zero))
        try compact.submitStep();try compact.synchronize()
        XCTAssertLessThan(compact.activeRigidContactPairs().count,before.count)
        compact.restoreSimulationSnapshot(checkpoint)
        XCTAssertEqual(Set(compact.activeRigidContactPairs().map { SIMD2($0.0,$0.1) }),before,
            "restored contact queries must not use a future frame's compact index list")
    }
}
