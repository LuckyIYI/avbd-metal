import XCTest
import simd
@testable import PhysicsAVBD
@testable import SimCore

final class FiniteGraspSpringTests: XCTestCase {
    func testActivatedSpringRetainsItsPhysicalStiffnessUnderGravity() throws {
        var scene=PhysicsScene(name:"finite grasp spring")
        scene.settings.dt=1/240
        scene.settings.iterations=16
        let body=scene.addBody(size:F3(repeating:0.1),density:1000,friction:0.5,position:F3(0,0,1),collisionEnabled:false)
        let slot=scene.addDragSlot()
        let gpu=try GPUSolver(scene:scene)
        gpu.setDrag(jointIndex:slot,body:body,worldTarget:F3(0,0,1),localAnchor:.zero,stiffness:1000)
        for _ in 0..<1200 {try gpu.submitStep()}
        try gpu.synchronize()
        XCTAssertEqual(gpu.bodyPosition(body).z,1-9.81/1000,accuracy:0.002)
        gpu.setJointWorldAnchor(slot,point:F3(0,0,1.2))
        for _ in 0..<2400 {try gpu.submitStep()}
        try gpu.synchronize()
        XCTAssertEqual(gpu.bodyPosition(body).z,1.2-9.81/1000,accuracy:0.002)
    }
    func testGraspCarriesAndReleasesDynamicPayload() throws {
        var scene=PhysicsScene(name:"physical grasp release")
        scene.settings.dt=1/240;scene.settings.iterations=16
        let hand=scene.addBody(size:F3(repeating:0.04),density:0,friction:0.5,position:F3(0,0,1),collisionEnabled:false)
        let payload=scene.addBody(size:F3(repeating:0.04),density:1000,friction:0.5,position:F3(0,0,0.9),collisionEnabled:false)
        let slot=scene.addDragSlot();let gpu=try GPUSolver(scene:scene)
        gpu.setGrasp(jointIndex:slot,parent:hand,body:payload,parentAnchor:F3(0,0,-0.1))
        for i in 0..<480 {
            let u=min(Float(i)/240,1)
            gpu.setDrivenBodyStates([.init(body:hand,position:F3(u*0.2,0,1+u*0.2),rotation:Quat(angle:u * .pi/2,axis:F3(0,1,0)))])
            try gpu.submitStep()
        }
        try gpu.synchronize()
        XCTAssertLessThan(distance(gpu.bodyPosition(payload),F3(0.1,0,1.2)),0.003)
        gpu.setGrasp(jointIndex:slot,parent:hand,body:nil)
        for _ in 0..<120 {try gpu.submitStep()}
        try gpu.synchronize()
        XCTAssertLessThan(gpu.bodyPosition(payload).z,0.3)
    }

}
