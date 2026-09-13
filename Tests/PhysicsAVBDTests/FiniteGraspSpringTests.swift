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

    func testHelicalJointBackdrivesUnderAxialForceAcrossMultipleTurns() throws {
        var scene=PhysicsScene(name:"backdrivable screw")
        scene.settings.gravity=0;scene.settings.dt=1/480;scene.settings.iterations=32
        let base=scene.addBody(size:F3(repeating:0.04),density:0,friction:0.5,position:.zero,collisionEnabled:false)
        let nut=scene.addBody(size:F3(repeating:0.02),density:1000,friction:0.5,position:F3(0,0,0.1),collisionEnabled:false)
        let screw=scene.addDragSlot(),pull=scene.addDragSlot();let gpu=try GPUSolver(scene:scene)
        gpu.setHelicalJoint(jointIndex:screw,parent:base,body:nut,parentAnchor:F3(0,0,0.1),pitch:0.02)
        gpu.setDrag(jointIndex:pull,body:nut,worldTarget:F3(0,0,0.14),localAnchor:.zero,stiffness:1000)
        for _ in 0..<2400 {try gpu.submitStep()}
        try gpu.synchronize()
        let travel=gpu.bodyPosition(nut).z-0.1,angle=gpu.helicalAngle(screw)
        XCTAssertGreaterThan(angle,3 * .pi)
        XCTAssertEqual(travel,0.04,accuracy:0.001)
        XCTAssertEqual(travel,angle * 0.02/(2 * .pi),accuracy:0.0003)
    }

    func testGraspTransmitsPureMultiTurnSpin() throws {
        var scene=PhysicsScene(name:"multi turn rigid grasp")
        scene.settings.gravity=0;scene.settings.dt=1/480;scene.settings.iterations=32
        let hand=scene.addBody(size:F3(repeating:0.04),density:0,friction:0.5,position:.zero,collisionEnabled:false)
        let load=scene.addBody(size:F3(repeating:0.1),density:0,friction:0.5,position:F3(0,0,-0.05),mass:0.64,diagonalInertia:F3(0.0025,0.0025,0.0023),collisionEnabled:false)
        let slot=scene.addDragSlot();let gpu=try GPUSolver(scene:scene)
        gpu.setGrasp(jointIndex:slot,parent:hand,body:load,parentAnchor:F3(0,0,-0.05),linearStiffness:100000,angularStiffness:500000)
        var previous:Float=0,total:Float=0
        for i in 0..<960 {
            let angle=Float(i)/960 * 4 * Float.pi
            gpu.setDrivenBodyStates([.init(body:hand,position:.zero,rotation:Quat(angle:angle,axis:F3(0,0,1)))])
            try gpu.submitStep();try gpu.synchronize()
            let q=gpu.bodyRotation(load);let a=2*atan2(q.imag.z,q.real)
            total+=atan2(sin(a-previous),cos(a-previous));previous=a
        }
        XCTAssertEqual(total,4 * .pi,accuracy:0.04)
    }

}
