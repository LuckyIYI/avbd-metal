import XCTest
import simd
@testable import SimCore
@testable import PhysicsAVBD

final class JointResponseTests: XCTestCase {
    func slide(response: JointResponse?, mass: Float=1, breakForce: Float?=nil, gravity: Float=0, dt: Float=1/120) throws -> GPUSolver {
        var scene=PhysicsScene(name:"passive slide")
        scene.settings.dt=dt;scene.settings.iterations=24;scene.settings.gravity=gravity
        let b=scene.addBody(size:F3(0.1,0.1,0.1),density:0,friction:0,position:F3(0,0,1),mass:mass,diagonalInertia:F3(repeating:0.01),collisionEnabled:false)
        var j=SceneJoint(bodyA:-1,bodyB:b,rA:F3(0,0,1),rB:.zero,stiffnessLin:.infinity,stiffnessAng:.infinity)
        j.prismaticAxis=F3(1,0,0);j.translationLimits = -0.01...1.1
        j.response=response
        if let breakForce {j.breakLoad=JointBreakLoad(force:breakForce)}
        scene.addJoint(j)
        return try GPUSolver(scene:scene)
    }
    func run(_ s:GPUSolver,_ count:Int) throws {
        for _ in 0..<count {try s.submitStep()};try s.synchronize()
        XCTAssertTrue(s.bodyPosition(0).x.isFinite)
    }
    func impulse(_ s:GPUSolver,_ v:Float) {
        s.applyLinearVelocityImpulses([.init(body:0,deltaVelocity:F3(v,0,0))])
    }
    func testCloserReturnsAcrossTimesteps() throws {
        for dt in [Float(1.0/60),Float(1.0/120)] {
            let s=try slide(response:JointResponse(knots:[[0,0],[1,-8]],damping:3,maxEffort:10),dt:dt)
            impulse(s,1);try run(s,Int(3/dt))
            XCTAssertLessThan(abs(s.bodyPosition(0).x),0.02)
        }
    }
    func testDetentResistsSmallPushAndClicksAfterLargerPush() throws {
        let response=JointResponse(knots:[[0,0],[0.25,-2],[0.5,0],[0.75,2],[1,0],[1.1,-2]],damping:0.3,maxEffort:3)
        let low=try slide(response:response);impulse(low,0.35);try run(low,600)
        XCTAssertLessThan(low.bodyPosition(0).x,0.1)
        let high=try slide(response:response);impulse(high,1.4);try run(high,900)
        XCTAssertGreaterThan(high.bodyPosition(0).x,0.9)
        XCTAssertLessThan(high.bodyPosition(0).x,1.05)
    }
    func testMagneticCaptureHasFiniteReach() throws {
        let law=JointResponse(knots:[[0,0],[0.025,-4],[0.1,0],[1,0]],damping:0,maxEffort:4)
        let low=try slide(response:law);impulse(low,0.3);try run(low,90)
        XCTAssertLessThan(low.bodyPosition(0).x,0.08)
        let high=try slide(response:law);impulse(high,1);try run(high,45)
        XCTAssertGreaterThan(high.bodyPosition(0).x,0.15)
        // Outside the capture range, the passive catch applies no force.
        XCTAssertEqual(law.effort(position:0.5,velocity:0),0)
    }
    func testPhysicalBreakLoadAndRepair() throws {
        let weak=try slide(response:nil,breakForce:2,gravity:-9.81)
        try run(weak,60);XCTAssertEqual(weak.brokenJointIndices(),[0])
        let z=weak.bodyPosition(0).z;try run(weak,30);XCTAssertLessThan(weak.bodyPosition(0).z,z)
        weak.repairJoints();XCTAssertTrue(weak.brokenJointIndices().isEmpty)
        let strong=try slide(response:nil,breakForce:40,gravity:-9.81)
        try run(strong,120);XCTAssertTrue(strong.brokenJointIndices().isEmpty)
        XCTAssertEqual(strong.bodyPosition(0).z,1,accuracy:0.005)
    }
    func testHingeResponseUsesTorqueAndTwoBodyReaction() throws {
        var scene=PhysicsScene(name:"two body hinge")
        scene.settings.gravity=0;scene.settings.dt=1/120;scene.settings.iterations=24
        for _ in 0..<2 {_ = scene.addBody(size:F3(0.2,0.2,0.2),density:0,friction:0,position:F3(0,0,1),mass:1,diagonalInertia:F3(repeating:0.05),collisionEnabled:false)}
        var j=SceneJoint(bodyA:0,bodyB:1,rA:.zero,rB:.zero,stiffnessLin:.infinity,stiffnessAng:.infinity,hingeAxis:F3(0,0,1))
        j.response=JointResponse(knots:[[-1,3],[0.5,0],[1,-1]],damping:0.2,maxEffort:3);scene.addJoint(j)
        let s=try GPUSolver(scene:scene);try run(s,480)
        XCTAssertEqual(s.motorStates([0])[0].angle,0.5,accuracy:0.025)
        let a=s.bodyRotation(0),b=s.bodyRotation(1)
        XCTAssertEqual(a.imag.z,-b.imag.z,accuracy:0.02)
    }
    func testResponseValidationAndClamp() throws {
        let valid=JointResponse(knots:[[0,0],[1,-10]],damping:2,maxEffort:3);try valid.validate()
        XCTAssertEqual(valid.effort(position:1,velocity:5),-3)
        XCTAssertThrowsError(try JointResponse(knots:[[0,1],[0,2]],maxEffort:1).validate())
        XCTAssertThrowsError(try JointBreakLoad(force:-1).validate())
    }
}

extension JointResponseTests {
    func testTorqueBreakChannelUsesAnchorTorque() throws {
        for threshold in [Float(0.5),Float(8)] {
            var scene=PhysicsScene(name:"torque break")
            scene.settings.dt=1/120;scene.settings.iterations=24
            let b=scene.addBody(size:F3(0.4,0.1,0.1),density:0,friction:0,position:F3(0.2,0,1),mass:1,diagonalInertia:F3(repeating:0.02),collisionEnabled:false)
            var joint=SceneJoint(bodyA:-1,bodyB:b,rA:F3(0,0,1),rB:F3(-0.2,0,0),stiffnessLin:.infinity,stiffnessAng:.infinity)
            joint.breakLoad=JointBreakLoad(torque:threshold);scene.addJoint(joint)
            let solver=try GPUSolver(scene:scene);try run(solver,120)
            XCTAssertEqual(solver.brokenJointIndices().isEmpty,threshold>2)
        }
    }
    func testSliderResponsePreservesTwoBodyCenterOfMass() throws {
        var scene=PhysicsScene(name:"two body slider");scene.settings.gravity=0;scene.settings.dt=1/120;scene.settings.iterations=24
        for _ in 0..<2 {_ = scene.addBody(size:F3(0.1,0.1,0.1),density:0,friction:0,position:F3(0,0,1),mass:1,diagonalInertia:F3(repeating:0.01),collisionEnabled:false)}
        var j=SceneJoint(bodyA:0,bodyB:1,rA:.zero,rB:.zero,stiffnessLin:.infinity,stiffnessAng:.infinity)
        j.prismaticAxis=F3(1,0,0);j.response=JointResponse(knots:[[-1,13],[0.3,0],[1,-7]],damping:2,maxEffort:15);scene.addJoint(j)
        let solver=try GPUSolver(scene:scene);try run(solver,480)
        XCTAssertEqual(solver.bodyPosition(1).x-solver.bodyPosition(0).x,0.3,accuracy:0.015)
        XCTAssertEqual(solver.bodyPosition(0).x+solver.bodyPosition(1).x,0,accuracy:0.003)
    }

    func testInvalidScalarResponseThrowsInsteadOfTrapping() throws {
        var scene=PhysicsScene(name:"invalid response import")
        let body=scene.addBody(size:F3(repeating:0.1),density:1,friction:0,position:.zero)
        var joint=SceneJoint(bodyA:-1,bodyB:body,rA:.zero,rB:.zero,
            stiffnessLin:.infinity,stiffnessAng:.infinity)
        joint.response=JointResponse(knots:[[0,0],[4.7,-1]],maxEffort:2)
        scene.addJoint(joint)
        XCTAssertThrowsError(try GPUSolver(scene:scene))
        scene.joints[0].hingeAxis=F3(0,0,1)
        XCTAssertThrowsError(try GPUSolver(scene:scene))
    }
}
