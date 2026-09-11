import XCTest
import simd
@testable import SimCore
@testable import PhysicsAVBD

final class HingeStopComplianceTests: XCTestCase {
    func testHistoricalStopStiffnessDefault() {
        let joint=SceneJoint(bodyA:-1,bodyB:0,rA:.zero,rB:.zero)
        XCTAssertEqual(joint.limitStiffness,40000)
    }
    private func deflection(_ stiffness:Float) throws -> Float {
        var scene=PhysicsScene(name:"small lid stop")
        scene.settings.dt=1/120;scene.settings.iterations=24
        let lid=scene.addBody(size:F3(0.04,0.04,0.004),density:0,friction:0,
            position:F3(0,-0.02,0.1),mass:0.006,
            diagonalInertia:F3(0.00000081,0.00000081,0.0000016),collisionEnabled:false)
        var joint=SceneJoint(bodyA:-1,bodyB:lid,rA:F3(0,0,0.1),rB:F3(0,0.02,0),
            stiffnessLin:.infinity,stiffnessAng:.infinity,
            hingeAxis:F3(-1,0,0),limitLo:0,limitHi:1.85)
        joint.limitStiffness=stiffness;scene.addJoint(joint)
        let solver=try GPUSolver(scene:scene)
        for _ in 0..<360 {try solver.submitStep()};try solver.synchronize()
        let q=solver.bodyRotation(lid),p=solver.bodyPosition(lid)
        XCTAssertTrue(p.x.isFinite && p.y.isFinite && p.z.isFinite)
        XCTAssertLessThan(length(p+q.act(F3(0,0.02,0))-F3(0,0,0.1)),0.003)
        return abs(2*atan2(q.imag.x,q.real))
    }
    func testAuthoredComplianceChangesPhysicalStopDeflection() throws {
        let stiff=try deflection(2),soft=try deflection(0.02)
        XCTAssertLessThan(stiff,0.01)
        XCTAssertGreaterThan(soft,stiff*3)
        XCTAssertLessThan(soft,0.15)
    }
}
