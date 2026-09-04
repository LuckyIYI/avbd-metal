import XCTest
import simd
@testable import SimCore
@testable import PhysicsAVBD

final class PrismaticJointTests:XCTestCase {
    private func scene(bounded:Bool=true,rotated:Bool=false)->PhysicsScene {
        var s=PhysicsScene(name:"prismatic rail")
        s.settings.dt=1/120; s.settings.iterations=16
        s.settings.betaLin=20000; s.settings.betaAng=500; s.settings.lambdaMax=1200
        let q=Quat(angle:rotated ? .pi/4 : 0,axis:F3(0,0,1)),axis=q.act(F3(1,0,0))
        let base=s.addBody(size:F3(repeating:0.1),density:0,friction:0.5,position:F3(0,0,1),rotation:q,collisionEnabled:false)
        let b=s.addBody(size:F3(repeating:0.1),density:1000,friction:0.5,position:F3(0,0,1),rotation:q,velocity:axis*0.4,collisionEnabled:false)
        var j=SceneJoint(bodyA:base,bodyB:b,rA:.zero,rB:.zero,stiffnessLin:.infinity,stiffnessAng:.infinity)
        j.prismaticAxis=F3(1,0,0)
        if bounded {j.translationLimits = -0.1...0.2}
        s.addJoint(j); return s
    }
    func testGPUFreeTranslationRetainsTransverseSupport() throws {
        let s=scene(bounded:false,rotated:true),gpu=try GPUSolver(scene:s)
        for _ in 0..<120 { try gpu.submitStep() }
        try gpu.synchronize()
        let p=gpu.bodyPosition(1),axis=normalize(F3(1,1,0))
        XCTAssertGreaterThan(dot(p-F3(0,0,1),axis),0.32)
        XCTAssertLessThan(abs(p.x-p.y),0.003)
        XCTAssertEqual(p.z,1,accuracy:0.008)
    }
    func testGPUTravelStopsAndReleaseDoNotLeaveAxialWarmstart() throws {
        let s=scene(),gpu=try GPUSolver(scene:s)
        for _ in 0..<240 {try gpu.submitStep()};try gpu.synchronize()
        XCTAssertLessThanOrEqual(gpu.bodyPosition(1).x,0.21)
        XCTAssertGreaterThan(gpu.bodyPosition(1).x,0.17)
        gpu.setBodyStates([.init(body:1,position:F3(0.15,0,1),rotation:Quat(real:1,imag:.zero),linearVelocity:F3(-0.25,0,0),angularVelocity:.zero)])
        for _ in 0..<90 {try gpu.submitStep()};try gpu.synchronize()
        XCTAssertLessThan(gpu.bodyPosition(1).x,0.02)
        XCTAssertGreaterThan(gpu.bodyPosition(1).x,-0.11)
        XCTAssertEqual(gpu.bodyPosition(1).z,1,accuracy:0.008)
    }
    func testCPURotatedRailAndTravelStops() throws {
        let s=scene(rotated:true),cpu=try s.makeCPUSolverChecked()
        for _ in 0..<240 {cpu.step()}
        let p=cpu.bodies[1].positionLin,axis=normalize(F3(1,1,0))
        XCTAssertLessThan(dot(p-F3(0,0,1),axis),0.212)
        XCTAssertGreaterThan(dot(p-F3(0,0,1),axis),0.17)
        XCTAssertEqual(p.z,1,accuracy:0.012)
        XCTAssertEqual(p.x,p.y,accuracy:0.003)
    }
}
