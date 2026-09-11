import XCTest
import Metal
import SimCore
import simd
@testable import PhysicsAVBD

final class RigidSleepingTests: XCTestCase {
    private func make() throws -> (GPUSolver, Int, Int) {
        guard MTLCreateSystemDefaultDevice() != nil else {throw XCTSkip("Metal required")}
        var scene=PhysicsScene(name:"rigid-sleep")
        scene.settings.gravity=0
        let a=scene.addBody(size:F3(repeating:0.2),density:1,friction:0.5,position:F3(0,0,1))
        let b=scene.addBody(size:F3(repeating:0.2),density:1,friction:0.5,position:F3(3,0,1))
        return (try GPUSolver(scene:scene),a,b)
    }
    func testQuietBodiesSleepAndImpulseWakesOnlyAffectedIsland() throws {
        let (s,a,b)=try make()
        var c=RigidSleepSettings();c.quietTime=0.1
        try s.configureRigidSleeping(c)
        for _ in 0..<60 {try s.submitStep();try s.synchronize()}
        XCTAssertEqual(s.sleepingRigidBodyCount,2)
        XCTAssertGreaterThan(s.sleepingSkippedFrameCount,0)
        let before=s.bodyPosition(a)
        s.applyLinearVelocityImpulses([.init(body:a,deltaVelocity:F3(0.1,0,0))])
        XCTAssertEqual(s.sleepingRigidBodyCount,1)
        try s.submitStep();try s.synchronize()
        XCTAssertGreaterThan(s.bodyPosition(a).x,before.x)
        XCTAssertEqual(s.bodyPosition(b),F3(3,0,1))
        try s.configureRigidSleeping(nil)
        XCTAssertEqual(s.sleepingRigidBodyCount,0)
    }
    func testExplicitGroupAndPoseWake() throws {
        let (s,a,b)=try make()
        var c=RigidSleepSettings();c.quietTime=0.1
        try s.configureRigidSleeping(c,groups:[[a,b]])
        for _ in 0..<60 {try s.submitStep();try s.synchronize()}
        XCTAssertEqual(s.sleepingRigidBodyCount,2)
        s.applyLinearVelocityImpulses([.init(body:a,deltaVelocity:F3(0.1,0,0))])
        XCTAssertEqual(s.sleepingRigidBodyCount,0)
        s.setBodyPose(a,position:F3(0,0,2),rotation:Quat(real:1,imag:.zero))
        try s.submitStep();try s.synchronize()
        XCTAssertGreaterThan(s.bodyPosition(a).z,1.9)
    }
    func testGravityChangeWakesAndFallingBodyDoesNotSleep() throws {
        let (s,_,_)=try make()
        var c=RigidSleepSettings();c.quietTime=0.1
        try s.configureRigidSleeping(c)
        for _ in 0..<60 {try s.submitStep();try s.synchronize()}
        s.settings.gravity = -9.81
        XCTAssertEqual(s.sleepingRigidBodyCount,0)
        for _ in 0..<20 {try s.submitStep();try s.synchronize()}
        XCTAssertEqual(s.sleepingRigidBodyCount,0)
    }
    func testInvalidSettingsRejected() throws {
        let (s,_,_)=try make()
        var c=RigidSleepSettings();c.quietTime = -1
        XCTAssertThrowsError(try s.configureRigidSleeping(c))
    }
    func testRestingBodySleepsAndMovingSupportWakesIt() throws {
        var scene=PhysicsScene(name:"sleep-support")
        let floor=scene.addBody(size:F3(5,5,0.2),density:0,friction:0.7,position:F3(0,0,-0.1))
        let box=scene.addBody(size:F3(repeating:0.2),density:100,friction:0.7,position:F3(0,0,0.11))
        let s=try GPUSolver(scene:scene)
        try s.configureRigidSleeping(RigidSleepSettings())
        for _ in 0..<300 {try s.submitStep();try s.synchronize()}
        XCTAssertEqual(s.sleepingRigidBodyCount,1)
        let z=s.bodyPosition(box).z
        s.setBodyPose(floor,position:F3(10,0,-0.1),rotation:Quat(real:1,imag:.zero))
        XCTAssertEqual(s.sleepingRigidBodyCount,0)
        for _ in 0..<20 {try s.submitStep();try s.synchronize()}
        XCTAssertLessThan(s.bodyPosition(box).z,z-0.01)
    }
    func testApproachingAwakeBodyWakesSleeper() throws {
        let (s,a,b)=try make()
        var c=RigidSleepSettings();c.quietTime=0.1
        try s.configureRigidSleeping(c)
        for _ in 0..<60 {try s.submitStep();try s.synchronize()}
        s.applyLinearVelocityImpulses([.init(body:b,deltaVelocity:F3(-2,0,0))])
        XCTAssertEqual(s.sleepingRigidBodyCount,1)
        for _ in 0..<100 {try s.submitStep();try s.synchronize()}
        XCTAssertGreaterThan(length(s.bodyPosition(a)-F3(0,0,1)),0.01)
    }
    func testShortestQuietWindowDoesNotFreezeInitialFreefall() throws {
        let (s,_,_)=try make()
        s.settings.gravity = -9.81
        var c=RigidSleepSettings();c.quietTime=s.settings.dt
        try s.configureRigidSleeping(c)
        for _ in 0..<3 {try s.submitStep();try s.synchronize()}
        XCTAssertEqual(s.sleepingRigidBodyCount,0)
        s.settings.gravity=0
        for _ in 0..<5 {try s.submitStep();try s.synchronize()}
    }
}
