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

    func testPickingSleepingBodyDoesNotWakeUnrelatedIslands() throws {
        let (solver,a,_) = try make()
        var settings=RigidSleepSettings(); settings.quietTime=0.1
        try solver.configureRigidSleeping(settings)
        for _ in 0..<30 {try solver.submitStep();try solver.synchronize()}
        XCTAssertEqual(solver.sleepingRigidBodyCount,2)
        let hit=solver.pick(origin:F3(0,-2,1),dir:F3(0,1,0))
        XCTAssertEqual(hit?.body,a)
        XCTAssertEqual(solver.sleepingRigidBodyCount,2,
            "Picking is a read-only query; interaction wakes the selected island")
    }
    func testLongSeparatedBodiesDoNotWakeFromBoundingSphereOverlap() throws {
        var scene=PhysicsScene(name:"tight-sleep-bounds")
        scene.settings.gravity=0
        scene.addBody(size:F3(4,0.2,0.2),density:1,friction:0.5,position:F3(0,0,1))
        let moving=scene.addBody(size:F3(4,0.2,0.2),density:1,friction:0.5,position:F3(0,1,1))
        let solver=try GPUSolver(scene:scene)
        var settings=RigidSleepSettings();settings.quietTime=0.1
        try solver.configureRigidSleeping(settings)
        for _ in 0..<30 {try solver.submitStep();try solver.synchronize()}
        XCTAssertEqual(solver.sleepingRigidBodyCount,2)
        solver.applyLinearVelocityImpulses([.init(body:moving,deltaVelocity:F3(0.1,0,0))])
        for _ in 0..<10 {try solver.submitStep();try solver.synchronize()}
        XCTAssertEqual(solver.sleepingRigidBodyCount,1)
        XCTAssertEqual(solver.rigidIslandStatistics["islands"],2)
    }
    func testUnchangedJointIslandReusesConnectivity() throws {
        let (solver,a,b)=try make()
        var settings=RigidSleepSettings();settings.quietTime=0.1
        try solver.configureRigidSleeping(settings,groups:[[a,b]])
        for _ in 0..<30 {try solver.submitStep();try solver.synchronize()}
        XCTAssertEqual(solver.rigidIslandStatistics["islands"],1)
        XCTAssertEqual(solver.rigidIslandStatistics["graph_rebuilds"],1)
        solver.applyLinearVelocityImpulses([.init(body:a,deltaVelocity:F3(0.1,0,0))])
        XCTAssertEqual(solver.sleepingRigidBodyCount,0)
    }

    func testEnergySleepUsesRotationalInertia() throws {
        var scene=PhysicsScene(name:"sleep-energy-scale")
        scene.settings.gravity=0
        let small=scene.addBody(size:F3(repeating:0.02),density:1,friction:0,position:F3(0,0,1))
        let large=scene.addBody(size:F3(repeating:2),density:1,friction:0,position:F3(4,0,1))
        let solver=try GPUSolver(scene:scene)
        solver.setBodyStates([small,large].map { .init(body:$0,position:scene.bodies[$0].position,
            rotation:Quat(real:1,imag:.zero),angularVelocity:F3(0,0,0.05)) })
        var settings=RigidSleepSettings();settings.quietTime=0.1;settings.energyThreshold=0.00005
        try solver.configureRigidSleeping(settings)
        for _ in 0..<30 {try solver.submitStep();try solver.synchronize()}
        XCTAssertEqual(solver.sleepingRigidBodyCount,1)
        XCTAssertEqual(solver.bodyAngularVelocity(small),.zero)
        XCTAssertGreaterThan(solver.bodyMass(small),0,"Sleeping preserves the public physical mass")
        XCTAssertGreaterThan(length(solver.bodyAngularVelocity(large)),0.04)
        settings.energyThreshold = -1
        XCTAssertThrowsError(try solver.configureRigidSleeping(settings))
    }

    func testOverlappingIndependentCollisionDomainsDoNotWakeEachOther() throws {
        var scene=PhysicsScene(name:"sleep-domains");scene.settings.gravity=0
        let a=scene.addBody(size:F3(repeating:0.2),density:1,friction:0,position:F3(0,0,1))
        let b=scene.addBody(size:F3(repeating:0.2),density:1,friction:0,position:F3(0,0,1))
        scene.colliders[0].collisionGroup=1;scene.colliders[1].collisionGroup=2
        let solver=try GPUSolver(scene:scene)
        var settings=RigidSleepSettings();settings.quietTime=0.1
        try solver.configureRigidSleeping(settings)
        for _ in 0..<30 {try solver.submitStep();try solver.synchronize()}
        solver.applyLinearVelocityImpulses([.init(body:b,deltaVelocity:F3(0.1,0,0))])
        for _ in 0..<5 {try solver.submitStep();try solver.synchronize()}
        XCTAssertEqual(solver.sleepingRigidBodyCount,1)
        XCTAssertEqual(solver.bodyPosition(a),F3(0,0,1))
    }
    func testBrokenJointAllowsRetainedIslandToSplitAfterWake() throws {
        var scene=PhysicsScene(name:"sleep-split");scene.settings.gravity=0
        let a=scene.addBody(size:F3(repeating:0.2),density:1,friction:0,position:F3(0,0,1))
        let b=scene.addBody(size:F3(repeating:0.2),density:1,friction:0,position:F3(3,0,1))
        scene.addJoint(SceneJoint(bodyA:a,bodyB:b,rA:F3(3,0,0),rB:.zero,
            stiffnessLin:.infinity,stiffnessAng:.infinity))
        let solver=try GPUSolver(scene:scene)
        var settings=RigidSleepSettings();settings.quietTime=0.1
        try solver.configureRigidSleeping(settings)
        for _ in 0..<30 {try solver.submitStep();try solver.synchronize()}
        XCTAssertEqual(solver.rigidIslandStatistics["islands"],1)
        solver.wakeRigidBodies([a])
        // Exercise topology invalidation independently of the load law,
        // whose physical break thresholds have separate integration tests.
        solver.joints.contents().assumingMemoryBound(to:JointGPU.self)[0].header.z=1
        for _ in 0..<30 {try solver.submitStep();try solver.synchronize()}
        XCTAssertEqual(solver.rigidIslandStatistics["islands"],2)
        solver.applyLinearVelocityImpulses([.init(body:a,deltaVelocity:F3(0.1,0,0))])
        XCTAssertEqual(solver.sleepingRigidBodyCount,1)
    }
}
