import XCTest
import simd
@testable import PhysicsAVBD
@testable import SimCore

final class ThreadFeatureContactTests: XCTestCase {
    func testAngularImpulsePreservesPoseAndIntegratesOnNextStep() throws {
        var scene=PhysicsScene(name:"angular impulse")
        scene.settings.gravity=0;scene.settings.dt=1/240
        let body=scene.addBody(size:F3(repeating:0.05),density:1,friction:0,position:F3(0,0,1),mass:0.08,diagonalInertia:F3(repeating:4e-5),collisionEnabled:false)
        let solver=try GPUSolver(scene:scene)
        let p=solver.bodyPosition(body),q=solver.bodyRotation(body)
        solver.applyAngularVelocityImpulses([(body:body,deltaVelocity:F3(0,0,1))])
        XCTAssertEqual(solver.bodyPosition(body),p)
        XCTAssertEqual(solver.bodyRotation(body).vector,q.vector)
        XCTAssertEqual(solver.bodyAngularVelocity(body).z,1,accuracy:1e-6)
        try solver.submitStep();try solver.synchronize()
        XCTAssertGreaterThan(solver.bodyRotation(body).imag.z,0)
        XCTAssertEqual(solver.bodyAngularVelocity(body).z,1,accuracy:1e-4)
    }

    func testCompoundBoundaryFilterRetainsSupportAndOpenGap() throws {
        let block=try JSONDecoder().decode(ConvexHullAsset.self,from:Data(#"{"boundingRadius":0.017320508137345314,"boundsMax":[0.009999999776482582,0.009999999776482582,0.009999999776482582],"boundsMin":[-0.009999999776482582,-0.009999999776482582,-0.009999999776482582],"centroid":[0.0,0.0,0.0],"digest":"7cebfc52599f0faf5cf322276f2f3cfd8fd7948fca8a20ca4f3885fef008eb19","edges":[{"faceA":0,"faceB":4,"vertexA":0,"vertexB":1},{"faceA":1,"faceB":2,"vertexA":0,"vertexB":2},{"faceA":0,"faceB":2,"vertexA":0,"vertexB":3},{"faceA":3,"faceB":5,"vertexA":0,"vertexB":4},{"faceA":3,"faceB":4,"vertexA":0,"vertexB":5},{"faceA":1,"faceB":5,"vertexA":0,"vertexB":6},{"faceA":0,"faceB":6,"vertexA":1,"vertexB":3},{"faceA":4,"faceB":6,"vertexA":1,"vertexB":5},{"faceA":2,"faceB":7,"vertexA":2,"vertexB":3},{"faceA":1,"faceB":7,"vertexA":2,"vertexB":6},{"faceA":6,"faceB":8,"vertexA":3,"vertexB":5},{"faceA":7,"faceB":9,"vertexA":3,"vertexB":6},{"faceA":8,"faceB":9,"vertexA":3,"vertexB":7},{"faceA":3,"faceB":10,"vertexA":4,"vertexB":5},{"faceA":5,"faceB":10,"vertexA":4,"vertexB":6},{"faceA":10,"faceB":11,"vertexA":5,"vertexB":6},{"faceA":8,"faceB":11,"vertexA":5,"vertexB":7},{"faceA":9,"faceB":11,"vertexA":6,"vertexB":7}],"stableID":"hull-7cebfc52599f0faf","triangles":[[0,1,3],[0,2,6],[0,3,2],[0,4,5],[0,5,1],[0,6,4],[1,5,3],[2,3,6],[3,5,7],[3,7,6],[4,6,5],[5,6,7]],"vertices":[[-0.009999999776482582,-0.009999999776482582,-0.009999999776482582],[-0.009999999776482582,-0.009999999776482582,0.009999999776482582],[-0.009999999776482582,0.009999999776482582,-0.009999999776482582],[-0.009999999776482582,0.009999999776482582,0.009999999776482582],[0.009999999776482582,-0.009999999776482582,-0.009999999776482582],[0.009999999776482582,-0.009999999776482582,0.009999999776482582],[0.009999999776482582,0.009999999776482582,-0.009999999776482582],[0.009999999776482582,0.009999999776482582,0.009999999776482582]],"volume":7.99999907030724e-06}"#.utf8))
        let small=try JSONDecoder().decode(ConvexHullAsset.self,from:Data(#"{"boundingRadius":0.0069282036274671555,"boundsMax":[0.004000000189989805,0.004000000189989805,0.004000000189989805],"boundsMin":[-0.004000000189989805,-0.004000000189989805,-0.004000000189989805],"centroid":[6.731611980827878e-20,-3.365805990413939e-20,-5.048709147179622e-20],"digest":"04b604c997e65b2f5c8df9adb6befe5bfab488dc1f9ba219a48b5666c351c812","edges":[{"faceA":0,"faceB":4,"vertexA":0,"vertexB":1},{"faceA":1,"faceB":2,"vertexA":0,"vertexB":2},{"faceA":0,"faceB":2,"vertexA":0,"vertexB":3},{"faceA":3,"faceB":5,"vertexA":0,"vertexB":4},{"faceA":3,"faceB":4,"vertexA":0,"vertexB":5},{"faceA":1,"faceB":5,"vertexA":0,"vertexB":6},{"faceA":0,"faceB":6,"vertexA":1,"vertexB":3},{"faceA":4,"faceB":6,"vertexA":1,"vertexB":5},{"faceA":2,"faceB":7,"vertexA":2,"vertexB":3},{"faceA":1,"faceB":7,"vertexA":2,"vertexB":6},{"faceA":6,"faceB":8,"vertexA":3,"vertexB":5},{"faceA":7,"faceB":9,"vertexA":3,"vertexB":6},{"faceA":8,"faceB":9,"vertexA":3,"vertexB":7},{"faceA":3,"faceB":10,"vertexA":4,"vertexB":5},{"faceA":5,"faceB":10,"vertexA":4,"vertexB":6},{"faceA":10,"faceB":11,"vertexA":5,"vertexB":6},{"faceA":8,"faceB":11,"vertexA":5,"vertexB":7},{"faceA":9,"faceB":11,"vertexA":6,"vertexB":7}],"stableID":"hull-04b604c997e65b2f","triangles":[[0,1,3],[0,2,6],[0,3,2],[0,4,5],[0,5,1],[0,6,4],[1,5,3],[2,3,6],[3,5,7],[3,7,6],[4,6,5],[5,6,7]],"vertices":[[-0.004000000189989805,-0.004000000189989805,-0.004000000189989805],[-0.004000000189989805,-0.004000000189989805,0.004000000189989805],[-0.004000000189989805,0.004000000189989805,-0.004000000189989805],[-0.004000000189989805,0.004000000189989805,0.004000000189989805],[0.004000000189989805,-0.004000000189989805,-0.004000000189989805],[0.004000000189989805,-0.004000000189989805,0.004000000189989805],[0.004000000189989805,0.004000000189989805,-0.004000000189989805],[0.004000000189989805,0.004000000189989805,0.004000000189989805]],"volume":5.120000992064888e-07}"#.utf8))
        for gap:Float in [0,0.02] {
            var scene=PhysicsScene(name:"compound boundary cavity")
            scene.settings.gravity = -9.81;scene.settings.dt=1/240
            scene.settings.iterations=16;scene.settings.collisionMargin=1e-5
            let base=scene.addBody(size:F3(repeating:0.06),density:0,friction:0,position:.zero,collisionEnabled:false)
            for sign:Float in [-1,1] {scene.addConvexCollider(body:base,asset:block,localPosition:F3(sign*(0.01+gap/2),0,0))}
            let probe=scene.addBody(size:F3(repeating:0.008),density:1,friction:0,position:F3(0,0,0.03),mass:0.01,diagonalInertia:F3(repeating:1e-7),collisionEnabled:false)
            scene.addConvexCollider(body:probe,asset:small)
            let solver=try GPUSolver(scene:scene)
            try solver.enableCompoundBoundaryFiltering(probeDistance:1e-6)
            for _ in 0..<120 {try solver.submitStep();try solver.synchronize()}
            if gap==0 {XCTAssertEqual(solver.bodyPosition(probe).z,0.014,accuracy:0.0002)}
            else {XCTAssertLessThan(solver.bodyPosition(probe).z,-0.1,"A real opening must remain passable")}
        }
    }

    // Captured disjoint thread/ring cells from the zero-friction lid fixture.
    // Independent convex half-space feasibility confirms positive clearance.
    func testDisjointMillimetreFeaturesDoNotApplyPhantomForces() throws {
        let a:[F3]=[F3(0.009930955156,-0.01736018807,0.02349619772),
F3(0.01092405067,-0.01909620687,0.02449619772),
F3(0.01092405067,-0.01909620687,0.02549619772),
F3(0.01114862209,-0.01660446403,0.02356463878),
F3(0.01226348429,-0.01826491043,0.02456463878),
F3(0.01226348429,-0.01826491043,0.02556463878)]
        let b:[F3]=[F3(0.01244477322,-0.01862491932,0.025),
F3(0.01363625601,-0.01777111482,0.025),
F3(0.01388925583,-0.02078674031,0.025),
F3(0.01521903573,-0.01983383351,0.025),
F3(0.01244477322,-0.01862491932,0.042),
F3(0.01363625601,-0.01777111482,0.042),
F3(0.01388925583,-0.02078674031,0.042),
F3(0.01521903573,-0.01983383351,0.042)]
        for scale:Float in [0.1,1,10] {
            var scene=PhysicsScene(name:"thread feature gap")
            scene.settings.gravity=0; scene.settings.dt=1/240
            scene.settings.iterations=16;scene.settings.collisionMargin=1e-5*scale
            let fixed=scene.addBody(size:F3(repeating:0.05*scale),density:0,friction:0,position:.zero,collisionEnabled:false)
            let moving=scene.addBody(size:F3(repeating:0.05*scale),density:1,friction:0,position:.zero,mass:0.08,diagonalInertia:F3(repeating:4e-5*scale*scale),collisionEnabled:false)
            scene.addConvexCollider(body:fixed,vertices:a.map{$0*scale})
            scene.addConvexCollider(body:moving,vertices:b.map{$0*scale})
            let solver=try GPUSolver(scene:scene)
            for _ in 0..<4 {try solver.submitStep();try solver.synchronize()}
            XCTAssertLessThan(length(solver.bodyPosition(moving)),1e-7*scale)
            XCTAssertLessThan(length(solver.bodyVelocity(moving)),1e-6*scale)
            XCTAssertTrue(solver.activeRigidContactNormalLoads().allSatisfy{$0.normalLoad < 1e-6})
        }
    }
}
