import Foundation
import Metal
import PhysicsAVBD
import SimCore
import XCTest
import simd

final class ConvexCapacityTests: XCTestCase {
    func fixture(_ name: String) throws -> ConvexHullAsset {
        let url = try XCTUnwrap(Bundle.module.url(forResource: name, withExtension: "json",
            subdirectory: "ConvexCapacity"))
        return try JSONDecoder().decode(ConvexHullAsset.self, from: Data(contentsOf: url))
    }

    func testSceneSizedWorkspacesAndLargeFaceContact() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal required") }
        for (sides, capacity) in [(16,32), (32,64), (64,128)] {
            let hull = try fixture("prism-\(sides)")
            XCTAssertEqual(hull.vertices.count, sides * 2)
            var scene = PhysicsScene(name: "Convex capacity \(sides)")
            scene.settings.dt = 1/60
            scene.settings.gravity = -9.81
            scene.settings.iterations = 8
            let ground = scene.addBody(size: F3(2,2,0.2), density: 0, friction: 0.6,
                position: .zero, collisionEnabled: false)
            scene.addConvexCollider(body: ground, asset: hull)
            let body = scene.addBody(size: F3(2,2,0.2), density: 100, friction: 0.6,
                position: F3(0,0,0.24),
                rotation: Quat(angle: .pi / Float(sides), axis: F3(0,0,1)),
                collisionEnabled: false)
            scene.addConvexCollider(body: body, asset: hull)
            let solver = try GPUSolver(scene: scene)
            XCTAssertEqual(solver.convexClipWorkspaceVertices, capacity)
            let start = Date()
            for _ in 0..<180 { try solver.submitStep() }
            try solver.synchronize()
            let elapsed = Date().timeIntervalSince(start)
            XCTAssertNil(solver.runtimeFailure)
            XCTAssertEqual(solver.bodyPosition(body).z, 0.2, accuracy: 0.015)
            XCTAssertLessThan(length(solver.bodyPosition(body) - F3(0,0,0.2)), 0.035)
            XCTAssertFalse(solver.activeRigidContactPairs().isEmpty)
            print("CONVEX_CAPACITY sides=\(sides) workspace=\(capacity) ms_per_step=\(1000*elapsed/180)")
        }
    }

    func testConvexPickingUsesSurfaceNotBoundingSphere() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal required") }
        var scene = PhysicsScene(name: "Convex mouse picking")
        let body = scene.addBody(size: F3(1,1,1), density: 1, friction: 0.6, position: .zero, collisionEnabled: false)
        scene.addConvexCollider(body: body, asset: try fixture("tetra"))
        let solver = try GPUSolver(scene: scene)
        XCTAssertNil(solver.pick(origin: F3(0.7,0.7,2), dir: F3(0,0,-1)))
        let hit = try XCTUnwrap(solver.pick(origin: F3(0.1,0.1,2), dir: F3(0,0,-1)))
        XCTAssertEqual(hit.body, body)
        XCTAssertEqual(hit.local.z, 0.8, accuracy: 1e-5)
    }
}
