import XCTest
import simd
@testable import PhysicsAVBD
@testable import SimCore

final class ThinConvexHullTests: XCTestCase {
    // A real generated plate-wall sector: a thin annulus above a solid foot.
    // Float construction makes some nominally coplanar vertices differ by a
    // few nanometres. The support cloud is still a valid solid convex set.
    private let sector: [F3] = [
        F3(0, 0, -0.008243507),
        F3(-0.038563777, 0.022264821, -0.008243507),
        F3(-0.04452962, -3.892903e-9, -0.008243507),
        F3(-0.07023556, 0.040550545, 0.008243507),
        F3(-0.08110105, -7.090079e-9, 0.008243507),
        F3(-0.07369966, 0.042550545, 0.008243507),
        F3(-0.08510105, -7.43977e-9, 0.008243507),
    ]

    func testThinRadialSectorBuildsAfterCenteringRotationAndScaling() throws {
        for scale: Float in [0.1, 1, 10] {
            for step in 0..<24 {
                let q = Quat(angle: Float(step) * .pi / 12, axis: normalize(F3(1, 2, 3)))
                let source = sector.map { q.act($0 * scale) }
                let lower = source.reduce(F3(repeating: .infinity), simd_min)
                let upper = source.reduce(F3(repeating: -.infinity), simd_max)
                let vertices = source.map { $0 - (lower + upper) * 0.5 }
                do {
                    let faces = try ConvexHullTopologyBuilder.triangulate(vertices: vertices)
                    XCTAssertGreaterThanOrEqual(faces.count, 4)
                    XCTAssertEqual(faces, try ConvexHullTopologyBuilder.triangulate(vertices: vertices))
                } catch {
                    XCTFail("scale \(scale), rotation \(step): \(error)")
                }
            }
        }
    }

    func testThinSectorCanBeUploadedAsAnInlineCollider() throws {
        var scene = PhysicsScene(name: "thin generated plate sector")
        let body = scene.addBody(size: F3(repeating: 0.1), density: 100,
                                 friction: 0.5, position: F3(0, 0, 1), collisionEnabled: false)
        scene.addConvexCollider(body: body, vertices: sector, collisionEnabled: true, isRendered: false)
        let gpu = try GPUSolver(scene: scene)
        try gpu.submitStep()
        try gpu.synchronize()
        XCTAssertTrue(gpu.bodyPosition(body).z.isFinite)
    }
}
