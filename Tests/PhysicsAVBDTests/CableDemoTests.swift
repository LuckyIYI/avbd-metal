import XCTest
import simd
import SimCore
import GPUSimDemos

final class CableDemoTests: XCTestCase {
    func testClipsHaveConformingVolumeAndNoCableAttachments() {
        let scene = Demos.cableGrippers()
        let cableBodies = Set(scene.cables.flatMap(\.bodyIDs))
        let particles = scene.bodies.indices.filter { scene.bodies[$0].isParticle }
        XCTAssertFalse(particles.isEmpty)
        XCTAssertTrue(particles.contains { scene.bodies[$0].density == 0 })
        XCTAssertTrue(particles.contains { scene.bodies[$0].density > 0 })
        for joint in scene.joints {
            // Retention/release must come from contact with the clip lips.
            XCTAssertEqual(cableBodies.contains(joint.bodyA), cableBodies.contains(joint.bodyB))
        }
        var faces: [[Int]: Int] = [:]
        var volume: Float = 0
        for tet in scene.tets {
            let ids = [tet.ids.0, tet.ids.1, tet.ids.2, tet.ids.3]
            let p = ids.map { scene.bodies[$0].position }
            let v = abs(dot(p[1] - p[0], cross(p[2] - p[0], p[3] - p[0]))) / 6
            XCTAssertGreaterThan(v, 1e-9)
            volume += v
            for omitted in 0..<4 {
                faces[ids.enumerated().filter { $0.offset != omitted }.map(\.element).sorted(), default: 0] += 1
            }
        }
        XCTAssertTrue(faces.values.allSatisfy { $0 == 1 || $0 == 2 })
        var boundaryEdges: [[Int]: Int] = [:]
        for (face, count) in faces where count == 1 {
            for (a, b) in [(0,1), (1,2), (2,0)] {
                boundaryEdges[[face[a], face[b]].sorted(), default: 0] += 1
            }
        }
        XCTAssertTrue(boundaryEdges.values.allSatisfy { $0 == 2 }, "clip skins must be watertight")
        // Four 270-degree annular clips, 120 mm long. Linear surface facets
        // approximate the analytic annulus within two percent.
        let expected: Float = 4 * 0.12 * 0.75 * .pi * (0.075 * 0.075 - 0.04 * 0.04)
        XCTAssertEqual(volume, expected, accuracy: expected * 0.02)
    }

    func testTwistFixturesDriveBothMaterialFrames() {
        let scene = Demos.cableTwisting(turnRate: 0.17)
        XCTAssertEqual(scene.cables.count, 2)
        XCTAssertEqual(scene.spinners.count, 1)
        XCTAssertEqual(scene.spinners[0].omega, 0.17 * 2 * .pi, accuracy: 1e-6)
        for cable in scene.cables {
            for body in [cable.bodyIDs[0], cable.bodyIDs.last!] {
                let weld = scene.joints.first { $0.bodyB == body && $0.cable == nil }
                XCTAssertNotNil(weld)
                XCTAssertTrue(weld?.stiffnessAng.isInfinite == true)
            }
        }
    }

    func testThreadingGuideClearanceAndDemoParameters() throws {
        for radius: Float in [0.08, 0.13, 0.20] {
            let scene = try XCTUnwrap(Demos.make("cablethreading", params: ["clearance": radius]))
            let rings = scene.colliders.filter { $0.shape == .torus }
            XCTAssertEqual(rings.count, 3)
            for ring in rings {
                XCTAssertEqual(ring.size.x - ring.size.y, radius, accuracy: 1e-6)
            }
        }
    }
}
