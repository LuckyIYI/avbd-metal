import XCTest
import simd
@testable import SimCore

final class CableAuthoringTests: XCTestCase {
    func testYieldMomentDoesNotDependOnResolution() {
        let material = CableMaterial(stretchRigidity: 2000, shearRigidity: 1000,
            bendRigidity: 0.8, twistRigidity: 0.1, dampingTime: 0.12, yieldCurvature: 1.5)
        for length: Float in [0.2, 0.1, 0.025] {
            let joint = CableJointMaterial(material: material, restLength: length)
            XCTAssertEqual(joint.angularStiffness.x * joint.yieldAngle, 1.2, accuracy: 1e-6)
        }
    }

    private let material = CableMaterial.circular(radius: 0.02, youngModulus: 1e6)

    func testRefinementExcludesOnlyLocalOverlappingMaterial() throws {
        var scene = PhysicsScene(name: "refined cable")
        let cable = try scene.addCable(points: (0...10).map { F3(Float($0) * 0.01, 0, 0) },
            radius: 0.02, density: 1000, material: material)
        XCTAssertFalse(scene.canPotentiallyCollide(colliderA: 0, colliderB: 2))
        XCTAssertFalse(scene.canPotentiallyCollide(colliderA: 0, colliderB: 4))
        XCTAssertTrue(scene.canPotentiallyCollide(colliderA: 0, colliderB: 6))
        let other = try scene.addCable(points: [F3(0,-0.05,0), F3(0,0.05,0)],
            radius: 0.02, density: 1000, material: material)
        XCTAssertTrue(scene.canPotentiallyCollide(colliderA: cable.bodyIDs[0], colliderB: other.bodyIDs[0]))
        let copy = scene.replicated(count: 2, spacing: F3(2,2,0)).scene
        XCTAssertEqual(copy.collisionExclusions.count, 2 * scene.collisionExclusions.count)
    }

    func testNonuniformLengthsMassFramesAndReplication() throws {
        var scene = PhysicsScene(name: "cable authoring")
        let cable = try scene.addCable(points: [.zero, F3(0, 0, 0.2), F3(0.3, 0, 0.6)],
            radius: 0.02, density: 1000, material: material, fixedSegments: [0])
        XCTAssertEqual(cable.restLengths[0], 0.2, accuracy: 1e-6)
        XCTAssertEqual(cable.restLengths[1], 0.5, accuracy: 1e-6)
        XCTAssertEqual(scene.bodies[0].mass, 0)
        XCTAssertEqual(scene.bodies[1].mass!, 1000 * .pi * 0.02 * 0.02 * 0.5, accuracy: 1e-6)
        XCTAssertEqual(scene.joints[0].cable!.linearStiffness.z,
                       material.stretchRigidity / 0.35, accuracy: 0.001)
        XCTAssertFalse(scene.canPotentiallyCollide(colliderA: 0, colliderB: 1))
        XCTAssertLessThan(length(scene.bodies[1].rotation.act(F3(0, 0, 1))
                                - F3(0.6, 0, 0.8)), 1e-5)
        let copy = scene.replicated(count: 2, spacing: F3(2, 2, 0))
        XCTAssertEqual(copy.scene.cables[1].bodyIDs, [2, 3])
        XCTAssertEqual(copy.scene.cables[1].jointIDs, [1])
        XCTAssertEqual(copy.scene.joints[1].cable, scene.joints[0].cable)
    }

    func testRejectedInputDoesNotPartiallyAppend() throws {
        var scene = PhysicsScene(name: "invalid cable")
        for points in [[F3.zero], [.zero, .zero], [.zero, F3(.infinity, 0, 0)]] {
            XCTAssertThrowsError(try scene.addCable(points: points, radius: 0.02,
                density: 1000, material: material))
            XCTAssertTrue(scene.bodies.isEmpty && scene.joints.isEmpty && scene.cables.isEmpty)
        }
        XCTAssertThrowsError(try scene.addCable(points: [.zero, F3(0, 0, 1)], radius: 0.02,
            density: 1000, material: material, fixedSegments: [1]))
        XCTAssertThrowsError(try scene.addCable(points: [.zero, F3(0, 0, 1)], radius: 0.02,
            density: 1000, material: material, rotations: [Quat(angle: 1, axis: F3(1, 0, 0))]))
        XCTAssertTrue(scene.bodies.isEmpty)
    }
}
