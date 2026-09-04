import XCTest
import simd
@testable import SimCore

final class SceneReplicationTests: XCTestCase {
    func testGridReplicationBoundsLargeBatchCoordinates() {
        var source = PhysicsScene(name: "grid-batch")
        let body = source.addBody(
            size: F3(repeating: 0.2), density: 1, friction: 1,
            position: F3(0.25, -0.5, 0.75))
        source.addJoint(SceneJoint(
            bodyA: -1, bodyB: body, rA: F3(0.25, -0.5, 0.75),
            rB: .zero, stiffnessLin: .infinity,
            stiffnessAng: .infinity))

        let result = source.replicated(
            count: 96, spacing: F3(10, 10, 0), columns: 10)
        XCTAssertEqual(result.replicas[95].worldOffset, F3(50, 90, 0))
        XCTAssertLessThanOrEqual(
            result.replicas.map { abs($0.worldOffset.x) }.max()!, 90)
        XCTAssertLessThanOrEqual(
            result.replicas.map { abs($0.worldOffset.y) }.max()!, 90)
        for replica in result.replicas {
            XCTAssertEqual(
                result.scene.bodies[replica.body(body)].position,
                source.bodies[body].position + replica.worldOffset)
            XCTAssertEqual(
                result.scene.joints[replica.joint(0)].rA,
                source.joints[0].rA + replica.worldOffset)
        }
    }

    func testReplicationPreservesNestedCollisionDomainsAndWorldAnchors() {
        var source = PhysicsScene(name: "nested-domains")
        let ground = source.addBody(
            size: F3(4, 4, 0.1), density: 0, friction: 1,
            position: F3(0, 0, -0.05), collisionGroup: 0)
        let robot = source.addBody(
            size: F3(repeating: 0.2), density: 1, friction: 1,
            position: F3(0, 0, 0.5), collisionGroup: 1)
        let mouth = source.addBody(
            size: F3(repeating: 0.1), density: 1, friction: 2,
            position: F3(0.2, 0, 0.5), collisionGroup: 2)
        let p0 = source.addParticle(
            radius: 0.02, mass: 0.01, friction: 2,
            position: F3(0.22, -0.03, 0.5))
        let p1 = source.addParticle(
            radius: 0.02, mass: 0.01, friction: 2,
            position: F3(0.22, 0.03, 0.5))
        let p2 = source.addParticle(
            radius: 0.02, mass: 0.01, friction: 2,
            position: F3(0.25, 0, 0.55))
        for collider in source.colliders.indices
            where [p0, p1, p2].contains(source.colliders[collider].body) {
            source.colliders[collider].collisionGroup = 2
        }
        source.addTri(SceneTri(ids: (p0, p1, p2)))
        source.addJoint(SceneJoint(
            bodyA: -1, bodyB: robot, rA: F3(0, 0, 0.5), rB: .zero,
            stiffnessLin: .infinity, stiffnessAng: .infinity))
        source.addCollisionExclusion(bodyA: robot, bodyB: mouth)
        source.rigidMeshes.append(SceneRigidMesh(
            body: mouth,
            mesh: SurfaceMesh(
                vertices: [.zero, F3(1, 0, 0), F3(0, 1, 0)],
                normals: [F3(0, 0, 1), F3(0, 0, 1), F3(0, 0, 1)],
                triangles: [(0, 1, 2)])))

        let result = source.replicated(
            count: 3, spacing: F3(0, 10, 0), includeVisuals: false)
        XCTAssertEqual(result.replicas.count, 3)
        XCTAssertTrue(result.scene.rigidMeshes.isEmpty)
        XCTAssertEqual(result.scene.bodies.count, source.bodies.count * 3)
        XCTAssertEqual(result.scene.tris.count, 3)
        XCTAssertEqual(result.scene.collisionExclusions.count, 3)

        for replica in result.replicas {
            XCTAssertEqual(
                result.scene.bodies[replica.body(ground)].position.y,
                10 * Float(replica.index), accuracy: 1e-6)
            XCTAssertEqual(
                result.scene.joints[replica.joint(0)].rA.y,
                10 * Float(replica.index), accuracy: 1e-6)
            XCTAssertEqual(
                result.scene.tris[replica.index].ids.0,
                replica.body(p0))
            let groups = source.colliders.indices.map {
                result.scene.colliders[replica.collider($0)].collisionGroup
            }
            XCTAssertEqual(groups, source.colliders.map(\.collisionGroup),
                "replication must not flatten robot/mouth/shared domains")
        }
    }
}
