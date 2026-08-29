import Metal
import SimCore
import simd
import XCTest
@testable import PhysicsAVBD

final class ClassicRigidBodiesDemoTests: XCTestCase {
    func testSceneUsesFourDetailedMeshesAndCookedCompoundsOnPrimitiveTable()
        throws
    {
        for spec in Demos.classicRigidBodySpecs {
            XCTAssertNotNil(GPUSolver.physicsResourceBundle.url(
                forResource: spec.assetName, withExtension: "obj",
                subdirectory: "Assets/classic"))
            XCTAssertNotNil(GPUSolver.physicsResourceBundle.url(
                forResource: spec.assetName,
                withExtension: "avbdconvex.json",
                subdirectory: "Assets/convex/classic"))
        }

        XCTAssertTrue(Demos.all.contains("classicrigids"))
        XCTAssertFalse(Demos.supportsScale("classicrigids"))
        XCTAssertTrue(Demos.supportsScale("stack"))
        let scene = try XCTUnwrap(Demos.make("classicrigids", scale: 1))
        XCTAssertEqual(scene.name, "classicrigids")
        XCTAssertEqual(scene.rigidMeshes.count,
                       Demos.classicRigidBodySpecs.count)
        XCTAssertEqual(scene.rigidMeshes.count, 4)
        XCTAssertTrue(scene.rigidMeshes.allSatisfy {
            !$0.vertices.isEmpty && !$0.triangles.isEmpty
                && $0.vertices.count == $0.normals.count
        })

        let tableTopCollider = try XCTUnwrap(scene.colliders.first { collider in
            collider.convexAssetID == nil
                && length(collider.size - Demos.classicRigidTableTopSize) < 1e-6
                && length(scene.bodies[collider.body].position
                    - Demos.classicRigidTableTopPosition) < 1e-6
        })
        XCTAssertFalse(scene.bodies[tableTopCollider.body].isDynamic)
        let primitiveTableColliders = scene.colliders.filter { collider in
            collider.convexAssetID == nil
                && collider.shape == .box
                && scene.bodies[collider.body].position.z >= 0
                && scene.bodies[collider.body].position.z <= 3
        }
        XCTAssertGreaterThanOrEqual(primitiveTableColliders.count, 5)

        for (spec, visual) in zip(
            Demos.classicRigidBodySpecs, scene.rigidMeshes
        ) {
            let body = scene.bodies[visual.body]
            XCTAssertEqual(body.isDynamic, spec.isDynamic, spec.assetName)
            if spec.isDynamic {
                XCTAssertEqual(try XCTUnwrap(body.mass), spec.targetMass,
                               accuracy: 1e-6, spec.assetName)
                let inertia = try XCTUnwrap(body.diagonalInertia)
                XCTAssertGreaterThan(inertia.x, 0, spec.assetName)
                XCTAssertGreaterThan(inertia.y, 0, spec.assetName)
                XCTAssertGreaterThan(inertia.z, 0, spec.assetName)
                XCTAssertLessThanOrEqual(inertia.x, inertia.y + inertia.z + 1e-5)
                XCTAssertLessThanOrEqual(inertia.y, inertia.x + inertia.z + 1e-5)
                XCTAssertLessThanOrEqual(inertia.z, inertia.x + inertia.y + 1e-5)
            } else {
                XCTAssertNil(body.mass, spec.assetName)
                XCTAssertNil(body.diagonalInertia, spec.assetName)
            }

            let collision = scene.colliders.filter {
                $0.body == visual.body && $0.convexAssetID != nil
            }
            XCTAssertFalse(collision.isEmpty, spec.assetName)
            XCTAssertTrue(collision.allSatisfy {
                $0.collisionEnabled && !$0.isRendered
                    && $0.convexHullVertices.isEmpty
            }, spec.assetName)
            XCTAssertTrue(collision.allSatisfy {
                length($0.localPosition - visual.localPosition) < 1e-6
                    && abs(dot($0.localRotation.vector,
                               visual.localRotation.vector)) > 0.999999
            }, "visual/collision frame mismatch for \(spec.assetName)")
        }

        let dynamic = zip(Demos.classicRigidBodySpecs, scene.rigidMeshes)
            .filter { $0.0.isDynamic }
        XCTAssertEqual(dynamic.count, 4)
        let dragon = try XCTUnwrap(zip(
            Demos.classicRigidBodySpecs, scene.rigidMeshes
        ).first { $0.0.assetName == "stanford-dragon" })
        XCTAssertTrue(scene.bodies[dragon.1.body].isDynamic)
        XCTAssertEqual(try XCTUnwrap(scene.bodies[dragon.1.body].mass),
                       dragon.0.targetMass, accuracy: 1e-6)
        XCTAssertGreaterThanOrEqual(scene.convexAssets.count, 4)
    }

    func testDynamicClassicMeshesSettleOnTableWithoutRuntimeFailure() throws {
        try requireMetal()
        let scene = Demos.classicRigidBodies()
        let instances = zip(
            Demos.classicRigidBodySpecs, scene.rigidMeshes
        ).map { (spec: $0.0, body: $0.1.body) }
        let dynamic = instances.filter { $0.0.isDynamic }
        let initialPositions = Dictionary(uniqueKeysWithValues: dynamic.map {
            ($0.body, scene.bodies[$0.body].position)
        })
        let tableTopBody = try XCTUnwrap(scene.colliders.first { collider in
            collider.convexAssetID == nil
                && length(collider.size - Demos.classicRigidTableTopSize) < 1e-6
        }).body
        let tableTopZ = Demos.classicRigidTableTopPosition.z
            + Demos.classicRigidTableTopSize.z * 0.5

        // Keep the detailed meshes attached: indexed upload is part of the
        // sustained end-to-end scene contract and no longer carries the old
        // 30+ MiB expanded-corner penalty.
        let solver = try GPUSolver(scene: scene)
        XCTAssertNotNil(solver.renderIndexedRigidMeshSurface)
        XCTAssertEqual(solver.materializedLegacyRigidMeshByteCount, 0)
        var tableContacts = Set<Int>()
        for frame in 0..<720 {
            try solver.submitStep()
            if frame % 24 == 23 {
                try solver.synchronize()
                for pair in solver.activeRigidContactPairs() {
                    if pair.0 == tableTopBody { tableContacts.insert(pair.1) }
                    if pair.1 == tableTopBody { tableContacts.insert(pair.0) }
                }
                let states = solver.bodyStates(dynamic.map { $0.body })
                for state in states {
                    XCTAssertTrue(isFinite(state.position))
                    XCTAssertTrue(isFinite(state.linearVelocity))
                    XCTAssertTrue(isFinite(state.angularVelocity))
                    XCTAssertTrue(state.rotation.vector.x.isFinite
                        && state.rotation.vector.y.isFinite
                        && state.rotation.vector.z.isFinite
                        && state.rotation.vector.w.isFinite)
                    XCTAssertEqual(simd_length(state.rotation.vector), 1,
                                   accuracy: 2e-3)
                    XCTAssertGreaterThan(state.position.z, tableTopZ - 0.3)
                    XCTAssertLessThan(abs(state.position.x),
                                      Demos.classicRigidTableTopSize.x * 0.5)
                    XCTAssertLessThan(abs(state.position.y),
                                      Demos.classicRigidTableTopSize.y * 0.5)
                }
            }
        }
        try solver.synchronize()

        XCTAssertNil(solver.runtimeFailure)
        XCTAssertEqual(solver.uniqueConvexAssetCount, scene.convexAssets.count)
        XCTAssertGreaterThan(solver.convexDebugTriangleVertexCount, 0)
        XCTAssertGreaterThan(solver.convexDebugEdgeVertexCount, 0)
        for (spec, body) in dynamic {
            let position = solver.bodyPosition(body)
            let initial = try XCTUnwrap(initialPositions[body])
            XCTAssertTrue(tableContacts.contains(body),
                          "\(spec.assetName) never contacted the tabletop")
            XCTAssertGreaterThan(position.z, tableTopZ - 0.12,
                                 "\(spec.assetName) fell through the table")
            XCTAssertLessThan(length(position - initial), 0.75,
                              "\(spec.assetName) drifted unexpectedly")
            XCTAssertLessThan(length(solver.bodyVelocity(body)), 0.45,
                              "\(spec.assetName) did not settle")
            XCTAssertLessThan(length(solver.bodyAngularVelocity(body)), 0.8,
                              "\(spec.assetName) kept spinning")
        }
    }

    func testClassicVisualsUseIndexedStorageWithoutLegacyExpansion() throws {
        try requireMetal()
        let solver = try GPUSolver(scene: Demos.classicRigidBodies())
        let indexed = try XCTUnwrap(solver.renderIndexedRigidMeshSurface)
        XCTAssertEqual(indexed.indexCount, solver.rigidMeshVertexCount)
        XCTAssertEqual(solver.materializedLegacyRigidMeshByteCount, 0)

        let expandedBytes = solver.rigidMeshVertexCount
            * MemoryLayout<RigidMeshVertexGPU>.stride
        XCTAssertGreaterThan(expandedBytes, 25 * 1_024 * 1_024)
        XCTAssertLessThan(solver.indexedRigidMeshStorageByteCount,
                          expandedBytes / 3)
        XCTAssertLessThan(solver.indexedRigidMeshStorageByteCount,
                          10 * 1_024 * 1_024)
        XCTAssertEqual(solver.materializedLegacyRigidMeshByteCount, 0,
            "the built-in indexed path must not allocate compatibility corners")
    }

    private func requireMetal() throws {
        if MTLCreateSystemDefaultDevice() == nil {
            throw XCTSkip("Metal is unavailable")
        }
    }

    private func isFinite(_ vector: F3) -> Bool {
        vector.x.isFinite && vector.y.isFinite && vector.z.isFinite
    }
}
