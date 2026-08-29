import XCTest
import simd
@testable import SimCore
@testable import PhysicsAVBD

final class CPUConvexColliderTests: XCTestCase {
    private let cubeVertices: [F3] = [
        F3(-0.5, -0.5, -0.5), F3(0.5, -0.5, -0.5),
        F3(-0.5, 0.5, -0.5), F3(0.5, 0.5, -0.5),
        F3(-0.5, -0.5, 0.5), F3(0.5, -0.5, 0.5),
        F3(-0.5, 0.5, 0.5), F3(0.5, 0.5, 0.5),
    ]

    private func manifolds(_ solver: CPUSolver) -> [CPUManifold] {
        solver.forces.compactMap { $0 as? CPUManifold }
    }

    private func isFinite(_ value: F3) -> Bool {
        value.x.isFinite && value.y.isFinite && value.z.isFinite
    }

    private func tetraAsset(offset: F3 = .zero) throws -> ConvexHullAsset {
        let vertices = [
            F3(-0.5, -0.5, -0.5) + offset,
            F3(-0.5, 0.5, 0.5) + offset,
            F3(0.5, -0.5, 0.5) + offset,
            F3(0.5, 0.5, -0.5) + offset,
        ]
        let triangles: [SIMD3<UInt32>] = [
            SIMD3(0, 1, 3), SIMD3(0, 2, 1),
            SIMD3(0, 3, 2), SIMD3(1, 2, 3),
        ]
        let edges = [
            ConvexHullEdge(vertexA: 0, vertexB: 1, faceA: 0, faceB: 1),
            ConvexHullEdge(vertexA: 0, vertexB: 2, faceA: 1, faceB: 2),
            ConvexHullEdge(vertexA: 0, vertexB: 3, faceA: 0, faceB: 2),
            ConvexHullEdge(vertexA: 1, vertexB: 2, faceA: 1, faceB: 3),
            ConvexHullEdge(vertexA: 1, vertexB: 3, faceA: 0, faceB: 3),
            ConvexHullEdge(vertexA: 2, vertexB: 3, faceA: 2, faceB: 3),
        ]
        let digest = ConvexHullAsset.geometryDigest(
            vertices: vertices, triangles: triangles)
        return try ConvexHullAsset(
            vertices: vertices, triangles: triangles, edges: edges,
            boundsMin: F3(repeating: -0.5) + offset,
            boundsMax: F3(repeating: 0.5) + offset,
            boundingRadius: sqrt(0.75), volume: 1.0 / 3.0,
            centroid: offset, digest: digest,
            stableID: "hull-" + digest.prefix(16))
    }

    func testLegacyDirectBodiesStillReceiveImplicitColliders() {
        let solver = CPUSolver()
        _ = solver.addBody(
            size: F3(repeating: 1), density: 0, friction: 0.5,
            position: .zero)
        _ = solver.addBody(
            size: F3(repeating: 1), density: 1, friction: 0.5,
            position: F3(0, 0, 0.8))

        XCTAssertEqual(solver.colliders.count, 2)
        solver.step()
        XCTAssertEqual(manifolds(solver).count, 1)
    }

    func testLegacyImplicitColliderTracksPublicBodyMutation() {
        let solver = CPUSolver()
        let body = solver.addBody(
            size: F3(repeating: 1), density: 1, friction: 0.2,
            position: .zero)
        body.size = F3(2, 0.5, 0.5)
        body.shape = .capsule
        body.friction = 0.8
        body.dynamicFriction = 0.3

        solver.step()

        XCTAssertEqual(solver.colliders.count, 1)
        XCTAssertEqual(solver.colliders[0].size, body.size)
        XCTAssertEqual(solver.colliders[0].shape, .capsule)
        XCTAssertEqual(solver.colliders[0].friction, 0.8)
        XCTAssertEqual(solver.colliders[0].dynamicFriction, 0.3)
        XCTAssertTrue(solver.colliders[0].usesWorldSpaceRoundAnchor)
        XCTAssertEqual(solver.colliders[0].boundingRadius, 1.5,
                       accuracy: 1e-6)
    }

    func testSceneConversionCopiesAuthoredColliderStateWithoutDuplicates() throws {
        var scene = PhysicsScene(name: "cpu-collider-copy")
        let bodyRotation = Quat(angle: .pi / 2, axis: F3(0, 0, 1))
        let localRotation = Quat(angle: .pi / 2, axis: F3(0, 1, 0))
        let body = scene.addBody(
            size: F3(repeating: 1), density: 1, friction: 0.1,
            position: F3(3, 4, 5), rotation: bodyRotation,
            collisionEnabled: false)
        let asset = try tetraAsset(offset: F3(2, 0, 0))
        _ = scene.addConvexCollider(
            body: body, asset: asset, friction: 0.81,
            dynamicFriction: 0.36, localPosition: F3(2, 0, 0),
            localRotation: localRotation, collisionGroup: 7,
            collidesWithSharedGeometry: false, collisionEnabled: false)

        let solver = scene.makeCPUSolver()
        XCTAssertEqual(solver.colliders.count, scene.colliders.count)
        XCTAssertEqual(solver.colliders.count, 1)
        XCTAssertEqual(solver.convexAssets, scene.convexAssets)

        let collider = solver.colliders[0]
        XCTAssertEqual(collider.body, body)
        XCTAssertEqual(collider.convexAssetID, 0)
        XCTAssertEqual(collider.friction, 0.81)
        XCTAssertEqual(collider.dynamicFriction, 0.36)
        XCTAssertEqual(collider.collisionGroup, 7)
        XCTAssertFalse(collider.collidesWithSharedGeometry)
        XCTAssertFalse(collider.collisionEnabled)

        let pose = collider.worldPose(solver.bodies[body])
        XCTAssertEqual(pose.position.x, 3, accuracy: 1e-5)
        XCTAssertEqual(pose.position.y, 6, accuracy: 1e-5)
        XCTAssertEqual(pose.position.z, 5, accuracy: 1e-5)
        let expectedAxis = (bodyRotation * localRotation).act(F3(1, 0, 0))
        let actualAxis = pose.orientation.act(F3(1, 0, 0))
        XCTAssertLessThan(length(actualAxis - expectedAxis), 1e-5)
    }

    func testInlineHullRoutesAgainstHullBoxSphereAndCapsule() {
        let cases: [(String, BodyShape, F3, Bool)] = [
            ("hull", .box, F3(repeating: 1), true),
            ("box", .box, F3(repeating: 0.7), false),
            ("sphere", .sphere, F3(repeating: 0.7), false),
            ("capsule", .capsule, F3(0.8, 0.25, 0), false),
        ]

        for (name, shape, size, useHull) in cases {
            var scene = PhysicsScene(name: "cpu-hull-\(name)")
            scene.settings.gravity = 0
            scene.settings.iterations = 1
            let hullBody = scene.addBody(
                size: F3(repeating: 1), density: 0, friction: 0.5,
                position: .zero, collisionEnabled: false)
            _ = scene.addConvexCollider(
                body: hullBody, vertices: cubeVertices, friction: 0.5)

            let otherBody = scene.addBody(
                size: size, density: 1, friction: 0.5,
                position: F3(useHull ? 0.75 : 0.55, 0, 0),
                shape: shape, collisionEnabled: !useHull)
            if useHull {
                _ = scene.addConvexCollider(
                    body: otherBody, vertices: cubeVertices, friction: 0.5)
            }

            let solver = scene.makeCPUSolver()
            solver.step()
            let pairManifolds = manifolds(solver).filter {
                guard let a = $0.bodyA?.index, let b = $0.bodyB?.index else {
                    return false
                }
                return Set([a, b]) == Set([hullBody, otherBody])
            }
            XCTAssertEqual(pairManifolds.count, 1,
                           "missing CPU hull contact against \(name)")
            guard let manifold = pairManifolds.first else { continue }
            XCTAssertFalse(manifold.contacts.isEmpty)
            XCTAssertEqual(length(manifold.basis.0), 1, accuracy: 1e-4)
            XCTAssertTrue(manifold.contacts.allSatisfy {
                isFinite($0.rA) && isFinite($0.rB) && isFinite($0.C0)
            })
        }
    }

    func testSharedOffOriginHullUsesSourceFrameAndAABBCenter() throws {
        var scene = PhysicsScene(name: "cpu-shared-off-origin")
        scene.settings.gravity = 0
        let hullBody = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 0.5,
            position: .zero, collisionEnabled: false)
        _ = scene.addConvexCollider(
            body: hullBody, asset: try tetraAsset(offset: F3(2, 0, 0)))
        let sphereBody = scene.addBody(
            size: F3(repeating: 0.7), density: 1, friction: 0.5,
            position: F3(2.1, 0, 0), shape: .sphere)

        let solver = scene.makeCPUSolver()
        XCTAssertEqual(solver.colliders[0].localBoundsCenter, F3(2, 0, 0))
        solver.step()
        XCTAssertTrue(manifolds(solver).contains {
            Set([$0.bodyA!.index, $0.bodyB!.index])
                == Set([hullBody, sphereBody])
        })
    }

    func testInlineHullCenterShiftUsesNormalizedAuthoredRotation() {
        let sourceCenter = F3(2, -3, 4)
        let vertices = cubeVertices.map { $0 + sourceCenter }
        let unitRotation = Quat(
            angle: 0.73, axis: normalize(F3(0.2, -0.6, 0.7)))
        let nonUnitRotation = Quat(vector: unitRotation.vector * 3.5)
        let authoredOffset = F3(-1.5, 0.25, 2.0)

        var scene = PhysicsScene(name: "inline-hull-normalized-transform")
        let body = scene.addBody(
            size: F3(repeating: 1), density: 1, friction: 0.5,
            position: .zero, collisionEnabled: false)
        let colliderID = scene.addConvexCollider(
            body: body, vertices: vertices,
            localPosition: authoredOffset,
            localRotation: nonUnitRotation)

        let collider = scene.colliders[colliderID]
        XCTAssertLessThan(
            simd_length(collider.localRotation.vector - unitRotation.vector),
            1e-6)
        for (source, centered) in zip(vertices, collider.convexHullVertices) {
            let reconstructed = collider.localPosition
                + collider.localRotation.act(centered)
            let expected = authoredOffset + unitRotation.act(source)
            XCTAssertLessThan(length(reconstructed - expected), 2e-5)
        }
    }

    func testCompoundBodyPairKeepsOneManifoldPerColliderPair() {
        var scene = PhysicsScene(name: "cpu-compound-pairs")
        scene.settings.gravity = 0
        scene.settings.iterations = 1
        let base = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 0.5,
            position: .zero, collisionEnabled: false)
        let moving = scene.addBody(
            size: F3(repeating: 1), density: 1, friction: 0.5,
            position: F3(0, 0, 0.8), collisionEnabled: false)
        for x in [Float(-1), Float(1)] {
            _ = scene.addCollider(
                body: base, size: F3(repeating: 1),
                localPosition: F3(x, 0, 0))
            _ = scene.addCollider(
                body: moving, size: F3(repeating: 1),
                localPosition: F3(x, 0, 0))
        }

        let solver = scene.makeCPUSolver()
        solver.step()
        let pairManifolds = manifolds(solver).filter {
            Set([$0.bodyA!.index, $0.bodyB!.index]) == Set([base, moving])
        }
        XCTAssertEqual(pairManifolds.count, 2)
        XCTAssertEqual(Set(pairManifolds.compactMap { $0.colliderPairKey }).count, 2)
    }

    func testColliderFiltersAndExplicitExclusionsReachCPUBroadphase() {
        func count(
            groupA: UInt32, groupB: UInt32,
            sharedA: Bool = true, sharedB: Bool = true,
            enabledB: Bool = true, excluded: Bool = false
        ) -> Int {
            var scene = PhysicsScene(name: "cpu-collider-filter")
            scene.settings.gravity = 0
            let a = scene.addBody(
                size: F3(repeating: 1), density: 0, friction: 0.5,
                position: .zero, collisionEnabled: false)
            let b = scene.addBody(
                size: F3(repeating: 1), density: 1, friction: 0.5,
                position: F3(0, 0, 0.8), collisionEnabled: false)
            _ = scene.addCollider(
                body: a, size: F3(repeating: 1), collisionGroup: groupA,
                collidesWithSharedGeometry: sharedA)
            _ = scene.addCollider(
                body: b, size: F3(repeating: 1), collisionGroup: groupB,
                collidesWithSharedGeometry: sharedB,
                collisionEnabled: enabledB)
            if excluded { scene.addCollisionExclusion(bodyA: a, bodyB: b) }
            let solver = scene.makeCPUSolver()
            solver.step()
            return manifolds(solver).count
        }

        XCTAssertEqual(count(groupA: 1, groupB: 1), 1)
        XCTAssertEqual(count(groupA: 1, groupB: 2), 0)
        XCTAssertEqual(count(groupA: 0, groupB: 1, sharedB: false), 0)
        XCTAssertEqual(count(groupA: 1, groupB: 0, sharedA: false), 0)
        XCTAssertEqual(count(groupA: 1, groupB: 1, enabledB: false), 0)
        XCTAssertEqual(count(groupA: 1, groupB: 1, excluded: true), 0)
    }

    func testColliderMaterialsDriveManifoldFriction() throws {
        var scene = PhysicsScene(name: "cpu-collider-materials")
        scene.settings.gravity = 0
        let a = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 0.01,
            position: .zero, collisionEnabled: false)
        let b = scene.addBody(
            size: F3(repeating: 1), density: 1, friction: 0.04,
            position: F3(0, 0, 0.8), collisionEnabled: false)
        _ = scene.addCollider(
            body: a, size: F3(repeating: 1), friction: 0.81,
            dynamicFriction: 0.36)
        _ = scene.addCollider(
            body: b, size: F3(repeating: 1), friction: 0.25,
            dynamicFriction: 0.04)

        let solver = scene.makeCPUSolver()
        solver.step()
        let manifold = try XCTUnwrap(manifolds(solver).first)
        XCTAssertEqual(manifold.staticFriction, 0.45, accuracy: 1e-6)
        XCTAssertEqual(manifold.dynamicFriction, 0.12, accuracy: 1e-6)
    }

    func testConvexWarmStartStateFollowsAnchorsInsteadOfSortIndices() throws {
        var scene = PhysicsScene(name: "cpu-convex-anchor-persistence")
        scene.settings.gravity = 0
        scene.settings.iterations = 1
        let a = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 0.5,
            position: .zero, collisionEnabled: false)
        let b = scene.addBody(
            size: F3(repeating: 1), density: 1, friction: 0.5,
            position: F3(0, 0, 0.8), collisionEnabled: false)
        _ = scene.addConvexCollider(
            body: a, vertices: cubeVertices, friction: 0.5)
        _ = scene.addConvexCollider(
            body: b, vertices: cubeVertices, friction: 0.5)

        let solver = scene.makeCPUSolver()
        XCTAssertEqual(solver.colliders.count, 2)
        let manifold = CPUManifold(
            solver: solver, colliderA: 0, colliderB: 1, pairKey: 1)
        XCTAssertTrue(try manifold.initialize().get())
        XCTAssertGreaterThan(manifold.contacts.count, 1)

        let anchors = manifold.contacts.map(\.rA)
        var expectedMarkerByAnchor = [Int: Float]()
        for index in manifold.contacts.indices {
            let marker = Float(index + 1)
            expectedMarkerByAnchor[index] = marker
            manifold.contacts[index].lambda = F3(marker, 0, 0)
            // Deliberately rotate the transient sort keys. A feature-key-only
            // merge transfers the marker to the wrong physical contact.
            manifold.contacts[index].featureKey = Int32(
                (index + 1) % manifold.contacts.count)
        }

        XCTAssertTrue(try manifold.initialize().get())
        let discount = solver.alpha * solver.gamma
        for contact in manifold.contacts {
            let nearest = anchors.indices.min {
                length_squared(anchors[$0] - contact.rA)
                    < length_squared(anchors[$1] - contact.rA)
            }
            let anchorIndex = try XCTUnwrap(nearest)
            XCTAssertEqual(
                contact.lambda.x,
                try XCTUnwrap(expectedMarkerByAnchor[anchorIndex]) * discount,
                accuracy: 1.0e-5)
        }
    }

    func testConvexQueryFailureRollsBackAndLatchesTypedError() throws {
        var scene = PhysicsScene(name: "cpu-convex-query-failure")
        scene.settings.gravity = 0
        let fixed = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 0.5,
            position: .zero, collisionEnabled: false)
        let moving = scene.addBody(
            size: F3(repeating: 1), density: 1, friction: 0.5,
            position: F3(0, 0, 0.8),
            rotation: Quat(angle: 0.2, axis: F3(0, 1, 0)),
            collisionEnabled: false)
        _ = scene.addConvexCollider(
            body: fixed, vertices: cubeVertices, friction: 0.5)
        _ = scene.addConvexCollider(
            body: moving, vertices: cubeVertices, friction: 0.5)

        let solver = scene.makeCPUSolver()
        let positions = solver.bodies.map(\.positionLin)
        let orientations = solver.bodies.map { $0.positionAng.vector }
        let injected = ConvexNarrowPhase.Failure.didNotConverge(stage: .mpr)
        let expected = CPUSolver.RuntimeFailure.convexQuery(
            colliderA: 0, colliderB: 1, failure: injected)
        solver.convexQueryFailureForTesting = injected

        XCTAssertThrowsError(try solver.stepChecked()) { error in
            XCTAssertEqual(error as? CPUSolver.RuntimeFailure, expected)
        }
        XCTAssertEqual(solver.runtimeFailure, expected)
        for index in solver.bodies.indices {
            XCTAssertEqual(solver.bodies[index].positionLin, positions[index])
            XCTAssertEqual(solver.bodies[index].positionAng.vector,
                           orientations[index])
        }

        // The legacy entry point is source-compatible and fail-closed. The
        // throwing companion keeps reporting the first terminal identity.
        solver.step()
        XCTAssertThrowsError(try solver.stepChecked()) { error in
            XCTAssertEqual(error as? CPUSolver.RuntimeFailure, expected)
        }
        XCTAssertEqual(solver.runtimeFailure, expected)
    }

    func testNonCollidingTorusHullPairsDoNotTripPreflight() {
        func makeScene(dynamicHull: Bool, excluded: Bool) -> PhysicsScene {
            var scene = PhysicsScene(name: "cpu-torus-hull-filter")
            let hullBody = scene.addBody(
                size: F3(repeating: 1), density: dynamicHull ? 1 : 0,
                friction: 0.5, position: .zero, collisionEnabled: false)
            _ = scene.addConvexCollider(
                body: hullBody, vertices: cubeVertices, friction: 0.5)
            let torusBody = scene.addBody(
                size: F3(0.6, 0.15, 0), density: 0, friction: 0.5,
                position: F3(0.2, 0, 0), shape: .torus)
            if excluded {
                scene.addCollisionExclusion(bodyA: hullBody, bodyB: torusBody)
            }
            return scene
        }

        XCTAssertEqual(try makeScene(dynamicHull: false, excluded: false)
            .makeCPUSolverChecked().bodies.count, 2)
        XCTAssertEqual(try makeScene(dynamicHull: true, excluded: true)
            .makeCPUSolverChecked().bodies.count, 2)
    }

    func testPotentialTorusHullPairFailsCheckedSceneConversion() {
        var scene = PhysicsScene(name: "cpu-torus-hull-unsupported")
        let hullBody = scene.addBody(
            size: F3(repeating: 1), density: 1, friction: 0.5,
            position: .zero, collisionEnabled: false)
        _ = scene.addConvexCollider(
            body: hullBody, vertices: cubeVertices, friction: 0.5)
        _ = scene.addBody(
            size: F3(0.6, 0.15, 0), density: 0, friction: 0.5,
            position: F3(0.2, 0, 0), shape: .torus)

        XCTAssertThrowsError(try scene.makeCPUSolverChecked()) { error in
            XCTAssertEqual(
                error as? CPUConvexCollisionError,
                .unsupportedTorusHull(colliderA: 0, colliderB: 1))
        }
    }

    func testCheckedSceneConversionRejectsRawInvalidConvexReferences() throws {
        func baseScene() -> (PhysicsScene, Int) {
            var scene = PhysicsScene(name: "cpu-invalid-raw-convex")
            let body = scene.addBody(
                size: F3(repeating: 1), density: 1, friction: 0.5,
                position: .zero, collisionEnabled: false)
            return (scene, body)
        }

        var (missing, missingBody) = baseScene()
        missing.colliders.append(SceneCollider(
            body: missingBody, size: F3(repeating: 1), friction: 0.5,
            convexAssetID: 7))
        XCTAssertThrowsError(try missing.makeCPUSolverChecked()) { error in
            XCTAssertEqual(
                error as? CPUConvexCollisionError,
                .invalidAssetReference(collider: 0, asset: 7))
        }

        var (conflicting, conflictingBody) = baseScene()
        let assetID = conflicting.registerConvexAsset(
            try tetraAsset(offset: .zero))
        conflicting.colliders.append(SceneCollider(
            body: conflictingBody, size: F3(repeating: 1), friction: 0.5,
            convexHullVertices: cubeVertices, convexAssetID: assetID))
        XCTAssertThrowsError(try conflicting.makeCPUSolverChecked()) { error in
            XCTAssertEqual(
                error as? CPUConvexCollisionError,
                .conflictingAssetSources(collider: 0))
        }
    }
}
