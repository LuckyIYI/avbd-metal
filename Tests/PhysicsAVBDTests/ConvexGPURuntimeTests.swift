import XCTest
import Metal
import SimCore
import simd
@testable import PhysicsAVBD
@testable import GPUSimDemos

final class ConvexGPURuntimeTests: XCTestCase {
    func testNearTouchingHullBoxUsesStableMPRGJKSwitchover() throws {
        try requireMetal()
        let source = Demos.convexDecomposition(scale: 1)
        let assetID = try XCTUnwrap(source.colliders[7].convexAssetID)
        let lowerBar = source.convexAssets[assetID]

        var scene = PhysicsScene(name: "convex-hull-box-switchover")
        scene.settings.gravity = 0
        let hullBody = scene.addBody(
            size: F3(repeating: 1), density: 1, friction: 0.7,
            position: F3(1.3667879, 0.09363741, 2.559786),
            rotation: Quat(vector: SIMD4(
                -0.018475592, -0.20643722, -0.10983613, 0.97209996)),
            collisionEnabled: false)
        _ = scene.addConvexCollider(
            body: hullBody, asset: lowerBar, friction: 0.7)
        let boxBody = scene.addBody(
            size: F3(repeating: 1.1), density: 0, friction: 0.65,
            position: F3(2.3304725, -0.02716027, 0.5507069),
            rotation: Quat(vector: SIMD4(
                -0.00385641, 0.0015234109, -0.084362164, 0.9964265)))

        let solver = try GPUSolver(scene: scene)
        try solver.submitStep()
        try solver.synchronize()

        XCTAssertNil(solver.runtimeFailure)
        XCTAssertTrue(solver.activeRigidContactPairs().contains {
            Set([$0.0, $0.1]) == Set([hullBody, boxBody])
        })
    }

    func testNearTouchingHullHullGJKIsRelativeFrameStable() throws {
        try requireMetal()
        let source = Demos.convexDecomposition(scale: 1)
        let staticAssetID = try XCTUnwrap(source.colliders[2].convexAssetID)
        let dynamicAssetID = try XCTUnwrap(source.colliders[5].convexAssetID)
        let staticPosition = F3(-2.4, 0, 1.5)
        let dynamicPosition = F3(1.1929938, 0.078434, 2.48551)
        let dynamicRotation = Quat(vector: SIMD4(
            -0.004394626, -0.2529939, -0.15325804, 0.95524174))

        var scene = PhysicsScene(name: "convex-hull-relative-frame")
        scene.settings.gravity = 0
        let staticBody = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 0.7,
            position: staticPosition, collisionEnabled: false)
        _ = scene.addConvexCollider(
            body: staticBody, asset: source.convexAssets[staticAssetID])
        let dynamicBody = scene.addBody(
            size: F3(repeating: 1), density: 1, friction: 0.7,
            position: dynamicPosition, rotation: dynamicRotation,
            collisionEnabled: false)
        _ = scene.addConvexCollider(
            body: dynamicBody, asset: source.convexAssets[dynamicAssetID])

        let oracle = ConvexNarrowPhase.query(
            shapeA: .convexHull(
                vertices: source.convexAssets[staticAssetID].vertices),
            poseA: .init(
                position: staticPosition,
                orientation: Quat(real: 1, imag: .zero)),
            shapeB: .convexHull(
                vertices: source.convexAssets[dynamicAssetID].vertices),
            poseB: .init(
                position: dynamicPosition, orientation: dynamicRotation),
            options: .init(contactThreshold: scene.settings.collisionMargin))
        guard case let .success(reference) = oracle else {
            return XCTFail("captured pose must have a finite CPU oracle: \(oracle)")
        }
        XCTAssertGreaterThan(reference.signedDistance, 1.0e-4,
            "the fixed-point regression requires robust positive separation")
        XCTAssertLessThan(reference.signedDistance, scene.settings.collisionMargin,
            "the separated pose must remain inside the contact threshold")

        let solver = try GPUSolver(scene: scene)
        try solver.submitStep()
        try solver.synchronize()

        XCTAssertNil(solver.runtimeFailure)
        XCTAssertTrue(solver.activeRigidContactPairs().contains {
            Set([$0.0, $0.1]) == Set([staticBody, dynamicBody])
        })
    }

    func testNearTouchingHullHullRetainsCertifiedFallbackContact() throws {
        try requireMetal()
        let source = Demos.convexDecomposition(scale: 1)
        let staticAssetID = try XCTUnwrap(source.colliders[2].convexAssetID)
        let dynamicAssetID = try XCTUnwrap(source.colliders[5].convexAssetID)
        let staticPosition = F3(-2.4, 0, 1.5)
        let dynamicPosition = F3(1.1921202, 0.08611235, 2.4814003)
        let dynamicRotation = Quat(vector: SIMD4(
            -0.0062861666, -0.25287876, -0.15026297, 0.95573735))

        var scene = PhysicsScene(name: "convex-hull-mpr-retry")
        scene.settings.gravity = 0
        let staticBody = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 0.7,
            position: staticPosition, collisionEnabled: false)
        _ = scene.addConvexCollider(
            body: staticBody, asset: source.convexAssets[staticAssetID])
        let dynamicBody = scene.addBody(
            size: F3(repeating: 1), density: 1, friction: 0.7,
            position: dynamicPosition, rotation: dynamicRotation,
            collisionEnabled: false)
        _ = scene.addConvexCollider(
            body: dynamicBody, asset: source.convexAssets[dynamicAssetID])

        let oracle = ConvexNarrowPhase.query(
            shapeA: .convexHull(
                vertices: source.convexAssets[staticAssetID].vertices),
            poseA: .init(
                position: staticPosition,
                orientation: Quat(real: 1, imag: .zero)),
            shapeB: .convexHull(
                vertices: source.convexAssets[dynamicAssetID].vertices),
            poseB: .init(
                position: dynamicPosition, orientation: dynamicRotation),
            options: .init(contactThreshold: scene.settings.collisionMargin))
        guard case let .success(reference) = oracle else {
            return XCTFail(
                "the bounded query must retain a finite CPU oracle: \(oracle)")
        }
        // Stable support recentering can move this threshold pose between the
        // certified MPR and GJK branches without changing its true geometry.
        // The contract is the finite positive witness inside the detect band,
        // not which bounded algorithm produced it.
        XCTAssertGreaterThan(reference.signedDistance, 0)
        XCTAssertLessThan(reference.signedDistance,
                          scene.settings.collisionMargin)
        let witness = try XCTUnwrap(reference.contacts.first)
        XCTAssertEqual(
            dot(witness.pointB - witness.pointA, witness.normalAtoB),
            witness.signedDistance, accuracy: 5.0e-5)

        let solver = try GPUSolver(scene: scene)
        try solver.submitStep()
        try solver.synchronize()

        XCTAssertNil(solver.runtimeFailure)
        XCTAssertTrue(solver.activeRigidContactPairs().contains {
            Set([$0.0, $0.1]) == Set([staticBody, dynamicBody])
        })
    }

    func testGroundHullDuplicateUsesBoundedAdaptiveMPRRetry() throws {
        try requireMetal()
        let source = Demos.convexDecomposition(scale: 1)
        let sourceHull = source.colliders[5]
        let assetID = try XCTUnwrap(sourceHull.convexAssetID)
        let asset = source.convexAssets[assetID]
        let groundCollider = source.colliders[0]
        let groundPose = ConvexNarrowPhase.Pose(
            position: source.bodies[groundCollider.body].position
                + source.bodies[groundCollider.body].rotation.act(
                    groundCollider.localPosition),
            orientation: source.bodies[groundCollider.body].rotation
                * groundCollider.localRotation)
        let capturedHullColliderPosition = F3(
            1.2241138, -0.0728105, 2.377979)
        let hullBodyRotation = Quat(vector: SIMD4(
            -0.035863057, -0.30239967, -0.01683997, 0.9523574))
        let hullPose = ConvexNarrowPhase.Pose(
            position: capturedHullColliderPosition,
            orientation: hullBodyRotation)
        let options = ConvexNarrowPhase.Options(
            contactThreshold: source.settings.collisionMargin)

        let cpu = ConvexNarrowPhase.query(
            shapeA: .box(halfExtents: groundCollider.size * 0.5),
            poseA: groundPose,
            shapeB: .convexHull(vertices: asset.vertices),
            poseB: hullPose,
            options: options)
        guard case let .success(reference) = cpu else {
            return XCTFail("captured adaptive-retry CPU query failed: \(cpu)")
        }
        XCTAssertEqual(reference.signedDistance, 0.2613225,
                       accuracy: 5.0e-4)
        let witness = try XCTUnwrap(reference.contacts.first)
        XCTAssertEqual(
            dot(witness.pointB - witness.pointA, witness.normalAtoB),
            witness.signedDistance, accuracy: 5.0e-5)

        let swappedCPU = ConvexNarrowPhase.query(
            shapeA: .convexHull(vertices: asset.vertices),
            poseA: hullPose,
            shapeB: .box(halfExtents: groundCollider.size * 0.5),
            poseB: groundPose,
            options: options)
        guard case let .success(swappedReference) = swappedCPU else {
            return XCTFail("swapped captured CPU query failed: \(swappedCPU)")
        }
        XCTAssertEqual(swappedReference.signedDistance,
                       reference.signedDistance, accuracy: 5.0e-4)

        var scene = PhysicsScene(name: "ground-hull-adaptive-retry")
        scene.settings.gravity = 0
        scene.settings.collisionMargin = source.settings.collisionMargin
        let ground = scene.addBody(
            size: groundCollider.size, density: 0, friction: 0.7,
            position: groundPose.position,
            rotation: groundPose.orientation)
        let hull = scene.addBody(
            size: source.bodies[sourceHull.body].size,
            density: 1, friction: 0.7,
            position: capturedHullColliderPosition,
            rotation: hullBodyRotation,
            collisionEnabled: false)
        _ = scene.addConvexCollider(body: hull, asset: asset)

        let solver = try GPUSolver(scene: scene)
        try solver.submitStep()
        try solver.synchronize()

        XCTAssertNil(solver.runtimeFailure)
        XCTAssertGreaterThan(solver.lastNumPairs, 0)
        XCTAssertFalse(solver.activeRigidContactPairs().contains {
            Set([$0.0, $0.1]) == Set([ground, hull])
        }, "the certified corrected separation is outside the detect band")
    }

    func testInflatedMPRUsesCorrectedDistanceForAdmission() throws {
        try requireMetal()
        var scene = PhysicsScene(name: "convex-mpr-corrected-distance")
        scene.settings.gravity = 0
        scene.settings.collisionMargin = 1.0e-6
        let hullBody = scene.addBody(
            size: F3(repeating: 1), density: 1, friction: 0.5,
            position: .zero, collisionEnabled: false)
        _ = scene.addConvexCollider(body: hullBody, asset: try cubeAsset())
        let boxBody = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 0.5,
            position: F3(1.00009, 0, 0))

        let solver = try GPUSolver(scene: scene)
        try solver.submitStep()
        try solver.synchronize()

        XCTAssertNil(solver.runtimeFailure)
        XCTAssertFalse(solver.activeRigidContactPairs().contains {
            Set([$0.0, $0.1]) == Set([hullBody, boxBody])
        })
    }

    func testHullBoxOverlapIsInvariantAtMillionUnitOrigins() throws {
        let origins = [
            F3.zero,
            F3(repeating: 1_000_000),
            F3(repeating: -1_000_000),
        ]
        let snapshots = try origins.map {
            try translatedConvexPairSnapshot(
                origin: $0, otherIsHull: false, centerDistance: 0.75,
                collisionMargin: 0.01)
        }
        let baseline = try XCTUnwrap(snapshots[0])
        XCTAssertEqual(baseline.contactCount, 4)
        XCTAssertTrue(baseline.finite)
        for snapshot in snapshots.dropFirst() {
            XCTAssertEqual(try XCTUnwrap(snapshot), baseline,
                "MPR/manifold output must not depend on world translation")
        }
    }

    func testHullHullOverlapIsInvariantAtMillionUnitOrigins() throws {
        let origins = [
            F3.zero,
            F3(repeating: 1_000_000),
            F3(repeating: -1_000_000),
        ]
        let snapshots = try origins.map {
            try translatedConvexPairSnapshot(
                origin: $0, otherIsHull: true, centerDistance: 0.75,
                collisionMargin: 0.01)
        }
        let baseline = try XCTUnwrap(snapshots[0])
        XCTAssertEqual(baseline.contactCount, 4)
        XCTAssertTrue(baseline.finite)
        for snapshot in snapshots.dropFirst() {
            XCTAssertEqual(try XCTUnwrap(snapshot), baseline,
                "hull-hull overlap must retain its local-frame result")
        }
    }

    func testNearThresholdHullBoxSeparationIsInvariantAtMillionUnitOrigins()
        throws
    {
        // One Float ULP at 1e6 is 0.0625. Keep the true surfaces one ULP apart
        // and the detect margin just below it, so every origin exercises the
        // separated GJK control without asking Float storage to encode a gap
        // that the scene representation itself cannot preserve.
        for origin in [
            F3.zero,
            F3(repeating: 1_000_000),
            F3(repeating: -1_000_000),
        ] {
            let snapshot = try translatedConvexPairSnapshot(
                origin: origin, otherIsHull: false,
                centerDistance: 1.0625, collisionMargin: 0.05)
            XCTAssertNil(snapshot,
                "a one-ULP positive gap outside the detect band must stay separated")
        }
    }

    func testLargeTranslatedHullBoxKeepsNewtonQueryTolerances() throws {
        try requireMetal()
        let source = Demos.convexDecomposition(scale: 1)
        let assetID = try XCTUnwrap(source.colliders[7].convexAssetID)
        let scale: Float = 1_000
        let translation = F3(10_000, -8_000, 6_000)

        var scene = PhysicsScene(name: "convex-large-translated-query")
        scene.settings.gravity = 0
        let hullBody = scene.addBody(
            size: F3(repeating: scale), density: 1, friction: 0.7,
            position: translation
                + F3(1.5379953, 0.027668802, 2.5985816) * scale,
            rotation: Quat(vector: SIMD4(
                -0.009982864, -0.14968342, -0.0861807, 0.9849203)),
            collisionEnabled: false)
        _ = scene.addConvexCollider(
            body: hullBody,
            vertices: source.convexAssets[assetID].vertices.map { $0 * scale })
        let boxBody = scene.addBody(
            size: F3(repeating: 1.1 * scale), density: 0, friction: 0.65,
            position: translation
                + F3(2.3231678, -0.032141834, 0.55020523) * scale,
            rotation: Quat(vector: SIMD4(
                0.0014651066, -0.004046518, -0.07832974, 0.9969182)))

        let solver = try GPUSolver(scene: scene)
        try solver.submitStep()
        try solver.synchronize()

        XCTAssertNil(solver.runtimeFailure)
        XCTAssertGreaterThan(solver.lastNumPairs, 0,
            "the regression must execute the generic convex query")
        XCTAssertFalse(solver.activeRigidContactPairs().contains {
            Set([$0.0, $0.1]) == Set([hullBody, boxBody])
        })
    }

    func testLargeSourceFrameInlineHullUsesStableUploadCenter() throws {
        try requireMetal()
        let base = F3(repeating: 1_000_000)
        var vertices: [F3] = []
        for x in 0..<4 {
            for y in 0..<4 {
                for z in 0..<4 {
                    vertices.append(base + F3(
                        Float(x) / 3, Float(y) / 3, Float(z) / 3))
                }
            }
        }

        var scene = PhysicsScene(name: "convex-large-source-frame")
        scene.settings.gravity = 0
        scene.settings.collisionMargin = 0.02
        let hullBody = scene.addBody(
            size: F3(repeating: 1), density: 1, friction: 0.5,
            position: -base, collisionEnabled: false)
        scene.colliders.append(SceneCollider(
            body: hullBody, size: F3(repeating: 1), friction: 0.5,
            shape: .box, convexHullVertices: vertices,
            collisionEnabled: true, isRendered: false))
        let boxBody = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 0.5,
            position: F3(1.50142, 0.5, 0.5))

        let solver = try GPUSolver(scene: scene)
        let positions = solver.colliderLocalPosition.contents().bindMemory(
            to: SIMD4<Float>.self, capacity: scene.colliders.count)
        XCTAssertEqual(positions[0].x, base.x + 0.5)
        XCTAssertEqual(positions[0].y, base.y + 0.5)
        XCTAssertEqual(positions[0].z, base.z + 0.5)

        try solver.submitStep()
        try solver.synchronize()

        XCTAssertNil(solver.runtimeFailure)
        XCTAssertTrue(solver.activeRigidContactPairs().contains {
            Set([$0.0, $0.1]) == Set([hullBody, boxBody])
        })
    }

    func testAsymmetricInlineTetraUsesInteriorSeedInBothOrders() throws {
        try requireMetal()
        // The AABB midpoint is outside this simplex. GPUSolver must recenter
        // its support range about the vertex mean, while preserving the
        // collider's authored world geometry.
        let tetra = [
            F3(0, 0, 0), F3(4, 0, 0),
            F3(0, 2, 0), F3(0, 0, 1),
        ]
        let analyticKinds: [BodyShape] = [.box, .sphere]
        for kind in analyticKinds {
            for separated in [false, true] {
                for analyticFirst in [false, true] {
                    var scene = PhysicsScene(name: "asymmetric-tetra-seed")
                    scene.settings.gravity = 0
                    scene.settings.collisionMargin = 0.02
                    let analyticPosition = separated
                        ? F3(4.105, 0, 0) : F3(0.25, 0.25, 0.25)
                    var hullBody = -1
                    var analyticBody = -1

                    func addHull(_ scene: inout PhysicsScene) -> Int {
                        let body = scene.addBody(
                            size: F3(repeating: 1), density: 1,
                            friction: 0.5, position: .zero,
                            collisionEnabled: false)
                        _ = scene.addConvexCollider(
                            body: body, vertices: tetra, friction: 0.5)
                        return body
                    }
                    func addAnalytic(_ scene: inout PhysicsScene) -> Int {
                        scene.addBody(
                            size: F3(repeating: 0.2), density: 0,
                            friction: 0.5, position: analyticPosition,
                            shape: kind)
                    }
                    if analyticFirst {
                        analyticBody = addAnalytic(&scene)
                        hullBody = addHull(&scene)
                    } else {
                        hullBody = addHull(&scene)
                        analyticBody = addAnalytic(&scene)
                    }

                    let solver = try GPUSolver(scene: scene)
                    try solver.submitStep()
                    try solver.synchronize()

                    XCTAssertNil(solver.runtimeFailure)
                    XCTAssertGreaterThan(solver.lastNumPairs, 0,
                        "the near-separated fixtures must execute the query")
                    let active = solver.activeRigidContactPairs().contains {
                        Set([$0.0, $0.1]) == Set([hullBody, analyticBody])
                    }
                    XCTAssertTrue(active,
                        "overlap and margin-separated fixtures must be admitted")
                }
            }
        }
    }

    private struct EdgeKey: Hashable, Comparable {
        var a: UInt32
        var b: UInt32

        init(_ a: UInt32, _ b: UInt32) {
            self.a = min(a, b)
            self.b = max(a, b)
        }

        static func < (lhs: EdgeKey, rhs: EdgeKey) -> Bool {
            lhs.a == rhs.a ? lhs.b < rhs.b : lhs.a < rhs.a
        }
    }

    private let cubeVertices: [F3] = [
        F3(-0.5, -0.5, -0.5), F3(0.5, -0.5, -0.5),
        F3(-0.5, 0.5, -0.5), F3(0.5, 0.5, -0.5),
        F3(-0.5, -0.5, 0.5), F3(0.5, -0.5, 0.5),
        F3(-0.5, 0.5, 0.5), F3(0.5, 0.5, 0.5),
    ]

    private func requireMetal() throws {
        if MTLCreateSystemDefaultDevice() == nil {
            throw XCTSkip("Metal is unavailable")
        }
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

    private func cubeAsset() throws -> ConvexHullAsset {
        let vertices: [F3] = [
            F3(-0.5, -0.5, -0.5), F3(-0.5, -0.5, 0.5),
            F3(-0.5, 0.5, -0.5), F3(-0.5, 0.5, 0.5),
            F3(0.5, -0.5, -0.5), F3(0.5, -0.5, 0.5),
            F3(0.5, 0.5, -0.5), F3(0.5, 0.5, 0.5),
        ]
        let triangles: [SIMD3<UInt32>] = [
            SIMD3(0, 1, 3), SIMD3(0, 2, 6),
            SIMD3(0, 3, 2), SIMD3(0, 4, 5),
            SIMD3(0, 5, 1), SIMD3(0, 6, 4),
            SIMD3(1, 5, 7), SIMD3(1, 7, 3),
            SIMD3(2, 3, 7), SIMD3(2, 7, 6),
            SIMD3(4, 6, 7), SIMD3(4, 7, 5),
        ]
        var incident: [EdgeKey: [UInt32]] = [:]
        for (face, triangle) in triangles.enumerated() {
            for edge in [EdgeKey(triangle.x, triangle.y),
                         EdgeKey(triangle.y, triangle.z),
                         EdgeKey(triangle.z, triangle.x)] {
                incident[edge, default: []].append(UInt32(face))
            }
        }
        let edges = incident.keys.sorted().map { edge in
            let faces = incident[edge]!.sorted()
            return ConvexHullEdge(
                vertexA: edge.a, vertexB: edge.b,
                faceA: faces[0], faceB: faces[1])
        }
        let digest = ConvexHullAsset.geometryDigest(
            vertices: vertices, triangles: triangles)
        return try ConvexHullAsset(
            vertices: vertices, triangles: triangles, edges: edges,
            boundsMin: F3(repeating: -0.5),
            boundsMax: F3(repeating: 0.5),
            boundingRadius: sqrt(0.75), volume: 1,
            centroid: .zero, digest: digest,
            stableID: "hull-" + digest.prefix(16))
    }

    /// Maximum-topology convex polyhedron under the public 64-vertex contract:
    /// a 62-gon bipyramid has 124 triangular faces and 186 manifold edges.
    private func maximumEdgeBipyramidAsset() throws -> ConvexHullAsset {
        let ringCount = 62
        var rawVertices: [F3] = []
        rawVertices.reserveCapacity(ringCount + 2)
        for index in 0..<ringCount {
            let angle = 2 * Float.pi * Float(index) / Float(ringCount)
            rawVertices.append(F3(
                0.5 * cos(angle), 0.5 * sin(angle), 0))
        }
        let top = UInt32(rawVertices.count)
        rawVertices.append(F3(0, 0, 0.5))
        let bottom = UInt32(rawVertices.count)
        rawVertices.append(F3(0, 0, -0.5))

        var rawTriangles: [SIMD3<UInt32>] = []
        rawTriangles.reserveCapacity(2 * ringCount)
        for index in 0..<ringCount {
            let current = UInt32(index)
            let next = UInt32((index + 1) % ringCount)
            rawTriangles.append(SIMD3(top, current, next))
            rawTriangles.append(SIMD3(bottom, next, current))
        }
        let ordered = rawVertices.indices.sorted {
            let a = rawVertices[$0], b = rawVertices[$1]
            if a.x != b.x { return a.x < b.x }
            if a.y != b.y { return a.y < b.y }
            return a.z < b.z
        }
        let vertices = ordered.map { rawVertices[$0] }
        var remap = [UInt32](repeating: 0, count: rawVertices.count)
        for (canonical, source) in ordered.enumerated() {
            remap[source] = UInt32(canonical)
        }
        func triangleLess(
            _ a: SIMD3<UInt32>, _ b: SIMD3<UInt32>
        ) -> Bool {
            if a.x != b.x { return a.x < b.x }
            if a.y != b.y { return a.y < b.y }
            return a.z < b.z
        }
        let triangles = rawTriangles.map { triangle -> SIMD3<UInt32> in
            let remapped = SIMD3(
                remap[Int(triangle.x)], remap[Int(triangle.y)],
                remap[Int(triangle.z)])
            return [
                remapped,
                SIMD3(remapped.y, remapped.z, remapped.x),
                SIMD3(remapped.z, remapped.x, remapped.y),
            ].min(by: triangleLess)!
        }.sorted(by: triangleLess)
        var incident: [EdgeKey: [UInt32]] = [:]
        for (face, triangle) in triangles.enumerated() {
            for edge in [EdgeKey(triangle.x, triangle.y),
                         EdgeKey(triangle.y, triangle.z),
                         EdgeKey(triangle.z, triangle.x)] {
                incident[edge, default: []].append(UInt32(face))
            }
        }
        let edges = incident.keys.sorted().map { edge in
            let faces = incident[edge]!.sorted()
            return ConvexHullEdge(
                vertexA: edge.a, vertexB: edge.b,
                faceA: faces[0], faceB: faces[1])
        }
        let lo = vertices.reduce(F3(repeating: .infinity), simd.min)
        let hi = vertices.reduce(F3(repeating: -.infinity), simd.max)
        let polygonArea = Float(ringCount) * 0.125
            * sin(2 * Float.pi / Float(ringCount))
        let volume = polygonArea / 3
        let digest = ConvexHullAsset.geometryDigest(
            vertices: vertices, triangles: triangles)
        return try ConvexHullAsset(
            vertices: vertices, triangles: triangles, edges: edges,
            boundsMin: lo, boundsMax: hi, boundingRadius: 0.5,
            volume: volume, centroid: .zero, digest: digest,
            stableID: "hull-" + digest.prefix(16))
    }

    private struct ManifoldSnapshot: Equatable {
        var bodyA: Int
        var bodyB: Int
        var contactCount: Int
        var features: [SIMD2<UInt32>]
        var finite: Bool
    }

    private func manifoldSnapshots(_ solver: GPUSolver) -> [ManifoldSnapshot] {
        solver.sync()
        let manifolds = solver.prevManifolds.contents().bindMemory(
            to: ManifoldGPU.self, capacity: solver.maxPairs)
        let features = solver.prevContactFeatures.contents().bindMemory(
            to: SIMD2<UInt32>.self,
            capacity: solver.maxPairs * AVBD_MAX_CONTACTS)
        var snapshots: [ManifoldSnapshot] = []
        for pairIndex in 0..<solver.lastNumPairs {
            let manifold = manifolds[pairIndex]
            let count = min(Int(manifold.header.z), AVBD_MAX_CONTACTS)
            guard count > 0 else { continue }
            let keys = (0..<count).map {
                features[pairIndex * AVBD_MAX_CONTACTS + $0]
            }
            var contactTuple = manifold.contacts
            let finite = [manifold.basisN, manifold.basisT1].allSatisfy {
                $0.x.isFinite && $0.y.isFinite && $0.z.isFinite
                    && $0.w.isFinite
            } && withUnsafeBytes(of: &contactTuple) { bytes in
                bytes.bindMemory(to: ContactGPU.self).prefix(count).allSatisfy {
                    [$0.rA, $0.rB, $0.C0, $0.lambda, $0.penalty]
                        .allSatisfy { value in
                            value.x.isFinite && value.y.isFinite
                                && value.z.isFinite && value.w.isFinite
                        }
                }
            }
            snapshots.append(ManifoldSnapshot(
                bodyA: Int(manifold.header.x), bodyB: Int(manifold.header.y),
                contactCount: count, features: keys, finite: finite))
        }
        return snapshots
    }

    private func broadFaceScene(otherIsHull: Bool) throws -> PhysicsScene {
        var scene = PhysicsScene(name: "convex-broad-face")
        scene.settings.gravity = 0
        scene.settings.dt = 1 / 120
        scene.settings.iterations = 10
        scene.settings.collisionMargin = 0.01
        let asset = try cubeAsset()
        let base = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 0.8,
            position: F3(0, 0, -0.5), collisionEnabled: false)
        _ = scene.addConvexCollider(body: base, asset: asset, friction: 0.8)
        let upper = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 0.8,
            position: F3(0, 0, 0.49), mass: 1,
            diagonalInertia: F3(repeating: 1 / 6),
            collisionEnabled: !otherIsHull)
        if otherIsHull {
            _ = scene.addConvexCollider(
                body: upper, asset: asset, friction: 0.8)
        }
        return scene
    }

    private func translatedConvexPairSnapshot(
        origin: F3,
        otherIsHull: Bool,
        centerDistance: Float,
        collisionMargin: Float
    ) throws -> ManifoldSnapshot? {
        try requireMetal()
        var scene = PhysicsScene(name: "convex-query-origin-invariance")
        scene.settings.gravity = 0
        scene.settings.iterations = 0
        scene.settings.dt = 1 / 120
        scene.settings.collisionMargin = collisionMargin
        let asset = try cubeAsset()
        let base = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 0.8,
            position: origin, collisionEnabled: false)
        _ = scene.addConvexCollider(body: base, asset: asset, friction: 0.8)
        let upper = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 0.8,
            position: origin + F3(0, 0, centerDistance), mass: 1,
            diagonalInertia: F3(repeating: 1 / 6),
            collisionEnabled: !otherIsHull)
        if otherIsHull {
            _ = scene.addConvexCollider(
                body: upper, asset: asset, friction: 0.8)
        }

        let solver = try GPUSolver(scene: scene)
        try solver.submitStep()
        try solver.synchronize()
        XCTAssertNil(solver.runtimeFailure)
        XCTAssertGreaterThan(solver.lastNumPairs, 0,
            "the regression must execute the generic convex query")
        let matches = manifoldSnapshots(solver).filter {
            Set([$0.bodyA, $0.bodyB]) == Set([base, upper])
        }
        XCTAssertLessThanOrEqual(matches.count, 1)
        return matches.first
    }

    func testConvexMetalABILayoutsRemainStable() {
        XCTAssertEqual(MemoryLayout<ConvexHullGPU>.stride, 64)
        XCTAssertEqual(MemoryLayout<ConvexFaceGPU>.stride, 32)
        XCTAssertEqual(MemoryLayout<ConvexEdgeGPU>.stride, 16)
        XCTAssertEqual(MemoryLayout<ManifoldGPU>.stride, 704)
    }

    func testAnalyticSceneUsesPlaceholderExactFeatureStorage() throws {
        try requireMetal()
        var analytic = PhysicsScene(name: "analytic-feature-placeholder")
        _ = analytic.addBody(
            size: F3(repeating: 1), density: 1, friction: 0.5,
            position: .zero)
        let analyticSolver = try GPUSolver(scene: analytic)
        XCTAssertEqual(analyticSolver.convexColliderCount, 0)
        XCTAssertTrue(
            analyticSolver.usesHullFreeAnalyticCompatibilityKernelForTesting)
        XCTAssertEqual(analyticSolver.convexFeatureStorageByteCountForTesting,
                       32, "two bindable 16-byte Metal placeholders suffice")

        var convex = PhysicsScene(name: "convex-feature-storage")
        let body = convex.addBody(
            size: F3(repeating: 1), density: 1, friction: 0.5,
            position: .zero, collisionEnabled: false)
        _ = convex.addConvexCollider(body: body, asset: try cubeAsset())
        _ = convex.addBody(
            size: F3(repeating: 1), density: 0, friction: 0.5,
            position: F3(0, 0, -0.9))
        let convexSolver = try GPUSolver(scene: convex)
        XCTAssertFalse(
            convexSolver.usesHullFreeAnalyticCompatibilityKernelForTesting)
        XCTAssertGreaterThan(
            convexSolver.convexFeatureStorageByteCountForTesting, 32)

        var renderOnly = PhysicsScene(name: "render-only-convex-placeholder")
        let renderBody = renderOnly.addBody(
            size: F3(repeating: 1), density: 1, friction: 0.5,
            position: .zero, collisionEnabled: false)
        _ = renderOnly.addConvexCollider(
            body: renderBody, asset: try cubeAsset(),
            collisionEnabled: false, isRendered: true)
        let renderOnlySolver = try GPUSolver(scene: renderOnly)
        XCTAssertEqual(renderOnlySolver.convexColliderCount, 1)
        XCTAssertTrue(
            renderOnlySolver.usesHullFreeAnalyticCompatibilityKernelForTesting)
        XCTAssertEqual(
            renderOnlySolver.convexFeatureStorageByteCountForTesting, 32)

        var filtered = PhysicsScene(name: "filtered-convex-placeholder")
        _ = filtered.addBody(
            size: F3(repeating: 1), density: 0, friction: 0.5,
            position: .zero)
        _ = filtered.addBody(
            size: F3(repeating: 1), density: 1, friction: 0.5,
            position: F3(0, 0, 0.9))
        let isolated = filtered.addBody(
            size: F3(repeating: 1), density: 1, friction: 0.5,
            position: F3(8, 0, 0), collisionEnabled: false)
        _ = filtered.addConvexCollider(
            body: isolated, asset: try cubeAsset(), collisionGroup: 7,
            collidesWithSharedGeometry: false)
        let filteredSolver = try GPUSolver(scene: filtered)
        XCTAssertEqual(filteredSolver.convexColliderCount, 1)
        XCTAssertTrue(
            filteredSolver.usesHullFreeAnalyticCompatibilityKernelForTesting)
        XCTAssertEqual(
            filteredSolver.convexFeatureStorageByteCountForTesting, 32)
        try filteredSolver.submitStep()
        try filteredSolver.synchronize()
        XCTAssertNil(filteredSolver.runtimeFailure)
        XCTAssertGreaterThan(filteredSolver.lastNumPairs, 0,
            "the unrelated analytic pair must still execute")
    }

    func testRawColliderRejectsConflictingConvexSources() throws {
        try requireMetal()
        var scene = PhysicsScene(name: "gpu-conflicting-convex-sources")
        let body = scene.addBody(
            size: F3(repeating: 1), density: 1, friction: 0.5,
            position: .zero, collisionEnabled: false)
        let assetID = scene.registerConvexAsset(try cubeAsset())
        scene.colliders.append(SceneCollider(
            body: body, size: F3(repeating: 1), friction: 0.5,
            convexHullVertices: cubeVertices, convexAssetID: assetID))

        XCTAssertThrowsError(try GPUSolver(scene: scene)) { error in
            XCTAssertEqual(
                error as? GPUSolver.ConvexCollisionError,
                .conflictingAssetSources(collider: 0))
        }
    }

    func testRenderedSharedAndInlineHullsUseOpaqueRigidMeshPath() throws {
        try requireMetal()
        var scene = PhysicsScene(name: "opaque-convex-render-path")

        let sharedOffset = F3(1.75, -0.6, 0.35)
        let sharedAsset = try tetraAsset(offset: sharedOffset)
        let sharedBody = scene.addBody(
            size: F3(repeating: 1), density: 1, friction: 0.5,
            position: F3(-3, 0, 1), collisionEnabled: false)
        let sharedLocalPosition = F3(0.3, -0.2, 0.7)
        let sharedLocalRotation = Quat(
            angle: 0.63, axis: normalize(F3(0.2, 0.7, 0.4)))
        let sharedColor = F3(0.12, 0.46, 0.83)
        _ = scene.addConvexCollider(
            body: sharedBody, asset: sharedAsset,
            localPosition: sharedLocalPosition,
            localRotation: sharedLocalRotation,
            collisionEnabled: false, isRendered: true,
            renderColor: sharedColor)

        let inlineBody = scene.addBody(
            size: F3(repeating: 1), density: 1, friction: 0.5,
            position: F3(0, 0, 1), collisionEnabled: false)
        let inlineOffset = F3(-0.8, 1.1, 0.45)
        let inlineLocalPosition = F3(-0.25, 0.4, 0.2)
        let inlineLocalRotation = Quat(
            angle: -0.41, axis: normalize(F3(0.6, 0.1, 0.7)))
        let inlineCollider = scene.addConvexCollider(
            body: inlineBody,
            vertices: cubeVertices.map { $0 + inlineOffset },
            localPosition: inlineLocalPosition,
            localRotation: inlineLocalRotation,
            collisionEnabled: false, isRendered: true)
        let inlineColor = F3(0.78, 0.31, 0.16)
        scene.colliders[inlineCollider].renderColor = inlineColor

        let hiddenBody = scene.addBody(
            size: F3(repeating: 1), density: 1, friction: 0.5,
            position: F3(3, 0, 1), collisionEnabled: false)
        _ = scene.addConvexCollider(
            body: hiddenBody, asset: sharedAsset,
            collisionEnabled: true, isRendered: false)

        let analyticBody = scene.addBody(
            size: F3(repeating: 0.8), density: 1, friction: 0.5,
            position: F3(5, 0, 1))

        let solver = try GPUSolver(scene: scene)
        XCTAssertEqual(solver.rigidMeshVertexCount, 12 + 36)
        XCTAssertEqual(solver.renderRigidBodyCount, 1,
            "only the ordinary box belongs in the analytic instance stream")
        XCTAssertEqual(solver.convexDebugTriangleVertexCount, 12,
            "collision debug remains enabled-collider-only")
        XCTAssertEqual(solver.convexDebugEdgeVertexCount, 12)
        XCTAssertEqual(solver.materializedConvexDebugByteCount, 0)
        XCTAssertEqual(solver.rigidMeshUniqueVertexCount, 48,
            "faceted inline/convex visuals intentionally keep split normals")
        let indexed = try XCTUnwrap(solver.renderIndexedRigidMeshSurface)
        XCTAssertEqual(indexed.indexCount, 48)
        XCTAssertEqual(solver.materializedLegacyRigidMeshByteCount, 0)

        let surface = try XCTUnwrap(solver.renderRigidMeshSurface)
        XCTAssertEqual(surface.vertexCount, 48)
        XCTAssertEqual(
            solver.materializedLegacyRigidMeshByteCount,
            48 * MemoryLayout<RigidMeshVertexGPU>.stride)
        XCTAssertEqual(solver.materializedConvexDebugByteCount, 0,
            "requesting opaque visuals must not materialize debug buffers")
        let vertices = surface.vertices.contents().bindMemory(
            to: RigidMeshVertexGPU.self, capacity: surface.vertexCount)
        let bodyIDs = (0..<surface.vertexCount).map {
            vertices[$0].positionBody.w.bitPattern
        }
        XCTAssertEqual(bodyIDs.filter { $0 == UInt32(sharedBody) }.count, 12)
        XCTAssertEqual(bodyIDs.filter { $0 == UInt32(inlineBody) }.count, 36)
        XCTAssertFalse(bodyIDs.contains(UInt32(hiddenBody)))
        XCTAssertFalse(bodyIDs.contains(UInt32(analyticBody)))

        let expectedFirstShared = sharedLocalPosition
            + sharedLocalRotation.act(sharedAsset.vertices[0])
        let firstShared = F3(
            vertices[0].positionBody.x,
            vertices[0].positionBody.y,
            vertices[0].positionBody.z)
        XCTAssertEqual(firstShared.x, expectedFirstShared.x, accuracy: 1e-6)
        XCTAssertEqual(firstShared.y, expectedFirstShared.y, accuracy: 1e-6)
        XCTAssertEqual(firstShared.z, expectedFirstShared.z, accuracy: 1e-6)
        XCTAssertEqual(vertices[0].color.x, sharedColor.x, accuracy: 1e-7)
        XCTAssertEqual(vertices[0].color.y, sharedColor.y, accuracy: 1e-7)
        XCTAssertEqual(vertices[0].color.z, sharedColor.z, accuracy: 1e-7)
        XCTAssertEqual(vertices[12].color.x, inlineColor.x, accuracy: 1e-7)
        XCTAssertEqual(vertices[12].color.y, inlineColor.y, accuracy: 1e-7)
        XCTAssertEqual(vertices[12].color.z, inlineColor.z, accuracy: 1e-7)

        for index in 0..<surface.vertexCount {
            let vertex = vertices[index]
            XCTAssertTrue(vertex.positionBody.x.isFinite)
            XCTAssertTrue(vertex.positionBody.y.isFinite)
            XCTAssertTrue(vertex.positionBody.z.isFinite)
            XCTAssertEqual(length(F3(
                vertex.normal.x, vertex.normal.y, vertex.normal.z)),
                1, accuracy: 2e-6)
        }
    }

    func testAssetPayloadIsDeduplicatedAndSourceFrameIsCentered() throws {
        try requireMetal()
        var scene = PhysicsScene(name: "convex-asset-dedup")
        let bodyA = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 0.5,
            position: .zero, mass: 1,
            diagonalInertia: F3(repeating: 1), collisionEnabled: false)
        let bodyB = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 0.5,
            position: F3(10, 0, 0), mass: 1,
            diagonalInertia: F3(repeating: 1), collisionEnabled: false)
        let asset = try tetraAsset(offset: F3(2, 0, 0))
        _ = scene.addConvexCollider(body: bodyA, asset: asset)
        _ = scene.addConvexCollider(body: bodyB, asset: asset)

        let solver = try GPUSolver(scene: scene)
        XCTAssertEqual(solver.uniqueConvexAssetCount, 1)
        XCTAssertEqual(solver.convexColliderCount, 2)
        XCTAssertEqual(solver.numConvexHullVertices, 4)
        XCTAssertEqual(solver.convexDebugTriangleVertexCount, 24)
        XCTAssertEqual(solver.convexDebugEdgeVertexCount, 24)
        XCTAssertEqual(solver.materializedConvexDebugByteCount, 0,
                       "headless replicas must not expand debug geometry")
        let debugSurface = try XCTUnwrap(solver.renderConvexCollisionSurface)
        XCTAssertEqual(debugSurface.triangleVertexCount, 24)
        XCTAssertEqual(debugSurface.edgeVertexCount, 24)
        XCTAssertGreaterThan(solver.materializedConvexDebugByteCount, 0)

        let ranges = solver.colliderHullRange.contents().bindMemory(
            to: SIMD2<UInt32>.self, capacity: 2)
        XCTAssertEqual(ranges[0], ranges[1])
        let positions = solver.colliderLocalPosition.contents().bindMemory(
            to: SIMD4<Float>.self, capacity: 2)
        XCTAssertEqual(positions[0].x, 2, accuracy: 1e-6)
        XCTAssertEqual(positions[1].x, 2, accuracy: 1e-6)
    }

    func testLegacyInlineHullReconstructsCollisionAndDebugTopology() throws {
        try requireMetal()
        var scene = PhysicsScene(name: "legacy-inline-convex-topology")
        let body = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 0.5,
            position: .zero, collisionEnabled: false)
        _ = scene.addConvexCollider(
            body: body, vertices: cubeVertices, friction: 0.5)

        let solver = try GPUSolver(scene: scene)
        XCTAssertEqual(solver.uniqueConvexAssetCount, 1)
        let header = solver.convexHullHeaders.contents().bindMemory(
            to: ConvexHullGPU.self, capacity: 1)[0]
        XCTAssertEqual(header.verticesFaces.w, 6)
        XCTAssertEqual(header.edgesLoops.y, 12)
        XCTAssertEqual(header.edgesLoops.w, 24)
        XCTAssertEqual(solver.convexDebugTriangleVertexCount, 36)
        XCTAssertEqual(solver.convexDebugEdgeVertexCount, 24)
        XCTAssertNotNil(solver.renderConvexCollisionSurface)
    }

    func testCollisionDebugSurfaceOmitsDisabledConvexColliders() throws {
        try requireMetal()
        var scene = PhysicsScene(name: "convex-debug-enabled-only")
        let asset = try tetraAsset()
        let enabled = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 0.5,
            position: .zero, collisionEnabled: false)
        _ = scene.addConvexCollider(
            body: enabled, asset: asset, collisionEnabled: true)
        let disabled = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 0.5,
            position: F3(3, 0, 0), collisionEnabled: false)
        _ = scene.addConvexCollider(
            body: disabled, asset: asset, collisionEnabled: false)

        let solver = try GPUSolver(scene: scene)
        XCTAssertEqual(solver.uniqueConvexAssetCount, 1)
        XCTAssertEqual(solver.convexColliderCount, 2)
        XCTAssertEqual(solver.convexDebugTriangleVertexCount, 12)
        XCTAssertEqual(solver.convexDebugEdgeVertexCount, 12)
        XCTAssertEqual(solver.rigidMeshVertexCount, 0,
            "hidden convex collision must not expand opaque render geometry")
        XCTAssertNil(solver.renderRigidMeshSurface)
    }

    func testHullHullAndHullRoundPairsReachGenericNarrowPhase() throws {
        try requireMetal()
        for other in [BodyShape.box, .sphere, .capsule] {
            var scene = PhysicsScene(name: "convex-generic-\(other)")
            scene.settings.gravity = 0
            scene.settings.collisionMargin = 0.02
            let hullBody = scene.addBody(
                size: F3(repeating: 1), density: 0, friction: 0.5,
                position: .zero, collisionEnabled: false)
            _ = scene.addConvexCollider(
                body: hullBody, vertices: cubeVertices, friction: 0.5)

            let otherBody = scene.addBody(
                size: other == .capsule ? F3(0.4, 0.2, 0.2)
                    : F3(repeating: 0.7),
                density: 0, friction: 0.5, position: F3(0.55, 0, 0),
                shape: other, mass: 1,
                diagonalInertia: F3(repeating: 0.2))
            if other == .box {
                let collider = scene.colliders.count - 1
                scene.colliders[collider].collisionEnabled = false
                _ = scene.addConvexCollider(
                    body: otherBody, vertices: cubeVertices, friction: 0.5)
            }

            let solver = try GPUSolver(scene: scene)
            solver.step()
            _ = solver.bodyStates([otherBody])
            XCTAssertTrue(solver.activeRigidContactPairs().contains {
                Set([$0.0, $0.1]) == Set([hullBody, otherBody])
            }, "missing generic convex contact against \(other)")
        }
    }

    func testSplitNarrowPhasePreservesAnalyticAndConvexManifoldSlots() throws {
        try requireMetal()
        var scene = PhysicsScene(name: "split-rigid-narrowphase")
        scene.settings.gravity = 0

        let analyticStatic = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 0.5,
            position: F3(-4, 0, 0))
        let analyticDynamic = scene.addBody(
            size: F3(repeating: 1), density: 1, friction: 0.5,
            position: F3(-3.25, 0, 0))

        let hullStatic = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 0.5,
            position: F3(4, 0, 0), collisionEnabled: false)
        _ = scene.addConvexCollider(
            body: hullStatic, vertices: cubeVertices, friction: 0.5)
        let hullDynamic = scene.addBody(
            size: F3(repeating: 1), density: 1, friction: 0.5,
            position: F3(4.75, 0, 0), shape: .sphere)

        let solver = try GPUSolver(scene: scene)
        XCTAssertEqual(solver.convexColliderCount, 1)
        XCTAssertFalse(
            solver.usesHullFreeAnalyticCompatibilityKernelForTesting,
            "mixed scenes must use disjoint analytic/convex pair ownership")
        solver.step()
        try solver.synchronize()

        XCTAssertNil(solver.runtimeFailure)
        XCTAssertEqual(solver.lastNumPairs, 2,
            "the isolated rigs should occupy two stable pair/manifold slots")
        let pairs = solver.activeRigidContactPairs().map {
            Set([$0.0, $0.1])
        }
        XCTAssertTrue(pairs.contains(Set([
            analyticStatic, analyticDynamic
        ])), "the analytic pass must retain its ordinary primitive contact")
        XCTAssertTrue(pairs.contains(Set([
            hullStatic, hullDynamic
        ])), "the convex pass must fill its shared pair-indexed manifold slot")
    }

    func testHullSphereDispatchIsSymmetric() throws {
        try requireMetal()
        for hullFirst in [true, false] {
            var scene = PhysicsScene(name: "convex-symmetric-dispatch")
            scene.settings.gravity = 0
            let sphere: Int
            let hull: Int
            if hullFirst {
                hull = scene.addBody(
                    size: F3(repeating: 1), density: 0, friction: 0.5,
                    position: .zero, collisionEnabled: false)
                _ = scene.addConvexCollider(
                    body: hull, vertices: cubeVertices, friction: 0.5)
                sphere = scene.addBody(
                    size: F3(repeating: 0.7), density: 0, friction: 0.5,
                    position: F3(0.55, 0, 0), shape: .sphere,
                    mass: 1, diagonalInertia: F3(repeating: 0.2))
            } else {
                sphere = scene.addBody(
                    size: F3(repeating: 0.7), density: 0, friction: 0.5,
                    position: F3(0.55, 0, 0), shape: .sphere,
                    mass: 1, diagonalInertia: F3(repeating: 0.2))
                hull = scene.addBody(
                    size: F3(repeating: 1), density: 0, friction: 0.5,
                    position: .zero, collisionEnabled: false)
                _ = scene.addConvexCollider(
                    body: hull, vertices: cubeVertices, friction: 0.5)
            }
            let solver = try GPUSolver(scene: scene)
            solver.step()
            _ = solver.bodyStates([sphere])
            XCTAssertTrue(solver.activeRigidContactPairs().contains {
                Set([$0.0, $0.1]) == Set([hull, sphere])
            })
        }
    }

    func testHullHullBroadFaceBuildsDeterministicFourPointManifold() throws {
        try requireMetal()
        var runs: [[ManifoldSnapshot]] = []
        for _ in 0..<2 {
            let solver = try GPUSolver(
                scene: broadFaceScene(otherIsHull: true))
            let header = solver.convexHullHeaders.contents().bindMemory(
                to: ConvexHullGPU.self, capacity: 1)[0]
            XCTAssertEqual(header.verticesFaces.w, 6,
                           "coplanar cube triangles must merge into six faces")
            XCTAssertEqual(header.edgesLoops.y, 12,
                           "triangulation diagonals are not collision edges")
            XCTAssertEqual(header.edgesLoops.w, 24)
            solver.step()
            let snapshots = manifoldSnapshots(solver)
            XCTAssertEqual(snapshots.count, 1)
            XCTAssertEqual(snapshots.first?.contactCount, 4)
            XCTAssertEqual(Set(snapshots.first?.features ?? []).count, 4)
            XCTAssertTrue(snapshots.first?.finite == true)
            XCTAssertTrue(snapshots.first?.features.allSatisfy {
                $0.x != UInt32.max && $0.y != UInt32.max
            } == true)
            runs.append(snapshots)
        }
        XCTAssertEqual(runs[0], runs[1],
                       "canonical topology must produce repeatable keys")
    }

    func testHullBoxBroadFaceBuildsFourPointManifold() throws {
        try requireMetal()
        let solver = try GPUSolver(
            scene: broadFaceScene(otherIsHull: false))
        solver.step()
        let snapshots = manifoldSnapshots(solver)
        XCTAssertEqual(snapshots.count, 1)
        XCTAssertEqual(snapshots.first?.contactCount, 4)
        XCTAssertEqual(Set(snapshots.first?.features ?? []).count, 4)
        XCTAssertTrue(snapshots.first?.finite == true)
    }

    func testHullHullSupportingEdgesBuildOneStableContact() throws {
        try requireMetal()
        var runs: [[ManifoldSnapshot]] = []
        for _ in 0..<2 {
            var scene = PhysicsScene(name: "convex-supporting-edges")
            scene.settings.gravity = 0
            scene.settings.dt = 1 / 120
            scene.settings.collisionMargin = 0.01
            let asset = try cubeAsset()
            let a = scene.addBody(
                size: F3(repeating: 1), density: 0, friction: 0.5,
                position: .zero, collisionEnabled: false)
            _ = scene.addConvexCollider(body: a, asset: asset)
            let n = simd_normalize(F3(1, 1, 0))
            let rotation = (Quat(
                angle: -Float.pi / 4, axis: F3(0, 0, 1))
                * Quat(angle: Float.pi / 6, axis: F3(1, 0, 0))).normalized
            let b = scene.addBody(
                size: F3(repeating: 1), density: 0, friction: 0.5,
                position: n * 1.38, rotation: rotation,
                mass: 1, diagonalInertia: F3(repeating: 1 / 6),
                collisionEnabled: false)
            _ = scene.addConvexCollider(body: b, asset: asset)

            let solver = try GPUSolver(scene: scene)
            solver.step()
            let snapshots = manifoldSnapshots(solver)
            XCTAssertEqual(snapshots.count, 1)
            XCTAssertEqual(snapshots.first?.contactCount, 1)
            XCTAssertTrue(snapshots.first?.finite == true)
            let key = try XCTUnwrap(snapshots.first?.features.first)
            XCTAssertEqual(key.x & 0xF0000000, 0x60000000)
            XCTAssertEqual(key.y & 0xF0000000, 0x60000000)
            XCTAssertGreaterThan(solver.lastConvexEdgePairTests, 0)
            XCTAssertLessThanOrEqual(
                solver.lastConvexEdgePairTests,
                solver.lastPairCandidates * 12 * 12,
                "the bounded cube edge search must never exceed E_a * E_b")
            runs.append(snapshots)
        }
        XCTAssertEqual(runs[0], runs[1])
    }

    func testMaximumTopologyHullKeepsEdgeExpansionBounded() throws {
        try requireMetal()
        let asset = try maximumEdgeBipyramidAsset()
        XCTAssertEqual(asset.vertices.count, 64)
        XCTAssertEqual(asset.triangles.count, 124)
        XCTAssertEqual(asset.edges.count, 186)

        var scene = PhysicsScene(name: "maximum-edge-hull")
        scene.settings.gravity = 0
        scene.settings.iterations = 0
        scene.settings.collisionMargin = 0.02
        let base = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 0.6,
            position: .zero, collisionEnabled: false)
        _ = scene.addConvexCollider(body: base, asset: asset)
        let moving = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 0.6,
            position: F3(0.82, 0.03, 0.02),
            rotation: Quat(
                angle: 0.37, axis: normalize(F3(0.2, 0.7, 0.4))),
            mass: 1, diagonalInertia: F3(repeating: 0.2),
            collisionEnabled: false)
        _ = scene.addConvexCollider(body: moving, asset: asset)

        let solver = try GPUSolver(scene: scene)
        try solver.submitStep()
        try solver.synchronize()

        XCTAssertNil(solver.runtimeFailure)
        XCTAssertGreaterThan(solver.lastPairCandidates, 0)
        XCTAssertLessThanOrEqual(
            solver.lastConvexEdgePairTests,
            solver.lastPairCandidates
                * ConvexHullGPU.maximumSupportingEdgePairTests)
        XCTAssertTrue(manifoldSnapshots(solver).allSatisfy(\.finite))
    }

    func testRigidColliderHierarchyPrunesManyPartBodyPair() throws {
        try requireMetal()
        var scene = PhysicsScene(name: "many-part-bvh")
        scene.settings.gravity = 0
        scene.settings.iterations = 0
        scene.settings.collisionMargin = 0.01
        let asset = try cubeAsset()
        let base = scene.addBody(
            size: F3(12, 12, 1), density: 0, friction: 0.7,
            position: .zero, collisionEnabled: false)
        let moving = scene.addBody(
            size: F3(12, 12, 1), density: 0, friction: 0.7,
            position: F3(0, 0, 0.9), mass: 1,
            diagonalInertia: F3(repeating: 10), collisionEnabled: false)
        for y in 0..<8 {
            for x in 0..<8 {
                let local = F3(
                    (Float(x) - 3.5) * 1.25,
                    (Float(y) - 3.5) * 1.25, 0)
                _ = scene.addConvexCollider(
                    body: base, asset: asset, localPosition: local)
                _ = scene.addConvexCollider(
                    body: moving, asset: asset, localPosition: local)
            }
        }

        let solver = try GPUSolver(scene: scene)
        XCTAssertTrue(solver.usesRigidColliderHierarchy)
        XCTAssertTrue(GPUSolver.requiredHierarchyKernelNames
            .isSubset(of: solver.pso.keys))
        XCTAssertGreaterThan(solver.rigidBroadphaseProxyCount, 2,
            "large compounds must distribute expansion across multiple proxies")
        XCTAssertEqual(solver.rigidBroadphaseBVHNodeCount, 254)
        try solver.submitStep()
        try solver.synchronize()

        XCTAssertNil(solver.runtimeFailure)
        XCTAssertGreaterThan(solver.lastPairCandidates, 0)
        XCTAssertLessThan(solver.lastPairCandidates, 1_024,
            "BVH leaf pruning must stay far below the 64×64 child product")
        XCTAssertTrue(solver.activeRigidContactPairs().contains {
            Set([$0.0, $0.1]) == Set([base, moving])
        })
        XCTAssertTrue(manifoldSnapshots(solver).allSatisfy(\.finite))
    }

    func testHierarchyFinalizationHasExactlyOneWriter() throws {
        try requireMetal()
        var scene = PhysicsScene(name: "hierarchy-finalizer-writer")
        let body = scene.addBody(size: F3(repeating: 1), density: 1,
            friction: 0.5, position: .zero, collisionEnabled: false)
        for _ in 0..<2 { scene.addCollider(body: body, size: F3(repeating: 0.5)) }
        let solver = try GPUSolver(scene: scene)
        let device = solver.device
        let counts = try XCTUnwrap(device.makeBuffer(length: 4096 * 4, options: .storageModeShared))
        let starts = try XCTUnwrap(device.makeBuffer(length: 4096 * 4, options: .storageModeShared))
        memset(counts.contents(), 0, counts.length)
        memset(starts.contents(), 0, starts.length)
        counts.contents().assumingMemoryBound(to: UInt32.self)[0] = 256
        // If another invocation mistakes the published contact count for a
        // proxy count, this valid allocation leads it to a zero sentinel.
        let counters = solver.counters.contents().assumingMemoryBound(to: UInt32.self)
        counters[GPUCounters.pairs] = 1
        counters[GPUCounters.pairCandidates] = 1
        let cmd = try XCTUnwrap(solver.queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(cmd.makeComputeCommandEncoder())
        encoder.setComputePipelineState(try XCTUnwrap(solver.pso["bp_finalize_hierarchy_pairs"]))
        encoder.setBuffer(counts, offset: 0, index: 0)
        encoder.setBuffer(starts, offset: 0, index: 1)
        encoder.setBuffer(solver.counters, offset: 0, index: 2)
        encoder.setBuffer(solver.dispatchArgs, offset: 0, index: 3)
        var params = solver.params
        encoder.setBytes(&params, length: MemoryLayout<SimParamsGPU>.stride, index: 4)
        // Exercise multiple scheduling waves with an oversized dispatch;
        // production dispatch1D also launches multiple SIMD groups.
        encoder.dispatchThreadgroups(MTLSize(width: 4096, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit(); cmd.waitUntilCompleted()
        XCTAssertEqual(cmd.status, .completed)
        XCTAssertEqual(counters[GPUCounters.pairs], 256)
        XCTAssertEqual(counters[GPUCounters.pairCandidates], 256)
        XCTAssertEqual(solver.dispatchArgs.contents().assumingMemoryBound(to: UInt32.self)[0], 4)
    }

    func testHierarchyScratchUsesBothProducerAndProxyPairBounds() throws {
        try requireMetal()
        for bodyCount in [1, 2, 5] {
            var scene = PhysicsScene(name: "hierarchy-small-proxy-scratch")
            scene.settings.gravity = 0
            scene.settings.iterations = 0
            for _ in 0..<bodyCount {
                let body = scene.addBody(size: F3(repeating: 1), density: 1,
                    friction: 0.5, position: .zero, collisionEnabled: false)
                for _ in 0..<2 {
                    scene.addCollider(body: body, size: F3(repeating: 0.5), shape: .sphere)
                }
            }
            let solver = try GPUSolver(scene: scene, maxPairsPerBody: 128)
            XCTAssertEqual(solver.rigidBroadphaseProxyCount, bodyCount)
            let pairCapacity = max(1, bodyCount * (bodyCount - 1) / 2)
            let scanCapacity = max(pairCapacity, bodyCount)
            XCTAssertEqual(solver.broadphaseProxyPairs.length, max(16, pairCapacity * 8))
            XCTAssertEqual(solver.pairCount.length, max(16, scanCapacity * 4))
            XCTAssertEqual(solver.pairStart.length, max(16, scanCapacity * 4))
            try solver.submitStep()
            try solver.synchronize()
            XCTAssertEqual(solver.lastPairCandidates, bodyCount * (bodyCount - 1) / 2 * 4)
        }
    }

    func testHierarchyRejectsProxyOverflowBeforeExpandingMissingPairs() throws {
        try requireMetal()
        var scene = PhysicsScene(name: "hierarchy-proxy-overflow")
        scene.settings.gravity = 0
        scene.settings.iterations = 0
        for _ in 0..<100 {
            let body = scene.addBody(size: F3(repeating: 1), density: 1,
                friction: 0.5, position: .zero, collisionEnabled: false)
            for _ in 0..<2 {
                scene.addCollider(body: body, size: F3(repeating: 0.5), shape: .sphere)
            }
        }
        let solver = try GPUSolver(scene: scene)
        XCTAssertEqual(solver.hierarchyPairCapacity, 4096)
        // 4,950 eligible proxy pairs exceed the cache's 4,096 entries;
        // expansion must read only initialized pairs and fail explicitly.
        try solver.submitStep()
        XCTAssertThrowsError(try solver.synchronize()) { error in
            guard case let GPUSolver.RuntimeFailure.rigidPairCapacity(frame, required, capacity) = error else {
                return XCTFail("unexpected failure: \(error)")
            }
            XCTAssertEqual(frame, 1)
            XCTAssertEqual(capacity, 4096)
            XCTAssertGreaterThan(required, capacity)
        }
    }

    func testHierarchySubtreesPreserveAllEligibleSpherePairs() throws {
        try requireMetal()
        var scene = PhysicsScene(name: "compound-subtree-pair-oracle")
        scene.settings.gravity = 0
        scene.settings.iterations = 0
        scene.settings.collisionMargin = 0.007
        let asset = try tetraAsset(offset: F3(0.07, -0.11, 0.03))
        var owners: [Int] = []
        // Unequal, non-power-of-two leaf counts exercise both frontier
        // branches. Rotated, mixed analytic/convex compounds cover radius
        // padding, local transforms, collision domains and owner exclusions.
        for (bodyIndex, count) in [67, 35, 19, 1].enumerated() {
            let body = scene.addBody(
                size: F3(repeating: 1), density: bodyIndex == 0 ? 0 : 1, friction: 0.5,
                position: F3(Float(bodyIndex) * 0.31, 0.17, 0.23),
                rotation: Quat(angle: Float(bodyIndex) * 0.19,
                               axis: normalize(F3(1, 2, 3))),
                collisionEnabled: false)
            owners.append(body)
            for index in 0..<count {
                let local = F3(Float(index % 7) * 0.61,
                               Float(index / 7) * 0.53, 0.13)
                let collider: Int
                if index % 3 == 0 {
                    collider = scene.addConvexCollider(
                        body: body, asset: asset, localPosition: local,
                        localRotation: Quat(angle: 0.21, axis: F3(0, 1, 0)))
                } else {
                    collider = scene.addCollider(
                        body: body, size: F3(0.83, 0.09, 0.07),
                        localPosition: local,
                        localRotation: Quat(angle: 0.37, axis: F3(1, 0, 0)))
                }
                scene.colliders[collider].collisionGroup = bodyIndex == 0
                    ? 0 : (index % 5 == 0 ? 2 : 1)
                scene.colliders[collider].collidesWithSharedGeometry = index % 4 != 0
            }
        }
        scene.addCollisionExclusion(bodyA: owners[1], bodyB: owners[3])
        let solver = try GPUSolver(scene: scene, maxPairsPerBody: 64)
        let nodeBuffer = solver.broadphaseBVHNodes.contents()
            .assumingMemoryBound(to: ColliderBVHNodeGPU.self)
        let roots = solver.broadphaseProxyRoot.contents()
            .assumingMemoryBound(to: UInt32.self)
        func leafIDs(_ root: UInt32) -> [UInt32] {
            let node = nodeBuffer[Int(root)]
            if node.links.w & 1 != 0 { return [node.links.z] }
            return leafIDs(node.links.x) + leafIDs(node.links.y)
        }
        var covered: [UInt32] = []
        for proxy in 0..<solver.rigidBroadphaseProxyCount {
            let leaves = leafIDs(roots[proxy])
            XCTAssertLessThanOrEqual(leaves.count,
                                    GPUSolver.rigidBroadphaseMaxLeavesPerProxy)
            covered += leaves
        }
        XCTAssertEqual(covered.sorted(), scene.colliders.indices
            .filter { scene.colliders[$0].collisionEnabled }.map(UInt32.init))

        let shape = solver.colliderShape.contents().assumingMemoryBound(to: SIMD4<Float>.self)
        let local = solver.colliderLocalPosition.contents().assumingMemoryBound(to: SIMD4<Float>.self)
        let shapeType = solver.colliderShapeType.contents().assumingMemoryBound(to: UInt32.self)
        func center(_ index: Int) -> F3 {
            let body = scene.bodies[scene.colliders[index].body]
            let p = local[index]
            return body.position + body.rotation.act(F3(p.x, p.y, p.z))
        }
        var expected = Set<SIMD2<UInt32>>()
        for a in scene.colliders.indices {
            for b in scene.colliders.indices where b > a {
                guard scene.canPotentiallyCollide(colliderA: a, colliderB: b) else { continue }
                let ra = abs(shape[a].w), rb = abs(shape[b].w)
                var radius = ra + rb
                if shapeType[a] & 15 == 4 || shapeType[b] & 15 == 4 {
                    radius += scene.settings.collisionMargin + min(0.25,
                        max(4 * scene.settings.collisionMargin, 3 * min(ra, rb)))
                }
                if simd_length_squared(center(a) - center(b)) <= radius * radius {
                    expected.insert(SIMD2(UInt32(a), UInt32(b)))
                }
            }
        }
        XCTAssertFalse(expected.isEmpty)
        let initial = solver.captureSimulationSnapshot()
        var first: [SIMD2<UInt32>]?
        for _ in 0..<3 {
            solver.restoreSimulationSnapshot(initial)
            try solver.submitStep()
            try solver.synchronize()
            XCTAssertNil(solver.runtimeFailure)
            let emitted = Array(UnsafeBufferPointer(
                start: solver.pairs.contents().assumingMemoryBound(to: SIMD2<UInt32>.self),
                count: solver.lastNumPairs))
            XCTAssertEqual(emitted.count, Set(emitted).count, "no duplicate collider pairs")
            XCTAssertTrue(Set(emitted) == expected,
                "subtrees must preserve all \(expected.count) pairs, got \(emitted.count)")
            if let first { XCTAssertTrue(emitted == first, "stable emission order") }
            else { first = emitted }
        }
    }

    func testHierarchyCachedPairsHandleFullTilesEmptyFramesAndOverflow() throws {
        try requireMetal()
        for leafCount in [16, 64, 65] {
            var scene = PhysicsScene(name: "hierarchy-cache-capacity-\(leafCount)")
            scene.settings.gravity = 0
            scene.settings.iterations = 0
            let base = scene.addBody(size: F3(repeating: 1), density: 0,
                friction: 0.5, position: .zero, collisionEnabled: false)
            let moving = scene.addBody(size: F3(repeating: 1), density: 1,
                friction: 0.5, position: F3(0, 0, 0.1), collisionEnabled: false)
            for index in 0..<leafCount {
                for body in [base, moving] {
                    scene.addCollider(body: body, size: F3(repeating: 0.5),
                        localPosition: F3(Float(index) * 0.0001, 0, 0), shape: .sphere)
                }
            }
            let solver = try GPUSolver(scene: scene)
            XCTAssertLessThan(solver.hierarchyPairCapacity, solver.maxPairs)
            try solver.submitStep()
            let required = leafCount * leafCount
            if required > solver.maxPairs {
                XCTAssertThrowsError(try solver.synchronize()) { error in
                    XCTAssertEqual(error as? GPUSolver.RuntimeFailure,
                        .rigidPairCapacity(frame: 1, required: required, capacity: 4096))
                }
                continue
            }
            try solver.synchronize()
            XCTAssertEqual(solver.lastPairCandidates, required)
            let emitted = Array(UnsafeBufferPointer(
                start: solver.pairs.contents().assumingMemoryBound(to: SIMD2<UInt32>.self),
                count: solver.lastNumPairs))
            XCTAssertEqual(Set(emitted).count, required,
                "all 256 entries in full tiles must decode to distinct pairs")
            // Shrink the live proxy list to zero after a full cache, then
            // restore contacts. Finalization must not read stale scan tails.
            solver.setBodyPose(moving, position: F3(100, 0, 0),
                               rotation: Quat(real: 1, imag: .zero))
            try solver.submitStep()
            try solver.synchronize()
            XCTAssertEqual(solver.lastPairCandidates, 0)
            solver.setBodyPose(moving, position: F3(0, 0, 0.1),
                               rotation: Quat(real: 1, imag: .zero))
            solver.profiling = true
            try solver.submitStep()
            try solver.synchronize()
            XCTAssertEqual(solver.lastPairCandidates, required)
        }
    }

    func testRigidColliderHierarchySupportsBodyAcrossCompoundSeam() throws {
        try requireMetal()
        var scene = PhysicsScene(name: "compound-bvh-seam-support")
        scene.settings.dt = 1 / 120
        scene.settings.gravity = -9.81
        scene.settings.iterations = 20
        scene.settings.collisionMargin = 0.01
        let asset = try cubeAsset()
        let table = scene.addBody(
            size: F3(2, 1, 1), density: 0, friction: 1,
            position: F3(0, 0, -0.5), collisionEnabled: false)
        _ = scene.addConvexCollider(
            body: table, asset: asset, friction: 1,
            localPosition: F3(-0.5, 0, 0))
        _ = scene.addConvexCollider(
            body: table, asset: asset, friction: 1,
            localPosition: F3(0.5, 0, 0))
        let falling = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 1,
            position: F3(0, 0, 1.6), mass: 1,
            diagonalInertia: F3(repeating: 1 / 6),
            collisionEnabled: false)
        _ = scene.addConvexCollider(body: falling, asset: asset, friction: 1)

        let solver = try GPUSolver(scene: scene)
        XCTAssertTrue(solver.usesRigidColliderHierarchy)
        for _ in 0..<300 { solver.step() }
        let state = solver.bodyStates([falling])[0]

        XCTAssertNil(solver.runtimeFailure)
        XCTAssertTrue(state.position.x.isFinite && state.position.z.isFinite)
        XCTAssertEqual(state.position.z, 0.5, accuracy: 0.08)
        XCTAssertLessThan(abs(state.position.x), 0.08)
        XCTAssertLessThan(abs(state.rotation.imag.x), 0.08)
        XCTAssertLessThan(abs(state.rotation.imag.y), 0.08)
        XCTAssertTrue(solver.activeRigidContactPairs().contains {
            Set([$0.0, $0.1]) == Set([table, falling])
        })
        XCTAssertTrue(manifoldSnapshots(solver).allSatisfy(\.finite))
    }

    func testDeterministicCPUAndGPUConvexParityCorpus() throws {
        try requireMetal()
        var scene = PhysicsScene(name: "convex-parity-corpus")
        scene.settings.gravity = 0
        scene.settings.iterations = 0
        scene.settings.collisionMargin = 0.01
        let asset = try cubeAsset()
        var expected: [(UInt64, Bool, Float)] = []
        var seed: UInt64 = 0xA17E_C0DE_5EED
        func randomUnit() -> Float {
            seed = seed &* 6_364_136_223_846_793_005 &+ 1
            return Float((seed >> 40) & 0xFFFFFF) / Float(0xFFFFFF)
        }
        func pairKey(_ a: Int, _ b: Int) -> UInt64 {
            UInt64(min(a, b)) << 32 | UInt64(max(a, b))
        }

        for index in 0..<24 {
            let group = UInt32(index + 1)
            let origin = F3(Float(index) * 4, 0, 0)
            let yawA = (randomUnit() - 0.5) * 2.5
            let yawB = (randomUnit() - 0.5) * 2.5
            let rotationA = Quat(angle: yawA, axis: F3(0, 0, 1))
            let rotationB = Quat(angle: yawB, axis: F3(0, 0, 1))
            let delta: Float = [-0.05, 0.005, 0.06][index % 3]
            let otherKind = index % 4
            let nominalDistance: Float
            let cpuShapeB: ConvexNarrowPhase.Shape
            switch otherKind {
            case 0:
                nominalDistance = 0.95
                cpuShapeB = .box(halfExtents: F3(repeating: 0.45))
            case 1:
                nominalDistance = 1.0
                cpuShapeB = .sphere(radius: 0.5)
            case 2:
                nominalDistance = 1.2
                cpuShapeB = .capsule(radius: 0.25, halfHeight: 0.45)
            default:
                nominalDistance = 1.0
                cpuShapeB = .convexHull(vertices: asset.vertices)
            }
            let positionB = origin + F3(0, 0, nominalDistance + delta)
            let cpu = ConvexNarrowPhase.query(
                shapeA: .convexHull(vertices: asset.vertices),
                poseA: .init(position: origin, orientation: rotationA),
                shapeB: cpuShapeB,
                poseB: .init(position: positionB, orientation: rotationB),
                options: .init(contactThreshold: scene.settings.collisionMargin))
            guard case let .success(manifold) = cpu else {
                return XCTFail("CPU corpus query \(index) failed: \(cpu)")
            }

            let bodyA = scene.addBody(
                size: F3(repeating: 1), density: 0, friction: 0.6,
                position: origin, rotation: rotationA,
                collisionEnabled: false)
            _ = scene.addConvexCollider(
                body: bodyA, asset: asset, collisionGroup: group,
                collidesWithSharedGeometry: false)
            let bodyB = scene.addBody(
                size: F3(repeating: 1), density: 0, friction: 0.6,
                position: positionB, rotation: rotationB, mass: 1,
                diagonalInertia: F3(repeating: 0.2),
                collisionEnabled: false)
            switch otherKind {
            case 0:
                _ = scene.addCollider(
                    body: bodyB, size: F3(repeating: 0.9), shape: .box,
                    collisionGroup: group,
                    collidesWithSharedGeometry: false)
            case 1:
                _ = scene.addCollider(
                    body: bodyB, size: F3(repeating: 1), shape: .sphere,
                    collisionGroup: group,
                    collidesWithSharedGeometry: false)
            case 2:
                _ = scene.addCollider(
                    body: bodyB, size: F3(0.9, 0.25, 0), shape: .capsule,
                    collisionGroup: group,
                    collidesWithSharedGeometry: false)
            default:
                _ = scene.addConvexCollider(
                    body: bodyB, asset: asset, collisionGroup: group,
                    collidesWithSharedGeometry: false)
            }
            expected.append((
                pairKey(bodyA, bodyB),
                manifold.signedDistance <= scene.settings.collisionMargin,
                manifold.signedDistance))
        }

        let solver = try GPUSolver(scene: scene)
        try solver.submitStep()
        try solver.synchronize()
        XCTAssertNil(solver.runtimeFailure)
        let active = Set(solver.activeRigidContactPairs().map {
            pairKey($0.0, $0.1)
        })
        XCTAssertTrue(expected.contains(where: \.1))
        XCTAssertTrue(expected.contains { !$0.1 })
        for (key, shouldContact, distance) in expected {
            XCTAssertEqual(active.contains(key), shouldContact,
                "CPU/GPU admission mismatch at signed distance \(distance)")
        }
    }

    func testReorderedClippedPatchWarmStartUsesBothAnchors() throws {
        try requireMetal()
        var scene = try broadFaceScene(otherIsHull: true)
        scene.settings.iterations = 0
        let solver = try GPUSolver(scene: scene)
        solver.step()
        try solver.synchronize()

        let previous = solver.prevManifolds.contents().bindMemory(
            to: ManifoldGPU.self, capacity: solver.maxPairs)
        XCTAssertEqual(solver.lastNumPairs, 1)
        XCTAssertGreaterThanOrEqual(previous[0].header.z, 2)
        previous[0].header.z = 2
        withUnsafeMutableBytes(of: &previous[0].contacts) { bytes in
            let contacts = bytes.bindMemory(to: ContactGPU.self)
            let current = contacts[0]
            contacts[0] = current
            contacts[0].rB.x += 0.08
            contacts[0].rB.w = 0
            contacts[0].lambda = SIMD4(1, 0, 0, 0)
            contacts[1] = current
            contacts[1].rA.x += 0.02
            contacts[1].rB.w = 0
            contacts[1].lambda = SIMD4(9, 0, 0, 0)
        }
        let previousFeatures = solver.prevContactFeatures.contents().bindMemory(
            to: SIMD2<UInt32>.self,
            capacity: solver.maxPairs * AVBD_MAX_CONTACTS)
        previousFeatures[0] = SIMD2(0xFFFF_FFF0, 0xFFFF_FFF1)
        previousFeatures[1] = SIMD2(0xFFFF_FFF2, 0xFFFF_FFF3)

        solver.step()
        try solver.synchronize()
        let current = solver.prevManifolds.contents().bindMemory(
            to: ManifoldGPU.self, capacity: solver.maxPairs)
        withUnsafeBytes(of: current[0].contacts) { bytes in
            let contacts = bytes.bindMemory(to: ContactGPU.self)
            XCTAssertGreaterThan(contacts[0].lambda.x, 4,
                "the symmetric metric must select the close rA/rB candidate")
        }
    }

    func testConvexHullRestsOnConvexHull() throws {
        try requireMetal()
        var scene = PhysicsScene(name: "convex-on-convex-rest")
        scene.settings.dt = 1 / 120
        scene.settings.gravity = -9.81
        scene.settings.iterations = 20
        let base = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 1,
            position: F3(0, 0, -0.5), collisionEnabled: false)
        _ = scene.addConvexCollider(
            body: base, vertices: cubeVertices, friction: 1)
        let falling = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 1,
            position: F3(0, 0, 1.5), mass: 1,
            diagonalInertia: F3(repeating: 1 / 6),
            collisionEnabled: false)
        _ = scene.addConvexCollider(
            body: falling, vertices: cubeVertices, friction: 1)

        let solver = try GPUSolver(scene: scene)
        for _ in 0..<360 { solver.step() }
        let state = solver.bodyStates([falling])[0]
        XCTAssertEqual(state.position.z, 0.5, accuracy: 0.08)
        XCTAssertLessThan(abs(state.rotation.imag.x), 0.08)
        XCTAssertLessThan(abs(state.rotation.imag.y), 0.08)
    }

    func testConvexHullBroadFaceStackRemainsFiniteAndUpright() throws {
        try requireMetal()
        var scene = PhysicsScene(name: "convex-broad-face-stack")
        scene.settings.dt = 1 / 120
        scene.settings.gravity = -9.81
        scene.settings.iterations = 20
        scene.settings.collisionMargin = 0.01
        let asset = try cubeAsset()
        let ground = scene.addBody(
            size: F3(8, 8, 1), density: 0, friction: 1,
            position: F3(0, 0, -0.5), shape: .box)
        var bodies: [Int] = []
        for level in 0..<4 {
            let body = scene.addBody(
                size: F3(repeating: 1), density: 0, friction: 1,
                position: F3(0, 0, 0.51 + Float(level) * 1.01),
                mass: 1, diagonalInertia: F3(repeating: 1 / 6),
                collisionEnabled: false)
            _ = scene.addConvexCollider(body: body, asset: asset, friction: 1)
            bodies.append(body)
        }

        let solver = try GPUSolver(scene: scene)
        for _ in 0..<300 { solver.step() }
        let states = solver.bodyStates(bodies)
        for (level, state) in states.enumerated() {
            XCTAssertTrue(state.position.x.isFinite
                && state.position.y.isFinite && state.position.z.isFinite)
            XCTAssertEqual(state.position.z, 0.5 + Float(level), accuracy: 0.13)
            XCTAssertLessThan(abs(state.rotation.imag.x), 0.12)
            XCTAssertLessThan(abs(state.rotation.imag.y), 0.12)
        }
        let snapshots = manifoldSnapshots(solver).filter {
            $0.bodyA == ground || $0.bodyB == ground
                || bodies.contains($0.bodyA) || bodies.contains($0.bodyB)
        }
        XCTAssertFalse(snapshots.isEmpty)
        XCTAssertTrue(snapshots.allSatisfy {
            $0.finite && (1...AVBD_MAX_CONTACTS).contains($0.contactCount)
                && Set($0.features).count == $0.features.count
        })
    }

    func testAnalyticBroadphasePreservesRawRadiusCandidateSemantics() throws {
        try requireMetal()
        var scene = PhysicsScene(name: "analytic-raw-radius-broadphase")
        scene.settings.gravity = 0
        scene.settings.collisionMargin = 0.02
        _ = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 0,
            position: .zero, shape: .sphere)
        _ = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 0,
            position: F3(1.01, 0, 0), shape: .sphere,
            mass: 1, diagonalInertia: F3(repeating: 0.1))
        let solver = try GPUSolver(scene: scene)
        solver.step()
        try solver.synchronize()
        XCTAssertEqual(solver.lastPairCandidates, 0,
            "analytic-only broadphase must retain origin/main raw bounds")
    }

    func testHullBroadphaseRetainsPairInsideCollisionMargin() throws {
        try requireMetal()
        var scene = PhysicsScene(name: "hull-margin-broadphase")
        scene.settings.gravity = 0
        scene.settings.collisionMargin = 0.02
        let asset = try tetraAsset()
        let hull = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 0,
            position: .zero, collisionEnabled: false)
        _ = scene.addConvexCollider(body: hull, asset: asset, friction: 0)
        let direction = normalize(F3(0.5, 0.5, -0.5))
        let sphere = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 0,
            position: direction * (sqrt(0.75) + 0.51), shape: .sphere,
            mass: 1, diagonalInertia: F3(repeating: 0.1))
        let solver = try GPUSolver(scene: scene)
        solver.step()
        _ = solver.bodyStates([sphere])
        XCTAssertGreaterThanOrEqual(solver.lastPairCandidates, 1)
        XCTAssertTrue(solver.activeRigidContactPairs().contains {
            Set([$0.0, $0.1]) == Set([hull, sphere])
        })
    }

    func testTorusHullPairFailsAtInitialization() throws {
        try requireMetal()
        var scene = PhysicsScene(name: "unsupported-torus-hull")
        let hullBody = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 0.5,
            position: .zero, collisionEnabled: false)
        _ = scene.addConvexCollider(
            body: hullBody, vertices: cubeVertices, friction: 0.5)
        _ = scene.addBody(
            size: F3(1, 0.2, 0.2), density: 1, friction: 0.5,
            position: F3(4, 0, 0), shape: .torus)

        XCTAssertThrowsError(try GPUSolver(scene: scene)) { error in
            guard case GPUSolver.ConvexCollisionError.unsupportedTorusHull = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }
    }

    func testStaticOrExcludedTorusHullPairsDoNotFailInitialization() throws {
        try requireMetal()

        func scene(dynamicHull: Bool, excluded: Bool) -> PhysicsScene {
            var scene = PhysicsScene(name: "filtered-torus-hull")
            let hull = scene.addBody(
                size: F3(repeating: 1), density: dynamicHull ? 1 : 0,
                friction: 0.5, position: .zero, collisionEnabled: false)
            _ = scene.addConvexCollider(
                body: hull, vertices: cubeVertices, friction: 0.5)
            let torus = scene.addBody(
                size: F3(0.6, 0.15, 0), density: 0, friction: 0.5,
                position: F3(0.1, 0, 0), shape: .torus)
            if excluded {
                scene.addCollisionExclusion(bodyA: hull, bodyB: torus)
            }
            return scene
        }

        XCTAssertNoThrow(try GPUSolver(
            scene: scene(dynamicHull: false, excluded: false)))
        XCTAssertNoThrow(try GPUSolver(
            scene: scene(dynamicHull: true, excluded: true)))
    }
}
