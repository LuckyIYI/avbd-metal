import XCTest
import Metal
import SimCore
import simd
@testable import PhysicsAVBD

final class ConvexDeformableCollisionTests: XCTestCase {
    private struct EdgeKey: Hashable, Comparable {
        var a: UInt32
        var b: UInt32

        init(_ a: UInt32, _ b: UInt32) {
            self.a = min(a, b)
            self.b = max(a, b)
        }

        static func < (lhs: Self, rhs: Self) -> Bool {
            lhs.a == rhs.a ? lhs.b < rhs.b : lhs.a < rhs.a
        }
    }

    private let identity = Quat(real: 1, imag: .zero)

    private func requireMetal() throws {
        if MTLCreateSystemDefaultDevice() == nil {
            throw XCTSkip("Metal is unavailable")
        }
    }

    @discardableResult
    private func addTriangle(
        _ scene: inout PhysicsScene,
        _ a: F3, _ b: F3, _ c: F3,
        radius: Float = 0.02,
        mass: Float
    ) -> [Int] {
        let vertices = [a, b, c].map {
            scene.addParticle(
                radius: radius, mass: mass, friction: 0.6, position: $0)
        }
        scene.addTri(SceneTri(ids: (vertices[0], vertices[1], vertices[2])))
        return vertices
    }

    private func configure(_ scene: inout PhysicsScene) {
        scene.settings.gravity = 0
        scene.settings.iterations = 0
        scene.settings.dt = 1 / 120
        scene.settings.collisionMargin = 0.01
    }

    private func setDomain(
        _ scene: inout PhysicsScene, bodies: [Int], group: UInt32,
        shared: Bool
    ) {
        let bodySet = Set(bodies)
        for collider in scene.colliders.indices
            where bodySet.contains(scene.colliders[collider].body) {
            scene.colliders[collider].collisionGroup = group
            scene.colliders[collider].collidesWithSharedGeometry = shared
        }
    }

    private func step(_ solver: GPUSolver) throws {
        try solver.submitStep()
        try solver.synchronize()
        XCTAssertNil(solver.runtimeFailure)
    }

    private func rtContacts(_ solver: GPUSolver) -> [SoftContactGPU] {
        let contacts = solver.prevSoftContacts.contents().bindMemory(
            to: SoftContactGPU.self, capacity: max(1, solver.maxSoft))
        return (0..<min(solver.lastNumSoft, solver.maxSoft)).compactMap {
            let contact = contacts[$0]
            let flags = contact.anchorA.w.bitPattern
            return ((flags >> 2) & 0x3) == 2 ? contact : nil
        }
    }

    private func tetraAsset(halfExtent h: Float = 0.5) throws
        -> ConvexHullAsset
    {
        let vertices = [
            F3(-h, -h, -h), F3(-h, h, h),
            F3(h, -h, h), F3(h, h, -h),
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
            boundsMin: F3(repeating: -h), boundsMax: F3(repeating: h),
            boundingRadius: sqrt(3) * h,
            volume: 8 * h * h * h / 3, centroid: .zero,
            digest: digest, stableID: "hull-" + digest.prefix(16))
    }

    private func boxAsset(center: F3, size: F3) throws -> ConvexHullAsset {
        let h = size * 0.5
        let lo = center - h
        let hi = center + h
        let vertices = [
            F3(lo.x, lo.y, lo.z), F3(lo.x, lo.y, hi.z),
            F3(lo.x, hi.y, lo.z), F3(lo.x, hi.y, hi.z),
            F3(hi.x, lo.y, lo.z), F3(hi.x, lo.y, hi.z),
            F3(hi.x, hi.y, lo.z), F3(hi.x, hi.y, hi.z),
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
            boundsMin: lo, boundsMax: hi, boundingRadius: length(h),
            volume: size.x * size.y * size.z, centroid: center,
            digest: digest, stableID: "hull-" + digest.prefix(16))
    }

    func testOffsetAnalyticColliderUsesColliderPoseAndBodyOwner() throws {
        try requireMetal()
        var scene = PhysicsScene(name: "rt-offset-collider")
        configure(&scene)
        let triangle = addTriangle(
            &scene, F3(-0.5, -0.5, 0), F3(0.5, -0.5, 0),
            F3(0, 0.5, 0), mass: 0)
        let owner = scene.addBody(
            size: F3(repeating: 0.2), density: 0, friction: 0.7,
            position: F3(4, 0, 0), mass: 1,
            diagonalInertia: F3(repeating: 0.1), collisionEnabled: false)
        _ = scene.addCollider(
            body: owner, size: F3(repeating: 0.08),
            localPosition: F3(0, 0, 8), shape: .sphere)
        let touchingCollider = scene.addCollider(
            body: owner, size: F3(repeating: 0.08),
            localPosition: F3(-4, 0, 0.04), shape: .sphere)
        XCTAssertNotEqual(touchingCollider, owner,
                          "the regression requires collider/body index skew")

        let solver = try GPUSolver(scene: scene)
        try step(solver)
        let first = try XCTUnwrap(rtContacts(solver).first {
            Int($0.ids.x) == owner && triangle.contains(Int($0.ids.y))
        })
        XCTAssertEqual(first.lambda.w.bitPattern & 0x0FFF_FFFF,
                       UInt32(touchingCollider))
        XCTAssertEqual(first.ids.x, UInt32(owner))
        XCTAssertEqual(first.anchorA.x, -4, accuracy: 2e-3)
        XCTAssertTrue(first.anchorA.y.isFinite && first.anchorA.z.isFinite)

        let firstKey = SIMD2(first.lambda.w.bitPattern,
                             first.penalty.w.bitPattern)
        try step(solver)
        let second = try XCTUnwrap(rtContacts(solver).first {
            $0.lambda.w.bitPattern == firstKey.x
                && $0.penalty.w.bitPattern == firstKey.y
        })
        XCTAssertEqual(second.ids.x, UInt32(owner))
    }

    func testCookedHullUsesSupportGeometryNotAuthoredAABB() throws {
        try requireMetal()
        let asset = try tetraAsset()

        func makeScene(height: Float, fakeSize: Float) -> PhysicsScene {
            var scene = PhysicsScene(name: "rt-hull-not-aabb")
            configure(&scene)
            _ = addTriangle(
                &scene, F3(-1, -1, 0), F3(1, -1, 0), F3(0, 1, 0),
                mass: 0)
            let owner = scene.addBody(
                size: F3(repeating: 0.001), density: 0, friction: 0.6,
                position: F3(0, 0, height), mass: 1,
                diagonalInertia: F3(repeating: 0.1),
                collisionEnabled: false)
            let collider = scene.addConvexCollider(body: owner, asset: asset)
            scene.colliders[collider].size = F3(repeating: fakeSize)
            return scene
        }

        let touching = try GPUSolver(scene: makeScene(
            height: 0.49, fakeSize: 0.001))
        try step(touching)
        XCTAssertFalse(rtContacts(touching).isEmpty,
            "a real hull contact must survive an intentionally tiny AABB")

        let separated = try GPUSolver(scene: makeScene(
            height: 2.0, fakeSize: 100))
        try step(separated)
        XCTAssertTrue(rtContacts(separated).isEmpty,
            "a huge authored AABB must not create a hull surface contact")
    }

    func testConcaveCompoundPartContactsTriangle() throws {
        try requireMetal()
        // Three independently validated hulls form one connected concave U.
        // Keeping this fixture in-process makes the runtime regression
        // independent of cooker/provenance-schema migrations.
        let parts = try [
            boxAsset(center: F3(-0.75, 0, 0.25), size: F3(0.5, 1, 2.5)),
            boxAsset(center: F3(0.75, 0, 0.25), size: F3(0.5, 1, 2.5)),
            boxAsset(center: F3(0, 0, -0.75), size: F3(1, 1, 0.5)),
        ]

        var scene = PhysicsScene(name: "rt-concave-compound-triangle")
        configure(&scene)
        let triangle = addTriangle(
            &scene,
            F3(-0.49, -0.25, 0), F3(-0.49, 0.25, 0),
            F3(-0.49, 0, 0.5), mass: 1)
        let owner = scene.addBody(
            size: F3(3, 1, 3), density: 0, friction: 0.7,
            position: .zero, collisionEnabled: false)
        let colliders = parts.map {
            scene.addConvexCollider(body: owner, asset: $0, friction: 0.7)
        }

        let solver = try GPUSolver(scene: scene)
        try step(solver)
        let contacts = rtContacts(solver).filter {
            $0.ids.x == UInt32(owner)
                && Set([$0.ids.y, $0.ids.z, $0.ids.w])
                    == Set(triangle.map(UInt32.init))
        }
        XCTAssertFalse(contacts.isEmpty,
            "a triangle at the U's inner wall must contact a cooked part")
        let colliderKeys = Set(contacts.map {
            Int($0.lambda.w.bitPattern & 0x0FFF_FFFF)
        })
        XCTAssertTrue(colliderKeys.isSubset(of: Set(colliders)))
        XCTAssertTrue(contacts.allSatisfy {
            [$0.normal, $0.anchorA, $0.C0, $0.lambda, $0.penalty]
                .allSatisfy { value in
                    value.x.isFinite && value.y.isFinite
                        && value.z.isFinite && value.w.isFinite
                }
        })
    }

    func testMoreThanFourCompoundPartsAreNeverSilentlyTruncated() throws {
        try requireMetal()
        let asset = try tetraAsset(halfExtent: 0.12)
        var scene = PhysicsScene(name: "rt-compound-over-four")
        configure(&scene)
        _ = addTriangle(
            &scene, F3(-1, -1, 0), F3(1, -1, 0), F3(0, 1, 0), mass: 1)
        let owner = scene.addBody(
            size: F3(1.2, 0.3, 0.3), density: 0, friction: 0.7,
            position: F3(0, 0, 0.08), collisionEnabled: false)
        let colliders = (0..<6).map { index in
            scene.addConvexCollider(
                body: owner, asset: asset,
                localPosition: F3(-0.5 + 0.2 * Float(index), 0, 0))
        }

        let solver = try GPUSolver(scene: scene)
        try step(solver)
        let keys = Set(rtContacts(solver).filter {
            $0.ids.x == UInt32(owner)
        }.map { Int($0.lambda.w.bitPattern & 0x0FFF_FFFF) })
        XCTAssertEqual(keys, Set(colliders),
            "all qualifying compound parts must reach exact demand accounting")
        XCTAssertGreaterThan(keys.count, 4)
    }

    func testRigidAndSurfaceCollisionDomainsAreSymmetricAndExact() throws {
        try requireMetal()
        let asset = try tetraAsset()

        func hasContact(
            surfaceGroup: UInt32, surfaceShared: Bool,
            rigidGroup: UInt32, rigidShared: Bool
        ) throws -> Bool {
            var scene = PhysicsScene(name: "rt-shared-domain")
            configure(&scene)
            let surface = addTriangle(
                &scene, F3(-1, -1, 0), F3(1, -1, 0), F3(0, 1, 0),
                mass: 1)
            setDomain(&scene, bodies: surface, group: surfaceGroup,
                      shared: surfaceShared)
            let owner = scene.addBody(
                size: F3(repeating: 1), density: 0, friction: 0.6,
                position: F3(0, 0, 0.49), collisionEnabled: false)
            _ = scene.addConvexCollider(
                body: owner, asset: asset, collisionGroup: rigidGroup,
                collidesWithSharedGeometry: rigidShared)
            let solver = try GPUSolver(scene: scene)
            try step(solver)
            return !rtContacts(solver).isEmpty
        }

        // Equal replicas collide regardless of their shared-geometry opt-in.
        XCTAssertTrue(try hasContact(
            surfaceGroup: 7, surfaceShared: false,
            rigidGroup: 7, rigidShared: false))
        // Different nonzero replicas never collide.
        XCTAssertFalse(try hasContact(
            surfaceGroup: 7, surfaceShared: true,
            rigidGroup: 8, rigidShared: true))
        // Surface is shared: the isolated rigid side owns the opt-in.
        XCTAssertTrue(try hasContact(
            surfaceGroup: 0, surfaceShared: false,
            rigidGroup: 7, rigidShared: true))
        XCTAssertFalse(try hasContact(
            surfaceGroup: 0, surfaceShared: true,
            rigidGroup: 7, rigidShared: false))
        // Rigid is shared: the isolated deformable side owns the opt-in.
        XCTAssertTrue(try hasContact(
            surfaceGroup: 7, surfaceShared: true,
            rigidGroup: 0, rigidShared: false))
        XCTAssertFalse(try hasContact(
            surfaceGroup: 7, surfaceShared: false,
            rigidGroup: 0, rigidShared: true))
    }

    func testSpatialRigidTriangleCandidatesStayLocalAsRigidCountScales()
        throws {
        try requireMetal()

        func run(farRigidCount: Int) throws
            -> (candidates: Int, contacts: Int, triangles: Int, rigids: Int)
        {
            var scene = PhysicsScene(name: "rt-spatial-scaling")
            configure(&scene)
            let side = 11
            var vertices = [Int]()
            vertices.reserveCapacity(side * side)
            for y in 0..<side {
                for x in 0..<side {
                    vertices.append(scene.addParticle(
                        radius: 0.015, mass: 1, friction: 0.6,
                        position: F3(
                            -0.5 + 0.1 * Float(x),
                            -0.5 + 0.1 * Float(y), 0)))
                }
            }
            for y in 0..<(side - 1) {
                for x in 0..<(side - 1) {
                    let i = y * side + x
                    scene.addTri(SceneTri(ids: (
                        vertices[i], vertices[i + 1],
                        vertices[i + side + 1])))
                    scene.addTri(SceneTri(ids: (
                        vertices[i], vertices[i + side + 1],
                        vertices[i + side])))
                }
            }
            setDomain(&scene, bodies: vertices, group: 7, shared: false)

            let nearBody = scene.addBody(
                size: F3(repeating: 0.04), density: 0, friction: 0.6,
                position: F3(0, 0, 0.035), collisionEnabled: false)
            _ = scene.addConvexCollider(
                body: nearBody,
                asset: try boxAsset(
                    center: .zero, size: F3(repeating: 0.04)),
                collisionGroup: 7, collidesWithSharedGeometry: false)
            for index in 0..<farRigidCount {
                _ = scene.addBody(
                    size: F3(repeating: 0.04), density: 0, friction: 0.6,
                    position: F3(
                        5 + 0.2 * Float(index % 24),
                        0.2 * Float(index / 24), 0.035),
                    collisionGroup: index.isMultiple(of: 2) ? 7 : 8)
            }

            let solver = try GPUSolver(scene: scene)
            try step(solver)
            let contacts = rtContacts(solver)
            XCTAssertFalse(contacts.isEmpty,
                "the local convex collider must retain RT contact")
            XCTAssertTrue(contacts.allSatisfy {
                $0.ids.x == UInt32(nearBody)
            }, "far and cross-domain hash entries must not alias into RT")
            return (
                solver.lastRigidTriangleCandidates, contacts.count,
                scene.tris.count, Int(solver.params.numHashedRigid))
        }

        let smaller = try run(farRigidCount: 40)
        let larger = try run(farRigidCount: 120)
        XCTAssertGreaterThan(smaller.candidates, 0)
        XCTAssertEqual(smaller.candidates, larger.candidates,
            "far hashed rigids must not increase exact RT work")
        XCTAssertEqual(smaller.contacts, larger.contacts)
        XCTAssertLessThan(
            larger.candidates,
            larger.triangles * larger.rigids / 16,
            "the spatial path must stay far below triangles times all rigids")
    }

    func testTriangleWithMixedAuthoredDomainsFailsAtInitialization() throws {
        try requireMetal()
        var scene = PhysicsScene(name: "rt-mixed-surface-domain")
        configure(&scene)
        let vertices = addTriangle(
            &scene, F3(-1, -1, 0), F3(1, -1, 0), F3(0, 1, 0), mass: 1)
        setDomain(&scene, bodies: [vertices[0]], group: 1, shared: true)
        setDomain(&scene, bodies: [vertices[1], vertices[2]], group: 2,
                  shared: true)

        XCTAssertThrowsError(try GPUSolver(scene: scene)) { error in
            guard case GPUSolver.SurfaceCollisionDomainError
                .inconsistentTriangle(let triangle, let reported) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(triangle, 0)
            XCTAssertEqual(reported, vertices)
        }
    }

    func testCookedHullContactsSynthesizedTetBoundary() throws {
        try requireMetal()
        let asset = try tetraAsset(halfExtent: 0.2)
        var scene = PhysicsScene(name: "rt-hull-tet-boundary")
        configure(&scene)
        let ids = [
            F3(-0.8, -0.8, 0), F3(0.8, -0.8, 0),
            F3(0, 0.8, 0), F3(0, 0, 1),
        ].map {
            scene.addParticle(radius: 0.02, mass: 1, friction: 0.6,
                              position: $0)
        }
        scene.addTet(SceneTet(
            ids: (ids[0], ids[1], ids[2], ids[3]),
            mu: 500, lambda: 1_000))
        let owner = scene.addBody(
            size: F3(repeating: 0.4), density: 0, friction: 0.7,
            position: F3(0, 0, 0.18), collisionEnabled: false)
        _ = scene.addConvexCollider(body: owner, asset: asset)

        let solver = try GPUSolver(scene: scene)
        try step(solver)
        XCTAssertTrue(rtContacts(solver).contains {
            $0.ids.x == UInt32(owner)
                && ids.contains(Int($0.ids.y))
                && ids.contains(Int($0.ids.z))
                && ids.contains(Int($0.ids.w))
        }, "synthesized tet boundary triangles must share the hull RT path")
    }

    func testLargeTriangleDoesNotDropContactingTopBoxCorners() throws {
        try requireMetal()
        var scene = PhysicsScene(name: "rt-box-eight-corners")
        configure(&scene)
        _ = addTriangle(
            &scene, F3(-10, -10, 0.5), F3(10, -10, 0.5),
            F3(0, 10, 0.5), mass: 1)
        let owner = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 0.7,
            position: .zero, collisionEnabled: false)
        let collider = scene.addCollider(
            body: owner, size: F3(repeating: 1), shape: .box)

        let solver = try GPUSolver(scene: scene)
        try step(solver)
        let contacts = rtContacts(solver).filter {
            $0.ids.x == UInt32(owner)
                && Int($0.lambda.w.bitPattern & 0x0FFF_FFFF) == collider
        }
        XCTAssertFalse(contacts.isEmpty,
            "bottom corners must not consume a staging cap before top contact")
        XCTAssertTrue(contacts.contains {
            ($0.penalty.w.bitPattern & 0xFF) >= 4
        }, "the emitted persistent feature must be a contacting top corner")
    }

    func testDegenerateSurfaceTriangleWithAnalyticRigidSkips() throws {
        try requireMetal()
        var scene = PhysicsScene(name: "rt-degenerate-triangle")
        configure(&scene)
        _ = addTriangle(
            &scene, F3(-1, 0, 0), F3(0, 0, 0), F3(1, 0, 0), mass: 1)
        _ = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 0.7,
            position: .zero, shape: .box)

        let solver = try GPUSolver(scene: scene)
        try step(solver)
        XCTAssertTrue(rtContacts(solver).isEmpty)
    }

    func testStaticHullAndStaticTriangleSkipConvexQuery() throws {
        try requireMetal()
        var scene = PhysicsScene(name: "rt-static-hull-static-triangle")
        configure(&scene)
        _ = addTriangle(
            &scene, F3(-1, -1, 0), F3(1, -1, 0), F3(0, 1, 0),
            mass: 0)
        let owner = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 0.7,
            position: F3(0, 0, 0.2), collisionEnabled: false)
        _ = scene.addConvexCollider(
            body: owner,
            asset: try boxAsset(center: .zero, size: F3(repeating: 1)))

        let solver = try GPUSolver(scene: scene)
        try step(solver)

        XCTAssertTrue(rtContacts(solver).isEmpty)
        XCTAssertEqual(solver.lastNumSoft, 0)
        XCTAssertEqual(
            solver.convexQueryPoison.contents()
                .bindMemory(to: UInt32.self, capacity: 1)[0],
            0, "a fully static pair must never enter the support query")
    }

    func testInflatedMPRKeepsPositiveCorrectedRTSeparation() throws {
        try requireMetal()
        var scene = PhysicsScene(name: "rt-mpr-corrected-separation")
        configure(&scene)
        _ = addTriangle(
            &scene, F3(-1, -1, 0), F3(1, -1, 0), F3(0, 1, 0),
            radius: 1.0e-8, mass: 1)
        let owner = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 0.7,
            position: F3(0, 0, 0.50007), collisionEnabled: false)
        _ = scene.addConvexCollider(
            body: owner,
            asset: try boxAsset(center: .zero, size: F3(repeating: 1)))

        let solver = try GPUSolver(scene: scene)
        try step(solver)
        let contact = try XCTUnwrap(rtContacts(solver).first {
            $0.ids.x == UInt32(owner)
        }, "the corrected positive gap remains inside the RT margin")

        XCTAssertGreaterThan(
            contact.C0.x, solver.params.elemMargin + 2.0e-5,
            "inflated MPR overlap must not clamp true separation to zero")
        XCTAssertLessThan(
            contact.C0.x, solver.params.elemMargin + 2.0e-4,
            "the fixture must stay in the corrected MPR boundary band")
    }

    func testFastHullOutsideRigidMarginReachesSpeculativeRTBand() throws {
        try requireMetal()
        var scene = PhysicsScene(name: "rt-fast-hull-speculative-band")
        configure(&scene)
        scene.settings.collisionMargin = 1.0e-4

        let triangleRadius: Float = 0.002
        let particleRadius: Float = 1.0e-6
        let surface = addTriangle(
            &scene,
            F3(triangleRadius, 0, 0),
            F3(-0.5 * triangleRadius, 0.8660254 * triangleRadius, 0),
            F3(-0.5 * triangleRadius, -0.8660254 * triangleRadius, 0),
            radius: particleRadius, mass: 0)
        setDomain(&scene, bodies: surface, group: 7, shared: false)
        // Keep the tiny particle colliders enabled as the authored collision-
        // domain carriers for the surface. Their geometric separation is
        // outside the rigid pair band; the final lastNumPairs assertion still
        // proves that only the triangle path detects this fixture.

        // A nearly axial hull keeps the center-sphere bound tight: this gap
        // is beyond the authored rigid margin (and even elemMargin), but an
        // approaching body must still be admitted by velocity inflation.
        let hullSize = F3(0.0005, 0.0005, 0.1)
        let gap: Float = 0.0025
        let speed: Float = 1.2
        let startZ = 0.5 * hullSize.z + particleRadius + gap
        let owner = scene.addBody(
            size: hullSize, density: 0, friction: 0.7,
            position: F3(0, 0, startZ), velocity: F3(0, 0, -speed),
            mass: 1, diagonalInertia: F3(repeating: 0.01),
            collisionEnabled: false)
        _ = scene.addConvexCollider(
            body: owner,
            asset: try boxAsset(center: .zero, size: hullSize),
            collisionGroup: 7, collidesWithSharedGeometry: false)

        // Force the spatial RT path as well as its per-candidate cull. These
        // bodies are far enough away that they cannot contribute contacts.
        for index in 0..<9 {
            _ = scene.addBody(
                size: F3(repeating: 0.02), density: 0, friction: 0.5,
                position: F3(5 + Float(index), 5, 5),
                collisionGroup: 7)
        }

        let solver = try GPUSolver(scene: scene)
        XCTAssertGreaterThan(solver.params.numHashedRigid, 8,
            "the regression must exercise the spatial RT prefilter")
        let triangleBound = triangleRadius + particleRadius
        let inflation = min(speed * scene.settings.dt, 0.5 * triangleBound)
        XCTAssertGreaterThan(gap, scene.settings.collisionMargin)
        XCTAssertGreaterThan(gap, solver.params.elemMargin,
            "velocity inflation, not elemMargin alone, must admit the pair")
        XCTAssertLessThan(gap, solver.params.elemMargin + inflation)
        let hullRadius = length(hullSize * 0.5)
        XCTAssertGreaterThan(
            startZ,
            triangleBound + hullRadius + scene.settings.collisionMargin,
            "the old collisionMargin-only center gate must reject this fixture")
        XCTAssertLessThan(
            startZ,
            triangleBound + hullRadius
                + max(scene.settings.collisionMargin,
                      solver.params.elemMargin + inflation))

        try step(solver)
        XCTAssertTrue(rtContacts(solver).contains {
            $0.ids.x == UInt32(owner)
        }, "the fast hull must reach the exact speculative RT query")
        XCTAssertGreaterThan(solver.lastRigidTriangleCandidates, 0)
        XCTAssertEqual(solver.lastNumPairs, 0,
            "particle colliders are disabled; only RT may detect this pair")
    }

    func testHullPenetratingTriangleEdgeUsesRadialNormal() throws {
        try requireMetal()
        var scene = PhysicsScene(name: "rt-hull-triangle-edge-normal")
        configure(&scene)
        _ = addTriangle(
            &scene, F3(-1, -1, 0), F3(1, -1, 0), F3(0, 1, 0),
            mass: 1)
        let owner = scene.addBody(
            size: F3(repeating: 0.6), density: 0, friction: 0.7,
            position: F3(0, -1.28, 0.05), collisionEnabled: false)
        _ = scene.addConvexCollider(
            body: owner,
            asset: try boxAsset(center: .zero, size: F3(repeating: 0.6)))

        let solver = try GPUSolver(scene: scene)
        try step(solver)
        let contact = try XCTUnwrap(rtContacts(solver).first {
            $0.ids.x == UInt32(owner)
        })

        XCTAssertLessThan(contact.normal.y, -0.7,
            "an edge overlap must eject laterally from the live witness")
        XCTAssertLessThan(abs(contact.normal.z), 0.3,
            "a boundary witness must not be forced onto the triangle normal")
        XCTAssertLessThan(contact.C0.x, solver.params.elemMargin,
            "the lateral witness must retain its physical penetration")
    }

    func testCookedHullIsSupportedByPinnedSurfaceOverSustainedSolve() throws {
        try requireMetal()
        var scene = PhysicsScene(name: "rt-hull-sustained-surface-support")
        scene.settings.dt = 1 / 120
        scene.settings.gravity = -9.81
        scene.settings.iterations = 14
        scene.settings.collisionMargin = 0.01
        scene.settings.rigidLinearDamping = 2
        scene.settings.rigidAngularDamping = 6

        let surface = addTriangle(
            &scene, F3(-2, -1.5, 0), F3(2, -1.5, 0), F3(0, 2, 0),
            radius: 0.02, mass: 0)
        let surfaceSet = Set(surface)
        // Particles normally own analytic sphere colliders. Disable those so
        // only the authored triangle can support the hull; otherwise this
        // gate could pass through the ordinary rigid-pair path.
        for collider in scene.colliders.indices
            where surfaceSet.contains(scene.colliders[collider].body) {
            scene.colliders[collider].collisionEnabled = false
        }

        let owner = scene.addBody(
            size: F3(repeating: 0.4), density: 0, friction: 1,
            position: F3(0.13, -0.07, 0.8), mass: 1,
            diagonalInertia: F3(repeating: 1), collisionEnabled: false)
        _ = scene.addConvexCollider(
            body: owner,
            asset: try boxAsset(center: .zero, size: F3(repeating: 0.4)),
            friction: 1)

        let solver = try GPUSolver(scene: scene)
        var minimumZ = Float.greatestFiniteMagnitude
        var minimumVerticalVelocity = Float.greatestFiniteMagnitude
        var contactFrames = 0
        for _ in 0..<180 {
            try step(solver)
            let state = solver.bodyStates([owner])[0]
            XCTAssertTrue(state.position.x.isFinite
                && state.position.y.isFinite && state.position.z.isFinite)
            XCTAssertTrue(state.linearVelocity.x.isFinite
                && state.linearVelocity.y.isFinite
                && state.linearVelocity.z.isFinite)
            minimumZ = min(minimumZ, state.position.z)
            minimumVerticalVelocity = min(
                minimumVerticalVelocity, state.linearVelocity.z)
            if rtContacts(solver).contains(where: {
                $0.ids.x == UInt32(owner)
            }) {
                contactFrames += 1
            }
            XCTAssertEqual(solver.lastNumPairs, 0,
                "ordinary rigid manifolds must not support this fixture")
        }

        let final = solver.bodyStates([owner])[0]
        XCTAssertLessThan(minimumVerticalVelocity, -1,
            "the hull must undergo a real gravity-driven impact")
        XCTAssertGreaterThan(contactFrames, 20,
            "rigid-triangle contacts must remain active during support")
        XCTAssertGreaterThan(minimumZ, 0.12,
            "the convex hull must never pass through the pinned surface")
        let expectedRestZ: Float = 0.2 + 0.02 - solver.params.elemMargin
        XCTAssertEqual(final.position.z, expectedRestZ, accuracy: 0.07)
        XCTAssertLessThan(abs(final.linearVelocity.z), 0.12)
        XCTAssertLessThan(length(final.linearVelocity), 0.18)
        XCTAssertNil(solver.runtimeFailure)
    }

    func testOverlappingHullAndPinnedTriangleRemainActiveAtMillionUnitOrigin()
        throws
    {
        try requireMetal()
        let origin = F3(repeating: 1_000_000)
        var scene = PhysicsScene(name: "rt-hull-million-origin-overlap")
        configure(&scene)
        let surface = addTriangle(
            &scene,
            origin + F3(-2, -2, 0),
            origin + F3(2, -2, 0),
            origin + F3(0, 4, 0),
            radius: 0.02, mass: 0)
        let surfaceSet = Set(surface)
        for collider in scene.colliders.indices
            where surfaceSet.contains(scene.colliders[collider].body) {
            scene.colliders[collider].collisionEnabled = false
        }

        // 0.1875 is exactly representable at this origin (three 0.0625 ULPs)
        // and puts the 0.4-wide hull in true geometric overlap with the prism.
        let owner = scene.addBody(
            size: F3(repeating: 0.4), density: 0, friction: 0.8,
            position: origin + F3(0, 0, 0.1875), mass: 1,
            diagonalInertia: F3(repeating: 1), collisionEnabled: false)
        _ = scene.addConvexCollider(
            body: owner,
            asset: try boxAsset(center: .zero, size: F3(repeating: 0.4)),
            friction: 0.8)

        let solver = try GPUSolver(scene: scene)
        for _ in 0..<24 {
            try step(solver)
            let contacts = rtContacts(solver).filter {
                $0.ids.x == UInt32(owner)
            }
            let contact = try XCTUnwrap(contacts.first,
                "the local MPR certificate must survive world remapping")
            XCTAssertTrue(contact.C0.x.isFinite)
            XCTAssertLessThan(contact.C0.x, solver.params.elemMargin,
                "the translated fixture must remain a true overlap")
            XCTAssertEqual(solver.lastNumPairs, 0,
                "only rigid-vs-triangle contact may support this fixture")
        }
        XCTAssertNil(solver.runtimeFailure)
    }

    func testGJKExhaustionPoisonsQueuedSuccessorAndRestoresPose() throws {
        try requireMetal()
        let asset = try tetraAsset()
        var scene = PhysicsScene(name: "convex-query-fail-closed")
        configure(&scene)
        scene.settings.gravity = -9.81
        let moving = scene.addBody(
            size: F3(repeating: 1), density: 0, friction: 0.6,
            position: F3(0.25, -0.5, 2),
            rotation: Quat(angle: 0.37, axis: F3(0, 1, 0)),
            velocity: F3(4, -2, 1), mass: 1,
            diagonalInertia: F3(repeating: 0.2), collisionEnabled: false)
        _ = scene.addConvexCollider(body: moving, asset: asset)

        let solver = try GPUSolver(scene: scene)
        let positions = solver.posLin.contents().bindMemory(
            to: SIMD4<Float>.self, capacity: solver.numBodies)
        let rotations = solver.posAng.contents().bindMemory(
            to: SIMD4<Float>.self, capacity: solver.numBodies)
        let initialPosition = positions[moving]
        let initialRotation = rotations[moving]
        solver.convexQueryFailureForTesting = true

        try solver.submitStep()
        solver.convexQueryFailureForTesting = false
        try solver.submitStep()
        XCTAssertEqual(solver.inflightCountForTesting, 2)
        XCTAssertThrowsError(try solver.synchronize()) { error in
            guard case let GPUSolver.RuntimeFailure.commandExecution(
                operation, frame, status, domain, code, message) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(operation, "convex narrowphase safety validation")
            XCTAssertEqual(frame, 1)
            XCTAssertEqual(status, 0)
            XCTAssertEqual(
                domain, GPUSolver.RuntimeFailure.convexQueryFailureDomain)
            XCTAssertEqual(
                code, GPUSolver.RuntimeFailure.convexQueryInconclusiveCode)
            XCTAssertTrue(message.contains("trustworthy collision witness"))
        }

        XCTAssertEqual(positions[moving].x, initialPosition.x, accuracy: 1e-7)
        XCTAssertEqual(positions[moving].y, initialPosition.y, accuracy: 1e-7)
        XCTAssertEqual(positions[moving].z, initialPosition.z, accuracy: 1e-7)
        XCTAssertEqual(rotations[moving].x, initialRotation.x, accuracy: 1e-7)
        XCTAssertEqual(rotations[moving].y, initialRotation.y, accuracy: 1e-7)
        XCTAssertEqual(rotations[moving].z, initialRotation.z, accuracy: 1e-7)
        XCTAssertEqual(rotations[moving].w, initialRotation.w, accuracy: 1e-7)
        let velocity = solver.velLin.contents().bindMemory(
            to: SIMD4<Float>.self, capacity: solver.numBodies)[moving]
        let angularVelocity = solver.velAng.contents().bindMemory(
            to: SIMD4<Float>.self, capacity: solver.numBodies)[moving]
        XCTAssertEqual(
            length(F3(velocity.x, velocity.y, velocity.z)), 0, accuracy: 1e-7)
        XCTAssertEqual(
            length(F3(
                angularVelocity.x, angularVelocity.y, angularVelocity.z)),
            0, accuracy: 1e-7)
        XCTAssertEqual(
            solver.runtimeFailure,
            .convexQueryInconclusive(frame: 1, offendingQueries: 1))
    }
}
