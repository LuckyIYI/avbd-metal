import Foundation
import simd
import XCTest
@testable import SimCore

final class ConvexAssetTests: XCTestCase {
    func testMeshUpAxisBakesRightHandedRuntimeZRotations() {
        let x = F3(1, 0, 0)
        let y = F3(0, 1, 0)
        let z = F3(0, 0, 1)

        XCTAssertEqual(orientMeshPoint(x, upAxis: .z), x)
        XCTAssertEqual(orientMeshPoint(y, upAxis: .y), z)
        XCTAssertEqual(orientMeshPoint(x, upAxis: .x), z)
        for axis: MeshUpAxis in [.x, .y, .z] {
            let ox = orientMeshPoint(x, upAxis: axis)
            let oy = orientMeshPoint(y, upAxis: axis)
            let oz = orientMeshPoint(z, upAxis: axis)
            XCTAssertEqual(cross(ox, oy), oz)
        }
    }

    private var fixtureURL: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // SimCoreTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // repository
            .appendingPathComponent(
                "Sources/PhysicsAVBD/Assets/convex/concave-u.avbdconvex.json"
            )
    }

    private func fixtureData() throws -> Data {
        try Data(contentsOf: fixtureURL)
    }

    func testPinnedCoACDConcaveUDecodesAndValidates() throws {
        let compound = try ConvexCompoundAsset.decode(from: fixtureData())

        XCTAssertEqual(compound.schemaVersion, 1)
        XCTAssertEqual(compound.kind, "avbd.convex-compound")
        XCTAssertEqual(compound.source.uri, "convex/concave-u.obj")
        XCTAssertEqual(
            compound.source.sha256,
            "7772476fe243e8f81e8750a17e75ab58025232bb7e94152d3b0ea1028df5382e"
        )
        XCTAssertEqual(compound.cooker.algorithm, "coacd")
        XCTAssertEqual(compound.cooker.backend, "coacd-python")
        XCTAssertEqual(compound.cooker.backendVersion, "1.0.11")
        XCTAssertEqual(compound.cooker.parameters.seed, 0)
        XCTAssertEqual(compound.cooker.parameters.maxHulls, 8)
        XCTAssertEqual(compound.cooker.parameters.maxVerticesPerHull, 64)
        XCTAssertEqual(compound.parts.count, 3)
        XCTAssertEqual(
            compound.cacheKey,
            "1e3cd5db0667603d389b69cdbb3f90d411e1ce752af2964fd8b3b659cf9ff260"
        )
        XCTAssertEqual(compound.cacheKey, try compound.expectedCacheKey())
        XCTAssertEqual(compound.digest,
                       ConvexCompoundAsset.contentDigest(parts: compound.parts))
        XCTAssertEqual(
            compound.parts.reduce(Float.zero) { $0 + $1.volume },
            7.0,
            accuracy: 0.02
        )

        for part in compound.parts {
            let _: [F3] = part.vertices
            let _: [SIMD3<UInt32>] = part.triangles
            XCTAssertEqual(part.digest, ConvexHullAsset.geometryDigest(
                vertices: part.vertices, triangles: part.triangles
            ))
            XCTAssertEqual(part.stableID, "hull-" + part.digest.prefix(16))
            XCTAssertGreaterThan(part.volume, 0)
            XCTAssertGreaterThan(part.boundingRadius, 0)
            XCTAssertEqual(part.edges.count,
                           part.vertices.count + part.triangles.count - 2)
        }
    }

    func testCodableRoundTripPreservesCanonicalGeometry() throws {
        let original = try ConvexCompoundAsset.decode(from: fixtureData())
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let encoded = try encoder.encode(original)
        let decoded = try ConvexCompoundAsset.decode(from: encoded)
        XCTAssertEqual(decoded, original)
    }

    func testBoundedRegularFileLoadMatchesInMemoryDecode() throws {
        XCTAssertEqual(
            try ConvexCompoundAsset.load(from: fixtureURL),
            try ConvexCompoundAsset.decode(from: fixtureData())
        )
    }

    func testLoadRejectsDenseAndSparseOversizedFilesBeforeDecode() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: directory) }

        let dense = directory.appendingPathComponent("dense.json")
        try Data(
            repeating: 0x20,
            count: ConvexAssetLimits.maximumEncodedBytes + 1
        ).write(to: dense)
        XCTAssertThrowsError(try ConvexCompoundAsset.load(from: dense)) { error in
            XCTAssertTrue(String(describing: error).contains("16 MiB"))
        }

        let sparse = directory.appendingPathComponent("sparse.json")
        XCTAssertTrue(FileManager.default.createFile(
            atPath: sparse.path, contents: Data()
        ))
        let sparseHandle = try FileHandle(forWritingTo: sparse)
        try sparseHandle.truncate(
            atOffset: UInt64(ConvexAssetLimits.maximumEncodedBytes + 1)
        )
        try sparseHandle.close()
        XCTAssertThrowsError(try ConvexCompoundAsset.load(from: sparse)) { error in
            XCTAssertTrue(String(describing: error).contains("16 MiB"))
        }
    }

    func testLoadRejectsDirectoriesAndSymlinks() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: directory) }

        XCTAssertThrowsError(try ConvexCompoundAsset.load(from: directory)) { error in
            XCTAssertTrue(String(describing: error).contains("regular file"))
        }
        XCTAssertThrowsError(try ConvexCompoundAsset.load(
            from: URL(fileURLWithPath: "/dev/null")
        )) { error in
            XCTAssertTrue(String(describing: error).contains("regular file"))
        }

        let link = directory.appendingPathComponent("asset-link.json")
        try FileManager.default.createSymbolicLink(
            at: link, withDestinationURL: fixtureURL
        )
        XCTAssertThrowsError(try ConvexCompoundAsset.load(from: link))
    }

    func testInvalidTriangleIndexAndDigestAreRejectedDuringDecode() throws {
        var root = try XCTUnwrap(
            JSONSerialization.jsonObject(with: fixtureData()) as? [String: Any]
        )
        var parts = try XCTUnwrap(root["parts"] as? [[String: Any]])
        var first = parts[0]
        var triangles = try XCTUnwrap(first["triangles"] as? [[Int]])
        triangles[0][0] = 4_294_967_295
        first["triangles"] = triangles
        parts[0] = first
        root["parts"] = parts
        let invalidIndex = try JSONSerialization.data(withJSONObject: root)
        XCTAssertThrowsError(try ConvexCompoundAsset.decode(from: invalidIndex))

        root = try XCTUnwrap(
            JSONSerialization.jsonObject(with: fixtureData()) as? [String: Any]
        )
        parts = try XCTUnwrap(root["parts"] as? [[String: Any]])
        first = parts[0]
        first["digest"] = String(repeating: "0", count: 64)
        first["stableID"] = "hull-" + String(repeating: "0", count: 16)
        parts[0] = first
        root["parts"] = parts
        let invalidDigest = try JSONSerialization.data(withJSONObject: root)
        XCTAssertThrowsError(try ConvexCompoundAsset.decode(from: invalidDigest))
    }

    func testOpenAdjacencyAndUnknownSchemaAreRejected() throws {
        var root = try XCTUnwrap(
            JSONSerialization.jsonObject(with: fixtureData()) as? [String: Any]
        )
        var parts = try XCTUnwrap(root["parts"] as? [[String: Any]])
        var first = parts[0]
        var edges = try XCTUnwrap(first["edges"] as? [[String: Any]])
        edges.removeLast()
        first["edges"] = edges
        parts[0] = first
        root["parts"] = parts
        let open = try JSONSerialization.data(withJSONObject: root)
        XCTAssertThrowsError(try ConvexCompoundAsset.decode(from: open))

        root = try XCTUnwrap(
            JSONSerialization.jsonObject(with: fixtureData()) as? [String: Any]
        )
        root["schemaVersion"] = 2
        let futureSchema = try JSONSerialization.data(withJSONObject: root)
        XCTAssertThrowsError(try ConvexCompoundAsset.decode(from: futureSchema))
    }

    func testPartsInternByFullDigestWithoutGeometryDuplication() throws {
        let compound = try ConvexCompoundAsset.decode(from: fixtureData())
        var table: [ConvexHullAsset] = []

        let firstIndices = try compound.internParts(into: &table)
        let secondIndices = try compound.internParts(into: &table)

        XCTAssertEqual(firstIndices, [0, 1, 2])
        XCTAssertEqual(secondIndices, firstIndices)
        XCTAssertEqual(table, compound.parts)
    }

    func testDigestDedupUsesExactGeometryAndCanonicalRegisteredMetadata() throws {
        let compound = try ConvexCompoundAsset.decode(from: fixtureData())
        let original = try XCTUnwrap(compound.parts.first)
        let diagonal = simd_length(original.boundsMax - original.boundsMin)
        let acceptedDelta = max(diagonal * 1e-5, 1e-7) * 0.25
        var alternateBoundsMin = original.boundsMin
        alternateBoundsMin.x += acceptedDelta
        let alternate = try ConvexHullAsset(
            vertices: original.vertices,
            triangles: original.triangles,
            edges: original.edges,
            boundsMin: alternateBoundsMin,
            boundsMax: original.boundsMax,
            boundingRadius: original.boundingRadius,
            volume: original.volume,
            centroid: original.centroid,
            digest: original.digest,
            stableID: original.stableID
        )

        XCTAssertNotEqual(alternate, original)
        XCTAssertTrue(alternate.hasSameDigestGeometry(as: original))

        var table = [alternate]
        let indices = try compound.internParts(into: &table)
        XCTAssertEqual(indices.first, 0)
        XCTAssertEqual(table.count, compound.parts.count)

        var scene = PhysicsScene(name: "convex-digest-canonical-metadata")
        let body = scene.addBody(
            size: F3(repeating: 1), density: 1, friction: 0.5,
            position: .zero, collisionEnabled: false)
        XCTAssertEqual(scene.registerConvexAsset(alternate), 0)
        let collider = scene.addConvexCollider(body: body, asset: original)
        XCTAssertEqual(scene.convexAssets.count, 1)
        let geometryBounds = original.geometryBounds()
        let canonicalSize = geometryBounds.max - geometryBounds.min
        XCTAssertEqual(
            scene.colliders[collider].size.x.bitPattern,
            canonicalSize.x.bitPattern)
        XCTAssertEqual(
            scene.colliders[collider].size.y.bitPattern,
            canonicalSize.y.bitPattern)
        XCTAssertEqual(
            scene.colliders[collider].size.z.bitPattern,
            canonicalSize.z.bitPattern)
    }

    func testRuntimeComplexityLimitsMatchCookerContract() throws {
        let parameters = try ConvexAssetCookerParameters(maxVerticesPerHull: 64)
        XCTAssertEqual(parameters.maxVerticesPerHull, 64)
        XCTAssertThrowsError(try ConvexAssetCookerParameters(
            maxVerticesPerHull: 3
        ))
        XCTAssertThrowsError(try ConvexAssetCookerParameters(
            maxVerticesPerHull: 65
        ))
        XCTAssertThrowsError(try ConvexAssetCookerParameters(maxHulls: 257))
        XCTAssertThrowsError(try ConvexAssetCookerParameters(seed: UInt32(Int32.max) + 1))
    }

    func testCacheSealRejectsForgedSourceMetadataAndCookerIdentity() throws {
        let compound = try ConvexCompoundAsset.decode(from: fixtureData())
        let forgedSource = try ConvexAssetSourceMetadata(
            uri: "renamed/concave-u.obj",
            byteCount: compound.source.byteCount,
            sha256: compound.source.sha256,
            geometrySHA256: compound.source.geometrySHA256,
            upAxis: compound.source.upAxis,
            bakedScale: compound.source.bakedScale
        )
        XCTAssertThrowsError(try ConvexCompoundAsset(
            source: forgedSource,
            cooker: compound.cooker,
            cacheKey: compound.cacheKey,
            digest: compound.digest,
            parts: compound.parts
        ))

        XCTAssertThrowsError(try ConvexAssetCookerMetadata(
            algorithm: "hull",
            backend: "coacd-python",
            backendVersion: "1.0.11",
            parameters: compound.cooker.parameters
        ))
        XCTAssertThrowsError(try ConvexAssetCookerMetadata(
            algorithm: "coacd",
            backend: "coacd-python",
            backendVersion: "1.0.10",
            parameters: compound.cooker.parameters
        ))
    }

    func testDuplicatePartsAreRejectedEvenWithMatchingCompoundDigest() throws {
        let compound = try ConvexCompoundAsset.decode(from: fixtureData())
        let parts = [compound.parts[0], compound.parts[0]]
        XCTAssertThrowsError(try ConvexCompoundAsset(
            source: compound.source,
            cooker: compound.cooker,
            cacheKey: compound.cacheKey,
            digest: ConvexCompoundAsset.contentDigest(parts: parts),
            parts: parts
        ))
    }

    func testEncodedAndNestedArrayBoundsFailClosed() throws {
        let oversized = Data(
            repeating: 0x20,
            count: ConvexAssetLimits.maximumEncodedBytes + 1
        )
        XCTAssertThrowsError(try ConvexCompoundAsset.decode(from: oversized))

        var root = try XCTUnwrap(
            JSONSerialization.jsonObject(with: fixtureData()) as? [String: Any]
        )
        var parts = try XCTUnwrap(root["parts"] as? [[String: Any]])
        var first = parts[0]
        var vertices = try XCTUnwrap(first["vertices"] as? [[NSNumber]])
        while vertices.count <= ConvexAssetLimits.maximumVerticesPerHull {
            vertices.append(vertices[0])
        }
        first["vertices"] = vertices
        parts[0] = first
        root["parts"] = parts
        XCTAssertThrowsError(try ConvexCompoundAsset.decode(
            from: JSONSerialization.data(withJSONObject: root)
        ))

        root = try XCTUnwrap(
            JSONSerialization.jsonObject(with: fixtureData()) as? [String: Any]
        )
        parts = try XCTUnwrap(root["parts"] as? [[String: Any]])
        while parts.count <= ConvexAssetLimits.maximumParts {
            parts.append(parts[0])
        }
        root["parts"] = parts
        XCTAssertThrowsError(try ConvexCompoundAsset.decode(
            from: JSONSerialization.data(withJSONObject: root)
        ))
    }

    func testJSONBooleansCannotMasqueradeAsIntegersOrFloats() throws {
        var root = try XCTUnwrap(
            JSONSerialization.jsonObject(with: fixtureData()) as? [String: Any]
        )
        root["schemaVersion"] = true
        XCTAssertThrowsError(try ConvexCompoundAsset.decode(
            from: JSONSerialization.data(withJSONObject: root)
        ))

        root = try XCTUnwrap(
            JSONSerialization.jsonObject(with: fixtureData()) as? [String: Any]
        )
        var parts = try XCTUnwrap(root["parts"] as? [[String: Any]])
        var first = parts[0]
        var triangles = try XCTUnwrap(first["triangles"] as? [[Any]])
        triangles[0][0] = false
        first["triangles"] = triangles
        parts[0] = first
        root["parts"] = parts
        XCTAssertThrowsError(try ConvexCompoundAsset.decode(
            from: JSONSerialization.data(withJSONObject: root)
        ))

        root = try XCTUnwrap(
            JSONSerialization.jsonObject(with: fixtureData()) as? [String: Any]
        )
        var cooker = try XCTUnwrap(root["cooker"] as? [String: Any])
        var parameters = try XCTUnwrap(cooker["parameters"] as? [String: Any])
        parameters["thresholdMeters"] = true
        cooker["parameters"] = parameters
        root["cooker"] = cooker
        XCTAssertThrowsError(try ConvexCompoundAsset.decode(
            from: JSONSerialization.data(withJSONObject: root)
        ))
    }

    func testSourceURIHasABoundedUTF8Representation() {
        XCTAssertThrowsError(try ConvexAssetSourceMetadata(
            uri: String(repeating: "a", count: ConvexAssetLimits.maximumSourceURIBytes + 1),
            byteCount: 1,
            sha256: String(repeating: "0", count: 64),
            geometrySHA256: String(repeating: "1", count: 64),
            upAxis: "z",
            bakedScale: F3(repeating: 1)
        ))
    }

    func testSeventeenGonPrismFaceLoopExceedsPerInputClipCap() {
        let sides = 17
        var vertices: [F3] = []
        for z: Float in [-0.5, 0.5] {
            for index in 0..<sides {
                let angle = Float(index) * 2 * .pi / Float(sides)
                vertices.append(F3(cos(angle), sin(angle), z))
            }
        }
        var triangles: [SIMD3<UInt32>] = []
        for index in 1..<(sides - 1) {
            triangles.append(SIMD3(0, UInt32(index + 1), UInt32(index)))
            triangles.append(SIMD3(
                UInt32(sides), UInt32(sides + index), UInt32(sides + index + 1)
            ))
        }
        for index in 0..<sides {
            let next = (index + 1) % sides
            triangles.append(SIMD3(
                UInt32(index), UInt32(next), UInt32(sides + next)
            ))
            triangles.append(SIMD3(
                UInt32(index), UInt32(sides + next), UInt32(sides + index)
            ))
        }

        XCTAssertThrowsError(try ConvexHullAsset.validateMergedFaceLoops(
            vertices: vertices, triangles: triangles
        )) { error in
            XCTAssertTrue(String(describing: error).contains("17 vertices"))
        }
    }

    func testTinyTetraAcceptsCanonicalVolumeButRejectsForgedMassScale() throws {
        let scale: Float = 1e-4
        let vertices: [F3] = [
            F3(0, 0, 0),
            F3(0, 0, scale),
            F3(0, scale, 0),
            F3(scale, 0, 0),
        ]
        let triangles: [SIMD3<UInt32>] = [
            SIMD3(0, 1, 2),
            SIMD3(0, 2, 3),
            SIMD3(0, 3, 1),
            SIMD3(1, 3, 2),
        ]
        let edges = [
            ConvexHullEdge(vertexA: 0, vertexB: 1, faceA: 0, faceB: 2),
            ConvexHullEdge(vertexA: 0, vertexB: 2, faceA: 0, faceB: 1),
            ConvexHullEdge(vertexA: 0, vertexB: 3, faceA: 1, faceB: 2),
            ConvexHullEdge(vertexA: 1, vertexB: 2, faceA: 0, faceB: 3),
            ConvexHullEdge(vertexA: 1, vertexB: 3, faceA: 2, faceB: 3),
            ConvexHullEdge(vertexA: 2, vertexB: 3, faceA: 1, faceB: 3),
        ]
        let digest = ConvexHullAsset.geometryDigest(
            vertices: vertices, triangles: triangles
        )

        func makeHull(volume: Float) throws -> ConvexHullAsset {
            try ConvexHullAsset(
                vertices: vertices,
                triangles: triangles,
                edges: edges,
                boundsMin: .zero,
                boundsMax: F3(repeating: scale),
                boundingRadius: sqrt(3) * scale * 0.5,
                volume: volume,
                centroid: F3(repeating: scale * 0.25),
                digest: digest,
                stableID: "hull-" + digest.prefix(16)
            )
        }

        XCTAssertNoThrow(try makeHull(volume: scale * scale * scale / 6))
        XCTAssertThrowsError(try makeHull(volume: 1e-8)) { error in
            XCTAssertTrue(String(describing: error).contains("stored volume"))
        }
    }

    func testTranslatedTetraUsesHullLocalVolumeReference() throws {
        let origin = F3(repeating: 1_000_000)
        let vertices: [F3] = [
            origin,
            origin + F3(0, 0, 1),
            origin + F3(0, 1, 0),
            origin + F3(1, 0, 0),
        ]
        let triangles: [SIMD3<UInt32>] = [
            SIMD3(0, 1, 2),
            SIMD3(0, 2, 3),
            SIMD3(0, 3, 1),
            SIMD3(1, 3, 2),
        ]
        let edges = [
            ConvexHullEdge(vertexA: 0, vertexB: 1, faceA: 0, faceB: 2),
            ConvexHullEdge(vertexA: 0, vertexB: 2, faceA: 0, faceB: 1),
            ConvexHullEdge(vertexA: 0, vertexB: 3, faceA: 1, faceB: 2),
            ConvexHullEdge(vertexA: 1, vertexB: 2, faceA: 0, faceB: 3),
            ConvexHullEdge(vertexA: 1, vertexB: 3, faceA: 2, faceB: 3),
            ConvexHullEdge(vertexA: 2, vertexB: 3, faceA: 1, faceB: 3),
        ]
        let digest = ConvexHullAsset.geometryDigest(
            vertices: vertices, triangles: triangles
        )

        XCTAssertNoThrow(try ConvexHullAsset(
            vertices: vertices,
            triangles: triangles,
            edges: edges,
            boundsMin: origin,
            boundsMax: origin + F3(repeating: 1),
            boundingRadius: sqrt(3) * 0.5,
            volume: 1.0 / 6.0,
            centroid: origin + F3(repeating: 0.25),
            digest: digest,
            stableID: "hull-" + digest.prefix(16)
        ))
    }
}
