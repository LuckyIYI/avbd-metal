import CryptoKit
import Darwin
import Foundation
import simd

public enum ConvexAssetLimits {
    public static let maximumEncodedBytes = 16 * 1024 * 1024
    public static let maximumSourceURIBytes = 4_096
    public static let maximumParts = 256
    public static let maximumVerticesPerHull = 64
    public static let maximumTrianglesPerHull = 124
    public static let maximumEdgesPerHull = 186
    /// Per-input-face cap: clipping two faces may consume their combined
    /// boundary size in the Metal kernel's fixed 32-vertex workspace.
    public static let maximumFaceVertices = 16
}

/// A malformed or incompatible offline-cooked convex asset.
public struct ConvexAssetValidationError: Error, Equatable, Sendable,
    CustomStringConvertible
{
    public let description: String

    public init(_ description: String) {
        self.description = description
    }
}

/// An undirected hull edge and the two triangles incident on it.
///
/// Vertices and faces are stored in ascending order. This compact adjacency is
/// used both by collision feature tracking and by wireframe debug rendering.
public struct ConvexHullEdge: Codable, Equatable, Hashable, Sendable {
    public let vertexA: UInt32
    public let vertexB: UInt32
    public let faceA: UInt32
    public let faceB: UInt32

    public init(vertexA: UInt32, vertexB: UInt32,
                faceA: UInt32, faceB: UInt32) {
        self.vertexA = vertexA
        self.vertexB = vertexB
        self.faceA = faceA
        self.faceB = faceB
    }
}

/// One canonical, closed convex hull in the source mesh coordinate frame.
///
/// The value owns geometry once. Scene colliders refer to an entry in a
/// scene-level `[ConvexHullAsset]`, allowing every replica to reuse CPU and GPU
/// ranges rather than copying vertex arrays onto each collider.
public struct ConvexHullAsset: Codable, Equatable, Sendable {
    public let vertices: [F3]
    public let triangles: [SIMD3<UInt32>]
    public let edges: [ConvexHullEdge]
    public let boundsMin: F3
    public let boundsMax: F3
    /// Radius about the local AABB center, not about the source origin.
    public let boundingRadius: Float
    public let volume: Float
    public let centroid: F3
    /// SHA-256 of canonical Float32 vertices and UInt32 triangles.
    public let digest: String
    /// Content-stable short identity: `hull-` plus 16 digest hex digits.
    public let stableID: String

    public init(
        vertices: [F3],
        triangles: [SIMD3<UInt32>],
        edges: [ConvexHullEdge],
        boundsMin: F3,
        boundsMax: F3,
        boundingRadius: Float,
        volume: Float,
        centroid: F3,
        digest: String,
        stableID: String
    ) throws {
        self.vertices = vertices
        self.triangles = triangles
        self.edges = edges
        self.boundsMin = boundsMin
        self.boundsMax = boundsMax
        self.boundingRadius = boundingRadius
        self.volume = volume
        self.centroid = centroid
        self.digest = digest
        self.stableID = stableID
        try validate()
    }

    private enum CodingKeys: String, CodingKey {
        case vertices, triangles, edges, boundsMin, boundsMax
        case boundingRadius, volume, centroid, digest, stableID
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        vertices = try Self.decodeFloat3Array(container, forKey: .vertices)
        triangles = try Self.decodeUInt3Array(container, forKey: .triangles)
        edges = try Self.decodeEdges(container, forKey: .edges)
        boundsMin = try Self.decodeFloat3(container, forKey: .boundsMin)
        boundsMax = try Self.decodeFloat3(container, forKey: .boundsMax)
        boundingRadius = try container.decode(Float.self, forKey: .boundingRadius)
        volume = try container.decode(Float.self, forKey: .volume)
        centroid = try Self.decodeFloat3(container, forKey: .centroid)
        digest = try container.decode(String.self, forKey: .digest)
        stableID = try container.decode(String.self, forKey: .stableID)
        do {
            try validate()
        } catch let error as ConvexAssetValidationError {
            throw DecodingError.dataCorrupted(.init(
                codingPath: decoder.codingPath,
                debugDescription: error.description
            ))
        }
    }

    public func encode(to encoder: Encoder) throws {
        try validate()
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(vertices.map { [$0.x, $0.y, $0.z] }, forKey: .vertices)
        try container.encode(triangles.map { [$0.x, $0.y, $0.z] }, forKey: .triangles)
        try container.encode(edges, forKey: .edges)
        try container.encode([boundsMin.x, boundsMin.y, boundsMin.z], forKey: .boundsMin)
        try container.encode([boundsMax.x, boundsMax.y, boundsMax.z], forKey: .boundsMax)
        try container.encode(boundingRadius, forKey: .boundingRadius)
        try container.encode(volume, forKey: .volume)
        try container.encode([centroid.x, centroid.y, centroid.z], forKey: .centroid)
        try container.encode(digest, forKey: .digest)
        try container.encode(stableID, forKey: .stableID)
    }

    public func validate() throws {
        guard vertices.count >= 4 else {
            throw ConvexAssetValidationError(
                "convex hull vertex count must be at least four"
            )
        }
        guard vertices.count <= ConvexAssetLimits.maximumVerticesPerHull else {
            throw ConvexAssetValidationError("convex hull exceeds the 64-vertex runtime limit")
        }
        guard triangles.count >= 4 else {
            throw ConvexAssetValidationError("convex hull requires at least four triangles")
        }
        guard triangles.count <= ConvexAssetLimits.maximumTrianglesPerHull else {
            throw ConvexAssetValidationError("convex hull exceeds the triangle runtime limit")
        }
        guard edges.count <= ConvexAssetLimits.maximumEdgesPerHull else {
            throw ConvexAssetValidationError("convex hull exceeds the edge runtime limit")
        }
        guard vertices.allSatisfy(Self.isFinite),
              Self.isFinite(boundsMin), Self.isFinite(boundsMax),
              Self.isFinite(centroid), boundingRadius.isFinite,
              volume.isFinite else {
            throw ConvexAssetValidationError("convex hull contains a non-finite value")
        }
        guard boundingRadius > 0, volume > 0 else {
            throw ConvexAssetValidationError("convex hull radius and volume must be positive")
        }
        guard Self.isLowercaseSHA256(digest) else {
            throw ConvexAssetValidationError("convex hull digest is not a lowercase SHA-256")
        }
        guard stableID == "hull-" + digest.prefix(16) else {
            throw ConvexAssetValidationError("convex hull stableID does not match its digest")
        }

        for index in vertices.indices.dropFirst() {
            guard Self.lexicographicallyPrecedes(vertices[index - 1], vertices[index]) else {
                throw ConvexAssetValidationError(
                    "convex hull vertices must be unique and lexicographically sorted"
                )
            }
        }

        var usedVertices = Set<UInt32>()
        var triangleKeys = Set<TriangleKey>()
        var previousTriangle: SIMD3<UInt32>?
        for (faceIndex, triangle) in triangles.enumerated() {
            let values = [triangle.x, triangle.y, triangle.z]
            guard values.allSatisfy({ $0 < UInt32(vertices.count) }) else {
                throw ConvexAssetValidationError(
                    "convex hull triangle \(faceIndex) has an out-of-range index"
                )
            }
            guard Set(values).count == 3 else {
                throw ConvexAssetValidationError(
                    "convex hull triangle \(faceIndex) repeats a vertex"
                )
            }
            let rotations = [
                triangle,
                SIMD3(triangle.y, triangle.z, triangle.x),
                SIMD3(triangle.z, triangle.x, triangle.y),
            ]
            guard triangle == rotations.min(by: Self.trianglePrecedes) else {
                throw ConvexAssetValidationError(
                    "convex hull triangle \(faceIndex) is not canonically rotated"
                )
            }
            if let previousTriangle,
               Self.trianglePrecedes(triangle, previousTriangle) {
                throw ConvexAssetValidationError(
                    "convex hull triangles are not canonically sorted"
                )
            }
            previousTriangle = triangle
            let key = TriangleKey(values.sorted())
            guard triangleKeys.insert(key).inserted else {
                throw ConvexAssetValidationError("convex hull contains duplicate triangles")
            }
            usedVertices.formUnion(values)
        }
        guard usedVertices.count == vertices.count else {
            throw ConvexAssetValidationError("convex hull contains an unused vertex")
        }

        let computedBoundsMin = vertices.reduce(F3(repeating: .infinity), min)
        let computedBoundsMax = vertices.reduce(F3(repeating: -.infinity), max)
        let diagonal = simd_length(computedBoundsMax - computedBoundsMin)
        guard diagonal.isFinite, diagonal > 0 else {
            throw ConvexAssetValidationError("convex hull scale is invalid")
        }
        let tolerance = max(diagonal * 1e-5, 1e-7)
        guard Self.approximatelyEqual(boundsMin, computedBoundsMin, tolerance),
              Self.approximatelyEqual(boundsMax, computedBoundsMax, tolerance) else {
            throw ConvexAssetValidationError("convex hull stored bounds are incorrect")
        }
        let boundsCenter = (computedBoundsMin + computedBoundsMax) * 0.5
        let computedRadius = vertices.reduce(Float.zero) {
            max($0, simd_length($1 - boundsCenter))
        }
        guard abs(boundingRadius - computedRadius) <= tolerance else {
            throw ConvexAssetValidationError("convex hull stored radius is incorrect")
        }

        var volume6: Float = 0
        var localCentroidNumerator = F3.zero
        // Root signed tetrahedra at a hull vertex. Computing about the source
        // origin catastrophically cancels for ordinary geometry translated far
        // from zero (for example CAD coordinates near 1e6).
        let volumeReference = boundsCenter
        var derivedEdges: [EdgeKey: [UInt32]] = [:]
        for (faceIndex, triangle) in triangles.enumerated() {
            let a = vertices[Int(triangle.x)]
            let b = vertices[Int(triangle.y)]
            let c = vertices[Int(triangle.z)]
            let normal = simd_cross(b - a, c - a)
            let normalLength = simd_length(normal)
            guard normalLength > tolerance * tolerance else {
                throw ConvexAssetValidationError(
                    "convex hull triangle \(faceIndex) is geometrically degenerate"
                )
            }
            for point in vertices {
                guard simd_dot(normal, point - a) <= tolerance * normalLength else {
                    throw ConvexAssetValidationError(
                        "convex hull triangle \(faceIndex) has a vertex outside its plane"
                    )
                }
            }
            let localA = a - volumeReference
            let localB = b - volumeReference
            let localC = c - volumeReference
            let tetra6 = simd_dot(localA, simd_cross(localB, localC))
            volume6 += tetra6
            localCentroidNumerator += (localA + localB + localC) * tetra6

            let pairs = [
                EdgeKey(triangle.x, triangle.y),
                EdgeKey(triangle.y, triangle.z),
                EdgeKey(triangle.z, triangle.x),
            ]
            for pair in pairs {
                derivedEdges[pair, default: []].append(UInt32(faceIndex))
            }
        }
        guard volume6 > 0, volume6.isFinite else {
            throw ConvexAssetValidationError("convex hull winding does not enclose positive volume")
        }
        let computedVolume = volume6 / 6
        let computedCentroid = volumeReference
            + localCentroidNumerator / (4 * volume6)
        // Volume needs a cubic tolerance. Reusing the positional tolerance
        // lets tiny hulls claim masses many orders of magnitude too large.
        // Eight Float ulps of the hull-scale cube covers the arithmetic floor;
        // the relative term covers accumulated face summation error.
        let scale = Double(diagonal)
        let dimensionalVolumeFloor = Float(max(
            Double(Float.leastNormalMagnitude),
            scale * scale * scale * Double(Float.ulpOfOne) * 8
        ))
        let volumeTolerance = max(
            dimensionalVolumeFloor,
            abs(computedVolume) * 5e-5
        )
        guard abs(volume - computedVolume) <= volumeTolerance else {
            throw ConvexAssetValidationError("convex hull stored volume is incorrect")
        }
        guard Self.approximatelyEqual(centroid, computedCentroid, tolerance * 4) else {
            throw ConvexAssetValidationError("convex hull stored centroid is incorrect")
        }

        var expectedEdges: [ConvexHullEdge] = []
        expectedEdges.reserveCapacity(derivedEdges.count)
        for key in derivedEdges.keys.sorted() {
            guard let incidentFaces = derivedEdges[key], incidentFaces.count == 2 else {
                throw ConvexAssetValidationError(
                    "convex hull is open or non-manifold at edge \(key.a)-\(key.b)"
                )
            }
            let sortedFaces = incidentFaces.sorted()
            expectedEdges.append(ConvexHullEdge(
                vertexA: key.a, vertexB: key.b,
                faceA: sortedFaces[0], faceB: sortedFaces[1]
            ))
        }
        guard edges == expectedEdges else {
            throw ConvexAssetValidationError("convex hull edge adjacency is not canonical")
        }
        guard vertices.count - edges.count + triangles.count == 2 else {
            throw ConvexAssetValidationError("convex hull topology is not spherical")
        }
        try Self.validateMergedFaceLoops(vertices: vertices, triangles: triangles)

        guard digest == Self.geometryDigest(vertices: vertices, triangles: triangles) else {
            throw ConvexAssetValidationError("convex hull digest does not match its geometry")
        }
    }

    /// Digest contract shared with `Tools/cook_convex_asset.py`.
    public static func geometryDigest(
        vertices: [F3], triangles: [SIMD3<UInt32>]
    ) -> String {
        var data = Data("avbd.convex-hull.v1\0".utf8)
        data.appendLittleEndian(UInt32(vertices.count))
        for vertex in vertices {
            data.appendLittleEndian(vertex.x.bitPattern)
            data.appendLittleEndian(vertex.y.bitPattern)
            data.appendLittleEndian(vertex.z.bitPattern)
        }
        data.appendLittleEndian(UInt32(triangles.count))
        for triangle in triangles {
            data.appendLittleEndian(triangle.x)
            data.appendLittleEndian(triangle.y)
            data.appendLittleEndian(triangle.z)
        }
        return Self.sha256(data)
    }

    /// Exact equality for the bytes covered by `geometryDigest`.
    ///
    /// Bounds, radius, volume, centroid, edges, and stable ID are validated
    /// derived metadata, but they are not part of the digest preimage. Two
    /// valid encodings of the same geometry may therefore differ slightly in
    /// those fields without constituting a SHA-256 collision.
    func hasSameDigestGeometry(as other: ConvexHullAsset) -> Bool {
        guard vertices.count == other.vertices.count,
              triangles.count == other.triangles.count else {
            return false
        }
        for (lhs, rhs) in zip(vertices, other.vertices) {
            guard lhs.x.bitPattern == rhs.x.bitPattern,
                  lhs.y.bitPattern == rhs.y.bitPattern,
                  lhs.z.bitPattern == rhs.z.bitPattern else {
                return false
            }
        }
        for (lhs, rhs) in zip(triangles, other.triangles) {
            guard lhs.x == rhs.x, lhs.y == rhs.y, lhs.z == rhs.z else {
                return false
            }
        }
        return true
    }

    /// Runtime broad-phase bounds derived from the digest-covered vertices.
    /// Serialized bounds and radius are validated metadata with a scale-aware
    /// tolerance; collision code must not let that tolerance shrink or shift
    /// the conservative runtime envelope.
    package func geometryBounds() -> (
        min: F3, max: F3, center: F3, radius: Float
    ) {
        let boundsMin = vertices.reduce(F3(repeating: .infinity), simd.min)
        let boundsMax = vertices.reduce(F3(repeating: -.infinity), simd.max)
        let center = (boundsMin + boundsMax) * 0.5
        let radius = vertices.reduce(Float.zero) {
            max($0, simd_length($1 - center))
        }
        return (boundsMin, boundsMax, center, radius)
    }

    private static func decodeFloat3<K: CodingKey>(
        _ container: KeyedDecodingContainer<K>, forKey key: K
    ) throws -> F3 {
        var values = try container.nestedUnkeyedContainer(forKey: key)
        guard values.count == nil || values.count == 3 else {
            throw DecodingError.dataCorruptedError(
                forKey: key, in: container,
                debugDescription: "expected exactly three Float components"
            )
        }
        let x = try values.decode(Float.self)
        let y = try values.decode(Float.self)
        let z = try values.decode(Float.self)
        guard values.isAtEnd else {
            throw DecodingError.dataCorruptedError(
                forKey: key, in: container,
                debugDescription: "expected exactly three Float components"
            )
        }
        return F3(x, y, z)
    }

    private static func decodeFloat3Array<K: CodingKey>(
        _ container: KeyedDecodingContainer<K>, forKey key: K
    ) throws -> [F3] {
        var values = try container.nestedUnkeyedContainer(forKey: key)
        if let count = values.count,
           count > ConvexAssetLimits.maximumVerticesPerHull {
            throw DecodingError.dataCorruptedError(
                forKey: key, in: container,
                debugDescription: "vertex count exceeds runtime limit"
            )
        }
        var result: [F3] = []
        result.reserveCapacity(min(
            values.count ?? 0, ConvexAssetLimits.maximumVerticesPerHull
        ))
        while !values.isAtEnd {
            guard result.count < ConvexAssetLimits.maximumVerticesPerHull else {
                throw DecodingError.dataCorruptedError(
                    forKey: key, in: container,
                    debugDescription: "vertex count exceeds runtime limit"
                )
            }
            var point = try values.nestedUnkeyedContainer()
            guard point.count == nil || point.count == 3 else {
                throw DecodingError.dataCorruptedError(
                    forKey: key, in: container,
                    debugDescription: "vertex \(result.count) does not have three components"
                )
            }
            let x = try point.decode(Float.self)
            let y = try point.decode(Float.self)
            let z = try point.decode(Float.self)
            guard point.isAtEnd else {
                throw DecodingError.dataCorruptedError(
                    forKey: key, in: container,
                    debugDescription: "vertex \(result.count) does not have three components"
                )
            }
            result.append(F3(x, y, z))
        }
        return result
    }

    private static func decodeUInt3Array<K: CodingKey>(
        _ container: KeyedDecodingContainer<K>, forKey key: K
    ) throws -> [SIMD3<UInt32>] {
        var values = try container.nestedUnkeyedContainer(forKey: key)
        if let count = values.count,
           count > ConvexAssetLimits.maximumTrianglesPerHull {
            throw DecodingError.dataCorruptedError(
                forKey: key, in: container,
                debugDescription: "triangle count exceeds runtime limit"
            )
        }
        var result: [SIMD3<UInt32>] = []
        result.reserveCapacity(min(
            values.count ?? 0, ConvexAssetLimits.maximumTrianglesPerHull
        ))
        while !values.isAtEnd {
            guard result.count < ConvexAssetLimits.maximumTrianglesPerHull else {
                throw DecodingError.dataCorruptedError(
                    forKey: key, in: container,
                    debugDescription: "triangle count exceeds runtime limit"
                )
            }
            var triangle = try values.nestedUnkeyedContainer()
            guard triangle.count == nil || triangle.count == 3 else {
                throw DecodingError.dataCorruptedError(
                    forKey: key, in: container,
                    debugDescription: "triangle \(result.count) does not have three indices"
                )
            }
            let a = try triangle.decode(UInt32.self)
            let b = try triangle.decode(UInt32.self)
            let c = try triangle.decode(UInt32.self)
            guard triangle.isAtEnd else {
                throw DecodingError.dataCorruptedError(
                    forKey: key, in: container,
                    debugDescription: "triangle \(result.count) does not have three indices"
                )
            }
            result.append(SIMD3(a, b, c))
        }
        return result
    }

    private static func decodeEdges<K: CodingKey>(
        _ container: KeyedDecodingContainer<K>, forKey key: K
    ) throws -> [ConvexHullEdge] {
        var values = try container.nestedUnkeyedContainer(forKey: key)
        if let count = values.count,
           count > ConvexAssetLimits.maximumEdgesPerHull {
            throw DecodingError.dataCorruptedError(
                forKey: key, in: container,
                debugDescription: "edge count exceeds runtime limit"
            )
        }
        var result: [ConvexHullEdge] = []
        result.reserveCapacity(min(
            values.count ?? 0, ConvexAssetLimits.maximumEdgesPerHull
        ))
        while !values.isAtEnd {
            guard result.count < ConvexAssetLimits.maximumEdgesPerHull else {
                throw DecodingError.dataCorruptedError(
                    forKey: key, in: container,
                    debugDescription: "edge count exceeds runtime limit"
                )
            }
            result.append(try values.decode(ConvexHullEdge.self))
        }
        return result
    }

    fileprivate static func isLowercaseSHA256(_ value: String) -> Bool {
        value.count == 64 && value.utf8.allSatisfy {
            (48...57).contains($0) || (97...102).contains($0)
        }
    }

    private static func isFinite(_ value: F3) -> Bool {
        value.x.isFinite && value.y.isFinite && value.z.isFinite
    }

    private static func approximatelyEqual(
        _ a: F3, _ b: F3, _ tolerance: Float
    ) -> Bool {
        abs(a.x - b.x) <= tolerance
            && abs(a.y - b.y) <= tolerance
            && abs(a.z - b.z) <= tolerance
    }

    private static func lexicographicallyPrecedes(_ a: F3, _ b: F3) -> Bool {
        if a.x != b.x { return a.x < b.x }
        if a.y != b.y { return a.y < b.y }
        return a.z < b.z
    }

    private static func trianglePrecedes(
        _ a: SIMD3<UInt32>, _ b: SIMD3<UInt32>
    ) -> Bool {
        if a.x != b.x { return a.x < b.x }
        if a.y != b.y { return a.y < b.y }
        return a.z < b.z
    }

    /// Mirrors `GPUSolver.makeConvexPolygonTopology`: maximal coplanar faces
    /// are the actual clipping features, so their loops must fit the Metal
    /// kernel's fixed 32-vertex workspace even when the triangle soup and the
    /// whole hull fit their independent limits. Each input is capped at 16
    /// because clipping two faces can require their combined boundary size.
    static func validateMergedFaceLoops(
        vertices: [F3], triangles: [SIMD3<UInt32>]
    ) throws {
        let lo = vertices.reduce(F3(repeating: .infinity), simd.min)
        let hi = vertices.reduce(F3(repeating: -.infinity), simd.max)
        let scale = max(simd_length(hi - lo), 1e-4)
        let planeTolerance = max(scale * 2e-6, 1e-7)
        let normalTolerance: Float = 5e-5
        var groups: [PlaneGroup] = []
        groups.reserveCapacity(triangles.count)

        for (triangleIndex, triangle) in triangles.enumerated() {
            let a = vertices[Int(triangle.x)]
            let b = vertices[Int(triangle.y)]
            let c = vertices[Int(triangle.z)]
            let crossValue = simd_cross(b - a, c - a)
            let crossLength = simd_length(crossValue)
            guard crossLength.isFinite, crossLength > 1e-12 else {
                throw ConvexAssetValidationError(
                    "convex hull triangle \(triangleIndex) is degenerate"
                )
            }
            let normal = crossValue / crossLength
            let distance = simd_dot(normal, a)
            if let groupIndex = groups.firstIndex(where: {
                simd_dot($0.normal, normal) > 0
                    && simd_length(simd_cross($0.normal, normal)) <= normalTolerance
                    && abs($0.distance - distance) <= planeTolerance
            }) {
                groups[groupIndex].triangles.append(triangle)
            } else {
                groups.append(PlaneGroup(
                    normal: normal,
                    distance: distance,
                    triangles: [triangle]
                ))
            }
        }

        var polygonEdges: [EdgeKey: [UInt32]] = [:]
        for (groupIndex, group) in groups.enumerated() {
            var counts: [EdgeKey: Int] = [:]
            for triangle in group.triangles {
                for edge in [
                    EdgeKey(triangle.x, triangle.y),
                    EdgeKey(triangle.y, triangle.z),
                    EdgeKey(triangle.z, triangle.x),
                ] {
                    counts[edge, default: 0] += 1
                }
            }
            let boundary = counts.compactMap { key, count in
                count == 1 ? key : nil
            }.sorted()
            guard boundary.count >= 3 else {
                throw ConvexAssetValidationError(
                    "coplanar face \(groupIndex) has no closed boundary"
                )
            }

            var adjacency: [UInt32: [UInt32]] = [:]
            for edge in boundary {
                adjacency[edge.a, default: []].append(edge.b)
                adjacency[edge.b, default: []].append(edge.a)
            }
            for vertex in adjacency.keys {
                adjacency[vertex]!.sort()
            }
            guard adjacency.values.allSatisfy({ $0.count == 2 }),
                  let start = adjacency.keys.min(),
                  let first = adjacency[start]?.first else {
                throw ConvexAssetValidationError(
                    "coplanar face \(groupIndex) is not one convex loop"
                )
            }

            var loop = [start]
            loop.reserveCapacity(boundary.count)
            var previous = start
            var current = first
            while current != start, loop.count <= boundary.count {
                loop.append(current)
                guard let neighbors = adjacency[current],
                      let next = neighbors.first(where: { $0 != previous }) else {
                    break
                }
                previous = current
                current = next
            }
            guard current == start, loop.count == boundary.count,
                  Set(loop).count == boundary.count else {
                throw ConvexAssetValidationError(
                    "coplanar face \(groupIndex) boundary is disconnected"
                )
            }
            guard loop.count <= ConvexAssetLimits.maximumFaceVertices else {
                throw ConvexAssetValidationError(
                    "coplanar face \(groupIndex) has \(loop.count) vertices; "
                        + "runtime supports at most \(ConvexAssetLimits.maximumFaceVertices)"
                )
            }
            for index in loop.indices {
                let next = loop[(index + 1) % loop.count]
                polygonEdges[EdgeKey(loop[index], next), default: []]
                    .append(UInt32(groupIndex))
            }
        }
        guard polygonEdges.values.allSatisfy({ $0.count == 2 }) else {
            throw ConvexAssetValidationError("merged convex polygon topology is not manifold")
        }
    }

    fileprivate static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private struct TriangleKey: Hashable {
        let a: UInt32
        let b: UInt32
        let c: UInt32

        init(_ values: [UInt32]) {
            a = values[0]
            b = values[1]
            c = values[2]
        }
    }

    private struct EdgeKey: Hashable, Comparable {
        let a: UInt32
        let b: UInt32

        init(_ a: UInt32, _ b: UInt32) {
            self.a = min(a, b)
            self.b = max(a, b)
        }

        static func < (lhs: EdgeKey, rhs: EdgeKey) -> Bool {
            lhs.a == rhs.a ? lhs.b < rhs.b : lhs.a < rhs.a
        }
    }

    private struct PlaneGroup {
        let normal: F3
        let distance: Float
        var triangles: [SIMD3<UInt32>]
    }
}

public struct ConvexAssetSourceMetadata: Codable, Equatable, Sendable {
    public let uri: String
    public let byteCount: Int
    public let sha256: String
    public let geometrySHA256: String
    public let upAxis: String
    public let bakedScale: F3

    public init(uri: String, byteCount: Int, sha256: String,
                geometrySHA256: String, upAxis: String, bakedScale: F3) throws {
        self.uri = uri
        self.byteCount = byteCount
        self.sha256 = sha256
        self.geometrySHA256 = geometrySHA256
        self.upAxis = upAxis
        self.bakedScale = bakedScale
        try validate()
    }

    private enum CodingKeys: String, CodingKey {
        case uri, byteCount, sha256, geometrySHA256, upAxis, bakedScale
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        uri = try container.decode(String.self, forKey: .uri)
        byteCount = try container.decode(Int.self, forKey: .byteCount)
        sha256 = try container.decode(String.self, forKey: .sha256)
        geometrySHA256 = try container.decode(String.self, forKey: .geometrySHA256)
        upAxis = try container.decode(String.self, forKey: .upAxis)
        var scale = try container.nestedUnkeyedContainer(forKey: .bakedScale)
        guard scale.count == nil || scale.count == 3 else {
            throw DecodingError.dataCorruptedError(
                forKey: .bakedScale, in: container,
                debugDescription: "baked scale must contain three values"
            )
        }
        let x = try scale.decode(Float.self)
        let y = try scale.decode(Float.self)
        let z = try scale.decode(Float.self)
        guard scale.isAtEnd else {
            throw DecodingError.dataCorruptedError(
                forKey: .bakedScale, in: container,
                debugDescription: "baked scale must contain three values"
            )
        }
        bakedScale = F3(x, y, z)
        do {
            try validate()
        } catch let error as ConvexAssetValidationError {
            throw DecodingError.dataCorrupted(.init(
                codingPath: decoder.codingPath,
                debugDescription: error.description
            ))
        }
    }

    public func encode(to encoder: Encoder) throws {
        try validate()
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(uri, forKey: .uri)
        try container.encode(byteCount, forKey: .byteCount)
        try container.encode(sha256, forKey: .sha256)
        try container.encode(geometrySHA256, forKey: .geometrySHA256)
        try container.encode(upAxis, forKey: .upAxis)
        try container.encode([bakedScale.x, bakedScale.y, bakedScale.z], forKey: .bakedScale)
    }

    public func validate() throws {
        guard !uri.isEmpty, !uri.contains("\0"),
              uri.lengthOfBytes(using: .utf8) <= ConvexAssetLimits.maximumSourceURIBytes else {
            throw ConvexAssetValidationError("convex source URI is invalid")
        }
        guard byteCount > 0 else {
            throw ConvexAssetValidationError("convex source byte count must be positive")
        }
        guard ConvexHullAsset.isLowercaseSHA256(sha256),
              ConvexHullAsset.isLowercaseSHA256(geometrySHA256) else {
            throw ConvexAssetValidationError("convex source digest is not a lowercase SHA-256")
        }
        guard ["x", "y", "z"].contains(upAxis) else {
            throw ConvexAssetValidationError("convex source up axis must be x, y, or z")
        }
        guard bakedScale.x.isFinite, bakedScale.y.isFinite, bakedScale.z.isFinite,
              bakedScale.x != 0, bakedScale.y != 0, bakedScale.z != 0 else {
            throw ConvexAssetValidationError("convex source baked scale is invalid")
        }
    }
}

public struct ConvexAssetCookerParameters: Codable, Equatable, Sendable {
    public let thresholdMeters: Float
    public let maxHulls: Int
    public let maxVerticesPerHull: Int
    public let seed: UInt32
    public let weldToleranceRelative: Float
    public let weldToleranceAbsolute: Float
    public let splitConnectedComponents: Bool
    public let coacdMCTSNodes: Int
    public let coacdMCTSIterations: Int
    public let coacdMCTSMaxDepth: Int
    public let coacdPreprocessMode: String
    public let coacdPreprocessResolution: Int
    public let coacdResolution: Int
    public let coacdMerge: Bool
    public let coacdDecimate: Bool
    public let coacdExtrude: Bool
    public let coacdExtrudeMargin: Float
    public let coacdPCA: Bool
    public let coacdApproximationMode: String
    public let coacdRealMetric: Bool

    public init(
        thresholdMeters: Float = 0.05,
        maxHulls: Int = 64,
        maxVerticesPerHull: Int = 64,
        seed: UInt32 = 0,
        weldToleranceRelative: Float = 1e-7,
        weldToleranceAbsolute: Float = 1e-9,
        splitConnectedComponents: Bool = true,
        coacdMCTSNodes: Int = 20,
        coacdMCTSIterations: Int = 150,
        coacdMCTSMaxDepth: Int = 3,
        coacdPreprocessMode: String = "auto",
        coacdPreprocessResolution: Int = 50,
        coacdResolution: Int = 2_000,
        coacdMerge: Bool = true,
        coacdDecimate: Bool = true,
        coacdExtrude: Bool = false,
        coacdExtrudeMargin: Float = 0.01,
        coacdPCA: Bool = false,
        coacdApproximationMode: String = "ch",
        coacdRealMetric: Bool = true
    ) throws {
        self.thresholdMeters = thresholdMeters
        self.maxHulls = maxHulls
        self.maxVerticesPerHull = maxVerticesPerHull
        self.seed = seed
        self.weldToleranceRelative = weldToleranceRelative
        self.weldToleranceAbsolute = weldToleranceAbsolute
        self.splitConnectedComponents = splitConnectedComponents
        self.coacdMCTSNodes = coacdMCTSNodes
        self.coacdMCTSIterations = coacdMCTSIterations
        self.coacdMCTSMaxDepth = coacdMCTSMaxDepth
        self.coacdPreprocessMode = coacdPreprocessMode
        self.coacdPreprocessResolution = coacdPreprocessResolution
        self.coacdResolution = coacdResolution
        self.coacdMerge = coacdMerge
        self.coacdDecimate = coacdDecimate
        self.coacdExtrude = coacdExtrude
        self.coacdExtrudeMargin = coacdExtrudeMargin
        self.coacdPCA = coacdPCA
        self.coacdApproximationMode = coacdApproximationMode
        self.coacdRealMetric = coacdRealMetric
        try validate()
    }

    public func validate() throws {
        guard thresholdMeters.isFinite, thresholdMeters > 0,
              weldToleranceRelative.isFinite, weldToleranceRelative > 0,
              weldToleranceAbsolute.isFinite, weldToleranceAbsolute > 0,
              coacdExtrudeMargin.isFinite, coacdExtrudeMargin > 0 else {
            throw ConvexAssetValidationError("convex cooker tolerances must be positive and finite")
        }
        guard (1...ConvexAssetLimits.maximumParts).contains(maxHulls),
              (4...ConvexAssetLimits.maximumVerticesPerHull).contains(maxVerticesPerHull),
              seed <= UInt32(Int32.max) else {
            throw ConvexAssetValidationError("convex cooker hull limits are invalid")
        }
        guard splitConnectedComponents else {
            throw ConvexAssetValidationError("convex cooker must split connected components")
        }
        guard (1...Int(UInt32.max)).contains(coacdMCTSNodes),
              (1...Int(UInt32.max)).contains(coacdMCTSIterations),
              coacdMCTSMaxDepth > 0, coacdPreprocessResolution > 0,
              coacdResolution > 0,
              coacdMCTSMaxDepth <= Int(UInt32.max),
              coacdPreprocessResolution <= Int(UInt32.max),
              coacdResolution <= Int(UInt32.max) else {
            throw ConvexAssetValidationError("convex cooker CoACD counts must be positive")
        }
        guard ["auto", "on", "off"].contains(coacdPreprocessMode) else {
            throw ConvexAssetValidationError("convex cooker preprocess mode is invalid")
        }
        guard ["ch", "box"].contains(coacdApproximationMode), coacdRealMetric else {
            throw ConvexAssetValidationError("convex cooker approximation/metric mode is invalid")
        }
    }
}

public struct ConvexAssetCookerMetadata: Codable, Equatable, Sendable {
    public let algorithm: String
    public let backend: String
    public let backendVersion: String
    public let parameters: ConvexAssetCookerParameters

    public init(algorithm: String, backend: String, backendVersion: String,
                parameters: ConvexAssetCookerParameters) throws {
        self.algorithm = algorithm
        self.backend = backend
        self.backendVersion = backendVersion
        self.parameters = parameters
        try validate()
    }

    public func validate() throws {
        switch algorithm {
        case "hull":
            guard backend == "avbd-incremental-hull", backendVersion == "1" else {
                throw ConvexAssetValidationError("builtin hull cooker identity is not pinned")
            }
        case "coacd":
            guard backend == "coacd-python", backendVersion == "1.0.11" else {
                throw ConvexAssetValidationError("CoACD cooker identity is not pinned")
            }
        default:
            throw ConvexAssetValidationError("convex cooker algorithm is invalid")
        }
        try parameters.validate()
    }
}

/// A provenance-sealed set of convex parts, all expressed in one source frame.
public struct ConvexCompoundAsset: Codable, Equatable, Sendable {
    public static let currentSchemaVersion = 1
    public static let kindIdentifier = "avbd.convex-compound"

    public let schemaVersion: Int
    public let kind: String
    public let source: ConvexAssetSourceMetadata
    public let cooker: ConvexAssetCookerMetadata
    public let cacheKey: String
    public let digest: String
    public let parts: [ConvexHullAsset]

    private enum CodingKeys: String, CodingKey {
        case schemaVersion, kind, source, cooker, cacheKey, digest, parts
    }

    public init(
        schemaVersion: Int = Self.currentSchemaVersion,
        kind: String = Self.kindIdentifier,
        source: ConvexAssetSourceMetadata,
        cooker: ConvexAssetCookerMetadata,
        cacheKey: String,
        digest: String,
        parts: [ConvexHullAsset]
    ) throws {
        self.schemaVersion = schemaVersion
        self.kind = kind
        self.source = source
        self.cooker = cooker
        self.cacheKey = cacheKey
        self.digest = digest
        self.parts = parts
        try validate()
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion = try container.decode(Int.self, forKey: .schemaVersion)
        kind = try container.decode(String.self, forKey: .kind)
        source = try container.decode(ConvexAssetSourceMetadata.self, forKey: .source)
        cooker = try container.decode(ConvexAssetCookerMetadata.self, forKey: .cooker)
        cacheKey = try container.decode(String.self, forKey: .cacheKey)
        digest = try container.decode(String.self, forKey: .digest)
        var encodedParts = try container.nestedUnkeyedContainer(forKey: .parts)
        if let count = encodedParts.count, count > ConvexAssetLimits.maximumParts {
            throw DecodingError.dataCorruptedError(
                forKey: .parts, in: container,
                debugDescription: "convex part count exceeds runtime limit"
            )
        }
        var decodedParts: [ConvexHullAsset] = []
        decodedParts.reserveCapacity(min(
            encodedParts.count ?? 0, ConvexAssetLimits.maximumParts
        ))
        while !encodedParts.isAtEnd {
            guard decodedParts.count < ConvexAssetLimits.maximumParts else {
                throw DecodingError.dataCorruptedError(
                    forKey: .parts, in: container,
                    debugDescription: "convex part count exceeds runtime limit"
                )
            }
            decodedParts.append(try encodedParts.decode(ConvexHullAsset.self))
        }
        parts = decodedParts
        do {
            try validate()
        } catch let error as ConvexAssetValidationError {
            throw DecodingError.dataCorrupted(.init(
                codingPath: decoder.codingPath,
                debugDescription: error.description
            ))
        }
    }

    public func encode(to encoder: Encoder) throws {
        try validate()
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(schemaVersion, forKey: .schemaVersion)
        try container.encode(kind, forKey: .kind)
        try container.encode(source, forKey: .source)
        try container.encode(cooker, forKey: .cooker)
        try container.encode(cacheKey, forKey: .cacheKey)
        try container.encode(digest, forKey: .digest)
        try container.encode(parts, forKey: .parts)
    }

    public func validate() throws {
        guard schemaVersion == Self.currentSchemaVersion else {
            throw ConvexAssetValidationError(
                "unsupported convex asset schema version \(schemaVersion)"
            )
        }
        guard kind == Self.kindIdentifier else {
            throw ConvexAssetValidationError("unsupported convex asset kind \(kind)")
        }
        try source.validate()
        try cooker.validate()
        guard ConvexHullAsset.isLowercaseSHA256(cacheKey),
              ConvexHullAsset.isLowercaseSHA256(digest) else {
            throw ConvexAssetValidationError("convex compound digest is not a lowercase SHA-256")
        }
        guard !parts.isEmpty,
              parts.count <= ConvexAssetLimits.maximumParts,
              parts.count <= cooker.parameters.maxHulls else {
            throw ConvexAssetValidationError("convex compound part count exceeds its cooker limit")
        }
        var previousDigest: String?
        for part in parts {
            try part.validate()
            guard part.vertices.count <= cooker.parameters.maxVerticesPerHull else {
                throw ConvexAssetValidationError(
                    "convex part \(part.stableID) exceeds its cooker vertex limit"
                )
            }
            if let previousDigest, part.digest <= previousDigest {
                throw ConvexAssetValidationError(
                    part.digest == previousDigest
                        ? "convex compound contains duplicate parts"
                        : "convex compound parts are not canonically sorted"
                )
            }
            previousDigest = part.digest
        }
        guard digest == Self.contentDigest(parts: parts) else {
            throw ConvexAssetValidationError("convex compound digest does not match its parts")
        }
        let computedCacheKey = try expectedCacheKey()
        guard cacheKey == computedCacheKey else {
            throw ConvexAssetValidationError(
                "convex asset cache key does not match source/cooker metadata"
            )
        }
    }

    public static func decode(from data: Data) throws -> Self {
        guard data.count <= ConvexAssetLimits.maximumEncodedBytes else {
            throw ConvexAssetValidationError(
                "convex asset exceeds the 16 MiB encoded-size limit"
            )
        }
        return try JSONDecoder().decode(Self.self, from: data)
    }

    public static func load(from url: URL) throws -> Self {
        guard url.isFileURL else {
            throw ConvexAssetValidationError(
                "convex asset URL must identify a regular local file"
            )
        }
        let descriptor = url.path.withCString {
            Darwin.open($0, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK)
        }
        guard descriptor >= 0 else {
            let code = errno
            throw POSIXError(POSIXErrorCode(rawValue: code) ?? .EIO)
        }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        defer { try? handle.close() }

        var metadata = stat()
        guard Darwin.fstat(descriptor, &metadata) == 0 else {
            let code = errno
            throw POSIXError(POSIXErrorCode(rawValue: code) ?? .EIO)
        }
        guard (metadata.st_mode & S_IFMT) == S_IFREG else {
            throw ConvexAssetValidationError(
                "convex asset source must be a regular file"
            )
        }
        guard metadata.st_size >= 0,
              UInt64(metadata.st_size) <= UInt64(ConvexAssetLimits.maximumEncodedBytes) else {
            throw ConvexAssetValidationError(
                "convex asset exceeds the 16 MiB encoded-size limit"
            )
        }

        // Size can change after fstat. Reading the already-open descriptor up
        // to limit+1 is both bounded and race-safe; looping also handles legal
        // short reads instead of treating one as EOF.
        let readLimit = ConvexAssetLimits.maximumEncodedBytes + 1
        var data = Data()
        data.reserveCapacity(Int(metadata.st_size))
        while data.count < readLimit {
            let request = min(64 * 1024, readLimit - data.count)
            guard let chunk = try handle.read(upToCount: request),
                  !chunk.isEmpty else {
                break
            }
            data.append(chunk)
        }
        guard data.count <= ConvexAssetLimits.maximumEncodedBytes else {
            throw ConvexAssetValidationError(
                "convex asset exceeds the 16 MiB encoded-size limit"
            )
        }
        return try decode(from: data)
    }

    /// Intern parts by their full content digest and return this compound's
    /// scene-table indices in part order. A SHA collision with unequal content
    /// is rejected instead of silently aliasing geometry.
    @discardableResult
    public func internParts(into table: inout [ConvexHullAsset]) throws -> [Int] {
        var indexByDigest: [String: Int] = [:]
        for (index, part) in table.enumerated() {
            if let prior = indexByDigest[part.digest],
               !table[prior].hasSameDigestGeometry(as: part) {
                throw ConvexAssetValidationError(
                    "scene convex table contains a digest collision for \(part.digest)"
                )
            }
            indexByDigest[part.digest] = index
        }
        return try parts.map { part in
            if let index = indexByDigest[part.digest] {
                guard table[index].hasSameDigestGeometry(as: part) else {
                    throw ConvexAssetValidationError(
                        "convex digest collision for \(part.digest)"
                    )
                }
                return index
            }
            let index = table.count
            table.append(part)
            indexByDigest[part.digest] = index
            return index
        }
    }

    public static func contentDigest(parts: [ConvexHullAsset]) -> String {
        var data = Data("avbd.convex-compound.v1\0".utf8)
        data.appendLittleEndian(UInt32(parts.count))
        for part in parts {
            data.append(contentsOf: Self.bytes(fromSHA256: part.digest))
        }
        return ConvexHullAsset.sha256(data)
    }

    /// Cross-language cache/provenance seal shared with
    /// `Tools/cook_convex_asset.py`. It deliberately includes the raw source
    /// identity as well as normalized geometry, so two differently-authored
    /// source files cannot alias one cache entry.
    public func expectedCacheKey() throws -> String {
        try source.validate()
        try cooker.validate()
        var data = Data("avbd.convex-cook-key.v1\0".utf8)
        data.append(contentsOf: try Self.sha256Bytes(source.sha256))
        data.appendLittleEndian(UInt64(source.byteCount))
        try data.appendLengthPrefixedUTF8(source.uri)
        data.append(contentsOf: try Self.sha256Bytes(source.geometrySHA256))
        data.appendFloat32(source.bakedScale.x)
        data.appendFloat32(source.bakedScale.y)
        data.appendFloat32(source.bakedScale.z)
        try data.appendLengthPrefixedUTF8(source.upAxis)
        try data.appendLengthPrefixedUTF8(cooker.algorithm)
        try data.appendLengthPrefixedUTF8(cooker.backend)
        try data.appendLengthPrefixedUTF8(cooker.backendVersion)
        data.appendLittleEndian(UInt32(1))

        let parameters = cooker.parameters
        data.appendFloat32(parameters.thresholdMeters)
        data.appendLittleEndian(UInt32(parameters.maxHulls))
        data.appendLittleEndian(UInt32(parameters.maxVerticesPerHull))
        data.appendLittleEndian(parameters.seed)
        data.appendFloat32(parameters.weldToleranceRelative)
        data.appendFloat32(parameters.weldToleranceAbsolute)
        data.appendBoolean(parameters.splitConnectedComponents)
        data.appendLittleEndian(UInt32(parameters.coacdMCTSNodes))
        data.appendLittleEndian(UInt32(parameters.coacdMCTSIterations))
        data.appendLittleEndian(UInt32(parameters.coacdMCTSMaxDepth))
        try data.appendLengthPrefixedUTF8(parameters.coacdPreprocessMode)
        data.appendLittleEndian(UInt32(parameters.coacdPreprocessResolution))
        data.appendLittleEndian(UInt32(parameters.coacdResolution))
        data.appendBoolean(parameters.coacdMerge)
        data.appendBoolean(parameters.coacdDecimate)
        data.appendBoolean(parameters.coacdExtrude)
        data.appendFloat32(parameters.coacdExtrudeMargin)
        data.appendBoolean(parameters.coacdPCA)
        try data.appendLengthPrefixedUTF8(parameters.coacdApproximationMode)
        data.appendBoolean(parameters.coacdRealMetric)
        return ConvexHullAsset.sha256(data)
    }

    private static func bytes(fromSHA256 digest: String) -> [UInt8] {
        stride(from: 0, to: digest.count, by: 2).compactMap { offset in
            let start = digest.index(digest.startIndex, offsetBy: offset)
            let end = digest.index(start, offsetBy: 2)
            return UInt8(digest[start..<end], radix: 16)
        }
    }

    private static func sha256Bytes(_ digest: String) throws -> [UInt8] {
        guard ConvexHullAsset.isLowercaseSHA256(digest) else {
            throw ConvexAssetValidationError("cache-key input is not a lowercase SHA-256")
        }
        let bytes = bytes(fromSHA256: digest)
        guard bytes.count == 32 else {
            throw ConvexAssetValidationError("cache-key SHA-256 could not be decoded")
        }
        return bytes
    }
}

private extension Data {
    mutating func appendLittleEndian(_ value: UInt32) {
        var littleEndian = value.littleEndian
        Swift.withUnsafeBytes(of: &littleEndian) { bytes in
            append(contentsOf: bytes)
        }
    }

    mutating func appendLittleEndian(_ value: UInt64) {
        var littleEndian = value.littleEndian
        Swift.withUnsafeBytes(of: &littleEndian) { bytes in
            append(contentsOf: bytes)
        }
    }

    mutating func appendFloat32(_ value: Float) {
        appendLittleEndian((value == 0 ? Float.zero : value).bitPattern)
    }

    mutating func appendBoolean(_ value: Bool) {
        append(value ? 1 : 0)
    }

    mutating func appendLengthPrefixedUTF8(_ value: String) throws {
        let encoded = Array(value.utf8)
        guard encoded.count <= Int(UInt32.max) else {
            throw ConvexAssetValidationError("cache-key string is too long")
        }
        appendLittleEndian(UInt32(encoded.count))
        append(contentsOf: encoded)
    }
}
