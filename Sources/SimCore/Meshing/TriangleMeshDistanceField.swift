import Foundation
import simd

/// Signed distance to a triangle mesh, negative inside.
///
/// Distance is the exact closest point over a uniform triangle grid with a
/// shell search. The sign depends on what the mesh is:
/// - a closed, consistently oriented 2-manifold uses the angle-weighted
///   pseudonormal of the closest feature (Baerentzen and Aanaes 2005),
///   which is exact and costs nothing beyond the closest point;
/// - anything else (holes, non-manifold edges, inconsistent winding, as
///   scans and vertex-clustered proxies routinely are) uses the generalised
///   winding number (Jacobson et al. 2013), which degrades gracefully.
/// A globally reversed mesh is detected from its signed volume and flipped,
/// so callers never get the padded exterior meshed by mistake.
///
/// Construction throws on input that cannot be interpreted at all:
/// non-finite vertices, triangle indices out of range, or no non-degenerate
/// triangle. Degenerate triangles are dropped and counted.
public struct TriangleMeshDistanceField: Sendable {
    public enum Error: Swift.Error, Equatable {
        case nonFiniteVertex(index: Int)
        case triangleIndexOutOfRange(triangle: Int)
        case noUsableTriangles
    }

    public enum SignMethod: Sendable, Equatable {
        /// Closed consistently oriented manifold: pseudonormal sign.
        case pseudonormal
        /// Open, non-manifold or inconsistently wound: winding-number sign.
        case windingNumber
    }

    /// What the constructor found in the input.
    public struct Topology: Sendable, Equatable {
        public var degenerateTriangles: Int
        public var boundaryEdges: Int
        public var nonManifoldEdges: Int
        public var inconsistentlyWoundEdges: Int
        /// True when the signed volume was negative and every triangle was
        /// flipped to make the surface face outward.
        public var orientationFlipped: Bool
        public var isClosedManifold: Bool {
            boundaryEdges == 0 && nonManifoldEdges == 0 && inconsistentlyWoundEdges == 0
        }
    }

    public let vertices: [F3]
    public let triangles: [SIMD3<Int32>]
    public let topology: Topology
    public let signMethod: SignMethod
    private let faceNormals: [F3]
    private let vertexNormals: [F3]          // angle weighted
    private let edgeNormals: [UInt64: F3]    // key = min << 32 | max
    private let origin: F3
    private let cell: Float
    private let dims: SIMD3<Int>
    private let cellStart: [Int32]
    private let cellTris: [Int32]

    public init(mesh: SurfaceMesh, cellSize: Float? = nil) throws {
        for (i, v) in mesh.vertices.enumerated()
            where !(v.x.isFinite && v.y.isFinite && v.z.isFinite) {
            throw Error.nonFiniteVertex(index: i)
        }
        let count = mesh.vertices.count
        var tris: [SIMD3<Int32>] = []
        tris.reserveCapacity(mesh.triangles.count)
        var degenerate = 0
        for (t, tri) in mesh.triangles.enumerated() {
            guard tri.0 >= 0, tri.0 < count, tri.1 >= 0, tri.1 < count,
                  tri.2 >= 0, tri.2 < count else {
                throw Error.triangleIndexOutOfRange(triangle: t)
            }
            let a = mesh.vertices[tri.0], b = mesh.vertices[tri.1], c = mesh.vertices[tri.2]
            if tri.0 == tri.1 || tri.1 == tri.2 || tri.0 == tri.2
                || simd_length_squared(simd_cross(b - a, c - a)) <= 1e-24 {
                degenerate += 1
                continue
            }
            tris.append(SIMD3<Int32>(Int32(tri.0), Int32(tri.1), Int32(tri.2)))
        }
        guard !tris.isEmpty else { throw Error.noUsableTriangles }

        // Global orientation from the divergence theorem: a closed outward
        // surface has positive signed volume. Open meshes still get a
        // meaningful sign when most of the surface is present.
        var signedVolume: Double = 0
        for tri in tris {
            let a = mesh.vertices[Int(tri.x)], b = mesh.vertices[Int(tri.y)], c = mesh.vertices[Int(tri.z)]
            signedVolume += Double(simd_dot(a, simd_cross(b, c))) / 6
        }
        let flipped = signedVolume < 0
        if flipped { tris = tris.map { SIMD3($0.x, $0.z, $0.y) } }

        // Edge census: manifoldness and winding consistency.
        func key(_ a: Int32, _ b: Int32) -> UInt64 {
            UInt64(min(a, b)) << 32 | UInt64(max(a, b))
        }
        var edgeUse: [UInt64: (count: Int, forward: Int)] = [:]
        edgeUse.reserveCapacity(tris.count * 3)
        for tri in tris {
            for (a, b) in [(tri.x, tri.y), (tri.y, tri.z), (tri.z, tri.x)] {
                var e = edgeUse[key(a, b)] ?? (0, 0)
                e.count += 1
                if a < b { e.forward += 1 }
                edgeUse[key(a, b)] = e
            }
        }
        var boundary = 0, nonManifold = 0, inconsistent = 0
        for e in edgeUse.values {
            if e.count == 1 { boundary += 1 }
            else if e.count > 2 { nonManifold += 1 }
            else if e.forward != 1 { inconsistent += 1 }   // both faces traverse it the same way
        }
        let topology = Topology(degenerateTriangles: degenerate, boundaryEdges: boundary,
                                nonManifoldEdges: nonManifold,
                                inconsistentlyWoundEdges: inconsistent,
                                orientationFlipped: flipped)

        vertices = mesh.vertices
        triangles = tris
        self.topology = topology
        signMethod = topology.isClosedManifold ? .pseudonormal : .windingNumber

        var fn = [F3](repeating: .zero, count: tris.count)
        var vn = [F3](repeating: .zero, count: count)
        var en: [UInt64: F3] = [:]
        var edgeSum: Float = 0
        for (t, tri) in tris.enumerated() {
            let a = vertices[Int(tri.x)], b = vertices[Int(tri.y)], c = vertices[Int(tri.z)]
            let unit = simd_normalize(simd_cross(b - a, c - a))
            fn[t] = unit
            let corners = [(tri.x, a, b, c), (tri.y, b, c, a), (tri.z, c, a, b)]
            for (v, p, q, r) in corners {
                let u = simd_normalize(q - p), w = simd_normalize(r - p)
                let angle = acos(max(-1, min(1, simd_dot(u, w))))
                vn[Int(v)] += unit * angle
            }
            en[key(tri.x, tri.y), default: .zero] += unit
            en[key(tri.y, tri.z), default: .zero] += unit
            en[key(tri.z, tri.x), default: .zero] += unit
            edgeSum += simd_length(b - a) + simd_length(c - b) + simd_length(a - c)
        }
        faceNormals = fn
        vertexNormals = vn
        edgeNormals = en

        let (lo, hi) = mesh.bounds()
        let meanEdge = edgeSum / Float(tris.count * 3)
        cell = max(cellSize ?? meanEdge * 2, 1e-5)
        origin = lo - F3(repeating: cell)
        let span = hi - lo + F3(repeating: 2 * cell)
        dims = SIMD3<Int>(max(1, Int(ceil(span.x / cell))),
                          max(1, Int(ceil(span.y / cell))),
                          max(1, Int(ceil(span.z / cell))))
        let cellCount = dims.x * dims.y * dims.z
        var counts = [Int32](repeating: 0, count: cellCount + 1)
        var entries: [(Int, Int32)] = []
        entries.reserveCapacity(tris.count * 2)
        for (t, tri) in tris.enumerated() {
            let a = vertices[Int(tri.x)], b = vertices[Int(tri.y)], c = vertices[Int(tri.z)]
            let tlo = simd_min(simd_min(a, b), c), thi = simd_max(simd_max(a, b), c)
            let c0 = Self.coord(tlo, origin, cell, dims), c1 = Self.coord(thi, origin, cell, dims)
            for i in c0.x...c1.x { for j in c0.y...c1.y { for k in c0.z...c1.z {
                let idx = (i * dims.y + j) * dims.z + k
                counts[idx + 1] += 1
                entries.append((idx, Int32(t)))
            } } }
        }
        for i in 0..<cellCount { counts[i + 1] += counts[i] }
        var cursor = counts
        var packed = [Int32](repeating: 0, count: entries.count)
        for (idx, t) in entries {
            packed[Int(cursor[idx])] = t
            cursor[idx] += 1
        }
        cellStart = counts
        cellTris = packed
    }

    private static func coord(_ p: F3, _ origin: F3, _ cell: Float,
                              _ dims: SIMD3<Int>) -> SIMD3<Int> {
        let q = (p - origin) / cell
        return SIMD3<Int>(
            min(dims.x - 1, max(0, Int(floor(q.x)))),
            min(dims.y - 1, max(0, Int(floor(q.y)))),
            min(dims.z - 1, max(0, Int(floor(q.z)))))
    }

    /// Closest point on triangle abc to p; feature 0-2 vertex, 3-5 edge
    /// (ab, bc, ca), 6 face.
    private static func closest(_ p: F3, _ a: F3, _ b: F3, _ c: F3)
        -> (point: F3, feature: Int) {
        let ab = b - a, ac = c - a, ap = p - a
        let d1 = simd_dot(ab, ap), d2 = simd_dot(ac, ap)
        if d1 <= 0 && d2 <= 0 { return (a, 0) }
        let bp = p - b
        let d3 = simd_dot(ab, bp), d4 = simd_dot(ac, bp)
        if d3 >= 0 && d4 <= d3 { return (b, 1) }
        let vc = d1 * d4 - d3 * d2
        if vc <= 0 && d1 >= 0 && d3 <= 0 { return (a + ab * (d1 / (d1 - d3)), 3) }
        let cp = p - c
        let d5 = simd_dot(ab, cp), d6 = simd_dot(ac, cp)
        if d6 >= 0 && d5 <= d6 { return (c, 2) }
        let vb = d5 * d2 - d1 * d6
        if vb <= 0 && d2 >= 0 && d6 <= 0 { return (a + ac * (d2 / (d2 - d6)), 5) }
        let va = d3 * d6 - d5 * d4
        if va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0 {
            let w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            return (b + (c - b) * w, 4)
        }
        let denom = 1 / (va + vb + vc)
        return (a + ab * (vb * denom) + ac * (vc * denom), 6)
    }

    /// Unsigned distance and the closest feature.
    private func closestFeature(_ p: F3) -> (distance: Float, tri: Int, point: F3, feature: Int) {
        var best = Float.infinity
        var bestTri = -1
        var bestPoint = F3.zero
        var bestFeature = 6
        let center = Self.coord(p, origin, cell, dims)
        let maxRing = max(dims.x, max(dims.y, dims.z))
        var ring = 0
        while ring <= maxRing {
            if ring > 0 && best.isFinite && best <= Float(ring - 1) * cell { break }
            for i in (center.x - ring)...(center.x + ring) {
                guard i >= 0, i < dims.x else { continue }
                for j in (center.y - ring)...(center.y + ring) {
                    guard j >= 0, j < dims.y else { continue }
                    let onSide = abs(i - center.x) == ring || abs(j - center.y) == ring
                    let ks: [Int] = onSide
                        ? Array((center.z - ring)...(center.z + ring))
                        : (ring == 0 ? [center.z] : [center.z - ring, center.z + ring])
                    for k in ks {
                        guard k >= 0, k < dims.z else { continue }
                        let idx = (i * dims.y + j) * dims.z + k
                        for e in Int(cellStart[idx])..<Int(cellStart[idx + 1]) {
                            let t = Int(cellTris[e])
                            let tri = triangles[t]
                            let (q, feature) = Self.closest(
                                p, vertices[Int(tri.x)], vertices[Int(tri.y)], vertices[Int(tri.z)])
                            let d = simd_length(q - p)
                            if d < best { best = d; bestTri = t; bestPoint = q; bestFeature = feature }
                        }
                    }
                }
            }
            ring += 1
        }
        return (best, bestTri, bestPoint, bestFeature)
    }

    /// Generalised winding number of the surface around p (1 deep inside a
    /// closed surface, 0 outside, fractional near holes).
    public func windingNumber(_ p: F3) -> Float {
        var total: Double = 0
        for tri in triangles {
            let a = vertices[Int(tri.x)] - p, b = vertices[Int(tri.y)] - p, c = vertices[Int(tri.z)] - p
            let la = simd_length(a), lb = simd_length(b), lc = simd_length(c)
            let num = simd_dot(a, simd_cross(b, c))
            let den = la * lb * lc + simd_dot(a, b) * lc + simd_dot(b, c) * la + simd_dot(c, a) * lb
            total += Double(atan2(num, den))
        }
        return Float(total / (2 * .pi))
    }

    /// Signed distance: negative inside.
    public func signedDistance(_ p: F3) -> Float {
        let (distance, t, point, feature) = closestFeature(p)
        guard t >= 0 else { return .infinity }
        let inside: Bool
        switch signMethod {
        case .windingNumber:
            inside = windingNumber(p) > 0.5
        case .pseudonormal:
            let tri = triangles[t]
            let normal: F3
            switch feature {
            case 0: normal = vertexNormals[Int(tri.x)]
            case 1: normal = vertexNormals[Int(tri.y)]
            case 2: normal = vertexNormals[Int(tri.z)]
            case 3: normal = edgeNormals[UInt64(min(tri.x, tri.y)) << 32 | UInt64(max(tri.x, tri.y))] ?? faceNormals[t]
            case 4: normal = edgeNormals[UInt64(min(tri.y, tri.z)) << 32 | UInt64(max(tri.y, tri.z))] ?? faceNormals[t]
            case 5: normal = edgeNormals[UInt64(min(tri.z, tri.x)) << 32 | UInt64(max(tri.z, tri.x))] ?? faceNormals[t]
            default: normal = faceNormals[t]
            }
            inside = simd_dot(p - point, normal) < 0
        }
        return inside ? -distance : distance
    }
}
