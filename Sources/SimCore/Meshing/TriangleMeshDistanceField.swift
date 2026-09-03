import Foundation
import simd

/// Signed distance to a closed triangle mesh. Distance comes from the
/// closest point over a uniform triangle grid; the sign comes from the
/// angle-weighted pseudonormal of the closest feature (Baerentzen and
/// Aanaes 2005), which is exact for closed manifolds and needs no ray casts.
public struct TriangleMeshDistanceField {
    public let vertices: [F3]
    public let triangles: [SIMD3<Int32>]
    private let faceNormals: [F3]
    private let vertexNormals: [F3]          // angle weighted
    private let edgeNormals: [UInt64: F3]    // key = min << 32 | max
    private let origin: F3
    private let cell: Float
    private let dims: SIMD3<Int>
    private let cellStart: [Int32]
    private let cellTris: [Int32]

    public init(mesh: SurfaceMesh, cellSize: Float? = nil) {
        vertices = mesh.vertices
        triangles = mesh.triangles.map {
            SIMD3<Int32>(Int32($0.0), Int32($0.1), Int32($0.2))
        }
        var fn = [F3](repeating: .zero, count: triangles.count)
        var vn = [F3](repeating: .zero, count: vertices.count)
        var en: [UInt64: F3] = [:]
        var edgeSum: Float = 0
        func key(_ a: Int32, _ b: Int32) -> UInt64 {
            UInt64(min(a, b)) << 32 | UInt64(max(a, b))
        }
        for (t, tri) in triangles.enumerated() {
            let a = vertices[Int(tri.x)], b = vertices[Int(tri.y)], c = vertices[Int(tri.z)]
            let n = simd_cross(b - a, c - a)
            let len = simd_length(n)
            let unit = len > 1e-20 ? n / len : F3(0, 0, 1)
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
        let meanEdge = edgeSum / Float(max(1, triangles.count * 3))
        cell = max(cellSize ?? meanEdge * 2, 1e-5)
        origin = lo - F3(repeating: cell)
        let span = hi - lo + F3(repeating: 2 * cell)
        dims = SIMD3<Int>(max(1, Int(ceil(span.x / cell))),
                          max(1, Int(ceil(span.y / cell))),
                          max(1, Int(ceil(span.z / cell))))
        let cellCount = dims.x * dims.y * dims.z
        var counts = [Int32](repeating: 0, count: cellCount + 1)
        var entries: [(Int, Int32)] = []
        entries.reserveCapacity(triangles.count * 2)
        for (t, tri) in triangles.enumerated() {
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
        var tris = [Int32](repeating: 0, count: entries.count)
        for (idx, t) in entries {
            tris[Int(cursor[idx])] = t
            cursor[idx] += 1
        }
        cellStart = counts
        cellTris = tris
    }

    private static func coord(_ p: F3, _ origin: F3, _ cell: Float,
                              _ dims: SIMD3<Int>) -> SIMD3<Int> {
        let q = (p - origin) / cell
        return SIMD3<Int>(
            min(dims.x - 1, max(0, Int(floor(q.x)))),
            min(dims.y - 1, max(0, Int(floor(q.y)))),
            min(dims.z - 1, max(0, Int(floor(q.z)))))
    }

    /// Closest point on triangle abc to p, with the closest feature encoded
    /// as (vertexIndex or -1, edge (i, j) or nil).
    private static func closest(_ p: F3, _ a: F3, _ b: F3, _ c: F3)
        -> (point: F3, feature: Int) {
        // feature: 0,1,2 = vertex a,b,c; 3,4,5 = edge ab,bc,ca; 6 = face
        let ab = b - a, ac = c - a, ap = p - a
        let d1 = simd_dot(ab, ap), d2 = simd_dot(ac, ap)
        if d1 <= 0 && d2 <= 0 { return (a, 0) }
        let bp = p - b
        let d3 = simd_dot(ab, bp), d4 = simd_dot(ac, bp)
        if d3 >= 0 && d4 <= d3 { return (b, 1) }
        let vc = d1 * d4 - d3 * d2
        if vc <= 0 && d1 >= 0 && d3 <= 0 {
            let v = d1 / (d1 - d3)
            return (a + ab * v, 3)
        }
        let cp = p - c
        let d5 = simd_dot(ab, cp), d6 = simd_dot(ac, cp)
        if d6 >= 0 && d5 <= d6 { return (c, 2) }
        let vb = d5 * d2 - d1 * d6
        if vb <= 0 && d2 >= 0 && d6 <= 0 {
            let w = d2 / (d2 - d6)
            return (a + ac * w, 5)
        }
        let va = d3 * d6 - d5 * d4
        if va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0 {
            let w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            return (b + (c - b) * w, 4)
        }
        let denom = 1 / (va + vb + vc)
        let v = vb * denom, w = vc * denom
        return (a + ab * v + ac * w, 6)
    }

    /// Signed distance: negative inside the closed surface.
    public func signedDistance(_ p: F3) -> Float {
        var best = Float.infinity
        var bestTri = -1
        var bestPoint = F3.zero
        var bestFeature = 6
        let center = Self.coord(p, origin, cell, dims)
        let maxRing = max(dims.x, max(dims.y, dims.z))
        var ring = 0
        while ring <= maxRing {
            // Anything in a farther ring is at least (ring) cells away from
            // the point's cell along some axis.
            if ring > 0 && best.isFinite {
                let reach = Float(ring - 1) * cell
                if best <= reach { break }
            }
            for i in (center.x - ring)...(center.x + ring) {
                guard i >= 0, i < dims.x else { continue }
                for j in (center.y - ring)...(center.y + ring) {
                    guard j >= 0, j < dims.y else { continue }
                    // Only the shell of the ring: full k range on the ring's
                    // side faces, the two end cells elsewhere.
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
                                p, vertices[Int(tri.x)], vertices[Int(tri.y)],
                                vertices[Int(tri.z)])
                            let d = simd_length(q - p)
                            if d < best {
                                best = d; bestTri = t; bestPoint = q
                                bestFeature = feature
                            }
                        }
                    }
                }
            }
            ring += 1
        }
        guard bestTri >= 0 else { return .infinity }
        let tri = triangles[bestTri]
        let normal: F3
        switch bestFeature {
        case 0: normal = vertexNormals[Int(tri.x)]
        case 1: normal = vertexNormals[Int(tri.y)]
        case 2: normal = vertexNormals[Int(tri.z)]
        case 3: normal = edgeNormals[UInt64(min(tri.x, tri.y)) << 32 | UInt64(max(tri.x, tri.y))] ?? faceNormals[bestTri]
        case 4: normal = edgeNormals[UInt64(min(tri.y, tri.z)) << 32 | UInt64(max(tri.y, tri.z))] ?? faceNormals[bestTri]
        case 5: normal = edgeNormals[UInt64(min(tri.z, tri.x)) << 32 | UInt64(max(tri.z, tri.x))] ?? faceNormals[bestTri]
        default: normal = faceNormals[bestTri]
        }
        return simd_dot(p - bestPoint, normal) >= 0 ? best : -best
    }
}
