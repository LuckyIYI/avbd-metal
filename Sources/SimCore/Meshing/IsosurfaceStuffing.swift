import Foundation
import simd

/// Isosurface stuffing (Labelle and Shewchuk 2007): fill the region where a
/// signed distance field is negative with a uniform tetrahedral mesh cut
/// from a body-centred cubic lattice. Lattice vertices that lie close to the
/// surface are warped onto it (alpha thresholds), which is what bounds the
/// dihedral angles; the remaining crossings are cut and each lattice tet is
/// clipped to its inside part. Clipped cells are tetrahedralised with a
/// face-local rule (fan from the smallest global vertex id), so shared faces
/// triangulate identically in both cells and the mesh is conforming.
///
/// Output quality depends on the warp thresholds only, not on the field's
/// smoothness; a coarse spacing simply rounds off features thinner than a
/// lattice cell. Offsetting the field outward (`fieldOffset` < 0) fattens
/// such features so a visual skin stays inside its simulation cage.
public enum IsosurfaceStuffing {
    public enum Error: Swift.Error, Equatable {
        case invalidOptions
        case invalidBounds
        /// The lattice would need more vertices than `Options.maxLatticeVertices`.
        case latticeTooLarge(vertices: Int, limit: Int)
        /// The field returned a non-finite value at a lattice vertex.
        case nonFiniteField(at: F3)
    }

    public struct Options {
        /// Lattice spacing (cube edge). BCC tets have edges of spacing and
        /// spacing * sqrt(3) / 2.
        public var spacing: Float
        /// Added to every field sample: negative grows the solid.
        public var fieldOffset: Float = 0
        /// Warp thresholds as fractions of edge length (paper defaults).
        public var alphaLong: Float = 0.24871
        public var alphaShort: Float = 0.41189
        /// Upper bound on lattice vertices (grid nodes plus cube centres);
        /// a request beyond it throws instead of allocating.
        public var maxLatticeVertices: Int = 20_000_000
        public init(spacing: Float) { self.spacing = spacing }
    }

    public struct TetMesh {
        public var nodes: [F3]
        public var tets: [SIMD4<Int32>]
        /// Outward-facing boundary triangles (faces used by one tet).
        public var boundaryFaces: [SIMD3<Int32>]

        public init(nodes: [F3], tets: [SIMD4<Int32>], boundaryFaces: [SIMD3<Int32>]) {
            self.nodes = nodes
            self.tets = tets
            self.boundaryFaces = boundaryFaces
        }

        public var volume: Float {
            tets.reduce(0) { acc, t in
                let a = nodes[Int(t.x)], b = nodes[Int(t.y)]
                let c = nodes[Int(t.z)], d = nodes[Int(t.w)]
                return acc + simd_dot(simd_cross(b - a, c - a), d - a) / 6
            }
        }

        /// Smallest and largest dihedral angle over the mesh, in degrees.
        public func dihedralRange() -> (min: Float, max: Float) {
            var lo: Float = 180, hi: Float = 0
            for t in tets {
                let p = [nodes[Int(t.x)], nodes[Int(t.y)], nodes[Int(t.z)], nodes[Int(t.w)]]
                let faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]
                var n = [F3]()
                for (a, b, c) in faces {
                    n.append(simd_normalize(simd_cross(p[b] - p[a], p[c] - p[a])))
                }
                for i in 0..<4 { for j in (i + 1)..<4 {
                    // outward normals: the dihedral is pi minus the angle
                    // between them
                    let cosang = max(-1, min(1, simd_dot(n[i], n[j])))
                    let angle = 180 - acos(cosang) * 180 / .pi
                    lo = min(lo, angle); hi = max(hi, angle)
                } }
            }
            return (lo, hi)
        }
    }

    /// Meshes the negative region of `field` inside `bounds`. Throws on
    /// non-finite or inverted bounds, non-finite options, a lattice beyond
    /// `Options.maxLatticeVertices`, or a non-finite field sample.
    public static func mesh(field: @Sendable (F3) -> Float, bounds: (lo: F3, hi: F3),
                            options: Options) throws -> TetMesh {
        let h = options.spacing
        guard h.isFinite, h > 0, options.fieldOffset.isFinite,
              options.alphaLong.isFinite, options.alphaShort.isFinite,
              options.alphaLong >= 0, options.alphaLong < 0.5,
              options.alphaShort >= 0, options.alphaShort < 0.5,
              options.maxLatticeVertices > 0 else { throw Error.invalidOptions }
        let finite = { (v: F3) in v.x.isFinite && v.y.isFinite && v.z.isFinite }
        guard finite(bounds.lo), finite(bounds.hi),
              bounds.lo.x < bounds.hi.x, bounds.lo.y < bounds.hi.y,
              bounds.lo.z < bounds.hi.z else { throw Error.invalidBounds }
        // Pad so the surface never touches the lattice boundary: one and a
        // half cells, plus whatever a negative offset grows the solid by.
        let pad = 1.5 * h + max(0, -options.fieldOffset)
        let origin = bounds.lo - F3(repeating: pad)
        let span = bounds.hi - bounds.lo + F3(repeating: 2 * pad)
        func cells(_ extent: Float) throws -> Int {
            let n = ceil(Double(extent) / Double(h))
            guard n.isFinite, n < Double(Int32.max) else {
                throw Error.latticeTooLarge(vertices: Int.max, limit: options.maxLatticeVertices)
            }
            return max(2, Int(n))
        }
        let nx = try cells(span.x), ny = try cells(span.y), nz = try cells(span.z)
        let (gridA, o1) = (nx + 1).multipliedReportingOverflow(by: ny + 1)
        let (gridCount, o2) = gridA.multipliedReportingOverflow(by: nz + 1)
        let (centA, o3) = nx.multipliedReportingOverflow(by: ny)
        let (centerCount, o4) = centA.multipliedReportingOverflow(by: nz)
        if o1 || o2 || o3 || o4 || gridCount + centerCount > options.maxLatticeVertices {
            throw Error.latticeTooLarge(
                vertices: (o1 || o2 || o3 || o4) ? Int.max : gridCount + centerCount,
                limit: options.maxLatticeVertices)
        }
        func gid(_ i: Int, _ j: Int, _ k: Int) -> Int { (i * (ny + 1) + j) * (nz + 1) + k }
        func cid(_ i: Int, _ j: Int, _ k: Int) -> Int { gridCount + (i * ny + j) * nz + k }
        var pos = [F3](repeating: .zero, count: gridCount + centerCount)
        for i in 0...nx { for j in 0...ny { for k in 0...nz {
            pos[gid(i, j, k)] = origin + F3(Float(i), Float(j), Float(k)) * h
        } } }
        for i in 0..<nx { for j in 0..<ny { for k in 0..<nz {
            pos[cid(i, j, k)] = origin + (F3(Float(i), Float(j), Float(k)) + F3(repeating: 0.5)) * h
        } } }
        // Field samples dominate the cost (a mesh-backed field walks a
        // triangle grid per query); they are independent, so spread them.
        var val = [Float](repeating: 0, count: pos.count)
        val.withUnsafeMutableBufferPointer { out in
            let base = out.baseAddress!
            DispatchQueue.concurrentPerform(iterations: pos.count) { i in
                base[i] = field(pos[i]) + options.fieldOffset
            }
        }
        if let bad = val.firstIndex(where: { !$0.isFinite }) {
            throw Error.nonFiniteField(at: pos[bad])
        }

        // BCC tets: two face-adjacent cube centres plus an edge of the shared
        // face. Every tet has two long (black) edges, centre-centre and
        // grid-grid, and four short (red) centre-grid edges.
        var tets: [SIMD4<Int32>] = []
        tets.reserveCapacity(centerCount * 12)
        let faceEdges: [[(SIMD3<Int>, SIMD3<Int>)]] = [
            // +x face of cube (i,j,k): grid nodes (i+1, j..j+1, k..k+1)
            [(SIMD3(1, 0, 0), SIMD3(1, 1, 0)), (SIMD3(1, 1, 0), SIMD3(1, 1, 1)),
             (SIMD3(1, 1, 1), SIMD3(1, 0, 1)), (SIMD3(1, 0, 1), SIMD3(1, 0, 0))],
            [(SIMD3(0, 1, 0), SIMD3(1, 1, 0)), (SIMD3(1, 1, 0), SIMD3(1, 1, 1)),
             (SIMD3(1, 1, 1), SIMD3(0, 1, 1)), (SIMD3(0, 1, 1), SIMD3(0, 1, 0))],
            [(SIMD3(0, 0, 1), SIMD3(1, 0, 1)), (SIMD3(1, 0, 1), SIMD3(1, 1, 1)),
             (SIMD3(1, 1, 1), SIMD3(0, 1, 1)), (SIMD3(0, 1, 1), SIMD3(0, 0, 1))],
        ]
        for i in 0..<nx { for j in 0..<ny { for k in 0..<nz {
            let c0 = cid(i, j, k)
            for axis in 0..<3 {
                let ni = i + (axis == 0 ? 1 : 0), nj = j + (axis == 1 ? 1 : 0)
                let nk = k + (axis == 2 ? 1 : 0)
                guard ni < nx, nj < ny, nk < nz else { continue }
                let c1 = cid(ni, nj, nk)
                for (ea, eb) in faceEdges[axis] {
                    let g0 = gid(i + ea.x, j + ea.y, k + ea.z)
                    let g1 = gid(i + eb.x, j + eb.y, k + eb.z)
                    tets.append(SIMD4(Int32(c0), Int32(c1), Int32(g0), Int32(g1)))
                }
            }
        } } }

        // Warp: a vertex whose incident crossing is closer than the alpha
        // fraction of that edge moves onto the crossing and becomes a zero.
        func isLong(_ a: Int, _ b: Int) -> Bool {
            (a < gridCount) == (b < gridCount)
        }
        var warpTarget = [F3?](repeating: nil, count: pos.count)
        var warpDistance = [Float](repeating: .infinity, count: pos.count)
        func consider(_ a: Int, _ b: Int) {
            let va = val[a], vb = val[b]
            guard (va < 0 && vb > 0) || (va > 0 && vb < 0) else { return }
            let t = va / (va - vb)
            let cut = pos[a] + (pos[b] - pos[a]) * t
            let alpha = isLong(a, b) ? options.alphaLong : options.alphaShort
            let len = simd_length(pos[b] - pos[a])
            let da = t * len, db = (1 - t) * len
            if t < alpha, da < warpDistance[a] { warpDistance[a] = da; warpTarget[a] = cut }
            if 1 - t < alpha, db < warpDistance[b] { warpDistance[b] = db; warpTarget[b] = cut }
        }
        var edgeSeen = Set<UInt64>()
        func edgeKey(_ a: Int, _ b: Int) -> UInt64 {
            UInt64(min(a, b)) << 32 | UInt64(max(a, b))
        }
        for t in tets {
            let v = [Int(t.x), Int(t.y), Int(t.z), Int(t.w)]
            for a in 0..<4 { for b in (a + 1)..<4 {
                if edgeSeen.insert(edgeKey(v[a], v[b])).inserted { consider(v[a], v[b]) }
            } }
        }
        for i in 0..<pos.count where warpTarget[i] != nil {
            pos[i] = warpTarget[i]!
            val[i] = 0
        }

        // Clip every lattice tet to its inside part. Cut points are shared
        // through the edge key so neighbouring cells reuse the same vertex.
        var outNodes = pos
        var cutIndex: [UInt64: Int] = [:]
        func cutVertex(_ a: Int, _ b: Int) -> Int {
            let key = edgeKey(a, b)
            if let id = cutIndex[key] { return id }
            let va = val[a], vb = val[b]
            let t = va / (va - vb)
            outNodes.append(pos[a] + (pos[b] - pos[a]) * t)
            cutIndex[key] = outNodes.count - 1
            return outNodes.count - 1
        }
        var outTets: [SIMD4<Int32>] = []
        outTets.reserveCapacity(tets.count)
        let tetFaces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]
        func emit(_ a: Int, _ b: Int, _ c: Int, _ d: Int) {
            let pa = outNodes[a], pb = outNodes[b], pc = outNodes[c], pd = outNodes[d]
            let vol6 = simd_dot(simd_cross(pb - pa, pc - pa), pd - pa)
            let scale = h * h * h
            guard abs(vol6) > 1e-6 * scale else { return }
            if vol6 > 0 {
                outTets.append(SIMD4(Int32(a), Int32(b), Int32(c), Int32(d)))
            } else {
                outTets.append(SIMD4(Int32(a), Int32(b), Int32(d), Int32(c)))
            }
        }
        for t in tets {
            let v = [Int(t.x), Int(t.y), Int(t.z), Int(t.w)]
            let s = v.map { val[$0] }
            if s.allSatisfy({ $0 > 0 }) { continue }
            let negatives = s.filter { $0 < 0 }.count
            if negatives == 0 { continue }            // zeros and positives: no volume
            if s.allSatisfy({ $0 <= 0 }) {
                emit(v[0], v[1], v[2], v[3]); continue
            }
            // Polytope faces: each tet face clipped, plus the cut polygon.
            var faces: [[Int]] = []
            var cutPolygon = Set<Int>()
            for (a, b, c) in tetFaces {
                let ring = [v[a], v[b], v[c]]
                var poly: [Int] = []
                for e in 0..<3 {
                    let p = ring[e], q = ring[(e + 1) % 3]
                    if val[p] <= 0 { poly.append(p) }
                    if val[p] == 0 { cutPolygon.insert(p) }
                    if (val[p] < 0 && val[q] > 0) || (val[p] > 0 && val[q] < 0) {
                        let cut = cutVertex(p, q)
                        poly.append(cut); cutPolygon.insert(cut)
                    }
                }
                if poly.count >= 3 { faces.append(poly) }
            }
            if cutPolygon.count >= 3 {
                // Planar section of a linear field: order by angle.
                let ids = Array(cutPolygon)
                let centroid = ids.reduce(F3.zero) { $0 + outNodes[$1] } / Float(ids.count)
                var normal = F3.zero
                for i in 1..<ids.count {
                    normal += simd_cross(outNodes[ids[i - 1]] - centroid, outNodes[ids[i]] - centroid)
                }
                if simd_length(normal) < 1e-20 { normal = F3(0, 0, 1) }
                normal = simd_normalize(normal)
                let u = simd_normalize(outNodes[ids[0]] - centroid)
                let w = simd_cross(normal, u)
                let ordered = ids.sorted {
                    let da = outNodes[$0] - centroid, db = outNodes[$1] - centroid
                    return atan2(simd_dot(da, w), simd_dot(da, u)) < atan2(simd_dot(db, w), simd_dot(db, u))
                }
                faces.append(ordered)
            }
            // Fan every face from its smallest id (face-local, hence
            // consistent across cells), then cone from the polytope's
            // smallest id. Faces containing the apex are degenerate.
            let apex = faces.flatMap { $0 }.min()!
            for poly in faces where !poly.contains(apex) {
                let start = poly.firstIndex(of: poly.min()!)!
                let n = poly.count
                for k in 1..<(n - 1) {
                    let a = poly[start], b = poly[(start + k) % n], c = poly[(start + k + 1) % n]
                    emit(apex, a, b, c)
                }
            }
        }

        // Compact node numbering and collect boundary faces.
        var remap = [Int32](repeating: -1, count: outNodes.count)
        var nodes: [F3] = []
        var finalTets: [SIMD4<Int32>] = []
        finalTets.reserveCapacity(outTets.count)
        for t in outTets {
            var m = SIMD4<Int32>(repeating: 0)
            for lane in 0..<4 {
                let id = Int(t[lane])
                if remap[id] < 0 { remap[id] = Int32(nodes.count); nodes.append(outNodes[id]) }
                m[lane] = remap[id]
            }
            finalTets.append(m)
        }
        var faceUse: [SIMD3<Int32>: (count: Int, oriented: SIMD3<Int32>)] = [:]
        for t in finalTets {
            // outward faces of a positively oriented tet
            let f = [SIMD3(t.x, t.z, t.y), SIMD3(t.x, t.y, t.w), SIMD3(t.x, t.w, t.z), SIMD3(t.y, t.z, t.w)]
            for face in f {
                let sorted = SIMD3<Int32>(min(face.x, min(face.y, face.z)),
                                          max(min(face.x, face.y), min(max(face.x, face.y), face.z)),
                                          max(face.x, max(face.y, face.z)))
                faceUse[sorted, default: (0, face)].count += 1
            }
        }
        let boundary = faceUse.values.filter { $0.count == 1 }.map(\.oriented)
            .sorted { ($0.x, $0.y, $0.z) < ($1.x, $1.y, $1.z) }
        return TetMesh(nodes: nodes, tets: finalTets, boundaryFaces: boundary)
    }
}
