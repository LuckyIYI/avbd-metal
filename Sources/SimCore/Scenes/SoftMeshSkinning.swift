import Foundation
import ModelIO
import simd

public enum MeshUpAxis {
    case x, y, z
}

public enum SurfaceMeshError: Error, CustomStringConvertible {
    case emptyAsset(String)
    case missingPositions(String)
    case unsupportedIndices(String)

    public var description: String {
        switch self {
        case .emptyAsset(let path): return "no mesh geometry in \(path)"
        case .missingPositions(let name): return "mesh \(name) has no float3 positions"
        case .unsupportedIndices(let name): return "mesh \(name) has unsupported index data"
        }
    }
}

public struct SurfaceMesh {
    public var vertices: [F3]
    public var normals: [F3]
    public var triangles: [(Int, Int, Int)]

    public init(vertices: [F3], normals: [F3], triangles: [(Int, Int, Int)]) {
        self.vertices = vertices
        self.normals = normals.count == vertices.count
            ? normals : Self.computeNormals(vertices: vertices, triangles: triangles)
        self.triangles = triangles
    }

    public static func load(path: String, upAxis: MeshUpAxis = .y) throws -> SurfaceMesh {
        let url = URL(fileURLWithPath: path)
        let asset = MDLAsset(url: url)
        var vertices: [F3] = []
        var normals: [F3] = []
        var triangles: [(Int, Int, Int)] = []

        func append(mesh: MDLMesh) throws {
            let desc = mesh.vertexDescriptor
            guard let posAttr = desc.attributeNamed(MDLVertexAttributePosition),
                  let posLayout = desc.layouts[posAttr.bufferIndex] as? MDLVertexBufferLayout
            else { throw SurfaceMeshError.missingPositions(mesh.name) }
            let posBuffer = mesh.vertexBuffers[posAttr.bufferIndex]
            let posBytes = posBuffer.map().bytes.bindMemory(to: UInt8.self,
                                                            capacity: posBuffer.length)

            let normAttr = desc.attributeNamed(MDLVertexAttributeNormal)
            let normLayout = normAttr.flatMap {
                desc.layouts[$0.bufferIndex] as? MDLVertexBufferLayout
            }
            let normBytes = normAttr.map {
                mesh.vertexBuffers[$0.bufferIndex].map().bytes.bindMemory(
                    to: UInt8.self, capacity: mesh.vertexBuffers[$0.bufferIndex].length)
            }

            let base = vertices.count
            vertices.reserveCapacity(vertices.count + mesh.vertexCount)
            normals.reserveCapacity(normals.count + mesh.vertexCount)
            for i in 0..<mesh.vertexCount {
                let pPtr = UnsafeRawPointer(posBytes.advanced(by: i * posLayout.stride
                                                              + posAttr.offset))
                    .bindMemory(to: Float.self, capacity: 3)
                vertices.append(orient(F3(pPtr[0], pPtr[1], pPtr[2]), upAxis: upAxis))
                if let normAttr, let normLayout, let normBytes {
                    let nPtr = UnsafeRawPointer(normBytes.advanced(by: i * normLayout.stride
                                                                   + normAttr.offset))
                        .bindMemory(to: Float.self, capacity: 3)
                    normals.append(safeNormalize(orient(F3(nPtr[0], nPtr[1], nPtr[2]),
                                                       upAxis: upAxis)))
                } else {
                    normals.append(.zero)
                }
            }

            guard let submeshes = mesh.submeshes, submeshes.count > 0 else {
                for i in stride(from: 0, to: mesh.vertexCount - 2, by: 3) {
                    triangles.append((base + i, base + i + 1, base + i + 2))
                }
                return
            }
            for case let submesh as MDLSubmesh in submeshes {
                guard submesh.geometryType == .triangles else { continue }
                let map = submesh.indexBuffer.map()
                let count = submesh.indexCount
                switch submesh.indexType {
                case .uInt8:
                    let p = map.bytes.bindMemory(to: UInt8.self, capacity: count)
                    for i in stride(from: 0, to: count - 2, by: 3) {
                        triangles.append((base + Int(p[i]), base + Int(p[i + 1]),
                                          base + Int(p[i + 2])))
                    }
                case .uInt16:
                    let p = map.bytes.bindMemory(to: UInt16.self, capacity: count)
                    for i in stride(from: 0, to: count - 2, by: 3) {
                        triangles.append((base + Int(p[i]), base + Int(p[i + 1]),
                                          base + Int(p[i + 2])))
                    }
                case .uInt32:
                    let p = map.bytes.bindMemory(to: UInt32.self, capacity: count)
                    for i in stride(from: 0, to: count - 2, by: 3) {
                        triangles.append((base + Int(p[i]), base + Int(p[i + 1]),
                                          base + Int(p[i + 2])))
                    }
                default:
                    throw SurfaceMeshError.unsupportedIndices(mesh.name)
                }
            }
        }

        func visit(_ object: MDLObject) throws {
            if let mesh = object as? MDLMesh { try append(mesh: mesh) }
            let children = object.children
            for i in 0..<children.count {
                try visit(children[i])
            }
        }
        for i in 0..<asset.count { try visit(asset.object(at: i)) }
        guard !vertices.isEmpty, !triangles.isEmpty else {
            throw SurfaceMeshError.emptyAsset(path)
        }
        let mesh = SurfaceMesh(vertices: vertices, normals: normals, triangles: triangles)
        return mesh.withConsistentNormals()
    }

    public func simplified(maxVertices: Int) -> SurfaceMesh {
        guard maxVertices > 0, vertices.count > maxVertices else { return self }
        let (mn, mx) = bounds()
        let extent = max(mx - mn, F3(repeating: 1e-6))

        struct Key: Hashable { var x, y, z: Int }
        func quantized(divisions div: Int) -> SurfaceMesh {
            var map: [Key: Int] = [:]
            var pos: [F3] = []
            var nrm: [F3] = []
            var counts: [Float] = []
            var remap = [Int](repeating: 0, count: vertices.count)
            for (i, p) in vertices.enumerated() {
                let q = clamp((p - mn) / extent, min: F3.zero, max: F3(repeating: 0.999999))
                let key = Key(x: Int(q.x * Float(div)), y: Int(q.y * Float(div)),
                              z: Int(q.z * Float(div)))
                if let id = map[key] {
                    pos[id] += p
                    nrm[id] += normals[i]
                    counts[id] += 1
                    remap[i] = id
                } else {
                    let id = pos.count
                    map[key] = id
                    pos.append(p)
                    nrm.append(normals[i])
                    counts.append(1)
                    remap[i] = id
                }
            }
            for i in 0..<pos.count {
                pos[i] /= counts[i]
                nrm[i] = safeNormalize(nrm[i])
            }
            var tris: [(Int, Int, Int)] = []
            tris.reserveCapacity(triangles.count)
            var seen = Set<UInt64>()
            for (a0, b0, c0) in triangles {
                let a = remap[a0], b = remap[b0], c = remap[c0]
                guard a != b, b != c, a != c else { continue }
                let sorted = [a, b, c].sorted()
                let key = UInt64(sorted[0]) << 42 | UInt64(sorted[1]) << 21 | UInt64(sorted[2])
                if seen.insert(key).inserted { tris.append((a, b, c)) }
            }
            return SurfaceMesh(vertices: pos, normals: nrm, triangles: tris).withConsistentNormals()
        }

        var div = max(2, Int(Double(maxVertices).squareRoot()))
        var best = quantized(divisions: div)
        while best.vertices.count > maxVertices && div > 2 {
            let next = max(2, Int(Float(div) * 0.78))
            if next == div { break }
            div = next
            best = quantized(divisions: div)
        }
        return best.triangles.isEmpty ? self : best
    }

    public func fitted(height: Float, center: F3) -> SurfaceMesh {
        let (mn, mx) = bounds()
        let extent = mx - mn
        let scale = height / max(extent.z, 1e-6)
        let anchor = F3((mn.x + mx.x) * 0.5, (mn.y + mx.y) * 0.5, mn.z)
        let pos = vertices.map { ($0 - anchor) * scale + center }
        return SurfaceMesh(vertices: pos, normals: normals, triangles: triangles)
    }

    public func translated(_ offset: F3) -> SurfaceMesh {
        SurfaceMesh(vertices: vertices.map { $0 + offset }, normals: normals,
                    triangles: triangles)
    }

    public func bounds() -> (F3, F3) {
        var mn = F3(repeating: Float.greatestFiniteMagnitude)
        var mx = F3(repeating: -Float.greatestFiniteMagnitude)
        for p in vertices {
            mn = min(mn, p)
            mx = max(mx, p)
        }
        return (mn, mx)
    }

    public func withConsistentNormals() -> SurfaceMesh {
        let computed = Self.computeNormals(vertices: vertices, triangles: triangles)
        var out = normals
        if out.count != vertices.count || out.allSatisfy({ length_squared($0) < 1e-12 }) {
            out = computed
        } else {
            for i in 0..<out.count {
                if length_squared(out[i]) < 1e-12 {
                    out[i] = computed[i]
                } else if dot(out[i], computed[i]) < 0 {
                    out[i] = -out[i]
                }
                out[i] = safeNormalize(out[i])
            }
        }
        return SurfaceMesh(vertices: vertices, normals: out, triangles: triangles)
    }

    static func computeNormals(vertices: [F3], triangles: [(Int, Int, Int)]) -> [F3] {
        var out = [F3](repeating: .zero, count: vertices.count)
        for (a, b, c) in triangles {
            let n = cross(vertices[b] - vertices[a], vertices[c] - vertices[a])
            out[a] += n; out[b] += n; out[c] += n
        }
        for i in 0..<out.count { out[i] = safeNormalize(out[i], fallback: F3(0, 0, 1)) }
        return out
    }

    private func contains(_ p: F3) -> Bool {
        let dir = safeNormalize(F3(1, 0.173, 0.097))
        var hits = 0
        for (ia, ib, ic) in triangles {
            if rayIntersectsTriangle(origin: p, dir: dir,
                                     a: vertices[ia], b: vertices[ib], c: vertices[ic]) {
                hits += 1
            }
        }
        return (hits & 1) == 1
    }

    private func distanceSquared(to p: F3) -> Float {
        var best = Float.greatestFiniteMagnitude
        for (ia, ib, ic) in triangles {
            best = min(best, pointTriangleDistanceSquared(p, vertices[ia],
                                                          vertices[ib], vertices[ic]))
        }
        return best
    }

    package func voxelCells(res: Int,
                           surfaceBand: Float = 0.45) -> (Set<SIMD3<Int>>, F3, Float) {
        let (mn, mx) = bounds()
        let extent = mx - mn
        let maxExtent = max(max(max(extent.x, extent.y), extent.z), 1e-5)
        let spacing = maxExtent / Float(max(2, res))
        let pad = 1
        let origin = mn - F3(repeating: Float(pad) * spacing)
        let nx = max(1, Int(ceil(extent.x / spacing)) + 2 * pad)
        let ny = max(1, Int(ceil(extent.y / spacing)) + 2 * pad)
        let nz = max(1, Int(ceil(extent.z / spacing)) + 2 * pad)
        let band = max(0, surfaceBand) * spacing
        let band2 = band * band
        let count = nx * ny * nz
        var surface = [UInt8](repeating: 0, count: count)
        var exterior = [UInt8](repeating: 0, count: count)
        func idx(_ i: Int, _ j: Int, _ k: Int) -> Int {
            (k * ny + j) * nx + i
        }
        func cell(_ p: F3, lower: Bool) -> SIMD3<Int> {
            let q = (p - origin) / spacing
            let e = lower ? floor(q) - F3(repeating: 1) : floor(q) + F3(repeating: 1)
            return SIMD3(Int(e.x), Int(e.y), Int(e.z))
        }

        // Surface voxelization: mark cells whose centers are within the
        // narrow triangle band. This turns O(cells * triangles) inside and
        // distance queries into local triangle AABB rasterization.
        for (ia, ib, ic) in triangles {
            let a = vertices[ia], b = vertices[ib], c = vertices[ic]
            let lo = min(min(a, b), c) - F3(repeating: band)
            let hi = max(max(a, b), c) + F3(repeating: band)
            let rawMn = cell(lo, lower: true)
            let rawMx = cell(hi, lower: false)
            let mnCell = SIMD3(max(0, rawMn.x), max(0, rawMn.y), max(0, rawMn.z))
            let mxCell = SIMD3(min(nx - 1, rawMx.x), min(ny - 1, rawMx.y),
                               min(nz - 1, rawMx.z))
            guard mnCell.x <= mxCell.x, mnCell.y <= mxCell.y,
                  mnCell.z <= mxCell.z else { continue }
            for i in mnCell.x...mxCell.x {
                for j in mnCell.y...mxCell.y {
                    for k in mnCell.z...mxCell.z {
                        let p = origin + (F3(Float(i), Float(j), Float(k))
                                          + F3(repeating: 0.5)) * spacing
                        if pointTriangleDistanceSquared(p, a, b, c) <= band2 + 1e-10 {
                            surface[idx(i, j, k)] = 1
                        }
                    }
                }
            }
        }

        var queue: [SIMD3<Int>] = []
        queue.reserveCapacity(count / 4)
        func pushExterior(_ i: Int, _ j: Int, _ k: Int) {
            let id = idx(i, j, k)
            if surface[id] == 0 && exterior[id] == 0 {
                exterior[id] = 1
                queue.append(SIMD3(i, j, k))
            }
        }
        for i in 0..<nx {
            for j in 0..<ny {
                pushExterior(i, j, 0)
                pushExterior(i, j, nz - 1)
            }
        }
        for i in 0..<nx {
            for k in 0..<nz {
                pushExterior(i, 0, k)
                pushExterior(i, ny - 1, k)
            }
        }
        for j in 0..<ny {
            for k in 0..<nz {
                pushExterior(0, j, k)
                pushExterior(nx - 1, j, k)
            }
        }
        let dirs = [
            SIMD3(1, 0, 0), SIMD3(-1, 0, 0), SIMD3(0, 1, 0),
            SIMD3(0, -1, 0), SIMD3(0, 0, 1), SIMD3(0, 0, -1)
        ]
        var head = 0
        while head < queue.count {
            let c = queue[head]
            head += 1
            for d in dirs {
                let n = c &+ d
                guard n.x >= 0, n.x < nx, n.y >= 0, n.y < ny,
                      n.z >= 0, n.z < nz else { continue }
                pushExterior(n.x, n.y, n.z)
            }
        }

        var occupied = Set<SIMD3<Int>>()
        occupied.reserveCapacity(count - queue.count)
        for i in 0..<nx {
            for j in 0..<ny {
                for k in 0..<nz where exterior[idx(i, j, k)] == 0 {
                    occupied.insert(SIMD3(i, j, k))
                }
            }
        }
        if occupied.isEmpty {
            occupied.insert(SIMD3(nx / 2, ny / 2, nz / 2))
        }
        return (occupied, origin, spacing)
    }
}

private func orient(_ p: F3, upAxis: MeshUpAxis) -> F3 {
    switch upAxis {
    case .z: return p
    case .y: return F3(p.x, -p.z, p.y)
    case .x: return F3(-p.y, p.z, p.x)
    }
}

private func safeNormalize(_ v: F3, fallback: F3 = .zero) -> F3 {
    let l2 = length_squared(v)
    return l2 > 1e-20 ? v * rsqrt(l2) : fallback
}

private func rayIntersectsTriangle(origin o: F3, dir d: F3,
                                   a: F3, b: F3, c: F3) -> Bool {
    let eps: Float = 1e-7
    let e1 = b - a
    let e2 = c - a
    let h = cross(d, e2)
    let det = dot(e1, h)
    if abs(det) < eps { return false }
    let invDet = 1 / det
    let s = o - a
    let u = invDet * dot(s, h)
    if u < 0 || u > 1 { return false }
    let q = cross(s, e1)
    let v = invDet * dot(d, q)
    if v < 0 || u + v > 1 { return false }
    let t = invDet * dot(e2, q)
    return t > eps
}

private func pointTriangleDistanceSquared(
    _ p: F3, _ a: F3, _ b: F3, _ c: F3
) -> Float {
    let ab = b - a, ac = c - a, ap = p - a
    let d1 = dot(ab, ap), d2 = dot(ac, ap)
    if d1 <= 0 && d2 <= 0 { return length_squared(ap) }
    let bp = p - b
    let d3 = dot(ab, bp), d4 = dot(ac, bp)
    if d3 >= 0 && d4 <= d3 { return length_squared(bp) }
    let vc = d1 * d4 - d3 * d2
    if vc <= 0 && d1 >= 0 && d3 <= 0 {
        let v = d1 / (d1 - d3)
        return length_squared(p - (a + ab * v))
    }
    let cp = p - c
    let d5 = dot(ab, cp), d6 = dot(ac, cp)
    if d6 >= 0 && d5 <= d6 { return length_squared(cp) }
    let vb = d5 * d2 - d1 * d6
    if vb <= 0 && d2 >= 0 && d6 <= 0 {
        let w = d2 / (d2 - d6)
        return length_squared(p - (a + ac * w))
    }
    let va = d3 * d6 - d5 * d4
    if va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return length_squared(p - (b + (c - b) * w))
    }
    let n = safeNormalize(cross(ab, ac))
    let dist = dot(p - a, n)
    return dist * dist
}
