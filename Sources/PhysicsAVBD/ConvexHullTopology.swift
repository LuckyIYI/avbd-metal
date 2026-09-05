import simd
import SimCore

/// Deterministic convex-hull boundary reconstruction for the legacy inline
/// point-cloud API. Cooked assets already carry canonical triangles; this
/// utility prevents older vertex-only colliders from losing broad-face
/// manifolds and collision-debug rendering.
enum ConvexHullTopologyBuilder {
    struct Failure: Error, Equatable, CustomStringConvertible {
        var reason: String
        var description: String { reason }
    }

    private struct IndexedPoint {
        var value: SIMD3<Double>
        var sourceIndex: UInt32
    }

    private struct Face {
        var a: Int
        var b: Int
        var c: Int
    }

    private struct EdgeKey: Hashable, Comparable {
        var a: Int
        var b: Int

        init(_ u: Int, _ v: Int) {
            a = min(u, v)
            b = max(u, v)
        }

        static func < (lhs: EdgeKey, rhs: EdgeKey) -> Bool {
            lhs.a == rhs.a ? lhs.b < rhs.b : lhs.a < rhs.a
        }
    }

    private static func dot(
        _ a: SIMD3<Double>, _ b: SIMD3<Double>
    ) -> Double {
        a.x * b.x + a.y * b.y + a.z * b.z
    }

    private static func cross(
        _ a: SIMD3<Double>, _ b: SIMD3<Double>
    ) -> SIMD3<Double> {
        SIMD3(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x)
    }

    private static func lengthSquared(_ value: SIMD3<Double>) -> Double {
        dot(value, value)
    }

    private static func normal(
        _ face: Face, points: [IndexedPoint]
    ) -> SIMD3<Double> {
        let a = points[face.a].value
        return cross(points[face.b].value - a, points[face.c].value - a)
    }

    private static func orientedFace(
        _ a: Int, _ b: Int, _ c: Int, points: [IndexedPoint],
        interior: SIMD3<Double>
    ) -> Face {
        var result = Face(a: a, b: b, c: c)
        if dot(normal(result, points: points), interior - points[a].value) > 0 {
            result = Face(a: a, b: c, c: b)
        }
        return result
    }

    private static func lexicographicallyPrecedes(
        _ lhs: IndexedPoint, _ rhs: IndexedPoint
    ) -> Bool {
        if lhs.value.x != rhs.value.x { return lhs.value.x < rhs.value.x }
        if lhs.value.y != rhs.value.y { return lhs.value.y < rhs.value.y }
        if lhs.value.z != rhs.value.z { return lhs.value.z < rhs.value.z }
        return lhs.sourceIndex < rhs.sourceIndex
    }

    /// Build a closed triangular boundary with the same bounded incremental
    /// hull algorithm used by the offline cooker. Geometry is evaluated in
    /// Double after accepting finite Float inputs; triangles retain indices
    /// into the original support array so collision support remains exactly
    /// the legacy point cloud, including harmless duplicate/interior samples.
    static func triangulate(vertices: [F3]) throws -> [SIMD3<UInt32>] {
        guard vertices.count >= 4 else {
            throw Failure(reason: "requires at least four vertices")
        }
        guard vertices.count <= ConvexAssetLimits.maximumVerticesPerHull else {
            throw Failure(reason: "has \(vertices.count) vertices; maximum is \(ConvexAssetLimits.maximumVerticesPerHull)")
        }
        guard vertices.allSatisfy({
            $0.x.isFinite && $0.y.isFinite && $0.z.isFinite
        }) else {
            throw Failure(reason: "contains a non-finite vertex")
        }

        var points = vertices.enumerated().map { index, point in
            IndexedPoint(
                value: SIMD3(Double(point.x), Double(point.y), Double(point.z)),
                sourceIndex: UInt32(index))
        }
        points.sort(by: lexicographicallyPrecedes)

        // Exact geometric duplicates do not belong in boundary topology.
        // Retain the lowest source index deterministically; all original
        // samples remain in the support buffer, so this does not change the
        // represented convex set.
        var unique: [IndexedPoint] = []
        unique.reserveCapacity(points.count)
        for point in points {
            if let last = unique.last,
               point.value.x == last.value.x,
               point.value.y == last.value.y,
               point.value.z == last.value.z {
                continue
            }
            unique.append(point)
        }
        points = unique
        guard points.count >= 4 else {
            throw Failure(reason: "requires at least four unique vertices")
        }

        let lower = points.reduce(
            SIMD3<Double>(repeating: .infinity)
        ) { SIMD3(min($0.x, $1.value.x), min($0.y, $1.value.y),
                  min($0.z, $1.value.z)) }
        let upper = points.reduce(
            SIMD3<Double>(repeating: -.infinity)
        ) { SIMD3(max($0.x, $1.value.x), max($0.y, $1.value.y),
                  max($0.z, $1.value.z)) }
        let diagonal = sqrt(lengthSquared(upper - lower))
        guard diagonal.isFinite && diagonal > 1.0e-7 else {
            throw Failure(reason: "does not span a finite three-dimensional hull")
        }
        let tolerance = max(diagonal * 1.0e-7, 1.0e-9)
        // Construction operates in Double. A point within the Float-sized
        // validation envelope of an early face can lie outside a later face
        // after the horizon changes. Skipping it with that same envelope can
        // therefore produce a hull that fails its own containment check.
        // Retain those near-coplanar extremes during construction, while
        // leaving the public geometry-validation tolerance unchanged.
        let visibilityTolerance = max(diagonal * 1.0e-12, 1.0e-15)

        // Deterministic maximal selections use lexicographic order as the
        // tie-breaker (the points array is already sorted ascending).
        let a = 0
        var b = 1
        var best = lengthSquared(points[b].value - points[a].value)
        for index in 2..<points.count {
            let score = lengthSquared(points[index].value - points[a].value)
            if score > best || score == best && index > b {
                b = index
                best = score
            }
        }
        let ab = points[b].value - points[a].value
        let abLengthSquared = lengthSquared(ab)
        guard abLengthSquared > tolerance * tolerance else {
            throw Failure(reason: "does not span a finite line segment")
        }

        var c: Int?
        var cScore = -Double.infinity
        for index in points.indices where index != a && index != b {
            let score = lengthSquared(cross(points[index].value
                - points[a].value, ab)) / abLengthSquared
            if score > cScore || score == cScore && index > (c ?? -1) {
                c = index
                cScore = score
            }
        }
        guard let c else {
            throw Failure(reason: "does not contain a third unique vertex")
        }
        let seedNormal = cross(points[b].value - points[a].value,
                               points[c].value - points[a].value)
        let seedNormalLength = sqrt(lengthSquared(seedNormal))
        guard seedNormalLength > tolerance * tolerance else {
            throw Failure(reason: "is collinear")
        }

        var d: Int?
        var dScore = -Double.infinity
        for index in points.indices
            where index != a && index != b && index != c {
            let score = abs(dot(seedNormal,
                points[index].value - points[a].value)) / seedNormalLength
            if score > dScore || score == dScore && index > (d ?? -1) {
                d = index
                dScore = score
            }
        }
        guard let d, dScore > tolerance else {
            throw Failure(reason: "is coplanar")
        }

        let interior = (points[a].value + points[b].value
            + points[c].value + points[d].value) * 0.25
        var faces = [
            orientedFace(a, b, c, points: points, interior: interior),
            orientedFace(a, d, b, points: points, interior: interior),
            orientedFace(a, c, d, points: points, interior: interior),
            orientedFace(b, d, c, points: points, interior: interior),
        ]
        let seed = Set([a, b, c, d])

        for pointIndex in points.indices where !seed.contains(pointIndex) {
            var visible: [Int] = []
            visible.reserveCapacity(faces.count)
            for (faceIndex, face) in faces.enumerated() {
                let faceNormal = normal(face, points: points)
                let normalLength = sqrt(lengthSquared(faceNormal))
                guard normalLength.isFinite && normalLength > 0 else {
                    throw Failure(reason: "generated a degenerate face")
                }
                let distance = dot(faceNormal,
                    points[pointIndex].value - points[face.a].value)
                if distance > visibilityTolerance * normalLength {
                    visible.append(faceIndex)
                }
            }
            if visible.isEmpty { continue }

            var counts: [EdgeKey: Int] = [:]
            var directions: [EdgeKey: (Int, Int)] = [:]
            for faceIndex in visible {
                let face = faces[faceIndex]
                for edge in [(face.a, face.b), (face.b, face.c),
                             (face.c, face.a)] {
                    let key = EdgeKey(edge.0, edge.1)
                    counts[key, default: 0] += 1
                    directions[key] = edge
                }
            }
            var horizon: [(Int, Int)] = []
            horizon.reserveCapacity(counts.count)
            for (key, count) in counts where count == 1 {
                if let direction = directions[key] {
                    horizon.append(direction)
                }
            }
            horizon.sort {
                $0.0 == $1.0 ? $0.1 < $1.1 : $0.0 < $1.0
            }
            guard horizon.count >= 3 else {
                throw Failure(reason: "generated hull horizon has fewer than three edges")
            }

            // A convex visible region has one simple closed horizon. Check it
            // explicitly rather than letting a numerical split produce an
            // open/non-manifold triangle soup downstream.
            var horizonAdjacency: [Int: [Int]] = [:]
            for edge in horizon {
                horizonAdjacency[edge.0, default: []].append(edge.1)
                horizonAdjacency[edge.1, default: []].append(edge.0)
            }
            guard horizonAdjacency.count == horizon.count,
                  horizonAdjacency.values.allSatisfy({ $0.count == 2 }) else {
                throw Failure(reason: "generated hull horizon is not one closed loop")
            }
            let start = horizonAdjacency.keys.min()!
            var visited: Set<Int> = [start]
            var previous = -1
            var current = start
            repeat {
                let neighbors = horizonAdjacency[current]!.sorted()
                guard let next = neighbors.first(where: { $0 != previous }) else {
                    throw Failure(reason: "generated hull horizon is disconnected")
                }
                previous = current
                current = next
                if current != start { visited.insert(current) }
            } while current != start && visited.count <= horizon.count
            guard current == start, visited.count == horizon.count else {
                throw Failure(reason: "generated hull horizon is disconnected")
            }

            let replacements = horizon.map { edge in
                orientedFace(edge.1, edge.0, pointIndex,
                             points: points, interior: interior)
            }
            // Adjacent Float extrema can cut a nanometre-sized sliver off
            // an otherwise valid face. The GPU clipping topology rejects
            // cross lengths <= 1e-12, including after Float centering. Keep
            // the existing boundary only when it already contains this
            // sample within the public validation envelope. Final full-
            // cloud containment and manifold checks still apply below.
            let createsSliver = replacements.contains { face in
                let a = SIMD3<Float>(points[face.a].value)
                let b = SIMD3<Float>(points[face.b].value)
                let c = SIMD3<Float>(points[face.c].value)
                return simd_length(simd_cross(b - a, c - a)) <= 1e-12
            }
            if createsSliver && faces.allSatisfy({ face in
                let n = normal(face, points: points)
                return dot(n, points[pointIndex].value - points[face.a].value)
                    <= tolerance * sqrt(lengthSquared(n))
            }) { continue }

            let visibleSet = Set(visible)
            faces = faces.enumerated().compactMap {
                visibleSet.contains($0.offset) ? nil : $0.element
            }
            faces.append(contentsOf: replacements)
        }

        guard faces.count >= 4,
              faces.count <= ConvexAssetLimits.maximumTrianglesPerHull else {
            throw Failure(reason: "generated hull has \(faces.count) triangles; runtime supports 4...\(ConvexAssetLimits.maximumTrianglesPerHull)")
        }

        // Keep the legacy reconstruction fail-closed just like cooked asset
        // validation: every triangle is finite and non-degenerate, every
        // undirected edge has exactly two incident faces, all source support
        // points lie within the tolerance envelope, and winding encloses a
        // positive three-dimensional volume.
        var edgeUses: [EdgeKey: Int] = [:]
        var volumeTimesSix = 0.0
        let reference = (lower + upper) * 0.5
        for (faceIndex, face) in faces.enumerated() {
            let faceNormal = normal(face, points: points)
            let normalLength = sqrt(lengthSquared(faceNormal))
            guard normalLength.isFinite,
                  normalLength > tolerance * tolerance else {
                throw Failure(reason: "generated face \(faceIndex) is degenerate")
            }
            for point in points {
                let distance = dot(faceNormal,
                    point.value - points[face.a].value)
                guard distance <= tolerance * normalLength else {
                    throw Failure(reason: "generated face \(faceIndex) does not contain every support point")
                }
            }
            for edge in [(face.a, face.b), (face.b, face.c),
                         (face.c, face.a)] {
                edgeUses[EdgeKey(edge.0, edge.1), default: 0] += 1
            }
            let pa = points[face.a].value - reference
            let pb = points[face.b].value - reference
            let pc = points[face.c].value - reference
            volumeTimesSix += dot(pa, cross(pb, pc))
        }
        guard edgeUses.values.allSatisfy({ $0 == 2 }) else {
            let edge = edgeUses.keys.sorted().first {
                edgeUses[$0] != 2
            }!
            throw Failure(reason: "generated hull edge \(edge.a)-\(edge.b) is not manifold")
        }
        guard volumeTimesSix.isFinite,
              volumeTimesSix > max(diagonal * diagonal * diagonal * 1.0e-12,
                                    1.0e-18) else {
            throw Failure(reason: "generated boundary has zero or inconsistent volume")
        }

        return faces.map { face in
            SIMD3(points[face.a].sourceIndex,
                  points[face.b].sourceIndex,
                  points[face.c].sourceIndex)
        }
    }
}
