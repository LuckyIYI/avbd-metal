import Foundation
import simd

extension SurfaceMesh {
    /// Rebuilds a triangle-corner normal field after welding repeated STL
    /// positions. Adjacent unit face normals are averaged only when they fall
    /// inside the requested crease angle.
    ///
    /// Equal face weighting prevents tessellation density from changing the
    /// apparent curvature: a large, irregular CAD triangle must not overpower
    /// several smaller neighbours that describe the same smooth surface.
    func withCreaseNormals(
        maxSmoothAngleRadians: Float,
        weldTolerance: Float = 1e-4
    ) -> SurfaceMesh {
        guard !vertices.isEmpty, !triangles.isEmpty else { return self }

        let tolerance = max(weldTolerance, Float.ulpOfOne)
        var positionToIndex: [SurfaceWeldKey: Int] = [:]
        var weldedPositions: [F3] = []
        var remap = [Int](repeating: 0, count: vertices.count)
        weldedPositions.reserveCapacity(vertices.count)

        for (index, position) in vertices.enumerated() {
            let key = SurfaceWeldKey(position, tolerance: tolerance)
            if let existing = positionToIndex[key] {
                remap[index] = existing
            } else {
                let next = weldedPositions.count
                positionToIndex[key] = next
                weldedPositions.append(position)
                remap[index] = next
            }
        }

        var weldedTriangles: [(Int, Int, Int)] = []
        var faceNormals: [F3] = []
        var adjacentFaceNormals = [[F3]](repeating: [], count: weldedPositions.count)
        weldedTriangles.reserveCapacity(triangles.count)
        faceNormals.reserveCapacity(triangles.count)
        for triangle in triangles {
            let welded = (remap[triangle.0], remap[triangle.1], remap[triangle.2])
            weldedTriangles.append(welded)
            let a = weldedPositions[welded.0]
            let b = weldedPositions[welded.1]
            let c = weldedPositions[welded.2]
            let normal = normalizedOrZero(cross(b - a, c - a))
            faceNormals.append(normal)
            adjacentFaceNormals[welded.0].append(normal)
            adjacentFaceNormals[welded.1].append(normal)
            adjacentFaceNormals[welded.2].append(normal)
        }

        let creaseDot = cos(max(0, min(maxSmoothAngleRadians, .pi)))
        var outputVertices: [F3] = []
        var outputNormals: [F3] = []
        var outputTriangles: [(Int, Int, Int)] = []
        outputVertices.reserveCapacity(weldedTriangles.count * 3)
        outputNormals.reserveCapacity(weldedTriangles.count * 3)
        outputTriangles.reserveCapacity(weldedTriangles.count)

        for (triangleIndex, triangle) in weldedTriangles.enumerated() {
            let faceNormal = faceNormals[triangleIndex]
            let base = outputVertices.count
            for vertexIndex in [triangle.0, triangle.1, triangle.2] {
                var sum = F3.zero
                for otherNormal in adjacentFaceNormals[vertexIndex]
                    where dot(faceNormal, otherNormal) > creaseDot {
                    sum += otherNormal
                }
                outputVertices.append(weldedPositions[vertexIndex])
                outputNormals.append(normalizedOrZero(sum, fallback: faceNormal))
            }
            outputTriangles.append((base, base + 1, base + 2))
        }

        return SurfaceMesh(
            vertices: outputVertices,
            normals: outputNormals,
            triangles: outputTriangles).withConsistentNormals()
    }
}

private struct SurfaceWeldKey: Hashable {
    var x: Int64
    var y: Int64
    var z: Int64

    init(_ value: F3, tolerance: Float) {
        // Match the reference mergeVertices rule: scale by 1/tolerance, add
        // half a cell, then truncate toward zero.
        let multiplier = 1 / Double(tolerance)
        let additive = 0.5
        x = Int64((Double(value.x) * multiplier + additive).rounded(.towardZero))
        y = Int64((Double(value.y) * multiplier + additive).rounded(.towardZero))
        z = Int64((Double(value.z) * multiplier + additive).rounded(.towardZero))
    }
}

private func normalizedOrZero(_ value: F3, fallback: F3 = .zero) -> F3 {
    let magnitudeSquared = length_squared(value)
    return magnitudeSquared > 1e-20 ? value * rsqrt(magnitudeSquared) : fallback
}
