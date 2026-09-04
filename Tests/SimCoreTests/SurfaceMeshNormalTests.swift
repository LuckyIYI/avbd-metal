import Foundation
import simd
import XCTest
@testable import SimCore

final class SurfaceMeshNormalTests: XCTestCase {
    func testAutomaticSTLNormalsSmoothCurvesButPreserveCreases() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let url = root.appendingPathComponent("two-hinges.stl")
        let cosine30 = sqrt(Float(3)) * 0.5
        let sine30: Float = 0.5
        let stl = """
        solid two_hinges
          facet normal 0 0 1
            outer loop
              vertex 0 0 0
              vertex 1 0 0
              vertex 0 1 0
            endloop
          endfacet
          facet normal 0 \(-sine30) \(cosine30)
            outer loop
              vertex 1 0 0
              vertex 0 0 0
              vertex 0 \(-cosine30) \(-sine30)
            endloop
          endfacet
          facet normal 0 0 1
            outer loop
              vertex 3 0 0
              vertex 4 0 0
              vertex 3 1 0
            endloop
          endfacet
          facet normal 0 -1 0
            outer loop
              vertex 4 0 0
              vertex 3 0 0
              vertex 3 0 -1
            endloop
          endfacet
        endsolid two_hinges
        """
        try stl.write(to: url, atomically: true, encoding: .utf8)

        let preserved = try SurfaceMesh.load(
            path: url.path, upAxis: .z,
            normalPolicy: .preserveImported)
        let flatAtSmoothEdge = normals(in: preserved, at: F3(0, 0, 0))
        XCTAssertEqual(flatAtSmoothEdge.count, 2)
        XCTAssertLessThan(dot(flatAtSmoothEdge[0], flatAtSmoothEdge[1]), 0.9,
            "facet-only STL normals expose the triangulation")

        let automatic = try SurfaceMesh.load(path: url.path, upAxis: .z)
        let smooth = normals(in: automatic, at: F3(0, 0, 0))
        let expected = normalize(F3(0, -sine30, 1 + cosine30))
        XCTAssertFalse(smooth.isEmpty)
        XCTAssertTrue(smooth.allSatisfy { dot($0, expected) > 0.9999 },
            "every corner on the 30-degree edge needs one smooth normal")

        let sharp = normals(in: automatic, at: F3(3, 0, 0))
        XCTAssertEqual(sharp.count, 2,
            "the 90-degree mechanical edge must remain split")
        XCTAssertLessThan(abs(dot(sharp[0], sharp[1])), 1e-5)
        XCTAssertTrue(automatic.normals.allSatisfy {
            $0.x.isFinite && $0.y.isFinite && $0.z.isFinite
                && abs(length($0) - 1) < 1e-5
        })
    }

    func testAutomaticImportPreservesAuthoredOBJNormals() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let url = root.appendingPathComponent("authored.obj")
        let obj = """
        v 0 0 0
        v 1 0 0
        v 0 1 0
        vn 0 1 0
        f 1//1 2//1 3//1
        """
        try obj.write(to: url, atomically: true, encoding: .utf8)

        let mesh = try SurfaceMesh.load(path: url.path, upAxis: .z)
        XCTAssertEqual(mesh.normals.count, mesh.vertices.count)
        XCTAssertTrue(mesh.normals.allSatisfy {
            dot($0, F3(0, 1, 0)) > 0.9999
        }, "authored non-STL normals must reach the renderer unchanged")
    }

    func testCreaseNormalsAverageUnitFacesRatherThanTriangleArea() {
        let cosine30 = sqrt(Float(3)) * 0.5
        let sine30: Float = 0.5
        let a = F3(0, 0, 0)
        let b = F3(1, 0, 0)
        let large = F3(0, 100, 0)
        let tiny = F3(0, -0.01 * cosine30, -0.01 * sine30)
        let mesh = SurfaceMesh(
            vertices: [a, b, large, b, a, tiny],
            normals: [F3](repeating: .zero, count: 6),
            triangles: [(0, 1, 2), (3, 4, 5)])

        let rebuilt = mesh.withCreaseNormals(
            maxSmoothAngleRadians: .pi / 5)
        let atSharedCorner = normals(in: rebuilt, at: a)
        let expected = normalize(F3(0, -sine30, 1 + cosine30))

        XCTAssertEqual(atSharedCorner.count, 2)
        XCTAssertTrue(atSharedCorner.allSatisfy {
            dot($0, expected) > 0.9999
        }, "normal smoothing must not depend on adjacent triangle area")
    }

    private func normals(in mesh: SurfaceMesh, at point: F3) -> [F3] {
        zip(mesh.vertices, mesh.normals).compactMap { vertex, normal in
            length_squared(vertex - point) < 1e-12 ? normal : nil
        }
    }
}
