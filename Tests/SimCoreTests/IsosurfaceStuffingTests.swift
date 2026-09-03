import XCTest
import simd
@testable import SimCore

final class IsosurfaceStuffingTests: XCTestCase {
    /// Boundary must be a closed 2-manifold: every boundary edge is used by
    /// exactly two boundary faces, and every interior face by two tets.
    private func assertConforming(_ mesh: IsosurfaceStuffing.TetMesh,
                                  _ label: String, file: StaticString = #filePath,
                                  line: UInt = #line) {
        var edgeUse: [SIMD2<Int32>: Int] = [:]
        for f in mesh.boundaryFaces {
            for (a, b) in [(f.x, f.y), (f.y, f.z), (f.z, f.x)] {
                edgeUse[SIMD2(min(a, b), max(a, b)), default: 0] += 1
            }
        }
        let open = edgeUse.values.filter { $0 != 2 }.count
        XCTAssertEqual(open, 0, "\(label): \(open) boundary edges not shared by two faces",
                       file: file, line: line)
        var inverted = 0
        for t in mesh.tets {
            let a = mesh.nodes[Int(t.x)], b = mesh.nodes[Int(t.y)]
            let c = mesh.nodes[Int(t.z)], d = mesh.nodes[Int(t.w)]
            if simd_dot(simd_cross(b - a, c - a), d - a) <= 0 { inverted += 1 }
        }
        XCTAssertEqual(inverted, 0, "\(label): inverted tets", file: file, line: line)
        var faceUse: [SIMD3<Int32>: Int] = [:]
        for t in mesh.tets {
            let ids = [t.x, t.y, t.z, t.w]
            for drop in 0..<4 {
                let f = ids.enumerated().filter { $0.offset != drop }.map(\.element).sorted()
                faceUse[SIMD3(f[0], f[1], f[2]), default: 0] += 1
            }
        }
        let overshared = faceUse.values.filter { $0 > 2 }.count
        XCTAssertEqual(overshared, 0, "\(label): faces used by more than two tets",
                       file: file, line: line)
        XCTAssertEqual(faceUse.values.filter { $0 == 1 }.count, mesh.boundaryFaces.count,
                       "\(label): boundary face count", file: file, line: line)
    }

    func testSphereIsFilledWithGoodTets() {
        let r: Float = 0.05
        for spacing: Float in [0.02, 0.01] {
            var options = IsosurfaceStuffing.Options(spacing: spacing)
            options.fieldOffset = 0
            let mesh = IsosurfaceStuffing.mesh(
                field: { simd_length($0) - r },
                bounds: (F3(repeating: -r), F3(repeating: r)), options: options)
            let exact = 4 / 3 * Float.pi * r * r * r
            let error = abs(mesh.volume - exact) / exact
            let angles = mesh.dihedralRange()
            print("sphere h=\(spacing): nodes \(mesh.nodes.count) tets \(mesh.tets.count) "
                + "volume error \(error) dihedral \(angles)")
            XCTAssertLessThan(error, spacing == 0.02 ? 0.08 : 0.03, "volume at h=\(spacing)")
            XCTAssertGreaterThan(angles.min, 8, "min dihedral at h=\(spacing)")
            XCTAssertLessThan(angles.max, 168, "max dihedral at h=\(spacing)")
            assertConforming(mesh, "sphere h=\(spacing)")
            // Every node must lie inside or on the (slightly grown) sphere.
            let outside = mesh.nodes.filter { simd_length($0) > r * 1.001 }.count
            XCTAssertEqual(outside, 0, "nodes outside the sphere at h=\(spacing)")
        }
    }

    func testFieldOffsetGrowsTheSolid() {
        let r: Float = 0.05
        var options = IsosurfaceStuffing.Options(spacing: 0.01)
        options.fieldOffset = -0.01
        let mesh = IsosurfaceStuffing.mesh(
            field: { simd_length($0) - r },
            bounds: (F3(repeating: -r), F3(repeating: r)), options: options)
        let grown = 4 / 3 * Float.pi * pow(r + 0.01, 3)
        XCTAssertLessThan(abs(mesh.volume - grown) / grown, 0.03)
        assertConforming(mesh, "grown sphere")
    }

    func testTriangleMeshDistanceFieldSignsAndDistances() {
        // Icosphere-like closed mesh: a cube surface with 12 triangles.
        let s: Float = 0.5
        let v = [F3(-s, -s, -s), F3(s, -s, -s), F3(s, s, -s), F3(-s, s, -s),
                 F3(-s, -s, s), F3(s, -s, s), F3(s, s, s), F3(-s, s, s)]
        let tris = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
                    (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
        let mesh = SurfaceMesh(vertices: v, normals: v.map { simd_normalize($0) }, triangles: tris)
        let field = TriangleMeshDistanceField(mesh: mesh)
        XCTAssertEqual(field.signedDistance(F3(0, 0, 0)), -0.5, accuracy: 1e-5)
        XCTAssertEqual(field.signedDistance(F3(0, 0, 1)), 0.5, accuracy: 1e-5)
        XCTAssertEqual(field.signedDistance(F3(0.4, 0.4, 0.4)), -0.1, accuracy: 1e-5)
        XCTAssertEqual(field.signedDistance(F3(1, 1, 1)), sqrt(0.75), accuracy: 1e-5)
        XCTAssertEqual(field.signedDistance(F3(0.6, 0.6, 0)), sqrt(0.02), accuracy: 1e-5)
        var options = IsosurfaceStuffing.Options(spacing: 0.1)
        let tet = IsosurfaceStuffing.mesh(
            field: field.signedDistance, bounds: mesh.bounds(), options: options)
        XCTAssertLessThan(abs(tet.volume - 1) / 1, 0.02, "cube volume")
        assertConforming(tet, "cube")
        options.spacing = 0.25
        let coarse = IsosurfaceStuffing.mesh(
            field: field.signedDistance, bounds: mesh.bounds(), options: options)
        assertConforming(coarse, "coarse cube")
    }
}

extension IsosurfaceStuffingTests {
    /// The plush armadillo at toy scale: a coarse cage must still close and
    /// keep its quality when the lattice is coarser than the ears and claws.
    func testArmadilloPlushCage() throws {
        // Meshing a 100k-triangle asset takes over ten minutes unoptimised;
        // run this one with `swift test -c release -Xswiftc -enable-testing`.
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["AVBD_MESHING_ASSET_TESTS"] != nil,
            "set AVBD_MESHING_ASSET_TESTS=1 (release build) to mesh the armadillo")
        let path = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/GPUSimDemos/Assets/classic/stanford-armadillo.obj").path
        let raw = try SurfaceMesh.load(path: path, upAxis: .z, normalPolicy: .preserveImported)
        let mesh = raw.withSmoothNormals()
        let (lo, hi) = mesh.bounds()
        let scale: Float = 0.10 / (hi - lo).max()
        let scaled = SurfaceMesh(vertices: mesh.vertices.map { ($0 - (lo + hi) * 0.5) * scale },
                                 normals: mesh.normals, triangles: mesh.triangles)
        let t0 = Date()
        let field = TriangleMeshDistanceField(mesh: scaled)
        let t1 = Date()
        for (spacing, offset) in [(Float(0.010), Float(-0.003)), (0.008, -0.003), (0.006, -0.002)] {
            var options = IsosurfaceStuffing.Options(spacing: spacing)
            options.fieldOffset = offset
            let start = Date()
            let cage = IsosurfaceStuffing.mesh(field: field.signedDistance,
                                               bounds: scaled.bounds(), options: options)
            let angles = cage.dihedralRange()
            print(String(format: "armadillo h=%.3f offset=%.3f: nodes %d tets %d boundary %d dihedral %.1f..%.1f volume %.1f cm3 mesh %.2fs (field %.2fs)",
                         spacing, offset, cage.nodes.count, cage.tets.count, cage.boundaryFaces.count,
                         angles.min, angles.max, cage.volume * 1e6, Date().timeIntervalSince(start),
                         t1.timeIntervalSince(t0)))
            assertConforming(cage, "armadillo h=\(spacing)")
            XCTAssertGreaterThan(angles.min, 8)
            XCTAssertLessThan(angles.max, 168)
        }
    }
}
