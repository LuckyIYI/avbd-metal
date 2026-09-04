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

    func testSphereIsFilledWithGoodTets() throws {
        let r: Float = 0.05
        for spacing: Float in [0.02, 0.01] {
            var options = IsosurfaceStuffing.Options(spacing: spacing)
            options.fieldOffset = 0
            let mesh = try IsosurfaceStuffing.mesh(
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

    func testFieldOffsetGrowsTheSolid() throws {
        let r: Float = 0.05
        var options = IsosurfaceStuffing.Options(spacing: 0.01)
        options.fieldOffset = -0.01
        let mesh = try IsosurfaceStuffing.mesh(
            field: { simd_length($0) - r },
            bounds: (F3(repeating: -r), F3(repeating: r)), options: options)
        let grown = 4 / 3 * Float.pi * pow(r + 0.01, 3)
        XCTAssertLessThan(abs(mesh.volume - grown) / grown, 0.03)
        assertConforming(mesh, "grown sphere")
    }

    func testTriangleMeshDistanceFieldSignsAndDistances() throws {
        // Icosphere-like closed mesh: a cube surface with 12 triangles.
        let s: Float = 0.5
        let v = [F3(-s, -s, -s), F3(s, -s, -s), F3(s, s, -s), F3(-s, s, -s),
                 F3(-s, -s, s), F3(s, -s, s), F3(s, s, s), F3(-s, s, s)]
        let tris = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
                    (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
        let mesh = SurfaceMesh(vertices: v, normals: v.map { simd_normalize($0) }, triangles: tris)
        let field = try TriangleMeshDistanceField(mesh: mesh)
        XCTAssertEqual(field.signedDistance(F3(0, 0, 0)), -0.5, accuracy: 1e-5)
        XCTAssertEqual(field.signedDistance(F3(0, 0, 1)), 0.5, accuracy: 1e-5)
        XCTAssertEqual(field.signedDistance(F3(0.4, 0.4, 0.4)), -0.1, accuracy: 1e-5)
        XCTAssertEqual(field.signedDistance(F3(1, 1, 1)), sqrt(0.75), accuracy: 1e-5)
        XCTAssertEqual(field.signedDistance(F3(0.6, 0.6, 0)), sqrt(0.02), accuracy: 1e-5)
        var options = IsosurfaceStuffing.Options(spacing: 0.1)
        let tet = try IsosurfaceStuffing.mesh(
            field: { field.signedDistance($0) }, bounds: mesh.bounds(), options: options)
        XCTAssertLessThan(abs(tet.volume - 1) / 1, 0.02, "cube volume")
        assertConforming(tet, "cube")
        options.spacing = 0.25
        let coarse = try IsosurfaceStuffing.mesh(
            field: { field.signedDistance($0) }, bounds: mesh.bounds(), options: options)
        assertConforming(coarse, "coarse cube")
    }
}

extension IsosurfaceStuffingTests {
    private func armadillo() throws -> SurfaceMesh {
        let path = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/GPUSimDemos/Assets/classic/stanford-armadillo.obj").path
        let raw = try SurfaceMesh.load(path: path, upAxis: .z, normalPolicy: .preserveImported)
        let mesh = raw.withSmoothNormals()
        let (lo, hi) = mesh.bounds()
        let scale: Float = 0.10 / (hi - lo).max()
        return SurfaceMesh(vertices: mesh.vertices.map { ($0 - (lo + hi) * 0.5) * scale },
                           normals: mesh.normals, triangles: mesh.triangles)
    }

    /// The real asset through the production proxy path: a clustered
    /// 3,000-vertex decimation (no longer a closed manifold, so the field
    /// uses winding numbers) stuffed at plush scale must close, stay well
    /// shaped and keep the asset's volume.
    func testArmadilloProxyCage() throws {
        let scaled = try armadillo()
        let proxy = scaled.simplified(maxVertices: 3000)
        let field = try TriangleMeshDistanceField(mesh: proxy)
        XCTAssertGreaterThan(field.topology.boundaryEdges + field.topology.nonManifoldEdges
                             + field.topology.inconsistentlyWoundEdges, 0,
                             "the clustered proxy is expected to be imperfect")
        XCTAssertEqual(field.signMethod, .windingNumber)
        XCTAssertFalse(field.topology.orientationFlipped)
        var options = IsosurfaceStuffing.Options(spacing: 0.010)
        options.fieldOffset = -0.003
        let cage = try IsosurfaceStuffing.mesh(
            field: { field.signedDistance($0) }, bounds: proxy.bounds(), options: options)
        let angles = cage.dihedralRange()
        print("armadillo proxy h=0.010: nodes \(cage.nodes.count) tets \(cage.tets.count) "
            + "dihedral \(angles) volume \(cage.volume * 1e6) cm3")
        assertConforming(cage, "armadillo proxy")
        XCTAssertGreaterThan(angles.min, 8)
        XCTAssertLessThan(angles.max, 168)
        XCTAssertGreaterThan(cage.nodes.count, 300)
        XCTAssertGreaterThan(cage.volume * 1e6, 90, "cm3; the asset is about 100 cm3 at this size")
        XCTAssertLessThan(cage.volume * 1e6, 140)
    }

    /// The full-resolution asset (100k triangles) at three spacings; too
    /// slow unoptimised, run with `swift test -c release -Xswiftc -enable-testing`.
    func testArmadilloPlushCage() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["AVBD_MESHING_ASSET_TESTS"] != nil,
            "set AVBD_MESHING_ASSET_TESTS=1 (release build) to mesh the full-resolution armadillo")
        let scaled = try armadillo()
        let t0 = Date()
        let field = try TriangleMeshDistanceField(mesh: scaled)
        let t1 = Date()
        for (spacing, offset) in [(Float(0.010), Float(-0.003)), (0.008, -0.003), (0.006, -0.002)] {
            var options = IsosurfaceStuffing.Options(spacing: spacing)
            options.fieldOffset = offset
            let start = Date()
            let cage = try IsosurfaceStuffing.mesh(field: { field.signedDistance($0) },
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

extension IsosurfaceStuffingTests {
    private func unitCube(reversed: Bool = false, dropFace: Bool = false) -> SurfaceMesh {
        let s: Float = 0.5
        let v = [F3(-s, -s, -s), F3(s, -s, -s), F3(s, s, -s), F3(-s, s, -s),
                 F3(-s, -s, s), F3(s, -s, s), F3(s, s, s), F3(-s, s, s)]
        var tris = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
                    (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
        if dropFace { tris.removeFirst(2) }           // open bottom
        if reversed { tris = tris.map { ($0.0, $0.2, $0.1) } }
        return SurfaceMesh(vertices: v, normals: v.map { simd_normalize($0) }, triangles: tris)
    }

    /// A negative offset grows the solid past the plain padding; the lattice
    /// must grow with it or the cage is clipped flat by its own boundary.
    func testNegativeOffsetIsNotClippedByTheLattice() throws {
        let r: Float = 0.03
        var options = IsosurfaceStuffing.Options(spacing: 0.01)
        options.fieldOffset = -0.05          // grown radius 0.08, far past 1.5 cells
        let mesh = try IsosurfaceStuffing.mesh(
            field: { simd_length($0) - r },
            bounds: (F3(repeating: -r), F3(repeating: r)), options: options)
        let grown = 4 / 3 * Float.pi * pow(r + 0.05, 3)
        XCTAssertLessThan(abs(mesh.volume - grown) / grown, 0.03)
        assertConforming(mesh, "grown sphere")
        let reach = mesh.nodes.map { simd_length($0) }.max() ?? 0
        XCTAssertGreaterThan(reach, 0.075, "the cage must extend to the grown radius")
    }

    /// A globally reversed but otherwise valid mesh must produce the same
    /// field and the same cage as its outward-facing twin.
    func testReversedMeshIsRepairedNotInverted() throws {
        let outward = try TriangleMeshDistanceField(mesh: unitCube())
        let reversed = try TriangleMeshDistanceField(mesh: unitCube(reversed: true))
        XCTAssertFalse(outward.topology.orientationFlipped)
        XCTAssertTrue(reversed.topology.orientationFlipped)
        XCTAssertEqual(outward.signMethod, .pseudonormal)
        XCTAssertEqual(reversed.signMethod, .pseudonormal)
        for p in [F3(0, 0, 0), F3(0.4, 0.4, 0.4), F3(0, 0, 1), F3(1, 1, 1), F3(0.6, 0.6, 0)] {
            XCTAssertEqual(outward.signedDistance(p), reversed.signedDistance(p), accuracy: 1e-6)
        }
        let cage = try IsosurfaceStuffing.mesh(
            field: { reversed.signedDistance($0) }, bounds: unitCube().bounds(),
            options: IsosurfaceStuffing.Options(spacing: 0.1))
        XCTAssertLessThan(abs(cage.volume - 1), 0.02, "a reversed cube must still mesh the cube")
    }

    /// A cube with its bottom missing is not a closed manifold: the field
    /// falls back to winding numbers and still meshes the interior.
    func testOpenMeshUsesWindingNumbersAndStillMeshes() throws {
        let field = try TriangleMeshDistanceField(mesh: unitCube(dropFace: true))
        XCTAssertEqual(field.topology.boundaryEdges, 4)
        XCTAssertEqual(field.signMethod, .windingNumber)
        XCTAssertLessThan(field.signedDistance(F3(0, 0, 0.2)), 0)
        XCTAssertGreaterThan(field.signedDistance(F3(0, 0, 1)), 0)
        XCTAssertGreaterThan(field.signedDistance(F3(1, 1, 1)), 0)
        let cage = try IsosurfaceStuffing.mesh(
            field: { field.signedDistance($0) }, bounds: unitCube().bounds(),
            options: IsosurfaceStuffing.Options(spacing: 0.1))
        XCTAssertGreaterThan(cage.volume, 0.8, "the open cube's interior must still be filled")
        assertConforming(cage, "open cube")
    }

    func testMalformedInputIsRejected() throws {
        var bad = unitCube()
        bad.vertices[0] = F3(.nan, 0, 0)
        XCTAssertThrowsError(try TriangleMeshDistanceField(mesh: bad)) {
            XCTAssertEqual($0 as? TriangleMeshDistanceField.Error, .nonFiniteVertex(index: 0))
        }
        var outOfRange = unitCube()
        outOfRange.triangles[3] = (0, 1, 99)
        XCTAssertThrowsError(try TriangleMeshDistanceField(mesh: outOfRange)) {
            XCTAssertEqual($0 as? TriangleMeshDistanceField.Error, .triangleIndexOutOfRange(triangle: 3))
        }
        let flat = SurfaceMesh(vertices: [F3(0, 0, 0), F3(1, 0, 0), F3(2, 0, 0)],
                               normals: [], triangles: [(0, 1, 2)])
        XCTAssertThrowsError(try TriangleMeshDistanceField(mesh: flat)) {
            XCTAssertEqual($0 as? TriangleMeshDistanceField.Error, .noUsableTriangles)
        }
        let sphere: @Sendable (F3) -> Float = { simd_length($0) - 0.05 }
        XCTAssertThrowsError(try IsosurfaceStuffing.mesh(
            field: sphere, bounds: (F3(repeating: 0.05), F3(repeating: -0.05)),
            options: IsosurfaceStuffing.Options(spacing: 0.01))) {
            XCTAssertEqual($0 as? IsosurfaceStuffing.Error, .invalidBounds)
        }
        XCTAssertThrowsError(try IsosurfaceStuffing.mesh(
            field: sphere, bounds: (F3(repeating: -0.05), F3(repeating: 0.05)),
            options: IsosurfaceStuffing.Options(spacing: -1))) {
            XCTAssertEqual($0 as? IsosurfaceStuffing.Error, .invalidOptions)
        }
        var huge = IsosurfaceStuffing.Options(spacing: 1e-5)
        huge.maxLatticeVertices = 1_000_000
        XCTAssertThrowsError(try IsosurfaceStuffing.mesh(
            field: sphere, bounds: (F3(repeating: -0.05), F3(repeating: 0.05)), options: huge)) {
            guard case .latticeTooLarge? = $0 as? IsosurfaceStuffing.Error else {
                return XCTFail("expected latticeTooLarge, got \($0)")
            }
        }
        XCTAssertThrowsError(try IsosurfaceStuffing.mesh(
            field: { _ in .nan }, bounds: (F3(repeating: -0.05), F3(repeating: 0.05)),
            options: IsosurfaceStuffing.Options(spacing: 0.02))) {
            guard case .nonFiniteField? = $0 as? IsosurfaceStuffing.Error else {
                return XCTFail("expected nonFiniteField, got \($0)")
            }
        }
    }
}
