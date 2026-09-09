import Metal
import ModelIO
import XCTest
import simd

@testable import GPUSimRenderer

@MainActor
final class AssetMaterialTests: XCTestCase {
  func testOBJPreservesUVsSubmeshMaterialsAndTransform() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    try """
    newmtl Red
    Kd 0.8 0.1 0.2
    d 1.0
    newmtl Green
    Kd 0.1 0.7 0.2
    d 1.0
    """.write(
      to: directory.appendingPathComponent("fixture.mtl"), atomically: true, encoding: .utf8)
    let url = directory.appendingPathComponent("fixture.obj")
    try """
    mtllib fixture.mtl
    v 0 0 0
    v 1 0 0
    v 0 1 0
    v 1 1 0
    vt 0 0
    vt 1 0
    vt 0 1
    vt 1 1
    vn 0 0 1
    usemtl Red
    f 1/1/1 2/2/1 3/3/1
    usemtl Green
    f 2/2/1 4/4/1 3/3/1
    """.write(to: url, atomically: true, encoding: .utf8)
    var transform = matrix_identity_float4x4
    transform.columns.0.x = -2
    transform.columns.3.x = 3
    let asset = try GPUSimAssetImporter.load(url: url, device: device, transform: transform)
    XCTAssertEqual(asset.parts.count, 2)
    XCTAssertEqual(Set(asset.parts.map(\.materialID)).count, 2)
    XCTAssertTrue(asset.materials.contains { $0.baseColor.x > 0.7 })
    XCTAssertTrue(asset.materials.contains { $0.baseColor.y > 0.6 })
    for part in asset.parts {
      XCTAssertEqual(part.textureCoordinates.count, part.mesh.vertices.count)
      XCTAssertTrue(part.mesh.vertices.allSatisfy { (1...3).contains($0.x) })
      XCTAssertTrue(part.mesh.normals.allSatisfy { abs($0.z - 1) < 0.001 })
      let t = try XCTUnwrap(part.mesh.triangles.first)
      let n = simd_cross(
        part.mesh.vertices[t.1] - part.mesh.vertices[t.0],
        part.mesh.vertices[t.2] - part.mesh.vertices[t.0])
      XCTAssertGreaterThan(n.z, 0, "mirrored transforms must reverse winding")
      XCTAssertEqual(part.rigidMesh(body: 0).materialID, part.materialID)
    }
    XCTAssertThrowsError(try GPUSimAssetImporter.load(url: url, device: device, vertexBudget: 1))
  }

  func testUSDHierarchyAndCapabilityReporting() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    XCTAssertTrue(GPUSimAssetImporter.canImport("obj"))
    XCTAssertTrue(GPUSimAssetImporter.canImport("usda"))
    XCTAssertFalse(GPUSimAssetImporter.canImport("not-a-format"))
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
      UUID().uuidString + ".usda")
    defer { try? FileManager.default.removeItem(at: url) }
    try """
    #usda 1.0
    (upAxis = "Z")
    def Xform "Parent" {
        double3 xformOp:translate = (2, 3, 4)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Mesh "Triangle" {
            point3f[] points = [(0,0,0),(1,0,0),(0,1,0)]
            int[] faceVertexCounts = [3]
            int[] faceVertexIndices = [0,1,2]
            uniform token subdivisionScheme = "none"
        }
    }
    """.write(to: url, atomically: true, encoding: .utf8)
    let asset = try GPUSimAssetImporter.load(url: url, device: device)
    XCTAssertEqual(asset.parts.count, 1)
    let p = try XCTUnwrap(asset.parts.first?.mesh.vertices.first)
    XCTAssertEqual(p.x, 2, accuracy: 0.001)
    XCTAssertEqual(p.y, 3, accuracy: 0.001)
    XCTAssertEqual(p.z, 4, accuracy: 0.001)
  }
}
