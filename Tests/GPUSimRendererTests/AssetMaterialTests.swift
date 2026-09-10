import CoreGraphics
import ImageIO
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

  private func writeImage(to url: URL, value: UInt8) throws {
    let pixels = Data((0..<16).flatMap { _ in [value, value, value, 255] })
    let provider = try XCTUnwrap(CGDataProvider(data: pixels as CFData))
    let image = try XCTUnwrap(
      CGImage(
        width: 4, height: 4, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 16,
        space: CGColorSpace(name: CGColorSpace.sRGB)!,
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue), provider: provider,
        decode: nil, shouldInterpolate: false, intent: .defaultIntent))
    let target = try XCTUnwrap(
      CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil))
    CGImageDestinationAddImage(target, image, nil)
    XCTAssertTrue(CGImageDestinationFinalize(target))
  }

  private func usdPreviewSurface(body: String) -> String {
    """
    #usda 1.0
    (upAxis = "Z")
    def Xform "Root" {
        def Mesh "Tri" (prepend apiSchemas = ["MaterialBindingAPI"]) {
            point3f[] points = [(0,0,0),(1,0,0),(0,1,0)]
            int[] faceVertexCounts = [3]
            int[] faceVertexIndices = [0,1,2]
            texCoord2f[] primvars:st = [(0,0),(1,0),(0,1)] (interpolation = "vertex")
            uniform token subdivisionScheme = "none"
            rel material:binding = </Root/Mat>
        }
        def Material "Mat" {
            token outputs:surface.connect = </Root/Mat/PBR.outputs:surface>
            def Shader "PBR" {
                uniform token info:id = "UsdPreviewSurface"
    \(body)
                token outputs:surface
            }
            def Shader "Reader" {
                uniform token info:id = "UsdPrimvarReader_float2"
                token inputs:varname = "st"
                float2 outputs:result
            }
            def Shader "ColorTex" {
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @color.png@
                float2 inputs:st.connect = </Root/Mat/Reader.outputs:result>
                float3 outputs:rgb
            }
            def Shader "RoughTex" {
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @rough.png@
                float2 inputs:st.connect = </Root/Mat/Reader.outputs:result>
                float outputs:r
            }
        }
    }
    """
  }

  /// Model I/O reports its own default grey `baseColor` beside a connected USD
  /// diffuse map; the importer must treat the map as replacing every constant.
  func testUSDTexturedMaterialUsesUnitFactorsAndConstantsPreferAuthoredInputs() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    try writeImage(to: directory.appendingPathComponent("color.png"), value: 200)
    try writeImage(to: directory.appendingPathComponent("rough.png"), value: 90)
    let textured = directory.appendingPathComponent("textured.usda")
    try usdPreviewSurface(
      body: """
                color3f inputs:diffuseColor = (1, 1, 1)
                color3f inputs:diffuseColor.connect = </Root/Mat/ColorTex.outputs:rgb>
                float inputs:roughness = 0.5
                float inputs:roughness.connect = </Root/Mat/RoughTex.outputs:r>
        """).write(to: textured, atomically: true, encoding: .utf8)
    let asset = try GPUSimAssetImporter.load(url: textured, device: device)
    let material = try XCTUnwrap(asset.materials.first)
    XCTAssertNotNil(material.baseColorTexture)
    XCTAssertNotNil(material.roughnessTexture)
    XCTAssertEqual(material.baseColor, SIMD3(repeating: 1), "a connected map replaces the constant")
    XCTAssertEqual(material.roughness, 1)
    XCTAssertEqual(material.metallic, 0)

    let constant = directory.appendingPathComponent("constant.usda")
    try usdPreviewSurface(
      body: """
                color3f inputs:diffuseColor = (0.9, 0.2, 0.1)
                float inputs:roughness = 0.25
                float inputs:metallic = 1.5
        """).write(to: constant, atomically: true, encoding: .utf8)
    let plain = try XCTUnwrap(try GPUSimAssetImporter.load(url: constant, device: device).materials.first)
    XCTAssertNil(plain.baseColorTexture)
    XCTAssertEqual(plain.baseColor.x, 0.9, accuracy: 0.001, "authored input beats Model I/O's default grey")
    XCTAssertEqual(plain.baseColor.y, 0.2, accuracy: 0.001)
    XCTAssertEqual(plain.roughness, 0.25, accuracy: 0.001)
    XCTAssertEqual(plain.metallic, 1, "out-of-range authored scalars are clamped, not rejected")
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
