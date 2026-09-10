import CoreGraphics
import ImageIO
import Metal
import XCTest

@testable import GPUSimRenderer

@MainActor
final class MaterialTextureTests: XCTestCase {
  private func image(at url: URL) throws {
    let pixels = Data((0..<16).flatMap { _ in [UInt8(64), 128, 192, 255] })
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

  func testPackedLinearChannelsNormalConventionAndMipGeneration() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
      UUID().uuidString + ".png")
    defer { try? FileManager.default.removeItem(at: url) }
    try image(at: url)
    let texture = try GPUSimMaterialLibrary.loadTexture(device: device, url: url, sRGB: false)
    XCTAssertEqual(texture.mipmapLevelCount, 3)
    XCTAssertThrowsError(
      try GPUSimMaterialLibrary.loadTexture(
        device: device, url: url, sRGB: false, maximumDecodedBytes: 16))
    var m = GPUSimSurfaceMaterial()
    m.roughnessTexture = texture
    m.metallicTexture = texture
    m.normalTexture = texture
    m.roughnessChannel = 1
    m.metallicChannel = 2
    m.metallic = 1
    m.invertNormalGreen = true
    m.uvScale.x = -1 // Mirroring the material UV transform also mirrors its tangent normal.
    let resources = try GPUSimMaterialLibrary(device: device, materials: [m])
    XCTAssertEqual(resources.textures.count, 1)
    XCTAssertThrowsError(
      try GPUSimMaterialLibrary(device: device, materials: [m], textureBudget: 1))
    let source =
      renderShaderSource + """
        kernel void packed_probe(device float4* result [[buffer(0)]],constant MaterialResources& resources [[buffer(10)]]) {
          MaterialContext context = {float3(0),float2(0.5),1,float4(0)};
          MaterialSample surface = {float3(1),1,0,float3(0),float3(0,0,1)};
          surface = evaluateMaterial(1,context,surface,resources);
          result[0]=float4(surface.roughness,surface.metallic,surface.normal.x,surface.normal.y);
        }
        """
    let library = try device.makeLibrary(source: source, options: nil)
    let pipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(library.makeFunction(name: "packed_probe")))
    let out = try XCTUnwrap(device.makeBuffer(length: 16, options: .storageModeShared))
    let command = try XCTUnwrap(device.makeCommandQueue()?.makeCommandBuffer())
    let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(out, offset: 0, index: 0)
    resources.bind(encoder)
    encoder.dispatchThreads(
      MTLSize(width: 1, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    XCTAssertEqual(command.status, .completed)
    let result = out.contents().assumingMemoryBound(to: SIMD4<Float>.self).pointee
    XCTAssertEqual(result.x, 128.0 / 255, accuracy: 0.004)
    XCTAssertEqual(result.y, 192.0 / 255, accuracy: 0.004)
    XCTAssertEqual(result.z, -(64.0 / 255 * 2 - 1), accuracy: 0.008)
    XCTAssertEqual(result.w, -(128.0 / 255 * 2 - 1), accuracy: 0.008)
  }

  func testDecodeBudgetEstimateFollowsSourceBitDepth() throws {
    // 8-bit sources decode to RGBA8, so an 8k map fits the default budget.
    let rgba8 = GPUSimMaterialLibrary.decodedByteEstimate(
      width: 8192, height: 4096, bitsPerComponent: 8, isFloat: false)
    func mipped(_ bytes: Int) -> Int { Int((Double(bytes) * 4 / 3).rounded(.up)) }
    XCTAssertEqual(rgba8, mipped(8192 * 4096 * 4))
    XCTAssertLessThan(rgba8, 512 * 1024 * 1024)
    XCTAssertEqual(
      GPUSimMaterialLibrary.decodedByteEstimate(
        width: 16, height: 16, bitsPerComponent: 16, isFloat: false), mipped(16 * 16 * 8))
    XCTAssertEqual(
      GPUSimMaterialLibrary.decodedByteEstimate(
        width: 16, height: 16, bitsPerComponent: 32, isFloat: true), mipped(16 * 16 * 16))
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
      UUID().uuidString + ".png")
    defer { try? FileManager.default.removeItem(at: url) }
    try image(at: url)
    // An exhausted budget is a budget error, not an invalid image.
    XCTAssertThrowsError(
      try GPUSimMaterialLibrary.loadTexture(
        device: device, url: url, sRGB: false, maximumDecodedBytes: 0)
    ) { error in
      guard case GPUSimMaterialLibrary.Failure.textureBudgetExceeded? =
        error as? GPUSimMaterialLibrary.Failure
      else { return XCTFail("unexpected \(error)") }
    }
  }

  func testOBJLoadsRelativeImageAndRejectsMissingTexture() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    try image(at: directory.appendingPathComponent("color.png"))
    let mtl = directory.appendingPathComponent("fixture.mtl")
    try "newmtl Image\nKd 1 1 1\nmap_Kd color.png\n".write(
      to: mtl, atomically: true, encoding: .utf8)
    let obj = directory.appendingPathComponent("fixture.obj")
    try """
    mtllib fixture.mtl
    v 0 0 0
    v 1 0 0
    v 0 1 0
    vt 0 0
    vt 1 0
    vt 0 1
    usemtl Image
    f 1/1 2/2 3/3
    """.write(to: obj, atomically: true, encoding: .utf8)
    let asset = try GPUSimAssetImporter.load(url: obj, device: device)
    let material = try XCTUnwrap(asset.materials.first)
    let texture = try XCTUnwrap(material.baseColorTexture)
    XCTAssertEqual(texture.width, 4)
    XCTAssertEqual(texture.mipmapLevelCount, 3)
    XCTAssertTrue([MTLPixelFormat.rgba8Unorm_srgb, .bgra8Unorm_srgb].contains(texture.pixelFormat))
    try "newmtl Image\nKd 1 1 1\nmap_Kd missing.png\n".write(
      to: mtl, atomically: true, encoding: .utf8)
    XCTAssertThrowsError(try GPUSimAssetImporter.load(url: obj, device: device)) { error in
      XCTAssertNotNil(error as? GPUSimAssetImporter.Failure, "importer errors use one enum: \(error)")
    }
    // Library validation failures are reported through the importer's enum too.
    try "newmtl Image\nKd 1 1 1\nmap_Kd color.png\n".write(
      to: mtl, atomically: true, encoding: .utf8)
    XCTAssertThrowsError(
      try GPUSimAssetImporter.load(url: obj, device: device, textureBudget: 1)
    ) { error in
      guard case GPUSimAssetImporter.Failure.textureBudgetExceeded? =
        error as? GPUSimAssetImporter.Failure
      else { return XCTFail("unexpected \(error)") }
    }
  }
}
