import Metal
import XCTest
import simd

@testable import GPUSimRenderer

@MainActor
final class ProceduralMaterialTests: XCTestCase {
  func testImageChannelsAndInjectedProgramOnGPU() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let desc = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: .rgba8Unorm_srgb, width: 2, height: 2, mipmapped: false)
    desc.usage = .shaderRead
    desc.storageMode = .shared
    let texture = try XCTUnwrap(device.makeTexture(descriptor: desc))
    let pixels: [UInt8] = [128, 0, 0, 255, 128, 0, 0, 255, 128, 0, 0, 255, 128, 0, 0, 255]
    pixels.withUnsafeBytes {
      texture.replace(
        region: MTLRegionMake2D(0, 0, 2, 2), mipmapLevel: 0, withBytes: $0.baseAddress!,
        bytesPerRow: 8)
    }
    var image = GPUSimSurfaceMaterial()
    image.baseColorTexture = texture
    image.roughness = 0.37
    var custom = image
    custom.program = 1
    custom.parameters = SIMD4(0.3, 0.7, 0.2, 0)
    let programs = [
      GPUSimMaterialProgram(
        body: "surface.color = tint(context.parameters.xyz); surface.metallic = 0.8;",
        supportingSource: "inline float3 tint(float3 value) { return value; }"),
      GPUSimMaterialProgram(
        body: "surface.color = tint(context.parameters.xyz);",
        supportingSource: "inline float3 tint(float3 value) { return value * 0.5; }"),
    ]
    let resources = try GPUSimMaterialLibrary(
      device: device, materials: [image, custom], programs: programs)
    XCTAssertEqual(resources.textures.count, 1)
    let source =
      makeRenderShaderSource(programs: programs) + """
        kernel void probe(device float4* output [[buffer(0)]], constant MaterialResources& resources [[buffer(10)]], uint i [[thread_position_in_grid]]) {
            MaterialContext context = { float3(0),float2(0.5),0,float4(0) };
            MaterialSample surface = {float3(0.9),0.6,0,float3(0),float3(0,0,1)};
            surface = evaluateMaterial(i,context,surface,resources);
            output[i] = float4(surface.color,surface.roughness);
            output[4+i] = float4(surface.metallic);
        }
        """
    let library = try device.makeLibrary(source: source, options: nil)
    let pipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(library.makeFunction(name: "probe")))
    let out = try XCTUnwrap(device.makeBuffer(length: 8 * 16, options: .storageModeShared))
    let command = try XCTUnwrap(device.makeCommandQueue()?.makeCommandBuffer())
    let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(out, offset: 0, index: 0)
    resources.bind(encoder)
    encoder.dispatchThreads(
      MTLSize(width: 4, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 4, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    XCTAssertEqual(command.status, .completed, "\(String(describing: command.error))")
    let result = out.contents().bindMemory(to: SIMD4<Float>.self, capacity: 8)
    XCTAssertEqual(result[0].x, 0.9, accuracy: 0.0001)
    XCTAssertEqual(result[1].x, 0.21586 * 0.9, accuracy: 0.003)  // hardware sRGB decode, not a gamma approximation
    XCTAssertEqual(result[1].w, 0.37, accuracy: 0.0001)
    XCTAssertEqual(result[2].x, 0.3, accuracy: 0.0001)
    XCTAssertEqual(result[2].y, 0.7, accuracy: 0.0001)
    XCTAssertEqual(result[6].x, 0.8, accuracy: 0.0001)
    XCTAssertEqual(result[3].x, 0.9, accuracy: 0.0001)  // unknown ID is a safe vertex-material fallback
    _ = try GPUSimRenderer(device: device, materials: resources)
  }

  func testInvalidMaterialAndProgramFailBeforeRendering() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    var m = GPUSimSurfaceMaterial()
    m.roughness = .nan
    XCTAssertThrowsError(try GPUSimMaterialLibrary(device: device, materials: [m]))
    m.roughness = 0.5
    m.program = 1
    XCTAssertThrowsError(try GPUSimMaterialLibrary(device: device, materials: [m]))
    let bad = try GPUSimMaterialLibrary(
      device: device, programs: [.init(body: "this is not valid Metal;")])
    XCTAssertThrowsError(try GPUSimRenderer(device: device, materials: bad))
  }
}
