import Metal
import XCTest

@testable import GPUSimRenderer

@MainActor
final class EnvironmentLightTests: XCTestCase {
  private func texture(_ device: MTLDevice, color: SIMD4<Float>) throws -> MTLTexture {
    let d = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: .rgba32Float, width: 32, height: 16, mipmapped: false)
    d.storageMode = .shared
    d.usage = .shaderRead
    let t = try XCTUnwrap(device.makeTexture(descriptor: d))
    let values = Array(repeating: color, count: 32 * 16)
    values.withUnsafeBytes {
      t.replace(
        region: MTLRegionMake2D(0, 0, 32, 16), mipmapLevel: 0, withBytes: $0.baseAddress!,
        bytesPerRow: 32 * 16)
    }
    return t
  }

  func testConstantHDRPreservesLinearRadianceAndDiffuseEnergy() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let tex = try texture(device, color: SIMD4(0.1, 0.3, 4, 1))
    let environment = try GPUSimEnvironmentLight(device: device, texture: tex, intensity: 2)
    let resources = try GPUSimMaterialLibrary(device: device, environment: environment)
    let source =
      renderShaderSource + """
        kernel void environment_probe(device float4* out [[buffer(0)]],constant Uniforms& u [[buffer(1)]],constant MaterialResources& resources [[buffer(10)]]) {
            out[0]=float4(materialEnvironment(float3(1,0,0),u,resources),1);
            out[1]=float4(materialDiffuseAmbient(float3(0,0,1),u,resources),1);
            out[2]=float4(materialDiffuseAmbient(normalize(float3(1,-2,3)),u,resources),1);
        }
        """
    let lib = try device.makeLibrary(source: source, options: nil)
    let p = try device.makeComputePipelineState(
      function: XCTUnwrap(lib.makeFunction(name: "environment_probe")))
    let out = try XCTUnwrap(device.makeBuffer(length: 48, options: .storageModeShared))
    let command = try XCTUnwrap(device.makeCommandQueue()?.makeCommandBuffer())
    let e = try XCTUnwrap(command.makeComputeCommandEncoder())
    let uniforms = try XCTUnwrap(
      device.makeBuffer(length: MemoryLayout<Uniforms>.stride, options: .storageModeShared))
    memset(uniforms.contents(), 0, uniforms.length)
    e.setComputePipelineState(p)
    e.setBuffer(out, offset: 0, index: 0)
    e.setBuffer(uniforms, offset: 0, index: 1)
    resources.bind(e)
    e.dispatchThreads(
      .init(width: 1, height: 1, depth: 1),
      threadsPerThreadgroup: .init(width: 1, height: 1, depth: 1))
    e.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    XCTAssertEqual(command.status, .completed, "\(String(describing:command.error))")
    let values = out.contents().assumingMemoryBound(to: SIMD4<Float>.self)
    for i in 0..<3 {
      XCTAssertEqual(values[i].x, 0.2, accuracy: 0.003)
      XCTAssertEqual(values[i].y, 0.6, accuracy: 0.003)
      XCTAssertEqual(
        values[i].z, 8, accuracy: 0.01, "HDR must not be clamped or encoded as sRGB twice")
    }
    XCTAssertEqual(resources.textureBytes, tex.allocatedSize)
  }

  func testEnvironmentValidationAndSharedTextureBudget() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let tex = try texture(device, color: SIMD4(repeating: 1))
    for intensity: Float in [-1, .nan, .infinity] {
      XCTAssertThrowsError(
        try GPUSimEnvironmentLight(device: device, texture: tex, intensity: intensity))
    }
    XCTAssertThrowsError(try GPUSimEnvironmentLight(device: device, texture: tex, rotation: .nan))
    XCTAssertThrowsError(
      try GPUSimEnvironmentLight(device: device, texture: tex, diffuseSamples: 1))
    let environment = try GPUSimEnvironmentLight(device: device, texture: tex)
    XCTAssertThrowsError(
      try GPUSimMaterialLibrary(device: device, textureBudget: 1, environment: environment))
    var m = GPUSimSurfaceMaterial()
    m.baseColorTexture = tex
    let shared = try GPUSimMaterialLibrary(device: device, materials: [m], environment: environment)
    XCTAssertEqual(shared.textureBytes, tex.allocatedSize, "A shared texture is budgeted once")
  }

  func testDefaultQualityAndBoundedOverrides() {
    let options = GPUSimRenderOptions()
    XCTAssertEqual(options.rayTracingQuality, .realtime)
    XCTAssertEqual(GPUSimRayTracingQuality(transmissionInterfaces: 0).resolved.transmissionInterfaces, 0)
    XCTAssertEqual(options.sunIntensity, 1)
    XCTAssertEqual(options.sunAngularRadius, 0)
    let q = GPUSimRayTracingQuality(
      shadowSamples: 999, reflectionSamples: -5, diffuseSamples: 999, transmissionInterfaces: 999
    ).resolved
    XCTAssertEqual(q.shadowSamples, 64)
    XCTAssertEqual(q.reflectionSamples, 1)
    XCTAssertEqual(q.diffuseSamples, 64)
    XCTAssertEqual(q.transmissionInterfaces, 32)
    var invalid = options
    invalid.sunAngularRadius = .nan
    invalid.sunIntensity = .nan
    let resolved = invalid.resolved(supportsHQ: true)
    XCTAssertEqual(resolved.sunAngularRadius, 0)
    XCTAssertEqual(resolved.sunIntensity, 1)
  }
}
