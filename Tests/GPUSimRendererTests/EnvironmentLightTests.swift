import Metal
import PhysicsAVBD
import SimCore
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
    let environment = try GPUSimEnvironmentLight(device: device, texture: tex)
    let materials = try GPUSimMaterialLibrary(device: device)
    let resources = try GPUSimLightingBindings(materials: materials, environment: environment)
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
    uniforms.contents().assumingMemoryBound(to: Uniforms.self).pointee.environmentSettings.x = 1
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
    XCTAssertThrowsError(
      try GPUSimEnvironmentLight(device: device, texture: tex, diffuseSamples: 1))
    let environment = try GPUSimEnvironmentLight(device: device, texture: tex)
    XCTAssertThrowsError(
      try GPUSimLightingBindings(materials: GPUSimMaterialLibrary(device: device, textureBudget: 1), environment: environment))
    var m = GPUSimSurfaceMaterial()
    m.baseColorTexture = tex
    let shared = try GPUSimLightingBindings(materials: GPUSimMaterialLibrary(device: device, materials: [m]), environment: environment)
    XCTAssertEqual(shared.textureBytes, tex.allocatedSize, "A shared texture is budgeted once")
  }

  func testEnvironmentReplacementKeepsMaterialsAndRayWorld() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let a = try GPUSimEnvironmentLight(device: device, texture: texture(device, color: SIMD4(1,0,0,1)))
    let b = try GPUSimEnvironmentLight(device: device, texture: texture(device, color: SIMD4(0,1,0,1)))
    let materials = try GPUSimMaterialLibrary(device: device)
    let shader = try materials.shaderLibrary()
    let renderer = try GPUSimRenderer(device: device, materials: materials, environment: a)
    let first = try GPUSimLightingBindings(materials: materials, environment: a)
    let oldArguments = Data(bytes: first.arguments.contents(), count: first.arguments.length)
    try renderer.setEnvironment(b)
    XCTAssertTrue(renderer.environment === b)
    XCTAssertTrue(try materials.shaderLibrary() === shader)
    XCTAssertEqual(Data(bytes: first.arguments.contents(), count: first.arguments.length), oldArguments,
                   "An in-flight binding must remain immutable after another camera swaps lighting")
    if device.supportsRaytracing {
      let solver = try GPUSolver(scene: PhysicsScene(name: "Environment ownership"), device: device)
      let world = try RayTracingScene.shared(scene: solver, materials: materials)
      try renderer.setEnvironment(a)
      XCTAssertTrue(try RayTracingScene.shared(scene: solver, materials: materials) === world)
    }
    let limited = try GPUSimRenderer(device: device,
      materials: GPUSimMaterialLibrary(device: device, textureBudget: 0))
    XCTAssertThrowsError(try limited.setEnvironment(a))
    XCTAssertNil(limited.environment, "Rejected lighting changes leave the renderer unchanged")
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
