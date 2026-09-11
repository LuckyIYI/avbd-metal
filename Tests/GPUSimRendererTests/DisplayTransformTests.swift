import Foundation
import Metal
import XCTest

@testable import GPUSimRenderer

@MainActor
final class DisplayTransformTests: XCTestCase {
  private func texture(_ device: MTLDevice, edge: Int, half: Bool = false) throws -> MTLTexture {
    let d = MTLTextureDescriptor()
    d.textureType = .type3D
    d.pixelFormat = half ? .rgba16Float : .rgba32Float
    d.width = edge
    d.height = edge
    d.depth = edge
    d.storageMode = .shared
    d.usage = .shaderRead
    return try XCTUnwrap(device.makeTexture(descriptor: d))
  }

  private func evaluate(
    _ transform: GPUSimDisplayTransform, inputs: [SIMD4<Float>], exposure: Float = 0
  ) throws -> [SIMD4<Float>] {
    let device = transform.device
    let library = try device.makeLibrary(
      source: makeRenderShaderSource(displayProgram: transform.program) + """
        kernel void display_probe(device const float4* input [[buffer(0)]],
            constant Uniforms& U [[buffer(1)]],device float4* output [[buffer(2)]],
            texture3d<float> lut [[texture(8)]],uint i [[thread_position_in_grid]]) {
            output[i]=float4(linearToSRGBExact(displayTonemap(input[i].rgb,U,lut)),1);
        }
        """, options: nil)
    let pipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(library.makeFunction(name: "display_probe")))
    let input = try XCTUnwrap(
      inputs.withUnsafeBytes {
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
      })
    let output = try XCTUnwrap(device.makeBuffer(length: input.length, options: .storageModeShared))
    let uniforms = try XCTUnwrap(
      device.makeBuffer(length: MemoryLayout<Uniforms>.stride, options: .storageModeShared))
    memset(uniforms.contents(), 0, uniforms.length)
    uniforms.contents().assumingMemoryBound(to: Uniforms.self).pointee.displaySettings = SIMD4(
      exposure, 1, 0, 0)
    let command = try XCTUnwrap(device.makeCommandQueue()?.makeCommandBuffer())
    let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(input, offset: 0, index: 0)
    encoder.setBuffer(uniforms, offset: 0, index: 1)
    encoder.setBuffer(output, offset: 0, index: 2)
    encoder.setTexture(transform.texture, index: 8)
    encoder.dispatchThreads(
      .init(width: inputs.count, height: 1, depth: 1),
      threadsPerThreadgroup: .init(width: 1, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    XCTAssertEqual(command.status, .completed, "\(String(describing:command.error))")
    return Array(
      UnsafeBufferPointer(
        start: output.contents().assumingMemoryBound(to: SIMD4<Float>.self), count: inputs.count))
  }

  func testValidationBudgetAndProgramCache() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let t = try texture(device, edge: 2)
    let program = GPUSimDisplayProgram(body: "return radiance/(1+radiance);")
    XCTAssertNoThrow(try GPUSimDisplayTransform(device: device, program: program, texture: t))
    XCTAssertNoThrow(try GPUSimDisplayTransform(device: device, program: program))
    XCTAssertThrowsError(
      try GPUSimDisplayTransform(device: device, program: GPUSimDisplayProgram(body: "")))
    XCTAssertThrowsError(
      try GPUSimDisplayTransform(device: device, program: program, texture: t, textureBudget: 0))
    let invalid = try XCTUnwrap(
      device.makeTexture(
        descriptor: MTLTextureDescriptor.texture2DDescriptor(
          pixelFormat: .rgba16Float, width: 2, height: 2, mipmapped: false)))
    XCTAssertThrowsError(
      try GPUSimDisplayTransform(device: device, program: program, texture: invalid))
    let resources = try GPUSimMaterialLibrary(device: device)
    let original = try resources.shaderLibrary()
    let custom = try resources.shaderLibrary(displayProgram: program)
    XCTAssertFalse(original === custom)
    XCTAssertTrue(try resources.shaderLibrary(displayProgram: program) === custom)
  }

  func testTexelCentersAxesExposureAndClampingOnGPU() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let t = try texture(device, edge: 2)
    var values = [SIMD4<Float>]()
    for b: Float in [0, 1] {
      for g: Float in [0, 1] { for r: Float in [0, 1] { values.append(SIMD4(r, g, b, 1)) } }
    }
    values.withUnsafeBytes {
      t.replace(
        region: MTLRegionMake3D(0, 0, 0, 2, 2, 2), mipmapLevel: 0, slice: 0,
        withBytes: $0.baseAddress!, bytesPerRow: 32, bytesPerImage: 64)
    }
    let program = GPUSimDisplayProgram(
      body:
        "constexpr sampler s(coord::normalized,address::clamp_to_edge,filter::linear); return lut.sample(s,clamp(radiance,0.0,1.0)*0.5+0.25).rgb;"
    )
    let transform = try GPUSimDisplayTransform(device: device, program: program, texture: t)
    let colors = try evaluate(
      transform, inputs: [SIMD4(0.25, 0.5, 0.75, 1), SIMD4(-1, 0, 0, 1), SIMD4(repeating: 100)])
    for (i, expected) in [Float(0.5370987), 0.735357, 0.880825].enumerated() {
      XCTAssertEqual(colors[0][i], expected, accuracy: 0.0001)
    }
    XCTAssertEqual(colors[1], SIMD4(0, 0, 0, 1))
    for j in 0..<3 { XCTAssertEqual(colors[2][j], 1, accuracy: 0.000001) }
    let brighter = try evaluate(
      transform, inputs: [SIMD4(0.25, 0.5, 0.75, 1), SIMD4(0, 0, 0, 1)], exposure: 1)
    for (i, expected) in [Float(0.735357), 1, 1].enumerated() {
      XCTAssertEqual(brighter[0][i], expected, accuracy: 0.0001)
    }
    XCTAssertEqual(brighter[1], SIMD4(0, 0, 0, 1))
  }

  func testOptionalOCIOReferenceOnGPU() throws {
    guard let path = ProcessInfo.processInfo.environment["AVBD_DISPLAY_LUT_FIXTURE"] else {
      throw XCTSkip("Set AVBD_DISPLAY_LUT_FIXTURE to an independently baked OCIO fixture")
    }
    let root = URL(fileURLWithPath: path)
    let metadata = try XCTUnwrap(
      JSONSerialization.jsonObject(
        with: Data(contentsOf: root.appendingPathComponent("display-lut.json"))) as? [String: Any])
    let n = try XCTUnwrap(metadata["size"] as? Int)
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let t = try texture(device, edge: n)
    let bytes = try Data(
      contentsOf: root.appendingPathComponent(try XCTUnwrap(metadata["file"] as? String)))
    XCTAssertEqual(bytes.count, n * n * n * 16)
    bytes.withUnsafeBytes {
      t.replace(
        region: MTLRegionMake3D(0, 0, 0, n, n, n), mipmapLevel: 0, slice: 0,
        withBytes: $0.baseAddress!, bytesPerRow: n * 16, bytesPerImage: n * n * 16)
    }
    let source = try String(
      contentsOf: root.appendingPathComponent(try XCTUnwrap(metadata["shader_file"] as? String)),
      encoding: .utf8)
    let program = GPUSimDisplayProgram(
      body: try XCTUnwrap(metadata["body"] as? String), supportingSource: source)
    let transform = try GPUSimDisplayTransform(device: device, program: program, texture: t)
    func floats(_ name: String) throws -> [Float] {
      try Data(contentsOf: root.appendingPathComponent(name)).withUnsafeBytes {
        Array($0.bindMemory(to: Float.self))
      }
    }
    let input = try floats("display-probe-input.f32")
    let reference = try floats("display-probe-reference.f32")
    XCTAssertEqual(input.count, reference.count)
    let colors = try evaluate(
      transform,
      inputs: stride(from: 0, to: input.count, by: 3).map {
        SIMD4(input[$0], input[$0 + 1], input[$0 + 2], 1)
      })
    var errors = [Float]()
    for i in colors.indices {
      for j in 0..<3 { errors.append(abs(colors[i][j] - reference[i * 3 + j]) * 255) }
    }
    errors.sort()
    print(
      "OCIO display LUT GPU: max \(errors.last!), p99 \(errors[Int(Double(errors.count-1)*0.99)]) 8-bit codes"
    )
    XCTAssertLessThan(errors.last!, 0.05)
    XCTAssertLessThan(errors[Int(Double(errors.count - 1) * 0.99)], 0.01)
  }

  func testSwitchingResourcesKeepsTheCompiledProgramContract() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let program = GPUSimDisplayProgram(body: "return radiance/(1+radiance);")
    let transform = try GPUSimDisplayTransform(device: device, program: program)
    let renderer = try GPUSimRenderer(device: device, displayTransform: transform)
    XCTAssertTrue(renderer.displayTransform === transform)
    try renderer.setDisplayTransform(nil)
    XCTAssertNil(renderer.displayTransform)
    try renderer.setDisplayTransform(transform)
    let incompatible = try GPUSimDisplayTransform(
      device: device, program: GPUSimDisplayProgram(body: "return float3(0);"))
    XCTAssertThrowsError(try renderer.setDisplayTransform(incompatible))
    XCTAssertTrue(
      renderer.displayTransform === transform,
      "A rejected program must not replace the active transform")
  }
}
