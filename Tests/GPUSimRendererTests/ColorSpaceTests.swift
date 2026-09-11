import Metal
import XCTest

@testable import GPUSimRenderer

@MainActor
final class ColorSpaceTests: XCTestCase {
  func testDisplayExposureScalesLinearHDRBeforeToneMapping() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let library = try device.makeLibrary(
      source: renderShaderSource + """
        kernel void exposure_probe(device float4* output [[buffer(0)]],
            constant Uniforms* uniforms [[buffer(1)]], uint i [[thread_position_in_grid]]) {
            output[i]=float4(displayTonemap(float3(0.18,0.5,2),uniforms[i]),1);
        }
        """, options: nil)
    let pipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(library.makeFunction(name: "exposure_probe")))
    let output = try XCTUnwrap(device.makeBuffer(length: 48, options: .storageModeShared))
    let stride = MemoryLayout<Uniforms>.stride
    let uniforms = try XCTUnwrap(device.makeBuffer(length: stride * 3, options: .storageModeShared))
    memset(uniforms.contents(), 0, uniforms.length)
    let offset = try XCTUnwrap(MemoryLayout<Uniforms>.offset(of: \.displaySettings))
    for (i, stops) in [Float(-1), 0, 1].enumerated() {
      uniforms.contents().storeBytes(of: stops, toByteOffset: i * stride + offset, as: Float.self)
    }
    let command = try XCTUnwrap(device.makeCommandQueue()?.makeCommandBuffer())
    let e = try XCTUnwrap(command.makeComputeCommandEncoder())
    e.setComputePipelineState(pipeline)
    e.setBuffer(output, offset: 0, index: 0)
    e.setBuffer(uniforms, offset: 0, index: 1)
    e.dispatchThreads(
      .init(width: 3, height: 1, depth: 1),
      threadsPerThreadgroup: .init(width: 1, height: 1, depth: 1))
    e.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    XCTAssertEqual(command.status, .completed)
    let result = output.contents().assumingMemoryBound(to: SIMD4<Float>.self)
    for (i, factor) in [Float(0.5), 1, 2].enumerated() {
      for (j, radiance) in [Float(0.18), 0.5, 2].enumerated() {
        let x = radiance * factor
        let expected = min(1, (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14))
        XCTAssertEqual(result[i][j], expected, accuracy: 0.00001)
      }
    }
    var options = GPUSimRenderOptions()
    XCTAssertEqual(options.displayExposure, 0)
    options.displayExposure = .nan
    XCTAssertEqual(options.resolved(supportsHQ: true).displayExposure, 0)
    options.displayExposure = 100
    XCTAssertEqual(options.resolved(supportsHQ: true).displayExposure, 16)
    options.displayExposure = -100
    XCTAssertEqual(options.resolved(supportsHQ: true).displayExposure, -16)
  }

  func testLinearPaletteEncodedForVertexABIIsDecodedOnce() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let library = try device.makeLibrary(
      source: renderShaderSource + """
        kernel void color_probe(device float4* output [[buffer(0)]]) {
            float3 palette=float3(0.18,0.5,0.8);
            output[0]=float4(srgbToLin(linearToSRGBExact(palette)),1);
            output[1]=float4(sRGBToLinearExact(linearToSRGBExact(palette)),1);
        }
        """, options: nil)
    let pipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(library.makeFunction(name: "color_probe")))
    let output = try XCTUnwrap(device.makeBuffer(length: 32, options: .storageModeShared))
    let command = try XCTUnwrap(device.makeCommandQueue()?.makeCommandBuffer())
    let e = try XCTUnwrap(command.makeComputeCommandEncoder())
    e.setComputePipelineState(pipeline)
    e.setBuffer(output, offset: 0, index: 0)
    e.dispatchThreads(
      .init(width: 1, height: 1, depth: 1),
      threadsPerThreadgroup: .init(width: 1, height: 1, depth: 1))
    e.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    XCTAssertEqual(command.status, .completed)
    let result = output.contents().assumingMemoryBound(to: SIMD4<Float>.self)
    for (i, expected) in [Float(0.18), 0.5, 0.8].enumerated() {
      XCTAssertEqual(
        result[0][i], expected, accuracy: 0.003,
        "The existing fast sRGB approximation must not darken linear palettes twice")
      XCTAssertEqual(result[1][i], expected, accuracy: 0.000001)
    }
  }
}
