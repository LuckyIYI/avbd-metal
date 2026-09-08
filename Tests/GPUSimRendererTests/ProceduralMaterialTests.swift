import Metal
import simd
import XCTest
@testable import GPUSimRenderer

final class ProceduralMaterialTests: XCTestCase {
    func testTextureVariationFilteringAndAmbientExposureOnGPU() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let library = try device.makeLibrary(source: renderShaderSource + """
        kernel void material_probe(device float4* out [[buffer(0)]], constant Uniforms* lighting [[buffer(1)]], uint i [[thread_position_in_grid]]) {
            float3 p = float3(float(i)*0.0037, float(i)*0.0071, 0.8225);
            float4 d = float4(1,1,0.5,0.0002);
            float fine = materialPattern(p,d,0.0001), coarse = filteredMaterialNoise(p*180,4);
            float3 base = pbrRadiance(float3(0.5),0.6,0,float3(0),float3(0,0,-1),normalize(float3(0.1,0,-1)),1,0,lighting[0]);
            float3 bright = pbrRadiance(float3(0.5),0.6,0,float3(0),float3(0,0,-1),normalize(float3(0.1,0,-1)),1,0,lighting[1]);
            out[i] = float4(fine,coarse,base.x,bright.x);
        }
        """, options: nil)
        let state = try device.makeComputePipelineState(function: XCTUnwrap(library.makeFunction(name: "material_probe")))
        let output = try XCTUnwrap(device.makeBuffer(length: 128*16, options: .storageModeShared))
        let uniforms = try XCTUnwrap(device.makeBuffer(length: MemoryLayout<Uniforms>.stride*2, options: .storageModeShared))
        memset(uniforms.contents(),0,uniforms.length)
        let u = uniforms.contents().bindMemory(to: Uniforms.self, capacity: 2)
        u[0].lightDir = SIMD4(0,0,-1,0); u[1].lightDir = SIMD4(0,0,-1,0); u[1].rayScene.y = 2
        let command = try XCTUnwrap(device.makeCommandQueue()?.makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        encoder.setBuffer(uniforms, offset: 0, index: 1)
        encoder.setComputePipelineState(state); encoder.setBuffer(output, offset: 0, index: 0)
        encoder.dispatchThreads(MTLSize(width: 128,height: 1,depth: 1), threadsPerThreadgroup: MTLSize(width: 32,height: 1,depth: 1))
        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed)
        let pixels = output.contents().bindMemory(to: SIMD4<Float>.self, capacity: 128)
        let values = (0..<128).map { pixels[$0].x }
        XCTAssertGreaterThan(values.max()!-values.min()!, 0.2)
        for i in 0..<128 {
            XCTAssertEqual(pixels[i].y,0.5)
            XCTAssertEqual(pixels[i].w,pixels[i].z*4,accuracy: 1e-5)
        }
    }
}
