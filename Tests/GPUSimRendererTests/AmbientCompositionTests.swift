import Metal
import simd
import XCTest
@testable import GPUSimRenderer

final class AmbientCompositionTests: XCTestCase {
    func testClothAndFloorPreserveAmbientWhenHQCorrectionIsZero() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        let library = try device.makeLibrary(source: renderShaderSource + """
        vertex VOut ambient_cloth(uint id [[vertex_id]]) {
            float2 p = float2((id << 1) & 2, id & 2)*2-1;
            VOut o = {}; o.position=float4(p,0.5,1); o.world=float3(0.5,0.5,0);
            o.normal=float3(0,0,1); o.albedo=float3(0.6,0.5,0.4); o.opacity=1;
            return o;
        }
        vertex FloorOut ambient_floor(uint id [[vertex_id]]) {
            float2 p = float2((id << 1) & 2, id & 2)*2-1;
            FloorOut o; o.position=float4(p,0.5,1); o.world=float3(0.5,0.5,0);
            return o;
        }
        """, options: nil)
        func texture(_ format: MTLPixelFormat) throws -> MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: 4, height: 4, mipmapped: false)
            d.storageMode = .shared; d.usage = [.renderTarget, .shaderRead]
            return try XCTUnwrap(device.makeTexture(descriptor: d))
        }
        let visibility = try texture(.rgba32Float), indirect = try texture(.rgba32Float)
        let normal = try texture(.rgba32Float)
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: 4, height: 4, mipmapped: false)
        let depth = try XCTUnwrap(device.makeTexture(descriptor: depthDescriptor))
        func fill(_ texture: MTLTexture, _ pixel: SIMD4<Float>) {
            let pixels = Array(repeating: pixel, count: 16)
            pixels.withUnsafeBytes { texture.replace(region: MTLRegionMake2D(0,0,4,4), mipmapLevel: 0,
                withBytes: $0.baseAddress!, bytesPerRow: 64) }
        }
        fill(normal, SIMD4(0,0,1,1))
        let queue = try XCTUnwrap(device.makeCommandQueue())
        for (vertex, fragment) in [("ambient_cloth", "soft_fragment"), ("ambient_floor", "floor_fragment")] {
            let d = MTLRenderPipelineDescriptor()
            d.vertexFunction = library.makeFunction(name: vertex); d.fragmentFunction = library.makeFunction(name: fragment)
            d.colorAttachments[0].pixelFormat = .rgba32Float
            let pipeline = try device.makeRenderPipelineState(descriptor: d)
            func render(hq: Bool, ao: Float = 1, delta: Float = 0, confidence: Float = 1) throws -> SIMD3<Float> {
                fill(visibility, SIMD4(ao,1,0,0)); fill(indirect, SIMD4(delta,delta,delta,confidence))
                let target = try texture(.rgba32Float)
                let pass = MTLRenderPassDescriptor()
                pass.colorAttachments[0].texture = target; pass.colorAttachments[0].storeAction = .store
                let command = try XCTUnwrap(queue.makeCommandBuffer())
                let encoder = try XCTUnwrap(command.makeRenderCommandEncoder(descriptor: pass))
                var u = Uniforms(viewProj: matrix_identity_float4x4, lightDir: SIMD4(0,0,1,0),
                    eye: SIMD4(0.5,0.5,1,0), screen: SIMD4(4,4,1,0), camRight: .zero, camUp: .zero,
                    prevViewProj: matrix_identity_float4x4, temporal: .zero,
                    shadowViewProj: matrix_identity_float4x4, shadowParams: .zero,
                    invViewProj: matrix_identity_float4x4, prevInvViewProj: matrix_identity_float4x4,
                    effects: SIMD4(1,0,0,0), rayTracing: SIMD4(1,0,0,0), aoProjection: .zero)
                // Use identical visibility inputs; toggle only the real HQ
                // diffuse composition, whose open-sky ray output is zero RGB.
                u.reconstruction.x = 1; u.diffuse.x = hq ? 1 : 0
                encoder.setRenderPipelineState(pipeline)
                encoder.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
                encoder.setFragmentTexture(visibility, index: 0); encoder.setFragmentTexture(depth, index: 1)
                encoder.setFragmentTexture(normal, index: 3); encoder.setFragmentTexture(depth, index: 4)
                encoder.setFragmentTexture(indirect, index: 6)
                encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
                encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
                XCTAssertEqual(command.status, .completed, "\(String(describing: command.error))")
                var pixel = SIMD4<Float>.zero
                target.getBytes(&pixel, bytesPerRow: 16, from: MTLRegionMake2D(1,1,1,1), mipmapLevel: 0)
                return SIMD3(pixel.x,pixel.y,pixel.z)
            }
            let fast = try render(hq: false), hq = try render(hq: true)
            let recovered = try render(hq: true, ao: 0.25)
            let bounce = try render(hq: true, delta: 0.1)
            let rejected = try render(hq: true, delta: 0.1, confidence: 0)
            let partial = try render(hq: true, delta: 0.1, confidence: 0.5)
            for channel in 0..<3 {
                XCTAssertGreaterThan(fast[channel], 0.01)
                XCTAssertEqual(hq[channel], fast[channel], accuracy: 0.00001, fragment)
                XCTAssertEqual(recovered[channel], fast[channel], accuracy: 0.00001, fragment)
                XCTAssertGreaterThan(bounce[channel], fast[channel])
                XCTAssertEqual(rejected[channel], fast[channel], accuracy: 0.00001)
                XCTAssertEqual(partial[channel], (fast[channel]+bounce[channel])*0.5, accuracy: 0.00001)
            }
        }
    }
}
