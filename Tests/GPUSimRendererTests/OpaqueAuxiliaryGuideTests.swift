import Metal
import simd
import XCTest
@testable import GPUSimRenderer

@MainActor
final class OpaqueAuxiliaryGuideTests: XCTestCase {
    func testCoplanarAuxiliaryWinsGuidesAndShadingWithoutOverwritingNearerGeometry() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        let renderer = try GPUSimRenderer(device: device)
        let library = try device.makeLibrary(source: renderShaderSource + """
        vertex VOut auxiliary_receiver_fixture(uint vid [[vertex_id]], constant float4& receiver [[buffer(0)]]) {
            const float2 xy[3] = {float2(-1,-1),float2(3,-1),float2(-1,3)};
            VOut o = {};
            o.position = float4(xy[vid],receiver.w,1);
            o.world = o.position.xyz; o.previousWorld = o.world;
            o.normal = float3(0,0,1); o.albedo = receiver.rgb; o.pbr = float2(0.5,0);
            return o;
        }
        fragment float4 auxiliary_receiver_color(VOut in [[stage_in]]) { return float4(in.albedo,1); }
        """, options: nil)
        let width = 17, height = 13
        func texture(_ format: MTLPixelFormat) throws -> MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: width, height: height, mipmapped: false)
            d.storageMode = .private; d.usage = [.renderTarget, .shaderRead]
            return try XCTUnwrap(device.makeTexture(descriptor: d))
        }
        let formats = MetalFXReconstruction.guideFormats
        let guideTextures = try formats.map(texture)
        let guideDepth = try texture(.depth32Float), mainDepth = try texture(.depth32Float)
        let mainColor = try texture(.rgba16Float)
        func pipeline(guides: Bool) throws -> MTLRenderPipelineState {
            let d = MTLRenderPipelineDescriptor()
            d.vertexFunction = library.makeFunction(name: "auxiliary_receiver_fixture")
            d.fragmentFunction = library.makeFunction(name: guides ? "reconstruction_fragment" : "auxiliary_receiver_color")
            for (i, format) in (guides ? formats : [.rgba16Float]).enumerated() { d.colorAttachments[i].pixelFormat = format }
            d.depthAttachmentPixelFormat = .depth32Float
            return try device.makeRenderPipelineState(descriptor: d)
        }
        let guidePipeline = try pipeline(guides: true), mainPipeline = try pipeline(guides: false)
        let queue = try XCTUnwrap(device.makeCommandQueue()), command = try XCTUnwrap(queue.makeCommandBuffer())
        var u = Uniforms(viewProj: matrix_identity_float4x4, lightDir: .zero, eye: SIMD4(0,0,2,0),
            screen: SIMD4(Float(width),Float(height),Float(height),0), camRight: SIMD4(1,0,0,0), camUp: SIMD4(0,-1,0,0),
            prevViewProj: matrix_identity_float4x4, temporal: .zero, shadowViewProj: matrix_identity_float4x4,
            shadowParams: .zero, invViewProj: matrix_identity_float4x4, prevInvViewProj: matrix_identity_float4x4,
            aoProjection: SIMD4(1,-0.1,1,1))
        var reconstruction = MetalFXReconstruction.GuideUniforms(current: matrix_identity_float4x4,
            previous: matrix_identity_float4x4, size: SIMD4(Float(width),Float(height),0,0))
        for guides in [true, false] {
            let pass = MTLRenderPassDescriptor()
            for (i, target) in (guides ? guideTextures : [mainColor]).enumerated() {
                pass.colorAttachments[i].texture = target; pass.colorAttachments[i].loadAction = .clear
                pass.colorAttachments[i].storeAction = .store
            }
            pass.depthAttachment.texture = guides ? guideDepth : mainDepth
            pass.depthAttachment.loadAction = .clear; pass.depthAttachment.clearDepth = 1
            pass.depthAttachment.storeAction = .dontCare
            let encoder = try XCTUnwrap(command.makeRenderCommandEncoder(descriptor: pass))
            try GPUSimMaterialLibrary(device: device).bind(encoder)
            encoder.setRenderPipelineState(guides ? guidePipeline : mainPipeline)
            encoder.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
            encoder.setFragmentBytes(&reconstruction, length: MemoryLayout<MetalFXReconstruction.GuideUniforms>.stride, index: 2)
            encoder.setDepthStencilState(renderer.depthState)
            // Equal-depth green auxiliary must replace red; blue is farther
            // away and must fail. Guides call the renderer's actual binding.
            for (index, receiver) in [SIMD4<Float>(0.75,0.1,0.15,0.5), SIMD4(0.1,0.5,0.2,0.5), SIMD4(0.2,0.3,0.9,0.7)].enumerated() {
                if index == 1 {
                    if guides { renderer.setOpaqueAuxiliaryDepth(on: encoder) }
                    else { encoder.setDepthStencilState(renderer.auxiliaryOpaqueDepthState) }
                }
                var value = receiver
                encoder.setVertexBytes(&value, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
                encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            }
            encoder.endEncoding()
        }
        let row = 256
        let buffers = try (0..<2).map { _ in try XCTUnwrap(device.makeBuffer(length: row * height, options: .storageModeShared)) }
        let blit = try XCTUnwrap(command.makeBlitCommandEncoder())
        for (i, target) in [guideTextures[2], mainColor].enumerated() {
            blit.copy(from: target, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(),
                sourceSize: MTLSize(width: width, height: height, depth: 1), to: buffers[i], destinationOffset: 0,
                destinationBytesPerRow: row, destinationBytesPerImage: row * height)
        }
        blit.endEncoding(); command.commit(); command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed, "\(String(describing: command.error))")
        let guide = buffers[0].contents().assumingMemoryBound(to: UInt16.self)
        let main = buffers[1].contents().assumingMemoryBound(to: UInt16.self)
        for y in 0..<height { for x in 0..<width {
            let index = y * row / 2 + x * 4
            XCTAssertEqual(guide[index + 1], Float16(0.5).bitPattern, "Guide must belong to the equal-depth auxiliary receiver")
            for channel in 0..<4 { XCTAssertEqual(guide[index + channel], main[index + channel]) }
        } }
    }
}
