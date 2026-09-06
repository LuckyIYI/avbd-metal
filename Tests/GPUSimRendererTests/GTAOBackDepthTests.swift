import Metal
import simd
import XCTest
@testable import GPUSimRenderer

final class GTAOBackDepthTests: XCTestCase {
    func testFirstExitIgnoresTriangleWindingAndDrawOrder() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal is unavailable") }
        let library = try device.makeLibrary(source: renderShaderSource + """
        vertex VOut back_depth_fixture(uint vid [[vertex_id]],
            constant Uniforms& U [[buffer(1)]], constant uint& configuration [[buffer(2)]]) {
            constexpr float2 corners[4] = {
                float2(-1, -1), float2(1, -1), float2(1, 1), float2(-1, 1)
            };
            constexpr uint indices[6] = { 0, 1, 2, 0, 2, 3 };
            uint plane = vid / 6;
            if ((configuration & 4u) != 0) plane = 2u - plane;
            uint winding = configuration & 3u;
            bool reverse = winding == 1u || (winding == 2u && ((vid / 3) & 1u) != 0);
            uint corner = vid % 3;
            if (reverse && corner > 0) corner = 3u - corner;
            uint index = indices[((vid % 6) / 3) * 3 + corner];
            // Entrance, first exit, and an unrelated farther exit. All are
            // parallel; reversing triangles never changes outward normals.
            float z = plane == 0u ? -1.0 : (plane == 1u ? -2.0 : -3.5);
            VOut out = collapse();
            out.world = float3(corners[index], z);
            out.position = U.viewProj * float4(out.world, 1);
            out.normal = float3(0, 0, plane == 0u ? 1.0 : -1.0);
            return out;
        }
        """, options: nil)
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = library.makeFunction(name: "back_depth_fixture")
        descriptor.fragmentFunction = library.makeFunction(name: "gtao_back_depth_fragment")
        descriptor.depthAttachmentPixelFormat = .depth32Float
        let pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        let depthDescriptor = MTLDepthStencilDescriptor()
        depthDescriptor.depthCompareFunction = .less
        depthDescriptor.isDepthWriteEnabled = true
        let depthState = try XCTUnwrap(device.makeDepthStencilState(descriptor: depthDescriptor))
        let width = 32, height = 32
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: width, height: height, mipmapped: false)
        textureDescriptor.storageMode = .private
        textureDescriptor.usage = [.renderTarget, .shaderRead]
        let depth = try XCTUnwrap(device.makeTexture(descriptor: textureDescriptor))
        // Orthographic projection gives device depth .2 for the entrance,
        // .4 for the first exit, and .7 for the farther exit.
        let projection = simd_float4x4(columns: (
            SIMD4(1, 0, 0, 0), SIMD4(0, 1, 0, 0),
            SIMD4(0, 0, -0.2, 0), SIMD4(0, 0, 0, 1)))
        var uniforms = Uniforms(viewProj: projection, lightDir: .zero, eye: .zero,
            screen: SIMD4(Float(width), Float(height), Float(height), 0),
            camRight: SIMD4(1, 0, 0, 0), camUp: SIMD4(0, -1, 0, 0),
            prevViewProj: projection, temporal: .zero,
            shadowViewProj: matrix_identity_float4x4, shadowParams: .zero,
            invViewProj: projection.inverse, prevInvViewProj: projection.inverse,
            aoProjection: .zero)
        // 0/1/2: normal, reversed, and mixed winding; bit2: reverse draw
        // order; bit3: draw only the open front plane with no finite exit.
        let configurations: [UInt32] = [0, 1, 2, 4, 5, 6, 8, 9, 10]
        let rowBytes = 256, imageBytes = rowBytes * height
        let readback = try XCTUnwrap(device.makeBuffer(
            length: imageBytes * configurations.count, options: .storageModeShared))
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        for (index, configuration) in configurations.enumerated() {
            let pass = MTLRenderPassDescriptor()
            pass.depthAttachment.texture = depth
            pass.depthAttachment.loadAction = .clear
            pass.depthAttachment.clearDepth = 1
            pass.depthAttachment.storeAction = .store
            let encoder = try XCTUnwrap(command.makeRenderCommandEncoder(descriptor: pass))
            encoder.setRenderPipelineState(pipeline)
            encoder.setDepthStencilState(depthState)
            encoder.setCullMode(.none)
            encoder.setVertexBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
            encoder.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
            var configuration = configuration
            encoder.setVertexBytes(&configuration, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0,
                vertexCount: configuration & 8 == 0 ? 18 : 6)
            encoder.endEncoding()
            let blit = try XCTUnwrap(command.makeBlitCommandEncoder())
            blit.copy(from: depth, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(),
                sourceSize: MTLSize(width: width, height: height, depth: 1), to: readback,
                destinationOffset: index * imageBytes, destinationBytesPerRow: rowBytes,
                destinationBytesPerImage: imageBytes)
            blit.endEncoding()
        }
        command.commit()
        command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed, "\(String(describing: command.error))")
        for (index, configuration) in configurations.enumerated() {
            let values = readback.contents().advanced(by: index * imageBytes).assumingMemoryBound(to: Float.self)
            var minimum: Float = 1, maximum: Float = 0
            for y in 3..<(height - 3) {
                for x in 3..<(width - 3) {
                    let value = values[y * (rowBytes / MemoryLayout<Float>.stride) + x]
                    minimum = min(minimum, value)
                    maximum = max(maximum, value)
                }
            }
            let expected: Float = configuration & 8 == 0 ? 0.4 : 1
            XCTAssertEqual(minimum, expected, accuracy: 1e-6, "Configuration \(configuration) minimum")
            XCTAssertEqual(maximum, expected, accuracy: 1e-6, "Configuration \(configuration) maximum")
        }
    }
}
