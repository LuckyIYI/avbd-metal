import Foundation
import Metal
import simd
import XCTest
@testable import GPUSimRenderer

final class GTAOShadingNormalTests: XCTestCase {
    func testSmoothShadingNormalsDoNotMakeAPlaneOccludeItself() throws {
        let image = try renderPlane()
        let values = (4..<(image.height - 4)).flatMap { y in
            (4..<(image.width - 4)).map { x in image.values[y * image.width + x] }
        }
        let mean = values.reduce(0, +) / Float(values.count)
        print("GTAO plane with smooth shading normals: mean \(mean), minimum \(values.min()!)")
        XCTAssertGreaterThan(mean, 0.99)
        XCTAssertGreaterThan(try XCTUnwrap(values.min()), 0.97,
            "The raw AO estimator must not interpret shading-normal interpolation as occluding geometry")
    }

    func testGeometricPlaneRejectionPreservesRaisedBlockerOcclusion() throws {
        let image = try renderPlane(raisedBlocker: true)
        // The block's projected right edge is x=111.5. These probes remain
        // on the receiving plane while facing a real raised occluder.
        let near = (48..<80).flatMap { y in (114..<122).map { x in image.values[y * image.width + x] } }
        let far = (48..<80).flatMap { y in (164..<180).map { x in image.values[y * image.width + x] } }
        let nearMean = near.reduce(0, +) / Float(near.count)
        let farMean = far.reduce(0, +) / Float(far.count)
        print("GTAO smooth-normal receiver: near blocker \(nearMean), far \(farMean)")
        XCTAssertLessThan(nearMean, 0.95, "Geometric rejection must preserve actual contact occlusion")
        XCTAssertGreaterThan(farMean, 0.99)
    }

    private func renderPlane(raisedBlocker: Bool = false) throws -> (width: Int, height: Int, values: [Float]) {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal is unavailable") }
        let width = 192, height = 128
        let shader = renderShaderSource + """
        struct ShadingNormalFixtureOut {
            float4 normal [[color(0)]];
            float depth [[depth(any)]];
        };
        fragment ShadingNormalFixtureOut shading_normal_fixture(FSOut in [[stage_in]],
            constant Uniforms& U [[buffer(1)]], constant uint& blocker [[buffer(2)]]) {
            ShadingNormalFixtureOut out;
            float4 clip = U.viewProj * float4(0, 0, -3, 1);
            out.depth = clip.z / clip.w;
            // A planar triangle can have tilted, smoothly interpolated vertex
            // normals, such as a vessel base sharing normals with its rim.
            // These affect shading without placing geometry above the plane.
            float2 tilt = float2(0.7 * (in.uv.x * 2 - 1), 0.5 * (in.uv.y * 2 - 1));
            out.normal = float4(normalize(float3(tilt, 1)), 0.2);
            float3 P = worldFromDepth(in.uv, out.depth, U.invViewProj);
            float3 top = P * (2.65 / 3.0);
            if (blocker != 0 && abs(top.x) < 0.3 && abs(top.y) < 0.65) {
                float4 topClip = U.viewProj * float4(top, 1);
                out.depth = topClip.z / topClip.w;
                out.normal = float4(0, 0, 1, 0.2);
            }
            return out;
        }
        """
        let library = try device.makeLibrary(source: shader, options: nil)
        func pipeline(_ fragment: String, _ format: MTLPixelFormat, depth: Bool = false) throws -> MTLRenderPipelineState {
            let descriptor = MTLRenderPipelineDescriptor()
            descriptor.vertexFunction = library.makeFunction(name: "fs_vertex")
            descriptor.fragmentFunction = library.makeFunction(name: fragment)
            descriptor.colorAttachments[0].pixelFormat = format
            if depth { descriptor.depthAttachmentPixelFormat = .depth32Float }
            return try device.makeRenderPipelineState(descriptor: descriptor)
        }
        func texture(_ format: MTLPixelFormat) throws -> MTLTexture {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: format, width: width, height: height, mipmapped: false)
            descriptor.storageMode = .private
            descriptor.usage = [.renderTarget, .shaderRead]
            return try XCTUnwrap(device.makeTexture(descriptor: descriptor))
        }
        let fixture = try pipeline("shading_normal_fixture", .rgba16Float, depth: true)
        let aoPipeline = try pipeline("gtao_fragment", .r8Unorm)
        let depth = try texture(.depth32Float), normal = try texture(.rgba16Float), ao = try texture(.r8Unorm)
        let descriptor = MTLDepthStencilDescriptor()
        descriptor.isDepthWriteEnabled = true
        descriptor.depthCompareFunction = .always
        let depthState = try XCTUnwrap(device.makeDepthStencilState(descriptor: descriptor))
        let y: Float = 1 / tan(25 * .pi / 180), near: Float = 0.1, far: Float = 1000
        let projection = simd_float4x4(columns: (
            SIMD4(y * Float(height) / Float(width), 0, 0, 0), SIMD4(0, y, 0, 0),
            SIMD4(0, 0, far / (near - far), -1), SIMD4(0, 0, near * far / (near - far), 0)))
        var uniforms = Uniforms(viewProj: projection, lightDir: .zero, eye: .zero,
            screen: SIMD4(Float(width), Float(height), Float(height) * y * 0.5, 0),
            camRight: SIMD4(1, 0, 0, 0), camUp: SIMD4(0, -1, 0, 0),
            prevViewProj: projection, temporal: SIMD4(0, 1, 0, 0),
            shadowViewProj: matrix_identity_float4x4, shadowParams: .zero,
            invViewProj: projection.inverse, prevInvViewProj: projection.inverse,
            aoProjection: SIMD4(-projection.columns.2.z, projection.columns.3.z,
                                1 / projection.columns.0.x, 1 / projection.columns.1.y))
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        let prepass = MTLRenderPassDescriptor()
        prepass.colorAttachments[0].texture = normal
        prepass.colorAttachments[0].loadAction = .dontCare
        prepass.colorAttachments[0].storeAction = .store
        prepass.depthAttachment.texture = depth
        prepass.depthAttachment.loadAction = .clear
        prepass.depthAttachment.clearDepth = 1
        prepass.depthAttachment.storeAction = .store
        let pre = try XCTUnwrap(command.makeRenderCommandEncoder(descriptor: prepass))
        pre.setRenderPipelineState(fixture)
        pre.setDepthStencilState(depthState)
        pre.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
        var blocker: UInt32 = raisedBlocker ? 1 : 0
        pre.setFragmentBytes(&blocker, length: MemoryLayout<UInt32>.stride, index: 2)
        pre.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        pre.endEncoding()
        let aoPass = MTLRenderPassDescriptor()
        aoPass.colorAttachments[0].texture = ao
        aoPass.colorAttachments[0].loadAction = .dontCare
        aoPass.colorAttachments[0].storeAction = .store
        let encoder = try XCTUnwrap(command.makeRenderCommandEncoder(descriptor: aoPass))
        encoder.setRenderPipelineState(aoPipeline)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
        encoder.setFragmentTexture(depth, index: 0)
        encoder.setFragmentTexture(normal, index: 1)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
        let stride = 256
        let output = try XCTUnwrap(device.makeBuffer(length: stride * height, options: .storageModeShared))
        let blit = try XCTUnwrap(command.makeBlitCommandEncoder())
        blit.copy(from: ao, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(),
            sourceSize: MTLSize(width: width, height: height, depth: 1), to: output,
            destinationOffset: 0, destinationBytesPerRow: stride, destinationBytesPerImage: stride * height)
        blit.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed, "\(String(describing: command.error))")
        let bytes = output.contents().assumingMemoryBound(to: UInt8.self)
        let values = (0..<height).flatMap { y in
            (0..<width).map { x in Float(bytes[y * stride + x]) / 255 }
        }
        return (width, height, values)
    }
}
