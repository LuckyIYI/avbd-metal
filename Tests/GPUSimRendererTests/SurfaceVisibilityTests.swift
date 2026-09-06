import Metal
import simd
import XCTest
@testable import GPUSimRenderer

final class SurfaceVisibilityTests: XCTestCase {
    func testThinForegroundWithoutAGuideDoesNotInheritBackgroundOcclusion() throws {
        let pixels = try render(mode: 0)
        for x in 14...17 {
            let pixel = pixels[8 * 32 + x]
            XCTAssertEqual(pixel.x, 1, accuracy: 0.001)
            XCTAssertEqual(pixel.y, 1, accuracy: 0.001)
            XCTAssertEqual(pixel.z, 0.1, accuracy: 0.001,
                "Ordinary bilinear upsampling would incorrectly darken this foreground surface")
        }
    }

    func testForegroundRenormalizesOnlyItsOwnVisibilitySamples() throws {
        let pixels = try render(mode: 1)
        // The full-resolution foreground extends beyond the two guide columns
        // that represent it. The other bilinear neighbor belongs to the back.
        for x in [13, 18] {
            let pixel = pixels[8 * 32 + x]
            XCTAssertEqual(pixel.x, 0.6, accuracy: 0.001)
            XCTAssertEqual(pixel.y, 0.8, accuracy: 0.001,
                "Real contact shadow visibility must survive surface ownership filtering")
            XCTAssertLessThan(pixel.z, pixel.x - 0.1)
        }
    }

    func testNormalDiscontinuityRejectsUnrelatedVisibility() throws {
        let pixel = try render(mode: 2)[8 * 32 + 15]
        XCTAssertEqual(pixel.x, 0.2, accuracy: 0.001)
        XCTAssertEqual(pixel.y, 0.4, accuracy: 0.001)
        XCTAssertGreaterThan(pixel.z, 0.3,
            "The control must blend across the normal discontinuity")
    }

    func testTiltedShadingNormalPreservesBilinearVisibilityOnFlatGeometry() throws {
        let pixels = try render(mode: 3)
        for x in 3..<29 {
            let pixel = pixels[8 * 32 + x]
            XCTAssertEqual(pixel.x, pixel.z, accuracy: 0.001,
                "A smoothly shaded planar receiver must not reject its own neighbors")
            XCTAssertEqual(pixel.y, pixel.w, accuracy: 0.001)
        }
    }

    func testHQVisibilityMatchesItsPixelAcrossJitterAndOddTargetSizes() throws {
        for size in [SIMD2(32,16), SIMD2(35,19)] {
            for frame: UInt32 in [0,17,31] {
                let pixels = try render(mode: 4, size: size,
                    jitter: MetalFXReconstruction.sampleJitter(frame), reconstructs: true)
                for pixel in pixels {
                    XCTAssertEqual(pixel.x,pixel.z)
                    XCTAssertEqual(pixel.y,pixel.w,
                        "Matching HQ guide and shading grids must retain exact per-pixel visibility at discontinuities")
                }
            }
        }
    }

    private func render(mode: UInt32, size: SIMD2<Int> = SIMD2(32,16),
                        jitter: SIMD2<Float> = .zero, reconstructs: Bool = false) throws -> [SIMD4<Float>] {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal is unavailable") }
        let width = size.x, height = size.y
        let source = renderShaderSource + """
        struct VisibilityFixtureOut {
            float4 normal [[color(0)]];
            float2 visibility [[color(1)]];
            float depth [[depth(any)]];
        };
        fragment VisibilityFixtureOut visibility_fixture(FSOut in [[stage_in]],
            constant Uniforms& U [[buffer(1)]], constant uint& mode [[buffer(2)]]) {
            bool foreground = mode >= 2 || (mode == 1 && abs(in.uv.x - 0.5) < 3.0 / 32.0);
            float4 clip = U.viewProj * float4(0, 0, foreground ? -2.0 : -4.0, 1);
            VisibilityFixtureOut out;
            out.depth = clip.z / clip.w;
            out.normal = float4(0, 0, 1, 0);
            out.visibility = foreground ? float2(0.6, 0.8) : float2(0.1, 0.2);
            if (mode == 2) {
                out.normal.xyz = in.uv.x < 0.5 ? float3(0, 0, 1) : float3(1, 0, 0);
                out.visibility = in.uv.x < 0.5 ? float2(0.2, 0.4) : float2(0.8, 0.9);
            }
            if (mode == 3) {
                out.normal.xyz = normalize(float3(0.7, 0.3, 0.65));
                out.visibility = float2(0.1 + 0.8 * in.uv.x, 0.4);
            }
            if (mode == 4) {
                bool even = ((uint(in.position.x) + uint(in.position.y)) & 1u) == 0;
                float4 surface = U.viewProj * float4(0,0,even ? -2.0 : -4.0,1);
                out.depth = surface.z / surface.w;
                out.normal.xyz = even ? float3(0,0,1) : normalize(float3(0.8,0.1,0.6));
                out.visibility = even ? float2(0.2,0.3) : float2(0.8,0.9);
            }
            return out;
        }
        fragment float4 visibility_reconstruction_fixture(FSOut in [[stage_in]],
            constant Uniforms& U [[buffer(1)]], constant uint& mode [[buffer(2)]],
            texture2d<float> visibility [[texture(0)]], depth2d<float> depth [[texture(1)]],
            texture2d<float> normal [[texture(2)]]) {
            // This is one full-resolution foreground triangle. Its interpolated
            // world position stays planar even where no guide covers it.
            float4 clip = U.viewProj * float4(0, 0, -2, 1);
            float3 P = worldFromDepth(in.uv, clip.z / clip.w, U.invViewProj);
            float3 N = mode == 3 ? normalize(float3(0.7, 0.3, 0.65)) : float3(0, 0, 1);
            if (mode == 4) {
                P = worldFromDepth(in.uv,depth.read(uint2(in.position.xy)),U.invViewProj);
                N = normal.read(uint2(in.position.xy)).xyz;
            }
            constexpr sampler linearSampler(filter::linear, address::clamp_to_edge);
            float2 reconstructed = surfaceVisibility(in.uv, P, N, U, visibility, depth, normal);
            float2 control = mode == 4 ? visibility.read(uint2(in.position.xy)).rg
                                      : visibility.sample(linearSampler, in.uv).rg;
            return float4(reconstructed, control);
        }
        """
        let library = try device.makeLibrary(source: source, options: nil)
        func pipeline(_ name: String, fixture: Bool) throws -> MTLRenderPipelineState {
            let descriptor = MTLRenderPipelineDescriptor()
            descriptor.vertexFunction = library.makeFunction(name: "fs_vertex")
            descriptor.fragmentFunction = library.makeFunction(name: name)
            descriptor.colorAttachments[0].pixelFormat = fixture ? .rgba16Float : .rgba32Float
            if fixture {
                descriptor.colorAttachments[1].pixelFormat = .rg32Float
                descriptor.depthAttachmentPixelFormat = .depth32Float
            }
            return try device.makeRenderPipelineState(descriptor: descriptor)
        }
        func texture(_ format: MTLPixelFormat, width: Int, height: Int) throws -> MTLTexture {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: format, width: width, height: height, mipmapped: false)
            descriptor.storageMode = .private
            descriptor.usage = [.shaderRead, .renderTarget]
            return try XCTUnwrap(device.makeTexture(descriptor: descriptor))
        }
        let fixturePipeline = try pipeline("visibility_fixture", fixture: true)
        let reconstructionPipeline = try pipeline("visibility_reconstruction_fixture", fixture: false)
        let guideWidth = reconstructs ? width : width / 2
        let guideHeight = reconstructs ? height : height / 2
        let normal = try texture(.rgba16Float, width: guideWidth, height: guideHeight)
        let visibility = try texture(.rg32Float, width: guideWidth, height: guideHeight)
        let depth = try texture(.depth32Float, width: guideWidth, height: guideHeight)
        let result = try texture(.rgba32Float, width: width, height: height)
        let y: Float = 1 / tan(25 * .pi / 180), near: Float = 0.1, far: Float = 100
        let projection = simd_float4x4(columns: (
            SIMD4(y * Float(height) / Float(width), 0, 0, 0), SIMD4(0, y, 0, 0),
            SIMD4(0, 0, far / (near - far), -1), SIMD4(0, 0, near * far / (near - far), 0)))
        var jittered = projection
        jittered.columns.2.x -= 2 * jitter.x / Float(width)
        jittered.columns.2.y += 2 * jitter.y / Float(height)
        var uniforms = Uniforms(viewProj: jittered, lightDir: .zero, eye: .zero,
            screen: SIMD4(Float(width), Float(height), Float(height) * y * 0.5, 0),
            camRight: SIMD4(1, 0, 0, 0), camUp: SIMD4(0, -1, 0, 0),
            prevViewProj: projection, temporal: .zero,
            shadowViewProj: matrix_identity_float4x4, shadowParams: .zero,
            invViewProj: jittered.inverse, prevInvViewProj: jittered.inverse,
            aoProjection: SIMD4(-projection.columns.2.z, projection.columns.3.z,
                                1 / projection.columns.0.x, 1 / projection.columns.1.y))
        uniforms.reconstruction = SIMD4(reconstructs ? 1 : 0,
            jitter.x / Float(width),jitter.y / Float(height),0)
        var fixtureMode = mode
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        let depthDescriptor = MTLDepthStencilDescriptor()
        depthDescriptor.isDepthWriteEnabled = true
        depthDescriptor.depthCompareFunction = .always
        let depthState = try XCTUnwrap(device.makeDepthStencilState(descriptor: depthDescriptor))
        let guides = MTLRenderPassDescriptor()
        for (index, texture) in [normal, visibility].enumerated() {
            guides.colorAttachments[index].texture = texture
            guides.colorAttachments[index].loadAction = .dontCare
            guides.colorAttachments[index].storeAction = .store
        }
        guides.depthAttachment.texture = depth
        guides.depthAttachment.loadAction = .clear
        guides.depthAttachment.clearDepth = 1
        guides.depthAttachment.storeAction = .store
        let guideEncoder = try XCTUnwrap(command.makeRenderCommandEncoder(descriptor: guides))
        guideEncoder.setRenderPipelineState(fixturePipeline)
        guideEncoder.setDepthStencilState(depthState)
        guideEncoder.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
        guideEncoder.setFragmentBytes(&fixtureMode, length: MemoryLayout<UInt32>.stride, index: 2)
        guideEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        guideEncoder.endEncoding()
        let output = MTLRenderPassDescriptor()
        output.colorAttachments[0].texture = result
        output.colorAttachments[0].loadAction = .dontCare
        output.colorAttachments[0].storeAction = .store
        let encoder = try XCTUnwrap(command.makeRenderCommandEncoder(descriptor: output))
        encoder.setRenderPipelineState(reconstructionPipeline)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
        encoder.setFragmentBytes(&fixtureMode, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setFragmentTexture(visibility, index: 0)
        encoder.setFragmentTexture(depth, index: 1)
        encoder.setFragmentTexture(normal, index: 2)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
        let stride = (width * MemoryLayout<SIMD4<Float>>.stride + 255) / 256 * 256
        let buffer = try XCTUnwrap(device.makeBuffer(length: stride * height, options: .storageModeShared))
        let blit = try XCTUnwrap(command.makeBlitCommandEncoder())
        blit.copy(from: result, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(),
            sourceSize: MTLSize(width: width, height: height, depth: 1), to: buffer,
            destinationOffset: 0, destinationBytesPerRow: stride, destinationBytesPerImage: stride * height)
        blit.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed, "\(String(describing: command.error))")
        return (0..<height).flatMap { y in
            Array(UnsafeBufferPointer(start: buffer.contents().advanced(by: y * stride)
                .assumingMemoryBound(to: SIMD4<Float>.self), count: width))
        }
    }
}
