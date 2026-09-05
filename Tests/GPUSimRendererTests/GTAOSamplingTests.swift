import Metal
import simd
import XCTest
@testable import GPUSimRenderer

final class GTAOSamplingTests: XCTestCase {
    func testEverySlidingFootprintReconstructsFixedSpatialSamples() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal is unavailable") }
        let size = 64
        let library = try device.makeLibrary(source: renderShaderSource + """
        kernel void sample_test(device float2* output [[buffer(0)]],
            uint3 p [[thread_position_in_grid]]) {
            // Independent invocations use the same spatial-only API.
            output[p.z * 4096 + p.y * 64 + p.x] = gtaoSampleNoise(p.xy);
        }
        struct SamplingFixtureOut {
            float4 visibility [[color(0)]];
            float4 normal [[color(1)]];
            float depth [[depth(any)]];
        };
        fragment SamplingFixtureOut sampling_fixture(FSOut in [[stage_in]],
            constant Uniforms& U [[buffer(1)]], constant uint& probe [[buffer(2)]]) {
            float2 sample = gtaoSampleNoise(uint2(in.position.xy));
            float visibility;
            if (probe == 0) {
                // Rectangle integral is exactly 0.37 * 0.61.
                visibility = sample.x < 0.37 && sample.y < 0.61 ? 1.0 : 0.0;
            } else {
                // Cosine-weighted hemisphere: z=sqrt(1-r). An occluder
                // covers 37% of azimuth below this fixed elevation.
                float z = sqrt(1.0 - sample.y);
                visibility = sample.x < 0.37 && z < sqrt(0.51) ? 0.0 : 1.0;
            }
            SamplingFixtureOut out;
            out.visibility = float4(visibility);
            out.normal = float4(0, 0, 1, 0.2);
            float4 clip = U.viewProj * float4(0, 0, -3, 1);
            out.depth = clip.z / clip.w;
            return out;
        }
        """, options: nil)
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let sampling = try device.makeComputePipelineState(function: XCTUnwrap(library.makeFunction(name: "sample_test")))
        let output = try XCTUnwrap(device.makeBuffer(
            length: size * size * 2 * MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared))
        let sampleCommand = try XCTUnwrap(queue.makeCommandBuffer())
        let compute = try XCTUnwrap(sampleCommand.makeComputeCommandEncoder())
        compute.setComputePipelineState(sampling)
        compute.setBuffer(output, offset: 0, index: 0)
        compute.dispatchThreads(MTLSize(width: size, height: size, depth: 2),
            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        compute.endEncoding()
        sampleCommand.commit()
        sampleCommand.waitUntilCompleted()
        XCTAssertEqual(sampleCommand.status, .completed, "\(String(describing: sampleCommand.error))")
        let samples = output.contents().assumingMemoryBound(to: SIMD2<Float>.self)
        for pixel in 0..<(size * size) {
            let sample = samples[pixel]
            XCTAssertTrue(sample.x.isFinite && sample.y.isFinite)
            XCTAssertTrue(sample.x >= 0 && sample.x < 1 && sample.y >= 0 && sample.y < 1)
            XCTAssertEqual(sample, samples[size * size + pixel], "Sampling cannot change between invocations")
        }
        // The closest reconstruction footprint must cover the full radial
        // range too. Four angles sharing one radial quartile form visible
        // blocks even when the larger 4x4 footprint has all sixteen pairs.
        for y in Swift.stride(from: 0, to: size, by: 2) {
            for x in Swift.stride(from: 0, to: size, by: 2) {
                var angles = Set<Int>(), radii = Set<Int>()
                for dy in 0..<2 {
                    for dx in 0..<2 {
                        let sample = samples[(y + dy) * size + x + dx]
                        angles.insert(Int(sample.x * 4))
                        radii.insert(Int(sample.y * 4))
                    }
                }
                XCTAssertEqual(angles, Set(0..<4), "Angle quartiles in quad at \(x),\(y)")
                XCTAssertEqual(radii, Set(0..<4), "Radius quartiles in quad at \(x),\(y)")
            }
        }
        for y in 0...(size - 4) {
            for x in 0...(size - 4) {
                var angles = Set<Int>(), radii = Set<Int>(), pairs = Set<Int>()
                for dy in 0..<4 {
                    for dx in 0..<4 {
                        let sample = samples[(y + dy) * size + x + dx]
                        let angle = Int(sample.x * 16), radius = Int(sample.y * 16)
                        angles.insert(angle); radii.insert(radius); pairs.insert(angle * 16 + radius)
                    }
                }
                XCTAssertEqual(angles.count, 16, "Angle strata at \(x),\(y)")
                XCTAssertEqual(radii.count, 16, "Radius strata at \(x),\(y)")
                XCTAssertEqual(pairs.count, 16, "Quadrature pairs at \(x),\(y)")
            }
        }

        func texture(_ format: MTLPixelFormat) throws -> MTLTexture {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: format, width: size, height: size, mipmapped: false)
            descriptor.storageMode = .private
            descriptor.usage = [.renderTarget, .shaderRead]
            return try XCTUnwrap(device.makeTexture(descriptor: descriptor))
        }
        let ambient = try texture(.r8Unorm), normal = try texture(.rgba16Float)
        let depth = try texture(.depth32Float), resolved = try texture(.rg8Unorm)
        let fixtureDescriptor = MTLRenderPipelineDescriptor()
        fixtureDescriptor.vertexFunction = library.makeFunction(name: "fs_vertex")
        fixtureDescriptor.fragmentFunction = library.makeFunction(name: "sampling_fixture")
        fixtureDescriptor.colorAttachments[0].pixelFormat = .r8Unorm
        fixtureDescriptor.colorAttachments[1].pixelFormat = .rgba16Float
        fixtureDescriptor.depthAttachmentPixelFormat = .depth32Float
        let fixture = try device.makeRenderPipelineState(descriptor: fixtureDescriptor)
        let resolveDescriptor = MTLRenderPipelineDescriptor()
        resolveDescriptor.vertexFunction = library.makeFunction(name: "fs_vertex")
        resolveDescriptor.fragmentFunction = library.makeFunction(name: "visibility_fragment")
        resolveDescriptor.colorAttachments[0].pixelFormat = .rg8Unorm
        let resolve = try device.makeRenderPipelineState(descriptor: resolveDescriptor)
        let depthDescriptor = MTLDepthStencilDescriptor()
        depthDescriptor.depthCompareFunction = .always
        depthDescriptor.isDepthWriteEnabled = true
        let depthState = try XCTUnwrap(device.makeDepthStencilState(descriptor: depthDescriptor))
        let focal: Float = 1 / tan(25 * .pi / 180)
        let near: Float = 0.1, far: Float = 1000
        let projection = simd_float4x4(columns: (
            SIMD4(focal, 0, 0, 0), SIMD4(0, focal, 0, 0),
            SIMD4(0, 0, far / (near - far), -1), SIMD4(0, 0, near * far / (near - far), 0)))
        var uniforms = Uniforms(viewProj: projection, lightDir: .zero, eye: .zero,
            screen: SIMD4(Float(size), Float(size), Float(size) * focal * 0.5, 0),
            camRight: SIMD4(1, 0, 0, 0), camUp: SIMD4(0, -1, 0, 0),
            prevViewProj: projection, temporal: .zero,
            shadowViewProj: matrix_identity_float4x4, shadowParams: .zero,
            invViewProj: projection.inverse, prevInvViewProj: projection.inverse,
            aoProjection: SIMD4(-projection.columns.2.z, projection.columns.3.z,
                                1 / projection.columns.0.x, 1 / projection.columns.1.y))
        let stride = 256
        let readback = try XCTUnwrap(device.makeBuffer(length: stride * size, options: .storageModeShared))
        for (probe, expected): (UInt32, Float) in [(0, 0.37 * 0.61), (1, 1 - 0.37 * 0.51)] {
            let command = try XCTUnwrap(queue.makeCommandBuffer())
            let prepass = MTLRenderPassDescriptor()
            for (index, target) in [ambient, normal].enumerated() {
                prepass.colorAttachments[index].texture = target
                prepass.colorAttachments[index].loadAction = .dontCare
                prepass.colorAttachments[index].storeAction = .store
            }
            prepass.depthAttachment.texture = depth
            prepass.depthAttachment.loadAction = .clear
            prepass.depthAttachment.clearDepth = 1
            prepass.depthAttachment.storeAction = .store
            let pre = try XCTUnwrap(command.makeRenderCommandEncoder(descriptor: prepass))
            pre.setRenderPipelineState(fixture)
            pre.setDepthStencilState(depthState)
            pre.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
            var probe = probe
            pre.setFragmentBytes(&probe, length: MemoryLayout<UInt32>.stride, index: 2)
            pre.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            pre.endEncoding()
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = resolved
            pass.colorAttachments[0].loadAction = .dontCare
            pass.colorAttachments[0].storeAction = .store
            let encoder = try XCTUnwrap(command.makeRenderCommandEncoder(descriptor: pass))
            encoder.setRenderPipelineState(resolve)
            encoder.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
            for (index, input) in [ambient, ambient, depth, normal].enumerated() {
                encoder.setFragmentTexture(input, index: index)
            }
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            encoder.endEncoding()
            let blit = try XCTUnwrap(command.makeBlitCommandEncoder())
            blit.copy(from: resolved, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(),
                sourceSize: MTLSize(width: size, height: size, depth: 1), to: readback,
                destinationOffset: 0, destinationBytesPerRow: stride, destinationBytesPerImage: stride * size)
            blit.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            XCTAssertEqual(command.status, .completed, "\(String(describing: command.error))")
            let bytes = readback.contents().assumingMemoryBound(to: UInt8.self)
            var values: [Float] = []
            values.reserveCapacity((size - 8) * (size - 8))
            for y in 4..<(size - 4) {
                for x in 4..<(size - 4) {
                    let index: Int = y * stride + x * 2
                    values.append(Float(bytes[index]) / 255.0)
                }
            }
            let mean = values.reduce(0, +) / Float(values.count)
            XCTAssertEqual(values.min(), values.max(), "A matched 4x4 resolve must leave no periodic spatial variation")
            XCTAssertEqual(mean, expected, accuracy: 1 / 16 + 1 / 255,
                "The 16-sample quadrature must reproduce the known visibility within one sample plus output quantization")
        }
    }
}
