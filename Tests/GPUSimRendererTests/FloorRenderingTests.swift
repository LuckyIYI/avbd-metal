import Metal
import simd
import XCTest
@testable import GPUSimRenderer

final class FloorRenderingTests: XCTestCase {
    /// Integrate by splitting the pixel interval at actual one-metre tile
    /// boundaries, independently of the shader's periodic filter formula.
    private func oddCoverage(center: Double, width: Double) -> Double {
        var x = center - width * 0.5
        let end = center + width * 0.5
        var covered = 0.0
        while x < end {
            let cell = floor(x)
            let next = min(end, cell + 1)
            if Int(cell) & 1 != 0 { covered += next - x }
            x = next
        }
        return covered / width
    }

    private func checkerReference(_ sample: SIMD4<Float>) -> Float {
        let x = oddCoverage(center: Double(sample.x), width: Double(sample.z))
        let y = oddCoverage(center: Double(sample.y), width: Double(sample.w))
        return Float(x * (1 - y) + (1 - x) * y)
    }

    private func render(_ samples: [SIMD4<Float>]) throws -> [Float] {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal is unavailable") }
        let source = renderShaderSource + """
        vertex FloorOut floor_filter_fixture(uint vid [[vertex_id]], constant float4& F [[buffer(2)]]) {
            float2 p = float2((vid << 1) & 2, vid & 2) * 2.0 - 1.0;
            float2 pixel = (p * float2(0.5, -0.5) + 0.5) * 4.0;
            FloorOut o; o.position = float4(p, 0.5, 1);
            o.world = float3(F.xy + (pixel - 1.5) * F.zw, 0.005);
            return o;
        }
        """
        let library = try device.makeLibrary(source: source, options: nil)
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = library.makeFunction(name: "floor_filter_fixture")
        descriptor.fragmentFunction = library.makeFunction(name: "floor_fragment")
        descriptor.colorAttachments[0].pixelFormat = .rgba32Float
        let pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        let whiteDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg8Unorm, width: 1, height: 1, mipmapped: false)
        whiteDescriptor.storageMode = .shared
        let white = try XCTUnwrap(device.makeTexture(descriptor: whiteDescriptor))
        var whitePixel: UInt16 = .max
        white.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: &whitePixel, bytesPerRow: 2)
        let shadowDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: 1, height: 1, mipmapped: false)
        let shadow = try XCTUnwrap(device.makeTexture(descriptor: shadowDescriptor))
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        var targets: [MTLTexture] = []
        // Calibrate the two tile radiances through the real floor fragment,
        // without duplicating its lighting or display conversion in the test.
        for sample in [SIMD4<Float>(0.5, 0.5, 0.01, 0.01), SIMD4(0.5, 1.5, 0.01, 0.01)] + samples {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: 4, height: 4, mipmapped: false)
            d.storageMode = .shared; d.usage = [.renderTarget]
            let target = try XCTUnwrap(device.makeTexture(descriptor: d))
            targets.append(target)
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = target
            pass.colorAttachments[0].loadAction = .dontCare; pass.colorAttachments[0].storeAction = .store
            let encoder = try XCTUnwrap(command.makeRenderCommandEncoder(descriptor: pass))
            var U = Uniforms(viewProj: matrix_identity_float4x4, lightDir: SIMD4(0, 0, -1, 0),
                eye: SIMD4(sample.x, sample.y, 1, 0), screen: SIMD4(4, 4, 1, 0), camRight: .zero, camUp: .zero,
                prevViewProj: matrix_identity_float4x4, temporal: .zero,
                shadowViewProj: matrix_identity_float4x4, shadowParams: .zero,
                invViewProj: matrix_identity_float4x4, prevInvViewProj: matrix_identity_float4x4,
                effects: SIMD4(1, 0, 0, 0), rayTracing: SIMD4(1, 0, 0, 0), aoProjection: .zero)
            var F = sample
            encoder.setRenderPipelineState(pipeline)
            encoder.setVertexBytes(&F, length: MemoryLayout.size(ofValue: F), index: 2)
            encoder.setFragmentBytes(&U, length: MemoryLayout.size(ofValue: U), index: 1)
            encoder.setFragmentTexture(white, index: 0); encoder.setFragmentTexture(shadow, index: 1)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            encoder.endEncoding()
        }
        command.commit(); command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed, "\(String(describing: command.error))")
        let values = targets.map { target -> Float in
            var pixel = SIMD4<Float>.zero
            target.getBytes(&pixel, bytesPerRow: 16, from: MTLRegionMake2D(1, 1, 1, 1), mipmapLevel: 0)
            return pixel.x
        }
        return values.dropFirst(2).map { ($0 - values[0]) / (values[1] - values[0]) }
    }

    func testFloorIntegratesTileBoundariesCornersAndSubpixelTiles() throws {
        var samples: [SIMD4<Float>] = []
        for width: Float in [0.01, 0.1, 0.8, 2, 3.7, 8] {
            for center: Float in [-2.001, -2, -1.999, -1, -0.001, 0, 0.001, 0.5, 0.999, 1, 1.001, 2] {
                samples.append(SIMD4(center, 0.5, width, 0.1))
                samples.append(SIMD4(center, center, width, width))
            }
        }
        let actual = try render(samples)
        let error = zip(actual, samples).map { abs($0 - checkerReference($1)) }.max()!
        XCTAssertLessThan(error, 0.001, "Pixel footprint must conserve tile coverage across wraps and corners")
    }

    func testSmallCameraMotionDoesNotPopAtCheckerWrap() throws {
        let samples = (-10...10).map { SIMD4<Float>(Float($0) * 0.0001, 0.5, 0.01, 0.01) }
        let actual = try render(samples)
        let largestChange = zip(actual, actual.dropFirst()).map { abs($0 - $1) }.max()!
        XCTAssertLessThan(largestChange, 0.011, "A 0.01-pixel movement must not toggle a tile abruptly")
        XCTAssertEqual(actual[10], 0.5, accuracy: 0.001)
    }
}
