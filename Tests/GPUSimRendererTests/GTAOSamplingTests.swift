import Metal
import XCTest
@testable import GPUSimRenderer

final class GTAOSamplingTests: XCTestCase {
    func testSpatialAndTemporalSamplesReconstructKnownVisibility() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal is unavailable") }
        let source = "#include <metal_stdlib>\nusing namespace metal;\n" + gtaoSamplingShaderSource + """
        kernel void sample_test(device float2* output [[buffer(0)]],
            uint3 p [[thread_position_in_grid]]) {
            output[p.z * 4096 + p.y * 64 + p.x] = gtaoSampleNoise(p.xy, p.z);
        }
        """
        let library = try device.makeLibrary(source: source, options: nil)
        let pipeline = try device.makeComputePipelineState(function: XCTUnwrap(library.makeFunction(name: "sample_test")))
        let frameCount = 8, size = 64
        let output = try XCTUnwrap(device.makeBuffer(
            length: size * size * frameCount * MemoryLayout<SIMD2<Float>>.stride,
            options: .storageModeShared))
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(output, offset: 0, index: 0)
        encoder.dispatchThreads(MTLSize(width: size, height: size, depth: frameCount),
            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed, "\(String(describing: command.error))")
        let samples = output.contents().assumingMemoryBound(to: SIMD2<Float>.self)

        // A rectangular visibility domain has a known integral. Treat each
        // pixel's samples as Monte Carlo estimates, then reconstruct using
        // AO's thirteen-tap kernel. This tests residual spatial clumping,
        // which a global histogram or a Hilbert-index round-trip misses.
        let expected: Float = 0.37 * 0.61
        func error(frames: Int) -> Float {
            var visibility = [Float](repeating: 0, count: size * size)
            for frame in 0..<frames {
                for pixel in 0..<(size * size) {
                    let sample = samples[frame * size * size + pixel]
                    XCTAssertTrue(sample.x >= 0 && sample.x < 1 && sample.y >= 0 && sample.y < 1)
                    if sample.x < 0.37 && sample.y < 0.61 { visibility[pixel] += 1 / Float(frames) }
                }
            }
            var squaredError: Float = 0
            for y in 2..<(size - 2) {
                for x in 2..<(size - 2) {
                    var total: Float = 0, weights: Float = 0
                    for dy in -2...2 {
                        for dx in -2...2 where abs(dx) + abs(dy) <= 2 {
                            let weight = exp(-0.5 * Float(dx * dx + dy * dy))
                            total += weight * visibility[(y + dy) * size + x + dx]
                            weights += weight
                        }
                    }
                    let delta = total / weights - expected
                    squaredError += delta * delta
                }
            }
            return sqrt(squaredError / Float((size - 4) * (size - 4)))
        }
        let firstError = error(frames: 1), accumulatedError = error(frames: frameCount)
        print("GTAO stratified visibility RMSE: frame 1 \(firstError), frame 8 \(accumulatedError)")
        XCTAssertLessThan(accumulatedError, 0.026)
        XCTAssertLessThan(accumulatedError, firstError * 0.3)
    }
}
