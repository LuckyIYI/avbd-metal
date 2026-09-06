import Foundation
import Metal
import simd
import XCTest
@testable import GPUSimRenderer

final class GTAOVisibilityIntegrationTests: XCTestCase {
    private struct Fixture {
        var normal: SIMD4<Float>
        var intervals: SIMD4<Float>

        init(_ angle: Float, _ first: ClosedRange<Float>, _ second: ClosedRange<Float> = 0...0) {
            normal = SIMD4(angle, 0, 0, 0)
            intervals = SIMD4(first.lowerBound, first.upperBound, second.lowerBound, second.upperBound)
        }
    }

    private struct Result {
        var occlusion: Float
        var openEnergy: Float
        var mask: UInt32
    }

    func testSectorIntegrationMatchesCosineQuadratureForTiltedReceivers() throws {
        let fixtures: [Fixture] = [
            Fixture(-1.45, -1.2 ... -0.4),
            Fixture(-0.85, -0.2 ... 0.5),
            Fixture(0, 0.5 ... 1.1),
            Fixture(0.75, -0.4 ... 0.3),
            Fixture(1.45, 0.4 ... 1.2),
            Fixture(1.2, -0.2 ... 0.2)
        ]
        let phases = 256
        let results = try evaluate(fixtures, phases: phases)
        for (index, fixture) in fixtures.enumerated() {
            let reference = quadrature(fixture)
            let samples = results[(index * phases)..<((index + 1) * phases)]
            let mean = samples.reduce(0.0) { $0 + Double($1.occlusion) } / Double(phases)
            XCTAssertEqual(mean, reference.occlusion, accuracy: 0.0003,
                "Sector occupancy must carry cosine energy, including near-grazing normals and intervals crossing the view direction")
            for sample in samples {
                XCTAssertEqual(Double(sample.openEnergy), reference.openEnergy, accuracy: 0.00002)
                XCTAssertLessThanOrEqual(abs(Double(sample.occlusion) - reference.occlusion),
                    reference.openEnergy / 32 + 0.00005,
                    "A single interval's error must remain within one equal-energy sector")
            }
        }
        // This fixture deliberately distinguishes cosine integration from an
        // unweighted angular popcount, even though both normalize an open arc.
        let unweighted = Double(fixtures[2].intervals.y - fixtures[2].intervals.x) / .pi
        XCTAssertGreaterThan(abs(quadrature(fixtures[2]).occlusion - unweighted), 0.05)
    }

    func testOverlappingAndDisjointBlockersIntegrateTheirUnion() throws {
        let fixtures: [Fixture] = [
            Fixture(0, -0.95 ... -0.55, 0.3 ... 0.65),
            Fixture(0.6, -0.55 ... 0.4, -0.1 ... 1.15),
            Fixture(-0.9, -1.8 ... -0.15, -1.4 ... -0.5),
            Fixture(1.4, -0.6 ... 0.3, 0.1 ... 1.0)
        ]
        let phases = 256
        let results = try evaluate(fixtures, phases: phases)
        for (index, fixture) in fixtures.enumerated() {
            let samples = results[(index * phases)..<((index + 1) * phases)]
            let mean = samples.reduce(0.0) { $0 + Double($1.occlusion) } / Double(phases)
            XCTAssertEqual(mean, quadrature(fixture).occlusion, accuracy: 0.0004,
                "Overlapping angular intervals must not count the same blocked light twice")
        }
    }

    func testEmptyAndFullHemispheresCoverBitmaskEndpointsExactly() throws {
        let fixtures: [Fixture] = [-1.5, 0, 1.5].flatMap { angle in
            [Fixture(Float(angle), 0...0), Fixture(Float(angle), -Float.pi...Float.pi)]
        }
        let phases = 32
        let results = try evaluate(fixtures, phases: phases)
        for (index, fixture) in fixtures.enumerated() {
            let expectedFull = index % 2 == 1
            for result in results[(index * phases)..<((index + 1) * phases)] {
                XCTAssertEqual(result.mask, expectedFull ? UInt32.max : 0,
                    "Empty and complete intervals must not invoke undefined shifts by 32")
                XCTAssertEqual(result.occlusion, expectedFull ? result.openEnergy : 0, accuracy: 0.000001)
            }
            if expectedFull {
                XCTAssertEqual(Double(results[index * phases].occlusion), quadrature(fixture).occlusion,
                    accuracy: 0.00002)
            }
        }
    }

    // Independently integrates the Lambertian slice using midpoint quadrature,
    // with a geometric union predicate. It does not reproduce the shader's
    // antiderivative, energy mapping, sector rounding, or bit operations.
    private func quadrature(_ fixture: Fixture) -> (occlusion: Double, openEnergy: Double) {
        let n = Double(fixture.normal.x), steps = 131_072
        let delta = Double.pi / Double(steps)
        var occlusion = 0.0, open = 0.0
        for step in 0..<steps {
            let h = n - .pi / 2 + (Double(step) + 0.5) * delta
            let energy = max(0, cos(h - n)) * abs(sin(h)) * delta
            open += energy
            let first = h >= Double(fixture.intervals.x) && h < Double(fixture.intervals.y)
            let second = h >= Double(fixture.intervals.z) && h < Double(fixture.intervals.w)
            if first || second { occlusion += energy }
        }
        return (occlusion, open)
    }

    private func evaluate(_ fixtures: [Fixture], phases: Int) throws -> [Result] {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal is unavailable") }
        let library = try device.makeLibrary(source: renderShaderSource + """
        struct VisibilityIntegralFixture { float4 normal; float4 intervals; };
        kernel void visibility_integral_fixture(
            device const VisibilityIntegralFixture* fixtures [[buffer(0)]],
            device uint4* results [[buffer(1)]], constant uint& phaseCount [[buffer(2)]],
            uint index [[thread_position_in_grid]]) {
            VisibilityIntegralFixture fixture = fixtures[index / phaseCount];
            float phase = (float(index % phaseCount) + 0.5) / float(phaseCount);
            float n = fixture.normal.x, cn = cos(n), sn = sin(n);
            float lower = n - M_PI_F * 0.5, upper = n + M_PI_F * 0.5;
            float start = gtaoArcEnergy(lower, n, cn, sn);
            float total = gtaoArcEnergy(upper, n, cn, sn) - start;
            uint mask = 0u;
            for (uint interval = 0; interval < 2; ++interval) {
                float a = clamp(fixture.intervals[interval * 2], lower, upper);
                float b = clamp(fixture.intervals[interval * 2 + 1], lower, upper);
                float lo = (gtaoArcEnergy(a, n, cn, sn) - start) / total;
                float hi = (gtaoArcEnergy(b, n, cn, sn) - start) / total;
                mask |= gtaoIntervalMask(lo, hi, phase);
            }
            float value = float(popcount(mask)) / 32.0 * total;
            results[index] = uint4(as_type<uint>(value), as_type<uint>(total), 0u, mask);
        }
        """, options: nil)
        let function = try XCTUnwrap(library.makeFunction(name: "visibility_integral_fixture"))
        let pipeline = try device.makeComputePipelineState(function: function)
        let inputs = try XCTUnwrap(fixtures.withUnsafeBytes { bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        })
        let count = fixtures.count * phases
        let output = try XCTUnwrap(device.makeBuffer(length: count * MemoryLayout<SIMD4<UInt32>>.stride,
            options: .storageModeShared))
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputs, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        var phaseCount = UInt32(phases)
        encoder.setBytes(&phaseCount, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(64, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed, "\(String(describing: command.error))")
        let values = output.contents().assumingMemoryBound(to: SIMD4<UInt32>.self)
        return (0..<count).map {
            Result(occlusion: Float(bitPattern: values[$0].x), openEnergy: Float(bitPattern: values[$0].y), mask: values[$0].w)
        }
    }
}
