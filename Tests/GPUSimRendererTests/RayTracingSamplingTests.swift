import Metal
import simd
import XCTest
@testable import GPUSimRenderer

final class RayTracingSamplingTests: XCTestCase {
    func testGGXSampleWeightsMatchIndependentHemisphereIntegration() throws {
        guard let device = MTLCreateSystemDefaultDevice(), device.supportsRaytracing else {
            throw XCTSkip("Metal ray tracing is unavailable")
        }
        let source = """
        kernel void furnace(device const float2* cases [[buffer(0)]], device float* result [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {
            uint row = i / 32768u, sampleIndex = i % 32768u;
            float alpha = cases[row].x*cases[row].x, cosine = cases[row].y;
            float3 V = float3(sqrt(1-cosine*cosine),0,cosine);
            uint bits = sampleIndex;
            bits = (bits << 16u) | (bits >> 16u);
            bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
            bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
            bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
            bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
            float2 random = float2((float(sampleIndex)+0.5)/32768.0,float(bits)*2.3283064365386963e-10);
            float3 H = rtGlossyNormal(V,alpha,random), L = reflect(-V,H);
            float lambdaV = rtSmithLambda(V.z,alpha);
            result[i] = L.z > 0 ? (1+lambdaV)/(1+lambdaV+rtSmithLambda(L.z,alpha)) : 0;
        }
        """
        let library = try device.makeLibrary(source: renderShaderSource + "\n" + rayTracingShaderSource + "\n" + source, options: nil)
        let pipeline = try device.makeComputePipelineState(function: XCTUnwrap(library.makeFunction(name: "furnace")))
        let cases: [SIMD2<Float>] = [SIMD2(0.4,0.15),SIMD2(0.4,0.5),SIMD2(0.4,1),SIMD2(0.6,0.15),SIMD2(0.6,0.5),SIMD2(0.6,1)]
        let inputs = try XCTUnwrap(cases.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) })
        let output = try XCTUnwrap(device.makeBuffer(length: cases.count*32768*4, options: .storageModeShared))
        let queue = try XCTUnwrap(device.makeCommandQueue()), command = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputs, offset: 0, index: 0); encoder.setBuffer(output, offset: 0, index: 1)
        encoder.dispatchThreads(MTLSize(width: cases.count*32768,height: 1,depth: 1),threadsPerThreadgroup: MTLSize(width: 64,height: 1,depth: 1))
        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
        XCTAssertEqual(command.status,.completed,"\(String(describing: command.error))")
        let samples = output.contents().assumingMemoryBound(to: Float.self)
        for (index, config) in cases.enumerated() {
            let alpha = Double(config.x*config.x), nv = Double(config.y)
            let view = SIMD3<Double>(sqrt(1-nv*nv),0,nv)
            func lambda(_ cosine: Double) -> Double { (sqrt(1+alpha*alpha*(1-cosine*cosine)/(cosine*cosine))-1)/2 }
            var integral = 0.0
            // Integrate f*cos over a uniform solid-angle grid, independently
            // of the production sampling transform and its simplified weight.
            for y in 0..<256 { for x in 0..<512 {
                let nl = (Double(y)+0.5)/256, phi = (Double(x)+0.5)/512 * 2 * Double.pi
                let light = SIMD3(sqrt(1-nl*nl)*cos(phi),sqrt(1-nl*nl)*sin(phi),nl)
                let half = normalize(view+light)
                let denominator = half.z*half.z*(alpha*alpha-1)+1
                let distribution = alpha*alpha / (Double.pi*denominator*denominator)
                let masking = 1 / (1+lambda(nv)+lambda(nl))
                integral += distribution*masking/(4*nv) * (2*Double.pi/(256*512))
            } }
            let mean = (0..<32768).reduce(0.0) { $0+Double(samples[index*32768+$1])/32768 }
            XCTAssertTrue(mean.isFinite)
            XCTAssertGreaterThan(mean,0)
            XCTAssertLessThanOrEqual(mean,1.0001,"a white furnace cannot gain energy")
            XCTAssertEqual(mean,integral,accuracy: 0.012,"roughness \(config.x), NdotV \(config.y)")
        }
    }
}
