import Metal
import simd
import XCTest
@testable import GPUSimRenderer

final class RayTracingSamplingTests: XCTestCase {
    func testDiffuseSpatialAndTemporalStrataCoverTheHemisphereWithoutBias() throws {
        guard let device = MTLCreateSystemDefaultDevice(), device.supportsRaytracing else {
            throw XCTSkip("Metal ray tracing is unavailable")
        }
        let source = """
        kernel void diffuse_samples(device float2* output [[buffer(0)]], uint i [[thread_position_in_grid]]) {
            uint receiver = (i%256u)/16u;
            uint2 pixel = i<256u ? uint2(receiver&3u,receiver>>2u) : uint2(7,11);
            uint frame = i<256u ? 0u : receiver;
            output[i] = diffuseSample(pixel,i%16u,frame,fract(float(frame)*0.6180339887));
        }
        """
        let library = try device.makeLibrary(source: renderShaderSource+"\n"+rayTracingShaderSource+"\n"+source,options: nil)
        let pipeline = try device.makeComputePipelineState(function: XCTUnwrap(library.makeFunction(name: "diffuse_samples")))
        let output = try XCTUnwrap(device.makeBuffer(length: 512*MemoryLayout<SIMD2<Float>>.stride,options: .storageModeShared))
        let command = try XCTUnwrap(device.makeCommandQueue()?.makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline); encoder.setBuffer(output,offset: 0,index: 0)
        encoder.dispatchThreads(MTLSize(width: 512,height: 1,depth: 1),threadsPerThreadgroup: MTLSize(width: 64,height: 1,depth: 1))
        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
        XCTAssertEqual(command.status,.completed,"\(String(describing: command.error))")
        let samples = output.contents().assumingMemoryBound(to: SIMD2<Float>.self)
        for start in [0,256] {
            var cells = Set<Int>()
            for i in start..<(start+256) {
                let p = samples[i]
                XCTAssertTrue(p.x>=0 && p.x<1 && p.y>=0 && p.y<1)
                cells.insert(Int(p.x*16)+Int(p.y*16)*16)
            }
            XCTAssertEqual(cells.count,256,"Both spatial neighbors and a receiver over time must cover every fine stratum")
            var mse: Float = 0, bias: Float = 0
            // A constant-radiance region cut by a sloped visibility boundary
            // has cosine-weighted integral a+b/2 in this uniform sample domain.
            for k in 0..<64 {
                let b = Float(k)/100-0.32, a: Float = 0.43
                let estimate = Float((start..<(start+256)).filter { samples[$0].y<a+b*samples[$0].x }.count)/256
                let error = estimate-(a+b/2)
                mse += error*error; bias += error
            }
            XCTAssertLessThan(sqrt(mse/64),0.008)
            XCTAssertLessThan(abs(bias/64),0.005)
        }
        for receiver in 0..<16 {
            let rows = Set((0..<16).map { Int(samples[receiver*16+$0].x*16) })
            let columns = Set((0..<16).map { Int(samples[receiver*16+$0].y*16) })
            XCTAssertEqual(rows.count,16); XCTAssertEqual(columns.count,16)
        }
    }

    func testSparseDiffuseStratificationReducesVisibilityVarianceWithoutBias() throws {
        guard let device = MTLCreateSystemDefaultDevice(), device.supportsRaytracing else {
            throw XCTSkip("Metal ray tracing is unavailable")
        }
        let source = """
        kernel void sparse_samples(device float2* output [[buffer(0)]], uint i [[thread_position_in_grid]]) {
            uint frame = i/4u;
            uint2 pixel = uint2(frame%127u,frame/127u);
            output[i] = sparseDiffuseSample(pixel,i%4u,4u,frame);
        }
        """
        let library = try device.makeLibrary(source: renderShaderSource+"\n"+rayTracingShaderSource+"\n"+source,options: nil)
        let pipeline = try device.makeComputePipelineState(function: XCTUnwrap(library.makeFunction(name: "sparse_samples")))
        let count = 16384
        let output = try XCTUnwrap(device.makeBuffer(length: count*MemoryLayout<SIMD2<Float>>.stride,options: .storageModeShared))
        let command = try XCTUnwrap(device.makeCommandQueue()?.makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline); encoder.setBuffer(output,offset: 0,index: 0)
        encoder.dispatchThreads(MTLSize(width: count,height: 1,depth: 1),threadsPerThreadgroup: MTLSize(width: 64,height: 1,depth: 1))
        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
        XCTAssertEqual(command.status,.completed)
        let samples = output.contents().assumingMemoryBound(to: SIMD2<Float>.self)
        // Integrate a sloped occluder with known cosine-domain area 0.45.
        var bias = 0.0, mse = 0.0
        for frame in 0..<(count/4) {
            var estimate = 0.0
            for i in 0..<4 {
                let p = samples[frame*4+i]
                XCTAssertTrue(p.x>=0 && p.x<1 && p.y>=0 && p.y<1)
                XCTAssertEqual(Int(p.x*2)+2*Int(p.y*2),i)
                if p.y < 0.3+0.3*p.x { estimate += 0.25 }
            }
            bias += estimate-0.45; mse += pow(estimate-0.45,2)
        }
        XCTAssertLessThan(abs(bias/Double(count/4)),0.015)
        XCTAssertLessThan(mse/Double(count/4),0.45*0.55*0.25)
    }

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
