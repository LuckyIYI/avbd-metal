import Foundation
import CoreGraphics
import ImageIO
import Metal
import simd
import XCTest
@testable import GPUSimRenderer

/// Execute the shipped shader, including depth reconstruction and the real
/// R8/RGBA8 denoising targets. A CPU port with ideal positions misses sampling
/// and precision errors in the production G-buffer path.
final class GTAOMetalTests: XCTestCase {
    private struct Fixture {
        var plane: SIMD4<Float>
        var boxMin = SIMD4<Float>(0, 0, 0, 0)
        var boxMax = SIMD4<Float>(0, 0, 0, 0)
    }

    private let fixtureShader = """
    struct AOFixture { float4 plane; float4 boxMin; float4 boxMax; };
    struct FixtureOut { float4 normal [[color(0)]]; float depth [[depth(any)]]; };
    fragment FixtureOut ao_fixture(FSOut in [[stage_in]],
        constant Uniforms& U [[buffer(1)]], constant AOFixture& F [[buffer(2)]]) {
        float3 ray = normalize(worldFromDepth(in.uv, 0.0, U.invViewProj) - U.eye.xyz);
        float denom = dot(F.plane.xyz, ray);
        float t = (F.plane.w - dot(F.plane.xyz, U.eye.xyz)) / denom;
        float3 N = F.plane.xyz;
        if (t <= 0.0) t = 1e20;
        if (F.boxMin.w > 0.0) {
            float3 a = (F.boxMin.xyz - U.eye.xyz) / ray;
            float3 b = (F.boxMax.xyz - U.eye.xyz) / ray;
            float3 lo = min(a, b), hi = max(a, b);
            float entry = max(lo.x, max(lo.y, lo.z));
            float leave = min(hi.x, min(hi.y, hi.z));
            if (entry > 0.0 && entry < leave && entry < t) {
                t = entry;
                N = float3(0);
                uint axis = lo.x >= lo.y && lo.x >= lo.z ? 0 : (lo.y >= lo.z ? 1 : 2);
                N[axis] = ray[axis] > 0.0 ? -1.0 : 1.0;
            }
        }
        FixtureOut out;
        float4 clip = U.viewProj * float4(U.eye.xyz + ray * t, 1);
        // Match raster near/far clipping. Writing an out-of-frustum depth
        // clamps it to 0/1 and invents geometry on the clip plane instead.
        out.depth = t < 1e19 && clip.z >= 0.0 && clip.z < clip.w ? clip.z / clip.w : 1.0;
        out.normal = float4(N, 0);
        return out;
    }
    """

    private struct Result {
        var raw: [Float]
        var resolved: [Float]
        var milliseconds: Double
    }

    private func uniforms(width: Int, height: Int, eye: SIMD3<Float> = .zero) -> Uniforms {
        let y: Float = 1 / tan(25 * .pi / 180)
        let near: Float = 0.1, far: Float = 1000
        let projection = simd_float4x4(columns: (
            SIMD4(y * Float(height) / Float(width), 0, 0, 0),
            SIMD4(0, y, 0, 0),
            SIMD4(0, 0, far / (near - far), -1),
            SIMD4(0, 0, near * far / (near - far), 0)))
        var view = matrix_identity_float4x4
        view.columns.3 = SIMD4(-eye, 1)
        let vp = projection * view
        return Uniforms(viewProj: vp, lightDir: .zero, eye: SIMD4(eye, 0),
            screen: SIMD4(Float(width), Float(height), Float(height) * y * 0.5, 0),
            camRight: SIMD4(1, 0, 0, 0), camUp: SIMD4(0, -1, 0, 0),
            prevViewProj: vp, temporal: SIMD4(0, 1, 0, 0),
            shadowViewProj: matrix_identity_float4x4, shadowParams: .zero,
            invViewProj: vp.inverse, prevInvViewProj: vp.inverse,
            aoProjection: SIMD4(-projection.columns.2.z, projection.columns.3.z,
                                1 / projection.columns.0.x, 1 / projection.columns.1.y))
    }

    private func render(_ fixture: Fixture, width: Int = 256, height: Int = 192,
                        frames: Int = 32, eye: SIMD3<Float> = .zero) throws -> Result {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal is unavailable") }
        // Opt-in A/B capture uses a saved Swift shader source; normal tests
        // always compile the exact source passed to the production renderer.
        let shader: String
        if let path = ProcessInfo.processInfo.environment["GTAO_TEST_SHADER"] {
            shader = try String(contentsOfFile: path).components(separatedBy: "\"\"\"")[1]
        } else { shader = renderShaderSource }
        let library = try device.makeLibrary(source: shader + fixtureShader, options: nil)
        func pipeline(_ fragment: String, _ format: MTLPixelFormat,
                      depth: Bool = false) throws -> MTLRenderPipelineState {
            let d = MTLRenderPipelineDescriptor()
            d.vertexFunction = library.makeFunction(name: "fs_vertex")
            d.fragmentFunction = library.makeFunction(name: fragment)
            d.colorAttachments[0].pixelFormat = format
            if depth { d.depthAttachmentPixelFormat = .depth32Float }
            return try device.makeRenderPipelineState(descriptor: d)
        }
        func texture(_ format: MTLPixelFormat) throws -> MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: format, width: width, height: height, mipmapped: false)
            d.storageMode = .private
            d.usage = [.renderTarget, .shaderRead]
            return try XCTUnwrap(device.makeTexture(descriptor: d))
        }
        let fixtureP = try pipeline("ao_fixture", .rgba16Float, depth: true)
        let aoP = try pipeline("gtao_fragment", .r8Unorm)
        let temporalP = try pipeline("temporal_fragment", .rgba8Unorm)
        let historical = ProcessInfo.processInfo.environment["GTAO_TEST_SHADER"] != nil
        let blurP = try pipeline(historical ? "blur_fragment" : "visibility_fragment", historical ? .r8Unorm : .rg8Unorm)
        let depth = try texture(.depth32Float), normal = try texture(.rgba16Float)
        let raw = try texture(.r8Unorm), history = try texture(.rgba8Unorm)
        let resolved = try texture(.rgba8Unorm), blurred = try texture(historical ? .r8Unorm : .rg8Unorm)
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let depthDesc = MTLDepthStencilDescriptor()
        depthDesc.isDepthWriteEnabled = true
        depthDesc.depthCompareFunction = .always
        let depthState = try XCTUnwrap(device.makeDepthStencilState(descriptor: depthDesc))
        var U = uniforms(width: width, height: height, eye: eye)
        var F = fixture
        var gpuTime: Double = 0
        for frame in 0..<frames {
            U.temporal = SIMD4((Float(frame % 64) * 0.6180339887).truncatingRemainder(dividingBy: 1),
                              frame == 0 ? 1 : 0.2, 0, 0)
            let cmd = try XCTUnwrap(queue.makeCommandBuffer())
            func pass(_ p: MTLRenderPipelineState, _ target: MTLTexture,
                      textures: [MTLTexture] = [], prepass: Bool = false) throws {
                let d = MTLRenderPassDescriptor()
                d.colorAttachments[0].texture = target
                d.colorAttachments[0].loadAction = .dontCare
                d.colorAttachments[0].storeAction = .store
                if prepass {
                    d.depthAttachment.texture = depth
                    d.depthAttachment.loadAction = .clear
                    d.depthAttachment.clearDepth = 1
                    d.depthAttachment.storeAction = .store
                }
                let enc = try XCTUnwrap(cmd.makeRenderCommandEncoder(descriptor: d))
                enc.setRenderPipelineState(p)
                if prepass {
                    enc.setDepthStencilState(depthState)
                    enc.setFragmentBytes(&F, length: MemoryLayout<Fixture>.stride, index: 2)
                }
                enc.setFragmentBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
                for (index, tex) in textures.enumerated() { enc.setFragmentTexture(tex, index: index) }
                enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
                enc.endEncoding()
            }
            if frame == 0 { try pass(fixtureP, normal, prepass: true) }
            try pass(aoP, raw, textures: [depth, normal])
            try pass(temporalP, resolved, textures: [raw, history, depth, depth, normal])
            let blit = try XCTUnwrap(cmd.makeBlitCommandEncoder())
            blit.copy(from: resolved, to: history)
            blit.endEncoding()
            try pass(blurP, blurred, textures: historical ? [resolved, depth, normal] : [resolved, raw, depth, normal])
            cmd.commit()
            cmd.waitUntilCompleted()
            XCTAssertEqual(cmd.status, .completed, "\(String(describing: cmd.error))")
            if frame >= 8 { gpuTime += cmd.gpuEndTime - cmd.gpuStartTime }
        }
        func read(_ tex: MTLTexture) throws -> [Float] {
            let channels = tex.pixelFormat == .rg8Unorm ? 2 : 1
            let stride = (width * channels + 255) / 256 * 256
            let buffer = try XCTUnwrap(device.makeBuffer(length: stride * height, options: .storageModeShared))
            let cmd = try XCTUnwrap(queue.makeCommandBuffer())
            let enc = try XCTUnwrap(cmd.makeBlitCommandEncoder())
            enc.copy(from: tex, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(),
                sourceSize: MTLSize(width: width, height: height, depth: 1), to: buffer,
                destinationOffset: 0, destinationBytesPerRow: stride, destinationBytesPerImage: stride * height)
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
            let bytes = buffer.contents().assumingMemoryBound(to: UInt8.self)
            return (0..<height).flatMap { y in (0..<width).map { x in Float(bytes[y * stride + x * channels]) / 255 } }
        }
        return try Result(raw: read(raw), resolved: read(blurred),
                          milliseconds: gpuTime * 1000 / Double(max(frames - 8, 1)))
    }

    func testUnoccludedTiltedPlanesStayUnoccluded() throws {
        for z: Float in [1, 0.5, 0.2, 0.08] {
            let normal = simd_normalize(SIMD3<Float>(sqrt(1 - z * z) * 0.8,
                                                    sqrt(1 - z * z) * 0.6, z))
            let result = try render(Fixture(plane: SIMD4(normal, -3 * normal.z)))
            // Include the entire viewport: clamped off-screen depth must not
            // fabricate occluders at borders, either.
            let values = result.resolved.sorted()
            let mean = values.reduce(0, +) / Float(values.count)
            let p01 = values[values.count / 100]
            print("GTAO plane z=\(z): mean=\(mean) p01=\(p01) min=\(values[0]) GPU=\(result.milliseconds)ms")
            XCTAssertGreaterThan(mean, 0.99)
            XCTAssertGreaterThan(p01, 0.97)
            XCTAssertGreaterThan(try XCTUnwrap(result.raw.min()), 0.97,
                                 "the raw estimator must be correct before denoising")
            try save(result.resolved, width: 256, height: 192, name: "plane-\(z)")
        }
    }

    /// An independent cosine-weighted ray integral over actual geometry,
    /// rather than another implementation of the horizon-search equations.
    private func rayVisibility(point: SIMD3<Float>, fixture: Fixture, samples: Int = 16384) -> Float {
        let n = simd_normalize(SIMD3<Double>(Double(fixture.plane.x), Double(fixture.plane.y), Double(fixture.plane.z)))
        let tangent = simd_normalize(simd_cross(n, abs(n.z) < 0.9 ? SIMD3(0, 0, 1) : SIMD3(0, 1, 0)))
        let bitangent = simd_cross(n, tangent)
        let origin = SIMD3<Double>(Double(point.x), Double(point.y), Double(point.z)) + n * 1e-6
        let bmin = SIMD3<Double>(Double(fixture.boxMin.x), Double(fixture.boxMin.y), Double(fixture.boxMin.z))
        let bmax = SIMD3<Double>(Double(fixture.boxMax.x), Double(fixture.boxMax.y), Double(fixture.boxMax.z))
        var visibility = 0.0
        for index in 0..<samples {
            let u = (Double(index) + 0.5) / Double(samples)
            let phi = 2 * Double.pi * (Double(index) * 0.6180339887498949).truncatingRemainder(dividingBy: 1)
            let tx: Double = sqrt(u) * cos(phi)
            let ty: Double = sqrt(u) * sin(phi)
            let nz: Double = sqrt(1 - u)
            let ray = tangent * tx + bitangent * ty + n * nz
            let a = (bmin - origin) / ray, b = (bmax - origin) / ray
            let lo = simd_min(a, b), hi = simd_max(a, b)
            let entry = max(lo.x, max(lo.y, lo.z)), leave = min(hi.x, min(hi.y, hi.z))
            let weight = entry > 0 && entry < leave ? max(0, min(1, (0.9 - entry) / (0.9 * 0.65))) : 0
            visibility += 1 - weight
        }
        return Float(visibility / Double(samples))
    }

    func testBoxContactMatchesRayTracedVisibility() throws {
        let fixture = Fixture(plane: SIMD4(0, 0, 1, -3),
                              boxMin: SIMD4(-0.2, -0.35, -3, 1),
                              boxMax: SIMD4(0.2, 0.35, -2.8, 1))
        let w = 384, h = 288
        // See the box side above the probes: a centered camera hides that
        // entire wall behind its top, which screen-space AO cannot recover.
        let eye = SIMD3<Float>(1.2, 0, 0)
        let result = try render(fixture, width: w, height: h, frames: 64, eye: eye)
        let U = uniforms(width: w, height: h, eye: eye)
        var nearVisibility: Float = 1
        for x: Float in [0.27, 0.4, 0.6, 1.2] {
            let requested = SIMD3<Float>(x, 0, -3)
            let clip = U.viewProj * SIMD4(requested, 1)
            let px = Int((clip.x / clip.w * 0.5 + 0.5) * Float(w))
            let py = h / 2
            let ray4 = U.invViewProj * SIMD4((Float(px) + 0.5) / Float(w) * 2 - 1,
                1 - (Float(py) + 0.5) / Float(h) * 2, 0, 1)
            let ray = SIMD3(ray4.x, ray4.y, ray4.z) / ray4.w - eye
            let point = eye + ray * (-3 / ray.z)
            let reference = rayVisibility(point: point, fixture: fixture)
            var ao: Float = 0
            for dy in -2...2 {
                for dx in -2...2 { ao += result.resolved[(py + dy) * w + px + dx] / 25 }
            }
            // Undo the renderer's artistic power curve for comparison with
            // physical cosine-weighted visibility.
            let measured = pow(ao, 1 / 1.25)
            print("GTAO contact x=\(x): ray=\(reference) GPU=\(measured)")
            // Finite-radius GTAO attenuates a sampled horizon rather than
            // tracing every hemisphere direction through unseen geometry.
            XCTAssertEqual(measured, reference, accuracy: 0.13)
            if x == 0.27 { nearVisibility = measured }
            if x == 1.2 {
                XCTAssertGreaterThan(measured, 0.98)
                XCTAssertLessThan(nearVisibility, 0.85, "contact shadow must survive the planar correction")
            }
        }
        try save(result.resolved, width: w, height: h, name: "box-contact")
    }

    func testPlanarDepthPrecisionAtDifferentDistancesAndWorldOrigins() throws {
        let n = simd_normalize(SIMD3<Float>(0.8, -0.4, 0.2))
        for eye in [SIMD3<Float>.zero, SIMD3<Float>(20, -30, 5)] {
            for distance: Float in [0.4, 3, 12] {
                let center = eye + SIMD3<Float>(0, 0, -distance)
                let fixture = Fixture(plane: SIMD4(n, simd_dot(n, center)))
                let result = try render(fixture, width: 255, height: 191, frames: 1, eye: eye)
                XCTAssertGreaterThan(try XCTUnwrap(result.raw.min()), 0.97,
                    "raw AO at distance \(distance), origin \(eye) must not self-occlude")
            }
        }
    }

    func testOccludedPlaneHasNoDirectionalNoiseBands() throws {
        // A wall invariant along Y makes true AO constant down each column.
        // The old shared IGN angle/radius seed leaves a repeated diagonal
        // pattern after both temporal accumulation and spatial filtering.
        let fixture = Fixture(plane: SIMD4(0, 0, 1, -3),
            boxMin: SIMD4(-10, -100, -3, 1), boxMax: SIMD4(0.2, 100, -2.7, 1))
        let w = 384, h = 288
        let result = try render(fixture, width: w, height: h, frames: 32, eye: SIMD3(1.2, 0, 0))
        let firstY = 32, lastY = h - 32
        var means = [Float](repeating: 0, count: w)
        for x in 0..<w {
            for y in firstY..<lastY { means[x] += result.resolved[y * w + x] / Float(lastY - firstY) }
        }
        let columns = (0..<w).filter { means[$0] > 0.65 && means[$0] < 0.95 }
        XCTAssertGreaterThan(columns.count, 8)
        var peak: Float = 0
        for dy in 4...12 {
            for dx in -12...12 {
                var product: Float = 0, energyA: Float = 0, energyB: Float = 0
                for x in columns where x + dx >= 0 && x + dx < w {
                    for y in firstY..<(lastY - dy) {
                        let a = result.resolved[y * w + x] - means[x]
                        let b = result.resolved[(y + dy) * w + x + dx] - means[x + dx]
                        product += a * b; energyA += a * a; energyB += b * b
                    }
                }
                if energyA * energyB > 1e-12 { peak = max(peak, abs(product) / sqrt(energyA * energyB)) }
            }
        }
        print("GTAO directional residual correlation=\(peak), columns=\(columns.count)")
        XCTAssertLessThan(peak, 0.5)
        try save(result.resolved, width: w, height: h, name: "wall-noise")
    }

    func testAOChainBenchmark() throws {
        guard ProcessInfo.processInfo.environment["GTAO_TEST_BENCHMARK"] == "1" else {
            throw XCTSkip("opt-in GPU timing; run shader A/B serially on an idle GPU")
        }
        print("GTAO benchmark device: \(MTLCreateSystemDefaultDevice()?.name ?? "unavailable")")
        let fixture = Fixture(plane: SIMD4(0, 0, 1, -3),
            boxMin: SIMD4(-10, -100, -3, 1), boxMax: SIMD4(0.2, 100, -2.7, 1))
        var times = [Double]()
        for _ in 0..<3 {
            let result = try render(fixture, width: 1440, height: 858,
                                    frames: 128, eye: SIMD3(1.2, 0, 0))
            times.append(result.milliseconds)
        }
        print("GTAO chain 1440x858 (24 depth taps, temporal, history copy, blur): \(times) ms; median=\(times.sorted()[1]) ms")
    }

    private func save(_ values: [Float], width: Int, height: Int, name: String) throws {
        guard let directory = ProcessInfo.processInfo.environment["GTAO_TEST_OUTPUT"] else { return }
        try FileManager.default.createDirectory(atPath: directory, withIntermediateDirectories: true)
        let bytes = values.map { UInt8(clamping: Int(($0 * 255).rounded())) }
        let provider = try XCTUnwrap(CGDataProvider(data: Data(bytes) as CFData))
        let image = try XCTUnwrap(CGImage(width: width, height: height, bitsPerComponent: 8,
            bitsPerPixel: 8, bytesPerRow: width, space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGBitmapInfo(rawValue: 0), provider: provider, decode: nil,
            shouldInterpolate: false, intent: .defaultIntent))
        let url = URL(fileURLWithPath: directory).appendingPathComponent(name + ".png")
        let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
    }
}
