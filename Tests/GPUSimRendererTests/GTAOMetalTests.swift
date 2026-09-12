import Foundation
import CoreGraphics
import ImageIO
import Metal
import simd
import XCTest
@testable import GPUSimRenderer

/// Execute the shipped shader, including depth reconstruction and the real
/// R8 AO and bilateral filtering targets. A CPU port with ideal positions misses sampling
/// and precision errors in the production G-buffer path.
@MainActor
final class GTAOMetalTests: XCTestCase {
    private struct Fixture {
        var plane: SIMD4<Float>
        var boxMin = SIMD4<Float>(0, 0, 0, 0)
        var boxMax = SIMD4<Float>(0, 0, 0, 0)
        var boxOrientation = SIMD4<Float>(1, 0, 0, 0)
    }

    private let fixtureShader = """
    struct AOFixture { float4 plane; float4 boxMin; float4 boxMax; float4 boxOrientation; };
    struct FixtureOut { float4 normal [[color(0)]]; float depth [[depth(any)]]; };
    fragment FixtureOut ao_fixture(FSOut in [[stage_in]],
        constant Uniforms& U [[buffer(1)]], constant AOFixture& F [[buffer(2)]]) {
        float3 ray = normalize(worldFromDepth(in.uv, 0.0, U.invViewProj) - U.eye.xyz);
        float denom = dot(F.plane.xyz, ray);
        float t = (F.plane.w - dot(F.plane.xyz, U.eye.xyz)) / denom;
        float3 N = F.plane.xyz;
        if (t <= 0.0) t = 1e20;
        if (F.boxMin.w > 0.0) {
            float2 X = F.boxOrientation.xy, Y = float2(-X.y, X.x);
            float3 origin = float3(dot(U.eye.xy, X), dot(U.eye.xy, Y), U.eye.z);
            float3 direction = float3(dot(ray.xy, X), dot(ray.xy, Y), ray.z);
            float3 a = (F.boxMin.xyz - origin) / direction;
            float3 b = (F.boxMax.xyz - origin) / direction;
            float3 lo = min(a, b), hi = max(a, b);
            float entry = max(lo.x, max(lo.y, lo.z));
            float leave = min(hi.x, min(hi.y, hi.z));
            if (entry > 0.0 && entry < leave && entry < t) {
                t = entry;
                N = float3(0);
                uint axis = lo.x >= lo.y && lo.x >= lo.z ? 0 : (lo.y >= lo.z ? 1 : 2);
                N[axis] = direction[axis] > 0.0 ? -1.0 : 1.0;
                N.xy = X * N.x + Y * N.y;
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
    struct FixtureBackOut { float depth [[depth(any)]]; };
    fragment FixtureBackOut ao_fixture_back(FSOut in [[stage_in]],
        constant Uniforms& U [[buffer(1)]], constant AOFixture& F [[buffer(2)]]) {
        FixtureBackOut out;
        out.depth = 1.0;
        if (F.boxMin.w <= 0.0) return out;
        float3 ray = normalize(worldFromDepth(in.uv, 0.0, U.invViewProj) - U.eye.xyz);
        float2 X = F.boxOrientation.xy, Y = float2(-X.y, X.x);
        float3 origin = float3(dot(U.eye.xy, X), dot(U.eye.xy, Y), U.eye.z);
        float3 direction = float3(dot(ray.xy, X), dot(ray.xy, Y), ray.z);
        float3 a = (F.boxMin.xyz - origin) / direction;
        float3 b = (F.boxMax.xyz - origin) / direction;
        float3 lo = min(a, b), hi = max(a, b);
        float entry = max(lo.x, max(lo.y, lo.z));
        float leave = min(hi.x, min(hi.y, hi.z));
        if (leave > max(entry, 0.0)) {
            float4 clip = U.viewProj * float4(U.eye.xyz + ray * leave, 1);
            if (clip.z >= 0.0 && clip.z < clip.w) out.depth = clip.z / clip.w;
        }
        // The open receiver plane has no backface. A box's exit is captured
        // independently of front visibility, like the real backface pass.
        return out;
    }
    """

    private struct Result {
        var raw: [Float]
        var resolved: [Float]
        var preciseResolved: [Float]?
        var snapshots: [Int: [Float]]
        var rawSnapshots: [Int: [Float]]
        var milliseconds: Double
    }

    private func uniforms(width: Int, height: Int, eye: SIMD3<Float> = .zero,
                          sceneScale: Float = 1) -> Uniforms {
        let y: Float = 1 / tan(25 * .pi / 180)
        let near: Float = 0.1 * sceneScale, far: Float = 1000 * sceneScale
        let projection = simd_float4x4(columns: (
            SIMD4(y * Float(height) / Float(width), 0, 0, 0),
            SIMD4(0, y, 0, 0),
            SIMD4(0, 0, far / (near - far), -1),
            SIMD4(0, 0, near * far / (near - far), 0)))
        var view = matrix_identity_float4x4
        view.columns.3 = SIMD4(-eye, 1)
        let vp = projection * view
        return Uniforms(viewProj: vp, lightDir: .zero, eye: SIMD4(eye, 0),
            screen: SIMD4(Float(width), Float(height), Float(height) * y * 0.5, sceneScale),
            camRight: SIMD4(1, 0, 0, 0), camUp: SIMD4(0, -1, 0, 0),
            prevViewProj: vp, temporal: SIMD4(0, 1, 0, 0),
            shadowViewProj: matrix_identity_float4x4, shadowParams: .zero,
            invViewProj: vp.inverse, prevInvViewProj: vp.inverse,
            aoProjection: SIMD4(-projection.columns.2.z, projection.columns.3.z,
                                1 / projection.columns.0.x, 1 / projection.columns.1.y))
    }

    private func render(_ fixture: Fixture, width: Int = 256, height: Int = 192,
                        frames: Int = 1, eye: SIMD3<Float> = .zero, finalFixture: Fixture? = nil,
                        captureFrames: Set<Int> = [],
                        camera: ((Int) -> SIMD3<Float>)? = nil,
                        capturePreciseResolve: Bool = false, sceneScale: Float = 1) throws -> Result {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal is unavailable") }
        // Exercise the shader library and pass chain used by the renderer.
        let library = try device.makeLibrary(source: renderShaderSource + fixtureShader, options: nil)
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
        let backP = try pipeline("ao_fixture_back", .invalid, depth: true)
        let preciseVisibilityP = try capturePreciseResolve ? pipeline("visibility_fragment", .rg32Float) : nil
        let preciseVisibility = try capturePreciseResolve ? texture(.rg32Float) : nil
        let effects = try ScreenSpacePipeline(device: device, library: library)
        let options = GPUSimRenderOptions(ambientOcclusion: true, contactShadows: false)
        try effects.prepare(size: CGSize(width: width * 2, height: height * 2), options: options)
        let depth = try XCTUnwrap(effects.depth)
        let backDepth = try XCTUnwrap(effects.aoBackDepth)
        let normal = try XCTUnwrap(effects.normal)
        let raw = try XCTUnwrap(effects.aoRaw)
        let spatial = try XCTUnwrap(effects.aoSpatial)
        let visibility = try XCTUnwrap(effects.visibility)
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let depthDesc = MTLDepthStencilDescriptor()
        depthDesc.isDepthWriteEnabled = true
        depthDesc.depthCompareFunction = .always
        let depthState = try XCTUnwrap(device.makeDepthStencilState(descriptor: depthDesc))
        var U = uniforms(width: width, height: height, eye: eye, sceneScale: sceneScale)
        var F = fixture
        var gpuTime: Double = 0
        var previousEye = camera?(0) ?? eye
        var snapshots = [Int: [Float]](), rawSnapshots = [Int: [Float]]()
        for frame in 0..<frames {
            let currentEye = camera?(frame) ?? eye
            let movedCamera = currentEye != previousEye
            let previousVP = U.viewProj
            U = uniforms(width: width, height: height, eye: currentEye, sceneScale: sceneScale)
            U.prevViewProj = frame == 0 ? U.viewProj : previousVP
            U.prevInvViewProj = U.prevViewProj.inverse
            U.temporal = SIMD4((Float(frame % 64) * 0.6180339887).truncatingRemainder(dividingBy: 1),
                              frame == 0 ? 1 : 0.2, Float(frame % 64), movedCamera ? 0 : 1)
            let cmd = try XCTUnwrap(queue.makeCommandBuffer())
            func pass(_ p: MTLRenderPipelineState, _ target: MTLTexture,
                      textures: [MTLTexture] = [], prepass: Bool = false,
                      backDepthPass: Bool = false) throws {
                let d = MTLRenderPassDescriptor()
                if !backDepthPass {
                    d.colorAttachments[0].texture = target
                    d.colorAttachments[0].loadAction = .dontCare
                    d.colorAttachments[0].storeAction = .store
                }
                if prepass || backDepthPass {
                    d.depthAttachment.texture = backDepthPass ? backDepth : depth
                    d.depthAttachment.loadAction = .clear
                    d.depthAttachment.clearDepth = 1
                    d.depthAttachment.storeAction = .store
                }
                let enc = try XCTUnwrap(cmd.makeRenderCommandEncoder(descriptor: d))
                enc.setRenderPipelineState(p)
                if prepass || backDepthPass {
                    enc.setDepthStencilState(depthState)
                    enc.setFragmentBytes(&F, length: MemoryLayout<Fixture>.stride, index: 2)
                }
                enc.setFragmentBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
                for (index, tex) in textures.enumerated() { enc.setFragmentTexture(tex, index: index) }
                enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
                enc.endEncoding()
            }
            if frame == frames - 1, let finalFixture { F = finalFixture }
            if frame == 0 || movedCamera || (frame == frames - 1 && finalFixture != nil) {
                try pass(fixtureP, normal, prepass: true)
                try pass(backP, backDepth, backDepthPass: true)
            }
            try effects.encodeBeforeLighting(command: cmd, uniforms: U, options: options)
            if frame == frames - 1, let preciseVisibilityP, let preciseVisibility {
                // Execute the same spatial shader with an FP32 target so
                // diagnostics can distinguish signal from R8 quantization.
                try pass(preciseVisibilityP, preciseVisibility, textures: [spatial, raw, depth, normal])
            }
            cmd.commit()
            cmd.waitUntilCompleted()
            XCTAssertEqual(cmd.status, .completed, "\(String(describing: cmd.error))")
            if frame >= 8 { gpuTime += cmd.gpuEndTime - cmd.gpuStartTime }
            if captureFrames.contains(frame + 1) {
                snapshots[frame + 1] = try read(visibility)
                rawSnapshots[frame + 1] = try read(raw)
            }
            previousEye = currentEye
        }
        func read(_ tex: MTLTexture) throws -> [Float] {
            let floatPixels = tex.pixelFormat == .rg32Float
            let bytesPerPixel = floatPixels ? 8 : (tex.pixelFormat == .rg8Unorm ? 2 : 1)
            let stride = (width * bytesPerPixel + 255) / 256 * 256
            let buffer = try XCTUnwrap(device.makeBuffer(length: stride * height, options: .storageModeShared))
            let cmd = try XCTUnwrap(queue.makeCommandBuffer())
            let enc = try XCTUnwrap(cmd.makeBlitCommandEncoder())
            enc.copy(from: tex, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(),
                sourceSize: MTLSize(width: width, height: height, depth: 1), to: buffer,
                destinationOffset: 0, destinationBytesPerRow: stride, destinationBytesPerImage: stride * height)
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
            if floatPixels {
                return (0..<height).flatMap { y in
                    let row = buffer.contents().advanced(by: y * stride).assumingMemoryBound(to: Float.self)
                    return (0..<width).map { x in row[x * 2] }
                }
            }
            let bytes = buffer.contents().assumingMemoryBound(to: UInt8.self)
            return (0..<height).flatMap { y in (0..<width).map { x in Float(bytes[y * stride + x * bytesPerPixel]) / 255 } }
        }
        return try Result(raw: read(raw), resolved: read(visibility),
                          preciseResolved: preciseVisibility.map { try read($0) },
                          snapshots: snapshots, rawSnapshots: rawSnapshots,
                          milliseconds: gpuTime * 1000 / Double(max(frames - 8, 1)))
    }

    func testSmallSceneKeepsEquivalentContactShading() throws {
        // Equivalent camera and geometry in metres and centimetres must have
        // the same screen-space contact shadow, including the raw estimator.
        let fixture = Fixture(plane: SIMD4(0, 0, 1, -3),
                              boxMin: SIMD4(-0.2, -0.35, -3, 1),
                              boxMax: SIMD4(0.2, 0.35, -2.8, 1))
        let eye = SIMD3<Float>(1.2, 0, 0)
        let reference = try render(fixture, eye: eye)
        let scale: Float = 0.01
        var small = fixture
        small.plane.w *= scale
        small.boxMin *= SIMD4(scale, scale, scale, 1)
        small.boxMax *= SIMD4(scale, scale, scale, 1)
        let result = try render(small, eye: eye * scale, sceneScale: scale)
        XCTAssertLessThan(try XCTUnwrap(reference.raw.min()), 0.9,
                          "the fixture must exercise real contact occlusion")
        for (a, b) in [(reference.raw, result.raw), (reference.resolved, result.resolved)] {
            let errors = zip(a, b).map { abs($0 - $1) }.sorted()
            XCTAssertLessThan(errors.reduce(0, +) / Float(errors.count), 0.006)
            XCTAssertLessThan(errors[errors.count * 99 / 100], 0.025)
        }
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
    private func rayVisibility(point: SIMD3<Float>, fixture: Fixture, samples: Int = 16384,
                               radius: Double = 0.9) -> Float {
        let n = simd_normalize(SIMD3<Double>(Double(fixture.plane.x), Double(fixture.plane.y), Double(fixture.plane.z)))
        let tangent = simd_normalize(simd_cross(n, abs(n.z) < 0.9 ? SIMD3(0, 0, 1) : SIMD3(0, 1, 0)))
        let bitangent = simd_cross(n, tangent)
        let worldOrigin = SIMD3<Double>(Double(point.x), Double(point.y), Double(point.z)) + n * 1e-6
        let c = Double(fixture.boxOrientation.x), s = Double(fixture.boxOrientation.y)
        let origin = SIMD3(c * worldOrigin.x + s * worldOrigin.y,
                           -s * worldOrigin.x + c * worldOrigin.y, worldOrigin.z)
        let bmin = SIMD3<Double>(Double(fixture.boxMin.x), Double(fixture.boxMin.y), Double(fixture.boxMin.z))
        let bmax = SIMD3<Double>(Double(fixture.boxMax.x), Double(fixture.boxMax.y), Double(fixture.boxMax.z))
        var visibility = 0.0
        for index in 0..<samples {
            let u = (Double(index) + 0.5) / Double(samples)
            let phi = 2 * Double.pi * (Double(index) * 0.6180339887498949).truncatingRemainder(dividingBy: 1)
            let tx: Double = sqrt(u) * cos(phi)
            let ty: Double = sqrt(u) * sin(phi)
            let nz: Double = sqrt(1 - u)
            let worldRay = tangent * tx + bitangent * ty + n * nz
            let ray = SIMD3(c * worldRay.x + s * worldRay.y, -s * worldRay.x + c * worldRay.y, worldRay.z)
            let a = (bmin - origin) / ray, b = (bmax - origin) / ray
            let lo = simd_min(a, b), hi = simd_max(a, b)
            let entry = max(lo.x, max(lo.y, lo.z)), leave = min(hi.x, min(hi.y, hi.z))
            let weight = entry > 0 && entry < leave ? max(0, min(1, (radius - entry) / (radius * 0.65))) : 0
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
        let result = try render(fixture, width: w, height: h, frames: 1, eye: eye)
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
            // The screen-space integral uses sparse finite-depth samples
            // rather than tracing every direction through complete geometry.
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
        // Spatial reconstruction must remove directional sampling patterns
        // in one frame without smoothing away the contact gradient.
        let fixture = Fixture(plane: SIMD4(0, 0, 1, -3),
            boxMin: SIMD4(-10, -100, -3, 1), boxMax: SIMD4(0.2, 100, -2.7, 1))
        let w = 384, h = 288
        let result = try render(fixture, width: w, height: h, frames: 1, eye: SIMD3(1.2, 0, 0),
                                capturePreciseResolve: true)
        let values = try XCTUnwrap(result.preciseResolved)
        let firstY = 32, lastY = h - 32
        func columnMeans(_ input: [Float]) -> [Float] {
            (0..<w).map { x in
                Float((firstY..<lastY).reduce(0.0) { $0 + Double(input[$1 * w + x]) } / Double(lastY - firstY))
            }
        }
        let means = columnMeans(values)
        let columns = (0..<w).filter { means[$0] > 0.65 && means[$0] < 0.95 }
        XCTAssertGreaterThan(columns.count, 8)
        func bandCorrelation(_ input: [Float]) -> Float {
            let averages = columnMeans(input)
            var peak: Float = 0
            for dy in 4...12 {
                for dx in -12...12 {
                    var product: Float = 0, energyA: Float = 0, energyB: Float = 0
                    var samples = 0
                    for x in columns where x + dx >= 0 && x + dx < w {
                        for y in firstY..<(lastY - dy) {
                            let a = input[y * w + x] - averages[x]
                            let b = input[(y + dy) * w + x + dx] - averages[x + dx]
                            product += a * b; energyA += a * a; energyB += b * b
                            samples += 1
                        }
                    }
                    // Correlation alone is scale invariant: nearly constant
                    // columns can score 1 despite invisible residuals. Use
                    // one output AO code as the variance floor, after testing
                    // the shipped spatial filter in FP32 rather than R8.
                    let floor = Float(samples) / (255 * 255)
                    peak = max(peak, abs(product) / max(sqrt(energyA * energyB), floor))
                }
            }
            return peak
        }
        let peak = bandCorrelation(values)
        var residualEnergy: Float = 0
        var blockEnergy: Float = 0
        var samples = 0
        for x in columns {
            for y in firstY..<(lastY - 4) {
                let residual = values[y*w+x] - means[x]
                var block: Float = 0
                for offset in 0..<4 { block += (values[(y+offset)*w+x] - means[x]) / 4 }
                residualEnergy += residual*residual
                blockEnergy += block*block
                samples += 1
            }
        }
        print("GTAO spatial RMS=\(sqrt(residualEnergy/Float(samples))), four-pixel RMS=\(sqrt(blockEnergy/Float(samples)))")
        print("GTAO visible directional residual correlation=\(peak), columns=\(columns.count)")
        XCTAssertLessThan(peak, 0.5)
        // The metric must still reject visible diagonal bands at the spatial
        // frequency of the original regression, even on a very clean input.
        let striped = values.enumerated().map { index, value in
            value + 0.015 * sin(2 * .pi * Float(index % w + index / w) / 8)
        }
        XCTAssertGreaterThan(bandCorrelation(striped), 0.8)
        try save(result.resolved, width: w, height: h, name: "wall-noise")
    }

    func testSpatialAOReconstructionReducesNoiseInOneFrame() throws {
        let fixture = Fixture(plane: SIMD4(0, 0, 1, -3),
            boxMin: SIMD4(-10, -100, -3, 1), boxMax: SIMD4(0.2, 100, -2.7, 1))
        let width = 256, height = 192
        let result = try render(fixture, width: width, height: height, eye: SIMD3(1.2, 0, 0))
        func variance(_ values: [Float]) -> Double {
            var total = 0.0
            for x in 0..<width {
                let column = (24..<(height - 24)).map { Double(values[$0 * width + x]) }
                let mean = column.reduce(0, +) / Double(column.count)
                total += column.reduce(0) { $0 + ($1 - mean) * ($1 - mean) }
            }
            return total
        }
        let ratio = variance(result.resolved) / variance(result.raw)
        print("Fast AO spatial/raw noise variance ratio in first frame: \(ratio)")
        XCTAssertLessThan(ratio, 0.5,
            "Fast AO must reconstruct its sampling noise spatially without waiting for temporal history")
    }

    func testThinNearbyOccluderHasSmoothAOWithoutBandsOrPatches() throws {
        // A thin vertical bar on an infinite plane models a rack wire in
        // front of a pan. Physical visibility is constant along Y and rises
        // monotonically away from the bar along X. Those two invariants let
        // us distinguish sampling patches from the real contact gradient.
        for normal in [SIMD3<Float>(0, 0, 1), SIMD3<Float>(0.8, 0, 0.6), SIMD3<Float>(-0.8, 0, 0.6)] {
            let fixture = Fixture(plane: SIMD4(normal, -normal.z),
                boxMin: SIMD4(-0.005, -100, -1.01, 1),
                boxMax: SIMD4(0.005, 100, -0.95, 1))
            let width = 384, height = 288
            let eye = SIMD3<Float>(0.2, 0, 0)
            let result = try render(fixture, width: width, height: height, eye: eye,
                capturePreciseResolve: true)
            let values = try XCTUnwrap(result.preciseResolved)
            try save(result.raw, width: width, height: height, name: "thin-nearby-bar-\(normal.x)-raw")
            try save(result.resolved, width: width, height: height, name: "thin-nearby-bar-\(normal.x)-resolved")

            let U = uniforms(width: width, height: height, eye: eye)
            func column(atWorldX x: Float) -> Int {
                let clip = U.viewProj * SIMD4<Float>(x, 0, -1 - normal.x / normal.z * x, 1)
                return Int((clip.x / clip.w * 0.5 + 0.5) * Float(width))
            }
            // Stay off the bar silhouette and the spatial filter's edge kernel.
            let columns = column(atWorldX: 0.02)..<column(atWorldX: 0.25)
            let rows = 48..<(height - 48)
            let means = (0..<width).map { x in
                rows.reduce(0.0) { $0 + Double(values[$1 * width + x]) } / Double(rows.count)
            }
            var pixelEnergy = 0.0, patchEnergy = 0.0
            var pixelCount = 0, patchCount = 0
            for x in columns {
                for y in rows {
                    let residual = Double(values[y * width + x]) - means[x]
                    pixelEnergy += residual * residual
                    pixelCount += 1
                }
                // Four-row averages preserve the coarse patches complained
                // about while suppressing harmless pixel-scale rounding.
                for y in stride(from: rows.lowerBound, to: rows.upperBound, by: 4) {
                    let patch = (0..<4).reduce(0.0) { $0 + Double(values[(y + $1) * width + x]) } / 4 - means[x]
                    patchEnergy += patch * patch
                    patchCount += 1
                }
            }
            let pixelRMS = sqrt(pixelEnergy / Double(pixelCount))
            let patchRMS = sqrt(patchEnergy / Double(patchCount))
            var maximumReversedStep = 0.0
            for x in columns {
                for offset in 1...8 where x + offset < columns.upperBound {
                    maximumReversedStep = max(maximumReversedStep, means[x] - means[x + offset])
                }
            }
            let near = column(atWorldX: 0.025)..<column(atWorldX: 0.04)
            let nearAO = near.reduce(0.0) { $0 + means[$1] } / Double(near.count)
            let referenceAO = near.reduce(0.0) { sum, x in
                let ray4 = U.invViewProj * SIMD4<Float>((Float(x) + 0.5) / Float(width) * 2 - 1, 0, 0, 1)
                let ray = SIMD3(ray4.x, ray4.y, ray4.z) / ray4.w - eye
                let point = eye + ray * ((fixture.plane.w - simd_dot(normal, eye)) / simd_dot(normal, ray))
                // Account for the finite projected-radius cap at this distance;
                // integration itself intersects independent three-dimensional rays.
                let radius = min(0.9, Double(96 * max(-point.z, 0.25) / U.screen.z))
                return sum + Double(rayVisibility(point: point, fixture: fixture, samples: 8192, radius: radius))
            } / Double(near.count)
            print("GTAO thin nearby bar normal=\(normal): pixel RMS=\(pixelRMS), 4-row patch RMS=\(patchRMS), reversed profile step=\(maximumReversedStep), near AO=\(nearAO), physical visibility=\(referenceAO)")
            XCTAssertLessThan(pixelRMS, 1.0 / 255,
                "A physically uniform vertical receiver must not acquire visible sampling noise")
            XCTAssertLessThan(patchRMS, 1.0 / 255,
                "A static thin occluder must not leave coarse patches after spatial reconstruction")
            XCTAssertLessThan(maximumReversedStep, 2.0 / 255,
                "AO must fade smoothly away from the bar, without alternating dark and light bands")
            XCTAssertEqual(pow(nearAO, 1 / 1.25), referenceAO, accuracy: 0.13,
                "Removing real contact occlusion must not satisfy the noise checks")
        }
    }

    func testDiagonalThinOccluderDoesNotLeaveGridAlignedPatches() throws {
        let diagonal: Float = sqrt(0.5)
        let fixture = Fixture(plane: SIMD4(0, 0, 1, -1),
            boxMin: SIMD4(-0.005, -100, -1, 1),
            boxMax: SIMD4(0.005, 100, -0.95, 1),
            boxOrientation: SIMD4(diagonal, diagonal, 0, 0))
        let width = 384, height = 288
        let eye = SIMD3<Float>(0.2, 0, 0)
        let result = try render(fixture, width: width, height: height, eye: eye,
            capturePreciseResolve: true)
        let values = try XCTUnwrap(result.preciseResolved)
        try save(result.raw, width: width, height: height, name: "diagonal-thin-bar-raw")
        try save(result.resolved, width: width, height: height, name: "diagonal-thin-bar-resolved")
        // With a 45-degree bar and square pixels, equal (x-y) means equal
        // geometric distance from the bar. Group exact pixel centers instead
        // of rotating/resampling an image, which would hide grid artifacts.
        var groups = Array(repeating: [Double](), count: width + height - 1)
        for y in 48..<(height - 48) {
            for x in 32..<(width - 32) {
                groups[x - y + height - 1].append(Double(values[y * width + x]))
            }
        }
        let U = uniforms(width: width, height: height, eye: eye)
        func group(atDistance distance: Float) -> Int {
            let k = sqrt(Float(2)) * U.screen.z * distance - U.screen.z * eye.x + Float(width - height) * 0.5
            return Int(k.rounded()) + height - 1
        }
        let selected = group(atDistance: 0.02)..<group(atDistance: 0.25)
        let means = groups.map { $0.isEmpty ? 1 : $0.reduce(0, +) / Double($0.count) }
        var energy = 0.0, patchEnergy = 0.0
        var count = 0, patchCount = 0
        for index in selected {
            let samples = groups[index]
            XCTAssertGreaterThan(samples.count, 64)
            for value in samples {
                energy += pow(value - means[index], 2)
                count += 1
            }
            for start in stride(from: 0, through: samples.count - 4, by: 4) {
                let average = samples[start..<(start + 4)].reduce(0, +) / 4
                patchEnergy += pow(average - means[index], 2)
                patchCount += 1
            }
        }
        let pixelRMS = sqrt(energy / Double(count))
        let patchRMS = sqrt(patchEnergy / Double(patchCount))
        var reversedStep = 0.0
        for index in selected {
            for offset in 1...8 where index + offset < selected.upperBound {
                reversedStep = max(reversedStep, means[index] - means[index + offset])
            }
        }
        print("GTAO diagonal thin bar: pixel RMS=\(pixelRMS), 4-pixel patch RMS=\(patchRMS), reversed profile step=\(reversedStep)")
        XCTAssertLessThan(pixelRMS, 1.0 / 255)
        XCTAssertLessThan(patchRMS, 1.0 / 255,
            "Reconstruction must remove coarse sampling patches at oblique occluder edges")
        XCTAssertLessThan(reversedStep, 2.0 / 255,
            "Rotating a thin occluder must not introduce bands across its smooth AO falloff")
        XCTAssertLessThan(try XCTUnwrap(selected.map { means[$0] }.min()), 0.97,
            "The fixture must retain real AO while testing its spatial smoothness")
    }

    func testFloatingThinBarMatchesFiniteGeometryVisibility() throws {
        // A rack wire has a real air gap behind it. A depth height field can
        // silently fill that gap and turn a thin rod into an opaque wall.
        // Flat, matching geometric/shading normals isolate that assumption
        // from shading-normal interpolation and tangent-plane rejection.
        let width = 384, height = 288
        let eye = SIMD3<Float>(0.2, 0, 0)
        let U = uniforms(width: width, height: height, eye: eye)
        let radius = min(0.9, Double(96 / U.screen.z))
        for halfWidth: Float in [0.005, 0.01] {
            for gap: Float in [0.02, 0.05] {
                let fixture = Fixture(plane: SIMD4(0, 0, 1, -1),
                    boxMin: SIMD4(-halfWidth, -100, -1 + gap, 1),
                    boxMax: SIMD4(halfWidth, 100, -1 + gap + 0.01, 1))
                let result = try render(fixture, width: width, height: height, eye: eye,
                    capturePreciseResolve: true)
                let values = try XCTUnwrap(result.preciseResolved)
                let name = "floating-bar-\(Int(halfWidth * 2000))mm-gap-\(Int(gap * 1000))mm"
                try save(result.raw, width: width, height: height, name: name + "-raw")
                try save(result.resolved, width: width, height: height, name: name + "-resolved")
                var maximumOverOcclusion = 0.0, maximumUnderOcclusion = 0.0
                var lowestReference = 1.0
                var visibleProbes = 0
                for requestedX: Float in [-0.08, -0.04, -0.02, 0, 0.02, 0.04, 0.08, 0.12] {
                    let clip = U.viewProj * SIMD4<Float>(requestedX, 0, -1, 1)
                    let x = Int((clip.x / clip.w * 0.5 + 0.5) * Float(width))
                    let ray4 = U.invViewProj * SIMD4<Float>((Float(x) + 0.5) / Float(width) * 2 - 1, 0, 0, 1)
                    let ray = SIMD3(ray4.x, ray4.y, ray4.z) / ray4.w - eye
                    let point = eye + ray * (-1 / ray.z)
                    // Only compare actual receiver pixels. Some points under
                    // the floating bar are hidden from this camera by the rod.
                    let toPoint = point - eye
                    let a = (SIMD3(fixture.boxMin.x, fixture.boxMin.y, fixture.boxMin.z) - eye) / toPoint
                    let b = (SIMD3(fixture.boxMax.x, fixture.boxMax.y, fixture.boxMax.z) - eye) / toPoint
                    let lo = simd_min(a, b), hi = simd_max(a, b)
                    let entry = max(lo.x, max(lo.y, lo.z)), leave = min(hi.x, min(hi.y, hi.z))
                    if entry > 0 && entry < 1 && entry < leave { continue }
                    let reference = Double(rayVisibility(point: point, fixture: fixture, radius: radius))
                    let ao = (48..<(height - 48)).reduce(0.0) { $0 + Double(values[$1 * width + x]) } / Double(height - 96)
                    let measured = pow(ao, 1 / 1.25)
                    maximumOverOcclusion = max(maximumOverOcclusion, reference - measured)
                    maximumUnderOcclusion = max(maximumUnderOcclusion, measured - reference)
                    lowestReference = min(lowestReference, reference)
                    visibleProbes += 1
                    print("GTAO \(name) x=\(requestedX): ray=\(reference), GPU=\(measured)")
                    XCTAssertEqual(measured, reference, accuracy: 0.13,
                        "A floating rod must preserve its air gap rather than cast AO as a solid wall: \(name), x=\(requestedX)")
                }
                print("GTAO \(name): max over-occlusion=\(maximumOverOcclusion), max under-occlusion=\(maximumUnderOcclusion)")
                XCTAssertGreaterThanOrEqual(visibleProbes, 5)
                XCTAssertLessThan(lowestReference, 0.95, "The finite rod must cast measurable reference AO")
            }
        }
    }

    func testStaticAOIsIdenticalFromTheFirstFrameRegardlessOfFrameIndex() throws {
        let fixture = Fixture(plane: SIMD4(0, 0, 1, -3),
            boxMin: SIMD4(-10, -100, -3, 1), boxMax: SIMD4(0.2, 100, -2.7, 1))
        let result = try render(fixture, frames: 66, eye: SIMD3(1.2, 0, 0),
            captureFrames: [1, 2, 17, 64, 65, 66])
        let firstRaw = try XCTUnwrap(result.rawSnapshots[1])
        let firstResolved = try XCTUnwrap(result.snapshots[1])
        XCTAssertLessThan(try XCTUnwrap(firstResolved.min()), 0.9,
            "The fixture must contain AO, so returning uniform white cannot satisfy stability")
        for frame in [2, 17, 64, 65, 66] {
            XCTAssertEqual(firstRaw, result.rawSnapshots[frame],
                "Stationary sampling must be frame-index independent before filtering (frame \(frame))")
            XCTAssertEqual(firstResolved, result.snapshots[frame],
                "Fast AO must be stable immediately, without convergence or a warm-up interval (frame \(frame))")
        }
    }

    func testMovingOccluderUpdatesAOOnAnUnchangedReceiverImmediately() throws {
        let fixture = Fixture(plane: SIMD4(0, 0, 1, -3),
            boxMin: SIMD4(-10, -100, -3, 1), boxMax: SIMD4(0.2, 100, -2.7, 1))
        let width = 256, height = 192
        let removed = Fixture(plane: fixture.plane)
        let result = try render(fixture, width: width, height: height, frames: 3,
            eye: SIMD3(1.2, 0, 0), finalFixture: removed, captureFrames: [2])
        let old = try XCTUnwrap(result.snapshots[2])
        // Receiver depth and normal stay unchanged when the neighbor leaves.
        let receivers = (48..<144).flatMap { y in (64..<84).map { x in y * width + x } }
        XCTAssertLessThan(try XCTUnwrap(receivers.map { old[$0] }.min()), 0.95,
            "The fixture must contain substantial AO before the blocker moves")
        let fresh = try render(removed, width: width, height: height, eye: SIMD3(1.2, 0, 0))
        XCTAssertEqual(result.raw, fresh.raw)
        XCTAssertEqual(result.resolved, fresh.resolved,
            "A removed neighboring occluder must leave no history ghost on its first frame")
        XCTAssertGreaterThan(try XCTUnwrap(result.resolved.min()), 0.99)

        var moved = fixture
        moved.boxMax.z -= 0.0048
        let changed = try render(fixture, width: width, height: height, frames: 3,
            eye: SIMD3(1.2, 0, 0), finalFixture: moved)
        let movedReference = try render(moved, width: width, height: height, eye: SIMD3(1.2, 0, 0))
        XCTAssertEqual(changed.raw, movedReference.raw)
        XCTAssertEqual(changed.resolved, movedReference.resolved,
            "Even small occluder motion must produce the current-view result without stale accumulation")
    }

    func testCameraMotionProducesTheCurrentViewWithoutAOHistory() throws {
        let fixture = Fixture(plane: SIMD4(0, 0, 1, -3),
            boxMin: SIMD4(-10, -100, -3, 1), boxMax: SIMD4(0.2, 100, -2.7, 1))
        let result = try render(fixture, frames: 12, captureFrames: [9, 10, 12],
            camera: { frame in SIMD3(1.2 + 0.0125 * Float(min(frame, 8)), 0, 0) })
        let stopped = try XCTUnwrap(result.snapshots[9])
        XCTAssertEqual(stopped, result.snapshots[10])
        XCTAssertEqual(stopped, result.snapshots[12])
        let reference = try render(fixture, eye: SIMD3(1.2 + 0.0125 * Float(8), 0, 0))
        XCTAssertEqual(result.raw, reference.raw)
        XCTAssertEqual(result.resolved, reference.resolved,
            "The first stationary frame must equal a fresh render of the new camera view")
    }

    func testAOChainBenchmark() throws {
        guard ProcessInfo.processInfo.environment["GTAO_TEST_BENCHMARK"] == "1" else {
            throw XCTSkip("opt-in GPU timing; run serially on an idle GPU")
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
        print("GTAO chain 1440x858 (48 taps, packed front/back depth, two spatial passes): \(times) ms; median=\(times.sorted()[1]) ms")
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
