import Foundation
import CoreImage
import Metal
import simd
import XCTest
@testable import GPUSimRenderer

@MainActor
final class ScreenSpaceLightingTests: XCTestCase {
    private struct Fixture {
        var plane = SIMD4<Float>(0, 1, 0, -0.7)
        var boxMin = SIMD4<Float>(-0.35, -0.7, -2.4, 1)
        var boxMax = SIMD4<Float>(0.35, 0, -1.8, 0)
        var material = SIMD4<Float>(0.6, 0.6, 0.6, 1)
        var parameters = SIMD4<Float>(0.1, 0, 0, 0)
    }
    private let fixtureSource = """
    struct LightingFixture { float4 plane; float4 boxMin; float4 boxMax; float4 material; float4 parameters; };
    struct LightingHit { float depth; float3 normal; float3 color; float roughness; float4 material; };
    inline LightingHit fixtureHit(float2 uv, constant Uniforms& U, constant LightingFixture& F) {
        float3 ray = normalize(worldFromDepth(uv, 0.0, U.invViewProj) - U.eye.xyz);
        float denom = dot(F.plane.xyz, ray);
        float t = (F.plane.w - dot(F.plane.xyz, U.eye.xyz)) / denom;
        float3 N = F.plane.xyz;
        if (t <= 0 || abs(denom) < 1e-6) t = 1e20;
        bool box = false;
        if (F.boxMin.w > 0) {
            float3 a = (F.boxMin.xyz - U.eye.xyz) / ray;
            float3 b = (F.boxMax.xyz - U.eye.xyz) / ray;
            float3 lo = min(a, b), hi = max(a, b);
            float entry = max(lo.x, max(lo.y, lo.z)), leave = min(hi.x, min(hi.y, hi.z));
            if (entry > 0 && entry < leave && entry < t) {
                t = entry; N = float3(0); box = true;
                uint axis = lo.x >= lo.y && lo.x >= lo.z ? 0 : (lo.y >= lo.z ? 1 : 2);
                N[axis] = ray[axis] > 0 ? -1 : 1;
            }
        }
        float4 clip = U.viewProj * float4(U.eye.xyz + ray*t, 1);
        LightingHit h;
        h.depth = t < 1e19 && clip.z >= 0 && clip.z < clip.w ? clip.z / clip.w : 1;
        h.normal = N;
        h.color = box ? float3(3.0, 0.05, 0.02) : float3(0.3);
        h.roughness = box ? 1.0 : F.parameters.x;
        h.material = box ? float4(0) : F.material;
        return h;
    }
    struct SurfaceFixtureOut { float4 normal [[color(0)]]; float4 material [[color(1)]]; float depth [[depth(any)]]; };
    fragment SurfaceFixtureOut lighting_surface(FSOut in [[stage_in]], constant Uniforms& U [[buffer(1)]],
        constant LightingFixture& F [[buffer(2)]]) {
        LightingHit h = fixtureHit(in.uv, U, F);
        SurfaceFixtureOut o; o.normal = float4(h.normal, h.roughness); o.material = h.material; o.depth = h.depth;
        return o;
    }
    struct ColorFixtureOut { float4 color [[color(0)]]; float depth [[depth(any)]]; };
    fragment ColorFixtureOut lighting_color(FSOut in [[stage_in]], constant Uniforms& U [[buffer(1)]],
        constant LightingFixture& F [[buffer(2)]]) {
        LightingHit h = fixtureHit(in.uv, U, F);
        ColorFixtureOut o; o.color = float4(h.color, 1); o.depth = h.depth; return o;
    }
    """

    private func uniforms(width: Int, height: Int, eye: SIMD3<Float> = .zero) -> Uniforms {
        let y: Float = 1 / tan(25 * .pi / 180), near: Float = 0.1, far: Float = 100
        let projection = simd_float4x4(columns: (
            SIMD4(y * Float(height) / Float(width), 0, 0, 0), SIMD4(0, y, 0, 0),
            SIMD4(0, 0, far / (near-far), -1), SIMD4(0, 0, near*far / (near-far), 0)))
        var view = matrix_identity_float4x4; view.columns.3 = SIMD4(-eye, 1)
        let vp = projection * view
        return Uniforms(viewProj: vp, lightDir: SIMD4(normalize(SIMD3<Float>(0.6, -1, 0.5)), 0),
            eye: SIMD4(eye, 0), screen: SIMD4(Float(width), Float(height), Float(height)*y*0.5, 0),
            camRight: SIMD4(1, 0, 0, 0), camUp: SIMD4(0, -1, 0, 0),
            prevViewProj: vp, temporal: SIMD4(0, 1, 0, 0), shadowViewProj: matrix_identity_float4x4,
            shadowParams: .zero, invViewProj: vp.inverse, prevInvViewProj: vp.inverse,
            effects: SIMD4(1, 0.2, 4, 0.65),
            aoProjection: SIMD4(-projection.columns.2.z, projection.columns.3.z,
                                1/projection.columns.0.x, 1/projection.columns.1.y))
    }

    private final class Harness {
        let effects: ScreenSpacePipeline
        let surface, color: MTLRenderPipelineState
        let queue: MTLCommandQueue
        let depthState: MTLDepthStencilState
        init(source: String, width: Int, height: Int) throws {
            guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal is unavailable") }
            let lib = try device.makeLibrary(source: renderShaderSource + source, options: nil)
            effects = try ScreenSpacePipeline(device: device, library: lib)
            try effects.prepare(size: CGSize(width: width, height: height), options: GPUSimRenderOptions(screenSpaceReflections: true))
            func pipe(_ name: String, material: Bool) throws -> MTLRenderPipelineState {
                let d = MTLRenderPipelineDescriptor()
                d.vertexFunction = lib.makeFunction(name: "fs_vertex")
                d.fragmentFunction = lib.makeFunction(name: name)
                d.colorAttachments[0].pixelFormat = .rgba16Float
                if material { d.colorAttachments[1].pixelFormat = .rgba8Unorm }
                d.depthAttachmentPixelFormat = .depth32Float
                return try device.makeRenderPipelineState(descriptor: d)
            }
            surface = try pipe("lighting_surface", material: true)
            color = try pipe("lighting_color", material: false)
            queue = try XCTUnwrap(device.makeCommandQueue())
            let d = MTLDepthStencilDescriptor(); d.depthCompareFunction = .always; d.isDepthWriteEnabled = true
            depthState = try XCTUnwrap(device.makeDepthStencilState(descriptor: d))
        }
    }

    private func render(_ fixture: Fixture, harness h: Harness, eye: SIMD3<Float> = .zero,
                        world: RayTracingScene? = nil, rayScene: RaySceneFixture? = nil,
                        auxiliary: [GPUSimRenderInstance] = [], appearances: [Int: GPUSimRenderAppearance] = [:],
                        lightDirection: SIMD3<Float>? = nil, screenShortcut: Bool = false, diffuseFrame: Int? = nil, accumulationFrame: Int = 0) throws -> ([Float], [SIMD4<Float>]) {
        let e = h.effects
        var options = GPUSimRenderOptions(ambientOcclusion: false, screenSpaceReflections: world == nil || screenShortcut)
        if world != nil { options.lightingMode = .qualityBeta }
        try e.prepare(size: CGSize(width: e.size.x, height: e.size.y), options: options)
        var U = uniforms(width: e.halfSize.x, height: e.halfSize.y, eye: eye)
        if let lightDirection { U.lightDir = SIMD4(normalize(lightDirection), 0) }
        U.rayTracing = SIMD4(world == nil ? 0 : 1, screenShortcut ? 1 : 0, 0, 0)
        if world != nil { U.reconstruction = SIMD4(1,0,0,Float(diffuseFrame ?? 0)) }
        if diffuseFrame != nil {
            let phase = Float(max(0,min(accumulationFrame,64)-1))*0.6180339887
            U.temporal = SIMD4(phase.truncatingRemainder(dividingBy: 1),accumulationFrame<=1 ? 1 : 1/Float(accumulationFrame),Float(accumulationFrame),0)
        }
        var F = fixture
        let cmd = try XCTUnwrap((world?.queue ?? h.queue).makeCommandBuffer())
        var rayInstances: MTLBuffer?
        func buffer<T>(_ values: [T]) throws -> MTLBuffer? {
            guard !values.isEmpty else { return nil }
            return try XCTUnwrap(values.withUnsafeBytes { e.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) })
        }
        let auxiliaryBuffer = try buffer(auxiliary)
        let appearanceBuffer = try buffer(appearances.isEmpty ? [] : (0..<(rayScene?.renderBodyCount ?? 0)).map { appearances[$0] ?? .init() })
        if let world, let rayScene {
            let buffer = try XCTUnwrap(e.device.makeBuffer(length: max(1,rayScene.values.count)*MemoryLayout<GPUSimRenderInstance>.stride, options: .storageModePrivate))
            try rayScene.encodeRenderInstances(cmd, instances: buffer, colorMode: .bodyIndex, appearanceOverrides: appearanceBuffer)
            try world.encodeUpdate(command: cmd, scene: rayScene, instances: buffer, auxiliary: auxiliaryBuffer,
                                   appearances: appearanceBuffer, appearanceValues: appearances)
            rayInstances = buffer
        }
        for isSurface in [true, false] {
            let d = MTLRenderPassDescriptor()
            d.colorAttachments[0].texture = isSurface ? e.normal : e.sceneColor
            d.colorAttachments[0].loadAction = .dontCare; d.colorAttachments[0].storeAction = .store
            if isSurface {
                d.colorAttachments[1].texture = e.material
                d.colorAttachments[1].loadAction = .dontCare; d.colorAttachments[1].storeAction = .store
            }
            d.depthAttachment.texture = isSurface ? e.depth : e.sceneDepth
            d.depthAttachment.loadAction = .clear; d.depthAttachment.clearDepth = 1; d.depthAttachment.storeAction = .store
            let enc = try XCTUnwrap(cmd.makeRenderCommandEncoder(descriptor: d))
            enc.setRenderPipelineState(isSurface ? h.surface : h.color)
            enc.setDepthStencilState(h.depthState)
            enc.setFragmentBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.setFragmentBytes(&F, length: MemoryLayout<Fixture>.stride, index: 2)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            enc.endEncoding()
        }
        if let world, let rayInstances {
            try world.encodeLighting(command: cmd, uniforms: U, screen: e, instances: rayInstances,
                                     auxiliary: auxiliaryBuffer, appearances: appearanceBuffer, reflections: false)
        }
        try e.encodeBeforeLighting(command: cmd, uniforms: U, options: options)
        if let world, let rayInstances {
            if screenShortcut { try e.encodeReflections(command: cmd, uniforms: U, filter: false) }
            try world.encodeLighting(command: cmd, uniforms: U, screen: e, instances: rayInstances,
                                     auxiliary: auxiliaryBuffer, appearances: appearanceBuffer, reflections: true)
            if diffuseFrame != nil {
                try world.encodeDiffuse(command: cmd, uniforms: U, screen: e, instances: rayInstances,
                                        auxiliary: auxiliaryBuffer, appearances: appearanceBuffer)
            }
        } else {
            try e.encodeReflections(command: cmd, uniforms: U)
        }
        cmd.commit(); cmd.waitUntilCompleted()
        XCTAssertEqual(cmd.status, .completed, "\(String(describing: cmd.error))")
        let contactBytes = try read(e.directVisibilityRaw!, queue: h.queue, bytesPerPixel: 1)
        let reflectionBytes = try read(e.reflection!, queue: h.queue, bytesPerPixel: 8)
        let reflection: [SIMD4<Float>] = reflectionBytes.withUnsafeBytes { raw in
            let values = raw.bindMemory(to: UInt16.self)
            return stride(from: 0, to: values.count, by: 4).map { i in
                SIMD4(Float(Float16(bitPattern: values[i])), Float(Float16(bitPattern: values[i+1])),
                      Float(Float16(bitPattern: values[i+2])), Float(Float16(bitPattern: values[i+3])))
            }
        }
        return (contactBytes.map { Float($0)/255 }, reflection)
    }

    // An empty RT scene still produces sampled environment radiance and coverage.
    // Compare against an independently built empty world, not a zero RGBA buffer.
    private func emptyReflections(_ fixture: Fixture, harness: Harness, eye: SIMD3<Float> = .zero) throws -> [SIMD4<Float>] {
        let scene = RaySceneFixture(device: harness.effects.device, values: [])
        let world = try RayTracingScene.shared(scene: scene)
        try world.prepare(scene: scene, revision: 0, auxiliary: [], ground: false)
        return try render(fixture, harness: harness, eye: eye, world: world, rayScene: scene,
                          lightDirection: SIMD3(0,0,-1)).1
    }

    private func read(_ texture: MTLTexture, queue: MTLCommandQueue, bytesPerPixel: Int) throws -> [UInt8] {
        let row = (texture.width * bytesPerPixel + 255) / 256 * 256
        let buffer = try XCTUnwrap(texture.device.makeBuffer(length: row*texture.height, options: .storageModeShared))
        let cmd = try XCTUnwrap(queue.makeCommandBuffer()), b = try XCTUnwrap(cmd.makeBlitCommandEncoder())
        b.copy(from: texture, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(),
               sourceSize: MTLSize(width: texture.width, height: texture.height, depth: 1), to: buffer,
               destinationOffset: 0, destinationBytesPerRow: row, destinationBytesPerImage: row*texture.height)
        b.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
        let bytes = buffer.contents().assumingMemoryBound(to: UInt8.self)
        return (0..<texture.height).flatMap { y in Array(UnsafeBufferPointer(start: bytes+y*row, count: texture.width*bytesPerPixel)) }
    }

    func testOpenPlanesHaveNeitherContactShadowsNorSelfReflections() throws {
        let h = try Harness(source: fixtureSource, width: 511, height: 383)
        for z: Float in [1, 0.5, 0.08] {
            var f = Fixture(); f.boxMin.w = 0
            let n = normalize(SIMD3<Float>(sqrt(1-z*z)*0.8, sqrt(1-z*z)*0.6, z))
            f.plane = SIMD4(n, -3*z)
            for eye in [SIMD3<Float>.zero, SIMD3(20, -30, 5)] {
                f.plane.w = -3*z + dot(n, eye)
                let (contact, reflection) = try render(f, harness: h, eye: eye)
                XCTAssertEqual(contact.min(), 1, "no striping even at grazing angles or viewport borders")
                XCTAssertTrue(reflection.allSatisfy { $0 == .zero }, "a plane must never reflect itself")
            }
        }
    }

    func testContactShadowTracksVisibleBlockerAndClearsImmediately() throws {
        let h = try Harness(source: fixtureSource, width: 512, height: 384)
        var f = Fixture()
        let blocked = try render(f, harness: h).0
        let shadowCount = blocked.filter { $0 < 0.5 }.count
        XCTAssertGreaterThan(shadowCount, 15, "visible short-range blocker must cast a contact shadow")
        f.boxMin.w = 0
        let open = try render(f, harness: h).0
        XCTAssertEqual(open.min(), 1, "removed blockers must not leave temporal trails")
    }

    func testReflectionsHitColoredGeometryAndRespectRoughnessAndMotion() throws {
        let h = try Harness(source: fixtureSource, width: 512, height: 384)
        var f = Fixture()
        let reflected = try render(f, harness: h).1
        let hits = reflected.filter { $0.w > 0.5 }
        print("SSR fixture confident hits: \(hits.count), peak red: \(reflected.map(\.x).max() ?? 0)")
        XCTAssertGreaterThan(hits.count, 50)
        XCTAssertGreaterThan(hits.map(\.x).max() ?? 0, 0.4, "HDR red reflected radiance must survive")
        XCTAssertTrue(hits.allSatisfy { $0.x > $0.y && $0.x > $0.z })
        f.parameters.x = 0.9
        XCTAssertTrue(try render(f, harness: h).1.allSatisfy { $0 == .zero }, "rough surfaces use the environment fallback")
        f.parameters.x = 0.1; f.boxMin.w = 0
        XCTAssertTrue(try render(f, harness: h).1.allSatisfy { $0 == .zero }, "no stale reflections after object removal")
    }

    private func boxIntersection(origin: SIMD3<Float>, direction: SIMD3<Float>, fixture: Fixture) -> Float? {
        let a = (SIMD3(fixture.boxMin.x, fixture.boxMin.y, fixture.boxMin.z) - origin) / direction
        let b = (SIMD3(fixture.boxMax.x, fixture.boxMax.y, fixture.boxMax.z) - origin) / direction
        let lo = simd_min(a, b), hi = simd_max(a, b)
        let entry = max(lo.x, max(lo.y, lo.z)), leave = min(hi.x, min(hi.y, hi.z))
        return entry > 0 && entry < leave ? entry : nil
    }

    func testReflectionFootprintMatchesIndependentGeometricRays() throws {
        let h = try Harness(source: fixtureSource, width: 511, height: 383)
        let f = Fixture()
        let pixels = try render(f, harness: h).1
        let width = h.effects.halfSize.x, height = h.effects.halfSize.y
        let u = uniforms(width: width, height: height)
        var expected = 0, detected = 0, matched = 0
        for y in 0..<height { for x in 0..<width {
            let clip = u.invViewProj * SIMD4((Float(x)+0.5)/Float(width)*2-1,
                1-(Float(y)+0.5)/Float(height)*2, 0, 1)
            let ray = normalize(SIMD3(clip.x, clip.y, clip.z) / clip.w)
            if ray.y >= 0 { continue }
            let planeT = f.plane.w / ray.y
            if let front = boxIntersection(origin: .zero, direction: ray, fixture: f), front < planeT { continue }
            let p = ray * planeT
            let reflected = SIMD3(ray.x, -ray.y, ray.z)
            let reference = boxIntersection(origin: p + SIMD3(0, 0.0001, 0), direction: reflected, fixture: f)
            let truth = reference.map { $0 < 3 } ?? false
            let hit = pixels[y*width+x].w > 0.25
            expected += truth ? 1 : 0; detected += hit ? 1 : 0; matched += truth && hit ? 1 : 0
        } }
        let precision = Float(matched) / Float(max(detected, 1))
        let recall = Float(matched) / Float(max(expected, 1))
        print("SSR geometric footprint precision=\(precision), recall=\(recall)")
        XCTAssertGreaterThan(expected, 100)
        XCTAssertGreaterThan(precision, 0.9)
        XCTAssertGreaterThan(recall, 0.85)
    }

    func testHDRCompositionPreservesNeutralColorAndWritesDisplayPixels() throws {
        let h = try Harness(source: fixtureSource, width: 127, height: 95)
        var f = Fixture(); f.boxMin.w = 0; f.parameters.x = 1
        _ = try render(f, harness: h)
        let e = h.effects
        let output = try e.texture(.bgra8Unorm_srgb, width: 127, height: 95, label: "Display test")
        let msaa = try e.texture(.bgra8Unorm_srgb, width: 127, height: 95, label: "Display MSAA test", samples: 4)
        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: 127, height: 95, mipmapped: false)
        depthDesc.textureType = .type2DMultisample; depthDesc.sampleCount = 4
        depthDesc.storageMode = .private; depthDesc.usage = .renderTarget
        let depth = try XCTUnwrap(e.device.makeTexture(descriptor: depthDesc))
        let cmd = try XCTUnwrap(h.queue.makeCommandBuffer())
        let clear = MTLRenderPassDescriptor()
        clear.depthAttachment.texture = depth; clear.depthAttachment.loadAction = .clear
        clear.depthAttachment.storeAction = .store
        let clearEncoder = try XCTUnwrap(cmd.makeRenderCommandEncoder(descriptor: clear)); clearEncoder.endEncoding()
        let display = MTLRenderPassDescriptor()
        display.colorAttachments[0].texture = msaa; display.colorAttachments[0].resolveTexture = output
        display.colorAttachments[0].storeAction = .multisampleResolve
        display.depthAttachment.texture = depth
        let encoder = try e.beginComposite(command: cmd, destination: display, uniforms: uniforms(width: 127, height: 95))
        encoder.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
        XCTAssertEqual(cmd.status, .completed, "\(String(describing: cmd.error))")
        let bytes = try read(output, queue: h.queue, bytesPerPixel: 4)
        // The callback exposes an sRGB Metal texture. Core Image samples
        // its decoded linear values, so labeling that input sRGB darkens it
        // a second time even though the renderer's drawable is correct.
        let ci = try XCTUnwrap(CIImage(mtlTexture: output, options: [.colorSpace: CGColorSpace(name: CGColorSpace.linearSRGB)!]))
        let context = CIContext(mtlDevice: e.device)
        var exported = [UInt8](repeating: 0, count: bytes.count)
        exported.withUnsafeMutableBytes {
            context.render(ci, toBitmap: $0.baseAddress!, rowBytes: 127*4, bounds: ci.extent,
                           format: .BGRA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
        }
        let rawMean = bytes.enumerated().filter { $0.offset % 4 != 3 }.reduce(0.0) { $0+Double($1.element) } / Double(127*95*3)
        let exportedMean = exported.enumerated().filter { $0.offset % 4 != 3 }.reduce(0.0) { $0+Double($1.element) } / Double(127*95*3)
        XCTAssertEqual(exportedMean,rawMean,accuracy: 0.01,"export must preserve the drawable's sRGB brightness")
        let value: Float = 0.3
        let mapped = (value * (2.51*value+0.03)) / (value * (2.43*value+0.59)+0.14)
        let encoded = 1.055 * pow(mapped, 1/2.4) - 0.055
        for channel in 0..<3 {
            let mean = stride(from: channel, to: bytes.count, by: 4).reduce(Float(0)) { $0 + Float(bytes[$1])/255 } / Float(127*95)
            XCTAssertEqual(mean, encoded, accuracy: 0.003, "tone map and sRGB conversion must each occur once")
        }
    }

    func testDisablingEffectsReleasesOptionalTargetsAndResizeInvalidatesAO() throws {
        let h = try Harness(source: fixtureSource, width: 512, height: 384)
        let e = h.effects
        _ = try render(Fixture(), harness: h)
        let off = GPUSimRenderOptions(ambientOcclusion: false, contactShadows: false, screenSpaceReflections: false)
        XCTAssertFalse(try e.prepare(size: CGSize(width: 512, height: 384), options: off))
        XCTAssertNil(e.sceneColor); XCTAssertNil(e.sceneDepth); XCTAssertNil(e.reflection); XCTAssertNil(e.directVisibilityRaw)
        XCTAssertTrue(try e.prepare(size: CGSize(width: 319, height: 201), options: GPUSimRenderOptions(screenSpaceReflections: true)))
        XCTAssertFalse(e.aoIsWhite)
        XCTAssertEqual(e.depth?.width, 159); XCTAssertEqual(e.depth?.height, 100)
        XCTAssertEqual(e.sceneColor?.width, 319); XCTAssertEqual(e.sceneDepth?.height, 201)
    }

    private final class RaySceneFixture: GPUSimRenderableScene {
        let renderDevice: MTLDevice
        var values: [GPUSimRenderInstance]
        var renderBodyCount: Int { values.count }
        var renderRigidInstanceCount: Int { values.count }
        var rendererStateIsValid: Bool { true }
        var renderStateRevision: UInt64?
        var renderCameraHint: GPUSimRenderCameraHint { .init() }
        var softRenderSurface: GPUSimSoftRenderSurface?
        var skinnedRenderSurface: GPUSimSkinnedRenderSurface?
        var rigidMeshRenderSurface: GPUSimRigidMeshRenderSurface?
        var convexDebugRenderSurface: GPUSimConvexDebugRenderSurface? { nil }
        init(device: MTLDevice, values: [GPUSimRenderInstance]) { renderDevice = device; self.values = values }
        func encodeRenderInstances(_ commandBuffer: MTLCommandBuffer, instances: MTLBuffer,
            colorMode: GPUSimRenderColorMode, appearanceOverrides: MTLBuffer?) throws {
            guard !values.isEmpty else { return }
            let data = try XCTUnwrap(values.withUnsafeBytes {
                renderDevice.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
            })
            let b = try XCTUnwrap(commandBuffer.makeBlitCommandEncoder())
            b.copy(from: data, sourceOffset: 0, to: instances, destinationOffset: 0, size: data.length)
            b.endEncoding()
        }
    }

    func testDiffuseBounceIntegratesOffscreenRadianceAndRejectsStaleLighting() throws {
        let h = try Harness(source: fixtureSource, width: 257, height: 193)
        let device = h.effects.device
        guard device.supportsRaytracing else { throw XCTSkip("Metal ray tracing unavailable") }
        let scene = RaySceneFixture(device: device,values: [])
        let world = try RayTracingScene.shared(scene: scene)
        var f = Fixture(); f.boxMin.w=0; f.plane=SIMD4(0,0,1,-3)
        f.material=SIMD4(0.6,0.6,0.6,0); f.parameters.x=1
        func sample(frame: Int = 0) throws -> SIMD3<Float> {
            try world.prepare(scene: scene,revision: 0,auxiliary: [],ground: false)
            _ = try render(f,harness: h,world: world,rayScene: scene,diffuseFrame: frame)
            let bytes = try read(h.effects.diffuse!,queue: h.queue,bytesPerPixel: 8)
            var total = SIMD3<Float>.zero
            let w=h.effects.diffuse!.width, hgt=h.effects.diffuse!.height
            bytes.withUnsafeBytes { data in
                let values=data.bindMemory(to: UInt16.self)
                for y in 8..<(hgt-8) { for x in 8..<(w-8) {
                    let i=(y*w+x)*4
                    total += SIMD3(Float(Float16(bitPattern: values[i])),Float(Float16(bitPattern: values[i+1])),Float(Float16(bitPattern: values[i+2])))
                } }
            }
            return total/Float((w-16)*(hgt-16))+SIMD3(0.30,0.33,0.38)
        }
        let open=try sample()
        XCTAssertLessThan(simd_length(open-SIMD3(0.30,0.33,0.38)),0.002,"Sparse sky samples preserve mean irradiance")
        // An effectively infinite constant-radiance plane behind the camera.
        // Lambert's cosine integral / pi is exactly that incoming radiance.
        scene.values=[GPUSimRenderInstance(primitive: .box(size: SIMD3(10000,10000,0.01)),
            position: SIMD3(0,0,1),color: .zero,emissive: SIMD3(1.5,0.1,0.05))]
        for frame in 0..<8 { _ = try sample(frame: frame) }
        let red=try sample(frame: 8)
        XCTAssertEqual(red.x,1.5,accuracy: 0.025)
        XCTAssertEqual(red.y,0.1,accuracy: 0.01)
        XCTAssertEqual(red.z,0.05,accuracy: 0.01)
        scene.values[0].material=SIMD4(0.05,0.1,1.5,1)
        let blue=try sample(frame: 9)
        XCTAssertLessThan(blue.x,0.08,"Lighting changes must clip away old red history immediately")
        XCTAssertGreaterThan(blue.z,1.45)
        scene.values.append(GPUSimRenderInstance(primitive: .box(size: SIMD3(10000,10000,0.01)),
            position: SIMD3(0,0,-1),color: .zero))
        let blocked=try sample(frame: 10)
        XCTAssertLessThan(simd_reduce_max(blocked),0.012,"A thin opaque wall must block the offscreen emitter")
        scene.values=[]
        let reset=try sample(frame: 11)
        XCTAssertLessThan(simd_length(reset-open),0.012,"A reset must discard stale occluded lighting")
        let options=GPUSimRenderOptions.lightweight
        try h.effects.prepare(size: CGSize(width: 257,height: 193),options: options)
        XCTAssertNil(h.effects.diffuse); XCTAssertNil(h.effects.diffuseRaw)
        try h.effects.prepare(size: CGSize(width: 131,height: 97),options: .lightweight)
        XCTAssertNil(h.effects.diffuse,"Fast does not allocate GI textures")
    }


    func testDiffuseHistoryRejectsAReceiverDepthChange() throws {
        let h = try Harness(source: fixtureSource,width: 257,height: 193)
        let device=h.effects.device
        guard device.supportsRaytracing else { throw XCTSkip("Metal ray tracing unavailable") }
        let scene=RaySceneFixture(device: device,values: [GPUSimRenderInstance(
            primitive: .box(size: SIMD3(3,3,0.1)),position: SIMD3(0,0,1),color: .zero,emissive: SIMD3(2,0.02,0.01))])
        let world=try RayTracingScene.shared(scene: scene)
        try world.prepare(scene: scene,revision: 0,auxiliary: [],ground: false)
        var f=Fixture(); f.boxMin.w=0; f.plane=SIMD4(0,0,1,-1); f.material=SIMD4(0.5,0.5,0.5,0); f.parameters.x=1
        for frame in 0..<12 { _ = try render(f,harness: h,world: world,rayScene: scene,diffuseFrame: frame) }
        let old=try read(h.effects.diffuse!,queue: h.queue,bytesPerPixel: 8)
        f.plane.w = -3
        _ = try render(f,harness: h,world: world,rayScene: scene,diffuseFrame: 12)
        let moved=try read(h.effects.diffuse!,queue: h.queue,bytesPerPixel: 8)
        let disabled=GPUSimRenderOptions.lightweight
        try h.effects.prepare(size: CGSize(width: 257,height: 193),options: disabled)
        // Same phase and geometry, with independently empty lighting history.
        _ = try render(f,harness: h,world: world,rayScene: scene,diffuseFrame: 12)
        let fresh=try read(h.effects.diffuse!,queue: h.queue,bytesPerPixel: 8)
        XCTAssertEqual(moved,fresh,"A moved receiver must use fresh irradiance, not another surface's history")
        XCTAssertNotEqual(old,fresh,"The fixture must expose a real lighting change")
    }

    func testQualityBetaTracesOffscreenGeometryAndUpdatesRigidTransforms() throws {
        let h = try Harness(source: fixtureSource, width: 511, height: 383)
        guard h.effects.device.supportsRaytracing else { throw XCTSkip("Metal ray tracing is unavailable") }
        let scene = RaySceneFixture(device: h.effects.device, values: [
            GPUSimRenderInstance(primitive: .box(size: SIMD3(100,100,0.1)), position: SIMD3(0,0,-3.05), color: .zero),
            GPUSimRenderInstance(primitive: .box(size: SIMD3(0.7,0.7,0.6)), position: SIMD3(0,0,0.8), color: .zero,
                                 emissive: SIMD3(3,0.05,0.02))
        ])
        let world = try RayTracingScene.shared(scene: scene)
        XCTAssertTrue(world === (try RayTracingScene.shared(scene: scene)), "cameras share a world's geometry and queue")
        try world.prepare(scene: scene, revision: 0, auxiliary: [], ground: false)
        var f = Fixture(); f.boxMin.w = 0; f.plane = SIMD4(0,0,1,-3)
        let screenOnly = try render(f, harness: h, lightDirection: SIMD3(0,0,-1))
        XCTAssertEqual(screenOnly.0.min(), 1)
        XCTAssertTrue(screenOnly.1.allSatisfy { $0 == .zero })
        let traced = try render(f, harness: h, world: world, rayScene: scene, lightDirection: SIMD3(0,0,-1))
        XCTAssertGreaterThan(traced.0.filter { $0 < 0.5 }.count, 500, "a blocker behind the camera casts a world-space shadow")
        let width = h.effects.halfSize.x, height = h.effects.halfSize.y
        let inverse = uniforms(width: width, height: height).invViewProj
        var shadowDisagreements = 0
        for y in 0..<height { for x in 0..<width {
            let clip = inverse * SIMD4((Float(x)+0.5)/Float(width)*2-1,1-(Float(y)+0.5)/Float(height)*2,0,1)
            let direction = SIMD3(clip.x,clip.y,clip.z)/clip.w
            let point = direction * (-3/direction.z)
            let referenceShadow = abs(point.x) < 0.35 && abs(point.y) < 0.35
            if referenceShadow != (traced.0[y*width+x] < 0.5) { shadowDisagreements += 1 }
        } }
        XCTAssertLessThan(Float(shadowDisagreements)/Float(width*height),0.001,
                          "world visibility must match analytic rays through the offscreen box")
        let hits = traced.1.filter { $0.w > 0.5 && $0.x > 0.1 }
        XCTAssertGreaterThan(hits.count, 100, "a mirror reflects an emitter behind the camera")
        XCTAssertTrue(hits.allSatisfy { $0.x > $0.y && $0.x > $0.z })
        let shortcut = try render(f, harness: h, world: world, rayScene: scene,
                                  lightDirection: SIMD3(0,0,-1), screenShortcut: true)
        XCTAssertEqual(shortcut.1,traced.1,"screen misses must fall back to the same world-space reflection")
        scene.values[1].model.columns.3.x = 100
        try world.prepare(scene: scene, revision: 0, auxiliary: [], ground: false)
        let moved = try render(f, harness: h, world: world, rayScene: scene, lightDirection: SIMD3(0,0,-1))
        XCTAssertEqual(moved.0.min(), 1, "GPU instance transforms update visibility immediately")
        XCTAssertEqual(moved.1, try emptyReflections(f, harness: h), "moving an offscreen emitter leaves no reflection history")
    }

    func testLightingPresetsAndResourceTransitions() throws {
        XCTAssertEqual(GPUSimRenderOptions(), .lightweight)
        XCTAssertFalse(GPUSimRenderOptions.lightweight.usesHDR)
        XCTAssertFalse(GPUSimRenderOptions.lightweight.screenSpaceReflections)
        XCTAssertTrue(GPUSimRenderOptions.qualityBeta.usesRayTracing)
        XCTAssertFalse(GPUSimRenderOptions.qualityBeta.screenSpaceReflections, "SSR is an optional shortcut, separate from world RT")
        let h = try Harness(source: fixtureSource, width: 319, height: 201)
        let e = h.effects
        for options in [GPUSimRenderOptions.lightweight, .qualityBeta, .lightweight, .qualityBeta] {
            try e.prepare(size: CGSize(width: 319,height: 201), options: options)
            XCTAssertEqual(e.sceneColor != nil, options.usesHDR)
            XCTAssertEqual(e.aoRaw == nil, options.usesRayTracing)
            XCTAssertEqual(e.depthHierarchy != nil, options.screenSpaceReflections)
            XCTAssertNotNil(e.directVisibilityRaw)
        }
    }

    func testAccelerationUpdatesAreSharedAcrossCamerasAndInvalidatedByChanges() throws {
        let h = try Harness(source: fixtureSource, width: 255, height: 191)
        let device = h.effects.device
        guard device.supportsRaytracing else { throw XCTSkip("Metal ray tracing is unavailable") }
        let scene = RaySceneFixture(device: device, values: [
            GPUSimRenderInstance(primitive: .box(size: SIMD3(100,100,0.1)), position: SIMD3(0,0,-3.05), color: .zero),
            GPUSimRenderInstance(primitive: .box(size: SIMD3(0.7,0.7,0.6)), position: SIMD3(0,0,0.8), color: .zero,
                                 emissive: SIMD3(3,0.05,0.02))
        ])
        scene.renderStateRevision = 41
        let world = try RayTracingScene.shared(scene: scene)
        try world.prepare(scene: scene, revision: 0, auxiliary: [], ground: false)
        var f = Fixture(); f.boxMin.w = 0; f.plane = SIMD4(0,0,1,-3)
        let first = try render(f, harness: h, world: world, rayScene: scene, lightDirection: SIMD3(0,0,-1))
        XCTAssertGreaterThan(first.0.filter { $0 < 0.5 }.count, 100)
        XCTAssertEqual(world.lastUpdate, .init(primitiveBuilds: 1, instanceBuilds: 1), "rigid boxes share one cached mesh structure")
        let scratch = world.scratch, structure = world.structure
        XCTAssertEqual(scratch?.hazardTrackingMode, .tracked)

        let otherCamera = try Harness(source: fixtureSource, width: 319, height: 201)
        _ = try render(f, harness: otherCamera, eye: SIMD3(0.02,0,0), world: world, rayScene: scene, lightDirection: SIMD3(0,0,-1))
        XCTAssertEqual(world.lastUpdate, .init(), "another camera and staging buffer do not change world geometry")
        let command = try XCTUnwrap(world.queue.makeCommandBuffer())
        let instances = try XCTUnwrap(device.makeBuffer(length: scene.values.count*MemoryLayout<GPUSimRenderInstance>.stride, options: .storageModePrivate))
        for _ in 0..<130 {
            try world.encodeUpdate(command: command, scene: scene, instances: instances, auxiliary: nil, appearances: nil)
            XCTAssertEqual(world.lastUpdate, .init(), "camera draws must not consume the periodic rebuild budget")
        }
        command.commit(); command.waitUntilCompleted()

        scene.values[1].model.columns.3.x = 100
        scene.renderStateRevision = 42
        try world.prepare(scene: scene, revision: 0, auxiliary: [], ground: false)
        let moved = try render(f, harness: h, world: world, rayScene: scene, lightDirection: SIMD3(0,0,-1))
        XCTAssertEqual(moved.0.min(), 1)
        XCTAssertEqual(moved.1, try emptyReflections(f, harness: h))
        XCTAssertEqual(world.lastUpdate, .init(instanceRefits: 1), "rigid motion refits instances without touching mesh structures")
        XCTAssertTrue(world.structure === structure); XCTAssertTrue(world.scratch === scratch)

        scene.values.append(GPUSimRenderInstance(primitive: .sphere(radius: 0.4), position: SIMD3(0,0,0.8), color: .zero))
        try world.prepare(scene: scene, revision: 0, auxiliary: [], ground: false)
        let added = try render(f, harness: h, world: world, rayScene: scene, lightDirection: SIMD3(0,0,-1))
        XCTAssertGreaterThan(added.0.filter { $0 < 0.5 }.count, 100)
        XCTAssertEqual(world.lastUpdate, .init(primitiveBuilds: 2, instanceBuilds: 1))
        world.invalidateBuilds()
        _ = try render(f, harness: h, world: world, rayScene: scene, lightDirection: SIMD3(0,0,-1))
        XCTAssertEqual(world.lastUpdate, .init(primitiveBuilds: 2, instanceBuilds: 1), "an aborted update cannot leave a reusable state key")
        scene.renderStateRevision = nil
        _ = try render(f, harness: h, world: world, rayScene: scene, lightDirection: SIMD3(0,0,-1))
        XCTAssertEqual(world.lastUpdate, .init(instanceRefits: 1), "unknown backend revisions always update")
    }

    func testPausedAccelerationReuseTracksAuxiliaryTransformsAndShadowMasks() throws {
        let h = try Harness(source: fixtureSource, width: 255, height: 191)
        guard h.effects.device.supportsRaytracing else { throw XCTSkip("Metal ray tracing is unavailable") }
        let scene = RaySceneFixture(device: h.effects.device, values: [
            GPUSimRenderInstance(primitive: .box(size: SIMD3(100,100,0.1)), position: SIMD3(0,0,-3.05), color: .zero)
        ])
        scene.renderStateRevision = 0
        var auxiliary = [GPUSimRenderInstance(primitive: .box(size: SIMD3(0.7,0.7,0.6)), position: SIMD3(0,0,0.8),
                                              color: .zero, emissive: SIMD3(3,0,0), castsShadow: true)]
        let world = try RayTracingScene.shared(scene: scene)
        var f = Fixture(); f.boxMin.w = 0; f.plane = SIMD4(0,0,1,-3)
        func draw() throws -> ([Float], [SIMD4<Float>]) {
            try world.prepare(scene: scene, revision: 0, auxiliary: auxiliary, ground: false)
            return try render(f, harness: h, world: world, rayScene: scene, auxiliary: auxiliary, lightDirection: SIMD3(0,0,-1))
        }
        let first = try draw()
        XCTAssertGreaterThan(first.0.filter { $0 < 0.5 }.count, 100)
        _ = try draw()
        XCTAssertEqual(world.lastUpdate, .init(), "identical auxiliary data in a fresh staging buffer is reusable")
        auxiliary[0].parameters.w = 0
        let unshadowed = try draw()
        XCTAssertEqual(unshadowed.0.min(), 1)
        XCTAssertGreaterThan(unshadowed.1.filter { $0.x > 0.1 }.count, 100, "shadow masks must leave reflections visible")
        XCTAssertEqual(world.lastUpdate, .init(instanceRefits: 1))
        auxiliary[0].model.columns.3.x = 100
        let moved = try draw()
        XCTAssertEqual(moved.1, try emptyReflections(f, harness: h), "auxiliary motion must invalidate unchanged physics state")
        XCTAssertEqual(world.lastUpdate, .init(instanceRefits: 1))
    }

    func testPausedAccelerationReuseTracksDeformedSurfaceAppearances() throws {
        let h = try Harness(source: fixtureSource, width: 255, height: 191)
        let device = h.effects.device
        guard device.supportsRaytracing else { throw XCTSkip("Metal ray tracing is unavailable") }
        func buffer<T>(_ values: [T]) throws -> MTLBuffer {
            try XCTUnwrap(values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) })
        }
        let scene = RaySceneFixture(device: device, values: [
            GPUSimRenderInstance(primitive: .box(size: SIMD3(100,100,0.1)), position: SIMD3(0,0,-3.05), color: .zero),
            GPUSimRenderInstance(primitive: .sphere(radius: 0), position: SIMD3(100,0,0), color: .zero),
            GPUSimRenderInstance(primitive: .sphere(radius: 0), position: SIMD3(100,0,0), color: .zero)
        ])
        scene.renderStateRevision = 0
        scene.softRenderSurface = GPUSimSoftRenderSurface(triangles: try buffer([UInt32(0),1,2]), triangleCount: 1,
            positions: try buffer([SIMD4<Float>(-0.5,-0.5,0.8,0), SIMD4(0.5,-0.5,0.8,0), SIMD4(0,0.5,0.8,0)]),
            normals: try buffer(Array(repeating: SIMD4<Float>(0,0,-1,0), count: 3)))
        let world = try RayTracingScene.shared(scene: scene)
        try world.prepare(scene: scene, revision: 0, auxiliary: [], ground: false)
        var f = Fixture(); f.boxMin.w = 0; f.plane = SIMD4(0,0,1,-3)
        let red = Dictionary(uniqueKeysWithValues: (0..<3).map { ($0, GPUSimRenderAppearance(color: .zero, emissive: SIMD3(3,0,0))) })
        let blue = Dictionary(uniqueKeysWithValues: (0..<3).map { ($0, GPUSimRenderAppearance(color: .zero, emissive: SIMD3(0,0,3))) })
        let first = try render(f, harness: h, world: world, rayScene: scene, appearances: red, lightDirection: SIMD3(0,0,-1))
        XCTAssertGreaterThan(first.1.filter { $0.x > 0.1 && $0.x > $0.z }.count, 100)
        _ = try render(f, harness: h, world: world, rayScene: scene, appearances: red, lightDirection: SIMD3(0,0,-1))
        XCTAssertEqual(world.lastUpdate, .init())
        let changed = try render(f, harness: h, world: world, rayScene: scene, appearances: blue, lightDirection: SIMD3(0,0,-1))
        XCTAssertGreaterThan(changed.1.filter { $0.z > 0.1 && $0.z > $0.x }.count, 100, "paused cameras must not reuse stale baked surface materials")
        XCTAssertEqual(world.lastUpdate, .init(primitiveRefits: 1, instanceRefits: 1))
    }


    func testRayTracedGroundDoesNotShadowItselfAsCameraMoves() throws {
        let h = try Harness(source: fixtureSource, width: 2240, height: 1520)
        let e = h.effects, device = e.device
        guard device.supportsRaytracing else { throw XCTSkip("Metal ray tracing is unavailable") }
        try e.prepare(size: CGSize(width: 2240,height: 1520),options: .qualityBeta)
        let scene = RaySceneFixture(device: device, values: [])
        let world = try RayTracingScene.shared(scene: scene)
        try world.prepare(scene: scene, revision: 0, auxiliary: [], ground: true)
        let library = try device.makeLibrary(source: renderShaderSource, options: nil)
        let p = MTLRenderPipelineDescriptor()
        p.vertexFunction = library.makeFunction(name: "floor_vertex")
        p.fragmentFunction = library.makeFunction(name: "floor_prepass_fragment")
        p.colorAttachments[0].pixelFormat = .rgba16Float
        p.depthAttachmentPixelFormat = .depth32Float
        let pipeline = try device.makeRenderPipelineState(descriptor: p)
        let renderer = try GPUSimRenderer(device: device)
        let instances = try XCTUnwrap(device.makeBuffer(length: MemoryLayout<GPUSimRenderInstance>.stride, options: .storageModePrivate))
        for offset: Float in [0, 0.001, 0.01, 0.1, 1, 7.999, 8, 8.001, 20] {
            let eye = SIMD3<Float>(offset, -2, 1)
            renderer.setCamera(position: eye, target: eye + SIMD3(0.2, 1, -0.1))
            let projection = renderer.projectionMatrix(aspect: Float(e.halfSize.x)/Float(e.halfSize.y))
            let vp = projection * renderer.viewMatrix
            var U = uniforms(width: e.halfSize.x, height: e.halfSize.y, eye: eye)
            U.viewProj = vp; U.invViewProj = vp.inverse
            U.screen.z = Float(e.halfSize.y)*projection.columns.1.y*0.5
            U.aoProjection = SIMD4(-projection.columns.2.z, projection.columns.3.z,
                                  1/projection.columns.0.x, 1/projection.columns.1.y)
            U.lightDir = SIMD4(normalize(SIMD3<Float>(0.4, 0.25, -0.85)), 0)
            let inverseView = renderer.viewMatrix.inverse
            U.camRight = inverseView.columns.0; U.camUp = -inverseView.columns.1
            let cmd = try XCTUnwrap(world.queue.makeCommandBuffer())
            try world.encodeUpdate(command: cmd, scene: scene, instances: instances, auxiliary: nil, appearances: nil)
            let materialClear = MTLRenderPassDescriptor()
            materialClear.colorAttachments[0].texture = e.material
            materialClear.colorAttachments[0].clearColor = MTLClearColorMake(0,0,0,0)
            materialClear.colorAttachments[0].loadAction = .clear; materialClear.colorAttachments[0].storeAction = .store
            let clearEncoder = try XCTUnwrap(cmd.makeRenderCommandEncoder(descriptor: materialClear)); clearEncoder.endEncoding()
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = e.normal
            pass.colorAttachments[0].loadAction = .clear; pass.colorAttachments[0].storeAction = .store
            pass.depthAttachment.texture = e.depth; pass.depthAttachment.clearDepth = 1
            pass.depthAttachment.loadAction = .clear; pass.depthAttachment.storeAction = .store
            let enc = try XCTUnwrap(cmd.makeRenderCommandEncoder(descriptor: pass))
            enc.setRenderPipelineState(pipeline); enc.setDepthStencilState(h.depthState)
            enc.setVertexBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: floorVertexCount); enc.endEncoding()
            try world.encodeLighting(command: cmd, uniforms: U, screen: e, instances: instances,
                                     auxiliary: nil, appearances: nil, reflections: false)
            try world.encodeDiffuse(command: cmd,uniforms: U,screen: e,instances: instances,auxiliary: nil,appearances: nil)
            cmd.commit(); cmd.waitUntilCompleted()
            XCTAssertEqual(cmd.status, .completed, "\(String(describing: cmd.error))")
            let visibility = try read(e.directVisibilityRaw!, queue: world.queue, bytesPerPixel: 1)
            let falseShadows = visibility.filter { $0 < 255 }.count
            XCTAssertEqual(falseShadows, 0, "An empty ground plane cannot cast shadows on itself; camera x=\(offset)")
            let bounce = try read(e.diffuseRaw!,queue: world.queue,bytesPerPixel: 8)
            bounce.withUnsafeBytes { raw in
                let values = raw.bindMemory(to: UInt16.self)
                let falseBounces = stride(from: 0,to: values.count,by: 4).filter {
                    values[$0] != 0 || values[$0+1] != 0 || values[$0+2] != 0
                }.count
                XCTAssertGreaterThan(stride(from: 0,to: values.count,by: 4).filter { values[$0+3] != 0 }.count,1000)
                XCTAssertEqual(falseBounces,0,"An open floor must receive analytic sky, without self-bounce noise; camera x=\(offset)")
            }
            let depthData = try read(e.depth!, queue: world.queue, bytesPerPixel: 4)
            depthData.withUnsafeBytes { raw in
                let depths = raw.bindMemory(to: Float.self)
                let right = SIMD3(U.camRight.x, U.camRight.y, U.camRight.z)
                let up = SIMD3(U.camUp.x, U.camUp.y, U.camUp.z)
                var missingFloor = 0, maxNearError: Float = 0
                for y in 0..<e.halfSize.y { for x in 0..<e.halfSize.x {
                    let uv = SIMD2((Float(x)+0.5)/Float(e.halfSize.x), (Float(y)+0.5)/Float(e.halfSize.y))
                    let ray = right*((uv.x*2-1)*U.aoProjection.z)
                        + up*((uv.y*2-1)*U.aoProjection.w) + cross(right, up)
                    let distance = (0.005-eye.z)/ray.z
                    let reference = eye + ray*distance
                    let depth = depths[y*e.halfSize.x+x]
                    if distance > 0.1 && distance < 999 && abs(reference.x) < 3999 && abs(reference.y) < 3999 {
                        if depth >= 1 { missingFloor += 1 }
                        if distance < 20 {
                            let point = U.invViewProj * SIMD4(uv.x*2-1, 1-uv.y*2, depth, 1)
                            maxNearError = max(maxNearError, abs(point.z/point.w - 0.005))
                        }
                    }
                } }
                XCTAssertEqual(missingFloor, 0, "Floor rings must cover the same plane without cracks")
                XCTAssertLessThan(maxNearError, 0.0002, "Within 20 m, raster depth must stay within 0.2 mm of the floor")
            }
        }
    }


    func testAnalyticGroundParticipatesInClosestHitsAndVisibility() throws {
        let h = try Harness(source: fixtureSource, width: 257, height: 193)
        guard h.effects.device.supportsRaytracing else { throw XCTSkip("Metal ray tracing unavailable") }
        let scene = RaySceneFixture(device: h.effects.device, values: [])
        let world = try RayTracingScene.shared(scene: scene)
        // A mirror below the floor sees it behind the camera. A nearer red
        // emitter then replaces that ground hit, exercising closest-hit order.
        var fixture = Fixture(); fixture.boxMin.w = 0
        fixture.plane = SIMD4(0,0,1,-3); fixture.parameters.x = 0.02
        try world.prepare(scene: scene, revision: 0, auxiliary: [], ground: true)
        let floor = try render(fixture, harness: h, world: world, rayScene: scene, lightDirection: SIMD3(0,0,-1))
        XCTAssertTrue(floor.0.allSatisfy { $0 < 0.5 }, "The ground blocks rays crossing its plane")
        XCTAssertGreaterThan(floor.1.filter { $0.w > 0.9 }.count, floor.1.count*9/10)
        scene.values = [GPUSimRenderInstance(primitive: .box(size: SIMD3(100,100,0.01)),
            position: SIMD3(0,0,-1), color: .zero, emissive: SIMD3(3,0,0))]
        try world.prepare(scene: scene, revision: 0, auxiliary: [], ground: true)
        let closer = try render(fixture, harness: h, world: world, rayScene: scene)
        XCTAssertGreaterThan(closer.1.filter { $0.x > 0.5 && $0.y < 0 && $0.w > 0.9 }.count, closer.1.count*9/10,
                             "Closer world geometry must replace the analytic ground radiance")
        scene.values = []
        for enabled in [false, true] {
            try world.prepare(scene: scene, revision: 0, auxiliary: [], ground: enabled)
            let empty = try render(fixture, harness: h, eye: SIMD3(enabled ? 4500 : 0,0,0),
                world: world, rayScene: scene, lightDirection: SIMD3(0,0,-1))
            XCTAssertTrue(empty.0.allSatisfy { $0 == 1 })
            XCTAssertEqual(empty.1, try emptyReflections(fixture, harness: h, eye: SIMD3(enabled ? 4500 : 0,0,0)), "Disabled ground and rays outside its finite bounds must match an empty world")
        }
    }

    func testRayTracingUpdatesRigidMeshesAndRefitsDeformingSurfaces() throws {
        let h = try Harness(source: fixtureSource, width: 255, height: 191)
        let device = h.effects.device
        guard device.supportsRaytracing else { throw XCTSkip("Metal ray tracing is unavailable") }
        func buffer<T>(_ values: [T]) throws -> MTLBuffer {
            try XCTUnwrap(values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) })
        }
        let points = [SIMD4<Float>(-0.5,-0.5,0.8,0), SIMD4(0.5,-0.5,0.8,0), SIMD4(0,0.5,0.8,0)]
        var fixture = Fixture(); fixture.boxMin.w = 0; fixture.plane = SIMD4(0,0,1,-3)
        for kind in 0..<3 {
            let scene = RaySceneFixture(device: device, values: [
                GPUSimRenderInstance(primitive: .box(size: SIMD3(100,100,0.1)), position: SIMD3(0,0,-3.05), color: .zero)
            ])
            let indices = try buffer([UInt32(0),1,2])
            let moving: MTLBuffer
            if kind == 0 {
                moving = try buffer([SIMD4<Float>.zero])
                let vertices = try buffer(points.map { GPUSimRigidMeshRenderVertex(positionBody: $0, normal: SIMD4(0,0,-1,0.45), color: SIMD4(1,0.1,0.1,0)) })
                scene.rigidMeshRenderSurface = GPUSimRigidMeshRenderSurface(vertices: vertices, indices: indices, indexCount: 3,
                    positions: moving, rotations: try buffer([SIMD4<Float>(0,0,0,1)]))
            } else if kind == 1 {
                moving = try buffer(points)
                scene.softRenderSurface = GPUSimSoftRenderSurface(triangles: indices, triangleCount: 1, positions: moving,
                    normals: try buffer(Array(repeating: SIMD4<Float>(0,0,-1,0), count: 3)))
            } else {
                moving = try buffer(points.map { GPUSimSkinRenderVertex(position: $0, normal: SIMD4(0,0,-1,0)) })
                scene.skinnedRenderSurface = GPUSimSkinnedRenderSurface(triangles: indices, triangleCount: 1, vertices: moving)
            }
            let world = try RayTracingScene.shared(scene: scene)
            try world.prepare(scene: scene, revision: 0, auxiliary: [], ground: false)
            let before = try render(fixture, harness: h, world: world, rayScene: scene, lightDirection: SIMD3(0,0,-1))
            XCTAssertGreaterThan(before.0.filter { $0 < 0.5 }.count, 100, "mesh kind \(kind) must cast a shadow")
            let originalGeometry = world.vertices
            if kind == 0 {
                moving.contents().assumingMemoryBound(to: SIMD4<Float>.self).pointee.x = 100
            } else if kind == 1 {
                let p = moving.contents().assumingMemoryBound(to: SIMD4<Float>.self)
                for i in 0..<3 { p[i].x += 100 }
            } else {
                let p = moving.contents().assumingMemoryBound(to: GPUSimSkinRenderVertex.self)
                for i in 0..<3 { p[i].position.x += 100 }
            }
            try world.prepare(scene: scene, revision: 0, auxiliary: [], ground: false)
            XCTAssertTrue(originalGeometry === world.vertices, "pose changes must not re-extract topology")
            let moved = try render(fixture, harness: h, world: world, rayScene: scene, lightDirection: SIMD3(0,0,-1))
            XCTAssertEqual(moved.0.min(), 1, "mesh kind \(kind) must update/refit without stale shadows")
            XCTAssertEqual(moved.1, try emptyReflections(fixture, harness: h))
            scene.values = []; scene.rigidMeshRenderSurface = nil; scene.softRenderSurface = nil; scene.skinnedRenderSurface = nil
            try world.prepare(scene: scene, revision: 1, auxiliary: [], ground: false)
            let empty = try render(fixture, harness: h, world: world, rayScene: scene, lightDirection: SIMD3(0,0,-1))
            XCTAssertEqual(empty.0.min(), 1)
            XCTAssertEqual(empty.1, try emptyReflections(fixture, harness: h), "reset to an empty scene must retire all old geometry")
        }
    }

}
