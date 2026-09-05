import Metal
import Foundation
import simd

/// Owns camera surfaces, visibility and radiance reconstruction. Geometry stays
/// with the renderer; effects share projection, allocation and pass encoding.
final class ScreenSpacePipeline {
    enum Failure: Error { case allocation(String), encoder(String), shaderFunction(String) }
    let device: MTLDevice
    private let aoPipeline, temporalPipeline, visibilityPipeline: MTLRenderPipelineState
    private let contactPipeline, reflectionPipeline, reflectionFilterPipeline, compositePipeline: MTLRenderPipelineState
    private let diffuseTemporalPipeline: MTLRenderPipelineState
    private let diffuseFilterPipeline: MTLRenderPipelineState
    private let antialiasingPipeline: MTLRenderPipelineState
    private let depthCopyPipeline, depthReducePipeline: MTLComputePipelineState
    private var depthLevels: [MTLTexture] = []
    private(set) var depthHierarchy: MTLTexture?
    private let white: MTLTexture
    private let noDepth: MTLDepthStencilState
    private(set) var size = SIMD2<Int>(0, 0)
    private(set) var halfSize = SIMD2<Int>(0, 0)
    private(set) var depth, normal, material: MTLTexture?
    private(set) var visibility, aoRaw, aoResolved, aoHistory: MTLTexture?
    private(set) var directVisibilityRaw, reflection, reflectionRaw, sceneColor, sceneMSAA, sceneDepth: MTLTexture?
    private(set) var reflectionScratch: MTLTexture?
    private(set) var diffuse, diffuseRaw, diffuseScratch: MTLTexture?
    private var diffuseHistory: MTLTexture?
    private(set) var displayColor: MTLTexture?
    private var displayEncoded: MTLTexture?
    private(set) var aoIsWhite = false
    private var visibilityIsWhite = false

    init(device: MTLDevice, library: MTLLibrary) throws {
        self.device = device
        func function(_ name: String) throws -> MTLFunction {
            guard let result = library.makeFunction(name: name) else { throw Failure.shaderFunction(name) }
            return result
        }
        func pipeline(_ name: String, format: MTLPixelFormat, samples: Int = 1) throws -> MTLRenderPipelineState {
            let d = MTLRenderPipelineDescriptor()
            d.label = name
            d.vertexFunction = try function("fs_vertex")
            d.fragmentFunction = try function(name)
            d.colorAttachments[0].pixelFormat = format
            d.rasterSampleCount = samples
            if samples > 1 { d.depthAttachmentPixelFormat = .depth32Float }
            return try device.makeRenderPipelineState(descriptor: d)
        }
        depthCopyPipeline = try device.makeComputePipelineState(function: function("screen_depth_copy"))
        depthReducePipeline = try device.makeComputePipelineState(function: function("screen_depth_reduce"))
        aoPipeline = try pipeline("gtao_fragment", format: .r8Unorm)
        temporalPipeline = try pipeline("temporal_fragment", format: .r16Float)
        visibilityPipeline = try pipeline("visibility_fragment", format: .rg8Unorm)
        contactPipeline = try pipeline("contact_fragment", format: .r8Unorm)
        reflectionPipeline = try pipeline("reflection_fragment", format: .rgba16Float)
        reflectionFilterPipeline = try pipeline("reflection_filter_fragment", format: .rgba16Float)
        compositePipeline = try pipeline("screen_composite_fragment", format: .bgra8Unorm_srgb, samples: GPUSimRenderer.sampleCount)
        diffuseTemporalPipeline = try pipeline("diffuse_temporal_fragment", format: .rgba16Float)
        diffuseFilterPipeline = try pipeline("diffuse_filter_fragment", format: .rgba16Float)
        antialiasingPipeline = try pipeline("edge_antialiasing_fragment", format: .bgra8Unorm_srgb, samples: GPUSimRenderer.sampleCount)
        let ds = MTLDepthStencilDescriptor()
        ds.depthCompareFunction = .always
        ds.isDepthWriteEnabled = false
        guard let state = device.makeDepthStencilState(descriptor: ds) else { throw Failure.allocation("depth state") }
        noDepth = state
        let wd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg8Unorm, width: 1, height: 1, mipmapped: false)
        wd.storageMode = .shared; wd.usage = .shaderRead
        guard let w = device.makeTexture(descriptor: wd) else { throw Failure.allocation("neutral visibility") }
        white = w
        var bytes: [UInt8] = [255, 255]
        bytes.withUnsafeMutableBytes { w.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: 2) }
    }

    func texture(_ format: MTLPixelFormat, width: Int, height: Int, label: String,
                 mipmapped: Bool = false, samples: Int = 1) throws -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format,
            width: width, height: height, mipmapped: mipmapped)
        d.usage = [.renderTarget, .shaderRead]
        if format == .r8Unorm || format == .rgba16Float || format == .rgba32Float { d.usage.insert(.shaderWrite) }
        d.storageMode = .private
        if samples > 1 {
            d.textureType = .type2DMultisample
            d.sampleCount = samples
            d.usage = .renderTarget
            if device.supportsFamily(.apple1) { d.storageMode = .memoryless }
        }
        guard let result = device.makeTexture(descriptor: d) else { throw Failure.allocation(label) }
        result.label = label
        return result
    }

    @discardableResult
    func prepare(size requested: CGSize, options: GPUSimRenderOptions) throws -> Bool {
        let next = SIMD2(max(Int(requested.width), 4), max(Int(requested.height), 4))
        let resized = next != size
        if resized {
            // Allocate into locals before publishing a complete surface set.
            let half = SIMD2(max(next.x / 2, 4), max(next.y / 2, 4))
            let d = try texture(.depth32Float, width: half.x, height: half.y, label: "Screen depth")
            let n = try texture(.rgba16Float, width: half.x, height: half.y, label: "Screen normal / roughness")
            let a = try texture(.rg8Unorm, width: half.x, height: half.y, label: "Ambient / direct visibility")
            let raw = try texture(.r8Unorm, width: half.x, height: half.y, label: "Raw AO")
            let r = try texture(.r16Float, width: half.x, height: half.y, label: "AO resolve")
            let h = try texture(.r16Float, width: half.x, height: half.y, label: "AO history")
            (size, halfSize) = (next, half)
            (depth, normal, visibility, aoRaw, aoResolved, aoHistory) = (d, n, a, raw, r, h)
            directVisibilityRaw = nil; material = nil; reflection = nil; reflectionRaw = nil
            reflectionScratch = nil
            sceneColor = nil; sceneMSAA = nil; sceneDepth = nil; depthHierarchy = nil; depthLevels = []
            displayColor = nil; displayEncoded = nil
            diffuse = nil; diffuseRaw = nil; diffuseScratch = nil
            diffuseHistory = nil
            aoIsWhite = false
            visibilityIsWhite = false
        }
        if options.contactShadows || options.usesRayTracing, directVisibilityRaw == nil {
            directVisibilityRaw = try texture(.r8Unorm, width: halfSize.x, height: halfSize.y, label: "Direct visibility")
        } else if !options.contactShadows && !options.usesRayTracing { directVisibilityRaw = nil }
        if options.usesHDR, sceneColor == nil || (sceneColor!.mipmapLevelCount > 1) != options.screenSpaceReflections {
            let m = try texture(.rgba8Unorm, width: halfSize.x, height: halfSize.y, label: "Screen albedo / metallic")
            let r = try texture(.rgba16Float, width: halfSize.x, height: halfSize.y, label: "Reflection radiance correction")
            let raw = try texture(.rgba16Float, width: halfSize.x, height: halfSize.y, label: "Raw reflected radiance")
            let c = try texture(.rgba16Float, width: size.x, height: size.y, label: "HDR scene", mipmapped: options.screenSpaceReflections)
            let s = try texture(.rgba16Float, width: size.x, height: size.y, label: "Transient HDR MSAA", samples: GPUSimRenderer.sampleCount)
            let d = try texture(.depth32Float, width: size.x, height: size.y, label: "Resolved scene depth")
            var hierarchy: MTLTexture?
            var levels: [MTLTexture] = []
            if options.screenSpaceReflections {
                let hd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: size.x, height: size.y, mipmapped: true)
                hd.mipmapLevelCount = min(hd.mipmapLevelCount, 7)
                hd.usage = [.shaderRead, .shaderWrite, .pixelFormatView]; hd.storageMode = .private
                guard let pyramid = device.makeTexture(descriptor: hd) else { throw Failure.allocation("depth hierarchy") }
                for level in 0..<hd.mipmapLevelCount {
                    guard let view = pyramid.makeTextureView(pixelFormat: .r32Float, textureType: .type2D,
                        levels: level..<(level+1), slices: 0..<1) else { throw Failure.allocation("depth mip") }
                    levels.append(view)
                }
                pyramid.label = "Screen depth hierarchy"
                hierarchy = pyramid
            }
            (material, reflection, reflectionRaw, sceneColor, sceneMSAA, sceneDepth) = (m, r, raw, c, s, d)
            depthHierarchy = hierarchy; depthLevels = levels
        } else if !options.usesHDR {
            material = nil; reflection = nil; reflectionRaw = nil; sceneColor = nil; sceneMSAA = nil; sceneDepth = nil
            depthHierarchy = nil; depthLevels = []
        }
        if options.usesRayTracing, reflectionScratch == nil {
            reflectionScratch = try texture(.rgba16Float, width: halfSize.x, height: halfSize.y, label: "Reflection filter scratch")
        } else if !options.usesRayTracing { reflectionScratch = nil }
        if options.usesDiffuseGI, diffuse == nil {
            let width = (halfSize.x+1)/2, height = (halfSize.y+1)/2
            let a = try texture(.rgba16Float, width: width, height: height, label: "Diffuse irradiance correction")
            let b = try texture(.rgba16Float, width: width, height: height, label: "Raw diffuse irradiance")
            let c = try texture(.rgba16Float, width: width, height: height, label: "Diffuse filter scratch")
            let h = try texture(.rgba16Float, width: width, height: height, label: "Diffuse lighting history")
            (diffuse, diffuseRaw, diffuseScratch, diffuseHistory) = (a,b,c,h)
        } else if !options.usesDiffuseGI {
            diffuse = nil; diffuseRaw = nil; diffuseScratch = nil
            diffuseHistory = nil
        }
        if options.edgeAntialiasing, displayColor == nil {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm_srgb, width: size.x, height: size.y, mipmapped: false)
            d.usage = [.renderTarget, .shaderRead, .pixelFormatView]
            d.storageMode = .private
            guard let color = device.makeTexture(descriptor: d),
                  let encoded = color.makeTextureView(pixelFormat: .bgra8Unorm) else { throw Failure.allocation("display AA color") }
            color.label = "Display color before edge AA"
            displayColor = color; displayEncoded = encoded
        } else if !options.edgeAntialiasing { displayColor = nil; displayEncoded = nil }
        return resized
    }

    func displayPass(using destination: MTLRenderPassDescriptor) -> MTLRenderPassDescriptor {
        let d = destination.copy() as! MTLRenderPassDescriptor
        d.colorAttachments[0].resolveTexture = displayColor
        return d
    }

    func encodeAntialiasing(command: MTLCommandBuffer, destination: MTLRenderPassDescriptor) throws {
        let d = destination.copy() as! MTLRenderPassDescriptor
        d.colorAttachments[0].loadAction = .dontCare
        d.depthAttachment.loadAction = .dontCare
        guard let e = command.makeRenderCommandEncoder(descriptor: d) else { throw Failure.encoder("edge AA") }
        e.label = "Display edge antialiasing"
        e.setRenderPipelineState(antialiasingPipeline)
        e.setDepthStencilState(noDepth)
        e.setFragmentTexture(displayEncoded, index: 0)
        e.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        e.endEncoding()
    }

    /// Fullscreen effects declare only their output and ordered inputs. Metal
    /// tracks the dependencies within the frame's existing command buffer.
    private func pass(_ pipeline: MTLRenderPipelineState, command: MTLCommandBuffer,
                      output: MTLTexture, inputs: [MTLTexture], uniforms: Uniforms,
                      parameters: SIMD4<Float>? = nil) throws {
        let d = MTLRenderPassDescriptor()
        d.colorAttachments[0].texture = output
        d.colorAttachments[0].loadAction = .dontCare
        d.colorAttachments[0].storeAction = .store
        guard let e = command.makeRenderCommandEncoder(descriptor: d) else { throw Failure.encoder(pipeline.label ?? "effect") }
        e.label = pipeline.label
        e.setRenderPipelineState(pipeline)
        var u = uniforms
        e.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        if var parameters { e.setFragmentBytes(&parameters, length: MemoryLayout<SIMD4<Float>>.stride, index: 2) }
        for (i, t) in inputs.enumerated() { e.setFragmentTexture(t, index: i) }
        e.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        e.endEncoding()
    }

    func encodeBeforeLighting(command: MTLCommandBuffer, uniforms: Uniforms, options: GPUSimRenderOptions) throws {
        if options.ambientOcclusion && uniforms.temporal.z<=64 {
            try pass(aoPipeline, command: command, output: aoRaw!, inputs: [depth!, normal!], uniforms: uniforms)
            if uniforms.temporal.z>0 {
                try pass(temporalPipeline, command: command, output: aoResolved!,
                         inputs: [aoRaw!, aoHistory!], uniforms: uniforms)
                guard let b = command.makeBlitCommandEncoder() else { throw Failure.encoder("AO history") }
                b.label = "AO history"
                b.copy(from: aoResolved!, to: aoHistory!)
                b.endEncoding()
            }
            aoIsWhite = false
        }
        if options.contactShadows && !options.usesRayTracing {
            try pass(contactPipeline, command: command, output: directVisibilityRaw!, inputs: [depth!, normal!], uniforms: uniforms)
        }
        if options.ambientOcclusion || options.contactShadows || options.usesRayTracing {
            try pass(visibilityPipeline, command: command, output: visibility!,
                inputs: [options.ambientOcclusion ? (uniforms.temporal.z>0 ? aoResolved! : aoRaw!) : white, directVisibilityRaw ?? white, depth!, normal!], uniforms: uniforms)
        } else if !visibilityIsWhite {
            // A fullscreen clear initializes both ambient and direct visibility channels.
            let d = MTLRenderPassDescriptor()
            d.colorAttachments[0].texture = visibility
            d.colorAttachments[0].loadAction = .clear
            d.colorAttachments[0].clearColor = MTLClearColor(red: 1, green: 1, blue: 1, alpha: 1)
            d.colorAttachments[0].storeAction = .store
            guard let e = command.makeRenderCommandEncoder(descriptor: d) else { throw Failure.encoder("clear visibility") }
            e.endEncoding()
        }
        aoIsWhite = !options.ambientOcclusion
        visibilityIsWhite = !options.ambientOcclusion && !options.contactShadows && !options.usesRayTracing
    }

    func encodeDiffuseFilter(command: MTLCommandBuffer, uniforms: Uniforms) throws {
        guard uniforms.temporal.z<=64 else { return }
        for (step, source, destination) in [(1, diffuseRaw!, diffuseScratch!), (2, diffuseScratch!, diffuse!)] {
            try pass(diffuseFilterPipeline, command: command, output: destination,
                     inputs: [source, depth!, normal!], uniforms: uniforms, parameters: SIMD4(Float(step),0,0,0))
        }
        try pass(diffuseTemporalPipeline, command: command, output: diffuseScratch!,
                 inputs: [diffuse!, diffuseHistory!], uniforms: uniforms)
        swap(&diffuse, &diffuseScratch)
        guard let blit = command.makeBlitCommandEncoder() else { throw Failure.encoder("diffuse history") }
        blit.copy(from: diffuse!, to: diffuseHistory!)
        blit.endEncoding()
    }

    func scenePass(using destination: MTLRenderPassDescriptor) -> MTLRenderPassDescriptor {
        let d = destination.copy() as! MTLRenderPassDescriptor
        d.colorAttachments[0].texture = sceneMSAA
        d.colorAttachments[0].resolveTexture = sceneColor
        d.colorAttachments[0].loadAction = .clear
        d.colorAttachments[0].storeAction = .multisampleResolve
        d.depthAttachment.resolveTexture = sceneDepth
        d.depthAttachment.depthResolveFilter = .min
        // Keep multisample depth for diagnostic/translucent draws after SSR.
        d.depthAttachment.storeAction = .storeAndMultisampleResolve
        return d
    }

    func encodeReflections(command: MTLCommandBuffer, uniforms: Uniforms, filter: Bool = true) throws {
        guard let c = command.makeComputeCommandEncoder() else { throw Failure.encoder("depth hierarchy") }
        c.label = "Screen depth hierarchy"
        for (index, level) in depthLevels.enumerated() {
            c.setComputePipelineState(index == 0 ? depthCopyPipeline : depthReducePipeline)
            c.setTexture(index == 0 ? sceneDepth : depthLevels[index-1], index: 0)
            c.setTexture(level, index: 1)
            c.dispatchThreads(MTLSize(width: level.width, height: level.height, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            c.memoryBarrier(scope: .textures)
        }
        c.endEncoding()
        guard let b = command.makeBlitCommandEncoder() else { throw Failure.encoder("scene color pyramid") }
        b.label = "Scene radiance pyramid"
        b.generateMipmaps(for: sceneColor!)
        b.endEncoding()
        try pass(reflectionPipeline, command: command, output: reflectionRaw!,
                 inputs: [depth!, normal!, material!, sceneColor!, sceneDepth!, visibility!, depthHierarchy!], uniforms: uniforms)
        if filter { try encodeReflectionFilter(command: command, uniforms: uniforms) }
    }

    func encodeReflectionFilter(command: MTLCommandBuffer, uniforms: Uniforms) throws {
        if uniforms.rayTracing.x > 0 {
            guard let scratch = reflectionScratch else { throw Failure.allocation("reflection filter scratch") }
            // Increasing tap spacing suppresses low-sample blotches without
            // retaining old geometry or reflected lighting across frames.
            for (input, output, parameters) in [
                (reflectionRaw!, reflection!, SIMD4<Float>(1, 1, 0, 0)),
                (reflection!, scratch, SIMD4<Float>(2, 0, 0, 0)),
                (scratch, reflection!, SIMD4<Float>(4, 0, 1, 0))
            ] {
                try pass(reflectionFilterPipeline, command: command, output: output,
                    inputs: [input, depth!, normal!, material!], uniforms: uniforms, parameters: parameters)
            }
        } else {
            try pass(reflectionFilterPipeline, command: command, output: reflection!,
                inputs: [reflectionRaw!, depth!, normal!, material!], uniforms: uniforms, parameters: SIMD4(1, 1, 1, 0))
        }
    }

    /// Returns the open display encoder so overlays share the tone-mapped MSAA
    /// pass. Translucency is composed afterward and never becomes an SSR receiver.
    func beginComposite(command: MTLCommandBuffer, destination: MTLRenderPassDescriptor,
                        uniforms: Uniforms) throws -> MTLRenderCommandEncoder {
        let d = destination.copy() as! MTLRenderPassDescriptor
        d.colorAttachments[0].loadAction = .dontCare
        d.depthAttachment.loadAction = .load
        guard let e = command.makeRenderCommandEncoder(descriptor: d) else { throw Failure.encoder("display composite") }
        e.label = "Reflections and display composite"
        e.setRenderPipelineState(compositePipeline)
        e.setDepthStencilState(noDepth)
        var u = uniforms
        e.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        for (i, t) in [sceneColor!, reflection!, depth!, normal!, sceneDepth!].enumerated() { e.setFragmentTexture(t, index: i) }
        e.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        return e
    }
}
