import Metal
import Foundation
import simd

/// Owns shared surfaces and the two screen-space stages: visibility before
/// lighting, then radiance effects before display conversion. Geometry stays
/// with the renderer; effects share projection, allocation and pass encoding.
final class ScreenSpacePipeline {
    enum Failure: Error { case allocation(String), encoder(String), shaderFunction(String) }
    let device: MTLDevice
    private let aoPipeline, temporalPipeline, visibilityPipeline: MTLRenderPipelineState
    private let contactPipeline, reflectionPipeline, reflectionFilterPipeline, compositePipeline: MTLRenderPipelineState
    private let depthCopyPipeline, depthReducePipeline: MTLComputePipelineState
    private var depthLevels: [MTLTexture] = []
    private(set) var depthHierarchy: MTLTexture?
    private let white: MTLTexture
    private let noDepth: MTLDepthStencilState
    private(set) var size = SIMD2<Int>(0, 0)
    private(set) var halfSize = SIMD2<Int>(0, 0)
    private(set) var depth, normal, material: MTLTexture?
    private(set) var visibility, aoRaw, aoResolved, aoHistory, previousDepth: MTLTexture?
    private(set) var directVisibilityRaw, reflection, reflectionRaw, sceneColor, sceneMSAA, sceneDepth: MTLTexture?
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
        temporalPipeline = try pipeline("temporal_fragment", format: .rgba8Unorm)
        visibilityPipeline = try pipeline("visibility_fragment", format: .rg8Unorm)
        contactPipeline = try pipeline("contact_fragment", format: .r8Unorm)
        reflectionPipeline = try pipeline("reflection_fragment", format: .rgba16Float)
        reflectionFilterPipeline = try pipeline("reflection_filter_fragment", format: .rgba16Float)
        compositePipeline = try pipeline("screen_composite_fragment", format: .bgra8Unorm_srgb, samples: GPUSimRenderer.sampleCount)
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
        if format == .r8Unorm || format == .rgba16Float { d.usage.insert(.shaderWrite) }
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
            let r = try texture(.rgba8Unorm, width: half.x, height: half.y, label: "AO resolve")
            let h = try texture(.rgba8Unorm, width: half.x, height: half.y, label: "AO history")
            let p = try texture(.depth32Float, width: half.x, height: half.y, label: "Previous screen depth")
            (size, halfSize) = (next, half)
            (depth, normal, visibility, aoRaw, aoResolved, aoHistory, previousDepth) = (d, n, a, raw, r, h, p)
            directVisibilityRaw = nil; material = nil; reflection = nil; reflectionRaw = nil
            sceneColor = nil; sceneMSAA = nil; sceneDepth = nil; depthHierarchy = nil; depthLevels = []
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
        return resized
    }

    /// Fullscreen effects declare only their output and ordered inputs. Metal
    /// tracks the dependencies within the frame's existing command buffer.
    private func pass(_ pipeline: MTLRenderPipelineState, command: MTLCommandBuffer,
                      output: MTLTexture, inputs: [MTLTexture], uniforms: Uniforms) throws {
        let d = MTLRenderPassDescriptor()
        d.colorAttachments[0].texture = output
        d.colorAttachments[0].loadAction = .dontCare
        d.colorAttachments[0].storeAction = .store
        guard let e = command.makeRenderCommandEncoder(descriptor: d) else { throw Failure.encoder(pipeline.label ?? "effect") }
        e.label = pipeline.label
        e.setRenderPipelineState(pipeline)
        var u = uniforms
        e.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        for (i, t) in inputs.enumerated() { e.setFragmentTexture(t, index: i) }
        e.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        e.endEncoding()
    }

    func encodeBeforeLighting(command: MTLCommandBuffer, uniforms: Uniforms, options: GPUSimRenderOptions) throws {
        if options.ambientOcclusion {
            try pass(aoPipeline, command: command, output: aoRaw!, inputs: [depth!, normal!], uniforms: uniforms)
            try pass(temporalPipeline, command: command, output: aoResolved!,
                     inputs: [aoRaw!, aoHistory!, depth!, previousDepth!, normal!], uniforms: uniforms)
            guard let b = command.makeBlitCommandEncoder() else { throw Failure.encoder("AO history") }
            b.label = "AO history"
            b.copy(from: aoResolved!, to: aoHistory!)
            b.copy(from: depth!, to: previousDepth!)
            b.endEncoding()
            aoIsWhite = false
        }
        if options.contactShadows && !options.usesRayTracing {
            try pass(contactPipeline, command: command, output: directVisibilityRaw!, inputs: [depth!, normal!], uniforms: uniforms)
        }
        if options.ambientOcclusion || options.contactShadows || options.usesRayTracing {
            try pass(visibilityPipeline, command: command, output: visibility!,
                inputs: [options.ambientOcclusion ? aoResolved! : white, directVisibilityRaw ?? white, depth!, normal!], uniforms: uniforms)
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
        try pass(reflectionFilterPipeline, command: command, output: reflection!,
                 inputs: [reflectionRaw!, depth!, normal!], uniforms: uniforms)
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
