import Metal
import MetalFX
import simd

/// Shared temporal reconstruction and upscaling. GPU snapshots refer to the last
/// submitted frame, not the previous simulation step, and stay on its queue.
final class MetalFXReconstruction {
    enum Failure: Error { case allocation(String), encoder, unsupported }
    static let guideFormats: [MTLPixelFormat] = [.rg16Float, .rgba16Float, .rgba16Float, .rgba16Float, .r16Float, .rgba8Unorm]
    let size: SIMD2<Int>
    let outputSize: SIMD2<Int>
    let denoising: Bool
    let material: MTLTexture
    let depth, motion, normal, diffuseAlbedo, specularAlbedo, roughness, output: MTLTexture
    private let device: MTLDevice
    private let encodeScaler: (MTLCommandBuffer, MTLTexture, SIMD2<Float>, Bool, simd_float4x4, simd_float4x4) -> Void
    private var snapshots: [String: MTLBuffer] = [:]
    private var sources: [String: MTLBuffer] = [:]
    private(set) var reset = true
    private var frame: UInt32 = 0
    private var view = matrix_identity_float4x4
    private var projection = matrix_identity_float4x4
    private var previousCamera: simd_float4x4?
    private var previousOptions: GPUSimRenderOptions?
    private(set) var jitter = SIMD2<Float>.zero
    private(set) var guideUniforms = GuideUniforms()
    struct GuideUniforms {
        var current = matrix_identity_float4x4
        var previous = matrix_identity_float4x4
        var size = SIMD4<Float>.zero
    }

    static func supports(device: MTLDevice, denoising: Bool) -> Bool {
        if denoising {
            if #available(macOS 26.0, iOS 26.0, *) {
                return MTLFXTemporalDenoisedScalerDescriptor.supportsDevice(device)
            }
            return false
        }
        return MTLFXTemporalScalerDescriptor.supportsDevice(device)
    }

    init(device: MTLDevice, size: SIMD2<Int>, denoising: Bool, outputSize: SIMD2<Int>? = nil, sharedDepth: MTLTexture? = nil, sharedNormal: MTLTexture? = nil, sharedMaterial: MTLTexture? = nil) throws {
        self.device = device; self.size = size; self.denoising = denoising
        let outputSize = outputSize ?? size
        self.outputSize = outputSize
        guard Self.supports(device: device, denoising: denoising) else { throw Failure.unsupported }
        func texture(_ format: MTLPixelFormat, _ usage: MTLTextureUsage, _ label: String, output: Bool = false) throws -> MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: output ? outputSize.x : size.x, height: output ? outputSize.y : size.y, mipmapped: false)
            d.storageMode = .private; d.usage = usage.union([.shaderRead, .renderTarget])
            guard let t = device.makeTexture(descriptor: d) else { throw Failure.allocation(label) }
            t.label = label; return t
        }
        material = try sharedMaterial ?? texture(.rgba8Unorm, [], "Reconstruction material")
        if denoising, #available(macOS 26.0, iOS 26.0, *) {
            let d = MTLFXTemporalDenoisedScalerDescriptor()
            d.inputWidth = size.x; d.inputHeight = size.y
            d.outputWidth = outputSize.x; d.outputHeight = outputSize.y
            d.colorTextureFormat = .rgba16Float; d.outputTextureFormat = .rgba16Float
            d.depthTextureFormat = .depth32Float; d.motionTextureFormat = .rg16Float
            d.normalTextureFormat = .rgba16Float; d.diffuseAlbedoTextureFormat = .rgba16Float
            d.specularAlbedoTextureFormat = .rgba16Float; d.roughnessTextureFormat = .r16Float
            d.isAutoExposureEnabled = false
            guard let scaler = d.makeTemporalDenoisedScaler(device: device) else { throw Failure.unsupported }
            depth = try sharedDepth ?? texture(.depth32Float, scaler.depthTextureUsage, "MetalFX depth")
            motion = try texture(.rg16Float, scaler.motionTextureUsage, "MetalFX motion")
            normal = try sharedNormal ?? texture(.rgba16Float, scaler.normalTextureUsage, "MetalFX normals")
            diffuseAlbedo = try texture(.rgba16Float, scaler.diffuseAlbedoTextureUsage, "MetalFX diffuse albedo")
            specularAlbedo = try texture(.rgba16Float, scaler.specularAlbedoTextureUsage, "MetalFX specular albedo")
            roughness = try texture(.r16Float, scaler.roughnessTextureUsage, "MetalFX roughness")
            output = try texture(.rgba16Float, scaler.outputTextureUsage, "MetalFX reconstructed HDR", output: true)
            scaler.depthTexture = depth; scaler.motionTexture = motion; scaler.normalTexture = normal
            scaler.diffuseAlbedoTexture = diffuseAlbedo; scaler.specularAlbedoTexture = specularAlbedo
            scaler.roughnessTexture = roughness; scaler.outputTexture = output
            scaler.motionVectorScaleX = 1; scaler.motionVectorScaleY = 1
            scaler.preExposure = 1
            encodeScaler = { command, color, jitter, reset, view, projection in
                scaler.colorTexture = color; scaler.jitterOffsetX = jitter.x; scaler.jitterOffsetY = jitter.y
                scaler.shouldResetHistory = reset; scaler.worldToViewMatrix = view; scaler.viewToClipMatrix = projection
                scaler.encode(commandBuffer: command)
            }
        } else {
            let d = MTLFXTemporalScalerDescriptor()
            d.inputWidth = size.x; d.inputHeight = size.y; d.outputWidth = outputSize.x; d.outputHeight = outputSize.y
            d.colorTextureFormat = .rgba16Float; d.outputTextureFormat = .rgba16Float
            d.depthTextureFormat = .depth32Float; d.motionTextureFormat = .rg16Float
            d.isInputContentPropertiesEnabled = true; d.inputContentMinScale = 1; d.inputContentMaxScale = 2
            d.isAutoExposureEnabled = false
            guard let scaler = d.makeTemporalScaler(device: device) else { throw Failure.unsupported }
            depth = try sharedDepth ?? texture(.depth32Float, scaler.depthTextureUsage, "MetalFX depth")
            motion = try texture(.rg16Float, scaler.motionTextureUsage, "MetalFX motion")
            normal = try sharedNormal ?? texture(.rgba16Float, [], "Reconstruction normals")
            diffuseAlbedo = try texture(.rgba16Float, [], "Reconstruction diffuse albedo")
            specularAlbedo = try texture(.rgba16Float, [], "Reconstruction specular albedo")
            roughness = try texture(.r16Float, [], "Reconstruction roughness")
            output = try texture(.rgba16Float, scaler.outputTextureUsage, "MetalFX reconstructed HDR", output: true)
            scaler.depthTexture = depth; scaler.motionTexture = motion; scaler.outputTexture = output
            scaler.motionVectorScaleX = 1; scaler.motionVectorScaleY = 1; scaler.preExposure = 1
            encodeScaler = { command, color, jitter, reset, view, projection in
                scaler.colorTexture = color; scaler.jitterOffsetX = jitter.x; scaler.jitterOffsetY = jitter.y
                scaler.reset = reset; scaler.encode(commandBuffer: command)
            }
        }
    }

    static func sampleJitter(_ frame: UInt32) -> SIMD2<Float> {
        func radicalInverse(_ n: UInt32, _ base: UInt32) -> Float {
            var n = n; var weight: Float = 1; var result: Float = 0
            while n > 0 { weight /= Float(base); result += Float(n % base) * weight; n /= base }
            return result
        }
        let n = frame % 32 + 1
        return SIMD2(radicalInverse(n, 2), radicalInverse(n, 3)) - 0.5
    }

    func beginFrame(camera: simd_float4x4, view: simd_float4x4 = matrix_identity_float4x4, projection: simd_float4x4 = matrix_identity_float4x4, options: GPUSimRenderOptions, invalidate: Bool) -> simd_float4x4 {
        self.view = view; self.projection = projection
        reset = invalidate || previousCamera == nil || previousOptions != options
        if reset { frame = 0 }
        sources.removeAll(keepingCapacity: true)
        jitter = Self.sampleJitter(frame)
        guideUniforms = GuideUniforms(current: camera, previous: reset ? camera : previousCamera!,
                                     size: SIMD4(Float(size.x), Float(size.y), 0, 0))
        // Shift homogeneous clip coordinates by the requested pixel jitter.
        var shift = matrix_identity_float4x4
        shift.columns.3.x = 2 * jitter.x / Float(size.x)
        shift.columns.3.y = -2 * jitter.y / Float(size.y)
        previousCamera = camera; previousOptions = options; frame &+= 1
        return shift * camera
    }

    func previous(_ source: MTLBuffer, key: String, command: MTLCommandBuffer) throws -> MTLBuffer {
        sources[key] = source
        if snapshots[key]?.length != source.length {
            guard let buffer = device.makeBuffer(length: source.length, options: .storageModePrivate),
                  let b = command.makeBlitCommandEncoder() else { throw Failure.allocation("previous geometry") }
            buffer.label = "Previous rendered \(key)"
            b.copy(from: source, sourceOffset: 0, to: buffer, destinationOffset: 0, size: source.length)
            b.endEncoding(); snapshots[key] = buffer; reset = true
        }
        return snapshots[key]!
    }

    func finishFrame(command: MTLCommandBuffer, color: MTLTexture) throws {
        encodeScaler(command, color, jitter, reset, view, projection)
        guard let b = command.makeBlitCommandEncoder() else { throw Failure.encoder }
        b.label = "Retain previous rendered geometry"
        for (key, source) in sources {
            b.copy(from: source, sourceOffset: 0, to: snapshots[key]!, destinationOffset: 0, size: source.length)
        }
        b.endEncoding()
        snapshots = snapshots.filter { sources[$0.key] != nil }
    }

    func guidePass() -> MTLRenderPassDescriptor {
        let d = MTLRenderPassDescriptor()
        for (i, t) in [motion, normal, diffuseAlbedo, specularAlbedo, roughness, material].enumerated() {
            d.colorAttachments[i].texture = t; d.colorAttachments[i].loadAction = .clear; d.colorAttachments[i].storeAction = .store
        }
        d.depthAttachment.texture = depth; d.depthAttachment.loadAction = .clear
        d.depthAttachment.storeAction = .store; d.depthAttachment.clearDepth = 1
        return d
    }
}

let reconstructionShaderSource = """
struct ReconstructionUniforms { float4x4 current; float4x4 previous; float4 size; };
struct ReconstructionOut {
    float2 motion [[color(0)]];
    float4 normal [[color(1)]];
    float4 diffuse [[color(2)]];
    float4 specular [[color(3)]];
    float roughness [[color(4)]];
    float4 material [[color(5)]];
};
inline ReconstructionOut reconstructionGuides(float3 world, float3 previous, float3 n, float3 albedo,
                                              float roughness, float metal, constant ReconstructionUniforms& R) {
    float4 c = R.current * float4(world, 1);
    float4 p = R.previous * float4(previous, 1);
    ReconstructionOut o;
    o.motion = p.w > 1e-5 && c.w > 1e-5 ? (p.xy / p.w - c.xy / c.w) * R.size.xy * float2(0.5, -0.5) : float2(0);
    o.normal = float4(normalize(n), roughness);
    o.material = float4(albedo, metal);
    o.diffuse = float4(albedo * (1.0-metal), 1);
    o.specular = float4(mix(float3(0.04), albedo, metal), 1);
    o.roughness = roughness;
    return o;
}
fragment ReconstructionOut reconstruction_fragment(VOut in [[stage_in]], constant Uniforms& U [[buffer(1)]],
                                                     constant ReconstructionUniforms& R [[buffer(2)]]) {
    float3 n = normalize(in.normal);
    if (in.flatShade > 0.5) n = normalize(cross(dfdx(in.world), dfdy(in.world)));
    if (dot(n, U.eye.xyz-in.world)<0) n = -n;
    return reconstructionGuides(in.world, in.previousWorld, n, in.albedo, clamp(in.pbr.x,0.02,1.0), saturate(in.pbr.y), R);
}
fragment ReconstructionOut soft_reconstruction_fragment(VOut in [[stage_in]], constant Uniforms& U [[buffer(1)]], constant ReconstructionUniforms& R [[buffer(2)]]) {
    float3 n = normalize(in.normal);
    if (in.flatShade > 0.5) n = normalize(cross(dfdx(in.world),dfdy(in.world)));
    if (dot(n,U.eye.xyz-in.world)<0) n = -n;
    return reconstructionGuides(in.world,in.previousWorld,n,in.albedo,0.72,0,R);
}
fragment ReconstructionOut floor_reconstruction_fragment(FloorOut in [[stage_in]], constant Uniforms& U [[buffer(1)]], constant ReconstructionUniforms& R [[buffer(2)]]) {
    float3 albedo = floorAlbedo(in.world,U);
    // The checker floor is diffuse-only; its guide must not advertise a specular lobe.
    ReconstructionOut result = reconstructionGuides(in.world, in.world, float3(0,0,1), albedo, 1, 0, R);
    result.specular = float4(0);
    return result;
}
struct ReconstructionDisplayOut { float4 color [[color(0)]]; float depth [[depth(any)]]; };
fragment ReconstructionDisplayOut reconstruction_display_fragment(FSOut in [[stage_in]], constant Uniforms& U [[buffer(1)]],
    texture2d<float> color [[texture(0)]], depth2d<float> depth [[texture(4)]]) {
    ReconstructionDisplayOut o;
    o.color = float4(displayColorSRGB8(acesTonemap(max(color.read(uint2(in.position.xy)).rgb,0.0)),in.position.xy),1);
    // Restore opaque depth for native-resolution diagnostic/translucent overlays.
    float2 uv = clamp(in.uv + U.reconstruction.yz, 0.0, 1.0);
    uint2 p = min(uint2(uv*float2(depth.get_width(),depth.get_height())),uint2(depth.get_width()-1,depth.get_height()-1));
    o.depth = depth.read(p);
    return o;
}
"""
