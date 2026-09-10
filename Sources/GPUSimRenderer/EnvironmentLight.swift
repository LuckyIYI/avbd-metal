import Metal
import simd

/// Immutable prepared image-based lighting, independently shared across renderers.
/// The texture is an equirectangular radiance map: longitude atan2(y,x), north +Z,
/// top-left UV origin. Floating-point formats preserve HDR; sRGB textures decode
/// through Metal's sampler. Intensity multiplies linear radiance, not encoded RGB.
@MainActor
public final class GPUSimEnvironmentLight {
  public enum Failure: Error {
    case invalidTexture, invalidParameters, allocation
    case preparation(String)
  }
  public let texture: MTLTexture
  let irradiance: MTLBuffer

  /// Nine cosine-convolved spherical-harmonic coefficients are computed once.
  /// `diffuseSamples` affects environment preparation only, never per-frame cost.
  public init(
    device: MTLDevice, texture: MTLTexture, diffuseSamples: Int = 4096
  ) throws {
    guard texture.device.registryID == device.registryID, texture.textureType == .type2D,
      texture.sampleCount == 1, texture.storageMode != .memoryless,
      texture.usage == .unknown || texture.usage.contains(.shaderRead)
    else { throw Failure.invalidTexture }
    guard (64...65536).contains(diffuseSamples)
    else { throw Failure.invalidParameters }
    self.texture = texture
    guard let buffer = device.makeBuffer(length: 9 * 16, options: .storageModeShared),
      let queue = device.makeCommandQueue(), let command = queue.makeCommandBuffer()
    else { throw Failure.allocation }
    irradiance = buffer
    let library = try device.makeLibrary(source: Self.preparationSource, options: nil)
    guard let function = library.makeFunction(name: "environment_sh") else {
      throw Failure.allocation
    }
    let pipeline = try device.makeComputePipelineState(function: function)
    guard let e = command.makeComputeCommandEncoder() else { throw Failure.allocation }
    var count = UInt32(diffuseSamples)
    e.setComputePipelineState(pipeline)
    e.setTexture(texture, index: 0)
    e.setBuffer(buffer, offset: 0, index: 0)
    e.setBytes(&count, length: 4, index: 1)
    e.dispatchThreads(
      .init(width: 9, height: 1, depth: 1),
      threadsPerThreadgroup: .init(width: 9, height: 1, depth: 1))
    e.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
      throw Failure.preparation(String(describing: command.error))
    }
  }

  nonisolated static let basisSource = """
    inline float environmentBasis(uint i,float3 n) {
        switch(i) {
        case 0:return 0.282094792;
        case 1:return 0.488602512*n.y;
        case 2:return 0.488602512*n.z;
        case 3:return 0.488602512*n.x;
        case 4:return 1.092548431*n.x*n.y;
        case 5:return 1.092548431*n.y*n.z;
        case 6:return 0.315391565*(3*n.z*n.z-1);
        case 7:return 1.092548431*n.x*n.z;
        default:return 0.546274215*(n.x*n.x-n.y*n.y);
        }
    }
    """
  private static let preparationSource = """
    #include <metal_stdlib>
    using namespace metal;
    \(basisSource)
    kernel void environment_sh(texture2d<float> map [[texture(0)]],device float4* result [[buffer(0)]],
        constant uint& count [[buffer(1)]],uint coefficient [[thread_position_in_grid]]) {
        constexpr sampler s(coord::normalized,s_address::repeat,t_address::clamp_to_edge,filter::linear);
        float3 total=float3(0);
        for(uint i=0;i<count;++i) {
            float z=1-2*(float(i)+0.5)/float(count);
            float phi=float(i)*2.399963229728653;
            float3 n=float3(sqrt(max(0.0,1-z*z))*float2(cos(phi),sin(phi)),z);
            float2 uv=float2(atan2(n.y,n.x)/(2*M_PI_F)+0.5,acos(clamp(n.z,-1.0,1.0))/M_PI_F);
            float3 radiance=max(map.sample(s,uv,level(0)).rgb,float3(0));
            total += select(float3(0),radiance,isfinite(radiance))*environmentBasis(coefficient,n);
        }
        float convolution=coefficient==0 ? 1 : (coefficient<4 ? 2.0/3.0 : 0.25);
        result[coefficient]=float4(total*(4*M_PI_F/float(count))*convolution,0);
    }
    """
}

/// Per-renderer bindings combine shared materials with independently owned lighting.
/// They are immutable, so replacing lighting cannot alter a submitted GPU frame.
@MainActor
final class GPUSimLightingBindings {
  let materials: GPUSimMaterialLibrary
  let environment: GPUSimEnvironmentLight?
  let arguments: MTLBuffer
  let textures: [MTLTexture]
  let textureBytes: Int

  init(materials: GPUSimMaterialLibrary, environment: GPUSimEnvironmentLight?) throws {
    self.materials = materials
    self.environment = environment
    var textures = materials.textures
    var bytes = materials.textureBytes
    if let environment {
      guard materials.usesArgumentBuffers else { throw GPUSimMaterialLibrary.Failure.unsupportedDevice }
      guard environment.texture.device.registryID == materials.device.registryID else {
        throw GPUSimRendererError.sceneDeviceMismatch
      }
      if !textures.contains(where: { $0 === environment.texture }) {
        guard environment.texture.allocatedSize <= max(0, materials.textureBudget - bytes) else {
          throw GPUSimMaterialLibrary.Failure.textureBudgetExceeded
        }
        textures.append(environment.texture)
        bytes += environment.texture.allocatedSize
      }
      arguments = try GPUSimMaterialLibrary.makeArguments(device: materials.device,
        records: materials.records, textures: materials.textures, count: materials.materialCount,
        environmentIrradiance: materials.environmentIrradiance, environment: environment)
    } else {
      arguments = materials.arguments
    }
    self.textures = textures
    textureBytes = bytes
  }

  func bind(_ encoder: MTLRenderCommandEncoder) {
    encoder.setFragmentBuffer(arguments, offset: 0, index: 10)
    guard materials.usesArgumentBuffers else { return }
    encoder.useResource(materials.records, usage: .read, stages: .fragment)
    encoder.useResource(environment?.irradiance ?? materials.environmentIrradiance, usage: .read, stages: .fragment)
    if !textures.isEmpty { encoder.useResources(textures, usage: .read, stages: .fragment) }
  }

  func bind(_ encoder: MTLComputeCommandEncoder) {
    encoder.setBuffer(arguments, offset: 0, index: 10)
    guard materials.usesArgumentBuffers else { return }
    encoder.useResource(materials.records, usage: .read)
    encoder.useResource(environment?.irradiance ?? materials.environmentIrradiance, usage: .read)
    if !textures.isEmpty { encoder.useResources(textures, usage: .read) }
  }
}
