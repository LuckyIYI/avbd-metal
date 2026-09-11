import Metal

/// A caller-authored display program. `radiance` is scene-linear RGB after
/// exposure; `lut` is the optional 3D texture. Return linear display RGB in 0...1.
/// Existing exact sRGB encode/decode helpers are available in the program.
public struct GPUSimDisplayProgram: Sendable, Hashable {
  public let body: String
  public let supportingSource: String
  public init(body: String, supportingSource: String = "") {
    self.body = body
    self.supportingSource = supportingSource
  }
}

/// Display-only processing after HDR reconstruction. The caller may supply
/// an OCIO-generated Metal program and its lookup table. The renderer has no
/// dependency on OCIO or on any particular application color configuration.
@MainActor
public final class GPUSimDisplayTransform {
  public enum Failure: Error {
    case invalidTexture, invalidParameters, textureBudgetExceeded, programMismatch
  }
  public let texture: MTLTexture?
  public let program: GPUSimDisplayProgram
  let device: MTLDevice

  public init(
    device: MTLDevice, program: GPUSimDisplayProgram,
    texture: MTLTexture? = nil, textureBudget: Int = 64 * 1024 * 1024
  ) throws {
    guard !program.body.isEmpty,
      program.body.utf8.count + program.supportingSource.utf8.count <= 256 * 1024,
      textureBudget >= 0
    else { throw Failure.invalidParameters }
    if let texture {
      guard texture.device.registryID == device.registryID, texture.textureType == .type3D,
        (2...129).contains(texture.width), texture.width == texture.height,
        texture.width == texture.depth, texture.mipmapLevelCount == 1,
        texture.sampleCount == 1, texture.storageMode != .memoryless,
        [.rgba16Float, .rgba32Float].contains(texture.pixelFormat),
        texture.usage == .unknown || texture.usage.contains(.shaderRead)
      else { throw Failure.invalidTexture }
      guard texture.allocatedSize <= textureBudget else { throw Failure.textureBudgetExceeded }
    }
    self.device = device
    self.texture = texture
    self.program = program
  }
}

func makeDisplayTransformShaderSource(_ program: GPUSimDisplayProgram?) -> String {
  guard let program else {
    return """
      inline float3 displayTonemap(float3 radiance,constant Uniforms& U,texture3d<float> lut) {
          return displayTonemap(radiance,U);
      }
      """
  }
  return """
    namespace gpuSimDisplay {
      \(program.supportingSource)
      inline float3 evaluate(float3 radiance,texture3d<float> lut) {
        \(program.body)
      }
    }
    inline float3 displayTonemap(float3 radiance,constant Uniforms& U,texture3d<float> lut) {
        if (U.displaySettings.y==0) return displayTonemap(radiance,U);
        return clamp(gpuSimDisplay::evaluate(radiance*exp2(U.displaySettings.x),lut),0.0,1.0);
    }
    """
}
