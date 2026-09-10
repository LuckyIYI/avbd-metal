import Foundation
import ImageIO
import Metal
import MetalKit
import ModelIO
import simd

/// A trusted, caller-authored Metal function body. Programs are compiled once with
/// the renderer, never accepted from a scene/network payload. `context` provides
/// position, UV, footprint and parameters; `surface` contains linear PBR channels.
/// Do not use fragment derivatives: this function also runs at ray intersections.
public struct GPUSimMaterialProgram: Sendable {
  public let body: String
  /// Helper functions/constants, scoped to this program to avoid symbol collisions.
  public let supportingSource: String
  public init(body: String, supportingSource: String = "") {
    self.body = body
    self.supportingSource = supportingSource
  }
}

/// Metallic/roughness material with optional HQ dielectric transmission.
/// Texture RGB is multiplied by its linear factor and vertex/body appearance tint.
/// Encoded color/emission images use an sRGB Metal format (linear HDR is also valid); scalar/normal maps use
/// linear formats. A nil texture leaves the factor unchanged. UV origin is top-left.
public struct GPUSimSurfaceMaterial {
  public var baseColor = SIMD3<Float>(repeating: 1)
  public var roughness: Float = 1
  public var metallic: Float = 0
  /// Experimental optical approximations, separate from the shared opaque PBR model.
  public var previewOptics = GPUSimPreviewOptics()
  public var emission = SIMD3<Float>.zero
  public var normalScale: Float = 1
  /// Select packed scalar channels (0 red, 1 green, 2 blue, 3 alpha).
  public var roughnessChannel: UInt32 = 0
  public var metallicChannel: UInt32 = 0
  public var invertNormalGreen = false
  public var baseColorTexture: MTLTexture?
  public var roughnessTexture: MTLTexture?
  public var metallicTexture: MTLTexture?
  public var normalTexture: MTLTexture?
  public var emissionTexture: MTLTexture?
  public var uvScale = SIMD2<Float>(repeating: 1)
  public var uvOffset = SIMD2<Float>.zero
  public var clampToEdge = false
  /// One-based program index, or zero for ordinary image/factor materials.
  public var program: UInt32 = 0
  public var parameters = SIMD4<Float>.zero
  public init() {}
}

/// Immutable, device-owned material resources shared by raster and ray shading.
/// Mesh material IDs are one-based; zero retains vertex color/PBR. Independent
/// image sizes/formats/mip chains are retained without atlas packing or resampling.
@MainActor
public final class GPUSimMaterialLibrary {
  public enum Failure: Error {
    case invalidImage, unsupportedDevice, allocation, tooManyTextures
    case invalidMaterial(Int)
    case incompatibleTexture(String)
    case textureBudgetExceeded
  }
  /// Distinct texture objects one library can bind. The argument-buffer layout,
  /// the MSL texture array and the validation guard all derive from this value.
  public nonisolated static let textureCapacity = 128
  /// Largest image dimension accepted by the loaders before decoding.
  public nonisolated static let maximumImageDimension = 16384
  /// Mirrors MSL `MaterialRecord` in `makeSurfaceMaterialShaderSource`; 128 bytes.
  struct Record {
    var color: SIMD4<Float>
    var emission: SIMD4<Float>
    var uv: SIMD4<Float>
    var maps: SIMD4<UInt32>
    var extra: SIMD4<UInt32>
    var parameters: SIMD4<Float>
    var channels: SIMD4<UInt32>
    var optics: SIMD4<Float>
  }
  let device: MTLDevice
  let programs: [GPUSimMaterialProgram]
  public let hasTransmission: Bool
  let environmentIrradiance: MTLBuffer
  let textureBudget: Int
  let records: MTLBuffer
  let textures: [MTLTexture]
  let arguments: MTLBuffer
  /// False on argument-buffer Tier 1 devices, where the shader declares a stub
  /// resource struct and the library must stay empty.
  let usesArgumentBuffers: Bool
  private struct ShaderKey: Hashable { let motionGuides, rays: Bool; let displayProgram: GPUSimDisplayProgram? }
  private var shaderLibraries: [ShaderKey: MTLLibrary] = [:]
  func shaderLibrary(motionGuides: Bool = false, rays: Bool = false, displayProgram: GPUSimDisplayProgram? = nil) throws -> MTLLibrary {
    let key = ShaderKey(motionGuides: motionGuides, rays: rays, displayProgram: displayProgram)
    if let library = shaderLibraries[key] { return library }
    let source =
      makeRenderShaderSource(
        motionGuides: motionGuides, programs: programs, argumentBuffers: usesArgumentBuffers, displayProgram: displayProgram)
      + (rays ? "\n" + rayTracingShaderSource : "")
    let library = try device.makeLibrary(source: source, options: nil)
    shaderLibraries[key] = library
    return library
  }
  public let materialCount: Int
  public let textureBytes: Int

  /// Validated, deduplicated library contents. Allocates no Metal objects.
  struct Prepared {
    var records: [Record]
    var textures: [MTLTexture]
    var bytes: Int
  }

  /// Validates material records, programs, and their textures without creating
  /// GPU resources. A renderer validates its environment and the combined texture
  /// budget separately when installing lighting.
  public static func validate(
    materials: [GPUSimSurfaceMaterial], programs: [GPUSimMaterialProgram] = [],
    device: MTLDevice, textureBudget: Int = 512 * 1024 * 1024
  ) throws {
    _ = try prepare(
      materials: materials, programs: programs, device: device, textureBudget: textureBudget)
  }

  static func prepare(
    materials: [GPUSimSurfaceMaterial], programs: [GPUSimMaterialProgram],
    device: MTLDevice, textureBudget: Int
  ) throws -> Prepared {
    var textures: [MTLTexture] = []
    var indices: [ObjectIdentifier: UInt32] = [:]
    var bytes = 0
    func index(_ texture: MTLTexture?) throws -> UInt32 {
      guard let texture else { return UInt32.max }
      if let old = indices[ObjectIdentifier(texture)] { return old }
      guard texture.device.registryID == device.registryID, texture.textureType == .type2D,
        texture.sampleCount == 1, texture.usage == .unknown || texture.usage.contains(.shaderRead),
        texture.storageMode != .memoryless
      else {
        throw Failure.incompatibleTexture(texture.label ?? "unnamed texture")
      }
      guard textures.count < textureCapacity else { throw Failure.tooManyTextures }
      guard texture.allocatedSize <= max(0, textureBudget - bytes) else {
        throw Failure.textureBudgetExceeded
      }
      bytes += texture.allocatedSize
      let result = UInt32(textures.count)
      indices[ObjectIdentifier(texture)] = result
      textures.append(texture)
      return result
    }
    var values: [Record] = []
    for (i, m) in materials.enumerated() {
      let floats = [
        m.baseColor.x, m.baseColor.y, m.baseColor.z, m.roughness, m.metallic,
        m.emission.x, m.emission.y, m.emission.z, m.normalScale,
        m.uvScale.x, m.uvScale.y, m.uvOffset.x, m.uvOffset.y,
        m.parameters.x, m.parameters.y, m.parameters.z, m.parameters.w,
        m.previewOptics.transmission, m.previewOptics.indexOfRefraction,
        m.previewOptics.clearcoat, m.previewOptics.sheen,
      ]
      guard floats.allSatisfy({ $0.isFinite }), (0...1).contains(m.roughness),
        (0...1).contains(m.metallic), m.program <= programs.count,
        (0...1).contains(m.previewOptics.transmission), (1...3).contains(m.previewOptics.indexOfRefraction),
        (0...1).contains(m.previewOptics.clearcoat), (0...1).contains(m.previewOptics.sheen),
        m.roughnessChannel < 4, m.metallicChannel < 4,
        m.baseColor.min() >= 0, m.emission.min() >= 0, m.normalScale >= 0
      else {
        throw Failure.invalidMaterial(i)
      }
      values.append(
        try Record(
          color: SIMD4(m.baseColor, m.roughness), emission: SIMD4(m.emission, m.metallic),
          uv: SIMD4(m.uvScale.x, m.uvScale.y, m.uvOffset.x, m.uvOffset.y),
          maps: SIMD4(
            index(m.baseColorTexture), index(m.roughnessTexture), index(m.metallicTexture),
            index(m.normalTexture)),
          extra: SIMD4(
            index(m.emissionTexture), m.program, m.normalScale.bitPattern, m.clampToEdge ? 1 : 0),
          parameters: m.parameters,
          channels: SIMD4(m.roughnessChannel, m.metallicChannel, m.invertNormalGreen ? 1 : 0, 0),
          optics: SIMD4(m.previewOptics.transmission, m.previewOptics.indexOfRefraction, m.previewOptics.clearcoat, m.previewOptics.sheen)))
    }
    return Prepared(records: values, textures: textures, bytes: bytes)
  }

  public init(
    device: MTLDevice, materials: [GPUSimSurfaceMaterial] = [],
    programs: [GPUSimMaterialProgram] = [], textureBudget: Int = 512 * 1024 * 1024
  ) throws {
    let tier2 = device.argumentBuffersSupport == .tier2
    guard materials.isEmpty || tier2 else { throw Failure.unsupportedDevice }
    self.device = device
    self.programs = programs
    self.textureBudget = textureBudget
    hasTransmission = materials.contains { $0.previewOptics.transmission > 0 }
    usesArgumentBuffers = tier2
    materialCount = materials.count
    let prepared = try GPUSimMaterialLibrary.prepare(
      materials: materials, programs: programs, device: device, textureBudget: textureBudget)
    guard let empty = device.makeBuffer(length: 9*16, options: .storageModeShared) else { throw Failure.allocation }
    memset(empty.contents(),0,empty.length)
    environmentIrradiance = empty
    textures = prepared.textures
    textureBytes = prepared.bytes
    var values = prepared.records
    if values.isEmpty {
      values.append(
        Record(
          color: .zero, emission: .zero, uv: .zero, maps: .zero, extra: .zero, parameters: .zero,
          channels: .zero, optics: SIMD4(0, 1.5, 0, 0)))
    }
    guard
      let records = values.withUnsafeBytes({
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
      })
    else { throw Failure.allocation }
    self.records = records
    guard tier2 else {
      // Tier 1 shaders declare `struct MaterialResources { uint4 info; }`.
      guard let stub = device.makeBuffer(length: 16, options: .storageModeShared) else {
        throw Failure.allocation
      }
      memset(stub.contents(), 0, stub.length)
      arguments = stub
      return
    }
    arguments = try Self.makeArguments(device: device, records: records, textures: textures,
      count: materials.count, environmentIrradiance: empty, environment: nil)
  }

  static func makeArguments(device: MTLDevice, records: MTLBuffer, textures: [MTLTexture],
                            count materialCount: Int, environmentIrradiance: MTLBuffer,
                            environment: GPUSimEnvironmentLight?) throws -> MTLBuffer {
    // The same argument layout is used in every fragment and compute pipeline.
    let capacity = GPUSimMaterialLibrary.textureCapacity
    let maps = MTLArgumentDescriptor()
    maps.index = 0
    maps.dataType = .texture
    maps.textureType = .type2D
    maps.arrayLength = capacity
    maps.access = .readOnly
    let table = MTLArgumentDescriptor()
    table.index = capacity
    table.dataType = .pointer
    table.access = .readOnly
    let count = MTLArgumentDescriptor()
    count.index = capacity + 1
    count.dataType = .uint
    let environmentMap = MTLArgumentDescriptor()
    environmentMap.index = capacity+2; environmentMap.dataType = .texture
    environmentMap.textureType = .type2D; environmentMap.access = .readOnly
    let irradiance = MTLArgumentDescriptor()
    irradiance.index = capacity+3; irradiance.dataType = .pointer; irradiance.access = .readOnly
    let settings = MTLArgumentDescriptor()
    settings.index = capacity+4; settings.dataType = .float4
    guard let encoder = device.makeArgumentEncoder(arguments: [maps, table, count, environmentMap, irradiance, settings]),
      let args = device.makeBuffer(length: encoder.encodedLength, options: .storageModeShared)
    else { throw Failure.allocation }
    memset(args.contents(), 0, args.length)
    encoder.setArgumentBuffer(args, offset: 0)
    for (i, t) in textures.enumerated() { encoder.setTexture(t, index: i) }
    encoder.setBuffer(records, offset: 0, index: capacity)
    encoder.constantData(at: capacity + 1).storeBytes(of: UInt32(materialCount), as: UInt32.self)
    encoder.setTexture(environment?.texture, index: capacity+2)
    encoder.setBuffer(environment?.irradiance ?? environmentIrradiance, offset: 0, index: capacity+3)
    encoder.constantData(at: capacity+4).storeBytes(of:
      SIMD4<Float>(1,0,1,environment == nil ? 0 : 1), as: SIMD4<Float>.self)
    return args
  }

  func bind(_ encoder: MTLRenderCommandEncoder) {
    encoder.setFragmentBuffer(arguments, offset: 0, index: 10)
    guard usesArgumentBuffers else { return }
    encoder.useResource(records, usage: .read, stages: .fragment)
    encoder.useResource(environmentIrradiance, usage: .read, stages: .fragment)
    if !textures.isEmpty { encoder.useResources(textures, usage: .read, stages: .fragment) }
  }
  func bind(_ encoder: MTLComputeCommandEncoder) {
    encoder.setBuffer(arguments, offset: 0, index: 10)
    guard usesArgumentBuffers else { return }
    encoder.useResource(records, usage: .read)
    encoder.useResource(environmentIrradiance, usage: .read)
    if !textures.isEmpty { encoder.useResources(textures, usage: .read) }
  }

  /// Upper bound on the bytes `MTKTextureLoader` allocates for a decoded image
  /// with a full mip chain: 8-bit sources decode to RGBA8, deeper integer
  /// sources to RGBA16, and floating-point sources are bounded by RGBA32F.
  public nonisolated static func decodedByteEstimate(
    width: Int, height: Int, bitsPerComponent: Int, isFloat: Bool
  ) -> Int {
    let bytesPerPixel = isFloat ? 16 : (bitsPerComponent > 8 ? 8 : 4)
    return Int((Double(width) * Double(height) * Double(bytesPerPixel) * 4 / 3).rounded(.up))
  }

  static func textureLoaderOptions(sRGB: Bool) -> [MTKTextureLoader.Option: Any] {
    [
      .SRGB: sRGB, .generateMipmaps: true, .origin: MTKTextureLoader.Origin.topLeft,
      .textureUsage: MTLTextureUsage.shaderRead.rawValue,
      .textureStorageMode: MTLStorageMode.private.rawValue,
    ]
  }

  /// Dimension and decode-budget check shared by the file and Model I/O loaders.
  static func checkDecodeBudget(
    width: Int, height: Int, bitsPerComponent: Int, isFloat: Bool, maximumDecodedBytes: Int
  ) throws {
    guard width > 0, height > 0, width <= maximumImageDimension, height <= maximumImageDimension,
      decodedByteEstimate(
        width: width, height: height, bitsPerComponent: bitsPerComponent, isFloat: isFloat)
        <= maximumDecodedBytes
    else { throw Failure.textureBudgetExceeded }
  }

  /// Decode an image once, with an explicit channel color space and a mip chain.
  public static func loadTexture(
    device: MTLDevice, url: URL, sRGB: Bool,
    maximumDecodedBytes: Int = 512 * 1024 * 1024
  ) throws -> MTLTexture {
    guard url.isFileURL else { throw Failure.invalidImage }
    guard maximumDecodedBytes > 0 else { throw Failure.textureBudgetExceeded }
    // Reject oversized ordinary images from metadata before allocating pixels.
    if let source = CGImageSourceCreateWithURL(url as CFURL, nil),
      let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
      let width = properties[kCGImagePropertyPixelWidth] as? Int,
      let height = properties[kCGImagePropertyPixelHeight] as? Int
    {
      try checkDecodeBudget(
        width: width, height: height,
        bitsPerComponent: properties[kCGImagePropertyDepth] as? Int ?? 8,
        isFloat: properties[kCGImagePropertyIsFloat] as? Bool ?? false,
        maximumDecodedBytes: maximumDecodedBytes)
    }
    let result = try MTKTextureLoader(device: device).newTexture(
      URL: url, options: textureLoaderOptions(sRGB: sRGB))
    guard result.allocatedSize <= maximumDecodedBytes else { throw Failure.textureBudgetExceeded }
    return result
  }

  /// Decode an image embedded in a Model I/O asset with the same policy as ``loadTexture(device:url:sRGB:maximumDecodedBytes:)``.
  public static func loadTexture(
    device: MTLDevice, texture: MDLTexture, sRGB: Bool,
    maximumDecodedBytes: Int = 512 * 1024 * 1024
  ) throws -> MTLTexture {
    guard maximumDecodedBytes > 0 else { throw Failure.textureBudgetExceeded }
    let bits: Int
    let isFloat: Bool
    switch texture.channelEncoding {
    case .uInt8: bits = 8; isFloat = false
    case .uInt16: bits = 16; isFloat = false
    case .uInt24, .uInt32: bits = 32; isFloat = false
    case .float16, .float16SR, .float32: bits = 32; isFloat = true
    @unknown default: bits = 32; isFloat = true
    }
    try checkDecodeBudget(
      width: Int(texture.dimensions.x), height: Int(texture.dimensions.y),
      bitsPerComponent: bits, isFloat: isFloat, maximumDecodedBytes: maximumDecodedBytes)
    let result = try MTKTextureLoader(device: device).newTexture(
      texture: texture, options: textureLoaderOptions(sRGB: sRGB))
    guard result.allocatedSize <= maximumDecodedBytes else { throw Failure.textureBudgetExceeded }
    return result
  }
}

extension GPUSimMaterialLibrary.Failure: CustomStringConvertible {
  public var description: String {
    switch self {
    case .invalidImage: return "Material image must be a local file"
    case .unsupportedDevice: return "Surface materials require Metal argument-buffer tier 2"
    case .allocation: return "Metal could not allocate material resources"
    case .tooManyTextures:
      return "Material library exceeds \(GPUSimMaterialLibrary.textureCapacity) distinct textures"
    case .invalidMaterial(let index):
      return "Invalid factors, channels or program index in material \(index + 1)"
    case .incompatibleTexture(let name):
      return "Texture \(name) is not a sampled 2D resource on this render device"
    case .textureBudgetExceeded:
      return "Material image allocations exceed the configured texture budget"
    }
  }
}
