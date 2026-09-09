import Foundation
import ImageIO
import Metal
import MetalKit
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

/// Opaque metallic/roughness material. Texture RGB is multiplied by its linear factor and vertex/body appearance tint.
/// Encoded color/emission images use an sRGB Metal format (linear HDR is also valid); scalar/normal maps use
/// linear formats. A nil texture leaves the factor unchanged. UV origin is top-left.
public struct GPUSimSurfaceMaterial {
  public var baseColor = SIMD3<Float>(repeating: 1)
  public var roughness: Float = 1
  public var metallic: Float = 0
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
  struct Record {
    var color: SIMD4<Float>
    var emission: SIMD4<Float>
    var uv: SIMD4<Float>
    var maps: SIMD4<UInt32>
    var extra: SIMD4<UInt32>
    var parameters: SIMD4<Float>
    var channels: SIMD4<UInt32>
  }
  let device: MTLDevice
  let programs: [GPUSimMaterialProgram]
  let records: MTLBuffer
  let textures: [MTLTexture]
  let arguments: MTLBuffer
  private var shaderLibraries: [String: MTLLibrary] = [:]
  func shaderLibrary(motionGuides: Bool = false, rays: Bool = false) throws -> MTLLibrary {
    let key = "\(motionGuides):\(rays)"
    if let library = shaderLibraries[key] { return library }
    let source =
      makeRenderShaderSource(motionGuides: motionGuides, programs: programs)
      + (rays ? "\n" + rayTracingShaderSource : "")
    let library = try device.makeLibrary(source: source, options: nil)
    shaderLibraries[key] = library
    return library
  }
  public let materialCount: Int
  public let textureBytes: Int

  public init(
    device: MTLDevice, materials: [GPUSimSurfaceMaterial] = [],
    programs: [GPUSimMaterialProgram] = [], textureBudget: Int = 512 * 1024 * 1024
  ) throws {
    guard materials.isEmpty || device.argumentBuffersSupport == .tier2 else {
      throw Failure.unsupportedDevice
    }
    self.device = device
    self.programs = programs
    materialCount = materials.count
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
      guard textures.count < 128 else { throw Failure.tooManyTextures }
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
      ]
      guard floats.allSatisfy({ $0.isFinite }), (0...1).contains(m.roughness),
        (0...1).contains(m.metallic), m.program <= programs.count,
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
          channels: SIMD4(m.roughnessChannel, m.metallicChannel, m.invertNormalGreen ? 1 : 0, 0)))
    }
    self.textures = textures
    textureBytes = bytes
    if values.isEmpty {
      values.append(
        Record(
          color: .zero, emission: .zero, uv: .zero, maps: .zero, extra: .zero, parameters: .zero,
          channels: .zero))
    }
    guard
      let records = values.withUnsafeBytes({
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
      })
    else { throw Failure.allocation }
    self.records = records
    // The same argument layout is used in every fragment and compute pipeline.
    let maps = MTLArgumentDescriptor()
    maps.index = 0
    maps.dataType = .texture
    maps.textureType = .type2D
    maps.arrayLength = 128
    maps.access = .readOnly
    let table = MTLArgumentDescriptor()
    table.index = 128
    table.dataType = .pointer
    table.access = .readOnly
    let count = MTLArgumentDescriptor()
    count.index = 129
    count.dataType = .uint
    guard let encoder = device.makeArgumentEncoder(arguments: [maps, table, count]),
      let args = device.makeBuffer(length: encoder.encodedLength, options: .storageModeShared)
    else { throw Failure.allocation }
    memset(args.contents(), 0, args.length)
    encoder.setArgumentBuffer(args, offset: 0)
    for (i, t) in textures.enumerated() { encoder.setTexture(t, index: i) }
    encoder.setBuffer(records, offset: 0, index: 128)
    encoder.constantData(at: 129).storeBytes(of: UInt32(materials.count), as: UInt32.self)
    arguments = args
  }

  func bind(_ encoder: MTLRenderCommandEncoder) {
    encoder.setFragmentBuffer(arguments, offset: 0, index: 10)
    encoder.useResource(records, usage: .read, stages: .fragment)
    for t in textures { encoder.useResource(t, usage: .read, stages: .fragment) }
  }
  func bind(_ encoder: MTLComputeCommandEncoder) {
    encoder.setBuffer(arguments, offset: 0, index: 10)
    encoder.useResource(records, usage: .read)
    for t in textures { encoder.useResource(t, usage: .read) }
  }

  /// Decode an image once, with an explicit channel color space and a mip chain.
  public static func loadTexture(
    device: MTLDevice, url: URL, sRGB: Bool,
    maximumDecodedBytes: Int = 512 * 1024 * 1024
  ) throws -> MTLTexture {
    guard url.isFileURL, maximumDecodedBytes > 0 else { throw Failure.invalidImage }
    // Reject oversized ordinary images from metadata before allocating pixels.
    if let source = CGImageSourceCreateWithURL(url as CFURL, nil),
      let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
      let width = properties[kCGImagePropertyPixelWidth] as? Int,
      let height = properties[kCGImagePropertyPixelHeight] as? Int
    {
      guard width > 0, height > 0, width <= 16384, height <= 16384,
        Double(width) * Double(height) * 16 * 4 / 3 <= Double(maximumDecodedBytes)
      else { throw Failure.textureBudgetExceeded }
    }
    let result = try MTKTextureLoader(device: device).newTexture(
      URL: url,
      options: [
        .SRGB: sRGB, .generateMipmaps: true, .origin: MTKTextureLoader.Origin.topLeft,
        .textureUsage: MTLTextureUsage.shaderRead.rawValue,
        .textureStorageMode: MTLStorageMode.private.rawValue,
      ])
    guard result.allocatedSize <= maximumDecodedBytes else { throw Failure.textureBudgetExceeded }
    return result
  }
}

extension GPUSimMaterialLibrary.Failure: CustomStringConvertible {
  public var description: String {
    switch self {
    case .invalidImage: return "Material image must be a local file with a positive decode budget"
    case .unsupportedDevice: return "Surface materials require Metal argument-buffer tier 2"
    case .allocation: return "Metal could not allocate material resources"
    case .tooManyTextures: return "Material library exceeds 128 distinct textures"
    case .invalidMaterial(let index):
      return "Invalid factors, channels or program index in material \(index + 1)"
    case .incompatibleTexture(let name):
      return "Texture \(name) is not a sampled 2D resource on this render device"
    case .textureBudgetExceeded:
      return "Material image allocations exceed the configured texture budget"
    }
  }
}
