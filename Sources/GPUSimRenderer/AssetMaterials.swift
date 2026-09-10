import Foundation
import MetalKit
import ModelIO
import SimCore
import simd

/// Imported visual submeshes. Positions/normals include the asset hierarchy's
/// transforms at time zero. Simulation colliders and joints remain caller-owned.
public struct GPUSimImportedAsset {
  public struct Part {
    public let name: String
    public let mesh: SurfaceMesh
    public let textureCoordinates: [SIMD2<Float>]
    public let materialID: UInt32
    public init(
      name: String, mesh: SurfaceMesh, textureCoordinates: [SIMD2<Float>], materialID: UInt32
    ) {
      self.name = name
      self.mesh = mesh
      self.textureCoordinates = textureCoordinates
      self.materialID = materialID
    }
    public func rigidMesh(body: Int) -> SceneRigidMesh {
      SceneRigidMesh(
        body: body, mesh: mesh, color: SIMD3(repeating: 1), textureCoordinates: textureCoordinates,
        materialID: materialID)
    }
  }
  public let parts: [Part]
  public let materials: [GPUSimSurfaceMaterial]
  /// Features outside the opaque metallic/roughness subset are reported here.
  public let diagnostics: [String]
  public init(parts: [Part], materials: [GPUSimSurfaceMaterial], diagnostics: [String] = []) {
    self.parts = parts
    self.materials = materials
    self.diagnostics = diagnostics
  }
}

@MainActor
public enum GPUSimAssetImporter {
  public enum Failure: Error {
    case unsupportedFormat(String), emptyAsset, invalidGeometry(String), unsupportedTopology(
      String), unsupportedOpacity(String), missingUV(String), missingTexture(String),
      textureBudgetExceeded
    /// A material or image failed ``GPUSimMaterialLibrary`` validation; the
    /// payload is that failure's description.
    case invalidMaterial(String)
  }
  /// Every error thrown by ``load`` is a ``Failure``; library and loader errors
  /// are mapped so callers can switch on one enum.
  static func mapped(_ error: Error, texture name: String? = nil) -> Failure {
    if let failure = error as? Failure { return failure }
    if let failure = error as? GPUSimMaterialLibrary.Failure {
      switch failure {
      case .textureBudgetExceeded: return .textureBudgetExceeded
      case .invalidImage, .incompatibleTexture:
        return .missingTexture(name.map { "\($0): \(failure.description)" } ?? failure.description)
      default: return .invalidMaterial(failure.description)
      }
    }
    if let name { return .missingTexture("\(name): \(error.localizedDescription)") }
    return .invalidMaterial(error.localizedDescription)
  }
  /// Model I/O capability, not an assertion that every material/animation in a
  /// format is supported. In particular FBX requires conversion on platforms
  /// whose Model I/O importer does not support it.
  public static func canImport(_ fileExtension: String) -> Bool {
    MDLAsset.canImportFileExtension(fileExtension.lowercased())
  }

  public static func load(
    url: URL, device: MTLDevice,
    transform: simd_float4x4 = matrix_identity_float4x4,
    textureBudget: Int = 512 * 1024 * 1024,
    vertexBudget: Int = 2_000_000
  ) throws -> GPUSimImportedAsset {
    guard canImport(url.pathExtension) else { throw Failure.unsupportedFormat(url.pathExtension) }
    let asset = MDLAsset(url: url)
    // Model I/O seeds every material with default-named properties (for example
    // a grey `baseColor`) alongside the authored inputs of USD materials. Those
    // names let the importer prefer authored constants over the defaults.
    let template = MDLMaterial(
      name: "", scatteringFunction: MDLPhysicallyPlausibleScatteringFunction())
    let defaultNames = Set((0..<template.count).compactMap { template[$0]?.name })
    var parts: [GPUSimImportedAsset.Part] = []
    var materials: [GPUSimSurfaceMaterial] = []
    var diagnostics: [String] = []
    var textureCache: [String: MTLTexture] = [:]
    var materialCache: [ObjectIdentifier: UInt32] = [:]
    var bytes = 0
    var verticesUsed = 0
    func texture(_ p: MDLMaterialProperty?, srgb: Bool) throws -> MTLTexture? {
      guard let p else { return nil }
      let source: URL?
      if p.type == .URL, let value = p.urlValue {
        source =
          value.isFileURL
          ? value
          : URL(fileURLWithPath: value.path, relativeTo: url.deletingLastPathComponent())
            .standardizedFileURL
      } else if p.type == .string, let path = p.stringValue {
        source =
          URL(fileURLWithPath: path, relativeTo: url.deletingLastPathComponent())
          .standardizedFileURL
      } else {
        source = nil
      }
      let key =
        "\(source?.absoluteString ?? p.textureSamplerValue?.texture.map { String(describing:ObjectIdentifier($0)) } ?? "none"):\(srgb)"
      if let cached = textureCache[key] { return cached }
      let remaining = textureBudget - bytes
      guard remaining > 0 else { throw Failure.textureBudgetExceeded }
      let result: MTLTexture
      do {
        if let source {
          result = try GPUSimMaterialLibrary.loadTexture(
            device: device, url: source, sRGB: srgb, maximumDecodedBytes: remaining)
        } else if let embedded = p.textureSamplerValue?.texture {
          result = try GPUSimMaterialLibrary.loadTexture(
            device: device, texture: embedded, sRGB: srgb, maximumDecodedBytes: remaining)
        } else {
          throw Failure.missingTexture(p.name)
        }
      } catch {
        throw mapped(error, texture: source?.lastPathComponent ?? p.name)
      }
      bytes += result.allocatedSize
      textureCache[key] = result
      return result
    }
    func scalar(_ p: MDLMaterialProperty?, fallback: Float) -> Float {
      guard let p, p.type == .float else { return fallback }
      return p.floatValue
    }
    func color(_ p: MDLMaterialProperty?, fallback: SIMD3<Float>) -> SIMD3<Float> {
      guard let p else { return fallback }
      switch p.type {
      case .float3: return p.float3Value
      case .float4: return SIMD3(p.float4Value.x, p.float4Value.y, p.float4Value.z)
      case .float: return SIMD3(repeating: p.floatValue)
      case .color:
        if let c = p.color?.converted(
          to: CGColorSpace(name: CGColorSpace.extendedLinearSRGB)!, intent: .defaultIntent,
          options: nil)?.components, c.count >= 3
        {
          return SIMD3(Float(c[0]), Float(c[1]), Float(c[2]))
        }
        return fallback
      default: return fallback
      }
    }
    func material(_ mdl: MDLMaterial?) throws -> UInt32 {
      guard let mdl else { return 0 }
      if let old = materialCache[ObjectIdentifier(mdl)] { return old }
      if let opacity = mdl.property(with: .opacity), opacity.type != .none,
        opacity.type != .float || opacity.floatValue < 0.999
      {
        throw Failure.unsupportedOpacity(mdl.name)
      }
      func image(_ semantic: MDLMaterialSemantic) -> MDLMaterialProperty? {
        mdl.properties(with: semantic).first { [.URL, .string, .texture].contains($0.type) }
      }
      /// The constant for a semantic. An image replaces any constant (USD
      /// connections override their fallback value, and Model I/O still
      /// reports its own default grey next to a connected map), and an authored
      /// USD input is preferred over Model I/O's default-named property.
      func factor(_ semantic: MDLMaterialSemantic) -> MDLMaterialProperty? {
        guard image(semantic) == nil else { return nil }
        let constants = mdl.properties(with: semantic).filter {
          [.float, .float3, .float4, .color].contains($0.type)
        }
        return constants.first { !defaultNames.contains($0.name) } ?? constants.first
      }
      var m = GPUSimSurfaceMaterial()
      m.baseColor = color(factor(.baseColor), fallback: SIMD3(repeating: 1))
      // Model I/O retains OBJ Kd on the texture property's float3Value.
      // Unlike a USD connection, map_Kd multiplies this authored constant.
      if url.pathExtension.lowercased() == "obj", let map = image(.baseColor) {
        m.baseColor = map.float3Value
      }
      m.roughness = min(max(scalar(factor(.roughness), fallback: image(.roughness) == nil ? 0.5 : 1), 0), 1)
      m.metallic = min(max(scalar(factor(.metallic), fallback: image(.metallic) == nil ? 0 : 1), 0), 1)
      m.emission = color(factor(.emission), fallback: .zero)
      m.baseColorTexture = try texture(image(.baseColor), srgb: true)
      m.roughnessTexture = try texture(image(.roughness), srgb: false)
      m.metallicTexture = try texture(image(.metallic), srgb: false)
      m.normalTexture = try texture(image(.tangentSpaceNormal), srgb: false)
      m.invertNormalGreen = true  // UV origin was flipped on import.
      m.emissionTexture = try texture(image(.emission), srgb: true)
      if m.emissionTexture != nil { m.emission = SIMD3(repeating: 1) }
      for i in 0..<mdl.count {
        guard let p = mdl[i], let sampler = p.textureSamplerValue else { continue }
        if sampler.transform != nil || sampler.hardwareFilter != nil {
          diagnostics.append(
            "\(mdl.name): per-map sampler/transform uses the renderer's repeat, linear mip-filtered defaults; author UVs or set material UV transform explicitly"
          )
        }
      }
      if mdl.property(with: .displacement) != nil {
        diagnostics.append("\(mdl.name): displacement is not imported")
      }
      materials.append(m)
      let id = UInt32(materials.count)
      materialCache[ObjectIdentifier(mdl)] = id
      return id
    }
    func visit(_ object: MDLObject) throws {
      if let mesh = object as? MDLMesh {
        guard mesh.vertexCount <= max(0, vertexBudget - verticesUsed) else {
          throw Failure.invalidGeometry("vertex budget exceeded")
        }
        verticesUsed += mesh.vertexCount
        guard
          let position = mesh.vertexAttributeData(
            forAttributeNamed: MDLVertexAttributePosition, as: .float3)
        else { throw Failure.invalidGeometry(mesh.name) }
        let normal = mesh.vertexAttributeData(
          forAttributeNamed: MDLVertexAttributeNormal, as: .float3)
        let uv = mesh.vertexAttributeData(
          forAttributeNamed: MDLVertexAttributeTextureCoordinate, as: .float2)
        let matrix = transform * MDLTransform.globalTransform(with: mesh, atTime: 0)
        let linear = simd_float3x3(
          SIMD3(matrix.columns.0.x, matrix.columns.0.y, matrix.columns.0.z),
          SIMD3(matrix.columns.1.x, matrix.columns.1.y, matrix.columns.1.z),
          SIMD3(matrix.columns.2.x, matrix.columns.2.y, matrix.columns.2.z))
        guard abs(linear.determinant) > 1e-12 else {
          throw Failure.invalidGeometry("singular transform: \(mesh.name)")
        }
        let normalMatrix = linear.inverse.transpose
        var positions: [SIMD3<Float>] = []
        var normals: [SIMD3<Float>] = []
        var uvs: [SIMD2<Float>] = []
        for i in 0..<mesh.vertexCount {
          let p = position.dataStart.advanced(by: i * position.stride).assumingMemoryBound(
            to: Float.self)
          let world = matrix * SIMD4(p[0], p[1], p[2], 1)
          guard world.x.isFinite, world.y.isFinite, world.z.isFinite else {
            throw Failure.invalidGeometry(mesh.name)
          }
          positions.append(SIMD3(world.x, world.y, world.z))
          if let normal {
            let n = normal.dataStart.advanced(by: i * normal.stride).assumingMemoryBound(
              to: Float.self)
            let value = normalMatrix * SIMD3(n[0], n[1], n[2])
            guard simd_length_squared(value) > 1e-12 else {
              throw Failure.invalidGeometry("invalid normal: \(mesh.name)")
            }
            normals.append(simd_normalize(value))
          }
          if let uv {
            let t = uv.dataStart.advanced(by: i * uv.stride).assumingMemoryBound(to: Float.self)
            guard t[0].isFinite, t[1].isFinite else {
              throw Failure.invalidGeometry("invalid UV: \(mesh.name)")
            }
            uvs.append(SIMD2(t[0], 1 - t[1]))  // Model I/O authored UVs use bottom-left origin.
          }
        }
        guard let submeshes = mesh.submeshes else { throw Failure.invalidGeometry(mesh.name) }
        for case let submesh as MDLSubmesh in submeshes {
          guard submesh.geometryType == .triangles, submesh.indexCount % 3 == 0 else {
            throw Failure.unsupportedTopology(mesh.name)
          }
          let buffer = submesh.indexBuffer(asIndexType: .uInt32)
          guard buffer.length >= submesh.indexCount * 4 else {
            throw Failure.invalidGeometry(mesh.name)
          }
          let map = buffer.map()
          let indices = map.bytes.assumingMemoryBound(to: UInt32.self)
          var triangles: [(Int, Int, Int)] = []
          for i in stride(from: 0, to: submesh.indexCount, by: 3) {
            let a = Int(indices[i])
            let b = Int(indices[i + 1])
            let c = Int(indices[i + 2])
            guard max(a, max(b, c)) < positions.count else {
              throw Failure.invalidGeometry(mesh.name)
            }
            triangles.append(linear.determinant < 0 ? (a, c, b) : (a, b, c))
          }
          let id = try material(submesh.material)
          if uvs.isEmpty, id > 0 {
            let m = materials[Int(id) - 1]
            if [
              m.baseColorTexture, m.roughnessTexture, m.metallicTexture, m.normalTexture,
              m.emissionTexture,
            ].contains(where: { $0 != nil }) {
              throw Failure.missingUV(mesh.name)
            }
          }
          parts.append(
            .init(
              name: mesh.name + "/" + submesh.name,
              mesh: SurfaceMesh(vertices: positions, normals: normals, triangles: triangles),
              textureCoordinates: uvs, materialID: id))
        }
      }
      for child in object.children.objects { try visit(child) }
    }
    for i in 0..<asset.count { try visit(asset.object(at: i)) }
    guard !parts.isEmpty else { throw Failure.emptyAsset }
    // Apply the same validation as renderer resources without allocating them.
    do {
      try GPUSimMaterialLibrary.validate(
        materials: materials, device: device, textureBudget: textureBudget)
    } catch {
      throw mapped(error)
    }
    return GPUSimImportedAsset(
      parts: parts, materials: materials, diagnostics: Array(Set(diagnostics)).sorted())
  }
}

extension GPUSimAssetImporter.Failure: CustomStringConvertible {
  public var description: String {
    switch self {
    case .unsupportedFormat(let name):
      return "Model I/O cannot import .\(name) on this platform; convert to a supported format"
    case .emptyAsset: return "Asset contains no triangle submeshes"
    case .invalidGeometry(let name): return "Invalid imported geometry: \(name)"
    case .unsupportedTopology(let name): return "Submesh requires triangulation: \(name)"
    case .unsupportedOpacity(let name):
      return "Material \(name) needs an opacity path; this importer supports opaque PBR"
    case .missingUV(let name): return "Textured mesh has no UVs: \(name)"
    case .missingTexture(let name): return "Material image could not be resolved: \(name)"
    case .textureBudgetExceeded: return "Imported images exceed the configured texture budget"
    case .invalidMaterial(let reason): return "Imported material rejected: \(reason)"
    }
  }
}
