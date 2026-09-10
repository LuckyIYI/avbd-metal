import simd

/// A finite, one-sided diffuse emitter. Radiance is linear RGB (W/sr/m²),
/// independent of emitter size. The normal points toward the illuminated side.
/// Emitters illuminate and appear in HQ reflections; they are not physics bodies.
public struct GPUSimAreaLight: Sendable, Equatable {
  public enum Shape: UInt32, Sendable { case rectangle, disk }
  public enum Failure: Error { case invalidParameters }
  public static let maximumCount = 8
  public let position, normal, up, radiance: SIMD3<Float>
  /// Full width/height in world units. A disk with unequal axes is an ellipse.
  public let size: SIMD2<Float>
  public let shape: Shape
  public let twoSided: Bool

  public init(
    position: SIMD3<Float>, normal: SIMD3<Float>, up: SIMD3<Float> = SIMD3(0, 0, 1),
    size: SIMD2<Float>, radiance: SIMD3<Float>, shape: Shape = .rectangle,
    twoSided: Bool = false
  ) throws {
    let values = [
      position.x, position.y, position.z, normal.x, normal.y, normal.z,
      up.x, up.y, up.z, size.x, size.y, radiance.x, radiance.y, radiance.z,
    ]
    guard values.allSatisfy({ $0.isFinite }), size.min() > 0, radiance.min() >= 0,
      (size.x * size.y).isFinite,
      length_squared(normal).isFinite, length_squared(up).isFinite,
      length_squared(normal) > 1e-12, length_squared(up) > 1e-12
    else { throw Failure.invalidParameters }
    let n = normalize(normal)
    let r = cross(normalize(up), n)
    guard length_squared(r) > 1e-8 else { throw Failure.invalidParameters }
    self.position = position
    self.normal = n
    self.up = normalize(cross(n, r))
    self.size = size
    self.radiance = radiance
    self.shape = shape
    self.twoSided = twoSided
  }

  var record: AreaLightRecord {
    AreaLightRecord(
      position: SIMD4(position, Float(shape.rawValue)),
      right: SIMD4(normalize(cross(up, normal)), size.x * 0.5),
      up: SIMD4(up, size.y * 0.5), radiance: SIMD4(radiance, twoSided ? 1 : 0))
  }
}

/// Mirrors the fixed 64-byte MSL area-light record. Eight fit in the existing
/// per-frame uniform upload; there is no light allocation or extra render queue.
struct AreaLightRecord {
  var position = SIMD4<Float>.zero
  var right = SIMD4<Float>.zero
  var up = SIMD4<Float>.zero
  var radiance = SIMD4<Float>.zero
}
