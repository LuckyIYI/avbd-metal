/// Experimental optical approximations for the hybrid preview renderer.
/// These are not a complete layered BSDF and are not evaluated uniformly on
/// primary, secondary and shadow rays. Defaults retain opaque PBR rendering.
public struct GPUSimPreviewOptics: Sendable, Equatable {
  /// HQ camera-path transmission only. Requires closed outward-normalled meshes;
  /// Fast remains opaque. No rough transmission, nested media or caustics.
  public var transmission: Float = 0
  public var indexOfRefraction: Float = 1.5
  /// Approximate primary-surface lobes; absent from secondary-hit evaluation.
  public var clearcoat: Float = 0
  public var sheen: Float = 0
  public init() {}
}
