/// Per-pixel ray budgets. Defaults preserve the existing HQ workload.
/// Increasing budgets trades throughput for lower noise within a single frame.
public struct GPUSimRayTracingQuality: Sendable, Equatable {
  public var shadowSamples: Int
  public var reflectionSamples: Int
  /// How finite emitters are selected. Defaults retain per-emitter stratification.
  public enum AreaLightSampling: Sendable, Equatable {
    case allLights
    /// Sample emitted-power-weighted lights; sample counts become a total budget.
    case powerWeighted
  }
  public var areaLightSampling: AreaLightSampling
  public var areaLightSamples: Int
  /// Zero inherits the primary budget. A positive value bounds work at secondary hits.
  public var secondaryAreaLightSamples: Int
  /// Zero retains adaptive HQ sampling (one direct-lit / four indirect rays).
  public var diffuseSamples: Int
  /// Zero disables dielectric camera transport; positive budgets resolve to 2...32.
  public var transmissionInterfaces: Int

  public init(
    shadowSamples: Int = 1, reflectionSamples: Int = 1,
    diffuseSamples: Int = 0, transmissionInterfaces: Int = 12, areaLightSamples: Int = 1,
    secondaryAreaLightSamples: Int = 0, areaLightSampling: AreaLightSampling = .allLights
  ) {
    self.areaLightSamples = areaLightSamples
    self.secondaryAreaLightSamples = secondaryAreaLightSamples
    self.areaLightSampling = areaLightSampling
    self.shadowSamples = shadowSamples
    self.reflectionSamples = reflectionSamples
    self.diffuseSamples = diffuseSamples
    self.transmissionInterfaces = transmissionInterfaces
  }
  public static let realtime = Self()
  public static let balanced = Self(
    shadowSamples: 4, reflectionSamples: 2, diffuseSamples: 4, areaLightSamples: 4)
  public static let high = Self(
    shadowSamples: 16, reflectionSamples: 8, diffuseSamples: 16, transmissionInterfaces: 24,
    areaLightSamples: 16)
  var resolved: Self {
    Self(
      shadowSamples: max(1, min(64, shadowSamples)),
      reflectionSamples: max(1, min(64, reflectionSamples)),
      diffuseSamples: max(0, min(64, diffuseSamples)),
      transmissionInterfaces: transmissionInterfaces == 0 ? 0 : max(2, min(32, transmissionInterfaces)),
      areaLightSamples: max(1, min(64, areaLightSamples)),
      secondaryAreaLightSamples: max(0, min(64, secondaryAreaLightSamples)),
      areaLightSampling: areaLightSampling)
  }
}
