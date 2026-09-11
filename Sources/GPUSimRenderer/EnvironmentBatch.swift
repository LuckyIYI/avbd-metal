import Foundation
import Metal
import PhysicsAVBD
import simd

/// Repeated rigid environments in one world-space overview. All entries share
/// identical local mesh topology, material IDs and body numbering, but can have
/// independent GPU poses. Screen-space effects still execute once per view.
///
/// This is a fixed-topology, Fast-only adapter. Recreate it after topology edits.
/// Offsets affect presentation only, not physics. Use one batch per renderer;
/// serialize calls and keep custom backend buffers valid until frame retirement.
public final class GPUSimEnvironmentBatch: GPUSimRenderableScene, RenderSubmissionProvider {
  private(set) var renderSubmissions: [MTLCommandBuffer] = []
  public struct Environment {
    public let scene: any GPUSimRenderableScene
    public let offset: SIMD3<Float>
    public init(scene: any GPUSimRenderableScene, offset: SIMD3<Float> = .zero) {
      self.scene = scene
      self.offset = offset
    }
  }
  public enum Failure: Error {
    case empty
    case incompatibleEnvironment(Int)
    case topologyChanged(Int)
    case allocation, encoder, commandFailed
  }
  public let environments: [Environment]
  public let renderDevice: MTLDevice
  public let renderBodyCount: Int
  public let renderRigidInstanceCount: Int
  public var renderSupportsRayTracing: Bool { false }
  public var renderGeometryRevision: UInt64 { 0 }
  public var renderStateRevision: UInt64? { nil }
  public var rendererStateIsValid: Bool {
    failureLock.lock()
    let valid = !producerFailed
    failureLock.unlock()
    return valid && environments.allSatisfy { $0.scene.rendererStateIsValid }
  }
  private let failureLock = NSLock()
  private var producerFailed = false
  public var renderSceneRequiresFrameRetirement: Bool {
    environments.contains {
      !($0.scene is GPUSolver) && $0.scene.renderSceneRequiresFrameRetirement
    }
  }
  public var renderCameraHint: GPUSimRenderCameraHint {
    var hint = environments[0].scene.renderCameraHint
    if let bounds = renderContentBounds {
      hint.target = bounds.center
      hint.distance = max(hint.distance, bounds.radius * 2.8)
    } else {
      var lower = environments[0].offset
      var upper = lower
      for environment in environments {
        lower = simd_min(lower, environment.offset)
        upper = simd_max(upper, environment.offset)
      }
      hint.target += (lower + upper) * 0.5
      hint.distance += simd_length(upper - lower) * 1.4
    }
    return hint
  }
  public var softRenderSurface: GPUSimSoftRenderSurface? { nil }
  public var skinnedRenderSurface: GPUSimSkinnedRenderSurface? { nil }
  public var convexDebugRenderSurface: GPUSimConvexDebugRenderSurface? { nil }
  public private(set) var rigidMeshRenderSurface: GPUSimRigidMeshRenderSurface?
  public var renderContentBounds: GPUSimContentBounds? {
    var lower = SIMD3<Float>(repeating: .infinity)
    var upper = -lower
    var cached: [ObjectIdentifier: GPUSimContentBounds] = [:]
    for environment in environments {
      let id = ObjectIdentifier(environment.scene)
      guard let bounds = cached[id] ?? environment.scene.renderContentBounds else { return nil }
      cached[id] = bounds
      let center = bounds.center + environment.offset
      lower = simd_min(lower, center - SIMD3(repeating: bounds.radius))
      upper = simd_max(upper, center + SIMD3(repeating: bounds.radius))
    }
    return GPUSimContentBounds(
      center: (lower + upper) * 0.5, radius: simd_length(upper - lower) * 0.5)
  }
  /// Immutable render geometry storage does not grow with environment count.
  public var sharedGeometryBytes: Int { (vertices?.length ?? 0) + (indices?.length ?? 0) }
  private let bodies, primitives, indexCount: Int
  private let vertices, indices: MTLBuffer?
  private let revisions: [UInt64]
  private let sourceVertices, sourceIndices: [MTLBuffer?]
  private let posePipeline, primitivePipeline: MTLComputePipelineState
  private let canonical: [Int]
  private let placementBuffer: MTLBuffer
  private let placementGroups: [Int: Range<Int>]
  private struct Placement {
    var offset: SIMD4<Float>
    var destination: SIMD4<UInt32>
  }
  private var slots: [Slot] = []
  private var nextSlot = 0

  private final class Slot {
    let available = DispatchSemaphore(value: 1)
    let positions, rotations: MTLBuffer
    var primitives: [Int: MTLBuffer] = [:]
    var snapshots: [Int: RenderSnapshot] = [:]
    var appearances: [Int: MTLBuffer] = [:]
    init(device: MTLDevice, bodyCount: Int) throws {
      guard let p = device.makeBuffer(length: max(1, bodyCount) * 16, options: .storageModePrivate),
        let r = device.makeBuffer(length: max(1, bodyCount) * 16, options: .storageModePrivate)
      else { throw Failure.allocation }
      positions = p
      rotations = r
    }
  }

  /// Validates matching geometry once using GPU readback. Simulation-owned
  /// mesh copies are never uploaded or copied again by this batch's frames.
  public init(environments: [Environment]) throws {
    guard let first = environments.first else { throw Failure.empty }
    self.environments = environments
    let device = first.scene.renderDevice
    renderDevice = device
    bodies = first.scene.renderBodyCount
    primitives = first.scene.renderRigidInstanceCount
    guard bodies >= 0, primitives >= 0,
      environments.count <= Int(UInt32.max) / max(1, bodies),
      environments.count <= Int(UInt32.max) / max(1, primitives)
    else { throw Failure.incompatibleEnvironment(0) }
    renderBodyCount = bodies * environments.count
    renderRigidInstanceCount = primitives * environments.count
    var seen: [ObjectIdentifier: Int] = [:]
    var canonical: [Int] = []
    for (i, environment) in environments.enumerated() {
      let source = environment.scene
      let id = ObjectIdentifier(source)
      if seen[id] == nil {
        if let solver = source as? GPUSolver { try solver.synchronize() }
        seen[id] = i
      }
      canonical.append(seen[id]!)
      guard source.renderDevice.registryID == renderDevice.registryID,
        source.renderBodyCount == bodies, source.renderRigidInstanceCount == primitives,
        source.softRenderSurface == nil, source.skinnedRenderSurface == nil,
        !(source is GPUSimEnvironmentBatch),
        (0..<3).allSatisfy({ environment.offset[$0].isFinite })
      else { throw Failure.incompatibleEnvironment(i) }
    }
    self.canonical = canonical
    var placements = environments.enumerated().map { i, e in
      Placement(offset: SIMD4(e.offset, Float(0)), destination: SIMD4(UInt32(i), 0, 0, 0))
    }
    var groups: [Int: Range<Int>] = [:]
    for key in Set(canonical).sorted() {
      let start = placements.count
      for i in environments.indices where canonical[i] == key { placements.append(placements[i]) }
      groups[key] = start..<placements.count
    }
    guard
      let placementBuffer = device.makeBuffer(
        bytes: placements,
        length: placements.count * MemoryLayout<Placement>.stride, options: .storageModeShared)
    else { throw Failure.allocation }
    self.placementBuffer = placementBuffer
    self.placementGroups = groups
    let mesh = first.scene.rigidMeshRenderSurface
    indexCount = mesh?.indexCount ?? 0
    guard indexCount >= 0 else { throw Failure.incompatibleEnvironment(0) }
    guard let queue = device.makeCommandQueue() else { throw Failure.allocation }
    // One shared immutable copy, with a temporary CPU-readable allocation
    // only during validation. Never read back per-frame poses.
    func copy(_ source: MTLBuffer) throws -> MTLBuffer {
      guard let result = device.makeBuffer(length: source.length, options: .storageModeShared),
        let command = queue.makeCommandBuffer(), let blit = command.makeBlitCommandEncoder()
      else { throw Failure.allocation }
      blit.copy(
        from: source, sourceOffset: 0, to: result, destinationOffset: 0, size: source.length)
      blit.endEncoding()
      command.commit()
      command.waitUntilCompleted()
      guard command.status == .completed else { throw Failure.commandFailed }
      return result
    }
    vertices = try mesh.map { try copy($0.vertices) }
    indices = try mesh.map { try copy($0.indices) }
    if let vertices, let indices {
      let stride = MemoryLayout<GPUSimRigidMeshRenderVertex>.stride
      guard vertices.length % stride == 0, indexCount <= indices.length / 4 else {
        throw Failure.incompatibleEnvironment(0)
      }
      let vertexCount = vertices.length / stride
      let ix = indices.contents().assumingMemoryBound(to: UInt32.self)
      let v = vertices.contents().assumingMemoryBound(to: GPUSimRigidMeshRenderVertex.self)
      for i in 0..<indexCount {
        guard Int(ix[i]) < vertexCount, Int(v[Int(ix[i])].positionBody.w.bitPattern) < bodies
        else { throw Failure.incompatibleEnvironment(0) }
      }
    }
    for (i, environment) in environments.enumerated() {
      let other = environment.scene.rigidMeshRenderSurface
      guard (other == nil) == (mesh == nil), other?.indexCount == mesh?.indexCount,
        (other?.instanceCount ?? 1) == 1,
        (other?.positions.length ?? bodies * 16) >= bodies * 16,
        (other?.rotations.length ?? bodies * 16) >= bodies * 16
      else { throw Failure.incompatibleEnvironment(i) }
      if canonical[i] != i { continue }
      if let other, let vertices, let indices {
        guard other.vertices.length == vertices.length, other.indices.length == indices.length
        else { throw Failure.incompatibleEnvironment(i) }
        if i != 0 {
          let v = try copy(other.vertices)
          let ix = try copy(other.indices)
          guard memcmp(v.contents(), vertices.contents(), v.length) == 0,
            memcmp(ix.contents(), indices.contents(), ix.length) == 0
          else { throw Failure.incompatibleEnvironment(i) }
        }
      }
    }
    revisions = environments.map { $0.scene.renderGeometryRevision }
    sourceVertices = environments.map { $0.scene.rigidMeshRenderSurface?.vertices }
    sourceIndices = environments.map { $0.scene.rigidMeshRenderSurface?.indices }
    let library = try renderDevice.makeLibrary(source: environmentBatchShaderSource, options: nil)
    guard let pose = library.makeFunction(name: "environment_poses"),
      let primitive = library.makeFunction(name: "environment_primitives")
    else { throw Failure.encoder }
    posePipeline = try renderDevice.makeComputePipelineState(function: pose)
    primitivePipeline = try renderDevice.makeComputePipelineState(function: primitive)
    slots = try (0..<3).map { _ in try Slot(device: renderDevice, bodyCount: renderBodyCount) }
    updateSurface(slots[0])
  }

  private func updateSurface(_ slot: Slot) {
    guard let vertices, let indices else { return }
    rigidMeshRenderSurface = .init(
      vertices: vertices, indices: indices, indexCount: indexCount,
      positions: slot.positions, rotations: slot.rotations,
      instanceCount: environments.count, bodiesPerInstance: bodies)
  }
  private func validate(_ scene: any GPUSimRenderableScene, index: Int) throws {
    let mesh = scene.rigidMeshRenderSurface
    guard scene.renderGeometryRevision == revisions[index], scene.renderBodyCount == bodies,
      scene.renderRigidInstanceCount == primitives,
      mesh?.vertices === sourceVertices[index], mesh?.indices === sourceIndices[index],
      (mesh?.indexCount ?? 0) == indexCount,
      scene.softRenderSurface == nil, scene.skinnedRenderSurface == nil
    else { throw Failure.topologyChanged(index) }
  }

  public func encodeRenderInstances(
    _ commandBuffer: MTLCommandBuffer, instances: MTLBuffer,
    colorMode: GPUSimRenderColorMode, appearanceOverrides: MTLBuffer?
  ) throws {
    guard commandBuffer.device.registryID == renderDevice.registryID,
      instances.device.registryID == renderDevice.registryID,
      instances.length >= max(1, renderRigidInstanceCount)
        * MemoryLayout<GPUSimRenderInstance>.stride,
      appearanceOverrides == nil
        || (appearanceOverrides!.device.registryID == renderDevice.registryID
          && appearanceOverrides!.length >= renderBodyCount
            * MemoryLayout<GPUSimRenderAppearance>.stride)
    else { throw Failure.incompatibleEnvironment(0) }
    let slot = slots[nextSlot]
    slot.available.wait()
    var submitted = false
    defer { if !submitted { slot.available.signal() } }
    var captured: [Int: any GPUSimRenderableScene] = [:]
    for (i, environment) in environments.enumerated() {
      try validate(environment.scene, index: i)
      let key = appearanceOverrides == nil ? canonical[i] : i
      let source: any GPUSimRenderableScene
      if let existing = captured[key] {
        source = existing
      } else {
        if slot.primitives[key] == nil {
          slot.primitives[key] = renderDevice.makeBuffer(
            length: max(1, primitives) * MemoryLayout<GPUSimRenderInstance>.stride,
            options: .storageModePrivate)
        }
        guard let staging = slot.primitives[key] else { throw Failure.allocation }
        var appearances: MTLBuffer?
        if let appearanceOverrides, bodies > 0, !(environment.scene is GPUSolver) {
          // Own the local body range until this GPU frame retires.
          let length = bodies * MemoryLayout<GPUSimRenderAppearance>.stride
          if slot.appearances[key] == nil {
            slot.appearances[key] = renderDevice.makeBuffer(
              length: length,
              options: appearanceOverrides.storageMode == .shared
                ? .storageModeShared : .storageModePrivate)
          }
          guard let slice = slot.appearances[key] else { throw Failure.allocation }
          if appearanceOverrides.storageMode == .shared && slice.storageMode == .shared {
            memcpy(
              slice.contents(), appearanceOverrides.contents().advanced(by: i * length), length)
          } else {
            guard let blit = commandBuffer.makeBlitCommandEncoder() else { throw Failure.encoder }
            blit.copy(
              from: appearanceOverrides, sourceOffset: i * length, to: slice, destinationOffset: 0,
              size: length)
            blit.endEncoding()
          }
          appearances = slice
        }
        if let solver = environment.scene as? GPUSolver {
          if slot.snapshots[key] == nil {
            slot.snapshots[key] = try RenderSnapshot(device: renderDevice)
          }
          let snapshot = slot.snapshots[key]!
          try snapshot.capture(
            solver: solver, colorMode: colorMode, appearances: appearanceOverrides,
            copyRigidGeometry: false,
            appearanceOffset: i * bodies * MemoryLayout<GPUSimRenderAppearance>.stride)
          source = snapshot
        } else {
          source = environment.scene
        }
        try source.encodeRenderInstances(
          commandBuffer, instances: staging,
          colorMode: colorMode, appearanceOverrides: appearances)
        try validate(environment.scene, index: i)
        captured[key] = source
      }
    }
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { throw Failure.encoder }
    encoder.label = "Pack environment poses and instances"
    for key in captured.keys.sorted() {
      let source = captured[key]!
      let staging = slot.primitives[key]!
      let group = appearanceOverrides == nil ? placementGroups[key]! : key..<(key + 1)
      encoder.setBuffer(
        placementBuffer, offset: group.lowerBound * MemoryLayout<Placement>.stride, index: 4)
      if let mesh = source.rigidMeshRenderSurface, bodies > 0 {
        encoder.setComputePipelineState(posePipeline)
        encoder.setBuffer(mesh.positions, offset: 0, index: 0)
        encoder.setBuffer(mesh.rotations, offset: 0, index: 1)
        encoder.setBuffer(slot.positions, offset: 0, index: 2)
        encoder.setBuffer(slot.rotations, offset: 0, index: 3)
        var count = UInt32(bodies)
        encoder.setBytes(&count, length: 4, index: 5)
        encoder.dispatchThreads(
          MTLSize(width: bodies, height: group.count, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
      }
      if primitives > 0 {
        encoder.setComputePipelineState(primitivePipeline)
        encoder.setBuffer(staging, offset: 0, index: 0)
        encoder.setBuffer(instances, offset: 0, index: 1)
        var count = UInt32(primitives)
        encoder.setBytes(&count, length: 4, index: 5)
        encoder.dispatchThreads(
          MTLSize(width: primitives, height: group.count, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
      }
    }
    encoder.endEncoding()
    updateSurface(slot)
    let producers = captured.values.compactMap { ($0 as? RenderSnapshot)?.submission }
    renderSubmissions = producers
    commandBuffer.addCompletedHandler { [self] _ in
      if producers.contains(where: { $0.status == .error }) {
        failureLock.lock()
        producerFailed = true
        failureLock.unlock()
      }
      slot.available.signal()
    }
    submitted = true
    nextSlot = (nextSlot + 1) % slots.count
  }
}

private let environmentBatchShaderSource = """
  #include <metal_stdlib>
  using namespace metal;
  struct Instance { float4x4 model; float4 color; float4 params; float4 material; };
  struct Placement { float4 offset; uint4 destination; };
  kernel void environment_poses(device const float4* positions [[buffer(0)]], device const float4* rotations [[buffer(1)]],
      device float4* outPositions [[buffer(2)]], device float4* outRotations [[buffer(3)]],
      device const Placement* placements [[buffer(4)]], constant uint& count [[buffer(5)]], uint2 pixel [[thread_position_in_grid]]) {
      uint id=pixel.x; if (id>=count) return;
      Placement placement=placements[pixel.y]; uint output=placement.destination.x*count+id;
      outPositions[output]=positions[id]+float4(placement.offset.xyz,0); outRotations[output]=rotations[id];
  }
  kernel void environment_primitives(device const Instance* source [[buffer(0)]], device Instance* destination [[buffer(1)]],
      device const Placement* placements [[buffer(4)]], constant uint& count [[buffer(5)]], uint2 pixel [[thread_position_in_grid]]) {
      uint id=pixel.x; if (id>=count) return;
      Placement placement=placements[pixel.y]; uint output=placement.destination.x*count+id;
      Instance result=source[id]; result.model[3].xyz+=placement.offset.xyz; destination[output]=result;
  }
  """
