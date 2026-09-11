import ImageIO
import MetalKit
import PhysicsAVBD
import SimCore
import XCTest

@testable import GPUSimRenderer

@MainActor
final class EnvironmentBatchTests: XCTestCase {
  final class Scene: GPUSimRenderableScene {
    let renderDevice: MTLDevice
    var renderBodyCount: Int
    var renderRigidInstanceCount: Int { records.count }
    var rendererStateIsValid: Bool { true }
    var renderCameraHint: GPUSimRenderCameraHint { .init() }
    var renderGeometryRevision: UInt64 = 0
    var rigidMeshRenderSurface: GPUSimRigidMeshRenderSurface?
    var softRenderSurface: GPUSimSoftRenderSurface? { nil }
    var skinnedRenderSurface: GPUSimSkinnedRenderSurface? { nil }
    var convexDebugRenderSurface: GPUSimConvexDebugRenderSurface? { nil }
    var renderContentBounds: GPUSimContentBounds?
    var records: [GPUSimRenderInstance]
    init(device: MTLDevice, offsets: [F3], z: Float = 0, segments: Int = 1) throws {
      renderDevice = device
      renderBodyCount = offsets.count
      records = offsets.map {
        .init(
          primitive: .box(size: F3(0.35, 0.35, 0.8)), position: $0 + F3(0, 0, 0.4 + z),
          color: F3(0.8, 0.5, 0.3))
      }
      var vertices: [GPUSimRigidMeshRenderVertex] = []
      var indices: [UInt32] = []
      for (body, _) in offsets.enumerated() {
        let base = UInt32(vertices.count)
        for y in 0...segments {
          for x in 0...segments {
            let p = F3(
              Float(x) / Float(segments) * 1.6 - 0.8, Float(y) / Float(segments) * 1.4 - 0.7, 0)
            vertices.append(
              .init(
                positionBody: SIMD4(p, Float(bitPattern: UInt32(body))),
                normal: SIMD4(0, 0, 1, 0.5), color: SIMD4(0.6, 0.7, 0.8, 0)))
          }
        }
        for y in 0..<segments {
          for x in 0..<segments {
            let v = base + UInt32(y * (segments + 1) + x)
            let row = UInt32(segments + 1)
            indices += [v, v + 1, v + row + 1, v, v + row + 1, v + row]
          }
        }
      }
      func buffer<T>(_ data: [T]) throws -> MTLBuffer {
        try data.withUnsafeBufferPointer {
          try XCTUnwrap(
            device.makeBuffer(
              bytes: $0.baseAddress!, length: $0.count * MemoryLayout<T>.stride,
              options: .storageModeShared))
        }
      }
      rigidMeshRenderSurface = .init(
        vertices: try buffer(vertices), indices: try buffer(indices), indexCount: indices.count,
        positions: try buffer(offsets.map { SIMD4($0 + F3(0, 0, z), Float(0)) }),
        rotations: try buffer(offsets.map { _ in SIMD4<Float>(0, 0, 0, 1) }))
      renderContentBounds = .init(center: .zero, radius: 1.3)
    }
    func encodeRenderInstances(
      _ commandBuffer: MTLCommandBuffer, instances: MTLBuffer,
      colorMode: GPUSimRenderColorMode, appearanceOverrides: MTLBuffer?
    ) throws {
      var records = records
      if let appearanceOverrides {
        let appearances = appearanceOverrides.contents().assumingMemoryBound(
          to: GPUSimRenderAppearance.self)
        for i in records.indices where appearances[i].albedo.w > 0 {
          records[i].color = SIMD4(
            appearances[i].albedo.x, appearances[i].albedo.y, appearances[i].albedo.z,
            records[i].color.w)
        }
      }
      let input = try XCTUnwrap(
        renderDevice.makeBuffer(
          bytes: records, length: records.count * 112, options: .storageModeShared))
      let blit = try XCTUnwrap(commandBuffer.makeBlitCommandEncoder())
      blit.copy(
        from: input, sourceOffset: 0, to: instances, destinationOffset: 0, size: input.length)
      blit.endEncoding()
    }
  }

  func testSharedMeshMatchesFlattenedRenderIncludingAOAndBodyOverrides() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let offsets = [F3(-1.1, 0, 0), F3(1.1, 0, 0.3)]
    let sources = try offsets.map { _ in try Scene(device: device, offsets: [.zero]) }
    let batch = try GPUSimEnvironmentBatch(
      environments: zip(sources, offsets).map { .init(scene: $0, offset: $1) })
    let flattened = try Scene(device: device, offsets: offsets)
    flattened.renderContentBounds = batch.renderContentBounds
    XCTAssertEqual(batch.rigidMeshRenderSurface?.instanceCount, 2)
    XCTAssertEqual(batch.renderCameraHint.target, batch.renderContentBounds?.center)
    XCTAssertGreaterThanOrEqual(
      batch.renderCameraHint.distance, batch.renderContentBounds!.radius * 2.8)
    XCTAssertEqual(
      batch.sharedGeometryBytes * 2,
      flattened.rigidMeshRenderSurface!.vertices.length
        + flattened.rigidMeshRenderSurface!.indices.length)
    for ao in [false, true] {
      let actual = try render(batch, ao: ao)
      let expected = try render(flattened, ao: ao)
      let difference = zip(actual, expected).map { abs(Int($0) - Int($1)) }
      XCTAssertLessThan(Double(difference.reduce(0, +)) / Double(difference.count), 0.02)
      XCTAssertLessThan(difference.max() ?? 0, 4)
    }
  }

  private func render(
    _ scene: any GPUSimRenderableScene, ao: Bool, compact: Bool = true, reflections: Bool = false
  ) throws -> [UInt8] {
    let device = scene.renderDevice
    let renderer = try GPUSimRenderer(device: device, scene: scene)
    renderer.enablesPrimitiveBatching = compact
    renderer.automaticallyFramesScene = false
    renderer.options = .lightweight
    renderer.options.ambientOcclusion = ao
    renderer.options.screenSpaceReflections = reflections
    renderer.auxiliaryInstances = [
      .init(primitive: .sphere(radius: 0.18), position: F3(0, -0.4, 0.4), color: F3(0.9, 0.2, 0.4))
    ]
    renderer.options.showsGroundPlane = false
    renderer.bodyAppearances = [1: .init(color: F3(0.2, 0.8, 0.4))]
    let bounds = scene.renderContentBounds ?? .init(center: .zero, radius: 1.3)
    let distance = max(4, bounds.radius * 2.5)
    renderer.setCamera(
      position: bounds.center + F3(distance * 0.5, -distance * 0.8, distance * 0.6),
      target: bounds.center, up: F3(0, 0, 1))
    let view = MTKView(frame: CGRect(x: 0, y: 0, width: 384, height: 256), device: device)
    renderer.configure(view)
    view.isPaused = true
    view.autoResizeDrawable = false
    view.drawableSize = CGSize(width: 384, height: 256)
    var pixels = [UInt8]()
    renderer.frameCompletionHandler = { texture, _ in
      pixels = [UInt8](repeating: 0, count: 384 * 256 * 4)
      pixels.withUnsafeMutableBytes {
        texture.getBytes(
          $0.baseAddress!, bytesPerRow: 384 * 4, from: MTLRegionMake2D(0, 0, 384, 256),
          mipmapLevel: 0)
      }
    }
    view.draw()
    XCTAssertNil(renderer.runtimeFailure)
    XCTAssertEqual(pixels.count, 384 * 256 * 4)
    return pixels
  }

  func testIndependentSolverPosesStayOwnedWhilePhysicsAdvances() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    func solver(_ x: Float) throws -> GPUSolver {
      var scene = PhysicsScene(name: "environment")
      let body = scene.addBody(
        size: F3(repeating: 0.2), density: 1000, friction: 0, position: F3(x, 0, 2))
      scene.addRigidMesh(
        .init(
          body: body,
          mesh: .init(
            vertices: [F3(0, 0, 0), F3(1, 0, 0), F3(0, 1, 0)], normals: [], triangles: [(0, 1, 2)]))
      )
      return try GPUSolver(scene: scene, device: device)
    }
    let a = try solver(0)
    let b = try solver(2)
    let batch = try GPUSimEnvironmentBatch(environments: [
      .init(scene: a, offset: F3(-5, 0, 0)), .init(scene: b, offset: F3(5, 0, 0)),
    ])
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let command = try XCTUnwrap(queue.makeCommandBuffer())
    let instances = try XCTUnwrap(
      device.makeBuffer(length: batch.renderRigidInstanceCount * 112, options: .storageModeShared))
    let gate = try XCTUnwrap(device.makeSharedEvent())
    let appearanceValues = [
      GPUSimRenderAppearance(color: F3(1, 0, 0)), GPUSimRenderAppearance(color: F3(0, 1, 0)),
    ]
    let appearanceSource = try XCTUnwrap(
      device.makeBuffer(bytes: appearanceValues, length: 64, options: .storageModeShared))
    let appearances = try XCTUnwrap(device.makeBuffer(length: 64, options: .storageModePrivate))
    let upload = try XCTUnwrap(queue.makeCommandBuffer())
    let copy = try XCTUnwrap(upload.makeBlitCommandEncoder())
    copy.copy(
      from: appearanceSource, sourceOffset: 0, to: appearances, destinationOffset: 0, size: 64)
    copy.endEncoding()
    upload.commit()
    upload.waitUntilCompleted()
    command.encodeWaitForEvent(gate, value: 1)
    try batch.encodeRenderInstances(
      command, instances: instances, colorMode: .bodyIndex, appearanceOverrides: appearances)
    let mesh = try XCTUnwrap(batch.rigidMeshRenderSurface)
    let poses = try XCTUnwrap(
      device.makeBuffer(length: batch.renderBodyCount * 16, options: .storageModeShared))
    let blit = try XCTUnwrap(command.makeBlitCommandEncoder())
    blit.copy(
      from: mesh.positions, sourceOffset: 0, to: poses, destinationOffset: 0, size: poses.length)
    blit.endEncoding()
    command.commit()
    DispatchQueue.global().asyncAfter(deadline: .now() + 5) { gate.signaledValue = 1 }
    defer {
      gate.signaledValue = 1
      command.waitUntilCompleted()
    }
    a.setBodyPose(0, position: F3(20, 0, 2), rotation: a.bodyRotation(0))
    b.setBodyPose(0, position: F3(30, 0, 2), rotation: b.bodyRotation(0))
    XCTAssertEqual(gate.signaledValue, 0)
    gate.signaledValue = 1
    command.waitUntilCompleted()
    XCTAssertEqual(command.status, .completed)
    let p = poses.contents().assumingMemoryBound(to: SIMD4<Float>.self)
    XCTAssertEqual(p[0].x, -5, accuracy: 0.001)
    XCTAssertEqual(p[a.bodyCount].x, 7, accuracy: 0.001)
    let primitive = instances.contents().assumingMemoryBound(to: GPUSimRenderInstance.self)
    XCTAssertEqual(primitive[0].model.columns.3.x, -5, accuracy: 0.001)
    XCTAssertEqual(primitive[a.renderRigidInstanceCount].model.columns.3.x, 7, accuracy: 0.001)
    XCTAssertEqual(primitive[0].color.x, 1, accuracy: 0.001)
    XCTAssertEqual(primitive[0].color.y, 0, accuracy: 0.001)
    XCTAssertEqual(primitive[a.renderRigidInstanceCount].color.x, 0, accuracy: 0.001)
    XCTAssertEqual(primitive[a.renderRigidInstanceCount].color.y, 1, accuracy: 0.001)
  }

  func testClassifiedPrimitivesMatchLegacyDraws() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let scene = try Scene(
      device: device,
      offsets: (0..<128).map { i in F3(Float(i % 16) * 0.2 - 1.5, Float(i / 16) * 0.2 - 0.7, 0) })
    for i in scene.records.indices {
      let p = scene.records[i].model.columns.3
      let shapes: [GPUSimRenderPrimitive] = [
        .box(size: F3(repeating: 0.16)), .sphere(radius: 0.08),
        .torus(majorRadius: 0.065, minorRadius: 0.02), .capsule(length: 0.16, radius: 0.05),
      ]
      scene.records[i] = .init(
        primitive: shapes[i % 4], position: F3(p.x, p.y, p.z), color: F3(0.8, 0.5, 0.3))
    }
    for reflections in [false, true] {
      let compact = try render(scene, ao: true, compact: true, reflections: reflections)
      let legacy = try render(scene, ao: true, compact: false, reflections: reflections)
      let difference = zip(compact, legacy).map { abs(Int($0) - Int($1)) }
      XCTAssertLessThan(Double(difference.reduce(0, +)) / Double(difference.count), 0.03)
    }
  }

  func testEnvironmentThroughputBenchmark() async throws {
    guard ProcessInfo.processInfo.environment["AVBD_ENV_BENCHMARK"] == "1" else {
      throw XCTSkip("Opt-in rendering benchmark")
    }
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let counts =
      ProcessInfo.processInfo.environment["AVBD_ENV_COUNTS"]?.split(separator: ",").compactMap {
        Int($0)
      }.filter { $0 > 0 && $0 <= 4096 } ?? [1, 16, 64, 128]
    for trial in 0..<3 {
      for count in counts {
        let columns = Int(ceil(sqrt(Double(count) * 2)))
        let offsets = (0..<count).map { i in F3(Float(i % columns) * 2, Float(i / columns) * 2, 0) }
        let prototype = try Scene(device: device, offsets: [.zero], segments: 32)
        let batch = try GPUSimEnvironmentBatch(
          environments: offsets.map { .init(scene: prototype, offset: $0) })
        let flat = try Scene(device: device, offsets: offsets, segments: 32)
        flat.renderContentBounds = batch.renderContentBounds
        let packed = try Scene(device: device, offsets: offsets)
        let packedPoses = try XCTUnwrap(packed.rigidMeshRenderSurface)
        let topology = try XCTUnwrap(prototype.rigidMeshRenderSurface)
        packed.rigidMeshRenderSurface = .init(
          vertices: topology.vertices, indices: topology.indices, indexCount: topology.indexCount,
          positions: packedPoses.positions, rotations: packedPoses.rotations, instanceCount: count,
          bodiesPerInstance: 1)
        packed.renderContentBounds = batch.renderContentBounds
        var renderers: [GPUSimRenderer] = []
        var views: [MTKView] = []
        for (name, scene) in [
          ("legacy", flat as any GPUSimRenderableScene),
          ("batched", batch as any GPUSimRenderableScene),
          ("packed", packed as any GPUSimRenderableScene),
        ] {
          let renderer = try GPUSimRenderer(device: device, scene: scene)
          renderer.options = .lightweight
          renderer.options.showsGroundPlane = false
          renderer.enablesPrimitiveBatching = name != "legacy"
          renderer.automaticallyFramesScene = false
          let bounds = try XCTUnwrap(batch.renderContentBounds)
          let center = bounds.center
          let distance = max(4, bounds.radius * 2.5)
          renderer.setCamera(
            position: center + F3(distance * 0.35, -distance * 0.7, distance * 0.7), target: center,
            up: F3(0, 0, 1))
          let view = MTKView(frame: CGRect(x: 0, y: 0, width: 1024, height: 768), device: device)
          renderer.configure(view)
          view.isPaused = true
          view.autoResizeDrawable = false
          view.drawableSize = CGSize(width: 1024, height: 768)
          renderers.append(renderer)
          views.append(view)
        }
        var timings = [[Double](), [Double](), [Double]()]
        for frame in 0..<272 {
          for step in 0..<3 {
            let index = (frame + step) % 3
            views[index].draw()
            XCTAssertNil(renderers[index].runtimeFailure)
            if frame >= 32 { timings[index].append(renderers[index].lastFrameGPUMilliseconds) }
          }
        }
        for index in 0..<3 {
          let scene: any GPUSimRenderableScene = [
            flat as any GPUSimRenderableScene, batch, packed,
          ][index]
          let mesh = try XCTUnwrap(scene.rigidMeshRenderSurface)
          print(
            "ENV_BENCH trial=\(trial) count=\(count) mode=\(["legacy","batched","packed"][index]) GPUms=\(timings[index].reduce(0,+)/Double(timings[index].count)) median=\(timings[index].sorted()[120]) geometryBytes=\(mesh.vertices.length+mesh.indices.length)"
          )
        }
        if count == 128, trial == 2,
          let directory = ProcessInfo.processInfo.environment["AVBD_ENV_OUTPUT"]
        {
          try FileManager.default.createDirectory(
            atPath: directory, withIntermediateDirectories: true)
          for index in 0..<2 {
            let path = directory + "/" + (index == 0 ? "legacy" : "batched") + ".png"
            renderers[index].frameCompletionHandler = { texture, _ in
              var pixels = [UInt8](repeating: 0, count: 1024 * 768 * 4)
              pixels.withUnsafeMutableBytes {
                texture.getBytes(
                  $0.baseAddress!, bytesPerRow: 1024 * 4, from: MTLRegionMake2D(0, 0, 1024, 768),
                  mipmapLevel: 0)
              }
              for i in stride(from: 0, to: pixels.count, by: 4) { pixels.swapAt(i, i + 2) }
              let provider = CGDataProvider(data: Data(pixels) as CFData)!
              let image = CGImage(
                width: 1024, height: 768, bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: 1024 * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
              let output = CGImageDestinationCreateWithURL(
                URL(fileURLWithPath: path) as CFURL, "public.png" as CFString, 1, nil)!
              CGImageDestinationAddImage(output, image, nil)
              XCTAssertTrue(CGImageDestinationFinalize(output))
            }
            views[index].draw()
          }
        }
      }
    }
  }

  func testRepeatedSourceMatches128CopiedEnvironments() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let offsets = (0..<128).map { F3(Float($0 % 16) * 2, Float($0 / 16) * 2, 0) }
    let prototype = try Scene(device: device, offsets: [.zero])
    let batch = try GPUSimEnvironmentBatch(
      environments: offsets.map { .init(scene: prototype, offset: $0) })
    let flattened = try Scene(device: device, offsets: offsets)
    flattened.renderContentBounds = batch.renderContentBounds
    let actual = try render(batch, ao: true)
    let expected = try render(flattened, ao: true, compact: false)
    let difference = zip(actual, expected).map { abs(Int($0) - Int($1)) }
    XCTAssertLessThan(Double(difference.reduce(0, +)) / Double(difference.count), 0.02)
    XCTAssertLessThan(difference.max() ?? 0, 4)
  }

  func testPrimitiveClassificationPreservesOrderAndResetsCounts() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let batch = try PrimitiveBatch(device: device)
    let queue = try XCTUnwrap(device.makeCommandQueue())
    for count in [129, 16, 300] {
      let records = (0..<count).map { i -> GPUSimRenderInstance in
        var instance = GPUSimRenderInstance(
          primitive: .box(size: F3(repeating: 1)), position: .zero, color: F3(repeating: 1))
        instance.color.w = i % 7 == 0 ? .nan : Float(i % 4)
        return instance
      }
      let input = try XCTUnwrap(
        device.makeBuffer(bytes: records, length: count * 112, options: .storageModeShared))
      let command = try XCTUnwrap(queue.makeCommandBuffer())
      try batch.encode(command: command, instances: input, count: count)
      let args = try XCTUnwrap(device.makeBuffer(length: 64, options: .storageModeShared))
      let ids = try XCTUnwrap(device.makeBuffer(length: count * 16, options: .storageModeShared))
      let blit = try XCTUnwrap(command.makeBlitCommandEncoder())
      blit.copy(from: batch.arguments, sourceOffset: 0, to: args, destinationOffset: 0, size: 64)
      blit.copy(
        from: batch.indices, sourceOffset: 0, to: ids, destinationOffset: 0, size: ids.length)
      blit.endEncoding()
      command.commit()
      command.waitUntilCompleted()
      XCTAssertEqual(command.status, .completed)
      let draws = args.contents().assumingMemoryBound(to: SIMD4<UInt32>.self)
      let indices = ids.contents().assumingMemoryBound(to: UInt32.self)
      for shape in 0..<4 {
        let expected = records.indices.filter { records[$0].color.w == Float(shape) }.map(
          UInt32.init)
        XCTAssertEqual(draws[shape].y, UInt32(expected.count))
        XCTAssertEqual(draws[shape].w, UInt32(shape * count))
        XCTAssertEqual(
          Array(
            UnsafeBufferPointer(start: indices.advanced(by: shape * count), count: expected.count)),
          expected)
      }
    }
  }

  func testRejectsIncompatibleTopologyAndTopologyEdits() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let a = try Scene(device: device, offsets: [.zero])
    let b = try Scene(device: device, offsets: [.zero, .zero])
    XCTAssertThrowsError(try GPUSimEnvironmentBatch(environments: []))
    XCTAssertThrowsError(
      try GPUSimEnvironmentBatch(environments: [.init(scene: a), .init(scene: b)]))
    let batch = try GPUSimEnvironmentBatch(environments: [.init(scene: a)])
    if GPUSimRenderer.supportsHQ(device: device) {
      let renderer = try GPUSimRenderer(device: device, scene: batch)
      renderer.options = .qualityBeta
      let view = MTKView(frame: CGRect(x: 0, y: 0, width: 64, height: 64), device: device)
      renderer.configure(view)
      view.isPaused = true
      view.draw()
      XCTAssertTrue(renderer.runtimeFailure?.contains("Fast") == true)
    }
    a.renderGeometryRevision += 1
    let command = try XCTUnwrap(device.makeCommandQueue()?.makeCommandBuffer())
    let output = try XCTUnwrap(device.makeBuffer(length: 112, options: .storageModeShared))
    XCTAssertThrowsError(
      try batch.encodeRenderInstances(
        command, instances: output, colorMode: .bodyIndex, appearanceOverrides: nil))
    XCTAssertFalse(batch.renderSupportsRayTracing)
  }
}
