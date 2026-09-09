import Metal
import PhysicsAVBD
import SimCore
import XCTest

@testable import GPUSimRenderer

@MainActor
final class RayMaterialTests: XCTestCase {
  func testRayHitsUseImageUVsAndCallerProgram() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    guard device.supportsRaytracing else { throw XCTSkip("Metal ray tracing unavailable") }
    var scene = PhysicsScene(name: "Ray material fixture")
    let body = scene.addBody(
      size: F3(repeating: 0.001), density: 0, friction: 0, position: F3(0, 0, -3))
    let mesh = SurfaceMesh(
      vertices: [F3(-1, -1, 3), F3(1, -1, 3), F3(0, 1, 3)],
      normals: Array(repeating: F3(0, 0, 1), count: 3), triangles: [(0, 1, 2)])
    scene.addRigidMesh(
      SceneRigidMesh(
        body: body, mesh: mesh, color: F3(repeating: 1),
        textureCoordinates: [SIMD2(0, 0), SIMD2(1, 0), SIMD2(0.5, 1)], materialID: 1))
    let solver = try GPUSolver(scene: scene, device: device)
    let td = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: .rgba8Unorm_srgb, width: 2, height: 2, mipmapped: false)
    td.storageMode = .shared
    td.usage = .shaderRead
    let texture = try XCTUnwrap(device.makeTexture(descriptor: td))
    let pixels: [UInt8] = [128, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255]
    pixels.withUnsafeBytes {
      texture.replace(
        region: MTLRegionMake2D(0, 0, 2, 2), mipmapLevel: 0, withBytes: $0.baseAddress!,
        bytesPerRow: 8)
    }
    for mode in 0..<3 {
      let procedural = mode > 0
      var m = GPUSimSurfaceMaterial()
      m.baseColor = .zero
      m.metallic = 1
      m.emission = F3(repeating: 1)
      m.emissionTexture = texture
      m.clampToEdge = true
      let basePrograms: [GPUSimMaterialProgram] =
        procedural ? [.init(body: "surface.emission = float3(0.2,0.4,0.8);")] : []
      let programs =
        mode == 2
        ? [
          GPUSimMaterialProgram(
            body: "surface.emission = surface.color; surface.color = float3(0);")
        ] : basePrograms
      if mode == 2 { m.baseColor = F3(0.5, 0.25, 0.75) }
      m.program = procedural ? 1 : 0
      let resources = try GPUSimMaterialLibrary(device: device, materials: [m], programs: programs)
      let world = try RayTracingScene.shared(scene: solver, materials: resources)
      try world.prepare(scene: solver, revision: 0, auxiliary: [], ground: false)
      let source =
        makeRenderShaderSource(programs: programs) + "\n" + rayTracingShaderSource + """
          kernel void ray_material_probe(instance_acceleration_structure scene [[buffer(0)]],constant Uniforms& U [[buffer(1)]],
              device const RTVertex* vertices [[buffer(2)]],device const RTObject* objects [[buffer(3)]],
              device const RTInstance* instances [[buffer(4)]],device const RenderInstance* rigid [[buffer(5)]],
              device const RenderAppearance* overrides [[buffer(6)]], device float4* result [[buffer(9)]],constant MaterialResources& materials [[buffer(10)]]) {
              ray r; r.origin=float3(-0.5,-0.5,1); r.direction=float3(0,0,-1); r.min_distance=0.001; r.max_distance=2;
              result[0]=rtIncoming(r,scene,U,vertices,objects,instances,rigid,rigid,overrides,1,materials);
          }
          """
      let lib = try device.makeLibrary(source: source, options: nil)
      let pipeline = try device.makeComputePipelineState(
        function: XCTUnwrap(lib.makeFunction(name: "ray_material_probe")))
      let command = try XCTUnwrap(world.queue.makeCommandBuffer())
      let instances = try XCTUnwrap(
        device.makeBuffer(
          length: solver.renderRigidInstanceCount * MemoryLayout<GPUSimRenderInstance>.stride,
          options: .storageModeShared))
      try solver.encodeRenderInstances(
        command, instances: instances, colorMode: .bodyIndex, appearanceOverrides: nil)
      try world.encodeUpdate(
        command: command, scene: solver, instances: instances, auxiliary: nil, appearances: nil)
      let u = try XCTUnwrap(
        device.makeBuffer(length: MemoryLayout<Uniforms>.stride, options: .storageModeShared))
      memset(u.contents(), 0, u.length)
      u.contents().assumingMemoryBound(to: Uniforms.self).pointee.screen.z = 1000
      u.contents().assumingMemoryBound(to: Uniforms.self).pointee.lightDir = SIMD4(0, 0, -1, 0)
      let output = try XCTUnwrap(device.makeBuffer(length: 16, options: .storageModeShared))
      let e = try XCTUnwrap(command.makeComputeCommandEncoder())
      e.setComputePipelineState(pipeline)
      e.setAccelerationStructure(world.structure, bufferIndex: 0)
      e.useResource(world.structure, usage: .read)
      e.setBuffer(u, offset: 0, index: 1)
      e.setBuffer(world.vertices, offset: 0, index: 2)
      e.setBuffer(world.objects, offset: 0, index: 3)
      e.setBuffer(world.descriptors, offset: 0, index: 4)
      e.setBuffer(instances, offset: 0, index: 5)
      var appearance = GPUSimRenderAppearance(color: F3(0.8, 0.6, 0.4))
      e.setBytes(&appearance, length: MemoryLayout<GPUSimRenderAppearance>.stride, index: 6)
      e.setBuffer(output, offset: 0, index: 9)
      resources.bind(e)
      e.dispatchThreads(
        MTLSize(width: 1, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
      e.endEncoding()
      command.commit()
      command.waitUntilCompleted()
      XCTAssertEqual(command.status, .completed, "\(String(describing:command.error))")
      let result = output.contents().assumingMemoryBound(to: SIMD4<Float>.self).pointee
      XCTAssertEqual(result.w, 1)
      let expected: F3 =
        mode == 2
        ? F3(0.8 * 0.8 * 0.94 * 0.5, 0.6 * 0.6 * 0.88 * 0.25, 0.4 * 0.4 * 0.82 * 0.75)
        : (procedural ? F3(0.2, 0.4, 0.8) : F3(0.21586, 0, 0))
      XCTAssertEqual(result.x, expected.x, accuracy: 0.003)
      XCTAssertEqual(result.y, expected.y, accuracy: 0.003)
      XCTAssertEqual(result.z, expected.z, accuracy: 0.003)
    }
  }
}
