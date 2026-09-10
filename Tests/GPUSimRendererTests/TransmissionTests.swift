import Metal
import PhysicsAVBD
import SimCore
import XCTest

@testable import GPUSimRenderer

@MainActor
final class TransmissionTests: XCTestCase {
  func testOpticsValidationAndDefaultOpaqueABI() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let base = GPUSimSurfaceMaterial()
    XCTAssertEqual(base.previewOptics.transmission, 0)
    XCTAssertEqual(MemoryLayout<GPUSimMaterialLibrary.Record>.stride, 128)
    for value: Float in [-0.1, 1.1, .nan, .infinity] {
      var m = base
      m.previewOptics.transmission = value
      XCTAssertThrowsError(try GPUSimMaterialLibrary.validate(materials: [m], device: device))
    }
    for value: Float in [0.9, 3.1, .nan, .infinity] {
      var m = base
      m.previewOptics.indexOfRefraction = value
      XCTAssertThrowsError(try GPUSimMaterialLibrary.validate(materials: [m], device: device))
    }
  }

  func testTwoInterfacesTransmitBackgroundAndIORDisplacesRay() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    guard device.supportsRaytracing else { throw XCTSkip("Metal ray tracing unavailable") }
    var scene = PhysicsScene(name: "Dielectric slab over split emissive target")
    let body = scene.addBody(
      size: F3(repeating: 0.001), density: 0, friction: 0, position: F3(0, 0, -5))
    func plane(z: Float, normal: Float, material: UInt32) {
      let mesh = SurfaceMesh(
        vertices: [F3(-3, -3, z + 5), F3(3, -3, z + 5), F3(3, 3, z + 5), F3(-3, 3, z + 5)],
        normals: Array(repeating: F3(0, 0, normal), count: 4),
        triangles: normal > 0 ? [(0, 1, 2), (0, 2, 3)] : [(0, 2, 1), (0, 3, 2)])
      scene.addRigidMesh(
        SceneRigidMesh(body: body, mesh: mesh, color: F3(repeating: 1), materialID: material))
    }
    plane(z: 0.25, normal: 1, material: 1)
    plane(z: -0.25, normal: -1, material: 1)
    plane(z: -1, normal: 1, material: 2)
    let solver = try GPUSolver(scene: scene, device: device)
    let programs = [
      GPUSimMaterialProgram(
        body: "surface.emission=context.position.x>0 ? float3(1,0,0) : float3(0,1,0);")
    ]
    var results = [SIMD4<Float>]()
    for mode in 0..<3 {
      var glass = GPUSimSurfaceMaterial()
      glass.previewOptics.transmission = mode == 0 ? 0 : 1
      glass.previewOptics.indexOfRefraction = mode == 2 ? 1.5 : 1
      var target = GPUSimSurfaceMaterial()
      target.baseColor = .zero
      target.metallic = 1
      target.program = 1
      let resources = try GPUSimMaterialLibrary(
        device: device, materials: [glass, target], programs: programs)
      let world = try RayTracingScene.shared(scene: solver, materials: resources)
      try world.prepare(scene: solver, revision: 0, auxiliary: [], ground: false)
      let source =
        makeRenderShaderSource(programs: programs) + "\n" + rayTracingShaderSource + """
          kernel void transmission_probe(instance_acceleration_structure scene [[buffer(0)]],constant Uniforms& U [[buffer(1)]],
              device const RTVertex* vertices [[buffer(2)]],device const RTObject* objects [[buffer(3)]],
              device const RTInstance* instances [[buffer(4)]],device const RenderInstance* rigid [[buffer(5)]],
              device const RenderAppearance* overrides [[buffer(6)]],device float4* result [[buffer(9)]],constant MaterialResources& materials [[buffer(10)]]) {
              ray r; r.origin=float3(-1.15,0,1); r.direction=normalize(float3(0.6,0,-1)); r.min_distance=0.001; r.max_distance=10;
              result[0]=rtTransmission(r,scene,U,vertices,objects,instances,rigid,rigid,overrides,0,materials);
          }
          """
      let lib = try device.makeLibrary(source: source, options: nil)
      let pipeline = try device.makeComputePipelineState(
        function: XCTUnwrap(lib.makeFunction(name: "transmission_probe")))
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
      e.setBuffer(instances, offset: 0, index: 6)
      e.setBuffer(output, offset: 0, index: 9)
      resources.bind(e)
      e.dispatchThreads(
        MTLSize(width: 1, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
      e.endEncoding()
      command.commit()
      command.waitUntilCompleted()
      XCTAssertEqual(command.status, .completed, "\(String(describing: command.error))")
      results.append(output.contents().assumingMemoryBound(to: SIMD4<Float>.self).pointee)
    }
    XCTAssertEqual(results[0], .zero, "Opaque receivers must retain the original shading path")
    XCTAssertEqual(results[1].w, 1)
    XCTAssertGreaterThan(results[1].x, 0.98, "IOR=1 should see the red half without displacement")
    XCTAssertLessThan(results[1].y, 0.01)
    XCTAssertEqual(results[2].w, 1)
    XCTAssertGreaterThan(
      results[2].y, 0.8, "A glass slab should refract the ray onto the green half")
    XCTAssertLessThan(results[2].x, 0.15)
    for r in results { XCTAssertTrue((0..<4).allSatisfy { r[$0].isFinite }) }
  }
}
