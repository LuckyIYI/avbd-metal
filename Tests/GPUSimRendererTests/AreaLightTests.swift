import Metal
import PhysicsAVBD
import SimCore
import XCTest
import simd

@testable import GPUSimRenderer

@MainActor
final class AreaLightTests: XCTestCase {
  func testValidationBudgetsAndUniformABI() throws {
    XCTAssertEqual(GPUSimRayTracingQuality().secondaryAreaLightSamples, 0)
    XCTAssertEqual(GPUSimRayTracingQuality().areaLightSampling, .allLights)
    XCTAssertEqual(GPUSimRayTracingQuality(secondaryAreaLightSamples: -2).resolved.secondaryAreaLightSamples, 0)
    XCTAssertEqual(GPUSimRayTracingQuality(secondaryAreaLightSamples: 100).resolved.secondaryAreaLightSamples, 64)
    XCTAssertEqual(MemoryLayout<AreaLightRecord>.stride, 64)
    XCTAssertEqual(
      MemoryLayout.offset(of: \Uniforms.aoProjection)! - MemoryLayout.offset(
        of: \Uniforms.areaLights)!, 512)
    XCTAssertTrue(GPUSimRenderOptions().areaLights.isEmpty)
    XCTAssertTrue(GPUSimRenderOptions().rayTracingDenoising)
    XCTAssertEqual(GPUSimRayTracingQuality().areaLightSamples, 1)
    XCTAssertEqual(GPUSimRayTracingQuality(areaLightSamples: 999).resolved.areaLightSamples, 64)
    let light = try GPUSimAreaLight(
      position: .zero, normal: SIMD3(0, 0, -1), up: SIMD3(0, 1, 0), size: SIMD2(2, 3),
      radiance: SIMD3(2, 3, 4))
    XCTAssertEqual(light.normal, SIMD3(0, 0, -1))
    var options = GPUSimRenderOptions()
    options.areaLights = Array(repeating: light, count: 20)
    XCTAssertThrowsError(try options.validateLighting())
    XCTAssertEqual(options.resolved(supportsHQ: true).areaLights.count, 20)
    for size: SIMD2<Float> in [
      .zero, SIMD2(-1, 1), SIMD2(.nan, 1), SIMD2(repeating: .greatestFiniteMagnitude),
    ] {
      XCTAssertThrowsError(
        try GPUSimAreaLight(
          position: .zero, normal: SIMD3(0, 0, -1), up: SIMD3(0, 1, 0), size: size,
          radiance: SIMD3(repeating: 1)))
    }
    XCTAssertThrowsError(
      try GPUSimAreaLight(
        position: .zero, normal: SIMD3(repeating: .greatestFiniteMagnitude),
        size: SIMD2(1, 1), radiance: SIMD3(repeating: 1)))
    XCTAssertThrowsError(
      try GPUSimAreaLight(
        position: .zero, normal: SIMD3(0, 0, 1), size: SIMD2(1, 1), radiance: SIMD3(repeating: 1)),
      "Parallel up and normal are invalid")
  }

  func testEmitterGeometryRadianceAndInverseSquareLimitOnGPU() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let library = try device.makeLibrary(
      source: renderShaderSource + """
        kernel void area_probe(device float4* output [[buffer(0)]],constant Uniforms& U [[buffer(1)]]) {
            output[0]=float4(rasterAreaLighting(float3(0),float3(0,0,1),float3(0,0,1),float3(0.6),1,0,U),1);
            output[1]=float4(rasterAreaLighting(float3(0,0,-3),float3(0,0,1),float3(0,0,1),float3(0.6),1,0,U),1);
            output[2]=areaIntersection(float3(0),float3(0,0,1),U);
            output[3]=areaIntersection(float3(0,0,6),float3(0,0,-1),U);
            output[4]=float4(cameraRayInterval(float3(0,0,1),U),cameraRayInterval(normalize(float3(1,0,1)),U));
        }
        """, options: nil)
    let pipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(library.makeFunction(name: "area_probe")))
    let light = try GPUSimAreaLight(
      position: SIMD3(0, 0, 3), normal: SIMD3(0, 0, -1), up: SIMD3(0, 1, 0),
      size: SIMD2(repeating: 0.01), radiance: SIMD3(repeating: 100))
    let u = try XCTUnwrap(
      device.makeBuffer(length: MemoryLayout<Uniforms>.stride, options: .storageModeShared))
    memset(u.contents(), 0, u.length)
    let values = u.contents().assumingMemoryBound(to: Uniforms.self)
    values.pointee.areaSettings = SIMD4(1, 64, 0, 0)
    values.pointee.areaLights.0 = light.record
    values.pointee.camRight = SIMD4(1,0,0,0)
    values.pointee.camUp = SIMD4(0,1,0,0)
    values.pointee.aoProjection = SIMD4(1.25,-2.5,1,1)
    let output = try XCTUnwrap(device.makeBuffer(length: 80, options: .storageModeShared))
    let command = try XCTUnwrap(device.makeCommandQueue()?.makeCommandBuffer())
    let e = try XCTUnwrap(command.makeComputeCommandEncoder())
    e.setComputePipelineState(pipeline)
    e.setBuffer(output, offset: 0, index: 0)
    e.setBuffer(u, offset: 0, index: 1)
    e.dispatchThreads(
      .init(width: 1, height: 1, depth: 1),
      threadsPerThreadgroup: .init(width: 1, height: 1, depth: 1))
    e.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    XCTAssertEqual(command.status, .completed)
    let result = output.contents().assumingMemoryBound(to: SIMD4<Float>.self)
    let brdf = (0.6 * 0.96 + 0.04 / 4) / Double.pi
    XCTAssertEqual(Double(result[0].x), brdf * 100 * 0.0001 / 9, accuracy: 0.000001)
    XCTAssertEqual(result[1].x / result[0].x, 0.25, accuracy: 0.0001)
    XCTAssertEqual(result[2], SIMD4(100, 100, 100, 3))
    XCTAssertGreaterThan(result[3].w, 1e10, "A one-sided emitter is invisible from behind")
    XCTAssertEqual(result[4].x, 2, accuracy: 0.0001)
    XCTAssertEqual(result[4].y, 10, accuracy: 0.0001)
    XCTAssertEqual(result[4].z, 2 * sqrt(2), accuracy: 0.0001)
    XCTAssertEqual(result[4].w, 10 * sqrt(2), accuracy: 0.0001)
  }

  func testPowerSelectionPreservesRGBEnergyAndEmitterSupport() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let library = try device.makeLibrary(source: renderShaderSource + """
      kernel void selection_probe(device float4* out [[buffer(0)]],constant Uniforms& U [[buffer(1)]]) {
          float weights[8];float sum=areaSelectionWeights(U,weights);float3 estimate=0,exact=0;
          for(uint i=0;i<3;++i) exact+=U.areaLights[i].radiance.rgb;
          for(uint j=0;j<65536;++j) {
              float p;uint i=areaSelectLight((float(j)+0.5)/65536,3,weights,sum,p);
              estimate+=U.areaLights[i].radiance.rgb/p/65536;
          }
          out[0]=float4(estimate,1);out[1]=float4(exact,1);
          // Light at z=3 faces downward. Receiver faces away, toward, and lies behind it.
          out[2]=float4(areaMayContribute(U.areaLights[0],float3(0),float3(0,0,-1)),
                        areaMayContribute(U.areaLights[0],float3(0),float3(0,0,1)),
                        areaMayContribute(U.areaLights[0],float3(0,0,4),float3(0,0,-1)),weights[2]/sum);
      }
      """, options: nil)
    let pipeline = try device.makeComputePipelineState(function: XCTUnwrap(library.makeFunction(name: "selection_probe")))
    let u = try XCTUnwrap(device.makeBuffer(length: MemoryLayout<Uniforms>.stride, options: .storageModeShared))
    memset(u.contents(), 0, u.length)
    let v = u.contents().assumingMemoryBound(to: Uniforms.self)
    v.pointee.areaSettings = SIMD4(3,1,1,1)
    func light(_ c: SIMD3<Float>, _ size: Float) throws -> AreaLightRecord {
      try GPUSimAreaLight(position: SIMD3(0,0,3), normal: SIMD3(0,0,-1), up: SIMD3(0,1,0), size: SIMD2(repeating:size), radiance:c).record
    }
    v.pointee.areaLights.0 = try light(SIMD3(10,1,0),1)
    v.pointee.areaLights.1 = try light(SIMD3(0,2,5),2)
    v.pointee.areaLights.2 = try light(.zero,1)
    let out = try XCTUnwrap(device.makeBuffer(length:48,options:.storageModeShared))
    let command = try XCTUnwrap(device.makeCommandQueue()?.makeCommandBuffer())
    let e = try XCTUnwrap(command.makeComputeCommandEncoder())
    e.setComputePipelineState(pipeline);e.setBuffer(out,offset:0,index:0);e.setBuffer(u,offset:0,index:1)
    e.dispatchThreads(.init(width:1,height:1,depth:1),threadsPerThreadgroup:.init(width:1,height:1,depth:1))
    e.endEncoding();command.commit();command.waitUntilCompleted()
    XCTAssertEqual(command.status,.completed,"\(String(describing:command.error))")
    let values = out.contents().assumingMemoryBound(to:SIMD4<Float>.self)
    for c in 0..<3 { XCTAssertEqual(values[0][c],values[1][c],accuracy:0.015) }
    XCTAssertEqual(values[2].x,0);XCTAssertEqual(values[2].y,1);XCTAssertEqual(values[2].z,0)
    XCTAssertGreaterThan(values[2].w,0)
  }

  func testWorldGeometryOccludesFiniteEmitter() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    guard device.supportsRaytracing else { throw XCTSkip("Metal ray tracing unavailable") }
    var scene = PhysicsScene(name: "Area light occlusion")
    let body = scene.addBody(
      size: F3(repeating: 0.01), density: 0, friction: 0, position: F3(0, 0, -10))
    let mesh = SurfaceMesh(
      vertices: [
        F3(-0.25, -0.25, 11), F3(0.25, -0.25, 11), F3(0.25, 0.25, 11), F3(-0.25, 0.25, 11),
      ],
      normals: Array(repeating: F3(0, 0, 1), count: 4), triangles: [(0, 1, 2), (0, 2, 3)])
    scene.addRigidMesh(SceneRigidMesh(body: body, mesh: mesh, color: F3(repeating: 1)))
    let solver = try GPUSolver(scene: scene, device: device)
    let resources = try GPUSimMaterialLibrary(device: device)
    let world = try RayTracingScene.shared(scene: solver, materials: resources)
    try world.prepare(scene: solver, revision: 0, auxiliary: [], ground: false)
    let library = try device.makeLibrary(
      source: renderShaderSource + "\n" + rayTracingShaderSource + """
        kernel void area_shadow_probe(instance_acceleration_structure scene [[buffer(0)]],constant Uniforms& U [[buffer(1)]],device float4* out [[buffer(9)]]) {
            out[0]=float4(rtAreaLighting(float3(0),float3(0,0,1),float3(0,0,1),float3(0.6),0.7,0,scene,U,uint2(0)),1);
            out[2]=float4(rtPrimaryPosition(float3(0,0,3),scene,U),1);
            out[1]=float4(rtAreaLighting(float3(2,0,0),float3(0,0,1),normalize(float3(-2,0,3)),float3(0.6),0.7,0,scene,U,uint2(1)),1);
        }
        """, options: nil)
    let pipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(library.makeFunction(name: "area_shadow_probe")))
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
    let values = u.contents().assumingMemoryBound(to: Uniforms.self)
    values.pointee.areaSettings = SIMD4(1, 64, 0, 0)
    values.pointee.areaLights.0 = try GPUSimAreaLight(
      position: F3(0, 0, 3), normal: F3(0, 0, -1), up: F3(0, 1, 0), size: SIMD2(repeating: 0.4),
      radiance: F3(repeating: 20), shape: .disk
    ).record
    // A second visible plane lies beyond a near-clipped foreground plane.
    // The recovery helper must not replace its position with the plane at z=1.
    values.pointee.eye = .zero
    values.pointee.camRight = SIMD4(1,0,0,0)
    values.pointee.camUp = SIMD4(0,1,0,0)
    values.pointee.aoProjection = SIMD4(1.25,-2.5,1,1) // near=2, far=10
    let output = try XCTUnwrap(device.makeBuffer(length: 48, options: .storageModeShared))
    let e = try XCTUnwrap(command.makeComputeCommandEncoder())
    e.setComputePipelineState(pipeline)
    e.setAccelerationStructure(world.structure, bufferIndex: 0)
    e.useResource(world.structure, usage: .read)
    e.setBuffer(u, offset: 0, index: 1)
    e.setBuffer(output, offset: 0, index: 9)
    e.dispatchThreads(
      .init(width: 1, height: 1, depth: 1),
      threadsPerThreadgroup: .init(width: 1, height: 1, depth: 1))
    e.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    XCTAssertEqual(command.status, .completed)
    let result = output.contents().assumingMemoryBound(to: SIMD4<Float>.self)
    XCTAssertEqual(
      result[0].x, 0, accuracy: 0.000001, "Opaque geometry must fully occlude this small disk")
    XCTAssertGreaterThan(result[1].x, 0.01, "An unobstructed receiver must see the finite emitter")
    XCTAssertEqual(result[2].z, 3, accuracy: 0.0001, "Near-clipped geometry cannot replace the raster receiver")
  }
}
