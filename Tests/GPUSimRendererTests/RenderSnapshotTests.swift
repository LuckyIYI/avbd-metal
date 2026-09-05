import MetalKit
import SimCore
import PhysicsAVBD
import XCTest
@testable import GPUSimRenderer

@MainActor
final class RenderSnapshotTests: XCTestCase {
    func testPhysicsAdvancesWhileRenderWaitsAndSnapshotKeepsOldGeometry() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        var scene = PhysicsScene(name: "snapshot-ownership")
        let body = scene.addBody(size: F3(repeating: 0.2), density: 1000, friction: 0, position: F3(0,0,2))
        let p = [F3(2,0,3),F3(2.5,0,3),F3(2,0.5,3),F3(2,0,3.5)].map {
            scene.addParticle(radius: 0.05, mass: 0.1, position: $0)
        }
        scene.addTet(SceneTet(ids: (p[0],p[1],p[2],p[3]), mu: 2000, lambda: 20000))
        let solver = try GPUSolver(scene: scene, device: device)
        let snapshot = try RenderSnapshot(device: device)
        try snapshot.capture(solver: solver, colorMode: .bodyIndex, appearances: nil)
        let surface = try XCTUnwrap(snapshot.softRenderSurface)
        XCTAssertFalse(surface.positions === solver.softRenderSurface?.positions)
        XCTAssertFalse(surface.normals === solver.softRenderSurface?.normals)
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let render = try XCTUnwrap(queue.makeCommandBuffer())
        let gate = try XCTUnwrap(device.makeSharedEvent())
        let instances = try XCTUnwrap(device.makeBuffer(length: max(1,snapshot.renderRigidInstanceCount)*112, options: .storageModeShared))
        let positions = try XCTUnwrap(device.makeBuffer(length: surface.positions.length, options: .storageModeShared))
        render.encodeWaitForEvent(gate, value: 1)
        try snapshot.encodeRenderInstances(render, instances: instances, colorMode: .bodyIndex, appearanceOverrides: nil)
        let blit = try XCTUnwrap(render.makeBlitCommandEncoder())
        blit.copy(from: surface.positions, sourceOffset: 0, to: positions, destinationOffset: 0, size: positions.length)
        blit.endEncoding(); render.commit()
        // Bound failures: this releases a broken implementation instead of hanging the test.
        DispatchQueue.global().asyncAfter(deadline: .now()+5) { gate.signaledValue = 1 }
        defer { gate.signaledValue = 1; render.waitUntilCompleted() }
        solver.setBodyPose(body, position: F3(4,0,2), rotation: solver.bodyRotation(body))
        try solver.submitStep(); try solver.synchronize()
        XCTAssertLessThan(gate.signaledValue,1,"physics must complete without waiting for the render consumer")
        XCTAssertNotEqual(render.status,.completed)
        XCTAssertEqual(solver.bodyPosition(body).x,4,accuracy: 0.001)
        gate.signaledValue = 1; render.waitUntilCompleted()
        XCTAssertEqual(render.status,.completed)
        let old = instances.contents().assumingMemoryBound(to: GPUSimRenderInstance.self).pointee
        XCTAssertEqual(old.model.columns.3.x,0,accuracy: 0.001)
        let oldPosition = positions.contents().assumingMemoryBound(to: SIMD4<Float>.self)[p[0]]
        XCTAssertEqual(oldPosition.z,3,accuracy: 0.001)
        XCTAssertLessThan(solver.bodyPosition(p[0]).z,oldPosition.z)
        // Reusing the retired slot captures the new state, without rebuilding storage.
        try snapshot.capture(solver: solver, colorMode: .bodyIndex, appearances: nil)
        XCTAssertTrue(snapshot.softRenderSurface?.positions === surface.positions)
        try solver.synchronize()
    }

    func testSkinnedAndRigidMeshSnapshotsOwnTheirBuffers() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        var scene = PhysicsScene(name: "snapshot-meshes")
        let body = scene.addBody(size: F3(repeating: 0.2), density: 1000, friction: 0, position: F3(0,0,2))
        scene.addRigidMesh(SceneRigidMesh(body: body, mesh: SurfaceMesh(
            vertices: [F3(0,0,0),F3(1,0,0),F3(0,1,0)], normals: [], triangles: [(0,1,2)])))
        let ids = (0..<4).map { scene.addParticle(radius: 0.01, mass: 0.1, position: F3(Float($0),0,4)) }
        scene.addSkinnedMesh(SceneSkinnedMesh(vertices: (0..<3).map { i in
            var weights = SIMD4<Float>.zero; weights[i] = 1
            return SceneSkinnedVertex(ids: (ids[0],ids[1],ids[2],ids[3]), weights: weights, restNormal: F3(0,0,1))
        }, triangles: [(0,1,2)], bodyIDs: ids))
        let solver = try GPUSolver(scene: scene, device: device)
        let snapshot = try RenderSnapshot(device: device)
        try snapshot.capture(solver: solver, colorMode: .bodyIndex, appearances: nil)
        try solver.synchronize()
        let skin = try XCTUnwrap(snapshot.skinnedRenderSurface)
        let mesh = try XCTUnwrap(snapshot.rigidMeshRenderSurface)
        XCTAssertFalse(skin.vertices === solver.skinnedRenderSurface?.vertices)
        XCTAssertFalse(mesh.positions === solver.rigidMeshRenderSurface?.positions)
        XCTAssertFalse(mesh.vertices === solver.rigidMeshRenderSurface?.vertices)
        func read(_ buffer: MTLBuffer) throws -> Data {
            let destination = try XCTUnwrap(device.makeBuffer(length: buffer.length, options: .storageModeShared))
            let command = try XCTUnwrap(device.makeCommandQueue()?.makeCommandBuffer())
            let blit = try XCTUnwrap(command.makeBlitCommandEncoder())
            blit.copy(from: buffer, sourceOffset: 0, to: destination, destinationOffset: 0, size: buffer.length)
            blit.endEncoding(); command.commit(); command.waitUntilCompleted()
            XCTAssertEqual(command.status,.completed)
            return Data(bytes: destination.contents(), count: destination.length)
        }
        let oldSkin = try read(skin.vertices), oldPose = try read(mesh.positions)
        solver.setBodyPose(body, position: F3(7,0,2), rotation: solver.bodyRotation(body))
        solver.setBodyPose(ids[0], position: F3(8,0,4), rotation: solver.bodyRotation(ids[0]))
        let next = try RenderSnapshot(device: device)
        try next.capture(solver: solver, colorMode: .bodyIndex, appearances: nil)
        try solver.synchronize()
        XCTAssertEqual(try read(skin.vertices),oldSkin)
        XCTAssertEqual(try read(mesh.positions),oldPose)
        XCTAssertNotEqual(try read(XCTUnwrap(next.skinnedRenderSurface).vertices),oldSkin)
        XCTAssertNotEqual(try read(XCTUnwrap(next.rigidMeshRenderSurface).positions),oldPose)
    }

    func testSolverFramesUseSnapshotRingAcrossFastHQAndResize() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        var scene = PhysicsScene(name: "snapshot-ring")
        _ = scene.addBody(size: F3(repeating: 0.2), density: 1000, friction: 0, position: F3(0,0,2))
        let solver = try GPUSolver(scene: scene, device: device)
        let world = device.supportsRaytracing ? try RayTracingScene.shared(scene: solver) : nil
        let renderer = try GPUSimRenderer(device: device, solver: solver)
        let view = MTKView(frame: CGRect(x: 0,y: 0,width: 128,height: 96),device: device)
        renderer.configure(view); view.isPaused = true
        let completed = expectation(description: "snapshot frames complete")
        completed.expectedFulfillmentCount = 10
        renderer.frameCompletionHandler = { _,_ in completed.fulfill() }
        for frame in 0..<10 {
            try solver.submitStep(); try solver.synchronize()
            renderer.options = frame<4 || frame>7 ? .lightweight : .qualityBeta
            if frame == 6 { view.drawableSize = CGSize(width: 160,height: 112) }
            view.draw()
            XCTAssertNil(renderer.runtimeFailure)
            if frame>4 && frame<8 && GPUSimRenderer.supportsHQ(device: device) {
                XCTAssertEqual(world?.lastUpdate.primitiveBuilds,0,"snapshot ring rotation must not rebuild topology")
                XCTAssertEqual(world?.lastUpdate.instanceRefits,1,"moving snapshots refit the existing world")
            }
        }
        await fulfillment(of: [completed], timeout: 10)
        XCTAssertNil(renderer.runtimeFailure)
    }
}
