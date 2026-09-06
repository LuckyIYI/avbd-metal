import Metal
import simd
import XCTest
@testable import GPUSimRenderer

@MainActor
final class TopologyHistoryTests: XCTestCase {
    private final class Scene: GPUSimRenderableScene {
        let renderDevice: MTLDevice
        var renderBodyCount = 4
        var renderRigidInstanceCount = 0
        var renderGeometryRevision: UInt64 = 0
        var renderStateRevision: UInt64? = 0
        var rendererStateIsValid: Bool { true }
        var renderCameraHint: GPUSimRenderCameraHint { .init() }
        var rigidMeshRenderSurface: GPUSimRigidMeshRenderSurface?
        var softRenderSurface: GPUSimSoftRenderSurface?
        var skinnedRenderSurface: GPUSimSkinnedRenderSurface?
        var convexDebugRenderSurface: GPUSimConvexDebugRenderSurface? { nil }
        init(_ device: MTLDevice) { renderDevice = device }
        func encodeRenderInstances(_ commandBuffer: MTLCommandBuffer, instances: MTLBuffer,
                                   colorMode: GPUSimRenderColorMode, appearanceOverrides: MTLBuffer?) throws {}
    }

    func testAuxiliaryRemovalAndCasterPartitionResetHistoryButMotionDoesNot() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        let renderer = try GPUSimRenderer(device: device)
        var values = (0..<10).map { i in
            GPUSimRenderInstance(model: matrix_identity_float4x4, color: SIMD4(1,1,1,0),
                                 parameters: SIMD4(0,0,1,Float(i % 2)))
        }
        renderer.updateAuxiliaryHistory(values)
        func check(_ reset: Bool) {
            renderer.prevVP = matrix_identity_float4x4
            renderer.updateAuxiliaryHistory(values)
            XCTAssertEqual(renderer.prevVP == nil, reset)
        }
        values[0].model.columns.3.x += 1
        check(false)
        values.removeFirst()
        check(true)
        check(false)
        values[0].parameters.w = 1 - values[0].parameters.w
        check(true)
        values[0].material.w = 0.5
        check(true)
        values.removeAll()
        check(true)
        check(false)
    }

    func testSameSizeTopologyEditsResetHistoryButMotionKeepsIt() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        let scene = Scene(device), renderer = try GPUSimRenderer(device: device)
        func buffer() throws -> MTLBuffer { try XCTUnwrap(device.makeBuffer(length: 256, options: .storageModeShared)) }
        let vertices = try buffer(), indices = try buffer(), corners = try buffer()
        let positions = try buffer(), rotations = try buffer()
        scene.rigidMeshRenderSurface = .init(vertices: vertices, indices: indices, indexCount: 3,
                                            positions: positions, rotations: rotations)
        scene.softRenderSurface = .init(triangles: corners, triangleCount: 1, positions: positions, normals: rotations)
        scene.skinnedRenderSurface = .init(triangles: corners, triangleCount: 1, vertices: vertices)
        renderer.updateHistoryTopology(scene)

        func assertTransition(resets: Bool, _ change: () throws -> Void,
                              file: StaticString = #filePath, line: UInt = #line) rethrows {
            renderer.prevVP = matrix_identity_float4x4
            try change()
            renderer.updateHistoryTopology(scene)
            XCTAssertEqual(renderer.prevVP == nil, resets, file: file, line: line)
        }

        try assertTransition(resets: false) {
            // Frame-owned pose/deformation buffers can rotate every frame.
            // Their identities and state revision must not stop accumulation.
            scene.renderStateRevision = 1
            scene.rigidMeshRenderSurface = try .init(vertices: vertices, indices: indices, indexCount: 3,
                positions: buffer(), rotations: buffer())
            scene.softRenderSurface = try .init(triangles: corners, triangleCount: 1,
                positions: buffer(), normals: buffer())
            scene.skinnedRenderSurface = try .init(triangles: corners, triangleCount: 1, vertices: buffer())
        }
        try assertTransition(resets: true) {
            scene.rigidMeshRenderSurface = try .init(vertices: buffer(), indices: indices, indexCount: 3,
                positions: positions, rotations: rotations)
        }
        try assertTransition(resets: true) {
            scene.rigidMeshRenderSurface = try .init(vertices: vertices, indices: buffer(), indexCount: 3,
                positions: positions, rotations: rotations)
        }
        try assertTransition(resets: true) {
            scene.softRenderSurface = try .init(triangles: buffer(), triangleCount: 1, positions: positions, normals: rotations)
        }
        try assertTransition(resets: true) {
            scene.skinnedRenderSurface = try .init(triangles: buffer(), triangleCount: 1, vertices: vertices)
        }
        assertTransition(resets: true) { scene.renderGeometryRevision += 1 }
        assertTransition(resets: false) {}
        assertTransition(resets: true) { scene.renderRigidInstanceCount += 1 }
        assertTransition(resets: true) { scene.softRenderSurface = nil }
    }
}
