import Metal
import PhysicsAVBD

/// One frame-owned copy of every buffer rasterization and RT may read.
/// Slots are recycled only after the consuming render command completes.
final class RenderSnapshot: GPUSimRenderableScene {
    enum Failure: Error { case allocation, encoder }
    let renderDevice: MTLDevice
    let ready: MTLSharedEvent
    private var generation: UInt64 = 0
    private var buffers: [ObjectIdentifier: MTLBuffer] = [:]
    private var primitiveInstances: MTLBuffer?
    private(set) var submission: MTLCommandBuffer?
    private(set) var renderBodyCount = 0
    private(set) var renderRigidInstanceCount = 0
    private(set) var renderGeometryRevision: UInt64 = 0
    private(set) var renderStateRevision: UInt64?
    private(set) var renderCameraHint = GPUSimRenderCameraHint()
    private(set) var renderContentBounds: GPUSimContentBounds?
    private(set) var softRenderSurface: GPUSimSoftRenderSurface?
    private(set) var skinnedRenderSurface: GPUSimSkinnedRenderSurface?
    private(set) var rigidMeshRenderSurface: GPUSimRigidMeshRenderSurface?
    private(set) var convexDebugRenderSurface: GPUSimConvexDebugRenderSurface?
    var rendererStateIsValid: Bool { submission?.status != .error }
    var renderSceneRequiresFrameRetirement: Bool { false }

    init(device: MTLDevice) throws {
        renderDevice = device
        guard let event = device.makeSharedEvent() else { throw Failure.allocation }
        ready = event
    }

    func capture(solver: GPUSolver, colorMode: GPUSimRenderColorMode,
                 appearances: MTLBuffer?, includeDebug: Bool = false) throws {
        let length = max(1, solver.renderRigidInstanceCount) * MemoryLayout<GPUSimRenderInstance>.stride
        if primitiveInstances?.length != length {
            primitiveInstances = renderDevice.makeBuffer(length: length, options: .storageModePrivate)
        }
        guard let primitives = primitiveInstances else { throw Failure.allocation }
        generation += 1
        let value = generation, event = ready
        submission = try solver.submitRenderSnapshot { command in
            renderBodyCount = solver.renderBodyCount
            renderRigidInstanceCount = solver.renderRigidInstanceCount
            renderGeometryRevision = solver.renderGeometryRevision
            renderStateRevision = solver.renderStateRevision
            renderCameraHint = solver.renderCameraHint
            renderContentBounds = solver.renderContentBounds
            try solver.encodeRenderInstances(command, instances: primitives, colorMode: colorMode,
                                             appearanceOverrides: appearances)
            do {
                guard let blit = command.makeBlitCommandEncoder() else { throw Failure.encoder }
                blit.label = "Copy immutable render surfaces"
                var copied: [ObjectIdentifier: MTLBuffer] = [:]
                func copy(_ source: MTLBuffer) throws -> MTLBuffer {
                    let key = ObjectIdentifier(source)
                    if let existing = copied[key] { return existing }
                    let destination: MTLBuffer
                    if let existing = buffers[key], existing.length == source.length { destination = existing }
                    else {
                        guard let allocated = renderDevice.makeBuffer(length: source.length, options: .storageModePrivate)
                        else { throw Failure.allocation }
                        allocated.label = "Render snapshot: \(source.label ?? "surface")"
                        destination = allocated
                    }
                    blit.copy(from: source, sourceOffset: 0, to: destination, destinationOffset: 0, size: source.length)
                    copied[key] = destination
                    return destination
                }
                defer { blit.endEncoding() }
                softRenderSurface = try solver.softRenderSurface.map {
                    try .init(triangles: copy($0.triangles), triangleCount: $0.triangleCount,
                              positions: copy($0.positions), normals: copy($0.normals))
                }
                skinnedRenderSurface = try solver.skinnedRenderSurface.map {
                    try .init(triangles: copy($0.triangles), triangleCount: $0.triangleCount, vertices: copy($0.vertices))
                }
                rigidMeshRenderSurface = try solver.rigidMeshRenderSurface.map {
                    try .init(vertices: copy($0.vertices), indices: copy($0.indices), indexCount: $0.indexCount,
                              positions: copy($0.positions), rotations: copy($0.rotations))
                }
                convexDebugRenderSurface = try (includeDebug ? solver.convexDebugRenderSurface : nil).map {
                    try .init(triangleVertices: copy($0.triangleVertices), triangleVertexCount: $0.triangleVertexCount,
                              edgeVertices: copy($0.edgeVertices), edgeVertexCount: $0.edgeVertexCount,
                              positions: copy($0.positions), rotations: copy($0.rotations))
                }
                buffers = copied
            }
            command.encodeSignalEvent(event, value: value)
            // A failed producer must release its consumer too; both completion
            // paths report the error rather than leaving a GPU event deadlocked.
            command.addCompletedHandler { finished in
                if finished.status == .error { event.signaledValue = value }
            }
        }
    }

    func encodeRenderInstances(_ commandBuffer: MTLCommandBuffer, instances: MTLBuffer,
                               colorMode: GPUSimRenderColorMode, appearanceOverrides: MTLBuffer?) throws {
        commandBuffer.encodeWaitForEvent(ready, value: generation)
        guard let source = primitiveInstances, let blit = commandBuffer.makeBlitCommandEncoder() else { throw Failure.encoder }
        blit.label = "Snapshot primitive instances"
        blit.copy(from: source, sourceOffset: 0, to: instances, destinationOffset: 0, size: source.length)
        blit.endEncoding()
    }
}
