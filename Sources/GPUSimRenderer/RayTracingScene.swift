import Foundation
import Metal
import simd

/// World geometry is shared by renderers viewing the same scene. Its queue
/// orders acceleration-structure updates and all readers across those cameras.
/// Screen-space targets and temporal history remain camera-owned.
@MainActor
final class RayTracingScene {
    enum Failure: Error { case unavailable, allocation(String), invalidGeometry, command(String) }
    struct Vertex {
        var position: SIMD4<Float>
        var normal: SIMD4<Float> // w = roughness
        var albedo: SIMD4<Float> // w = metallic
        var emissive = SIMD4<Float>.zero
    }
    struct Object {
        var vertexStart: UInt32
        var source: UInt32 // 0 rigid primitive, 1 auxiliary, 2 rigid mesh, 3 world-space mesh, 4 ground
        var index: UInt32
        var asset: UInt32
    }
    private struct Asset {
        let descriptor: MTLPrimitiveAccelerationStructureDescriptor
        let structure: MTLAccelerationStructure
        let scratch: MTLBuffer
        let vertexStart, vertexCount: Int
        let deformation: Int // 0 static, 1 soft, 2 skinned
        var built = false
    }
    private final class WeakEntry {
        weak var value: RayTracingScene?
        init(_ value: RayTracingScene) { self.value = value }
    }
    private static var worlds: [ObjectIdentifier: WeakEntry] = [:]
    static func shared(scene: any GPUSimRenderableScene) throws -> RayTracingScene {
        worlds = worlds.filter { $0.value.value?.scene != nil }
        let key = ObjectIdentifier(scene)
        if let existing = worlds[key]?.value { return existing }
        let result = try RayTracingScene(scene: scene)
        worlds[key] = WeakEntry(result)
        return result
    }

    weak var scene: (any GPUSimRenderableScene)?
    let device: MTLDevice
    let queue: MTLCommandQueue
    private let instancesPipeline, deformationPipeline, shadowPipeline, reflectionPipeline: MTLComputePipelineState
    private var assets: [Asset] = []
    private(set) var vertices, objects, descriptors: MTLBuffer!
    private(set) var structure: MTLAccelerationStructure!
    private var topDescriptor: MTLInstanceAccelerationStructureDescriptor!
    private var topScratch: MTLBuffer!
    private var topBuilt = false
    private var updateCount = 0
    private var topologyKey = ""
    private var objectCount = 0
    private let dummy: MTLBuffer

    private init(scene: any GPUSimRenderableScene) throws {
        let device = scene.renderDevice
        guard device.supportsRaytracing else { throw Failure.unavailable }
        self.scene = scene
        self.device = device
        guard let queue = device.makeCommandQueue(),
              let dummy = device.makeBuffer(length: 256, options: .storageModeShared) else { throw Failure.allocation("ray tracing queue") }
        self.queue = queue
        self.dummy = dummy
        let library = try device.makeLibrary(source: renderShaderSource + "\n" + rayTracingShaderSource, options: nil)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let f = library.makeFunction(name: name) else { throw GPUSimRendererError.shaderFunction(name) }
            return try device.makeComputePipelineState(function: f)
        }
        instancesPipeline = try pipeline("rt_instances")
        deformationPipeline = try pipeline("rt_deform")
        shadowPipeline = try pipeline("rt_shadows")
        reflectionPipeline = try pipeline("rt_reflections")
    }

    private func buffer<T>(_ values: [T], label: String) throws -> MTLBuffer {
        guard !values.isEmpty, let result = values.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }) else { throw Failure.allocation(label) }
        result.label = label
        return result
    }

    /// Topology extraction/readback occurs only on scene or geometry revision.
    /// Frame updates below consume the same GPU poses used by rasterization.
    func prepare(scene: any GPUSimRenderableScene, revision: Int,
                 auxiliary: [GPUSimRenderInstance], ground: Bool) throws {
        let mesh = scene.rigidMeshRenderSurface
        let soft = scene.softRenderSurface
        let skin = scene.skinnedRenderSurface
        func identity(_ buffer: MTLBuffer?) -> String { buffer.map { String(describing: ObjectIdentifier($0)) } ?? "nil" }
        let auxKey = auxiliary.map { "\($0.color.w):\($0.parameters.x):\($0.parameters.y)" }.joined(separator: ",")
        let key = "\(revision):\(scene.renderGeometryRevision):\(scene.renderRigidInstanceCount):\(identity(mesh?.vertices)):\(identity(mesh?.indices)):\(mesh?.indexCount ?? 0):\(identity(soft?.triangles)):\(soft?.triangleCount ?? 0):\(identity(skin?.triangles)):\(skin?.triangleCount ?? 0):\(auxKey):\(ground)"
        guard key != topologyKey else { return }
        guard let command = queue.makeCommandBuffer() else { throw Failure.allocation("topology command") }
        let count = scene.renderRigidInstanceCount
        guard let primitiveData = device.makeBuffer(length: max(1, count) * MemoryLayout<GPUSimRenderInstance>.stride,
                                                     options: .storageModeShared) else { throw Failure.allocation("primitive metadata") }
        try scene.encodeRenderInstances(command, instances: primitiveData, colorMode: .bodyIndex, appearanceOverrides: nil)
        var meshVertices: MTLBuffer?, meshIndices: MTLBuffer?
        if let mesh {
            guard let v = device.makeBuffer(length: mesh.vertices.length, options: .storageModeShared),
                  let i = device.makeBuffer(length: mesh.indexCount * 4, options: .storageModeShared),
                  let blit = command.makeBlitCommandEncoder() else { throw Failure.allocation("mesh topology") }
            blit.copy(from: mesh.vertices, sourceOffset: 0, to: v, destinationOffset: 0, size: v.length)
            blit.copy(from: mesh.indices, sourceOffset: 0, to: i, destinationOffset: 0, size: i.length)
            blit.endEncoding()
            meshVertices = v; meshIndices = i
        }
        command.commit(); command.waitUntilCompleted()
        guard command.status == .completed else { throw Failure.command(String(describing: command.error)) }

        var allVertices: [Vertex] = [], allObjects: [Object] = []
        var ranges: [(start: Int, count: Int, deformation: Int)] = []
        var primitiveAssets: [SIMD4<Float>: Int] = [:]
        func addAsset(_ data: [Vertex], deformation: Int = 0) -> Int {
            let index = ranges.count
            ranges.append((allVertices.count, data.count, deformation))
            allVertices.append(contentsOf: data)
            return index
        }
        func addPrimitive(_ instance: GPUSimRenderInstance, source: UInt32, index: Int) {
            let kind = instance.color.w
            let dims = SIMD4(kind, kind >= 2 ? instance.parameters.x : 0,
                             kind >= 2 ? instance.parameters.y : 0, 0)
            let asset: Int
            if let cached = primitiveAssets[dims] { asset = cached }
            else { asset = addAsset(Self.primitiveVertices(instance)); primitiveAssets[dims] = asset }
            allObjects.append(Object(vertexStart: UInt32(ranges[asset].start), source: source, index: UInt32(index), asset: UInt32(asset)))
        }
        let primitives = primitiveData.contents().assumingMemoryBound(to: GPUSimRenderInstance.self)
        for i in 0..<count {
            guard [Float(0),1,2,3].contains(primitives[i].color.w),
                  primitives[i].parameters.x.isFinite, primitives[i].parameters.y.isFinite else { throw Failure.invalidGeometry }
            addPrimitive(primitives[i], source: 0, index: i)
        }
        for (i, instance) in auxiliary.enumerated() { addPrimitive(instance, source: 1, index: i) }
        if let mesh, let meshVertices, let meshIndices {
            let v = meshVertices.contents().assumingMemoryBound(to: GPUSimRigidMeshRenderVertex.self)
            let indices = meshIndices.contents().assumingMemoryBound(to: UInt32.self)
            var groups: [UInt32: [Vertex]] = [:]
            for triangle in stride(from: 0, to: mesh.indexCount, by: 3) {
                guard triangle + 2 < mesh.indexCount else { throw Failure.invalidGeometry }
                let ids = (0..<3).map { Int(indices[triangle + $0]) }
                guard ids.allSatisfy({ $0 < meshVertices.length / MemoryLayout<GPUSimRigidMeshRenderVertex>.stride }) else { throw Failure.invalidGeometry }
                let body = v[ids[0]].positionBody.w.bitPattern
                guard Int(body) < mesh.positions.length / 16, Int(body) < mesh.rotations.length / 16 else { throw Failure.invalidGeometry }
                guard ids.allSatisfy({ v[$0].positionBody.w.bitPattern == body }) else { throw Failure.invalidGeometry }
                for id in ids {
                    let input = v[id]
                    let c = SIMD3(input.color.x, input.color.y, input.color.z)
                    let rough = input.normal.w > 0 ? max(0.02, min(input.normal.w, 1)) : 0.45
                    let metal = input.normal.w > 0 ? max(0, min(input.color.w, 1)) : 0
                    groups[body, default: []].append(Vertex(position: SIMD4(input.positionBody.x, input.positionBody.y, input.positionBody.z, 1),
                        normal: SIMD4(input.normal.x, input.normal.y, input.normal.z, rough),
                        albedo: SIMD4(c*c*(SIMD3(repeating: 0.7)+c*0.3), metal)))
                }
            }
            for body in groups.keys.sorted() {
                let asset = addAsset(groups[body]!)
                allObjects.append(Object(vertexStart: UInt32(ranges[asset].start), source: 2, index: body, asset: UInt32(asset)))
            }
        }
        for (count, deformation) in [(soft?.triangleCount ?? 0, 1), (skin?.triangleCount ?? 0, 2)] where count > 0 {
            let vertex = Vertex(position: .zero, normal: SIMD4(0,0,1,0.72), albedo: SIMD4(0.5,0.5,0.5,0))
            let asset = addAsset(Array(repeating: vertex, count: count*3), deformation: deformation)
            allObjects.append(Object(vertexStart: UInt32(ranges[asset].start), source: 3, index: 0, asset: UInt32(asset)))
        }
        if ground {
            let data = [SIMD3<Float>(-4000,-4000,0.005), SIMD3(4000,-4000,0.005), SIMD3(4000,4000,0.005), SIMD3(-4000,-4000,0.005), SIMD3(4000,4000,0.005), SIMD3(-4000,4000,0.005)]
            let asset = addAsset(data.map { Vertex(position: SIMD4($0,1), normal: SIMD4(0,0,1,1), albedo: SIMD4(0.4,0.4,0.4,0)) })
            allObjects.append(Object(vertexStart: UInt32(ranges[asset].start), source: 4, index: 0, asset: UInt32(asset)))
        }
        // A masked degenerate instance supports empty scenes without invalid zero-count AS builds.
        if allObjects.isEmpty {
            let asset = addAsset(Array(repeating: Vertex(position: .zero, normal: .zero, albedo: .zero), count: 3))
            allObjects.append(Object(vertexStart: 0, source: 5, index: 0, asset: UInt32(asset)))
        }
        let newVertices = try buffer(allVertices, label: "RT object-local vertices")
        let newObjects = try buffer(allObjects, label: "RT object bindings")
        var newAssets: [Asset] = []
        for range in ranges {
            let geometry = MTLAccelerationStructureTriangleGeometryDescriptor()
            geometry.vertexBuffer = newVertices
            geometry.vertexBufferOffset = range.start * MemoryLayout<Vertex>.stride
            geometry.vertexStride = MemoryLayout<Vertex>.stride
            geometry.vertexFormat = .float3
            geometry.triangleCount = range.count / 3
            geometry.opaque = true
            let descriptor = MTLPrimitiveAccelerationStructureDescriptor()
            descriptor.geometryDescriptors = [geometry]
            if range.deformation != 0 { descriptor.usage = .refit }
            let sizes = device.accelerationStructureSizes(descriptor: descriptor)
            guard let structure = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
                  let scratch = device.makeBuffer(length: max(256, max(sizes.buildScratchBufferSize, sizes.refitScratchBufferSize)), options: .storageModePrivate)
            else { throw Failure.allocation("object acceleration structure") }
            newAssets.append(Asset(descriptor: descriptor, structure: structure, scratch: scratch,
                                   vertexStart: range.start, vertexCount: range.count, deformation: range.deformation))
        }
        let top = MTLInstanceAccelerationStructureDescriptor()
        top.usage = .refit
        top.instanceCount = allObjects.count
        top.instancedAccelerationStructures = newAssets.map(\.structure)
        guard let instances = device.makeBuffer(length: allObjects.count * MemoryLayout<MTLAccelerationStructureInstanceDescriptor>.stride,
                                                options: .storageModePrivate) else { throw Failure.allocation("RT instances") }
        top.instanceDescriptorBuffer = instances
        let sizes = device.accelerationStructureSizes(descriptor: top)
        guard let newTop = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
              let scratch = device.makeBuffer(length: max(256, max(sizes.buildScratchBufferSize, sizes.refitScratchBufferSize)), options: .storageModePrivate)
        else { throw Failure.allocation("scene acceleration structure") }
        (vertices, objects, descriptors) = (newVertices, newObjects, instances)
        (assets, topDescriptor, structure, topScratch) = (newAssets, top, newTop, scratch)
        objectCount = allObjects.count
        topBuilt = false
        updateCount = 0
        topologyKey = key
    }

    func invalidateBuilds() {
        topBuilt = false
        for i in assets.indices { assets[i].built = false }
    }

    func encodeUpdate(command: MTLCommandBuffer, scene: any GPUSimRenderableScene,
                      instances: MTLBuffer, auxiliary: MTLBuffer?, appearances: MTLBuffer?) throws {
        guard let e = command.makeComputeCommandEncoder() else { throw Failure.allocation("RT update encoder") }
        e.label = "World instance transforms and deformation"
        e.setComputePipelineState(instancesPipeline)
        e.setBuffer(objects, offset: 0, index: 0)
        e.setBuffer(descriptors, offset: 0, index: 1)
        e.setBuffer(instances, offset: 0, index: 2)
        e.setBuffer(auxiliary ?? dummy, offset: 0, index: 3)
        e.setBuffer(scene.rigidMeshRenderSurface?.positions ?? dummy, offset: 0, index: 4)
        e.setBuffer(scene.rigidMeshRenderSurface?.rotations ?? dummy, offset: 0, index: 5)
        var count = UInt32(objectCount)
        e.setBytes(&count, length: 4, index: 6)
        e.dispatchThreads(MTLSize(width: objectCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        for asset in assets where asset.deformation != 0 {
            e.setComputePipelineState(deformationPipeline)
            e.setBuffer(vertices, offset: asset.vertexStart * MemoryLayout<Vertex>.stride, index: 0)
            if asset.deformation == 1, let soft = scene.softRenderSurface {
                e.setBuffer(soft.triangles, offset: 0, index: 1)
                e.setBuffer(soft.positions, offset: 0, index: 2)
                e.setBuffer(soft.normals, offset: 0, index: 3)
            } else if let skin = scene.skinnedRenderSurface {
                e.setBuffer(skin.triangles, offset: 0, index: 1)
                e.setBuffer(skin.vertices, offset: 0, index: 2)
                e.setBuffer(dummy, offset: 0, index: 3)
            }
            var config = SIMD4<UInt32>(UInt32(asset.vertexCount), UInt32(asset.deformation), appearances == nil ? 0 : 1, 0)
            e.setBytes(&config, length: 16, index: 4)
            e.setBuffer(appearances ?? dummy, offset: 0, index: 5)
            e.dispatchThreads(MTLSize(width: asset.vertexCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        }
        e.endEncoding()
        for i in assets.indices where !assets[i].built || assets[i].deformation != 0 {
            guard let build = command.makeAccelerationStructureCommandEncoder() else { throw Failure.allocation("object build encoder") }
            let asset = assets[i]
            if asset.built && updateCount % 120 != 0 {
                build.refit(sourceAccelerationStructure: asset.structure, descriptor: asset.descriptor,
                            destinationAccelerationStructure: nil, scratchBuffer: asset.scratch, scratchBufferOffset: 0)
            } else {
                build.build(accelerationStructure: asset.structure, descriptor: asset.descriptor,
                            scratchBuffer: asset.scratch, scratchBufferOffset: 0)
            }
            build.endEncoding()
            assets[i].built = true
        }
        guard let build = command.makeAccelerationStructureCommandEncoder() else { throw Failure.allocation("scene build encoder") }
        // Periodic rebuilds bound traversal degradation after large motion
        // and folding; ordinary frames only refit existing structures.
        if topBuilt && updateCount % 120 != 0 {
            build.refit(sourceAccelerationStructure: structure, descriptor: topDescriptor,
                        destinationAccelerationStructure: nil, scratchBuffer: topScratch, scratchBufferOffset: 0)
        } else {
            build.build(accelerationStructure: structure, descriptor: topDescriptor, scratchBuffer: topScratch, scratchBufferOffset: 0)
        }
        build.endEncoding()
        topBuilt = true
        updateCount += 1
    }

    func encodeLighting(command: MTLCommandBuffer, uniforms: Uniforms, screen: ScreenSpacePipeline,
                        instances: MTLBuffer, auxiliary: MTLBuffer?, appearances: MTLBuffer?, reflections: Bool) throws {
        guard let e = command.makeComputeCommandEncoder() else { throw Failure.allocation("ray lighting encoder") }
        e.label = reflections ? "Selective world reflections" : "World directional visibility"
        e.setComputePipelineState(reflections ? reflectionPipeline : shadowPipeline)
        e.setAccelerationStructure(structure, bufferIndex: 0)
        e.useResource(structure, usage: .read)
        for asset in assets { e.useResource(asset.structure, usage: .read) }
        var u = uniforms
        e.setBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        e.setBuffer(vertices, offset: 0, index: 2)
        e.setBuffer(objects, offset: 0, index: 3)
        e.setBuffer(descriptors, offset: 0, index: 4)
        e.setBuffer(instances, offset: 0, index: 5)
        e.setBuffer(auxiliary ?? dummy, offset: 0, index: 6)
        e.setBuffer(appearances ?? dummy, offset: 0, index: 7)
        var hasAppearance: UInt32 = appearances == nil ? 0 : 1
        e.setBytes(&hasAppearance, length: 4, index: 8)
        e.setTexture(screen.depth, index: 0)
        e.setTexture(screen.normal, index: 1)
        e.setTexture(reflections ? screen.reflectionRaw : screen.directVisibilityRaw, index: 2)
        if reflections {
            e.setTexture(screen.material, index: 3)
            e.setTexture(screen.visibility, index: 4)
        }
        e.dispatchThreads(MTLSize(width: screen.halfSize.x, height: screen.halfSize.y, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        e.endEncoding()
    }
}
