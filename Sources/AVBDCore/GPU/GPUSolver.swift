import Foundation
import Metal
import simd

// Metal GPU implementation of Augmented Vertex Block Descent.
//
// Per-step pipeline (matches the CPU reference and paper Algorithm 1):
//   1. Broadphase: spatial-hash grouping (count/scan/scatter) + pair gen
//   2. Narrowphase: SAT OBB-OBB, warm-started from previous frame manifolds
//      via a pair-keyed hash map and contact feature IDs
//   3. Joint + body warm starting (Eq. 19, adaptive initialization)
//   4. CSR adjacency rebuild + incremental parallel greedy coloring
//   5. n iterations of: per-color primal solve (6x6 LDL) + dual update
//   6. BDF1 velocity finalize
public final class GPUSolver {
    public let device: MTLDevice
    let queue: MTLCommandQueue

    public var settings = SimSettings()

    // Capacities
    let numBodies: Int
    let numJoints: Int
    let numSprings: Int
    let maxPairs: Int
    let mapCapacity: Int
    let gridHashSize: Int

    var params = SimParamsGPU()

    // Body SoA buffers
    var posLin, posAng, initLin, initAng, inertLin, inertAng: MTLBuffer
    var velLin, velAng, prevVelLin: MTLBuffer
    var props, shape: MTLBuffer

    // Constraints
    var joints: MTLBuffer
    var springs: MTLBuffer
    var manifolds: MTLBuffer       // current frame
    var prevManifolds: MTLBuffer   // previous frame (swapped)

    // Broadphase
    var hashedIdx, globalIdx: MTLBuffer
    var cellCount, cellStart, cellBodies: MTLBuffer
    var bodyCellSlot: MTLBuffer
    var pairs: MTLBuffer
    var exclusions: MTLBuffer
    var numExclusions: UInt32 = 0

    // Persistence map
    var mapKeyA, mapKeyB, mapVal: MTLBuffer

    // Adjacency + coloring
    var degrees, adjStart, adjCursor, adjList: MTLBuffer
    var colorsA, colorsB, bodySlot, colorStart, colorList: MTLBuffer
    var changedFlag: MTLBuffer

    // Control
    var counters: MTLBuffer
    var dispatchArgs: MTLBuffer    // 9 uints (pairs / forces / diag)
    var colorArgs: MTLBuffer       // MAX_COLORS * 3 uints
    var scanBlockSums: MTLBuffer
    var scanTotal: MTLBuffer
    var diag: MTLBuffer

    // Pipelines
    var pso: [String: MTLComputePipelineState] = [:]

    // Cached per-frame color counts (read back once per step)
    public private(set) var lastColorCounts: [Int] = []
    public private(set) var lastNumPairs: Int = 0
    // Start pessimistic so the first frame covers every possible color.
    var lastMaxColorUsed: Int = AVBD_MAX_COLORS - 1
    public private(set) var frameIndex: Int = 0

    public init(scene: PhysicsScene, device: MTLDevice? = nil,
                maxPairsPerBody: Int = 16) throws {
        guard let dev = device ?? MTLCreateSystemDefaultDevice() else {
            throw AVBDError.noDevice
        }
        self.device = dev
        guard let q = dev.makeCommandQueue() else { throw AVBDError.noDevice }
        self.queue = q

        self.settings = scene.settings
        self.numBodies = scene.bodies.count
        self.numJoints = scene.joints.count
        self.numSprings = scene.springs.count
        self.maxPairs = max(64, scene.bodies.count * maxPairsPerBody)
        self.mapCapacity = Self.nextPow2(2 * maxPairs)
        self.gridHashSize = Self.nextPow2(max(64, 2 * numBodies))

        func makeBuf(_ length: Int, _ label: String) throws -> MTLBuffer {
            guard let b = dev.makeBuffer(length: max(16, length), options: .storageModeShared) else {
                throw AVBDError.allocFailed(label)
            }
            b.label = label
            return b
        }

        let nb = numBodies
        posLin = try makeBuf(nb * 16, "posLin")
        posAng = try makeBuf(nb * 16, "posAng")
        initLin = try makeBuf(nb * 16, "initLin")
        initAng = try makeBuf(nb * 16, "initAng")
        inertLin = try makeBuf(nb * 16, "inertLin")
        inertAng = try makeBuf(nb * 16, "inertAng")
        velLin = try makeBuf(nb * 16, "velLin")
        velAng = try makeBuf(nb * 16, "velAng")
        prevVelLin = try makeBuf(nb * 16, "prevVelLin")
        props = try makeBuf(nb * 16, "props")
        shape = try makeBuf(nb * 16, "shape")

        joints = try makeBuf(max(1, numJoints) * MemoryLayout<JointGPU>.stride, "joints")
        springs = try makeBuf(max(1, numSprings) * MemoryLayout<SpringGPU>.stride, "springs")
        manifolds = try makeBuf(maxPairs * MemoryLayout<ManifoldGPU>.stride, "manifolds")
        prevManifolds = try makeBuf(maxPairs * MemoryLayout<ManifoldGPU>.stride, "prevManifolds")

        hashedIdx = try makeBuf(nb * 4, "hashedIdx")
        globalIdx = try makeBuf(nb * 4, "globalIdx")
        cellCount = try makeBuf(gridHashSize * 4, "cellCount")
        cellStart = try makeBuf(gridHashSize * 4, "cellStart")
        cellBodies = try makeBuf(nb * 4, "cellBodies")
        bodyCellSlot = try makeBuf(nb * 8, "bodyCellSlot")
        pairs = try makeBuf(maxPairs * 8, "pairs")
        exclusions = try makeBuf(max(1, scene.joints.count + scene.springs.count) * 8, "exclusions")

        mapKeyA = try makeBuf(mapCapacity * 4, "mapKeyA")
        mapKeyB = try makeBuf(mapCapacity * 4, "mapKeyB")
        mapVal = try makeBuf(mapCapacity * 4, "mapVal")

        degrees = try makeBuf(nb * 4, "degrees")
        adjStart = try makeBuf(nb * 4, "adjStart")
        adjCursor = try makeBuf(nb * 4, "adjCursor")
        adjList = try makeBuf(2 * (numJoints + numSprings + maxPairs) * 4, "adjList")
        colorsA = try makeBuf(nb * 4, "colorsA")
        colorsB = try makeBuf(nb * 4, "colorsB")
        bodySlot = try makeBuf(nb * 4, "bodySlot")
        colorStart = try makeBuf((AVBD_MAX_COLORS + 1) * 4, "colorStart")
        colorList = try makeBuf(nb * 4, "colorList")
        changedFlag = try makeBuf(4, "changedFlag")

        counters = try makeBuf(GPUCounters.total * 4, "counters")
        dispatchArgs = try makeBuf(9 * 4, "dispatchArgs")
        colorArgs = try makeBuf(AVBD_MAX_COLORS * 3 * 4, "colorArgs")
        let maxScanCount = max(gridHashSize, nb)
        scanBlockSums = try makeBuf(((maxScanCount + 1023) / 1024 + 1) * 4, "scanBlockSums")
        scanTotal = try makeBuf(4, "scanTotal")
        diag = try makeBuf(4, "diag")

        try buildPipelines()
        try upload(scene: scene)
    }

    public enum AVBDError: Error {
        case noDevice
        case allocFailed(String)
        case shaderCompile(String)
        case kernelMissing(String)
    }

    static func nextPow2(_ v: Int) -> Int {
        var p = 1
        while p < v { p <<= 1 }
        return p
    }

    // MARK: - Shader compilation

    private func buildPipelines() throws {
        let lib: MTLLibrary
        do {
            lib = try Self.makeLibrary(device: device)
        } catch {
            throw AVBDError.shaderCompile("\(error)")
        }
        for name in lib.functionNames {
            guard let fn = lib.makeFunction(name: name) else { continue }
            pso[name] = try device.makeComputePipelineState(function: fn)
        }
    }

    /// Concatenates the bundled .metal sources (filename order) and compiles.
    static func makeLibrary(device: MTLDevice) throws -> MTLLibrary {
        let urls = (Bundle.module.urls(forResourcesWithExtension: "metal", subdirectory: nil) ?? [])
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        precondition(!urls.isEmpty, "no .metal resources found in bundle")
        var source = ""
        for url in urls {
            var text = try String(contentsOf: url, encoding: .utf8)
            // Strip duplicate includes/usings; one set at the top is enough.
            text = text.replacingOccurrences(of: "#include <metal_stdlib>", with: "")
            text = text.replacingOccurrences(of: "using namespace metal;", with: "")
            source += text + "\n"
        }
        source = "#include <metal_stdlib>\nusing namespace metal;\n" + source
        let options = MTLCompileOptions()
        if #available(macOS 15.0, *) {
            options.mathMode = .fast
        } else {
            options.fastMathEnabled = true
        }
        return try device.makeLibrary(source: source, options: options)
    }

    // MARK: - Scene upload

    private func upload(scene: PhysicsScene) throws {
        let pl = posLin.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        let pa = posAng.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        let vl = velLin.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        let va = velAng.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        let pv = prevVelLin.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        let pr = props.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        let sh = shape.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        let colA = colorsA.contents().bindMemory(to: UInt32.self, capacity: numBodies)
        let colB = colorsB.contents().bindMemory(to: UInt32.self, capacity: numBodies)

        var radii: [Float] = []
        for (i, b) in scene.bodies.enumerated() {
            let mass = b.density > 0 ? b.size.x * b.size.y * b.size.z * b.density : 0
            let moment = F3(
                (b.size.y * b.size.y + b.size.z * b.size.z) / 12 * mass,
                (b.size.x * b.size.x + b.size.z * b.size.z) / 12 * mass,
                (b.size.x * b.size.x + b.size.y * b.size.y) / 12 * mass
            )
            let radius = length(b.size * 0.5)
            pl[i] = SIMD4(b.position, mass)
            pa[i] = SIMD4(b.rotation.imag, b.rotation.real)
            vl[i] = SIMD4(b.velocity, 0)
            va[i] = .zero
            pv[i] = SIMD4(b.velocity, 0)
            pr[i] = SIMD4(moment, b.friction)
            sh[i] = SIMD4(b.size, radius)
            colA[i] = UInt32(i % AVBD_MAX_COLORS)  // initial guess; refined per frame
            colB[i] = colA[i]
            if mass > 0 { radii.append(radius) }
        }

        // Partition into hashed vs global bodies. Globals: statics or bodies
        // far larger than the median dynamic radius (keeps grid cells tight).
        let medianRadius = radii.sorted()[max(0, radii.count / 2 - (radii.isEmpty ? 0 : 0))]
        let threshold = medianRadius * 4
        var hashed: [UInt32] = []
        var globals: [UInt32] = []
        for (i, b) in scene.bodies.enumerated() {
            let radius = length(b.size * 0.5)
            let isStatic = !(b.density > 0)
            if isStatic || radius > threshold {
                globals.append(UInt32(i))
            } else {
                hashed.append(UInt32(i))
            }
        }
        // Cell size: 2x the max hashed radius (sphere-bound broadphase)
        var maxHashedRadius: Float = 0.5
        for i in hashed {
            maxHashedRadius = max(maxHashedRadius, sh[Int(i)].w)
        }

        hashedIdx.contents().bindMemory(to: UInt32.self, capacity: max(1, hashed.count))
            .update(from: hashed.isEmpty ? [0] : hashed, count: max(1, hashed.count))
        globalIdx.contents().bindMemory(to: UInt32.self, capacity: max(1, globals.count))
            .update(from: globals.isEmpty ? [0] : globals, count: max(1, globals.count))

        // Joints
        let jp = joints.contents().bindMemory(to: JointGPU.self, capacity: max(1, numJoints))
        for (i, j) in scene.joints.enumerated() {
            var g = JointGPU()
            let aIdx: UInt32 = j.bodyA >= 0 ? UInt32(j.bodyA) : 0xFFFFFFFF
            // Flag bits avoid inf comparisons under fast math:
            // 1 = hard linear, 2 = hard angular, 4 = breakable
            var flags: UInt32 = 0
            if j.stiffnessLin.isInfinite { flags |= 1 }
            if j.stiffnessAng.isInfinite { flags |= 2 }
            if j.fracture.isFinite { flags |= 4 }
            g.header = SIMD4(aIdx, UInt32(j.bodyB), 0, flags)
            let bigK: Float = 3.0e10
            g.rA = SIMD4(j.rA, min(j.stiffnessLin, bigK))
            g.rB = SIMD4(j.rB, min(j.stiffnessAng, bigK))
            let sizeA = j.bodyA >= 0 ? scene.bodies[j.bodyA].size : .zero
            let torqueArm = length_squared(sizeA + scene.bodies[j.bodyB].size)
            g.C0Lin = SIMD4(0, 0, 0, torqueArm)
            g.C0Ang = SIMD4(0, 0, 0, min(j.fracture, 3.0e18))
            jp[i] = g
        }

        // Springs (rest length resolved here if negative)
        let sp = springs.contents().bindMemory(to: SpringGPU.self, capacity: max(1, numSprings))
        for (i, s) in scene.springs.enumerated() {
            var g = SpringGPU()
            g.header = SIMD4(UInt32(s.bodyA), UInt32(s.bodyB), 0, 0)
            var rest = s.rest
            if rest < 0 {
                let a = scene.bodies[s.bodyA], b = scene.bodies[s.bodyB]
                let pA = a.position + a.rotation.act(s.rA)
                let pB = b.position + b.rotation.act(s.rB)
                rest = length(pA - pB)
            }
            g.rA = SIMD4(s.rA, s.stiffness)
            g.rB = SIMD4(s.rB, rest)
            sp[i] = g
        }

        // Collision exclusions: sorted (min,max) pairs of jointed/springed bodies
        var excl: Set<UInt64> = []
        for j in scene.joints where j.bodyA >= 0 {
            let lo = UInt64(min(j.bodyA, j.bodyB)), hi = UInt64(max(j.bodyA, j.bodyB))
            excl.insert(lo << 32 | hi)
        }
        for s in scene.springs {
            let lo = UInt64(min(s.bodyA, s.bodyB)), hi = UInt64(max(s.bodyA, s.bodyB))
            excl.insert(lo << 32 | hi)
        }
        let sortedExcl = excl.sorted()
        let ep = exclusions.contents().bindMemory(to: SIMD2<UInt32>.self, capacity: max(1, sortedExcl.count))
        for (i, key) in sortedExcl.enumerated() {
            ep[i] = SIMD2(UInt32(key >> 32), UInt32(key & 0xFFFFFFFF))
        }
        numExclusions = UInt32(sortedExcl.count)

        // Clear manifolds + map
        memset(manifolds.contents(), 0, manifolds.length)
        memset(prevManifolds.contents(), 0, prevManifolds.length)
        memset(mapKeyA.contents(), 0, mapKeyA.length)
        memset(counters.contents(), 0, counters.length)

        // Params
        params.numBodies = UInt32(numBodies)
        params.numJoints = UInt32(numJoints)
        params.numSprings = UInt32(numSprings)
        params.mapCapacity = UInt32(mapCapacity)
        params.maxManifolds = UInt32(maxPairs)
        params.maxPairs = UInt32(maxPairs)
        params.cellSize = maxHashedRadius * 2
        params.gridHashSize = UInt32(gridHashSize)
        params.numHashed = UInt32(hashed.count)
        params.numGlobals = UInt32(globals.count)
    }

    // MARK: - Dispatch helpers

    private func ps(_ name: String) -> MTLComputePipelineState {
        guard let p = pso[name] else { fatalError("missing kernel \(name)") }
        return p
    }

    private func syncParams() {
        params.dt = settings.dt
        params.gravity = settings.gravity
        params.alpha = settings.alpha
        params.betaLin = settings.betaLin
        params.betaAng = settings.betaAng
        params.gamma = settings.gamma
    }

    private func dispatch1D(_ enc: MTLComputeCommandEncoder, _ name: String, _ count: Int,
                            _ setup: (MTLComputeCommandEncoder) -> Void) {
        guard count > 0 else { return }
        let p = ps(name)
        enc.setComputePipelineState(p)
        setup(enc)
        let tg = min(p.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreadgroups(MTLSize(width: (count + tg - 1) / tg, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
    }

    private func dispatchIndirect(_ enc: MTLComputeCommandEncoder, _ name: String,
                                  argsOffset: Int,
                                  _ setup: (MTLComputeCommandEncoder) -> Void) {
        let p = ps(name)
        enc.setComputePipelineState(p)
        setup(enc)
        enc.dispatchThreadgroups(indirectBuffer: dispatchArgs,
                                 indirectBufferOffset: argsOffset * 4,
                                 threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }

    /// Exclusive scan: input -> output (counts of `count` uints).
    private func encodeScan(_ enc: MTLComputeCommandEncoder,
                            input: MTLBuffer, output: MTLBuffer, count: Int) {
        var c = UInt32(count)
        let blocks = (count + 1023) / 1024
        let p1 = ps("scan_blocks")
        enc.setComputePipelineState(p1)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(output, offset: 0, index: 1)
        enc.setBuffer(scanBlockSums, offset: 0, index: 2)
        enc.setBytes(&c, length: 4, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: blocks, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        var nb = UInt32(blocks)
        let p2 = ps("scan_block_sums")
        enc.setComputePipelineState(p2)
        enc.setBuffer(scanBlockSums, offset: 0, index: 0)
        enc.setBytes(&nb, length: 4, index: 1)
        enc.setBuffer(scanTotal, offset: 0, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        dispatch1D(enc, "scan_add_offsets", count) { e in
            e.setBuffer(output, offset: 0, index: 0)
            e.setBuffer(self.scanBlockSums, offset: 0, index: 1)
            e.setBytes(&c, length: 4, index: 2)
        }
    }

    // MARK: - Step

    public func step() {
        syncParams()
        frameIndex += 1

        // ---- Command buffer 1: collision + warm start + adjacency + coloring
        guard let cmd1 = queue.makeCommandBuffer() else { return }

        if let blit = cmd1.makeBlitCommandEncoder() {
            blit.fill(buffer: counters, range: 0..<counters.length, value: 0)
            blit.fill(buffer: cellCount, range: 0..<(gridHashSize * 4), value: 0)
            blit.endEncoding()
        }

        guard let enc = cmd1.makeComputeCommandEncoder() else { return }
        var P = params
        var nExcl = numExclusions

        // Broadphase
        dispatch1D(enc, "bp_count", Int(P.numHashed)) { e in
            e.setBuffer(self.posLin, offset: 0, index: 0)
            e.setBuffer(self.hashedIdx, offset: 0, index: 1)
            e.setBuffer(self.cellCount, offset: 0, index: 2)
            e.setBuffer(self.bodyCellSlot, offset: 0, index: 3)
            e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 4)
        }
        encodeScan(enc, input: cellCount, output: cellStart, count: gridHashSize)
        dispatch1D(enc, "bp_scatter", Int(P.numHashed)) { e in
            e.setBuffer(self.hashedIdx, offset: 0, index: 0)
            e.setBuffer(self.bodyCellSlot, offset: 0, index: 1)
            e.setBuffer(self.cellStart, offset: 0, index: 2)
            e.setBuffer(self.cellBodies, offset: 0, index: 3)
            e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 4)
        }
        dispatch1D(enc, "bp_gen_pairs", Int(P.numHashed)) { e in
            e.setBuffer(self.posLin, offset: 0, index: 0)
            e.setBuffer(self.shape, offset: 0, index: 1)
            e.setBuffer(self.hashedIdx, offset: 0, index: 2)
            e.setBuffer(self.cellStart, offset: 0, index: 3)
            e.setBuffer(self.cellCount, offset: 0, index: 4)
            e.setBuffer(self.cellBodies, offset: 0, index: 5)
            e.setBuffer(self.globalIdx, offset: 0, index: 6)
            e.setBuffer(self.exclusions, offset: 0, index: 7)
            e.setBytes(&nExcl, length: 4, index: 8)
            e.setBuffer(self.counters, offset: 0, index: 9)
            e.setBuffer(self.pairs, offset: 0, index: 10)
            e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 11)
        }
        dispatch1D(enc, "bp_gen_global_pairs", Int(P.numGlobals)) { e in
            e.setBuffer(self.posLin, offset: 0, index: 0)
            e.setBuffer(self.shape, offset: 0, index: 1)
            e.setBuffer(self.globalIdx, offset: 0, index: 2)
            e.setBuffer(self.exclusions, offset: 0, index: 3)
            e.setBytes(&nExcl, length: 4, index: 4)
            e.setBuffer(self.counters, offset: 0, index: 5)
            e.setBuffer(self.pairs, offset: 0, index: 6)
            e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 7)
        }
        dispatch1D(enc, "bp_finalize_pairs", 1) { e in
            e.setBuffer(self.counters, offset: 0, index: 0)
            e.setBuffer(self.dispatchArgs, offset: 0, index: 1)
            e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 2)
        }

        // Narrowphase (warm-start from prev manifolds + map)
        dispatchIndirect(enc, "np_collide", argsOffset: 0) { e in
            e.setBuffer(self.posLin, offset: 0, index: 0)
            e.setBuffer(self.posAng, offset: 0, index: 1)
            e.setBuffer(self.shape, offset: 0, index: 2)
            e.setBuffer(self.props, offset: 0, index: 3)
            e.setBuffer(self.pairs, offset: 0, index: 4)
            e.setBuffer(self.counters, offset: 0, index: 5)
            e.setBuffer(self.manifolds, offset: 0, index: 6)
            e.setBuffer(self.prevManifolds, offset: 0, index: 7)
            e.setBuffer(self.mapKeyA, offset: 0, index: 8)
            e.setBuffer(self.mapKeyB, offset: 0, index: 9)
            e.setBuffer(self.mapVal, offset: 0, index: 10)
            e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 11)
        }

        // Rebuild persistence map from THIS frame's manifolds (for next frame)
        dispatch1D(enc, "pm_clear", mapCapacity) { e in
            e.setBuffer(self.mapKeyA, offset: 0, index: 0)
            var cap = UInt32(self.mapCapacity)
            e.setBytes(&cap, length: 4, index: 1)
        }
        dispatchIndirect(enc, "pm_insert", argsOffset: 0) { e in
            e.setBuffer(self.manifolds, offset: 0, index: 0)
            e.setBuffer(self.mapKeyA, offset: 0, index: 1)
            e.setBuffer(self.mapKeyB, offset: 0, index: 2)
            e.setBuffer(self.mapVal, offset: 0, index: 3)
            e.setBuffer(self.counters, offset: 0, index: 4)
            e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 5)
        }

        // Warm start joints (before body prediction; uses start-of-step poses)
        dispatch1D(enc, "warmstart_joints", numJoints) { e in
            e.setBuffer(self.posLin, offset: 0, index: 0)
            e.setBuffer(self.posAng, offset: 0, index: 1)
            e.setBuffer(self.joints, offset: 0, index: 2)
            e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 3)
        }
        dispatch1D(enc, "warmstart_bodies", numBodies) { e in
            e.setBuffer(self.posLin, offset: 0, index: 0)
            e.setBuffer(self.posAng, offset: 0, index: 1)
            e.setBuffer(self.initLin, offset: 0, index: 2)
            e.setBuffer(self.initAng, offset: 0, index: 3)
            e.setBuffer(self.inertLin, offset: 0, index: 4)
            e.setBuffer(self.inertAng, offset: 0, index: 5)
            e.setBuffer(self.velLin, offset: 0, index: 6)
            e.setBuffer(self.velAng, offset: 0, index: 7)
            e.setBuffer(self.prevVelLin, offset: 0, index: 8)
            e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 9)
        }

        // Adjacency
        var nb32 = UInt32(numBodies)
        dispatch1D(enc, "adj_clear_degrees", numBodies) { e in
            e.setBuffer(self.degrees, offset: 0, index: 0)
            e.setBytes(&nb32, length: 4, index: 1)
        }
        dispatchIndirect(enc, "adj_count", argsOffset: 3) { e in
            e.setBuffer(self.posLin, offset: 0, index: 0)
            e.setBuffer(self.joints, offset: 0, index: 1)
            e.setBuffer(self.springs, offset: 0, index: 2)
            e.setBuffer(self.manifolds, offset: 0, index: 3)
            e.setBuffer(self.degrees, offset: 0, index: 4)
            e.setBuffer(self.counters, offset: 0, index: 5)
            e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 6)
        }
        encodeScan(enc, input: degrees, output: adjStart, count: numBodies)
        dispatch1D(enc, "adj_copy_cursor", numBodies) { e in
            e.setBuffer(self.adjStart, offset: 0, index: 0)
            e.setBuffer(self.adjCursor, offset: 0, index: 1)
            e.setBytes(&nb32, length: 4, index: 2)
        }
        dispatchIndirect(enc, "adj_scatter", argsOffset: 3) { e in
            e.setBuffer(self.posLin, offset: 0, index: 0)
            e.setBuffer(self.joints, offset: 0, index: 1)
            e.setBuffer(self.springs, offset: 0, index: 2)
            e.setBuffer(self.manifolds, offset: 0, index: 3)
            e.setBuffer(self.adjCursor, offset: 0, index: 4)
            e.setBuffer(self.adjList, offset: 0, index: 5)
            e.setBuffer(self.counters, offset: 0, index: 6)
            e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 7)
        }

        // Coloring (8 Jacobi rounds, ping-pong)
        var src = colorsA, dst = colorsB
        for _ in 0..<8 {
            dispatch1D(enc, "color_iterate", numBodies) { e in
                e.setBuffer(self.posLin, offset: 0, index: 0)
                e.setBuffer(self.joints, offset: 0, index: 1)
                e.setBuffer(self.springs, offset: 0, index: 2)
                e.setBuffer(self.manifolds, offset: 0, index: 3)
                e.setBuffer(self.adjStart, offset: 0, index: 4)
                e.setBuffer(self.degrees, offset: 0, index: 5)
                e.setBuffer(self.adjList, offset: 0, index: 6)
                e.setBuffer(src, offset: 0, index: 7)
                e.setBuffer(dst, offset: 0, index: 8)
                e.setBuffer(self.changedFlag, offset: 0, index: 9)
                e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 10)
            }
            swap(&src, &dst)
        }
        let finalColors = src

        dispatch1D(enc, "color_count", numBodies) { e in
            e.setBuffer(self.posLin, offset: 0, index: 0)
            e.setBuffer(finalColors, offset: 0, index: 1)
            e.setBuffer(self.counters, offset: 0, index: 2)
            e.setBuffer(self.bodySlot, offset: 0, index: 3)
            e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 4)
        }
        dispatch1D(enc, "color_scan", AVBD_MAX_COLORS) { e in
            e.setBuffer(self.counters, offset: 0, index: 0)
            e.setBuffer(self.colorStart, offset: 0, index: 1)
            e.setBuffer(self.colorArgs, offset: 0, index: 2)
        }
        dispatch1D(enc, "color_scatter", numBodies) { e in
            e.setBuffer(self.posLin, offset: 0, index: 0)
            e.setBuffer(finalColors, offset: 0, index: 1)
            e.setBuffer(self.bodySlot, offset: 0, index: 2)
            e.setBuffer(self.colorStart, offset: 0, index: 3)
            e.setBuffer(self.colorList, offset: 0, index: 4)
            e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 5)
        }
        // ---- Solver iterations: per-color primal (indirect; empty colors
        // dispatch zero threadgroups) + dual updates. Same command buffer —
        // no CPU sync mid-step. The color loop bound comes from the previous
        // frame (colors change slowly; stale bound is safe since dispatch
        // size always comes from the GPU-side colorArgs).
        let colorBound = min(AVBD_MAX_COLORS, max(lastMaxColorUsed + 2, 12))
        for _ in 0..<settings.iterations {
            for c in 0..<colorBound {
                var cIdx = UInt32(c)
                let p = ps("primal_solve")
                enc.setComputePipelineState(p)
                enc.setBuffer(posLin, offset: 0, index: 0)
                enc.setBuffer(posAng, offset: 0, index: 1)
                enc.setBuffer(initLin, offset: 0, index: 2)
                enc.setBuffer(initAng, offset: 0, index: 3)
                enc.setBuffer(inertLin, offset: 0, index: 4)
                enc.setBuffer(inertAng, offset: 0, index: 5)
                enc.setBuffer(props, offset: 0, index: 6)
                enc.setBuffer(joints, offset: 0, index: 7)
                enc.setBuffer(springs, offset: 0, index: 8)
                enc.setBuffer(manifolds, offset: 0, index: 9)
                enc.setBuffer(adjStart, offset: 0, index: 10)
                enc.setBuffer(degrees, offset: 0, index: 11)
                enc.setBuffer(adjList, offset: 0, index: 12)
                enc.setBuffer(colorList, offset: 0, index: 13)
                enc.setBuffer(colorStart, offset: 0, index: 14)
                enc.setBytes(&cIdx, length: 4, index: 15)
                enc.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 16)
                enc.dispatchThreadgroups(indirectBuffer: colorArgs,
                                         indirectBufferOffset: c * 12,
                                         threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            }

            dispatch1D(enc, "dual_joints", numJoints) { e in
                e.setBuffer(self.posLin, offset: 0, index: 0)
                e.setBuffer(self.posAng, offset: 0, index: 1)
                e.setBuffer(self.joints, offset: 0, index: 2)
                e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 3)
            }
            dispatchIndirect(enc, "dual_manifolds", argsOffset: 0) { e in
                e.setBuffer(self.posLin, offset: 0, index: 0)
                e.setBuffer(self.posAng, offset: 0, index: 1)
                e.setBuffer(self.initLin, offset: 0, index: 2)
                e.setBuffer(self.initAng, offset: 0, index: 3)
                e.setBuffer(self.manifolds, offset: 0, index: 4)
                e.setBuffer(self.counters, offset: 0, index: 5)
                e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 6)
            }
        }

        dispatch1D(enc, "finalize_velocities", numBodies) { e in
            e.setBuffer(self.posLin, offset: 0, index: 0)
            e.setBuffer(self.posAng, offset: 0, index: 1)
            e.setBuffer(self.initLin, offset: 0, index: 2)
            e.setBuffer(self.initAng, offset: 0, index: 3)
            e.setBuffer(self.velLin, offset: 0, index: 4)
            e.setBuffer(self.velAng, offset: 0, index: 5)
            e.setBuffer(self.prevVelLin, offset: 0, index: 6)
            e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 7)
        }

        enc.endEncoding()
        cmd1.commit()
        cmd1.waitUntilCompleted()

        // Keep canonical colors in colorsA for next frame
        if finalColors !== colorsA {
            memcpy(colorsA.contents(), finalColors.contents(), numBodies * 4)
        }

        // Read back stats + next frame's color loop bound (shared memory)
        let ctr = counters.contents().bindMemory(to: UInt32.self, capacity: GPUCounters.total)
        lastNumPairs = Int(ctr[GPUCounters.pairs])
        var colorCounts: [Int] = []
        var maxColorUsed = -1
        for c in 0..<AVBD_MAX_COLORS {
            let n = Int(ctr[GPUCounters.colorBase + c])
            colorCounts.append(n)
            if n > 0 { maxColorUsed = c }
        }
        lastColorCounts = colorCounts
        lastMaxColorUsed = maxColorUsed

        swap(&manifolds, &prevManifolds)
    }

    // MARK: - State access (shared memory)

    public func bodyPosition(_ i: Int) -> F3 {
        let p = posLin.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        return F3(p[i].x, p[i].y, p[i].z)
    }

    public func bodyRotation(_ i: Int) -> Quat {
        let p = posAng.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        return Quat(real: p[i].w, imag: F3(p[i].x, p[i].y, p[i].z))
    }

    public func bodyMass(_ i: Int) -> Float {
        let p = posLin.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        return p[i].w
    }

    public func bodyVelocity(_ i: Int) -> F3 {
        let p = velLin.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        return F3(p[i].x, p[i].y, p[i].z)
    }

    public func bodyAngularVelocity(_ i: Int) -> F3 {
        let p = velAng.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        return F3(p[i].x, p[i].y, p[i].z)
    }

    // MARK: - Interaction & rendering support

    /// Activate / update a drag joint. Scenes add an inert slot via
    /// PhysicsScene.addDragSlot() (stiffness 0 keeps it disabled until used).
    public func setDrag(jointIndex: Int, body: Int?, worldTarget: F3, localAnchor: F3,
                        stiffness: Float = 5000) {
        guard jointIndex < numJoints else { return }
        let jp = joints.contents().bindMemory(to: JointGPU.self, capacity: numJoints)
        var j = jp[jointIndex]
        if let body {
            j.header = SIMD4(0xFFFFFFFF, UInt32(body), 0, 0)   // active, soft
            j.rA = SIMD4(worldTarget, stiffness)
            j.rB = SIMD4(localAnchor, 0)
            j.C0Ang = SIMD4(0, 0, 0, Float.greatestFiniteMagnitude)
            // soft (finite) constraint: no flags, penalty ramps to stiffness
            j.penaltyLin = SIMD4(repeating: 0)
            j.penaltyLin = SIMD4(1, 1, 1, 0)
            j.lambdaLin = .zero
        } else {
            j.header.z = 1   // broken = disabled
            j.penaltyLin = .zero
            j.lambdaLin = .zero
        }
        jp[jointIndex] = j
    }

    /// Ray-cast against body OBBs (CPU, shared buffers). Returns (body, local hit).
    public func pick(origin: F3, dir: F3) -> (body: Int, local: F3)? {
        let pl = posLin.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        let pa = posAng.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        let sh = shape.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        var bestT = Float.infinity
        var best: (Int, F3)? = nil
        for i in 0..<numBodies where pl[i].w > 0 {
            let q = Quat(real: pa[i].w, imag: F3(pa[i].x, pa[i].y, pa[i].z))
            let inv = q.conjugate
            let o = inv.act(origin - F3(pl[i].x, pl[i].y, pl[i].z))
            let d = inv.act(dir)
            let half = F3(sh[i].x, sh[i].y, sh[i].z) * 0.5
            var tEnter: Float = 0
            var tExit = Float.infinity
            var hit = true
            for k in 0..<3 {
                if abs(d[k]) < 1e-6 {
                    if o[k] < -half[k] || o[k] > half[k] { hit = false; break }
                    continue
                }
                var t0 = (-half[k] - o[k]) / d[k]
                var t1 = (half[k] - o[k]) / d[k]
                if t0 > t1 { swap(&t0, &t1) }
                tEnter = max(tEnter, t0)
                tExit = min(tExit, t1)
                if tEnter > tExit { hit = false; break }
            }
            if !hit { continue }
            let t = tEnter >= 0 ? tEnter : tExit
            if t >= 0 && t < bestT {
                bestT = t
                best = (i, o + d * t)
            }
        }
        return best
    }

    /// Encode instance-transform building into a render command buffer.
    public func encodeBuildInstances(_ cmd: MTLCommandBuffer, instances: MTLBuffer,
                                     colorMode: UInt32 = 0) {
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        var nb = UInt32(numBodies)
        var cm = colorMode
        let p = ps("build_instances")
        enc.setComputePipelineState(p)
        enc.setBuffer(posLin, offset: 0, index: 0)
        enc.setBuffer(posAng, offset: 0, index: 1)
        enc.setBuffer(shape, offset: 0, index: 2)
        enc.setBuffer(instances, offset: 0, index: 3)
        enc.setBytes(&nb, length: 4, index: 4)
        enc.setBytes(&cm, length: 4, index: 5)
        enc.setBuffer(colorsA, offset: 0, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: (numBodies + 255) / 256, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    public var bodyCount: Int { numBodies }

    /// Max constraint error: hard-joint violation + contact penetration depth.
    public func maxConstraintError() -> Float {
        syncParams()
        guard let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else { return 0 }
        var P = params
        dispatch1D(enc, "diag_clear", 1) { e in
            e.setBuffer(self.diag, offset: 0, index: 0)
        }
        // manifolds was swapped to prevManifolds after step; use prev
        dispatch1D(enc, "diag_error", numJoints + lastNumPairs) { e in
            e.setBuffer(self.posLin, offset: 0, index: 0)
            e.setBuffer(self.posAng, offset: 0, index: 1)
            e.setBuffer(self.joints, offset: 0, index: 2)
            e.setBuffer(self.prevManifolds, offset: 0, index: 3)
            e.setBuffer(self.counters, offset: 0, index: 4)
            e.setBuffer(self.diag, offset: 0, index: 5)
            e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 6)
        }
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        let bits = diag.contents().load(as: UInt32.self)
        return Float(bitPattern: bits)
    }
}
