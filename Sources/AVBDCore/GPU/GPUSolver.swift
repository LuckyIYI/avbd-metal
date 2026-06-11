import Foundation
import Metal
import QuartzCore
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
    let numTets: Int
    let maxPairs: Int
    let mapCapacity: Int
    let gridHashSize: Int
    // Cloth elements
    let numTris: Int
    var numEdges: Int = 0
    var numParticles: Int = 0
    let maxSoft: Int
    let softMapCapacity: Int
    let elemHashSize: Int

    var params = SimParamsGPU()

    // Body SoA buffers
    var posLin, posAng, initLin, initAng, inertLin, inertAng: MTLBuffer
    var velLin, velAng, prevVelLin: MTLBuffer
    var props, shape: MTLBuffer
    var shapeType: MTLBuffer       // 0 box, 1 sphere, 2 torus
    var spinVel: MTLBuffer         // angular velocity of kinematic spinners

    // Constraints
    var joints: MTLBuffer
    var springs: MTLBuffer
    var manifolds: MTLBuffer       // current frame
    var prevManifolds: MTLBuffer   // previous frame (swapped)

    // Broadphase
    var tets: MTLBuffer
    var hashedIdx, globalIdx: MTLBuffer
    var cellCount, cellStart, cellBodies: MTLBuffer
    var bodyCellSlot: MTLBuffer
    var pairs: MTLBuffer
    var exclusions: MTLBuffer
    var numExclusions: UInt32 = 0
    var spinners: [SceneSpinner] = []

    // Persistence map
    var mapKeyA, mapKeyB, mapVal: MTLBuffer

    // Cloth elements: triangles/edges, topology CSR, element grid, soft
    // contacts (double-buffered) + their persistence map
    var trisBuf, edgesBuf, particleIdxBuf: MTLBuffer
    var nbrStart, nbrCount, nbrList: MTLBuffer
    var elemCellCount, elemCellStart, elemCells, elemSlot: MTLBuffer
    var softContacts, prevSoftContacts: MTLBuffer
    var softMapKeyA, softMapKeyB, softMapVal: MTLBuffer
    var membranes, bends: MTLBuffer

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
    public private(set) var lastNumSoft: Int = 0
    // Start pessimistic so the first frame covers every possible color.
    public internal(set) var lastMaxColorUsed: Int = AVBD_MAX_COLORS - 1
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
        self.numTets = scene.tets.count
        self.maxPairs = max(64, scene.bodies.count * maxPairsPerBody)
        self.mapCapacity = Self.nextPow2(2 * maxPairs)
        self.gridHashSize = Self.nextPow2(max(64, 2 * numBodies))
        self.numTris = scene.tris.count
        // capacity bound: V-T (4/vertex) + rigid-T (4/tri) + E-E (2/edge,
        // edges <= 3 per tri)
        let particleEstimate = scene.bodies.lazy.filter { $0.isParticle }.count
        self.maxSoft = numTris == 0 ? 1
            : 4 * particleEstimate + 4 * numTris + 6 * numTris + 256
        self.softMapCapacity = Self.nextPow2(max(64, 2 * maxSoft))
        self.elemHashSize = Self.nextPow2(max(64, 2 * 4 * numTris))

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
        shapeType = try makeBuf(nb * 4, "shapeType")
        spinVel = try makeBuf(nb * 16, "spinVel")

        joints = try makeBuf(max(1, numJoints) * MemoryLayout<JointGPU>.stride, "joints")
        springs = try makeBuf(max(1, numSprings) * MemoryLayout<SpringGPU>.stride, "springs")
        tets = try makeBuf(max(1, numTets) * MemoryLayout<TetGPU>.stride, "tets")
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

        // Cloth element buffers (edges/neighbors sized after derivation below;
        // worst case edges = 3 per triangle)
        let maxEdges = max(1, 3 * numTris)
        trisBuf = try makeBuf(max(1, numTris) * 16, "tris")
        edgesBuf = try makeBuf(maxEdges * 8, "edges")
        particleIdxBuf = try makeBuf(nb * 4, "particleIdx")
        nbrStart = try makeBuf(nb * 4, "nbrStart")
        nbrCount = try makeBuf(nb * 4, "nbrCount")
        nbrList = try makeBuf(max(1, numTris * 6) * 4, "nbrList")
        elemCellCount = try makeBuf(elemHashSize * 4, "elemCellCount")
        elemCellStart = try makeBuf(elemHashSize * 4, "elemCellStart")
        elemCells = try makeBuf((numTris + maxEdges + 1) * 4, "elemCells")
        elemSlot = try makeBuf((numTris + maxEdges + 1) * 8, "elemSlot")
        softContacts = try makeBuf(maxSoft * MemoryLayout<SoftContactGPU>.stride, "softContacts")
        prevSoftContacts = try makeBuf(maxSoft * MemoryLayout<SoftContactGPU>.stride, "prevSoftContacts")
        softMapKeyA = try makeBuf(softMapCapacity * 4, "softMapKeyA")
        softMapKeyB = try makeBuf(softMapCapacity * 4, "softMapKeyB")
        softMapVal = try makeBuf(softMapCapacity * 4, "softMapVal")
        membranes = try makeBuf(max(1, numTris) * MemoryLayout<MembraneGPU>.stride, "membranes")
        bends = try makeBuf(max(1, maxEdges) * MemoryLayout<BendGPU>.stride, "bends")

        degrees = try makeBuf(nb * 4, "degrees")
        adjStart = try makeBuf(nb * 4, "adjStart")
        adjCursor = try makeBuf(nb * 4, "adjCursor")
        adjList = try makeBuf((2 * (numJoints + numSprings + maxPairs) + 4 * numTets
                               + 4 * maxSoft) * 4, "adjList")
        colorsA = try makeBuf(nb * 4, "colorsA")
        colorsB = try makeBuf(nb * 4, "colorsB")
        bodySlot = try makeBuf(nb * 4, "bodySlot")
        colorStart = try makeBuf((AVBD_MAX_COLORS + 1) * 4, "colorStart")
        colorList = try makeBuf(nb * 4, "colorList")
        changedFlag = try makeBuf(4, "changedFlag")

        counters = try makeBuf(GPUCounters.total * 4, "counters")
        dispatchArgs = try makeBuf(9 * 4, "dispatchArgs")
        colorArgs = try makeBuf(AVBD_MAX_COLORS * 3 * 4, "colorArgs")
        let maxScanCount = max(max(gridHashSize, elemHashSize), nb)
        scanBlockSums = try makeBuf(((maxScanCount + 1023) / 1024 + 1) * 4, "scanBlockSums")
        scanTotal = try makeBuf(4, "scanTotal")
        diag = try makeBuf(4, "diag")

        try buildPipelines()
        try upload(scene: scene)
        self.spinners = scene.spinners
        // expose spinner angular velocity to the contact solver so friction
        // sees the kinematic surface motion
        memset(spinVel.contents(), 0, spinVel.length)
        let sv = spinVel.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        for sp in scene.spinners {
            sv[sp.body] = SIMD4(sp.axis * sp.omega, 1)
        }
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
        var urls = (Bundle.module.urls(forResourcesWithExtension: "metal", subdirectory: nil) ?? [])
        if urls.isEmpty {   // .copy resource rule keeps the Shaders/ subdir
            urls = Bundle.module.urls(forResourcesWithExtension: "metal",
                                      subdirectory: "Shaders") ?? []
        }
        urls.sort { $0.lastPathComponent < $1.lastPathComponent }
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
        let fastMath = ProcessInfo.processInfo.environment["AVBD_SAFE_MATH"] == nil
        if #available(macOS 15.0, *) {
            options.mathMode = fastMath ? .fast : .safe
        } else {
            options.fastMathEnabled = fastMath
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
        let st = shapeType.contents().bindMemory(to: UInt32.self, capacity: numBodies)
        let colA = colorsA.contents().bindMemory(to: UInt32.self, capacity: numBodies)
        let colB = colorsB.contents().bindMemory(to: UInt32.self, capacity: numBodies)

        var radii: [Float] = []
        for (i, b) in scene.bodies.enumerated() {
            let mass: Float
            let moment: F3
            let radius: Float
            switch b.shape {
            case .box:
                mass = b.density > 0 ? b.size.x * b.size.y * b.size.z * b.density : 0
                moment = F3(
                    (b.size.y * b.size.y + b.size.z * b.size.z) / 12 * mass,
                    (b.size.x * b.size.x + b.size.z * b.size.z) / 12 * mass,
                    (b.size.x * b.size.x + b.size.y * b.size.y) / 12 * mass
                )
                radius = length(b.size * 0.5)
            case .sphere:
                let r = b.size.x / 2
                mass = b.density > 0 ? 4.0 / 3.0 * Float.pi * r * r * r * b.density : 0
                moment = F3(repeating: 0.4 * mass * r * r)
                radius = r
            case .torus:
                let R = b.size.x, r = b.size.y
                mass = b.density > 0 ? 2 * Float.pi * Float.pi * R * r * r * b.density : 0
                let iDia = mass * (R * R / 2 + 5 * r * r / 8)
                let iAxis = mass * (R * R + 3 * r * r / 4)
                moment = F3(iDia, iDia, iAxis)
                radius = R + r
            case .capsule:
                let L = b.size.x, r = b.size.y
                mass = b.density > 0 ? Float.pi * r * r * (L + 4 * r / 3) * b.density : 0
                let iAxis = 0.5 * mass * r * r
                let iPerp = mass * (L * L / 12 + r * r / 4)
                moment = F3(iPerp, iPerp, iAxis)
                radius = L / 2 + r
            }
            pl[i] = SIMD4(b.position, mass)
            pa[i] = SIMD4(b.rotation.imag, b.rotation.real)
            vl[i] = SIMD4(b.velocity, 0)
            va[i] = .zero
            pv[i] = SIMD4(b.velocity, 0)
            pr[i] = SIMD4(moment, b.friction)
            sh[i] = SIMD4(b.size, b.isParticle ? -radius : radius)
            switch b.shape {
            case .box: st[i] = 0
            case .sphere: st[i] = 1
            case .torus: st[i] = 2
            case .capsule: st[i] = 3
            }
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
        // Only oversized bodies go to the brute-forced global list; normal
        // sized statics live in the spatial hash like everything else.
        for (i, b) in scene.bodies.enumerated() {
            let radius: Float
            switch b.shape {
            case .sphere: radius = b.size.x / 2
            case .torus: radius = b.size.x + b.size.y
            case .capsule: radius = b.size.x / 2 + b.size.y
            case .box: radius = length(b.size * 0.5)
            }
            if radius > threshold {
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
            if j.fractureLinear { flags |= 8 }
            g.header = SIMD4(aIdx, UInt32(j.bodyB), 0, flags)
            let bigK: Float = 3.0e10
            g.rA = SIMD4(j.rA, min(j.stiffnessLin, bigK))
            g.rB = SIMD4(j.rB, min(j.stiffnessAng, bigK))
            let sizeA = j.bodyA >= 0 ? scene.bodies[j.bodyA].size : .zero
            let torqueArm = length_squared(sizeA + scene.bodies[j.bodyB].size)
            g.C0Lin = SIMD4(0, 0, 0, torqueArm)
            g.C0Ang = SIMD4(0, 0, 0, min(j.fracture, 3.0e18))
            // rest relative rotation: angular welds preserve spawn alignment
            let qA0 = j.bodyA >= 0 ? scene.bodies[j.bodyA].rotation : Quat(real: 1, imag: .zero)
            let rel = (qA0.inverse * scene.bodies[j.bodyB].rotation).normalized
            g.restRel = SIMD4(rel.imag, rel.real)
            if let axis = j.hingeAxis {
                g.hingeAxis = SIMD4(axis, 1)
                if j.motorTorque > 0 {
                    g.motor = SIMD4(j.motorTarget, j.motorTorque, 0, 400)
                }
                if j.limitLo < j.limitHi {
                    g.limits = SIMD4(j.limitLo, j.limitHi, 0, 0)
                }
            }
            // soft (finite) joints ARE their stiffness from frame one; only
            // hard (AL) constraints ramp from PENALTY_MIN per the paper
            if j.stiffnessLin > 0 && j.stiffnessLin.isFinite {
                g.penaltyLin = SIMD4(repeating: min(j.stiffnessLin, 1e9))
            }
            if j.stiffnessAng > 0 && j.stiffnessAng.isFinite {
                g.penaltyAng = SIMD4(repeating: min(j.stiffnessAng, 1e9))
            }
            jp[i] = g
        }

        // Springs (rest length resolved here if negative)
        let sp = springs.contents().bindMemory(to: SpringGPU.self, capacity: max(1, numSprings))
        for (i, s) in scene.springs.enumerated() {
            var g = SpringGPU()
            g.header = SIMD4(UInt32(s.bodyA), UInt32(s.bodyB), s.hard ? 1 : 0, 0)
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

        let tp = tets.contents().bindMemory(to: TetGPU.self, capacity: max(1, numTets))
        for (i, t) in scene.tets.enumerated() {
            var g = TetGPU()
            g.ids = SIMD4(UInt32(t.ids.0), UInt32(t.ids.1), UInt32(t.ids.2), UInt32(t.ids.3))
            // rest matrix Dm from spawn positions; DmInv rows + material
            let x0 = scene.bodies[t.ids.0].position
            let d0 = scene.bodies[t.ids.1].position - x0
            let d1 = scene.bodies[t.ids.2].position - x0
            let d2 = scene.bodies[t.ids.3].position - x0
            let vol = abs(dot(d0, cross(d1, d2))) / 6
            let Dm = simd_float3x3(columns: (d0, d1, d2))
            let DmInv = Dm.inverse
            // rows of DmInv
            let r0 = F3(DmInv.columns.0.x, DmInv.columns.1.x, DmInv.columns.2.x)
            let r1 = F3(DmInv.columns.0.y, DmInv.columns.1.y, DmInv.columns.2.y)
            let r2 = F3(DmInv.columns.0.z, DmInv.columns.1.z, DmInv.columns.2.z)
            g.r0 = SIMD4(r0, vol)
            g.r1 = SIMD4(r1, t.mu * vol)
            g.r2 = SIMD4(r2, t.lambda * vol)
            tp[i] = g
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

        // ---- Cloth element topology ----
        // Triangles, unique edges, per-vertex topological neighborhoods (the
        // V-T/E-E exclusion sets: vertices sharing a triangle), particle list.
        let tp4 = trisBuf.contents().bindMemory(to: SIMD4<UInt32>.self,
                                                capacity: max(1, numTris))
        var edgeSet: Set<UInt64> = []
        var nbrSets = [Set<Int>](repeating: [], count: numBodies)
        var maxElemR: Float = 0
        for (i, t) in scene.tris.enumerated() {
            let (a, b, c) = t.ids
            tp4[i] = SIMD4(UInt32(a), UInt32(b), UInt32(c), 0)
            for (u, v) in [(a, b), (b, c), (a, c)] {
                edgeSet.insert(UInt64(min(u, v)) << 32 | UInt64(max(u, v)))
            }
            nbrSets[a].formUnion([b, c]); nbrSets[b].formUnion([a, c])
            nbrSets[c].formUnion([a, b])
            let pa = scene.bodies[a].position, pb = scene.bodies[b].position
            let pc = scene.bodies[c].position
            let m = (pa + pb + pc) / 3
            let thick = max(scene.bodies[a].size.x, max(scene.bodies[b].size.x,
                                                        scene.bodies[c].size.x)) / 2
            maxElemR = max(maxElemR, max(distance(m, pa), max(distance(m, pb),
                                                              distance(m, pc))) + thick)
        }
        let sortedEdges = edgeSet.sorted()
        numEdges = sortedEdges.count
        let ep2 = edgesBuf.contents().bindMemory(to: SIMD2<UInt32>.self,
                                                 capacity: max(1, numEdges))
        for (i, key) in sortedEdges.enumerated() {
            ep2[i] = SIMD2(UInt32(key >> 32), UInt32(key & 0xFFFFFFFF))
            let a = scene.bodies[Int(key >> 32)], b = scene.bodies[Int(key & 0xFFFFFFFF)]
            maxElemR = max(maxElemR, distance(a.position, b.position) / 2
                           + max(a.size.x, b.size.x) / 2)
        }
        // CSR neighborhoods (sorted per vertex for binary search)
        let nsP = nbrStart.contents().bindMemory(to: UInt32.self, capacity: numBodies)
        let ncP = nbrCount.contents().bindMemory(to: UInt32.self, capacity: numBodies)
        var flatNbrs: [UInt32] = []
        for i in 0..<numBodies {
            nsP[i] = UInt32(flatNbrs.count)
            let sorted = nbrSets[i].sorted()
            ncP[i] = UInt32(sorted.count)
            flatNbrs.append(contentsOf: sorted.map(UInt32.init))
        }
        precondition(flatNbrs.count <= nbrList.length / 4,
                     "neighbor CSR exceeded 6-per-triangle bound")
        if !flatNbrs.isEmpty {
            nbrList.contents().bindMemory(to: UInt32.self, capacity: flatNbrs.count)
                .update(from: flatNbrs, count: flatNbrs.count)
        }
        // particle list: V-T query points (any 3-DOF particle, cloth or tet)
        var particles: [UInt32] = []
        for (i, b) in scene.bodies.enumerated() where b.isParticle {
            particles.append(UInt32(i))
        }
        numParticles = numTris == 0 ? 0 : particles.count
        if !particles.isEmpty {
            particleIdxBuf.contents().bindMemory(to: UInt32.self, capacity: particles.count)
                .update(from: particles, count: particles.count)
        }
        // element grid cell size: 2x max element radius with stretch headroom
        params.elemCellSize = max(0.2, 2 * maxElemR * 1.5)
        params.elemHashSize = UInt32(elemHashSize)
        params.numTris = UInt32(numTris)
        params.numEdges = UInt32(numEdges)
        params.numParticles = UInt32(numParticles)
        params.maxSoft = UInt32(maxSoft)
        params.softMapCapacity = UInt32(softMapCapacity)

        // Clear manifolds + map
        memset(manifolds.contents(), 0, manifolds.length)
        memset(prevManifolds.contents(), 0, prevManifolds.length)
        memset(mapKeyA.contents(), 0, mapKeyA.length)
        memset(counters.contents(), 0, counters.length)
        memset(softContacts.contents(), 0, softContacts.length)
        memset(prevSoftContacts.contents(), 0, prevSoftContacts.length)
        memset(softMapKeyA.contents(), 0, softMapKeyA.length)

        // Params
        params.numBodies = UInt32(numBodies)
        params.numJoints = UInt32(numJoints)
        params.numSprings = UInt32(numSprings)
        params.numTets = UInt32(numTets)
        params.mapCapacity = UInt32(mapCapacity)
        params.maxManifolds = UInt32(maxPairs)
        params.maxPairs = UInt32(maxPairs)
        params.cellSize = maxHashedRadius * 2
        params.gridHashSize = UInt32(gridHashSize)
        params.numHashed = UInt32(hashed.count)
        params.numGlobals = UInt32(globals.count)
        // Anti-tunneling: cap speed so nothing crosses the thinnest static
        // geometry in one frame. Heuristic: a couple of cells per step.
        params.maxSpeed = max(30, 1.5 * params.cellSize / settings.dt * 0.5)
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
        params.lambdaMax = settings.lambdaMax
        params.iterations = UInt32(settings.iterations)
        params.rodDecayPow = settings.rodDecayPow
        if let env = ProcessInfo.processInfo.environment["AVBD_ROD_DECAY"],
           let v = Float(env) {
            params.rodDecayPow = v
        }
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

    // ---- GPU profiling: per-stage timestamps (encoder-boundary sampling,
    // the only granularity Apple GPUs support) ----
    public var profiling = false
    public private(set) var profileNS: [String: Double] = [:]
    public private(set) var profileFrames = 0
    private var counterBuf: MTLCounterSampleBuffer?
    private var stageNames: [String] = []

    // ---- async pipelining: step() never blocks; the queue serializes GPU
    // work, and CPU access to shared buffers syncs lazily ----
    private var inflight: [MTLCommandBuffer] = []
    private let statsLock = NSLock()

    /// Robotics: render parallel Push-T pixel observations via the
    /// analytic top-down compute kernel. envTable: PushTEnvGPU records.
    public func renderPushTObs(envTable: MTLBuffer, numEnvs: Int,
                               out: MTLBuffer, res: Int) {
        guard let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else { return }
        let p = ps("pusht_obs")
        enc.setComputePipelineState(p)
        enc.setBuffer(posLin, offset: 0, index: 0)
        enc.setBuffer(posAng, offset: 0, index: 1)
        enc.setBuffer(envTable, offset: 0, index: 2)
        enc.setBuffer(out, offset: 0, index: 3)
        var r32 = UInt32(res)
        enc.setBytes(&r32, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: (res + 7) / 8, height: (res + 7) / 8,
                                         depth: numEnvs),
                                 threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    public var metalDevice: MTLDevice { device }

    /// Robotics: teleport a body (resets its velocity).
    public func setBodyPose(_ i: Int, position: F3, rotation: Quat) {
        sync()
        let pl = posLin.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        let pa = posAng.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        let vl = velLin.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        let va = velAng.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        pl[i] = SIMD4(position, pl[i].w)
        pa[i] = SIMD4(rotation.imag, rotation.real)
        vl[i] = .zero
        va[i] = .zero
    }

    /// Robotics: move a world-anchored joint's target point (Cartesian
    /// position actuator — the joint's bounded force does the rest).
    public func setJointWorldAnchor(_ jointIndex: Int, point: F3) {
        sync()
        let jp = joints.contents().bindMemory(to: JointGPU.self, capacity: max(1, numJoints))
        jp[jointIndex].rA = SIMD4(point, jp[jointIndex].rA.w)
    }

    /// Robotics: set a motor joint's target angle at runtime.
    public func setMotorTarget(_ jointIndex: Int, angle: Float) {
        sync()
        let jp = joints.contents().bindMemory(to: JointGPU.self, capacity: max(1, numJoints))
        jp[jointIndex].motor.x = angle
    }

    /// Robotics: set a motor joint's torque limit at runtime.
    public func setMotorTorque(_ jointIndex: Int, torque: Float) {
        sync()
        let jp = joints.contents().bindMemory(to: JointGPU.self, capacity: max(1, numJoints))
        jp[jointIndex].motor.y = torque
    }

    /// Robotics: read a motor joint's current twist angle.
    public func motorAngle(_ jointIndex: Int) -> Float {
        sync()
        let jp = joints.contents().bindMemory(to: JointGPU.self, capacity: max(1, numJoints))
        let j = jp[jointIndex]
        let a = j.header.x
        let pa = posAng.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        let qB = Quat(real: pa[Int(j.header.y)].w,
                      imag: F3(pa[Int(j.header.y)].x, pa[Int(j.header.y)].y, pa[Int(j.header.y)].z))
        let qA: Quat = a == 0xFFFFFFFF
            ? Quat(real: 1, imag: .zero)
            : Quat(real: pa[Int(a)].w, imag: F3(pa[Int(a)].x, pa[Int(a)].y, pa[Int(a)].z))
        let rest = Quat(real: j.restRel.w, imag: F3(j.restRel.x, j.restRel.y, j.restRel.z))
        var r = (qA * rest).inverse * qB
        if r.real < 0 { r = Quat(real: -r.real, imag: -r.imag) }
        let axis = F3(j.hingeAxis.x, j.hingeAxis.y, j.hingeAxis.z)
        return 2 * atan2(dot(r.imag, axis), r.real)
    }

    /// Debug: worst joints by linear lambda, with endpoints.
    public func debugWorstJoints(_ n: Int = 5) -> [(Int, Int, Float)] {
        sync()
        let jp = joints.contents().bindMemory(to: JointGPU.self, capacity: max(1, numJoints))
        var all: [(Int, Int, Float)] = []
        for i in 0..<numJoints where jp[i].header.z == 0 {
            let l = length(F3(jp[i].lambdaLin.x, jp[i].lambdaLin.y, jp[i].lambdaLin.z))
            all.append((Int(jp[i].header.x), Int(jp[i].header.y), l))
        }
        return Array(all.sorted { $0.2 > $1.2 }.prefix(n))
    }

    /// Debug: count of broken (fractured) joints.
    public func debugBrokenJoints() -> Int {
        sync()
        let jp = joints.contents().bindMemory(to: JointGPU.self, capacity: max(1, numJoints))
        var n = 0
        for i in 0..<numJoints where jp[i].header.z != 0 { n += 1 }
        return n
    }

    /// Debug: max joint lambda magnitudes (lin, ang) across live joints.
    public func debugMaxLambda() -> (Float, Float) {
        sync()
        let jp = joints.contents().bindMemory(to: JointGPU.self, capacity: max(1, numJoints))
        var ml: Float = 0, ma: Float = 0
        for i in 0..<numJoints where jp[i].header.z == 0 {
            ml = max(ml, length(F3(jp[i].lambdaLin.x, jp[i].lambdaLin.y, jp[i].lambdaLin.z)))
            ma = max(ma, length(F3(jp[i].lambdaAng.x, jp[i].lambdaAng.y, jp[i].lambdaAng.z)))
        }
        return (ml, ma)
    }

    /// Debug: current body colors (post-step, canonical buffer).
    public func debugColors() -> [Int] {
        sync()
        let c = colorsA.contents().bindMemory(to: UInt32.self, capacity: numBodies)
        return (0..<numBodies).map { Int(c[$0]) }
    }

    /// Cloth diagnostics (CPU, brute force over the shared buffers):
    /// - minGap: worst vertex-to-triangle-surface clearance among
    ///   non-topologically-adjacent pairs. 0 = surfaces touching; values
    ///   below -(rv+rt) mean a vertex CENTER crossed the midsurface.
    /// - maxStretch: worst stiff-spring strain |len/rest - 1| (stiffness >=
    ///   1000 selects structural edges; shear/bend sets are soft by design).
    public func debugClothMetrics() -> (minGap: Float, maxStretch: Float) {
        sync()
        let pl = posLin.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        let sh = shape.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        let tp = trisBuf.contents().bindMemory(to: SIMD4<UInt32>.self,
                                               capacity: max(1, numTris))
        let pidx = particleIdxBuf.contents().bindMemory(to: UInt32.self,
                                                        capacity: max(1, numParticles))
        let ns = nbrStart.contents().bindMemory(to: UInt32.self, capacity: numBodies)
        let nc = nbrCount.contents().bindMemory(to: UInt32.self, capacity: numBodies)
        let nl = nbrList.contents().bindMemory(to: UInt32.self,
                                               capacity: max(1, nbrList.length / 4))

        func closest(_ p: F3, _ a: F3, _ b: F3, _ c: F3) -> F3 {
            let ab = b - a, ac = c - a, ap = p - a
            let d1 = dot(ab, ap), d2 = dot(ac, ap)
            if d1 <= 0 && d2 <= 0 { return a }
            let bp = p - b
            let d3 = dot(ab, bp), d4 = dot(ac, bp)
            if d3 >= 0 && d4 <= d3 { return b }
            let vc = d1 * d4 - d3 * d2
            if vc <= 0 && d1 >= 0 && d3 <= 0 { return a + ab * (d1 / max(d1 - d3, 1e-12)) }
            let cp = p - c
            let d5 = dot(ab, cp), d6 = dot(ac, cp)
            if d6 >= 0 && d5 <= d6 { return c }
            let vb = d5 * d2 - d1 * d6
            if vb <= 0 && d2 >= 0 && d6 <= 0 { return a + ac * (d2 / max(d2 - d6, 1e-12)) }
            let va = d3 * d6 - d5 * d4
            if va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0 {
                let w = (d4 - d3) / max((d4 - d3) + (d5 - d6), 1e-12)
                return b + (c - b) * w
            }
            let denom = va + vb + vc
            if abs(denom) < 1e-20 { return a }
            return a + ab * (vb / denom) + ac * (vc / denom)
        }
        func isNbr(_ v: Int, _ x: UInt32) -> Bool {
            let s = Int(ns[v]), e = s + Int(nc[v])
            for k in s..<e where nl[k] == x { return true }
            return false
        }

        var minGap: Float = .greatestFiniteMagnitude
        for g in 0..<numParticles {
            let v = Int(pidx[g])
            let p = F3(pl[v].x, pl[v].y, pl[v].z)
            let rv = abs(sh[v].w)
            for t in 0..<numTris {
                let id = tp[t]
                if id.x == UInt32(v) || id.y == UInt32(v) || id.z == UInt32(v) { continue }
                if isNbr(v, id.x) || isNbr(v, id.y) || isNbr(v, id.z) { continue }
                let a = F3(pl[Int(id.x)].x, pl[Int(id.x)].y, pl[Int(id.x)].z)
                let b = F3(pl[Int(id.y)].x, pl[Int(id.y)].y, pl[Int(id.y)].z)
                let c = F3(pl[Int(id.z)].x, pl[Int(id.z)].y, pl[Int(id.z)].z)
                let q = closest(p, a, b, c)
                let rt = max(abs(sh[Int(id.x)].w), max(abs(sh[Int(id.y)].w), abs(sh[Int(id.z)].w)))
                minGap = min(minGap, distance(p, q) - (rv + rt))
            }
        }

        var maxStretch: Float = 0
        let sp = springs.contents().bindMemory(to: SpringGPU.self, capacity: max(1, numSprings))
        let pa = posAng.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        for i in 0..<numSprings {
            let s = sp[i]
            guard s.rA.w >= 1000 || s.header.z != 0 else { continue }
            let a = Int(s.header.x), b = Int(s.header.y)
            let qa = Quat(real: pa[a].w, imag: F3(pa[a].x, pa[a].y, pa[a].z))
            let qb = Quat(real: pa[b].w, imag: F3(pa[b].x, pa[b].y, pa[b].z))
            let wa = F3(pl[a].x, pl[a].y, pl[a].z) + qa.act(F3(s.rA.x, s.rA.y, s.rA.z))
            let wb = F3(pl[b].x, pl[b].y, pl[b].z) + qb.act(F3(s.rB.x, s.rB.y, s.rB.z))
            let rest = s.rB.w
            if rest > 1e-6 {
                let st = abs(distance(wa, wb) / rest - 1)
                if st > maxStretch {
                    maxStretch = st
                    lastWorstSpring = (a, b)
                }
            }
        }
        return (minGap == .greatestFiniteMagnitude ? 0 : minGap, maxStretch)
    }

    /// Endpoints of the worst stiff spring from the last debugClothMetrics call.
    public private(set) var lastWorstSpring: (Int, Int) = (-1, -1)

    /// Count live soft contacts by kind (VT, RT, EE) — CPU read.
    public func debugSoftKinds() -> (vt: Int, rt: Int, ee: Int) {
        sync()
        let ctr = counters.contents().bindMemory(to: UInt32.self, capacity: GPUCounters.total)
        // step() swapped buffers; the contacts of the LAST step live in prev
        let n = min(Int(ctr[GPUCounters.soft]), maxSoft)
        let sc = prevSoftContacts.contents().bindMemory(to: SoftContactGPU.self,
                                                        capacity: max(1, maxSoft))
        var vt = 0, rt = 0, ee = 0
        for i in 0..<n {
            let kind = (sc[i].anchorA.w.bitPattern >> 2) & 0x7
            if kind == 1 { vt += 1 } else if kind == 2 { rt += 1 }
            else if kind == 3 { ee += 1 }
        }
        return (vt, rt, ee)
    }

    /// Wait for all committed steps (no-op when already complete).
    public func sync() {
        for c in inflight { c.waitUntilCompleted() }
        inflight.removeAll()
    }

    /// Cap the pipeline depth: deeper queues make the async color-bound
    /// readback stale enough to skip colors (observed physics regressions).
    private func throttle() {
        while inflight.count >= 2 {
            inflight.removeFirst().waitUntilCompleted()
        }
    }

    public func resetProfile() {
        profileNS = [:]
        profileFrames = 0
    }

    private func makeCounterBuf() -> MTLCounterSampleBuffer? {
        if let counterBuf { return counterBuf }
        guard let set = device.counterSets?.first(where: { $0.name.lowercased().contains("timestamp") })
        else { return nil }
        let d = MTLCounterSampleBufferDescriptor()
        d.counterSet = set
        d.storageMode = .shared
        d.sampleCount = 128
        counterBuf = try? device.makeCounterSampleBuffer(descriptor: d)
        return counterBuf
    }

    public func step() {
        syncParams()
        frameIndex += 1
        throttle()
        if !spinners.isEmpty { sync() }   // spinner poses are CPU writes
        advanceSpinners()

        // ---- Command buffer 1: collision + warm start + adjacency + coloring
        guard let cmd1 = queue.makeCommandBuffer() else { return }

        if let blit = cmd1.makeBlitCommandEncoder() {
            blit.fill(buffer: counters, range: 0..<counters.length, value: 0)
            blit.fill(buffer: cellCount, range: 0..<(gridHashSize * 4), value: 0)
            if numTris > 0 {
                blit.fill(buffer: elemCellCount, range: 0..<(elemHashSize * 4), value: 0)
            }
            blit.endEncoding()
        }

        let sampleBuf = makeCounterBuf()
        stageNames = []
        func makeEncoder(_ name: String) -> MTLComputeCommandEncoder? {
            guard let sampleBuf, stageNames.count < 63 else {
                let e = cmd1.makeComputeCommandEncoder()
                e?.label = name
                return e
            }
            let pd = MTLComputePassDescriptor()
            let att = pd.sampleBufferAttachments[0]!
            att.sampleBuffer = sampleBuf
            att.startOfEncoderSampleIndex = stageNames.count * 2
            att.endOfEncoderSampleIndex = stageNames.count * 2 + 1
            stageNames.append(name)
            let e = cmd1.makeComputeCommandEncoder(descriptor: pd)
            e?.label = name
            return e
        }
        guard var enc = makeEncoder("broadphase") else { return }
        // ALWAYS split encoders at stage boundaries: measured 2.5-3x faster
        // than one mega-encoder — intra-encoder hazard barriers over long
        // dispatch chains drain the pipe far harder than encoder boundaries
        func stage(_ name: String) {
            enc.endEncoding()
            enc = makeEncoder(name) ?? enc
        }
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

        stage("narrowphase")
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
            e.setBuffer(self.shapeType, offset: 0, index: 12)
            e.setBuffer(self.spinVel, offset: 0, index: 13)
        }

        stage("persistence-map")
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

        // ---- Cloth element contacts: bin triangles into the element grid,
        // emit V-T and rigid-feature-T records (warm-started from the soft
        // persistence map), then rebuild the map for next frame. Runs at
        // start-of-step poses like the rigid narrowphase.
        if numTris > 0 {
            stage("cloth-detect")
            dispatch1D(enc, "el_count", numTris + Int(P.numEdges)) { e in
                e.setBuffer(self.posLin, offset: 0, index: 0)
                e.setBuffer(self.trisBuf, offset: 0, index: 1)
                e.setBuffer(self.edgesBuf, offset: 0, index: 2)
                e.setBuffer(self.elemCellCount, offset: 0, index: 3)
                e.setBuffer(self.elemSlot, offset: 0, index: 4)
                e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 5)
            }
            encodeScan(enc, input: elemCellCount, output: elemCellStart, count: elemHashSize)
            dispatch1D(enc, "el_scatter", numTris + Int(P.numEdges)) { e in
                e.setBuffer(self.elemSlot, offset: 0, index: 0)
                e.setBuffer(self.elemCellStart, offset: 0, index: 1)
                e.setBuffer(self.elemCells, offset: 0, index: 2)
                e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 3)
            }
            if ProcessInfo.processInfo.environment["AVBD_NO_VT"] == nil {
            dispatch1D(enc, "vt_emit", numParticles) { e in
                e.setBuffer(self.posLin, offset: 0, index: 0)
                e.setBuffer(self.shape, offset: 0, index: 1)
                e.setBuffer(self.props, offset: 0, index: 2)
                e.setBuffer(self.velLin, offset: 0, index: 3)
                e.setBuffer(self.particleIdxBuf, offset: 0, index: 4)
                e.setBuffer(self.trisBuf, offset: 0, index: 5)
                e.setBuffer(self.elemCellStart, offset: 0, index: 6)
                e.setBuffer(self.elemCellCount, offset: 0, index: 7)
                e.setBuffer(self.elemCells, offset: 0, index: 8)
                e.setBuffer(self.nbrStart, offset: 0, index: 9)
                e.setBuffer(self.nbrCount, offset: 0, index: 10)
                e.setBuffer(self.nbrList, offset: 0, index: 11)
                e.setBuffer(self.softContacts, offset: 0, index: 12)
                e.setBuffer(self.counters, offset: 0, index: 13)
                e.setBuffer(self.prevSoftContacts, offset: 0, index: 14)
                e.setBuffer(self.softMapKeyA, offset: 0, index: 15)
                e.setBuffer(self.softMapKeyB, offset: 0, index: 16)
                e.setBuffer(self.softMapVal, offset: 0, index: 17)
                e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 18)
            }
            }
            if ProcessInfo.processInfo.environment["AVBD_NO_EE"] == nil {
            dispatch1D(enc, "ee_emit", numEdges) { e in
                e.setBuffer(self.posLin, offset: 0, index: 0)
                e.setBuffer(self.shape, offset: 0, index: 1)
                e.setBuffer(self.props, offset: 0, index: 2)
                e.setBuffer(self.velLin, offset: 0, index: 3)
                e.setBuffer(self.edgesBuf, offset: 0, index: 4)
                e.setBuffer(self.elemCellStart, offset: 0, index: 5)
                e.setBuffer(self.elemCellCount, offset: 0, index: 6)
                e.setBuffer(self.elemCells, offset: 0, index: 7)
                e.setBuffer(self.nbrStart, offset: 0, index: 8)
                e.setBuffer(self.nbrCount, offset: 0, index: 9)
                e.setBuffer(self.nbrList, offset: 0, index: 10)
                e.setBuffer(self.softContacts, offset: 0, index: 11)
                e.setBuffer(self.counters, offset: 0, index: 12)
                e.setBuffer(self.prevSoftContacts, offset: 0, index: 13)
                e.setBuffer(self.softMapKeyA, offset: 0, index: 14)
                e.setBuffer(self.softMapKeyB, offset: 0, index: 15)
                e.setBuffer(self.softMapVal, offset: 0, index: 16)
                e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 17)
            }
            }
            if ProcessInfo.processInfo.environment["AVBD_NO_RT"] == nil {
            dispatch1D(enc, "rt_emit", numTris) { e in
                e.setBuffer(self.posLin, offset: 0, index: 0)
                e.setBuffer(self.posAng, offset: 0, index: 1)
                e.setBuffer(self.shape, offset: 0, index: 2)
                e.setBuffer(self.props, offset: 0, index: 3)
                e.setBuffer(self.velLin, offset: 0, index: 4)
                e.setBuffer(self.shapeType, offset: 0, index: 5)
                e.setBuffer(self.trisBuf, offset: 0, index: 6)
                e.setBuffer(self.cellStart, offset: 0, index: 7)
                e.setBuffer(self.cellCount, offset: 0, index: 8)
                e.setBuffer(self.cellBodies, offset: 0, index: 9)
                e.setBuffer(self.globalIdx, offset: 0, index: 10)
                e.setBuffer(self.exclusions, offset: 0, index: 11)
                e.setBytes(&nExcl, length: 4, index: 12)
                e.setBuffer(self.softContacts, offset: 0, index: 13)
                e.setBuffer(self.counters, offset: 0, index: 14)
                e.setBuffer(self.prevSoftContacts, offset: 0, index: 15)
                e.setBuffer(self.softMapKeyA, offset: 0, index: 16)
                e.setBuffer(self.softMapKeyB, offset: 0, index: 17)
                e.setBuffer(self.softMapVal, offset: 0, index: 18)
                e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 19)
            }
            }
            dispatch1D(enc, "soft_finalize", 1) { e in
                e.setBuffer(self.counters, offset: 0, index: 0)
                e.setBuffer(self.dispatchArgs, offset: 0, index: 1)
                e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 2)
            }
            dispatch1D(enc, "softmap_clear", softMapCapacity) { e in
                e.setBuffer(self.softMapKeyA, offset: 0, index: 0)
                e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 1)
            }
            dispatch1D(enc, "softmap_insert", maxSoft) { e in
                e.setBuffer(self.softContacts, offset: 0, index: 0)
                e.setBuffer(self.softMapKeyA, offset: 0, index: 1)
                e.setBuffer(self.softMapKeyB, offset: 0, index: 2)
                e.setBuffer(self.softMapVal, offset: 0, index: 3)
                e.setBuffer(self.counters, offset: 0, index: 4)
                e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 5)
            }
        }

        stage("warmstart")
        // Warm start joints (before body prediction; uses start-of-step poses)
        dispatch1D(enc, "warmstart_joints", numJoints + numSprings) { e in
            e.setBuffer(self.posLin, offset: 0, index: 0)
            e.setBuffer(self.posAng, offset: 0, index: 1)
            e.setBuffer(self.joints, offset: 0, index: 2)
            e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 3)
            e.setBuffer(self.springs, offset: 0, index: 4)
            e.setBuffer(self.velLin, offset: 0, index: 5)
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

        stage("adjacency")
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
            e.setBuffer(self.tets, offset: 0, index: 7)
            e.setBuffer(self.softContacts, offset: 0, index: 8)
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
            e.setBuffer(self.tets, offset: 0, index: 8)
            e.setBuffer(self.softContacts, offset: 0, index: 9)
        }

        stage("coloring")
        // Coloring (Jacobi rounds, ping-pong). Dense soft-body graphs
        // (cloth: degree ~16) need more rounds to clear conflicts —
        // uncleared conflicts mean connected bodies update simultaneously,
        // which INJECTS energy into stiff constraint networks.
        var src = colorsA, dst = colorsB
        for _ in 0..<20 {
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
                e.setBuffer(self.tets, offset: 0, index: 11)
                e.setBuffer(self.softContacts, offset: 0, index: 12)
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
        stage("solver-iterations")
        let persistPSO = ps("solve_persistent")
        if numBodies <= persistPSO.maxTotalThreadsPerThreadgroup {
            // small scene: the whole solve loop in ONE dispatch — hundreds
            // of per-dispatch launch/barrier latencies become threadgroup
            // barriers (see kernel comment)
            enc.setComputePipelineState(persistPSO)
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
            enc.setBuffer(counters, offset: 0, index: 15)
            enc.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 16)
            enc.setBuffer(shape, offset: 0, index: 17)
            enc.setBuffer(tets, offset: 0, index: 18)
            enc.setBuffer(softContacts, offset: 0, index: 19)
            let w = persistPSO.threadExecutionWidth
            let tg = min(persistPSO.maxTotalThreadsPerThreadgroup,
                         ((max(numBodies, 64) + w - 1) / w) * w)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        } else {
        // tight bound: +1 headroom over what the colorer actually produced
        // last frame (a stale bound only delays a few bodies by one frame;
        // every extra slot costs an empty dispatch + barrier per iteration)
        let colorBound = min(AVBD_MAX_COLORS, max(lastMaxColorUsed + 2, 4))
        let primalPSO = ps("primal_solve")
        for it in 0..<settings.iterations {
            enc.setComputePipelineState(primalPSO)
            do {
                // rebind each iteration (dual_all clobbers low indices), but
                // hoisted out of the color loop: only cIdx changes per color
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
                enc.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 16)
                enc.setBuffer(shape, offset: 0, index: 17)
                enc.setBuffer(tets, offset: 0, index: 18)
                enc.setBuffer(softContacts, offset: 0, index: 19)
            }
            _ = it
            for c in 0..<colorBound {
                var cIdx = UInt32(c)
                enc.setBytes(&cIdx, length: 4, index: 15)
                enc.dispatchThreadgroups(indirectBuffer: colorArgs,
                                         indirectBufferOffset: c * 12,
                                         threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            }
            // tail: bodies in colors >= colorBound (the async bound can be
            // stale; skipped bodies would fly ballistic — see kernel comment)
            do {
                let p = ps("primal_tail")
                enc.setComputePipelineState(p)
                var cb = UInt32(colorBound)
                enc.setBytes(&cb, length: 4, index: 15)
                enc.dispatchThreadgroups(MTLSize(width: (numBodies + 63) / 64, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                enc.setComputePipelineState(primalPSO)
            }

            dispatchIndirect(enc, "dual_all", argsOffset: 6) { e in
                e.setBuffer(self.posLin, offset: 0, index: 0)
                e.setBuffer(self.posAng, offset: 0, index: 1)
                e.setBuffer(self.initLin, offset: 0, index: 2)
                e.setBuffer(self.initAng, offset: 0, index: 3)
                e.setBuffer(self.joints, offset: 0, index: 4)
                e.setBuffer(self.manifolds, offset: 0, index: 5)
                e.setBuffer(self.counters, offset: 0, index: 6)
                e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 7)
                e.setBuffer(self.springs, offset: 0, index: 8)
                e.setBuffer(self.softContacts, offset: 0, index: 9)
            }
        }

        }
        stage("finalize")
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
        // canonicalize colors on the GPU (was a CPU memcpy needing a sync)
        if finalColors !== colorsA, let blit = cmd1.makeBlitCommandEncoder() {
            blit.copy(from: finalColors, sourceOffset: 0,
                      to: colorsA, destinationOffset: 0, size: numBodies * 4)
            blit.endEncoding()
        }
        let readStats = { [weak self] in
            guard let self else { return }
            let ctr = self.counters.contents()
                .bindMemory(to: UInt32.self, capacity: GPUCounters.total)
            let pairs = Int(ctr[GPUCounters.pairs])
            let softN = min(Int(ctr[GPUCounters.soft]), self.maxSoft)
            var counts: [Int] = []
            var maxUsed = -1
            for c in 0..<AVBD_MAX_COLORS {
                let n = Int(ctr[GPUCounters.colorBase + c])
                counts.append(n)
                if n > 0 { maxUsed = c }
            }
            self.statsLock.lock()
            self.lastNumPairs = pairs
            self.lastNumSoft = softN
            self.lastColorCounts = counts
            self.lastMaxColorUsed = maxUsed
            self.statsLock.unlock()
        }
        if profiling {
            cmd1.commit()
            cmd1.waitUntilCompleted()
        } else {
            cmd1.addCompletedHandler { _ in readStats() }
            cmd1.commit()
            inflight.append(cmd1)
        }

        if profiling, let sampleBuf,
           let data = try? sampleBuf.resolveCounterRange(0..<(stageNames.count * 2)) {
            data.withUnsafeBytes { raw in
                let ts = raw.bindMemory(to: UInt64.self)
                // calibrate raw GPU ticks to the command buffer's wall time
                var sum: Double = 0
                var deltas: [Double] = []
                for i in 0..<stageNames.count {
                    let d = Double(ts[i * 2 + 1] &- ts[i * 2])
                    deltas.append(d)
                    sum += d
                }
                let wallNS = (cmd1.gpuEndTime - cmd1.gpuStartTime) * 1e9
                let k = sum > 0 ? wallNS / sum : 0
                for (i, name) in stageNames.enumerated() {
                    profileNS[name, default: 0] += deltas[i] * k
                }
            }
            profileFrames += 1
        }

        if profiling { readStats() }

        swap(&manifolds, &prevManifolds)
        if numTris > 0 { swap(&softContacts, &prevSoftContacts) }
    }

    /// Advance kinematic spinners (static bodies with prescribed rotation).
    private func advanceSpinners() {
        guard !spinners.isEmpty else { return }
        let pa = posAng.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        for sp in spinners {
            let q = Quat(real: pa[sp.body].w,
                         imag: F3(pa[sp.body].x, pa[sp.body].y, pa[sp.body].z))
            let dq = Quat(angle: sp.omega * settings.dt, axis: sp.axis)
            let nq = (dq * q).normalized
            pa[sp.body] = SIMD4(nq.imag, nq.real)
        }
    }

    // MARK: - State access (shared memory)

    public func bodyPosition(_ i: Int) -> F3 {
        sync()
        let p = posLin.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        return F3(p[i].x, p[i].y, p[i].z)
    }

    public func bodyRotation(_ i: Int) -> Quat {
        sync()
        let p = posAng.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        return Quat(real: p[i].w, imag: F3(p[i].x, p[i].y, p[i].z))
    }

    public func bodyMass(_ i: Int) -> Float {
        sync()
        let p = posLin.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        return p[i].w
    }

    public func bodyVelocity(_ i: Int) -> F3 {
        sync()
        let p = velLin.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        return F3(p[i].x, p[i].y, p[i].z)
    }

    public func bodyAngularVelocity(_ i: Int) -> F3 {
        sync()
        let p = velAng.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        return F3(p[i].x, p[i].y, p[i].z)
    }

    // MARK: - Interaction & rendering support

    /// Activate / update a drag joint. Scenes add an inert slot via
    /// PhysicsScene.addDragSlot() (stiffness 0 keeps it disabled until used).
    public func setDrag(jointIndex: Int, body: Int?, worldTarget: F3, localAnchor: F3,

                        stiffness: Float = 5000) {
        sync()
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
        sync()
        let pl = posLin.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        let pa = posAng.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        let sh = shape.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        let st = shapeType.contents().bindMemory(to: UInt32.self, capacity: numBodies)
        var bestT = Float.infinity
        var best: (Int, F3)? = nil
        for i in 0..<numBodies where pl[i].w > 0 {
            let q = Quat(real: pa[i].w, imag: F3(pa[i].x, pa[i].y, pa[i].z))
            let inv = q.conjugate
            let o = inv.act(origin - F3(pl[i].x, pl[i].y, pl[i].z))
            let d = inv.act(dir)
            let stp = shapeType.contents().bindMemory(to: UInt32.self, capacity: numBodies)
            if stp[i] != 0 {
                // sphere/torus: pick against bounding sphere (good enough for grab)
                let r = sh[i].w
                let b = dot(o, d)
                let cc = dot(o, o) - r * r
                let disc = b * b - cc
                if disc < 0 { continue }
                let t = -b - disc.squareRoot()
                if t >= 0 && t < bestT {
                    bestT = t
                    best = (i, o + d * t)
                }
                continue
            }
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
        // no sync: instances are built on the GPU; queue order serializes
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
        enc.setBuffer(shapeType, offset: 0, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: (numBodies + 255) / 256, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    public var bodyCount: Int { numBodies }

    /// Max threadgroup width of the persistent solver kernel (scenes at or
    /// under this run the whole solve in one dispatch).
    public var persistentCapacity: Int { ps("solve_persistent").maxTotalThreadsPerThreadgroup }

    /// Max constraint error: hard-joint violation + contact penetration depth.
    public func maxConstraintError() -> Float {
        sync()
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
