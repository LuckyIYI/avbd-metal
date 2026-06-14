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
    public let tetBoundaryTris: [(Int, Int, Int)]
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
    var hashedIdx, globalIdx, hashedRigidIdx: MTLBuffer
    var cellCount, cellStart, cellBodies, cellRigid: MTLBuffer
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

    // Render surface meshes (cloth triangles + tet boundary faces):
    // packed corner ids (body | component<<24), per-vertex incidence CSR
    // for smooth normals, and the per-body surfaced flag.
    var surfTriBuf, surfVertsBuf, surfVtStart, surfVtCount, surfVtList: MTLBuffer
    var surfacedFlags, softNormalsBuf, faceNormalsBuf, renderTriBuf: MTLBuffer
    var clothGroupBuf: MTLBuffer
    // Visual-only tetrahedral skinning buffers. These draw arbitrary mesh
    // surfaces embedded in tets; collisions still use `trisBuf`.
    let skinnedVertexCount: Int
    let skinnedTriCount: Int
    var skinBindingBuf, skinVertexBuf, skinTriBuf: MTLBuffer
    // Voronoi temporal tracking: per-vertex / per-edge persistent closest-
    // element candidate sets, plus the topology they propagate through.
    var vtTrackBuf, eeTrackBuf, triAdjBuf: MTLBuffer
    var clothVertFlag: MTLBuffer
    var boundsBuf, ogcPrevBuf, ogcArgsBuf: MTLBuffer
    var nbr2Start, nbr2Count, nbr2List: MTLBuffer
    var vertEdgeStart, vertEdgeCount, vertEdgeList: MTLBuffer
    public private(set) var surfaceTriCount: Int = 0
    public private(set) var renderTriCount: Int = 0
    var surfVertCount: Int = 0
    public private(set) var staticUsedColors: Int = 1
    // Contact-aware per-frame coloring for every scene WITHOUT soft
    // elements: rigid stacking (stack/jenga/gearclock) provably needs
    // strict GS contact ordering, and springs were always part of the
    // dynamic coloring graph. Only cloth/tet scenes use the static palette.
    var usesDynamicColoring: Bool { numTris == 0 && numTets == 0 }
    var dynColorSrc: MTLBuffer?

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
    public internal(set) var lastColorCounts: [Int] = []
    public private(set) var lastNumPairs: Int = 0
    private var warnedPairSaturation = false
    private var warnedSoftSaturation = false
    public private(set) var lastNumSoft: Int = 0
    // Start pessimistic so the first frame covers every possible color.
    public internal(set) var lastMaxColorUsed: Int = AVBD_MAX_COLORS - 1
    public private(set) var frameIndex: Int = 0
    /// (joint index, rad/s) — servo targets advanced each step
    private var rateMotors: [(Int, Float)] = []

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
        // Floor of 4096: small scenes are cheap (688B/slot) but elongated
        // shapes (jenga blocks) legitimately exceed 16 candidates/body, and
        // a saturated pair list silently drops a scheduling-random subset of
        // pairs each frame — bodies lose ground support and sink/freefall.
        self.maxPairs = max(4096, scene.bodies.count * maxPairsPerBody)
        self.mapCapacity = Self.nextPow2(2 * maxPairs)
        self.gridHashSize = Self.nextPow2(max(64, 2 * numBodies))
        // Tet BOUNDARY faces are collision triangles too: soft-soft contact
        // is element-based (soft V-V sphere pairs are banned at broadphase),
        // so without them tet bodies would pass through each other and
        // rigids would only touch their corner features. Collision-only —
        // membranes and the render extractor keep using scene.tris/tets.
        self.tetBoundaryTris = Self.tetBoundaryFaces(scene)
        self.numTris = scene.tris.count + tetBoundaryTris.count
        self.skinnedVertexCount = scene.skinnedMeshes.reduce(0) { $0 + $1.vertices.count }
        self.skinnedTriCount = scene.skinnedMeshes.reduce(0) { $0 + $1.triangles.count }
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
        hashedRigidIdx = try makeBuf(nb * 4, "hashedRigidIdx")
        clothVertFlag = try makeBuf(nb * 4, "clothVertFlag")
        boundsBuf = try makeBuf(nb * 4, "ogcBounds")
        ogcPrevBuf = try makeBuf(nb * 16, "ogcPrev")
        ogcArgsBuf = try makeBuf(3 * 4, "ogcArgs")
        // 2-ring CSR sized after derivation below (~19/vertex upper bound)
        nbr2Start = try makeBuf(nb * 4, "nbr2Start")
        nbr2Count = try makeBuf(nb * 4, "nbr2Count")
        nbr2List = try makeBuf(max(1, numTris * 24) * 4, "nbr2List")
        cellCount = try makeBuf(gridHashSize * 4, "cellCount")
        cellRigid = try makeBuf(gridHashSize * 4, "cellRigid")
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
        // AABB multi-cell: cover loops are clamped to 3x3x3 per element,
        // so 27x capacity makes overflow impossible
        elemCells = try makeBuf(max(1, 27 * (numTris + maxEdges)) * 4, "elemCells")
        elemSlot = try makeBuf(elemHashSize * 4, "elemCursor")
        softContacts = try makeBuf(maxSoft * MemoryLayout<SoftContactGPU>.stride, "softContacts")
        prevSoftContacts = try makeBuf(maxSoft * MemoryLayout<SoftContactGPU>.stride, "prevSoftContacts")
        softMapKeyA = try makeBuf(softMapCapacity * 4, "softMapKeyA")
        softMapKeyB = try makeBuf(softMapCapacity * 4, "softMapKeyB")
        softMapVal = try makeBuf(softMapCapacity * 4, "softMapVal")
        membranes = try makeBuf(max(1, numTris) * MemoryLayout<MembraneGPU>.stride, "membranes")
        bends = try makeBuf(max(1, maxEdges) * MemoryLayout<BendGPU>.stride, "bends")

        // render surface capacity: cloth tris + worst-case tet boundary;
        // the render list adds back layers + hem rims for thin sheets
        let maxSurfTris = numTris + 4 * numTets
        surfTriBuf = try makeBuf(max(1, maxSurfTris) * 12, "surfTris")
        surfVertsBuf = try makeBuf(nb * 4, "surfVerts")
        surfVtStart = try makeBuf(nb * 4, "surfVtStart")
        surfVtCount = try makeBuf(nb * 4, "surfVtCount")
        surfVtList = try makeBuf(max(1, maxSurfTris) * 12, "surfVtList")
        surfacedFlags = try makeBuf(nb * 4, "surfacedFlags")
        softNormalsBuf = try makeBuf(nb * 16, "softNormals")
        faceNormalsBuf = try makeBuf(max(1, maxSurfTris) * 16, "faceNormals")
        renderTriBuf = try makeBuf(max(1, 8 * numTris + 4 * numTets) * 12, "renderTris")
        clothGroupBuf = try makeBuf(nb * 4, "clothGroup")
        skinBindingBuf = try makeBuf(max(1, skinnedVertexCount) * MemoryLayout<SkinBindingGPU>.stride,
                                     "skinBindings")
        skinVertexBuf = try makeBuf(max(1, skinnedVertexCount) * MemoryLayout<SkinVertexGPU>.stride,
                                    "skinVertices")
        skinTriBuf = try makeBuf(max(1, skinnedTriCount) * 12, "skinTris")
        vtTrackBuf = try makeBuf(nb * 16, "vtTrack")
        eeTrackBuf = try makeBuf(max(1, maxEdges) * 16, "eeTrack")
        triAdjBuf = try makeBuf(max(1, numTris) * 16, "triAdj")
        vertEdgeStart = try makeBuf(nb * 4, "vertEdgeStart")
        vertEdgeCount = try makeBuf(nb * 4, "vertEdgeCount")
        vertEdgeList = try makeBuf(max(1, maxEdges) * 8, "vertEdgeList")

        degrees = try makeBuf(nb * 4, "degrees")
        adjStart = try makeBuf(nb * 4, "adjStart")
        adjCursor = try makeBuf(nb * 4, "adjCursor")
        adjList = try makeBuf((2 * (numJoints + numSprings + maxPairs) + 4 * numTets
                               + 4 * maxSoft + 3 * numTris + 4 * maxEdges) * 4, "adjList")
        colorsA = try makeBuf(nb * 4, "colorsA")
        colorsB = try makeBuf(nb * 4, "colorsB")
        bodySlot = try makeBuf(nb * 4, "bodySlot")
        colorStart = try makeBuf((AVBD_MAX_COLORS + 2) * 4, "colorStart")
        colorList = try makeBuf(nb * 4, "colorList")
        changedFlag = try makeBuf(24 * 4, "changedFlag")  // per-pass slots

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

    private struct UInt64UniqueBuilder {
        var table: [UInt64]
        var mask: Int

        init(capacity: Int) {
            var n = 1
            while n < max(2, capacity * 2) { n <<= 1 }
            table = [UInt64](repeating: 0, count: n)
            mask = n - 1
        }

        static func hash(_ x: UInt64) -> Int {
            var h = x
            h ^= h >> 33
            h &*= 0xff51afd7ed558ccd
            h ^= h >> 33
            h &*= 0xc4ceb9fe1a85ec53
            h ^= h >> 33
            return Int(truncatingIfNeeded: h)
        }

        mutating func insert(_ key: UInt64) -> Bool {
            let stored = key &+ 1
            var slot = Self.hash(key) & mask
            while true {
                let cur = table[slot]
                if cur == stored { return false }
                if cur == 0 {
                    table[slot] = stored
                    return true
                }
                slot = (slot + 1) & mask
            }
        }
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
        shaderLib = lib
        for name in lib.functionNames {
            guard let fn = lib.makeFunction(name: name) else { continue }
            pso[name] = try device.makeComputePipelineState(function: fn)
        }
    }
    var shaderLib: MTLLibrary?

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

    /// Boundary faces of the tet meshes (faces used by exactly one tet),
    /// wound outward against the opposite vertex. These become collision
    /// triangles so V-T/E-E own soft-soft and rigid-face-vs-soft contact.
    static func tetBoundaryFaces(_ scene: PhysicsScene) -> [(Int, Int, Int)] {
        guard !scene.tets.isEmpty else { return [] }
        var faces: [SIMD3<Int>: (tri: (Int, Int, Int), opp: Int, count: Int)] = [:]
        for t in scene.tets {
            let ids = [t.ids.0, t.ids.1, t.ids.2, t.ids.3]
            for (i, j, k, o) in [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 3, 1), (1, 2, 3, 0)] {
                let tri = (ids[i], ids[j], ids[k])
                let key = SIMD3([tri.0, tri.1, tri.2].sorted())
                if var e = faces[key] { e.count += 1; faces[key] = e }
                else { faces[key] = (tri, ids[o], 1) }
            }
        }
        var out: [(Int, Int, Int)] = []
        for (_, f) in faces where f.count == 1 {
            var (a, b, c) = f.tri
            let pa = scene.bodies[a].position
            let n = cross(scene.bodies[b].position - pa, scene.bodies[c].position - pa)
            if dot(n, scene.bodies[f.opp].position - pa) > 0 { swap(&b, &c) }
            out.append((a, b, c))
        }
        // deterministic order (dictionary iteration is not)
        return out.sorted { $0 < $1 }
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
        // Cell size: 2x the max hashed radius (sphere-bound broadphase).
        // shape.w is NEGATIVE for particles — take |.| or a particles-only
        // hashed set never raises the floor and the old 0.5 floor put a
        // whole mattress of particles into a handful of 1 m cells (atomic
        // contention in bp_count + thousands-long cell lists in pair gen).
        var maxHashedRadius: Float = 0.05
        for i in hashed {
            maxHashedRadius = max(maxHashedRadius, abs(sh[Int(i)].w))
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
                    if j.motorRate != 0 { rateMotors.append((i, j.motorRate)) }
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

        func sortedUnique(_ input: [UInt64]) -> [UInt64] {
            guard !input.isEmpty else { return [] }
            var keys = input
            keys.sort()
            var out: [UInt64] = []
            out.reserveCapacity(keys.count)
            var last = keys[0]
            out.append(last)
            for key in keys.dropFirst() where key != last {
                out.append(key)
                last = key
            }
            return out
        }

        func sortSmallUInt32(_ a: inout [UInt32]) {
            guard a.count > 1 else { return }
            for i in 1..<a.count {
                let x = a[i]
                var j = i
                while j > 0 && a[j - 1] > x {
                    a[j] = a[j - 1]
                    j -= 1
                }
                a[j] = x
            }
        }

        // Collision exclusions: sorted (min,max) pairs of jointed/springed bodies
        var excl: [UInt64] = []
        excl.reserveCapacity(scene.joints.count + scene.springs.count)
        for j in scene.joints where j.bodyA >= 0 {
            let lo = UInt64(min(j.bodyA, j.bodyB)), hi = UInt64(max(j.bodyA, j.bodyB))
            excl.append(lo << 32 | hi)
        }
        for s in scene.springs {
            let lo = UInt64(min(s.bodyA, s.bodyB)), hi = UInt64(max(s.bodyA, s.bodyB))
            excl.append(lo << 32 | hi)
        }
        let sortedExcl = sortedUnique(excl)
        let ep = exclusions.contents().bindMemory(to: SIMD2<UInt32>.self, capacity: max(1, sortedExcl.count))
        for (i, key) in sortedExcl.enumerated() {
            ep[i] = SIMD2(UInt32(key >> 32), UInt32(key & 0xFFFFFFFF))
        }
        numExclusions = UInt32(sortedExcl.count)

        // ---- Cloth element topology ----
        // Triangles (scene cloth tris + synthesized tet boundary faces),
        // unique edges, per-vertex topological neighborhoods (the V-T/E-E
        // exclusion sets: vertices sharing a triangle), particle list.
        let collTris: [(Int, Int, Int)] = scene.tris.map { $0.ids } + tetBoundaryTris
        let tp4 = trisBuf.contents().bindMemory(to: SIMD4<UInt32>.self,
                                                capacity: max(1, numTris))
        var edgeSet = UInt64UniqueBuilder(capacity: collTris.count * 3)
        var edgeKeys: [UInt64] = []
        edgeKeys.reserveCapacity(collTris.count * 3)
        var maxElemR: Float = 0
        for (i, ids) in collTris.enumerated() {
            let (a, b, c) = ids
            tp4[i] = SIMD4(UInt32(a), UInt32(b), UInt32(c), 0)
            for (u, v) in [(a, b), (b, c), (a, c)] {
                let key = UInt64(min(u, v)) << 32 | UInt64(max(u, v))
                if edgeSet.insert(key) { edgeKeys.append(key) }
            }
            let pa = scene.bodies[a].position, pb = scene.bodies[b].position
            let pc = scene.bodies[c].position
            let m = (pa + pb + pc) / 3
            let thick = max(scene.bodies[a].size.x, max(scene.bodies[b].size.x,
                                                        scene.bodies[c].size.x)) / 2
            maxElemR = max(maxElemR, max(distance(m, pa), max(distance(m, pb),
                                                                              distance(m, pc))) + thick)
        }
        numEdges = edgeKeys.count
        let ep2 = edgesBuf.contents().bindMemory(to: SIMD2<UInt32>.self,
                                                 capacity: max(1, numEdges))
        var nbrArrays = [[UInt32]](repeating: [], count: numBodies)
        var vertEdges: [[UInt32]] = Array(repeating: [], count: numBodies)
        for (i, key) in edgeKeys.enumerated() {
            let ai = Int(key >> 32)
            let bi = Int(key & 0xFFFFFFFF)
            ep2[i] = SIMD2(UInt32(ai), UInt32(bi))
            nbrArrays[ai].append(UInt32(bi))
            nbrArrays[bi].append(UInt32(ai))
            vertEdges[ai].append(UInt32(i))
            vertEdges[bi].append(UInt32(i))
            let a = scene.bodies[ai], b = scene.bodies[bi]
            maxElemR = max(maxElemR, distance(a.position, b.position) / 2
                           + max(a.size.x, b.size.x) / 2)
        }
        for i in 0..<numBodies {
            sortSmallUInt32(&nbrArrays[i])
        }
        // triangle adjacency (shared edges, ALL triangles) for tracker
        // propagation
        var edgeToTris: [UInt64: (Int, Int)] = [:]
        for (ti, ids3) in collTris.enumerated() {
            let ids = [ids3.0, ids3.1, ids3.2]
            for (u, v) in [(ids[0], ids[1]), (ids[1], ids[2]), (ids[0], ids[2])] {
                let key = UInt64(min(u, v)) << 32 | UInt64(max(u, v))
                if var e = edgeToTris[key] { e.1 = ti; edgeToTris[key] = e }
                else { edgeToTris[key] = (ti, -1) }
            }
        }
        var triAdj = [SIMD4<UInt32>](repeating: SIMD4(repeating: 0xFFFFFFFF),
                                     count: max(1, numTris))
        var triAdjFill = [Int](repeating: 0, count: max(1, numTris))
        for (_, e) in edgeToTris where e.1 >= 0 {
            if triAdjFill[e.0] < 3 { triAdj[e.0][triAdjFill[e.0]] = UInt32(e.1); triAdjFill[e.0] += 1 }
            if triAdjFill[e.1] < 3 { triAdj[e.1][triAdjFill[e.1]] = UInt32(e.0); triAdjFill[e.1] += 1 }
        }
        if numTris > 0 {
            triAdjBuf.contents().bindMemory(to: SIMD4<UInt32>.self, capacity: numTris)
                .update(from: triAdj, count: numTris)
        }
        let vesP = vertEdgeStart.contents().bindMemory(to: UInt32.self, capacity: numBodies)
        let vecP = vertEdgeCount.contents().bindMemory(to: UInt32.self, capacity: numBodies)
        var veFlat: [UInt32] = []
        for i in 0..<numBodies {
            vesP[i] = UInt32(veFlat.count)
            vecP[i] = UInt32(vertEdges[i].count)
            veFlat.append(contentsOf: vertEdges[i])
        }
        if !veFlat.isEmpty {
            precondition(veFlat.count <= vertEdgeList.length / 4)
            vertEdgeList.contents().bindMemory(to: UInt32.self, capacity: veFlat.count)
                .update(from: veFlat, count: veFlat.count)
        }
        memset(vtTrackBuf.contents(), 0xFF, vtTrackBuf.length)
        memset(eeTrackBuf.contents(), 0xFF, eeTrackBuf.length)

        // CSR neighborhoods (sorted per vertex for binary search)
        let nsP = nbrStart.contents().bindMemory(to: UInt32.self, capacity: numBodies)
        let ncP = nbrCount.contents().bindMemory(to: UInt32.self, capacity: numBodies)
        var flatNbrs: [UInt32] = []
        for i in 0..<numBodies {
            nsP[i] = UInt32(flatNbrs.count)
            ncP[i] = UInt32(nbrArrays[i].count)
            flatNbrs.append(contentsOf: nbrArrays[i])
        }
        precondition(flatNbrs.count <= nbrList.length / 4,
                     "neighbor CSR exceeded 6-per-triangle bound")
        if !flatNbrs.isEmpty {
            nbrList.contents().bindMemory(to: UInt32.self, capacity: flatNbrs.count)
                .update(from: flatNbrs, count: flatNbrs.count)
        }
        // 2-RING neighborhoods (sorted, for the OGC conservative bounds):
        // bounds must EXCLUDE near-geodesic same-sheet elements — the 2-ring
        // sits at ~2 edge-lengths permanently and would cap every bound at a
        // crawl. Safe to exclude: cloth thickness is clamped below
        // 0.38*spacing, so 2-ring rest distances stay outside contact range.
        let n2sP = nbr2Start.contents().bindMemory(to: UInt32.self, capacity: numBodies)
        let n2cP = nbr2Count.contents().bindMemory(to: UInt32.self, capacity: numBodies)
        var flat2: [UInt32] = []
        var marks = [UInt32](repeating: 0, count: numBodies)
        var mark: UInt32 = 1
        var two: [UInt32] = []
        two.reserveCapacity(32)
        for i in 0..<numBodies {
            n2sP[i] = UInt32(flat2.count)
            if nbrArrays[i].isEmpty {
                n2cP[i] = 0
                continue
            }
            mark &+= 1
            if mark == 0 {
                marks = [UInt32](repeating: 0, count: numBodies)
                mark = 1
            }
            marks[i] = mark
            two.removeAll(keepingCapacity: true)
            for nb1 in nbrArrays[i] {
                let n1 = Int(nb1)
                if marks[n1] != mark {
                    marks[n1] = mark
                    two.append(nb1)
                }
                for nb2 in nbrArrays[n1] {
                    let n2 = Int(nb2)
                    if marks[n2] != mark {
                        marks[n2] = mark
                        two.append(nb2)
                    }
                }
            }
            sortSmallUInt32(&two)
            n2cP[i] = UInt32(two.count)
            flat2.append(contentsOf: two)
        }
        if !flat2.isEmpty {
            precondition(flat2.count <= nbr2List.length / 4,
                         "2-ring CSR exceeded bound")
            nbr2List.contents().bindMemory(to: UInt32.self, capacity: flat2.count)
                .update(from: flat2, count: flat2.count)
        }

        // Collision-surface vertices: V-T queries should come from the
        // deformable collision surface, not from interior FEM nodes. Interior
        // tet particles are governed by volume elements; letting them emit
        // surface contacts creates nonphysical self-pressure and extra work.
        var surfaceParticleFlags = [UInt8](repeating: 0, count: numBodies)
        for (a, b, c) in collTris {
            if scene.bodies[a].isParticle { surfaceParticleFlags[a] = 1 }
            if scene.bodies[b].isParticle { surfaceParticleFlags[b] = 1 }
            if scene.bodies[c].isParticle { surfaceParticleFlags[c] = 1 }
        }
        var particles: [UInt32] = []
        particles.reserveCapacity(numBodies)
        for i in 0..<numBodies where surfaceParticleFlags[i] != 0 {
            particles.append(UInt32(i))
        }
        numParticles = numTris == 0 ? 0 : particles.count
        if !particles.isEmpty {
            particleIdxBuf.contents().bindMemory(to: UInt32.self, capacity: particles.count)
                .update(from: particles, count: particles.count)
        }
        // ---- Membrane + bending elements (tris with material) ----
        var mems: [MembraneGPU] = []
        var bendEls: [BendGPU] = []
        var edgeTris: [UInt64: [(Int, Int, Float)]] = [:]   // edge -> (tri, oppVert, bendK)
        for t in scene.tris where t.mu > 0 {
            let (i0, i1, i2) = t.ids
            let x0 = scene.bodies[i0].position
            let x1 = scene.bodies[i1].position
            let x2 = scene.bodies[i2].position
            let d1 = x1 - x0, d2 = x2 - x0
            let nrm = cross(d1, d2)
            let area = length(nrm) / 2
            guard area > 1e-10 else { continue }
            let e1 = normalize(d1)
            let e2 = normalize(cross(normalize(nrm), e1))
            // rest 2x2 (columns d1, d2 in the local frame) and its inverse
            let a = dot(d1, e1), b = dot(d1, e2)
            let c = dot(d2, e1), dd = dot(d2, e2)
            let det = a * dd - b * c
            guard abs(det) > 1e-12 else { continue }
            // inverse, stored column-major (ia, ib, ic, id):
            // F col1 = d0*ia + d1*ib, col2 = d0*ic + d1*id
            let ia = dd / det, ib = -b / det
            let ic = -c / det, id = a / det
            var m = MembraneGPU()
            m.ids = SIMD4(UInt32(i0), UInt32(i1), UInt32(i2), 0)
            m.dm = SIMD4(ia, ib, ic, id)
            m.mat = SIMD4(t.mu * area, t.lambda * area, area, 0)
            mems.append(m)
            if t.bend > 0 {
                for (u, v, opp) in [(i0, i1, i2), (i1, i2, i0), (i0, i2, i1)] {
                    let key = UInt64(min(u, v)) << 32 | UInt64(max(u, v))
                    edgeTris[key, default: []].append((mems.count - 1, opp, t.bend))
                }
            }
        }
        // hinges: numeric rank-1 K from the INTRINSIC (unfolded) rest shape;
        // null space of {constants, in-plane positions}, scaled to the
        // cotangent convention (flap coefficient = sum of its triangle's
        // edge-adjacent cotangents)
        for (key, list) in edgeTris where list.count == 2 {
            let v0 = Int(key >> 32), v1 = Int(key & 0xFFFFFFFF)
            let (_, opp2, k1) = list[0]
            let (_, opp3, k2) = list[1]
            let p0 = scene.bodies[v0].position
            let p1 = scene.bodies[v1].position
            let L = distance(p0, p1)
            guard L > 1e-9 else { continue }
            // unfold both flaps into the plane from rest edge lengths
            func unfold(_ opp: Int, side: Float) -> SIMD2<Float>? {
                let l0 = distance(scene.bodies[opp].position, p0)
                let l1 = distance(scene.bodies[opp].position, p1)
                let ax = (L * L + l0 * l0 - l1 * l1) / (2 * L)
                let h2 = l0 * l0 - ax * ax
                guard h2 > 1e-12 else { return nil }
                return SIMD2(ax, side * h2.squareRoot())
            }
            guard let q2 = unfold(opp2, side: 1), let q3 = unfold(opp3, side: -1)
            else { continue }
            let q0 = SIMD2<Float>(0, 0), q1 = SIMD2<Float>(L, 0)
            // K = null space of rows {1,1,1,1}, {x...}, {y...} (generalized
            // 4D cross product via 3x3 minors)
            let r1 = SIMD4<Float>(1, 1, 1, 1)
            let r2 = SIMD4<Float>(q0.x, q1.x, q2.x, q3.x)
            let r3 = SIMD4<Float>(q0.y, q1.y, q2.y, q3.y)
            func minor(_ c0: Int, _ c1: Int, _ c2: Int) -> Float {
                let m = simd_float3x3(
                    SIMD3(r1[c0], r2[c0], r3[c0]),
                    SIMD3(r1[c1], r2[c1], r3[c1]),
                    SIMD3(r1[c2], r2[c2], r3[c2]))
                return m.determinant
            }
            var K = SIMD4<Float>(minor(1, 2, 3), -minor(0, 2, 3),
                                 minor(0, 1, 3), -minor(0, 1, 2))
            // scale: flap-2 coefficient = cot(angle at q0 in T1) + cot(at q1)
            func cot(_ apex: SIMD2<Float>, _ a: SIMD2<Float>, _ b: SIMD2<Float>) -> Float {
                let u = a - apex, v = b - apex
                let cr = u.x * v.y - u.y * v.x
                return abs(cr) > 1e-12 ? (u.x * v.x + u.y * v.y) / abs(cr) : 0
            }
            let want = cot(q0, q1, q2) + cot(q1, q0, q2)
            guard abs(K.z) > 1e-12 else { continue }
            K *= want / K.z
            let a1 = abs((q1.x - q0.x) * q2.y) / 2     // unfolded areas
            let a2 = abs((q1.x - q0.x) * q3.y) / 2
            var bd = BendGPU()
            bd.ids = SIMD4(UInt32(v0), UInt32(v1), UInt32(opp2), UInt32(opp3))
            bd.K = K
            bd.mat = SIMD4(3 * (k1 + k2) / 2 / max(a1 + a2, 1e-9), 0, 0, 0)
            bendEls.append(bd)
        }
        if !mems.isEmpty {
            membranes.contents().bindMemory(to: MembraneGPU.self, capacity: mems.count)
                .update(from: mems, count: mems.count)
        }
        if !bendEls.isEmpty {
            precondition(bendEls.count <= bends.length / MemoryLayout<BendGPU>.stride)
            bends.contents().bindMemory(to: BendGPU.self, capacity: bendEls.count)
                .update(from: bendEls, count: bendEls.count)
        }
        params.numMembranes = UInt32(mems.count)
        params.numBends = UInt32(bendEls.count)

        // Element-contact margin scales with the cloth skin (the rigid
        // 1 cm margin exceeds 2r at high resolution and bloats detection)
        var maxPartR: Float = 0
        var edgeLenSum: Float = 0
        for key in edgeKeys {
            let a = scene.bodies[Int(key >> 32)], b = scene.bodies[Int(key & 0xFFFFFFFF)]
            maxPartR = max(maxPartR, max(a.size.x, b.size.x) / 2)
            edgeLenSum += distance(a.position, b.position)
        }
        let meanEdge = numEdges > 0 ? edgeLenSum / Float(numEdges) : 0.2
        params.elemMargin = min(0.01, max(0.002, 0.5 * maxPartR))
        // Detection-radius cells (AABB multi-cell insertion): reach must
        // cover skin + margin + velocity inflation (<= 0.3 cell, hence /0.7).
        // The broadphase bins fat element AABBs and clamps cover loops to
        // 3 cells/axis, so the cell also has to cover the largest element
        // radius plus collision padding.
        let reach = (2 * maxPartR + params.elemMargin) / 0.7
        params.elemCellSize = max(0.02, max(max(reach, meanEdge * 1.05),
                                            maxElemR + maxPartR
                                                + params.elemMargin))
        params.elemHashSize = UInt32(elemHashSize)
        params.numTris = UInt32(numTris)
        params.numEdges = UInt32(numEdges)
        params.numParticles = UInt32(numParticles)
        params.maxSoft = UInt32(maxSoft)
        params.softMapCapacity = UInt32(softMapCapacity)

        // ---- Render surface meshes: cloth triangles as authored + the
        // boundary faces of tet bodies (faces owned by exactly one tet),
        // wound outward against the opposite rest vertex. Connected
        // components give each sheet/jelly its own color.
        var skinnedBodies = Set<Int>()
        for mesh in scene.skinnedMeshes {
            skinnedBodies.formUnion(mesh.bodyIDs)
            for v in mesh.vertices {
                skinnedBodies.insert(v.ids.0)
                skinnedBodies.insert(v.ids.1)
                skinnedBodies.insert(v.ids.2)
                skinnedBodies.insert(v.ids.3)
            }
        }
        var surfCorners: [UInt32] = []          // 3 per tri: body | comp<<24
        var triVerts: [[Int]] = []              // per surface tri, its bodies
        var isCloth: [Bool] = []                // thin sheet (two render layers)
        for t in scene.tris {
            triVerts.append([t.ids.0, t.ids.1, t.ids.2])
            isCloth.append(true)
        }
        let allTetsSkinned = !scene.tets.isEmpty && !skinnedBodies.isEmpty
            && scene.tets.allSatisfy {
                skinnedBodies.contains($0.ids.0) && skinnedBodies.contains($0.ids.1)
                    && skinnedBodies.contains($0.ids.2) && skinnedBodies.contains($0.ids.3)
            }
        if !allTetsSkinned {
            var faceCount: [UInt64: (count: Int, tri: (Int, Int, Int), opp: Int)] = [:]
            for tet in scene.tets {
                let ids = [tet.ids.0, tet.ids.1, tet.ids.2, tet.ids.3]
                for (f, o) in [((0, 1, 2), 3), ((0, 3, 1), 2), ((1, 3, 2), 0), ((0, 2, 3), 1)] {
                    let tri = (ids[f.0], ids[f.1], ids[f.2])
                    let sorted3 = [tri.0, tri.1, tri.2].sorted()
                    let key = UInt64(sorted3[0]) << 42 | UInt64(sorted3[1]) << 21 | UInt64(sorted3[2])
                    if var e = faceCount[key] {
                        e.count += 1
                        faceCount[key] = e
                    } else {
                        faceCount[key] = (1, tri, ids[o])
                    }
                }
            }
            for (_, e) in faceCount where e.count == 1 {
                var (a, b, c) = e.tri
                // outward winding: flip if the rest normal points toward the
                // opposite (interior) vertex
                let pa = scene.bodies[a].position
                let n = cross(scene.bodies[b].position - pa, scene.bodies[c].position - pa)
                if dot(n, scene.bodies[e.opp].position - pa) > 0 { swap(&b, &c) }
                if !skinnedBodies.isEmpty && skinnedBodies.contains(a)
                    && skinnedBodies.contains(b) && skinnedBodies.contains(c) {
                    continue
                }
                triVerts.append([a, b, c])
                isCloth.append(false)
            }
        }
        surfaceTriCount = triVerts.count
        if surfaceTriCount > 0 {
            // connected components (union-find over surface vertices)
            var parent: [Int: Int] = [:]
            func find(_ x: Int) -> Int {
                var r = x
                while let p = parent[r], p != r { r = p }
                var c = x
                while let p = parent[c], p != r { parent[c] = r; c = p }
                if parent[r] == nil { parent[r] = r }
                return r
            }
            for tv in triVerts {
                let r = find(tv[0])
                parent[find(tv[1])] = r
                parent[find(tv[2])] = r
            }
            var compIdx: [Int: Int] = [:]
            defer { params.numSoftGroups = UInt32(compIdx.count) }
            var incidence: [Int: [Int]] = [:]   // body -> surface tri indices
            var triComp: [UInt32] = []
            let groupP = clothGroupBuf.contents().bindMemory(to: UInt32.self,
                                                             capacity: numBodies)
            let clothP = clothVertFlag.contents().bindMemory(to: UInt32.self,
                                                             capacity: numBodies)
            for i in 0..<numBodies { groupP[i] = 0; clothP[i] = 0 }
            for (ti, tv) in triVerts.enumerated() {
                let root = find(tv[0])
                if compIdx[root] == nil { compIdx[root] = compIdx.count }
                let comp = UInt32(min(compIdx[root]!, 255))
                triComp.append(comp)
                for v in tv {
                    surfCorners.append(UInt32(v) | (comp << 24))
                    incidence[v, default: []].append(ti)
                    groupP[v] = comp + 1      // same-surface V-V is handled
                                              // by V-T/E-E, never spheres
                    if isCloth[ti] { clothP[v] = 1 }  // thin sheet: render
                                              // thickness is OPT-IN
                }
            }
            surfTriBuf.contents().bindMemory(to: UInt32.self, capacity: surfCorners.count)
                .update(from: surfCorners, count: surfCorners.count)

            // Render triangle list. Thin sheets get a front layer (+r along
            // the smooth normal), a back layer (-r, reversed winding) and
            // rim quads along boundary edges; tet boundaries are already
            // closed and get a single outward-offset layer. Bit 23 of each
            // packed corner is the side flag.
            precondition(numBodies < (1 << 21), "flag bits collide with body id")
            let side: UInt32 = 1 << 23
            let flat: UInt32 = 1 << 21      // tet boundaries: sharp edges,
                                            // shade with face normals
            var renderCorners: [UInt32] = []
            var clothEdgeUse: [UInt64: (Int, Int, UInt32)] = [:]  // first use winding
            for (ti, tv) in triVerts.enumerated() {
                let comp = triComp[ti] << 24
                let flatBit = isCloth[ti] ? 0 : flat
                let (a, b, c) = (UInt32(tv[0]) | flatBit, UInt32(tv[1]) | flatBit,
                                 UInt32(tv[2]) | flatBit)
                renderCorners.append(contentsOf: [a | comp, b | comp, c | comp])
                if isCloth[ti] {
                    renderCorners.append(contentsOf: [(a | side) | comp,
                                                      (c | side) | comp,
                                                      (b | side) | comp])
                    for (u, v) in [(tv[0], tv[1]), (tv[1], tv[2]), (tv[2], tv[0])] {
                        let key = UInt64(min(u, v)) << 32 | UInt64(max(u, v))
                        if clothEdgeUse.removeValue(forKey: key) == nil {
                            clothEdgeUse[key] = (u, v, comp)   // directed as authored
                        }
                    }
                }
            }
            // hem rims: edges used by exactly one cloth triangle (bit 22
            // marks rim corners so the AO prepass can skip them)
            let rim: UInt32 = 1 << 22
            for (_, (u, v, comp)) in clothEdgeUse {
                let uF = UInt32(u) | comp | rim, vF = UInt32(v) | comp | rim
                let uB = uF | side, vB = vF | side
                renderCorners.append(contentsOf: [uF, vF, vB, uF, vB, uB])
            }
            renderTriCount = renderCorners.count / 3
            precondition(renderCorners.count * 4 <= renderTriBuf.length,
                         "render corner list exceeded capacity")
            renderTriBuf.contents().bindMemory(to: UInt32.self, capacity: renderCorners.count)
                .update(from: renderCorners, count: renderCorners.count)
            // incidence CSR over the surfaced-vertex list + flags
            let flags = surfacedFlags.contents().bindMemory(to: UInt32.self, capacity: numBodies)
            for i in 0..<numBodies { flags[i] = 0 }
            var verts: [UInt32] = []
            var starts: [UInt32] = []
            var counts: [UInt32] = []
            var list: [UInt32] = []
            for (v, tris) in incidence.sorted(by: { $0.key < $1.key }) {
                verts.append(UInt32(v))
                starts.append(UInt32(list.count))
                counts.append(UInt32(tris.count))
                list.append(contentsOf: tris.map(UInt32.init))
                flags[v] = 1
            }
            surfVertCount = verts.count
            surfVertsBuf.contents().bindMemory(to: UInt32.self, capacity: verts.count)
                .update(from: verts, count: verts.count)
            surfVtStart.contents().bindMemory(to: UInt32.self, capacity: starts.count)
                .update(from: starts, count: starts.count)
            surfVtCount.contents().bindMemory(to: UInt32.self, capacity: counts.count)
                .update(from: counts, count: counts.count)
            surfVtList.contents().bindMemory(to: UInt32.self, capacity: max(1, list.count))
                .update(from: list.isEmpty ? [0] : list, count: max(1, list.count))
        } else {
            memset(surfacedFlags.contents(), 0, surfacedFlags.length)
            memset(clothGroupBuf.contents(), 0, clothGroupBuf.length)
            memset(clothVertFlag.contents(), 0, clothVertFlag.length)
        }
        memset(softNormalsBuf.contents(), 0, softNormalsBuf.length)

        if skinnedVertexCount > 0 {
            precondition(skinnedVertexCount < (1 << 24), "skin vertex id exceeds render packing")
            var bindings: [SkinBindingGPU] = []
            bindings.reserveCapacity(skinnedVertexCount)
            var skinCorners: [UInt32] = []
            skinCorners.reserveCapacity(skinnedTriCount * 3)
            let flags = surfacedFlags.contents().bindMemory(to: UInt32.self, capacity: numBodies)
            var base = 0
            for (mi, mesh) in scene.skinnedMeshes.enumerated() {
                let comp = UInt32(min(mi, 255)) << 24
                for id in mesh.bodyIDs { flags[id] = 1 }
                for tri in mesh.triangles {
                    skinCorners.append(UInt32(base + tri.0) | comp)
                    skinCorners.append(UInt32(base + tri.1) | comp)
                    skinCorners.append(UInt32(base + tri.2) | comp)
                }
                for v in mesh.vertices {
                    let ids = [v.ids.0, v.ids.1, v.ids.2, v.ids.3]
                    for id in ids { flags[id] = 1 }
                    var g = SkinBindingGPU()
                    g.ids = SIMD4(UInt32(v.ids.0), UInt32(v.ids.1),
                                  UInt32(v.ids.2), UInt32(v.ids.3))
                    g.weights = v.weights
                    let n = length_squared(v.restNormal) > 1e-12
                        ? normalize(v.restNormal) : F3(0, 0, 1)
                    g.restNormal = SIMD4(n, 0)
                    if length_squared(v.restInv0) + length_squared(v.restInv1)
                        + length_squared(v.restInv2) > 1e-16 {
                        g.inv0 = SIMD4(v.restInv0, 0)
                        g.inv1 = SIMD4(v.restInv1, 0)
                        g.inv2 = SIMD4(v.restInv2, 0)
                    } else {
                        let x0 = scene.bodies[v.ids.0].position
                        let d0 = scene.bodies[v.ids.1].position - x0
                        let d1 = scene.bodies[v.ids.2].position - x0
                        let d2 = scene.bodies[v.ids.3].position - x0
                        let dm = simd_float3x3(columns: (d0, d1, d2))
                        let inv = abs(dm.determinant) > 1e-12
                            ? dm.inverse : matrix_identity_float3x3
                        g.inv0 = SIMD4(F3(inv.columns.0.x, inv.columns.1.x,
                                          inv.columns.2.x), 0)
                        g.inv1 = SIMD4(F3(inv.columns.0.y, inv.columns.1.y,
                                          inv.columns.2.y), 0)
                        g.inv2 = SIMD4(F3(inv.columns.0.z, inv.columns.1.z,
                                          inv.columns.2.z), 0)
                    }
                    bindings.append(g)
                }
                base += mesh.vertices.count
            }
            precondition(bindings.count == skinnedVertexCount)
            precondition(skinCorners.count == skinnedTriCount * 3)
            skinBindingBuf.contents().bindMemory(to: SkinBindingGPU.self,
                                                 capacity: bindings.count)
                .update(from: bindings, count: bindings.count)
            if !skinCorners.isEmpty {
                skinTriBuf.contents().bindMemory(to: UInt32.self,
                                                 capacity: skinCorners.count)
                    .update(from: skinCorners, count: skinCorners.count)
            }
        }

        // ---- Static topology coloring (computed ONCE; contacts do not
        // constrain colors — same-color contact pairs degrade to Jacobi,
        // which penalty-based contacts tolerate; stiff elements keep strict
        // Gauss-Seidel ordering). Replaces 20 GPU Jacobi rounds per frame.
        var adjacencySets = [Set<Int>](repeating: [], count: numBodies)
        func link2(_ a: Int, _ b: Int) {
            guard a >= 0, b >= 0,
                  scene.bodies[a].density > 0 || scene.bodies[b].density > 0 else { return }
            if scene.bodies[a].density > 0 && scene.bodies[b].density > 0 {
                adjacencySets[a].insert(b)
                adjacencySets[b].insert(a)
            }
        }
        // Inert joints (stiffness 0, e.g. collision-exclusion markers and
        // disabled drag slots) impose no solve ordering: the adjacency
        // kernel skips them, so the palette must too (a 4x4x4 soft block's
        // exclusion lattice alone pushed 3 colors to 9).
        for j in scene.joints where j.stiffnessLin != 0 || j.stiffnessAng != 0
                                 || j.motorTorque != 0 {
            link2(j.bodyA, j.bodyB)
        }
        for sp2 in scene.springs { link2(sp2.bodyA, sp2.bodyB) }
        // Tets keep strict Gauss-Seidel ordering: Jacobi-accepting them was
        // tried and FAILED the battery outright (volume preservation, block
        // stacking, bunny, tire drive — Neo-Hookean at mu 2.5-9k oscillates
        // under simultaneous neighbor updates).
        for t in scene.tets {
            let ids = [t.ids.0, t.ids.1, t.ids.2, t.ids.3]
            for i in 0..<4 { for j in (i + 1)..<4 { link2(ids[i], ids[j]) } }
        }
        // MEMBRANES are Jacobi-accepted (like contacts and bends): their
        // 3-cliques were the only reason cloth needed a third color over
        // the bipartite rod grid, and at mu ~300 with 20 iterations every
        // cloth gate and KE envelope holds without the ordering. Palette
        // 3 -> 2 = a third of the primal dispatch chain gone.
        // AVBD_GS_ELEMENTS=1 restores the old strict ordering for A/B.
        if ProcessInfo.processInfo.environment["AVBD_GS_ELEMENTS"] != nil {
            for t in scene.tris where t.mu > 0 {
                let ids = [t.ids.0, t.ids.1, t.ids.2]
                for i in 0..<3 { for j in (i + 1)..<3 { link2(ids[i], ids[j]) } }
            }
        }
        // Bends were ALWAYS excluded from coloring conflicts: their 4-vertex
        // hinge cliques over the 2-ring inflate the palette ~2x, and their
        // forces (kappa ~ 5e-4) are orders below membrane/rod scale.
        var staticColors = [Int](repeating: 0, count: numBodies)
        var maxColor = 0
        for v in 0..<numBodies where scene.bodies[v].density > 0 {
            var used: UInt64 = 0
            for nb in adjacencySets[v] where nb < v {
                let c = staticColors[nb]
                if c < 64 { used |= (1 << UInt64(c)) }
            }
            var c = 0
            while c < 63 && (used & (1 << UInt64(c))) != 0 { c += 1 }
            staticColors[v] = c
            maxColor = max(maxColor, c)
        }
        staticUsedColors = maxColor + 1
        // colorList/colorStart in final form, uploaded once
        var buckets = [[UInt32]](repeating: [], count: AVBD_MAX_COLORS)
        for v in 0..<numBodies where scene.bodies[v].density > 0 {
            buckets[staticColors[v]].append(UInt32(v))
        }
        var clist: [UInt32] = []
        var cstart: [UInt32] = []
        for c in 0..<AVBD_MAX_COLORS {
            cstart.append(UInt32(clist.count))
            clist.append(contentsOf: buckets[c])
        }
        cstart.append(UInt32(clist.count))
        cstart.append(UInt32(staticUsedColors))
        colorStart.contents().bindMemory(to: UInt32.self, capacity: cstart.count)
            .update(from: cstart, count: cstart.count)
        if !clist.isEmpty {
            colorList.contents().bindMemory(to: UInt32.self, capacity: clist.count)
                .update(from: clist, count: clist.count)
        }
        // indirect dispatch args per color, fixed for the scene's lifetime
        let cargs = colorArgs.contents().bindMemory(to: UInt32.self,
                                                    capacity: AVBD_MAX_COLORS * 3)
        for c in 0..<AVBD_MAX_COLORS {
            cargs[c * 3 + 0] = UInt32((buckets[c].count + 63) / 64)
            cargs[c * 3 + 1] = 1
            cargs[c * 3 + 2] = 1
        }
        for i in 0..<numBodies { colA[i] = UInt32(staticColors[i]) }
        lastColorCounts = buckets.map { $0.count }
        lastMaxColorUsed = staticUsedColors - 1

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
        let hashedRigid = hashed.filter { !scene.bodies[Int($0)].isParticle }
        params.numHashedRigid = UInt32(hashedRigid.count)
        hashedRigidIdx.contents().bindMemory(to: UInt32.self, capacity: max(1, hashedRigid.count))
            .update(from: hashedRigid.isEmpty ? [0] : hashedRigid, count: max(1, hashedRigid.count))
        // Anti-tunneling: cap speed so nothing crosses the thinnest static
        // geometry in one frame. Heuristic: a couple of cells per step.
        params.maxSpeed = max(30, 1.5 * params.cellSize / settings.dt * 0.5)
    }

    // NOTE on indirect command buffers: the obvious dispatch-storm fix
    // (pre-encode iterations x colors of solve commands once, replay per
    // frame) is blocked on macOS — CPU-side compute-ICB encoding
    // (indirectComputeCommand(at:)) is iOS-only, and GPU-side encoding
    // cannot switch pipelines per command. Attacked instead by shrinking
    // the static palette (bends/contacts Jacobi-accepted).

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
        params.particleDamping = settings.particleDamping
        params.frame = UInt32(truncatingIfNeeded: frameIndex)

        if let env = ProcessInfo.processInfo.environment["AVBD_ROD_DECAY"],
           let v = Float(env) {
            params.rodDecayPow = v
        }
        if let env = ProcessInfo.processInfo.environment["AVBD_PDAMP"],
           let v = Float(env) {
            params.particleDamping = v
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
    /// - maxStretch: worst stiff-spring EXTENSION max(len/rest - 1, 0)
    ///   (stiffness >= 1000 or hard selects structural edges). Compression
    ///   is buckling — folds — and intentionally free for tension-only rods.
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
                let st = max(distance(wa, wb) / rest - 1, 0)
                if st > maxStretch {
                    maxStretch = st
                    lastWorstSpring = (a, b)
                    lastWorstSpringIdx = i
                }
            }
        }
        return (minGap == .greatestFiniteMagnitude ? 0 : minGap, maxStretch)
    }

    /// Endpoints of the worst stiff spring from the last debugClothMetrics call.
    public private(set) var lastWorstSpring: (Int, Int) = (-1, -1)
    public private(set) var lastWorstSpringIdx: Int = -1

    /// Dual state of a spring/rod: (lambda, penalty, C0, rest).
    public func debugSpringDual(_ i: Int) -> (Float, Float, Float, Float) {
        sync()
        let sp = springs.contents().bindMemory(to: SpringGPU.self, capacity: max(1, numSprings))
        return (sp[i].dual.x, sp[i].dual.y, sp[i].dual.z, sp[i].rB.w)
    }

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
            blit.fill(buffer: changedFlag, range: 0..<changedFlag.length, value: 0)
            blit.fill(buffer: cellCount, range: 0..<(gridHashSize * 4), value: 0)
            blit.fill(buffer: cellRigid, range: 0..<(gridHashSize * 4), value: 0)
            if numTris > 0 {
                // OGC conservative bounds: reset to "far" (0x7F7F7F7F ~ 3e38)
                blit.fill(buffer: boundsBuf, range: 0..<(numBodies * 4), value: 0x7F)
            }
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
        if profiling { stage("bp-count") }
        dispatch1D(enc, "bp_count", Int(P.numHashed)) { e in
            e.setBuffer(self.posLin, offset: 0, index: 0)
            e.setBuffer(self.hashedIdx, offset: 0, index: 1)
            e.setBuffer(self.cellCount, offset: 0, index: 2)
            e.setBuffer(self.bodyCellSlot, offset: 0, index: 3)
            e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 4)
            e.setBuffer(self.shape, offset: 0, index: 5)
            e.setBuffer(self.cellRigid, offset: 0, index: 6)
        }
        if profiling { stage("bp-scan") }
        encodeScan(enc, input: cellCount, output: cellStart, count: gridHashSize)
        if profiling { stage("bp-scatter") }
        dispatch1D(enc, "bp_scatter", Int(P.numHashed)) { e in
            e.setBuffer(self.hashedIdx, offset: 0, index: 0)
            e.setBuffer(self.bodyCellSlot, offset: 0, index: 1)
            e.setBuffer(self.cellStart, offset: 0, index: 2)
            e.setBuffer(self.cellBodies, offset: 0, index: 3)
            e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 4)
        }
        if profiling { stage("bp-pairs") }
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
            e.setBuffer(self.clothGroupBuf, offset: 0, index: 12)
            e.setBuffer(self.cellRigid, offset: 0, index: 13)
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
            e.setBuffer(self.clothGroupBuf, offset: 0, index: 8)
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
            e.setBuffer(self.velLin, offset: 0, index: 14)
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
            stage("el-bin")
            dispatch1D(enc, "el_count", numTris + Int(P.numEdges)) { e in
                e.setBuffer(self.posLin, offset: 0, index: 0)
                e.setBuffer(self.trisBuf, offset: 0, index: 1)
                e.setBuffer(self.edgesBuf, offset: 0, index: 2)
                e.setBuffer(self.shape, offset: 0, index: 3)
                e.setBuffer(self.elemCellCount, offset: 0, index: 4)
                e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 5)
            }
            encodeScan(enc, input: elemCellCount, output: elemCellStart, count: elemHashSize)
            var ehs = UInt32(elemHashSize)
            dispatch1D(enc, "adj_copy_cursor", elemHashSize) { e in
                e.setBuffer(self.elemCellStart, offset: 0, index: 0)
                e.setBuffer(self.elemSlot, offset: 0, index: 1)
                e.setBytes(&ehs, length: 4, index: 2)
            }
            dispatch1D(enc, "el_scatter", numTris + Int(P.numEdges)) { e in
                e.setBuffer(self.posLin, offset: 0, index: 0)
                e.setBuffer(self.trisBuf, offset: 0, index: 1)
                e.setBuffer(self.edgesBuf, offset: 0, index: 2)
                e.setBuffer(self.shape, offset: 0, index: 3)
                e.setBuffer(self.elemSlot, offset: 0, index: 4)
                e.setBuffer(self.elemCells, offset: 0, index: 5)
                e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 6)
            }
            stage("vt-emit")
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
                e.setBuffer(self.vtTrackBuf, offset: 0, index: 19)
                e.setBuffer(self.triAdjBuf, offset: 0, index: 20)
                e.setBuffer(self.boundsBuf, offset: 0, index: 21)
                e.setBuffer(self.nbr2Start, offset: 0, index: 22)
                e.setBuffer(self.nbr2Count, offset: 0, index: 23)
                e.setBuffer(self.nbr2List, offset: 0, index: 24)
            }
            }
            stage("ee-emit")
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
                e.setBuffer(self.eeTrackBuf, offset: 0, index: 18)
                e.setBuffer(self.vertEdgeStart, offset: 0, index: 19)
                e.setBuffer(self.vertEdgeCount, offset: 0, index: 20)
                e.setBuffer(self.vertEdgeList, offset: 0, index: 21)
                e.setBuffer(self.boundsBuf, offset: 0, index: 22)
                e.setBuffer(self.nbr2Start, offset: 0, index: 23)
                e.setBuffer(self.nbr2Count, offset: 0, index: 24)
                e.setBuffer(self.nbr2List, offset: 0, index: 25)
            }
            }
            stage("rt-emit")
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
                e.setBuffer(self.hashedRigidIdx, offset: 0, index: 20)
            }
            }
            stage("softmap")
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
            e.setBuffer(self.ogcPrevBuf, offset: 0, index: 10)
            e.setBuffer(self.boundsBuf, offset: 0, index: 11)
            var doOgc: UInt32 = self.numTris > 0 ? 1 : 0
            e.setBytes(&doOgc, length: 4, index: 12)
            e.setBuffer(self.shape, offset: 0, index: 13)
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
            e.setBuffer(self.membranes, offset: 0, index: 9)
            e.setBuffer(self.bends, offset: 0, index: 10)
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
            e.setBuffer(self.membranes, offset: 0, index: 10)
            e.setBuffer(self.bends, offset: 0, index: 11)
        }

        // Coloring is scene-adaptive:
        // - Soft scenes (cloth/tets): STATIC topology palette from init,
        //   Newton-style — contacts never constrain colors (same-color
        //   contact pairs degrade to Jacobi, which gram-scale penalty
        //   contacts tolerate; measured: all cloth gates green, 1.3-1.7 ms
        //   of per-frame recoloring gone, palette 14-20 -> 3).
        // - Pure rigid scenes: the original contact-aware per-frame GPU
        //   coloring — box stacks and gear trains genuinely need strict
        //   Gauss-Seidel contact ordering (stack/gearclock fail without).
        if usesDynamicColoring {
            stage("coloring")
            var src = colorsA, dst = colorsB
            for pass in 0..<20 {
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
                    e.setBuffer(self.membranes, offset: 0, index: 13)
                    e.setBuffer(self.bends, offset: 0, index: 14)
                    var pi = UInt32(pass)
                    e.setBytes(&pi, length: 4, index: 15)
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
            if finalColors !== colorsA {
                dynColorSrc = finalColors
            }
        }
        stage("solver-iterations")
        let persistPSO = ps("solve_persistent")
        let multiPSO = ps("solve_persistent_multi")
        // Multi-threadgroup persistent path: whole solve in one dispatch of
        // a few co-resident threadgroups (device-scope spin barrier). Covers
        // small AND mid scenes; huge scenes amortize per-color dispatches.
        // EXPERIMENTAL, opt-in: deadlocked on first contact with reality —
        // Metal guarantees no forward progress between threadgroups, and the
        // arrive-and-spin barrier wedged the queue. Kept for further study.
        let multiOK = ProcessInfo.processInfo.environment["AVBD_MULTI"] != nil
        let noPersist = ProcessInfo.processInfo.environment["AVBD_NO_PERSIST"] != nil
        if multiOK && numBodies <= 4096 {
            let tgW = min(256, multiPSO.maxTotalThreadsPerThreadgroup)
            var ntg = UInt32(min(8, max(1, (numBodies + tgW - 1) / tgW + 1)))
            enc.setComputePipelineState(multiPSO)
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
            enc.setBuffer(membranes, offset: 0, index: 20)
            enc.setBuffer(bends, offset: 0, index: 21)
            enc.setBytes(&ntg, length: 4, index: 22)
            enc.setBuffer(boundsBuf, offset: 0, index: 26)
            enc.setBuffer(ogcPrevBuf, offset: 0, index: 27)
            enc.setBuffer(counters, offset: 0, index: 28)
            enc.dispatchThreadgroups(MTLSize(width: Int(ntg), height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tgW, height: 1, depth: 1))
        } else if !noPersist && numBodies <= persistPSO.maxTotalThreadsPerThreadgroup {
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
            enc.setBuffer(membranes, offset: 0, index: 20)
            enc.setBuffer(bends, offset: 0, index: 21)
            enc.setBuffer(boundsBuf, offset: 0, index: 26)
            enc.setBuffer(ogcPrevBuf, offset: 0, index: 27)
            enc.setBuffer(counters, offset: 0, index: 28)
            let w = persistPSO.threadExecutionWidth
            let tg = min(persistPSO.maxTotalThreadsPerThreadgroup,
                         ((max(numBodies, 64) + w - 1) / w) * w)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        } else {
        let colorBound = usesDynamicColoring
            ? min(AVBD_MAX_COLORS, max(lastMaxColorUsed + 2, 4))
            : staticUsedColors
        // Rigid scenes use the 8-lane cooperative split primal too: dense
        // piles average 12-27 manifolds/body and the one-thread-per-body
        // kernel was latency-bound walking them serially (boxpile x12
        // solve-primal 9.5 ms). AVBD_NO_RSPLIT restores the old kernel
        // (the x8 indirect args early-out harmlessly).
        let noRSplit = ProcessInfo.processInfo.environment["AVBD_NO_RSPLIT"] != nil
        let primalPSO = (usesDynamicColoring && noRSplit) ? ps("primal_solve")
                                                          : ps("primal_particles_split")
        // static palettes have fixed per-color counts: dispatch directly
        // (8 lanes per body for the split kernel)
        let splitSizes: [Int] = usesDynamicColoring ? []
            : (0..<staticUsedColors).map { lastColorCounts[$0] * 8 }
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
                enc.setBuffer(membranes, offset: 0, index: 20)
                enc.setBuffer(bends, offset: 0, index: 21)
                enc.setBuffer(self.boundsBuf, offset: 0, index: 22)
                enc.setBuffer(self.ogcPrevBuf, offset: 0, index: 23)
                enc.setBuffer(self.counters, offset: 0, index: 24)
            }
            _ = it
            if profiling { stage("solve-primal") ; enc.setComputePipelineState(primalPSO)
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
                enc.setBuffer(membranes, offset: 0, index: 20)
                enc.setBuffer(bends, offset: 0, index: 21)
                enc.setBuffer(self.boundsBuf, offset: 0, index: 22)
                enc.setBuffer(self.ogcPrevBuf, offset: 0, index: 23)
                enc.setBuffer(self.counters, offset: 0, index: 24)
            }
            for c in 0..<colorBound {
                var cIdx = UInt32(c)
                enc.setBytes(&cIdx, length: 4, index: 15)
                if usesDynamicColoring {
                    enc.dispatchThreadgroups(indirectBuffer: colorArgs,
                                             indirectBufferOffset: c * 12,
                                             threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                } else {
                    enc.dispatchThreadgroups(
                        MTLSize(width: (splitSizes[c] + 63) / 64, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                }
            }
            // tail: bodies in colors >= colorBound (the async bound can be
            // stale; skipped bodies would fly ballistic — see kernel
            // comment). Static palettes are exact: no tail needed.
            if usesDynamicColoring {
                let p = ps("primal_tail")
                enc.setComputePipelineState(p)
                var cb = UInt32(colorBound)
                enc.setBytes(&cb, length: 4, index: 15)
                enc.dispatchThreadgroups(MTLSize(width: (numBodies + 63) / 64, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                enc.setComputePipelineState(primalPSO)
            }

            if profiling { stage("solve-dual") }
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
            // OGC conditional refresh (paper Alg 3): a one-thread kernel
            // turns the exceed counter into indirect dispatch args, so the
            // refresh runs ONLY in iterations where >1% of particles were
            // bound-limited — settled scenes pay an empty dispatch
            if numParticles > 0 && it + 1 < settings.iterations {
                dispatch1D(enc, "ogc_refresh_args", 1) { e in
                    e.setBuffer(self.counters, offset: 0, index: 0)
                    e.setBuffer(self.ogcArgsBuf, offset: 0, index: 1)
                    e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 2)
                }
                let pr = ps("ogc_bounds_refresh")
                enc.setComputePipelineState(pr)
                enc.setBuffer(posLin, offset: 0, index: 0)
                enc.setBuffer(particleIdxBuf, offset: 0, index: 1)
                enc.setBuffer(trisBuf, offset: 0, index: 2)
                enc.setBuffer(edgesBuf, offset: 0, index: 3)
                enc.setBuffer(vtTrackBuf, offset: 0, index: 4)
                enc.setBuffer(eeTrackBuf, offset: 0, index: 5)
                enc.setBuffer(vertEdgeStart, offset: 0, index: 6)
                enc.setBuffer(vertEdgeCount, offset: 0, index: 7)
                enc.setBuffer(vertEdgeList, offset: 0, index: 8)
                enc.setBuffer(boundsBuf, offset: 0, index: 9)
                enc.setBuffer(ogcPrevBuf, offset: 0, index: 10)
                enc.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 11)
                enc.setBuffer(nbr2Start, offset: 0, index: 12)
                enc.setBuffer(nbr2Count, offset: 0, index: 13)
                enc.setBuffer(nbr2List, offset: 0, index: 14)
                enc.dispatchThreadgroups(indirectBuffer: ogcArgsBuf,
                                         indirectBufferOffset: 0,
                                         threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
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
            e.setBuffer(self.shape, offset: 0, index: 8)
        }
        enc.endEncoding()
        var visc = settings.clothViscosity
        if let env = ProcessInfo.processInfo.environment["AVBD_VISC"],
           let v = Float(env) { visc = v }
        if numTris > 0 && visc > 0 {
            // scratch copy for the gather (inertLin is dead after the solve
            // and rewritten by next frame's warmstart)
            if let blit = cmd1.makeBlitCommandEncoder() {
                blit.copy(from: velLin, sourceOffset: 0,
                          to: inertLin, destinationOffset: 0, size: numBodies * 16)
                blit.endEncoding()
            }
            if let e2 = cmd1.makeComputeCommandEncoder() {
                let p = ps("smooth_particle_velocities")
                e2.setComputePipelineState(p)
                e2.setBuffer(velLin, offset: 0, index: 0)
                e2.setBuffer(inertLin, offset: 0, index: 1)
                e2.setBuffer(particleIdxBuf, offset: 0, index: 2)
                e2.setBuffer(nbrStart, offset: 0, index: 3)
                e2.setBuffer(nbrCount, offset: 0, index: 4)
                e2.setBuffer(nbrList, offset: 0, index: 5)
                e2.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 6)
                e2.setBytes(&visc, length: 4, index: 7)
                e2.dispatchThreadgroups(MTLSize(width: (max(1, numParticles) + 63) / 64,
                                                height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                e2.endEncoding()
            }
        }
        if let src2 = dynColorSrc, let blit = cmd1.makeBlitCommandEncoder() {
            blit.copy(from: src2, sourceOffset: 0,
                      to: colorsA, destinationOffset: 0, size: numBodies * 4)
            blit.endEncoding()
            dynColorSrc = nil
        }
        let dynColors = usesDynamicColoring
        let readStats = { [weak self] in
            guard let self else { return }
            let ctr = self.counters.contents()
                .bindMemory(to: UInt32.self, capacity: GPUCounters.total)
            let pairs = Int(ctr[GPUCounters.pairs])
            let softN = min(Int(ctr[GPUCounters.soft]), self.maxSoft)
            // Saturated capacity = contacts silently dropped (random subset,
            // GPU arrival order) = non-deterministic sink-through. Never hide.
            if pairs >= self.maxPairs && !self.warnedPairSaturation {
                self.warnedPairSaturation = true
                print("AVBD WARNING: pair list saturated (\(pairs)/\(self.maxPairs)) — contacts dropped; raise maxPairsPerBody")
            }
            if Int(ctr[GPUCounters.soft]) >= self.maxSoft && !self.warnedSoftSaturation {
                self.warnedSoftSaturation = true
                print("AVBD WARNING: soft contact list saturated (\(self.maxSoft)) — contacts dropped")
            }
            self.statsLock.lock()
            self.lastNumPairs = pairs
            self.lastNumSoft = softN
            if dynColors {
                var counts: [Int] = []
                var maxUsed = -1
                for c in 0..<AVBD_MAX_COLORS {
                    let n = Int(ctr[GPUCounters.colorBase + c])
                    counts.append(n)
                    if n > 0 { maxUsed = c }
                }
                self.lastColorCounts = counts
                self.lastMaxColorUsed = maxUsed
            }
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
        // velocity motors: advance the bounded-torque servo target (wrapped;
        // the motor constraint wraps its error the same way)
        if !rateMotors.isEmpty {
            let jp = joints.contents().bindMemory(to: JointGPU.self,
                                                  capacity: max(1, numJoints))
            for (ji, rate) in rateMotors {
                var t = jp[ji].motor.x + rate * settings.dt
                t -= (2 * Float.pi) * (t / (2 * .pi)).rounded()
                jp[ji].motor.x = t
            }
        }
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
            if st[i] != 0 {
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
        enc.setBuffer(surfacedFlags, offset: 0, index: 8)
        enc.dispatchThreadgroups(MTLSize(width: (numBodies + 255) / 256, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        if surfVertCount > 0 {
            let pf = ps("soft_face_normals")
            enc.setComputePipelineState(pf)
            enc.setBuffer(posLin, offset: 0, index: 0)
            enc.setBuffer(surfTriBuf, offset: 0, index: 1)
            enc.setBuffer(faceNormalsBuf, offset: 0, index: 2)
            var nt = UInt32(surfaceTriCount)
            enc.setBytes(&nt, length: 4, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: (surfaceTriCount + 255) / 256, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

            let pn = ps("soft_normals")
            enc.setComputePipelineState(pn)
            enc.setBuffer(posLin, offset: 0, index: 0)
            enc.setBuffer(surfVertsBuf, offset: 0, index: 1)
            enc.setBuffer(surfVtStart, offset: 0, index: 2)
            enc.setBuffer(surfVtCount, offset: 0, index: 3)
            enc.setBuffer(surfVtList, offset: 0, index: 4)
            enc.setBuffer(surfTriBuf, offset: 0, index: 5)
            enc.setBuffer(softNormalsBuf, offset: 0, index: 6)
            var nv = UInt32(surfVertCount)
            enc.setBytes(&nv, length: 4, index: 7)
            enc.setBuffer(faceNormalsBuf, offset: 0, index: 8)
            enc.setBuffer(shape, offset: 0, index: 9)
            enc.setBuffer(clothVertFlag, offset: 0, index: 10)
            var cs = settings.clothRenderScale
            enc.setBytes(&cs, length: 4, index: 11)
            enc.dispatchThreadgroups(MTLSize(width: (surfVertCount + 255) / 256, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        if skinnedVertexCount > 0 {
            let ps = ps("skin_deform")
            enc.setComputePipelineState(ps)
            enc.setBuffer(posLin, offset: 0, index: 0)
            enc.setBuffer(skinBindingBuf, offset: 0, index: 1)
            enc.setBuffer(skinVertexBuf, offset: 0, index: 2)
            var nv = UInt32(skinnedVertexCount)
            enc.setBytes(&nv, length: 4, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: (skinnedVertexCount + 255) / 256,
                                             height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        enc.endEncoding()
    }

    /// Soft-surface render data (cloth + tet boundary meshes), nil if none.
    /// Corners pack body-id | side<<23 | component<<24; thin sheets carry
    /// front/back layers + hem rims, offset by the thickness in normals.w.
    /// Positions index the live posLin buffer; normals are refreshed by
    /// encodeBuildInstances.
    public var renderSurface: (tris: MTLBuffer, triCount: Int,
                               positions: MTLBuffer, normals: MTLBuffer)? {
        guard renderTriCount > 0 else { return nil }
        return (renderTriBuf, renderTriCount, posLin, softNormalsBuf)
    }

    /// Skinned visual mesh render data, nil if the scene has no embedded
    /// visual surfaces. Vertices are refreshed by encodeBuildInstances.
    public var renderSkinnedSurface: (tris: MTLBuffer, triCount: Int,
                                      vertices: MTLBuffer)? {
        guard skinnedTriCount > 0 else { return nil }
        return (skinTriBuf, skinnedTriCount, skinVertexBuf)
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
