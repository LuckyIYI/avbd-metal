import Foundation
import Metal
import QuartzCore
import simd
import SimCore

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
    /// Collision-safe displacement limiter for deformable surface V-T/E-E
    /// motion. Automatic selection uses Planar-DAT for shell-only scenes and
    /// opt-in volumetric self-collision. Ordinary closed tet bodies retain the
    /// lower-overhead isotropic OGC bound, including mixed shell + tet scenes.
    /// The explicit modes are stable in-process A/B paths.
    public enum SurfaceTruncationMode: UInt32, Sendable {
        case isotropicDAT = 1
        case planarDAT = 2
    }

    /// How a scene chooses its concrete surface limiter. This is separate
    /// from `SurfaceTruncationMode` so adding automatic policy does not add a
    /// case to the established public enum and break exhaustive client
    /// switches.
    public enum SurfaceTruncationSelection: Sendable, Equatable {
        case automatic
        case explicit(SurfaceTruncationMode)
    }

    private static let jointMotorModeMask: UInt32 = 3 << 4
    private static let jointMotorModeImplicitPositionPD: UInt32 = 1 << 4
    private static let jointMotorModeExplicitTorquePD: UInt32 = 2 << 4
    private static let jointMotorModeVelocity: UInt32 = 3 << 4
    private static let orderedColoringBodyLimit = 1_024

    public let device: MTLDevice
    let queue: MTLCommandQueue

    public var settings = SimSettings()

    // Capacities
    let numBodies: Int
    let numColliders: Int
    let numConvexHullVertices: Int
    let numJoints: Int
    let numSprings: Int
    let numTets: Int
    let hasAuthoredSurfaceTriangles: Bool
    let hasVolumetricSelfCollision: Bool
    let maxPairs: Int
    let mapCapacity: Int
    let gridHashSize: Int
    // Cloth elements
    let numTris: Int
    public let tetBoundaryTris: [(Int, Int, Int)]
    var numEdges: Int = 0
    var numPlanarDATEdges: Int = 0
    var numParticles: Int = 0
    let maxSoft: Int
    let softMapCapacity: Int
    let maxIsotropicSoft: Int
    let isotropicSoftMapCapacity: Int
    let elemHashSize: Int
    let maxPlanarDATPairs: Int
    var isotropicElemCellSize: Float = 1
    var planarDATElemCellSize: Float = 1

    public var surfaceTruncationSelection: SurfaceTruncationSelection = .automatic

    /// Optional signed-volume limiter for tetrahedra. This is deliberately
    /// independent of surface Planar-DAT: Newton applies Planar-DAT to
    /// particle-surface V-T/E-E motion but does not enable the paper's
    /// underspecified tet-inversion extension. Keeping this opt-in prevents a
    /// mixed shell scene from silently changing its volumetric material path.
    public var tetInversionPreventionEnabled = false

    /// Backward-compatible explicit mode API. Assigning this property opts
    /// out of automatic selection; reading it returns the effective mode.
    public var surfaceTruncationMode: SurfaceTruncationMode {
        get { effectiveSurfaceTruncationMode }
        set { surfaceTruncationSelection = .explicit(newValue) }
    }

    /// Concrete surface limiter selected for the current scene. Selection is
    /// structural rather than demo-name based, so imported scenes receive the
    /// same policy as built-ins with equivalent topology.
    public var effectiveSurfaceTruncationMode: SurfaceTruncationMode {
        switch surfaceTruncationSelection {
        case .automatic:
            return hasVolumetricSelfCollision
                || (hasAuthoredSurfaceTriangles && numTets == 0)
                ? .planarDAT : .isotropicDAT
        case .explicit(let mode):
            return mode
        }
    }

    enum PlanarDATPassSite: Equatable {
        case predictor
        case color(iteration: Int, color: Int)
    }

    var params = SimParamsGPU()

    // Body SoA buffers
    var posLin, posAng, initLin, initAng, inertLin, inertAng: MTLBuffer
    var velLin, velAng, prevVelLin: MTLBuffer
    var props, shape: MTLBuffer
    var gravityScale: MTLBuffer
    var shapeType: MTLBuffer       // 0 box, 1 sphere, 2 torus
    var spinVel: MTLBuffer         // angular velocity of kinematic spinners
    // Collision geometry is independent of body inertia. Every legacy body
    // contributes one identity-local collider; imported links may contribute
    // zero or many offset primitives.
    var colliderOwner, colliderShape, colliderShapeType, colliderGroup: MTLBuffer
    var colliderSharedCollision: MTLBuffer
    var colliderLocalPosition, colliderLocalRotation: MTLBuffer
    var colliderRenderColor: MTLBuffer
    var colliderFriction: MTLBuffer
    var colliderHullRange, convexHullVertices: MTLBuffer

    // Constraints
    var joints: MTLBuffer
    /// Adaptive joint penalties are solver state, not scene state. Keep the
    /// authored first-frame values so an in-place environment reset can be
    /// bitwise equivalent to a fresh episode instead of inheriting the
    /// previous fall's augmented-Lagrangian conditioning.
    private var initialJointPenaltyLin: [SIMD4<Float>] = []
    private var initialJointPenaltyAng: [SIMD4<Float>] = []
    var springs: MTLBuffer
    var manifolds: MTLBuffer       // current frame
    var prevManifolds: MTLBuffer   // previous frame (swapped)

    // Broadphase
    var tets: MTLBuffer
    var hashedIdx, globalIdx, hashedRigidIdx: MTLBuffer
    var cellCount, cellStart, cellBodies, cellRigid: MTLBuffer
    var bodyCellSlot: MTLBuffer
    /// Per-producer candidate counts and exclusive offsets. Pair emission is
    /// count/scan/scatter instead of atomic append so manifold identity and
    /// contact accumulation order are reproducible across Metal schedules.
    var pairCount, pairStart: MTLBuffer
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
    var renderBodyIdxBuf: MTLBuffer
    public private(set) var renderRigidBodyCount: Int = 0
    var clothGroupBuf: MTLBuffer
    // Visual-only tetrahedral skinning buffers. These draw arbitrary mesh
    // surfaces embedded in tets; collisions still use `trisBuf`.
    let skinnedVertexCount: Int
    let skinnedTriCount: Int
    var skinBindingBuf, skinVertexBuf, skinTriBuf: MTLBuffer
    // Visual-only rigid CAD meshes are expanded to triangle corners once.
    // Each corner carries its body id and is transformed from live solver
    // state in the renderer; it never enters collision or solver topology.
    let rigidMeshVertexCount: Int
    var rigidMeshVertexBuf: MTLBuffer
    // Voronoi temporal tracking: per-vertex / per-edge persistent closest-
    // element candidate sets, plus the topology they propagate through.
    var vtTrackBuf, eeTrackBuf, triAdjBuf: MTLBuffer
    var clothVertFlag: MTLBuffer
    /// Per-particle policy for collision with non-adjacent primitives in the
    /// same connected deformable component. Thin shells enable this by
    /// construction; tet bodies opt in through SceneTet.
    var softSelfCollisionFlag: MTLBuffer
    var boundsBuf, ogcPrevBuf, ogcArgsBuf: MTLBuffer
    // Full compact V-T/E-E safety neighborhood used by Planar-DAT. Contact
    // emission consumes the same stream, so safety and force generation
    // cannot silently disagree about which nearby primitives exist.
    var planarDATPairsBuf, planarDATPairCountsBuf: MTLBuffer
    var planarDATTBuf, planarDATArgsBuf: MTLBuffer
    // Accepted-pose Planar pairs indexed by participating particle. The
    // fixed pair set is reused after every VBD color; this CSR prevents each
    // color from rescanning the complete global pair stream. Its pair-index
    // list is intentionally a placeholder unless the initial scene selects
    // the high-color incidence path; low-color/global/isotropic scenes must
    // not pay for a second full Planar pair stream.
    var planarDATBodyCountBuf, planarDATBodyStartBuf: MTLBuffer
    var planarDATBodyCursorBuf, planarDATBodyPairsBuf: MTLBuffer
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
    var degrees, adjStart, adjCursor, adjList, adjNeighbor: MTLBuffer
    var colorsA, colorsB, bodySlot, colorStart, colorList: MTLBuffer
    var changedFlag: MTLBuffer

    // Control
    var counters: MTLBuffer
    /// Frame-owned readback slots. The solver permits at most two in-flight
    /// submissions, so parity selects a slot only after its previous owner
    /// has been retired by throttling.
    let counterReadbacks: [MTLBuffer]
    var dispatchArgs: MTLBuffer    // 9 uints (pairs / forces / diag)
    var colorArgs: MTLBuffer       // MAX_COLORS * 3 uints
    var scanBlockSums: MTLBuffer
    var scanTotal: MTLBuffer
    var diag: MTLBuffer

    // Pipelines
    var pso: [String: MTLComputePipelineState] = [:]

    /// Every compute entry point reachable through this solver, including
    /// optional cloth, rendering, diagnostics, and robotics paths. Validate
    /// the complete contract at initialization so a Swift/Metal source drift
    /// is reported as a throwing construction error instead of reaching the
    /// internal `ps(_:)` invariant and terminating the process later.
    static let requiredKernelNames: Set<String> = [
        "adj_clear_degrees",
        "adj_copy_cursor",
        "adj_count",
        "adj_extract_neighbors",
        "adj_scatter",
        "adj_sort",
        "bp_count",
        "bp_count_pairs_deterministic",
        "bp_emit_pairs_deterministic",
        "bp_finalize_deterministic_pairs",
        "bp_scatter",
        "bp_sort_cells",
        "build_instances",
        "color_count",
        "color_iterate",
        "color_repair_greedy",
        "color_scan",
        "color_scatter",
        "color_validate",
        "dat_apply",
        "dat_apply_color",
        "dat_apply_tet_inversion",
        "dat_build_ee_pairs",
        "dat_build_vt_pairs",
        "dat_clear_element_grid",
        "dat_clear_pair_counts",
        "dat_emit_contacts",
        "dat_finalize_pairs",
        "dat_incidence_count",
        "dat_incidence_scatter",
        "dat_reanchor",
        "dat_reduce",
        "dat_reduce_incident_color",
        "dat_reduce_tet_inversion",
        "dat_restore_failed_surface",
        "diag_clear",
        "diag_error",
        "dual_all",
        "ee_emit",
        "el_count",
        "el_scatter",
        "finalize_velocities",
        "manifold_solver_pack",
        "np_collide",
        "ogc_bounds_refresh",
        "ogc_refresh_args",
        "pm_clear",
        "pm_insert",
        "primal_particles_split",
        "primal_rigid_split",
        "primal_solve",
        "pusht_obs",
        "rt_emit",
        "scan_add_offsets",
        "scan_block_sums",
        "scan_blocks",
        "skin_deform",
        "smooth_particle_velocities",
        "soft_face_normals",
        "soft_finalize",
        "soft_normals",
        "softmap_clear",
        "softmap_insert",
        "solve_persistent",
        "solve_persistent_multi",
        "vt_emit",
        "warmstart_bodies",
        "warmstart_joints",
    ]

    // Cached per-frame color counts (read back once per step)
    public internal(set) var lastColorCounts: [Int] = []
    public private(set) var lastNumPairs: Int = 0
    public private(set) var lastPairCandidates: Int = 0
    public private(set) var lastNumSoft: Int = 0
    public private(set) var lastSoftCandidates: Int = 0
    public private(set) var lastPlanarDATPairs: Int = 0
    public private(set) var lastPlanarDATVertexTrianglePairs: Int = 0
    public private(set) var lastPlanarDATEdgeEdgePairs: Int = 0
    public private(set) var lastPlanarDATTruncations: Int = 0
    public private(set) var lastPlanarDATVertexTriangleDegeneracies: Int = 0
    public private(set) var lastPlanarDATEdgeEdgeDegeneracies: Int = 0
    public private(set) var lastPlanarDATNonfiniteValues: Int = 0
    public private(set) var lastPlanarDATTetDegeneracies: Int = 0
    // Start pessimistic so the first frame covers every possible color.
    public internal(set) var lastMaxColorUsed: Int = AVBD_MAX_COLORS - 1
    public private(set) var frameIndex: Int = 0
    /// Legacy position-servo targets advanced each step as `(joint, rad/s)`.
    private var rateMotors: [(Int, Float)] = []
    public convenience init(scene: PhysicsScene, device: MTLDevice? = nil,
                            maxPairsPerBody: Int = 16) throws {
        try self.init(
            scene: scene, device: device, maxPairsPerBody: maxPairsPerBody,
            optionalPlanarDATBodyPairAllocator: nil)
    }

    init(
        scene: PhysicsScene,
        device: MTLDevice? = nil,
        maxPairsPerBody: Int = 16,
        optionalPlanarDATBodyPairAllocator:
            ((MTLDevice, Int) -> MTLBuffer?)?
    ) throws {
        precondition(scene.settings.collisionMargin >= 0
            && scene.settings.collisionMargin.isFinite,
            "collision margin must be finite and nonnegative")
        precondition(scene.settings.rigidLinearDamping >= 0
            && scene.settings.rigidLinearDamping.isFinite
            && scene.settings.rigidAngularDamping >= 0
            && scene.settings.rigidAngularDamping.isFinite,
            "rigid damping must be finite and nonnegative")
        guard let dev = device ?? MTLCreateSystemDefaultDevice() else {
            throw AVBDError.noDevice
        }
        self.device = dev
        guard let q = dev.makeCommandQueue() else { throw AVBDError.noDevice }
        self.queue = q

        self.settings = scene.settings
        self.numBodies = scene.bodies.count
        self.numColliders = scene.colliders.count
        self.numConvexHullVertices = scene.colliders.reduce(0) {
            $0 + $1.convexHullVertices.count
        }
        self.numJoints = scene.joints.count
        self.numSprings = scene.springs.count
        self.numTets = scene.tets.count
        self.hasAuthoredSurfaceTriangles = !scene.tris.isEmpty
        self.hasVolumetricSelfCollision = scene.tets.contains {
            $0.selfCollisionEnabled
        }
        let enabledColliderCount = scene.colliders.lazy.filter(\.collisionEnabled).count
        // Floor of 4096: small scenes are cheap (688B/slot) but elongated
        // shapes (jenga blocks) legitimately exceed 16 candidates/body, and
        // a saturated pair list silently drops a scheduling-random subset of
        // pairs each frame — bodies lose ground support and sink/freefall.
        // Visual-only/imported proxy colliders do not enter broad phase and
        // therefore must not size the 688-byte manifold pools. This matters
        // for batched robots: H1 keeps its analytic MJCF proxies for replay,
        // while only three exact USD hulls per environment are active.
        self.maxPairs = max(4096, enabledColliderCount * maxPairsPerBody)
        self.mapCapacity = Self.nextPow2(2 * maxPairs)
        self.gridHashSize = Self.nextPow2(
            max(64, 2 * max(1, enabledColliderCount)))
        // Tet BOUNDARY faces are collision triangles too: soft-soft contact
        // is element-based (soft V-V sphere pairs are banned at broadphase),
        // so without them tet bodies would pass through each other and
        // rigids would only touch their corner features. Collision-only —
        // membranes and the render extractor keep using scene.tris/tets.
        let tetBoundaryTris = Self.tetBoundaryFaces(scene)
        self.tetBoundaryTris = tetBoundaryTris
        self.numTris = scene.tris.count + tetBoundaryTris.count
        self.skinnedVertexCount = scene.skinnedMeshes.reduce(0) { $0 + $1.vertices.count }
        self.skinnedTriCount = scene.skinnedMeshes.reduce(0) { $0 + $1.triangles.count }
        self.rigidMeshVertexCount = scene.rigidMeshes.reduce(0) {
            $0 + 3 * $1.triangles.count
        }
        // Derive the actual surface and unique edge counts instead of relying
        // on the old triangle heuristic, which under-allocated disconnected
        // and open meshes.
        let collisionTriangles = scene.tris.map(\.ids) + tetBoundaryTris
        var surfaceParticles = Set<Int>()
        for (a, b, c) in collisionTriangles {
            if scene.bodies[a].isParticle { surfaceParticles.insert(a) }
            if scene.bodies[b].isParticle { surfaceParticles.insert(b) }
            if scene.bodies[c].isParticle { surfaceParticles.insert(c) }
        }
        var contactEdges = Set<UInt64>()
        for (a, b, c) in collisionTriangles {
            for (u, v) in [(a, b), (b, c), (a, c)] {
                contactEdges.insert(UInt64(min(u, v)) << 32
                    | UInt64(max(u, v)))
            }
        }
        var isotropicContactEdges = Set<UInt64>()
        for tri in scene.tris {
            let (a, b, c) = tri.ids
            for (u, v) in [(a, b), (b, c), (a, c)] {
                isotropicContactEdges.insert(UInt64(min(u, v)) << 32
                    | UInt64(max(u, v)))
            }
        }
        let planarPairCapacity = max(
            1,
            AVBD_PLANAR_DAT_PAIRS_PER_PARTICLE * surfaceParticles.count
                + AVBD_PLANAR_DAT_PAIRS_PER_EDGE * contactEdges.count)
        self.maxPlanarDATPairs = planarPairCapacity
        // Full contact emission is no longer capped at four per owner. Keep a
        // generous linear operational budget while the independent raw count
        // remains exact and terminal on overflow; sizing this to the much
        // larger complete safety-pair capacity would waste hundreds of MB on
        // folds.
        let planarContactBudget = min(
            planarPairCapacity,
            8 * surfaceParticles.count + 8 * contactEdges.count)
        self.maxSoft = max(1, 4 * numTris + planarContactBudget)
        self.softMapCapacity = Self.nextPow2(max(64, 2 * maxSoft))
        self.maxIsotropicSoft = max(1, 4 * surfaceParticles.count
            + 4 * numTris + 4 * isotropicContactEdges.count)
        self.isotropicSoftMapCapacity = Self.nextPow2(
            max(64, 2 * maxIsotropicSoft))
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
        gravityScale = try makeBuf(nb * 4, "gravityScale")
        shapeType = try makeBuf(nb * 4, "shapeType")
        spinVel = try makeBuf(nb * 16, "spinVel")
        colliderOwner = try makeBuf(numColliders * 4, "colliderOwner")
        colliderShape = try makeBuf(numColliders * 16, "colliderShape")
        colliderShapeType = try makeBuf(numColliders * 4, "colliderShapeType")
        colliderGroup = try makeBuf(numColliders * 4, "colliderGroup")
        colliderSharedCollision = try makeBuf(
            numColliders * 4, "colliderSharedCollision")
        colliderLocalPosition = try makeBuf(numColliders * 16, "colliderLocalPosition")
        colliderLocalRotation = try makeBuf(numColliders * 16, "colliderLocalRotation")
        colliderRenderColor = try makeBuf(numColliders * 16, "colliderRenderColor")
        colliderFriction = try makeBuf(
            numColliders * MemoryLayout<SIMD2<Float>>.stride,
            "colliderFriction")
        colliderHullRange = try makeBuf(
            numColliders * MemoryLayout<SIMD2<UInt32>>.stride,
            "colliderHullRange")
        convexHullVertices = try makeBuf(
            max(1, numConvexHullVertices) * MemoryLayout<SIMD4<Float>>.stride,
            "convexHullVertices")

        joints = try makeBuf(max(1, numJoints) * MemoryLayout<JointGPU>.stride, "joints")
        springs = try makeBuf(max(1, numSprings) * MemoryLayout<SpringGPU>.stride, "springs")
        tets = try makeBuf(max(1, numTets) * MemoryLayout<TetGPU>.stride, "tets")
        manifolds = try makeBuf(maxPairs * MemoryLayout<ManifoldGPU>.stride, "manifolds")
        prevManifolds = try makeBuf(maxPairs * MemoryLayout<ManifoldGPU>.stride, "prevManifolds")

        hashedIdx = try makeBuf(numColliders * 4, "hashedIdx")
        globalIdx = try makeBuf(numColliders * 4, "globalIdx")
        hashedRigidIdx = try makeBuf(numColliders * 4, "hashedRigidIdx")
        clothVertFlag = try makeBuf(nb * 4, "clothVertFlag")
        softSelfCollisionFlag = try makeBuf(nb * 4, "softSelfCollisionFlag")
        boundsBuf = try makeBuf(nb * 4, "ogcBounds")
        ogcPrevBuf = try makeBuf(nb * 16, "ogcPrev")
        ogcArgsBuf = try makeBuf(3 * 4, "ogcArgs")
        planarDATPairsBuf = try makeBuf(
            maxPlanarDATPairs * 16, "planarDATPairs")
        planarDATPairCountsBuf = try makeBuf(3 * 4, "planarDATPairCounts")
        planarDATTBuf = try makeBuf(nb * 4, "planarDATT")
        planarDATTBuf.contents().bindMemory(
            to: UInt32.self, capacity: nb
        ).initialize(repeating: Float(1).bitPattern, count: nb)
        planarDATArgsBuf = try makeBuf(3 * 4, "planarDATArgs")
        planarDATBodyCountBuf = try makeBuf(nb * 4, "planarDATBodyCount")
        planarDATBodyStartBuf = try makeBuf(nb * 4, "planarDATBodyStart")
        planarDATBodyCursorBuf = try makeBuf(nb * 4, "planarDATBodyCursor")
        planarDATBodyPairsBuf = try makeBuf(0, "planarDATBodyPairs")
        // 2-ring CSR sized after derivation below (~19/vertex upper bound)
        nbr2Start = try makeBuf(nb * 4, "nbr2Start")
        nbr2Count = try makeBuf(nb * 4, "nbr2Count")
        nbr2List = try makeBuf(max(1, numTris * 24) * 4, "nbr2List")
        cellCount = try makeBuf(gridHashSize * 4, "cellCount")
        cellRigid = try makeBuf(gridHashSize * 4, "cellRigid")
        cellStart = try makeBuf(gridHashSize * 4, "cellStart")
        cellBodies = try makeBuf(numColliders * 4, "cellBodies")
        bodyCellSlot = try makeBuf(numColliders * 8, "bodyCellSlot")
        pairCount = try makeBuf(numColliders * 4, "pairCount")
        pairStart = try makeBuf(numColliders * 4, "pairStart")
        pairs = try makeBuf(maxPairs * 8, "pairs")
        exclusions = try makeBuf(max(1, scene.joints.count + scene.springs.count
                                     + scene.collisionExclusions.count) * 8,
                                 "exclusions")

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
        // AABB multi-cell: Planar-DAT permits a 4x4x4 cover (the isotropic
        // path retains its legacy 3x3x3 logical span), so 64x capacity makes
        // every healthy scatter exact.
        elemCells = try makeBuf(max(1, 64 * (numTris + maxEdges)) * 4,
                                "elemCells")
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
        renderBodyIdxBuf = try makeBuf(max(1, numColliders) * 4, "renderColliderIdx")
        clothGroupBuf = try makeBuf(nb * 4, "clothGroup")
        skinBindingBuf = try makeBuf(max(1, skinnedVertexCount) * MemoryLayout<SkinBindingGPU>.stride,
                                     "skinBindings")
        skinVertexBuf = try makeBuf(max(1, skinnedVertexCount) * MemoryLayout<SkinVertexGPU>.stride,
                                    "skinVertices")
        skinTriBuf = try makeBuf(max(1, skinnedTriCount) * 12, "skinTris")
        rigidMeshVertexBuf = try makeBuf(
            max(1, rigidMeshVertexCount) * MemoryLayout<RigidMeshVertexGPU>.stride,
            "rigidMeshVertices")
        vtTrackBuf = try makeBuf(nb * 16, "vtTrack")
        eeTrackBuf = try makeBuf(max(1, maxEdges) * 16, "eeTrack")
        triAdjBuf = try makeBuf(max(1, numTris) * 16, "triAdj")
        vertEdgeStart = try makeBuf(nb * 4, "vertEdgeStart")
        vertEdgeCount = try makeBuf(nb * 4, "vertEdgeCount")
        vertEdgeList = try makeBuf(max(1, maxEdges) * 8, "vertEdgeList")

        degrees = try makeBuf(nb * 4, "degrees")
        adjStart = try makeBuf(nb * 4, "adjStart")
        adjCursor = try makeBuf(nb * 4, "adjCursor")
        let adjacencyCapacity = 2 * (numJoints + numSprings + maxPairs)
            + 4 * numTets + 4 * maxSoft + 3 * numTris + 4 * maxEdges
        adjList = try makeBuf(adjacencyCapacity * 4, "adjList")
        let compactNeighborCapacity = numTris == 0 && numTets == 0
            && numBodies > Self.orderedColoringBodyLimit
            ? adjacencyCapacity : 1
        adjNeighbor = try makeBuf(compactNeighborCapacity * 4, "adjNeighbor")
        colorsA = try makeBuf(nb * 4, "colorsA")
        colorsB = try makeBuf(nb * 4, "colorsB")
        bodySlot = try makeBuf(nb * 4, "bodySlot")
        colorStart = try makeBuf((AVBD_MAX_COLORS + 2) * 4, "colorStart")
        colorList = try makeBuf(nb * 4, "colorList")
        changedFlag = try makeBuf(24 * 4, "changedFlag")  // per-pass slots

        counters = try makeBuf(GPUCounters.total * 4, "counters")
        counterReadbacks = try (0..<2).map {
            try makeBuf(GPUCounters.total * 4, "counterReadback[\($0)]")
        }
        dispatchArgs = try makeBuf(9 * 4, "dispatchArgs")
        colorArgs = try makeBuf(AVBD_MAX_COLORS * 3 * 4, "colorArgs")
        let maxScanCount = max(max(gridHashSize, elemHashSize), nb)
        scanBlockSums = try makeBuf(((maxScanCount + 1023) / 1024 + 1) * 4, "scanBlockSums")
        scanTotal = try makeBuf(4, "scanTotal")
        diag = try makeBuf(4, "diag")

        try buildPipelines()
        try upload(scene: scene)
        // This CSR is only a high-color optimization. If Metal cannot
        // reserve it, retain the placeholder and use the global reducer.
        if effectiveSurfaceTruncationMode == .planarDAT,
           staticUsedColors > 4,
           ProcessInfo.processInfo.environment["AVBD_DAT_GLOBAL_COLOR"] == nil {
            let bodyPairBuffer: MTLBuffer?
            if let allocator = optionalPlanarDATBodyPairAllocator {
                bodyPairBuffer = allocator(dev, planarDATBodyPairStorageBytes)
            } else {
                bodyPairBuffer = dev.makeBuffer(
                    length: planarDATBodyPairStorageBytes,
                    options: .storageModeShared)
            }
            if let bodyPairBuffer {
                bodyPairBuffer.label = "planarDATBodyPairs"
                planarDATBodyPairsBuf = bodyPairBuffer
            }
        }
        self.spinners = scene.spinners
        // expose spinner angular velocity to the contact solver so friction
        // sees the kinematic surface motion
        memset(spinVel.contents(), 0, spinVel.length)
        let sv = spinVel.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        for sp in scene.spinners {
            sv[sp.body] = SIMD4(sp.axis * sp.omega, 1)
        }
    }

    public enum AVBDError: Error, Equatable, LocalizedError {
        case noDevice
        case allocFailed(String)
        case shaderCompile(String)
        case kernelMissing(String)

        public var errorDescription: String? {
            switch self {
            case .noDevice:
                return "Metal device or command queue is unavailable"
            case .allocFailed(let label):
                return "Metal buffer allocation failed: \(label)"
            case .shaderCompile(let message):
                return "Metal shader compilation failed: \(message)"
            case .kernelMissing(let name):
                return "Metal kernel is unavailable: \(name)"
            }
        }
    }

    /// Terminal failures discovered while encoding or retiring solver work.
    /// These are separate from the original construction-time `AVBDError`
    /// surface so clients can continue to switch exhaustively over it.
    public enum RuntimeFailure: Error, Equatable, LocalizedError {
        case commandBufferCreation(operation: String, frame: Int)
        case commandEncoderCreation(operation: String, stage: String,
                                    frame: Int)
        case commandExecution(operation: String, frame: Int, status: Int,
                              domain: String, code: Int, message: String)
        case rigidPairCapacity(frame: Int, required: Int, capacity: Int)
        case softContactCapacity(frame: Int, required: Int, capacity: Int)
        case staticColorCapacity(body: Int, required: Int, capacity: Int)
        case unresolvedColoring(frame: Int, conflictingBodies: Int)

        // Planar-DAT host validation failures use the pre-existing extensible
        // commandExecution payload rather than adding enum cases. That keeps
        // exhaustive switches over RuntimeFailure source-compatible while the
        // domain/code pair remains machine-readable.
        static let planarDATFailureDomain = "PhysicsAVBD.PlanarDAT"
        static let planarDATPairCapacityCode = 1
        static let planarDATElementGridSpanCode = 2
        static let planarDATInvalidAnchorCode = 3

        public static func planarDATPairCapacity(
            frame: Int, required: Int, capacity: Int
        ) -> Self {
            .commandExecution(
                operation: "Planar-DAT safety validation", frame: frame,
                status: 0, domain: planarDATFailureDomain,
                code: planarDATPairCapacityCode,
                message: "Planar-DAT safety-pair demand \(required) exceeds capacity \(capacity)")
        }

        public static func planarDATElementGridSpan(
            frame: Int, offendingElements: Int
        ) -> Self {
            .commandExecution(
                operation: "Planar-DAT safety validation", frame: frame,
                status: 0, domain: planarDATFailureDomain,
                code: planarDATElementGridSpanCode,
                message: "\(offendingElements) Planar-DAT element AABBs exceeded the supported 4x4x4 grid span")
        }

        public static func planarDATInvalidAnchor(
            frame: Int, offendingPairs: Int
        ) -> Self {
            planarDATInvalidAnchor(
                frame: frame, offendingPairs: offendingPairs,
                vertexTriangle: 0, edgeEdge: 0, tet: 0, nonfinite: 0)
        }

        public static func planarDATInvalidAnchor(
            frame: Int, offendingPairs: Int,
            vertexTriangle: Int, edgeEdge: Int,
            tet: Int, nonfinite: Int
        ) -> Self {
            let breakdown = "V-T coincident \(vertexTriangle), "
                + "E-E degenerate \(edgeEdge), tet degenerate \(tet), "
                + "non-finite \(nonfinite)"
            return .commandExecution(
                operation: "Planar-DAT safety validation", frame: frame,
                status: 0, domain: planarDATFailureDomain,
                code: planarDATInvalidAnchorCode,
                message: "\(offendingPairs) Planar-DAT anchors were invalid "
                    + "(\(breakdown))")
        }

        public var errorDescription: String? {
            switch self {
            case .commandBufferCreation(let operation, let frame):
                return "\(operation) frame \(frame): command-buffer creation failed"
            case .commandEncoderCreation(let operation, let stage, let frame):
                return "\(operation) frame \(frame): encoder creation failed at \(stage)"
            case .commandExecution(let operation, let frame, let status,
                                   let domain, let code, let message):
                if domain == Self.planarDATFailureDomain {
                    return "physics frame \(frame): \(message)"
                }
                let detail = domain.isEmpty ? message
                    : "\(domain) \(code): \(message)"
                return "\(operation) frame \(frame): Metal status \(status) (\(detail))"
            case .rigidPairCapacity(let frame, let required, let capacity):
                return "physics frame \(frame): rigid-pair demand \(required) exceeds capacity \(capacity)"
            case .softContactCapacity(let frame, let required, let capacity):
                return "physics frame \(frame): soft-contact demand \(required) exceeds capacity \(capacity)"
            case .staticColorCapacity(let body, let required, let capacity):
                return "greedy static solver coloring at body \(body) exhausted \(capacity) colors (next color \(required))"
            case .unresolvedColoring(let frame, let conflictingBodies):
                return "physics frame \(frame): dynamic coloring retained \(conflictingBodies) conflicting bodies"
            }
        }
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
        } catch let error as AVBDError {
            throw error
        } catch {
            throw AVBDError.shaderCompile(error.localizedDescription)
        }
        shaderLib = lib
        for name in lib.functionNames {
            guard let fn = lib.makeFunction(name: name) else { continue }
            pso[name] = try device.makeComputePipelineState(function: fn)
        }
        try Self.validateRequiredKernelNames(Set(pso.keys))
    }
    var shaderLib: MTLLibrary?

    static func validateRequiredKernelNames(_ available: Set<String>) throws {
        if let missing = requiredKernelNames.subtracting(available).min() {
            throw AVBDError.kernelMissing(missing)
        }
    }

    static func validateShaderResourceURLs(_ urls: [URL]) throws {
        guard !urls.isEmpty else {
            throw AVBDError.shaderCompile(
                "no .metal resources were found in the PhysicsAVBD bundle")
        }
    }

    /// SwiftPM executables place target bundles beside the executable, while
    /// Xcode applications place them in `Contents/Resources`. Check both
    /// distributable layouts before falling back to SwiftPM's build-time
    /// accessor, whose absolute development path is not portable.
    private static var packagedResourceBundle: Bundle? {
        let name = "avbd-metal_PhysicsAVBD.bundle"
        let roots = [Bundle.main.resourceURL, Bundle.main.bundleURL]
        for root in roots.compactMap({ $0 }) {
            if let bundle = Bundle(url: root.appendingPathComponent(name)) {
                return bundle
            }
        }
        return nil
    }

    /// Concatenates the bundled .metal sources (filename order) and compiles.
    static func makeLibrary(device: MTLDevice) throws -> MTLLibrary {
        let resources = packagedResourceBundle ?? Bundle.module
        var urls = (resources.urls(forResourcesWithExtension: "metal", subdirectory: nil) ?? [])
        if urls.isEmpty {   // .copy resource rule keeps the Shaders/ subdir
            urls = resources.urls(forResourcesWithExtension: "metal",
                                  subdirectory: "Shaders") ?? []
        }
        urls.sort { $0.lastPathComponent < $1.lastPathComponent }
        try validateShaderResourceURLs(urls)
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
        if #available(macOS 15.0, iOS 18.0, *) {
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
        let gs = gravityScale.contents().bindMemory(to: Float.self,
                                                    capacity: numBodies)
        let colA = colorsA.contents().bindMemory(to: UInt32.self, capacity: numBodies)
        let colB = colorsB.contents().bindMemory(to: UInt32.self, capacity: numBodies)

        for (i, b) in scene.bodies.enumerated() {
            var mass: Float
            var moment: F3
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
            if let explicitMass = b.mass,
               let explicitInertia = b.diagonalInertia {
                mass = explicitMass
                moment = explicitInertia
            }
            pl[i] = SIMD4(b.position, mass)
            pa[i] = SIMD4(b.rotation.imag, b.rotation.real)
            vl[i] = SIMD4(b.velocity, 0)
            va[i] = .zero
            pv[i] = SIMD4(b.velocity, 0)
            pr[i] = SIMD4(moment, b.friction)
            sh[i] = SIMD4(b.size, b.isParticle ? -radius : radius)
            gs[i] = b.gravityScale
            switch b.shape {
            case .box: st[i] = 0
            case .sphere: st[i] = 1
            case .torus: st[i] = 2
            case .capsule: st[i] = 3
            }
            colA[i] = UInt32(i % AVBD_MAX_COLORS)  // initial guess; refined per frame
            colB[i] = colA[i]
        }

        let co = colliderOwner.contents().bindMemory(to: UInt32.self,
                                                     capacity: numColliders)
        let cs = colliderShape.contents().bindMemory(to: SIMD4<Float>.self,
                                                     capacity: numColliders)
        let ct = colliderShapeType.contents().bindMemory(to: UInt32.self,
                                                         capacity: numColliders)
        let cg = colliderGroup.contents().bindMemory(to: UInt32.self,
                                                     capacity: numColliders)
        let csc = colliderSharedCollision.contents().bindMemory(
            to: UInt32.self, capacity: numColliders)
        let cp = colliderLocalPosition.contents().bindMemory(to: SIMD4<Float>.self,
                                                             capacity: numColliders)
        let cq = colliderLocalRotation.contents().bindMemory(to: SIMD4<Float>.self,
                                                             capacity: numColliders)
        let crc = colliderRenderColor.contents().bindMemory(to: SIMD4<Float>.self,
                                                            capacity: numColliders)
        let cf = colliderFriction.contents().bindMemory(
            to: SIMD2<Float>.self, capacity: numColliders)
        let chr = colliderHullRange.contents().bindMemory(
            to: SIMD2<UInt32>.self, capacity: numColliders)
        let chv = convexHullVertices.contents().bindMemory(
            to: SIMD4<Float>.self, capacity: max(1, numConvexHullVertices))
        func radius(of c: SceneCollider) -> Float {
            if !c.convexHullVertices.isEmpty {
                return c.convexHullVertices.reduce(0) { max($0, length($1)) }
            }
            switch c.shape {
            case .sphere: return c.size.x / 2
            case .torus: return c.size.x + c.size.y
            case .capsule: return c.size.x / 2 + c.size.y
            case .box: return length(c.size * 0.5)
            }
        }
        var radii: [Float] = []
        radii.reserveCapacity(numColliders)
        var hullVertexOffset = 0
        for (i, c) in scene.colliders.enumerated() {
            precondition(scene.bodies.indices.contains(c.body),
                         "collider owner out of range")
            let r = radius(of: c)
            let particle = scene.bodies[c.body].isParticle
            co[i] = UInt32(c.body)
            cg[i] = c.collisionGroup
            csc[i] = c.collidesWithSharedGeometry ? 1 : 0
            cs[i] = SIMD4(c.size, particle ? -r : r)
            cp[i] = SIMD4(c.localPosition, c.friction)
            cq[i] = SIMD4(c.localRotation.imag, c.localRotation.real)
            crc[i] = c.renderColor.map { SIMD4($0, 1) } ?? .zero
            cf[i] = SIMD2(c.friction, c.dynamicFriction)
            if c.convexHullVertices.isEmpty {
                chr[i] = .zero
            } else {
                precondition(c.convexHullVertices.count <= 64,
                             "Metal convex contact supports at most 64 vertices")
                chr[i] = SIMD2(UInt32(hullVertexOffset),
                               UInt32(c.convexHullVertices.count))
                for vertex in c.convexHullVertices {
                    chv[hullVertexOffset] = SIMD4(vertex, 0)
                    hullVertexOffset += 1
                }
            }
            var flags: UInt32
            if !c.convexHullVertices.isEmpty {
                flags = 4
            } else {
                switch c.shape {
                case .box: flags = 0
                case .sphere: flags = 1
                case .torus: flags = 2
                case .capsule: flags = 3
                }
            }
            if particle { flags |= 0x10 }
            if c.usesWorldSpaceRoundAnchor { flags |= 0x20 }
            ct[i] = flags
            if c.collisionEnabled && scene.bodies[c.body].isDynamic {
                radii.append(r)
            }
        }
        precondition(hullVertexOffset == numConvexHullVertices)

        // Partition collision primitives into hashed vs global sets. Globals:
        // primitives far larger than the median dynamic radius.
        // far larger than the median dynamic radius (keeps grid cells tight).
        let sortedRadii = radii.sorted()
        let medianRadius = sortedRadii.isEmpty ? 0.5 : sortedRadii[sortedRadii.count / 2]
        let threshold = medianRadius * 4
        var hashed: [UInt32] = []
        var globals: [UInt32] = []
        // Only oversized bodies go to the brute-forced global list; normal
        // sized statics live in the spatial hash like everything else.
        let hasIsolatedGroups = scene.colliders.contains {
            $0.collisionEnabled && $0.collisionGroup != 0
        }
        for (i, c) in scene.colliders.enumerated() {
            guard c.collisionEnabled else { continue }
            // Shared-domain geometry must be visible to every isolated hash
            // domain, so keep it in the normally tiny global list.
            if radius(of: c) > threshold
                || (hasIsolatedGroups && c.collisionGroup == 0) {
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
            maxHashedRadius = max(maxHashedRadius, abs(cs[Int(i)].w))
        }

        hashedIdx.contents().bindMemory(to: UInt32.self, capacity: max(1, hashed.count))
            .update(from: hashed.isEmpty ? [0] : hashed, count: max(1, hashed.count))
        globalIdx.contents().bindMemory(to: UInt32.self, capacity: max(1, globals.count))
            .update(from: globals.isEmpty ? [0] : globals, count: max(1, globals.count))

        // Joints
        let jp = joints.contents().bindMemory(to: JointGPU.self, capacity: max(1, numJoints))
        for (i, j) in scene.joints.enumerated() {
            if j.motorTorque > 0 {
                precondition(j.hingeAxis != nil,
                             "powered joint \(i) must define a hinge axis")
                precondition(j.motorTarget.isFinite
                    && j.motorTorque.isFinite
                    && j.motorStiffness.isFinite
                    && j.motorDamping.isFinite
                    && j.armature.isFinite,
                    "powered joint \(i) has non-finite motor parameters")
                precondition(j.motorTorque > 0 && j.motorStiffness >= 0
                    && j.motorDamping >= 0 && j.armature >= 0,
                    "powered joint \(i) has negative motor parameters")
                switch j.motorMode {
                case .implicitPositionPD, .explicitTorquePD:
                    precondition(j.motorStiffness > 0
                        && j.motorDamping >= 0,
                        "position-PD joint \(i) requires physical kp/kd gains")
                case .velocity:
                    precondition(j.motorStiffness == 0
                        && j.motorDamping > 0,
                        "velocity joint \(i) requires kp=0 and kd>0")
                }
            }
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
                g.dynamics.x = j.armature
                if j.motorTorque > 0 {
                    g.motor = SIMD4(j.motorTarget, j.motorTorque, 0,
                                    j.motorStiffness)
                    g.limits = SIMD4(j.limitLo, j.limitHi,
                                    j.motorDamping, 0)
                    switch j.motorMode {
                    case .implicitPositionPD:
                        g.header.w |= Self.jointMotorModeImplicitPositionPD
                    case .explicitTorquePD:
                        g.header.w |= Self.jointMotorModeExplicitTorquePD
                    case .velocity:
                        g.header.w |= Self.jointMotorModeVelocity
                    }
                    if j.motorRate != 0 {
                        precondition(j.motorMode == .implicitPositionPD,
                            "legacy motor rates require implicit position PD")
                        rateMotors.append((i, j.motorRate))
                    }
                }
                if j.limitLo < j.limitHi && j.motorTorque == 0 {
                    g.limits = SIMD4(j.limitLo, j.limitHi, 0, 0)
                }
            }
            // A hard articulation constraint must be load-bearing on its
            // first frame. Starting every hard joint at PENALTY_MIN lets an
            // articulated robot compress under gravity until the adaptive
            // penalty has ramped, so its first reset has different dynamics
            // from every later reset. Use the same effective-mass / dt^2
            // floor as newly created contacts; AVBD can still adapt upward.
            let massA = j.bodyA >= 0 ? pl[j.bodyA].w : 0
            let massB = pl[j.bodyB].w
            let dynamicMasses = [massA, massB].filter { $0 > 0 }
            let effectiveMass = dynamicMasses.min() ?? 0
            let hardPenaltyFloor = min(
                1.0e9,
                max(1, effectiveMass
                    / max(settings.dt * settings.dt, 1.0e-12)))
            if j.stiffnessLin.isInfinite {
                g.penaltyLin = SIMD4(repeating: hardPenaltyFloor)
            }
            if j.stiffnessAng.isInfinite {
                g.penaltyAng = SIMD4(repeating: hardPenaltyFloor)
            }
            // Soft (finite) joints are their physical stiffness from frame
            // one; they do not use the adaptive hard-constraint floor.
            if j.stiffnessLin > 0 && j.stiffnessLin.isFinite {
                g.penaltyLin = SIMD4(repeating: min(j.stiffnessLin, 1e9))
            }
            if j.stiffnessAng > 0 && j.stiffnessAng.isFinite {
                g.penaltyAng = SIMD4(repeating: min(j.stiffnessAng, 1e9))
            }
            jp[i] = g
        }
        initialJointPenaltyLin = (0..<numJoints).map { jp[$0].penaltyLin }
        initialJointPenaltyAng = (0..<numJoints).map { jp[$0].penaltyAng }

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
            let signedSixVolume = dot(d0, cross(d1, d2))
            let vol = abs(signedSixVolume) / 6
            let Dm = simd_float3x3(columns: (d0, d1, d2))
            let DmInv = Dm.inverse
            // rows of DmInv
            let r0 = F3(DmInv.columns.0.x, DmInv.columns.1.x, DmInv.columns.2.x)
            let r1 = F3(DmInv.columns.0.y, DmInv.columns.1.y, DmInv.columns.2.y)
            let r2 = F3(DmInv.columns.0.z, DmInv.columns.1.z, DmInv.columns.2.z)
            g.r0 = SIMD4(r0, signedSixVolume)
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

        // Collision exclusions: sorted (min,max) pairs of jointed/springed
        // bodies plus explicit source-asset exclusions.
        var excl: [UInt64] = []
        excl.reserveCapacity(scene.joints.count + scene.springs.count
                             + scene.collisionExclusions.count)
        for j in scene.joints where j.bodyA >= 0 {
            let lo = UInt64(min(j.bodyA, j.bodyB)), hi = UInt64(max(j.bodyA, j.bodyB))
            excl.append(lo << 32 | hi)
        }
        for s in scene.springs {
            let lo = UInt64(min(s.bodyA, s.bodyB)), hi = UInt64(max(s.bodyA, s.bodyB))
            excl.append(lo << 32 | hi)
        }
        for e in scene.collisionExclusions {
            let lo = UInt64(min(e.bodyA, e.bodyB)), hi = UInt64(max(e.bodyA, e.bodyB))
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
        var topoEdgeSet = UInt64UniqueBuilder(capacity: collTris.count * 3)
        var topoEdgeKeys: [UInt64] = []
        topoEdgeKeys.reserveCapacity(collTris.count * 3)
        var contactEdgeSet = UInt64UniqueBuilder(capacity: collTris.count * 3)
        var contactEdgeKeys: [UInt64] = []
        contactEdgeKeys.reserveCapacity(scene.tris.count * 3)
        var maxElemR: Float = 0
        var maxCollisionParticleRadius: Float = 0
        for (i, ids) in collTris.enumerated() {
            let (a, b, c) = ids
            tp4[i] = SIMD4(UInt32(a), UInt32(b), UInt32(c), 0)
            let emitsEE = i < scene.tris.count
            for (u, v) in [(a, b), (b, c), (a, c)] {
                let key = UInt64(min(u, v)) << 32 | UInt64(max(u, v))
                if topoEdgeSet.insert(key) { topoEdgeKeys.append(key) }
                if emitsEE && contactEdgeSet.insert(key) {
                    contactEdgeKeys.append(key)
                }
            }
            let pa = scene.bodies[a].position, pb = scene.bodies[b].position
            let pc = scene.bodies[c].position
            let m = (pa + pb + pc) / 3
            let thick = max(scene.bodies[a].size.x, max(scene.bodies[b].size.x,
                                                        scene.bodies[c].size.x)) / 2
            maxCollisionParticleRadius = max(maxCollisionParticleRadius, thick)
            maxElemR = max(maxElemR, max(distance(m, pa), max(distance(m, pb),
                                                                              distance(m, pc))) + thick)
        }
        // Preserve authored cloth edges as the force-model prefix, then append
        // synthesized tet-boundary edges for Planar-DAT safety only. V-T alone
        // cannot detect every triangle-triangle crossing: an edge-edge event
        // can occur with no vertex entering the opposing triangle. Keeping
        // `numEdges` at the authored count also preserves the established
        // contract that closed tet solids do not acquire cloth E-E forces.
        numEdges = contactEdgeKeys.count
        for key in topoEdgeKeys where contactEdgeSet.insert(key) {
            contactEdgeKeys.append(key)
        }
        numPlanarDATEdges = contactEdgeKeys.count
        let ep2 = edgesBuf.contents().bindMemory(to: SIMD2<UInt32>.self,
                                                 capacity: max(1, numPlanarDATEdges))
        var nbrArrays = [[UInt32]](repeating: [], count: numBodies)
        var vertEdges: [[UInt32]] = Array(repeating: [], count: numBodies)
        for key in topoEdgeKeys {
            let ai = Int(key >> 32)
            let bi = Int(key & 0xFFFFFFFF)
            nbrArrays[ai].append(UInt32(bi))
            nbrArrays[bi].append(UInt32(ai))
        }
        for (i, key) in contactEdgeKeys.enumerated() {
            let ai = Int(key >> 32)
            let bi = Int(key & 0xFFFFFFFF)
            ep2[i] = SIMD2(UInt32(ai), UInt32(bi))
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
        // Soft collision groups are part of the contact model, not rendering.
        // Skinned tet meshes intentionally hide their generated collision
        // shell from the renderer; deriving these groups from render surfaces
        // made skinned soft bodies fall back to expensive particle-pair
        // broadphase in addition to V-T/E-E element contacts.
        let groupP = clothGroupBuf.contents().bindMemory(to: UInt32.self,
                                                         capacity: numBodies)
        let selfCollisionP = softSelfCollisionFlag.contents().bindMemory(
            to: UInt32.self, capacity: numBodies)
        for i in 0..<numBodies { groupP[i] = 0 }
        for i in 0..<numBodies { selfCollisionP[i] = 0 }
        if !collTris.isEmpty {
            var parent: [Int: Int] = [:]
            func findCollisionRoot(_ x: Int) -> Int {
                var r = x
                while let p = parent[r], p != r { r = p }
                var c = x
                while let p = parent[c], p != r { parent[c] = r; c = p }
                if parent[r] == nil { parent[r] = r }
                return r
            }
            for (a, b, c) in collTris {
                let r = findCollisionRoot(a)
                parent[findCollisionRoot(b)] = r
                parent[findCollisionRoot(c)] = r
            }
            // Thin shells self-collide by default. Volumetric bodies do not:
            // they already have internal elasticity, and their dense
            // boundary-vs-boundary stream is useful only for assets that
            // explicitly author self-collision. Inter-component soft contact
            // remains unconditional in the pair kernels.
            var selfCollisionRoots = Set<Int>()
            for tri in scene.tris {
                selfCollisionRoots.insert(findCollisionRoot(tri.ids.0))
            }
            for tet in scene.tets where tet.selfCollisionEnabled {
                for id in [tet.ids.0, tet.ids.1, tet.ids.2, tet.ids.3]
                    where parent[id] != nil {
                    selfCollisionRoots.insert(findCollisionRoot(id))
                }
            }
            var groupIndex: [Int: UInt32] = [:]
            for (a, b, c) in collTris {
                let root = findCollisionRoot(a)
                if groupIndex[root] == nil {
                    groupIndex[root] = UInt32(groupIndex.count + 1)
                }
                let g = groupIndex[root]!
                if scene.bodies[a].isParticle { groupP[a] = g }
                if scene.bodies[b].isParticle { groupP[b] = g }
                if scene.bodies[c].isParticle { groupP[c] = g }
                let selfCollision = selfCollisionRoots.contains(root)
                    ? UInt32(1) : UInt32(0)
                if scene.bodies[a].isParticle { selfCollisionP[a] = selfCollision }
                if scene.bodies[b].isParticle { selfCollisionP[b] = selfCollision }
                if scene.bodies[c].isParticle { selfCollisionP[c] = selfCollision }
            }
            params.numSoftGroups = UInt32(groupIndex.count)
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
        var maxPartR = maxCollisionParticleRadius
        var edgeLenSum: Float = 0
        for key in topoEdgeKeys {
            let a = scene.bodies[Int(key >> 32)], b = scene.bodies[Int(key & 0xFFFFFFFF)]
            maxPartR = max(maxPartR, max(a.size.x, b.size.x) / 2)
            edgeLenSum += distance(a.position, b.position)
        }
        let meanEdge = topoEdgeKeys.isEmpty ? 0.2 : edgeLenSum / Float(topoEdgeKeys.count)
        params.elemMargin = min(0.01, max(0.002, 0.5 * maxPartR))
        // Planar-DAT queries a wider safety neighborhood than the active
        // OGC force band. Anything beyond this radius is protected by the
        // unconditional 0.5*gamma*rq accumulated-displacement cap.
        let forceReach = 2 * maxPartR + params.elemMargin
        params.planarDATQueryRadius = max(1.5 * forceReach,
                                          0.75 * meanEdge)
        params.planarDATRelaxation = 0.85
        params.maxPlanarDATPairs = UInt32(maxPlanarDATPairs)
        // Detection-radius cells (AABB multi-cell insertion): reach must
        // cover skin + margin + velocity inflation (<= 0.3 cell, hence /0.7).
        // The broadphase bins fat element AABBs and clamps cover loops to
        // 3 cells/axis, so the cell also has to cover the largest element
        // radius plus collision padding.
        let reach = (2 * maxPartR + params.elemMargin) / 0.7
        isotropicElemCellSize = max(
            0.02, max(max(reach, meanEdge * 1.05),
                      maxElemR + maxPartR + params.elemMargin))
        params.surfaceContactCellSize = isotropicElemCellSize
        planarDATElemCellSize = max(
            0.02, max(max(reach, meanEdge * 1.05),
                      2 * maxElemR + params.planarDATQueryRadius))
        params.elemCellSize = planarDATElemCellSize
        params.elemHashSize = UInt32(elemHashSize)
        params.numTris = UInt32(numTris)
        params.numEdges = UInt32(numPlanarDATEdges)
        params.numSurfaceContactEdges = UInt32(numEdges)
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
            var incidence: [Int: [Int]] = [:]   // body -> surface tri indices
            var triComp: [UInt32] = []
            let clothP = clothVertFlag.contents().bindMemory(to: UInt32.self,
                                                             capacity: numBodies)
            for i in 0..<numBodies { clothP[i] = 0 }
            for (ti, tv) in triVerts.enumerated() {
                let root = find(tv[0])
                if compIdx[root] == nil { compIdx[root] = compIdx.count }
                let comp = UInt32(min(compIdx[root]!, 255))
                triComp.append(comp)
                for v in tv {
                    surfCorners.append(UInt32(v) | (comp << 24))
                    incidence[v, default: []].append(ti)
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
        if rigidMeshVertexCount > 0 {
            var corners: [RigidMeshVertexGPU] = []
            corners.reserveCapacity(rigidMeshVertexCount)
            for mesh in scene.rigidMeshes {
                precondition(scene.bodies.indices.contains(mesh.body),
                             "rigid mesh owner out of range")
                precondition(mesh.normals.count == mesh.vertices.count,
                             "rigid mesh normal count mismatch")
                let bodyBits = Float(bitPattern: UInt32(mesh.body))
                let color = SIMD4(mesh.color, 1)
                for triangle in mesh.triangles {
                    for index in [triangle.0, triangle.1, triangle.2] {
                        precondition(mesh.vertices.indices.contains(index),
                                     "rigid mesh triangle index out of range")
                        var corner = RigidMeshVertexGPU()
                        let position = mesh.localPosition
                            + mesh.localRotation.act(mesh.vertices[index])
                        let normal = normalize(
                            mesh.localRotation.act(mesh.normals[index]))
                        corner.positionBody = SIMD4(position, bodyBits)
                        corner.normal = SIMD4(normal, 0)
                        corner.color = color
                        corners.append(corner)
                    }
                }
            }
            precondition(corners.count == rigidMeshVertexCount)
            rigidMeshVertexBuf.contents().bindMemory(
                to: RigidMeshVertexGPU.self, capacity: corners.count)
                .update(from: corners, count: corners.count)
        }
        let flags = surfacedFlags.contents().bindMemory(to: UInt32.self,
                                                        capacity: numBodies)
        var renderColliderIDs: [UInt32] = []
        renderColliderIDs.reserveCapacity(numColliders)
        for (i, c) in scene.colliders.enumerated()
            where c.isRendered && flags[c.body] == 0 {
            renderColliderIDs.append(UInt32(i))
        }
        renderRigidBodyCount = renderColliderIDs.count
        renderBodyIdxBuf.contents().bindMemory(to: UInt32.self,
                                               capacity: max(1, renderColliderIDs.count))
            .update(from: renderColliderIDs.isEmpty ? [0] : renderColliderIDs,
                    count: max(1, renderColliderIDs.count))

        // ---- Static topology coloring (computed ONCE; contacts do not
        // constrain colors — same-color contact pairs degrade to Jacobi,
        // which penalty-based contacts tolerate; stiff elements keep strict
        // Gauss-Seidel ordering). Replaces 20 GPU Jacobi rounds per frame.
        var adjacencySets = [Set<Int>](repeating: [], count: numBodies)
        func link2(_ a: Int, _ b: Int) {
            guard a >= 0, b >= 0,
                  scene.bodies[a].isDynamic || scene.bodies[b].isDynamic else { return }
            if scene.bodies[a].isDynamic && scene.bodies[b].isDynamic {
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
        for v in 0..<numBodies where scene.bodies[v].isDynamic {
            var used: UInt64 = 0
            for nb in adjacencySets[v] where nb < v {
                let c = staticColors[nb]
                if c < 64 { used |= (1 << UInt64(c)) }
            }
            guard used != UInt64.max else {
                throw RuntimeFailure.staticColorCapacity(
                    body: v, required: AVBD_MAX_COLORS + 1,
                    capacity: AVBD_MAX_COLORS)
            }
            var c = 0
            while (used & (1 << UInt64(c))) != 0 { c += 1 }
            staticColors[v] = c
            maxColor = max(maxColor, c)
        }
        staticUsedColors = maxColor + 1
        // colorList/colorStart in final form, uploaded once
        var buckets = [[UInt32]](repeating: [], count: AVBD_MAX_COLORS)
        for v in 0..<numBodies where scene.bodies[v].isDynamic {
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
        let hashedRigid = hashed.filter {
            !scene.bodies[scene.colliders[Int($0)].body].isParticle
        }
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
        params.frictionCombineMode = settings.frictionCombineMode.rawValue
        params.collisionMargin = max(settings.collisionMargin, 0)
        params.rigidLinearDamping = max(settings.rigidLinearDamping, 0)
        params.rigidAngularDamping = max(settings.rigidAngularDamping, 0)
        let effectiveTruncationMode = effectiveSurfaceTruncationMode
        params.surfaceTruncationMode = numParticles > 0
            ? effectiveTruncationMode.rawValue : 0
        params.tetInversionPreventionEnabled =
            effectiveTruncationMode == .planarDAT
                && tetInversionPreventionEnabled ? 1 : 0
        params.maxPlanarDATPairs = UInt32(max(0, min(
            maxPlanarDATPairs,
            planarDATPairCapacityForTesting ?? maxPlanarDATPairs)))
        params.numEdges = UInt32(effectiveTruncationMode == .planarDAT
            ? numPlanarDATEdges : numEdges)
        params.elemCellSize = effectiveTruncationMode == .planarDAT
            ? planarDATElemCellSize : isotropicElemCellSize
        let selectedSoftCapacity = effectiveTruncationMode == .planarDAT
            ? maxSoft : maxIsotropicSoft
        params.maxSoft = UInt32(max(
            0, min(selectedSoftCapacity,
                   planarDATSoftCapacityForTesting ?? selectedSoftCapacity)))
        params.softMapCapacity = UInt32(effectiveTruncationMode == .planarDAT
            ? softMapCapacity : isotropicSoftMapCapacity)

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

    // ---- async pipelining: submitStep() never blocks; the queue serializes
    // GPU work, and checked CPU access synchronizes lazily ----
    private struct StepSubmission {
        let commandBuffer: MTLCommandBuffer
        let counterSnapshot: MTLBuffer
        let frame: Int
        let usesDynamicColoring: Bool
        let softCapacity: Int
        let planarDATCapacity: Int
    }
    private var inflight: [StepSubmission] = []
    private let statsLock = NSLock()
    private let failureLock = NSLock()
    private var latchedRuntimeFailure: RuntimeFailure?

    /// A runtime failure is terminal because a failed Metal command may have
    /// partially modified solver state. Rebuild the solver; never clear and
    /// continue a training rollout.
    public var runtimeFailure: RuntimeFailure? {
        failureLock.lock()
        defer { failureLock.unlock() }
        return latchedRuntimeFailure
    }

    // Instance-scoped deterministic failure injection for unit tests. The
    // hooks are read only while encoding a new submission; changing one
    // between submissions is safe even if an older command is still in flight.
    var commandBufferFactoryForTesting: (() -> MTLCommandBuffer?)?
    var deniedEncoderStageForTesting: String?
    var planarDATPairCapacityForTesting: Int?
    var planarDATSoftCapacityForTesting: Int?
    var planarDATBodyIncidenceForTesting = false
    var planarDATPassObserverForTesting: ((PlanarDATPassSite) -> Void)?
    var completionFailureForTesting: ((String, Int) -> RuntimeFailure?)?
    var inflightCountForTesting: Int { inflight.count }

    private var planarDATBodyPairStorageBytes: Int {
        max(16, maxPlanarDATPairs * 4 * MemoryLayout<UInt32>.stride)
    }

    /// Whether the optional per-body incidence list can represent the full
    /// accepted Planar pair stream. A missing list is never a correctness
    /// failure: encoding falls back to the global pair reducer.
    var hasPlanarDATBodyIncidenceStorage: Bool {
        numTris > 0
            && planarDATBodyPairsBuf.length >= planarDATBodyPairStorageBytes
    }

    /// Focused tests may force the incidence reducer on ordinary low-color
    /// shells, but must explicitly opt into its otherwise-unneeded storage.
    func enablePlanarDATBodyIncidenceForTesting() throws {
        if !hasPlanarDATBodyIncidenceStorage {
            guard let buffer = device.makeBuffer(
                length: planarDATBodyPairStorageBytes,
                options: .storageModeShared
            ) else {
                throw AVBDError.allocFailed("planarDATBodyPairs")
            }
            buffer.label = "planarDATBodyPairs"
            planarDATBodyPairsBuf = buffer
        }
        planarDATBodyIncidenceForTesting = true
    }

    func motorTargetForTesting(_ joint: Int) -> Float {
        sync()
        let values = joints.contents().bindMemory(
            to: JointGPU.self, capacity: max(1, numJoints))
        return values[joint].motor.x
    }

    private func makeRuntimeCommandBuffer() -> MTLCommandBuffer? {
        if let factory = commandBufferFactoryForTesting { return factory() }
        return queue.makeCommandBuffer()
    }

    /// Robotics: render parallel Push-T pixel observations via the
    /// analytic top-down compute kernel. envTable: PushTEnvGPU records.
    public func renderPushTObs(envTable: MTLBuffer, numEnvs: Int,
                               out: MTLBuffer, res: Int) {
        do {
            try renderPushTObsChecked(
                envTable: envTable, numEnvs: numEnvs, out: out, res: res)
        } catch {
            fatalError("Push-T observation rendering failed: \(error.localizedDescription)")
        }
    }

    public func renderPushTObsChecked(envTable: MTLBuffer, numEnvs: Int,
                                      out: MTLBuffer, res: Int) throws {
        try synchronize()
        guard let cmd = makeRuntimeCommandBuffer() else {
            throw latch(.commandBufferCreation(
                operation: "Push-T observation", frame: frameIndex))
        }
        guard deniedEncoderStageForTesting != "Push-T observation",
              let enc = cmd.makeComputeCommandEncoder() else {
            throw latch(.commandEncoderCreation(
                operation: "Push-T observation", stage: "render",
                frame: frameIndex))
        }
        cmd.label = "AVBD Push-T observation"
        enc.label = "Push-T observation"
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
        try waitForCompletion(
            cmd, operation: "Push-T observation", frame: frameIndex)
    }

    public var metalDevice: MTLDevice { device }

    public struct BodyPoseUpdate {
        public var body: Int
        public var position: F3
        public var rotation: Quat

        public init(body: Int, position: F3, rotation: Quat) {
            self.body = body
            self.position = position
            self.rotation = rotation
        }
    }

    /// Batched pose-and-velocity write used for deterministic resets and
    /// physical disturbance injection. Unlike an impulse shortcut, a launched
    /// object enters the ordinary broadphase/contact solver on the next step.
    public struct BodyStateUpdate {
        public var body: Int
        public var position: F3
        public var rotation: Quat
        public var linearVelocity: F3
        public var angularVelocity: F3

        public init(body: Int, position: F3, rotation: Quat,
                    linearVelocity: F3 = .zero,
                    angularVelocity: F3 = .zero) {
            self.body = body
            self.position = position
            self.rotation = rotation
            self.linearVelocity = linearVelocity
            self.angularVelocity = angularVelocity
        }
    }

    public struct JointAnchorUpdate {
        public var joint: Int
        public var point: F3

        public init(joint: Int, point: F3) {
            self.joint = joint
            self.point = point
        }
    }

    public struct MotorTargetUpdate {
        public var joint: Int
        public var angle: Float

        public init(joint: Int, angle: Float) {
            self.joint = joint
            self.angle = angle
        }
    }

    public struct RigidBodyState {
        public var position: F3
        public var rotation: Quat
        public var linearVelocity: F3
        public var angularVelocity: F3

        public init(position: F3, rotation: Quat, linearVelocity: F3,
                    angularVelocity: F3) {
            self.position = position
            self.rotation = rotation
            self.linearVelocity = linearVelocity
            self.angularVelocity = angularVelocity
        }
    }

    /// Robotics: teleport several bodies after a single GPU fence. Batched
    /// resets must use this API; synchronizing once per body destroys the
    /// throughput advantage of vectorized environments.
    public func setBodyPoses(_ updates: [BodyPoseUpdate]) {
        setBodyStates(updates.map {
            BodyStateUpdate(body: $0.body, position: $0.position,
                            rotation: $0.rotation)
        })
    }

    /// Robotics: update several rigid states after one GPU fence. Constraint
    /// warm starts incident to those bodies are cleared exactly as for pose
    /// resets, so a relaunched projectile cannot inherit stale impulses.
    public func setBodyStates(_ updates: [BodyStateUpdate]) {
        guard !updates.isEmpty else { return }
        sync()
        let pl = posLin.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        let pa = posAng.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        let vl = velLin.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        let va = velAng.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        let pvl = prevVelLin.contents().bindMemory(
            to: SIMD4<Float>.self, capacity: numBodies)
        for u in updates {
            precondition(u.body >= 0 && u.body < numBodies, "body index out of range")
            pl[u.body] = SIMD4(u.position, pl[u.body].w)
            pa[u.body] = SIMD4(u.rotation.imag, u.rotation.real)
            vl[u.body] = SIMD4(u.linearVelocity, 0)
            va[u.body] = SIMD4(u.angularVelocity, 0)
            // Adaptive body initialization estimates acceleration from this
            // history slot. Carrying a pre-teleport velocity here makes the
            // first post-reset prediction depend on the previous episode.
            pvl[u.body] = SIMD4(u.linearVelocity, 0)
        }
        // A teleported episode must not inherit augmented-Lagrangian state
        // from its previous trajectory. Clear warm starts for every
        // constraint incident to a reset body while leaving other replicas'
        // solver state untouched. Motor gains are fixed scene data.
        let resetBodies = Set(updates.map { UInt32($0.body) })
        if numJoints > 0 {
            let jp = joints.contents().bindMemory(to: JointGPU.self, capacity: numJoints)
            for i in 0..<numJoints {
                let a = jp[i].header.x, b = jp[i].header.y
                if resetBodies.contains(b)
                    || (a != UInt32.max && resetBodies.contains(a)) {
                    jp[i].lambdaLin = .zero
                    jp[i].lambdaAng = .zero
                    jp[i].motor.z = 0
                    jp[i].dynamics.y = 0
                    jp[i].dynamics.z = 0
                    jp[i].penaltyLin = initialJointPenaltyLin[i]
                    jp[i].penaltyAng = initialJointPenaltyAng[i]
                }
            }
        }
        if numSprings > 0 {
            let sp = springs.contents().bindMemory(to: SpringGPU.self, capacity: numSprings)
            for i in 0..<numSprings where resetBodies.contains(sp[i].header.x)
                    || resetBodies.contains(sp[i].header.y) {
                sp[i].dual = .zero
            }
        }
        // Contact multipliers, penalties, and static-friction anchors are
        // also temporal solver state. The persistence hash may continue to
        // point at these slots until the next narrow phase; a zero contact
        // count makes that lookup deliberately cold without rebuilding the
        // global map or disturbing other vectorized environments.
        for buffer in [manifolds, prevManifolds] {
            let mp = buffer.contents().bindMemory(
                to: ManifoldGPU.self, capacity: maxPairs)
            for i in 0..<maxPairs {
                if resetBodies.contains(mp[i].header.x)
                    || resetBodies.contains(mp[i].header.y) {
                    mp[i] = ManifoldGPU()
                }
            }
        }
    }

    /// Robotics: teleport a body (resets its velocity).
    public func setBodyPose(_ i: Int, position: F3, rotation: Quat) {
        setBodyPoses([BodyPoseUpdate(body: i, position: position, rotation: rotation)])
    }

    /// Robotics: move a world-anchored joint's target point (Cartesian
    /// position actuator — the joint's bounded force does the rest).
    public func setJointWorldAnchors(_ updates: [JointAnchorUpdate]) {
        guard !updates.isEmpty else { return }
        sync()
        let jp = joints.contents().bindMemory(to: JointGPU.self, capacity: max(1, numJoints))
        for u in updates {
            precondition(u.joint >= 0 && u.joint < numJoints, "joint index out of range")
            jp[u.joint].rA = SIMD4(u.point, jp[u.joint].rA.w)
        }
    }

    public func setJointWorldAnchor(_ jointIndex: Int, point: F3) {
        setJointWorldAnchors([JointAnchorUpdate(joint: jointIndex, point: point)])
    }

    /// Robotics: set a motor joint's target angle at runtime.
    public func setMotorTargets(_ updates: [MotorTargetUpdate]) {
        guard !updates.isEmpty else { return }
        sync()
        let jp = joints.contents().bindMemory(to: JointGPU.self, capacity: max(1, numJoints))
        for u in updates {
            precondition(u.joint >= 0 && u.joint < numJoints, "joint index out of range")
            precondition(u.angle.isFinite, "motor angle must be finite")
            let mode = jp[u.joint].header.w & Self.jointMotorModeMask
            // Dense robot action layouts may retain welded/disabled joints.
            // Their commands are intentional no-ops, never an implicit
            // request to create a motor after static coloring.
            if mode == 0 {
                precondition(jp[u.joint].motor.y == 0,
                             "active motor is missing an authored mode")
                continue
            }
            precondition(mode == Self.jointMotorModeImplicitPositionPD
                || mode == Self.jointMotorModeExplicitTorquePD,
                         "setMotorTargets requires a position-PD motor")
            jp[u.joint].motor.x = u.angle
        }
    }

    public func setMotorTarget(_ jointIndex: Int, angle: Float) {
        setMotorTargets([MotorTargetUpdate(joint: jointIndex, angle: angle)])
    }

    public struct MotorVelocityUpdate {
        public var joint: Int
        public var radiansPerSecond: Float

        public init(joint: Int, radiansPerSecond: Float) {
            self.joint = joint
            self.radiansPerSecond = radiansPerSecond
        }
    }

    /// Set commands for physically defined velocity motors.
    public func setMotorVelocities(_ updates: [MotorVelocityUpdate]) {
        guard !updates.isEmpty else { return }
        sync()
        let jp = joints.contents().bindMemory(
            to: JointGPU.self, capacity: max(1, numJoints))
        for update in updates {
            precondition(update.joint >= 0 && update.joint < numJoints,
                         "joint index out of range")
            precondition(update.radiansPerSecond.isFinite,
                         "motor velocity must be finite")
            precondition((jp[update.joint].header.w
                          & Self.jointMotorModeMask)
                         == Self.jointMotorModeVelocity,
                         "setMotorVelocities requires a velocity motor")
            jp[update.joint].motor.x = update.radiansPerSecond
        }
    }

    public func setMotorVelocity(_ jointIndex: Int,
                                 radiansPerSecond: Float) {
        setMotorVelocities([MotorVelocityUpdate(
            joint: jointIndex, radiansPerSecond: radiansPerSecond)])
    }

    public struct MotorTorqueUpdate {
        public var joint: Int
        public var torque: Float

        public init(joint: Int, torque: Float) {
            self.joint = joint
            self.torque = torque
        }
    }

    /// Robotics: set several motor effort limits behind one GPU fence.
    public func setMotorTorques(_ updates: [MotorTorqueUpdate]) {
        guard !updates.isEmpty else { return }
        sync()
        let jp = joints.contents().bindMemory(to: JointGPU.self, capacity: max(1, numJoints))
        for update in updates {
            precondition(update.torque >= 0 && update.torque.isFinite,
                         "motor torque limit must be finite and nonnegative")
            precondition(update.joint >= 0 && update.joint < numJoints,
                         "joint index out of range")
            precondition((jp[update.joint].header.w
                            & Self.jointMotorModeMask) != 0,
                         "cannot enable a motor that was not authored in the scene")
            jp[update.joint].motor.y = update.torque
        }
    }

    /// Robotics: set one motor joint's torque limit at runtime.
    public func setMotorTorque(_ jointIndex: Int, torque: Float) {
        setMotorTorques([MotorTorqueUpdate(
            joint: jointIndex, torque: torque)])
    }

    /// Robotics: read a motor joint's current twist angle.
    public func motorAngle(_ jointIndex: Int) -> Float {
        motorAngles([jointIndex])[0]
    }

    /// Read several hinge angles behind one GPU fence.
    public func motorAngles(_ jointIndices: [Int]) -> [Float] {
        motorStates(jointIndices).map(\.angle)
    }

    /// Read reduced hinge position and velocity behind one GPU fence. The
    /// velocity is the relative angular velocity of the child and parent,
    /// projected onto the current world-space hinge axis. This avoids noisy
    /// finite differences of the projected twist angle in RL observations
    /// and acceleration rewards.
    public func motorStates(_ jointIndices: [Int])
        -> [(angle: Float, velocity: Float)] {
        guard !jointIndices.isEmpty else { return [] }
        sync()
        let jp = joints.contents().bindMemory(to: JointGPU.self, capacity: max(1, numJoints))
        let pa = posAng.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        let va = velAng.contents().bindMemory(to: SIMD4<Float>.self,
                                              capacity: numBodies)
        return jointIndices.map { jointIndex in
            precondition(jointIndex >= 0 && jointIndex < numJoints, "joint index out of range")
            let j = jp[jointIndex]
            let a = j.header.x
            let b = Int(j.header.y)
            let qB = Quat(real: pa[b].w,
                          imag: F3(pa[b].x, pa[b].y, pa[b].z))
            let qA: Quat = a == 0xFFFFFFFF
                ? Quat(real: 1, imag: .zero)
                : Quat(real: pa[Int(a)].w,
                       imag: F3(pa[Int(a)].x, pa[Int(a)].y, pa[Int(a)].z))
            let rest = Quat(real: j.restRel.w,
                            imag: F3(j.restRel.x, j.restRel.y, j.restRel.z))
            var r = (qA * rest).inverse * qB
            if r.real < 0 { r = Quat(real: -r.real, imag: -r.imag) }
            let axis = F3(j.hingeAxis.x, j.hingeAxis.y, j.hingeAxis.z)
            let angle = 2 * atan2(dot(r.imag, axis), r.real)
            let childVelocity = F3(va[b].x, va[b].y, va[b].z)
            let parentVelocity: F3 = a == 0xFFFFFFFF
                ? .zero : F3(va[Int(a)].x, va[Int(a)].y, va[Int(a)].z)
            let velocity = dot(childVelocity - parentVelocity,
                               qB.act(axis))
            return (angle, velocity)
        }
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

    @discardableResult
    private func latch(_ failure: RuntimeFailure) -> RuntimeFailure {
        failureLock.lock()
        defer { failureLock.unlock() }
        if latchedRuntimeFailure == nil { latchedRuntimeFailure = failure }
        return latchedRuntimeFailure!
    }

    private func requireHealthy() throws {
        if let failure = runtimeFailure { throw failure }
    }

    private func waitForCompletion(_ command: MTLCommandBuffer,
                                   operation: String, frame: Int) throws {
        command.waitUntilCompleted()
        if let injected = completionFailureForTesting?(operation, frame) {
            throw latch(injected)
        }
        if command.status != .completed || command.error != nil {
            let nsError = command.error as NSError?
            throw latch(.commandExecution(
                operation: operation, frame: frame,
                status: Int(command.status.rawValue),
                domain: nsError?.domain ?? "",
                code: nsError?.code ?? 0,
                message: nsError?.localizedDescription
                    ?? "command did not complete successfully"))
        }
    }

    private func retire(_ submission: StepSubmission) throws {
        let command = submission.commandBuffer
        try waitForCompletion(
            command, operation: "physics", frame: submission.frame)

        let ctr = submission.counterSnapshot.contents().bindMemory(
            to: UInt32.self, capacity: GPUCounters.total)
        let pairCandidates = Int(ctr[GPUCounters.pairCandidates])
        let softCandidates = Int(ctr[GPUCounters.softCandidates])
        let pairs = Int(ctr[GPUCounters.pairs])
        let soft = Int(ctr[GPUCounters.soft])
        let colorConflicts = Int(ctr[GPUCounters.colorConflicts])
        let planarDATPairs = Int(ctr[GPUCounters.planarDATPairs])
        let planarDATVT = Int(ctr[GPUCounters.planarDATVertexTrianglePairs])
        let planarDATEE = Int(ctr[GPUCounters.planarDATEdgeEdgePairs])
        let planarDATGridOverflows = Int(
            ctr[GPUCounters.planarDATGridOverflows])
        let planarDATInvalidAnchors = Int(
            ctr[GPUCounters.planarDATInvalidAnchors])
        let planarDATTruncations = Int(ctr[GPUCounters.planarDATTruncations])
        let planarDATVTDegeneracies = Int(
            ctr[GPUCounters.planarDATVertexTriangleDegeneracies])
        let planarDATEEDegeneracies = Int(
            ctr[GPUCounters.planarDATEdgeEdgeDegeneracies])
        let planarDATNonfinite = Int(
            ctr[GPUCounters.planarDATNonfiniteValues])
        let planarDATTetDegeneracies = Int(
            ctr[GPUCounters.planarDATTetDegeneracies])

        statsLock.lock()
        lastPairCandidates = pairCandidates
        lastNumPairs = pairs
        lastSoftCandidates = softCandidates
        lastNumSoft = soft
        lastPlanarDATPairs = planarDATPairs
        lastPlanarDATVertexTrianglePairs = planarDATVT
        lastPlanarDATEdgeEdgePairs = planarDATEE
        lastPlanarDATTruncations = planarDATTruncations
        lastPlanarDATVertexTriangleDegeneracies = planarDATVTDegeneracies
        lastPlanarDATEdgeEdgeDegeneracies = planarDATEEDegeneracies
        lastPlanarDATNonfiniteValues = planarDATNonfinite
        lastPlanarDATTetDegeneracies = planarDATTetDegeneracies
        if submission.usesDynamicColoring {
            var counts = [Int]()
            counts.reserveCapacity(AVBD_MAX_COLORS)
            var maxUsed = -1
            for color in 0..<AVBD_MAX_COLORS {
                let count = Int(ctr[GPUCounters.colorBase + color])
                counts.append(count)
                if count > 0 { maxUsed = color }
            }
            lastColorCounts = counts
            lastMaxColorUsed = maxUsed
        }
        statsLock.unlock()

        if pairCandidates > maxPairs {
            throw latch(.rigidPairCapacity(
                frame: submission.frame, required: pairCandidates,
                capacity: maxPairs))
        }
        if softCandidates > submission.softCapacity {
            throw latch(.softContactCapacity(
                frame: submission.frame, required: softCandidates,
                capacity: submission.softCapacity))
        }
        if planarDATPairs > submission.planarDATCapacity {
            throw latch(.planarDATPairCapacity(
                frame: submission.frame, required: planarDATPairs,
                capacity: submission.planarDATCapacity))
        }
        if planarDATGridOverflows > 0 {
            throw latch(.planarDATElementGridSpan(
                frame: submission.frame,
                offendingElements: planarDATGridOverflows))
        }
        if planarDATInvalidAnchors > 0 {
            throw latch(.planarDATInvalidAnchor(
                frame: submission.frame,
                offendingPairs: planarDATInvalidAnchors,
                vertexTriangle: planarDATVTDegeneracies,
                edgeEdge: planarDATEEDegeneracies,
                tet: planarDATTetDegeneracies,
                nonfinite: planarDATNonfinite))
        }
        if colorConflicts > 0 {
            throw latch(.unresolvedColoring(
                frame: submission.frame,
                conflictingBodies: colorConflicts))
        }
    }

    /// Wait for every submitted physics frame, validate Metal completion,
    /// then publish frame-owned statistics. The earliest failure is sticky
    /// and permanently invalidates this solver instance.
    public func synchronize() throws {
        let pending = inflight
        inflight.removeAll(keepingCapacity: true)
        var observedFailure: RuntimeFailure?
        for submission in pending {
            do {
                try retire(submission)
            } catch let failure as RuntimeFailure {
                if observedFailure == nil { observedFailure = failure }
            }
        }
        if let failure = observedFailure ?? runtimeFailure { throw failure }
    }

    /// Source-compatible fail-closed wrapper. Production code that can
    /// recover at a run boundary should call `synchronize()` directly.
    public func sync() {
        do {
            try synchronize()
        } catch {
            fatalError("GPUSolver synchronization failed: \(error.localizedDescription)")
        }
    }

    /// Opaque rigid-scene checkpoint for batched speculative control. Unlike
    /// `setBodyStates`, this preserves the temporal solver state that makes a
    /// fork dynamically equivalent to continuing the original rollout:
    /// adaptive velocity history, joint/contact multipliers, persistence-map
    /// identity, and incremental coloring. It deliberately excludes authored
    /// scene data and soft-body state; callers must use it only with the same
    /// solver instance and a rigid scene.
    package struct RigidSpeculationSnapshot {
        fileprivate var posLin: Data
        fileprivate var posAng: Data
        fileprivate var initLin: Data
        fileprivate var initAng: Data
        fileprivate var inertLin: Data
        fileprivate var inertAng: Data
        fileprivate var velLin: Data
        fileprivate var velAng: Data
        fileprivate var prevVelLin: Data
        fileprivate var joints: Data
        fileprivate var springs: Data
        fileprivate var manifolds: Data
        fileprivate var prevManifolds: Data
        fileprivate var mapKeyA: Data
        fileprivate var mapKeyB: Data
        fileprivate var mapVal: Data
        fileprivate var colorsA: Data
        fileprivate var colorsB: Data
        fileprivate var counters: Data
        fileprivate var frameIndex: Int
        fileprivate var lastColorCounts: [Int]
        fileprivate var lastNumPairs: Int
        fileprivate var lastPairCandidates: Int
        fileprivate var lastNumSoft: Int
        fileprivate var lastSoftCandidates: Int
        fileprivate var lastMaxColorUsed: Int
    }

    package func captureRigidSpeculationSnapshot() -> RigidSpeculationSnapshot {
        precondition(numTris == 0 && numTets == 0,
            "rigid speculation snapshots do not include soft-body state")
        sync()
        func copy(_ buffer: MTLBuffer) -> Data {
            Data(bytes: buffer.contents(), count: buffer.length)
        }
        statsLock.lock()
        let statistics = (
            lastColorCounts, lastNumPairs, lastPairCandidates,
            lastNumSoft, lastSoftCandidates, lastMaxColorUsed)
        statsLock.unlock()
        return RigidSpeculationSnapshot(
            posLin: copy(posLin), posAng: copy(posAng),
            initLin: copy(initLin), initAng: copy(initAng),
            inertLin: copy(inertLin), inertAng: copy(inertAng),
            velLin: copy(velLin), velAng: copy(velAng),
            prevVelLin: copy(prevVelLin), joints: copy(joints),
            springs: copy(springs), manifolds: copy(manifolds),
            prevManifolds: copy(prevManifolds), mapKeyA: copy(mapKeyA),
            mapKeyB: copy(mapKeyB), mapVal: copy(mapVal),
            colorsA: copy(colorsA), colorsB: copy(colorsB),
            counters: copy(counters), frameIndex: frameIndex,
            lastColorCounts: statistics.0, lastNumPairs: statistics.1,
            lastPairCandidates: statistics.2,
            lastNumSoft: statistics.3,
            lastSoftCandidates: statistics.4,
            lastMaxColorUsed: statistics.5)
    }

    package func restoreRigidSpeculationSnapshot(
        _ snapshot: RigidSpeculationSnapshot
    ) {
        precondition(numTris == 0 && numTets == 0,
            "rigid speculation snapshots do not include soft-body state")
        sync()
        func restore(_ data: Data, to buffer: MTLBuffer) {
            precondition(data.count == buffer.length,
                "speculation snapshot buffer size mismatch")
            data.withUnsafeBytes { bytes in
                guard let source = bytes.baseAddress else { return }
                memcpy(buffer.contents(), source, data.count)
            }
        }
        restore(snapshot.posLin, to: posLin)
        restore(snapshot.posAng, to: posAng)
        restore(snapshot.initLin, to: initLin)
        restore(snapshot.initAng, to: initAng)
        restore(snapshot.inertLin, to: inertLin)
        restore(snapshot.inertAng, to: inertAng)
        restore(snapshot.velLin, to: velLin)
        restore(snapshot.velAng, to: velAng)
        restore(snapshot.prevVelLin, to: prevVelLin)
        restore(snapshot.joints, to: joints)
        restore(snapshot.springs, to: springs)
        restore(snapshot.manifolds, to: manifolds)
        restore(snapshot.prevManifolds, to: prevManifolds)
        restore(snapshot.mapKeyA, to: mapKeyA)
        restore(snapshot.mapKeyB, to: mapKeyB)
        restore(snapshot.mapVal, to: mapVal)
        restore(snapshot.colorsA, to: colorsA)
        restore(snapshot.colorsB, to: colorsB)
        restore(snapshot.counters, to: counters)
        frameIndex = snapshot.frameIndex
        statsLock.lock()
        lastColorCounts = snapshot.lastColorCounts
        lastNumPairs = snapshot.lastNumPairs
        lastPairCandidates = snapshot.lastPairCandidates
        lastNumSoft = snapshot.lastNumSoft
        lastSoftCandidates = snapshot.lastSoftCandidates
        lastMaxColorUsed = snapshot.lastMaxColorUsed
        statsLock.unlock()
    }

    /// Cap the pipeline depth to the two frame-owned counter readback slots.
    private func throttleChecked() throws {
        while inflight.count >= 2 {
            let oldest = inflight.removeFirst()
            do {
                try retire(oldest)
            } catch let failure as RuntimeFailure {
                // A terminal error invalidates every queued frame. Drain the
                // queue before returning so no GPU work outlives its solver
                // resources, while preserving the earliest failure.
                let remaining = inflight
                inflight.removeAll(keepingCapacity: true)
                for submission in remaining { try? retire(submission) }
                throw failure
            }
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

    /// Source-compatible fail-closed wrapper. RL and evaluation code should
    /// use `submitStep()` and propagate its typed error at the run boundary.
    public func step() {
        do {
            try submitStep()
        } catch {
            fatalError("GPUSolver step failed: \(error.localizedDescription)")
        }
    }

    /// Encode and asynchronously submit one physics frame. Synchronous
    /// creation/encoding failures are thrown immediately; asynchronous Metal,
    /// contact-capacity, and coloring failures surface at `synchronize()` or
    /// when pipeline throttling retires the frame.
    public func submitStep() throws {
        try requireHealthy()
        try throttleChecked()
        if profiling || !spinners.isEmpty || !rateMotors.isEmpty {
            // Profiling retires its own submission synchronously. Drain any
            // older asynchronous owner before selecting a parity readback
            // slot; CPU-authored kinematic targets likewise require a fence.
            try synchronize()
        }
        do {
            try encodeAndSubmitStep()
        } catch let failure as RuntimeFailure {
            // A synchronous failure in frame N must still retire every older
            // submitted frame before it reaches the caller. Otherwise the
            // throwing API would return while Metal continued mutating the
            // solver behind an already-failed rollout boundary.
            let pending = inflight
            inflight.removeAll(keepingCapacity: true)
            var earlierFailure: RuntimeFailure?
            for submission in pending {
                do {
                    try retire(submission)
                } catch let prior as RuntimeFailure {
                    if earlierFailure == nil { earlierFailure = prior }
                }
            }
            if let earlierFailure { throw earlierFailure }
            throw latch(failure)
        }
    }

    private func encodeAndSubmitStep() throws {
        syncParams()
        let submittedFrame = frameIndex + 1

        // ---- Command buffer 1: collision + warm start + adjacency + coloring
        guard let cmd1 = makeRuntimeCommandBuffer() else {
            throw RuntimeFailure.commandBufferCreation(
                operation: "physics", frame: submittedFrame)
        }
        cmd1.label = "AVBD physics frame \(submittedFrame)"

        guard let clearEncoder = cmd1.makeBlitCommandEncoder() else {
            throw RuntimeFailure.commandEncoderCreation(
                operation: "physics", stage: "clear", frame: submittedFrame)
        }
        clearEncoder.label = "clear"
        clearEncoder.fill(buffer: counters, range: 0..<counters.length, value: 0)
        clearEncoder.fill(buffer: changedFlag, range: 0..<changedFlag.length, value: 0)
        clearEncoder.fill(buffer: cellCount, range: 0..<(gridHashSize * 4), value: 0)
        clearEncoder.fill(buffer: cellRigid, range: 0..<(gridHashSize * 4), value: 0)
        if numTris > 0 {
            // OGC conservative bounds: reset to "far" (0x7F7F7F7F ~ 3e38)
            clearEncoder.fill(buffer: boundsBuf,
                              range: 0..<(numBodies * 4), value: 0x7F)
            clearEncoder.fill(buffer: elemCellCount,
                              range: 0..<(elemHashSize * 4), value: 0)
        }
        clearEncoder.endEncoding()

        let sampleBuf = profiling ? makeCounterBuf() : nil
        stageNames = []
        func makeEncoder(_ name: String) -> MTLComputeCommandEncoder? {
            if deniedEncoderStageForTesting == name { return nil }
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
        guard var enc = makeEncoder("broadphase") else {
            throw RuntimeFailure.commandEncoderCreation(
                operation: "physics", stage: "broadphase",
                frame: submittedFrame)
        }
        // ALWAYS split encoders at stage boundaries: measured 2.5-3x faster
        // than one mega-encoder — intra-encoder hazard barriers over long
        // dispatch chains drain the pipe far harder than encoder boundaries
        func stage(_ name: String) throws {
            enc.endEncoding()
            guard let next = makeEncoder(name) else {
                throw RuntimeFailure.commandEncoderCreation(
                    operation: "physics", stage: name,
                    frame: submittedFrame)
            }
            enc = next
        }
        var P = params
        var nExcl = numExclusions
        let isPlanarDAT = P.surfaceTruncationMode
            == SurfaceTruncationMode.planarDAT.rawValue
        // Both kernels implement the same fixed-plane reduction. A global
        // pair-parallel pass is cheaper for the 2-3 color palettes typical
        // of thin shells; a body-incidence pass avoids rereading the stream
        // across the 5-7 colors required by tetrahedral element cliques.
        // Select from solver structure rather than demo names.
        let usePlanarBodyIncidence = (staticUsedColors > 4
            || planarDATBodyIncidenceForTesting)
            && hasPlanarDATBodyIncidenceStorage
            && ProcessInfo.processInfo.environment[
                "AVBD_DAT_GLOBAL_COLOR"] == nil

        func encodeElementGrid(_ encoder: MTLComputeCommandEncoder,
                               clearFirst: Bool) {
            if clearFirst {
                dispatch1D(encoder, "dat_clear_element_grid", elemHashSize) { e in
                    e.setBuffer(self.elemCellCount, offset: 0, index: 0)
                    e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride,
                               index: 1)
                }
            }
            dispatch1D(encoder, "el_count", numTris + Int(P.numEdges)) { e in
                e.setBuffer(self.posLin, offset: 0, index: 0)
                e.setBuffer(self.trisBuf, offset: 0, index: 1)
                e.setBuffer(self.edgesBuf, offset: 0, index: 2)
                e.setBuffer(self.shape, offset: 0, index: 3)
                e.setBuffer(self.elemCellCount, offset: 0, index: 4)
                e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride,
                           index: 5)
                e.setBuffer(self.counters, offset: 0, index: 6)
            }
            encodeScan(encoder, input: elemCellCount,
                       output: elemCellStart, count: elemHashSize)
            var hashSize = UInt32(elemHashSize)
            dispatch1D(encoder, "adj_copy_cursor", elemHashSize) { e in
                e.setBuffer(self.elemCellStart, offset: 0, index: 0)
                e.setBuffer(self.elemSlot, offset: 0, index: 1)
                e.setBytes(&hashSize, length: 4, index: 2)
            }
            dispatch1D(encoder, "el_scatter", numTris + Int(P.numEdges)) { e in
                e.setBuffer(self.posLin, offset: 0, index: 0)
                e.setBuffer(self.trisBuf, offset: 0, index: 1)
                e.setBuffer(self.edgesBuf, offset: 0, index: 2)
                e.setBuffer(self.shape, offset: 0, index: 3)
                e.setBuffer(self.elemSlot, offset: 0, index: 4)
                e.setBuffer(self.elemCells, offset: 0, index: 5)
                e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride,
                           index: 6)
            }
        }

        func encodePlanarPairBuild(_ encoder: MTLComputeCommandEncoder) {
            dispatch1D(encoder, "dat_clear_pair_counts", 3) { e in
                e.setBuffer(self.planarDATPairCountsBuf, offset: 0, index: 0)
            }
            encoder.memoryBarrier(resources: [planarDATPairCountsBuf])
            if numParticles > 0
                && ProcessInfo.processInfo.environment["AVBD_NO_VT"] == nil {
                encoder.setComputePipelineState(ps("dat_build_vt_pairs"))
                encoder.setBuffer(self.posLin, offset: 0, index: 0)
                encoder.setBuffer(self.particleIdxBuf, offset: 0, index: 1)
                encoder.setBuffer(self.trisBuf, offset: 0, index: 2)
                encoder.setBuffer(self.elemCellStart, offset: 0, index: 3)
                encoder.setBuffer(self.elemCellCount, offset: 0, index: 4)
                encoder.setBuffer(self.elemCells, offset: 0, index: 5)
                // Newton's default topology threshold: one-edge-ring
                // primitives are excluded once, then the exact same retained
                // pair set feeds OGC forces and Planar-DAT.
                encoder.setBuffer(self.nbrStart, offset: 0, index: 6)
                encoder.setBuffer(self.nbrCount, offset: 0, index: 7)
                encoder.setBuffer(self.nbrList, offset: 0, index: 8)
                encoder.setBuffer(self.shape, offset: 0, index: 9)
                encoder.setBuffer(self.clothGroupBuf, offset: 0, index: 10)
                encoder.setBuffer(self.softSelfCollisionFlag, offset: 0,
                                  index: 11)
                encoder.setBuffer(self.planarDATPairsBuf, offset: 0, index: 12)
                encoder.setBuffer(self.planarDATPairCountsBuf, offset: 0, index: 13)
                encoder.setBuffer(self.counters, offset: 0, index: 14)
                encoder.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride,
                                 index: 15)
                encoder.setBuffer(self.edgesBuf, offset: 0, index: 19)
                encoder.dispatchThreadgroups(
                    MTLSize(width: numParticles, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 64, height: 1,
                                                   depth: 1))
            }
            if P.numEdges > 0
                && ProcessInfo.processInfo.environment["AVBD_NO_EE"] == nil {
                encoder.setComputePipelineState(ps("dat_build_ee_pairs"))
                encoder.setBuffer(self.posLin, offset: 0, index: 0)
                encoder.setBuffer(self.edgesBuf, offset: 0, index: 1)
                encoder.setBuffer(self.trisBuf, offset: 0, index: 2)
                encoder.setBuffer(self.elemCellStart, offset: 0, index: 3)
                encoder.setBuffer(self.elemCellCount, offset: 0, index: 4)
                encoder.setBuffer(self.elemCells, offset: 0, index: 5)
                encoder.setBuffer(self.nbrStart, offset: 0, index: 6)
                encoder.setBuffer(self.nbrCount, offset: 0, index: 7)
                encoder.setBuffer(self.nbrList, offset: 0, index: 8)
                encoder.setBuffer(self.shape, offset: 0, index: 9)
                encoder.setBuffer(self.clothGroupBuf, offset: 0, index: 10)
                encoder.setBuffer(self.softSelfCollisionFlag, offset: 0,
                                  index: 11)
                encoder.setBuffer(self.planarDATPairsBuf, offset: 0, index: 12)
                encoder.setBuffer(self.planarDATPairCountsBuf, offset: 0, index: 13)
                encoder.setBuffer(self.counters, offset: 0, index: 14)
                encoder.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride,
                                 index: 15)
                encoder.dispatchThreadgroups(
                    MTLSize(width: Int(P.numEdges), height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 64, height: 1,
                                                   depth: 1))
            }
            // Pair builders publish payload and relaxed atomic counts from
            // thousands of lanes. Make both visible before the single-lane
            // finalizer snapshots the count and writes indirect arguments;
            // relying on dispatch order alone is not a Metal memory barrier.
            encoder.memoryBarrier(resources: [planarDATPairsBuf,
                                               planarDATPairCountsBuf])
            dispatch1D(encoder, "dat_finalize_pairs", 1) { e in
                e.setBuffer(self.planarDATPairCountsBuf, offset: 0, index: 0)
                e.setBuffer(self.counters, offset: 0, index: 1)
                e.setBuffer(self.planarDATArgsBuf, offset: 0, index: 2)
                e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride,
                           index: 3)
            }
            encoder.memoryBarrier(resources: [planarDATPairsBuf,
                                               planarDATPairCountsBuf,
                                               planarDATArgsBuf])
        }

        func encodePlanarContacts(_ encoder: MTLComputeCommandEncoder,
                                  pairs: MTLBuffer,
                                  pairCounts: MTLBuffer,
                                  positions: MTLBuffer,
                                  referencePositions: MTLBuffer) {
            let contactPSO = ps("dat_emit_contacts")
            encoder.setComputePipelineState(contactPSO)
            encoder.setBuffer(pairs, offset: 0, index: 0)
            encoder.setBuffer(positions, offset: 0, index: 1)
            encoder.setBuffer(shape, offset: 0, index: 2)
            encoder.setBuffer(props, offset: 0, index: 3)
            encoder.setBuffer(velLin, offset: 0, index: 4)
            encoder.setBuffer(trisBuf, offset: 0, index: 5)
            encoder.setBuffer(edgesBuf, offset: 0, index: 6)
            encoder.setBuffer(softContacts, offset: 0, index: 7)
            encoder.setBuffer(counters, offset: 0, index: 8)
            encoder.setBuffer(prevSoftContacts, offset: 0, index: 9)
            encoder.setBuffer(softMapKeyA, offset: 0, index: 10)
            encoder.setBuffer(softMapKeyB, offset: 0, index: 11)
            encoder.setBuffer(softMapVal, offset: 0, index: 12)
            encoder.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride,
                             index: 13)
            encoder.setBuffer(pairCounts, offset: 0, index: 14)
            // initLin deliberately stores only coordinates (w == 0). Keep
            // dynamic mass classification bound to the authoritative state.
            encoder.setBuffer(posLin, offset: 0, index: 15)
            // Contacts are detected at the accepted predictor pose but the
            // solver linearizes from initLin. The kernel reconstructs C0 from
            // these exact step-start coordinates rather than double-counting
            // predictor displacement.
            encoder.setBuffer(referencePositions, offset: 0, index: 16)
            encoder.dispatchThreadgroups(
                indirectBuffer: planarDATArgsBuf,
                indirectBufferOffset: 0,
                threadsPerThreadgroup: MTLSize(width: 64, height: 1,
                                               depth: 1))
        }

        func encodePlanarIncidence(_ encoder: MTLComputeCommandEncoder) {
            var bodyCount = UInt32(numBodies)
            dispatch1D(encoder, "adj_clear_degrees", numBodies) { e in
                e.setBuffer(self.planarDATBodyCountBuf, offset: 0, index: 0)
                e.setBytes(&bodyCount, length: 4, index: 1)
            }
            encoder.setComputePipelineState(ps("dat_incidence_count"))
            encoder.setBuffer(planarDATPairsBuf, offset: 0, index: 0)
            encoder.setBuffer(planarDATPairCountsBuf, offset: 0, index: 1)
            encoder.setBuffer(planarDATBodyCountBuf, offset: 0, index: 2)
            encoder.setBuffer(trisBuf, offset: 0, index: 3)
            encoder.setBuffer(edgesBuf, offset: 0, index: 4)
            encoder.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride,
                             index: 5)
            encoder.dispatchThreadgroups(
                indirectBuffer: planarDATArgsBuf,
                indirectBufferOffset: 0,
                threadsPerThreadgroup: MTLSize(width: 64, height: 1,
                                               depth: 1))
            encodeScan(encoder, input: planarDATBodyCountBuf,
                       output: planarDATBodyStartBuf, count: numBodies)
            dispatch1D(encoder, "adj_copy_cursor", numBodies) { e in
                e.setBuffer(self.planarDATBodyStartBuf, offset: 0, index: 0)
                e.setBuffer(self.planarDATBodyCursorBuf, offset: 0, index: 1)
                e.setBytes(&bodyCount, length: 4, index: 2)
            }
            encoder.setComputePipelineState(ps("dat_incidence_scatter"))
            encoder.setBuffer(planarDATPairsBuf, offset: 0, index: 0)
            encoder.setBuffer(planarDATPairCountsBuf, offset: 0, index: 1)
            encoder.setBuffer(planarDATBodyCursorBuf, offset: 0, index: 2)
            encoder.setBuffer(planarDATBodyPairsBuf, offset: 0, index: 3)
            encoder.setBuffer(trisBuf, offset: 0, index: 4)
            encoder.setBuffer(edgesBuf, offset: 0, index: 5)
            encoder.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride,
                             index: 6)
            encoder.dispatchThreadgroups(
                indirectBuffer: planarDATArgsBuf,
                indirectBufferOffset: 0,
                threadsPerThreadgroup: MTLSize(width: 64, height: 1,
                                               depth: 1))
        }

        func encodeSoftContactMap(_ encoder: MTLComputeCommandEncoder) {
            dispatch1D(encoder, "soft_finalize", 1) { e in
                e.setBuffer(self.counters, offset: 0, index: 0)
                e.setBuffer(self.dispatchArgs, offset: 0, index: 1)
                e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride,
                           index: 2)
            }
            var clearCapacity = UInt32(self.softMapCapacity)
            dispatch1D(encoder, "softmap_clear", self.softMapCapacity) { e in
                e.setBuffer(self.softMapKeyA, offset: 0, index: 0)
                e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride,
                           index: 1)
                e.setBytes(&clearCapacity, length: 4, index: 2)
            }
            dispatch1D(encoder, "softmap_insert", Int(P.maxSoft)) { e in
                e.setBuffer(self.softContacts, offset: 0, index: 0)
                e.setBuffer(self.softMapKeyA, offset: 0, index: 1)
                e.setBuffer(self.softMapKeyB, offset: 0, index: 2)
                e.setBuffer(self.softMapVal, offset: 0, index: 3)
                e.setBuffer(self.counters, offset: 0, index: 4)
                e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride,
                           index: 5)
            }
        }

        func encodePlanarPass(_ encoder: MTLComputeCommandEncoder,
                              pairs: MTLBuffer,
                              pairCounts: MTLBuffer,
                              activeColor: Int? = nil) {
            let reducePSO = ps(activeColor == nil
                || !usePlanarBodyIncidence
                ? "dat_reduce" : "dat_reduce_incident_color")
            encoder.setComputePipelineState(reducePSO)
            encoder.setBuffer(pairs, offset: 0, index: 0)
            encoder.setBuffer(posLin, offset: 0, index: 1)
            encoder.setBuffer(ogcPrevBuf, offset: 0, index: 2)
            encoder.setBuffer(trisBuf, offset: 0, index: 3)
            encoder.setBuffer(edgesBuf, offset: 0, index: 4)
            encoder.setBuffer(planarDATTBuf, offset: 0, index: 5)
            encoder.setBuffer(pairCounts, offset: 0, index: 6)
            encoder.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride,
                             index: 7)
            encoder.setBuffer(counters, offset: 0, index: 8)
            if let activeColor {
                var color = UInt32(activeColor)
                if usePlanarBodyIncidence {
                    encoder.setBuffer(self.colorList, offset: 0, index: 9)
                    encoder.setBuffer(self.colorStart, offset: 0, index: 10)
                    encoder.setBytes(&color, length: 4, index: 11)
                    encoder.setBuffer(self.planarDATBodyStartBuf, offset: 0,
                                      index: 12)
                    encoder.setBuffer(self.planarDATBodyCountBuf, offset: 0,
                                      index: 13)
                    encoder.setBuffer(self.planarDATBodyPairsBuf, offset: 0,
                                      index: 14)
                    let threads = self.lastColorCounts[activeColor] * 8
                    encoder.dispatchThreadgroups(
                        MTLSize(width: (threads + 63) / 64,
                                height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 64, height: 1,
                                                       depth: 1))
                } else {
                    encoder.setBuffer(colorsA, offset: 0, index: 9)
                    encoder.setBytes(&color, length: 4, index: 10)
                    encoder.dispatchThreadgroups(
                        indirectBuffer: planarDATArgsBuf,
                        indirectBufferOffset: 0,
                        threadsPerThreadgroup: MTLSize(width: 64, height: 1,
                                                       depth: 1))
                }
                encoder.setComputePipelineState(ps("dat_apply_color"))
                encoder.setBuffer(self.posLin, offset: 0, index: 0)
                encoder.setBuffer(self.ogcPrevBuf, offset: 0, index: 1)
                encoder.setBuffer(self.colorList, offset: 0, index: 2)
                encoder.setBuffer(self.colorStart, offset: 0, index: 3)
                encoder.setBuffer(self.clothGroupBuf, offset: 0, index: 4)
                encoder.setBuffer(self.planarDATTBuf, offset: 0, index: 5)
                encoder.setBuffer(pairCounts, offset: 0, index: 6)
                encoder.setBuffer(self.counters, offset: 0, index: 7)
                encoder.setBytes(&P,
                                 length: MemoryLayout<SimParamsGPU>.stride,
                                 index: 8)
                encoder.setBytes(&color, length: 4, index: 9)
                encoder.dispatchThreadgroups(
                    indirectBuffer: self.colorArgs,
                    indirectBufferOffset: activeColor * 12,
                    threadsPerThreadgroup: MTLSize(width: 64, height: 1,
                                                   depth: 1))
            } else {
                encoder.setBuffer(colorsA, offset: 0, index: 9)
                var fullPass = UInt32.max
                encoder.setBytes(&fullPass, length: 4, index: 10)
                encoder.dispatchThreadgroups(
                    indirectBuffer: planarDATArgsBuf,
                    indirectBufferOffset: 0,
                    threadsPerThreadgroup: MTLSize(width: 64, height: 1,
                                                   depth: 1))
                dispatch1D(encoder, "dat_apply", numParticles) { e in
                e.setBuffer(self.posLin, offset: 0, index: 0)
                e.setBuffer(self.ogcPrevBuf, offset: 0, index: 1)
                e.setBuffer(self.particleIdxBuf, offset: 0, index: 2)
                e.setBuffer(self.planarDATTBuf, offset: 0, index: 3)
                e.setBuffer(pairCounts, offset: 0, index: 4)
                e.setBuffer(self.counters, offset: 0, index: 5)
                e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride,
                           index: 6)
                }
            }
        }

        func encodeTetPredictorInversion(
            _ encoder: MTLComputeCommandEncoder
        ) {
            guard numTets > 0 else { return }
            dispatch1D(encoder, "dat_reduce_tet_inversion", numTets) { e in
                e.setBuffer(self.posLin, offset: 0, index: 0)
                e.setBuffer(self.initLin, offset: 0, index: 1)
                e.setBuffer(self.tets, offset: 0, index: 2)
                e.setBuffer(self.planarDATTBuf, offset: 0, index: 3)
                e.setBuffer(self.counters, offset: 0, index: 4)
                e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride,
                           index: 5)
            }
            dispatch1D(encoder, "dat_apply_tet_inversion", numBodies) { e in
                e.setBuffer(self.posLin, offset: 0, index: 0)
                e.setBuffer(self.initLin, offset: 0, index: 1)
                e.setBuffer(self.shape, offset: 0, index: 2)
                e.setBuffer(self.planarDATTBuf, offset: 0, index: 3)
                e.setBuffer(self.counters, offset: 0, index: 4)
                e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride,
                           index: 5)
            }
        }

        // Broadphase
        if profiling { try stage("bp-count") }
        dispatch1D(enc, "bp_count", Int(P.numHashed)) { e in
            e.setBuffer(self.posLin, offset: 0, index: 0)
            e.setBuffer(self.hashedIdx, offset: 0, index: 1)
            e.setBuffer(self.cellCount, offset: 0, index: 2)
            e.setBuffer(self.bodyCellSlot, offset: 0, index: 3)
            e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 4)
            e.setBuffer(self.colliderShape, offset: 0, index: 5)
            e.setBuffer(self.cellRigid, offset: 0, index: 6)
            e.setBuffer(self.colliderOwner, offset: 0, index: 7)
            e.setBuffer(self.colliderLocalPosition, offset: 0, index: 8)
            e.setBuffer(self.posAng, offset: 0, index: 9)
            e.setBuffer(self.colliderGroup, offset: 0, index: 10)
        }
        if profiling { try stage("bp-scan") }
        encodeScan(enc, input: cellCount, output: cellStart, count: gridHashSize)
        if profiling { try stage("bp-scatter") }
        dispatch1D(enc, "bp_scatter", Int(P.numHashed)) { e in
            e.setBuffer(self.hashedIdx, offset: 0, index: 0)
            e.setBuffer(self.bodyCellSlot, offset: 0, index: 1)
            e.setBuffer(self.cellStart, offset: 0, index: 2)
            e.setBuffer(self.cellBodies, offset: 0, index: 3)
            e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 4)
        }
        var hashSize32 = UInt32(gridHashSize)
        dispatch1D(enc, "bp_sort_cells", gridHashSize) { e in
            e.setBuffer(self.cellStart, offset: 0, index: 0)
            e.setBuffer(self.cellCount, offset: 0, index: 1)
            e.setBuffer(self.cellBodies, offset: 0, index: 2)
            e.setBytes(&hashSize32, length: 4, index: 3)
        }
        if profiling { try stage("bp-pairs") }
        let pairProducerCount = Int(P.numHashed + P.numGlobals)
        if pairProducerCount > 0 {
            dispatch1D(enc, "bp_count_pairs_deterministic", pairProducerCount) { e in
                e.setBuffer(self.posLin, offset: 0, index: 0)
                e.setBuffer(self.colliderShape, offset: 0, index: 1)
                e.setBuffer(self.hashedIdx, offset: 0, index: 2)
                e.setBuffer(self.cellStart, offset: 0, index: 3)
                e.setBuffer(self.cellCount, offset: 0, index: 4)
                e.setBuffer(self.cellBodies, offset: 0, index: 5)
                e.setBuffer(self.globalIdx, offset: 0, index: 6)
                e.setBuffer(self.exclusions, offset: 0, index: 7)
                e.setBytes(&nExcl, length: 4, index: 8)
                e.setBuffer(self.pairCount, offset: 0, index: 9)
                e.setBuffer(self.pairs, offset: 0, index: 10)
                e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 11)
                e.setBuffer(self.clothGroupBuf, offset: 0, index: 12)
                e.setBuffer(self.cellRigid, offset: 0, index: 13)
                e.setBuffer(self.colliderOwner, offset: 0, index: 14)
                e.setBuffer(self.colliderLocalPosition, offset: 0, index: 15)
                e.setBuffer(self.posAng, offset: 0, index: 16)
                e.setBuffer(self.colliderGroup, offset: 0, index: 17)
                e.setBuffer(
                    self.colliderSharedCollision, offset: 0, index: 18)
            }
            encodeScan(enc, input: pairCount, output: pairStart,
                       count: pairProducerCount)
            dispatch1D(enc, "bp_emit_pairs_deterministic", pairProducerCount) { e in
                e.setBuffer(self.posLin, offset: 0, index: 0)
                e.setBuffer(self.colliderShape, offset: 0, index: 1)
                e.setBuffer(self.hashedIdx, offset: 0, index: 2)
                e.setBuffer(self.cellStart, offset: 0, index: 3)
                e.setBuffer(self.cellCount, offset: 0, index: 4)
                e.setBuffer(self.cellBodies, offset: 0, index: 5)
                e.setBuffer(self.globalIdx, offset: 0, index: 6)
                e.setBuffer(self.exclusions, offset: 0, index: 7)
                e.setBytes(&nExcl, length: 4, index: 8)
                e.setBuffer(self.pairStart, offset: 0, index: 9)
                e.setBuffer(self.pairs, offset: 0, index: 10)
                e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 11)
                e.setBuffer(self.clothGroupBuf, offset: 0, index: 12)
                e.setBuffer(self.cellRigid, offset: 0, index: 13)
                e.setBuffer(self.colliderOwner, offset: 0, index: 14)
                e.setBuffer(self.colliderLocalPosition, offset: 0, index: 15)
                e.setBuffer(self.posAng, offset: 0, index: 16)
                e.setBuffer(self.colliderGroup, offset: 0, index: 17)
                e.setBuffer(
                    self.colliderSharedCollision, offset: 0, index: 18)
            }
        }
        dispatch1D(enc, "bp_finalize_deterministic_pairs", 1) { e in
            e.setBuffer(self.pairCount, offset: 0, index: 0)
            e.setBuffer(self.pairStart, offset: 0, index: 1)
            e.setBuffer(self.counters, offset: 0, index: 2)
            e.setBuffer(self.dispatchArgs, offset: 0, index: 3)
            e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 4)
        }

        try stage("narrowphase")
        // Narrowphase (warm-start from prev manifolds + map)
        dispatchIndirect(enc, "np_collide", argsOffset: 0) { e in
            e.setBuffer(self.posLin, offset: 0, index: 0)
            e.setBuffer(self.posAng, offset: 0, index: 1)
            e.setBuffer(self.colliderShape, offset: 0, index: 2)
            e.setBuffer(self.props, offset: 0, index: 3)
            e.setBuffer(self.pairs, offset: 0, index: 4)
            e.setBuffer(self.counters, offset: 0, index: 5)
            e.setBuffer(self.manifolds, offset: 0, index: 6)
            e.setBuffer(self.prevManifolds, offset: 0, index: 7)
            e.setBuffer(self.mapKeyA, offset: 0, index: 8)
            e.setBuffer(self.mapKeyB, offset: 0, index: 9)
            e.setBuffer(self.mapVal, offset: 0, index: 10)
            e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 11)
            e.setBuffer(self.colliderShapeType, offset: 0, index: 12)
            e.setBuffer(self.spinVel, offset: 0, index: 13)
            e.setBuffer(self.velLin, offset: 0, index: 14)
            e.setBuffer(self.velAng, offset: 0, index: 15)
            e.setBuffer(self.colliderOwner, offset: 0, index: 16)
            e.setBuffer(self.colliderLocalPosition, offset: 0, index: 17)
            e.setBuffer(self.colliderLocalRotation, offset: 0, index: 18)
            e.setBuffer(self.colliderHullRange, offset: 0, index: 19)
            e.setBuffer(self.convexHullVertices, offset: 0, index: 20)
            e.setBuffer(self.colliderFriction, offset: 0, index: 21)
        }

        try stage("persistence-map")
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

            // ---- Soft contacts. Planar-DAT replaces the legacy best-four
            // VT/EE trackers with exact endpoint queries; rigid-triangle
            // records remain on their existing start-pose path. The accepted
            // Planar stream is emitted after predictor acceptance below.
        if numTris > 0 {
            if !isPlanarDAT {
            try stage("el-bin")
            encodeElementGrid(enc, clearFirst: false)
            try stage("vt-emit")
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
                e.setBuffer(self.clothGroupBuf, offset: 0, index: 25)
                e.setBuffer(self.clothVertFlag, offset: 0, index: 26)
            }
            }
            try stage("ee-emit")
            if ProcessInfo.processInfo.environment["AVBD_NO_EE"] == nil {
            dispatch1D(enc, "ee_emit", Int(P.numEdges)) { e in
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
                e.setBuffer(self.clothGroupBuf, offset: 0, index: 26)
                e.setBuffer(self.clothVertFlag, offset: 0, index: 27)
            }
            }
            }
            else {
                try stage("el-bin")
                encodeElementGrid(enc, clearFirst: false)
                try stage("dat-pairs")
                encodePlanarPairBuild(enc)
            }
            try stage("rt-emit")
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
            if !isPlanarDAT {
                try stage("softmap")
                encodeSoftContactMap(enc)
            }
        }

        try stage("warmstart")
        // Warm start joints (before body prediction; uses start-of-step poses)
        dispatch1D(enc, "warmstart_joints", numJoints + numSprings) { e in
            e.setBuffer(self.posLin, offset: 0, index: 0)
            e.setBuffer(self.posAng, offset: 0, index: 1)
            e.setBuffer(self.joints, offset: 0, index: 2)
            e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride, index: 3)
            e.setBuffer(self.springs, offset: 0, index: 4)
            e.setBuffer(self.velLin, offset: 0, index: 5)
            e.setBuffer(self.velAng, offset: 0, index: 6)
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
            var truncationMode = P.surfaceTruncationMode
            e.setBytes(&truncationMode, length: 4, index: 12)
            e.setBuffer(self.shape, offset: 0, index: 13)
            e.setBuffer(self.gravityScale, offset: 0, index: 14)
        }

        if isPlanarDAT {
            // The predictor moves all tet vertices concurrently, so protect
            // signed volume with exact cubic trajectory isolation before the
            // surface pair pass. Colored VBD updates use a cheaper affine
            // signed-volume line limit because one tet vertex moves at a
            // time under the static topology coloring.
            if P.tetInversionPreventionEnabled != 0 {
                encodeTetPredictorInversion(enc)
            }
            // Start from the exact rq neighborhood. Since the predictor caps
            // each side to R=0.5*gamma*rq, a pair outside this query cannot
            // close its positive gap before the accepted-pose query below.
            try stage("dat-predict")
            let predictorSite = PlanarDATPassSite.predictor
            planarDATPassObserverForTesting?(predictorSite)
            encodePlanarPass(enc, pairs: planarDATPairsBuf,
                             pairCounts: planarDATPairCountsBuf)

            // Newton's two-detection schedule: rebuild at the accepted
            // predictor pose, then use this exact rq set for OGC forces and
            // every per-color DAT pass. Clearing these working counts does
            // not clear the sticky peak/failure counters, so an overflow in
            // either query remains terminal and restores the failed frame.
            try stage("dat-accepted-grid")
            encodeElementGrid(enc, clearFirst: true)
            try stage("dat-accepted-pairs")
            encodePlanarPairBuild(enc)
            if usePlanarBodyIncidence {
                try stage("dat-incidence")
                encodePlanarIncidence(enc)
            }
            try stage("vt-ee-emit")
            encodePlanarContacts(
                enc, pairs: planarDATPairsBuf,
                pairCounts: planarDATPairCountsBuf,
                positions: posLin, referencePositions: initLin)
            try stage("softmap")
            encodeSoftContactMap(enc)
            try stage("dat-reanchor")
            dispatch1D(enc, "dat_reanchor", numParticles) { e in
                e.setBuffer(self.posLin, offset: 0, index: 0)
                e.setBuffer(self.particleIdxBuf, offset: 0, index: 1)
                e.setBuffer(self.ogcPrevBuf, offset: 0, index: 2)
                e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride,
                           index: 3)
            }
        }

        try stage("adjacency")
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
        dispatch1D(enc, "adj_sort", numBodies) { e in
            e.setBuffer(self.adjStart, offset: 0, index: 0)
            e.setBuffer(self.degrees, offset: 0, index: 1)
            e.setBuffer(self.adjList, offset: 0, index: 2)
            e.setBytes(&nb32, length: 4, index: 3)
        }
        if usesDynamicColoring
            && numBodies > Self.orderedColoringBodyLimit {
            dispatch1D(enc, "adj_extract_neighbors", numBodies) { e in
                e.setBuffer(self.joints, offset: 0, index: 0)
                e.setBuffer(self.springs, offset: 0, index: 1)
                e.setBuffer(self.manifolds, offset: 0, index: 2)
                e.setBuffer(self.adjStart, offset: 0, index: 3)
                e.setBuffer(self.degrees, offset: 0, index: 4)
                e.setBuffer(self.adjList, offset: 0, index: 5)
                e.setBuffer(self.adjNeighbor, offset: 0, index: 6)
                e.setBytes(&nb32, length: 4, index: 7)
            }
        }

        let noRSplit = ProcessInfo.processInfo.environment[
            "AVBD_NO_RSPLIT"] != nil
        let useCompactManifoldSolve = usesDynamicColoring
            && numParticles == 0
            && numBodies > Self.orderedColoringBodyLimit
            && !noRSplit
        let useWideRigidSplit = useCompactManifoldSolve
            && ProcessInfo.processInfo.environment[
                "AVBD_RIGID_SPLIT8"] == nil
        var primalThreadsPerBody: UInt32 = usesDynamicColoring && noRSplit
            ? 1 : (useWideRigidSplit ? 16 : 8)

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
            try stage("coloring")
            var src = colorsA, dst = colorsB
            let parallelColorPasses = 20
            for pass in 0..<parallelColorPasses {
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
                    e.setBuffer(self.adjNeighbor, offset: 0, index: 16)
                }
                swap(&src, &dst)
            }
            let finalColors = src
            // Ordered parallel relaxation is fast on the warm-started graph,
            // but a newly formed long contact chain can need one hop per
            // pass. If the last parallel pass still changed anything, finish
            // with an exact in-place serial greedy sweep on the GPU. This is
            // rare after settling and guarantees that validation never sees
            // an arbitrary iteration-limit artifact.
            var finalPass = UInt32(parallelColorPasses - 1)
            dispatch1D(enc, "color_repair_greedy", 1) { e in
                e.setBuffer(self.posLin, offset: 0, index: 0)
                e.setBuffer(self.joints, offset: 0, index: 1)
                e.setBuffer(self.springs, offset: 0, index: 2)
                e.setBuffer(self.manifolds, offset: 0, index: 3)
                e.setBuffer(self.adjStart, offset: 0, index: 4)
                e.setBuffer(self.degrees, offset: 0, index: 5)
                e.setBuffer(self.adjList, offset: 0, index: 6)
                e.setBuffer(finalColors, offset: 0, index: 7)
                e.setBuffer(self.changedFlag, offset: 0, index: 8)
                e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride,
                           index: 9)
                e.setBuffer(self.tets, offset: 0, index: 10)
                e.setBuffer(self.softContacts, offset: 0, index: 11)
                e.setBuffer(self.membranes, offset: 0, index: 12)
                e.setBuffer(self.bends, offset: 0, index: 13)
                e.setBytes(&finalPass, length: 4, index: 14)
                e.setBuffer(self.adjNeighbor, offset: 0, index: 15)
            }
            dispatch1D(enc, "color_validate", numBodies) { e in
                e.setBuffer(self.posLin, offset: 0, index: 0)
                e.setBuffer(self.joints, offset: 0, index: 1)
                e.setBuffer(self.springs, offset: 0, index: 2)
                e.setBuffer(self.manifolds, offset: 0, index: 3)
                e.setBuffer(self.adjStart, offset: 0, index: 4)
                e.setBuffer(self.degrees, offset: 0, index: 5)
                e.setBuffer(self.adjList, offset: 0, index: 6)
                e.setBuffer(finalColors, offset: 0, index: 7)
                e.setBuffer(self.counters, offset: 0, index: 8)
                e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride,
                           index: 9)
                e.setBuffer(self.tets, offset: 0, index: 10)
                e.setBuffer(self.softContacts, offset: 0, index: 11)
                e.setBuffer(self.membranes, offset: 0, index: 12)
                e.setBuffer(self.bends, offset: 0, index: 13)
                e.setBuffer(self.adjNeighbor, offset: 0, index: 14)
            }
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
                e.setBytes(&primalThreadsPerBody, length: 4, index: 3)
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
        // Narrowphase is finished with prevManifolds at this point. Reuse
        // that 704-byte/slot buffer for a 48-byte header stream followed by
        // eight slot-major 64-byte contact streams. This gives the repeated
        // solver passes a compact working set without another allocation.
        let solverContactOffset = maxPairs * 48
        var compactManifoldFlag: UInt32 = useCompactManifoldSolve ? 1 : 0
        if useCompactManifoldSolve {
            try stage("manifold-pack")
            dispatchIndirect(enc, "manifold_solver_pack", argsOffset: 0) { e in
                e.setBuffer(self.manifolds, offset: 0, index: 0)
                e.setBuffer(self.prevManifolds, offset: 0, index: 1)
                e.setBuffer(self.prevManifolds,
                            offset: solverContactOffset, index: 2)
                e.setBuffer(self.counters, offset: 0, index: 3)
                e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride,
                           index: 4)
            }
        }
        try stage("solver-iterations")
        let persistPSO = ps("solve_persistent")
        let multiPSO = ps("solve_persistent_multi")
        // Multi-threadgroup persistent path: whole solve in one dispatch of
        // a few co-resident threadgroups (device-scope spin barrier). Covers
        // small AND mid scenes; huge scenes amortize per-color dispatches.
        // EXPERIMENTAL, opt-in: deadlocked on first contact with reality —
        // Metal guarantees no forward progress between threadgroups, and the
        // arrive-and-spin barrier wedged the queue. Kept for further study.
        let multiOK = ProcessInfo.processInfo.environment["AVBD_MULTI"] != nil
        // The dispatched solver is the canonical path at every scene size.
        // Switching to a different numerical kernel at the threadgroup-size
        // boundary made otherwise identical vectorized RL replicas diverge.
        // Keep the scalar persistent kernel as an explicit benchmark/debug
        // mode only; production simulation and replay must share one path.
        let persistentRequested = ProcessInfo.processInfo.environment[
            "AVBD_PERSIST"] != nil
            && ProcessInfo.processInfo.environment["AVBD_NO_PERSIST"] == nil
        if multiOK && !isPlanarDAT && numBodies <= 4096 {
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
        } else if persistentRequested && !isPlanarDAT
                    && numBodies <= persistPSO.maxTotalThreadsPerThreadgroup {
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
        // Dynamic indirect dispatch sizes are produced by this same command
        // buffer. Dispatch every representable color: zero-count indirect
        // calls are empty, while relying on an asynchronous CPU color bound
        // previously required a racy in-place "tail" solve.
        let colorBound = usesDynamicColoring
            ? AVBD_MAX_COLORS : staticUsedColors
        // Rigid scenes use the 8-lane cooperative split primal too: dense
        // piles average 12-27 manifolds/body and the one-thread-per-body
        // kernel was latency-bound walking them serially (boxpile x12
        // solve-primal 9.5 ms). AVBD_NO_RSPLIT restores the old kernel.
        let primalPSO: MTLComputePipelineState
        if usesDynamicColoring && noRSplit {
            primalPSO = ps("primal_solve")
        } else if useWideRigidSplit {
            primalPSO = ps("primal_rigid_split")
        } else {
            primalPSO = ps("primal_particles_split")
        }
        // static palettes have fixed per-color counts: dispatch directly
        // (8 lanes per body for the split kernel)
        let splitSizes: [Int] = usesDynamicColoring ? []
            : (0..<staticUsedColors).map {
                lastColorCounts[$0] * Int(primalThreadsPerBody)
            }
        for it in 0..<settings.iterations {
            var writeBackCompactManifoldFlag: UInt32 =
                useCompactManifoldSolve && it + 1 == settings.iterations ? 1 : 0
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
                if primalThreadsPerBody > 1 {
                    enc.setBuffer(self.prevManifolds, offset: 0, index: 25)
                    enc.setBuffer(self.prevManifolds,
                                  offset: solverContactOffset, index: 26)
                    enc.setBytes(&compactManifoldFlag, length: 4, index: 27)
                }
            }
            _ = it
            if profiling { try stage("solve-primal") ; enc.setComputePipelineState(primalPSO)
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
                if primalThreadsPerBody > 1 {
                    enc.setBuffer(self.prevManifolds, offset: 0, index: 25)
                    enc.setBuffer(self.prevManifolds,
                                  offset: solverContactOffset, index: 26)
                    enc.setBytes(&compactManifoldFlag, length: 4, index: 27)
                }
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
                if isPlanarDAT {
                    // A VBD color is a Gauss-Seidel iteration in DAT's
                    // terminology: later colors must never observe a trial
                    // pose that already crossed one of the fixed division
                    // planes. Reduce and apply immediately after every color.
                    let passSite = PlanarDATPassSite.color(
                        iteration: it, color: c)
                    planarDATPassObserverForTesting?(passSite)
                    encodePlanarPass(
                        enc, pairs: planarDATPairsBuf,
                        pairCounts: planarDATPairCountsBuf,
                        activeColor: c)
                    if c + 1 < colorBound {
                        // Body-local DAT reuses the primal slots through 14.
                        // Restore every overwritten binding before the next
                        // color; cIdx and the higher bindings stay intact.
                        enc.setComputePipelineState(primalPSO)
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
                    }
                }
            }
            if profiling { try stage("solve-dual") }
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
                e.setBuffer(self.prevManifolds, offset: 0, index: 10)
                e.setBuffer(self.prevManifolds,
                            offset: solverContactOffset, index: 11)
                e.setBytes(&compactManifoldFlag, length: 4, index: 12)
                e.setBytes(&writeBackCompactManifoldFlag,
                           length: 4, index: 13)
            }
            // OGC conditional refresh (paper Alg 3): a one-thread kernel
            // turns the exceed counter into indirect dispatch args, so the
            // refresh runs ONLY in iterations where >1% of particles were
            // bound-limited — settled scenes pay an empty dispatch
            if !isPlanarDAT && numParticles > 0
                && it + 1 < settings.iterations {
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
        try stage("finalize")
        if isPlanarDAT {
            dispatch1D(enc, "dat_restore_failed_surface", numBodies) { e in
                e.setBuffer(self.posLin, offset: 0, index: 0)
                e.setBuffer(self.initLin, offset: 0, index: 1)
                e.setBuffer(self.shape, offset: 0, index: 2)
                e.setBuffer(self.counters, offset: 0, index: 3)
                e.setBytes(&P, length: MemoryLayout<SimParamsGPU>.stride,
                           index: 4)
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
            e.setBuffer(self.shape, offset: 0, index: 8)
            e.setBuffer(self.gravityScale, offset: 0, index: 9)
        }
        enc.endEncoding()
        var visc = settings.clothViscosity
        if let env = ProcessInfo.processInfo.environment["AVBD_VISC"],
           let v = Float(env) { visc = v }
        if numTris > 0 && visc > 0 {
            // scratch copy for the gather (inertLin is dead after the solve
            // and rewritten by next frame's warmstart)
            guard let blit = cmd1.makeBlitCommandEncoder() else {
                throw RuntimeFailure.commandEncoderCreation(
                    operation: "physics", stage: "viscosity-copy",
                    frame: submittedFrame)
            }
            blit.label = "viscosity-copy"
            blit.copy(from: velLin, sourceOffset: 0,
                      to: inertLin, destinationOffset: 0, size: numBodies * 16)
            blit.endEncoding()
            guard deniedEncoderStageForTesting != "velocity-smoothing",
                  let e2 = cmd1.makeComputeCommandEncoder() else {
                throw RuntimeFailure.commandEncoderCreation(
                    operation: "physics", stage: "velocity-smoothing",
                    frame: submittedFrame)
            }
            e2.label = "velocity-smoothing"
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
            e2.dispatchThreadgroups(
                MTLSize(width: (max(1, numParticles) + 63) / 64,
                        height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            e2.endEncoding()
        }
        guard let readbackEncoder = cmd1.makeBlitCommandEncoder() else {
            throw RuntimeFailure.commandEncoderCreation(
                operation: "physics", stage: "readback",
                frame: submittedFrame)
        }
        readbackEncoder.label = "readback"
        if let source = dynColorSrc {
            readbackEncoder.copy(from: source, sourceOffset: 0,
                                 to: colorsA, destinationOffset: 0,
                                 size: numBodies * 4)
        }
        let counterSnapshot = counterReadbacks[(submittedFrame - 1) & 1]
        readbackEncoder.copy(from: counters, sourceOffset: 0,
                             to: counterSnapshot, destinationOffset: 0,
                             size: counters.length)
        readbackEncoder.endEncoding()
        dynColorSrc = nil

        // CPU-authored kinematic poses are advanced only once the entire
        // frame encoded successfully, so creation failures do not consume
        // simulation time or move spinners.
        advanceSpinners()
        frameIndex = submittedFrame
        let submission = StepSubmission(
            commandBuffer: cmd1, counterSnapshot: counterSnapshot,
            frame: submittedFrame,
            usesDynamicColoring: usesDynamicColoring,
            softCapacity: Int(P.maxSoft),
            planarDATCapacity: Int(P.maxPlanarDATPairs))
        cmd1.commit()

        // Buffer identity is a submitted-frame contract even if retirement
        // later discovers a terminal Metal or capacity failure.
        swap(&manifolds, &prevManifolds)
        if numTris > 0 { swap(&softContacts, &prevSoftContacts) }

        if profiling {
            try retire(submission)
        } else {
            inflight.append(submission)
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

    }

    /// Advance compatibility rate motors and kinematic spinners.
    private func advanceSpinners() {
        if !rateMotors.isEmpty {
            let jp = joints.contents().bindMemory(
                to: JointGPU.self, capacity: max(1, numJoints))
            for (joint, rate) in rateMotors {
                var target = jp[joint].motor.x + rate * settings.dt
                target -= (2 * Float.pi)
                    * (target / (2 * .pi)).rounded()
                jp[joint].motor.x = target
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

    public func bodyDiagonalInertia(_ i: Int) -> F3 {
        sync()
        let p = props.contents().bindMemory(to: SIMD4<Float>.self,
                                            capacity: numBodies)
        return F3(p[i].x, p[i].y, p[i].z)
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

    /// Read an arbitrary set of body states after one GPU fence. The order
    /// of the result matches `bodyIndices`, which makes this directly usable
    /// for row-major environment observations.
    public func bodyStates(_ bodyIndices: [Int]) -> [RigidBodyState] {
        guard !bodyIndices.isEmpty else { return [] }
        sync()
        let pl = posLin.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        let pa = posAng.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        let vl = velLin.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        let va = velAng.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        return bodyIndices.map { i in
            precondition(i >= 0 && i < numBodies, "body index out of range")
            return RigidBodyState(
                position: F3(pl[i].x, pl[i].y, pl[i].z),
                rotation: Quat(real: pa[i].w, imag: F3(pa[i].x, pa[i].y, pa[i].z)),
                linearVelocity: F3(vl[i].x, vl[i].y, vl[i].z),
                angularVelocity: F3(va[i].x, va[i].y, va[i].z))
        }
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

    /// Ray-cast against collision primitives (CPU, shared buffers). Returns
    /// the owning body and a body-local anchor suitable for dragging.
    public func pick(origin: F3, dir: F3) -> (body: Int, local: F3)? {
        sync()
        let pl = posLin.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        let pa = posAng.contents().bindMemory(to: SIMD4<Float>.self, capacity: numBodies)
        let sh = colliderShape.contents().bindMemory(to: SIMD4<Float>.self,
                                                     capacity: numColliders)
        let st = colliderShapeType.contents().bindMemory(to: UInt32.self,
                                                         capacity: numColliders)
        let owners = colliderOwner.contents().bindMemory(to: UInt32.self,
                                                         capacity: numColliders)
        let lp = colliderLocalPosition.contents().bindMemory(to: SIMD4<Float>.self,
                                                             capacity: numColliders)
        let lq = colliderLocalRotation.contents().bindMemory(to: SIMD4<Float>.self,
                                                             capacity: numColliders)
        var bestT = Float.infinity
        var best: (Int, F3)? = nil
        for i in 0..<numColliders {
            let body = Int(owners[i])
            guard pl[body].w > 0 else { continue }
            let bodyQ = Quat(real: pa[body].w,
                             imag: F3(pa[body].x, pa[body].y, pa[body].z))
            let localQ = Quat(real: lq[i].w, imag: F3(lq[i].x, lq[i].y, lq[i].z))
            let q = bodyQ * localQ
            let center = F3(pl[body].x, pl[body].y, pl[body].z)
                + bodyQ.act(F3(lp[i].x, lp[i].y, lp[i].z))
            let inv = q.conjugate
            let o = inv.act(origin - center)
            let d = inv.act(dir)
            var hitT: Float?
            if (st[i] & 0xF) != 0 {
                // sphere/torus: pick against bounding sphere (good enough for grab)
                let r = abs(sh[i].w)
                let b = dot(o, d)
                let cc = dot(o, o) - r * r
                let disc = b * b - cc
                if disc < 0 { continue }
                let t = -b - disc.squareRoot()
                if t >= 0 { hitT = t }
            } else {
                let half = F3(sh[i].x, sh[i].y, sh[i].z) * 0.5
                var tEnter: Float = 0
                var tExit = Float.infinity
                var hit = true
                for k in 0..<3 {
                    if abs(d[k]) < 1e-6 {
                        if o[k] < -half[k] || o[k] > half[k] {
                            hit = false
                            break
                        }
                        continue
                    }
                    var t0 = (-half[k] - o[k]) / d[k]
                    var t1 = (half[k] - o[k]) / d[k]
                    if t0 > t1 { swap(&t0, &t1) }
                    tEnter = max(tEnter, t0)
                    tExit = min(tExit, t1)
                    if tEnter > tExit { hit = false; break }
                }
                if hit {
                    let t = tEnter >= 0 ? tEnter : tExit
                    if t >= 0 { hitT = t }
                }
            }
            guard let t = hitT else { continue }
            if t >= 0 && t < bestT {
                bestT = t
                let worldHit = origin + dir * t
                let bodyLocal = bodyQ.conjugate.act(
                    worldHit - F3(pl[body].x, pl[body].y, pl[body].z))
                best = (body, bodyLocal)
            }
        }
        return best
    }

    /// Encode instance-transform building into a render command buffer.
    public func encodeBuildInstances(_ cmd: MTLCommandBuffer, instances: MTLBuffer,
                                     colorMode: UInt32 = 0) {
        do {
            try encodeBuildInstancesChecked(
                cmd, instances: instances, colorMode: colorMode)
        } catch {
            fatalError("Render-instance encoding failed: \(error.localizedDescription)")
        }
    }

    /// Checked render-instance encoding. A render encoder failure invalidates
    /// the visual frame, but it does not poison an otherwise healthy physics
    /// solver because the caller owns this separate render command buffer.
    public func encodeBuildInstancesChecked(
        _ cmd: MTLCommandBuffer, instances: MTLBuffer,
        colorMode: UInt32 = 0
    ) throws {
        // The caller may own a different Metal queue. Retire all preceding
        // physics before that queue reads solver buffers; the caller must in
        // turn finish this render command before submitting the next physics
        // mutation (AVBDApp enforces that at the frame boundary).
        try synchronize()
        guard deniedEncoderStageForTesting != "render-instances",
              let enc = cmd.makeComputeCommandEncoder() else {
            throw RuntimeFailure.commandEncoderCreation(
                operation: "render instances", stage: "build-instances",
                frame: frameIndex)
        }
        enc.label = "build-instances"
        var cm = colorMode
        if renderRigidBodyCount > 0 {
            var nb = UInt32(renderRigidBodyCount)
            let p = ps("build_instances")
            enc.setComputePipelineState(p)
            enc.setBuffer(posLin, offset: 0, index: 0)
            enc.setBuffer(posAng, offset: 0, index: 1)
            enc.setBuffer(colliderShape, offset: 0, index: 2)
            enc.setBuffer(instances, offset: 0, index: 3)
            enc.setBytes(&nb, length: 4, index: 4)
            enc.setBytes(&cm, length: 4, index: 5)
            enc.setBuffer(colorsA, offset: 0, index: 6)
            enc.setBuffer(colliderShapeType, offset: 0, index: 7)
            enc.setBuffer(renderBodyIdxBuf, offset: 0, index: 8)
            enc.setBuffer(colliderOwner, offset: 0, index: 9)
            enc.setBuffer(colliderLocalPosition, offset: 0, index: 10)
            enc.setBuffer(colliderLocalRotation, offset: 0, index: 11)
            enc.setBuffer(colliderRenderColor, offset: 0, index: 12)
            enc.dispatchThreadgroups(MTLSize(width: (renderRigidBodyCount + 255) / 256,
                                             height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256,
                                                                    height: 1,
                                                                    depth: 1))
        }
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

    /// Visual-only CAD triangles attached to live rigid bodies. Positions and
    /// normals are local to each owning body's inertial frame.
    public var renderRigidMeshSurface: (
        vertices: MTLBuffer, vertexCount: Int,
        positions: MTLBuffer, rotations: MTLBuffer
    )? {
        guard rigidMeshVertexCount > 0 else { return nil }
        return (rigidMeshVertexBuf, rigidMeshVertexCount, posLin, posAng)
    }

    public var bodyCount: Int { numBodies }

    /// Active rigid contact owner pairs from the last completed step. This is
    /// a diagnostic/readback API for asset validation and tests; training
    /// loops should use task observations and avoid the synchronization.
    public func activeRigidContactPairs() -> [(Int, Int)] {
        sync()
        let m = prevManifolds.contents().bindMemory(
            to: ManifoldGPU.self, capacity: maxPairs)
        var result: [(Int, Int)] = []
        result.reserveCapacity(lastNumPairs)
        for i in 0..<lastNumPairs where m[i].header.z > 0 {
            result.append((Int(m[i].header.x), Int(m[i].header.y)))
        }
        return result
    }

    /// Solver normal-load magnitude for every active rigid manifold from the
    /// last completed step. A narrowphase manifold can remain active inside
    /// the contact margin while carrying essentially no support force; the
    /// accumulated `-lambda.x` separates that state from a load-bearing
    /// contact without using geometric height heuristics.
    public func activeRigidContactNormalLoads()
        -> [(bodyA: Int, bodyB: Int, normalLoad: Float)] {
        sync()
        let manifolds = prevManifolds.contents().bindMemory(
            to: ManifoldGPU.self, capacity: maxPairs)
        var result = [(bodyA: Int, bodyB: Int, normalLoad: Float)]()
        result.reserveCapacity(lastNumPairs)
        for index in 0..<lastNumPairs
            where manifolds[index].header.z > 0 {
            var contacts = manifolds[index].contacts
            let contactCount = min(
                Int(manifolds[index].header.z), AVBD_MAX_CONTACTS)
            let normalLoad = withUnsafeBytes(of: &contacts) { bytes in
                let values = bytes.bindMemory(to: ContactGPU.self)
                return (0..<contactCount).reduce(Float(0)) {
                    $0 + max(-values[$1].lambda.x, 0)
                }
            }
            result.append((
                bodyA: Int(manifolds[index].header.x),
                bodyB: Int(manifolds[index].header.y),
                normalLoad: normalLoad))
        }
        return result
    }

    /// Deepest active rigid-contact violation from the last completed step.
    /// Intended for convergence diagnostics in dense cable/contact rigs.
    public func debugWorstRigidContactPenetration()
        -> (bodyA: Int, bodyB: Int, depth: Float)? {
        sync()
        let manifolds = prevManifolds.contents().bindMemory(
            to: ManifoldGPU.self, capacity: maxPairs)
        let positions = posLin.contents().bindMemory(
            to: SIMD4<Float>.self, capacity: numBodies)
        let rotations = posAng.contents().bindMemory(
            to: SIMD4<Float>.self, capacity: numBodies)
        var worst: (bodyA: Int, bodyB: Int, depth: Float)?
        for index in 0..<lastNumPairs where manifolds[index].header.z > 0 {
            let manifold = manifolds[index]
            let a = Int(manifold.header.x), b = Int(manifold.header.y)
            let qA = Quat(real: rotations[a].w,
                          imag: F3(rotations[a].x, rotations[a].y, rotations[a].z))
            let qB = Quat(real: rotations[b].w,
                          imag: F3(rotations[b].x, rotations[b].y, rotations[b].z))
            let pA = F3(positions[a].x, positions[a].y, positions[a].z)
            let pB = F3(positions[b].x, positions[b].y, positions[b].z)
            let roundA = (manifold.header.w & 2) != 0
            let roundB = (manifold.header.w & 4) != 0
            let normal = F3(manifold.basisN.x, manifold.basisN.y,
                            manifold.basisN.z)
            var contacts = manifold.contacts
            let count = min(Int(manifold.header.z), AVBD_MAX_CONTACTS)
            withUnsafeBytes(of: &contacts) { bytes in
                let values = bytes.bindMemory(to: ContactGPU.self)
                for contact in values.prefix(count) {
                    let rA = F3(contact.rA.x, contact.rA.y, contact.rA.z)
                    let rB = F3(contact.rB.x, contact.rB.y, contact.rB.z)
                    let xA = pA + (roundA ? rA : qA.act(rA))
                    let xB = pB + (roundB ? rB : qB.act(rB))
                    let depth = max(0, -(dot(normal, xA - xB)
                                         + settings.collisionMargin))
                    if depth > (worst?.depth ?? 0) {
                        worst = (a, b, depth)
                    }
                }
            }
        }
        return worst
    }

    /// Max threadgroup width of the persistent solver kernel (scenes at or
    /// under this run the whole solve in one dispatch).
    public var persistentCapacity: Int { ps("solve_persistent").maxTotalThreadsPerThreadgroup }

    /// Max constraint error: hard-joint violation + contact penetration depth.
    public func maxConstraintError() -> Float {
        do {
            return try maxConstraintErrorChecked()
        } catch {
            fatalError("Constraint diagnostics failed: \(error.localizedDescription)")
        }
    }

    /// Checked diagnostics never translate a Metal failure into a false
    /// perfect zero error.
    public func maxConstraintErrorChecked() throws -> Float {
        try synchronize()
        syncParams()
        guard let cmd = makeRuntimeCommandBuffer() else {
            throw latch(.commandBufferCreation(
                operation: "constraint diagnostics", frame: frameIndex))
        }
        guard deniedEncoderStageForTesting != "constraint diagnostics",
              let enc = cmd.makeComputeCommandEncoder() else {
            throw latch(.commandEncoderCreation(
                operation: "constraint diagnostics", stage: "diagnostics",
                frame: frameIndex))
        }
        cmd.label = "AVBD constraint diagnostics"
        enc.label = "constraint diagnostics"
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
        try waitForCompletion(
            cmd, operation: "constraint diagnostics", frame: frameIndex)
        let bits = diag.contents().load(as: UInt32.self)
        return Float(bitPattern: bits)
    }
}
