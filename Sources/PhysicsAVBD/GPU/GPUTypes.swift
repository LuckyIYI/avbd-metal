import simd
import SimCore

// Swift mirrors of the Metal struct layouts (00_common.metal).
// All structs are float4/uint4-granular so layouts match exactly.

public let AVBD_MAX_COLORS = 64
public let AVBD_MAX_CONTACTS = 8
/// Default compact safety-pair budget. Unlike Newton's per-owner 32/64
/// slices, this is one global stream: sparse rows can lend capacity to dense
/// rows, while the raw demand counter still makes overflow fail closed.
// Allocation heuristics, not silent per-owner caps: both exact endpoint
// queries append into one reusable global stream and fail closed if either
// query's total demand exceeds it. Keep enough headroom for folded multilayer
// scenes while retaining memory linear in mesh size.
public let AVBD_PLANAR_DAT_PAIRS_PER_PARTICLE = 64
public let AVBD_PLANAR_DAT_PAIRS_PER_EDGE = 128

public struct SimParamsGPU {
    public var dt: Float = 1.0 / 60.0
    public var gravity: Float = -10.0
    public var alpha: Float = 0.99
    public var betaLin: Float = 5000.0
    public var betaAng: Float = 100.0
    public var gamma: Float = 0.999
    public var numBodies: UInt32 = 0
    public var numJoints: UInt32 = 0
    public var numSprings: UInt32 = 0
    public var mapCapacity: UInt32 = 0
    public var maxManifolds: UInt32 = 0
    public var maxPairs: UInt32 = 0
    public var cellSize: Float = 1.0
    public var gridHashSize: UInt32 = 0
    public var numHashed: UInt32 = 0
    public var numGlobals: UInt32 = 0
    public var maxSpeed: Float = 100
    public var lambdaMax: Float = 1.0e6
    public var iterations: UInt32 = 0
    public var numTets: UInt32 = 0
    // --- cloth / element pipeline (mirrors 00_common.metal) ---
    public var numTris: UInt32 = 0
    public var numEdges: UInt32 = 0
    public var numParticles: UInt32 = 0
    public var maxSoft: UInt32 = 0
    public var softMapCapacity: UInt32 = 0
    public var numMembranes: UInt32 = 0
    public var numBends: UInt32 = 0
    public var elemCellSize: Float = 1.0
    public var elemHashSize: UInt32 = 0
    public var rodDecayPow: Float = 0
    public var particleDamping: Float = 0
    public var numHashedRigid: UInt32 = 0
    public var elemMargin: Float = 0.01
    public var numSoftGroups: UInt32 = 0
    public var frame: UInt32 = 0
    public var frictionCombineMode: UInt32 = 0
    public var collisionMargin: Float = 0.01
    /// Retains the byte-frozen analytic-kernel parameter layout. The former
    /// sphere-patch switch is intentionally not part of the public model.
    public var contactMaterialReserved: UInt32 = 0
    public var rigidLinearDamping: Float = 0
    public var rigidAngularDamping: Float = 0
    /// 0 = disabled/no surface, 1 = legacy isotropic OGC bound,
    /// 2 = direction-aware Planar-DAT.
    public var surfaceTruncationMode: UInt32 = 0
    public var maxPlanarDATPairs: UInt32 = 0
    /// Authored cloth edges eligible for the OGC E-E force model. Planar-DAT
    /// may additionally protect synthesized tet-boundary edges.
    public var numSurfaceContactEdges: UInt32 = 0
    public var planarDATQueryRadius: Float = 0
    public var planarDATRelaxation: Float = 0.85
    /// Legacy OGC velocity-inflation cap. Kept separate from the wider
    /// Planar-DAT broadphase cell so changing safety-query geometry cannot
    /// silently widen the force model.
    public var surfaceContactCellSize: Float = 1
    /// Independent opt-in for the experimental signed-volume limiter. The
    /// Planar-DAT paper/Newton surface path does not require this extension.
    public var tetInversionPreventionEnabled: UInt32 = 0
}

extension SimParamsGPU: Equatable {}

public struct JointGPU {
    public var header: SIMD4<UInt32> = .zero  // bodyA, bodyB, broken, pad
    public var rA: SIMD4<Float> = .zero       // w = stiffnessLin
    public var rB: SIMD4<Float> = .zero       // w = stiffnessAng
    public var C0Lin: SIMD4<Float> = .zero    // w = torqueArm
    public var C0Ang: SIMD4<Float> = .zero    // w = fracture
    public var lambdaLin: SIMD4<Float> = .zero
    public var lambdaAng: SIMD4<Float> = .zero
    public var penaltyLin: SIMD4<Float> = .zero
    public var penaltyAng: SIMD4<Float> = .zero
    public var restRel: SIMD4<Float> = SIMD4(0, 0, 0, 1)
    public var hingeAxis: SIMD4<Float> = .zero
    public var motor: SIMD4<Float> = .zero       // target, effort cap, pad, gain
    public var limits: SIMD4<Float> = .zero      // lo, hi, position-PD kd, pad
    public var dynamics: SIMD4<Float> = .zero    // armature, predicted twist, explicit effort, pad
}

public struct TetGPU {
    public var ids: SIMD4<UInt32> = .zero
    public var r0: SIMD4<Float> = .zero // DmInv row 0; w = signed 6x rest volume
    public var r1: SIMD4<Float> = .zero
    public var r2: SIMD4<Float> = .zero
}

public struct SkinBindingGPU {
    public var ids: SIMD4<UInt32> = .zero
    public var weights: SIMD4<Float> = .zero
    public var restNormal: SIMD4<Float> = .zero
    public var inv0: SIMD4<Float> = .zero
    public var inv1: SIMD4<Float> = .zero
    public var inv2: SIMD4<Float> = .zero
}

public struct SkinVertexGPU {
    public var position: SIMD4<Float> = .zero
    public var normal: SIMD4<Float> = .zero
}

/// Expanded visual-only rigid-mesh corner. `position.w` stores the owning
/// body id as Float bit-pattern so the Metal vertex shader can fetch the live
/// body pose without a per-mesh draw call or duplicated transforms.
public struct RigidMeshVertexGPU {
    public var positionBody: SIMD4<Float> = .zero
    public var normal: SIMD4<Float> = .zero
    public var color: SIMD4<Float> = .zero
}

public struct SoftContactGPU {
    public var ids: SIMD4<UInt32> = .zero
    public var normal: SIMD4<Float> = .zero
    public var anchorA: SIMD4<Float> = .zero
    public var weights: SIMD4<Float> = .zero
    public var C0: SIMD4<Float> = .zero
    public var lambda: SIMD4<Float> = .zero
    public var penalty: SIMD4<Float> = .zero
}

public struct MembraneGPU {
    public var ids: SIMD4<UInt32> = .zero
    public var dm: SIMD4<Float> = .zero
    public var mat: SIMD4<Float> = .zero
}

public struct BendGPU {
    public var ids: SIMD4<UInt32> = .zero
    public var K: SIMD4<Float> = .zero
    public var mat: SIMD4<Float> = .zero
}

public struct SpringGPU {
    public var header: SIMD4<UInt32> = .zero  // bodyA, bodyB, hard flag
    public var rA: SIMD4<Float> = .zero       // w = stiffness
    public var rB: SIMD4<Float> = .zero       // w = rest
    public var dual: SIMD4<Float> = .zero     // lambda, penalty, C0
}

public struct ContactGPU {
    public var rA: SIMD4<Float> = .zero       // w = feature bits
    public var rB: SIMD4<Float> = .zero       // w = stick
    public var C0: SIMD4<Float> = .zero
    public var lambda: SIMD4<Float> = .zero
    public var penalty: SIMD4<Float> = .zero
}

/// Immutable, deduplicated convex support-map payload. Vertex positions live
/// in the existing `convexHullVertices` float4 buffer so the legacy
/// `colliderHullRange` ABI remains valid while asset-backed colliders share
/// one cooked range.
public struct ConvexHullGPU {
    /// Maximum polygon loop accepted from either a cooked asset or the legacy
    /// inline-hull reconstruction path. Keep this in sync with SimCore's
    /// public asset contract.
    public static let maximumSourceFaceVertices = 16

    /// Per-thread Sutherland-Hodgman workspace. Clipping a valid source face
    /// can temporarily add vertices, so this is deliberately larger than the
    /// authored/cooked face limit and remains a shader implementation detail.
    public static let maximumClipWorkspaceVertices = 32

    /// Optional supporting-edge manifold enrichment is bounded per pair.
    /// The certified MPR/face witness remains the fallback above this budget.
    public static let maximumSupportingEdgePairTests = 1_024

    /// vertexStart, vertexCount, faceStart, faceCount
    public var verticesFaces: SIMD4<UInt32> = .zero
    /// edgeStart, edgeCount, faceLoopStart, faceLoopCount
    public var edgesLoops: SIMD4<UInt32> = .zero
    /// Centered local AABB minimum; w is the centered support radius.
    public var boundsMinRadius: SIMD4<Float> = .zero
    /// Centered local AABB maximum; w is positive hull volume.
    public var boundsMaxVolume: SIMD4<Float> = .zero
}

/// One canonical polygon face in a cooked convex asset.
public struct ConvexFaceGPU {
    /// Outward local plane: dot(normal, point) <= distance.
    public var plane: SIMD4<Float> = .zero
    /// loopStart, loopCount, stable face index, reserved.
    public var loop: SIMD4<UInt32> = .zero
}

/// One unique undirected hull edge and its adjacent polygon faces.
public struct ConvexEdgeGPU {
    /// vertex0, vertex1, face0, face1 (UInt32.max for an invalid boundary).
    public var endpointsFaces: SIMD4<UInt32> = .zero
}

/// One node of the immutable body-local broadphase hierarchy. Bounds are
/// spheres so live rigid transforms require only one rotation/translation.
/// links: left node, right node, leaf collider, flags (bit 0 leaf, bit 1 hull).
public struct ColliderBVHNodeGPU {
    public var centerRadius: SIMD4<Float> = .zero
    public var links: SIMD4<UInt32> = .zero

    public init() {}
}

public struct ManifoldGPU {
    public var header: SIMD4<UInt32> = .zero  // bodyA, bodyB, numContacts, active
    public var colliderPair: SIMD4<UInt32> = .zero // collision-geom identity
    public var basisN: SIMD4<Float> = .zero   // w = friction
    public var basisT1: SIMD4<Float> = .zero
    public var contacts: (ContactGPU, ContactGPU, ContactGPU, ContactGPU,
                          ContactGPU, ContactGPU, ContactGPU, ContactGPU) =
        (ContactGPU(), ContactGPU(), ContactGPU(), ContactGPU(),
         ContactGPU(), ContactGPU(), ContactGPU(), ContactGPU())
}

// Counter layout (must match 00_common.metal)
public enum GPUCounters {
    public static let pairs = 0
    public static let soft = 1
    /// Candidate demand before storage-capacity clipping. Keeping this
    /// separate from `pairs` is what distinguishes a valid exact fill from
    /// a physics-invalid dropped pair.
    public static let pairCandidates = 4
    /// Element-contact demand before storage-capacity clipping.
    public static let softCandidates = 5
    /// Dynamic bodies that still share a color with a graph neighbor after
    /// the final coloring pass. Any nonzero value invalidates the solve.
    public static let colorConflicts = 6
    /// Raw compact Planar-DAT pair demand before storage clipping.
    public static let planarDATPairs = 7
    public static let planarDATVertexTrianglePairs = 8
    public static let planarDATEdgeEdgePairs = 9
    /// Element AABB exceeded the truncation mode's supported grid span.
    public static let planarDATGridOverflows = 10
    public static let planarDATInvalidAnchors = 11
    public static let planarDATTruncations = 12
    /// Diagnostic partition of `planarDATInvalidAnchors`.
    public static let planarDATVertexTriangleDegeneracies = 13
    public static let planarDATEdgeEdgeDegeneracies = 14
    public static let planarDATNonfiniteValues = 15
    public static let planarDATTetDegeneracies = 16
    public static let colorBase = 17
    /// A support-mapped query could not produce a trustworthy separation or
    /// penetration witness. Such a frame is restored and retired as failed.
    public static let convexQueryFailures = 17 + 2 * AVBD_MAX_COLORS
    /// Exact rigid-triangle narrow-phase candidates after spatial lookup.
    public static let rigidTriangleCandidates = 18 + 2 * AVBD_MAX_COLORS
    /// Supporting edge pairs considered by generic convex manifold builds.
    public static let convexEdgePairTests = 19 + 2 * AVBD_MAX_COLORS
    public static let total = 20 + 2 * AVBD_MAX_COLORS
}
