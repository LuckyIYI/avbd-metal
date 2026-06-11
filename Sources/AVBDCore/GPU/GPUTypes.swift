import simd

// Swift mirrors of the Metal struct layouts (00_common.metal).
// All structs are float4/uint4-granular so layouts match exactly.

public let AVBD_MAX_COLORS = 64
public let AVBD_MAX_CONTACTS = 8

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
}

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
    public var motor: SIMD4<Float> = .zero
    public var limits: SIMD4<Float> = .zero
}

public struct TetGPU {
    public var ids: SIMD4<UInt32> = .zero
    public var r0: SIMD4<Float> = .zero
    public var r1: SIMD4<Float> = .zero
    public var r2: SIMD4<Float> = .zero
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

public struct ManifoldGPU {
    public var header: SIMD4<UInt32> = .zero  // bodyA, bodyB, numContacts, active
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
    public static let colorBase = 8
    public static let total = 8 + 2 * AVBD_MAX_COLORS
}
