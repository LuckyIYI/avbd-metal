#include <metal_stdlib>
using namespace metal;

// ============================================================================
// AVBD Metal — common types, math, and constants.
// Shader files are concatenated in filename order and compiled at runtime.
// All struct layouts are float4/uint4-granular and mirrored in Swift
// (GPUTypes.swift); any change here must be reflected there.
// ============================================================================

#define PENALTY_MIN 1.0f
#define PENALTY_MAX 1.0e10f
// Tangential (friction) penalty cap: friction force is cone-bounded, so
// stiffness beyond ~1e6 adds nothing physically but destroys the fp32
// conditioning of the rolling mode (linear/angular cancellation).
#define PENALTY_MAX_T 1.0e6f
#define COLLISION_MARGIN 0.01f
#define STICK_THRESH 0.00001f
#define MAX_COLORS 64
// shapeType encoding: low nibble = shape kind, bit 4 = particle (3-DOF)
#define SHAPE_KIND_MASK 0xFu
#define SHAPE_PARTICLE  0x10u
#define MAX_CONTACTS 8

// Force kinds packed into adjacency entries (top 4 bits)
#define FK_JOINT 0u
#define FK_SPRING 1u
#define FK_MANIFOLD 2u
#define FK_TET 3u
#define FK_SOFT 4u          // element contact (V-T / E-E / rigid-T): 4-slot stencil
#define FK_MEMBRANE 5u      // StVK triangle membrane element
#define FK_BEND 6u          // quadratic (Bergou) bending element
#define ADJ_KIND_SHIFT 28
#define ADJ_INDEX_MASK 0x0FFFFFFFu

// Soft-contact kinds (stored in SoftContactGPU flags)
#define SC_VT 1u            // particle vertex vs cloth triangle
#define SC_RT 2u            // rigid feature point vs cloth triangle
#define SC_EE 3u            // cloth edge vs cloth edge
#define SC_RE 4u            // rigid edge point vs cloth edge

#define WORLD_BODY 0xFFFFFFFFu

// ----------------------------------------------------------------------------
// Row-major 3x3 matrix (matches CPU Mat3Rows semantics: mul(v) = rows . v)
// ----------------------------------------------------------------------------
struct M3 {
    float3 r0, r1, r2;
};

inline M3 m3_zero() { return M3{float3(0), float3(0), float3(0)}; }
inline M3 m3_diag(float3 d) { return M3{float3(d.x,0,0), float3(0,d.y,0), float3(0,0,d.z)}; }
inline M3 m3_identity() { return m3_diag(float3(1)); }
inline float3 m3_mul(M3 m, float3 v) { return float3(dot(m.r0,v), dot(m.r1,v), dot(m.r2,v)); }
inline float3 m3_col(M3 m, int i) {
    return float3(i==0?m.r0.x:(i==1?m.r0.y:m.r0.z),
                  i==0?m.r1.x:(i==1?m.r1.y:m.r1.z),
                  i==0?m.r2.x:(i==1?m.r2.y:m.r2.z));
}
inline M3 m3_transpose(M3 m) { return M3{m3_col(m,0), m3_col(m,1), m3_col(m,2)}; }
inline M3 m3_mulm(M3 a, M3 b) {
    M3 bt = m3_transpose(b); // bt rows are b's columns
    return M3{
        float3(dot(a.r0,bt.r0), dot(a.r0,bt.r1), dot(a.r0,bt.r2)),
        float3(dot(a.r1,bt.r0), dot(a.r1,bt.r1), dot(a.r1,bt.r2)),
        float3(dot(a.r2,bt.r0), dot(a.r2,bt.r1), dot(a.r2,bt.r2))};
}
inline M3 m3_add(M3 a, M3 b) { return M3{a.r0+b.r0, a.r1+b.r1, a.r2+b.r2}; }
inline M3 m3_scale(M3 a, float s) { return M3{a.r0*s, a.r1*s, a.r2*s}; }
inline M3 m3_outer(float3 a, float3 b) { return M3{b*a.x, b*a.y, b*a.z}; }
inline M3 m3_skew(float3 r) {
    // m3_mul(m3_skew(r), v) == cross(r, v)
    return M3{float3(0,-r.z,r.y), float3(r.z,0,-r.x), float3(-r.y,r.x,0)};
}
inline M3 m3_diagonalize(M3 m) {
    return m3_diag(float3(length(m3_col(m,0)), length(m3_col(m,1)), length(m3_col(m,2))));
}

// ----------------------------------------------------------------------------
// Quaternion ops on float4 (x,y,z imag; w real) — paper Eq. 20-21
// ----------------------------------------------------------------------------
inline float4 q_mul(float4 a, float4 b) {
    return float4(
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z);
}
inline float4 q_conj(float4 q) { return float4(-q.xyz, q.w); }
inline float4 q_inv(float4 q) { return q_conj(q) / dot(q, q); }
inline float3 q_rotate(float4 q, float3 v) {
    float3 t = 2.0f * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}
// q_a - q_b in tangent space (Eq. 20). The relative quaternion is
// canonicalized to the w >= 0 hemisphere: without this, a freely spinning
// hinge crossing 180 deg of relative rotation flips the constraint
// gradient's sign (double cover) and the joint explodes.
inline float3 q_sub(float4 a, float4 b) {
    float4 r = q_mul(a, q_inv(b));
    if (r.w < 0.0f) r = -r;
    return r.xyz * 2.0f;
}
// q + w (Eq. 21)
inline float4 q_addw(float4 q, float3 w) {
    float4 dq = q_mul(float4(w, 0.0f), q);
    return normalize(q + 0.5f * dq);
}
inline float3 xform(float3 p, float4 q, float3 v) { return q_rotate(q, v) + p; }

// ----------------------------------------------------------------------------
// 6x6 SPD solve via LDL^T (port of reference maths.h)
// aCross = lower-left block: rows angular, cols linear
// ----------------------------------------------------------------------------
// Fast-math-safe finiteness test (isfinite() folds to true under -ffast-math)
inline bool finite_bits(float x) {
    return (as_type<uint>(x) & 0x7F800000u) != 0x7F800000u;
}
inline bool finite3(float3 v) {
    return finite_bits(v.x) && finite_bits(v.y) && finite_bits(v.z);
}

inline void solve6x6(M3 aLin, M3 aAng, M3 aCross,
                     float3 bLin, float3 bAng,
                     thread float3 &xLin, thread float3 &xAng)
{
    // Jacobi (diagonal) preconditioning: solve (SAS)(S^-1 x) = S b with
    // S = diag(1/sqrt(A_ii)). Makes the LDL^T factorization scale-invariant,
    // which is essential in fp32 when penalty terms (up to 1e10) meet tiny
    // rod inertias in the same block.
    float s1 = rsqrt(max(aLin.r0.x, 1e-30f));
    float s2 = rsqrt(max(aLin.r1.y, 1e-30f));
    float s3 = rsqrt(max(aLin.r2.z, 1e-30f));
    float s4 = rsqrt(max(aAng.r0.x, 1e-30f));
    float s5 = rsqrt(max(aAng.r1.y, 1e-30f));
    float s6 = rsqrt(max(aAng.r2.z, 1e-30f));

    float A11 = 1.0f;
    float A21 = aLin.r1.x * s2 * s1, A22 = 1.0f;
    float A31 = aLin.r2.x * s3 * s1, A32 = aLin.r2.y * s3 * s2, A33 = 1.0f;
    float A41 = aCross.r0.x * s4 * s1, A42 = aCross.r0.y * s4 * s2, A43 = aCross.r0.z * s4 * s3, A44 = 1.0f;
    float A51 = aCross.r1.x * s5 * s1, A52 = aCross.r1.y * s5 * s2, A53 = aCross.r1.z * s5 * s3, A54 = aAng.r1.x * s5 * s4, A55 = 1.0f;
    float A61 = aCross.r2.x * s6 * s1, A62 = aCross.r2.y * s6 * s2, A63 = aCross.r2.z * s6 * s3, A64 = aAng.r2.x * s6 * s4, A65 = aAng.r2.y * s6 * s5, A66 = 1.0f;

    bLin = float3(bLin.x * s1, bLin.y * s2, bLin.z * s3);
    bAng = float3(bAng.x * s4, bAng.y * s5, bAng.z * s6);

    float L21 = A21 / A11;
    float L31 = A31 / A11;
    float L41 = A41 / A11;
    float L51 = A51 / A11;
    float L61 = A61 / A11;

    // Preconditioned matrix has unit diagonal, so pivots are O(1); anything
    // below this floor is numerically rank-deficient — clamp for a damped,
    // stable quasi-Newton step instead of an explosive one.
    float pivotEps = 1e-6f;
    float D1 = max(A11, pivotEps);
    float D2 = max(A22 - L21*L21*D1, pivotEps);

    float L32 = (A32 - L21*L31*D1) / D2;
    float L42 = (A42 - L21*L41*D1) / D2;
    float L52 = (A52 - L21*L51*D1) / D2;
    float L62 = (A62 - L21*L61*D1) / D2;

    float D3 = max(A33 - (L31*L31*D1 + L32*L32*D2), pivotEps);

    float L43 = (A43 - L31*L41*D1 - L32*L42*D2) / D3;
    float L53 = (A53 - L31*L51*D1 - L32*L52*D2) / D3;
    float L63 = (A63 - L31*L61*D1 - L32*L62*D2) / D3;

    float D4 = max(A44 - (L41*L41*D1 + L42*L42*D2 + L43*L43*D3), pivotEps);

    float L54 = (A54 - L41*L51*D1 - L42*L52*D2 - L43*L53*D3) / D4;
    float L64 = (A64 - L41*L61*D1 - L42*L62*D2 - L43*L63*D3) / D4;

    float D5 = max(A55 - (L51*L51*D1 + L52*L52*D2 + L53*L53*D3 + L54*L54*D4), pivotEps);

    float L65 = (A65 - L51*L61*D1 - L52*L62*D2 - L53*L63*D3 - L54*L64*D4) / D5;

    float D6 = max(A66 - (L61*L61*D1 + L62*L62*D2 + L63*L63*D3 + L64*L64*D4 + L65*L65*D5), pivotEps);

    float y1 = bLin.x;
    float y2 = bLin.y - L21*y1;
    float y3 = bLin.z - L31*y1 - L32*y2;
    float y4 = bAng.x - L41*y1 - L42*y2 - L43*y3;
    float y5 = bAng.y - L51*y1 - L52*y2 - L53*y3 - L54*y4;
    float y6 = bAng.z - L61*y1 - L62*y2 - L63*y3 - L64*y4 - L65*y5;

    float z1 = y1 / D1;
    float z2 = y2 / D2;
    float z3 = y3 / D3;
    float z4 = y4 / D4;
    float z5 = y5 / D5;
    float z6 = y6 / D6;

    xAng.z = z6;
    xAng.y = z5 - L65*xAng.z;
    xAng.x = z4 - L54*xAng.y - L64*xAng.z;
    xLin.z = z3 - L43*xAng.x - L53*xAng.y - L63*xAng.z;
    xLin.y = z2 - L32*xLin.z - L42*xAng.x - L52*xAng.y - L62*xAng.z;
    xLin.x = z1 - L21*xLin.y - L31*xLin.z - L41*xAng.x - L51*xAng.y - L61*xAng.z;

    // Undo preconditioning: x = S y
    xLin = float3(xLin.x * s1, xLin.y * s2, xLin.z * s3);
    xAng = float3(xAng.x * s4, xAng.y * s5, xAng.z * s6);
}

// Orthonormal basis: rows (n, t1, t2), matches CPU orthonormalBasis
inline void orthonormal(float3 n, thread float3 &t1, thread float3 &t2) {
    t1 = fabs(n.x) > fabs(n.z) ? float3(-n.y, n.x, 0) : float3(0, -n.z, n.y);
    t1 = normalize(t1);
    t2 = cross(n, t1);
}

// ----------------------------------------------------------------------------
// GPU structs (mirrored in Swift)
// ----------------------------------------------------------------------------
struct SimParams {
    float dt;
    float gravity;          // along -z when negative
    float alpha;
    float betaLin;
    float betaAng;
    float gamma;
    uint numBodies;
    uint numJoints;
    uint numSprings;
    uint mapCapacity;       // pair hash map size (pow2)
    uint maxManifolds;
    uint maxPairs;
    float cellSize;
    uint gridHashSize;      // pow2
    uint numHashed;         // bodies in spatial hash
    uint numGlobals;        // oversized/static bodies tested brute-force
    float maxSpeed;         // velocity clamp (anti-tunneling safety)
    float lambdaMax;        // dual variable bound (paper Sec 4)
    uint iterations;        // solver iterations (persistent kernel path)
    uint numTets;
    // --- cloth / element pipeline ---
    uint numTris;           // cloth triangles (contact surface elements)
    uint numEdges;          // unique cloth mesh edges (E-E contacts)
    uint numParticles;      // entries in particleIdx (V-T query points)
    uint maxSoft;           // soft contact capacity
    uint softMapCapacity;   // soft persistence map size (pow2)
    uint numMembranes;      // StVK membrane elements
    uint numBends;          // quadratic bending elements
    float elemCellSize;     // element grid cell size (>= 2x max element radius)
    uint elemHashSize;      // element grid hash size (pow2)
};

struct JointGPU {
    uint4 header;       // bodyA (WORLD_BODY=world), bodyB, broken flag, flags
    float4 rA;          // w = stiffnessLin
    float4 rB;          // w = stiffnessAng
    float4 C0Lin;       // w = torqueArm
    float4 C0Ang;       // w = fracture
    float4 lambdaLin;
    float4 lambdaAng;
    float4 penaltyLin;
    float4 penaltyAng;
    float4 restRel;     // rest relative rotation qA^-1 * qB (quaternion)
    float4 hingeAxis;   // xyz: axis in B local; w != 0 -> 1-DOF hinge
    float4 motor;       // x = target angle, y = max |lambda| (torque limit),
                        // z = lambda, w = penalty (0 = no motor)
    float4 limits;      // x = lo, y = hi (twist range; lo < hi enables)
};

struct SpringGPU {
    uint4 header;       // bodyA, bodyB, hard flag, pad
    float4 rA;          // w = stiffness (soft) / penalty cap (hard)
    float4 rB;          // w = rest length
    float4 dual;        // x = lambda, y = penalty, z = C0, w = pad
};

// Stable Neo-Hookean tetrahedron (Smith et al. 2018) on 3-DOF particles.
struct TetGPU {
    uint4 ids;          // 4 particle body indices
    float4 r0;          // DmInv row 0; w = rest volume
    float4 r1;          // DmInv row 1; w = mu  * volume
    float4 r2;          // DmInv row 2; w = lambda * volume
};

// Element contact (V-T, E-E, rigid-feature-vs-element). ONE record per
// contact point, up to 4 participating bodies with signed weights:
//   C = C0 (1-alpha) + sum_i w_i [n|t1|t2] . dq_i  (+ slot-0 angular terms
//   when slot 0 is a rigid body). Same bounded-dual / ramped-penalty /
//   warm-start treatment as rigid manifold contacts.
struct SoftContactGPU {
    uint4 ids;          // body slots; WORLD_BODY = unused slot
    float4 normal;      // xyz = contact normal (pushes slot 0 along +n), w = friction
    float4 anchorA;     // xyz = slot-0 LOCAL anchor (rigid slot 0 only); w = flag bits
    float4 weights;     // signed weight per slot (gradient_i = w_i * basis row)
    float4 C0;          // xyz = (n,t1,t2) constraint at detection; w = lambda cap
    float4 lambda;      // xyz = duals; w = persistence key A (bits)
    float4 penalty;     // xyz = penalties; w = persistence key B (bits)
};
// anchorA.w flag bits
#define SCF_RIGID_A 1u      // slot 0 is a 6-DOF rigid body (use anchorA)
#define SCF_SIDE_NEG 2u     // sign memory: vertex started on -triN side
#define SCF_KIND_SHIFT 2    // bits 2.. = SC_* kind

// StVK triangle membrane element over 3 particles (phase: real elements).
struct MembraneGPU {
    uint4 ids;          // 3 particle indices + pad
    float4 dm;          // inverse rest-shape 2x2 (column major: a c / b d)
    float4 mat;         // x = mu*area, y = lambda*area, z = area
};

// Quadratic (Bergou) bending element: rank-1 Hessian K K^T * s over the
// 4-vertex hinge stencil of an interior edge. Constant, SPD, isotropic.
struct BendGPU {
    uint4 ids;          // x0, x1 = edge ends; x2, x3 = opposite vertices
    float4 K;           // cotangent coefficient vector
    float4 mat;         // x = 3*kappa/(A1+A2)
};

struct ContactGPU {
    float4 rA;          // w = featureKey (as bits)
    float4 rB;          // w = stick flag
    float4 C0;
    float4 lambda;
    float4 penalty;
};

struct ManifoldGPU {
    uint4 header;       // bodyA, bodyB, numContacts, active
    float4 basisN;      // w = friction
    float4 basisT1;     // t2 = cross(n, t1)
    ContactGPU contacts[MAX_CONTACTS];
};

// Counters layout (single uint buffer)
#define CTR_PAIRS 0
#define CTR_SOFT 1                 // element (soft) contact count
#define CTR_COLOR_BASE 8           // MAX_COLORS entries
#define CTR_SCATTER_BASE (8 + MAX_COLORS)  // MAX_COLORS scatter cursors
#define CTR_TOTAL (8 + 2 * MAX_COLORS)
