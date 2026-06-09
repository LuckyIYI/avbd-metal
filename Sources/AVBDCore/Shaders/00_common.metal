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
#define COLLISION_MARGIN 0.01f
#define STICK_THRESH 0.00001f
#define MAX_COLORS 64
#define MAX_CONTACTS 8

// Force kinds packed into adjacency entries (top 3 bits)
#define FK_JOINT 0u
#define FK_SPRING 1u
#define FK_MANIFOLD 2u
#define ADJ_KIND_SHIFT 28
#define ADJ_INDEX_MASK 0x0FFFFFFFu

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
// q_a - q_b in tangent space (Eq. 20)
inline float3 q_sub(float4 a, float4 b) {
    return q_mul(a, q_inv(b)).xyz * 2.0f;
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
inline void solve6x6(M3 aLin, M3 aAng, M3 aCross,
                     float3 bLin, float3 bAng,
                     thread float3 &xLin, thread float3 &xAng)
{
    float A11 = aLin.r0.x;
    float A21 = aLin.r1.x, A22 = aLin.r1.y;
    float A31 = aLin.r2.x, A32 = aLin.r2.y, A33 = aLin.r2.z;
    float A41 = aCross.r0.x, A42 = aCross.r0.y, A43 = aCross.r0.z, A44 = aAng.r0.x;
    float A51 = aCross.r1.x, A52 = aCross.r1.y, A53 = aCross.r1.z, A54 = aAng.r1.x, A55 = aAng.r1.y;
    float A61 = aCross.r2.x, A62 = aCross.r2.y, A63 = aCross.r2.z, A64 = aAng.r2.x, A65 = aAng.r2.y, A66 = aAng.r2.z;

    float L21 = A21 / A11;
    float L31 = A31 / A11;
    float L41 = A41 / A11;
    float L51 = A51 / A11;
    float L61 = A61 / A11;

    float D1 = A11;
    float D2 = A22 - L21*L21*D1;

    float L32 = (A32 - L21*L31*D1) / D2;
    float L42 = (A42 - L21*L41*D1) / D2;
    float L52 = (A52 - L21*L51*D1) / D2;
    float L62 = (A62 - L21*L61*D1) / D2;

    float D3 = A33 - (L31*L31*D1 + L32*L32*D2);

    float L43 = (A43 - L31*L41*D1 - L32*L42*D2) / D3;
    float L53 = (A53 - L31*L51*D1 - L32*L52*D2) / D3;
    float L63 = (A63 - L31*L61*D1 - L32*L62*D2) / D3;

    float D4 = A44 - (L41*L41*D1 + L42*L42*D2 + L43*L43*D3);

    float L54 = (A54 - L41*L51*D1 - L42*L52*D2 - L43*L53*D3) / D4;
    float L64 = (A64 - L41*L61*D1 - L42*L62*D2 - L43*L63*D3) / D4;

    float D5 = A55 - (L51*L51*D1 + L52*L52*D2 + L53*L53*D3 + L54*L54*D4);

    float L65 = (A65 - L51*L61*D1 - L52*L62*D2 - L53*L63*D3 - L54*L64*D4) / D5;

    float D6 = A66 - (L61*L61*D1 + L62*L62*D2 + L63*L63*D3 + L64*L64*D4 + L65*L65*D5);

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
};

struct JointGPU {
    uint4 header;       // bodyA (WORLD_BODY=world), bodyB, broken flag, pad
    float4 rA;          // w = stiffnessLin
    float4 rB;          // w = stiffnessAng
    float4 C0Lin;       // w = torqueArm
    float4 C0Ang;       // w = fracture
    float4 lambdaLin;
    float4 lambdaAng;
    float4 penaltyLin;
    float4 penaltyAng;
};

struct SpringGPU {
    uint4 header;       // bodyA, bodyB, pad, pad
    float4 rA;          // w = stiffness
    float4 rB;          // w = rest
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
#define CTR_COLOR_BASE 8           // MAX_COLORS entries
#define CTR_SCATTER_BASE (8 + MAX_COLORS)  // MAX_COLORS scatter cursors
#define CTR_TOTAL (8 + 2 * MAX_COLORS)
