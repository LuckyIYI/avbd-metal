import simd

// Math helpers matching the AVBD reference implementation (avbd-demo3d/maths.h)
// and the paper's rigid-body formulation (Sec. 4, Eq. 20-21).

public typealias F3 = SIMD3<Float>
public typealias Quat = simd_quatf

/// Right-handed orthographic projection using Metal's `[0, 1]` clip-depth
/// convention. View-space `-near` maps to 0 and `-far` maps to 1.
///
/// Kept in the testable core math layer because renderer depth conventions are
/// easy to get subtly wrong and directly affect shadow comparison sampling.
@inlinable public func metalOrthographicProjection(
    left: Float, right: Float,
    bottom: Float, top: Float,
    near: Float, far: Float
) -> simd_float4x4 {
    simd_float4x4(columns: (
        SIMD4(2 / (right - left), 0, 0, 0),
        SIMD4(0, 2 / (top - bottom), 0, 0),
        SIMD4(0, 0, 1 / (near - far), 0),
        SIMD4(-(right + left) / (right - left),
              -(top + bottom) / (top - bottom),
              near / (near - far), 1)
    ))
}

@inlinable public func skew(_ r: F3) -> simd_float3x3 {
    // Row-major semantic: result * v == cross(r, v)
    simd_float3x3(columns: (
        F3(0, r.z, -r.y),
        F3(-r.z, 0, r.x),
        F3(r.y, -r.x, 0)
    ))
}

@inlinable public func outer(_ a: F3, _ b: F3) -> simd_float3x3 {
    // a * b^T : column j is a * b[j]
    simd_float3x3(columns: (a * b.x, a * b.y, a * b.z))
}

@inlinable public func diagonal(_ x: Float, _ y: Float, _ z: Float) -> simd_float3x3 {
    simd_float3x3(diagonal: F3(x, y, z))
}

@inlinable public func diagonal(_ v: F3) -> simd_float3x3 {
    simd_float3x3(diagonal: v)
}

/// Diagonal matrix from column norms (paper Sec 3.5 — SPD approximation of G_ij).
@inlinable public func diagonalize(_ m: simd_float3x3) -> simd_float3x3 {
    diagonal(length(m.columns.0), length(m.columns.1), length(m.columns.2))
}

/// Orthonormal basis with `normal` as first row (matches demo's orthonormal()).
/// Returns matrix whose ROWS are (n, t1, t2); stored row-major as 3 float3 "rows".
@inlinable public func orthonormalBasis(_ normal: F3) -> (F3, F3, F3) {
    var t1 = abs(normal.x) > abs(normal.z) ? F3(-normal.y, normal.x, 0) : F3(0, -normal.z, normal.y)
    t1 = normalize(t1)
    let t2 = cross(normal, t1)
    return (normal, t1, t2)
}

// Quaternion "tangent space" ops from the paper (Eq. 20-21):
//   q_a - q_b := 2 * vec(q_a * q_b^-1)
//   q + w    := normalize(q + 0.5 * quat(w,0) * q)
@inlinable public func quatSub(_ a: Quat, _ b: Quat) -> F3 {
    // canonicalize to w >= 0: avoids the double-cover gradient sign flip
    // when a hinge spins past 180 deg of relative rotation
    let d = a * b.inverse
    return (d.real < 0 ? -d.imag : d.imag) * 2
}

@inlinable public func quatAdd(_ q: Quat, _ w: F3) -> Quat {
    let dq = Quat(real: 0, imag: w) * q
    let r = Quat(real: q.real + 0.5 * dq.real, imag: q.imag + 0.5 * dq.imag)
    return r.normalized
}

@inlinable public func rotate(_ q: Quat, _ v: F3) -> F3 {
    q.act(v)
}

/// Principal-axis inertia rotated into the world-space angular tangent basis.
@inlinable func worldInertiaRows(_ q: Quat, _ principal: F3) -> Mat3Rows {
    let c0 = q.act(F3(1, 0, 0))
    let c1 = q.act(F3(0, 1, 0))
    let c2 = q.act(F3(0, 0, 1))
    let matrix = outer(c0, c0) * principal.x
        + outer(c1, c1) * principal.y
        + outer(c2, c2) * principal.z
    return Mat3Rows(rowMajor: matrix)
}

@inlinable public func transform(_ p: F3, _ q: Quat, _ v: F3) -> F3 {
    q.act(v) + p
}

/// 6x6 symmetric LDL^T solve over [linear; angular] blocks.
/// lhsLin = upper-left 3x3, lhsAng = lower-right 3x3, lhsCross = lower-left 3x3
/// (rows angular, cols linear), matching the reference solve().
/// Matrices use ROW-major semantics here: m[i][j] = element at row i, col j.
/// We pass them as tuples of rows to keep this exact.
public struct Mat3Rows {
    public var r0: F3, r1: F3, r2: F3
    @inlinable public init() { r0 = .zero; r1 = .zero; r2 = .zero }
    @inlinable public init(_ r0: F3, _ r1: F3, _ r2: F3) { self.r0 = r0; self.r1 = r1; self.r2 = r2 }
    @inlinable public subscript(i: Int) -> F3 {
        get { i == 0 ? r0 : (i == 1 ? r1 : r2) }
        set { if i == 0 { r0 = newValue } else if i == 1 { r1 = newValue } else { r2 = newValue } }
    }
    @inlinable public static func += (a: inout Mat3Rows, b: Mat3Rows) {
        a.r0 += b.r0; a.r1 += b.r1; a.r2 += b.r2
    }
    @inlinable public static func diagonal(_ v: F3) -> Mat3Rows {
        Mat3Rows(F3(v.x, 0, 0), F3(0, v.y, 0), F3(0, 0, v.z))
    }
    /// From simd row-major intent: build from simd_float3x3 treated as row-major
    /// (i.e. simd columns are our rows' transposes). Helper to convert.
    @inlinable public init(rowMajor m: simd_float3x3) {
        // simd_float3x3 stores columns; rows are gathered:
        let c0 = m.columns.0, c1 = m.columns.1, c2 = m.columns.2
        r0 = F3(c0.x, c1.x, c2.x)
        r1 = F3(c0.y, c1.y, c2.y)
        r2 = F3(c0.z, c1.z, c2.z)
    }
    @inlinable public func mul(_ v: F3) -> F3 {
        F3(dot(r0, v), dot(r1, v), dot(r2, v))
    }
    @inlinable public func col(_ i: Int) -> F3 {
        F3(r0[i], r1[i], r2[i])
    }
    @inlinable public var transposed: Mat3Rows {
        Mat3Rows(col(0), col(1), col(2))
    }
    @inlinable public func mul(_ b: Mat3Rows) -> Mat3Rows {
        Mat3Rows(
            F3(dot(r0, b.col(0)), dot(r0, b.col(1)), dot(r0, b.col(2))),
            F3(dot(r1, b.col(0)), dot(r1, b.col(1)), dot(r1, b.col(2))),
            F3(dot(r2, b.col(0)), dot(r2, b.col(1)), dot(r2, b.col(2)))
        )
    }
    @inlinable public static func * (a: Mat3Rows, s: Float) -> Mat3Rows {
        Mat3Rows(a.r0 * s, a.r1 * s, a.r2 * s)
    }
    @inlinable public static prefix func - (a: Mat3Rows) -> Mat3Rows {
        Mat3Rows(-a.r0, -a.r1, -a.r2)
    }
    @inlinable public static var identity: Mat3Rows { Mat3Rows(F3(1, 0, 0), F3(0, 1, 0), F3(0, 0, 1)) }
    /// skew(r) in row semantics: mul(v) == cross(r, v)
    @inlinable public static func skewRows(_ r: F3) -> Mat3Rows {
        Mat3Rows(F3(0, -r.z, r.y), F3(r.z, 0, -r.x), F3(-r.y, r.x, 0))
    }
    /// Diagonal of column norms (paper Sec 3.5)
    @inlinable public var diagonalized: Mat3Rows {
        Mat3Rows.diagonal(F3(length(col(0)), length(col(1)), length(col(2))))
    }
}

/// Solve the 6x6 SPD system via LDL^T, ported exactly from maths.h solve().
/// aCross holds rows 4..6, cols 1..3 (angular-linear coupling block).
public func solve6x6(
    _ aLin: Mat3Rows, _ aAng: Mat3Rows, _ aCross: Mat3Rows,
    _ bLin: F3, _ bAng: F3,
    _ xLin: inout F3, _ xAng: inout F3
) {
    let A11 = aLin[0][0]
    let A21 = aLin[1][0], A22 = aLin[1][1]
    let A31 = aLin[2][0], A32 = aLin[2][1], A33 = aLin[2][2]
    let A41 = aCross[0][0], A42 = aCross[0][1], A43 = aCross[0][2], A44 = aAng[0][0]
    let A51 = aCross[1][0], A52 = aCross[1][1], A53 = aCross[1][2], A54 = aAng[1][0], A55 = aAng[1][1]
    let A61 = aCross[2][0], A62 = aCross[2][1], A63 = aCross[2][2], A64 = aAng[2][0], A65 = aAng[2][1], A66 = aAng[2][2]

    let L21 = A21 / A11
    let L31 = A31 / A11
    let L41 = A41 / A11
    let L51 = A51 / A11
    let L61 = A61 / A11

    let D1 = A11
    let D2 = A22 - L21 * L21 * D1

    let L32 = (A32 - L21 * L31 * D1) / D2
    let L42 = (A42 - L21 * L41 * D1) / D2
    let L52 = (A52 - L21 * L51 * D1) / D2
    let L62 = (A62 - L21 * L61 * D1) / D2

    let D3 = A33 - (L31 * L31 * D1 + L32 * L32 * D2)

    let L43 = (A43 - L31 * L41 * D1 - L32 * L42 * D2) / D3
    let L53 = (A53 - L31 * L51 * D1 - L32 * L52 * D2) / D3
    let L63 = (A63 - L31 * L61 * D1 - L32 * L62 * D2) / D3

    let D4 = A44 - (L41 * L41 * D1 + L42 * L42 * D2 + L43 * L43 * D3)

    let L54 = (A54 - L41 * L51 * D1 - L42 * L52 * D2 - L43 * L53 * D3) / D4
    let L64 = (A64 - L41 * L61 * D1 - L42 * L62 * D2 - L43 * L63 * D3) / D4

    let D5 = A55 - (L51 * L51 * D1 + L52 * L52 * D2 + L53 * L53 * D3 + L54 * L54 * D4)

    let L65 = (A65 - L51 * L61 * D1 - L52 * L62 * D2 - L53 * L63 * D3 - L54 * L64 * D4) / D5

    let D6 = A66 - (L61 * L61 * D1 + L62 * L62 * D2 + L63 * L63 * D3 + L64 * L64 * D4 + L65 * L65 * D5)

    let y1 = bLin[0]
    let y2 = bLin[1] - L21 * y1
    let y3 = bLin[2] - L31 * y1 - L32 * y2
    let y4 = bAng[0] - L41 * y1 - L42 * y2 - L43 * y3
    let y5 = bAng[1] - L51 * y1 - L52 * y2 - L53 * y3 - L54 * y4
    let y6 = bAng[2] - L61 * y1 - L62 * y2 - L63 * y3 - L64 * y4 - L65 * y5

    let z1 = y1 / D1
    let z2 = y2 / D2
    let z3 = y3 / D3
    let z4 = y4 / D4
    let z5 = y5 / D5
    let z6 = y6 / D6

    xAng[2] = z6
    xAng[1] = z5 - L65 * xAng[2]
    xAng[0] = z4 - L54 * xAng[1] - L64 * xAng[2]
    xLin[2] = z3 - L43 * xAng[0] - L53 * xAng[1] - L63 * xAng[2]
    xLin[1] = z2 - L32 * xLin[2] - L42 * xAng[0] - L52 * xAng[1] - L62 * xAng[2]
    xLin[0] = z1 - L21 * xLin[1] - L31 * xLin[2] - L41 * xAng[0] - L51 * xAng[1] - L61 * xAng[2]
}
