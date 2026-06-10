import simd

// Joint, Spring, and Manifold force types — CPU reference port.

func geometricStiffnessBallSocket(_ k: Int, _ v: F3) -> Mat3Rows {
    var m = Mat3Rows.diagonal(F3(repeating: -v[k]))
    m.r0[k] += v[0]
    m.r1[k] += v[1]
    m.r2[k] += v[2]
    return m
}

public final class CPUJoint: CPUForce {
    public var rA: F3, rB: F3
    public var C0Lin: F3 = .zero, C0Ang: F3 = .zero
    public var penaltyLin: F3 = .zero, penaltyAng: F3 = .zero
    public var lambdaLin: F3 = .zero, lambdaAng: F3 = .zero
    public var stiffnessLin: Float, stiffnessAng: Float, fracture: Float
    public var torqueArm: Float
    public var broken = false
    /// Rest relative rotation captured at creation: the angular constraint
    /// preserves the spawn alignment instead of forcing qA == qB.
    public var restRel: Quat
    /// Hinge axis in B local (nil = full weld). Rotation about it stays free.
    public var hingeAxis: F3? = nil

    init(solver: CPUSolver, bodyA: CPURigid?, bodyB: CPURigid, rA: F3, rB: F3,
         stiffnessLin: Float, stiffnessAng: Float, fracture: Float) {
        self.rA = rA
        self.rB = rB
        self.stiffnessLin = stiffnessLin
        self.stiffnessAng = stiffnessAng
        self.fracture = fracture
        self.torqueArm = length_squared((bodyA?.size ?? .zero) + bodyB.size)
        let qA0 = bodyA?.positionAng ?? Quat(real: 1, imag: .zero)
        self.restRel = (qA0.inverse * bodyB.positionAng).normalized
        super.init(solver: solver, bodyA: bodyA, bodyB: bodyB)
    }

    func currentCLin() -> F3 {
        guard let bodyB else { return .zero }
        let pA = bodyA.map { transform($0.positionLin, $0.positionAng, rA) } ?? rA
        return pA - transform(bodyB.positionLin, bodyB.positionAng, rB)
    }

    func currentCAng() -> F3 {
        guard let bodyB else { return .zero }
        let qA = bodyA?.positionAng ?? Quat(real: 1, imag: .zero)
        var C = quatSub((qA * restRel).normalized, bodyB.positionAng) * torqueArm
        if let axisL = hingeAxis {
            // project out the free rotation about the hinge axis
            let axisW = bodyB.positionAng.act(axisL)
            C -= axisW * dot(axisW, C)
        }
        return C
    }

    override func initialize() -> Bool {
        C0Lin = currentCLin()
        C0Ang = currentCAng()

        // Warmstart dual variables and penalty (Eq. 19)
        lambdaLin *= solver.alpha * solver.gamma
        lambdaAng *= solver.alpha * solver.gamma
        penaltyLin = simd_clamp(penaltyLin * solver.gamma,
                                F3(repeating: AVBDConstants.penaltyMin),
                                F3(repeating: AVBDConstants.penaltyMax))
        penaltyAng = simd_clamp(penaltyAng * solver.gamma,
                                F3(repeating: AVBDConstants.penaltyMin),
                                F3(repeating: AVBDConstants.penaltyMax))
        penaltyLin = simd_min(penaltyLin, F3(repeating: stiffnessLin))
        penaltyAng = simd_min(penaltyAng, F3(repeating: stiffnessAng))
        return !broken
    }

    override func updatePrimal(_ body: CPURigid, _ alpha: Float,
                               _ lhsLin: inout Mat3Rows, _ lhsAng: inout Mat3Rows, _ lhsCross: inout Mat3Rows,
                               _ rhsLin: inout F3, _ rhsAng: inout F3) {
        guard let bodyB else { return }
        let isA = body === bodyA

        // Linear constraint
        if length_squared(penaltyLin) > 0 {
            var C = currentCLin()
            if stiffnessLin.isInfinite { C -= C0Lin * alpha }
            let F = penaltyLin * C + lambdaLin

            let jLin = isA ? Mat3Rows.identity : -Mat3Rows.identity
            let jAng = isA
                ? Mat3Rows.skewRows(-rotate(bodyA!.positionAng, rA))
                : Mat3Rows.skewRows(rotate(bodyB.positionAng, rB))

            let jLinT = jLin.transposed
            let jAngT = jAng.transposed
            let K = Mat3Rows.diagonal(penaltyLin)
            let jAngTk = jAngT.mul(K)

            lhsLin += jLinT.mul(K).mul(jLin)
            lhsAng += jAngTk.mul(jAng)
            lhsCross += jAngTk.mul(jLin)

            // SPD diagonal approximation of geometric stiffness (paper Sec 3.5)
            let r = isA ? rotate(bodyA!.positionAng, rA) : -rotate(bodyB.positionAng, rB)
            var H = geometricStiffnessBallSocket(0, r) * F[0]
            H += geometricStiffnessBallSocket(1, r) * F[1]
            H += geometricStiffnessBallSocket(2, r) * F[2]
            lhsAng += H.diagonalized

            rhsLin += jLinT.mul(F)
            rhsAng += jAngT.mul(F)
        }

        // Angular constraint
        if length_squared(penaltyAng) > 0 {
            var C = currentCAng()
            if stiffnessAng.isInfinite { C -= C0Ang * alpha }
            var F = penaltyAng * C + lambdaAng

            let s: Float = (isA ? 1 : -1) * torqueArm
            if let axisL = hingeAxis {
                // hinge: no stiffness or force about the free axis
                let a = bodyB.positionAng.act(axisL)
                F -= a * dot(a, F)
                let k = (penaltyAng.x + penaltyAng.y + penaltyAng.z) / 3 * (s * s)
                var P = Mat3Rows.diagonal(F3(repeating: k))
                P += outerRows(a, a) * (-k)
                lhsAng += P
            } else {
                lhsAng += Mat3Rows.diagonal(penaltyAng * (s * s))
            }
            rhsAng += F * s
        }
    }

    override func updateDual(_ alpha: Float) {
        if length_squared(penaltyLin) > 0 {
            var C = currentCLin()
            if stiffnessLin.isInfinite {
                C -= C0Lin * alpha
                lambdaLin = penaltyLin * C + lambdaLin
            }
            let cap = min(stiffnessLin, AVBDConstants.penaltyMax)
            penaltyLin = simd_min(penaltyLin + abs(C) * solver.betaLin, F3(repeating: cap))
        }

        if length_squared(penaltyAng) > 0 {
            var C = currentCAng()
            if stiffnessAng.isInfinite {
                C -= C0Ang * alpha
                lambdaAng = penaltyAng * C + lambdaAng
            }
            let cap = min(stiffnessAng, AVBDConstants.penaltyMax)
            penaltyAng = simd_min(penaltyAng + abs(C) * solver.betaAng, F3(repeating: cap))
        }

        if length_squared(lambdaAng) > fracture * fracture {
            penaltyLin = .zero
            penaltyAng = .zero
            lambdaLin = .zero
            lambdaAng = .zero
            broken = true
        }
    }
}

public final class CPUSpring: CPUForce {
    public var rA: F3, rB: F3
    public var rest: Float
    public var stiffness: Float

    init(solver: CPUSolver, bodyA: CPURigid, bodyB: CPURigid, rA: F3, rB: F3,
         stiffness: Float, rest: Float) {
        self.rA = rA
        self.rB = rB
        self.stiffness = stiffness
        if rest < 0 {
            let pA = transform(bodyA.positionLin, bodyA.positionAng, rA)
            let pB = transform(bodyB.positionLin, bodyB.positionAng, rB)
            self.rest = length(pA - pB)
        } else {
            self.rest = rest
        }
        super.init(solver: solver, bodyA: bodyA, bodyB: bodyB)
    }

    override func updatePrimal(_ body: CPURigid, _ alpha: Float,
                               _ lhsLin: inout Mat3Rows, _ lhsAng: inout Mat3Rows, _ lhsCross: inout Mat3Rows,
                               _ rhsLin: inout F3, _ rhsAng: inout F3) {
        guard let bodyA, let bodyB else { return }
        let pA = transform(bodyA.positionLin, bodyA.positionAng, rA)
        let pB = transform(bodyB.positionLin, bodyB.positionAng, rB)
        let d = pA - pB
        let dLen = length(d)
        if dLen <= 1e-6 { return }

        let n = d / dLen
        let f = stiffness * (dLen - rest)

        let isA = body === bodyA
        let rWorld = isA ? rotate(bodyA.positionAng, rA) : rotate(bodyB.positionAng, rB)
        let jLin = isA ? n : -n
        let jAng = isA ? cross(rWorld, n) : -cross(rWorld, n)

        lhsLin += outerRows(jLin, jLin) * stiffness
        lhsAng += outerRows(jAng, jAng) * stiffness
        lhsCross += outerRows(jAng, jLin) * stiffness
        rhsLin += jLin * f
        rhsAng += jAng * f
    }
}

/// outer(a, b) in row semantics: row i = a[i] * b
@inlinable func outerRows(_ a: F3, _ b: F3) -> Mat3Rows {
    Mat3Rows(b * a.x, b * a.y, b * a.z)
}
