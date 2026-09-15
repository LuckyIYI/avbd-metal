import simd
import SimCore

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
    public var fractureLinear = false
    /// Rest relative rotation captured at creation: the angular constraint
    /// preserves the spawn alignment instead of forcing qA == qB.
    public var restRel: Quat
    /// Hinge axis in B local (nil = full weld). Rotation about it stays free.
    public var hingeAxis: F3? = nil
    public var prismaticAxis: F3? = nil
    public var translationLimits: ClosedRange<Float>? = nil
    // Bound owning the axial multiplier: -1 lower, 0 free, +1 upper.
    private var prismaticWarmStartStop = 0

    init(solver: CPUSolver, bodyA: CPURigid?, bodyB: CPURigid, rA: F3, rB: F3,
         stiffnessLin: Float, stiffnessAng: Float, fracture: Float) {
        self.rA = rA
        self.rB = rB
        self.stiffnessLin = stiffnessLin
        self.stiffnessAng = stiffnessAng
        self.fracture = fracture
        self.torqueArm = length_squared((bodyA?.size ?? .zero) + bodyB.size)
        // A hard constraint must be load-bearing on its first frame: seed
        // its penalty at the same effective-mass / dt^2 floor as the GPU
        // and as new contacts, instead of PENALTY_MIN (which let a hanging
        // chain sag on frame one until the adaptive ramp caught up).
        let dynamicMasses = [bodyA?.mass ?? 0, bodyB.mass].filter { $0 > 0 }
        let hardPenaltyFloor = min(
            1.0e9,
            max(1, (dynamicMasses.min() ?? 0)
                / max(solver.dt * solver.dt, 1.0e-12)))
        if stiffnessLin.isInfinite {
            penaltyLin = F3(repeating: hardPenaltyFloor)
        } else if stiffnessLin > 0 {
            penaltyLin = F3(repeating: min(stiffnessLin, 1e9))
        }
        if stiffnessAng.isInfinite {
            penaltyAng = F3(repeating: hardPenaltyFloor)
        } else if stiffnessAng > 0 {
            penaltyAng = F3(repeating: min(stiffnessAng, 1e9))
        }
        let qA0 = bodyA?.positionAng ?? Quat(real: 1, imag: .zero)
        self.restRel = (qA0.inverse * bodyB.positionAng).normalized
        super.init(solver: solver, bodyA: bodyA, bodyB: bodyB)
    }

    func currentCLin() -> F3 {
        guard let bodyB else { return .zero }
        let pA = bodyA.map { transform($0.positionLin, $0.positionAng, rA) } ?? rA
        let delta = pA - transform(bodyB.positionLin, bodyB.positionAng, rB)
        guard let axis = prismaticAxis else { return delta }
        let d = (bodyA?.positionAng ?? Quat(real: 1, imag: .zero)).inverse.act(delta)
        let t = -dot(d, axis)
        let target = translationLimits.map { min(max(t, $0.lowerBound), $0.upperBound) } ?? t
        return d + axis * target
    }

    private func prismaticActiveStop() -> Int {
        guard let axis = prismaticAxis, let limits = translationLimits,
              let bodyB else { return 0 }
        let qA = bodyA?.positionAng ?? Quat(real: 1, imag: .zero)
        let pA = bodyA.map { transform($0.positionLin, $0.positionAng, rA) } ?? rA
        let pB = transform(bodyB.positionLin, bodyB.positionAng, rB)
        let t = -dot(qA.inverse.act(pA - pB), axis)
        return t < limits.lowerBound ? -1 : (t > limits.upperBound ? 1 : 0)
    }

    private func prismaticProjection(activeStop: Int) -> Mat3Rows {
        guard let axis = prismaticAxis, activeStop == 0 else { return .identity }
        return Mat3Rows(rowMajor: matrix_identity_float3x3 - outer(axis, axis))
    }

    private func prismaticWarmStart(activeStop: Int) -> F3 {
        guard let axis = prismaticAxis else { return lambdaLin }
        // The projection is the identity at either stop, so it alone cannot
        // reject a multiplier carried directly across the whole free interval.
        if activeStop == 0 || activeStop != prismaticWarmStartStop {
            return lambdaLin - axis * dot(lambdaLin, axis)
        }
        return lambdaLin
    }

    func currentCAng() -> F3 {
        guard let bodyB else { return .zero }
        let qA = bodyA?.positionAng ?? Quat(real: 1, imag: .zero)
        if let axisL = hingeAxis {
            // hinge: axis-alignment error (spin-invariant)
            let aB = bodyB.positionAng.act(axisL)
            let aA = ((qA * restRel).normalized).act(axisL)
            return cross(aA, aB) * torqueArm
        }
        return quatSub((qA * restRel).normalized, bodyB.positionAng) * torqueArm
    }

    override func initialize() -> Result<Bool, CPUSolver.RuntimeFailure> {
        C0Lin = currentCLin()
        C0Ang = currentCAng()

        // Warmstart dual variables and penalty (Eq. 19)
        if prismaticAxis != nil {
            let stop = prismaticActiveStop()
            lambdaLin = prismaticWarmStart(activeStop: stop)
            prismaticWarmStartStop = stop
        }
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
        return .success(!broken)
    }

    override func updatePrimal(_ body: CPURigid, _ alpha: Float,
                               _ lhsLin: inout Mat3Rows, _ lhsAng: inout Mat3Rows, _ lhsCross: inout Mat3Rows,
                               _ rhsLin: inout F3, _ rhsAng: inout F3) {
        guard let bodyB else { return }
        let isA = body === bodyA

        // Linear constraint
        if length_squared(penaltyLin) > 0, prismaticAxis != nil {
            let qA = bodyA?.positionAng ?? Quat(real: 1, imag: .zero)
            let xA = bodyA?.positionLin ?? .zero
            let pB = transform(bodyB.positionLin, bodyB.positionAng, rB)
            let stop = prismaticActiveStop()
            let projection = prismaticProjection(activeStop: stop)
            let basis = projection.mul(Mat3Rows(rowMajor: simd_float3x3(qA).transpose))
            var C = currentCLin()
            if stiffnessLin.isInfinite { C -= C0Lin * alpha }
            // Prediction or a primal update can switch stops after initialize.
            let F = penaltyLin * C + prismaticWarmStart(activeStop: stop)
            let jLin = basis * (isA ? 1 : -1)
            let jAng = basis.mul(.skewRows(isA ? xA - pB : bodyB.positionAng.act(rB)))
            let jLinT = jLin.transposed, jAngT = jAng.transposed
            let K = Mat3Rows.diagonal(penaltyLin), jAngTk = jAngT.mul(K)
            lhsLin += jLinT.mul(K).mul(jLin)
            lhsAng += jAngTk.mul(jAng)
            lhsCross += jAngTk.mul(jLin)
            rhsLin += jLinT.mul(F)
            rhsAng += jAngT.mul(F)
        } else if length_squared(penaltyLin) > 0 {
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

            var s: Float = (isA ? 1 : -1) * torqueArm
            if hingeAxis != nil { s = -s }   // cross-constraint sign flip
            lhsAng += Mat3Rows.diagonal(penaltyAng * (s * s))
            rhsAng += F * s
        }
    }

    override func updateDual(_ alpha: Float) {
        guard let bodyB else { return }
        // Dual bound scaled to the lighter participant, as for contacts
        // (twice the contact scale so a structural joint wins a fight with
        // a contact). Mirrors the GPU `dual_joint_one`; the CPU previously
        // accumulated lambda without any bound.
        let big = Float.greatestFiniteMagnitude
        let mA = bodyA.map { $0.mass > 0 ? $0.mass : big } ?? big
        let mB = bodyB.mass > 0 ? bodyB.mass : big
        let mMin = min(mA, mB)
        let lamCap = mMin == big ? solver.lambdaMax
            : min(solver.lambdaMax, max(10, 2.0e5 * mMin))
        let lamLo = F3(repeating: -lamCap), lamHi = F3(repeating: lamCap)

        if length_squared(penaltyLin) > 0 {
            if prismaticAxis != nil {
                let stop = prismaticActiveStop()
                lambdaLin = prismaticWarmStart(activeStop: stop)
                prismaticWarmStartStop = stop
            }
            var C = currentCLin()
            if stiffnessLin.isInfinite {
                C -= C0Lin * alpha
                lambdaLin = simd_clamp(penaltyLin * C + lambdaLin, lamLo, lamHi)
            }
            let cap = min(stiffnessLin, AVBDConstants.penaltyMax)
            penaltyLin = simd_min(penaltyLin + abs(C) * solver.betaLin, F3(repeating: cap))
        }

        if length_squared(penaltyAng) > 0 {
            var C = currentCAng()
            if stiffnessAng.isInfinite {
                C -= C0Ang * alpha
                lambdaAng = simd_clamp(penaltyAng * C + lambdaAng, lamLo, lamHi)
            }
            let cap = min(stiffnessAng, AVBDConstants.penaltyMax)
            penaltyAng = simd_min(penaltyAng + abs(C) * solver.betaAng, F3(repeating: cap))
        }

        let lin2 = fractureLinear ? length_squared(lambdaLin) : 0
        if length_squared(lambdaAng) + lin2 > fracture * fracture {
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
    public var hard = false
    var lambda: Float = 0
    var penalty: Float = 0
    var C0: Float = 0

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

    override func initialize() -> Result<Bool, CPUSolver.RuntimeFailure> {
        guard hard, let bodyA, let bodyB else { return .success(true) }
        let pA = transform(bodyA.positionLin, bodyA.positionAng, rA)
        let pB = transform(bodyB.positionLin, bodyB.positionAng, rB)
        C0 = length(pA - pB) - rest
        lambda *= solver.alpha * solver.gamma
        penalty = min(simd_clamp(penalty * solver.gamma,
                                 AVBDConstants.penaltyMin, AVBDConstants.penaltyMax),
                      stiffness)
        return .success(true)
    }

    override func updateDual(_ alpha: Float) {
        guard hard, let bodyA, let bodyB else { return }
        let pA = transform(bodyA.positionLin, bodyA.positionAng, rA)
        let pB = transform(bodyB.positionLin, bodyB.positionAng, rB)
        let C = length(pA - pB) - rest - C0 * alpha
        // Tension-only, leaky, mass-capped dual: mirrors the GPU rod dual
        // (see `dual_all`). A hard rod is inextensible, not incompressible.
        let big = Float.greatestFiniteMagnitude
        let mA = bodyA.mass > 0 ? bodyA.mass : big
        let mB = bodyB.mass > 0 ? bodyB.mass : big
        let mMin = min(mA, mB)
        let cap = mMin == big ? solver.lambdaMax
            : min(solver.lambdaMax, max(2, 5.0e3 * mMin))
        lambda = min(max(0.98 * lambda + penalty * C, 0), cap)
        if C > 0 {
            penalty = min(penalty + C * solver.betaLin,
                          min(stiffness, AVBDConstants.penaltyMax))
        }
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
        var k = stiffness
        var f: Float
        if hard {
            k = penalty
            f = penalty * (dLen - rest - C0 * alpha) + lambda
            // TENSION-ONLY inextensible element (matches the GPU rod): a
            // compressed rod buckles freely; enforcing compression turned
            // buckled pairs into perpetual oscillators.
            if f <= 0 { return }
        } else {
            f = stiffness * (dLen - rest)
        }

        let isA = body === bodyA
        let rWorld = isA ? rotate(bodyA.positionAng, rA) : rotate(bodyB.positionAng, rB)
        let jLin = isA ? n : -n
        let jAng = isA ? cross(rWorld, n) : -cross(rWorld, n)

        lhsLin += outerRows(jLin, jLin) * k
        lhsAng += outerRows(jAng, jAng) * k
        lhsCross += outerRows(jAng, jLin) * k
        rhsLin += jLin * f
        rhsAng += jAng * f
    }
}

/// outer(a, b) in row semantics: row i = a[i] * b
@inlinable func outerRows(_ a: F3, _ b: F3) -> Mat3Rows {
    Mat3Rows(b * a.x, b * a.y, b * a.z)
}
