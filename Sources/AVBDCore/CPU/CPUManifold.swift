import simd

// Contact manifold with friction (paper Sec 3.2-3.3) and OBB SAT collision
// with feature-ID based warm-start persistence. CPU reference port.

public struct ContactPoint {
    public var featureKey: Int32 = 0
    public var rA: F3 = .zero   // contact offset in A local space
    public var rB: F3 = .zero   // contact offset in B local space
    public var C0: F3 = .zero
    public var penalty: F3 = .zero
    public var lambda: F3 = .zero
    public var stick: Bool = false
}

public final class CPUManifold: CPUForce {
    public var contacts: [ContactPoint] = []
    public var numContacts: Int { contacts.count }
    // Basis rows: normal (B->A flipped to point A-ward), tangent1, tangent2
    public var basis: (F3, F3, F3) = (.zero, .zero, .zero)
    public var staticFriction: Float = 0
    public var dynamicFriction: Float = 0
    /// Compatibility alias for the original single Coulomb coefficient.
    /// Reads the static coefficient; writes update both coefficients so
    /// legacy callers retain their original single-material semantics.
    public var friction: Float {
        get { staticFriction }
        set {
            staticFriction = newValue
            dynamicFriction = newValue
        }
    }

    init(solver: CPUSolver, bodyA: CPURigid, bodyB: CPURigid) {
        super.init(solver: solver, bodyA: bodyA, bodyB: bodyB)
    }

    func anchorWorld(_ body: CPURigid, _ r: F3) -> F3 {
        body.shape != .box ? body.positionLin + r
                              : transform(body.positionLin, body.positionAng, r)
    }

    func anchorOffsetWorld(_ body: CPURigid, _ r: F3) -> F3 {
        body.shape != .box ? r : rotate(body.positionAng, r)
    }

    func currentPenetration(_ i: Int) -> Float {
        guard let bodyA, let bodyB else { return 0 }
        let xA = anchorWorld(bodyA, contacts[i].rA)
        let xB = anchorWorld(bodyB, contacts[i].rB)
        return dot(basis.0, xA - xB) + solver.collisionMargin
    }

    override func initialize() -> Bool {
        guard let bodyA, let bodyB else { return false }
        staticFriction = solver.frictionCombineMode.combine(
            bodyA.friction, bodyB.friction)
        dynamicFriction = solver.frictionCombineMode.combine(
            bodyA.dynamicFriction, bodyB.dynamicFriction)

        var newContacts: [ContactPoint] = []
        let count = Self.collide(
            bodyA, bodyB, margin: solver.collisionMargin,
            &newContacts, &basis)

        // Merge old contact data via feature keys (warm-start persistence).
        // Stick anchors are never restored for spheres: anchors rotate with
        // the body, which is wrong for rolling contacts.
        let anySphere = bodyA.shape != .box || bodyB.shape != .box
        for i in 0..<count {
            for old in contacts where old.featureKey == newContacts[i].featureKey {
                let newRA = newContacts[i].rA
                let newRB = newContacts[i].rB
                newContacts[i] = old
                if !old.stick || anySphere {
                    newContacts[i].rA = newRA
                    newContacts[i].rB = newRB
                }
                break
            }
        }
        contacts = newContacts

        // kinematic spinner surface motion (see GPU spinSurfaceShift)
        var wA = F3.zero, wB = F3.zero
        for sp in solver.spinners {
            if solver.bodies[sp.body] === bodyA { wA = sp.axis * sp.omega }
            if solver.bodies[sp.body] === bodyB { wB = sp.axis * sp.omega }
        }
        for i in 0..<contacts.count {
            let xA = anchorWorld(bodyA, contacts[i].rA)
            let xB = anchorWorld(bodyB, contacts[i].rB)
            let d = xA - xB
            contacts[i].C0 = F3(dot(basis.0, d) + solver.collisionMargin,
                                dot(basis.1, d), dot(basis.2, d))
            let sA = cross(wA, xA - bodyA.positionLin) * solver.dt
            let sB = cross(wB, xB - bodyB.positionLin) * solver.dt
            // pre-compensate the solver's alpha discount (shift is a target)
            let rel = (sB - sA) / max(1 - solver.alpha, 0.01)
            contacts[i].C0.y -= dot(basis.1, rel)
            contacts[i].C0.z -= dot(basis.2, rel)

            contacts[i].lambda *= solver.alpha * solver.gamma
            let minMass = [bodyA.mass, bodyB.mass].filter { $0 > 0 }.min()
            let kFloor = min(AVBDConstants.penaltyMax,
                             max(AVBDConstants.penaltyMin,
                                 (minMass ?? 0) / max(solver.dt * solver.dt, 1e-12)))
            let penaltyMin = F3(kFloor,
                                min(kFloor, AVBDConstants.penaltyMaxTangent),
                                min(kFloor, AVBDConstants.penaltyMaxTangent))
            let penaltyMax = F3(AVBDConstants.penaltyMax,
                                AVBDConstants.penaltyMaxTangent,
                                AVBDConstants.penaltyMaxTangent)
            contacts[i].penalty = simd_clamp(contacts[i].penalty * solver.gamma,
                                             penaltyMin, penaltyMax)
        }

        return !contacts.isEmpty
    }

    /// Taylor-approximated constraint and clamped force for contact i (Sec 4).
    private func forceAndC(_ i: Int, _ alpha: Float,
                           _ dqALin: F3, _ dqAAng: F3, _ dqBLin: F3, _ dqBAng: F3,
                           _ rAW: F3, _ rBW: F3) -> (F: F3, C: F3, frictionScale: Float, bounds: Float) {
        let c = contacts[i]
        // J rows for A: basis rows; angular rows: cross(rAW, basisRow)
        let jA0 = basis.0, jA1 = basis.1, jA2 = basis.2
        var C = c.C0 * (1 - alpha)
        C += F3(dot(jA0, dqALin), dot(jA1, dqALin), dot(jA2, dqALin))
        C -= F3(dot(jA0, dqBLin), dot(jA1, dqBLin), dot(jA2, dqBLin))
        C += F3(dot(cross(rAW, jA0), dqAAng), dot(cross(rAW, jA1), dqAAng), dot(cross(rAW, jA2), dqAAng))
        C += F3(dot(cross(rBW, -jA0), dqBAng), dot(cross(rBW, -jA1), dqBAng), dot(cross(rBW, -jA2), dqBAng))

        var F = c.penalty * C + c.lambda
        F[0] = min(F[0], 0)
        let staticBounds = abs(F[0]) * staticFriction
        let dynamicBounds = abs(F[0]) * dynamicFriction
        let frictionScale = length(SIMD2<Float>(F[1], F[2]))
        if frictionScale > staticBounds && frictionScale > 0 {
            F[1] *= dynamicBounds / frictionScale
            F[2] *= dynamicBounds / frictionScale
        }
        return (F, C, frictionScale, staticBounds)
    }

    override func updatePrimal(_ body: CPURigid, _ alpha: Float,
                               _ lhsLin: inout Mat3Rows, _ lhsAng: inout Mat3Rows, _ lhsCross: inout Mat3Rows,
                               _ rhsLin: inout F3, _ rhsAng: inout F3) {
        guard let bodyA, let bodyB else { return }
        let dqALin = bodyA.positionLin - bodyA.initialLin
        let dqAAng = quatSub(bodyA.positionAng, bodyA.initialAng)
        let dqBLin = bodyB.positionLin - bodyB.initialLin
        let dqBAng = quatSub(bodyB.positionAng, bodyB.initialAng)
        let isA = body === bodyA

        for i in 0..<contacts.count {
            let rAW = anchorOffsetWorld(bodyA, contacts[i].rA)
            let rBW = anchorOffsetWorld(bodyB, contacts[i].rB)
            let (F, _, _, _) = forceAndC(i, alpha, dqALin, dqAAng, dqBLin, dqBAng, rAW, rBW)

            let s: Float = isA ? 1 : -1
            let jLin = Mat3Rows(basis.0 * s, basis.1 * s, basis.2 * s)
            let rW = isA ? rAW : rBW
            let jAng = Mat3Rows(cross(rW, jLin.r0), cross(rW, jLin.r1), cross(rW, jLin.r2))

            let K = Mat3Rows.diagonal(contacts[i].penalty)
            let jLinT = jLin.transposed
            let jAngT = jAng.transposed
            let jAngTk = jAngT.mul(K)

            lhsLin += jLinT.mul(K).mul(jLin)
            lhsAng += jAngTk.mul(jAng)
            lhsCross += jAngTk.mul(jLin)

            rhsLin += jLinT.mul(F)
            rhsAng += jAngT.mul(F)
        }
    }

    override func updateDual(_ alpha: Float) {
        guard let bodyA, let bodyB else { return }
        let dqALin = bodyA.positionLin - bodyA.initialLin
        let dqAAng = quatSub(bodyA.positionAng, bodyA.initialAng)
        let dqBLin = bodyB.positionLin - bodyB.initialLin
        let dqBAng = quatSub(bodyB.positionAng, bodyB.initialAng)

        for i in 0..<contacts.count {
            let rAW = anchorOffsetWorld(bodyA, contacts[i].rA)
            let rBW = anchorOffsetWorld(bodyB, contacts[i].rB)
            let (F, C, frictionScale, bounds) =
                forceAndC(i, alpha, dqALin, dqAAng, dqBLin, dqBAng, rAW, rBW)

            var Fb = F
            Fb[0] = max(Fb[0], -solver.lambdaMax)
            contacts[i].lambda = Fb

            if F[0] < 0 {
                contacts[i].penalty[0] = min(contacts[i].penalty[0] + solver.betaLin * abs(C[0]),
                                             AVBDConstants.penaltyMax)
            }
            if frictionScale <= bounds {
                contacts[i].penalty[1] = min(contacts[i].penalty[1] + solver.betaLin * abs(C[1]),
                                             AVBDConstants.penaltyMaxTangent)
                contacts[i].penalty[2] = min(contacts[i].penalty[2] + solver.betaLin * abs(C[2]),
                                             AVBDConstants.penaltyMaxTangent)
                contacts[i].stick = length(SIMD2<Float>(C[1], C[2])) < AVBDConstants.stickThresh
            }
        }
    }
}
