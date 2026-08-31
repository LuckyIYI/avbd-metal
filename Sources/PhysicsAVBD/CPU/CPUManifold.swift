import simd
import SimCore

// Contact manifold with friction (paper Sec 3.2-3.3). Analytic contacts use
// feature-ID persistence; clipped generic-convex patches use one-to-one local
// anchor matching because their deterministic sort order is not an identity.

public struct ContactPoint {
    public var featureKey: Int32 = 0
    public var rA: F3 = .zero   // contact offset in A local space
    public var rB: F3 = .zero   // contact offset in B local space
    public var C0: F3 = .zero
    public var penalty: F3 = .zero
    public var lambda: F3 = .zero
    public var stick: Bool = false
    /// Aggregate solver-level twisting friction for this manifold. Only the
    /// first contact stores it; collision geometry stays exact.
    public var torsionLambda: Float = 0
    public var torsionPenalty: Float = 0
}

public final class CPUManifold: CPUForce {
    public var contacts: [ContactPoint] = []
    public var numContacts: Int { contacts.count }
    // Basis rows: normal (B->A flipped to point A-ward), tangent1, tangent2
    public var basis: (F3, F3, F3) = (.zero, .zero, .zero)
    public var staticFriction: Float = 0
    public var dynamicFriction: Float = 0
    /// Effective torsional contact radius. The twisting-torque bound is
    /// `normalLoad * torsionalFriction`.
    public var torsionalFriction: Float = 0
    let colliderAIndex: Int?
    let colliderBIndex: Int?
    let colliderPairKey: UInt64?
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
        colliderAIndex = nil
        colliderBIndex = nil
        colliderPairKey = nil
        super.init(solver: solver, bodyA: bodyA, bodyB: bodyB)
    }

    init(solver: CPUSolver, colliderA: Int, colliderB: Int, pairKey: UInt64) {
        precondition(solver.colliders.indices.contains(colliderA)
            && solver.colliders.indices.contains(colliderB))
        let a = solver.colliders[colliderA]
        let b = solver.colliders[colliderB]
        colliderAIndex = colliderA
        colliderBIndex = colliderB
        colliderPairKey = pairKey
        super.init(solver: solver,
                   bodyA: solver.bodies[a.body], bodyB: solver.bodies[b.body])
    }

    private func collider(isA: Bool) -> CPUCollider? {
        let index = isA ? colliderAIndex : colliderBIndex
        guard let index else { return nil }
        return solver.colliders[index]
    }

    func anchorWorld(_ body: CPURigid, _ r: F3, isA: Bool) -> F3 {
        if let collider = collider(isA: isA) {
            return collider.usesWorldSpaceRoundAnchor
                ? body.positionLin + r
                : transform(body.positionLin, body.positionAng, r)
        }
        return body.shape != .box
            ? body.positionLin + r
            : transform(body.positionLin, body.positionAng, r)
    }

    func anchorOffsetWorld(_ body: CPURigid, _ r: F3, isA: Bool) -> F3 {
        if let collider = collider(isA: isA) {
            return collider.usesWorldSpaceRoundAnchor ? r : rotate(body.positionAng, r)
        }
        return body.shape != .box ? r : rotate(body.positionAng, r)
    }

    func currentPenetration(_ i: Int) -> Float {
        guard let bodyA, let bodyB else { return 0 }
        let xA = anchorWorld(bodyA, contacts[i].rA, isA: true)
        let xB = anchorWorld(bodyB, contacts[i].rB, isA: false)
        return dot(basis.0, xA - xB) + solver.collisionMargin
    }

    override func initialize() -> Result<Bool, CPUSolver.RuntimeFailure> {
        guard let bodyA, let bodyB else { return .success(false) }
        let colliderA = collider(isA: true)
        let colliderB = collider(isA: false)
        let previousContacts = contacts
        let previousBasis = basis
        let previousTorsionLambda = previousContacts.first?.torsionLambda ?? 0
        let previousTorsionPenalty = previousContacts.first?.torsionPenalty ?? 0
        staticFriction = solver.frictionCombineMode.combine(
            colliderA?.friction ?? bodyA.friction,
            colliderB?.friction ?? bodyB.friction)
        dynamicFriction = solver.frictionCombineMode.combine(
            colliderA?.dynamicFriction ?? bodyA.dynamicFriction,
            colliderB?.dynamicFriction ?? bodyB.dynamicFriction)
        // This coefficient has length units, so multiplicative friction
        // combination would be dimensionally wrong. Match Newton XPBD's
        // symmetric per-shape average for torsional/rolling materials.
        torsionalFriction = 0.5 * (
            (colliderA?.torsionalFriction ?? bodyA.torsionalFriction)
            + (colliderB?.torsionalFriction ?? bodyB.torsionalFriction))

        var newContacts: [ContactPoint] = []
        let count: Int
        if let colliderA, let colliderB {
            let query = Self.collideColliders(
                solver: solver,
                bodyA: bodyA, colliderA: colliderA,
                bodyB: bodyB, colliderB: colliderB,
                margin: solver.collisionMargin,
                &newContacts, &basis)
            switch query {
            case .success(let value):
                count = value
            case .failure(let failure):
                return .failure(.convexQuery(
                    colliderA: colliderA.index,
                    colliderB: colliderB.index,
                    failure: failure))
            }
        } else {
            count = Self.collide(
                bodyA, bodyB, margin: solver.collisionMargin,
                &newContacts, &basis)
        }

        // Merge old contact data for warm-start persistence. The analytic
        // narrow phase has stable feature IDs. A clipped generic-convex
        // manifold does not: a small rotation can reorder its world-space
        // points without changing the physical patch. Match that path
        // one-to-one by owner-local anchors while the normal is compatible,
        // mirroring the GPU path, so sticky state cannot jump between points.
        // Stick anchors are never restored for round shapes: their anchors are
        // world-space offsets and must follow rolling contact.
        let anyWorldSpaceRoundAnchor: Bool
        let usesGenericConvexManifold: Bool
        if let colliderA, let colliderB {
            anyWorldSpaceRoundAnchor = colliderA.usesWorldSpaceRoundAnchor
                || colliderB.usesWorldSpaceRoundAnchor
            usesGenericConvexManifold = colliderA.hasConvexHull
                || colliderB.hasConvexHull
        } else {
            anyWorldSpaceRoundAnchor = bodyA.shape != .box || bodyB.shape != .box
            usesGenericConvexManifold = false
        }

        if usesGenericConvexManifold {
            var matchedPrevious = Set<Int>()
            let normalCompatible = dot(previousBasis.0, basis.0) > 0.95
            let scale = max(colliderA?.boundingRadius ?? 0,
                            colliderB?.boundingRadius ?? 0)
            let matchRadius = 0.08 * scale + 2 * solver.collisionMargin
            for i in 0..<count where normalCompatible {
                let newRA = newContacts[i].rA
                let newRB = newContacts[i].rB
                var bestIndex: Int?
                var bestDistance = matchRadius
                for oldIndex in previousContacts.indices
                    where !matchedPrevious.contains(oldIndex) {
                    let old = previousContacts[oldIndex]
                    let anchorDistance = max(length(old.rA - newRA),
                                             length(old.rB - newRB))
                    if anchorDistance < bestDistance - 1.0e-7
                        || (abs(anchorDistance - bestDistance) <= 1.0e-7
                            && oldIndex < (bestIndex ?? .max)) {
                        bestDistance = anchorDistance
                        bestIndex = oldIndex
                    }
                }
                guard let bestIndex else { continue }
                matchedPrevious.insert(bestIndex)
                let old = previousContacts[bestIndex]
                var restored = old
                restored.featureKey = newContacts[i].featureKey

                // Tangential multipliers are coordinates in the old basis;
                // rotate them into the current manifold basis before reuse.
                let worldTangent = previousBasis.1 * old.lambda.y
                    + previousBasis.2 * old.lambda.z
                restored.lambda.y = dot(basis.1, worldTangent)
                restored.lambda.z = dot(basis.2, worldTangent)
                if !old.stick || anyWorldSpaceRoundAnchor {
                    restored.rA = newRA
                    restored.rB = newRB
                }
                newContacts[i] = restored
            }
        } else {
            for i in 0..<count {
                for old in previousContacts
                    where old.featureKey == newContacts[i].featureKey {
                    let newRA = newContacts[i].rA
                    let newRB = newContacts[i].rB
                    newContacts[i] = old
                    if !old.stick || anyWorldSpaceRoundAnchor {
                        newContacts[i].rA = newRA
                        newContacts[i].rB = newRB
                    }
                    break
                }
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
            let xA = anchorWorld(bodyA, contacts[i].rA, isA: true)
            let xB = anchorWorld(bodyB, contacts[i].rB, isA: false)
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

        // Torsional friction is one aggregate material mode per manifold,
        // independent of how many geometric witnesses its narrow phase
        // emits. This avoids both sphere-specific fake contacts and torque
        // multiplication when a face is tessellated into more points.
        if !contacts.isEmpty, torsionalFriction > 0 {
            var effectiveMoment = Float.greatestFiniteMagnitude
            for candidate in [bodyA, bodyB] where candidate.mass > 0 {
                let moment = min(candidate.moment.x,
                                 min(candidate.moment.y, candidate.moment.z))
                if moment > 0, moment.isFinite {
                    effectiveMoment = min(effectiveMoment, moment)
                }
            }
            if effectiveMoment == Float.greatestFiniteMagnitude {
                effectiveMoment = 0
            }
            let floor = min(
                AVBDConstants.penaltyMaxTangent,
                max(AVBDConstants.penaltyMin,
                    effectiveMoment / max(solver.dt * solver.dt, 1.0e-12)))
            let normalCompatible = dot(previousBasis.0, basis.0) > 0.95
            contacts[0].torsionLambda = normalCompatible
                ? previousTorsionLambda * solver.alpha * solver.gamma : 0
            contacts[0].torsionPenalty = simd_clamp(
                (normalCompatible ? previousTorsionPenalty : 0) * solver.gamma,
                floor, AVBDConstants.penaltyMaxTangent)
            for index in contacts.indices.dropFirst() {
                contacts[index].torsionLambda = 0
                contacts[index].torsionPenalty = 0
            }
        } else {
            for index in contacts.indices {
                contacts[index].torsionLambda = 0
                contacts[index].torsionPenalty = 0
            }
        }

        return .success(!contacts.isEmpty)
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

    private func torsionForceAndC(
        _ contact: ContactPoint, normalForce: Float,
        _ dqAAng: F3, _ dqBAng: F3
    ) -> (torque: Float, C: Float, unconstrained: Float, bound: Float) {
        let C = dot(basis.0, dqAAng - dqBAng)
        let unconstrained = contact.torsionPenalty * C
            + contact.torsionLambda
        let bound = abs(normalForce) * torsionalFriction
        return (simd_clamp(unconstrained, -bound, bound), C,
                unconstrained, bound)
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
        var totalNormalLoad: Float = 0

        for i in 0..<contacts.count {
            let rAW = anchorOffsetWorld(bodyA, contacts[i].rA, isA: true)
            let rBW = anchorOffsetWorld(bodyB, contacts[i].rB, isA: false)
            let (F, _, _, _) = forceAndC(i, alpha, dqALin, dqAAng, dqBLin, dqBAng, rAW, rBW)
            totalNormalLoad += abs(F.x)

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
        if let torsion = contacts.first, torsionalFriction > 0 {
            let twist = torsionForceAndC(
                torsion, normalForce: totalNormalLoad, dqAAng, dqBAng)
            let jTwist = basis.0 * (isA ? 1 : -1)
            lhsAng += outerRows(jTwist, jTwist) * torsion.torsionPenalty
            rhsAng += jTwist * twist.torque
        }
    }

    override func updateDual(_ alpha: Float) {
        guard let bodyA, let bodyB else { return }
        let dqALin = bodyA.positionLin - bodyA.initialLin
        let dqAAng = quatSub(bodyA.positionAng, bodyA.initialAng)
        let dqBLin = bodyB.positionLin - bodyB.initialLin
        let dqBAng = quatSub(bodyB.positionAng, bodyB.initialAng)
        var totalNormalLoad: Float = 0

        for i in 0..<contacts.count {
            let rAW = anchorOffsetWorld(bodyA, contacts[i].rA, isA: true)
            let rBW = anchorOffsetWorld(bodyB, contacts[i].rB, isA: false)
            let (F, C, frictionScale, bounds) =
                forceAndC(i, alpha, dqALin, dqAAng, dqBLin, dqBAng, rAW, rBW)
            totalNormalLoad += abs(F.x)

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
        if !contacts.isEmpty, torsionalFriction > 0 {
            let twist = torsionForceAndC(
                contacts[0], normalForce: totalNormalLoad, dqAAng, dqBAng)
            contacts[0].torsionLambda = twist.torque
            if abs(twist.unconstrained) <= twist.bound {
                contacts[0].torsionPenalty = min(
                    contacts[0].torsionPenalty
                        + solver.betaAng * abs(twist.C),
                    AVBDConstants.penaltyMaxTangent)
            }
        }
    }
}
