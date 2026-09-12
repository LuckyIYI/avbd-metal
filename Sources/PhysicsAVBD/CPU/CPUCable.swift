import simd
import SimCore

/// Exact linear strain Jacobians, including the rotating parent frame.
@inline(__always)
package func cableLinearStrain(xA: F3, qA: Quat, xB: F3, qB: Quat,
                               rA: F3, rB: F3, isA: Bool)
    -> (value: F3, linear: Mat3Rows, angular: Mat3Rows) {
    let rBW = qB.act(rB), pB = xB + rBW
    let basis = Mat3Rows(rowMajor: simd_float3x3(qA).transpose)
    return (basis.mul(pB - xA) - rA,
            basis * (isA ? -1 : 1),
            basis.mul(Mat3Rows.skewRows(isA ? pB - xA : -rBW)))
}

/// Principal SO(3) logarithm and its exact derivative under a left angular
/// perturbation. Series avoids cancellation at rest; no numerical derivatives.
package func cableRotationLog(_ rotation: Quat) -> (value: F3, derivative: Mat3Rows) {
    let q = rotation.real < 0 ? Quat(vector: -rotation.vector) : rotation
    let s2 = length_squared(q.imag)
    let value: F3
    let coefficient: Float
    if s2 < 1e-8 {
        value = q.imag * (2 + s2 / 3)
        coefficient = 1 / 12 + s2 / 180
    } else {
        let s = sqrt(s2)
        let angle = 2 * atan2(s, q.real)
        value = q.imag * (angle / s)
        coefficient = (1 - 0.5 * angle * q.real / s) / (angle * angle)
    }
    let skew = Mat3Rows.skewRows(value)
    var derivative = Mat3Rows.identity
    derivative += skew * -0.5
    derivative += skew.mul(skew) * coefficient
    return (value, derivative)
}

/// Radial return in the two bending coordinates. The yielded moment is the
/// gradient of the incremental Huber energy; its tangent has zero stiffness
/// along continued plastic loading and positive transverse stiffness.
package func cableBendResponse(_ strain: F3, plastic: F3, yieldAngle: Float)
    -> (elastic: F3, tangent: Mat3Rows) {
    let trial = strain - plastic
    let bend = F3(trial.x, trial.y, 0), magnitude = length(bend)
    guard yieldAngle > 0 && magnitude > yieldAngle else {
        return (trial, .identity)
    }
    let direction = bend / magnitude, scale = yieldAngle / magnitude
    var tangent = Mat3Rows.diagonal(F3(scale, scale, 1))
    tangent += Mat3Rows(rowMajor: outer(direction, direction)) * -scale
    return (F3(bend.x * scale, bend.y * scale, trial.z), tangent)
}

/// Finite material energy in the AVBD body blocks shared with contacts and
/// hard joints. Physical coefficients are never penalty-ramped or capped.
final class CPUCable: CPUForce {
    let rA: F3, rB: F3
    let material: CableJointMaterial
    let restRel: Quat
    var initialLinear: F3 = .zero
    var initialAngular: F3 = .zero
    var plasticAngular: F3 = .zero

    init(solver: CPUSolver, bodyA: CPURigid?, bodyB: CPURigid,
         rA: F3, rB: F3, material: CableJointMaterial) {
        self.rA = rA; self.rB = rB; self.material = material
        restRel = ((bodyA?.positionAng ?? Quat(real: 1, imag: .zero)).inverse
                   * bodyB.positionAng).normalized
        super.init(solver: solver, bodyA: bodyA, bodyB: bodyB)
    }

    override func initialize() -> Result<Bool, CPUSolver.RuntimeFailure> {
        guard let bodyB else { return .success(false) }
        guard material.dampingTime > 0 || material.yieldAngle > 0 else { return .success(true) }
        let qA = bodyA?.positionAng ?? Quat(real: 1, imag: .zero)
        let xA = bodyA?.positionLin ?? .zero
        let pB = transform(bodyB.positionLin, bodyB.positionAng, rB)
        initialLinear = qA.inverse.act(pB - xA) - rA
        if material.angularStiffness.max() > 0 {
            initialAngular = cableRotationLog((qA * restRel).inverse * bodyB.positionAng).value
            if material.yieldAngle > 0 {
                // Commit only the preceding accepted pose, never an iterate.
                let response = cableBendResponse(initialAngular, plastic: plasticAngular,
                                                 yieldAngle: material.yieldAngle)
                plasticAngular = initialAngular - response.elastic
            }
        }
        return .success(true)
    }

    override func updatePrimal(_ body: CPURigid, _ alpha: Float,
                               _ lhsLin: inout Mat3Rows, _ lhsAng: inout Mat3Rows,
                               _ lhsCross: inout Mat3Rows,
                               _ rhsLin: inout F3, _ rhsAng: inout F3) {
        guard let bodyB else { return }
        let isA = body === bodyA
        let qA = bodyA?.positionAng ?? Quat(real: 1, imag: .zero)
        let xA = bodyA?.positionLin ?? .zero
        let strain = cableLinearStrain(xA: xA, qA: qA,
            xB: bodyB.positionLin, qB: bodyB.positionAng, rA: rA, rB: rB, isA: isA)
        let c = strain.value, jLin = strain.linear, jAng = strain.angular
        let damping = material.dampingTime / solver.dt
        let k = material.linearStiffness
        let force = k * (c + damping * (c - initialLinear))
        let stiffness = Mat3Rows.diagonal(k * (1 + damping))
        let jLinTK = jLin.transposed.mul(stiffness)
        let jAngTK = jAng.transposed.mul(stiffness)
        lhsLin += jLinTK.mul(jLin)
        lhsAng += jAngTK.mul(jAng)
        lhsCross += jAngTK.mul(jLin)
        rhsLin += jLin.transposed.mul(force)
        rhsAng += jAng.transposed.mul(force)

        if material.angularStiffness.max() > 0 {
            let frame = (qA * restRel).normalized
            let strain = cableRotationLog(frame.inverse * bodyB.positionAng)
            let j = strain.derivative.mul(
                Mat3Rows(rowMajor: simd_float3x3(frame).transpose)) * (isA ? -1 : 1)
            let kAng = material.angularStiffness
            let response = cableBendResponse(strain.value, plastic: plasticAngular,
                                             yieldAngle: material.yieldAngle)
            let f = kAng * (response.elastic + damping * (strain.value - initialAngular))
            var tangent = Mat3Rows.diagonal(kAng).mul(response.tangent)
            tangent += Mat3Rows.diagonal(kAng * damping)
            lhsAng += j.transposed.mul(tangent).mul(j)
            rhsAng += j.transposed.mul(f)
        }
    }
}
