import Foundation
import simd
import SimCore
import PhysicsAVBD

// Isolated torsional ring-down has a closed-form Kelvin–Voigt solution.
// It detects energy injection and damping-history bugs without any contact.
func validateCableMaterials(gpu: Bool) throws {
    let label = gpu ? "Metal" : "CPU"
    for tau: Float in [0, 0.025, 0.12] {
        let mat = CableMaterial(stretchRigidity: 2000, shearRigidity: 1000,
            bendRigidity: 0.1, twistRigidity: 0.0001, dampingTime: tau)
        let s = try scene([.zero, F3(0,0,0.2), F3(0,0,0.4)], fixed: [0], material: mat)
        let solver = gpu ? try GPUSolver(scene: s) : nil
        let cpu = gpu ? nil : try s.makeCPUSolverChecked()
        let q = Quat(angle: 0.4, axis: F3(0,0,1))
        if let solver {
            solver.setBodyStates([.init(body: 1, position: s.bodies[1].position,
                rotation: q, linearVelocity: .zero, angularVelocity: .zero)])
        } else { cpu!.bodies[1].positionAng = q }
        let k = s.joints[0].cable!.angularStiffness.z
        let inertia = s.bodies[1].diagonalInertia!.z
        let omega2 = k / inertia, decay = tau * omega2 / 2
        let frequency = sqrt(omega2 - decay * decay)
        let initialEnergy: Float = 0.5 * k * 0.4 * 0.4
        var previousEnergy = initialEnergy, maximumError: Float = 0
        for frame in 1...3600 {
            if let solver { try solver.submitStep() } else { try cpu!.stepChecked() }
            let rotation = solver?.bodyRotation(1) ?? cpu!.bodies[1].positionAng
            let velocity = solver?.bodyAngularVelocity(1).z ?? cpu!.bodies[1].velocityAng.z
            let angle = 2 * atan2(rotation.imag.z, rotation.real)
            let energy = 0.5 * (k * angle * angle + inertia * velocity * velocity)
            try require(energy.isFinite && energy <= previousEnergy + initialEnergy * 0.0002,
                        "\(label) ring-down injects energy: \(energy) after \(previousEnergy)")
            previousEnergy = energy
            let time = Float(frame) * s.settings.dt
            let analytic = 0.4 * exp(-decay * time)
                * (cos(frequency * time) + decay / frequency * sin(frequency * time))
            maximumError = max(maximumError, abs(angle - analytic))
        }
        // Backward Euler is dissipative even with tau=0; its first-order
        // truncation error accumulates over the 15-second trajectory.
        try require(maximumError < 0.03, "\(label) Kelvin–Voigt analytic error \(maximumError)")
        print("PASS \(label) ring-down tau=\(tau)s energyFraction=\(previousEnergy / initialEnergy) maxAnalyticError=\(maximumError)rad")
    }

    // Bulk translation must survive internal damping; only environmental
    // drag should reduce it. Check the exact exponential drag law.
    for drag: Float in [0, 0.8] {
        var s = try scene([.zero, F3(0,0,0.2), F3(0,0,0.4)])
        s.settings.rigidLinearDamping = drag
        for i in s.bodies.indices { s.bodies[i].velocity = F3(1,0,0) }
        let solver = gpu ? try GPUSolver(scene: s) : nil
        let cpu = gpu ? nil : try s.makeCPUSolverChecked()
        for _ in 0..<240 {
            if let solver { try solver.submitStep() } else { try cpu!.stepChecked() }
        }
        let speed = solver?.bodyVelocity(1).x ?? cpu!.bodies[1].velocityLin.x
        try require(abs(speed - exp(-drag)) < 0.002, "\(label) bulk drag: \(speed)")
        print("PASS \(label) rigid translation drag=\(drag)/s speed=\(speed)m/s")
    }

    // Impose a pure bend with coincident connector anchors, then release.
    // A yielded cable retains angle - yield; the elastic control springs back.
    for yield: Float? in [nil, 0.5] {
        let mat = CableMaterial(stretchRigidity: 2000, shearRigidity: 1000,
            bendRigidity: 0.04, twistRigidity: 0.02,
            dampingTime: 0.12, yieldCurvature: yield)
        var s = try scene([.zero, F3(0,0,0.2), F3(0,0,0.4)], fixed: [0], material: mat)
        s.settings.rigidLinearDamping = 0.8; s.settings.rigidAngularDamping = 0.8
        let solver = gpu ? try GPUSolver(scene: s) : nil
        let cpu = gpu ? nil : try s.makeCPUSolverChecked()
        let q = Quat(angle: 0.5, axis: F3(1,0,0)), p = F3(0,0,0.2) + q.act(F3(0,0,0.1))
        if let solver {
            solver.setDrivenBodyStates([.init(body: 1, position: p, rotation: q,
                linearVelocity: .zero, angularVelocity: .zero)])
        } else { cpu!.bodies[1].positionLin = p; cpu!.bodies[1].positionAng = q }
        for _ in 0..<2400 {
            if let solver { try solver.submitStep() } else { try cpu!.stepChecked() }
        }
        let rotation = solver?.bodyRotation(1) ?? cpu!.bodies[1].positionAng
        let retained = cableRotationLog(rotation).value.x
        let expected: Float = yield == nil ? 0 : 0.4
        try require(abs(retained - expected) < 0.015,
                    "\(label) permanent bend \(retained) expected \(expected)")
        print("PASS \(label) \(yield == nil ? "elastic" : "plastic") release retainedAngle=\(retained)rad")
        if let solver, yield != nil {
            solver.setBodyStates(s.bodies.enumerated().map { i, b in
                .init(body: i, position: b.position, rotation: b.rotation,
                      linearVelocity: .zero, angularVelocity: .zero)
            })
            for _ in 0..<120 { try solver.submitStep() }
            try require(length(quatSub(solver.bodyRotation(1), s.bodies[1].rotation)) < 0.001,
                        "episode reset must clear plastic state")
        }
    }
    try validateBendReturnMap()
}

func validateBendReturnMap() throws {
    let yield: Float = 0.1
    // Loading, unloading and reverse loading: dissipated work is nonnegative.
    var plastic = F3.zero, dissipation: Float = 0
    for i in 0...400 {
        let angle: Float = i <= 100 ? Float(i) * 0.005
            : (i <= 300 ? 0.5 - Float(i - 100) * 0.005 : -0.5 + Float(i - 300) * 0.005)
        let total = F3(angle, 0, 0)
        let response = cableBendResponse(total, plastic: plastic, yieldAngle: yield)
        let updated = total - response.elastic
        let work = dot(response.elastic, updated - plastic)
        try require(work >= -1e-7, "plastic return added energy")
        dissipation += work; plastic = updated
    }
    try require(dissipation > 0.1, "loading cycle must dissipate work")
    // Full two-axis bending tangent, independent finite differences.
    let value = F3(0.3, -0.4, 0.2)
    let response = cableBendResponse(value, plastic: .zero, yieldAngle: yield)
    for axis in [F3(1,0,0), F3(0,1,0), F3(0,0,1)] {
        let h: Float = 0.0001
        let derivative = (cableBendResponse(value + h * axis, plastic: .zero, yieldAngle: yield).elastic
            - cableBendResponse(value - h * axis, plastic: .zero, yieldAngle: yield).elastic) / (2 * h)
        try require(length(derivative - response.tangent.mul(axis)) < 0.001,
                    "plastic tangent disagrees with force derivative")
    }
    print("PASS plastic loading/unloading dissipation and two-axis consistent tangent")
}
