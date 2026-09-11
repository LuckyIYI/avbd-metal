import Foundation
import Metal
import simd
import SimCore
import PhysicsAVBD

enum ValidationFailure: Error, CustomStringConvertible {
    case failed(String)
    var description: String { switch self { case .failed(let message): return message } }
}

func require(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() { throw ValidationFailure.failed(message) }
}

let material = CableMaterial(stretchRigidity: 2000, shearRigidity: 1000,
                             bendRigidity: 0.5, twistRigidity: 0.2,
                             dampingTime: 0.03)

func scene(_ points: [F3], fixed: Set<Int> = [],
           material: CableMaterial = material, collision: Bool = false) throws -> PhysicsScene {
    var scene = PhysicsScene(name: "cable validation")
    scene.settings.dt = 1 / 240
    scene.settings.iterations = 24
    scene.settings.gravity = 0
    scene.settings.rigidLinearDamping = 0
    scene.settings.rigidAngularDamping = 0
    try scene.addCable(points: points, radius: 0.03, density: 1000,
                       material: material, fixedSegments: fixed,
                       collisionEnabled: collision)
    return scene
}

func validateAuthoring() throws {
    var s = try scene([.zero, F3(0, 0, 0.2), F3(0.1, 0, 0.5)])
    let cable = s.cables[0]
    let expectedMass = 1000 * Float.pi * 0.03 * 0.03 * cable.restLengths.reduce(0, +)
    try require(abs(s.bodies.reduce(0) { $0 + $1.mass! } - expectedMass) < 1e-5,
                "cylinder mass must not double-count capsule end caps")
    try require(!s.colliders[0].usesWorldSpaceRoundAnchor, "cable friction needs material anchors")
    let copy = s.replicated(count: 2, spacing: F3(4, 4, 0))
    try require(copy.scene.cables[1].bodyIDs == [2, 3]
                && copy.scene.cables[1].jointIDs == [1], "replicated cable IDs")
    for points in [[F3.zero], [.zero, .zero], [.zero, F3(.nan, 0, 0)]] {
        let count = s.bodies.count
        do {
            try s.addCable(points: points, radius: 0.03, density: 1000, material: material)
            throw ValidationFailure.failed("invalid cable accepted")
        } catch is CableAuthoringError {}
        try require(s.bodies.count == count, "failed authoring must be atomic")
    }
    print("PASS authoring, mass, collision anchors, replication, invalid input")
}

func validateRotationDerivative() throws {
    for angle: Float in [0, 1e-5, 0.01, 0.7, 2.7, 3.13] {
        let q = Quat(angle: angle, axis: normalize(F3(0.2, -0.5, 0.7)))
        let log = cableRotationLog(q)
        for axis in [F3(1, 0, 0), F3(0, 1, 0), F3(0, 0, 1)] {
            let epsilon: Float = 0.0005
            let plus = cableRotationLog(Quat(angle: epsilon, axis: axis) * q).value
            let minus = cableRotationLog(Quat(angle: -epsilon, axis: axis) * q).value
            let numerical = (plus - minus) / (2 * epsilon)
            try require(length(numerical - log.derivative.mul(axis)) < 0.0015,
                        "SO(3) analytic derivative at angle \(angle)")
        }
        try require(length(log.value - cableRotationLog(Quat(vector: -q.vector)).value) < 1e-6,
                    "quaternion sign invariance")
    }
    print("PASS analytic SO(3) Jacobian (rest through near-pi), quaternion sign")
}

func validateLinearDerivatives() throws {
    let xA = F3(-0.2, 0.3, 0.1), xB = F3(0.5, -0.4, 0.3)
    let qA = Quat(angle: 0.8, axis: normalize(F3(1, -2, 3)))
    let qB = Quat(angle: 1.6, axis: normalize(F3(-2, 1, 1)))
    let rA = F3(0, 0, 0.12), rB = F3(0, 0, -0.23)
    let epsilon: Float = 0.0005
    for isA in [false, true] {
        let exact = cableLinearStrain(xA: xA, qA: qA, xB: xB, qB: qB,
                                     rA: rA, rB: rB, isA: isA)
        for axis in [F3(1, 0, 0), F3(0, 1, 0), F3(0, 0, 1)] {
            func perturbed(_ h: Float, angular: Bool) -> F3 {
                let delta = Quat(angle: h, axis: axis)
                return cableLinearStrain(
                    xA: xA + (isA && !angular ? axis * h : .zero),
                    qA: isA && angular ? delta * qA : qA,
                    xB: xB + (!isA && !angular ? axis * h : .zero),
                    qB: !isA && angular ? delta * qB : qB,
                    rA: rA, rB: rB, isA: isA).value
            }
            for angular in [false, true] {
                let numerical = (perturbed(epsilon, angular: angular)
                                 - perturbed(-epsilon, angular: angular)) / (2 * epsilon)
                let analytic = (angular ? exact.angular : exact.linear).mul(axis)
                try require(length(numerical - analytic) < 0.0005,
                            "linear strain Jacobian, parent=\(isA), angular=\(angular)")
            }
        }
    }
    print("PASS linear strain Jacobians for all 12 body DOFs")
}

func run(_ s: PhysicsScene, frames: Int, gpu: Bool) throws -> [(F3, Quat)] {
    if gpu {
        let solver = try GPUSolver(scene: s)
        for _ in 0..<frames { try solver.submitStep() }
        try solver.synchronize()
        return s.bodies.indices.map { (solver.bodyPosition($0), solver.bodyRotation($0)) }
    }
    let solver = try s.makeCPUSolverChecked()
    for _ in 0..<frames { try solver.stepChecked() }
    return solver.bodies.map { ($0.positionLin, $0.positionAng) }
}

func validateRest(gpu: Bool) throws {
    let s = try scene([F3(0, 0, 1), F3(0.2, 0, 1.1), F3(0.3, 0.1, 1.3), F3(0.3, 0.3, 1.4)])
    let poses = try run(s, frames: 60, gpu: gpu)
    for i in s.bodies.indices {
        try require(length(poses[i].0 - s.bodies[i].position) < 2e-4, "curved rest position")
        try require(length(quatSub(poses[i].1, s.bodies[i].rotation)) < 0.002, "curved rest rotation")
    }
    print("PASS \(gpu ? "Metal" : "CPU") curved stress-free rest shape")
}

func validateAxial(gpu: Bool) throws {
    var s = try scene([F3(0, 0, 1.5), F3(0, 0, 1), F3(0, 0, 0.5)], fixed: [0])
    s.settings.gravity = -9.81
    let poses = try run(s, frames: 480, gpu: gpu)
    let expected = s.bodies[1].mass! * 9.81 / s.joints[0].cable!.linearStiffness.z
    let measured = s.bodies[1].position.z - poses[1].0.z
    try require(abs(measured - expected) < max(expected * 0.04, 1e-5),
                "axial Hooke equilibrium: \(measured) vs \(expected)")
    try require(length(poses[0].0 - s.bodies[0].position) < 1e-7, "fixed segment moved")
    print("PASS \(gpu ? "Metal" : "CPU") axial Hooke equilibrium: \(measured) m (expected \(expected))")
}

func validateTwist(gpu: Bool) throws {
    var s = try scene([.zero, F3(0, 0, 0.25), F3(0, 0, 0.5), F3(0, 0, 0.75)], fixed: [0, 2])
    // Capture the untwisted rest state, then rotate the fixed tip in live state.
    let target = Quat(angle: 0.6, axis: F3(0, 0, 1))
    let middle: Quat
    if gpu {
        let solver = try GPUSolver(scene: s)
        solver.setBodyStates([.init(body: 2, position: s.bodies[2].position,
            rotation: target, linearVelocity: .zero, angularVelocity: .zero)])
        for _ in 0..<360 { try solver.submitStep() }
        try solver.synchronize()
        middle = solver.bodyRotation(1)
    } else {
        let solver = try s.makeCPUSolverChecked()
        solver.bodies[2].positionAng = target
        for _ in 0..<360 { try solver.stepChecked() }
        middle = solver.bodies[1].positionAng
    }
    let twist = 2 * atan2(middle.imag.z, middle.real)
    try require(abs(twist - 0.3) < 0.006, "uniform torsion: \(twist) vs 0.3")
    print("PASS \(gpu ? "Metal" : "CPU") uniform torsion: \(twist) rad")
    // Setting twist rigidity to zero must leave the middle segment free.
    for i in s.joints.indices {
        s.joints[i].cable = CableJointMaterial(material: CableMaterial(
            stretchRigidity: 2000, shearRigidity: 1000, bendRigidity: 0.5,
            twistRigidity: 0), restLength: 0.25)
    }
    let freeRotation = Quat(angle: 0.4, axis: F3(0, 0, 1))
    let free: Quat
    if gpu {
        let solver = try GPUSolver(scene: s)
        solver.setBodyStates([.init(body: 1, position: s.bodies[1].position,
            rotation: freeRotation, linearVelocity: .zero, angularVelocity: .zero)])
        for _ in 0..<60 { try solver.submitStep() }
        try solver.synchronize(); free = solver.bodyRotation(1)
    } else {
        let solver = try s.makeCPUSolverChecked()
        solver.bodies[1].positionAng = freeRotation
        for _ in 0..<60 { try solver.stepChecked() }
        free = solver.bodies[1].positionAng
    }
    try require(length(quatSub(free, freeRotation)) < 0.001, "zero twist must be free")
    print("PASS \(gpu ? "Metal" : "CPU") zero twist rigidity permits material spin")
}

func validateBending(gpu: Bool) throws {
    var s = try scene([F3(0, 0, 1), F3(0.25, 0, 1), F3(0.5, 0, 1)], fixed: [0])
    s.settings.gravity = -9.81
    let poses = try run(s, frames: 720, gpu: gpu)
    let tangent = poses[1].1.act(F3(0, 0, 1))
    let angle = atan2(-tangent.z, tangent.x)
    let loadMoment = s.bodies[1].mass! * 9.81 * 0.125
    let k = s.joints[0].cable!.angularStiffness.x
    var expected: Float = loadMoment / k
    for _ in 0..<10 {
        expected -= (k * expected - loadMoment * cos(expected))
            / (k + loadMoment * sin(expected))
    }
    try require(abs(angle - expected) < 0.003,
                "cantilever moment balance: \(angle) vs \(expected)")
    print("PASS \(gpu ? "Metal" : "CPU") nonlinear cantilever moment: \(angle) rad (expected \(expected))")
}

func validateContact(gpu: Bool) throws {
    var s = try scene((0...4).map { F3(Float($0) * 0.15, 0, 0.3) }, collision: true)
    s.settings.gravity = -9.81
    s.settings.collisionMargin = 0.002
    _ = s.addBody(size: F3(4, 4, 0.2), density: 0, friction: 0.5,
                  position: F3(0, 0, -0.1))
    let poses = try run(s, frames: 480, gpu: gpu)
    for i in s.cables[0].bodyIDs {
        let z = poses[i].0.z
        try require(z > 0.026 && z < 0.04, "capsule/floor contact: z=\(z)")
    }
    print("PASS \(gpu ? "Metal" : "CPU") cable drop onto rigid floor")

    // Two perpendicular cable segments meet at an interior capsule witness.
    // This exercises the same pair path used by nonadjacent self-contact.
    var crossing = try scene([F3(-0.3, 0, 0.1), F3(0.3, 0, 0.1)],
                             fixed: [0], collision: true)
    crossing.settings.gravity = -9.81
    crossing.settings.collisionMargin = 0.002
    try crossing.addCable(points: [F3(0, -0.3, 0.4), F3(0, 0.3, 0.4)],
        radius: 0.03, density: 1000, material: material)
    // Isolate the normal response. An unrestrained round segment can roll
    // sideways off this finite support, so height alone is not a contact test.
    var guide = SceneJoint(bodyA: -1, bodyB: 1, rA: F3(0, 0, 0.4), rB: .zero,
                           stiffnessAng: .infinity)
    guide.prismaticAxis = F3(0, 0, 1)
    crossing.addJoint(guide)
    let crossed = try run(crossing, frames: 480, gpu: gpu)
    let separation = crossed[1].0.z - crossed[0].0.z
    try require(abs(separation - 0.058) < 0.004,
                "crossed cable contact: \(separation), center=\(crossed[1].0), tangent=\(crossed[1].1.act(F3(0,0,1)))")
    print("PASS \(gpu ? "Metal" : "CPU") interior cable/cable contact")
}

func validateParityAndReset() throws {
    var s = try scene([F3(0, 0, 1), F3(0.25, 0, 1), F3(0.5, 0, 1)], fixed: [0])
    s.settings.gravity = -9.81
    let cpu = try s.makeCPUSolverChecked(), gpu = try GPUSolver(scene: s)
    for _ in 0..<60 { try cpu.stepChecked(); try gpu.submitStep() }
    try gpu.synchronize()
    let error = length(cpu.bodies[1].positionLin - gpu.bodyPosition(1))
    try require(error < 0.003, "CPU/Metal transient parity: \(error)")
    gpu.setBodyStates(s.bodies.enumerated().map { i, body in
        .init(body: i, position: body.position, rotation: body.rotation,
              linearVelocity: .zero, angularVelocity: .zero)
    })
    let fresh = try GPUSolver(scene: s)
    for _ in 0..<30 { try gpu.submitStep(); try fresh.submitStep() }
    try gpu.synchronize(); try fresh.synchronize()
    try require(length(gpu.bodyPosition(1) - fresh.bodyPosition(1)) < 0.0001,
                "reset must preserve all material stiffness components")
    print("PASS CPU/Metal transient parity (\(error) m), reset/fresh equivalence")
}

func benchmark() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw ValidationFailure.failed("Metal device unavailable")
    }
    print("BENCH device=\(device.name), dt=1/240, iterations=12, 60 warmup + 5 x 120 timed frames")
    for contact in [false, true] {
        for count in [1, 16, 64] {
            var base = try scene((0...16).map { F3(Float($0) * 0.1, 0, contact ? 0.08 : 2) },
                                 fixed: contact ? [] : [0], collision: contact)
            base.settings.gravity = -9.81
            base.settings.iterations = 12
            base.settings.collisionMargin = 0.002
            if contact {
                _ = base.addBody(size: F3(2, 2, 0.2), density: 0, friction: 0.5,
                                  position: F3(0.8, 0, -0.1))
            }
            let batch = base.replicated(count: count, spacing: F3(3, 3, 0), columns: 8).scene
            let solver = try GPUSolver(scene: batch, device: device)
            for _ in 0..<60 { try solver.submitStep() }
            try solver.synchronize()
            var samples: [Double] = []
            for _ in 0..<5 {
                let start = ProcessInfo.processInfo.systemUptime
                for _ in 0..<120 { try solver.submitStep() }
                try solver.synchronize()
                samples.append((ProcessInfo.processInfo.systemUptime - start) * 1000 / 120)
            }
            samples.sort()
            let ms = samples[2]
            var maxGap: Float = 0
            for cable in batch.cables {
                for id in cable.jointIDs {
                    let joint = batch.joints[id]
                    let pA = solver.bodyPosition(joint.bodyA)
                        + solver.bodyRotation(joint.bodyA).act(joint.rA)
                    let pB = solver.bodyPosition(joint.bodyB)
                        + solver.bodyRotation(joint.bodyB).act(joint.rB)
                    let gap = length(pB - pA)
                    try require(gap.isFinite, "benchmark produced non-finite state")
                    maxGap = max(maxGap, gap)
                }
            }
            try require(maxGap < 0.01, "benchmark connector error exceeds 10% of segment length")
            print(String(format: "BENCH contact=%@ cables=%d segments=%d median_wall_ms/frame=%.4f segments/s=%.0f pairs=%d max_gap_m=%.6f",
                         String(contact), count, 16 * count, ms, Double(16 * count) * 1000 / ms,
                         solver.lastNumPairs, maxGap))
        }
    }
}

@main struct CableValidation {
    static func main() throws {
        setbuf(stdout, nil)
        if CommandLine.arguments.contains("--compatibility") {
            try validateGearClockCompatibility()
            return
        }
        if CommandLine.arguments.contains("--collision-stress") {
            try validateTwistingContact()
            return
        }
        if CommandLine.arguments.contains("--demos") {
            try validateCableDemos()
            return
        }
        if CommandLine.arguments.contains("--materials") {
            for gpu in CommandLine.arguments.contains("--cpu-only") ? [false] : [false, true] {
                try validateCableMaterials(gpu: gpu)
            }
            return
        }
        if CommandLine.arguments.contains("--rigid-contact") {
            for gpu in CommandLine.arguments.contains("--cpu-only") ? [false] : [false, true] {
                try validateSpinningRigidContact(gpu: gpu)
            }
            return
        }
        try validateAuthoring()
        try validateRotationDerivative()
        try validateLinearDerivatives()
        let cpuOnly = CommandLine.arguments.contains("--cpu-only")
        for gpu in cpuOnly ? [false] : [false, true] {
            try validateRest(gpu: gpu)
            try validateAxial(gpu: gpu)
            try validateTwist(gpu: gpu)
            try validateBending(gpu: gpu)
            try validateContact(gpu: gpu)
        }
        if !cpuOnly { try validateParityAndReset() }
        if CommandLine.arguments.contains("--benchmark") && !cpuOnly { try benchmark() }
        print("All cable validation checks passed.")
    }
}
