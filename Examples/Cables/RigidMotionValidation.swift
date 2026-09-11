import Foundation
import PhysicsAVBD
import SimCore
import simd

/// Independent load and ballistic oracles for the optional common-motion
/// solve. Neither test prescribes any deformable vertex trajectory.
func validateRigidMotionGroups() throws {
    func cube() -> PhysicsScene {
        var s = PhysicsScene(name: "SI stiff cube")
        s.settings.dt = 1 / 480
        s.settings.iterations = 12
        s.settings.gravity = -9.81
        s.settings.deterministic = true
        s.settings.gamma = 1
        let points: [F3] = [
            F3(0, 0, 0), F3(1, 0, 0), F3(1, 1, 0), F3(0, 1, 0),
            F3(0, 0, 1), F3(1, 0, 1), F3(1, 1, 1), F3(0, 1, 1),
        ]
        for p in points {
            _ = s.addParticle(
                radius: 0.000025, mass: 0, position: (p - F3(repeating: 0.5)) * 0.01 + F3(0, 0, 0.05))
        }
        let mu: Float = 2.4e9 / (2 * (1 + 0.37))
        let lambda: Float = 2.4e9 * 0.37 / ((1 + 0.37) * (1 - 2 * 0.37))
        for t in [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)] {
            s.addTet(SceneTet(ids: t, mu: mu, lambda: lambda))
            for i in [t.0, t.1, t.2, t.3] {
                let r = s.bodies[i].size.x / 2
                s.bodies[i].density += (1200e-6 / 24) / (4 * .pi / 3 * r * r * r)
            }
        }
        for i in s.colliders.indices { s.colliders[i].collisionEnabled = false }
        s.rigidMotionGroups = [Array(0..<8)]
        return s
    }
    func mean(_ solver: GPUSolver) -> F3 {
        (0..<8).reduce(F3.zero) { $0 + solver.bodyPosition($1) } / 8
    }
    func volumeRatio(_ solver: GPUSolver, _ scene: PhysicsScene) -> Float {
        scene.tets.map { t -> Float in
            let p = [t.ids.0, t.ids.1, t.ids.2, t.ids.3].map { solver.bodyPosition($0) }
            let r = [t.ids.0, t.ids.1, t.ids.2, t.ids.3].map { scene.bodies[$0].position }
            return dot(p[1] - p[0], cross(p[2] - p[0], p[3] - p[0]))
                / dot(r[1] - r[0], cross(r[2] - r[0], r[3] - r[0]))
        }.min()!
    }
    let free = cube()
    let fall = try GPUSolver(scene: free)
    for _ in 0..<120 { try fall.submitStep() }
    let t = Float(120) * free.settings.dt
    let expected = Float(0.05) - 0.5 * 9.81 * t * (t + free.settings.dt)
    try require(
        abs(mean(fall).z - expected) < 0.0003, "common-motion free fall: \(mean(fall).z) vs \(expected)")
    try require(volumeRatio(fall, free) > 0.999, "common-motion free-fall volume drift")

    var supported = cube()
    supported.settings.particleDamping = 20
    let wrist = supported.addBody(
        size: F3(repeating: 0.01), density: 0, friction: 0, position: F3(0, 0, 0.05))
    let tool = supported.addBody(
        size: F3(repeating: 0.01), density: 60000, friction: 0, position: F3(0, 0, 0.05))
    let anchors = [F3(0, 0.008, 0), F3(0, -0.004, 0.00693), F3(0, -0.004, -0.00693)]
    for a in anchors {
        supported.addJoint(SceneJoint(bodyA: wrist, bodyB: tool, rA: a, rB: a, stiffnessLin: 8000))
    }
    for i in 0..<8 {
        supported.addJoint(
            SceneJoint(
                bodyA: tool, bodyB: i,
                rA: supported.bodies[i].position - supported.bodies[tool].position, rB: .zero,
                stiffnessLin: 1e5))
    }
    for i in supported.colliders.indices { supported.colliders[i].collisionEnabled = false }
    supported.rigidMotionGroups = [[tool] + Array(0..<8)]
    let hold = try GPUSolver(scene: supported)
    for _ in 0..<960 { try hold.submitStep() }
    let f = anchors.reduce(F3.zero) {
        $0 + 8000
            * (supported.bodies[wrist].position + $1 - hold.bodyPosition(tool)
                - hold.bodyRotation(tool).act($1))
    }
    let weight: Float = (0.06 + 0.0012) * 9.81
    try require(
        abs(f.z - weight) < weight * 0.01 && length(F3(f.x, f.y, 0)) < 0.01,
        "common-motion support reaction \(f) vs weight \(weight)N")
    try require(volumeRatio(hold, supported) > 0.999, "support deformed a stiff cube")

    var invalid = cube()
    invalid.rigidMotionGroups = [[0, 1]]
    do {
        _ = try GPUSolver(scene: invalid)
        throw ValidationFailure.failed("partial-tet motion group accepted")
    } catch is GPUSolver.RigidMotionGroupError {}
    print(
        "PASS common-motion SI free fall, volume preservation, weight reaction \(f.z)N and partial-element rejection"
    )
}
