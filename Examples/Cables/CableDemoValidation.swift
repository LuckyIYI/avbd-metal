import Foundation
import simd
import SimCore
import PhysicsAVBD
import GPUSimDemos

private func endpoint(_ solver: GPUSolver, _ cable: SceneCable, start: Bool) -> F3 {
    let body = start ? cable.bodyIDs[0] : cable.bodyIDs.last!
    return solver.bodyPosition(body) + solver.bodyRotation(body).act(start ? cable.startAnchor : cable.endAnchor)
}

private func checkState(_ solver: GPUSolver, _ scene: PhysicsScene) throws -> Float {
    var gap: Float = 0
    for body in scene.bodies.indices {
        let p = solver.bodyPosition(body), q = solver.bodyRotation(body)
        try require(p.x.isFinite && p.y.isFinite && p.z.isFinite && length(q.vector).isFinite,
                    "non-finite body \(body) in \(scene.name)")
        try require(length(p) < 20, "escaped body \(body) in \(scene.name)")
    }
    for cable in scene.cables {
        for id in cable.jointIDs {
            let j = scene.joints[id]
            gap = max(gap, length(solver.bodyPosition(j.bodyA) + solver.bodyRotation(j.bodyA).act(j.rA)
                - solver.bodyPosition(j.bodyB) - solver.bodyRotation(j.bodyB).act(j.rB)))
        }
    }
    try require(gap < 0.025, "cable connector gap \(gap) in \(scene.name)")
    return gap
}

func validateCableDemos() throws {
    try validatePlasticDemo()
    for name in ["cablethreading", "cabletwisting", "cablegrippers"] {
        var s = Demos.make(name)!
        let slot = s.addDragSlot()
        let solver = try GPUSolver(scene: s)
        let cable = s.cables[0]
        let substeps = Int((1 / (60 * s.settings.dt)).rounded())
        let isGrip = name == "cablegrippers"
        let body = cable.bodyIDs.last!
        let local = cable.endAnchor
        let begin = ProcessInfo.processInfo.systemUptime
        let clips = [F3(-1.3, -0.175, 1.55), F3(-1.3, -0.175, 2.18),
                     F3(-0.55, -0.175, 0.9), F3(0.70, -0.175, 0.9)]
        func occupiedClips(maxDistance: Float = 0.065) -> Int {
            clips.filter { clip in
                cable.bodyIDs.enumerated().contains { i, body in
                    let center = solver.bodyPosition(body)
                    let tangent = solver.bodyRotation(body).act(F3(0, 0, 1))
                    let half = cable.restLengths[i] / 2
                    let t = min(max(dot(clip - center, tangent), -half), half)
                    return length(clip - center - tangent * t) < maxDistance
                }
            }.count
        }
        func clipDeformation() -> Float {
            s.bodies.indices.filter { s.bodies[$0].isParticle }.reduce(Float(0)) {
                max($0, length(solver.bodyPosition($1) - s.bodies[$1].position))
            }
        }
        for _ in 0..<(60 * substeps) { try solver.submitStep() }
        try solver.synchronize()
        print("DEMO \(name) settled tip=\(endpoint(solver, cable, start: false)) gap=\(try checkState(solver, s))")
        if isGrip {
            try require(occupiedClips(maxDistance: 0.02) == 4,
                        "all four clip bores must retain the settled cable; occupied=\(occupiedClips(maxDistance: 0.02))")
        }
        var peakDeformation: Float = 0
        if name == "cabletwisting" {
            for _ in 0..<600 { try solver.submitStep() }
        } else {
            let p = endpoint(solver, cable, start: false)
            let hit = solver.pick(origin: p + F3(0, -1, 0), dir: F3(0, 1, 0))
            try require(hit?.body == body, "gold handle must be pickable")
            // Same 50 N/m spring as the app; update at 60 Hz.
            // Threading lifts the tip before feeding it horizontally. Grippers
            // peel upward from the free end without modifying contacts or pads.
            for frame in 0..<360 {
                let t = min(Float(frame) / 300, 1)
                let target: F3
                if isGrip {
                    target = p + F3(1.0 * t, -2.2 * t, 0.7 * t)
                } else if t < 0.2 {
                    target = mix(p, F3(-0.3, 0, 0.86), t: F3(repeating: t / 0.2))
                } else {
                    target = F3(-0.3 + 2.6 * (t - 0.2) / 0.8, 0, 0.86)
                }
                solver.setDrag(jointIndex: slot, body: body, worldTarget: target,
                               localAnchor: local, stiffness: 50)
                for _ in 0..<substeps { try solver.submitStep() }
                if frame % 60 == 59 {
                    print("DEMO \(name) drag \(frame + 1) tip=\(endpoint(solver, cable, start: false)) gap=\(try checkState(solver, s))")
                    if isGrip {
                        peakDeformation = max(peakDeformation, clipDeformation())
                        print("CLIPS occupied=\(occupiedClips()) deformation=\(clipDeformation())")
                    }
                }
            }
            let tip = endpoint(solver, cable, start: false)
            try require(isGrip ? tip.y < -1.3 : tip.x > 1.8, "mouse drag failed: \(tip)")
            if isGrip {
                try require(occupiedClips() <= 2, "mouse pull must pop cable out of at least two clips")
                try require(peakDeformation > 0.002 && peakDeformation < 0.12,
                            "clip lips must flex without escaping: \(peakDeformation)")
            }
            solver.setDrag(jointIndex: slot, body: nil, worldTarget: .zero, localAnchor: .zero)
            for _ in 0..<120 { try solver.submitStep() }
        }
        try solver.synchronize()
        let gap = try checkState(solver, s)
        let elapsed = ProcessInfo.processInfo.systemUptime - begin
        print("PASS DEMO \(name) bodies=\(s.bodies.count) tets=\(s.tets.count) gap=\(gap) total_wall_s=\(elapsed)")
    }
}

// Full interactive-sized cantilevers: grab the same local tip with the same
// mouse spring, release, and compare recovery with permanent deformation.
func validatePlasticDemo() throws {
    var s = Demos.cablePlastic()
    let slots = [s.addDragSlot(), s.addDragSlot()]
    let solver = try GPUSolver(scene: s)
    for _ in 0..<240 { try solver.submitStep() }
    let initial = s.cables.map { endpoint(solver, $0, start: false) }
    for i in 0..<2 {
        let authored = s.bodies[s.cables[i].bodyIDs.last!].position
            + s.bodies[s.cables[i].bodyIDs.last!].rotation.act(s.cables[i].endAnchor)
        try require(length(initial[i] - authored) < 0.04, "cantilever must not buckle before grabbing")
    }
    for frame in 0..<300 {
        let t = min(Float(frame) / 180, 1)
        for i in 0..<2 {
            let cable = s.cables[i]
            solver.setDrag(jointIndex: slots[i], body: cable.bodyIDs.last!,
                worldTarget: initial[i] + F3(0, -0.9 * t, -0.65 * t),
                localAnchor: cable.endAnchor, stiffness: 50)
        }
        for _ in 0..<4 { try solver.submitStep() }
    }
    print("PLASTIC DEMO held tips=\(s.cables.map { endpoint(solver, $0, start: false) }) gap=\(try checkState(solver, s))")
    for slot in slots { solver.setDrag(jointIndex: slot, body: nil, worldTarget: .zero, localAnchor: .zero) }
    for _ in 0..<2400 { try solver.submitStep() }
    let released = s.cables.map { endpoint(solver, $0, start: false) }
    let recovery = length(released[0] - initial[0])
    let retained = length(released[1] - initial[1])
    try require(recovery < 0.10, "elastic cable must recover: \(recovery)m")
    try require(retained > 0.25, "plastic cable must retain a visible bend: \(retained)m")
    print("PASS DEMO cableplastic elasticOffset=\(recovery)m plasticOffset=\(retained)m gap=\(try checkState(solver, s))")
}
