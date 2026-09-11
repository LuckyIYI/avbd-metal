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
    try validateEthernetDemo()
    try validatePlasticDemo()
    for name in ["cabletwisting", "cablegrippers"] {
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
            // Peel upward from the free end without modifying contacts or pads.
            for frame in 0..<360 {
                let t = min(Float(frame) / 300, 1)
                let target = p + F3(t, -2.2 * t, 0.7 * t)
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
            try require(tip.y < -1.3, "mouse drag failed: \(tip)")
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

/// Two minutes unattended, sampling every simulation step. This catches
/// detached links that a short pull-and-release demonstration can miss.
func validateRoutingSoak() throws {
    let scene = Demos.cableGrippers()
    let solver = try GPUSolver(scene: scene)
    var peakGap: Float = 0
    for step in 0..<28800 {
        try solver.submitStep()
        peakGap = max(peakGap, try checkState(solver, scene))
        if (step + 1) % 7200 == 0 {
            print("ROUTING idle_s=\((step + 1) / 240) peak_gap_m=\(peakGap)")
        }
    }
    print("PASS two-minute routing soak, peak connector gap=\(peakGap)m")
}

func validateEthernetDemo() throws {
    try validateEthernetTopology()
    var scene = Demos.cableEthernet()
    let slot = scene.addDragSlot()
    let solver = try GPUSolver(scene: scene)
    let cable = scene.cables[0], boot = cable.bodyIDs.last!
    let nodes = scene.bodies.indices.filter { scene.bodies[$0].isParticle }
    let hit = solver.pick(origin: F3(-0.477, -0.059, 1.4), dir: F3(0,0,-1))
    try require(hit.map { scene.bodies[$0.body].isParticle } == true,
                "visible soft plug faces must be pickable between tiny nodal spheres")
    let nose = nodes.filter { scene.bodies[$0].position.x > -0.29 }
    func tip() -> F3 { nose.reduce(F3.zero) { $0 + solver.bodyPosition($1) } / Float(nose.count) }
    var minJ: Float = 1, peakGap: Float = 0
    func check() throws {
        peakGap = max(peakGap, try checkState(solver, scene))
        for tet in scene.tets {
            let ids = [tet.ids.0, tet.ids.1, tet.ids.2, tet.ids.3]
            let p = ids.map { solver.bodyPosition($0) }, r = ids.map { scene.bodies[$0].position }
            let j = dot(p[1]-p[0], cross(p[2]-p[0], p[3]-p[0]))
                / dot(r[1]-r[0], cross(r[2]-r[0], r[3]-r[0]))
            minJ = min(minJ, j)
            try require(j > 0.1, "Ethernet tet inversion/collapse: J=\(j)")
        }
    }
    func drag(_ target: F3, frames: Int) throws {
        let start = endpoint(solver, cable, start: false)
        for frame in 0..<frames {
            let t = min(Float(frame+1) / Float(frames) * 1.25, 1)
            solver.setDrag(jointIndex: slot, body: boot,
                worldTarget: mix(start, target, t: F3(repeating: t)),
                localAnchor: cable.endAnchor, stiffness: 50)
            for _ in 0..<4 { try solver.submitStep() }
            if frame % 30 == 29 { try check() }
        }
        print("ETHERNET target=\(target) boot=\(endpoint(solver, cable, start: false)) nose=\(tip()) minJ=\(minJ)")
    }
    for _ in 0..<240 { try solver.submitStep() }
    try check()
    print("ETHERNET settled nose=\(tip()) boot=\(endpoint(solver, cable, start: false))")
    let initial = endpoint(solver, cable, start: false)
    try drag(F3(0.22, 0, initial.z), frames: 360)
    try require(tip().x > 0.50, "Ethernet plug must enter the rigid socket: \(tip())")
    try require(nose.allSatisfy { solver.bodyPosition($0).x < 0.622 }, "plug passed through back wall")
    try drag(initial, frames: 360)
    try require(tip().x < -0.20, "plug must withdraw intact: \(tip())")
    solver.setDrag(jointIndex: slot, body: nil, worldTarget: .zero, localAnchor: .zero)
    for _ in 0..<480 { try solver.submitStep() }
    try check()
    try require(minJ < 0.95, "soft plug must actually deform under contact")
    try validateEthernetBlockedInsertion()
    print("PASS DEMO cableethernet insertion/withdrawal nodes=\(nodes.count) tets=\(scene.tets.count) minJ=\(minJ) gap=\(peakGap)")
}

func validateEthernetTopology() throws {
    let scene = Demos.cableEthernet()
    var faces: [[Int]: Int] = [:]
    var volume: Float = 0
    for tet in scene.tets {
        let ids = [tet.ids.0,tet.ids.1,tet.ids.2,tet.ids.3]
        let p = ids.map { scene.bodies[$0].position }
        let v = abs(dot(p[1]-p[0], cross(p[2]-p[0],p[3]-p[0]))) / 6
        try require(v > 1e-9, "degenerate Ethernet tet")
        volume += v
        for omitted in 0..<4 {
            faces[ids.enumerated().filter { $0.offset != omitted }.map(\.element).sorted(), default: 0] += 1
        }
    }
    try require(faces.values.allSatisfy { $0 == 1 || $0 == 2 }, "nonmanifold plug faces")
    var edges: [[Int]: Int] = [:]
    for (face,count) in faces where count == 1 {
        for (a,b) in [(0,1),(1,2),(2,0)] { edges[[face[a],face[b]].sorted(), default: 0] += 1 }
    }
    try require(edges.values.allSatisfy { $0 == 2 }, "plug/ribs/latch boundary must be watertight")
    let nodes = scene.bodies.filter(\.isParticle)
    let mass = nodes.reduce(Float(0)) { $0 + $1.density * (4 * .pi / 3) * pow($1.size.x/2, 3) }
    try require(abs(mass-volume*30) < 0.001, "tet mass must follow rest volume")
    print("PASS Ethernet conforming topology, volume=\(volume)m3 mass=\(mass)kg")
}

private func validateEthernetBlockedInsertion() throws {
    var scene = Demos.cableEthernet()
    // Hold the entire cable/plug assembly off-axis, with gravity disabled to
    // isolate the socket wall response from a fall off the insertion bed.
    scene.settings.gravity = 0
    for i in scene.bodies.indices where scene.bodies[i].isDynamic {
        scene.bodies[i].position.y += 0.22
    }
    let slot = scene.addDragSlot(), cable = scene.cables[0]
    let solver = try GPUSolver(scene: scene)
    let nose = scene.bodies.indices.filter { scene.bodies[$0].isParticle && scene.bodies[$0].position.x > -0.29 }
    for frame in 0..<240 {
        let t = min(Float(frame+1)/180, 1)
        solver.setDrag(jointIndex: slot, body: cable.bodyIDs.last!,
            worldTarget: F3(-0.72 + 0.94*t, 0.22, 0.9), localAnchor: cable.endAnchor, stiffness: 50)
        for _ in 0..<4 { try solver.submitStep() }
        if frame % 30 == 29 { _ = try checkState(solver, scene) }
    }
    let center = nose.reduce(F3.zero) { $0 + solver.bodyPosition($1) } / Float(nose.count)
    try require(center.x < 0.30, "misaligned plug passed through socket face: \(center)")
    // Independent oriented-box point oracle over all soft vertices. The
    // allowed 6 mm includes contact tolerance and discretization, not a wall.
    var penetration: Float = 0
    for id in scene.bodies.indices where scene.bodies[id].isParticle {
        let p = solver.bodyPosition(id)
        for c in scene.colliders where !scene.bodies[c.body].isDynamic && c.collisionEnabled && c.shape == .box {
            let b = scene.bodies[c.body]
            let local = c.localRotation.inverse.act(b.rotation.inverse.act(p-b.position)-c.localPosition)
            let inside = c.size/2 - abs(local)
            penetration = max(penetration, min(inside.x,min(inside.y,inside.z)))
        }
    }
    try require(penetration < 0.006, "soft plug penetrated rigid wall by \(penetration)m")
    print("PASS off-axis insertion blocked: nose=\(center), max wall penetration=\(penetration)m")
}
