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
    let env = ProcessInfo.processInfo.environment
    let offset = env["ETHERNET_OFFSET_MM"].flatMap(Float.init) ?? 0
    var task = Demos.ethernetInsertionTask(lateralError: offset * 0.001)
    if let v = env["ETHERNET_ITERS"].flatMap(Int.init) { task.scene.settings.iterations = v }
    if let v = env["ETHERNET_DT"].flatMap(Float.init) {
        task.scene.settings.clothViscosity = 1-pow(1-task.scene.settings.clothViscosity,v/task.scene.settings.dt)
        task.scene.settings.dt = v
    }
    if let v = env["ETHERNET_VISCOSITY"].flatMap(Float.init) { task.scene.settings.clothViscosity = v }
    let scene = task.scene
    let solver = try GPUSolver(scene: scene)
    solver.profiling = env["ETHERNET_PROFILE"] != nil
    var run = EthernetInsertionRun(task: task)
    let begin = ProcessInfo.processInfo.systemUptime
    var minJ: Float = 1, peakPenetration: Float = 0, peakGap: Float = 0
    let tip = task.latchNodes.min { scene.bodies[$0].position.x < scene.bodies[$1].position.x }!
    let tipRest = scene.bodies[task.toolBody].rotation.inverse.act(
        scene.bodies[tip].position-scene.bodies[task.toolBody].position)
    var freeTipExcursion: Float = 0, remoteGap: Float = 0
    var holdMin = F3(repeating: Float.infinity), holdMax = F3(repeating: -Float.infinity)
    let remoteBond = scene.joints.first { $0.bodyA == task.remoteConnectorBody && $0.bodyB == task.cable.bodyIDs[0] }!
    let bonds = scene.joints.filter { Set(task.plugNodes).contains($0.bodyB) }
    var trace = ["time_s,command_x_m,nose_x_m,nose_y_m,nose_z_m,fx_N,fy_N,fz_N,stopped,latch_tip_x_m,latch_tip_y_m,latch_tip_z_m"]
    var stoppedFrames = 0
    for step in 0..<Int(ceil(EthernetInsertionTask.duration / scene.settings.dt)) {
        let c = run.command
        solver.setDrivenBodyStates([.init(body: task.wristBody, position: c.position,
            rotation: c.rotation, linearVelocity: c.linearVelocity, angularVelocity: c.angularVelocity)])
        try solver.submitStep()
        let nose = task.noseNodes.reduce(F3.zero) { $0 + solver.bodyPosition($1) } / Float(task.noseNodes.count)
        run.observe(toolPosition: solver.bodyPosition(task.toolBody),
                    toolRotation: solver.bodyRotation(task.toolBody), noseCenter: nose)
        let toolP = solver.bodyPosition(task.toolBody), toolQ = solver.bodyRotation(task.toolBody)
        let tipLocal = toolQ.inverse.act(solver.bodyPosition(tip)-toolP)
        let physicalTime = Float(step)*scene.settings.dt
        if physicalTime < 3.5 { freeTipExcursion = max(freeTipExcursion,length(tipLocal-tipRest)) }
        if physicalTime > 7.5 { holdMin = min(holdMin,tipLocal); holdMax = max(holdMax,tipLocal) }
        remoteGap = max(remoteGap,distance(endpoint(solver,task.cable,start:true),
            scene.bodies[task.remoteConnectorBody].position+remoteBond.rA))
        trace.append("\(Float(step)*scene.settings.dt),\(c.position.x),\(nose.x),\(nose.y),\(nose.z),\(run.force.x),\(run.force.y),\(run.force.z),\(run.stopped),\(tipLocal.x),\(tipLocal.y),\(tipLocal.z)")
        if step % 120 == 119 || run.stopped {
            let positions = scene.bodies.indices.map { solver.bodyPosition($0) }
            for tet in scene.tets {
                let ids = [tet.ids.0,tet.ids.1,tet.ids.2,tet.ids.3]
                let p = ids.map { positions[$0] }, r = ids.map { scene.bodies[$0].position }
                let j = dot(p[1]-p[0],cross(p[2]-p[0],p[3]-p[0])) / dot(r[1]-r[0],cross(r[2]-r[0],r[3]-r[0]))
                minJ = min(minJ,j)
            }
            for bond in bonds {
                peakGap = max(peakGap, length(positions[bond.bodyA]
                    + solver.bodyRotation(bond.bodyA).act(bond.rA) - positions[bond.bodyB]))
            }
            // Independent analytic signed distances, including the particles'
            // contact skin and the socket's rounded latch entry lip.
            for collider in scene.colliders where collider.body == task.socketBody && collider.collisionEnabled {
                let center = scene.bodies[task.socketBody].position + collider.localPosition
                for id in task.plugNodes {
                    let local = collider.localRotation.inverse.act(positions[id]-center)
                    let distance: Float
                    if collider.shape == .capsule {
                        let q=F3(0,0,min(max(local.z,-collider.size.x/2),collider.size.x/2))
                        distance=length(local-q)-collider.size.y
                    } else {
                        let q=abs(local)-collider.size/2
                        distance=length(max(q,F3.zero))+min(0,q.max())
                    }
                    peakPenetration=max(peakPenetration,max(0,scene.bodies[id].size.x/2-distance))
                }
            }
            if step % 120 == 119 {
                print("ETHERNET t=\(Float(step+1)*scene.settings.dt) phase=\(run.phase) depth_mm=\(nose.x*1000) force_N=\(length(run.force)) peak_N=\(run.peakForce) minJ=\(minJ) penetration_um=\(peakPenetration*1e6) bond_um=\(peakGap*1e6)")
            }
            try require(minJ > 0.98, "excessive PC compression or inversion: J=\(minJ)")
            let cableGap = try checkState(solver,scene)
            try require(cableGap < 0.00025,"Ethernet cable link gap exceeds 0.25 mm")
        }
        if run.stopped {
            stoppedFrames += 1
            if stoppedFrames >= Int(0.25/scene.settings.dt) { break }
        }
    }
    let tracePath = "/tmp/ethernet-trace-\(offset)-\(scene.settings.iterations)-\(scene.settings.dt).csv"
    try trace.joined(separator: "\n").write(toFile: tracePath, atomically: true, encoding: .utf8)
    print("ETHERNET result phase=\(run.phase) nose=\(run.noseCenter) peak_N=\(run.peakForce) minJ=\(minJ) penetration_um=\(peakPenetration*1e6) wall_s=\(ProcessInfo.processInfo.systemUptime-begin) trace=\(tracePath)")
    let holdRange = holdMin.x.isFinite ? (holdMax-holdMin)*1e6 : .zero
    print("LATCH precontact_excursion_um=\(freeTipExcursion*1e6) hold_range_um=\(holdRange) remote_anchor_gap_um=\(remoteGap*1e6)")
    try require(remoteGap < 0.000025,"remote cable termination detached")
    try require(freeTipExcursion < 0.00015,"latch flutter before contact exceeds 0.15 mm")
    if offset == 0 {
        try require((holdMax-holdMin).max() < 0.00005,"seated latch vibration exceeds 50 micrometers")
        try require(run.seated, "nominal insertion must seat without exceeding the force limit")
        try require(peakPenetration < 0.00005, "socket penetration including contact skins: \(peakPenetration)m")
        let deflections = task.contactWires.map { wire -> Float in
            let b = wire.bodyIDs[0]
            let current = solver.bodyPosition(b) + solver.bodyRotation(b).act(wire.endAnchor)
            let initial = scene.bodies[b].position + scene.bodies[b].rotation.act(wire.endAnchor)
            return current.z-initial.z
        }
        print("CONTACT spring deflections_mm=\(deflections.map { $0*1000 })")
        try require(deflections.allSatisfy { $0 > 0.00015 && $0 < 0.002 }, "all eight contacts must physically deflect")
        // A wire may lift its tip while its middle still tunnels into the PC.
        // Check sampled axes against independently inverted world-space tets.
        let tets = scene.tets.map { t -> (F3,simd_float3x3) in
            let p=[t.ids.0,t.ids.1,t.ids.2,t.ids.3].map { solver.bodyPosition($0) }
            return (p[0],simd_float3x3(columns:(p[1]-p[0],p[2]-p[0],p[3]-p[0])).inverse)
        }
        for wire in task.contactWires {
            let a=endpoint(solver,wire,start:true), b=endpoint(solver,wire,start:false)
            for i in 0...64 {
                let point=a+(b-a)*(Float(i)/64)
                for (origin,inv) in tets {
                    let v=inv*(point-origin)
                    try require(!(v.min()>0.0001 && v.x+v.y+v.z<0.9999),
                                "contact wire axis entered the PC housing")
                }
            }
        }
        let toolP=solver.bodyPosition(task.toolBody), toolQ=solver.bodyRotation(task.toolBody)
        let latchBend=task.latchNodes.map { id -> Float in
            let rest=scene.bodies[task.toolBody].rotation.inverse.act(
                scene.bodies[id].position-scene.bodies[task.toolBody].position)
            return length(solver.bodyPosition(id)-toolP-toolQ.act(rest))
        }.max()!
        print("LATCH deformation_mm=\(latchBend*1000)")
        try require(latchBend>0.00008 && latchBend<0.002,"latch must bend under insertion contact")
    } else {
        try require(run.stopped && !run.seated && run.noseCenter.x < 0.010,
                    "misalignment must stop before seating")
    }
    if solver.profiling { print("PROFILE \(solver.profileFrames) \(solver.profileNS.sorted { $0.value > $1.value })") }
    print("PASS Ethernet offset=\(offset)mm iterations=\(scene.settings.iterations)")
}

func validateEthernetTopology() throws {
    let task=Demos.ethernetInsertionTask(), scene=task.scene
    var faces:[[Int]:Int]=[:]
    var volume:Float=0
    for t in scene.tets {
        let ids=[t.ids.0,t.ids.1,t.ids.2,t.ids.3], p=ids.map{scene.bodies[$0].position}
        let v=abs(dot(p[1]-p[0],cross(p[2]-p[0],p[3]-p[0])))/6
        try require(v>1e-15,"degenerate SI plug tet")
        volume += v
        for o in 0..<4 {faces[ids.enumerated().filter{$0.offset != o}.map(\.element).sorted(),default:0] += 1}
    }
    var edges:[[Int]:Int]=[:]
    for (face,count) in faces {
        try require(count<=2,"nonmanifold face")
        if count==1 {for (a,b) in [(0,1),(1,2),(2,0)] {edges[[face[a],face[b]].sorted(),default:0] += 1}}
    }
    try require(edges.values.allSatisfy{$0==2},"open/nonmanifold plug boundary")
    let mass=task.plugNodes.reduce(Float(0)){sum,i in
        let b=scene.bodies[i],r=b.size.x/2
        return sum+b.density*(4 * .pi/3)*r*r*r
    }
    let latchVolume = scene.tris.reduce(Float(0)) { sum, tri in
        let p = [tri.ids.0,tri.ids.1,tri.ids.2].map { scene.bodies[$0].position }
        return sum + length(cross(p[1]-p[0],p[2]-p[0]))/2 * 0.00047
    }
    try require(abs(mass-(volume+latchVolume)*1200)<1e-7,"SI nodal mass mismatch")
    let rootBond = scene.joints.first { $0.bodyA == task.remoteConnectorBody && $0.bodyB == task.cable.bodyIDs[0] }
    try require(rootBond != nil && !scene.bodies[task.remoteConnectorBody].isDynamic,"remote connector must anchor the cable")
    let root = scene.bodies[task.remoteConnectorBody].position+rootBond!.rA
    let endCommand = task.command(at:EthernetInsertionTask.duration)
    let end = endCommand.position+endCommand.rotation.act(F3(-0.004,0,0))
    try require(task.cable.restLengths.reduce(0,+)-distance(root,end)>0.04,"insufficient service-loop slack")
    print("PASS SI plug nodes=\(task.plugNodes.count) tets=\(scene.tets.count) latch_triangles=\(scene.tris.count) mass_g=\(mass*1000)")
}
