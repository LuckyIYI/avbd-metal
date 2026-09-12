import Foundation
import simd
import SimCore
import PhysicsAVBD
import GPUSimDemos

// Independent double-precision geometric oracle; never uses solver contacts.
private func segmentDistance(_ af: F3, _ bf: F3, _ cf: F3, _ df: F3) -> Float {
    let a = SIMD3<Double>(af), b = SIMD3<Double>(bf), c = SIMD3<Double>(cf), d = SIMD3<Double>(df)
    let u = b - a, v = d - c, w = a - c
    let aa = dot(u,u), bb = dot(u,v), cc = dot(v,v), dd = dot(u,w), ee = dot(v,w)
    let det = aa * cc - bb * bb
    var best = Double.infinity
    func point(_ p: SIMD3<Double>, _ x: SIMD3<Double>, _ y: SIMD3<Double>) -> Double {
        let e = y - x
        let t = min(max(dot(p - x,e) / dot(e,e), 0), 1)
        return length(p - x - t * e)
    }
    for value in [point(a,c,d), point(b,c,d), point(c,a,b), point(d,a,b)] { best = min(best, value) }
    if det > 1e-14 * aa * cc {
        let s = (bb * ee - cc * dd) / det, t = (aa * ee - bb * dd) / det
        if s >= 0 && s <= 1 && t >= 0 && t <= 1 { best = min(best, length(w + s * u - t * v)) }
    }
    return Float(best)
}

// The analytic shader's existing gear-clock regression, runnable on CLT
// installations that provide Metal but do not include XCTest.
func validateGearClockCompatibility() throws {
    let scene = Demos.gearclock()
    let drive = scene.bodies.firstIndex { $0.shape == .torus && abs($0.size.x - 1.25) < 0.01 }!
    let hand = scene.bodies.firstIndex { $0.shape == .torus && abs($0.size.x - 0.65) < 0.01 }!
    let solver = try GPUSolver(scene: scene)
    func angle(_ body: Int) -> Float {
        let axis = solver.bodyRotation(body).act(F3(1,0,0))
        return atan2(axis.z, axis.x)
    }
    var previous = SIMD2(angle(drive),angle(hand)), total = SIMD2<Float>.zero
    for _ in 0..<1500 {
        try solver.submitStep()
        let current = SIMD2(angle(drive),angle(hand))
        for i in 0..<2 {
            var delta = current[i] - previous[i]
            while delta > .pi { delta -= 2 * .pi }
            while delta < -.pi { delta += 2 * .pi }
            total[i] += delta
        }
        previous = current
    }
    try require(total.x < -0.2 && total.y > 0.4, "gear clock must advance and counter-rotate: \(total)")
    try require(abs(abs(total.y / total.x) - 1.25 / 0.65) < 0.35, "gear pitch ratio changed: \(total)")
    print("PASS analytic gear-clock compatibility, drive/hand rotations=\(total)")
}

func validateTwistingContact() throws {
    let args = CommandLine.arguments
    func value(_ key: String, default fallback: Int) -> Int {
        guard let i = args.firstIndex(of: key), i+1 < args.count else { return fallback }
        return Int(args[i+1]) ?? fallback
    }
    var s = Demos.cableTwisting(segments: value("--segments", default: 40))
    if args.contains("--mixed") {
        let signs = [-1, 1]
        var vertices: [F3] = []
        vertices.reserveCapacity(8)
        for x in signs {
            for y in signs {
                for z in signs {
                    let point = F3(Float(x), Float(y), Float(z))
                    vertices.append(point * 0.1)
                }
            }
        }
        let hull = s.addBody(size: F3(repeating: 0.2), density: 0, friction: 0.5,
                             position: F3(10,10,0), collisionEnabled: false)
        _ = s.addCollider(body: hull, size: F3(repeating: 0.2), convexHullVertices: vertices)
    }
    if let i = args.firstIndex(of: "--alpha"), i+1 < args.count, let value = Float(args[i+1]) {
        s.settings.alpha = value
    }
    if let i = args.firstIndex(of: "--iterations"), i+1 < args.count, let value = Int(args[i+1]) {
        s.settings.iterations = value
    }
    let cpuOnly = args.contains("--cpu-only")
    let solver: GPUSolver? = cpuOnly ? nil : try GPUSolver(scene: s)
    let cpu: CPUSolver? = cpuOnly ? try s.makeCPUSolverChecked() : nil
    func position(_ body: Int) -> F3 { cpu?.bodies[body].positionLin ?? solver!.bodyPosition(body) }
    func rotation(_ body: Int) -> Quat { cpu?.bodies[body].positionAng ?? solver!.bodyRotation(body) }
    print("TWIST backend=\(cpuOnly ? "CPU" : "Metal") segments=\(s.cables[0].bodyIDs.count) mixed=\(args.contains("--mixed")) iterations=\(s.settings.iterations) alpha=\(s.settings.alpha)")
    var worst: Float = 0, maximumGap: Float = 0
    let excluded = Set(s.collisionExclusions.map {
        UInt64(min($0.bodyA,$0.bodyB)) << 32 | UInt64(max($0.bodyA,$0.bodyB))
    })
    for frame in 0..<value("--steps", default: 3600) {
        if let cpu { try cpu.stepChecked() }
        else { try solver!.submitStep(); try solver!.synchronize() }
        var segments: [(a: F3,b: F3,r: Float,body: Int,cable: Int,index: Int)] = []
        for (c, cable) in s.cables.enumerated() {
            for (i, body) in cable.bodyIDs.enumerated() {
                let p = position(body)
                let q = rotation(body)
                try require(p.x.isFinite && p.y.isFinite && p.z.isFinite
                    && length(p) < 20 && length(q.vector).isFinite,
                    "twisting body \(body) is non-finite or escaped at step \(frame)")
                let axis = q.act(F3(0, 0, cable.restLengths[i] / 2))
                segments.append((p-axis,p+axis,cable.radius,body,c,i))
            }
            for id in cable.jointIDs {
                let j = s.joints[id]
                let gap = length(position(j.bodyA) + rotation(j.bodyA).act(j.rA)
                    - position(j.bodyB) - rotation(j.bodyB).act(j.rB))
                try require(gap.isFinite && gap < 0.025, "twisting connector gap \(gap)")
                maximumGap = max(maximumGap, gap)
            }
        }
        var deepest: Float = 0, pair = (0,0)
        for i in segments.indices {
            for j in (i+1)..<segments.count {
                let a = segments[i], b = segments[j]
                if a.cable == b.cable && abs(a.index-b.index) <= 1 { continue }
                if excluded.contains(UInt64(min(a.body,b.body)) << 32 | UInt64(max(a.body,b.body))) { continue }
                let depth = a.r+b.r-segmentDistance(a.a,a.b,b.a,b.b)
                if depth > deepest { deepest = depth; pair = (a.body,b.body) }
            }
        }
        if deepest > worst + 0.001 || frame % 120 == 119 {
            let contact = solver?.activeRigidContactNormalLoads().first {
                ($0.bodyA == pair.0 && $0.bodyB == pair.1) || ($0.bodyA == pair.1 && $0.bodyB == pair.0)
            }
            print("TWIST step=\(frame+1) penetration=\(deepest) pair=\(pair) load=\(String(describing: contact?.normalLoad)) solverDepth=\(String(describing: solver?.debugWorstRigidContactPenetration()))")

        }
        worst = max(worst, deepest)
    }
    try require(worst < 0.006, "twisting cables geometrically interpenetrate by \(worst) m")
    print("PASS twisting capsule geometry, worst penetration=\(worst), maximum connector gap=\(maximumGap)")
}

// Pure axial spin does not change capsule geometry. A fast spinning segment
// dropped onto a box must come to rest at the same height as a nonspinning one.
func validateSpinningRigidContact(gpu: Bool) throws {
    for spin: Float in [0, 80] {
        var s = try scene([F3(-0.2,0,0.12), F3(0.2,0,0.12)], collision: true)
        s.settings.dt = 1 / 120; s.settings.gravity = -9.81
        s.settings.collisionMargin = 0.0015
        s.colliders[0].friction = 0; s.colliders[0].dynamicFriction = 0
        _ = s.addBody(size: F3(4,4,0.2), density: 0, friction: 0, position: F3(0,0,-0.1))
        let solver = gpu ? try GPUSolver(scene: s) : nil
        let cpu = gpu ? nil : try s.makeCPUSolverChecked()
        if let solver {
            solver.setBodyStates([.init(body: 0, position: s.bodies[0].position,
                rotation: s.bodies[0].rotation, linearVelocity: .zero, angularVelocity: F3(spin,0,0))])
        } else { cpu!.bodies[0].velocityAng = F3(spin,0,0) }
        var worst: Float = 0
        for _ in 0..<600 {
            if let solver { try solver.submitStep() } else { try cpu!.stepChecked() }
            let p = solver?.bodyPosition(0) ?? cpu!.bodies[0].positionLin
            let q = solver?.bodyRotation(0) ?? cpu!.bodies[0].positionAng
            let height = p.z - abs(q.act(F3(0,0,0.2)).z)
            try require(height.isFinite, "spinning support contact became non-finite")
            worst = max(worst, 0.03 - height)
        }
        try require(worst < 0.004, "spinning cable/box overlap \(worst)m")
        print("PASS \(gpu ? "Metal" : "CPU") cable/box spin=\(spin)rad/s maxOverlap=\(worst)m")
    }
}
