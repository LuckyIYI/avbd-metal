import simd

// The ultimate marble rollercoaster: wire-style rails built from capsule
// segments, descending through a helix, under a hinged windmill, through a
// zigzag, over a hinged seesaw balancer, down a final spiral into a pool.

extension Demos {

    /// Catch funnel: V of two plates + back wall, guiding a falling ball
    /// onto the groove that starts at `railStart` heading `yaw`.
    static func addCatchFunnel(_ s: inout PhysicsScene, over railStart: F3, yaw: Float) {
        let d = F3(cos(yaw), sin(yaw), 0)
        let side = normalize(cross(F3(0, 0, 1), d))
        let base = railStart + F3(0, 0, 0.55)
        for sgn in [Float(-1), 1] {
            // V slot must pass the marble: inner edges leave a 0.7 gap
            let q = (Quat(angle: yaw, axis: F3(0, 0, 1))
                     * Quat(angle: sgn * 0.85, axis: F3(1, 0, 0))).normalized
            _ = s.addBody(size: F3(2.4, 1.2, 0.1), density: 0, friction: 0.05,
                          position: base + side * (sgn * 0.78) + d * 0.2,
                          rotation: q)
        }
        let qb = (Quat(angle: yaw, axis: F3(0, 0, 1))
                  * Quat(angle: 0.25, axis: F3(0, 1, 0))).normalized
        _ = s.addBody(size: F3(0.1, 1.4, 1.2), density: 0, friction: 0.1,
                      position: base + d * 1.25 + F3(0, 0, 0.1), rotation: qb)
    }

    public static func marblerun(marbles: Int = 12) -> PhysicsScene {
        var s = PhysicsScene(name: "marblerun")
        s.settings.iterations = 15
        addGround(&s, friction: 0.5)

        var samples: [(F3, F3, Float)] = []  // (groove pos, dir, turn sign)
        var pos = F3(-10, 0, 12)
        var yaw: Float = 0
        let step: Float = 0.5

        func dirAt(_ pitch: Float) -> F3 {
            normalize(F3(cos(yaw) * cos(pitch), sin(yaw) * cos(pitch), sin(pitch)))
        }
        func straight(_ len: Float, _ pitch: Float) {
            let n = max(1, Int(len / step))
            for _ in 0..<n {
                let d = dirAt(pitch)
                samples.append((pos, d, 0))
                pos += d * step
            }
        }
        func arc(_ totalTurn: Float, _ radius: Float, _ pitch: Float) {
            let n = max(2, Int(abs(totalTurn) * radius / step))
            let dYaw = totalTurn / Float(n)
            for _ in 0..<n {
                yaw += dYaw / 2
                let d = dirAt(pitch)
                samples.append((pos, d, totalTurn > 0 ? 1 : -1))
                pos += d * step
                yaw += dYaw / 2
            }
        }
        func helix(_ turns: Float, _ radius: Float, _ dropPerTurn: Float) {
            arc(turns * 2 * .pi, radius, -atan(dropPerTurn / (2 * .pi * radius)))
        }

        // ---------------- the course ----------------
        straight(9.5, -0.06)                   // start queue (fits the marble train)
        let windmillAt = pos + dirAt(0) * 1.6  // remember a spot mid-straight
        _ = windmillAt
        helix(2, 3.0, 1.6)                     // big spiral down
        straight(2.5, -0.07)
        let windmillPos = pos + dirAt(0) * 1.2
        let windmillYaw = yaw
        straight(2.5, -0.07)
        arc(.pi / 2, 2.2, -0.13)               // zigzag
        arc(-.pi / 2, 2.2, -0.13)
        arc(-.pi / 2, 2.2, -0.13)
        arc(.pi / 2, 2.2, -0.13)
        straight(1.5, -0.06)
        let seesawDrop = pos                   // rails end here; ball drops
        let seesawYaw = yaw

        // seesaw: just below the rail end, slightly ahead
        let seesawC = seesawDrop + dirAt(0) * 1.3 + F3(0, 0, -0.9)

        // resume rails: pick up right under the tipped seesaw's far end
        // (tilt = atan(stop drop / half length) ≈ 0.33 rad)
        pos = seesawC + F3(cos(seesawYaw), sin(seesawYaw), 0) * 1.95 + F3(0, 0, -0.75)
        let resumeStart = pos
        _ = resumeStart
        straight(2.4, -0.10)
        helix(1.5, 2.4, 1.55)                  // final spiral
        straight(3.0, -0.09)
        let poolAt = pos + dirAt(0) * 1.0

        // ---------------- emit rails ----------------
        let gap: Float = 0.42
        let railR: Float = 0.062
        for (p, d, turn) in samples {
            let side = normalize(cross(F3(0, 0, 1), d))
            let axis = cross(F3(0, 0, 1), d)
            let q: Quat = length(axis) < 1e-5
                ? Quat(real: 1, imag: .zero)
                : Quat(angle: acos(simd_clamp(dot(F3(0, 0, 1), d), -1, 1)),
                       axis: normalize(axis))
            for sgn in [Float(-1), 1] {
                _ = s.addCapsule(length: 0.72, radius: railR, density: 0,
                                 friction: 0.18,
                                 position: p + side * (sgn * gap / 2), rotation: q)
            }
            // guard rail on the OUTER side of curves, at ball-belly height,
            // so fast marbles bank against it instead of flying off
            if turn != 0 {
                let outer = -turn      // left turn (+) -> guard on the right
                _ = s.addCapsule(length: 0.72, radius: railR, density: 0,
                                 friction: 0.02,
                                 position: p + side * (outer * (gap / 2 + 0.30))
                                     + F3(0, 0, 0.38),
                                 rotation: q)
            }
        }

        // ---------------- windmill (hinge joint!) ----------------
        do {
            let d = F3(cos(windmillYaw), sin(windmillYaw), 0)
            let side = normalize(cross(F3(0, 0, 1), d))
            // hub low enough that marbles strike the blades square-on and
            // spin the wheel as they pass (it is a free hinge: tippable)
            let hubPos = windmillPos + F3(0, 0, 1.02)
            let core = s.addBody(size: F3(0.22, 0.22, 0.22), density: 0.4,
                                 friction: 0.3, position: hubPos)
            // hinge axis = side (horizontal, perpendicular to travel)
            s.addJoint(SceneJoint(bodyA: -1, bodyB: core, rA: hubPos, rB: .zero,
                                  stiffnessLin: .infinity, stiffnessAng: .infinity,
                                  hingeAxis: side))
            for k in 0..<4 {
                let a = Float(k) * .pi / 2
                // blades rotate in the (d, z) plane
                let bladeDir = d * cos(a) + F3(0, 0, 1) * sin(a)
                let blade = s.addBody(size: F3(0.6, 0.3, 0.06), density: 0.1,
                                      friction: 0.4,
                                      position: hubPos + bladeDir * 0.42,
                                      rotation: (Quat(angle: windmillYaw, axis: F3(0,0,1))
                                                 * Quat(angle: -a, axis: F3(0,1,0))).normalized)
                s.addJoint(SceneJoint(bodyA: core, bodyB: blade,
                                      rA: .zero, rB: F3(-0.42, 0, 0),
                                      stiffnessLin: .infinity, stiffnessAng: .infinity))
            }
        }

        // ---------------- seesaw balancer (hinge joint!) ----------------
        do {
            let d = F3(cos(seesawYaw), sin(seesawYaw), 0)
            let qYaw = Quat(angle: seesawYaw, axis: F3(0, 0, 1))
            let side = normalize(cross(F3(0, 0, 1), d))
            let plank = s.addBody(size: F3(3.4, 0.7, 0.1), density: 0.5,
                                  friction: 0.3, position: seesawC, rotation: qYaw)
            s.addJoint(SceneJoint(bodyA: -1, bodyB: plank, rA: seesawC, rB: .zero,
                                  stiffnessLin: .infinity, stiffnessAng: .infinity,
                                  hingeAxis: F3(0, 1, 0)))    // local y = world side
            // side lips keep the marble on the plank
            for sgn in [Float(-1), 1] {
                let lip = s.addBody(size: F3(3.4, 0.08, 0.22), density: 0.3,
                                    friction: 0.2,
                                    position: seesawC + side * (sgn * 0.36) + F3(0, 0, 0.12),
                                    rotation: qYaw)
                s.addJoint(SceneJoint(bodyA: plank, bodyB: lip,
                                      rA: F3(0, sgn * 0.36, 0.12), rB: .zero,
                                      stiffnessLin: .infinity, stiffnessAng: .infinity))
                s.addJoint(SceneJoint(bodyA: plank, bodyB: lip,
                                      rA: F3(1.2, sgn * 0.36, 0.12), rB: F3(1.2, 0, 0),
                                      stiffnessLin: .infinity, stiffnessAng: .infinity))
            }
            // tilt stops (static) under both ends
            for sgn in [Float(-1), 1] {
                _ = s.addBody(size: F3(0.25, 1.0, 0.5), density: 0, friction: 0.2,
                              position: seesawC + d * (sgn * 1.45) + F3(0, 0, -0.62))
            }
            // flanking walls: the whole seesaw region is a corridor, so a
            // bouncing marble cannot scatter sideways
            for sgn in [Float(-1), 1] {
                _ = s.addBody(size: F3(5.2, 0.12, 2.2), density: 0, friction: 0.1,
                              position: seesawC + side * (sgn * 0.85) + F3(0, 0, 0.5),
                              rotation: qYaw)
            }
            // low wall behind the landing spot (kills backward bounces) —
            // kept BELOW the incoming flight path
            _ = s.addBody(size: F3(0.12, 1.6, 1.1), density: 0, friction: 0.1,
                          position: seesawC - d * 2.0 + F3(0, 0, -0.1), rotation: qYaw)
        }
        // entry chute: both-side guard rails over the first stretch of the
        // resumed track catch any sloppy seesaw exit
        do {
            let d = F3(cos(seesawYaw), sin(seesawYaw), 0)
            let side = normalize(cross(F3(0, 0, 1), d))
            let q = Quat(angle: seesawYaw, axis: F3(0, 0, 1))
                .normalized
            for i in 0..<5 {
                let p = resumeStart + d * (Float(i) * 0.5)
                for sgn in [Float(-1), 1] {
                    _ = s.addCapsule(length: 0.72, radius: 0.062, density: 0,
                                     friction: 0.02,
                                     position: p + side * (sgn * 0.51) + F3(0, 0, 0.38),
                                     rotation: (q * Quat(angle: .pi / 2, axis: F3(0, 1, 0))).normalized)
                }
            }
        }

        // ---------------- pool ----------------
        do {
            let poolCenter = poolAt + F3(1.2, 0, 0)
            let entryDirection = dirAt(0)
            _ = s.addBody(size: F3(4.5, 4.5, 0.25), density: 0, friction: 0.3,
                          position: poolCenter + F3(0, 0, -0.4))
            for (dx, dy, sx, sy) in [(2.2, 0.0, 0.25, 4.5), (-2.2, 0.0, 0.25, 4.5),
                                     (0.0, 2.2, 4.5, 0.25), (0.0, -2.2, 4.5, 0.25)] {
                let offset = F3(Float(dx), Float(dy), 0)
                // Leave the wall facing the incoming track open. The old
                // four-wall pool put a static box directly across the final
                // rail and turned the last helix into a marble traffic jam.
                if dot(normalize(offset), entryDirection) < -0.5 { continue }
                _ = s.addBody(size: F3(Float(sx), Float(sy), 0.8), density: 0,
                              friction: 0.3,
                              position: poolCenter + offset)
            }
        }

        // ---------------- marbles ----------------
        for k in 0..<marbles {
            _ = s.addSphere(diameter: 0.6, density: 1.2, friction: 0.22,
                            position: F3(-10 + Float(k) * 0.78, 0, 12 + 0.30),
                            velocity: F3(0.5, 0, 0))
        }
        return s
    }
}
