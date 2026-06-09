import simd

// Showcase demos: hanging bridge, tensegrity, box-chainmail, swirl chute,
// mechanically driven treadmill, jenga, spiral dominoes.

extension Demos {

    /// Hanging plank bridge ("навесной мост"): plank chain anchored to two
    /// static platforms, with heavy boxes dropped onto the middle.
    public static func bridge(planks: Int = 16, drops: Int = 4) -> PhysicsScene {
        var s = PhysicsScene(name: "bridge")
        addGround(&s)
        let plankLen: Float = 1.0, plankW: Float = 2.0, plankT: Float = 0.15
        let span = Float(planks) * plankLen
        let deckZ: Float = 5

        // platforms
        _ = s.addBody(size: F3(4, 4, deckZ), density: 0, friction: 0.8,
                      position: F3(-span / 2 - 2, 0, deckZ / 2))
        _ = s.addBody(size: F3(4, 4, deckZ), density: 0, friction: 0.8,
                      position: F3(span / 2 + 2, 0, deckZ / 2))

        // plank chain, anchored at both ends to world
        var prev = -1
        var first = -1
        for i in 0..<planks {
            let x = -span / 2 + (Float(i) + 0.5) * plankLen
            let idx = s.addBody(size: F3(plankLen * 0.98, plankW, plankT),
                                density: 0.8, friction: 0.8,
                                position: F3(x, 0, deckZ))
            if i == 0 { first = idx }
            if prev >= 0 {
                // hinge along the shared edge: two ball joints across the width
                for yo in [-plankW * 0.4, plankW * 0.4] {
                    s.addJoint(SceneJoint(bodyA: prev, bodyB: idx,
                                          rA: F3(plankLen / 2, yo, 0),
                                          rB: F3(-plankLen / 2, yo, 0)))
                }
            }
            prev = idx
        }
        // world anchors at platform edges
        for yo in [-plankW * 0.4, plankW * 0.4] {
            s.addJoint(SceneJoint(bodyA: -1, bodyB: first,
                                  rA: F3(-span / 2, yo, deckZ),
                                  rB: F3(-plankLen / 2, yo, 0)))
            s.addJoint(SceneJoint(bodyA: -1, bodyB: prev,
                                  rA: F3(span / 2, yo, deckZ),
                                  rB: F3(plankLen / 2, yo, 0)))
        }

        // cargo dropped on the middle
        var rng = SplitMix64(seed: 7)
        for k in 0..<drops {
            _ = s.addBody(size: F3(repeating: 0.8), density: 2, friction: 0.6,
                          position: F3((rng.nextFloat() - 0.5) * span * 0.4,
                                       (rng.nextFloat() - 0.5) * plankW * 0.5,
                                       deckZ + 3 + Float(k) * 1.5))
        }
        return s
    }

    /// Three-strut tensegrity prism: rigid struts suspended purely by
    /// pre-tensioned cables (springs). It should settle, hover, and wobble
    /// elastically when poked.
    public static func tensegrity(towers: Int = 1) -> PhysicsScene {
        var s = PhysicsScene(name: "tensegrity")
        addGround(&s, friction: 0.9)

        let R: Float = 1.0
        let H: Float = 1.6
        let strutLen: Float = 2.5
        let cableK: Float = 3e4
        let preTension: Float = 0.99    // barely taut; gravity provides the rest

        for tower in 0..<towers {
            let baseZ = Float(0.2)
            let cx = Float(tower) * 4
            var bottoms: [F3] = []
            var tops: [F3] = []
            for k in 0..<3 {
                let a = Float(k) * 2 * .pi / 3
                bottoms.append(F3(cx + R * cos(a), R * sin(a), baseZ))
                let at = a + 5 * .pi / 6   // 150° twist
                tops.append(F3(cx + R * cos(at), R * sin(at), baseZ + H))
            }

            var struts: [Int] = []
            for k in 0..<3 {
                let b = bottoms[k], t = tops[k]
                let mid = (b + t) / 2
                let dir = normalize(t - b)
                // rotation aligning local +z to dir
                let zAxis = F3(0, 0, 1)
                let c = dot(zAxis, dir)
                let axis = cross(zAxis, dir)
                let rot: Quat = length(axis) < 1e-5
                    ? Quat(real: 1, imag: .zero)
                    : Quat(angle: acos(simd_clamp(c, -1, 1)), axis: normalize(axis))
                struts.append(s.addBody(size: F3(0.12, 0.12, strutLen), density: 1,
                                        friction: 0.6, position: mid, rotation: rot))
            }

            func cable(_ sa: Int, _ ea: F3, _ sb: Int, _ eb: F3) {
                let pa = s.bodies[sa].position + s.bodies[sa].rotation.act(ea)
                let pb = s.bodies[sb].position + s.bodies[sb].rotation.act(eb)
                s.addSpring(SceneSpring(bodyA: sa, bodyB: sb, rA: ea, rB: eb,
                                        stiffness: cableK,
                                        rest: length(pa - pb) * preTension))
            }
            let bot = F3(0, 0, -strutLen / 2)
            let top = F3(0, 0, strutLen / 2)
            for k in 0..<3 {
                let n = (k + 1) % 3
                cable(struts[k], bot, struts[n], bot)   // bottom triangle
                cable(struts[k], top, struts[n], top)   // top triangle
                cable(struts[k], top, struts[n], bot)   // saddle: t_k - b_{k+1}
            }
        }
        return s
    }

    /// Chainmail sheet from interlocking rigid square rings (each ring is
    /// four bars welded by hard joints), plus primitives dropped on top.
    public static func chainmail(rings: Int = 6, drops: Int = 3) -> PhysicsScene {
        var s = PhysicsScene(name: "chainmail")
        s.settings.iterations = 20      // deep contact chains need convergence
        addGround(&s, friction: 0.6)

        let L: Float = 1.1          // ring outer side
        let c: Float = 0.2          // bar cross-section (thick: interlocks
                                    // survive deeper transient penetration)
        let sheetZ: Float = 8.0

        /// Welded square ring in a plane. orient: 0 = flat (xy), 1 = vertical
        /// xz, 2 = vertical yz. Returns body indices.
        func addRing(center: F3, orient: Int) -> [Int] {
            // bars in ring-local frame: square in xy plane
            let half = L / 2
            let barLen = L
            var bodies: [Int] = []
            let q: Quat
            switch orient {
            case 1: q = Quat(angle: .pi / 2, axis: F3(1, 0, 0))   // xy -> xz
            case 2: q = (Quat(angle: .pi / 2, axis: F3(0, 1, 0)) * Quat(angle: .pi / 2, axis: F3(1, 0, 0))).normalized
            default: q = Quat(real: 1, imag: .zero)
            }
            // local bar definitions: (center offset, size)
            let bars: [(F3, F3)] = [
                (F3(0, -half + c / 2, 0), F3(barLen, c, c)),    // bottom (along x)
                (F3(0, half - c / 2, 0), F3(barLen, c, c)),     // top
                (F3(-half + c / 2, 0, 0), F3(c, barLen - 2 * c, c)),  // left (along y)
                (F3(half - c / 2, 0, 0), F3(c, barLen - 2 * c, c)),   // right
            ]
            for (off, size) in bars {
                bodies.append(s.addBody(size: size, density: 1, friction: 0.4,
                                        position: center + q.act(off), rotation: q))
            }
            // weld corners. Corner points in RING-local frame; bar-local
            // anchor = corner - bar center offset (bars share the ring
            // rotation, so local frames are translated copies).
            let cp = half - c / 2
            let corners: [(Int, Int, F3)] = [
                (0, 2, F3(-cp, -cp, 0)),    // bottom + left
                (0, 3, F3(cp, -cp, 0)),     // bottom + right
                (1, 2, F3(-cp, cp, 0)),     // top + left
                (1, 3, F3(cp, cp, 0)),      // top + right
            ]
            for (a, b, corner) in corners {
                s.addJoint(SceneJoint(bodyA: bodies[a], bodyB: bodies[b],
                                      rA: corner - bars[a].0, rB: corner - bars[b].0,
                                      stiffnessLin: .infinity, stiffnessAng: .infinity))
            }
            // exclude the two non-jointed bar pairs (parallel bars) — they
            // never touch but share the ring; keep collisions off within ring
            s.addJoint(SceneJoint(bodyA: bodies[0], bodyB: bodies[1],
                                  rA: .zero, rB: .zero, stiffnessLin: 0, stiffnessAng: 0))
            s.addJoint(SceneJoint(bodyA: bodies[2], bodyB: bodies[3],
                                  rA: .zero, rB: .zero, stiffnessLin: 0, stiffnessAng: 0))
            return bodies
        }

        // corner posts ("sticks") that pin the sheet corners in the air
        let pitch = L * 1.25   // tight weave: less slack, less sag
        let n = rings
        let extent = Float(n - 1) * pitch / 2
        var cornerRingCenters: [F3] = []
        for sx in [Float(-1), 1] {
            for sy in [Float(-1), 1] {
                let cx = sx * extent, cy = sy * extent
                cornerRingCenters.append(F3(cx, cy, sheetZ))
                // post under each corner, slightly outward
                _ = s.addBody(size: F3(0.25, 0.25, sheetZ + 0.5), density: 0,
                              friction: 0.5,
                              position: F3(cx + sx * 0.9, cy + sy * 0.9,
                                           (sheetZ + 0.5) / 2))
            }
        }
        var cornerBars = [Int?](repeating: nil, count: 4)
        var edgeAnchors: [(Int, F3)] = []      // (bar body, world anchor)
        let mid = (n - 1) / 2
        for j in 0..<n {
            for i in 0..<n {
                let center = F3(Float(i) * pitch - Float(n - 1) * pitch / 2,
                                Float(j) * pitch - Float(n - 1) * pitch / 2,
                                sheetZ)
                let ringBodies = addRing(center: center, orient: 0)
                if (i == 0 || i == n - 1) && (j == 0 || j == n - 1) {
                    let ri = (i == 0 ? 0 : 2) + (j == 0 ? 0 : 1)
                    cornerBars[ri] = ringBodies[0]
                } else if (i == mid && (j == 0 || j == n - 1))
                       || (j == mid && (i == 0 || i == n - 1)) {
                    // mid-edge anchors keep the perimeter from folding in
                    edgeAnchors.append((ringBodies[0],
                                        center + F3(0, -L / 2 + c / 2, 0)))
                }
                if i + 1 < n {
                    _ = addRing(center: center + F3(pitch / 2, 0, 0), orient: 1)
                }
                if j + 1 < n {
                    _ = addRing(center: center + F3(0, pitch / 2, 0), orient: 2)
                }
            }
        }

        // tie each corner ring's bottom bar to world at the sheet height —
        // the visible posts stand right next to the anchors, "holding" them
        for (ri, center) in cornerRingCenters.enumerated() {
            if let bar = cornerBars[ri] {
                let barWorld = center + F3(0, -L / 2 + c / 2, 0)
                s.addJoint(SceneJoint(bodyA: -1, bodyB: bar,
                                      rA: barWorld, rB: .zero))
            }
        }
        for (bar, anchor) in edgeAnchors {
            s.addJoint(SceneJoint(bodyA: -1, bodyB: bar, rA: anchor, rB: .zero))
        }

        // primitives falling on top: mixed boxes and balls
        var rng = SplitMix64(seed: 11)
        for k in 0..<drops {
            let sz = 0.8 + rng.nextFloat() * 0.8
            let pos = F3((rng.nextFloat() - 0.5) * Float(n) * pitch * 0.5,
                         (rng.nextFloat() - 0.5) * Float(n) * pitch * 0.5,
                         sheetZ + 3 + Float(k) * 2)
            if k % 2 == 0 {
                _ = s.addSphere(diameter: sz, density: 1.5, friction: 0.5, position: pos)
            } else {
                _ = s.addBody(size: F3(repeating: sz), density: 1.5, friction: 0.5,
                              position: pos)
            }
        }
        return s
    }

    /// Vortex funnel ("swirl"): balls launched tangentially orbit a banked
    /// cone, spiral inward as friction bleeds energy, and drop through the
    /// center hole. Built from wedged static plates.
    public static func swirl(turns: Int = 3, balls: Int = 40) -> PhysicsScene {
        var s = PhysicsScene(name: "swirl")
        addGround(&s, friction: 0.3)

        let plates = 20
        let holeR: Float = 1.4
        let rimR: Float = 5.6
        let cone: Float = 0.38             // funnel slope (rad)
        let baseZ: Float = 2.6             // hole height above ground
        let rm = (holeR + rimR) / 2
        let radialLen = (rimR - holeR) / cos(cone) + 0.3

        for k in 0..<plates {
            let a = Float(k) / Float(plates) * 2 * .pi
            let mid = F3(rm * cos(a), rm * sin(a), baseZ + (rm - holeR) * tan(cone))
            // plate local x = radial, y = tangent
            let qYaw = Quat(angle: a, axis: F3(0, 0, 1))
            let tangentAxis = F3(-sin(a), cos(a), 0)
            let qCone = Quat(angle: -cone, axis: tangentAxis)  // outer edge up
            let width = 2 * .pi * rm / Float(plates) * 1.18
            _ = s.addBody(size: F3(radialLen, width, 0.2), density: 0, friction: 0.25,
                          position: mid, rotation: (qCone * qYaw).normalized)
        }
        // outer rim wall to keep fast balls in
        for k in 0..<plates {
            let a = (Float(k) + 0.5) / Float(plates) * 2 * .pi
            let z = baseZ + (rimR - holeR) * tan(cone) + 0.8
            let qYaw = Quat(angle: a + .pi / 2, axis: F3(0, 0, 1))
            _ = s.addBody(size: F3(2 * .pi * rimR / Float(plates) * 1.18, 0.2, 2.0),
                          density: 0, friction: 0.05,
                          position: F3(rimR * cos(a), rimR * sin(a), z), rotation: qYaw)
        }

        // balls: tangential launch onto the funnel
        var rng = SplitMix64(seed: 3)
        for k in 0..<balls {
            let a0 = Float(k) / Float(balls) * 2 * .pi
            let r0 = 3.6 + rng.nextFloat() * 1.2
            let z0 = baseZ + (r0 - holeR) * tan(cone) + 1.0 + Float(k % 5) * 0.5
            let speed = 4.5 + rng.nextFloat() * 2
            _ = s.addSphere(diameter: 0.7, density: 1, friction: 0.25,
                            position: F3(r0 * cos(a0), r0 * sin(a0), z0),
                            velocity: F3(-sin(a0), cos(a0), 0) * speed)
        }
        _ = turns
        return s
    }

    /// Treadmill: a row of kinematically spinning rollers conveys boxes.
    /// The surface motion comes purely from collisions with the rotating
    /// rollers (friction drag), not from scripted box velocities.
    public static func treadmill(boxes: Int = 12) -> PhysicsScene {
        var s = PhysicsScene(name: "treadmill")
        addGround(&s)

        let rollerCount = 8
        let rollerSpacing: Float = 1.4
        let beltY: Float = 0
        let beltZ: Float = 1.2

        // Crossed paddle wheels: contact anchors can't convey via friction
        // from a kinematic surface (anchors re-detect each frame), so the
        // conveying is honest mechanics — paddles push boxes along.
        for i in 0..<rollerCount {
            let x = Float(i) * rollerSpacing - Float(rollerCount - 1) * rollerSpacing / 2
            for half in 0..<2 {
                let q = Quat(angle: Float(half) * .pi / 2 + Float(i) * 0.4,
                             axis: F3(0, 1, 0))
                let idx = s.addBody(size: F3(1.6, 5.0, 0.25), density: 0, friction: 0.3,
                                    position: F3(x, beltY, beltZ), rotation: q)
                s.addSpinner(SceneSpinner(body: idx, axis: F3(0, 1, 0), omega: 3.0))
            }
        }

        // slick bed under the paddle wheels: boxes settle on it inside the
        // paddle sweep and get pushed along mechanically
        _ = s.addBody(size: F3(Float(rollerCount) * rollerSpacing + 2, 5.0, 0.7),
                      density: 0, friction: 0.05,
                      position: F3(0, beltY, 0.35))

        // side rails
        for yo in [Float(-2.8), 2.8] {
            _ = s.addBody(size: F3(Float(rollerCount) * rollerSpacing + 1, 0.3, 1.6),
                          density: 0, friction: 0.1,
                          position: F3(0, yo, beltZ + 0.5))
        }

        // boxes dropped at the upstream end, conveyed by roller friction
        var rng = SplitMix64(seed: 21)
        for k in 0..<boxes {
            _ = s.addBody(size: F3(repeating: 0.55 + rng.nextFloat() * 0.3),
                          density: 1, friction: 0.8,
                          position: F3(-Float(rollerCount - 2) * rollerSpacing / 2
                                           + (rng.nextFloat() - 0.5),
                                       (rng.nextFloat() - 0.5) * 3.5,
                                       beltZ + 1.5 + Float(k) * 0.9))
        }
        return s
    }

    /// Jenga tower: alternating 3-block layers. Pull blocks out with the mouse.
    public static func jenga(levels: Int = 18) -> PhysicsScene {
        var s = PhysicsScene(name: "jenga")
        addGround(&s, friction: 0.5)
        let bl: Float = 3.0, bw: Float = 1.0, bh: Float = 0.6
        for level in 0..<levels {
            let z = bh / 2 + Float(level) * bh
            for i in -1...1 {
                if level % 2 == 0 {
                    _ = s.addBody(size: F3(bl, bw * 0.99, bh * 0.99), density: 0.7,
                                  friction: 0.45, position: F3(0, Float(i) * bw, z))
                } else {
                    _ = s.addBody(size: F3(bw * 0.99, bl, bh * 0.99), density: 0.7,
                                  friction: 0.45, position: F3(Float(i) * bw, 0, z))
                }
            }
        }
        return s
    }

    /// Spiral domino run, kicked off by a rolling ball.
    public static func dominoes(count: Int = 80) -> PhysicsScene {
        var s = PhysicsScene(name: "dominoes")
        addGround(&s, friction: 0.5)

        let h: Float = 1.2, w: Float = 0.65, t: Float = 0.18
        let spacing: Float = 0.75
        var arc: Float = 0
        var radius: Float = 2.2
        var firstPos = F3.zero
        var firstDir = F3(1, 0, 0)

        for k in 0..<count {
            let a = arc / radius
            let pos = F3(radius * cos(a), radius * sin(a), h / 2)
            let qa = Quat(angle: a + .pi / 2, axis: F3(0, 0, 1))   // face along tangent
            _ = s.addBody(size: F3(t, w, h), density: 0.8, friction: 0.4,
                          position: pos, rotation: qa)
            if k == 0 {
                firstPos = pos
                firstDir = F3(-sin(a), cos(a), 0)   // travel direction (tangent)
            }
            arc += spacing
            radius += 0.016
        }

        // starter ball approaching the first domino from behind
        _ = s.addBody(size: F3(repeating: 0.6), density: 5, friction: 0.2,
                      position: firstPos - firstDir * 2 + F3(0, 0, -0.2),
                      velocity: firstDir * 8)
        return s
    }
}
