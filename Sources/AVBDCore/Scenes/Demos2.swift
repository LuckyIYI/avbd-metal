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

    /// Chainmail sheet from interlocking REAL torus rings (one rigid torus
    /// body per ring, exact implicit collision). The perimeter hangs from
    /// posts; boxes and balls are dropped onto the sheet.
    public static func chainmail(rings: Int = 6, drops: Int = 3) -> PhysicsScene {
        var s = PhysicsScene(name: "chainmail")
        s.settings.iterations = 25
        s.settings.betaLin = 20000  // fast contact stiffening for snap loads
        s.settings.lambdaMax = 1500 // wedged links must not stockpile force
        addGround(&s, friction: 0.6)

        // Threading geometry: the connector tube crosses the flat ring's
        // plane at |pitch/2 - R| from its center; that offset + r must stay
        // under the hole radius (R - r) or the rings spawn interpenetrating.
        let R: Float = 0.45         // major (spine) radius
        let r: Float = 0.15         // minor (tube) radius
        let sheetZ: Float = 8.0
        let pitch: Float = 1.15     // (1.15/2-0.45)+0.15 = 0.275 < 0.30 ok
        let n = rings
        let extent = Float(n - 1) * pitch / 2

        // posts at corners (visual support for the anchors)
        for sx in [Float(-1), 1] {
            for sy in [Float(-1), 1] {
                _ = s.addBody(size: F3(0.25, 0.25, sheetZ + 0.5), density: 0,
                              friction: 0.5,
                              position: F3(sx * (extent + 0.9), sy * (extent + 0.9),
                                           (sheetZ + 0.5) / 2))
            }
        }

        let qFlat = Quat(real: 1, imag: .zero)                       // xy plane
        let qX = Quat(angle: .pi / 2, axis: F3(1, 0, 0))             // xz plane
        let qY = (Quat(angle: .pi / 2, axis: F3(0, 0, 1)) * qX).normalized  // yz plane

        var anchors: [(Int, F3)] = []
        for j in 0..<n {
            for i in 0..<n {
                let cx = Float(i) * pitch - extent
                let cy = Float(j) * pitch - extent
                let ring = s.addTorus(major: R, minor: r, density: 1, friction: 0.3,
                                      position: F3(cx, cy, sheetZ), rotation: qFlat)
                if i == 0 || i == n - 1 || j == 0 || j == n - 1 {
                    // two anchor points so the perimeter rings can't flip
                    anchors.append((ring, F3(cx + R, cy, sheetZ)))
                    anchors.append((ring, F3(cx - R, cy, sheetZ)))
                }
                if i + 1 < n {
                    _ = s.addTorus(major: R, minor: r, density: 1, friction: 0.3,
                                   position: F3(cx + pitch / 2, cy, sheetZ), rotation: qX)
                }
                if j + 1 < n {
                    _ = s.addTorus(major: R, minor: r, density: 1, friction: 0.3,
                                   position: F3(cx, cy + pitch / 2, sheetZ), rotation: qY)
                }
            }
        }
        var flip = false
        for (ring, anchor) in anchors {
            s.addJoint(SceneJoint(bodyA: -1, bodyB: ring, rA: anchor,
                                  rB: F3(flip ? -R : R, 0, 0)))
            flip.toggle()
        }

        // primitives falling on top: mixed boxes and balls
        var rng = SplitMix64(seed: 11)
        for k in 0..<drops {
            let sz = 0.9 + rng.nextFloat() * 0.7
            // drop AFTER the sheet's own settle transient (high spawn = late
            // arrival), or impacts superpose with the settling oscillation
            let pos = F3((rng.nextFloat() - 0.5) * Float(n) * pitch * 0.5,
                         (rng.nextFloat() - 0.5) * Float(n) * pitch * 0.5,
                         sheetZ + 3 + Float(k) * 2)
            if k % 2 == 0 {
                _ = s.addSphere(diameter: sz, density: 1.0, friction: 0.5, position: pos)
            } else {
                _ = s.addBody(size: F3(repeating: sz), density: 1.0, friction: 0.5,
                              position: pos)
            }
        }
        return s
    }

    /// Waterpark tube slide: a banked U-channel spiraling down; balls are
    /// released at the top, race down the chute, and shoot out the runout.
    public static func swirl(turns: Int = 3, balls: Int = 30) -> PhysicsScene {
        var s = PhysicsScene(name: "swirl")
        addGround(&s, friction: 0.4)

        let R: Float = 5.0                  // slide radius
        let dropPerTurn: Float = 4.6
        let nTurns = Float(turns)
        let segs = Int(nTurns * 30)
        let topZ = nTurns * dropPerTurn + 2.5
        let chanW: Float = 1.5              // channel floor width
        let wallTilt: Float = 0.85          // side plates angle (rad)
        let slope = atan(dropPerTurn / (2 * .pi * R))

        for k in 0..<segs {
            let t = Float(k) / 30.0          // turns travelled
            let a = t * 2 * .pi
            let z = topZ - t * dropPerTurn
            let cpos = F3(R * cos(a), R * sin(a), z)
            let qYaw = Quat(angle: a + .pi / 2, axis: F3(0, 0, 1))
            let radial = F3(cos(a), sin(a), 0)
            let tangent = F3(-sin(a), cos(a), 0)
            let qSlope = Quat(angle: -slope, axis: radial)      // descend along +tangent
            let qBank = Quat(angle: -0.22, axis: tangent)       // mild bank toward mid-lane
            let qSeg = (qBank * qSlope * qYaw).normalized

            // channel floor
            _ = s.addBody(size: F3(1.25, chanW, 0.18), density: 0, friction: 0.25,
                          position: cpos, rotation: qSeg)
            // side plates (angled up-and-out), in segment-local frame
            for side in [Float(-1), 1] {
                let qSide = (qSeg * Quat(angle: side * wallTilt, axis: F3(1, 0, 0))).normalized
                let off = qSeg.act(F3(0, side * (chanW / 2 + 0.32), 0.32))
                _ = s.addBody(size: F3(1.25, 1.0, 0.15), density: 0, friction: 0.15,
                              position: cpos + off, rotation: qSide)
            }
        }

        // balls released at the slide entrance with a push down the chute
        var rng = SplitMix64(seed: 5)
        for k in 0..<balls {
            // spread along the first half-turn so they don't pile up
            let t0f = Float(k % 12) / 30.0
            let a0 = 0.15 + t0f * 2 * .pi
            let t0 = F3(-sin(a0), cos(a0), 0)
            _ = s.addSphere(diameter: 0.55, density: 1, friction: 0.2,
                            position: F3(R * cos(a0), R * sin(a0),
                                         topZ - t0f * dropPerTurn + 0.8 + Float(k / 12) * 0.7)
                                + F3((rng.nextFloat() - 0.5) * 0.4,
                                     (rng.nextFloat() - 0.5) * 0.4, 0),
                            velocity: t0 * 5)
        }
        return s
    }

    /// Conveyor treadmill: a closed loop of hinged planks (a real belt)
    /// wrapped around two spinning paddle wheels. The wheels engage the
    /// belt's inner surface mechanically and drive it around; boxes ride
    /// the moving belt.
    public static func treadmill(boxes: Int = 8) -> PhysicsScene {
        var s = PhysicsScene(name: "treadmill")
        addGround(&s)

        let wheelX: Float = 2.6          // wheel centers at ±wheelX
        let wheelZ: Float = 2.6
        let wrapR: Float = 1.0           // belt wrap radius around wheels
        let beltW: Float = 3.2
        let span = 2 * wheelX
        let circumference = 2 * span + 2 * .pi * wrapR
        let plankCount = 26
        let plankLen = circumference / Float(plankCount)

        // belt path: top span (+x), right wrap, bottom span (-x), left wrap
        func pathPoint(_ sIn: Float) -> (F3, Float) {   // (pos, pitch about y)
            var t = sIn.truncatingRemainder(dividingBy: circumference)
            if t < 0 { t += circumference }
            if t < span {
                return (F3(-wheelX + t, 0, wheelZ + wrapR), 0)
            }
            t -= span
            if t < .pi * wrapR {
                let a = .pi / 2 - t / wrapR     // from +90 (top) to -90 (bottom)
                return (F3(wheelX + wrapR * cos(a), 0, wheelZ + wrapR * sin(a)),
                        .pi / 2 - a)
            }
            t -= .pi * wrapR
            if t < span {
                return (F3(wheelX - t, 0, wheelZ - wrapR), .pi)
            }
            t -= span
            let a = -.pi / 2 - t / wrapR        // from -90 back to -270 (top)
            return (F3(-wheelX + wrapR * cos(a), 0, wheelZ + wrapR * sin(a)),
                    .pi / 2 - a)
        }

        var planks: [Int] = []
        for i in 0..<plankCount {
            let sMid = (Float(i) + 0.5) * plankLen
            let (pos, pitch) = pathPoint(sMid)
            let q = Quat(angle: pitch, axis: F3(0, 1, 0))
            let plank = s.addBody(size: F3(plankLen * 0.98, beltW, 0.1),
                                  density: 0.6, friction: 0.9,
                                  position: pos, rotation: q)
            planks.append(plank)
        }
        // hinge consecutive planks (closed loop), two joints across the width
        for i in 0..<plankCount {
            let nx = (i + 1) % plankCount
            for yo in [-beltW * 0.4, beltW * 0.4] {
                s.addJoint(SceneJoint(bodyA: planks[i], bodyB: planks[nx],
                                      rA: F3(plankLen / 2, yo, 0),
                                      rB: F3(-plankLen / 2, yo, 0)))
            }
        }

        // Kinematic octagon drums: the contact solver now understands
        // spinner surface velocity, so plain friction drives the belt.
        for wx in [-wheelX, wheelX] {
            for k in 0..<4 {
                let idx = s.addBody(size: F3(2 * (wrapR - 0.06), beltW * 0.8, 0.3),
                                    density: 0, friction: 1.2,
                                    position: F3(wx, 0, wheelZ),
                                    rotation: Quat(angle: Float(k) * .pi / 4,
                                                   axis: F3(0, 1, 0)))
                s.addSpinner(SceneSpinner(body: idx, axis: F3(0, 1, 0), omega: 1.2))
            }
        }

        // slick beds: under the top span (carrying side) and under the
        // bottom span (return side), like a conveyor's support + return bed
        _ = s.addBody(size: F3(span - 2.6, beltW, 0.25), density: 0, friction: 0.02,
                      position: F3(0, 0, wheelZ + wrapR - 0.26))
        _ = s.addBody(size: F3(span - 2.6, beltW, 0.25), density: 0, friction: 0.02,
                      position: F3(0, 0, wheelZ - wrapR - 0.45))

        // cargo dropped onto the upstream end of the belt
        var rng = SplitMix64(seed: 21)
        for k in 0..<boxes {
            _ = s.addBody(size: F3(repeating: 0.5 + rng.nextFloat() * 0.25),
                          density: 0.8, friction: 0.9,
                          position: F3(-wheelX + 0.7 + (rng.nextFloat() - 0.5) * 0.6,
                                       (rng.nextFloat() - 0.5) * (beltW - 1),
                                       wheelZ + wrapR + 1.0 + Float(k) * 0.8))
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
