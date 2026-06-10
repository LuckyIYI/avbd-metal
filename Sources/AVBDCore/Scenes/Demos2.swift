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

    /// Chainmail from interlocking torus rings, hung PURELY MECHANICALLY:
    /// no world constraints on the cloth. Perimeter rings are threaded
    /// around flexible posts (segmented stacks joined by hard constraints,
    /// so they bend slightly but hold straight) and rest on small
    /// orthogonal cross-sticks welded near the post tops.
    public static func chainmail(rings: Int = 6, drops: Int = 3) -> PhysicsScene {
        var s = PhysicsScene(name: "chainmail")
        s.settings.iterations = 25
        s.settings.betaLin = 20000  // fast contact stiffening for snap loads
        s.settings.lambdaMax = 1500 // wedged links must not stockpile force
        addGround(&s, friction: 0.6)

        let R: Float = 0.45         // major (spine) radius
        let r: Float = 0.15         // minor (tube) radius
        let sheetZ: Float = 8.0
        let pitch: Float = 1.15     // (1.15/2-0.45)+0.15 = 0.275 < 0.30 ok
        let n = rings
        let extent = Float(n - 1) * pitch / 2
        let mid = (n - 1) / 2

        /// Flexible post through (x, y): static base + hard-jointed segment
        /// stack + welded orthogonal peg sticks just below the sheet height.
        func addPost(x: Float, y: Float) {
            let segH: Float = 1.6
            let cross: Float = 0.22
            let pegZ = sheetZ - 0.3
            let topZ = sheetZ + 1.0
            var prev = s.addBody(size: F3(cross, cross, segH), density: 0,
                                 friction: 0.4, position: F3(x, y, segH / 2))
            var z = segH
            while z < topZ {
                let h = min(segH, topZ - z)
                let seg = s.addBody(size: F3(cross, cross, h), density: 4,
                                    friction: 0.4, position: F3(x, y, z + h / 2))
                // hard weld: bends a little under load, holds mostly straight
                s.addJoint(SceneJoint(bodyA: prev, bodyB: seg,
                                      rA: F3(0, 0, s.bodies[prev].size.z / 2),
                                      rB: F3(0, 0, -h / 2),
                                      stiffnessLin: .infinity, stiffnessAng: .infinity))
                prev = seg
                z += h
            }
            // orthogonal peg sticks (a small +) welded to the carrying segment
            let segTopZ = z
            let pegHost = prev
            let hostC = F3(x, y, segTopZ - min(segH, segTopZ) / 2)
            _ = hostC
            for axis in 0..<2 {
                let size = axis == 0 ? F3(1.1, 0.14, 0.14) : F3(0.14, 1.1, 0.14)
                let peg = s.addBody(size: size, density: 2, friction: 0.5,
                                    position: F3(x, y, pegZ))
                // weld peg to the post segment at the peg height
                let segCenterZ = s.bodies[pegHost].position.z
                s.addJoint(SceneJoint(bodyA: pegHost, bodyB: peg,
                                      rA: F3(0, 0, pegZ - segCenterZ), rB: .zero,
                                      stiffnessLin: .infinity, stiffnessAng: .infinity))
                s.addJoint(SceneJoint(bodyA: pegHost, bodyB: peg,
                                      rA: F3(0, 0, pegZ - segCenterZ)
                                          + (axis == 0 ? F3(0.4, 0, 0) : F3(0, 0.4, 0)),
                                      rB: axis == 0 ? F3(0.4, 0, 0) : F3(0, 0.4, 0),
                                      stiffnessLin: .infinity, stiffnessAng: .infinity))
            }
        }

        // posts at the perimeter corner + mid-edge ring positions
        var postCells: Set<Int> = []
        for j in 0..<n {
            for i in 0..<n {
                let corner = (i == 0 || i == n - 1) && (j == 0 || j == n - 1)
                let edgeMid = (i == mid && (j == 0 || j == n - 1))
                           || (j == mid && (i == 0 || i == n - 1))
                if corner || edgeMid { postCells.insert(j * n + i) }
            }
        }
        for cell in postCells {
            let i = cell % n, j = cell / n
            addPost(x: Float(i) * pitch - extent, y: Float(j) * pitch - extent)
        }

        let qFlat = Quat(real: 1, imag: .zero)
        let qX = Quat(angle: .pi / 2, axis: F3(1, 0, 0))
        let qY = (Quat(angle: .pi / 2, axis: F3(0, 0, 1)) * qX).normalized

        for j in 0..<n {
            for i in 0..<n {
                let cx = Float(i) * pitch - extent
                let cy = Float(j) * pitch - extent
                // rings at post cells spawn threaded AROUND the post, just
                // above the peg; gravity drops them onto the cross-sticks
                _ = s.addTorus(major: R, minor: r, density: 1, friction: 0.3,
                               position: F3(cx, cy, sheetZ), rotation: qFlat)
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

        // primitives falling on top (after the sheet settles onto the pegs)
        var rng = SplitMix64(seed: 11)
        for k in 0..<drops {
            let sz = 0.9 + rng.nextFloat() * 0.7
            let pos = F3((rng.nextFloat() - 0.5) * Float(n) * pitch * 0.4,
                         (rng.nextFloat() - 0.5) * Float(n) * pitch * 0.4,
                         sheetZ + 4 + Float(k) * 2)
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

    /// Roller conveyor: the surface IS spinning blocks on motors — a row of
    /// octagonal rollers (kinematic spinners); spin-aware contact friction
    /// conveys whatever rides on them.
    public static func treadmill(boxes: Int = 10) -> PhysicsScene {
        var s = PhysicsScene(name: "treadmill")
        addGround(&s)

        let rollerCount = 10
        let spacing: Float = 1.05
        let rollerZ: Float = 1.1
        let rollerW: Float = 5.0
        let diam: Float = 0.95

        for i in 0..<rollerCount {
            let x = Float(i) * spacing - Float(rollerCount - 1) * spacing / 2
            // proper octagonal roller: 4 slabs with thickness = diam*cos(22.5°)
            // (regular octagon = intersection of 4 such slabs) -> only ~8%
            // radius ripple, smooth ride
            for k in 0..<4 {
                let idx = s.addBody(size: F3(diam, rollerW, diam * 0.924), density: 0,
                                    friction: 1.0,
                                    position: F3(x, 0, rollerZ),
                                    rotation: Quat(angle: Float(k) * .pi / 4,
                                                   axis: F3(0, 1, 0)))
                s.addSpinner(SceneSpinner(body: idx, axis: F3(0, 1, 0), omega: 2.5))
            }
        }

        // side rails
        for yo in [Float(-2.7), 2.7] {
            _ = s.addBody(size: F3(Float(rollerCount) * spacing + 1.5, 0.25, 1.4),
                          density: 0, friction: 0.05,
                          position: F3(0, yo, rollerZ + 0.6))
        }

        // cargo dropped at the upstream end
        var rng = SplitMix64(seed: 21)
        for k in 0..<boxes {
            _ = s.addBody(size: F3(repeating: 0.5 + rng.nextFloat() * 0.3),
                          density: 0.8, friction: 0.8,
                          position: F3(-Float(rollerCount - 2) * spacing / 2
                                           + Float(k % 3) * 0.9,
                                       (rng.nextFloat() - 0.5) * 4,
                                       rollerZ + 1.3 + Float(k / 3) * 0.9))
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
