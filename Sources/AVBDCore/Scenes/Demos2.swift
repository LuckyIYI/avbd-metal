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

extension Demos {
    /// A faithful mechanical car: no engine, no wheel joints. Each wheel is
    /// an inner torus (hub) threaded around an axle stick welded to the
    /// body — retained by an end cap, spinning freely as a real bearing —
    /// with spokes out to an outer torus that is the tire. The car rolls
    /// down a curly banked track on inertia alone.
    /// Assemble the mechanical car at `carPos`. Returns the chassis index.
    @discardableResult
    public static func buildCar(_ s: inout PhysicsScene, carPos: F3, v0: F3,
                                rot: Quat = Quat(real: 1, imag: .zero)) -> Int {
        let wheelR: Float = 0.5         // outer torus major radius
        let tire: Float = 0.13          // outer tube radius
        let hubR: Float = 0.27          // inner torus major radius
        let hubTube: Float = 0.115      // thick tube: deep penetration budget
        let axleR: Float = 0.11         // round axle (capsule), hole 0.155

        func place(_ off: F3) -> F3 { carPos + rot.act(off) }
        func orient(_ q: Quat) -> Quat { (rot * q).normalized }
        let qId = Quat(real: 1, imag: .zero)

        let chassis = s.addBody(size: F3(2.6, 1.1, 0.35), density: 2, friction: 0.3,
                                position: place(.zero), rotation: orient(qId), velocity: v0)

        let qWheel = Quat(angle: .pi / 2, axis: F3(1, 0, 0))   // torus axis -> y

        for sx in [Float(-1), 1] {          // front/back
            for sy in [Float(-1), 1] {      // left/right
                let wheelC = place(F3(sx * 0.95, sy * 0.85, 0))

                // round axle (capsule along y) welded to the chassis
                let qAxle = Quat(angle: .pi / 2, axis: F3(1, 0, 0))   // z -> y
                let axle = s.addCapsule(length: 0.55, radius: axleR,
                                        density: 2, friction: 0.02,
                                        position: place(F3(sx * 0.95, sy * 0.825, 0)),
                                        rotation: orient(qAxle), velocity: v0)
                // capsule local +z maps to world -y under qAxle
                s.addJoint(SceneJoint(bodyA: chassis, bodyB: axle,
                                      rA: F3(sx * 0.95, sy * 0.55, 0),
                                      rB: F3(0, 0, sy * 0.275),
                                      stiffnessLin: .infinity, stiffnessAng: .infinity))
                // end cap so the hub can't slide off
                let cap = s.addBody(size: F3(0.52, 0.09, 0.52), density: 2,
                                    friction: 0.02,
                                    position: place(F3(sx * 0.95, sy * 1.14, 0)),
                                    rotation: orient(qId), velocity: v0)
                s.addJoint(SceneJoint(bodyA: axle, bodyB: cap,
                                      rA: F3(0, 0, -sy * 0.275), rB: F3(0, -sy * 0.04, 0),
                                      stiffnessLin: .infinity, stiffnessAng: .infinity))

                // hubs: TWO inner tori around the axle, spaced like a real
                // double-row bearing — a single thin hub can cant and wedge
                let hub = s.addTorus(major: hubR, minor: hubTube, density: 1.5,
                                     friction: 0.02,
                                     position: wheelC + rot.act(F3(0, -sy * 0.12, 0)),
                                     rotation: orient(qWheel), velocity: v0)
                let hub2 = s.addTorus(major: hubR, minor: hubTube, density: 1.5,
                                      friction: 0.02,
                                      position: wheelC + rot.act(F3(0, sy * 0.12, 0)),
                                      rotation: orient(qWheel), velocity: v0)
                // local z maps to world -y under qWheel: anchors flipped
                s.addJoint(SceneJoint(bodyA: hub, bodyB: hub2,
                                      rA: F3(hubR, 0, -sy * 0.12), rB: F3(hubR, 0, sy * 0.12),
                                      stiffnessLin: .infinity, stiffnessAng: .infinity))
                // tire: outer torus
                let outer = s.addTorus(major: wheelR, minor: tire, density: 1,
                                       friction: 1.2, position: wheelC,
                                       rotation: orient(qWheel), velocity: v0)
                // spokes: three sticks welded hub -> tire
                for k in 0..<3 {
                    let a = Float(k) * 2 * .pi / 3
                    let dir = rot.act(F3(cos(a), 0, sin(a)))    // wheel plane
                    let mid = wheelC + dir * (hubR + wheelR) / 2
                    let spoke = s.addBody(size: F3(wheelR - hubR + 0.1, 0.06, 0.06),
                                          density: 1, friction: 0.1,
                                          position: mid,
                                          rotation: orient(Quat(angle: -a, axis: F3(0, 1, 0))),
                                          velocity: v0)
                    // weld to hub at hub rim (hub1 sits -0.12 off the wheel
                    // plane; world +y = hub-local -z under qWheel)
                    s.addJoint(SceneJoint(bodyA: hub, bodyB: spoke,
                                          rA: F3(cos(a) * hubR, sin(a) * hubR, -sy * 0.12),
                                          rB: F3(-(wheelR - hubR + 0.1) / 2 + 0.05, 0, 0),
                                          stiffnessLin: .infinity, stiffnessAng: .infinity))
                    s.addJoint(SceneJoint(bodyA: spoke, bodyB: outer,
                                          rA: F3((wheelR - hubR + 0.1) / 2 - 0.05, 0, 0),
                                          rB: F3(cos(a) * wheelR, sin(a) * wheelR, 0),
                                          stiffnessLin: .infinity, stiffnessAng: .infinity))
                }
                // exclusions: same-assembly parts that come close but must
                // never collide (hubs/tire, spokes vs axle+cap+chassis)
                for h in [hub, hub2] {
                    s.addJoint(SceneJoint(bodyA: h, bodyB: outer, rA: .zero, rB: .zero,
                                          stiffnessLin: 0, stiffnessAng: 0))
                }
                // NOTE: hub-chassis collisions stay ON — the chassis side is
                // the natural inner retention flange of the bearing
                let spokeStart = s.bodies.count - 3
                for sp in spokeStart..<s.bodies.count {
                    for other in [axle, cap, chassis, hub2] {
                        s.addJoint(SceneJoint(bodyA: other, bodyB: sp,
                                              rA: .zero, rB: .zero,
                                              stiffnessLin: 0, stiffnessAng: 0))
                    }
                }
            }
        }
        return chassis
    }

    public static func car(trackScale: Int = 1) -> PhysicsScene {
        var s = PhysicsScene(name: "car")
        s.settings.iterations = 20
        s.settings.betaLin = 20000      // bearings need fast contact stiffening
        addGround(&s, friction: 0.6)

        let chassisZ0: Float = 10.47
        let startX: Float = -20.0
        let wheelR: Float = 0.5
        // flat launch platform start: car spawns resting on its wheels
        buildCar(&s, carPos: F3(startX, 0, chassisZ0), v0: F3(4.0, 0, 0))

        // ---------- the track ----------
        // piecewise path: steep ramp -> banked right curve -> straight ->
        // banked left curve -> runout. Heading/pitch/bank per segment.
        struct Seg { var len: Float; var turn: Float; var pitch: Float; var bank: Float }
        let L = Float(max(1, trackScale))
        let segs: [Seg] = [
            Seg(len: 5, turn: 0, pitch: -0.01, bank: 0),    // launch platform
            Seg(len: 16, turn: 0, pitch: -0.38, bank: 0),
            Seg(len: 10 * L, turn: -.pi / 2, pitch: -0.10, bank: -0.3),
            Seg(len: 7, turn: 0, pitch: -0.18, bank: 0),
            Seg(len: 10 * L, turn: .pi / 2, pitch: -0.10, bank: 0.3),
            Seg(len: 12, turn: 0, pitch: -0.03, bank: 0),
        ]

        var pos = F3(startX - 1.5, 0, chassisZ0 - wheelR - 0.2)
        var heading: Float = 0
        let step: Float = 0.9
        for seg in segs {
            let n = Int(seg.len / step)
            for k in 0..<n {
                let dHeading = seg.turn / Float(max(1, n))
                heading += dHeading
                let dir = F3(cos(heading), sin(heading), 0)
                pos += dir * step + F3(0, 0, sin(seg.pitch) * step)
                if pos.z < 0.1 { pos.z = 0.1 }

                let qYaw = Quat(angle: heading, axis: F3(0, 0, 1))
                let right = F3(sin(heading), -cos(heading), 0)
                let qPitch = Quat(angle: seg.pitch, axis: right)
                let qBank = Quat(angle: seg.bank, axis: dir)
                let q = (qBank * qPitch * qYaw).normalized
                _ = s.addBody(size: F3(1.35, 3.2, 0.16), density: 0, friction: 0.8,
                              position: pos, rotation: q)
                // side walls
                for wside in [Float(-1), 1] {
                    let woff = q.act(F3(0, wside * 1.65, 0.3))
                    let qWall = (q * Quat(angle: wside * 0.15, axis: F3(1, 0, 0))).normalized
                    _ = s.addBody(size: F3(1.35, 0.14, 0.7), density: 0, friction: 0.1,
                                  position: pos + woff, rotation: qWall)
                }
                _ = k
            }
        }
        return s
    }
}
