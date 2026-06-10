import simd

// Showcase demos: hanging bridge, tensegrity, box-chainmail, swirl chute,
// mechanically driven treadmill, jenga, spiral dominoes.

extension Demos {

    /// Hanging plank bridge ("навесной мост"): plank chain anchored to two
    /// static platforms, with heavy boxes dropped onto the middle.
    /// Suspension bridge with real construction: towers, main catenary
    /// cables (capsule-link ropes pinned at tower saddles, anchored to the
    /// ground), vertical hanger ropes carrying a segmented deck. The deck
    /// hangs entirely on rope tension, like the real thing.
    public static func bridge(planks: Int = 14, drops: Int = 4) -> PhysicsScene {
        var s = PhysicsScene(name: "bridge")
        s.settings.iterations = 20
        s.settings.lambdaMax = 6000
        addGround(&s, friction: 0.8)

        let plankLen: Float = 1.3, plankW: Float = 2.6, plankT: Float = 0.14
        let span = Float(planks) * plankLen           // suspended main span
        let deckZ: Float = 2.6
        let towerX = span / 2 + 0.6
        let towerTopZ: Float = 7.4
        let anchorX = towerX + 4.2
        let cableY = plankW / 2 + 0.12                // cables run outside the deck
        let sag: Float = 2.9                           // mid-span cable droop

        // ---- towers: two legs + crossbeam per side, static ----
        for tx in [-towerX, towerX] {
            for ty in [-cableY, cableY] {
                _ = s.addBody(size: F3(0.45, 0.45, towerTopZ), density: 0, friction: 0.5,
                              position: F3(tx, ty, towerTopZ / 2))
            }
            _ = s.addBody(size: F3(0.45, cableY * 2 + 0.45, 0.4), density: 0, friction: 0.5,
                          position: F3(tx, 0, towerTopZ - 1.1))
            // ground anchor blocks
            for ax in [anchorX, -anchorX] where (ax > 0) == (tx > 0) {
                _ = s.addBody(size: F3(1.2, cableY * 2 + 0.6, 0.7), density: 0, friction: 0.8,
                              position: F3(ax, 0, 0.35))
            }
        }

        /// rope of capsule links along a world polyline. Ends are pinned to
        /// (body, localAnchor) pairs; body -1 pins to the world at the
        /// polyline end point.
        func addRope(_ pts: [F3], radius: Float, density: Float,
                     pinA: (Int, F3)? = (-1, .zero),
                     pinB: (Int, F3)? = (-1, .zero)) -> [Int] {
            var links: [Int] = []
            var prev = -3                       // -3 = nothing yet
            var prevAnchor = F3.zero
            for k in 0..<(pts.count - 1) {
                let a = pts[k]
                let b = pts[k + 1]
                let d = b - a
                let len = length(d)
                let dir = d / len
                let axis = cross(F3(0, 0, 1), dir)
                var q = Quat(real: 1, imag: .zero)
                if length(axis) > 1e-5 {
                    let c = simd_clamp(dot(F3(0, 0, 1), dir), -1, 1)
                    q = Quat(angle: acos(c), axis: normalize(axis))
                }
                let mid = (a + b) * 0.5
                let link = s.addCapsule(length: len * 0.8, radius: radius,
                                        density: density, friction: 0.3,
                                        position: mid, rotation: q)
                let rB = F3(0, 0, -len / 2)
                if prev == -3 {
                    if let (pb, pr) = pinA {
                        let rA = pb < 0 ? pts[0] : pr
                        s.addJoint(SceneJoint(bodyA: pb, bodyB: link, rA: rA, rB: rB))
                    }
                } else {
                    s.addJoint(SceneJoint(bodyA: prev, bodyB: link,
                                          rA: prevAnchor, rB: rB))
                    s.addJoint(SceneJoint(bodyA: prev, bodyB: link, rA: .zero, rB: .zero,
                                          stiffnessLin: 0, stiffnessAng: 0))
                }
                links.append(link)
                prev = link
                prevAnchor = F3(0, 0, len / 2)
            }
            if let (pb, pr) = pinB {
                let rB2 = pb < 0 ? pts[pts.count - 1] : pr
                s.addJoint(SceneJoint(bodyA: pb, bodyB: links.last!,
                                      rA: rB2, rB: prevAnchor))
            }
            return links
        }

        // ---- main cables: anchor -> tower saddle -> catenary -> saddle -> anchor
        func cableHeight(_ x: Float) -> Float {
            towerTopZ - sag * (1 - (x / towerX) * (x / towerX))
        }
        var cableLinks: [[Int]] = []   // per side: links of the suspended part
        var cableXs: [Float] = []      // x of each suspended sample midpoint
        for sy in [Float(-1), 1] {
            let y = sy * cableY
            // side span: anchor up to the saddle (straight, 5 segments)
            var pts: [F3] = []
            let pA = F3(sy * anchorX, y, 0.45)
            let pB = F3(sy * towerX, y, towerTopZ)
            for k in 0...5 {
                let t = Float(k) / 5
                pts.append(pA + (pB - pA) * t)
            }
            _ = addRope(pts, radius: 0.07, density: 3)
            // main span: catenary between the saddles, fine sampling
            pts = []
            let nSamp = planks * 2
            for k in 0...nSamp {
                let t: Float = Float(k) / Float(nSamp)
                let x: Float = -towerX + t * 2 * towerX
                pts.append(F3(x, y, cableHeight(x)))
            }
            let links = addRope(pts, radius: 0.07, density: 3)
            if sy > 0 {
                cableXs = (0..<links.count).map { (i: Int) -> Float in
                    let t: Float = (Float(i) + 0.5) / Float(nSamp)
                    return -towerX + t * 2 * towerX
                }
            }
            cableLinks.append(links)
        }

        // ---- deck: segmented planks hinged edge to edge ----
        var plankIdx: [Int] = []
        var prev = -1
        for i in 0..<planks {
            let x = -span / 2 + (Float(i) + 0.5) * plankLen
            let idx = s.addBody(size: F3(plankLen * 0.97, plankW, plankT),
                                density: 0.7, friction: 0.8,
                                position: F3(x, 0, deckZ))
            plankIdx.append(idx)
            if prev >= 0 {
                for yo in [-plankW * 0.42, plankW * 0.42] {
                    s.addJoint(SceneJoint(bodyA: prev, bodyB: idx,
                                          rA: F3(plankLen / 2, yo, 0),
                                          rB: F3(-plankLen / 2, yo, 0)))
                }
            }
            prev = idx
        }
        // deck ends tied to short static ramps at the towers
        for (endIdx, sx) in [(plankIdx[0], Float(-1)), (plankIdx[planks - 1], 1)] {
            let ramp = s.addBody(size: F3(1.6, plankW, 0.3), density: 0, friction: 0.8,
                                 position: F3(sx * (span / 2 + 0.85), 0, deckZ - 0.1))
            for yo in [-plankW * 0.42, plankW * 0.42] {
                s.addJoint(SceneJoint(bodyA: ramp, bodyB: endIdx,
                                      rA: F3(-sx * 0.8, yo, 0.1),
                                      rB: F3(sx * plankLen / 2, yo, 0)))
            }
        }

        // ---- hangers: vertical ropes from cable links down to plank edges
        for (sideIdx, sy) in [Float(-1), 1].enumerated() {
            let links = cableLinks[sideIdx]
            for (i, plank) in plankIdx.enumerated() {
                let px = -span / 2 + (Float(i) + 0.5) * plankLen
                var best = 0
                var bestD = Float.greatestFiniteMagnitude
                for (li, lx) in cableXs.enumerated() where abs(lx - px) < bestD {
                    bestD = abs(lx - px)
                    best = li
                }
                let cl = links[best]
                let top = s.bodies[cl].position
                let bot = F3(px, sy * plankW * 0.46, deckZ + plankT / 2)
                let hgt = top.z - bot.z
                if hgt < 0.3 { continue }
                let n = max(1, Int(hgt / 0.55 + 0.5))
                var hp: [F3] = []
                let topP = F3(top.x, sy * cableY, top.z)
                for k in 0...n {
                    let t = Float(k) / Float(n)
                    hp.append(topP + (bot - topP) * t)
                }
                _ = addRope(hp, radius: 0.045, density: 2,
                            pinA: (cl, F3(0, 0, 0)),
                            pinB: (plank, F3(0, sy * plankW * 0.46, plankT / 2)))
            }
        }

        // cargo dropped on the middle
        var rng = SplitMix64(seed: 7)
        for k in 0..<drops {
            _ = s.addBody(size: F3(repeating: 0.8), density: 1.5, friction: 0.6,
                          position: F3((rng.nextFloat() - 0.5) * span * 0.4,
                                       (rng.nextFloat() - 0.5) * plankW * 0.4,
                                       deckZ + 3 + Float(k) * 1.5))
        }
        return s
    }

    /// Tensegrity sculpture: a floating C-frame that hangs entirely on
    /// chains. The short center chain (in tension, hidden in plain sight)
    /// carries the weight from the static arm's tip; the long outer chains
    /// only prevent tipping. No springs, no tricks — pure chain tension.
    public static func tensegrity(towers: Int = 1) -> PhysicsScene {
        var s = PhysicsScene(name: "tensegrity")
        s.settings.iterations = 20
        s.settings.lambdaMax = 4000
        addGround(&s, friction: 0.8)

        /// chain of capsule links between two anchor points on two bodies
        func addChain(from bodyA: Int, _ rA: F3, to bodyB: Int, _ rB: F3,
                      slack: Float = 1.01) {
            let pA = s.bodies[bodyA].position + s.bodies[bodyA].rotation.act(rA)
            let pB = s.bodies[bodyB].position + s.bodies[bodyB].rotation.act(rB)
            let span = pB - pA
            let dist = length(span)
            // quantize the link count, then stretch links so the chain spans
            // the gap EXACTLY (times slack): taut from frame one, no snap
            let n = max(2, Int((dist * slack) / 0.30 + 0.5))
            let linkLen = dist * slack / Float(n)
            let dir = span / dist
            // orientation: capsule local z along the chain
            let axis = cross(F3(0, 0, 1), dir)
            let q: Quat = length(axis) < 1e-5
                ? Quat(real: 1, imag: .zero)
                : Quat(angle: acos(simd_clamp(dot(F3(0, 0, 1), dir), -1, 1)),
                       axis: normalize(axis))
            var prev = bodyA
            var prevAnchor = rA
            for k in 0..<n {
                let t = (Float(k) + 0.5) / Float(n)
                let link = s.addCapsule(length: linkLen * 0.72, radius: 0.045,
                                        density: 2, friction: 0.3,
                                        position: pA + dir * (dist * slack * t), rotation: q)
                s.addJoint(SceneJoint(bodyA: prev, bodyB: link,
                                      rA: prevAnchor, rB: F3(0, 0, -linkLen / 2)))
                // links of one chain never collide with each other
                s.addJoint(SceneJoint(bodyA: prev, bodyB: link, rA: .zero, rB: .zero,
                                      stiffnessLin: 0, stiffnessAng: 0))
                prev = link
                prevAnchor = F3(0, 0, linkLen / 2)
            }
            s.addJoint(SceneJoint(bodyA: prev, bodyB: bodyB,
                                  rA: prevAnchor, rB: rB))
            s.addJoint(SceneJoint(bodyA: prev, bodyB: bodyB, rA: .zero, rB: .zero,
                                  stiffnessLin: 0, stiffnessAng: 0))
        }

        for tower in 0..<towers {
            let ox = Float(tower) * 6

            // ---- static lower structure: base, post, inward arm ----
            let base = s.addBody(size: F3(3.0, 3.0, 0.22), density: 0, friction: 0.8,
                                 position: F3(ox, 0, 0.11))
            _ = s.addBody(size: F3(0.34, 0.34, 2.7), density: 0, friction: 0.5,
                          position: F3(ox + 0.75, 0, 1.57))
            let armS = s.addBody(size: F3(1.8, 0.34, 0.3), density: 0, friction: 0.5,
                                 position: F3(ox + 0.0, 0, 3.05))
            // static tip hangs the center chain at its -x end
            _ = armS

            // ---- floating upper structure: welded C-frame ----
            // lower hook (under the static tip), arm out, post up, top bar over
            let hookP = F3(ox - 0.75, 0, 2.30)
            // mass distribution is the whole trick: the hanging frame is only
            // stable if its CG sits BELOW the hook (else it's an inverted
            // pendulum and capsizes through the unrestrained direction) and
            // LEFT of it (so gravity's moment keeps the outer chains taut).
            // Heavy bottom arm, feather-light everything above.
            let hook = s.addBody(size: F3(0.32, 0.32, 0.32), density: 15, friction: 0.4,
                                 position: hookP)
            let armF = s.addBody(size: F3(1.2, 0.34, 0.34), density: 15, friction: 0.4,
                                 position: hookP + F3(-0.74, 0, 0))
            let postF = s.addBody(size: F3(0.3, 0.3, 2.3), density: 0.12, friction: 0.4,
                                  position: F3(ox - 1.62, 0, 3.57))
            let barF = s.addBody(size: F3(3.1, 0.3, 0.28), density: 0.12, friction: 0.4,
                                 position: F3(ox - 0.2, 0, 4.84))
            func weld(_ a: Int, _ b: Int, _ world: F3) {
                let ba = s.bodies[a], bb = s.bodies[b]
                s.addJoint(SceneJoint(bodyA: a, bodyB: b,
                                      rA: ba.rotation.inverse.act(world - ba.position),
                                      rB: bb.rotation.inverse.act(world - bb.position),
                                      stiffnessLin: .infinity, stiffnessAng: .infinity))
            }
            // crossbar at the bar's far end (like the original): the two
            // outer chains attach at its tips, wide apart in y — that spread
            // blocks the roll mode about the hook-to-anchor axis
            let crossF = s.addBody(size: F3(0.3, 1.9, 0.26), density: 0.12, friction: 0.4,
                                   position: F3(ox + 1.3, 0, 4.84))
            weld(hook, armF, hookP + F3(-0.18, 0, 0))
            weld(armF, postF, F3(ox - 1.62, 0, 2.3))
            weld(postF, barF, F3(ox - 1.62, 0, 4.78))
            weld(barF, crossF, F3(ox + 1.3, 0, 4.84))

            // ---- chains ----
            // center chain: static arm tip DOWN to the floating hook — this
            // single short chain carries the whole floating frame
            addChain(from: armS, F3(-0.82, 0, -0.15),
                     to: hook, F3(0, 0, 0.16), slack: 1.0)
            // three ropes total: the center chain carries the weight, the two
            // outer chains (crossbar tips -> base corners) carry the tipping
            // moment — gravity's moment about the hook keeps them taut
            for sy in [Float(-1), 1] {
                addChain(from: crossF, F3(0, sy * 0.85, 0),
                         to: base, F3(1.25, sy * 1.25, 0.11), slack: 1.0)
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

extension Demos {
    /// A faithful mechanical car: no engine, no wheel joints. Each wheel is
    /// an inner torus (hub) threaded around an axle stick welded to the
    /// body — retained by an end cap, spinning freely as a real bearing —
    /// with spokes out to an outer torus that is the tire. The car rolls
    /// down a curly banked track on inertia alone.
    /// Assemble the car at `carPos`. Wheels are welded composites (hub torus,
    /// three spokes, tire torus) attached to the chassis by 1-DOF HINGE
    /// joints — they spin freely about their axles. Returns the chassis index.
    @discardableResult
    public static func buildCar(_ s: inout PhysicsScene, carPos: F3, v0: F3,
                                rot: Quat = Quat(real: 1, imag: .zero)) -> Int {
        let wheelR: Float = 0.5         // tire major radius
        let tire: Float = 0.13
        let hubR: Float = 0.22
        let hubTube: Float = 0.09

        func place(_ off: F3) -> F3 { carPos + rot.act(off) }
        func orient(_ q: Quat) -> Quat { (rot * q).normalized }
        let qId = Quat(real: 1, imag: .zero)

        let chassis = s.addBody(size: F3(2.6, 1.1, 0.35), density: 2, friction: 0.3,
                                position: place(.zero), rotation: orient(qId), velocity: v0)

        let qWheel = Quat(angle: .pi / 2, axis: F3(1, 0, 0))   // torus axis -> y

        for sx in [Float(-1), 1] {
            for sy in [Float(-1), 1] {
                let wheelC = place(F3(sx * 0.95, sy * 0.85, 0))

                // decorative stub axle welded to the chassis
                let qAxle = Quat(angle: .pi / 2, axis: F3(1, 0, 0))
                let axle = s.addCapsule(length: 0.62, radius: 0.08,
                                        density: 2, friction: 0.1,
                                        position: place(F3(sx * 0.95, sy * 0.80, 0)),
                                        rotation: orient(qAxle), velocity: v0)
                s.addJoint(SceneJoint(bodyA: chassis, bodyB: axle,
                                      rA: F3(sx * 0.95, sy * 0.49, 0),
                                      rB: F3(0, 0, sy * 0.31),
                                      stiffnessLin: .infinity, stiffnessAng: .infinity))

                // hub torus, HINGED to the chassis (free spin about its axis)
                let hub = s.addTorus(major: hubR, minor: hubTube, density: 1.5,
                                     friction: 0.1, position: wheelC,
                                     rotation: orient(qWheel), velocity: v0)
                s.addJoint(SceneJoint(bodyA: chassis, bodyB: hub,
                                      rA: F3(sx * 0.95, sy * 0.85, 0), rB: .zero,
                                      stiffnessLin: .infinity, stiffnessAng: .infinity,
                                      hingeAxis: F3(0, 0, 1)))   // torus local axis

                // tire + spokes welded into the wheel composite
                let outer = s.addTorus(major: wheelR, minor: tire, density: 1,
                                       friction: 0.5, position: wheelC,
                                       rotation: orient(qWheel), velocity: v0)
                for k in 0..<3 {
                    let a = Float(k) * 2 * .pi / 3
                    let dir = rot.act(F3(cos(a), 0, sin(a)))
                    let spokeLen = wheelR - hubR + 0.1
                    let mid = wheelC + dir * (hubR + wheelR) / 2
                    let spoke = s.addBody(size: F3(spokeLen, 0.06, 0.06),
                                          density: 1, friction: 0.1,
                                          position: mid,
                                          rotation: orient(Quat(angle: -a, axis: F3(0, 1, 0))),
                                          velocity: v0)
                    s.addJoint(SceneJoint(bodyA: hub, bodyB: spoke,
                                          rA: F3(cos(a) * hubR, sin(a) * hubR, 0),
                                          rB: F3(-spokeLen / 2 + 0.05, 0, 0),
                                          stiffnessLin: .infinity, stiffnessAng: .infinity))
                    s.addJoint(SceneJoint(bodyA: spoke, bodyB: outer,
                                          rA: F3(spokeLen / 2 - 0.05, 0, 0),
                                          rB: F3(cos(a) * wheelR, sin(a) * wheelR, 0),
                                          stiffnessLin: .infinity, stiffnessAng: .infinity))
                    // spokes never collide with the chassis/axle
                    let sp = s.bodies.count - 1
                    for other in [chassis, axle] {
                        s.addJoint(SceneJoint(bodyA: other, bodyB: sp,
                                              rA: .zero, rB: .zero,
                                              stiffnessLin: 0, stiffnessAng: 0))
                    }
                }
                // wheel parts never collide with their own axle/chassis
                for part in [hub, outer] {
                    for other in [chassis, axle] {
                        s.addJoint(SceneJoint(bodyA: other, bodyB: part,
                                              rA: .zero, rB: .zero,
                                              stiffnessLin: 0, stiffnessAng: 0))
                    }
                }
                s.addJoint(SceneJoint(bodyA: hub, bodyB: outer, rA: .zero, rB: .zero,
                                      stiffnessLin: 0, stiffnessAng: 0))
            }
        }
        return chassis
    }

    public static func car(trackScale: Int = 1) -> PhysicsScene {
        var s = PhysicsScene(name: "car")
        s.settings.iterations = 20
        s.settings.betaLin = 20000
        s.settings.lambdaMax = 1200     // wall-seam wedges must not stockpile
                                        // force and launch the car
        addGround(&s, friction: 0.6)

        let chassisZ0: Float = 7.2
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
            // straight schuss: platform -> drop -> long runout. Unsteered
            // rigid cars corner and jump chaotically; this stays honest.
            Seg(len: 5, turn: 0, pitch: -0.01, bank: 0),
            Seg(len: 12 * L, turn: 0, pitch: -0.15, bank: 0),
            Seg(len: 30, turn: 0, pitch: -0.012, bank: 0),
        ]

        // three passes: collect per-sample (turn, pitch, bank) targets,
        // SMOOTH pitch and bank (kinked pitch = concave ridges across the
        // lane that eat all the car's energy), then integrate positions
        var targets: [(Float, Float, Float)] = []   // dHeading, pitch, bank
        let step: Float = 0.9
        for seg in segs {
            let n = Int(seg.len / step)
            for _ in 0..<n {
                targets.append((seg.turn / Float(max(1, n)), seg.pitch, seg.bank))
            }
        }
        func smooth(_ v: [Float]) -> [Float] {
            var out = v
            for i in 0..<v.count {
                var acc: Float = 0; var cnt: Float = 0
                for j in max(0, i - 5)...min(v.count - 1, i + 5) {
                    acc += v[j]; cnt += 1
                }
                out[i] = acc / cnt
            }
            return out
        }
        let pitches = smooth(targets.map { $0.1 })
        let banksArr = smooth(targets.map { $0.2 })

        var samples: [(F3, Float, Float, Float)] = []
        var pos = F3(startX - 1.5, 0, chassisZ0 - wheelR - 0.2)
        var heading: Float = 0
        for (i, t) in targets.enumerated() {
            heading += t.0
            // exact 3D unit step along the slat plane: consecutive slats are
            // perfectly coplanar on straights (no micro-ridges to bounce on)
            let cp = cos(pitches[i]), sp = sin(pitches[i])
            let d3 = F3(cos(heading) * cp, sin(heading) * cp, sp)
            pos += d3 * step
            if pos.z < 0.1 { pos.z = 0.1 }
            samples.append((pos, heading, pitches[i], banksArr[i]))
        }
        for (i, sm) in samples.enumerated() {
            let (p, hdg, pitch, _) = sm
            let bank = samples[i].3
            _ = bank
            let qYaw = Quat(angle: hdg, axis: F3(0, 0, 1))
            let right = F3(sin(hdg), -cos(hdg), 0)
            let dir = F3(cos(hdg), sin(hdg), 0)
            let qPitch = Quat(angle: pitch, axis: right)
            let qBank = Quat(angle: samples[i].3, axis: dir)
            let q = (qBank * qPitch * qYaw).normalized
            _ = s.addBody(size: F3(1.35, 3.2, 0.16), density: 0, friction: 0.8,
                          position: p, rotation: q)
            // pinewood-derby center guide rail: the wheels straddle it with
            // little play, keeping the unsteered car laterally captured
            _ = s.addBody(size: F3(1.35, 1.18, 0.26), density: 0, friction: 0.03,
                          position: p + q.act(F3(0, 0, 0.21)), rotation: q)
            for wside in [Float(-1), 1] {
                // tall near-vertical guide walls (an outward tilt becomes a
                // launch ramp at speed), slick so the car grinds smoothly
                let woff = q.act(F3(0, wside * 1.74, 0.62))
                let qWall = (q * Quat(angle: wside * 0.06, axis: F3(1, 0, 0))).normalized
                _ = s.addBody(size: F3(1.7, 0.14, 1.6), density: 0, friction: 0.03,
                              position: p + woff, rotation: qWall)
            }
        }
        // parking pocket at the runout end: a grippy patch + end bumper
        if let last = samples.last {
            let endP = last.0
            _ = s.addBody(size: F3(6, 3.4, 0.16), density: 0, friction: 1.4,
                          position: endP + F3(3.2, 0, 0))
            _ = s.addBody(size: F3(0.3, 3.4, 1.2), density: 0, friction: 0.5,
                          position: endP + F3(6.4, 0, 0.55))
            for wside in [Float(-1), 1] {
                _ = s.addBody(size: F3(6.6, 0.14, 1.2), density: 0, friction: 0.05,
                              position: endP + F3(3.2, wside * 1.74, 0.55))
            }
        }
        return s
    }
}

extension Demos {
    /// A free-spinning gear on a static capsule axle: double torus-hub
    /// bearing (like the car wheel), three spokes, a rim, and N teeth
    /// welded around it. Returns the rim body index.
    @discardableResult
    static func addGear(_ s: inout PhysicsScene, center: F3, R: Float, teeth: Int,
                        phase: Float = 0, handLen: Float = 0) -> Int {
        let qWheel = Quat(angle: .pi / 2, axis: F3(1, 0, 0))   // axis -> y

        // rim on a world HINGE joint (1-DOF revolute about its own axis):
        // simpler and stiffer than the contact bearing for machinery
        let rim = s.addTorus(major: R, minor: 0.1, density: 0.35, friction: 0.3,
                             position: center, rotation: qWheel)
        s.addJoint(SceneJoint(bodyA: -1, bodyB: rim, rA: center, rB: .zero,
                              stiffnessLin: .infinity, stiffnessAng: .infinity,
                              hingeAxis: F3(0, 0, 1)))    // torus local z

        // decorative axle (static, visual)
        _ = s.addCapsule(length: 0.9, radius: 0.09, density: 0, friction: 0.1,
                         position: center, rotation: qWheel)

        // spokes (welded, decorative + carry the hand)
        let hubR: Float = 0.18
        let spokeLen = R - hubR + 0.1
        for k in 0..<3 {
            let a = Float(k) * 2 * .pi / 3 + phase
            let dir = F3(cos(a), 0, sin(a))
            let spoke = s.addBody(size: F3(spokeLen, 0.06, 0.06), density: 0.3,
                                  friction: 0.1,
                                  position: center + dir * (hubR + R) / 2,
                                  rotation: Quat(angle: -a, axis: F3(0, 1, 0)))
            s.addJoint(SceneJoint(bodyA: rim, bodyB: spoke,
                                  rA: F3(cos(a) * R, sin(a) * R, 0),
                                  rB: F3(spokeLen / 2 - 0.05, 0, 0),
                                  stiffnessLin: .infinity, stiffnessAng: .infinity))
        }

        // squared teeth like a real gear: radial boxes welded to the rim.
        // Tooth width = ~40% of the pitch arc so the gaps swallow the
        // mating teeth; hinge-fixed centers keep the mesh aligned.
        let toothLen: Float = 0.34
        let pitchArc = 2 * .pi * R / Float(teeth)
        let toothW = pitchArc * 0.40
        for k in 0..<teeth {
            let a = Float(k) / Float(teeth) * 2 * .pi + phase
            let dir = F3(cos(a), 0, sin(a))
            let tooth = s.addBody(size: F3(toothLen, 0.42, toothW), density: 0.35,
                                  friction: 0.15,
                                  position: center + dir * (R + toothLen / 2),
                                  rotation: Quat(angle: -a, axis: F3(0, 1, 0)))
            s.addJoint(SceneJoint(bodyA: rim, bodyB: tooth,
                                  rA: F3(cos(a) * R, sin(a) * R, 0),
                                  rB: F3(-toothLen / 2, 0, 0),
                                  stiffnessLin: .infinity, stiffnessAng: .infinity))
        }

        // clock hand hard-attached to the rim
        if handLen > 0 {
            let handY: Float = 0.45
            let hand = s.addBody(size: F3(handLen, 0.1, 0.14), density: 0.3,
                                 friction: 0.1,
                                 position: center + F3(handLen / 2, handY, 0))
            s.addJoint(SceneJoint(bodyA: rim, bodyB: hand,
                                  rA: F3(0, 0, -handY), rB: F3(-handLen / 2, 0, 0),
                                  stiffnessLin: .infinity, stiffnessAng: .infinity))
            s.addJoint(SceneJoint(bodyA: rim, bodyB: hand,
                                  rA: F3(handLen * 0.7, 0, -handY),
                                  rB: F3(handLen * 0.2, 0, 0),
                                  stiffnessLin: .infinity, stiffnessAng: .infinity))
        }
        return rim
    }

    /// Gravity-powered gear clock: no motors anywhere. A ball drops onto a
    /// lever; the lever's far end pushes the big gear's teeth; tooth
    /// collisions transfer the motion to a smaller gear that carries a
    /// clock hand. Pure collision + friction mechanics.
    public static func gearclock(scale: Int = 1) -> PhysicsScene {
        var s = PhysicsScene(name: "gearclock")
        s.settings.iterations = 20
        s.settings.betaLin = 20000
        addGround(&s, friction: 0.6)

        // gear train in the xz-plane (axles along y)
        let g1c = F3(0, 0, 3.4)
        let g2c = F3(2.40, 0, 3.4)        // tooth tips overlap ~0.18
        _ = addGear(&s, center: g1c, R: 1.25, teeth: 12)
        _ = addGear(&s, center: g2c, R: 0.65, teeth: 6, phase: 0.524, handLen: 1.5)

        // crank pin on the big gear's face (offset from the teeth plane):
        // the lever lifts this pin instead of poking between teeth
        let pinAngle: Float = 3.32      // rad: pin floats just above the lever
        let pinR: Float = 0.95
        let rims = s.bodies.enumerated().filter { $0.element.shape == .torus && abs($0.element.size.x - 1.25) < 0.01 }
        let rim1 = rims[0].offset
        let pinPos = g1c + F3(pinR * cos(pinAngle), -0.5, pinR * sin(pinAngle))
        let pin = s.addBody(size: F3(0.16, 0.5, 0.16), density: 0.6, friction: 0.01,
                            position: pinPos)
        s.addJoint(SceneJoint(bodyA: rim1, bodyB: pin,
                              rA: F3(pinR * cos(pinAngle), pinR * sin(pinAngle), 0.5),
                              rB: .zero,
                              stiffnessLin: .infinity, stiffnessAng: .infinity))

        // rocker lever on a world hinge in the pin's plane (y = -0.55)
        let leverY: Float = -0.55
        let pivot = F3(-2.6, leverY, 2.6)
        _ = s.addCapsule(length: 1.1, radius: 0.11, density: 0, friction: 0.02,
                         position: pivot, rotation: Quat(angle: .pi / 2, axis: F3(1, 0, 0)))
        let lever = s.addBody(size: F3(4.4, 0.5, 0.15), density: 0.35, friction: 0.4,
                              position: pivot + F3(0, 0, 0.42))
        s.addJoint(SceneJoint(bodyA: -1, bodyB: lever,
                              rA: pivot + F3(0, 0, 0.42), rB: .zero,
                              stiffnessLin: .infinity, stiffnessAng: .infinity,
                              hingeAxis: F3(0, 1, 0)))

        // pan bucket on the left end: low walls on all four sides, spaced
        // wider than the ball so it lands on the pan floor
        let panX: Float = -2.6 - 1.8
        for xo in [Float(-0.72), 0.72] {
            let wall = s.addBody(size: F3(0.12, 0.74, 0.4), density: 0.4, friction: 0.4,
                                 position: F3(panX + xo, leverY, pivot.z + 0.42 + 0.25))
            s.addJoint(SceneJoint(bodyA: lever, bodyB: wall,
                                  rA: F3(panX + xo - pivot.x, 0, 0.175), rB: F3(0, 0, -0.15),
                                  stiffnessLin: .infinity, stiffnessAng: .infinity))
        }
        for yo in [Float(-0.31), 0.31] {
            let wall = s.addBody(size: F3(1.32, 0.12, 0.4), density: 0.4, friction: 0.4,
                                 position: F3(panX, leverY + yo, pivot.z + 0.42 + 0.25))
            s.addJoint(SceneJoint(bodyA: lever, bodyB: wall,
                                  rA: F3(panX - pivot.x, yo, 0.175), rB: F3(0, 0, -0.15),
                                  stiffnessLin: .infinity, stiffnessAng: .infinity))
        }
        // static rest stop under the gear-side lever end (lever level at rest)
        _ = s.addBody(size: F3(0.3, 1.0, 2.3), density: 0, friction: 0.1,
                      position: F3(-0.7, leverY, 1.75))

        // the trigger: a heavy box dropped from high — the lever whips and
        // kicks the crank pin, spinning the gear train up to coast
        _ = s.addBody(size: F3(1.0, 1.0, 1.0), density: 8, friction: 0.8,
                      position: F3(panX, leverY, 7.0))

        _ = scale
        return s
    }
}