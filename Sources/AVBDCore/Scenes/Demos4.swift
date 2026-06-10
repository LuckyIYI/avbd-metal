import simd

// Wrecking-ball demo: an open-frame furnished building (columns + slabs,
// no walls) meets a crane whose ball spawns hoisted AWAY from the building,
// so it drops at t=0 and swings straight into the structure.

extension Demos {

    public static func wreckingball(floors: Int = 3) -> PhysicsScene {
        var s = PhysicsScene(name: "wreckingball")
        s.settings.iterations = 20
        s.settings.betaLin = 20000
        s.settings.lambdaMax = 50000      // let the ball hit HARD, no launches
        addGround(&s, friction: 0.9)

        // ---------- building shell: columns + slabs, fully dynamic ----------
        let W: Float = 14, D: Float = 10          // footprint
        let floorH: Float = 2.7
        let slabT: Float = 0.3
        let colS: Float = 0.55
        let nF = max(2, floors)

        func column(_ x: Float, _ y: Float, _ z0: Float) {
            _ = s.addBody(size: F3(colS, colS, floorH - slabT), density: 1.6,
                          friction: 0.9,
                          position: F3(x, y, z0 + (floorH - slabT) / 2))
        }
        var slabZ: [Float] = []                    // top surface of each slab
        for f in 0..<nF {
            let z0 = Float(f) * floorH             // column base height
            for gx in 0...2 {
                for gy in 0...1 {
                    column(-W / 2 + Float(gx) * W / 2, -D / 2 + Float(gy) * D, z0)
                }
            }
            // mid-edge columns front/back for slab support
            column(-W / 4, -D / 2, z0); column(W / 4, -D / 2, z0)
            column(-W / 4, D / 2, z0); column(W / 4, D / 2, z0)
            let sz = z0 + floorH - slabT / 2
            _ = s.addBody(size: F3(W + 0.8, D + 0.8, slabT), density: 0.55,
                          friction: 0.9, position: F3(0, 0, sz))
            slabZ.append(z0 + floorH)
        }

        // ---------- furniture (welded box composites) ----------
        func weld(_ a: Int, _ b: Int, _ world: F3) {
            let ba = s.bodies[a], bb = s.bodies[b]
            s.addJoint(SceneJoint(bodyA: a, bodyB: b,
                                  rA: ba.rotation.inverse.act(world - ba.position),
                                  rB: bb.rotation.inverse.act(world - bb.position),
                                  stiffnessLin: .infinity, stiffnessAng: .infinity))
            // welded parts overlap by design: exclude their mutual collision
            // or the contacts fight the weld and the piece jitters/jumps
            s.addJoint(SceneJoint(bodyA: a, bodyB: b, rA: .zero, rB: .zero,
                                  stiffnessLin: 0, stiffnessAng: 0))
        }
        func box(_ size: F3, _ at: F3, d: Float = 0.45, yaw: Float = 0) -> Int {
            s.addBody(size: size, density: d, friction: 0.7, position: at,
                      rotation: Quat(angle: yaw, axis: F3(0, 0, 1)))
        }

        /// p = floor position of the piece's center, z = floor surface height
        func sofa(_ p: F3, yaw: Float = 0) {
            let q = Quat(angle: yaw, axis: F3(0, 0, 1))
            let seat = box(F3(2.0, 0.85, 0.4), p + q.act(F3(0, 0, 0.2)), yaw: yaw)
            let back = box(F3(2.0, 0.25, 0.55), p + q.act(F3(0, -0.32, 0.65)), yaw: yaw)
            let armL = box(F3(0.25, 0.85, 0.3), p + q.act(F3(-0.93, 0, 0.55)), yaw: yaw)
            let armR = box(F3(0.25, 0.85, 0.3), p + q.act(F3(0.93, 0, 0.55)), yaw: yaw)
            weld(seat, back, p + q.act(F3(0, -0.32, 0.42)))
            weld(seat, armL, p + q.act(F3(-0.93, 0, 0.41)))
            weld(seat, armR, p + q.act(F3(0.93, 0, 0.41)))
            // arms touch the back but aren't welded to it — exclude those too
            for arm in [armL, armR] {
                s.addJoint(SceneJoint(bodyA: back, bodyB: arm, rA: .zero, rB: .zero,
                                      stiffnessLin: 0, stiffnessAng: 0))
            }
        }
        func table(_ p: F3, top: F3 = F3(1.7, 1.0, 0.08), h: Float = 0.74) {
            let t = box(top, p + F3(0, 0, h - top.z / 2), d: 0.6)
            for sx in [Float(-1), 1] {
                for sy in [Float(-1), 1] {
                    let lp = p + F3(sx * (top.x / 2 - 0.1), sy * (top.y / 2 - 0.1),
                                    (h - top.z) / 2)
                    let leg = box(F3(0.09, 0.09, h - top.z), lp, d: 0.6)
                    weld(t, leg, lp + F3(0, 0, (h - top.z) / 2))
                }
            }
        }
        func chair(_ p: F3, yaw: Float = 0) {
            let q = Quat(angle: yaw, axis: F3(0, 0, 1))
            let seat = box(F3(0.45, 0.45, 0.08), p + q.act(F3(0, 0, 0.45)), yaw: yaw)
            let legs = box(F3(0.38, 0.38, 0.41), p + q.act(F3(0, 0, 0.205)),
                           d: 0.25, yaw: yaw)
            let back = box(F3(0.45, 0.07, 0.5), p + q.act(F3(0, -0.19, 0.74)), yaw: yaw)
            weld(seat, legs, p + q.act(F3(0, 0, 0.41)))
            weld(seat, back, p + q.act(F3(0, -0.19, 0.49)))
        }
        func bed(_ p: F3, yaw: Float = 0) {
            let q = Quat(angle: yaw, axis: F3(0, 0, 1))
            let base = box(F3(2.0, 1.5, 0.35), p + q.act(F3(0, 0, 0.175)), d: 0.5, yaw: yaw)
            let mat = box(F3(1.9, 1.4, 0.2), p + q.act(F3(0, 0, 0.45)), d: 0.2, yaw: yaw)
            let head = box(F3(0.12, 1.5, 0.75), p + q.act(F3(-1.0, 0, 0.375)), d: 0.5, yaw: yaw)
            weld(base, mat, p + q.act(F3(0, 0, 0.35)))
            weld(base, head, p + q.act(F3(-1.0, 0, 0.3)))
            s.addJoint(SceneJoint(bodyA: mat, bodyB: head, rA: .zero, rB: .zero,
                                  stiffnessLin: 0, stiffnessAng: 0))
        }
        func toilet(_ p: F3, yaw: Float = 0) {
            let q = Quat(angle: yaw, axis: F3(0, 0, 1))
            let bowl = box(F3(0.45, 0.55, 0.42), p + q.act(F3(0, 0, 0.21)), d: 0.8, yaw: yaw)
            let tank = box(F3(0.45, 0.18, 0.5), p + q.act(F3(0, -0.32, 0.62)), d: 0.8, yaw: yaw)
            weld(bowl, tank, p + q.act(F3(0, -0.3, 0.42)))
        }
        func fridge(_ p: F3) { _ = box(F3(0.75, 0.75, 1.85), p + F3(0, 0, 0.925), d: 0.8) }
        func counter(_ p: F3, len: Float, yaw: Float = 0) {
            _ = box(F3(len, 0.65, 0.95), p + F3(0, 0, 0.475), d: 0.7, yaw: yaw)
        }
        func shelf(_ p: F3, yaw: Float = 0) {
            _ = box(F3(1.2, 0.35, 1.7), p + F3(0, 0, 0.85), d: 0.5, yaw: yaw)
        }
        func tv(_ p: F3, yaw: Float = 0) {
            let q = Quat(angle: yaw, axis: F3(0, 0, 1))
            let stand = box(F3(1.5, 0.45, 0.5), p + q.act(F3(0, 0, 0.25)), d: 0.6, yaw: yaw)
            let panel = box(F3(1.3, 0.08, 0.75), p + q.act(F3(0, 0, 0.9)), d: 0.3, yaw: yaw)
            weld(stand, panel, p + q.act(F3(0, 0, 0.5)))
        }
        func diningSet(_ p: F3) {
            table(p)
            chair(p + F3(0, 0.85, 0), yaw: .pi)
            chair(p + F3(0, -0.85, 0), yaw: 0)
            chair(p + F3(1.15, 0, 0), yaw: .pi / 2)
            chair(p + F3(-1.15, 0, 0), yaw: -.pi / 2)
        }

        // ---------- rooms per floor (variant per level) ----------
        for f in 0..<nF {
            let z = Float(f) * floorH + (f == 0 ? 0 : 0)
            let zz = f == 0 ? Float(0) : slabZ[f - 1]
            _ = z
            switch f % 3 {
            case 0:   // kitchen + dining + living
                counter(F3(-5.8, -3.5, zz), len: 3.4)
                fridge(F3(-3.6, -4.2, zz))
                counter(F3(-6.3, -1.2, zz), len: 2.0, yaw: .pi / 2)
                diningSet(F3(-1.5, -2.2, zz))
                sofa(F3(3.5, 3.2, zz), yaw: .pi)
                tv(F3(3.5, -0.5, zz))
                shelf(F3(6.2, 2.0, zz), yaw: .pi / 2)
            case 1:   // guest room + bathroom
                bed(F3(-4.5, 2.5, zz))
                shelf(F3(-6.4, -1.0, zz), yaw: .pi / 2)
                sofa(F3(0.5, -3.3, zz))
                table(F3(0.5, -1.2, zz), top: F3(0.9, 0.9, 0.07), h: 0.45)
                toilet(F3(5.8, -3.8, zz), yaw: .pi / 2)
                counter(F3(5.8, -1.8, zz), len: 1.4, yaw: .pi / 2)
                bed(F3(4.5, 2.8, zz), yaw: .pi / 2)
            default:  // office / lounge
                table(F3(-4.5, -3.0, zz), top: F3(1.6, 0.8, 0.07))
                chair(F3(-4.5, -1.9, zz), yaw: .pi)
                shelf(F3(-6.4, 0.5, zz), yaw: .pi / 2)
                shelf(F3(-6.4, 2.5, zz), yaw: .pi / 2)
                sofa(F3(1.0, 2.8, zz), yaw: .pi)
                sofa(F3(-1.5, 0.0, zz), yaw: -.pi / 2)
                tv(F3(3.0, -3.5, zz))
                diningSet(F3(4.5, 1.5, zz))
            }
        }

        // ---------- the wrecking machine ----------
        // static gantry: base, mast, jib reaching over toward the building
        let jibTipX: Float = 8.5
        let jibZ: Float = Float(nF) * floorH + 4.6
        let ballR: Float = 1.9                     // a properly giant ball
        // chain reaches down to mid-height of the building
        let chainLen: Float = jibZ - Float(nF) * floorH / 2 - ballR
        let theta: Float = 1.1                     // hoist angle from vertical
        // mast far enough back that the hoisted ball clears it
        let mastX: Float = jibTipX + sin(theta) * (chainLen + ballR) + ballR + 1.2
        _ = s.addBody(size: F3(5.5, 5.5, 0.8), density: 0, friction: 0.8,
                      position: F3(mastX + 1.0, 0, 0.4))
        _ = s.addBody(size: F3(1.1, 1.1, jibZ), density: 0, friction: 0.5,
                      position: F3(mastX, 0, jibZ / 2))
        _ = s.addBody(size: F3(mastX - jibTipX + 2.4, 0.9, 0.9), density: 0,
                      friction: 0.5,
                      position: F3((mastX + jibTipX) / 2, 0, jibZ + 0.45))
        // counterweight block on the back of the jib (decorative, static)
        _ = s.addBody(size: F3(2.0, 2.2, 1.6), density: 0, friction: 0.5,
                      position: F3(mastX + 2.6, 0, jibZ - 0.5))

        // chain from the jib tip, spawned hoisted AWAY from the building:
        // gravity does the rest
        let tip = F3(jibTipX, 0, jibZ)
        let hang = chainLen + ballR               // pivot to ball center
        let ballC = tip + F3(sin(theta) * hang, 0, -cos(theta) * hang)
        let ball = s.addSphere(diameter: ballR * 2, density: 9, friction: 0.4,
                               position: ballC)

        let dir = normalize(ballC - tip)
        let n = max(3, Int(chainLen / 0.55 + 0.5))
        let linkLen = chainLen / Float(n)
        var prev = -1
        var prevAnchor = tip
        let axis = cross(F3(0, 0, 1), dir)
        let q: Quat = length(axis) < 1e-5
            ? Quat(real: 1, imag: .zero)
            : Quat(angle: acos(simd_clamp(dot(F3(0, 0, 1), dir), -1, 1)),
                   axis: normalize(axis))
        for k in 0..<n {
            let mid = tip + dir * ((Float(k) + 0.5) * linkLen)
            let link = s.addCapsule(length: linkLen * 0.8, radius: 0.07,
                                    density: 6, friction: 0.3,
                                    position: mid, rotation: q)
            if prev < 0 {
                s.addJoint(SceneJoint(bodyA: -1, bodyB: link,
                                      rA: prevAnchor, rB: F3(0, 0, -linkLen / 2)))
            } else {
                s.addJoint(SceneJoint(bodyA: prev, bodyB: link,
                                      rA: prevAnchor, rB: F3(0, 0, -linkLen / 2)))
                s.addJoint(SceneJoint(bodyA: prev, bodyB: link, rA: .zero, rB: .zero,
                                      stiffnessLin: 0, stiffnessAng: 0))
            }
            prev = link
            prevAnchor = F3(0, 0, linkLen / 2)
        }
        s.addJoint(SceneJoint(bodyA: prev, bodyB: ball,
                              rA: prevAnchor, rB: F3(0, 0, ballR)))
        s.addJoint(SceneJoint(bodyA: prev, bodyB: ball, rA: .zero, rB: .zero,
                              stiffnessLin: 0, stiffnessAng: 0))
        return s
    }
}

// MARK: - Trebuchet siege

extension Demos {
    /// Counterweight trebuchet vs a fracture-jointed castle. Gravity-only:
    /// the cocked arm is released at t=0, the hung counterweight drops, the
    /// chain sling whips the boulder around, and a breakable joint releases
    /// it at the top of the whip.
    public static func trebuchet(castleScale: Int = 1) -> PhysicsScene {
        var s = PhysicsScene(name: "trebuchet")
        s.settings.iterations = 20
        s.settings.betaLin = 20000
        s.settings.lambdaMax = 80000
        addGround(&s, friction: 0.9)

        // scale = a BATTERY of trebuchets and a wider, taller castle
        let nT = min(max(castleScale, 1), 5)
        let spread: Float = 7.0
        for ti in 0..<nT {
            let oy = (Float(ti) - Float(nT - 1) / 2) * spread
            buildTrebuchetMachine(&s, oy: oy)
        }
        buildCastle(&s, nT: nT)
        return s
    }

    static func buildTrebuchetMachine(_ s: inout PhysicsScene, oy: Float) {
        // ---------- the machine (at x ~ 0, throws toward +x) ----------
        let pivotZ: Float = 4.2
        let shortArm: Float = 1.5      // counterweight side (+x)
        let longArm: Float = 5.2       // sling side (-x)

        // A-frame: two static legs + crossbeam (the axle housing)
        for sy in [Float(-1), 1] {
            _ = s.addBody(size: F3(0.5, 0.5, pivotZ), density: 0, friction: 0.8,
                          position: F3(0, oy + sy * 1.5, pivotZ / 2))
        }
        let frame = s.addBody(size: F3(0.6, 3.6, 0.6), density: 0, friction: 0.5,
                              position: F3(0, oy, pivotZ))

        // throwing arm, spawned COCKED: long end down toward +x... rotated
        // about the pivot so the counterweight end is UP. Arm local +x = long
        // (sling) direction.
        let cock: Float = 0.98          // radians above horizontal, long end down
        let qArm = Quat(angle: -cock, axis: F3(0, 1, 0))
        let armCenterOff = (longArm - shortArm) / 2   // pivot offset from center
        let armC = F3(0, oy, pivotZ) + qArm.act(F3(-armCenterOff, 0, 0))
        let arm = s.addBody(size: F3(longArm + shortArm, 0.4, 0.35), density: 2.5,
                            friction: 0.5, position: armC, rotation: qArm)
        s.addJoint(SceneJoint(bodyA: frame, bodyB: arm,
                              rA: F3(0, 0, 0), rB: F3(armCenterOff, 0, 0),
                              stiffnessLin: .infinity, stiffnessAng: .infinity,
                              hingeAxis: F3(0, 1, 0)))
        s.addJoint(SceneJoint(bodyA: frame, bodyB: arm, rA: .zero, rB: .zero,
                              stiffnessLin: 0, stiffnessAng: 0))

        // counterweight box HUNG from the short end (free-hanging CW is the
        // efficient classic design)
        let cwAnchor = F3(0, oy, pivotZ) + qArm.act(F3(shortArm, 0, 0))
        let cwC = cwAnchor + F3(0, 0, -1.15)
        let cw = s.addBody(size: F3(1.7, 1.7, 1.5), density: 10, friction: 0.6,
                           position: cwC)
        s.addJoint(SceneJoint(bodyA: arm, bodyB: cw,
                              rA: F3(shortArm + armCenterOff, 0, 0),
                              rB: F3(0, 0, 0.75 + 0.4)))
        s.addJoint(SceneJoint(bodyA: arm, bodyB: cw, rA: .zero, rB: .zero,
                              stiffnessLin: 0, stiffnessAng: 0))

        // spoon cup at the long arm tip carrying the boulder; a static stop
        // beam halts the arm mid-swing and the boulder departs ballistically
        // (onager-style release: purely mechanical, no magic timing)
        let tipW = F3(0, oy, pivotZ) + qArm.act(F3(-longArm, 0, 0))
        let boulderR: Float = 0.55
        func cupPart(_ size: F3, _ worldAt: F3) {
            let part = s.addBody(size: size, density: 0.6, friction: 0.5,
                                 position: worldAt, rotation: qArm)
            s.addJoint(SceneJoint(bodyA: arm, bodyB: part,
                                  rA: qArm.inverse.act(worldAt - armC),
                                  rB: .zero,
                                  stiffnessLin: .infinity, stiffnessAng: .infinity))
            s.addJoint(SceneJoint(bodyA: arm, bodyB: part, rA: .zero, rB: .zero,
                                  stiffnessLin: 0, stiffnessAng: 0))
        }
        // cup local frame = arm frame: bottom plate sits on the arm's
        // swing-leading face, lips trail and flank
        let up = qArm.act(F3(0, 0, 1))               // arm-local z in world
        let out = qArm.act(F3(-1, 0, 0))             // along the long arm
        cupPart(F3(1.5, 1.3, 0.12), tipW + up * 0.24)              // bottom
        cupPart(F3(0.14, 1.3, 0.6), tipW - out * 0.72 + up * 0.55) // back lip
        // low front lip: cages the boulder through the swing; the stop-beam
        // deceleration throws it over
        cupPart(F3(0.14, 1.3, 0.42), tipW + out * 0.72 + up * 0.46)
        cupPart(F3(1.5, 0.12, 0.5), tipW + up * 0.5 + qArm.act(F3(0, 0.66, 0)))
        cupPart(F3(1.5, 0.12, 0.5), tipW + up * 0.5 + qArm.act(F3(0, -0.66, 0)))
        _ = out
        let boulder = s.addSphere(diameter: boulderR * 2, density: 3,
                                  friction: 0.4,
                                  position: tipW + up * (0.3 + boulderR))

        // the stop: a static padded beam the arm slams into
        _ = s.addBody(size: F3(0.7, 4.2, 0.7), density: 0, friction: 0.9,
                      position: F3(-1.9, oy, pivotZ + 2.9))
    }

    /// diagnostic: build one castle structure type in isolation
    static func buildCastlePiece(_ s: inout PhysicsScene, variant: Int) {
        buildCastleImpl(&s, nT: 1, pieceOnly: variant)
    }

    static func buildCastle(_ s: inout PhysicsScene, nT: Int) {
        buildCastleImpl(&s, nT: nT, pieceOnly: -1)
    }

    static nonisolated(unsafe) var unbreakable = false

    static func buildCastleImpl(_ s: inout PhysicsScene, nT: Int, pieceOnly: Int) {
        // A proper medieval castle downrange at +x: tall curtain walls with
        // crenellated battlements, a gatehouse with a real gate opening,
        // round-ish corner towers with merlon crowns, and an inner keep.
        // All weak-mortar fracture brick construction.
        let cx: Float = 26
        let halfW: Float = 5.6 + Float(nT - 1) * 3.5
        let brick = F3(1.1, 0.55, 0.55)

        /// weld with MIDPOINT anchors: zero rest error, so lambda stays
        /// honest (center-to-center anchors carry a permanent C0(1-alpha)
        /// bias that ratchets lambda to the cap on perfectly static walls)
        func mortar(_ u: Int, _ b: Int, fracture: Float) {
            let bu = s.bodies[u], bb = s.bodies[b]
            let mid = (bu.position + bb.position) * 0.5
            s.addJoint(SceneJoint(bodyA: u, bodyB: b,
                                  rA: bu.rotation.inverse.act(mid - bu.position),
                                  rB: bb.rotation.inverse.act(mid - bb.position),
                                  stiffnessLin: .infinity, stiffnessAng: .infinity,
                                  fracture: fracture))
            s.addJoint(SceneJoint(bodyA: u, bodyB: b, rA: .zero, rB: .zero,
                                  stiffnessLin: 0, stiffnessAng: 0))
        }
        let wallH = 10 + (nT - 1)             // properly tall curtain walls

        /// brick wall with optional crenellation and an optional gate gap
        /// (gap given in brick-columns at the wall center, gapH rows tall)
        func wall(_ at: F3, len: Int, height: Int, yaw: Float,
                  crenellated: Bool = true, gateW: Int = 0, gateH: Int = 0) {
            let q = Quat(angle: yaw, axis: F3(0, 0, 1))
            var below: [Int] = []
            for h in 0..<height {
                var row: [Int] = []
                let off: Float = h % 2 == 0 ? 0 : brick.x / 2
                for i in 0..<len {
                    let lx = -Float(len - 1) / 2 * brick.x + Float(i) * brick.x + off
                    // stepped corbel ARCH over the gate: opening narrows row
                    // by row above gateH
                    let gw: Float
                    if gateW > 0 && h < gateH { gw = Float(gateW) }
                    else if gateW > 0 && h == gateH { gw = Float(gateW) - 0.9 }
                    else if gateW > 0 && h == gateH + 1 { gw = Float(gateW) - 1.6 }
                    else { gw = 0 }
                    if gw > 0 && abs(lx) < gw * brick.x / 2 { continue }
                    // arrow slits: skip slim openings on an upper course
                    if height > 6 && h == height - 3 && i % 4 == 1 { continue }
                    // string course: a protruding (deeper) decorative row
                    let deep = height > 6 && h == height - 4
                    let bsize = deep ? F3(brick.x, brick.y * 1.4, brick.z) : brick
                    let p = at + q.act(F3(lx, 0, Float(h) * brick.z + brick.z / 2))
                    let b = s.addBody(size: bsize * 0.93, density: 0.8,
                                      friction: 0.85, position: p, rotation: q)
                    if h > 0 {
                        for u in below where abs(s.bodies[u].position.x - p.x) < brick.x
                            && abs(s.bodies[u].position.y - p.y) < brick.x {
                            mortar(u, b, fracture: unbreakable ? .infinity : 60)
                        }
                    }
                    row.append(b)
                }
                below = row
            }
            // battlements: merlons every other brick on top
            if crenellated {
                for i in stride(from: 0, to: len, by: 2) {
                    let lx = -Float(len - 1) / 2 * brick.x + Float(i) * brick.x
                    let p = at + q.act(F3(lx, 0, Float(height) * brick.z + brick.z * 0.45))
                    let m = s.addBody(size: F3(brick.x * 0.7, brick.y, brick.z * 0.9) ,
                                      density: 0.6, friction: 0.85,
                                      position: p, rotation: q)
                    for u in below where abs(s.bodies[u].position.x - p.x) < brick.x
                        && abs(s.bodies[u].position.y - p.y) < brick.x {
                        mortar(u, m, fracture: 50)
                    }
                }
            }
        }

        /// corner tower: square brick shaft, taller than the walls, merlon
        /// crown on top
        func tower(_ at: F3, height: Int) {
            // flat-stacked 2x2 shaft: full face contact every layer (rotated
            // star plans rest on corner points and topple themselves)
            let tb = F3(0.66, 0.66, 0.5)
            var below: [Int] = []
            for h in 0..<height {
                var row: [Int] = []
                for gx in [Float(-0.34), 0.34] {
                    for gy in [Float(-0.34), 0.34] {
                        let p = at + F3(gx, gy, Float(h) * tb.z + tb.z / 2)
                        let b = s.addBody(size: tb * 0.93, density: 0.85,
                                          friction: 0.9, position: p)
                        if h > 0 {
                            for u in below
                                where distance(s.bodies[u].position, p) < 0.8 {
                                mortar(u, b, fracture: 90)
                            }
                        }
                        row.append(b)
                    }
                }
                below = row
            }
            // merlon crown on the rim
            for (mx, my) in [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)] {
                let p = at + F3(Float(mx), Float(my), Float(height) * 0.5 + 0.21)
                let m = s.addBody(size: F3(0.34, 0.34, 0.42), density: 0.6,
                                  friction: 0.9, position: p)
                for u in below where distance(s.bodies[u].position, p) < 0.9 {
                    mortar(u, m, fracture: 50)
                }
            }
        }

        let frontLen = Int(2 * halfW / 1.1) - 1
        if pieceOnly == 0 {
            wall(F3(cx, 0, 0), len: 10, height: wallH, yaw: .pi / 2)
            return
        }
        if pieceOnly == 3 {
            unbreakable = true
            wall(F3(cx, 0, 0), len: 10, height: wallH, yaw: .pi / 2)
            unbreakable = false
            return
        }
        if pieceOnly == 1 {
            tower(F3(cx, 0, 0), height: wallH + 8)
            return
        }
        if pieceOnly == 2 {
            wall(F3(cx + 7.5, 2.3, 0), len: 4, height: wallH + 2, yaw: .pi / 2)
            wall(F3(cx + 7.5, -2.3, 0), len: 4, height: wallH + 2, yaw: .pi / 2)
            wall(F3(cx + 5.3, 0, 0), len: 3, height: wallH + 2, yaw: 0)
            wall(F3(cx + 9.7, 0, 0), len: 3, height: wallH + 2, yaw: 0)
            return
        }
        // front curtain wall WITH GATEHOUSE: gate 2 columns wide, 4 rows tall
        wall(F3(cx, 0, 0), len: frontLen, height: wallH, yaw: .pi / 2,
             gateW: 2, gateH: 4)
        // gatehouse flanking towers, proud of the wall face
        tower(F3(cx - 2.3, 2.8, 0), height: wallH + 8)
        tower(F3(cx - 2.3, -2.8, 0), height: wallH + 8)
        // side + back curtain walls
        wall(F3(cx + 6, halfW, 0), len: 10, height: wallH - 2, yaw: 0)
        wall(F3(cx + 6, -halfW, 0), len: 10, height: wallH - 2, yaw: 0)
        wall(F3(cx + 12, 0, 0), len: frontLen, height: wallH - 2, yaw: .pi / 2)
        // corner towers, taller than everything, outside the wall corners
        for sy in [Float(-1), 1] {
            tower(F3(cx - 1.5, sy * (halfW + 1.6), 0), height: wallH + 8)
            tower(F3(cx + 13.5, sy * (halfW + 1.6), 0), height: wallH + 8)
        }
        // inner keep: a stout crenellated block in the courtyard
        wall(F3(cx + 7.5, 2.3, 0), len: 4, height: wallH + 2, yaw: .pi / 2,
             crenellated: true)
        wall(F3(cx + 7.5, -2.3, 0), len: 4, height: wallH + 2, yaw: .pi / 2,
             crenellated: true)
        wall(F3(cx + 5.3, 0, 0), len: 3, height: wallH + 2, yaw: 0,
             crenellated: true)
        wall(F3(cx + 9.7, 0, 0), len: 3, height: wallH + 2, yaw: 0,
             crenellated: true)
    }
}

// MARK: - Rube Goldberg machine

extension Demos {
    /// A causal cascade: marble rail -> drop chute -> growing domino run ->
    /// seesaw catapult -> catch basin -> bowling ramp -> tower collapse.
    public static func rubegoldberg(scale: Int = 1) -> PhysicsScene {
        var s = PhysicsScene(name: "rubegoldberg")
        s.settings.iterations = 20
        s.settings.betaLin = 20000
        s.settings.lambdaMax = 3000
        addGround(&s, friction: 0.7)

        func wall(_ size: F3, _ at: F3, yaw: Float = 0, pitch: Float = 0,
                  friction: Float = 0.3) {
            let q = (Quat(angle: yaw, axis: F3(0, 0, 1)) *
                     Quat(angle: pitch, axis: F3(0, 1, 0))).normalized
            _ = s.addBody(size: size, density: 0, friction: friction,
                          position: at, rotation: q)
        }

        // ---- S1: elevated start rail (capsule pair) ----
        let railLen: Float = 9.0
        let drop: Float = 0.30                  // pitch of the start rail
        let z0: Float = 4.4
        let gap: Float = 0.42
        let nSeg = Int(railLen / 0.7)
        for k in 0..<nSeg {
            let t = (Float(k) + 0.5) / Float(nSeg)
            let x = t * railLen
            let z = z0 - sin(drop) * railLen * t
            let q = Quat(angle: drop, axis: F3(0, 1, 0))
            for sy in [Float(-1), 1] {
                _ = s.addBody(size: F3(0.78, 0.06, 0.06) * 1.0, density: 0,
                              friction: 0.15,
                              position: F3(x, sy * gap / 2, z),
                              rotation: q)
            }
        }
        let marble = s.addSphere(diameter: 0.6, density: 1.5, friction: 0.3,
                                 position: F3(0.4, 0, z0 + 0.36),
                                 velocity: F3(1.2, 0, 0))

        // ---- S2: the rail feeds a raised TABLE ----
        let endX = railLen + 0.2
        let tableTop: Float = 1.30
        _ = s.addBody(size: F3(8.6, 2.2, 0.18), density: 0, friction: 0.6,
                      position: F3(endX + 4.1, 0, tableTop - 0.09))
        for sy in [Float(-1), 1] {     // table-top guides
            _ = s.addBody(size: F3(8.6, 0.1, 0.5), density: 0, friction: 0.2,
                          position: F3(endX + 4.1, sy * 1.0, tableTop + 0.25))
        }

        // ---- S3: uniform domino run across the table ----
        // (uniform height: growing chains jam into a leaning pile)
        let dh: Float = 1.05
        var dx = endX + 1.6
        for _ in 0..<6 {
            // 0.78h spacing + low friction: tighter packs jam into a
            // mutually-leaning static wedge instead of falling
            _ = s.addBody(size: F3(dh * 0.2, dh * 0.5, dh), density: 1.0,
                          friction: 0.5,
                          position: F3(dx, 0, tableTop + dh / 2))
            dx += dh * 0.75
        }

        // ---- S4: the last domino's tip lands ON the ball's rear quadrant —
        // the wedge squeeze squirts even a slow fall forward over the pin.
        // Light ball: the table drop re-energizes it, mass is irrelevant.
        let payload = s.addSphere(diameter: 0.8, density: 0.25, friction: 0.4,
                                  position: F3(dx + 0.06, 0, tableTop + 0.4))
        // no stop pin needed: the table is level, so the ball rests until
        // nudged, then creeps to the edge — the drop supplies the energy

        // ---- S5: catch ramp under the edge turns the fall into a strike.
        // Gentle pitch + high catch point: minimal bounce, maximal roll.
        let tableEnd = endX + 4.1 + 4.3
        let rampPitch: Float = 0.32
        let rLen: Float = 1.0 / sin(rampPitch)
        let rampC = F3(tableEnd + 0.3, 0, 1.0)
            + F3(cos(rampPitch), 0, -sin(rampPitch)) * (rLen / 2)
        wall(F3(rLen, 1.7, 0.1), rampC, pitch: rampPitch, friction: 0.4)
        let runoutX = tableEnd + 0.3 + rLen * cos(rampPitch)
        for sy in [Float(-1), 1] {
            wall(F3(6.0, 0.1, 0.6), F3(tableEnd + 2.6, sy * 0.92, 0.3),
                 friction: 0.1)
        }

        // ---- S6: finale: a rank of tall thin monoliths (tipping angle
        // ~6 degrees) felled by the ball. With scale, the rank grows into a
        // long ARC of ever-taller monoliths — a traveling collapse wave that
        // curves around and drops the perched golden ball at the end.
        let nMon = 4 + 4 * min(max(scale, 1) - 1, 4)
        var mp = F3(runoutX + 1.0, 0, 0)
        var heading: Float = 0
        let turnPer: Float = nMon > 4 ? 1.8 / Float(nMon) : 0
        var h: Float = 1.8
        var lastP = mp
        var lastH = h
        for _ in 0..<nMon {
            let q = Quat(angle: heading, axis: F3(0, 0, 1))
            _ = s.addBody(size: F3(0.16, 1.1, h), density: 0.12, friction: 0.5,
                          position: F3(mp.x, mp.y, h / 2), rotation: q)
            lastP = mp
            lastH = h
            let dir = F3(cos(heading), sin(heading), 0)
            mp += dir * (h * 0.62)
            heading += turnPer
            h += 0.12
        }
        _ = s.addSphere(diameter: 0.5, density: 0.3, friction: 0.5,
                        position: F3(lastP.x, lastP.y, lastH + 0.25))

        _ = marble; _ = payload
        return s
    }
}
