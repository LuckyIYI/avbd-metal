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
