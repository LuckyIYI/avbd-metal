import simd

// Giant brick android statue (the classic invader sprite, voxelized from
// REAL dynamic bricks with weak mortar — fully destructible) with a marble
// coaster hugging its silhouette in descending stadium loops.

extension Demos {

    /// Shared statue builder: voxelized invader sprite from dynamic bricks
    /// with weak mortar. Brick size bs scales the whole statue; fracture
    /// thresholds scale with brick mass (~bs^3). Returns the statue top z.
    @discardableResult
    static func buildAndroidStatue(_ s: inout PhysicsScene, bs: Float) -> Float {
        let k3 = (bs / 0.62) * (bs / 0.62) * (bs / 0.62)
        let sprite: [[Int]] = [
            [0,0,1,0,0,0,0,0,1,0,0],
            [0,0,0,1,0,0,0,1,0,0,0],
            [0,0,1,1,1,1,1,1,1,0,0],
            [0,1,1,0,1,1,1,0,1,1,0],
            [1,1,1,1,1,1,1,1,1,1,1],
            [1,0,1,1,1,1,1,1,1,0,1],
            [1,0,1,0,0,0,0,0,1,0,1],
            [0,0,0,1,1,0,1,1,0,0,0],
        ]
        let depth = 2
        let rows = sprite.count
        let cols = sprite[0].count
        let baseZ: Float = 0.9 * (bs / 0.62)
        let plinth = s.addBody(size: F3(Float(cols) * bs + 1.6,
                                        Float(depth) * bs + 1.6, baseZ),
                               density: 0, friction: 0.8,
                               position: F3(0, 0, baseZ / 2))

        func mortar(_ a: Int, _ b: Int, fracture: Float) {
            let ba = s.bodies[a], bb = s.bodies[b]
            let mid = (ba.position + bb.position) * 0.5
            s.addJoint(SceneJoint(bodyA: a, bodyB: b,
                                  rA: mid - ba.position, rB: mid - bb.position,
                                  stiffnessLin: .infinity, stiffnessAng: .infinity,
                                  fracture: fracture))
            s.addJoint(SceneJoint(bodyA: a, bodyB: b, rA: .zero, rB: .zero,
                                  stiffnessLin: 0, stiffnessAng: 0))
        }

        var ids = [[[Int]]](repeating: [[Int]](repeating: [Int](repeating: -1, count: depth),
                                               count: cols), count: rows)
        for (r, row) in sprite.enumerated() {
            for (c, v) in row.enumerated() where v == 1 {
                let x = (Float(c) - Float(cols - 1) / 2) * bs
                let z = baseZ + (Float(rows - 1 - r) + 0.5) * bs
                for d in 0..<depth {
                    let y = (Float(d) - Float(depth - 1) / 2) * bs
                    ids[r][c][d] = s.addBody(size: F3(repeating: bs * 0.93),
                                             density: 0.8, friction: 0.6,
                                             position: F3(x, y, z))
                }
            }
        }
        // weak mortar between face-adjacent bricks; bottom row to the plinth
        for r in 0..<rows {
            for c in 0..<cols {
                for d in 0..<depth {
                    let a = ids[r][c][d]
                    if a < 0 { continue }
                    if r + 1 < rows, ids[r + 1][c][d] >= 0 {
                        mortar(a, ids[r + 1][c][d], fracture: 55 * k3)      // vertical
                    }
                    if c + 1 < cols, ids[r][c + 1][d] >= 0 {
                        mortar(a, ids[r][c + 1][d], fracture: 55 * k3)      // lateral
                    }
                    if d + 1 < depth, ids[r][c][d + 1] >= 0 {
                        mortar(a, ids[r][c][d + 1], fracture: 55 * k3)      // depth
                    }
                    if r == rows - 1 {
                        // anchor the feet to the plinth
                        let ba = s.bodies[a]
                        let anchor = F3(ba.position.x, ba.position.y, baseZ)
                        s.addJoint(SceneJoint(bodyA: plinth, bodyB: a,
                                              rA: anchor - s.bodies[plinth].position,
                                              rB: anchor - ba.position,
                                              stiffnessLin: .infinity,
                                              stiffnessAng: .infinity, fracture: 80 * k3))
                    }
                }
            }
        }
        return baseZ + Float(rows) * bs
    }

    /// Statue-only showcase: the brick android on its plinth, brick size
    /// scaling with the Size picker. Fully destructible (weak mortar) —
    /// throw things at it.
    public static func android(scale: Int = 1) -> PhysicsScene {
        var s = PhysicsScene(name: "android")
        s.settings.iterations = 10
        s.settings.betaLin = 20000
        s.settings.lambdaMax = 6000
        addGround(&s, friction: 0.6)
        let k = Float(max(1, scale)).squareRoot()
        let top = buildAndroidStatue(&s, bs: 0.62 * k)
        s.settings.cameraDistance = top * 2.2 + 4
        s.settings.cameraTargetZ = top * 0.5
        return s
    }

    /// The original marble-coaster scene (kept for the coaster tests):
    /// the statue with a stadium-loop rail hugging its silhouette.
    public static func androidCoaster(scale: Int = 1) -> PhysicsScene {
        var s = PhysicsScene(name: "android")
        s.settings.iterations = 20
        s.settings.betaLin = 20000
        s.settings.lambdaMax = 6000
        addGround(&s, friction: 0.6)

        // ---------------- the statue (built by the shared builder) -------
        let bs: Float = 0.62
        let statueTopShared = buildAndroidStatue(&s, bs: bs)
        let depth = 2
        let cols = 11
        let statueTop = statueTopShared

        // ---------------- the coaster: tight stadium loops ----------------
        var samples: [(F3, F3, Float)] = []
        let step: Float = 0.5
        let loopY: Float = Float(depth - 1) / 2 * bs + 0.62 + 1.35   // rail offset
        let loopX: Float = Float(cols - 1) / 2 * bs + 0.4            // arc centers
        var pos = F3(-loopX, -loopY, statueTop + 1.6)
        var yaw: Float = 0

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

        // stadium laps hugging the statue silhouette
        let laps = 3
        let lapDrop: Float = 1.35
        let lapLen = 2 * (2 * loopX) + 2 * .pi * loopY
        let lapPitch = -atan(lapDrop / lapLen)
        for _ in 0..<laps {
            straight(2 * loopX, lapPitch)
            arc(.pi, loopY, lapPitch)
            straight(2 * loopX, lapPitch)
            arc(.pi, loopY, lapPitch)
        }
        // exit: peel away and run out
        straight(2 * loopX, lapPitch)
        arc(-.pi * 0.4, 3.2, -0.05)
        straight(7.0, -0.045)
        arc(.pi * 0.35, 4.0, -0.03)
        straight(3.0, -0.02)
        let endPos = pos

        // emit rails
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
            if turn != 0 {
                let outer = -turn
                _ = s.addCapsule(length: 0.72, radius: railR, density: 0,
                                 friction: 0.02,
                                 position: p + side * (outer * (gap / 2 + 0.30))
                                     + F3(0, 0, 0.38),
                                 rotation: q)
            }
        }

        // ground-level catch pen with an open mouth facing the track
        let endDir = dirAt(0)
        let penC = F3(endPos.x, endPos.y, 0) + endDir * 3.4
        let wallDefs: [(F3, F3)] = [
            (F3(2.5, 0, 0), F3(0.2, 5.0, 1.1)), (F3(-2.5, 0, 0), F3(0.2, 5.0, 1.1)),
            (F3(0, 2.5, 0), F3(5.0, 0.2, 1.1)), (F3(0, -2.5, 0), F3(5.0, 0.2, 1.1)),
        ]
        for (off, size) in wallDefs {
            if dot(normalize(off), endDir) < -0.5 { continue }
            _ = s.addBody(size: size, density: 0, friction: 0.5,
                          position: penC + off + F3(0, 0, 0.55))
        }

        // balls at the top of the route
        let first = samples[0]
        for k in 0...max(1, scale) {
            _ = s.addSphere(diameter: 0.6, density: 1.4, friction: 0.22,
                            position: first.0 + F3(0, 0, 0.36) - first.1 * (Float(k) * 0.9),
                            velocity: first.1 * 0.8)
        }
        return s
    }
}
