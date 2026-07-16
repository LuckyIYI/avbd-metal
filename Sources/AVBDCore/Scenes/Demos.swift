import simd

// Built-in demo scenes (drawn from the paper's evaluation set).

/// A user-tunable scene parameter (exposed as a slider in the app's
/// expandable "Demo parameters" section and applied at scene build).
public struct DemoParam: Identifiable {
    public var id: String { key }
    public let key: String
    public let label: String
    public let range: ClosedRange<Float>
    public let def: Float
    public init(_ key: String, _ label: String, _ range: ClosedRange<Float>, _ def: Float) {
        self.key = key; self.label = label; self.range = range; self.def = def
    }
}

public enum Demos {
    public static var all: [String] {
        ["stack", "ratiostack", "wall", "pyramid", "pendulum", "boxpile",
         "cardhouse", "fracture", "bridge", "tensegrity", "chainmail",
         "treadmill", "jenga", "dominoes", "car", "marblerun",
         "wreckingball", "trebuchet", "rubegoldberg",
         "slopefriction", "softbody", "skinnedbunny", "meshclothdrop",
         "softwheel", "android",
         "clothfold", "boxoncloth", "hammock", "drape", "multidrape",
         "clothcombo", "flagwhip", "ribbons", "twist", "bed"]
    }

    /// Tunable parameters per demo (empty = none). Keys are looked up in
    /// `make(_:scale:res:params:)`.
    public static func tunables(_ name: String) -> [DemoParam] {
        let fric = { (d: Float) in DemoParam("friction", "Friction", 0...1, d) }
        // cloth material: StVK membrane stretch stiffness + quadratic
        // bending stiffness (the actual element parameters)
        let membrane = DemoParam("membrane", "Membrane µ", 30...1500, 300)
        let bending = DemoParam("bending", "Bending ×1e-4", 0...40, 5)
        switch name {
        case "clothfold": return [fric(0.9), membrane, bending]
        case "boxoncloth": return [fric(0.8),
                                   DemoParam("massRatio", "Box mass ×cloth", 1...16, 8),
                                   membrane, bending]
        case "hammock": return [fric(0.8), membrane, bending]
        case "drape": return [fric(0.6), membrane, bending]
        case "multidrape": return [fric(0.5), membrane, bending]
        case "clothcombo": return [fric(0.8), membrane, bending,
                                   DemoParam("blockMu", "Block µ", 500...12000, 4000)]
        case "flagwhip": return [DemoParam("kick", "Kick speed", 0...6, 3),
                                 membrane, bending]
        case "ribbons": return [fric(0.35), membrane, bending]
        case "twist": return [DemoParam("turnRate", "Turns / s", 0...1, 0.3),
                              membrane, bending]
        case "softbody": return [fric(0.8),
                                 DemoParam("stiffness", "Stiffness µ", 500...8000, 2500)]
        case "skinnedbunny": return [fric(0.8),
                                     DemoParam("stiffness", "Stiffness µ", 500...8000, 2500)]
        case "meshclothdrop": return [fric(0.8),
                                      DemoParam("stiffness", "Soft µ", 500...8000, 2500),
                                      membrane, bending]
        case "bed": return [fric(0.7),
                            DemoParam("mattressMu", "Mattress µ", 600...8000, 2500),
                            DemoParam("pillowMu", "Pillow µ", 100...2500, 600),
                            membrane, bending]
        case "softwheel": return [DemoParam("drive", "Drive speed", 0...12, 6),
                                  DemoParam("tireMu", "Tire µ", 2000...30000, 9000)]
        default: return []
        }
    }

    /// Every demo scales for stress testing: 1 = small (original size),
    /// 2 = medium, 4 = large, 8 = giant. `res` overrides cloth resolution.
    /// `params` carries the per-demo tunables (missing key or a value of
    /// -1 = scale-derived default).
    public static func make(_ name: String, scale: Int = 1, res: Int? = nil,
                            params: [String: Float] = [:]) -> PhysicsScene? {
        let s = max(1, scale)
        func p(_ key: String, _ def: Float) -> Float {
            let v = params[key] ?? -1
            return v < 0 ? def : v
        }
        func skinnedCount(_ scale: Int) -> Int {
            switch scale {
            case 1: return 1
            case 2: return 4
            case 4: return 10
            case 8: return 22
            default: return 36
            }
        }
        switch name {
        case "stack": return stack(height: 10 * s)
        case "ratiostack": return ratiostack(levels: min(8, 3 + s))
        case "slopefriction": return slopefriction(count: 6 + 2 * s)
        case "softwheel": return softwheel(res: res ?? (9 + s),
                                           drive: p("drive", 6),
                                           tireMu: p("tireMu", 9000))
        case "bed": return bed(res: res ?? min(66, 30 + 6 * s),
                               friction: p("friction", 0.7),
                               mattressMu: p("mattressMu", 2500),
                               pillowMu: p("pillowMu", 600),
                               membraneMu: p("membrane", 300),
                               bend: p("bending", 5) * 1e-4)
        case "wall": return wall(width: 8 * s, height: 6 * s)
        case "pyramid": return pyramid(base: 8 * s)
        case "pendulum": return pendulum(links: 20 * s, massRatio: 100)
        case "boxpile": return boxpile(count: 200 * s * s)
        case "cardhouse": return cardhouse(levels: 3 + s)
        case "fracture": return fractureWall(width: 10 * s, height: 8 * s)
        case "bridge": return bridge(planks: 16 * s, drops: 4 * s)
        case "tensegrity": return tensegrity(towers: s)
        case "chainmail": return chainmail(rings: 5 + s, drops: 3 * s)
        case "treadmill": return treadmill(boxes: 12 * s)
        case "jenga": return jenga(levels: 18 * s)
        case "dominoes": return dominoes(count: 80 * s)
        case "car": return car(trackScale: s)
        case "rubegoldberg": return rubegoldberg(scale: s)
        case "android": return android(scale: s)
        case "softbody": return softbody(res: res ?? min(22, 9 + 2 * s),
                                         stiffness: p("stiffness", 2500),
                                         friction: p("friction", 0.8))
        case "skinnedbunny": return skinnedbunny(res: res ?? min(7, 5 + s / 4),
                                                 count: skinnedCount(s),
                                                 stiffness: p("stiffness", 2500),
                                                 friction: p("friction", 0.8))
        case "meshclothdrop": return meshclothdrop(res: res ?? min(36, 18 + 2 * s),
                                                   scale: s,
                                                   stiffness: p("stiffness", 2500),
                                                   friction: p("friction", 0.8),
                                                   membraneMu: p("membrane", 300),
                                                   bend: p("bending", 5) * 1e-4)
        case "clothfold": return clothfold(res: res ?? (36 + 12 * s),
                                           friction: p("friction", 0.9),
                                           membraneMu: p("membrane", 300),
                                           bend: p("bending", 5) * 1e-4)
        case "boxoncloth": return boxoncloth(res: res ?? (18 + 6 * s),
                                             massRatio: p("massRatio", 8),
                                             friction: p("friction", 0.8),
                                             membraneMu: p("membrane", 300),
                                             bend: p("bending", 5) * 1e-4)
        case "hammock": return hammock(res: res ?? (16 + 4 * s),
                                       cubes: Int(p("cubes", Float(s))),
                                       friction: p("friction", 0.8),
                                       membraneMu: p("membrane", 300),
                                       bend: p("bending", 5) * 1e-4)
        case "drape": return drape(res: res ?? (22 + 6 * s),
                                   friction: p("friction", 0.6),
                                   membraneMu: p("membrane", 300),
                                   bend: p("bending", 5) * 1e-4)
        case "multidrape": return multidrape(res: res ?? (20 + 5 * s),
                                             friction: p("friction", 0.5),
                                             membraneMu: p("membrane", 300),
                                             bend: p("bending", 5) * 1e-4)
        case "clothcombo": return clothcombo(res: res ?? min(48, 16 + 4 * s),
                                             friction: p("friction", 0.8),
                                             membraneMu: p("membrane", 300),
                                             bend: p("bending", 5) * 1e-4,
                                             blockMu: p("blockMu", 4000))
        case "flagwhip": return flagwhip(res: res ?? (14 + 2 * s),
                                         kick: p("kick", 3),
                                         membraneMu: p("membrane", 300),
                                         bend: p("bending", 5) * 1e-4)
        case "twist": return twist(res: res ?? (32 + 8 * s),
                                   turnRate: p("turnRate", 0.3),
                                   membraneMu: p("membrane", 300),
                                   bend: p("bending", 5) * 1e-4)
        case "ribbons": return ribbons(len: res ?? (18 + 4 * s),
                                       count: Int(p("count", Float(4 + 2 * s))),
                                       friction: p("friction", 0.35),
                                       membraneMu: p("membrane", 300),
                                       bend: p("bending", 5) * 1e-4)
        case "trebuchet": return trebuchet(castleScale: s)
        case "wreckingball": return wreckingball(floors: 2 + s)
        case "marblerun": return marblerun(marbles: 10 * s)
        default: return nil
        }
    }

    @discardableResult
    static func addGround(_ s: inout PhysicsScene,
                          friction: Float = 0.7) -> Int {
        s.addBody(size: F3(200, 200, 2), density: 0, friction: friction,
                  position: F3(0, 0, -1))
    }

    public static func ground(count: Int = 1) -> PhysicsScene {
        var s = PhysicsScene(name: "ground")
        addGround(&s)
        let side = Int(Double(count).squareRoot().rounded(.up))
        var placed = 0
        for j in 0..<side {
            for i in 0..<side where placed < count {
                _ = s.addBody(size: F3(1, 1, 1), density: 1, friction: 0.5,
                              position: F3(Float(i - side / 2) * 1.5,
                                           Float(j - side / 2) * 1.5, 3))
                placed += 1
            }
        }
        return s
    }

    public static func stack(height: Int) -> PhysicsScene {
        var s = PhysicsScene(name: "stack")
        addGround(&s)
        for i in 0..<height {
            _ = s.addBody(size: F3(1, 1, 1), density: 1, friction: 0.7,
                          position: F3(0, 0, 0.5 + Float(i)))
        }
        return s
    }

    public static func wall(width: Int, height: Int) -> PhysicsScene {
        var s = PhysicsScene(name: "wall")
        addGround(&s)
        let bw: Float = 2, bh: Float = 1, bd: Float = 1
        for j in 0..<height {
            let offset: Float = (j % 2 == 0) ? 0 : bw / 2
            for i in 0..<width {
                _ = s.addBody(size: F3(bw * 0.98, bd, bh * 0.98), density: 1, friction: 0.7,
                              position: F3(Float(i) * bw + offset - Float(width) * bw / 2,
                                           0, bh / 2 + Float(j) * bh))
            }
        }
        return s
    }

    /// Static-friction correctness, proportioned after avbd-demo3d's
    /// sceneStaticFriction: a LONG 30° ramp (mu = 1) with boxes starting
    /// high up-slope, friction band straddling the stick threshold. PAIR
    /// friction (geometric mean = sqrt(mu_box)) sweeps 0.30..0.78 around
    /// tan(30°) ≈ 0.577 — the slick boxes run the whole ramp onto the
    /// floor, mid ones decelerate to graded stops, the grippy ones hold.
    public static func slopefriction(count: Int = 8) -> PhysicsScene {
        var s = PhysicsScene(name: "slopefriction")
        s.settings.iterations = 10
        addGround(&s, friction: 0.8)
        let tilt: Float = .pi / 6                       // 30 degrees
        let rot = Quat(angle: tilt, axis: F3(0, 1, 0))
        let slopeT: Float = 0.4
        let slopeLen: Float = 32
        let centerZ = slopeLen / 2 * sin(tilt) + 0.4    // bottom edge near floor
        _ = s.addBody(size: F3(slopeLen, Float(count) * 1.3 + 1.5, slopeT),
                      density: 0, friction: 1.0,
                      position: F3(0, 0, centerZ), rotation: rot)
        let n = max(2, count)
        let boxSide: Float = 0.8
        let lift = rot.act(F3(0, 0, slopeT / 2 + boxSide / 2 + 0.01))
        for k in 0..<n {
            let pair = 0.30 + 0.48 * Float(k) / Float(n - 1)
            let mu = pair * pair
            let y = (Float(k) - Float(n - 1) / 2) * 1.3
            _ = s.addBody(size: F3(repeating: boxSide), density: 1,
                          friction: mu,
                          position: F3(0, 0, centerZ)
                                    + rot.act(F3(-slopeLen * 0.32, y, 0))
                                    + lift,
                          rotation: rot)
        }
        s.settings.cameraDistance = 38
        s.settings.cameraTargetZ = 5.5
        return s
    }

    /// Mass-ratio stack: each box LARGER than the one beneath — the heavy
    /// top must neither squeeze the light base out nor sink through it.
    public static func ratiostack(levels: Int = 5, growth: Float = 1.45) -> PhysicsScene {
        var s = PhysicsScene(name: "ratiostack")
        // each level is ~3x the mass of the one below; at 8 levels the
        // bottom contact carries ~2000x the top — needs converged contacts
        s.settings.iterations = 25
        s.settings.betaLin = 40000
        addGround(&s, friction: 0.8)
        var side: Float = 0.5
        var z: Float = 0
        for _ in 0..<max(2, levels) {
            z += side / 2
            _ = s.addBody(size: F3(repeating: side * 0.98), density: 1,
                          friction: 0.8, position: F3(0, 0, z))
            z += side / 2
            side *= growth
        }
        s.settings.cameraDistance = max(10, z * 2.2)
        s.settings.cameraTargetZ = z * 0.45
        return s
    }

    public static func pyramid(base: Int) -> PhysicsScene {
        var s = PhysicsScene(name: "pyramid")
        addGround(&s)
        let b: Float = 1.0
        for j in 0..<base {
            for i in 0..<(base - j) {
                _ = s.addBody(size: F3(b * 0.95, b * 0.95, b * 0.95), density: 1, friction: 0.8,
                              position: F3(Float(i) * b + Float(j) * b / 2 - Float(base) * b / 2,
                                           0, b / 2 + Float(j) * b))
            }
        }
        return s
    }

    /// Chain of links with a heavy bob — the paper's high-mass-ratio test (Fig. 7).
    public static func pendulum(links: Int, massRatio: Float) -> PhysicsScene {
        var s = PhysicsScene(name: "pendulum")
        let linkLen: Float = 0.5
        // anchor high enough that the fully extended rope + bob clears the
        // floor at the bottom of the swing
        let anchorZ = Float(links) * linkLen + 2.0
        var prev = -1
        for i in 0..<links {
            let x = Float(i + 1) * linkLen
            let idx = s.addBody(size: F3(linkLen, 0.1, 0.1), density: 1, friction: 0.5,
                                position: F3(x - linkLen / 2, 0, anchorZ))
            s.addJoint(SceneJoint(bodyA: prev, bodyB: idx,
                                  rA: prev < 0 ? F3(0, 0, anchorZ) : F3(linkLen / 2, 0, 0),
                                  rB: F3(-linkLen / 2, 0, 0)))
            prev = idx
        }
        // heavy bob
        let bobSize: Float = 1.0
        let density = massRatio * (linkLen * 0.1 * 0.1) / (bobSize * bobSize * bobSize)
        let x = Float(links) * linkLen + bobSize / 2
        let bob = s.addBody(size: F3(repeating: bobSize), density: density, friction: 0.5,
                            position: F3(x, 0, anchorZ))
        s.addJoint(SceneJoint(bodyA: prev, bodyB: bob,
                              rA: F3(linkLen / 2, 0, 0), rB: F3(-bobSize / 2, 0, 0)))
        s.settings.cameraDistance = Float(links) * linkLen * 2.4 + 4
        s.settings.cameraTargetZ = anchorZ * 0.55
        return s
    }

    public static func chain(links: Int) -> PhysicsScene {
        var s = PhysicsScene(name: "chain")
        addGround(&s)
        let linkLen: Float = 0.6
        var prev = -1
        for i in 0..<links {
            let z = 20 - Float(i) * linkLen
            let idx = s.addBody(size: F3(0.15, 0.15, linkLen), density: 1, friction: 0.5,
                                position: F3(0, 0, z - linkLen / 2))
            s.addJoint(SceneJoint(bodyA: prev, bodyB: idx,
                                  rA: prev < 0 ? F3(0, 0, 20) : F3(0, 0, -linkLen / 2),
                                  rB: F3(0, 0, linkLen / 2)))
            prev = idx
        }
        return s
    }

    public static func boxpile(count: Int) -> PhysicsScene {
        var s = PhysicsScene(name: "boxpile")
        addGround(&s)
        var rng = SplitMix64(seed: 42)
        let perLayer = max(1, Int(Double(count).squareRoot() * 1.5))
        var placed = 0
        var layer = 0
        while placed < count {
            for _ in 0..<perLayer {
                if placed >= count { break }
                let x = (rng.nextFloat() - 0.5) * 12
                let y = (rng.nextFloat() - 0.5) * 12
                let z = 1.0 + Float(layer) * 1.2 + rng.nextFloat() * 0.3
                let sz = 0.4 + rng.nextFloat() * 0.6
                _ = s.addBody(size: F3(repeating: sz), density: 1, friction: 0.6,
                              position: F3(x, y, z))
                placed += 1
            }
            layer += 1
        }
        return s
    }

    /// Chain of blocks connected by springs of alternating stiffness with a
    /// 10,000x ratio (paper Fig. 2/4). More blocks = harder test.
    public static func springRatio(blocks: Int = 3) -> PhysicsScene {
        var s = PhysicsScene(name: "spring")
        let zTop = Float(4 + 2 * blocks)
        var prev = s.addBody(size: F3(1, 1, 0.5), density: 0, friction: 0.5,
                             position: F3(0, 0, zTop))
        for i in 1..<blocks {
            let b = s.addBody(size: F3(1, 1, 0.5), density: 1, friction: 0.5,
                              position: F3(0, 0, zTop - Float(i) * 2))
            let stiff: Float = i % 2 == 1 ? 1e6 : 1e2
            s.addSpring(SceneSpring(bodyA: prev, bodyB: b, rA: .zero, rB: .zero,
                                    stiffness: stiff))
            prev = b
        }
        return s
    }

    /// Lightweight cards held by friction (paper Fig. 6 style).
    /// Each level: Λ-pairs of cards leaning against each other (~20° off
    /// vertical), bridged by flat separator cards that carry the next level.
    public static func cardhouse(levels: Int) -> PhysicsScene {
        var s = PhysicsScene(name: "cardhouse")
        addGround(&s, friction: 0.9)
        let cardW: Float = 1.2, cardT: Float = 0.05, depth: Float = 1.0
        let theta: Float = 1.22                     // ~70° from the ground
        let footSpan = cardW * cos(theta)           // horizontal span per card
        let H = cardW * sin(theta)                  // apex height
        let pitch = 2 * footSpan + 0.15             // distance between apexes
        let yAxis = F3(0, 1, 0)

        for level in 0..<levels {
            let n = levels - level
            let z0 = Float(level) * (H + cardT)
            let x0 = -Float(n - 1) * pitch / 2      // first apex x
            for i in 0..<n {
                let cx = x0 + Float(i) * pitch
                // left card: foot at cx-footSpan, apex at cx (local +x runs up-slope)
                _ = s.addBody(size: F3(cardW, depth, cardT), density: 0.2, friction: 0.9,
                              position: F3(cx - footSpan / 2, 0, z0 + H / 2),
                              rotation: Quat(angle: -theta, axis: yAxis))
                // right card: foot at cx+footSpan, apex at cx
                _ = s.addBody(size: F3(cardW, depth, cardT), density: 0.2, friction: 0.9,
                              position: F3(cx + footSpan / 2, 0, z0 + H / 2),
                              rotation: Quat(angle: theta, axis: yAxis))
            }
            // flat separators spanning adjacent apexes
            if level < levels - 1 {
                for i in 0..<(n - 1) {
                    let cx = x0 + (Float(i) + 0.5) * pitch
                    _ = s.addBody(size: F3(pitch * 1.05, depth, cardT), density: 0.2,
                                  friction: 0.9,
                                  position: F3(cx, 0, z0 + H + cardT / 2))
                }
            }
        }
        return s
    }

    /// Wall of bricks with breakable attachments + a heavy ball (paper Fig. 13).
    public static func fractureWall(width: Int = 10, height: Int = 8) -> PhysicsScene {
        var s = PhysicsScene(name: "fracture")
        addGround(&s)
        let bw: Float = 1, bh: Float = 0.5, bd: Float = 0.5
        var grid: [[Int]] = []
        for j in 0..<height {
            var row: [Int] = []
            for i in 0..<width {
                let idx = s.addBody(size: F3(bw, bd, bh), density: 1, friction: 0.6,
                                    position: F3(Float(i) * bw - Float(width) * bw / 2,
                                                 0, bh / 2 + Float(j) * bh))
                row.append(idx)
            }
            grid.append(row)
        }
        // breakable joints between neighbors
        // Settle-phase joint torques are ~0.003; ball impact peaks at ~500.
        let fractureForce: Float = 100
        for j in 0..<height {
            for i in 0..<width {
                if i + 1 < width {
                    s.addJoint(SceneJoint(bodyA: grid[j][i], bodyB: grid[j][i + 1],
                                          rA: F3(bw / 2, 0, 0), rB: F3(-bw / 2, 0, 0),
                                          stiffnessLin: .infinity, stiffnessAng: .infinity,
                                          fracture: fractureForce))
                }
                if j + 1 < height {
                    s.addJoint(SceneJoint(bodyA: grid[j][i], bodyB: grid[j + 1][i],
                                          rA: F3(0, 0, bh / 2), rB: F3(0, 0, -bh / 2),
                                          stiffnessLin: .infinity, stiffnessAng: .infinity,
                                          fracture: fractureForce))
                }
            }
        }
        // heavy ball with momentum, sized with the wall
        let ballSize = 2 * Float(width) / 10
        _ = s.addBody(size: F3(repeating: ballSize), density: 4, friction: 0.5,
                      position: F3(0, -15, Float(height) * bh * 0.4),
                      velocity: F3(0, 30, 0))
        return s
    }
}

/// Deterministic RNG for reproducible scenes.
public struct SplitMix64 {
    var state: UInt64
    public init(seed: UInt64) { state = seed }
    public mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
    public mutating func nextFloat() -> Float {
        Float(next() >> 40) / Float(1 << 24)
    }
}
