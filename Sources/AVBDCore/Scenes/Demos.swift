import simd

// Built-in demo scenes (drawn from the paper's evaluation set).

public enum Demos {
    public static var all: [String] {
        ["ground", "stack", "wall", "pyramid", "pendulum", "chain", "boxpile",
         "spring", "cardhouse", "fracture", "bridge", "tensegrity", "chainmail",
         "swirl", "treadmill", "jenga", "dominoes", "car", "gearclock", "marblerun", "wreckingball", "trebuchet", "rubegoldberg", "cloth", "softbody", "android",
         "clothfold", "boxoncloth", "hammock", "drape", "clothcombo",
         "flagwhip", "ribbons"]
    }

    /// Every demo scales for stress testing: 1 = small (original size),
    /// 2 = medium, 4 = large, 8 = giant. `res` overrides cloth resolution.
    public static func make(_ name: String, scale: Int = 1, res: Int? = nil) -> PhysicsScene? {
        let s = max(1, scale)
        switch name {
        case "ground": return ground(count: s * s)
        case "stack": return stack(height: 10 * s)
        case "wall": return wall(width: 8 * s, height: 6 * s)
        case "pyramid": return pyramid(base: 8 * s)
        case "pendulum": return pendulum(links: 20 * s, massRatio: 100)
        case "chain": return chain(links: 30 * s)
        case "boxpile": return boxpile(count: 200 * s * s)
        case "spring": return springRatio(blocks: 2 + s)
        case "cardhouse": return cardhouse(levels: 3 + s)
        case "fracture": return fractureWall(width: 10 * s, height: 8 * s)
        case "bridge": return bridge(planks: 16 * s, drops: 4 * s)
        case "tensegrity": return tensegrity(towers: s)
        case "chainmail": return chainmail(rings: 5 + s, drops: 3 * s)
        case "swirl": return swirl(turns: 2 + s, balls: 40 * s)
        case "treadmill": return treadmill(boxes: 12 * s)
        case "jenga": return jenga(levels: 18 * s)
        case "dominoes": return dominoes(count: 80 * s)
        case "car": return car(trackScale: s)
        case "gearclock": return gearclock(scale: s)
        case "rubegoldberg": return rubegoldberg(scale: s)
        case "android": return android(scale: s)
        case "cloth": return cloth(res: 12 + 4 * s)
        case "softbody": return softbody(count: 2 + s)
        case "clothfold": return clothfold(res: res ?? (18 + 6 * s))
        case "boxoncloth": return boxoncloth(res: res ?? (18 + 6 * s))
        case "hammock": return hammock(res: res ?? (16 + 4 * s), cubes: s)
        case "drape": return drape(res: res ?? (22 + 6 * s))
        case "clothcombo": return clothcombo(res: res ?? (16 + 4 * s))
        case "flagwhip": return flagwhip(res: res ?? (14 + 2 * s))
        case "ribbons": return ribbons(len: res ?? (18 + 4 * s), count: 4 + 2 * s)
        case "trebuchet": return trebuchet(castleScale: s)
        case "wreckingball": return wreckingball(floors: 2 + s)
        case "marblerun": return marblerun(marbles: 10 * s)
        default: return nil
        }
    }

    static func addGround(_ s: inout PhysicsScene, friction: Float = 0.7) {
        _ = s.addBody(size: F3(200, 200, 2), density: 0, friction: friction,
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
        var prev = -1
        for i in 0..<links {
            let x = Float(i + 1) * linkLen
            let idx = s.addBody(size: F3(linkLen, 0.1, 0.1), density: 1, friction: 0.5,
                                position: F3(x - linkLen / 2, 0, 10))
            s.addJoint(SceneJoint(bodyA: prev, bodyB: idx,
                                  rA: prev < 0 ? F3(0, 0, 10) : F3(linkLen / 2, 0, 0),
                                  rB: F3(-linkLen / 2, 0, 0)))
            prev = idx
        }
        // heavy bob
        let bobSize: Float = 1.0
        let density = massRatio * (linkLen * 0.1 * 0.1) / (bobSize * bobSize * bobSize)
        let x = Float(links) * linkLen + bobSize / 2
        let bob = s.addBody(size: F3(repeating: bobSize), density: density, friction: 0.5,
                            position: F3(x, 0, 10))
        s.addJoint(SceneJoint(bodyA: prev, bodyB: bob,
                              rA: F3(linkLen / 2, 0, 0), rB: F3(-bobSize / 2, 0, 0)))
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
