import simd

// Built-in demo scenes (drawn from the paper's evaluation set).

public enum Demos {
    public static var all: [String] {
        ["ground", "stack", "wall", "pyramid", "pendulum", "chain", "boxpile", "spring", "cardhouse", "fracture"]
    }

    public static func make(_ name: String, scale: Int = 1) -> PhysicsScene? {
        switch name {
        case "ground": return ground()
        case "stack": return stack(height: 10 * scale)
        case "wall": return wall(width: 8 * scale, height: 6 * scale)
        case "pyramid": return pyramid(base: 8 * scale)
        case "pendulum": return pendulum(links: 20, massRatio: 100)
        case "chain": return chain(links: 30 * scale)
        case "boxpile": return boxpile(count: 200 * scale * scale)
        case "spring": return springRatio()
        case "cardhouse": return cardhouse(levels: 4)
        case "fracture": return fractureWall()
        default: return nil
        }
    }

    static func addGround(_ s: inout PhysicsScene, friction: Float = 0.7) {
        _ = s.addBody(size: F3(200, 200, 2), density: 0, friction: friction,
                      position: F3(0, 0, -1))
    }

    public static func ground() -> PhysicsScene {
        var s = PhysicsScene(name: "ground")
        addGround(&s)
        _ = s.addBody(size: F3(1, 1, 1), density: 1, friction: 0.5, position: F3(0, 0, 3))
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

    /// Three blocks connected by springs with a 10,000x stiffness ratio (paper Fig. 2).
    public static func springRatio() -> PhysicsScene {
        var s = PhysicsScene(name: "spring")
        let top = s.addBody(size: F3(1, 1, 0.5), density: 0, friction: 0.5, position: F3(0, 0, 10))
        let mid = s.addBody(size: F3(1, 1, 0.5), density: 1, friction: 0.5, position: F3(0, 0, 8))
        let bot = s.addBody(size: F3(1, 1, 0.5), density: 1, friction: 0.5, position: F3(0, 0, 6))
        s.addSpring(SceneSpring(bodyA: top, bodyB: mid, rA: .zero, rB: .zero, stiffness: 1e6))
        s.addSpring(SceneSpring(bodyA: mid, bodyB: bot, rA: .zero, rB: .zero, stiffness: 1e2))
        return s
    }

    /// Lightweight cards held by friction (paper Fig. 6 style).
    public static func cardhouse(levels: Int) -> PhysicsScene {
        var s = PhysicsScene(name: "cardhouse")
        addGround(&s, friction: 0.9)
        let cardW: Float = 1.2, cardT: Float = 0.05
        let lean: Float = 0.35
        for level in 0..<levels {
            let z = Float(level) * (cardW * 0.92 + cardT)
            let n = levels - level
            for i in 0..<n {
                let x = Float(i) * cardW * 1.3 - Float(n) * cardW * 0.65
                let qa = Quat(angle: lean, axis: F3(0, 1, 0))
                let qb = Quat(angle: -lean, axis: F3(0, 1, 0))
                _ = s.addBody(size: F3(cardW, 1.0, cardT), density: 0.2, friction: 0.9,
                              position: F3(x - 0.25, 0, z + cardW * 0.46), rotation: qa)
                _ = s.addBody(size: F3(cardW, 1.0, cardT), density: 0.2, friction: 0.9,
                              position: F3(x + 0.25, 0, z + cardW * 0.46), rotation: qb)
            }
            if level < levels - 1 {
                for i in 0..<(n - 1) {
                    let x = Float(i) * cardW * 1.3 - Float(n) * cardW * 0.65 + cardW * 0.65
                    _ = s.addBody(size: F3(cardW * 1.2, 1.0, cardT), density: 0.2, friction: 0.9,
                                  position: F3(x, 0, z + cardW * 0.92 + cardT / 2))
                }
            }
        }
        return s
    }

    /// Wall of bricks with breakable attachments + a heavy ball (paper Fig. 13).
    public static func fractureWall() -> PhysicsScene {
        var s = PhysicsScene(name: "fracture")
        addGround(&s)
        let bw: Float = 1, bh: Float = 0.5, bd: Float = 0.5
        let width = 10, height = 8
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
        let fractureForce: Float = 800
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
        // heavy ball with momentum
        _ = s.addBody(size: F3(2, 2, 2), density: 4, friction: 0.5,
                      position: F3(0, -15, 3), velocity: F3(0, 30, 0))
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
