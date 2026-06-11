import simd

// Cloth gate scenes: surface-element collision (V-T, rigid-T, E-E),
// inextensibility, and combined soft/rigid coupling. These back the
// ClothTests gates and double as app demos.

extension Demos {

    /// Low-level cloth builder over explicit node positions. Structural
    /// edges are stiff springs, shear/bend the usual soft set, and every
    /// quad contributes two collision triangles (alternating diagonal).
    @discardableResult
    static func addClothGrid(_ s: inout PhysicsScene, positions: [[F3]],
                             thickness: Float, massPerNode: Float,
                             friction: Float = 0.7,
                             structuralK: Float = 5000,
                             shearK: Float = 60, bendK: Float = 8) -> [[Int]] {
        let nu = positions.count, nv = positions[0].count
        var grid: [[Int]] = []
        for i in 0..<nu {
            var row: [Int] = []
            for j in 0..<nv {
                row.append(s.addParticle(radius: thickness, mass: massPerNode,
                                         friction: friction, position: positions[i][j]))
            }
            grid.append(row)
        }
        func link(_ a: Int, _ b: Int, _ k: Float) {
            s.addSpring(SceneSpring(bodyA: a, bodyB: b, rA: .zero, rB: .zero,
                                    stiffness: k))
            s.addJoint(SceneJoint(bodyA: a, bodyB: b, rA: .zero, rB: .zero,
                                  stiffnessLin: 0, stiffnessAng: 0))
        }
        for i in 0..<nu {
            for j in 0..<nv {
                if i + 1 < nu { link(grid[i][j], grid[i + 1][j], structuralK) }
                if j + 1 < nv { link(grid[i][j], grid[i][j + 1], structuralK) }
                if i + 1 < nu && j + 1 < nv {
                    link(grid[i][j], grid[i + 1][j + 1], shearK)
                    link(grid[i + 1][j], grid[i][j + 1], shearK)
                }
                if i + 2 < nu { link(grid[i][j], grid[i + 2][j], bendK) }
                if j + 2 < nv { link(grid[i][j], grid[i][j + 2], bendK) }
            }
        }
        for i in 0..<(nu - 1) {
            for j in 0..<(nv - 1) {
                let v00 = grid[i][j], v10 = grid[i + 1][j]
                let v01 = grid[i][j + 1], v11 = grid[i + 1][j + 1]
                if (i + j) % 2 == 0 {
                    s.addTri(SceneTri(ids: (v00, v10, v11)))
                    s.addTri(SceneTri(ids: (v00, v11, v01)))
                } else {
                    s.addTri(SceneTri(ids: (v00, v10, v01)))
                    s.addTri(SceneTri(ids: (v10, v11, v01)))
                }
            }
        }
        return grid
    }

    /// Gate 1a: a sheet spawned as an S-fold (three layers) settles on the
    /// ground. Without V-T contacts the layers pass through each other
    /// (nodes interleave through the holes); with them the stack keeps its
    /// layer order and thickness.
    public static func clothfold(res: Int = 24) -> PhysicsScene {
        var s = PhysicsScene(name: "clothfold")
        s.settings.iterations = 20
        s.settings.betaLin = 20000
        addGround(&s, friction: 0.9)

        let nu = max(18, res), nv = max(12, res / 2)
        let spacing: Float = 0.12
        let r: Float = 0.045
        let rb: Float = 2.5 * r          // fold bend radius
        let lenTotal = Float(nu - 1) * spacing
        let straight = (lenTotal - 2 * Float.pi * rb) / 3

        // S-fold arc-length profile in (x, z), draped just above the ground
        func profile(_ sArc: Float) -> (Float, Float) {
            let a1 = straight
            let b1 = a1 + Float.pi * rb
            let a2 = b1 + straight
            let b2 = a2 + Float.pi * rb
            if sArc < a1 { return (sArc, 0) }
            if sArc < b1 {
                let t = (sArc - a1) / rb
                return (a1 + rb * sin(t), rb - rb * cos(t))
            }
            if sArc < a2 { return (a1 - (sArc - b1), 2 * rb) }
            if sArc < b2 {
                let t = (sArc - a2) / rb
                return (a1 - straight - rb * sin(t), 2 * rb + rb * (1 - cos(t)))
            }
            return (a1 - straight + (sArc - b2), 4 * rb)
        }

        var positions: [[F3]] = []
        for i in 0..<nu {
            let (x, z) = profile(Float(i) * spacing)
            var row: [F3] = []
            for j in 0..<nv {
                row.append(F3(x - straight / 2, Float(j) * spacing - Float(nv - 1) * spacing / 2,
                              z + r + 0.02))
            }
            positions.append(row)
        }
        addClothGrid(&s, positions: positions, thickness: r,
                     massPerNode: 0.008, friction: 0.9)
        return s
    }

    /// Gate 1b: cloth draped over a rigid pedestal, then a heavy box dropped
    /// on top. The box face must rest on the cloth SURFACE (corner-vs-triangle
    /// contacts), not dimple through the node gaps — and never poke through
    /// even at 8x the total cloth mass.
    public static func boxoncloth(res: Int = 24, massRatio: Float = 8) -> PhysicsScene {
        var s = PhysicsScene(name: "boxoncloth")
        s.settings.iterations = 20
        s.settings.betaLin = 20000
        addGround(&s, friction: 0.8)

        // pedestal: static box the cloth drapes over
        _ = s.addBody(size: F3(1.2, 1.2, 0.8), density: 0, friction: 0.6,
                      position: F3(0, 0, 0.4))

        let n = max(16, res)
        let size: Float = 2.6
        let spacing = size / Float(n - 1)
        let r: Float = min(0.05, 0.3 * spacing)
        let massPerNode: Float = 0.01
        var positions: [[F3]] = []
        for i in 0..<n {
            var row: [F3] = []
            for j in 0..<n {
                row.append(F3(Float(i) * spacing - size / 2,
                              Float(j) * spacing - size / 2,
                              0.8 + r + 0.02))
            }
            positions.append(row)
        }
        addClothGrid(&s, positions: positions, thickness: r,
                     massPerNode: massPerNode, friction: 0.8)

        // the test article: a box weighing massRatio x the whole cloth
        let clothMass = Float(n * n) * massPerNode
        let boxSide: Float = 0.55
        let density = massRatio * clothMass / (boxSide * boxSide * boxSide)
        _ = s.addBody(size: F3(repeating: boxSide), density: density,
                      friction: 0.7, position: F3(0, 0, 1.45))
        return s
    }

    /// Gate 2: hammock strung between two static posts; a rigid box rides
    /// in the middle. Inextensibility gate: structural stretch < 2% under
    /// load.
    public static func hammock(res: Int = 20, structuralK: Float = 2e5,
                               hardRods: Bool = false) -> PhysicsScene {
        var s = PhysicsScene(name: "hammock")
        s.settings.iterations = 20
        s.settings.betaLin = 20000
        addGround(&s, friction: 0.8)

        let nu = max(16, res), nv = max(10, res * 2 / 3)
        let spacing: Float = 0.14
        let r: Float = min(0.05, 0.3 * spacing)
        let spanZ: Float = 1.6
        var positions: [[F3]] = []
        for i in 0..<nu {
            var row: [F3] = []
            for j in 0..<nv {
                row.append(F3(Float(i) * spacing - Float(nu - 1) * spacing / 2,
                              Float(j) * spacing - Float(nv - 1) * spacing / 2,
                              spanZ))
            }
            positions.append(row)
        }
        let grid = addClothGrid(&s, positions: positions, thickness: r,
                                massPerNode: 0.012, friction: 0.8,
                                structuralK: structuralK)
        if hardRods {
            // flip structural springs to hard AL rods (set via flag)
            for i in 0..<s.springs.count where s.springs[i].stiffness == structuralK {
                s.springs[i].hard = true
            }
        }
        // pin the two short ends to the world (taut hammock)
        for j in 0..<nv {
            for i in [0, nu - 1] {
                let node = grid[i][j]
                s.addJoint(SceneJoint(bodyA: -1, bodyB: node,
                                      rA: s.bodies[node].position, rB: .zero))
            }
        }
        // cargo: rigid box dropped into the middle
        _ = s.addBody(size: F3(0.5, 0.5, 0.5), density: 8, friction: 0.6,
                      position: F3(0, 0, spanZ + 0.6))
        return s
    }

    /// Drape quality scene: cloth falls over a sphere; folds should look
    /// like fabric, not crystal facets.
    public static func drape(res: Int = 28, friction: Float = 0.6) -> PhysicsScene {
        var s = PhysicsScene(name: "drape")
        s.settings.iterations = 20
        s.settings.betaLin = 20000
        addGround(&s, friction: friction)
        _ = s.addSphere(diameter: 1.1, density: 0, friction: friction,
                        position: F3(0, 0, 1.0))
        let n = max(20, res)
        let size: Float = 2.8
        let spacing = size / Float(n - 1)
        let r: Float = min(0.05, 0.3 * spacing)
        var positions: [[F3]] = []
        for i in 0..<n {
            var row: [F3] = []
            for j in 0..<n {
                row.append(F3(Float(i) * spacing - size / 2,
                              Float(j) * spacing - size / 2, 1.75))
            }
            positions.append(row)
        }
        addClothGrid(&s, positions: positions, thickness: r,
                     massPerNode: 0.008, friction: friction)
        return s
    }

    /// Gate 4 scene: tets + cloth + rigid bodies in one solve. A soft block
    /// sits on the ground, cloth drapes over it, rigid boxes drop on top.
    public static func clothcombo(res: Int = 20) -> PhysicsScene {
        var s = PhysicsScene(name: "clothcombo")
        s.settings.iterations = 20
        s.settings.betaLin = 20000
        s.settings.lambdaMax = 1.0e6
        addGround(&s, friction: 0.8)

        addSoftBlock(&s, center: F3(0, 0, 0.7), nx: 4, ny: 4, nz: 4,
                     spacing: 0.34, mu: 4000, lambda: 40000,
                     massPerNode: 0.05)

        let n = max(16, res)
        let size: Float = 2.6
        let spacing = size / Float(n - 1)
        let r: Float = min(0.05, 0.3 * spacing)
        var positions: [[F3]] = []
        for i in 0..<n {
            var row: [F3] = []
            for j in 0..<n {
                row.append(F3(Float(i) * spacing - size / 2,
                              Float(j) * spacing - size / 2, 1.6))
            }
            positions.append(row)
        }
        addClothGrid(&s, positions: positions, thickness: r,
                     massPerNode: 0.01, friction: 0.8)

        _ = s.addBody(size: F3(0.45, 0.45, 0.45), density: 2, friction: 0.7,
                      position: F3(-0.1, 0.1, 2.6))
        _ = s.addSphere(diameter: 0.5, density: 1.5, friction: 0.5,
                        position: F3(0.5, -0.4, 3.1))
        return s
    }
}
