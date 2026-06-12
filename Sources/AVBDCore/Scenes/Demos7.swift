import Foundation
import simd

// Cloth gate scenes: surface-element collision (V-T, rigid-T, E-E),
// inextensibility, and combined soft/rigid coupling. These back the
// ClothTests gates and double as app demos.

extension Demos {

    /// Low-level cloth builder over explicit node positions. Structural
    /// edges are HARD AL rods (exact inextensibility — the rod-pumping
    /// experiment cleared them once the color-skip and dual-cap bugs were
    /// fixed), shear/bend the usual soft set, and every quad contributes
    /// two collision triangles (alternating diagonal).
    @discardableResult
    static func addClothGrid(_ s: inout PhysicsScene, positions: [[F3]],
                             thickness: Float, massPerNode: Float,
                             friction: Float = 0.7,
                             structuralK: Float = 5000,
                             hardRods: Bool = true,
                             shearK: Float = 60, bendK: Float = 8,
                             membraneMu: Float = 300,
                             membraneBend: Float = 5e-4) -> [[Int]] {
        // experiment override: AVBD_SOFT_CLOTH forces stiff-spring structure
        let hardRods = hardRods
            && ProcessInfo.processInfo.environment["AVBD_SOFT_CLOTH"] == nil
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
        let rodPen = ProcessInfo.processInfo.environment["AVBD_ROD_PEN"]
            .flatMap(Float.init)
        func link(_ a: Int, _ b: Int, _ k: Float, hard: Bool = false) {
            // For hard rods `stiffness` is the PENALTY CAP, and modest is
            // right: lambda carries the rod tension exactly; letting the
            // penalty ramp toward 1e6 on gram nodes reproduces the stiff-
            // spring overshoot explosion the duals exist to avoid.
            // (No exclusion joints: intra-surface sphere pairs are dropped
            // wholesale at the broadphase — elements own self-collision.)
            s.addSpring(SceneSpring(bodyA: a, bodyB: b, rA: .zero, rB: .zero,
                                    stiffness: hard ? (rodPen ?? k) : k, hard: hard))
        }
        for i in 0..<nu {
            for j in 0..<nv {
                if i + 1 < nu { link(grid[i][j], grid[i + 1][j], structuralK, hard: hardRods) }
                if j + 1 < nv { link(grid[i][j], grid[i][j + 1], structuralK, hard: hardRods) }
                // membrane elements replace the crossed-diagonal/2-ring
                // spring zoo with isotropic StVK shear + quadratic bending
                if membraneMu <= 0 {
                    if i + 1 < nu && j + 1 < nv {
                        link(grid[i][j], grid[i + 1][j + 1], shearK)
                        link(grid[i + 1][j], grid[i][j + 1], shearK)
                    }
                    if i + 2 < nu { link(grid[i][j], grid[i + 2][j], bendK) }
                    if j + 2 < nv { link(grid[i][j], grid[i][j + 2], bendK) }
                }
            }
        }
        for i in 0..<(nu - 1) {
            for j in 0..<(nv - 1) {
                let v00 = grid[i][j], v10 = grid[i + 1][j]
                let v01 = grid[i][j + 1], v11 = grid[i + 1][j + 1]
                let tri: (Int, Int, Int) -> SceneTri = { a, b, c in
                    SceneTri(ids: (a, b, c), mu: membraneMu,
                             lambda: membraneMu, bend: membraneBend)
                }
                if (i + j) % 2 == 0 {
                    s.addTri(tri(v00, v10, v11))
                    s.addTri(tri(v00, v11, v01))
                } else {
                    s.addTri(tri(v00, v10, v01))
                    s.addTri(tri(v10, v11, v01))
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
        s.settings.particleDamping = 1.5
        s.settings.clothViscosity = 0.25
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
        s.settings.particleDamping = 1.5
        s.settings.clothViscosity = 0.25
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
    public static func hammock(res: Int = 20, structuralK: Float = 5000,
                               hardRods: Bool = true) -> PhysicsScene {
        var s = PhysicsScene(name: "hammock")
        s.settings.iterations = 20
        s.settings.betaLin = 20000
        s.settings.particleDamping = 1.5
        s.settings.clothViscosity = 0.25
        addGround(&s, friction: 0.8)

        let nu = max(16, res), nv = max(10, res * 2 / 3)
        let spacing: Float = 0.14
        let r: Float = min(0.05, 0.3 * spacing)
        let spanZ: Float = 1.6
        // spawn with catenary slack: an INEXTENSIBLE sheet pinned taut is an
        // infinite-tension geometry — real hammocks hang with sag
        let span = Float(nu - 1) * spacing
        let sag: Float = 0.22 * span
        var positions: [[F3]] = []
        for i in 0..<nu {
            var row: [F3] = []
            let t = Float(i) / Float(nu - 1) * 2 - 1        // -1..1 across
            let dip = sag * (1 - t * t)
            for j in 0..<nv {
                row.append(F3(Float(i) * spacing - span / 2,
                              Float(j) * spacing - Float(nv - 1) * spacing / 2,
                              spanZ - dip))
            }
            positions.append(row)
        }
        let grid = addClothGrid(&s, positions: positions, thickness: r,
                                massPerNode: 0.012, friction: 0.8,
                                structuralK: structuralK, hardRods: hardRods)
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
    public static func drape(res: Int = 28, friction: Float = 0.6,
                             membrane: Bool = true) -> PhysicsScene {
        var s = PhysicsScene(name: "drape")
        s.settings.iterations = 20
        s.settings.betaLin = 20000
        s.settings.particleDamping = 1.5
        s.settings.clothViscosity = 0.25
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
                     massPerNode: 0.008, friction: friction,
                     membraneMu: membrane ? 300 : 0)
        return s
    }

    /// Inextensibility experiment: a sheet pinned at two top corners, given
    /// a hard sideways kick so it swings and whips. The structural-edge
    /// mechanism is selectable: stiff soft springs (k) or hard AL rods.
    /// Pumping diagnosis: KE envelope must decay, never grow.
    public static func flagwhip(res: Int = 16, structuralK: Float = 5000,
                                hardRods: Bool = true) -> PhysicsScene {
        var s = PhysicsScene(name: "flagwhip")
        s.settings.iterations = 20
        s.settings.betaLin = 20000
        s.settings.particleDamping = 1.5
        s.settings.clothViscosity = 0.25
        addGround(&s, friction: 0.6)

        let n = max(12, res)
        let spacing: Float = 2.0 / Float(n - 1)
        let r: Float = min(0.05, 0.3 * spacing)
        let topZ: Float = 3.2
        var positions: [[F3]] = []
        for i in 0..<n {                  // i: across (y), j: down (z)
            var row: [F3] = []
            for j in 0..<n {
                row.append(F3(0, Float(i) * spacing - 1.0,
                              topZ - Float(j) * spacing))
            }
            positions.append(row)
        }
        let grid = addClothGrid(&s, positions: positions, thickness: r,
                                massPerNode: 0.008, friction: 0.6,
                                structuralK: structuralK, hardRods: hardRods)
        // pin the two top corners
        for i in [0, n - 1] {
            let node = grid[i][0]
            s.addJoint(SceneJoint(bodyA: -1, bodyB: node,
                                  rA: s.bodies[node].position, rB: .zero))
        }
        // sideways kick: the sheet swings like a pendulum and whips
        let kick: Float = ProcessInfo.processInfo.environment["AVBD_NO_KICK"] != nil ? 0 : 3.0
        for i in 0..<n {
            for j in 0..<n {
                s.bodies[grid[i][j]].velocity = F3(kick, 0, 0) * (Float(j) / Float(n - 1))
            }
        }
        return s
    }

    /// Gate 3: two narrow cloth strips crossing at 90 degrees. Strip A is
    /// strung taut between world anchors; strip B lies across it and gets
    /// dragged by its leading corners (drag joints returned for the
    /// harness). Their long edges rub corner-over-corner at the crossing —
    /// E-E territory. Returns (scene, dragJointIndices, stripANodes, stripBNodes).
    public static func eecross(len: Int = 18, wid: Int = 4)
        -> (PhysicsScene, [Int], [Int], [Int]) {
        var s = PhysicsScene(name: "eecross")
        s.settings.iterations = 20
        s.settings.betaLin = 20000
        s.settings.particleDamping = 1.5
        s.settings.clothViscosity = 0.25
        addGround(&s, friction: 0.3)

        // A sits just above the ground: there is no room to wrap or swing
        // underneath it, so "below A" can only mean "passed through".
        let spacing: Float = 0.12
        let r: Float = 0.04
        let zA: Float = 0.12

        // strip A: along x, pinned taut at both short ends
        var posA: [[F3]] = []
        for i in 0..<len {
            var row: [F3] = []
            for j in 0..<wid {
                row.append(F3(Float(i) * spacing - Float(len - 1) * spacing / 2,
                              Float(j) * spacing - Float(wid - 1) * spacing / 2,
                              zA))
            }
            posA.append(row)
        }
        let gridA = addClothGrid(&s, positions: posA, thickness: r,
                                 massPerNode: 0.01, friction: 0.3)
        for i in [0, len - 1] {
            for j in 0..<wid {
                let node = gridA[i][j]
                s.addJoint(SceneJoint(bodyA: -1, bodyB: node,
                                      rA: s.bodies[node].position, rB: .zero))
            }
        }

        // strip B: along y, resting across A, slightly above
        var posB: [[F3]] = []
        for i in 0..<len {
            var row: [F3] = []
            for j in 0..<wid {
                row.append(F3(Float(j) * spacing - Float(wid - 1) * spacing / 2,
                              Float(i) * spacing - Float(len - 1) * spacing / 2,
                              zA + 2 * r + 0.02))
            }
            posB.append(row)
        }
        let gridB = addClothGrid(&s, positions: posB, thickness: r,
                                 massPerNode: 0.01, friction: 0.3)
        var dragJoints: [Int] = []
        for j in 0..<wid {                  // leading edge = last row (max y)
            let node = gridB[len - 1][j]
            dragJoints.append(s.joints.count)
            s.addJoint(SceneJoint(bodyA: -1, bodyB: node,
                                  rA: s.bodies[node].position, rB: .zero,
                                  stiffnessLin: 3000, stiffnessAng: 0))
        }
        let bNodes = gridB.flatMap { $0 }
        let aNodes = gridA.flatMap { $0 }
        return (s, dragJoints, aNodes, bNodes)
    }

    /// Gate 4 scene: tets + cloth + rigid bodies in one solve. A soft block
    /// sits on the ground, cloth drapes over it, rigid boxes drop on top.
    public static func clothcombo(res: Int = 20) -> PhysicsScene {
        var s = PhysicsScene(name: "clothcombo")
        s.settings.iterations = 20
        s.settings.betaLin = 20000
        s.settings.particleDamping = 1.5
        s.settings.clothViscosity = 0.25
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
                     massPerNode: 0.01, friction: 0.8, membraneMu: 300)

        _ = s.addBody(size: F3(0.45, 0.45, 0.45), density: 2, friction: 0.7,
                      position: F3(-0.1, 0.1, 2.6))
        _ = s.addSphere(diameter: 0.5, density: 1.5, friction: 0.5,
                        position: F3(0.5, -0.4, 3.1))
        return s
    }
}
