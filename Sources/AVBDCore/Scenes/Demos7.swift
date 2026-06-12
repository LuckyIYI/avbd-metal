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
    public static func clothfold(res: Int = 48, friction: Float = 0.9,
                                 membraneMu: Float = 300,
                                 bend: Float = 5e-4) -> PhysicsScene {
        var s = PhysicsScene(name: "clothfold")
        s.settings.iterations = 20
        s.settings.betaLin = 20000
        s.settings.particleDamping = 1.5
        s.settings.clothViscosity = 0.25
        addGround(&s, friction: friction)

        // a folded two-towel sheet: FIXED physical size (2.2 m x ~1.1 m),
        // resolution only refines the mesh (Colossal ~26k vertices)
        let nu = max(36, res), nv = max(18, res / 2)
        let lenTotal: Float = 2.2
        let spacing = lenTotal / Float(nu - 1)
        let r: Float = min(0.045, 0.36 * spacing)
        let rb: Float = max(0.055, 2.6 * r)   // fold bend radius
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

        // towel-like areal density (~0.5 kg/m^2) independent of resolution
        let massPerNode = 0.5 * spacing * spacing
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
                     massPerNode: massPerNode, friction: friction,
                     membraneMu: membraneMu, membraneBend: bend)
        s.settings.cameraDistance = 3.2
        s.settings.cameraTargetZ = 0.25
        return s
    }

    /// Gate 1b: cloth draped over a rigid pedestal, then a heavy box dropped
    /// on top. The box face must rest on the cloth SURFACE (corner-vs-triangle
    /// contacts), not dimple through the node gaps — and never poke through
    /// even at 8x the total cloth mass.
    public static func boxoncloth(res: Int = 24, massRatio: Float = 8,
                                  friction: Float = 0.8,
                                  membraneMu: Float = 300,
                                  bend: Float = 5e-4) -> PhysicsScene {
        var s = PhysicsScene(name: "boxoncloth")
        s.settings.iterations = 20
        s.settings.betaLin = 20000
        s.settings.particleDamping = 1.5
        s.settings.clothViscosity = 0.25
        addGround(&s, friction: friction)

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
                     massPerNode: massPerNode, friction: friction,
                     membraneMu: membraneMu, membraneBend: bend)

        // the test article: a box weighing massRatio x the whole cloth
        let clothMass = Float(n * n) * massPerNode
        let boxSide: Float = 0.55
        let density = massRatio * clothMass / (boxSide * boxSide * boxSide)
        _ = s.addBody(size: F3(repeating: boxSide), density: density,
                      friction: 0.7, position: F3(0, 0, 1.45))
        s.settings.cameraDistance = 7
        s.settings.cameraTargetZ = 0.8
        return s
    }

    /// Gate 2: hammock strung between two static posts; a rigid box rides
    /// in the middle. Inextensibility gate: structural stretch < 2% under
    /// load.
    public static func hammock(res: Int = 20, structuralK: Float = 5000,
                               hardRods: Bool = true, cubes: Int = 1,
                               friction: Float = 0.8,
                               membraneMu: Float = 300,
                               bend: Float = 5e-4) -> PhysicsScene {
        var s = PhysicsScene(name: "hammock")
        s.settings.iterations = 20
        s.settings.betaLin = 20000
        s.settings.particleDamping = 1.5
        s.settings.clothViscosity = 0.25
        addGround(&s, friction: friction)

        let nu = max(16, res), nv = max(10, res * 2 / 3)
        let spacing: Float = 0.14
        let r: Float = min(0.05, 0.3 * spacing)
        // spawn with catenary slack: an INEXTENSIBLE sheet pinned taut is an
        // infinite-tension geometry — real hammocks hang with sag
        let span = Float(nu - 1) * spacing
        let sag: Float = 0.22 * span
        // pins rise with the span: sag + cargo dip must clear the floor at
        // any scale (a Colossal hammock pinned at 1.6 just lies on the ground)
        let spanZ: Float = sag + 1.1
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
                                massPerNode: 0.012, friction: friction,
                                structuralK: structuralK, hardRods: hardRods,
                                membraneMu: membraneMu, membraneBend: bend)
        // pin the two short ends to the world (taut hammock)
        for j in 0..<nv {
            for i in [0, nu - 1] {
                let node = grid[i][j]
                s.addJoint(SceneJoint(bodyA: -1, bodyB: node,
                                      rA: s.bodies[node].position, rB: .zero))
            }
        }
        // cargo: rigid boxes cascade into the sag, spread along the span so
        // bigger hammocks carry proportionally more freight (drop height
        // cycles in tiers — no skyscraper drops at big cube counts)
        let nC = max(1, cubes)
        var rng = SplitMix64(seed: 7)
        for k in 0..<nC {
            let t = nC == 1 ? 0 : (Float(k) / Float(nC - 1) * 2 - 1) * 0.45
            let jitter = F3((rng.nextFloat() - 0.5) * 0.2,
                            (rng.nextFloat() - 0.5) * 0.3, 0)
            _ = s.addBody(size: F3(0.5, 0.5, 0.5), density: 8, friction: 0.6,
                          position: F3(t * span, 0,
                                       spanZ + 0.6 + Float(k % 4) * 0.7)
                                    + jitter,
                          rotation: Quat(angle: rng.nextFloat() * 0.5,
                                         axis: normalize(F3(rng.nextFloat(),
                                                            rng.nextFloat(), 1))))
        }
        s.settings.cameraDistance = span * 1.9 + 2.5
        s.settings.cameraTargetZ = spanZ * 0.55
        return s
    }

    /// Drape quality scene: cloth falls over a sphere; folds should look
    /// like fabric, not crystal facets.
    public static func drape(res: Int = 28, friction: Float = 0.6,
                             membrane: Bool = true,
                             membraneMu: Float = 300,
                             bend: Float = 5e-4) -> PhysicsScene {
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
                     membraneMu: membrane ? membraneMu : 0, membraneBend: bend)
        s.settings.cameraDistance = 7
        s.settings.cameraTargetZ = 1.0
        return s
    }

    /// Inextensibility experiment: a sheet pinned at two top corners, given
    /// a hard sideways kick so it swings and whips. The structural-edge
    /// mechanism is selectable: stiff soft springs (k) or hard AL rods.
    /// Pumping diagnosis: KE envelope must decay, never grow.
    public static func flagwhip(res: Int = 16, structuralK: Float = 5000,
                                hardRods: Bool = true, kick: Float = 3.0,
                                membraneMu: Float = 300,
                                bend: Float = 5e-4) -> PhysicsScene {
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
                                structuralK: structuralK, hardRods: hardRods,
                                membraneMu: membraneMu, membraneBend: bend)
        // pin the two top corners
        for i in [0, n - 1] {
            let node = grid[i][0]
            s.addJoint(SceneJoint(bodyA: -1, bodyB: node,
                                  rA: s.bodies[node].position, rB: .zero))
        }
        // sideways kick: the sheet swings like a pendulum and whips
        let kick: Float = ProcessInfo.processInfo.environment["AVBD_NO_KICK"] != nil ? 0 : kick
        for i in 0..<n {
            for j in 0..<n {
                s.bodies[grid[i][j]].velocity = F3(kick, 0, 0) * (Float(j) / Float(n - 1))
            }
        }
        s.settings.cameraDistance = 7
        s.settings.cameraTargetZ = 2.0
        return s
    }

    /// Layered drape: three sheets of decreasing size fall in sequence
    /// onto a sphere — the bottom one drapes the ball, the others drape
    /// the cloth below them. Cloth-on-cloth stacking at a glance (each
    /// sheet is its own component, so each gets its own color).
    public static func multidrape(res: Int = 26, friction: Float = 0.5,
                                  membraneMu: Float = 300,
                                  bend: Float = 5e-4) -> PhysicsScene {
        var s = PhysicsScene(name: "multidrape")
        s.settings.iterations = 20
        s.settings.betaLin = 20000
        s.settings.particleDamping = 1.5
        s.settings.clothViscosity = 0.25
        addGround(&s, friction: friction)
        _ = s.addSphere(diameter: 1.2, density: 0, friction: friction,
                        position: F3(0, 0, 1.0))

        let n = max(18, res)
        for k in 0..<3 {
            let size: Float = 3.0 - Float(k) * 0.45
            let spacing = size / Float(n - 1)
            let r: Float = min(0.05, 0.3 * spacing)
            let z = 1.95 + Float(k) * 0.55
            let ang = Float(k) * 0.5      // each sheet rotated for variety
            let ca = cos(ang), sa = sin(ang)
            var positions: [[F3]] = []
            for i in 0..<n {
                var row: [F3] = []
                for j in 0..<n {
                    let u = Float(i) * spacing - size / 2
                    let v = Float(j) * spacing - size / 2
                    row.append(F3(ca * u - sa * v, sa * u + ca * v, z))
                }
                positions.append(row)
            }
            addClothGrid(&s, positions: positions, thickness: r,
                         massPerNode: 0.008, friction: friction,
                         membraneMu: membraneMu, membraneBend: bend)
        }
        s.settings.cameraDistance = 8
        s.settings.cameraTargetZ = 1.1
        return s
    }

    /// E-E showcase: long ribbons rain criss-cross onto a thin capsule
    /// cross; they fold over the bars, slide off and tangle with each
    /// other — long free edges rubbing at every angle is edge-edge
    /// territory (each ribbon is its own surface component, so each gets
    /// its own color).
    public static func ribbons(len: Int = 22, count: Int = 6,
                               friction: Float = 0.35,
                               membraneMu: Float = 300,
                               bend: Float = 5e-4) -> PhysicsScene {
        var s = PhysicsScene(name: "ribbons")
        s.settings.iterations = 20
        s.settings.betaLin = 20000
        s.settings.particleDamping = 1.5
        s.settings.clothViscosity = 0.25
        addGround(&s, friction: friction)

        // a low cross of thin static capsules catches and folds the ribbons
        let pegZ: Float = 2.3
        _ = s.addCapsule(length: 5.0, radius: 0.12, density: 0, friction: friction,
                         position: F3(0, 0, pegZ),
                         rotation: Quat(angle: .pi / 2, axis: F3(0, 1, 0)))
        _ = s.addCapsule(length: 5.0, radius: 0.12, density: 0, friction: friction,
                         position: F3(0, 0, pegZ),
                         rotation: Quat(angle: .pi / 2, axis: F3(1, 0, 0)))

        let wid = 4
        let spacing: Float = 0.22
        // ribbons are CLOTH: keep the skin thin (render thickness == contact
        // thickness, so a fat radius reads as rubber strips, not fabric)
        let r: Float = 0.028
        var rng = SplitMix64(seed: 11)
        for k in 0..<count {
            let ang = Float(k) * (.pi / Float(max(1, count))) + 0.2
            let dir = F3(cos(ang), sin(ang), 0)
            let side = F3(-sin(ang), cos(ang), 0)
            let off = side * ((rng.nextFloat() - 0.5) * 1.0)
            let z = pegZ + 1.0 + Float(k) * 0.55
            var pos: [[F3]] = []
            for i in 0..<len {
                var row: [F3] = []
                let u = Float(i) * spacing - Float(len - 1) * spacing / 2
                for j in 0..<wid {
                    let v = Float(j) * spacing - Float(wid - 1) * spacing / 2
                    row.append(dir * u + side * v + off + F3(0, 0, z))
                }
                pos.append(row)
            }
            addClothGrid(&s, positions: pos, thickness: r, massPerNode: 0.01,
                         friction: friction)
        }
        s.settings.cameraDistance = 12
        s.settings.cameraTargetZ = 1.6
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
    public static func clothcombo(res: Int = 20, friction: Float = 0.8,
                                  membraneMu: Float = 300,
                                  bend: Float = 5e-4,
                                  blockMu: Float = 4000) -> PhysicsScene {
        var s = PhysicsScene(name: "clothcombo")
        s.settings.iterations = 20
        s.settings.betaLin = 20000
        s.settings.particleDamping = 1.5
        s.settings.clothViscosity = 0.25
        s.settings.lambdaMax = 1.0e6
        addGround(&s, friction: friction)

        // block lattice at CLOTH scale: a coarse 0.34 lattice makes its big
        // particles set the element-grid cell size for the whole scene and
        // the fine cloth elements cram ~40 per cell (vt+ee went 1 -> 8.7 ms)
        addSoftBlock(&s, center: F3(0, 0, 0.7), nx: 7, ny: 7, nz: 7,
                     spacing: 0.17, mu: blockMu, lambda: 10 * blockMu,
                     massPerNode: 0.0095)

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
                     massPerNode: 0.01, friction: friction,
                     membraneMu: membraneMu, membraneBend: bend)

        _ = s.addBody(size: F3(0.45, 0.45, 0.45), density: 2, friction: 0.7,
                      position: F3(-0.1, 0.1, 2.6))
        _ = s.addSphere(diameter: 0.5, density: 1.5, friction: 0.5,
                        position: F3(0.5, -0.4, 3.1))
        s.settings.cameraDistance = 8
        s.settings.cameraTargetZ = 1.0
        return s
    }
}
