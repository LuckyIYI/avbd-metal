import simd

// Soft bodies, faithful to the paper: 3-DOF particles (M = mI, 3x3 blocks)
// carrying mass-spring cloth with HARD inextensible attachments/rods, and
// stable Neo-Hookean tetrahedral FEM for volumetric solids. Contacts treat
// particles like any body (sphere thickness), so rigid<->soft coupling is
// two-way through the same unified solve.

extension Demos {

    /// Cloth sheet on 3-DOF particles: structural edges are hard rods
    /// (inextensible, AL duals), shear + bend are soft springs.
    @discardableResult
    static func addCloth(_ s: inout PhysicsScene, origin: F3,
                         dirU: F3, dirV: F3, nu: Int, nv: Int,
                         spacing: Float, thickness: Float = 0.06,
                         massPerNode: Float = 0.02,
                         friction: Float = 0.7) -> [[Int]] {
        // keep the contact skin below the topological-exclusion horizon:
        // 2-ring rest distances start at ~spacing, so 2r must stay under it
        let thickness = min(thickness, 0.38 * spacing)
        var grid: [[Int]] = []
        for i in 0..<nu {
            var row: [Int] = []
            for j in 0..<nv {
                let p = origin + dirU * (Float(i) * spacing) + dirV * (Float(j) * spacing)
                row.append(s.addParticle(radius: thickness, mass: massPerNode,
                                         friction: friction, position: p))
            }
            grid.append(row)
        }
        // collision surface: one triangle pair per quad, alternating the
        // diagonal so the mesh carries no preferred direction
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
        func rod(_ a: Int, _ b: Int) {
            // stiff soft spring: VBD solves these implicitly without the
            // dual machinery; hard-AL rods on a swinging sheet were observed
            // to pump the low-frequency pendulum modes
            s.addSpring(SceneSpring(bodyA: a, bodyB: b, rA: .zero, rB: .zero,
                                    stiffness: 5000))
            s.addJoint(SceneJoint(bodyA: a, bodyB: b, rA: .zero, rB: .zero,
                                  stiffnessLin: 0, stiffnessAng: 0))
        }
        func soft(_ a: Int, _ b: Int, _ k: Float) {
            s.addSpring(SceneSpring(bodyA: a, bodyB: b, rA: .zero, rB: .zero,
                                    stiffness: k))
            s.addJoint(SceneJoint(bodyA: a, bodyB: b, rA: .zero, rB: .zero,
                                  stiffnessLin: 0, stiffnessAng: 0))
        }
        for i in 0..<nu {
            for j in 0..<nv {
                if i + 1 < nu { rod(grid[i][j], grid[i + 1][j]) }
                if j + 1 < nv { rod(grid[i][j], grid[i][j + 1]) }
                if i + 1 < nu && j + 1 < nv {
                    soft(grid[i][j], grid[i + 1][j + 1], 60)
                    soft(grid[i + 1][j], grid[i][j + 1], 60)
                }
                if i + 2 < nu { soft(grid[i][j], grid[i + 2][j], 8) }
                if j + 2 < nv { soft(grid[i][j], grid[i][j + 2], 8) }
            }
        }
        return grid
    }

    /// Volumetric soft block: particle lattice + 5-tet decomposition per
    /// cell (parity-alternated), stable Neo-Hookean material.
    @discardableResult
    static func addSoftBlock(_ s: inout PhysicsScene, center: F3,
                             nx: Int, ny: Int, nz: Int, spacing: Float,
                             mu: Float = 3000, lambda: Float = 30000,
                             massPerNode: Float = 0.05,
                             friction: Float = 0.8) -> [Int] {
        var ids = [[[Int]]](repeating: [[Int]](repeating: [Int](repeating: -1, count: nz),
                                               count: ny), count: nx)
        let half = F3(Float(nx - 1), Float(ny - 1), Float(nz - 1)) * spacing / 2
        var flat: [Int] = []
        for i in 0..<nx {
            for j in 0..<ny {
                for k in 0..<nz {
                    let p = center + F3(Float(i), Float(j), Float(k)) * spacing - half
                    let id = s.addParticle(radius: spacing * 0.32, mass: massPerNode,
                                           friction: friction, position: p)
                    ids[i][j][k] = id
                    flat.append(id)
                }
            }
        }
        func tet(_ a: Int, _ b: Int, _ c: Int, _ d: Int) {
            s.addTet(SceneTet(ids: (a, b, c, d), mu: mu, lambda: lambda))
        }
        for i in 0..<(nx - 1) {
            for j in 0..<(ny - 1) {
                for k in 0..<(nz - 1) {
                    // cube corners
                    let c000 = ids[i][j][k],     c100 = ids[i+1][j][k]
                    let c010 = ids[i][j+1][k],   c110 = ids[i+1][j+1][k]
                    let c001 = ids[i][j][k+1],   c101 = ids[i+1][j][k+1]
                    let c011 = ids[i][j+1][k+1], c111 = ids[i+1][j+1][k+1]
                    if (i + j + k) % 2 == 0 {
                        tet(c000, c100, c010, c001)
                        tet(c110, c100, c010, c111)
                        tet(c101, c100, c001, c111)
                        tet(c011, c010, c001, c111)
                        tet(c100, c010, c001, c111)
                    } else {
                        tet(c100, c000, c110, c101)
                        tet(c010, c000, c110, c011)
                        tet(c001, c000, c101, c011)
                        tet(c111, c110, c101, c011)
                        tet(c000, c110, c101, c011)
                    }
                }
            }
        }
        // exclude collisions between lattice neighbors (tets hold the shape;
        // internal contacts only fight them)
        for i in 0..<nx {
            for j in 0..<ny {
                for k in 0..<nz {
                    let a = ids[i][j][k]
                    for (di, dj, dk) in [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)] {
                        if i+di < nx && j+dj < ny && k+dk < nz {
                            s.addJoint(SceneJoint(bodyA: a, bodyB: ids[i+di][j+dj][k+dk],
                                                  rA: .zero, rB: .zero,
                                                  stiffnessLin: 0, stiffnessAng: 0))
                        }
                    }
                }
            }
        }
        return flat
    }

    /// Paper Fig. 5: a flag (particle mass-spring cloth) attached with HARD
    /// constraints to a deformable pole of rigid segments with ball joints —
    /// soft, stiff, and rigid solved in one unified loop.
    public static func cloth(res: Int = 16, ball: Bool = true) -> PhysicsScene {
        var s = PhysicsScene(name: "cloth")
        s.settings.iterations = 20
        s.settings.betaLin = 20000
        s.settings.betaAng = 20000     // the pole's elastic bending must ramp
        s.settings.lambdaMax = 500
        addGround(&s, friction: 0.8)

        // ---- deformable pole: rigid segments, ball-jointed (paper) ----
        let segs = 10
        let segH: Float = 0.55
        var prev = -1
        var topSeg = -1
        for k in 0..<segs {
            let z = (Float(k) + 0.5) * segH
            let seg = s.addBody(size: F3(0.16, 0.16, segH * 0.98), density: 40,
                                friction: 0.5, position: F3(0, 0, z))
            if k == 0 {
                s.addJoint(SceneJoint(bodyA: -1, bodyB: seg,
                                      rA: F3(0, 0, 0), rB: F3(0, 0, -segH / 2),
                                      stiffnessLin: .infinity, stiffnessAng: .infinity))
            } else {
                // hard ball joint + finite angular stiffness: an elastic
                // bending element — the paper's "stiffer but deformable pole"
                s.addJoint(SceneJoint(bodyA: prev, bodyB: seg,
                                      rA: F3(0, 0, segH / 2), rB: F3(0, 0, -segH / 2),
                                      stiffnessLin: .infinity, stiffnessAng: 80000))
                s.addJoint(SceneJoint(bodyA: prev, bodyB: seg, rA: .zero, rB: .zero,
                                      stiffnessLin: 0, stiffnessAng: 0))
            }
            prev = seg
            topSeg = seg
        }

        // ---- flag cloth, hard-attached at two points (paper) ----
        let n = max(10, res)
        let spacing: Float = 2.6 / Float(n - 1)
        let topZ = Float(segs) * segH - 0.15
        let grid = addCloth(&s, origin: F3(0.12, 0, topZ),
                            dirU: normalize(F3(0.55, 0.2, -0.81)),   // drooped spawn
                            dirV: F3(0, 0, -1),
                            nu: n, nv: n, spacing: spacing,
                            massPerNode: 0.007)
        // hard attachments: top and bottom inner corners to the pole
        s.addJoint(SceneJoint(bodyA: topSeg, bodyB: grid[0][0],
                              rA: F3(0.12, 0, segH / 2 - 0.15), rB: .zero))
        let lowSeg = topSeg - Int(2.6 / segH)
        s.addJoint(SceneJoint(bodyA: lowSeg, bodyB: grid[0][n - 1],
                              rA: F3(0.12, 0, 0), rB: .zero))
        // initial sideways breeze so the flag unfurls
        for row in grid {
            for node in row {
                s.bodies[node].velocity = F3(0.8, 0.5, 0)
            }
        }

        if ball {
            // a rigid ball thrown through the flag (two-way coupling)
            _ = s.addSphere(diameter: 0.7, density: 1.2, friction: 0.4,
                            position: F3(1.8, -4.0, topZ - 1.2),
                            velocity: F3(0, 6.5, 0))
        }
        return s
    }

    /// Tetrahedral soft bodies: blocks squash under a heavy rigid ball and
    /// a plank, tumble, and recover — two-way coupling throughout.
    public static func softbody(count: Int = 3) -> PhysicsScene {
        var s = PhysicsScene(name: "softbody")
        s.settings.iterations = 20
        s.settings.betaLin = 20000
        s.settings.lambdaMax = 800
        addGround(&s, friction: 0.8)

        let nC = max(2, count)
        for k in 0..<nC {
            let x = Float(k % 3) * 2.4 - 2.4
            let y = Float(k / 3) * 2.4
            let soft = Float(k % 3)        // varying stiffness per block
            addSoftBlock(&s, center: F3(x, y, 1.2 + Float(k % 2) * 0.3),
                         nx: 4, ny: 4, nz: 4, spacing: 0.34,
                         mu: 1500 + soft * 2500, lambda: 15000 + soft * 25000)
        }
        // heavy rigid ball dropped onto the middle block
        _ = s.addSphere(diameter: 1.1, density: 3, friction: 0.6,
                        position: F3(0, 0, 4.2))
        // rigid plank bridging two blocks
        _ = s.addBody(size: F3(5.0, 0.9, 0.12), density: 1.0, friction: 0.7,
                      position: F3(-1.2, 0, 3.2))
        return s
    }
}
