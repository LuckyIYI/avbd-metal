import SimCore
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
        // collision surface AND material: one StVK membrane pair per quad
        // (alternating diagonal — no preferred direction), with quadratic
        // bending across shared edges. Replaces the shear/bend spring zoo.
        for i in 0..<(nu - 1) {
            for j in 0..<(nv - 1) {
                let v00 = grid[i][j], v10 = grid[i + 1][j]
                let v01 = grid[i][j + 1], v11 = grid[i + 1][j + 1]
                let tri: (Int, Int, Int) -> SceneTri = {
                    SceneTri(ids: ($0, $1, $2), mu: 300, lambda: 300, bend: 5e-4)
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
        func rod(_ a: Int, _ b: Int) {
            // hard AL rod: exact inextensibility. The historical "rods pump
            // swinging sheets" observation was root-caused to the stale
            // color-bound skip + unscaled dual caps (both fixed); the
            // 60-second whip experiment shows a decaying KE envelope.
            // Stiffness here is the PENALTY CAP — modest by design: lambda
            // carries the rod tension, the penalty only conditions it.
            s.addSpring(SceneSpring(bodyA: a, bodyB: b, rA: .zero, rB: .zero,
                                    stiffness: 5000, hard: true))
        }
        func soft(_ a: Int, _ b: Int, _ k: Float) {
            s.addSpring(SceneSpring(bodyA: a, bodyB: b, rA: .zero, rB: .zero,
                                    stiffness: k))
        }
        _ = soft     // membrane elements carry shear + bending now
        for i in 0..<nu {
            for j in 0..<nv {
                if i + 1 < nu { rod(grid[i][j], grid[i + 1][j]) }
                if j + 1 < nv { rod(grid[i][j], grid[i][j + 1]) }
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
                             friction: Float = 0.8,
                             selfCollisionEnabled: Bool = false) -> [Int] {
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
            s.addTet(SceneTet(ids: (a, b, c, d), mu: mu, lambda: lambda,
                              selfCollisionEnabled: selfCollisionEnabled))
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

    /// Voxelized soft body over an implicit shape: lattice cells whose
    /// center satisfies `solid` get the 5-tet split with shared nodes —
    /// same recipe as addSoftBlock, arbitrary silhouette.
    @discardableResult
    static func addSoftVoxelShape(_ s: inout PhysicsScene, origin: F3,
                                  spacing: Float, nx: Int, ny: Int, nz: Int,
                                  mu: Float, lambda: Float,
                                  massPerNode: Float, friction: Float = 0.8,
                                  solid: (F3) -> Bool) -> [Int] {
        // occupancy first, then keep only the LARGEST 6-connected component:
        // coarse voxelizations shed orphan fragments (a 1-cell tail) that
        // would drop beside the body as their own little blobs
        var occupied: Set<SIMD3<Int>> = []
        for i in 0..<nx {
            for j in 0..<ny {
                for k in 0..<nz {
                    let center = origin + (F3(Float(i), Float(j), Float(k))
                                           + F3(repeating: 0.5)) * spacing
                    if solid(center) { occupied.insert(SIMD3(i, j, k)) }
                }
            }
        }
        var unvisited = occupied
        var keep: Set<SIMD3<Int>> = []
        while let seed = unvisited.first {
            var comp: Set<SIMD3<Int>> = []
            var stack = [seed]
            unvisited.remove(seed)
            while let c = stack.popLast() {
                comp.insert(c)
                for d in [SIMD3(1, 0, 0), SIMD3(-1, 0, 0), SIMD3(0, 1, 0),
                          SIMD3(0, -1, 0), SIMD3(0, 0, 1), SIMD3(0, 0, -1)] {
                    let n = c &+ d
                    if unvisited.remove(n) != nil { stack.append(n) }
                }
            }
            if comp.count > keep.count { keep = comp }
        }

        var nodeId: [SIMD3<Int>: Int] = [:]
        var flat: [Int] = []
        func node(_ i: Int, _ j: Int, _ k: Int) -> Int {
            let key = SIMD3(i, j, k)
            if let id = nodeId[key] { return id }
            let p = origin + F3(Float(i), Float(j), Float(k)) * spacing
            let id = s.addParticle(radius: spacing * 0.32, mass: massPerNode,
                                   friction: friction, position: p)
            nodeId[key] = id
            flat.append(id)
            return id
        }
        func tet(_ a: Int, _ b: Int, _ c: Int, _ d: Int) {
            s.addTet(SceneTet(ids: (a, b, c, d), mu: mu, lambda: lambda))
        }
        for i in 0..<nx {
            for j in 0..<ny {
                for k in 0..<nz {
                    guard keep.contains(SIMD3(i, j, k)) else { continue }
                    let c000 = node(i, j, k),         c100 = node(i + 1, j, k)
                    let c010 = node(i, j + 1, k),     c110 = node(i + 1, j + 1, k)
                    let c001 = node(i, j, k + 1),     c101 = node(i + 1, j, k + 1)
                    let c011 = node(i, j + 1, k + 1), c111 = node(i + 1, j + 1, k + 1)
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
        // lattice-neighbor exclusion joints (inert): internal sphere contacts
        // under deep squash only fight the tets
        for (key, a) in nodeId {
            for d in [SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 0, 1),
                      SIMD3(1, 1, 0), SIMD3(1, 0, 1), SIMD3(0, 1, 1),
                      SIMD3(1, 1, 1)] {
                if let b = nodeId[key &+ d] {
                    s.addJoint(SceneJoint(bodyA: a, bodyB: b, rA: .zero, rB: .zero,
                                          stiffnessLin: 0, stiffnessAng: 0))
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
        s.settings.particleDamping = 1.5
        s.settings.clothViscosity = 0.25
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

    /// Tetrahedral soft-body showcase: a voxelized BUNNY (body, head, ears,
    /// tail, feet as an implicit ellipsoid union) flops onto the ground and
    /// a rigid ball drops onto its back — ears wobble, body dents and
    /// recovers, two-way coupling throughout.
    public static func softbody(res: Int = 11, stiffness: Float = 2500,
                                friction: Float = 0.8) -> PhysicsScene {
        var s = PhysicsScene(name: "softbody")
        s.settings.iterations = 10
        s.settings.betaLin = 20000
        s.settings.lambdaMax = 1000
        addGround(&s, friction: friction)

        func ellipsoid(_ p: F3, _ c: F3, _ r: F3) -> Bool {
            let d = (p - c) / r
            return dot(d, d) <= 1
        }
        // bunny in a unit-ish frame, ~1.7 tall with ears, sitting pose
        func bunny(_ p: F3) -> Bool {
            return ellipsoid(p, F3(0.00, 0, 0.52), F3(0.62, 0.46, 0.44)) ||   // body
                   ellipsoid(p, F3(0.52, 0, 1.00), F3(0.30, 0.26, 0.27)) ||   // head
                   ellipsoid(p, F3(0.46, 0.14, 1.42), F3(0.11, 0.08, 0.34)) ||  // ear L
                   ellipsoid(p, F3(0.46, -0.14, 1.42), F3(0.11, 0.08, 0.34)) || // ear R
                   ellipsoid(p, F3(-0.60, 0, 0.42), F3(0.16, 0.16, 0.16)) ||  // tail
                   ellipsoid(p, F3(0.30, 0.26, 0.16), F3(0.30, 0.13, 0.14)) ||  // foot L
                   ellipsoid(p, F3(0.30, -0.26, 0.16), F3(0.30, 0.13, 0.14))    // foot R
        }
        let cells = max(8, min(22, res))
        let h = 1.8 / Float(cells)
        let nodes = addSoftVoxelShape(&s, origin: F3(-1.0, -0.7, 0.06),
                                      spacing: h,
                                      nx: Int(2.0 / h), ny: Int(1.4 / h),
                                      nz: Int(1.9 / h),
                                      mu: stiffness, lambda: 10 * stiffness,
                                      massPerNode: 28 * h * h * h,
                                      friction: friction, solid: bunny)
        _ = nodes
        // rigid ball dropped onto the bunny's back
        _ = s.addSphere(diameter: 0.55, density: 2.2, friction: 0.6,
                        position: F3(-0.15, 0, 2.6))
        s.settings.cameraDistance = 6.5
        s.settings.cameraTargetZ = 0.8
        return s
    }
}
