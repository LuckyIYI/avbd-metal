import Foundation
import simd

// Coupled-material showcases: rigid, volumetric soft, and cloth working in
// one unified solve.

extension Demos {

    /// Steel-and-rubber drive: a cart with rigid capsule hubs hinged to the
    /// chassis (velocity motors) and voxel-torus RUBBER tires whose inner
    /// ring is hard-jointed to the hubs. The tires flatten at the contact
    /// patch, grip through friction, and carry the motor torque to the
    /// ground — over a couple of ridges for good measure.
    public static func softwheel(res: Int = 10, drive: Float = 6,
                                 tireMu: Float = 9000) -> PhysicsScene {
        var s = PhysicsScene(name: "softwheel")
        s.settings.iterations = 10
        s.settings.betaLin = 20000
        s.settings.lambdaMax = 5000
        addGround(&s, friction: 0.9)

        let R: Float = 0.30, tube: Float = 0.14
        let axleZ = R + tube + 0.02
        let track: Float = 0.55                  // half wheel spacing (y)
        let base: Float = 0.62                   // half wheel spacing (x)
        let start = F3(-4.0, 0, 0)               // drive across the camera

        let chassis = s.addBody(size: F3(1.7, 0.68, 0.2), density: 350,
                                friction: 0.4,
                                position: start + F3(0, 0, axleZ + 0.16))
        let qWheel = Quat(angle: .pi / 2, axis: F3(1, 0, 0))   // capsule axis -> y

        for sx in [Float(-1), 1] {
            for sy in [Float(-1), 1] {
                let hubC = start + F3(sx * base, sy * track, axleZ)
                let hub = s.addCapsule(length: 0.2, radius: 0.12, density: 2000,
                                       friction: 0.3, position: hubC,
                                       rotation: qWheel)
                // hinge to the chassis, velocity motor on the axle
                s.addJoint(SceneJoint(bodyA: chassis, bodyB: hub,
                                      rA: F3(sx * base, sy * track, -0.16),
                                      rB: .zero,
                                      stiffnessLin: .infinity,
                                      stiffnessAng: .infinity,
                                      hingeAxis: F3(0, 0, 1),
                                      motorTarget: -drive,
                                      motorTorque: 300,
                                      motorDamping: 50,
                                      motorMode: .velocity))

                // rubber tire: voxelized torus around the hub, axis along y
                let h = 2 * (R + tube) / Float(max(8, res))
                let halfX = R + tube + h, halfY = tube + h
                let nodes = addSoftVoxelShape(
                    &s, origin: hubC - F3(halfX, halfY, halfX), spacing: h,
                    nx: Int(2 * halfX / h) + 1, ny: Int(2 * halfY / h) + 1,
                    nz: Int(2 * halfX / h) + 1,
                    mu: tireMu, lambda: 10 * tireMu,
                    massPerNode: 150 * h * h * h, friction: 1.0) { p in
                        let q = p - hubC
                        let rr = sqrt(q.x * q.x + q.z * q.z)
                        return (rr - R) * (rr - R) + q.y * q.y <= tube * tube
                    }
                // weld the inner ring to the hub: the joint anchors are
                // hub-LOCAL, so they spin with it — that IS the drive path
                for n in nodes {
                    let q = s.bodies[n].position - hubC
                    let rr = sqrt(q.x * q.x + q.z * q.z)
                    if rr < R - 0.3 * tube {
                        s.addJoint(SceneJoint(bodyA: hub, bodyB: n,
                                              rA: qWheel.inverse.act(q),
                                              rB: .zero))
                    }
                }
            }
        }

        // ridges to climb (static, lying across the path)
        for x in [Float(1.0), 3.0] {
            _ = s.addBody(size: F3(0.16, 2.4, 0.16), density: 0, friction: 0.8,
                          position: F3(x, 0, 0),
                          rotation: Quat(angle: .pi / 4, axis: F3(0, 1, 0)))
        }
        s.settings.cameraDistance = 9
        s.settings.cameraTargetZ = 0.7
        return s
    }

    /// OGC stress test (the paper's twisting-cloth regime): a hanging
    /// sheet clamped to a spinning bar at the top and a free-hanging
    /// weight bar at the bottom. The twist propagates down, the sheet
    /// winds into a tight multi-layered rope, and the weight bar rises as
    /// the rope shortens — layers press and slide without pass-through
    /// (conservative bounds + log barrier).
    public static func twist(res: Int = 40, turnRate: Float = 0.3,
                             membraneMu: Float = 300,
                             bend: Float = 5e-4) -> PhysicsScene {
        var s = PhysicsScene(name: "twist")
        s.settings.iterations = 20
        s.settings.betaLin = 20000
        s.settings.particleDamping = 1.5
        s.settings.clothViscosity = 0.25
        addGround(&s, friction: 0.5)

        let L: Float = 2.2                        // hanging length
        let nu = max(30, res), nv = max(16, res / 2)
        let spacing = L / Float(nu - 1)
        let W = Float(nv - 1) * spacing
        let r: Float = min(0.03, 0.34 * spacing)
        let topZ: Float = 3.0

        // spinner bar (static, kinematic rotation about the vertical axis)
        let barA = s.addBody(size: F3(W + 0.3, 0.12, 0.12), density: 0,
                             friction: 0.4, position: F3(0, 0, topZ))
        s.addSpinner(SceneSpinner(body: barA, axis: F3(0, 0, 1),
                                  omega: turnRate * 2 * .pi))
        // weight bar (dynamic): hangs from the sheet, rises as it winds
        let barB = s.addBody(size: F3(W + 0.3, 0.12, 0.12), density: 1.2,
                             friction: 0.4,
                             position: F3(0, 0, topZ - L - 0.1))

        var positions: [[F3]] = []
        for i in 0..<nu {                          // i: down the sheet
            var row: [F3] = []
            for j in 0..<nv {
                row.append(F3(Float(j) * spacing - W / 2, 0,
                              topZ - 0.04 - Float(i) * spacing))
            }
            positions.append(row)
        }
        let grid = addClothGrid(&s, positions: positions, thickness: r,
                                massPerNode: 0.5 * spacing * spacing,
                                friction: 0.5,
                                membraneMu: membraneMu, membraneBend: bend)
        for j in 0..<nv {
            for (end, bar) in [(0, barA), (nu - 1, barB)] {
                let node = grid[end][j]
                s.addJoint(SceneJoint(bodyA: bar, bodyB: node,
                                      rA: s.bodies[node].position
                                          - s.bodies[bar].position,
                                      rB: .zero))
            }
        }
        s.settings.cameraDistance = 5.5
        s.settings.cameraTargetZ = topZ * 0.6
        return s
    }

    /// All three together: rigid bed frame, soft mattress, two softer
    /// pillows, and a cloth blanket draped over — then a ball bounces in.
    public static func bed(res: Int = 36, friction: Float = 0.7,
                           mattressMu: Float = 2500, pillowMu: Float = 600,
                           membraneMu: Float = 300,
                           bend: Float = 5e-4) -> PhysicsScene {
        var s = PhysicsScene(name: "bed")
        s.settings.iterations = 10
        s.settings.betaLin = 20000
        s.settings.particleDamping = 1.5
        s.settings.clothViscosity = 0.25
        s.settings.lambdaMax = 5000
        addGround(&s, friction: friction)

        // ---- frame (static rigid construction) ----
        let baseTop: Float = 0.34
        _ = s.addBody(size: F3(2.14, 1.64, 0.1), density: 0, friction: friction,
                      position: F3(0, 0, baseTop - 0.05))            // slat base
        for sy in [Float(-1), 1] {                                    // side rails
            _ = s.addBody(size: F3(2.3, 0.07, 0.2), density: 0, friction: friction,
                          position: F3(0, sy * 0.86, baseTop + 0.0))
        }
        _ = s.addBody(size: F3(0.08, 1.8, 0.62), density: 0, friction: friction,
                      position: F3(-1.19, 0, 0.31))                  // headboard
        _ = s.addBody(size: F3(0.08, 1.8, 0.34), density: 0, friction: friction,
                      position: F3(1.19, 0, 0.17))                   // footboard
        for (sx, sy) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] {       // legs
            _ = s.addBody(size: F3(0.09, 0.09, baseTop),
                          density: 0, friction: friction,
                          position: F3(Float(sx) * 1.06, Float(sy) * 0.8,
                                       baseTop / 2 - 0.05))
        }

        // ---- mattress (soft block) ----
        let mh: Float = 0.095
        let mattressTop = baseTop + 0.28
        _ = addSoftBlock(&s, center: F3(0, 0, baseTop + 0.14 + 0.01),
                         nx: 22, ny: 17, nz: 4, spacing: mh,
                         mu: mattressMu, lambda: 10 * mattressMu,
                         massPerNode: 25 * mh * mh * mh * 8,
                         friction: friction)

        // ---- pillows (softer, lighter) ----
        let ph: Float = 0.06
        for sy in [Float(-1), 1] {
            _ = addSoftBlock(&s, center: F3(-0.78, sy * 0.38, mattressTop + 0.1),
                             nx: 9, ny: 7, nz: 3, spacing: ph,
                             mu: pillowMu, lambda: 10 * pillowMu,
                             massPerNode: 20 * ph * ph * ph * 8,
                             friction: friction)
        }

        // ---- blanket (cloth) draped over the lower two thirds ----
        let n = max(48, res)
        let sizeX: Float = 1.7, sizeY: Float = 2.0
        let spacing = max(sizeX, sizeY) / Float(n - 1)
        let nx = Int(sizeX / spacing), ny = n
        let r: Float = min(0.04, 0.3 * spacing)
        var positions: [[F3]] = []
        for i in 0..<nx {
            var row: [F3] = []
            for j in 0..<ny {
                row.append(F3(Float(i) * spacing - sizeX / 2 + 0.3,
                              Float(j) * spacing - sizeY / 2,
                              mattressTop + 0.32))
            }
            positions.append(row)
        }
        addClothGrid(&s, positions: positions, thickness: r,
                     massPerNode: 0.008, friction: friction,
                     membraneMu: membraneMu, membraneBend: bend)

        _ = s.addSphere(diameter: 0.42, density: 1.8, friction: friction,
                        position: F3(0.35, 0, mattressTop + 1.0))

        s.settings.cameraDistance = 7.5
        s.settings.cameraTargetZ = 0.8
        return s
    }
}
