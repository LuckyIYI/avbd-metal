import simd

// CPU reference implementation of Augmented Vertex Block Descent (AVBD).
// Faithful port of Chris Giles' avbd-demo3d, used as ground truth for the
// Metal GPU solver. Paper: Giles, Diaz, Yuksel, "Augmented Vertex Block
// Descent", SIGGRAPH 2025.

public enum AVBDConstants {
    public static let penaltyMin: Float = 1.0
    public static let penaltyMax: Float = 1.0e10
    public static let penaltyMaxTangent: Float = 1.0e6
    public static let collisionMargin: Float = 0.01
    public static let stickThresh: Float = 0.00001
}

public final class CPURigid {
    public var positionLin: F3
    public var positionAng: Quat
    public var initialLin: F3 = .zero
    public var initialAng: Quat = Quat(real: 1, imag: .zero)
    public var inertialLin: F3 = .zero
    public var inertialAng: Quat = Quat(real: 1, imag: .zero)
    public var velocityLin: F3
    public var velocityAng: F3 = .zero
    public var prevVelocityLin: F3
    public var size: F3        // full extents
    public var mass: Float
    public var moment: F3
    public var friction: Float
    public var dynamicFriction: Float
    public var gravityScale: Float
    public var radius: Float
    public var shape: BodyShape
    public var isParticle = false
    public var forces: [CPUForce] = []
    public let index: Int

    public init(index: Int, size: F3, density: Float, friction: Float,
                dynamicFriction: Float? = nil, position: F3,
                rotation: Quat = Quat(real: 1, imag: .zero), velocity: F3 = .zero,
                shape: BodyShape = .box, mass explicitMass: Float? = nil,
                diagonalInertia: F3? = nil, gravityScale: Float = 1) {
        self.index = index
        self.size = size
        self.friction = friction
        self.dynamicFriction = dynamicFriction ?? friction
        self.gravityScale = gravityScale
        self.positionLin = position
        self.positionAng = rotation
        self.velocityLin = velocity
        self.prevVelocityLin = velocity
        self.shape = shape
        switch shape {
        case .box:
            let m = size.x * size.y * size.z * density
            self.mass = density > 0 ? m : 0
            self.moment = F3(
                (size.y * size.y + size.z * size.z) / 12 * m,
                (size.x * size.x + size.z * size.z) / 12 * m,
                (size.x * size.x + size.y * size.y) / 12 * m
            )
            self.radius = length(size * 0.5)
        case .sphere:
            let r = size.x / 2
            let m = density > 0 ? 4.0 / 3.0 * Float.pi * r * r * r * density : 0
            self.mass = m
            self.moment = F3(repeating: 0.4 * m * r * r)
            self.radius = r
        case .torus:
            let R = size.x, r = size.y
            let m = density > 0 ? 2 * Float.pi * Float.pi * R * r * r * density : 0
            self.mass = m
            // solid torus: I_axis = m(R^2 + 3/4 r^2); I_diameter = m(R^2/2 + 5/8 r^2)
            let iDia = m * (R * R / 2 + 5 * r * r / 8)
            let iAxis = m * (R * R + 3 * r * r / 4)
            self.moment = F3(iDia, iDia, iAxis)
            self.radius = R + r
        case .capsule:
            let L = size.x, r = size.y
            let m = density > 0 ? Float.pi * r * r * (L + 4 * r / 3) * density : 0
            self.mass = m
            let iAxis = 0.5 * m * r * r
            let iPerp = m * (L * L / 12 + r * r / 4)
            self.moment = F3(iPerp, iPerp, iAxis)
            self.radius = L / 2 + r
        }
        if let explicitMass, let diagonalInertia {
            self.mass = explicitMass
            self.moment = diagonalInertia
        }
    }

    public func constrainedTo(_ other: CPURigid) -> Bool {
        forces.contains { f in
            (f.bodyA === self && f.bodyB === other) || (f.bodyA === other && f.bodyB === self)
        }
    }
}

public class CPUForce {
    public weak var bodyA: CPURigid?
    public weak var bodyB: CPURigid?
    unowned let solver: CPUSolver

    init(solver: CPUSolver, bodyA: CPURigid?, bodyB: CPURigid?) {
        self.solver = solver
        self.bodyA = bodyA
        self.bodyB = bodyB
        solver.forces.append(self)
        bodyA?.forces.append(self)
        bodyB?.forces.append(self)
    }

    /// Returns false when the force should be removed from the solver.
    func initialize() -> Bool { true }
    func updatePrimal(_ body: CPURigid, _ alpha: Float,
                      _ lhsLin: inout Mat3Rows, _ lhsAng: inout Mat3Rows, _ lhsCross: inout Mat3Rows,
                      _ rhsLin: inout F3, _ rhsAng: inout F3) {}
    func updateDual(_ alpha: Float) {}
}

public final class CPUSolver {
    public var dt: Float = 1.0 / 60.0
    public var gravity: Float = -10.0
    public var iterations: Int = 10
    public var collisionMargin: Float = AVBDConstants.collisionMargin
    public var alpha: Float = 0.99
    public var betaLin: Float = 5000.0
    public var betaAng: Float = 100.0
    public var gamma: Float = 0.999
    public var lambdaMax: Float = 1.0e6
    public var frictionCombineMode: FrictionCombineMode = .geometricMean
    public var rigidLinearDamping: Float = 0
    public var rigidAngularDamping: Float = 0

    public private(set) var bodies: [CPURigid] = []
    public var forces: [CPUForce] = []
    public var spinners: [SceneSpinner] = []

    public init() {}

    @discardableResult
    public func addBody(size: F3, density: Float, friction: Float,
                        dynamicFriction: Float? = nil, position: F3,
                        rotation: Quat = Quat(real: 1, imag: .zero), velocity: F3 = .zero,
                        shape: BodyShape = .box, mass: Float? = nil,
                        diagonalInertia: F3? = nil,
                        gravityScale: Float = 1) -> CPURigid {
        let b = CPURigid(index: bodies.count, size: size, density: density,
                         friction: friction,
                         dynamicFriction: dynamicFriction,
                         position: position, rotation: rotation, velocity: velocity,
                         shape: shape, mass: mass,
                         diagonalInertia: diagonalInertia,
                         gravityScale: gravityScale)
        bodies.append(b)
        return b
    }

    @discardableResult
    public func addJoint(_ bodyA: CPURigid?, _ bodyB: CPURigid, rA: F3, rB: F3,
                         stiffnessLin: Float = .infinity, stiffnessAng: Float = 0,
                         fracture: Float = .infinity) -> CPUJoint {
        CPUJoint(solver: self, bodyA: bodyA, bodyB: bodyB, rA: rA, rB: rB,
                 stiffnessLin: stiffnessLin, stiffnessAng: stiffnessAng, fracture: fracture)
    }

    @discardableResult
    public func addSpring(_ bodyA: CPURigid, _ bodyB: CPURigid, rA: F3, rB: F3,
                          stiffness: Float, rest: Float = -1) -> CPUSpring {
        CPUSpring(solver: self, bodyA: bodyA, bodyB: bodyB, rA: rA, rB: rB,
                  stiffness: stiffness, rest: rest)
    }

    func remove(force: CPUForce) {
        forces.removeAll { $0 === force }
        force.bodyA?.forces.removeAll { $0 === force }
        force.bodyB?.forces.removeAll { $0 === force }
    }

    public func step() {
        for sp in spinners {
            let body = bodies[sp.body]
            let dq = Quat(angle: sp.omega * dt, axis: sp.axis)
            body.positionAng = (dq * body.positionAng).normalized
        }

        // Broadphase: O(n^2) sphere check (reference path; GPU uses spatial hash)
        for i in 0..<bodies.count {
            for j in (i + 1)..<bodies.count {
                let a = bodies[i], b = bodies[j]
                let dp = a.positionLin - b.positionLin
                let r = a.radius + b.radius
                if dot(dp, dp) <= r * r && !a.constrainedTo(b) {
                    _ = CPUManifold(solver: self, bodyA: a, bodyB: b)
                }
            }
        }

        // Initialize & warmstart forces; drop inactive ones
        var keep: [CPUForce] = []
        keep.reserveCapacity(forces.count)
        for force in forces {
            if force.initialize() {
                keep.append(force)
            } else {
                force.bodyA?.forces.removeAll { $0 === force }
                force.bodyB?.forces.removeAll { $0 === force }
            }
        }
        forces = keep

        // Initialize & warmstart bodies (Eq. 2 + adaptive warmstart)
        for body in bodies {
            body.inertialLin = body.positionLin + body.velocityLin * dt
            if body.mass > 0 {
                body.inertialLin += F3(0, 0, gravity * body.gravityScale)
                    * (dt * dt)
            }
            body.inertialAng = quatAdd(body.positionAng, body.velocityAng * dt)

            let accel = (body.velocityLin - body.prevVelocityLin) / dt
            let accelExt = accel.z * (gravity < 0 ? Float(-1) : (gravity > 0 ? 1 : 0))
            var accelWeight = simd_clamp(accelExt / abs(gravity), 0.0, 1.0)
            if !accelWeight.isFinite { accelWeight = 0 }

            body.initialLin = body.positionLin
            body.initialAng = body.positionAng
            if body.mass > 0 {
                body.positionLin += body.velocityLin * dt
                    + F3(0, 0, gravity * body.gravityScale)
                    * (accelWeight * dt * dt)
                body.positionAng = quatAdd(body.positionAng, body.velocityAng * dt)
            }
        }

        // Main solver loop
        for _ in 0..<iterations {
            // Primal update (Gauss-Seidel over bodies on CPU)
            for body in bodies where body.mass > 0 {
                var lhsLin = Mat3Rows.diagonal(F3(repeating: body.mass / (dt * dt)))
                let inertia = worldInertiaRows(body.positionAng, body.moment)
                    * (1 / (dt * dt))
                var lhsAng = inertia
                var lhsCross = Mat3Rows()

                var rhsLin = (body.positionLin - body.inertialLin) * (body.mass / (dt * dt))
                var rhsAng = inertia.mul(
                    quatSub(body.positionAng, body.inertialAng))

                for force in body.forces {
                    force.updatePrimal(body, alpha, &lhsLin, &lhsAng, &lhsCross, &rhsLin, &rhsAng)
                }

                var dxLin = F3.zero, dxAng = F3.zero
                if body.isParticle {
                    // 3-DOF particle: zero the angular system (3x3 solve)
                    lhsAng = .identity
                    lhsCross = Mat3Rows(.zero, .zero, .zero)
                    rhsAng = .zero
                }
                solve6x6(lhsLin, lhsAng, lhsCross, -rhsLin, -rhsAng, &dxLin, &dxAng)
                body.positionLin += dxLin
                body.positionAng = quatAdd(body.positionAng, dxAng)
            }

            // Dual update
            for force in forces {
                force.updateDual(alpha)
            }
        }

        // BDF1 velocity update
        for body in bodies {
            body.prevVelocityLin = body.velocityLin
            if body.mass > 0 {
                body.velocityLin = (body.positionLin - body.initialLin) / dt
                body.velocityAng = quatSub(body.positionAng, body.initialAng) / dt
                if !body.isParticle {
                    body.velocityLin *= exp(-rigidLinearDamping * dt)
                    body.velocityAng *= exp(-rigidAngularDamping * dt)
                }
            }
        }
    }

    /// Max constraint violation across all joints and contact normals (for tests).
    public func maxConstraintError() -> Float {
        var err: Float = 0
        for force in forces {
            if let j = force as? CPUJoint {
                if j.stiffnessLin == 0 && j.stiffnessAng == 0 { continue }
                err = max(err, length(j.currentCLin()))
            } else if let m = force as? CPUManifold {
                for c in 0..<m.numContacts {
                    err = max(err, max(0, -m.currentPenetration(c)))
                }
            }
        }
        return err
    }
}
