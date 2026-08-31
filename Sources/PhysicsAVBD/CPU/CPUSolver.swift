import Foundation
import simd
import SimCore

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

    @_disfavoredOverload
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

    /// Source-compatible initializer for the original single-friction body
    /// model. New material and inertia fields retain their legacy defaults.
    public convenience init(index: Int, size: F3, density: Float,
                            friction: Float, position: F3,
                            rotation: Quat = Quat(real: 1, imag: .zero),
                            velocity: F3 = .zero,
                            shape: BodyShape = .box) {
        self.init(index: index, size: size, density: density,
                  friction: friction, dynamicFriction: friction,
                  position: position, rotation: rotation, velocity: velocity,
                  shape: shape, mass: nil, diagonalInertia: nil,
                  gravityScale: 1)
    }

    public func constrainedTo(_ other: CPURigid) -> Bool {
        forces.contains { f in
            if f is CPUManifold { return false }
            return (f.bodyA === self && f.bodyB === other)
                || (f.bodyA === other && f.bodyB === self)
        }
    }
}

/// CPU-side collision instance. Geometry and material belong to the collider;
/// inertia and motion remain on the owning `CPURigid`.
struct CPUCollider {
    let index: Int
    let body: Int
    var size: F3
    var friction: Float
    var dynamicFriction: Float
    var localPosition: F3
    var localRotation: Quat
    var shape: BodyShape
    var convexHullVertices: [F3]
    var convexAssetID: Int?
    var collisionGroup: UInt32
    var collidesWithSharedGeometry: Bool
    var collisionEnabled: Bool
    var usesWorldSpaceRoundAnchor: Bool
    var isLegacyImplicit: Bool
    var localBoundsCenter: F3
    var boundingRadius: Float

    var hasConvexHull: Bool {
        convexAssetID != nil || !convexHullVertices.isEmpty
    }

    func worldPose(_ owner: CPURigid) -> (position: F3, orientation: Quat) {
        let orientation = (owner.positionAng * localRotation).normalized
        let position = owner.positionLin + owner.positionAng.act(localPosition)
        return (position, orientation)
    }

    func worldBoundsCenter(_ owner: CPURigid) -> F3 {
        let pose = worldPose(owner)
        return pose.position + pose.orientation.act(localBoundsCenter)
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

    /// Returns `false` when the force should be removed from the solver. A
    /// numerical collision-query failure is propagated explicitly so the
    /// solver can roll the entire frame back before latching it.
    func initialize() -> Result<Bool, CPUSolver.RuntimeFailure> { .success(true) }
    func updatePrimal(_ body: CPURigid, _ alpha: Float,
                      _ lhsLin: inout Mat3Rows, _ lhsAng: inout Mat3Rows, _ lhsCross: inout Mat3Rows,
                      _ rhsLin: inout F3, _ rhsAng: inout F3) {}
    func updateDual(_ alpha: Float) {}
}

public final class CPUSolver {
    public enum RuntimeFailure: Error, Equatable, LocalizedError {
        case convexQuery(
            colliderA: Int, colliderB: Int,
            failure: ConvexNarrowPhase.Failure)

        public var errorDescription: String? {
            switch self {
            case .convexQuery(let colliderA, let colliderB, let failure):
                return "CPU convex query failed for colliders \(colliderA) and "
                    + "\(colliderB): \(failure)"
            }
        }
    }

    public var dt: Float = 1.0 / 60.0
    public var gravity: Float = -10.0
    public var iterations: Int = 10
    public var collisionMargin: Float = AVBDConstants.collisionMargin
    /// A numerical narrow-phase failure is terminal. The legacy nonthrowing
    /// `step()` API remains source-compatible and becomes a no-op after the
    /// first failure; callers that need immediate propagation use
    /// `stepChecked()`.
    public private(set) var runtimeFailure: RuntimeFailure?
    var convexQueryFailureForTesting: ConvexNarrowPhase.Failure?
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
    private(set) var colliders: [CPUCollider] = []
    var convexAssets: [ConvexHullAsset] = []
    var collisionExclusions: Set<UInt64> = []
    private var colliderManifolds: [UInt64: CPUManifold] = [:]

    public init() {}

    @discardableResult
    @_disfavoredOverload
    public func addBody(size: F3, density: Float, friction: Float,
                        dynamicFriction: Float? = nil, position: F3,
                        rotation: Quat = Quat(real: 1, imag: .zero), velocity: F3 = .zero,
                        shape: BodyShape = .box, mass: Float? = nil,
                        diagonalInertia: F3? = nil,
                        gravityScale: Float = 1) -> CPURigid {
        addBodyImpl(
            size: size, density: density, friction: friction,
            dynamicFriction: dynamicFriction, position: position,
            rotation: rotation, velocity: velocity, shape: shape,
            mass: mass, diagonalInertia: diagonalInertia,
            gravityScale: gravityScale, addImplicitCollider: true)
    }

    /// Scene conversion supplies authored colliders separately, so it must not
    /// synthesize the legacy one-collider-per-body fallback.
    @discardableResult
    func addSceneBody(size: F3, density: Float, friction: Float,
                      dynamicFriction: Float? = nil, position: F3,
                      rotation: Quat = Quat(real: 1, imag: .zero),
                      velocity: F3 = .zero, shape: BodyShape = .box,
                      mass: Float? = nil, diagonalInertia: F3? = nil,
                      gravityScale: Float = 1) -> CPURigid {
        addBodyImpl(
            size: size, density: density, friction: friction,
            dynamicFriction: dynamicFriction, position: position,
            rotation: rotation, velocity: velocity, shape: shape,
            mass: mass, diagonalInertia: diagonalInertia,
            gravityScale: gravityScale, addImplicitCollider: false)
    }

    private func addBodyImpl(
        size: F3, density: Float, friction: Float,
        dynamicFriction: Float?, position: F3, rotation: Quat,
        velocity: F3, shape: BodyShape, mass: Float?,
        diagonalInertia: F3?, gravityScale: Float,
        addImplicitCollider: Bool
    ) -> CPURigid {
        let b = CPURigid(index: bodies.count, size: size, density: density,
                         friction: friction,
                         dynamicFriction: dynamicFriction,
                         position: position, rotation: rotation, velocity: velocity,
                         shape: shape, mass: mass,
                         diagonalInertia: diagonalInertia,
                         gravityScale: gravityScale)
        bodies.append(b)
        if addImplicitCollider {
            _ = addCollider(
                body: b.index, size: size, friction: friction,
                dynamicFriction: dynamicFriction ?? friction,
                shape: shape,
                usesWorldSpaceRoundAnchor: shape != .box,
                isLegacyImplicit: true)
        }
        return b
    }

    /// Source-compatible body insertion using the original material model.
    @discardableResult
    public func addBody(size: F3, density: Float, friction: Float,
                        position: F3,
                        rotation: Quat = Quat(real: 1, imag: .zero),
                        velocity: F3 = .zero,
                        shape: BodyShape = .box) -> CPURigid {
        addBody(size: size, density: density, friction: friction,
                dynamicFriction: friction, position: position,
                rotation: rotation, velocity: velocity, shape: shape,
                mass: nil, diagonalInertia: nil, gravityScale: 1)
    }

    @discardableResult
    func addCollider(
        body: Int, size: F3, friction: Float,
        dynamicFriction: Float,
        localPosition: F3 = .zero,
        localRotation: Quat = Quat(real: 1, imag: .zero),
        shape: BodyShape = .box,
        convexHullVertices: [F3] = [],
        convexAssetID: Int? = nil,
        collisionGroup: UInt32 = 0,
        collidesWithSharedGeometry: Bool = true,
        collisionEnabled: Bool = true,
        usesWorldSpaceRoundAnchor: Bool = false,
        isLegacyImplicit: Bool = false
    ) -> Int {
        precondition(bodies.indices.contains(body), "CPU collider owner out of range")
        precondition(convexAssetID == nil || convexAssets.indices.contains(convexAssetID!),
                     "CPU convex asset out of range")
        precondition(convexAssetID == nil || convexHullVertices.isEmpty,
                     "CPU collider cannot use inline and shared hulls together")

        let boundsCenter: F3
        let radius: Float
        if let convexAssetID {
            let asset = convexAssets[convexAssetID]
            let geometryBounds = asset.geometryBounds()
            boundsCenter = geometryBounds.center
            radius = geometryBounds.radius
        } else if let first = convexHullVertices.first {
            var lower = first
            var upper = first
            for vertex in convexHullVertices.dropFirst() {
                lower = simd_min(lower, vertex)
                upper = simd_max(upper, vertex)
            }
            boundsCenter = (lower + upper) * 0.5
            radius = convexHullVertices.reduce(Float.zero) {
                max($0, length($1 - boundsCenter))
            }
        } else {
            boundsCenter = .zero
            switch shape {
            case .box: radius = length(size * 0.5)
            case .sphere: radius = size.x * 0.5
            case .torus: radius = size.x + size.y
            case .capsule: radius = size.x * 0.5 + size.y
            }
        }

        let index = colliders.count
        colliders.append(CPUCollider(
            index: index, body: body, size: size,
            friction: friction, dynamicFriction: dynamicFriction,
            localPosition: localPosition,
            localRotation: localRotation.normalized,
            shape: shape, convexHullVertices: convexHullVertices,
            convexAssetID: convexAssetID,
            collisionGroup: collisionGroup,
            collidesWithSharedGeometry: collidesWithSharedGeometry,
            collisionEnabled: collisionEnabled,
            usesWorldSpaceRoundAnchor: usesWorldSpaceRoundAnchor,
            isLegacyImplicit: isLegacyImplicit,
            localBoundsCenter: boundsCenter,
            boundingRadius: radius))
        return index
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

    static func bodyPairKey(_ a: Int, _ b: Int) -> UInt64 {
        let lower = UInt64(min(a, b))
        let upper = UInt64(max(a, b))
        return lower << 32 | upper
    }

    private static func colliderPairKey(_ a: Int, _ b: Int) -> UInt64 {
        let lower = UInt64(min(a, b))
        let upper = UInt64(max(a, b))
        return lower << 32 | upper
    }

    private static func collisionDomainsCompatible(
        _ a: CPUCollider, _ b: CPUCollider
    ) -> Bool {
        if a.collisionGroup != 0 && b.collisionGroup != 0
            && a.collisionGroup != b.collisionGroup { return false }
        if a.collisionGroup == 0 && b.collisionGroup != 0 {
            return b.collidesWithSharedGeometry
        }
        if b.collisionGroup == 0 && a.collisionGroup != 0 {
            return a.collidesWithSharedGeometry
        }
        return true
    }

    /// Throwing companion to the legacy `step()` entry point.
    public func stepChecked() throws {
        step()
        if let runtimeFailure { throw runtimeFailure }
    }

    public func step() {
        guard runtimeFailure == nil else { return }
        // Collision queries run after CPU-authored kinematic spinners move.
        // Preserve the pre-step pose so a terminal numeric query failure can
        // restore the whole frame before any dynamic prediction is published.
        let poseSnapshot = bodies.map { ($0.positionLin, $0.positionAng) }
        for sp in spinners {
            let body = bodies[sp.body]
            let dq = Quat(angle: sp.omega * dt, axis: sp.axis)
            body.positionAng = (dq * body.positionAng).normalized
        }

        // The original direct CPUSolver API stored collision geometry on the
        // public mutable body. Keep its synthesized collider synchronized so
        // introducing authored compound colliders does not change that legacy
        // behavior. Scene-authored colliders remain immutable snapshots.
        for index in colliders.indices where colliders[index].isLegacyImplicit {
            let body = bodies[colliders[index].body]
            colliders[index].size = body.size
            colliders[index].shape = body.shape
            colliders[index].friction = body.friction
            colliders[index].dynamicFriction = body.dynamicFriction
            colliders[index].usesWorldSpaceRoundAnchor = body.shape != .box
            colliders[index].localBoundsCenter = .zero
            switch body.shape {
            case .box:
                colliders[index].boundingRadius = length(body.size * 0.5)
            case .sphere:
                colliders[index].boundingRadius = body.size.x * 0.5
            case .torus:
                colliders[index].boundingRadius = body.size.x + body.size.y
            case .capsule:
                colliders[index].boundingRadius = body.size.x * 0.5 + body.size.y
            }
        }

        // Collider broadphase: O(n^2) bounding-sphere check. Keeping this
        // reference path collider-based is essential for compound bodies: one
        // body pair may own several independent contact manifolds.
        var candidateColliderPairs = Set<UInt64>()
        if colliders.count >= 2 {
            for i in 0..<(colliders.count - 1) {
                let colliderA = colliders[i]
                guard colliderA.collisionEnabled else { continue }
                for j in (i + 1)..<colliders.count {
                    let colliderB = colliders[j]
                    guard colliderB.collisionEnabled,
                          colliderA.body != colliderB.body,
                          Self.collisionDomainsCompatible(colliderA, colliderB) else { continue }
                    let bodyA = bodies[colliderA.body]
                    let bodyB = bodies[colliderB.body]
                    guard bodyA.mass > 0 || bodyB.mass > 0,
                          !bodyA.constrainedTo(bodyB),
                          !collisionExclusions.contains(
                            Self.bodyPairKey(bodyA.index, bodyB.index)) else { continue }

                    let delta = colliderA.worldBoundsCenter(bodyA)
                        - colliderB.worldBoundsCenter(bodyB)
                    let radius = colliderA.boundingRadius + colliderB.boundingRadius
                        + collisionMargin
                    guard dot(delta, delta) <= radius * radius else { continue }

                    let key = Self.colliderPairKey(i, j)
                    candidateColliderPairs.insert(key)
                    if colliderManifolds[key] == nil {
                        colliderManifolds[key] = CPUManifold(
                            solver: self, colliderA: i, colliderB: j,
                            pairKey: key)
                    }
                }
            }
        }

        let stalePairs = colliderManifolds.keys.filter {
            !candidateColliderPairs.contains($0)
        }
        for key in stalePairs {
            if let manifold = colliderManifolds.removeValue(forKey: key) {
                remove(force: manifold)
            }
        }

        // Initialize & warmstart forces; drop inactive ones
        var keep: [CPUForce] = []
        keep.reserveCapacity(forces.count)
        forceInitialization: for force in forces {
            let active: Bool
            switch force.initialize() {
            case .success(let value):
                active = value
            case .failure(let failure):
                runtimeFailure = runtimeFailure ?? failure
                for (body, pose) in zip(bodies, poseSnapshot) {
                    body.positionLin = pose.0
                    body.positionAng = pose.1
                }
                break forceInitialization
            }
            if active {
                keep.append(force)
            } else {
                if let manifold = force as? CPUManifold,
                   let key = manifold.colliderPairKey {
                    colliderManifolds.removeValue(forKey: key)
                }
                force.bodyA?.forces.removeAll { $0 === force }
                force.bodyB?.forces.removeAll { $0 === force }
            }
        }

        guard runtimeFailure == nil else { return }
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
