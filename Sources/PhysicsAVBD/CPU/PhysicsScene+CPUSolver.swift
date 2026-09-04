import simd
import SimCore

public enum CPUConvexCollisionError: Error, Equatable, Sendable,
    CustomStringConvertible
{
    case invalidAssetReference(collider: Int, asset: Int)
    case conflictingAssetSources(collider: Int)
    case unsupportedTorusHull(colliderA: Int, colliderB: Int)

    public var description: String {
        switch self {
        case .invalidAssetReference(let collider, let asset):
            return "CPU convex collider \(collider) references missing asset \(asset)"
        case .conflictingAssetSources(let collider):
            return "CPU convex collider \(collider) supplies both shared and inline geometry"
        case .unsupportedTorusHull(let colliderA, let colliderB):
            return "CPU convex hull collision does not support torus pair "
                + "\(colliderA)/\(colliderB)"
        }
    }
}

public extension PhysicsScene {
    /// Compatibility entry point for trusted, statically authored scenes.
    /// Importers should use `makeCPUSolverChecked()` so unsupported collision
    /// pairs are reported as data errors instead of terminating the process.
    func makeCPUSolver() -> CPUSolver {
        do {
            return try makeCPUSolverChecked()
        } catch {
            preconditionFailure("CPU scene conversion failed: \(error)")
        }
    }

    /// Build the AVBD CPU reference backend from a validated neutral scene.
    ///
    /// Torus is non-convex and intentionally absent from the generic support
    /// map. Potentially colliding torus/hull pairs therefore fail before any
    /// solver state is allocated or mutated.
    func makeCPUSolverChecked() throws -> CPUSolver {
        for (index, collider) in colliders.enumerated() {
            if let asset = collider.convexAssetID {
                guard collider.convexHullVertices.isEmpty else {
                    throw CPUConvexCollisionError.conflictingAssetSources(
                        collider: index)
                }
                guard convexAssets.indices.contains(asset) else {
                    throw CPUConvexCollisionError.invalidAssetReference(
                        collider: index, asset: asset)
                }
            }
        }
        let hullColliders = colliders.indices.filter {
            colliders[$0].collisionEnabled
                && (colliders[$0].convexAssetID != nil
                    || !colliders[$0].convexHullVertices.isEmpty)
        }
        let torusColliders = colliders.indices.filter {
            colliders[$0].collisionEnabled
                && colliders[$0].shape == .torus
                && colliders[$0].convexAssetID == nil
                && colliders[$0].convexHullVertices.isEmpty
        }
        for hull in hullColliders {
            for torus in torusColliders where canPotentiallyCollide(
                colliderA: hull, colliderB: torus)
            {
                throw CPUConvexCollisionError.unsupportedTorusHull(
                    colliderA: hull, colliderB: torus)
            }
        }

        let solver = CPUSolver()
        solver.dt = settings.dt
        solver.gravity = settings.gravity
        solver.iterations = settings.iterations
        solver.alpha = settings.alpha
        solver.betaLin = settings.betaLin
        solver.betaAng = settings.betaAng
        solver.gamma = settings.gamma
        solver.lambdaMax = settings.lambdaMax
        solver.collisionMargin = max(settings.collisionMargin, 0)
        solver.frictionCombineMode = settings.frictionCombineMode
        solver.rigidLinearDamping = settings.rigidLinearDamping
        solver.rigidAngularDamping = settings.rigidAngularDamping
        solver.convexAssets = convexAssets

        for body in bodies {
            let rigidBody = solver.addSceneBody(
                size: body.size,
                density: body.density,
                friction: body.friction,
                dynamicFriction: body.dynamicFriction,
                torsionalFriction: body.torsionalFriction,
                position: body.position,
                rotation: body.rotation,
                velocity: body.velocity,
                shape: body.shape,
                mass: body.mass,
                diagonalInertia: body.diagonalInertia,
                gravityScale: body.gravityScale
            )
            rigidBody.isParticle = body.isParticle
        }
        for collider in colliders {
            _ = solver.addCollider(
                body: collider.body, size: collider.size,
                friction: collider.friction,
                dynamicFriction: collider.dynamicFriction,
                torsionalFriction: collider.torsionalFriction,
                localPosition: collider.localPosition,
                localRotation: collider.localRotation,
                shape: collider.shape,
                convexHullVertices: collider.convexHullVertices,
                convexAssetID: collider.convexAssetID,
                collisionGroup: collider.collisionGroup,
                collidesWithSharedGeometry: collider.collidesWithSharedGeometry,
                collisionEnabled: collider.collisionEnabled,
                usesWorldSpaceRoundAnchor: collider.usesWorldSpaceRoundAnchor)
        }
        solver.collisionExclusions = Set(collisionExclusions.map {
            CPUSolver.bodyPairKey($0.bodyA, $0.bodyB)
        })
        for joint in joints {
            let cpuJoint = solver.addJoint(
                joint.bodyA >= 0 ? solver.bodies[joint.bodyA] : nil,
                solver.bodies[joint.bodyB],
                rA: joint.rA,
                rB: joint.rB,
                stiffnessLin: joint.stiffnessLin,
                stiffnessAng: joint.stiffnessAng,
                fracture: joint.fracture
            )
            cpuJoint.hingeAxis = joint.hingeAxis
            if let axis = joint.prismaticAxis {
                precondition(axis.x.isFinite && axis.y.isFinite && axis.z.isFinite
                    && length_squared(axis) > 1e-12)
                precondition(joint.hingeAxis == nil && joint.motorTorque == 0)
                cpuJoint.prismaticAxis = normalize(axis)
                cpuJoint.translationLimits = joint.translationLimits
            }
            cpuJoint.fractureLinear = joint.fractureLinear
        }
        for spring in springs {
            let cpuSpring = solver.addSpring(
                solver.bodies[spring.bodyA],
                solver.bodies[spring.bodyB],
                rA: spring.rA,
                rB: spring.rB,
                stiffness: spring.stiffness,
                rest: spring.rest
            )
            cpuSpring.hard = spring.hard
        }
        solver.spinners = spinners

        return solver
    }
}
