import SimCore

public extension PhysicsScene {
    /// Build the AVBD CPU reference backend from this neutral scene.
    func makeCPUSolver() -> CPUSolver {
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

        for body in bodies {
            let rigidBody = solver.addBody(
                size: body.size,
                density: body.density,
                friction: body.friction,
                dynamicFriction: body.dynamicFriction,
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
