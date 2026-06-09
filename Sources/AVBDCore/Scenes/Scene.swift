import simd

// Backend-agnostic scene description, consumed by both the CPU reference
// solver and the Metal GPU solver (and by tests for parity checks).

public struct SceneBody {
    public var size: F3
    public var density: Float       // 0 = static
    public var friction: Float
    public var position: F3
    public var rotation: Quat
    public var velocity: F3

    public init(size: F3, density: Float, friction: Float, position: F3,
                rotation: Quat = Quat(real: 1, imag: .zero), velocity: F3 = .zero) {
        self.size = size
        self.density = density
        self.friction = friction
        self.position = position
        self.rotation = rotation
        self.velocity = velocity
    }
}

public struct SceneJoint {
    public var bodyA: Int           // -1 = world
    public var bodyB: Int
    public var rA: F3
    public var rB: F3
    public var stiffnessLin: Float
    public var stiffnessAng: Float
    public var fracture: Float

    public init(bodyA: Int, bodyB: Int, rA: F3, rB: F3,
                stiffnessLin: Float = .infinity, stiffnessAng: Float = 0,
                fracture: Float = .infinity) {
        self.bodyA = bodyA
        self.bodyB = bodyB
        self.rA = rA
        self.rB = rB
        self.stiffnessLin = stiffnessLin
        self.stiffnessAng = stiffnessAng
        self.fracture = fracture
    }
}

public struct SceneSpring {
    public var bodyA: Int
    public var bodyB: Int
    public var rA: F3
    public var rB: F3
    public var stiffness: Float
    public var rest: Float          // < 0: use current distance

    public init(bodyA: Int, bodyB: Int, rA: F3, rB: F3, stiffness: Float, rest: Float = -1) {
        self.bodyA = bodyA
        self.bodyB = bodyB
        self.rA = rA
        self.rB = rB
        self.stiffness = stiffness
        self.rest = rest
    }
}

public struct SimSettings {
    public var dt: Float = 1.0 / 60.0
    public var gravity: Float = -10.0
    public var iterations: Int = 10
    public var alpha: Float = 0.99
    public var betaLin: Float = 10000.0
    public var betaAng: Float = 100.0
    public var gamma: Float = 0.999

    public init() {}
}

public struct PhysicsScene {
    public var name: String
    public var bodies: [SceneBody] = []
    public var joints: [SceneJoint] = []
    public var springs: [SceneSpring] = []
    public var settings = SimSettings()

    public init(name: String) {
        self.name = name
    }

    @discardableResult
    public mutating func addBody(size: F3, density: Float, friction: Float, position: F3,
                                 rotation: Quat = Quat(real: 1, imag: .zero),
                                 velocity: F3 = .zero) -> Int {
        bodies.append(SceneBody(size: size, density: density, friction: friction,
                                position: position, rotation: rotation, velocity: velocity))
        return bodies.count - 1
    }

    public mutating func addJoint(_ j: SceneJoint) { joints.append(j) }

    /// Adds an inert joint slot for interactive dragging (stiffness 0 keeps
    /// it disabled until GPUSolver.setDrag activates it). Returns its index.
    @discardableResult
    public mutating func addDragSlot() -> Int {
        joints.append(SceneJoint(bodyA: -1, bodyB: 0, rA: .zero, rB: .zero,
                                 stiffnessLin: 0, stiffnessAng: 0))
        return joints.count - 1
    }
    public mutating func addSpring(_ s: SceneSpring) { springs.append(s) }

    /// Build a CPU reference solver from this scene.
    public func makeCPUSolver() -> CPUSolver {
        let solver = CPUSolver()
        solver.dt = settings.dt
        solver.gravity = settings.gravity
        solver.iterations = settings.iterations
        solver.alpha = settings.alpha
        solver.betaLin = settings.betaLin
        solver.betaAng = settings.betaAng
        solver.gamma = settings.gamma

        for b in bodies {
            solver.addBody(size: b.size, density: b.density, friction: b.friction,
                           position: b.position, rotation: b.rotation, velocity: b.velocity)
        }
        for j in joints {
            solver.addJoint(j.bodyA >= 0 ? solver.bodies[j.bodyA] : nil, solver.bodies[j.bodyB],
                            rA: j.rA, rB: j.rB,
                            stiffnessLin: j.stiffnessLin, stiffnessAng: j.stiffnessAng,
                            fracture: j.fracture)
        }
        for s in springs {
            solver.addSpring(solver.bodies[s.bodyA], solver.bodies[s.bodyB],
                             rA: s.rA, rB: s.rB, stiffness: s.stiffness, rest: s.rest)
        }
        return solver
    }
}
