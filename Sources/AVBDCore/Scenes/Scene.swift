import simd

// Backend-agnostic scene description, consumed by both the CPU reference
// solver and the Metal GPU solver (and by tests for parity checks).

public enum BodyShape: Equatable {
    case box
    case sphere     // size.x = diameter (size.y/z ignored)
    case torus      // size.x = major (spine) radius, size.y = minor (tube) radius
    case capsule    // size.x = cylinder length (along local z), size.y = radius
}

public struct SceneBody {
    public var size: F3
    public var density: Float       // 0 = static
    public var friction: Float
    public var position: F3
    public var rotation: Quat
    public var velocity: F3
    public var shape: BodyShape

    public init(size: F3, density: Float, friction: Float, position: F3,
                rotation: Quat = Quat(real: 1, imag: .zero), velocity: F3 = .zero,
                shape: BodyShape = .box) {
        switch shape {
        case .sphere: self.size = F3(repeating: size.x)
        default: self.size = size
        }
        self.density = density
        self.friction = friction
        self.position = position
        self.rotation = rotation
        self.velocity = velocity
        self.shape = shape
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

/// Kinematic rotation applied to a static body every step (e.g. rollers,
/// paddle wheels). The body stays mass-0 for the solver; only its pose is
/// advanced, and contacts react to the new pose mechanically.
public struct SceneSpinner {
    public var body: Int
    public var axis: F3
    public var omega: Float   // rad/s

    public init(body: Int, axis: F3, omega: Float) {
        self.body = body
        self.axis = normalize(axis)
        self.omega = omega
    }
}

public struct SimSettings {
    public var dt: Float = 1.0 / 60.0
    public var gravity: Float = -10.0
    public var iterations: Int = 10
    public var alpha: Float = 0.99
    public var betaLin: Float = 5000.0
    public var betaAng: Float = 100.0
    public var gamma: Float = 0.999
    /// Dual variable bound (paper Sec. 4): prevents unbounded force
    /// accumulation when conflicting contacts cannot all be satisfied
    /// (e.g. wedged chainmail links). Large = effectively off.
    public var lambdaMax: Float = 1.0e6

    public init() {}
}

public struct PhysicsScene {
    public var name: String
    public var bodies: [SceneBody] = []
    public var joints: [SceneJoint] = []
    public var springs: [SceneSpring] = []
    public var spinners: [SceneSpinner] = []
    public var settings = SimSettings()

    public init(name: String) {
        self.name = name
    }

    @discardableResult
    public mutating func addBody(size: F3, density: Float, friction: Float, position: F3,
                                 rotation: Quat = Quat(real: 1, imag: .zero),
                                 velocity: F3 = .zero, shape: BodyShape = .box) -> Int {
        bodies.append(SceneBody(size: size, density: density, friction: friction,
                                position: position, rotation: rotation, velocity: velocity,
                                shape: shape))
        return bodies.count - 1
    }

    @discardableResult
    public mutating func addSphere(diameter: Float, density: Float, friction: Float,
                                   position: F3, velocity: F3 = .zero) -> Int {
        addBody(size: F3(repeating: diameter), density: density, friction: friction,
                position: position, velocity: velocity, shape: .sphere)
    }

    @discardableResult
    public mutating func addCapsule(length: Float, radius: Float, density: Float,
                                    friction: Float, position: F3,
                                    rotation: Quat = Quat(real: 1, imag: .zero),
                                    velocity: F3 = .zero) -> Int {
        addBody(size: F3(length, radius, 0), density: density, friction: friction,
                position: position, rotation: rotation, velocity: velocity, shape: .capsule)
    }

    @discardableResult
    public mutating func addTorus(major: Float, minor: Float, density: Float,
                                  friction: Float, position: F3,
                                  rotation: Quat = Quat(real: 1, imag: .zero),
                                  velocity: F3 = .zero) -> Int {
        addBody(size: F3(major, minor, 0), density: density, friction: friction,
                position: position, rotation: rotation, velocity: velocity, shape: .torus)
    }

    public mutating func addJoint(_ j: SceneJoint) { joints.append(j) }
    public mutating func addSpinner(_ sp: SceneSpinner) { spinners.append(sp) }

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
        solver.lambdaMax = settings.lambdaMax

        for b in bodies {
            solver.addBody(size: b.size, density: b.density, friction: b.friction,
                           position: b.position, rotation: b.rotation, velocity: b.velocity,
                           shape: b.shape)
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
        solver.spinners = spinners
        return solver
    }
}
