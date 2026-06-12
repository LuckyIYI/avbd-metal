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
    /// 3-DOF particle (paper: vertices with M = mI, 3x3 blocks). Collides
    /// as a sphere of `size.x/2` (cloth/soft thickness) but carries no
    /// rotational state.
    public var isParticle: Bool = false

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
    /// Include linear lambda in the fracture criterion (tension release,
    /// e.g. sling straps). Default false: linear lambda is penalty-scaled
    /// and spikes transiently on hard welds, so it would pop mortar.
    public var fractureLinear: Bool
    /// Hinge axis in body B's LOCAL frame. When set, the angular constraint
    /// leaves rotation about this axis free (1-DOF revolute joint).
    public var hingeAxis: F3?
    /// Servo motor on the hinge: drive the twist angle toward motorTarget
    /// with |lambda| bounded by motorTorque. 0 torque = no motor.
    public var motorTarget: Float
    public var motorTorque: Float
    /// Continuous drive: the servo target ADVANCES at this rate (rad/s)
    /// every step — a velocity motor built on the bounded-torque servo
    /// (wheels, drums). 0 = positional servo only.
    public var motorRate: Float
    /// Hinge twist limits (radians). lo < hi enables; real arms have them —
    /// without limits a decelerating arm can tumble over the top and wedge.
    public var limitLo: Float
    public var limitHi: Float

    public init(bodyA: Int, bodyB: Int, rA: F3, rB: F3,
                stiffnessLin: Float = .infinity, stiffnessAng: Float = 0,
                fracture: Float = .infinity, fractureLinear: Bool = false,
                hingeAxis: F3? = nil, motorTarget: Float = 0,
                motorTorque: Float = 0, motorRate: Float = 0,
                limitLo: Float = 1, limitHi: Float = -1) {
        self.bodyA = bodyA
        self.bodyB = bodyB
        self.rA = rA
        self.rB = rB
        self.stiffnessLin = stiffnessLin
        self.stiffnessAng = stiffnessAng
        self.fracture = fracture
        self.fractureLinear = fractureLinear
        self.hingeAxis = hingeAxis.map(normalize)
        self.motorTarget = motorTarget
        self.motorTorque = motorTorque
        self.motorRate = motorRate
        self.limitLo = limitLo
        self.limitHi = limitHi
    }
}

/// Cloth surface triangle over three 3-DOF particles. Triangles are the
/// collision surface of thin soft bodies (V-T and E-E contacts, plus rigid
/// features against the face), and — when mu > 0 — StVK membrane elements
/// with quadratic bending across shared edges (bend > 0).
public struct SceneTri {
    public var ids: (Int, Int, Int)
    public var mu: Float        // membrane shear/stretch stiffness (0 = contact only)
    public var lambda: Float    // membrane area-preservation stiffness
    public var bend: Float      // bending stiffness across shared edges

    public init(ids: (Int, Int, Int), mu: Float = 0, lambda: Float = 0,
                bend: Float = 0) {
        self.ids = ids
        self.mu = mu
        self.lambda = lambda
        self.bend = bend
    }
}

/// Stable Neo-Hookean tetrahedron over four 3-DOF particles.
public struct SceneTet {
    public var ids: (Int, Int, Int, Int)
    public var mu: Float
    public var lambda: Float

    public init(ids: (Int, Int, Int, Int), mu: Float, lambda: Float) {
        self.ids = ids
        self.mu = mu
        self.lambda = lambda
    }
}

public struct SceneSpring {
    public var bodyA: Int
    public var bodyB: Int
    public var rA: F3
    public var rB: F3
    public var stiffness: Float
    public var rest: Float          // < 0: use current distance
    /// Hard rod: inextensible distance element handled with the augmented
    /// Lagrangian dual machinery (paper's hard constraints).
    public var hard: Bool

    public init(bodyA: Int, bodyB: Int, rA: F3, rB: F3, stiffness: Float,
                rest: Float = -1, hard: Bool = false) {
        self.bodyA = bodyA
        self.bodyB = bodyB
        self.rA = rA
        self.rB = rB
        self.stiffness = stiffness
        self.rest = rest
        self.hard = hard
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
    // 10000 matches the reference avbd-demo3d default (its in-code note
    // calls the higher, unit-split betas "a minor upgrade from the paper");
    // 5000 was an early-port value and converges visibly slower on stacks.
    public var betaLin: Float = 10000.0
    public var betaAng: Float = 100.0
    public var gamma: Float = 0.999
    /// Dual variable bound (paper Sec. 4): prevents unbounded force
    /// accumulation when conflicting contacts cannot all be satisfied
    /// (e.g. wedged chainmail links). Large = effectively off.
    public var lambdaMax: Float = 1.0e6
    /// Hard-rod dual decay exponent vs per-frame rod rotation (0 = off).
    /// Carried lambda re-applied along rotated rod directions pumps the
    /// pendulum modes of swinging sheets.
    public var rodDecayPow: Float = 0
    /// Per-second velocity damping applied to 3-DOF particles only (air
    /// drag + internal viscosity of thin sheets). Inextensible cloth has
    /// no material compliance to bleed energy through; without a touch of
    /// drag, free skirts and hems flutter forever.
    public var particleDamping: Float = 0
    /// Internal sheet viscosity: per-frame blend of each particle velocity
    /// toward its topological 1-ring average (0..1). Damps relative flutter
    /// (bending-rate viscosity), preserves bulk motion.
    public var clothViscosity: Float = 0
    /// Default camera framing for this scene: orbit distance and target
    /// height (0 = renderer default). Small cloth rigs drown at the
    /// rigid-rig default distance of 30.
    public var cameraDistance: Float = 0
    public var cameraTargetZ: Float = 0
    /// Cloth render thickness as a fraction of the contact radius.
    /// 0 (default) = flat sheets: one layer, no extrusion, no hem rims.
    /// 1 = extrude the full contact skin (layered cloth visually touches).
    /// Tet-boundary surfaces always keep their outward offset.
    public var clothRenderScale: Float = 0

    public init() {}
}

public struct PhysicsScene {
    public var name: String
    public var bodies: [SceneBody] = []
    public var joints: [SceneJoint] = []
    public var springs: [SceneSpring] = []
    public var tets: [SceneTet] = []
    public var tris: [SceneTri] = []
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
    /// Add a 3-DOF particle (soft-body vertex). `mass` is explicit since
    /// particles represent lumped vertex mass, not volume.
    @discardableResult
    public mutating func addParticle(radius: Float, mass: Float, friction: Float = 0.5,
                                     position: F3, velocity: F3 = .zero) -> Int {
        // density chosen so the sphere volume formula reproduces `mass`
        let vol = 4.0 / 3.0 * Float.pi * radius * radius * radius
        var b = SceneBody(size: F3(repeating: radius * 2), density: mass / vol,
                          friction: friction, position: position,
                          velocity: velocity, shape: .sphere)
        b.isParticle = true
        bodies.append(b)
        return bodies.count - 1
    }

    public mutating func addSpring(_ s: SceneSpring) { springs.append(s) }
    public mutating func addTet(_ t: SceneTet) { tets.append(t) }
    public mutating func addTri(_ t: SceneTri) { tris.append(t) }

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
            let rb = solver.addBody(size: b.size, density: b.density, friction: b.friction,
                                    position: b.position, rotation: b.rotation,
                                    velocity: b.velocity, shape: b.shape)
            rb.isParticle = b.isParticle
        }
        for j in joints {
            let cj = solver.addJoint(j.bodyA >= 0 ? solver.bodies[j.bodyA] : nil,
                                     solver.bodies[j.bodyB],
                                     rA: j.rA, rB: j.rB,
                                     stiffnessLin: j.stiffnessLin,
                                     stiffnessAng: j.stiffnessAng,
                                     fracture: j.fracture)
            cj.hingeAxis = j.hingeAxis
            cj.fractureLinear = j.fractureLinear
        }
        for s in springs {
            let sp = solver.addSpring(solver.bodies[s.bodyA], solver.bodies[s.bodyB],
                                      rA: s.rA, rB: s.rB, stiffness: s.stiffness,
                                      rest: s.rest)
            sp.hard = s.hard
        }
        solver.spinners = spinners
        return solver
    }
}
