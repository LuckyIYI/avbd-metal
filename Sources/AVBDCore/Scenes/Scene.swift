import simd

// Backend-agnostic scene description, consumed by both the CPU reference
// solver and the Metal GPU solver (and by tests for parity checks).

public enum BodyShape: Equatable {
    case box
    case sphere     // size.x = diameter (size.y/z ignored)
    case torus      // size.x = major (spine) radius, size.y = minor (tube) radius
    case capsule    // size.x = cylinder length (along local z), size.y = radius
}

/// Contact-material coefficient combination. The legacy AVBD demos use the
/// geometric mean; PhysX scenes can request their authored combine mode
/// without pre-distorting either collider's material values.
public enum FrictionCombineMode: UInt32, Equatable, Sendable {
    case geometricMean = 0
    case multiply = 1
    case minimum = 2
    case maximum = 3
    case average = 4

    public func combine(_ a: Float, _ b: Float) -> Float {
        switch self {
        case .geometricMean: return sqrt(max(a * b, 0))
        case .multiply: return a * b
        case .minimum: return min(a, b)
        case .maximum: return max(a, b)
        case .average: return 0.5 * (a + b)
        }
    }
}

public struct SceneBody {
    public var size: F3
    public var density: Float       // 0 = static
    /// Static friction coefficient. `dynamicFriction` defaults to this value
    /// for legacy scenes that author one Coulomb coefficient.
    public var friction: Float
    public var dynamicFriction: Float
    public var position: F3
    public var rotation: Quat
    public var velocity: F3
    public var shape: BodyShape
    /// Optional authoritative inertial properties in the body's local frame.
    /// Importers use these instead of deriving mass and inertia from the
    /// collision primitive's density. Both values must be supplied together.
    public var mass: Float?
    public var diagonalInertia: F3?
    /// Multiplier applied to world gravity for this body. Articulated robot
    /// controllers commonly balance passive forces by disabling gravity on
    /// their links while manipulated objects retain ordinary gravity.
    public var gravityScale: Float
    /// 3-DOF particle (paper: vertices with M = mI, 3x3 blocks). Collides
    /// as a sphere of `size.x/2` (cloth/soft thickness) but carries no
    /// rotational state.
    public var isParticle: Bool = false

    public var isDynamic: Bool { (mass ?? (density > 0 ? 1 : 0)) > 0 }

    public init(size: F3, density: Float, friction: Float,
                dynamicFriction: Float? = nil, position: F3,
                rotation: Quat = Quat(real: 1, imag: .zero), velocity: F3 = .zero,
                shape: BodyShape = .box, mass: Float? = nil,
                diagonalInertia: F3? = nil, gravityScale: Float = 1) {
        precondition(gravityScale >= 0 && gravityScale.isFinite,
                     "gravity scale must be finite and nonnegative")
        precondition((mass == nil) == (diagonalInertia == nil),
                     "explicit mass and diagonal inertia must be supplied together")
        if let mass, let diagonalInertia {
            precondition(mass >= 0 && diagonalInertia.x >= 0
                && diagonalInertia.y >= 0 && diagonalInertia.z >= 0,
                "inertial properties must be nonnegative")
            precondition(mass == 0 || (diagonalInertia.x > 0
                && diagonalInertia.y > 0 && diagonalInertia.z > 0),
                "dynamic bodies require positive principal inertia")
        }
        switch shape {
        case .sphere: self.size = F3(repeating: size.x)
        default: self.size = size
        }
        self.density = density
        self.friction = friction
        self.dynamicFriction = dynamicFriction ?? friction
        self.position = position
        self.rotation = rotation
        self.velocity = velocity
        self.shape = shape
        self.mass = mass
        self.diagonalInertia = diagonalInertia
        self.gravityScale = gravityScale
    }
}

/// A collision primitive rigidly attached to a `SceneBody`.
///
/// Keeping collision geometry separate from inertial bodies is required by
/// real robot assets: one link can have several deliberately simplified
/// contact shapes, while its mass and principal inertia come from the source
/// MJCF/URDF instead of those shapes. `localPosition` and `localRotation` are
/// expressed in the owning body's frame.
public struct SceneCollider {
    public var body: Int
    public var size: F3
    public var friction: Float
    public var dynamicFriction: Float
    public var localPosition: F3
    public var localRotation: Quat
    public var shape: BodyShape
    /// Optional convex-hull vertices in the collider's centered local frame.
    /// When non-empty they replace the analytic `shape` for collision while
    /// `size` remains the local AABB extent used by broad phase/debug tools.
    /// Keeping hull data on colliders (rather than inertial bodies) permits
    /// exact imported robot contact geometry without changing link mass.
    public var convexHullVertices: [F3]
    /// Collision domain used by batched simulations. Group zero is shared
    /// geometry; nonzero groups collide with shared geometry and themselves,
    /// but never with a different nonzero group.
    public var collisionGroup: UInt32
    /// Whether this analytic primitive participates in broad-phase and
    /// contact generation. Keeping this independent of `isRendered` lets
    /// imported robot models retain their visual proxy geometry while a task
    /// selects a reduced collision model (for example MuJoCo Playground's
    /// feet-only H1 training scene).
    public var collisionEnabled: Bool
    /// Physics collision remains enabled when false; only analytic debug
    /// rendering omits this primitive. Imported robot assets use this for
    /// protective contact geoms that look misleading without the visual mesh.
    public var isRendered: Bool
    /// Preserve the legacy rotation-invariant contact anchor for a centered
    /// standalone round body. Offset compound shapes use material/body-local
    /// anchors so their centers rotate correctly with the link.
    public var usesWorldSpaceRoundAnchor: Bool

    public init(body: Int, size: F3, friction: Float,
                dynamicFriction: Float? = nil,
                localPosition: F3 = .zero,
                localRotation: Quat = Quat(real: 1, imag: .zero),
                shape: BodyShape = .box,
                convexHullVertices: [F3] = [],
                collisionGroup: UInt32 = 0,
                collisionEnabled: Bool = true,
                usesWorldSpaceRoundAnchor: Bool = false,
                isRendered: Bool = true) {
        precondition(body >= 0, "collider requires a valid body owner")
        switch shape {
        case .sphere: self.size = F3(repeating: size.x)
        default: self.size = size
        }
        self.body = body
        self.friction = friction
        self.dynamicFriction = dynamicFriction ?? friction
        self.localPosition = localPosition
        self.localRotation = localRotation.normalized
        self.shape = shape
        self.convexHullVertices = convexHullVertices
        self.collisionGroup = collisionGroup
        self.collisionEnabled = collisionEnabled
        self.isRendered = isRendered
        self.usesWorldSpaceRoundAnchor = usesWorldSpaceRoundAnchor
    }
}

/// How a powered hinge converts its command into bounded joint effort.
///
/// All modes have fixed, authored gains. Solver conditioning is never used
/// as actuator state, so the plant remains stationary across timesteps,
/// solver iteration counts, and episode resets.
public enum JointMotorMode: Sendable, Equatable {
    /// Position PD stamped implicitly into the position-level solve. This is
    /// robust for stiff interactive mechanisms and larger timesteps.
    case implicitPositionPD
    /// Position PD evaluated once at the beginning of the physics step and
    /// applied as a constant bounded torque, matching MuJoCo/Isaac control
    /// loops and common robot low-level controllers.
    case explicitTorquePD
    /// Velocity feedback evaluated once per physics step:
    /// `torque = kd * (targetVelocity - jointVelocity)`.
    case velocity
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
    /// Motor command. This is a target angle in the two position-PD modes
    /// and a target angular velocity in velocity mode.
    public var motorTarget: Float
    /// Absolute joint-effort limit. Zero disables the motor.
    public var motorTorque: Float
    /// Fixed physical gains. Position modes use both `motorStiffness` (kp)
    /// and `motorDamping` (kd). Velocity mode requires zero stiffness and
    /// uses `motorDamping` as its velocity gain. Damping has units of torque
    /// per angular velocity in every mode.
    public var motorStiffness: Float
    public var motorDamping: Float
    public var motorMode: JointMotorMode
    /// Reflected rotor/joint inertia about the hinge coordinate (kg m^2).
    /// This is a joint-space mass term, distinct from damping and the two
    /// links' spatial inertia.
    public var armature: Float
    /// Hinge twist limits (radians). lo < hi enables; real arms have them —
    /// without limits a decelerating arm can tumble over the top and wedge.
    public var limitLo: Float
    public var limitHi: Float

    public init(bodyA: Int, bodyB: Int, rA: F3, rB: F3,
                stiffnessLin: Float = .infinity, stiffnessAng: Float = 0,
                fracture: Float = .infinity, fractureLinear: Bool = false,
                hingeAxis: F3? = nil, motorTarget: Float = 0,
                motorTorque: Float = 0, motorStiffness: Float = 0,
                motorDamping: Float = 0,
                motorMode: JointMotorMode = .implicitPositionPD,
                armature: Float = 0,
                limitLo: Float = 1, limitHi: Float = -1) {
        precondition(motorTarget.isFinite && motorTorque.isFinite
            && motorStiffness.isFinite && motorDamping.isFinite
            && armature.isFinite,
            "motor parameters and armature must be finite")
        precondition(motorTorque >= 0 && motorStiffness >= 0
            && motorDamping >= 0 && armature >= 0,
            "motor parameters and armature must be nonnegative")
        if motorTorque > 0 {
            precondition(hingeAxis != nil,
                         "a powered joint must define a hinge axis")
            switch motorMode {
            case .implicitPositionPD, .explicitTorquePD:
                precondition(motorStiffness > 0,
                             "position PD requires positive stiffness")
            case .velocity:
                precondition(motorStiffness == 0 && motorDamping > 0,
                    "velocity control requires zero stiffness and positive damping")
            }
        }
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
        self.motorStiffness = motorStiffness
        self.motorDamping = motorDamping
        self.motorMode = motorMode
        self.armature = armature
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

/// A visual-only vertex embedded in a simulated tetrahedron. The four ids
/// point at 3-DOF soft-body particles; `weights` are rest-pose barycentric
/// coordinates in that tet. Rendering updates this vertex from the current
/// tet deformation every frame, while collision remains owned by the tet
/// boundary triangles.
public struct SceneSkinnedVertex {
    public var ids: (Int, Int, Int, Int)
    public var weights: SIMD4<Float>
    public var restNormal: F3
    public var restInv0: F3
    public var restInv1: F3
    public var restInv2: F3

    public init(ids: (Int, Int, Int, Int), weights: SIMD4<Float>,
                restNormal: F3,
                restInv0: F3 = .zero,
                restInv1: F3 = .zero,
                restInv2: F3 = .zero) {
        self.ids = ids
        self.weights = weights
        self.restNormal = restNormal
        self.restInv0 = restInv0
        self.restInv1 = restInv1
        self.restInv2 = restInv2
    }
}

/// Visual mesh skinned to a tetrahedral soft body. Triangles index the local
/// `vertices` array. This mesh is not used for contact generation.
public struct SceneSkinnedMesh {
    public var vertices: [SceneSkinnedVertex]
    public var triangles: [(Int, Int, Int)]
    public var bodyIDs: [Int]

    public init(vertices: [SceneSkinnedVertex],
                triangles: [(Int, Int, Int)],
                bodyIDs: [Int] = []) {
        self.vertices = vertices
        self.triangles = triangles
        self.bodyIDs = bodyIDs
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

/// A rigid-body pair that must not generate contacts. This complements the
/// automatic exclusion of directly jointed and spring-connected bodies and
/// maps directly to MJCF `<exclude>` entries.
public struct SceneCollisionExclusion {
    public var bodyA: Int
    public var bodyB: Int

    public init(bodyA: Int, bodyB: Int) {
        precondition(bodyA >= 0 && bodyB >= 0 && bodyA != bodyB)
        self.bodyA = bodyA
        self.bodyB = bodyB
    }
}

public struct SimSettings {
    public var dt: Float = 1.0 / 60.0
    public var gravity: Float = -10.0
    public var iterations: Int = 10
    /// Normal contact slop/detection margin in scene length units. Legacy
    /// meter-scale demos use 1 cm; centimeter-scale robotics tasks should use
    /// a smaller value so thin workpieces do not rest deeply interpenetrated.
    public var collisionMargin: Float = 0.01
    public var frictionCombineMode: FrictionCombineMode = .geometricMean
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
    /// Per-second damping applied to non-ballistic 3-DOF particle velocity
    /// residuals only. Uniform gravity is preserved so soft bodies and
    /// rigid bodies free-fall at the same rate; damping bleeds
    /// solver/contact/internal jitter.
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
    /// Optional scene-specific camera framing. NaN means that the renderer's
    /// interactive default remains in charge. These values affect rendering
    /// only; training scenes and physics never consume them.
    public var cameraTargetX: Float = .nan
    public var cameraTargetY: Float = .nan
    public var cameraAzimuth: Float = .nan
    public var cameraElevation: Float = .nan
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
    public var colliders: [SceneCollider] = []
    public var joints: [SceneJoint] = []
    public var springs: [SceneSpring] = []
    public var tets: [SceneTet] = []
    public var tris: [SceneTri] = []
    public var skinnedMeshes: [SceneSkinnedMesh] = []
    public var spinners: [SceneSpinner] = []
    public var collisionExclusions: [SceneCollisionExclusion] = []
    public var settings = SimSettings()

    public init(name: String) {
        self.name = name
    }

    @discardableResult
    public mutating func addBody(size: F3, density: Float, friction: Float,
                                 dynamicFriction: Float? = nil, position: F3,
                                 rotation: Quat = Quat(real: 1, imag: .zero),
                                 velocity: F3 = .zero, shape: BodyShape = .box,
                                 mass: Float? = nil,
                                 diagonalInertia: F3? = nil,
                                 gravityScale: Float = 1,
                                 collisionGroup: UInt32 = 0,
                                 collisionEnabled: Bool = true) -> Int {
        bodies.append(SceneBody(size: size, density: density, friction: friction,
                                dynamicFriction: dynamicFriction,
                                position: position, rotation: rotation, velocity: velocity,
                                shape: shape, mass: mass,
                                diagonalInertia: diagonalInertia,
                                gravityScale: gravityScale))
        let body = bodies.count - 1
        if collisionEnabled {
            colliders.append(SceneCollider(
                body: body, size: size, friction: friction,
                dynamicFriction: dynamicFriction, shape: shape,
                collisionGroup: collisionGroup,
                usesWorldSpaceRoundAnchor: shape != .box))
        }
        return body
    }

    /// Attach an additional collision primitive without changing body mass or
    /// inertia. Imported robot links normally call `addBody(...,
    /// collisionEnabled: false)` and then add every source collision geom.
    @discardableResult
    public mutating func addCollider(body: Int, size: F3,
                                     friction: Float? = nil,
                                     dynamicFriction: Float? = nil,
                                     localPosition: F3 = .zero,
                                     localRotation: Quat = Quat(real: 1, imag: .zero),
                                     shape: BodyShape = .box,
                                     convexHullVertices: [F3] = [],
                                     collisionGroup: UInt32 = 0,
                                     collisionEnabled: Bool = true,
                                     isRendered: Bool = true) -> Int {
        precondition(bodies.indices.contains(body), "collider owner out of range")
        colliders.append(SceneCollider(
            body: body, size: size, friction: friction ?? bodies[body].friction,
            dynamicFriction: dynamicFriction
                ?? bodies[body].dynamicFriction,
            localPosition: localPosition, localRotation: localRotation,
            shape: shape, convexHullVertices: convexHullVertices,
            collisionGroup: collisionGroup,
            collisionEnabled: collisionEnabled,
            usesWorldSpaceRoundAnchor: false,
            isRendered: isRendered))
        return colliders.count - 1
    }

    /// Attach an authored convex hull to a rigid body. `vertices` are in the
    /// supplied collider frame and need not be centered; this method derives
    /// a tight AABB and shifts the collider origin without changing geometry.
    /// The current GPU narrow phase supports convex-vs-box contact, covering
    /// robot hulls against flat/boxed terrain and box projectiles.
    @discardableResult
    public mutating func addConvexCollider(
        body: Int, vertices: [F3], friction: Float? = nil,
        dynamicFriction: Float? = nil,
        localPosition: F3 = .zero,
        localRotation: Quat = Quat(real: 1, imag: .zero),
        collisionGroup: UInt32 = 0,
        collisionEnabled: Bool = true,
        isRendered: Bool = false
    ) -> Int {
        precondition(bodies.indices.contains(body), "collider owner out of range")
        precondition(vertices.count >= 4 && vertices.count <= 64,
                     "convex collider requires 4...64 cooked hull vertices")
        var lo = F3(repeating: .greatestFiniteMagnitude)
        var hi = F3(repeating: -.greatestFiniteMagnitude)
        for vertex in vertices {
            precondition(vertex.x.isFinite && vertex.y.isFinite && vertex.z.isFinite,
                         "convex collider vertices must be finite")
            lo = simd_min(lo, vertex)
            hi = simd_max(hi, vertex)
        }
        let center = (lo + hi) * 0.5
        let centered = vertices.map { $0 - center }
        return addCollider(
            body: body, size: hi - lo, friction: friction,
            dynamicFriction: dynamicFriction,
            localPosition: localPosition + localRotation.act(center),
            localRotation: localRotation, shape: .box,
            convexHullVertices: centered,
            collisionGroup: collisionGroup,
            collisionEnabled: collisionEnabled, isRendered: isRendered)
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
    public mutating func addCollisionExclusion(bodyA: Int, bodyB: Int) {
        precondition(bodies.indices.contains(bodyA) && bodies.indices.contains(bodyB),
                     "collision exclusion body out of range")
        collisionExclusions.append(.init(bodyA: bodyA, bodyB: bodyB))
    }

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
        let body = bodies.count - 1
        colliders.append(SceneCollider(
            body: body, size: b.size, friction: friction, shape: .sphere,
            usesWorldSpaceRoundAnchor: true))
        return body
    }

    public mutating func addSpring(_ s: SceneSpring) { springs.append(s) }
    public mutating func addTet(_ t: SceneTet) { tets.append(t) }
    public mutating func addTri(_ t: SceneTri) { tris.append(t) }
    public mutating func addSkinnedMesh(_ m: SceneSkinnedMesh) { skinnedMeshes.append(m) }

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
        solver.collisionMargin = settings.collisionMargin
        solver.frictionCombineMode = settings.frictionCombineMode

        for b in bodies {
            let rb = solver.addBody(size: b.size, density: b.density,
                                    friction: b.friction,
                                    dynamicFriction: b.dynamicFriction,
                                    position: b.position, rotation: b.rotation,
                                    velocity: b.velocity, shape: b.shape,
                                    mass: b.mass, diagonalInertia: b.diagonalInertia,
                                    gravityScale: b.gravityScale)
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
