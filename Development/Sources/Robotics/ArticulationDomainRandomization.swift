import SimCore

/// Closed scalar multiplier range used by deterministic articulation-domain
/// randomization. A degenerate range such as `1...1` disables that channel.
public struct DynamicsMultiplierRange: Sendable, Equatable {
    public var lowerBound: Float
    public var upperBound: Float

    public init(_ lowerBound: Float = 1, _ upperBound: Float = 1) {
        precondition(lowerBound.isFinite && upperBound.isFinite
            && lowerBound > 0 && upperBound >= lowerBound,
            "dynamics multiplier range must be finite, positive, and ordered")
        self.lowerBound = lowerBound
        self.upperBound = upperBound
    }

    fileprivate func sample(using rng: inout SplitMix64) -> Float {
        lowerBound + (upperBound - lowerBound) * rng.nextFloat()
    }
}

/// Robot-agnostic, seeded sim-to-real plant randomization. Sampling occurs
/// once per replica at scene construction, so the hot Metal simulation path
/// remains branch-free and every experiment is exactly reproducible. Runtime
/// per-episode actuator delay/noise is intentionally task-owned because it is
/// part of the policy's sensor/control contract rather than the MJCF asset.
public struct ArticulationDomainRandomization: Sendable, Equatable {
    public var mass = DynamicsMultiplierRange()
    public var inertia = DynamicsMultiplierRange()
    public var friction = DynamicsMultiplierRange()
    public var motorTorque = DynamicsMultiplierRange()
    public var motorStiffness = DynamicsMultiplierRange()
    public var motorDamping = DynamicsMultiplierRange()
    public var armature = DynamicsMultiplierRange()

    public init(mass: DynamicsMultiplierRange = .init(),
                inertia: DynamicsMultiplierRange = .init(),
                friction: DynamicsMultiplierRange = .init(),
                motorTorque: DynamicsMultiplierRange = .init(),
                motorStiffness: DynamicsMultiplierRange = .init(),
                motorDamping: DynamicsMultiplierRange = .init(),
                armature: DynamicsMultiplierRange = .init()) {
        self.mass = mass
        self.inertia = inertia
        self.friction = friction
        self.motorTorque = motorTorque
        self.motorStiffness = motorStiffness
        self.motorDamping = motorDamping
        self.armature = armature
    }

    public func sample(seed: UInt64) -> MJCFDynamicsScale {
        var rng = SplitMix64(seed: seed)
        return MJCFDynamicsScale(
            mass: mass.sample(using: &rng),
            inertia: inertia.sample(using: &rng),
            friction: friction.sample(using: &rng),
            motorTorque: motorTorque.sample(using: &rng),
            motorStiffness: motorStiffness.sample(using: &rng),
            motorDamping: motorDamping.sample(using: &rng),
            armature: armature.sample(using: &rng))
    }

    /// Conservative starting envelope for a printed robot whose exact link
    /// masses, servo strength, drivetrain friction, and reflected inertia will
    /// be calibrated later from measurements.
    public static let conservativeSimToReal = ArticulationDomainRandomization(
        mass: .init(0.90, 1.10),
        inertia: .init(0.90, 1.10),
        friction: .init(0.75, 1.25),
        motorTorque: .init(0.85, 1.05),
        motorStiffness: .init(0.85, 1.15),
        motorDamping: .init(0.80, 1.20),
        armature: .init(0.80, 1.20))
}
