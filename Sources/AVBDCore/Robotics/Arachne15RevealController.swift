import simd

/// A deterministic commissioning sequence that physically curls Arachne-15
/// into its compact guard pose and deploys it back to the learned-policy home
/// pose. It produces relative joint targets only: the normal torque-limited
/// motors, contact solver, gravity, and friction execute every transition.
///
/// The compact pose deliberately uses mechanical reserve travel outside the
/// learned policy's action envelope. Walking still sees the exact revision-6
/// `[-0.35, 0.35]` hip and `[-0.45, 0.45]` knee contract.
public final class Arachne15RevealController {
    public enum Phase: String, Sendable, Equatable {
        case folding
        case compactHold
        case unfolding
        case settling
        case complete
    }

    public struct Configuration: Sendable, Equatable {
        /// Each pair lifts clear of the floor, sweeps in yaw, then plants.
        public var liftSteps: Int
        public var sweepSteps: Int
        public var plantSteps: Int
        /// Delay between balanced diagonal waves. One four-leg support set
        /// remains planted while its complement transforms.
        public var pairStaggerSteps: Int
        public var unfoldWaveStaggerSteps: Int
        /// Whole-body transition into and out of the low transport crouch.
        /// All eight knees share the phone/chassis load during these ramps.
        public var transportSteps: Int
        public var compactHoldSteps: Int
        public var settlingSteps: Int
        public var liftedKneeTarget: Float

        public init(
            liftSteps: Int = 14,
            sweepSteps: Int = 18,
            plantSteps: Int = 26,
            pairStaggerSteps: Int = 58,
            unfoldWaveStaggerSteps: Int = 58,
            transportSteps: Int = 40,
            compactHoldSteps: Int = 70,
            settlingSteps: Int = 100,
            liftedKneeTarget: Float = -0.38
        ) {
            precondition(liftSteps >= 2 && sweepSteps >= 2 && plantSteps >= 2
                && pairStaggerSteps >= liftSteps + sweepSteps + plantSteps
                && unfoldWaveStaggerSteps
                    >= liftSteps + sweepSteps + plantSteps
                && transportSteps >= 2
                && compactHoldSteps >= 0
                && settlingSteps >= 1
                && liftedKneeTarget >= Arachne15RevealController
                    .kneeMechanicalLowerLimit
                && liftedKneeTarget <= Arachne15RevealController
                    .kneeMechanicalUpperLimit)
            self.liftSteps = liftSteps
            self.sweepSteps = sweepSteps
            self.plantSteps = plantSteps
            self.pairStaggerSteps = pairStaggerSteps
            self.unfoldWaveStaggerSteps = unfoldWaveStaggerSteps
            self.transportSteps = transportSteps
            self.compactHoldSteps = compactHoldSteps
            self.settlingSteps = settlingSteps
            self.liftedKneeTarget = liftedKneeTarget
        }
    }

    /// Mechanical limits authored in both generated MJCF profiles.
    public static let hipMechanicalLimit: Float = 0.55
    public static let kneeMechanicalLowerLimit: Float = -0.70
    public static let kneeMechanicalUpperLimit: Float = 0.90
    public static let commissioningTorqueScale: Float = 2.0

    /// Rear coxae sweep backward, front coxae forward, and every tibia curls
    /// inward beneath its hip. The 65-degree authored tibia mount plus 0.82 rad
    /// puts the beam at 112.0 degrees, visibly tucking each foot beneath the
    /// body while retaining 0.08 rad of hard-stop margin.
    public static let compactJointTargets: [Float] = [
        -0.52, 0.82, -0.52, 0.82,  0.52, 0.82,  0.52, 0.82,
         0.52, 0.82,  0.52, 0.82, -0.52, 0.82, -0.52, 0.82,
    ]
    public static let deployedJointTargets =
        [Float](repeating: 0, count: Arachne15PolicyContract.actionDimension)
    public static let transportKneeTarget: Float = -0.18

    /// Complementary diagonal waves keep forces left/right symmetric while
    /// their four planted legs span the COM.
    public static let foldOrder = [[0, 2, 5, 7], [1, 3, 4, 6]]
    public static let unfoldOrder = [[1, 3, 4, 6], [0, 2, 5, 7]]

    public let configuration: Configuration
    public let initialJointTargets: [Float]
    public private(set) var stepIndex = 0

    public init(
        initialJointTargets: [Float] = deployedJointTargets,
        configuration: Configuration = .init()
    ) {
        precondition(initialJointTargets.count
            == Arachne15PolicyContract.actionDimension)
        precondition(initialJointTargets.allSatisfy(\.isFinite))
        self.initialJointTargets = initialJointTargets
        self.configuration = configuration
    }

    public var pairMotionSteps: Int {
        perPairMotionSteps
            + configuration.pairStaggerSteps * (Self.foldOrder.count - 1)
    }

    public var perPairMotionSteps: Int {
        configuration.liftSteps + configuration.sweepSteps
            + configuration.plantSteps
    }

    public var totalSteps: Int {
        foldingSteps + configuration.compactHoldSteps + unfoldingSteps
            + configuration.settlingSteps
    }

    public var foldingSteps: Int {
        pairMotionSteps
    }

    public var unfoldingSteps: Int {
        2 * configuration.transportSteps + unfoldWaveMotionSteps
    }

    public var unfoldWaveMotionSteps: Int {
        perPairMotionSteps + configuration.unfoldWaveStaggerSteps
    }

    public var phase: Phase {
        if stepIndex >= totalSteps { return .complete }
        if stepIndex < foldingSteps { return .folding }
        if stepIndex < foldingSteps + configuration.compactHoldSteps {
            return .compactHold
        }
        if stepIndex < foldingSteps + configuration.compactHoldSteps
            + unfoldingSteps { return .unfolding }
        return .settling
    }

    public var progress: Float {
        min(Float(stepIndex) / Float(max(totalSteps, 1)), 1)
    }

    public var isComplete: Bool { phase == .complete }

    /// Return the next sixteen relative joint targets, then advance one 50 Hz
    /// control tick. Calling this after completion safely holds home pose.
    public func nextJointTargets() -> ContiguousArray<Float> {
        let targets = jointTargets(at: stepIndex)
        stepIndex = min(stepIndex + 1, totalSteps)
        return ContiguousArray(targets)
    }

    /// Resume a controller paused in its compact hold at the first deployment
    /// tick. Policy Replay uses this to expose folding and unfolding as two
    /// explicit physical actions without duplicating the trajectory.
    public func beginUnfolding() {
        stepIndex = min(
            max(stepIndex, foldingSteps + configuration.compactHoldSteps),
            totalSteps)
    }

    public func jointTargets(at step: Int) -> [Float] {
        let clampedStep = min(max(step, 0), totalSteps)
        var targets: [Float]
        if clampedStep < pairMotionSteps {
            targets = articulatedGroupTargets(
                at: clampedStep, from: initialJointTargets,
                to: Self.compactJointTargets,
                groups: Self.foldOrder,
                staggerSteps: configuration.pairStaggerSteps)
        } else if clampedStep < foldingSteps
                    + configuration.compactHoldSteps {
            targets = Self.compactJointTargets
        } else if clampedStep < foldingSteps
                    + configuration.compactHoldSteps + unfoldingSteps {
            let local = clampedStep - foldingSteps
                - configuration.compactHoldSteps
            let compactTransport = jointTargets(
                hipsFrom: Self.compactJointTargets,
                knee: Self.transportKneeTarget)
            let deployedTransport = jointTargets(
                hipsFrom: Self.deployedJointTargets,
                knee: Self.transportKneeTarget)
            if local < configuration.transportSteps {
                targets = interpolate(
                    from: Self.compactJointTargets,
                    to: compactTransport,
                    step: local, duration: configuration.transportSteps)
            } else if local < configuration.transportSteps
                        + unfoldWaveMotionSteps {
                targets = articulatedGroupTargets(
                    at: local - configuration.transportSteps,
                    from: compactTransport,
                    to: deployedTransport,
                    groups: Self.unfoldOrder,
                    staggerSteps: configuration.unfoldWaveStaggerSteps)
            } else {
                targets = interpolate(
                    from: deployedTransport,
                    to: Self.deployedJointTargets,
                    step: local - configuration.transportSteps
                        - unfoldWaveMotionSteps,
                    duration: configuration.transportSteps)
            }
        } else {
            targets = Self.deployedJointTargets
        }
        precondition(Self.targetsRespectMechanicalLimits(targets))
        return targets
    }

    public static func targetsRespectMechanicalLimits(
        _ targets: [Float]
    ) -> Bool {
        guard targets.count == Arachne15PolicyContract.actionDimension else {
            return false
        }
        return targets.indices.allSatisfy { index in
            let value = targets[index]
            guard value.isFinite else { return false }
            return index.isMultiple(of: 2)
                ? abs(value) <= hipMechanicalLimit
                : value >= kneeMechanicalLowerLimit
                    && value <= kneeMechanicalUpperLimit
        }
    }

    private func articulatedGroupTargets(
        at localStep: Int, from: [Float], to: [Float],
        groups: [[Int]], staggerSteps: Int
    ) -> [Float] {
        var result = from
        for (sequence, legs) in groups.enumerated() {
            let delayed = localStep
                - sequence * staggerSteps
            for leg in legs {
                guard delayed >= 0 else { continue }
                let hip = 2 * leg
                let knee = hip + 1
                let liftEnd = configuration.liftSteps
                let sweepEnd = liftEnd + configuration.sweepSteps
                let plantEnd = sweepEnd + configuration.plantSteps
                if delayed < liftEnd {
                    let t = smoothstep(
                        Float(delayed) / Float(configuration.liftSteps))
                    result[hip] = from[hip]
                    result[knee] = mix(
                        from[knee], configuration.liftedKneeTarget, t)
                } else if delayed < sweepEnd {
                    let t = smoothstep(
                        Float(delayed - liftEnd)
                            / Float(configuration.sweepSteps))
                    result[hip] = mix(from[hip], to[hip], t)
                    result[knee] = configuration.liftedKneeTarget
                } else if delayed < plantEnd {
                    let t = smoothstep(
                        Float(delayed - sweepEnd)
                            / Float(configuration.plantSteps))
                    result[hip] = to[hip]
                    result[knee] = mix(
                        configuration.liftedKneeTarget, to[knee], t)
                } else {
                    result[hip] = to[hip]
                    result[knee] = to[knee]
                }
            }
        }
        return result
    }

    private func jointTargets(hipsFrom source: [Float], knee: Float)
        -> [Float] {
        source.enumerated().map { index, value in
            index.isMultiple(of: 2) ? value : knee
        }
    }

    private func interpolate(
        from: [Float], to: [Float], step: Int, duration: Int
    ) -> [Float] {
        let t = smoothstep(Float(step) / Float(duration))
        return zip(from, to).map { mix($0, $1, t) }
    }

    private func smoothstep(_ value: Float) -> Float {
        let t = simd_clamp(value, 0, 1)
        return t * t * (3 - 2 * t)
    }

    private func mix(_ from: Float, _ to: Float, _ t: Float) -> Float {
        from + t * (to - from)
    }
}
