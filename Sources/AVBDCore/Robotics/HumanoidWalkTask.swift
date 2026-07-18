import simd

public struct HumanoidWalkTaskConfig: Sendable {
    public var numEnvironments: Int
    public var seed: UInt64
    public var maxEpisodeSteps: Int
    public var controlDecimation: Int
    public var minimumCommandSpeed: Float
    public var maximumCommandSpeed: Float
    public var commandCurriculumControlSteps: Int
    /// Training-only fraction of the episode horizon used to seed the first
    /// batch's episode ages. A value of one uniformly distributes initial
    /// ages over the complete horizon, preventing every healthy replica from
    /// producing a synchronized timeout/reset wave in one PPO minibatch.
    /// Subsequent resets start at age zero, and evaluation/replay never apply
    /// this randomization.
    public var trainingInitialEpisodeAgeFraction: Float
    /// Seeded fraction of straight velocity-command episodes assigned an
    /// exact zero command. This mirrors Isaac Lab's `rel_standing_envs` and
    /// teaches a distinct balance mode that continuous uniform sampling would
    /// almost surely never request exactly.
    public var standingCommandProbability: Float
    /// Training-only linear ramp from no standing episodes to the configured
    /// cohort probability. Evaluation always samples the full probability.
    public var standingCommandCurriculumControlSteps: Int
    /// Opt into PPO's two-expert actor. Exact standing commands route to a
    /// specialized actor while the verified moving actor remains frozen.
    public var commandGatedActor: Bool
    /// Split the specialized branch into learned low-speed and exact-stand
    /// actors. Cruise commands use the base actor, positive commands below
    /// `expertGateCommandSpeed` use the low-speed actor, and an exact zero
    /// command uses the stand actor. Routing contains no scripted actions.
    public var threeModeActor: Bool
    /// Commands below this speed route to the specialist actor. A point-goal
    /// arrival can therefore train the complete braking band while ordinary
    /// cruise commands remain on the frozen verified expert.
    public var expertGateCommandSpeed: Float
    /// Width of an optional smooth transition below the gate speed. Zero
    /// preserves the original hard router. A positive width prevents a
    /// one-frame action discontinuity when a goal command decelerates from
    /// the cruise expert into the balance/braking expert.
    public var expertGateBlendWidth: Float
    /// Optional measured-speed transition between the low-speed locomotion
    /// expert and the exact-stand expert after a point goal sets its command
    /// to zero. At or above `standExpertBlendStartSpeed` the verified braking
    /// expert remains fully active. Over `standExpertBlendWidth` below that
    /// speed, routing changes smoothly to the learned stand expert. Zeros
    /// preserve the legacy immediate zero-command switch.
    public var standExpertBlendStartSpeed: Float
    public var standExpertBlendWidth: Float
    /// Delay the learned stand branch until both feet have measured ground
    /// contact. Contact is a causal robot sensor, not a reference phase or a
    /// scripted action; it prevents a low-speed mid-swing state from being
    /// treated as equivalent to a planted stance.
    public var standExpertRequiresDoubleSupport: Bool
    /// Use the full measured planar base speed for the stand handoff. The
    /// policy frame's first velocity channel is only goal/heading projected;
    /// treating it as a magnitude can switch to standing while the robot is
    /// still sliding laterally.
    public var standExpertUsesPlanarSpeed: Bool
    /// Keep the base/cruise actor byte-stable during a specialist curriculum.
    /// This is useful for straight arrival and disturbance stages after a
    /// moving policy has passed its own gate. Full steering curricula leave
    /// it false so the cruise expert can learn non-zero yaw commands.
    public var freezeBasePolicyExpert: Bool
    /// Freeze the learned creep/braking branch while exact-zero arrival
    /// transitions specialize only the independent stand actor.
    public var freezeLowSpeedPolicyExpert: Bool
    /// Opt into updating the moving/base actor in a straight mixed-command
    /// curriculum. The legacy straight-task behavior freezes that accepted
    /// actor while learning only a zero-command specialist. Enabling this
    /// flag instead lets the base actor cover the full non-zero speed range
    /// while the specialist remains independently routed at exact zero.
    public var trainBasePolicyExpert: Bool
    /// Width of the planar command-tracking kernel. The published H1-style
    /// default is broad enough for cruise-gait discovery; a narrower value is
    /// useful for a dedicated creep expert that must distinguish low speeds.
    public var velocityTrackingStandardDeviation: Float
    /// Per-second signed L2 penalty on planar velocity-command error. Zero
    /// preserves the published exponential-only objective; a positive value
    /// supplies dense ordering when a transferred gait begins outside a
    /// narrow tracking kernel.
    public var velocityTrackingErrorPenaltyWeight: Float
    /// Per-second weight on unwanted planar base velocity as the command
    /// approaches zero. It is exactly inactive for commands at or above
    /// 0.2 m/s, preserving the accepted nominal locomotion objective.
    public var standStillVelocityPenaltyWeight: Float
    /// Weight on the sum of absolute deviations of every joint from the
    /// model's default standing pose near a zero command. This matches the
    /// command-conditioned H1 stand-still regularizer in MuJoCo Playground;
    /// it is a static posture cost, not a motion reference, phase clock, or
    /// scripted controller.
    public var standStillJointDeviationPenaltyWeight: Float
    /// Per-second reward for simultaneous physical contact at both feet as
    /// the command approaches zero. It is exactly inactive for locomotion
    /// commands and uses measured manifolds rather than a prescribed pose.
    public var standStillDoubleSupportRewardWeight: Float
    /// One-time terminal cost for falling while a near-zero command asks the
    /// robot to remain standing. Without this finite-horizon correction, an
    /// agent can rationally terminate early instead of paying velocity costs
    /// for the remaining episode. It fades continuously to zero at 0.2 m/s.
    public var standStillFallPenalty: Float
    public var lateralPenaltyWarmupControlSteps: Int
    public var lateralPenaltyRampControlSteps: Int
    /// Standard deviation of the straight A-to-B lane-tracking kernel. The
    /// kernel gates locomotion reward as lateral path error grows, so a fast
    /// diagonal trajectory cannot outperform commanded travel to the goal.
    public var laneTrackingStandardDeviation: Float
    /// Event reward for an opposite-foot touchdown after a real swing.  The
    /// event is multiplied by velocity tracking, so it cannot replace the
    /// commanded locomotion objective with in-place stepping.
    public var alternatingTouchdownRewardWeight: Float
    /// Per-second cost while neither foot has a physical ground manifold.
    /// This distinguishes walking from a symmetric hopping exploit without a
    /// phase schedule or reference motion.
    public var flightPenaltyWeight: Float
    /// Seeded whole-body reset perturbations. They rotate the complete
    /// articulated spawn about the pelvis, preserving every joint anchor
    /// while requiring the policy to recover from small balance and heading
    /// errors. This is the A-to-B analogue of the randomized base pose used
    /// by modern Isaac locomotion tasks.
    public var initialRollPitchRange: Float
    public var initialYawRange: Float
    /// Maximum absolute angle between world +X and the episode's point-goal
    /// direction. Zero preserves the accepted straight A-to-B task exactly;
    /// pi samples the complete horizontal circle.
    public var maximumGoalDirectionAngle: Float
    /// Training-only starting half-range for goal direction. This permits a
    /// validated steering policy to expand from (for example) +/-90 degrees
    /// to the full circle without first collapsing its learned distribution
    /// back to a straight-line-only curriculum. Evaluation always uses the
    /// configured maximum range.
    public var initialGoalDirectionAngle: Float
    /// Training-only ramp from the initial to the maximum goal-angle range.
    public var goalDirectionCurriculumControlSteps: Int
    /// Radius around the commanded endpoint required by point-goal success.
    public var goalRadius: Float
    /// Distance at which a point-goal command begins reducing cruise speed.
    /// The policy receives the reduced command and a bounded proximity
    /// signal, so it can learn to brake rather than hit a timed endpoint.
    public var goalSlowdownDistance: Float
    /// Replace the legacy point-goal proximity slot with measured lateral
    /// base velocity in the instantaneous goal frame. The commanded-speed
    /// channel already identifies the slowdown state, while this velocity is
    /// required for a Markovian braking/balance response. Disabled by default
    /// so existing checkpoints retain their exact observation semantics.
    public var goalObservationUsesLateralVelocity: Bool
    /// Append a nine-step lateral-velocity history while retaining every
    /// legacy observation channel at its original flat index. This is the
    /// transfer-safe form of the same signal: inherited policy behavior is
    /// initially exact because new first-layer weights start at zero.
    public var goalObservationIncludesLateralVelocity: Bool
    /// Optional task-space route range in meters. Positive values decouple
    /// point-goal geometry from the cruise command and episode timeout, so
    /// the horizon can include physically necessary braking and dwell time.
    /// Zeros preserve legacy `commandedSpeed * horizon` geometry.
    public var minimumGoalDistanceMeters: Float
    public var maximumGoalDistanceMeters: Float
    /// Training-only fraction of the full evaluation goal distance used at
    /// the beginning of a distance curriculum. Short early routes provide
    /// more genuine arrival and braking transitions per PPO update without
    /// changing the final task or its deterministic evaluation geometry.
    public var initialGoalDistanceScale: Float
    /// Control steps over which training goals expand to their full distance.
    public var goalDistanceCurriculumControlSteps: Int
    /// Consecutive stable control steps required inside the goal radius.
    public var goalDwellSteps: Int
    /// Maximum planar root speed counted as stable goal occupancy.
    public var maximumGoalArrivalSpeed: Float
    /// Positive radial command immediately outside the success circle. Zero
    /// preserves the legacy asymptotic controller. A value no greater than
    /// `maximumGoalArrivalSpeed` gives a learned policy a finite, admissible
    /// crossing velocity before the command becomes zero for the dwell.
    public var goalBoundaryCommandSpeed: Float
    /// Per-second dense credit for a control step that already satisfies every
    /// strict dwell predicate (inside radius, slow, upright, not fallen).
    /// The episode still succeeds only after `goalDwellSteps` consecutive
    /// valid steps; this merely shortens the temporal credit-assignment path.
    public var goalStableDwellRewardWeight: Float
    /// Fraction of episodes containing one physically simulated box launch.
    /// Zero omits projectile bodies entirely from the nominal walk scene.
    public var projectileProbability: Float
    public var projectileCurriculumControlSteps: Int
    public var minimumProjectileSpeed: Float
    public var maximumProjectileSpeed: Float
    public var minimumProjectileLaunchStep: Int
    public var maximumProjectileLaunchStep: Int
    /// Fraction of each newly requested normalized joint target applied per
    /// control step. `1` is the direct position-target interface used by the
    /// public H1 joystick tasks; values below one opt into a first-order
    /// actuator filter for experiments that explicitly require it.
    public var actionTargetResponse: Float
    public var autoReset: Bool

    public init(numEnvironments: Int, seed: UInt64 = 1,
                maxEpisodeSteps: Int = 1_000, controlDecimation: Int = 4,
                minimumCommandSpeed: Float = 0.45,
                maximumCommandSpeed: Float = 0.65,
                commandCurriculumControlSteps: Int = 0,
                trainingInitialEpisodeAgeFraction: Float = 0,
                standingCommandProbability: Float = 0,
                standingCommandCurriculumControlSteps: Int = 0,
                commandGatedActor: Bool = false,
                threeModeActor: Bool = false,
                expertGateCommandSpeed: Float = 0.20,
                expertGateBlendWidth: Float = 0,
                standExpertBlendStartSpeed: Float = 0,
                standExpertBlendWidth: Float = 0,
                standExpertRequiresDoubleSupport: Bool = false,
                standExpertUsesPlanarSpeed: Bool = false,
                freezeBasePolicyExpert: Bool = false,
                freezeLowSpeedPolicyExpert: Bool = false,
                trainBasePolicyExpert: Bool = false,
                velocityTrackingStandardDeviation: Float =
                    HumanoidLocomotionObjective.velocityTrackingStandardDeviation,
                velocityTrackingErrorPenaltyWeight: Float = 0,
                standStillVelocityPenaltyWeight: Float = 2,
                standStillJointDeviationPenaltyWeight: Float = 1,
                standStillDoubleSupportRewardWeight: Float = 1,
                standStillFallPenalty: Float = 0,
                lateralPenaltyWarmupControlSteps: Int = 0,
                lateralPenaltyRampControlSteps: Int = 0,
                laneTrackingStandardDeviation: Float = 0.30,
                alternatingTouchdownRewardWeight: Float = 2,
                flightPenaltyWeight: Float = 1,
                initialRollPitchRange: Float = 0.015,
                initialYawRange: Float = 0.05,
                maximumGoalDirectionAngle: Float = 0,
                initialGoalDirectionAngle: Float = 0,
                goalDirectionCurriculumControlSteps: Int = 0,
                goalRadius: Float = 1.5,
                goalSlowdownDistance: Float = 3,
                goalObservationUsesLateralVelocity: Bool = false,
                goalObservationIncludesLateralVelocity: Bool = false,
                minimumGoalDistanceMeters: Float = 0,
                maximumGoalDistanceMeters: Float = 0,
                initialGoalDistanceScale: Float = 1,
                goalDistanceCurriculumControlSteps: Int = 0,
                goalDwellSteps: Int = 25,
                maximumGoalArrivalSpeed: Float = 0.25,
                goalBoundaryCommandSpeed: Float = 0,
                goalStableDwellRewardWeight: Float = 0,
                projectileProbability: Float = 0,
                projectileCurriculumControlSteps: Int = 0,
                minimumProjectileSpeed: Float = 4,
                maximumProjectileSpeed: Float = 6,
                minimumProjectileLaunchStep: Int = 200,
                maximumProjectileLaunchStep: Int = 700,
                actionTargetResponse: Float = 1,
                autoReset: Bool = true) {
        self.numEnvironments = numEnvironments
        self.seed = seed
        self.maxEpisodeSteps = maxEpisodeSteps
        self.controlDecimation = controlDecimation
        self.minimumCommandSpeed = minimumCommandSpeed
        self.maximumCommandSpeed = maximumCommandSpeed
        self.commandCurriculumControlSteps = commandCurriculumControlSteps
        self.trainingInitialEpisodeAgeFraction =
            trainingInitialEpisodeAgeFraction
        self.standingCommandProbability = standingCommandProbability
        self.standingCommandCurriculumControlSteps =
            standingCommandCurriculumControlSteps
        self.commandGatedActor = commandGatedActor
        self.threeModeActor = threeModeActor
        self.expertGateCommandSpeed = expertGateCommandSpeed
        self.expertGateBlendWidth = expertGateBlendWidth
        self.standExpertBlendStartSpeed = standExpertBlendStartSpeed
        self.standExpertBlendWidth = standExpertBlendWidth
        self.standExpertRequiresDoubleSupport =
            standExpertRequiresDoubleSupport
        self.standExpertUsesPlanarSpeed = standExpertUsesPlanarSpeed
        self.freezeBasePolicyExpert = freezeBasePolicyExpert
        self.freezeLowSpeedPolicyExpert = freezeLowSpeedPolicyExpert
        self.trainBasePolicyExpert = trainBasePolicyExpert
        self.velocityTrackingStandardDeviation =
            velocityTrackingStandardDeviation
        self.velocityTrackingErrorPenaltyWeight =
            velocityTrackingErrorPenaltyWeight
        self.standStillVelocityPenaltyWeight = standStillVelocityPenaltyWeight
        self.standStillJointDeviationPenaltyWeight =
            standStillJointDeviationPenaltyWeight
        self.standStillDoubleSupportRewardWeight =
            standStillDoubleSupportRewardWeight
        self.standStillFallPenalty = standStillFallPenalty
        self.lateralPenaltyWarmupControlSteps = lateralPenaltyWarmupControlSteps
        self.lateralPenaltyRampControlSteps = lateralPenaltyRampControlSteps
        self.laneTrackingStandardDeviation = laneTrackingStandardDeviation
        self.alternatingTouchdownRewardWeight = alternatingTouchdownRewardWeight
        self.flightPenaltyWeight = flightPenaltyWeight
        self.initialRollPitchRange = initialRollPitchRange
        self.initialYawRange = initialYawRange
        self.maximumGoalDirectionAngle = maximumGoalDirectionAngle
        self.initialGoalDirectionAngle = initialGoalDirectionAngle
        self.goalDirectionCurriculumControlSteps =
            goalDirectionCurriculumControlSteps
        self.goalRadius = goalRadius
        self.goalSlowdownDistance = goalSlowdownDistance
        self.goalObservationUsesLateralVelocity =
            goalObservationUsesLateralVelocity
        self.goalObservationIncludesLateralVelocity =
            goalObservationIncludesLateralVelocity
        self.minimumGoalDistanceMeters = minimumGoalDistanceMeters
        self.maximumGoalDistanceMeters = maximumGoalDistanceMeters
        self.initialGoalDistanceScale = initialGoalDistanceScale
        self.goalDistanceCurriculumControlSteps =
            goalDistanceCurriculumControlSteps
        self.goalDwellSteps = goalDwellSteps
        self.maximumGoalArrivalSpeed = maximumGoalArrivalSpeed
        self.goalBoundaryCommandSpeed = goalBoundaryCommandSpeed
        self.goalStableDwellRewardWeight = goalStableDwellRewardWeight
        self.projectileProbability = projectileProbability
        self.projectileCurriculumControlSteps = projectileCurriculumControlSteps
        self.minimumProjectileSpeed = minimumProjectileSpeed
        self.maximumProjectileSpeed = maximumProjectileSpeed
        self.minimumProjectileLaunchStep = minimumProjectileLaunchStep
        self.maximumProjectileLaunchStep = maximumProjectileLaunchStep
        self.actionTargetResponse = actionTargetResponse
        self.autoReset = autoReset
    }
}

public struct HumanoidState {
    public var root: GPUSolver.RigidBodyState
    public var torso: GPUSolver.RigidBodyState
    public var leftFoot: GPUSolver.RigidBodyState
    public var rightFoot: GPUSolver.RigidBodyState
    public var jointAngles: [Float]
    public var jointVelocities: [Float]
}

/// Measured state used by whole-body manipulation tasks.  `leftHand` and
/// `rightHand` are the actual terminal collision-sphere centers, not link COM
/// poses; `object` is the physical rigid body owned by the same replica.
public struct HumanoidManipulationState {
    public var object: GPUSolver.RigidBodyState
    public var leftHand: GPUSolver.RigidBodyState
    public var rightHand: GPUSolver.RigidBodyState
}

public struct HumanoidLocomotionSuccessComponents: Equatable, Sendable {
    public var fullHorizon: Bool
    public var distanceBand: Bool
    public var lateralCorridor: Bool
    public var heading: Bool
    public var alternatingGait: Bool
    public var goalReached: Bool = true

    public var allPassed: Bool {
        fullHorizon && distanceBand && lateralCorridor && heading
            && alternatingGait && goalReached
    }
}

/// Reference-free locomotion objective expressed only in measured simulator
/// state. Reward terms are integrated in seconds so their scale does not
/// silently change with control decimation. The dense objective follows the
/// current MuJoCo Playground H1 joystick structure: commanded planar velocity
/// and commanded yaw-rate tracking plus orientation, vertical velocity,
/// action rate, and foot-slip regularization. The straight task commands zero
/// yaw; the point-goal task derives a bounded yaw command from measured
/// relative heading. Exact world displacement remains an
/// independently reported success metric, not a second progress reward that
/// can make a short forward lunge attractive.
public enum HumanoidLocomotionObjective {
    public static let velocityTrackingStandardDeviation: Float = 0.5
    /// MuJoCo Playground's H1 joystick task shapes a moving swing foot toward
    /// 13 cm of geometric ground clearance. This depends only on measured
    /// foot state: there is no phase clock, reference trajectory, or selected
    /// support leg.
    public static let targetFootClearance: Float = 0.13
    /// Isaac's current H1 reward penalizes the sum of contact-foot speed
    /// norms, not squared speeds, with weight -0.25. It is zero for a planted
    /// support foot and does not select a leg, phase, pose, or trajectory.
    public static let feetSlidePenaltyWeight: Float = 0.25
    /// The official H1 model spawns at 1.06 m. MuJoCo Playground terminates
    /// when the torso/pelvis origin falls below 0.92 m, and Isaac Lab ends on
    /// Catastrophic-tunneling guard below the physical torso-ground contact
    /// termination. Ordinary falls terminate from an actual contact manifold,
    /// not a task-tuned root-height heuristic.
    public static let minimumPelvisHeight: Float = 0.55
    public static let terminationPenalty: Float = 4
    public static let minimumDistanceRatio: Float = 0.60
    public static let maximumDistanceRatio: Float = 1.40
    public static let minimumHeadingAlignment: Float = 0.75
    public static let maximumLateralDrift: Float = 0.30
    /// Evaluation threshold for an explicit standing command. This is an
    /// episode-average planar translation rate, not a reward term.
    public static let maximumStandingDriftSpeed: Float = 0.10
    /// One genuine forward foot exchange per second is a deliberately modest
    /// lower bound for a 20 s walking trial. The former value of six admitted
    /// low-cadence shuffling as successful locomotion.
    public static let minimumAlternatingSteps = 20
    /// A touchdown is a forward step only after that foot has physically
    /// passed the other foot along the measured torso heading. This rejects a
    /// split-stance pogo whose contact manifolds happen to alternate while
    /// preserving either leg as the freely chosen first mover.
    public static let minimumForwardFootExchange: Float = 0.08

    public static func velocityTracking(
        measured: Float, commanded: Float,
        standardDeviation: Float =
            HumanoidLocomotionObjective.velocityTrackingStandardDeviation
    ) -> Float {
        precondition(standardDeviation > 0)
        let error = measured - commanded
        let variance = standardDeviation * standardDeviation
        return exp(-(error * error) / variance)
    }

    /// Gait-specific rewards are meaningful only when locomotion is
    /// commanded. Isaac-style command tasks gate air-time/step bonuses near
    /// zero velocity so walking in place cannot beat a quiet balanced stance.
    public static func movementCommandWeight(
        commandedSpeed: Float, fullWeightSpeed: Float = 0.20
    ) -> Float {
        precondition(commandedSpeed >= 0 && fullWeightSpeed > 0)
        return simd_clamp(commandedSpeed / fullWeightSpeed, 0, 1)
    }

    /// Command-conditioned cost for unwanted planar base motion. A zero-speed
    /// command should prefer a quiet balanced stance, while ordinary walking
    /// commands must preserve the accepted locomotion objective exactly.
    /// Fading the cost with the same gate as gait rewards makes the transition
    /// from stand to walk continuous and keeps this reference-free.
    public static func standStillVelocityCost(
        commandedSpeed: Float, measuredVelocity: F3,
        fullWeightSpeed: Float = 0.20
    ) -> Float {
        let standWeight = 1 - movementCommandWeight(
            commandedSpeed: commandedSpeed,
            fullWeightSpeed: fullWeightSpeed)
        return standWeight * (measuredVelocity.x * measuredVelocity.x
            + measuredVelocity.y * measuredVelocity.y)
    }

    public static func standStillJointDeviationCost(
        commandedSpeed: Float, jointDeviationAbsolute: Float,
        fullWeightSpeed: Float = 0.20
    ) -> Float {
        precondition(jointDeviationAbsolute >= 0)
        let standWeight = 1 - movementCommandWeight(
            commandedSpeed: commandedSpeed,
            fullWeightSpeed: fullWeightSpeed)
        return standWeight * jointDeviationAbsolute
    }

    /// Contact-conditioned standing signal. This closes the valid but
    /// unwanted one-leg-balance optimum without choosing a support leg,
    /// prescribing joint motion, or affecting commanded locomotion.
    public static func standStillDoubleSupportReward(
        commandedSpeed: Float, bothFeetInContact: Bool,
        fullWeightSpeed: Float = 0.20
    ) -> Float {
        guard bothFeetInContact else { return 0 }
        return 1 - movementCommandWeight(
            commandedSpeed: commandedSpeed,
            fullWeightSpeed: fullWeightSpeed)
    }

    public static func standStillFallCost(
        commandedSpeed: Float, fallen: Bool, penalty: Float,
        fullWeightSpeed: Float = 0.20
    ) -> Float {
        precondition(penalty >= 0)
        guard fallen else { return 0 }
        return penalty * (1 - movementCommandWeight(
            commandedSpeed: commandedSpeed,
            fullWeightSpeed: fullWeightSpeed))
    }

    /// Proportional heading command used by the point-goal task. This is a
    /// task-space velocity command, not an action or reference pose: PPO must
    /// still discover the coordinated joint torques that realize the turn.
    public static func pointGoalYawRate(relativeHeading: F3,
                                        gain: Float = 2,
                                        maximumRate: Float = 1.2) -> Float {
        precondition(gain >= 0 && maximumRate > 0)
        let angleFromHeadingToGoal = -atan2(
            relativeHeading.y, relativeHeading.x)
        return PointGoalNavigator.boundedYawRate(
            bearing: angleFromHeadingToGoal,
            gain: gain, maximumRate: maximumRate)
    }

    /// Smooth task-space speed command for genuine point navigation. Cruise
    /// speed is unchanged away from the target. The legacy default reaches
    /// zero at the success radius; an optional admissible boundary speed lets
    /// the policy cross the success boundary before receiving a zero command.
    public static func pointGoalCommandSpeed(
        remainingDistance: Float, cruiseSpeed: Float,
        goalRadius: Float, slowdownDistance: Float,
        boundaryCommandSpeed: Float = 0
    ) -> Float {
        precondition(remainingDistance >= 0 && cruiseSpeed >= 0)
        precondition(goalRadius > 0 && slowdownDistance > goalRadius)
        precondition(boundaryCommandSpeed >= 0)
        let boundarySpeed = min(boundaryCommandSpeed, cruiseSpeed)
        return PointGoalNavigator.commandSpeed(
            remainingDistance: remainingDistance,
            cruiseSpeed: cruiseSpeed,
            goalRadius: goalRadius,
            slowdownDistance: slowdownDistance,
            boundarySpeed: boundarySpeed)
    }

    /// Bounded observation channel that is zero through normal cruising and
    /// rises to one at the goal radius. Zero matches the accepted walker's
    /// lateral-offset channel at transfer initialization.
    public static func pointGoalProximity(
        remainingDistance: Float, goalRadius: Float,
        slowdownDistance: Float
    ) -> Float {
        precondition(remainingDistance >= 0)
        precondition(goalRadius > 0 && slowdownDistance > goalRadius)
        return PointGoalNavigator.proximity(
            remainingDistance: remainingDistance,
            goalRadius: goalRadius,
            slowdownDistance: slowdownDistance)
    }

    /// Smooth arrival-and-balance shaping used near a point goal. It depends
    /// only on task-space distance and measured body state, never a reference
    /// pose or scripted action. The strict success velocity and dwell tests
    /// remain separate, so shaping cannot turn a fly-through into success.
    public static func pointGoalArrivalQuality(
        proximity: Float, planarSpeed: Float, upright: Float,
        arrivalSpeed: Float
    ) -> Float {
        precondition(proximity >= 0 && proximity <= 1)
        precondition(planarSpeed >= 0 && arrivalSpeed > 0)
        let speedScore = exp(-(planarSpeed * planarSpeed)
            / (arrivalSpeed * arrivalSpeed))
        let uprightScore = simd_clamp((upright - 0.35) / 0.45, 0, 1)
        return proximity * speedScore * uprightScore * uprightScore
    }

    public static func laneTracking(lateralDisplacementSquared: Float,
                                    standardDeviation: Float) -> Float {
        precondition(lateralDisplacementSquared >= 0 && standardDeviation > 0)
        return exp(-lateralDisplacementSquared
            / (standardDeviation * standardDeviation))
    }

    /// Reference-free moving-foot clearance cost used by the current MuJoCo
    /// Playground H1 task. `sqrt(speed)` keeps the signal smooth enough near
    /// touchdown while making a stationary stance foot contribute zero.
    public static func footClearanceCost(clearance: Float,
                                         horizontalSpeedSquared: Float) -> Float {
        let speed = sqrt(max(horizontalSpeedSquared, 0))
        let error = clearance - targetFootClearance
        return error * error * sqrt(speed)
    }

    /// Phase-free dense contact reward used by Isaac Lab's
    /// `feet_air_time_positive_biped`: reward only exact single stance and
    /// use the shorter of the stance foot's current contact time and the
    /// swing foot's current air time. A hop, double support, or prescribed
    /// left/right schedule receives zero.
    public static func positiveBipedAirTime(
        leftInContact: Bool, rightInContact: Bool,
        leftAirTime: Float, rightAirTime: Float,
        leftContactTime: Float, rightContactTime: Float,
        threshold: Float = 0.6
    ) -> Float {
        guard leftInContact != rightInContact else { return 0 }
        let leftModeTime = leftInContact ? leftContactTime : leftAirTime
        let rightModeTime = rightInContact ? rightContactTime : rightAirTime
        return min(min(leftModeTime, rightModeTime), threshold)
    }

    public static func isLeadingTouchdown(
        touchdownFootPosition: F3, otherFootPosition: F3,
        heading: F3,
        minimumExchange: Float = minimumForwardFootExchange
    ) -> Bool {
        let horizontalHeading = F3(heading.x, heading.y, 0)
        let headingLength = simd_length(horizontalHeading)
        guard headingLength > 1e-6, minimumExchange >= 0 else { return false }
        let forward = horizontalHeading / headingLength
        return simd_dot(touchdownFootPosition - otherFootPosition, forward)
            >= minimumExchange
    }

    public static func reward(controlStep: Float, commandedSpeed: Float,
                              measuredVelocity: F3, forwardDisplacement: Float,
                              tiltSquared: Float, angularVelocityXYSquared: Float,
                              yawAngularVelocitySquared: Float,
                              headingErrorSquared: Float,
                              actionRateSquared: Float, feetAirTime: Float,
                              alternatingTouchdown: Float = 0,
                              alternatingTouchdownWeight: Float = 0,
                              feetFlight: Float = 0,
                              feetFlightPenaltyWeight: Float = 0,
                              feetSlideSpeed: Float, jointDeviation: Float,
                              fallen: Bool,
                              standStillVelocityPenaltyWeight: Float = 2,
                              standStillJointDeviationAbsolute: Float = 0,
                              standStillJointDeviationPenaltyWeight: Float = 1,
                              bothFeetInContact: Bool = false,
                              standStillDoubleSupportRewardWeight: Float = 1,
                              standStillFallPenalty: Float = 0,
                              velocityTrackingStandardDeviation: Float =
                                HumanoidLocomotionObjective
                                    .velocityTrackingStandardDeviation,
                              velocityTrackingErrorPenaltyWeight: Float = 0,
                              lateralDisplacementSquared: Float = 0,
                              laneTrackingStandardDeviation: Float = .infinity,
                              feetClearanceCost: Float = 0) -> Float {
        _ = forwardDisplacement
        precondition(velocityTrackingStandardDeviation > 0)
        precondition(velocityTrackingErrorPenaltyWeight >= 0)
        let variance = velocityTrackingStandardDeviation
            * velocityTrackingStandardDeviation
        let planarError = (measuredVelocity.x - commandedSpeed)
            * (measuredVelocity.x - commandedSpeed)
            + measuredVelocity.y * measuredVelocity.y
        let tracking = exp(-planarError / variance)
            * laneTracking(
                lateralDisplacementSquared: lateralDisplacementSquared,
                standardDeviation: laneTrackingStandardDeviation)
        // The caller supplies squared error relative to its task-space yaw
        // command. For the accepted straight task that command remains exact
        // zero; point-goal locomotion instead rewards the turn needed to face
        // the endpoint. This matches the planar/yaw command decomposition in
        // published H1 joystick tasks without prescribing any joint motion.
        let yawTracking = exp(-yawAngularVelocitySquared / variance)
        let movementCommandWeight = movementCommandWeight(
            commandedSpeed: commandedSpeed)
        let regularizedRate = tracking + yawTracking
            - velocityTrackingErrorPenaltyWeight * planarError
            - standStillVelocityPenaltyWeight * standStillVelocityCost(
                commandedSpeed: commandedSpeed,
                measuredVelocity: measuredVelocity)
            - standStillJointDeviationPenaltyWeight
                * standStillJointDeviationCost(
                    commandedSpeed: commandedSpeed,
                    jointDeviationAbsolute: standStillJointDeviationAbsolute)
            + standStillDoubleSupportRewardWeight
                * standStillDoubleSupportReward(
                    commandedSpeed: commandedSpeed,
                    bothFeetInContact: bothFeetInContact)
            - 2.00 * measuredVelocity.z * measuredVelocity.z
            - 0.05 * angularVelocityXYSquared
            - 0.10 * yawAngularVelocitySquared
            // Match Isaac Lab H1's flat-orientation scale. MuJoCo
            // Playground uses -5 together with an additional +0.8 yaw-rate
            // tracking reward; retaining the former after deliberately
            // removing the latter starved an otherwise stable straight-line
            // policy of dense reward.
            - 1.00 * tiltSquared
            - 0.50 * headingErrorSquared
            - 1.00 * lateralDisplacementSquared
            // Isaac Lab's H1 task uses -0.005 for action-rate L2 together
            // with a -200 termination term (integrated at dt, i.e. -4 at
            // this task's control rate). The previous -0.20 coefficient was
            // borrowed from MuJoCo Playground while retaining Isaac's much
            // stronger termination penalty. That hybrid clipped essentially
            // every exploratory running reward to zero and taught only
            // standing/survival. Keep the coherent Isaac-scale regularizer.
            - 0.005 * actionRateSquared
            - feetSlidePenaltyWeight * feetSlideSpeed
            - 0.50 * feetClearanceCost
            - jointDeviation
        // The dense positive-biped term alone can be maximized by holding one
        // foot up forever.  Give an event reward only after a real swing lands
        // on the opposite foot, and gate it by measured command tracking so
        // in-place stepping cannot substitute for locomotion.  There is no
        // clock, reference pose, or selected lead leg.
        let contactReward = movementCommandWeight * (
            controlStep * feetAirTime
                + alternatingTouchdownWeight * alternatingTouchdown * tracking)
        // Keep the signed Isaac-style objective with the Isaac-scale terminal
        // cost. Combining that -4 fall cost with MuJoCo Playground's
        // nonnegative clipping erased the ordering among all poor exploratory
        // motions: PPO could only distinguish "stand" from "eventually fall".
        // Signed running costs preserve gradients from slip, tilt, and action
        // smoothness before a viable gait exists; the terminal cost still
        // makes deliberately ending an episode unprofitable.
        let runningReward = controlStep * regularizedRate + contactReward
            - controlStep * feetFlightPenaltyWeight * feetFlight
        return runningReward - (fallen ? terminationPenalty : 0)
            - standStillFallCost(
                commandedSpeed: commandedSpeed, fallen: fallen,
                penalty: standStillFallPenalty)
    }

    public static func isSuccessful(timedOut: Bool, fallen: Bool,
                                    forwardDistance: Float,
                                    lateralDistance: Float,
                                    headingAlignment: Float,
                                    alternatingSteps: Int,
                                    commandedSpeed: Float,
                                    elapsed: Float) -> Bool {
        successComponents(
            timedOut: timedOut, fallen: fallen,
            forwardDistance: forwardDistance,
            lateralDistance: lateralDistance,
            headingAlignment: headingAlignment,
            alternatingSteps: alternatingSteps,
            commandedSpeed: commandedSpeed, elapsed: elapsed).allPassed
    }

    public static func successComponents(
        timedOut: Bool, fallen: Bool, forwardDistance: Float,
        lateralDistance: Float, headingAlignment: Float,
        alternatingSteps: Int, commandedSpeed: Float, elapsed: Float,
        goalDistanceError: Float? = nil, goalRadius: Float = .infinity
    ) -> HumanoidLocomotionSuccessComponents {
        let fullHorizon = timedOut && !fallen
        let commandedDistance = commandedSpeed * elapsed
        let standing = commandedSpeed < 0.20
        let standingDriftSpeed = abs(forwardDistance) / max(elapsed, 1e-6)
        return HumanoidLocomotionSuccessComponents(
            fullHorizon: fullHorizon,
            distanceBand: standing
                ? standingDriftSpeed <= maximumStandingDriftSpeed
                : forwardDistance >= minimumDistanceRatio * commandedDistance
                    && forwardDistance <= maximumDistanceRatio * commandedDistance,
            lateralCorridor: abs(lateralDistance) <= maximumLateralDrift,
            heading: headingAlignment >= minimumHeadingAlignment,
            alternatingGait: standing
                || alternatingSteps >= minimumAlternatingSteps,
            goalReached: goalDistanceError.map { $0 <= goalRadius } ?? true)
    }
}

/// The official 19-DoF Unitree H1 MuJoCo-Menagerie articulation, replicated
/// into one Metal solve. Source COM/principal inertias, joint frames, limits,
/// torque bounds, and compound collision primitives are imported rather than
/// reconstructed from hand-sized boxes.
public enum HumanoidControlProfile: String, Sendable {
    /// Isaac Lab's lower-gain H1 implicit-actuator setup and H1_MINIMAL
    /// collision model (ankles plus torso). Kept as the default for
    /// compatibility with existing tasks and checkpoints.
    case isaacLab
    /// MuJoCo Playground's H1 joystick contract: 4 ms physics, its exact home
    /// keyframe, compiler-resolved critically damped position servos, and the
    /// reduced feet-only collision model used during published training.
    case mujocoPlayground
    /// Unitree RL Gym's public recurrent H1 deployment contract: 2 ms
    /// physics, 50 Hz policy updates, the published 10-joint gains, and a
    /// rigid upper body. This profile exists for unchanged-policy sim-to-sim
    /// validation rather than native AVBD training.
    case unitreeRLGym
}

public final class HumanoidWalkEnv {
    private static let h1FootCapsuleEndpoints: [F3] = [
        F3(-0.035, 0, -0.056), F3(0.020, 0, -0.045),
        F3(0.115, 0, -0.056),
        F3(0.140, -0.030, -0.056), F3(0.140, 0.030, -0.056),
    ]
    private static let h1FootCapsuleRadius: Float = 0.014

    public struct EnvRefs {
        public var center: F3
        public var root: Int
        public var torso: Int
        public var leftFoot: Int
        public var rightFoot: Int
        /// The public 19-DoF H1 asset has no wrist joint.  Its terminal hand
        /// collision sphere is authored on the elbow/forearm rigid body.
        public var leftHand: Int
        public var rightHand: Int
        public var rootFrame: MJCFLinkFrame
        public var torsoFrame: MJCFLinkFrame
        public var leftFootFrame: MJCFLinkFrame
        public var rightFootFrame: MJCFLinkFrame
        /// Frames of the centers of the terminal 33 mm hand collision
        /// spheres.  The public H1 model has no wrist body, so reading the
        /// elbow body's COM would put manipulation observations roughly a
        /// forearm length away from the actual contact point.
        public var leftHandFrame: MJCFLinkFrame
        public var rightHandFrame: MJCFLinkFrame
        public var bodies: [Int]
        public var motors: [Int]
        public var projectile: Int?
        public var carryPedestal: Int?
        /// Additional static parts belonging to the same carry support. The
        /// primary body above is the load-bearing top; these are ordinary
        /// colliders such as table legs, stored with top-relative offsets so
        /// the complete prop can move during episode layout.
        public var carryPedestalParts: [Int]
        public var carryPedestalPartOffsets: [F3]
        /// Optional receiving support for transport/manipulation tasks. It is
        /// kept distinct from the source support so task success can be based
        /// on the exact physical manifold that supports the object.
        public var carryDestinationPedestal: Int?
        public var carryDestinationPedestalParts: [Int]
        public var carryDestinationPedestalPartOffsets: [F3]
        public var startMarker: Int?
        public var goalMarker: Int?
    }

    public static let jointRanges: [(Float, Float)] = [
        (-0.43, 0.43), (-0.43, 0.43), (-1.29, 1.85),
        (-1.05, 1.26), (-0.35, 1.04),
        (-0.43, 0.43), (-0.43, 0.43), (-1.29, 1.85),
        (-1.05, 1.26), (-0.35, 1.04),
        (-2.35, 2.35),
        (-2.87, 2.87), (-0.34, 3.11), (-1.30, 4.45), (-1.25, 2.61),
        (-2.87, 2.87), (-3.11, 0.34), (-4.45, 1.30), (-1.25, 2.61),
    ]
    /// Normalized policy actions are offsets from the stable standing pose,
    /// not midpoints of asymmetric joint limits (which would command both
    /// knees to 0.65 rad for a zero action).
    public static let defaultJointPositions: [Float] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]
    /// Published Isaac Lab and MuJoCo Playground H1 tasks both use a uniform
    /// 0.5-radian position-target scale around the model's home pose.
    public static let actionScales: [Float] = [Float](repeating: 0.5, count: 19)

    /// IsaacLab's public H1 crouched initialization, rebased to zero in the
    /// policy coordinate system by `MJCFAsset.instantiate`.
    private static let isaacLabHomePositions: [String: Float] = [
        "left_hip_pitch": -0.28, "left_knee": 0.79, "left_ankle": -0.52,
        "right_hip_pitch": -0.28, "right_knee": 0.79, "right_ankle": -0.52,
        "left_shoulder_pitch": 0.28, "left_elbow": 0.52,
        "right_shoulder_pitch": 0.28, "right_elbow": 0.52,
    ]

    /// Exact `home` keyframe from MuJoCo Playground's
    /// `scene_mjx_feetonly.xml`, excluding the free-root coordinates.
    private static let mujocoPlaygroundHomePositions: [String: Float] = [
        "left_hip_pitch": -0.267, "left_knee": 0.8,
        "left_ankle": -0.543,
        "right_hip_pitch": -0.267, "right_knee": 0.8,
        "right_ankle": -0.543,
        "left_shoulder_roll": 0.2, "left_elbow": 1.14,
        "right_shoulder_roll": -0.2, "right_elbow": 1.14,
    ]

    private static let isaacLabMotorGains: [String: MJCFMotorGain] = {
        var gains: [String: MJCFMotorGain] = [:]
        for side in ["left", "right"] {
            gains["\(side)_hip_yaw"] = .init(stiffness: 150, damping: 5)
            gains["\(side)_hip_roll"] = .init(stiffness: 150, damping: 5)
            gains["\(side)_hip_pitch"] = .init(stiffness: 200, damping: 5)
            gains["\(side)_knee"] = .init(stiffness: 200, damping: 5)
            gains["\(side)_ankle"] = .init(stiffness: 20, damping: 4)
            gains["\(side)_shoulder_pitch"] = .init(stiffness: 40, damping: 10)
            gains["\(side)_shoulder_roll"] = .init(stiffness: 40, damping: 10)
            gains["\(side)_shoulder_yaw"] = .init(stiffness: 40, damping: 10)
            gains["\(side)_elbow"] = .init(stiffness: 40, damping: 10)
        }
        gains["torso"] = .init(stiffness: 200, damping: 5)
        return gains
    }()

    /// MuJoCo 3.3.7 compiler output for `kp` plus `dampratio=1` at the public
    /// H1 home keyframe. The source joint's passive damping of 1 Nms/rad is
    /// added because AVBD's fixed-PD motor has one combined damping channel.
    private static let mujocoPlaygroundMotorGains: [String: MJCFMotorGain] = [
        "left_hip_yaw": .init(stiffness: 800, damping: 26.398067),
        "left_hip_roll": .init(stiffness: 800, damping: 60.444678),
        "left_hip_pitch": .init(stiffness: 800, damping: 58.791980),
        "left_knee": .init(stiffness: 800, damping: 28.597559),
        "left_ankle": .init(stiffness: 800, damping: 19.341247),
        "right_hip_yaw": .init(stiffness: 800, damping: 26.398067),
        "right_hip_roll": .init(stiffness: 800, damping: 60.444678),
        "right_hip_pitch": .init(stiffness: 800, damping: 58.791980),
        "right_knee": .init(stiffness: 800, damping: 28.597559),
        "right_ankle": .init(stiffness: 800, damping: 19.341247),
        "torso": .init(stiffness: 400, damping: 31.782940),
        "left_shoulder_pitch": .init(stiffness: 400, damping: 22.282847),
        "left_shoulder_roll": .init(stiffness: 400, damping: 21.697023),
        "left_shoulder_yaw": .init(stiffness: 400, damping: 15.306674),
        "left_elbow": .init(stiffness: 400, damping: 15.034399),
        "right_shoulder_pitch": .init(stiffness: 400, damping: 22.282847),
        "right_shoulder_roll": .init(stiffness: 400, damping: 21.697023),
        "right_shoulder_yaw": .init(stiffness: 400, damping: 15.306674),
        "right_elbow": .init(stiffness: 400, damping: 15.034399),
    ]

    private static let unitreeRLGymMotorGains: [String: MJCFMotorGain] = [
        "left_hip_yaw": .init(stiffness: 150, damping: 2),
        "left_hip_roll": .init(stiffness: 150, damping: 2),
        "left_hip_pitch": .init(stiffness: 150, damping: 2),
        "left_knee": .init(stiffness: 200, damping: 4),
        "left_ankle": .init(stiffness: 40, damping: 2),
        "right_hip_yaw": .init(stiffness: 150, damping: 2),
        "right_hip_roll": .init(stiffness: 150, damping: 2),
        "right_hip_pitch": .init(stiffness: 150, damping: 2),
        "right_knee": .init(stiffness: 200, damping: 4),
        "right_ankle": .init(stiffness: 40, damping: 2),
    ]

    public let numEnvironments: Int
    public let controlProfile: HumanoidControlProfile
    public let projectileMass: Float
    public let solver: GPUSolver
    public let scene: PhysicsScene
    public let groundBody: Int
    public private(set) var refs: [EnvRefs] = []
    private let spawnPoses: [(F3, Quat)]
    /// body -> environment * 3 + (left foot, right foot, torso)
    private let groundContactSlots: [Int: Int]
    private let projectileOwners: [Int: Int]
    private let robotBodyOwners: [Int: Int]
    /// terminal hand body -> environment * 2 + (left, right)
    private let handContactSlots: [Int: Int]

    private static func addCarrySupport(
        to scene: inout PhysicsScene, size: F3, center: F3,
        friction: Float, includeLegs: Bool, collisionGroup: UInt32
    ) -> (top: Int, parts: [Int], offsets: [F3]) {
        let top = scene.addBody(
            size: size, density: 0, friction: friction,
            dynamicFriction: friction, position: center,
            collisionGroup: collisionGroup)
        guard includeLegs else { return (top, [], []) }

        let legHeight = center.z - 0.5 * size.z
        let legSize = F3(
            min(0.055, 0.2 * size.x), min(0.055, 0.2 * size.y),
            legHeight)
        let inset = F3(
            0.5 * (size.x - legSize.x) - 0.025,
            0.5 * (size.y - legSize.y) - 0.025, 0)
        let legZ = -0.5 * (size.z + legHeight)
        var parts = [Int]()
        var offsets = [F3]()
        for xSign: Float in [-1, 1] {
            for ySign: Float in [-1, 1] {
                let offset = F3(xSign * inset.x, ySign * inset.y, legZ)
                parts.append(scene.addBody(
                    size: legSize, density: 0, friction: friction,
                    dynamicFriction: friction, position: center + offset,
                    collisionGroup: collisionGroup))
                offsets.append(offset)
            }
        }
        return (top, parts, offsets)
    }

    public init(numEnvironments: Int, seed: UInt64 = 1,
                includeProjectile: Bool = false,
                projectileSize: Float = 0.25,
                projectileDimensions: F3? = nil,
                projectileMass: Float = 8,
                projectileFriction: Float = 0.7,
                carryPedestalSize: F3? = nil,
                carryPedestalCenter: F3 = F3(0.62, 0, 0.32),
                carryPedestalLegs: Bool = false,
                carryDestinationPedestalSize: F3? = nil,
                carryDestinationPedestalCenter: F3 = F3(1.40, 0, 0.32),
                carryDestinationPedestalLegs: Bool = false,
                preserveMinimalTerrainContactProfile: Bool = false,
                controlProfile: HumanoidControlProfile = .isaacLab,
                solverIterations: Int? = nil) throws {
        precondition(numEnvironments > 0)
        let projectileDimensions = projectileDimensions
            ?? F3(repeating: projectileSize)
        precondition(projectileSize > 0 && projectileMass > 0
            && projectileFriction >= 0
            && projectileDimensions.x > 0
            && projectileDimensions.y > 0
            && projectileDimensions.z > 0)
        if let carryPedestalSize {
            precondition(carryPedestalSize.x > 0
                && carryPedestalSize.y > 0
                && carryPedestalSize.z > 0)
            if carryPedestalLegs {
                precondition(carryPedestalCenter.z
                    > 0.5 * carryPedestalSize.z)
            }
        }
        if let carryDestinationPedestalSize {
            precondition(carryDestinationPedestalSize.x > 0
                && carryDestinationPedestalSize.y > 0
                && carryDestinationPedestalSize.z > 0)
            if carryDestinationPedestalLegs {
                precondition(carryDestinationPedestalCenter.z
                    > 0.5 * carryDestinationPedestalSize.z)
            }
        }
        precondition(!preserveMinimalTerrainContactProfile
            || (includeProjectile && controlProfile == .isaacLab))
        self.numEnvironments = numEnvironments
        self.controlProfile = controlProfile
        self.projectileMass = projectileMass
        var built = PhysicsScene(name: "humanoid-walk")
        switch controlProfile {
        case .mujocoPlayground: built.settings.dt = 0.004
        case .unitreeRLGym: built.settings.dt = 0.002
        case .isaacLab: built.settings.dt = 1 / 200
        }
        // Both public source environments use SI gravity. The generic AVBD
        // demo default is -10 m/s², but that 1.9% plant mismatch compounds in
        // a transferred 50 Hz feedback policy.
        built.settings.gravity = -9.81
        built.settings.iterations = solverIterations ?? 20
        built.settings.frictionCombineMode = controlProfile == .isaacLab
            ? .multiply : .geometricMean
        built.settings.betaLin = 20_000
        built.settings.betaAng = 400
        built.settings.lambdaMax = 1_200
        // Policy Replay should frame a single humanoid closely enough that
        // the fixed A/B ground gates and individual footfalls are legible.
        // Batched training is headless, so this has no simulation cost.
        built.settings.cameraDistance = 15.0
        built.settings.cameraTargetX = 6.5
        built.settings.cameraTargetY = 0
        built.settings.cameraTargetZ = 0.90
        built.settings.cameraAzimuth = -.pi / 2
        built.settings.cameraElevation = 0.16
        // Collision groups make replicas physically independent while every
        // robot occupies the exact same near-origin coordinates. This avoids
        // both cross-replica contacts and Float world-offset drift.
        let groundBody = built.addBody(
            size: F3(80, 40, 2),
            density: 0,
            friction: controlProfile == .mujocoPlayground ? 1.1 : 1.0,
            position: F3(10, 0, -1))
        let asset = try MJCFAsset.bundledUnitreeH1()
        var builtRefs = [EnvRefs]()
        builtRefs.reserveCapacity(numEnvironments)
        for e in 0..<numEnvironments {
            let center = F3.zero
            let firstReplicaCollider = built.colliders.count
            let motorGains: [String: MJCFMotorGain]
            let homePositions: [String: Float]
            switch controlProfile {
            case .isaacLab:
                motorGains = Self.isaacLabMotorGains
                homePositions = Self.isaacLabHomePositions
            case .mujocoPlayground:
                motorGains = Self.mujocoPlaygroundMotorGains
                homePositions = Self.mujocoPlaygroundHomePositions
            case .unitreeRLGym:
                motorGains = Self.unitreeRLGymMotorGains
                homePositions = [:]
            }
            let imported = try asset.instantiate(
                in: &built,
                options: MJCFInstantiationOptions(
                    worldOffset: center,
                    motorGains: motorGains,
                    jointHomePositions: homePositions,
                    // IsaacLab's H1 articulation configuration disables
                    // internal self-collision; terrain contacts remain live.
                    selfCollisions: false,
                    // The public H1 USD copies Menagerie's diagonal inertia
                    // and COM values but omits `physics:principalAxes`.
                    inertiaFrame: controlProfile == .isaacLab
                        ? .linkAligned : .principal))
            if controlProfile == .isaacLab {
                // H1_CFG uses simulation effort caps of 300 Nm for the legs,
                // torso, and arms, and 100 Nm for the ankles. The vendored
                // MuJoCo asset carries different hardware/control ranges, so
                // retaining those values would no longer be the Isaac
                // implicit-actuator contract even when Kp/Kd match.
                for (name, joint) in zip(
                    imported.actuatorNames, imported.actuatorJoints) {
                    built.joints[joint].motorTorque =
                        name.hasSuffix("_ankle") ? 100 : 300
                }
            }
            if controlProfile == .unitreeRLGym {
                // Unitree's deployment XML exposes only ten leg hinges; its
                // torso and arm meshes are rigidly attached to the pelvis.
                // Welding the corresponding full-H1 links preserves their
                // composite mass and visual geometry without inventing nine
                // policy outputs. The source deploy model also uses 0.01
                // kg*m^2 reflected armature on every leg hinge.
                for (index, joint) in imported.actuatorJoints.enumerated() {
                    if index < 10 {
                        built.joints[joint].armature = 0.01
                    } else {
                        built.joints[joint].hingeAxis = nil
                        built.joints[joint].motorTorque = 0
                        built.joints[joint].motorStiffness = 0
                        built.joints[joint].motorDamping = 0
                        built.joints[joint].armature = 0
                    }
                }
            }
            let leftFoot = imported.bodiesByName["left_ankle_link"]!
            let rightFoot = imported.bodiesByName["right_ankle_link"]!
            let torso = imported.bodiesByName["torso_link"]!
            let leftHand = imported.bodiesByName["left_elbow_link"]!
            let rightHand = imported.bodiesByName["right_elbow_link"]!
            let handTipInLink = F3(0.28, 0, -0.015)
            func handFrame(_ linkFrame: MJCFLinkFrame) -> MJCFLinkFrame {
                MJCFLinkFrame(
                    position: linkFrame.position
                        + linkFrame.rotation.act(handTipInLink),
                    rotation: linkFrame.rotation)
            }
            for collider in firstReplicaCollider..<built.colliders.count {
                built.colliders[collider].collisionGroup = UInt32(e + 1)
                let body = built.colliders[collider].body
                switch controlProfile {
                case .isaacLab:
                    // Nominal Flat replay retains Isaac Lab's H1_MINIMAL
                    // collision preset through the cooked hulls below.  A
                    // projectile scene instead enables every authored source
                    // primitive: pelvis, legs, shoulders, arms, forearms and
                    // the terminal hand spheres.  The importer already added
                    // body-pair exclusions for `selfCollisions: false`, so
                    // this expands real external contact without making
                    // neighboring robot links collide with one another.
                    if preserveMinimalTerrainContactProfile {
                        // A manual replay probe must not change the nominal
                        // H1_MINIMAL floor/contact trajectory before launch.
                        // Keep the cooked ankle/torso hulls below and expose
                        // every other authored primitive only to same-replica
                        // bodies such as the projectile, never shared terrain.
                        let coveredByMinimalHull = body == leftFoot
                            || body == rightFoot || body == torso
                        built.colliders[collider].collisionEnabled =
                            !coveredByMinimalHull
                        built.colliders[collider]
                            .collidesWithSharedGeometry = false
                    } else {
                        built.colliders[collider].collisionEnabled =
                            includeProjectile
                    }
                case .mujocoPlayground:
                    built.colliders[collider].collisionEnabled =
                        body == leftFoot || body == rightFoot
                case .unitreeRLGym:
                    // Unitree's MuJoCo deployment uses collision meshes on
                    // every leg and on the rigid upper-body compound.
                    built.colliders[collider].collisionEnabled = true
                }
            }
            if controlProfile == .isaacLab {
                // H1_MINIMAL_CFG authors one PhysX convex hull on each ankle
                // and the torso. The decoded vertices are in link frames;
                // MJCFInstantiation supplies the selected source-simulator
                // link->COM-body transform used for joints and inertia.
                let hulls: [(Int, MJCFLinkFrame, [F3])] = [
                    (leftFoot, imported.linkFramesInBody["left_ankle_link"]!,
                     IsaacH1CollisionHulls.leftAnkle),
                    (rightFoot, imported.linkFramesInBody["right_ankle_link"]!,
                     IsaacH1CollisionHulls.rightAnkle),
                    (torso, imported.linkFramesInBody["torso_link"]!,
                     IsaacH1CollisionHulls.torso),
                ]
                for (body, frame, vertices) in hulls {
                    _ = built.addConvexCollider(
                        body: body, vertices: vertices,
                        friction: 0.8, dynamicFriction: 0.6,
                        localPosition: frame.position,
                        localRotation: frame.rotation,
                        collisionGroup: UInt32(e + 1),
                        // Full-body projectile scenes use the complete set of
                        // rendered MJCF primitives above.  Do not layer the
                        // three H1_MINIMAL hulls over those same links: that
                        // would create duplicate contact manifolds and an
                        // artificially rigid impact response.
                        collisionEnabled: !includeProjectile
                            || preserveMinimalTerrainContactProfile,
                        isRendered: false)
                }
            }
            let importedBodies = asset.bodyNames.map {
                imported.bodiesByName[$0]!
            }
            // H1's source collision model puts a protective 5 cm sphere on
            // each upper calf. It is useful when a shin hits the terrain, but
            // without the source STL visual meshes rendering it as a literal
            // ball falsely looks like an extra knee/link. Keep both colliders
            // fully active and omit only their analytic debug visualization.
            for name in ["left_knee_link", "right_knee_link"] {
                let kneeBody = imported.bodiesByName[name]!
                for collider in built.colliders.indices
                    where built.colliders[collider].body == kneeBody
                        && built.colliders[collider].shape == .sphere {
                    built.colliders[collider].isRendered = false
                }
            }
            // Translate the complete articulation to the task's authored
            // root height while preserving every joint anchor exactly. The
            // two public profiles intentionally use different source poses.
            let verticalCorrection: Float
            if controlProfile == .mujocoPlayground {
                let pelvis = imported.bodiesByName["pelvis"]!
                let frame = imported.linkFramesInBody["pelvis"]!
                let linkPosition = built.bodies[pelvis].position
                    + built.bodies[pelvis].rotation.act(frame.position)
                // Exact free-root height in the public `home` keyframe.
                verticalCorrection = linkPosition.z - 0.97
            } else {
                let pelvis = imported.bodiesByName["pelvis"]!
                let frame = imported.linkFramesInBody["pelvis"]!
                let linkPosition = built.bodies[pelvis].position
                    + built.bodies[pelvis].rotation.act(frame.position)
                // Exact H1 Flat init-state root height. Do not derive this
                // from contact clearance: PhysX applies the crouched joint
                // pose at the authored 1.05 m root and the published policy
                // was trained through that initial contact transient.
                verticalCorrection = linkPosition.z - 1.05
            }
            for body in importedBodies {
                built.bodies[body].position.z -= verticalCorrection
            }
            var ref = EnvRefs(
                center: center,
                root: imported.bodiesByName["pelvis"]!,
                torso: imported.bodiesByName["torso_link"]!,
                leftFoot: imported.bodiesByName["left_ankle_link"]!,
                rightFoot: imported.bodiesByName["right_ankle_link"]!,
                leftHand: leftHand,
                rightHand: rightHand,
                rootFrame: imported.linkFramesInBody["pelvis"]!,
                torsoFrame: imported.linkFramesInBody["torso_link"]!,
                leftFootFrame: imported.linkFramesInBody["left_ankle_link"]!,
                rightFootFrame: imported.linkFramesInBody["right_ankle_link"]!,
                leftHandFrame: handFrame(
                    imported.linkFramesInBody["left_elbow_link"]!),
                rightHandFrame: handFrame(
                    imported.linkFramesInBody["right_elbow_link"]!),
                bodies: importedBodies,
                motors: imported.actuatorJoints,
                projectile: nil,
                carryPedestal: nil,
                carryPedestalParts: [],
                carryPedestalPartOffsets: [],
                carryDestinationPedestal: nil,
                carryDestinationPedestalParts: [],
                carryDestinationPedestalPartOffsets: [],
                startMarker: nil, goalMarker: nil)
            if includeProjectile {
                let density = projectileMass
                    / (projectileDimensions.x * projectileDimensions.y
                        * projectileDimensions.z)
                ref.projectile = built.addBody(
                    size: projectileDimensions,
                    density: density, friction: projectileFriction,
                    dynamicFriction: projectileFriction,
                    position: center + F3(0, 0, -4),
                    collisionGroup: UInt32(e + 1))
            }
            if let carryPedestalSize {
                let support = Self.addCarrySupport(
                    to: &built, size: carryPedestalSize,
                    center: center + carryPedestalCenter,
                    friction: projectileFriction, includeLegs: carryPedestalLegs,
                    collisionGroup: UInt32(e + 1))
                ref.carryPedestal = support.top
                ref.carryPedestalParts = support.parts
                ref.carryPedestalPartOffsets = support.offsets
            }
            if let carryDestinationPedestalSize {
                let support = Self.addCarrySupport(
                    to: &built, size: carryDestinationPedestalSize,
                    center: center + carryDestinationPedestalCenter,
                    friction: projectileFriction,
                    includeLegs: carryDestinationPedestalLegs,
                    collisionGroup: UInt32(e + 1))
                ref.carryDestinationPedestal = support.top
                ref.carryDestinationPedestalParts = support.parts
                ref.carryDestinationPedestalPartOffsets = support.offsets
            }
            if numEnvironments <= 4 {
                // Visible acceptance gates for Policy Replay: the learned
                // humanoid must move its root from A to its command-scaled B
                // under a fixed side camera. The task moves B to the exact
                // episode target after sampling the speed command.
                ref.startMarker = built.addBody(
                    size: F3(0.04, 0.40, 0.04), density: 0, friction: 0,
                    position: center + F3(0, -0.55, 0.02),
                    collisionEnabled: false)
                ref.goalMarker = built.addBody(
                    size: F3(0.04, 0.40, 0.04), density: 0, friction: 0,
                    position: center + F3(6.0, -0.55, 0.02),
                    collisionEnabled: false)
            }
            builtRefs.append(ref)
        }
        refs = builtRefs
        scene = built
        self.groundBody = groundBody
        var contactSlots = [Int: Int]()
        for (environment, ref) in builtRefs.enumerated() {
            contactSlots[ref.leftFoot] = environment * 3
            contactSlots[ref.rightFoot] = environment * 3 + 1
            contactSlots[ref.torso] = environment * 3 + 2
        }
        groundContactSlots = contactSlots
        var projectileOwners = [Int: Int]()
        var robotBodyOwners = [Int: Int]()
        var handContactSlots = [Int: Int]()
        for (environment, ref) in builtRefs.enumerated() {
            if let projectile = ref.projectile {
                projectileOwners[projectile] = environment
            }
            for body in ref.bodies { robotBodyOwners[body] = environment }
            handContactSlots[ref.leftHand] = environment * 2
            handContactSlots[ref.rightHand] = environment * 2 + 1
        }
        self.projectileOwners = projectileOwners
        self.robotBodyOwners = robotBodyOwners
        self.handContactSlots = handContactSlots
        spawnPoses = built.bodies.map { ($0.position, $0.rotation) }
        solver = try GPUSolver(scene: built)
        _ = seed
    }

    private static func footClearance(in scene: PhysicsScene, body: Int,
                                      frame: MJCFLinkFrame) -> Float {
        let state = scene.bodies[body]
        let linkPosition = state.position + state.rotation.act(frame.position)
        let linkRotation = (state.rotation * frame.rotation).normalized
        let lowestCenter = h1FootCapsuleEndpoints.reduce(Float.infinity) {
            min($0, (linkPosition + linkRotation.act($1)).z)
        }
        return lowestCenter - h1FootCapsuleRadius
    }

    /// Physical ground contacts from the solver's last manifold build. The
    /// task uses these for gait rewards and torso-contact termination; the
    /// geometric foot-clearance calculation remains diagnostics only.
    public func groundContacts() -> (feet: [[Bool]], torso: [Bool]) {
        var feet = [[Bool]](repeating: [false, false], count: numEnvironments)
        var torso = [Bool](repeating: false, count: numEnvironments)
        for (a, b) in solver.activeRigidContactPairs() {
            let other: Int
            if a == groundBody {
                other = b
            } else if b == groundBody {
                other = a
            } else {
                continue
            }
            guard let slot = groundContactSlots[other] else { continue }
            let environment = slot / 3
            switch slot % 3 {
            case 0: feet[environment][0] = true
            case 1: feet[environment][1] = true
            default: torso[environment] = true
            }
        }
        return (feet, torso)
    }

    /// Whether each environment's physical projectile touched any body in
    /// its own articulation during the last completed solver step. Ground
    /// contacts do not count, and collision groups prevent cross-replica hits.
    public func projectileRobotContacts() -> [Bool] {
        var contacts = [Bool](repeating: false, count: numEnvironments)
        guard !projectileOwners.isEmpty else { return contacts }
        for (a, b) in solver.activeRigidContactPairs() {
            if let environment = projectileOwners[a],
               robotBodyOwners[b] == environment {
                contacts[environment] = true
            } else if let environment = projectileOwners[b],
                      robotBodyOwners[a] == environment {
                contacts[environment] = true
            }
        }
        return contacts
    }

    /// Per-hand object contact from the solver's active physical manifolds.
    /// The signal is intentionally diagnostic/task state rather than a
    /// geometric distance proxy: a hand counts only after actual contact.
    public func boxHandContacts() -> (left: [Bool], right: [Bool]) {
        var left = [Bool](repeating: false, count: numEnvironments)
        var right = [Bool](repeating: false, count: numEnvironments)
        guard !projectileOwners.isEmpty else { return (left, right) }
        for (a, b) in solver.activeRigidContactPairs() {
            let slot: Int?
            if let environment = projectileOwners[a],
               let candidate = handContactSlots[b], candidate / 2 == environment {
                slot = candidate
            } else if let environment = projectileOwners[b],
                      let candidate = handContactSlots[a],
                      candidate / 2 == environment {
                slot = candidate
            } else {
                slot = nil
            }
            guard let slot else { continue }
            if slot.isMultiple(of: 2) { left[slot / 2] = true }
            else { right[slot / 2] = true }
        }
        return (left, right)
    }

    /// Physical support contacts for each task-owned box.  A carry task must
    /// not infer a drop from height alone: after the box has moved beyond a
    /// pedestal, its top plane no longer exists beneath the box.  Reporting
    /// the solver manifolds keeps drop termination tied to geometry that is
    /// actually present in the scene.
    public func boxCarrySupportContacts() -> (
        ground: [Bool], source: [Bool], destination: [Bool]
    ) {
        var ground = [Bool](repeating: false, count: numEnvironments)
        var source = [Bool](repeating: false, count: numEnvironments)
        var destination = [Bool](repeating: false, count: numEnvironments)
        guard !projectileOwners.isEmpty else {
            return (ground, source, destination)
        }
        for (a, b) in solver.activeRigidContactPairs() {
            let environment: Int
            let other: Int
            if let owner = projectileOwners[a] {
                environment = owner
                other = b
            } else if let owner = projectileOwners[b] {
                environment = owner
                other = a
            } else {
                continue
            }
            if other == groundBody {
                ground[environment] = true
            }
            if refs[environment].carryPedestal == other
                || refs[environment].carryPedestalParts.contains(other) {
                source[environment] = true
            }
            if refs[environment].carryDestinationPedestal == other
                || refs[environment].carryDestinationPedestalParts.contains(other) {
                destination[environment] = true
            }
        }
        return (ground, source, destination)
    }

    /// Compatibility view for tasks that only distinguish world support from
    /// unsupported flight. New transport tasks should use the source- and
    /// destination-specific contact API above.
    public func boxSupportContacts() -> (ground: [Bool], pedestal: [Bool]) {
        let contacts = boxCarrySupportContacts()
        return (
            contacts.ground,
            zip(contacts.source, contacts.destination).map { $0 || $1 })
    }

    private static func buildOne(_ s: inout PhysicsScene, center c: F3) -> EnvRefs {
        // Start from an anchor-consistent, mildly crouched home pose. Straight
        // knees put a position-controlled biped at a kinematic singularity:
        // small independent motor noise can only buckle the legs, which made
        // PPO's initial exploration fall backward instead of discovering a
        // recoverable step. This is a single static robot home pose (like an
        // MJCF keyframe), not a gait phase or a reference trajectory.
        let thighRotation = Quat(angle: -0.20, axis: F3(0, 1, 0))
        let shinRotation = Quat(angle: 0.20, axis: F3(0, 1, 0))
        let identity = Quat(real: 1, imag: .zero)
        // Solve the leg chain once with its hip at z=0, then translate it so
        // the 10 cm foot begins 5 mm above the ground. Every joint anchor is
        // constructed from the same chain below, eliminating reset impulses.
        let temporaryHip = F3.zero
        let temporaryThigh = temporaryHip - thighRotation.act(F3(0, 0, 0.19))
        let temporaryKnee = temporaryThigh + thighRotation.act(F3(0, 0, -0.19))
        let temporaryShin = temporaryKnee - shinRotation.act(F3(0, 0, 0.19))
        let temporaryShinBottom = temporaryShin + shinRotation.act(F3(0, 0, -0.19))
        let temporaryAnkle = temporaryShinBottom - shinRotation.act(F3(0, 0, 0.04))
        let temporaryFootAnchor = temporaryAnkle + shinRotation.act(F3(0, 0, -0.03))
        let temporaryFoot = temporaryFootAnchor - F3(-0.08, 0, 0.045)
        let neutralHipHeight: Float = 0.055 - temporaryFoot.z
        let pelvisPosition = c + F3(0, 0, neutralHipHeight + 0.07)
        let torsoPosition = pelvisPosition + F3(0, 0, 0.34)
        let headPosition = torsoPosition + F3(0, 0, 0.37)

        let pelvis = s.addBody(size: F3(0.34, 0.28, 0.20), density: 850,
                               friction: 0.8, position: pelvisPosition)
        let torso = s.addBody(size: F3(0.40, 0.25, 0.48), density: 360,
                              friction: 0.6, position: torsoPosition)
        let head = s.addSphere(diameter: 0.24, density: 420, friction: 0.5,
                               position: headPosition)
        // A tiny welded nose makes the physical +X/front direction visible in
        // replay. It is not observed or rewarded and its ~4 g mass is
        // negligible relative to the robot; unlike a floating UI glyph it
        // follows the exact simulated head pose.
        let nose = s.addSphere(diameter: 0.055, density: 50, friction: 0.2,
                               position: headPosition + F3(0.1475, 0, 0))
        s.addJoint(SceneJoint(bodyA: pelvis, bodyB: torso,
                              rA: F3(0, 0, 0.10), rB: F3(0, 0, -0.24),
                              stiffnessLin: .infinity, stiffnessAng: .infinity))
        s.addJoint(SceneJoint(bodyA: torso, bodyB: head,
                              rA: F3(0, 0, 0.25), rB: F3(0, 0, -0.12),
                              stiffnessLin: .infinity, stiffnessAng: .infinity))
        s.addJoint(SceneJoint(bodyA: head, bodyB: nose,
                              rA: F3(0.12, 0, 0), rB: F3(-0.0275, 0, 0),
                              stiffnessLin: .infinity, stiffnessAng: .infinity))

        var bodies = [pelvis, torso, head, nose]
        var motors = [Int]()
        func hinge(_ a: Int, _ b: Int, rA: F3, rB: F3,
                   axis: F3 = F3(0, 1, 0), torque: Float,
                   stiffness: Float, damping: Float,
                   limits: (Float, Float)) {
            motors.append(s.joints.count)
            s.addJoint(SceneJoint(bodyA: a, bodyB: b, rA: rA, rB: rB,
                                  stiffnessLin: .infinity, stiffnessAng: .infinity,
                                  hingeAxis: axis, motorTarget: 0,
                                  motorTorque: torque,
                                  motorStiffness: stiffness,
                                  motorDamping: damping,
                                  limitLo: limits.0, limitHi: limits.1))
        }
        func excludeCollision(_ a: Int, _ b: Int) {
            s.addJoint(SceneJoint(bodyA: a, bodyB: b, rA: .zero, rB: .zero,
                                  stiffnessLin: 0, stiffnessAng: 0))
        }

        var feet = [Int]()
        var hands = [Int]()
        for (sideIndex, side) in [Float(-1), 1].enumerated() {
            let y = side * 0.105
            let hipGimbalPosition = c + F3(0, y, neutralHipHeight)
            let hipYawPosition = hipGimbalPosition + F3(0, 0, 0.06)
            let thighPosition = hipGimbalPosition
                - thighRotation.act(F3(0, 0, 0.19))
            let kneePosition = thighPosition
                + thighRotation.act(F3(0, 0, -0.19))
            let shinPosition = kneePosition
                - shinRotation.act(F3(0, 0, 0.19))
            let shinBottom = shinPosition
                + shinRotation.act(F3(0, 0, -0.19))
            let anklePosition = shinBottom
                - shinRotation.act(F3(0, 0, 0.04))
            let footAnchor = anklePosition
                + shinRotation.act(F3(0, 0, -0.03))
            let footPosition = footAnchor - F3(-0.08, 0, 0.045)
            let hipYaw = s.addSphere(diameter: 0.10, density: 500,
                                     friction: 0.4,
                                     position: hipYawPosition)
            let hipGimbal = s.addSphere(diameter: 0.11, density: 500,
                                        friction: 0.4,
                                        position: hipGimbalPosition)
            let thigh = s.addCapsule(length: 0.38, radius: 0.075, density: 720,
                                     friction: 0.7, position: thighPosition,
                                     rotation: thighRotation)
            let shin = s.addCapsule(length: 0.38, radius: 0.065, density: 650,
                                    friction: 0.7, position: shinPosition,
                                    rotation: shinRotation)
            let ankleGimbal = s.addBody(
                size: F3(repeating: 0.09), density: 450, friction: 0.8,
                position: anklePosition, rotation: shinRotation, shape: .sphere)
            let foot = s.addBody(size: F3(0.34, 0.14, 0.10), density: 650,
                                 friction: 1.25, position: footPosition,
                                 rotation: identity)
            bodies.append(contentsOf: [hipYaw, hipGimbal, thigh, shin,
                                       ankleGimbal, foot])
            feet.append(foot)
            let base = sideIndex * 6
            hinge(pelvis, hipYaw, rA: F3(0, y, -0.01), rB: .zero,
                  axis: F3(0, 0, 1), torque: 160, stiffness: 400, damping: 20,
                  limits: jointRanges[base])
            hinge(hipYaw, hipGimbal, rA: F3(0, 0, -0.06), rB: .zero,
                  axis: F3(1, 0, 0), torque: 160, stiffness: 400, damping: 20,
                  limits: jointRanges[base + 1])
            hinge(hipGimbal, thigh, rA: .zero, rB: F3(0, 0, 0.19),
                  torque: 180, stiffness: 400, damping: 20,
                  limits: jointRanges[base + 2])
            hinge(thigh, shin, rA: F3(0, 0, -0.19), rB: F3(0, 0, 0.19),
                  torque: 150, stiffness: 400, damping: 20,
                  limits: jointRanges[base + 3])
            hinge(shin, ankleGimbal, rA: F3(0, 0, -0.19), rB: F3(0, 0, 0.04),
                  torque: 100, stiffness: 400, damping: 20,
                  limits: jointRanges[base + 4])
            hinge(ankleGimbal, foot, rA: F3(0, 0, -0.03),
                  rB: F3(-0.08, 0, 0.045), axis: F3(1, 0, 0),
                  torque: 90, stiffness: 400, damping: 20,
                  limits: jointRanges[base + 5])
            // Non-consecutive neighboring shapes overlap at the compact hip.
            // Inert joints are collision filters only; they exert no force.
            excludeCollision(pelvis, hipGimbal)
            excludeCollision(pelvis, thigh)
            excludeCollision(hipYaw, thigh)
        }

        for (sideIndex, side) in [Float(-1), 1].enumerated() {
            let shoulderAnchor = torsoPosition + F3(0, side * 0.15, 0.13)
            let upperPosition = shoulderAnchor - F3(0, 0, 0.145)
            let elbowAnchor = upperPosition + F3(0, 0, -0.145)
            let lowerPosition = elbowAnchor - F3(0, 0, 0.135)
            let upper = s.addCapsule(length: 0.29, radius: 0.055, density: 500,
                                     friction: 0.5, position: upperPosition)
            let lower = s.addCapsule(length: 0.27, radius: 0.045, density: 450,
                                     friction: 0.5, position: lowerPosition)
            bodies.append(contentsOf: [upper, lower])
            hands.append(lower)
            let base = 12 + sideIndex * 2
            hinge(torso, upper, rA: F3(0, side * 0.15, 0.13),
                  rB: F3(0, 0, 0.145), torque: 55,
                  stiffness: 40, damping: 10, limits: jointRanges[base])
            hinge(upper, lower, rA: F3(0, 0, -0.145), rB: F3(0, 0, 0.135),
                  torque: 40, stiffness: 40, damping: 10,
                  limits: jointRanges[base + 1])
            excludeCollision(torso, lower)
            excludeCollision(pelvis, upper)
            excludeCollision(pelvis, lower)
        }

        return EnvRefs(center: c, root: pelvis, torso: torso,
                       leftFoot: feet[0], rightFoot: feet[1],
                       leftHand: hands[0], rightHand: hands[1],
                       rootFrame: .init(position: .zero,
                                        rotation: Quat(real: 1, imag: .zero)),
                       torsoFrame: .init(position: .zero,
                                         rotation: Quat(real: 1, imag: .zero)),
                       leftFootFrame: .init(position: .zero,
                                           rotation: Quat(real: 1, imag: .zero)),
                       rightFootFrame: .init(position: .zero,
                                            rotation: Quat(real: 1, imag: .zero)),
                       leftHandFrame: .init(position: .zero,
                                           rotation: Quat(real: 1, imag: .zero)),
                       rightHandFrame: .init(position: .zero,
                                            rotation: Quat(real: 1, imag: .zero)),
                       bodies: bodies, motors: motors, projectile: nil,
                       carryPedestal: nil,
                       carryPedestalParts: [],
                       carryPedestalPartOffsets: [],
                       carryDestinationPedestal: nil,
                       carryDestinationPedestalParts: [],
                       carryDestinationPedestalPartOffsets: [],
                       startMarker: nil, goalMarker: nil)
    }

    /// Render-only destination markers. They are absent from batched training
    /// scenes and never enter observations, rewards, contacts, or success.
    public func setGoalMarkers(environmentIDs: [Int], directions: [F3],
                               distances: [Float], origins: [F3]? = nil) {
        precondition(environmentIDs.count == distances.count
            && environmentIDs.count == directions.count)
        precondition(origins == nil || origins?.count == environmentIDs.count)
        var poses = [GPUSolver.BodyPoseUpdate]()
        for (offset, e) in environmentIDs.enumerated() {
            let direction = directions[offset]
            let yaw = atan2(direction.y, direction.x)
            let rotation = Quat(angle: yaw, axis: F3(0, 0, 1))
            let markerOffset = F3(direction.y, -direction.x, 0) * 0.55
            let origin = origins?[offset] ?? refs[e].center
            if let marker = refs[e].startMarker {
                poses.append(.init(
                    body: marker,
                    position: origin + markerOffset + F3(0, 0, 0.02),
                    rotation: rotation))
            }
            guard let marker = refs[e].goalMarker else { continue }
            poses.append(.init(
                body: marker,
                position: origin + direction * distances[offset]
                    + markerOffset + F3(0, 0, 0.02),
                rotation: rotation))
        }
        if !poses.isEmpty { solver.setBodyPoses(poses) }
    }

    /// Launch one real rigid box per requested replica. Collision groups keep
    /// overlapping batched worlds independent; the box still contacts its own
    /// robot and the shared terrain through the ordinary Metal solver.
    public func throwProjectiles(environmentIDs: [Int], positions: [F3],
                                 velocities: [F3], angularVelocities: [F3]) {
        precondition(environmentIDs.count == positions.count
            && environmentIDs.count == velocities.count
            && environmentIDs.count == angularVelocities.count)
        var updates = [GPUSolver.BodyStateUpdate]()
        updates.reserveCapacity(environmentIDs.count)
        for (offset, e) in environmentIDs.enumerated() {
            guard let projectile = refs[e].projectile else { continue }
            updates.append(.init(
                body: projectile, position: positions[offset],
                rotation: Quat(real: 1, imag: .zero),
                linearVelocity: velocities[offset],
                angularVelocity: angularVelocities[offset]))
        }
        solver.setBodyStates(updates)
    }

    /// Place task-owned boxes and clear their velocities plus incident
    /// warm-start state.  Unlike `throwProjectiles`, this name states the
    /// reset semantics used by manipulation curricula.
    public func placeCarryBoxes(
        environmentIDs: [Int], positions: [F3], rotations: [Quat]? = nil,
        linearVelocities: [F3]? = nil, angularVelocities: [F3]? = nil
    ) {
        precondition(environmentIDs.count == positions.count)
        precondition(rotations == nil || rotations?.count == positions.count)
        precondition(linearVelocities == nil
            || linearVelocities?.count == positions.count)
        precondition(angularVelocities == nil
            || angularVelocities?.count == positions.count)
        let identity = Quat(real: 1, imag: .zero)
        solver.setBodyStates(environmentIDs.enumerated().compactMap { offset, e in
            refs[e].projectile.map { projectile in
                GPUSolver.BodyStateUpdate(
                    body: projectile, position: positions[offset],
                    rotation: rotations?[offset] ?? identity,
                    linearVelocity: linearVelocities?[offset] ?? .zero,
                    angularVelocity: angularVelocities?[offset] ?? .zero)
            }
        })
    }

    /// Move the static support and its dynamic box together at episode reset.
    /// The support remains ordinary collision geometry; this only authors the
    /// initial task layout before control begins.
    public func placeCarryStations(
        environmentIDs: [Int], pedestalCenters: [F3], boxCenters: [F3],
        destinationPedestalCenters: [F3]? = nil
    ) {
        precondition(environmentIDs.count == pedestalCenters.count
            && environmentIDs.count == boxCenters.count)
        precondition(destinationPedestalCenters == nil
            || destinationPedestalCenters?.count == environmentIDs.count)
        let identity = Quat(real: 1, imag: .zero)
        var poses = [GPUSolver.BodyPoseUpdate]()
        var boxes = [GPUSolver.BodyStateUpdate]()
        for (offset, e) in environmentIDs.enumerated() {
            if let pedestal = refs[e].carryPedestal {
                poses.append(.init(
                    body: pedestal, position: pedestalCenters[offset],
                    rotation: identity))
            }
            for (part, localOffset) in zip(
                refs[e].carryPedestalParts,
                refs[e].carryPedestalPartOffsets
            ) {
                poses.append(.init(
                    body: part,
                    position: pedestalCenters[offset] + localOffset,
                    rotation: identity))
            }
            if let destination = refs[e].carryDestinationPedestal,
               let destinationPedestalCenters {
                poses.append(.init(
                    body: destination,
                    position: destinationPedestalCenters[offset],
                    rotation: identity))
            }
            if let destinationPedestalCenters {
                for (part, localOffset) in zip(
                    refs[e].carryDestinationPedestalParts,
                    refs[e].carryDestinationPedestalPartOffsets
                ) {
                    poses.append(.init(
                        body: part,
                        position: destinationPedestalCenters[offset] + localOffset,
                        rotation: identity))
                }
            }
            if let box = refs[e].projectile {
                boxes.append(.init(
                    body: box, position: boxCenters[offset],
                    rotation: identity))
            }
        }
        if !poses.isEmpty { solver.setBodyPoses(poses) }
        if !boxes.isEmpty { solver.setBodyStates(boxes) }
    }

    /// Launch reusable replay boxes from robot-local left/right toward the
    /// measured moving torso. This is an external disturbance only: no robot
    /// pose, joint target, observation, or policy action is modified.
    public func throwBoxes(
        environmentIDs: [Int], sideSigns: [Float],
        launchDistance: Float = 1.2, speed: Float = 6
    ) {
        precondition(environmentIDs.count == sideSigns.count)
        precondition(Set(environmentIDs).count == environmentIDs.count
            && environmentIDs.allSatisfy(refs.indices.contains))
        precondition(launchDistance.isFinite && launchDistance > 0
            && speed.isFinite && speed > 0
            && sideSigns.allSatisfy { $0.isFinite && $0 != 0 })
        let measured = states()
        let flightTime = launchDistance / speed
        let gravity = F3(0, 0, scene.settings.gravity)
        var positions = [F3]()
        var velocities = [F3]()
        var angularVelocities = [F3]()
        positions.reserveCapacity(environmentIDs.count)
        velocities.reserveCapacity(environmentIDs.count)
        angularVelocities.reserveCapacity(environmentIDs.count)
        for (offset, environment) in environmentIDs.enumerated() {
            let state = measured[environment]
            let heading = state.root.rotation.act(F3(1, 0, 0))
            let headingLength = max(
                sqrt(heading.x * heading.x + heading.y * heading.y), 1e-6)
            let forward = F3(
                heading.x / headingLength, heading.y / headingLength, 0)
            let lateral = F3(-forward.y, forward.x, 0)
            let side: Float = sideSigns[offset] > 0 ? 1 : -1
            let target = state.torso.position
            let launch = target + lateral * (launchDistance * side)
                + forward * 0.15
            let predictedTarget = target
                + state.torso.linearVelocity * flightTime
            positions.append(launch)
            velocities.append((predictedTarget - launch
                - 0.5 * gravity * flightTime * flightTime) / flightTime)
            angularVelocities.append(F3(
                side * 2.5, -side * 1.5, side * 3.5))
        }
        throwProjectiles(
            environmentIDs: environmentIDs, positions: positions,
            velocities: velocities, angularVelocities: angularVelocities)
    }

    /// Park selected replay boxes below their replica and clear all velocity
    /// and incident warm-start state through the solver's reset API.
    public func hideBoxes(environmentIDs: [Int]) {
        precondition(Set(environmentIDs).count == environmentIDs.count
            && environmentIDs.allSatisfy(refs.indices.contains))
        solver.setBodyStates(environmentIDs.compactMap { environment in
            refs[environment].projectile.map { projectile in
                GPUSolver.BodyStateUpdate(
                    body: projectile,
                    position: refs[environment].center + F3(0, 0, -4),
                    rotation: Quat(real: 1, imag: .zero))
            }
        })
    }

    public func boxRobotContacts() -> [Bool] {
        projectileRobotContacts()
    }

    public func step(normalizedActions: ContiguousArray<Float>, decimation: Int,
                     clampActions: Bool = true,
                     clampTargetsToLimits: Bool = true) {
        precondition(normalizedActions.count == numEnvironments * Self.jointRanges.count)
        var commands = [GPUSolver.MotorTargetUpdate]()
        commands.reserveCapacity(normalizedActions.count)
        for e in 0..<numEnvironments {
            for j in 0..<Self.jointRanges.count {
                let rawAction = normalizedActions[
                    e * Self.jointRanges.count + j]
                let a = clampActions
                    ? simd_clamp(rawAction, -1, 1) : rawAction
                // Home-pose rebasing shifts source joint limits differently
                // for the Isaac and MuJoCo profiles. Read the instantiated
                // joint limits instead of applying one static approximation
                // to both profiles.
                let joint = scene.joints[refs[e].motors[j]]
                let requestedTarget = Self.defaultJointPositions[j]
                    + a * Self.actionScales[j]
                let target = clampTargetsToLimits
                    ? simd_clamp(requestedTarget,
                                 joint.limitLo, joint.limitHi)
                    : requestedTarget
                commands.append(.init(joint: refs[e].motors[j], angle: target))
            }
        }
        solver.setMotorTargets(commands)
        for _ in 0..<decimation { solver.step() }
    }

    /// Step absolute source-joint position targets. Imported policies use
    /// their own offset/scale contract and must not be remapped through the
    /// native task's normalized 0.5-radian action convention.
    public func step(jointPositionTargets: ContiguousArray<Float>,
                     decimation: Int, clampTargetsToLimits: Bool = true) {
        precondition(jointPositionTargets.count
            == numEnvironments * Self.jointRanges.count)
        precondition(decimation > 0)
        var commands = [GPUSolver.MotorTargetUpdate]()
        commands.reserveCapacity(jointPositionTargets.count)
        for environment in 0..<numEnvironments {
            for jointOffset in 0..<Self.jointRanges.count {
                let jointIndex = refs[environment].motors[jointOffset]
                let requested = jointPositionTargets[
                    environment * Self.jointRanges.count + jointOffset]
                let joint = scene.joints[jointIndex]
                let target = clampTargetsToLimits
                    ? simd_clamp(requested, joint.limitLo, joint.limitHi)
                    : requested
                commands.append(.init(joint: jointIndex, angle: target))
            }
        }
        solver.setMotorTargets(commands)
        for _ in 0..<decimation { solver.step() }
    }

    public func reset(_ environmentIDs: [Int], seeds: [UInt64],
                      initialRollPitchRange: Float = 0,
                      initialYawRange: Float = 0) {
        precondition(environmentIDs.count == seeds.count)
        var poses = [GPUSolver.BodyPoseUpdate]()
        var motors = [GPUSolver.MotorTargetUpdate]()
        for (i, e) in environmentIDs.enumerated() {
            // Use an independent stream from the task command sampler so a
            // particular requested speed is not correlated with a particular
            // tilt. Rotating every body about the pelvis keeps the articulated
            // reset exactly joint-consistent and avoids projection impulses.
            var rng = SplitMix64(seed: seeds[i] ^ 0xD1B54A32D192ED03)
            let roll = (2 * rng.nextFloat() - 1)
                * initialRollPitchRange
            let pitch = (2 * rng.nextFloat() - 1)
                * initialRollPitchRange
            let yaw = (2 * rng.nextFloat() - 1)
                * initialYawRange
            let perturbation = (
                Quat(angle: yaw, axis: F3(0, 0, 1))
                    * Quat(angle: pitch, axis: F3(0, 1, 0))
                    * Quat(angle: roll, axis: F3(1, 0, 0))).normalized
            let pivot = spawnPoses[refs[e].root].0
            for body in refs[e].bodies {
                let spawn = spawnPoses[body]
                poses.append(.init(body: body,
                                   position: pivot
                                       + perturbation.act(spawn.0 - pivot),
                                   rotation: (perturbation
                                       * spawn.1).normalized))
            }
            if let projectile = refs[e].projectile {
                let spawn = spawnPoses[projectile]
                poses.append(.init(body: projectile, position: spawn.0,
                                   rotation: spawn.1))
            }
            for (j, joint) in refs[e].motors.enumerated() {
                motors.append(.init(joint: joint,
                                    angle: Self.defaultJointPositions[j]))
            }
        }
        solver.setBodyPoses(poses)
        solver.setMotorTargets(motors)
    }

    public func states() -> [HumanoidState] {
        var bodyIDs = [Int]()
        var jointIDs = [Int]()
        bodyIDs.reserveCapacity(numEnvironments * 4)
        jointIDs.reserveCapacity(numEnvironments * Self.jointRanges.count)
        for r in refs {
            bodyIDs.append(contentsOf: [r.root, r.torso, r.leftFoot, r.rightFoot])
            jointIDs.append(contentsOf: r.motors)
        }
        let bodies = solver.bodyStates(bodyIDs)
        let jointStates = solver.motorStates(jointIDs)
        return (0..<numEnvironments).map { e in
            let b = e * 4
            let j = e * Self.jointRanges.count
            let r = refs[e]
            return HumanoidState(root: Self.linkState(bodies[b], r.rootFrame),
                                 torso: Self.linkState(bodies[b + 1], r.torsoFrame),
                                 leftFoot: Self.linkState(bodies[b + 2], r.leftFootFrame),
                                 rightFoot: Self.linkState(bodies[b + 3], r.rightFootFrame),
                                 jointAngles: jointStates[
                                    j..<(j + Self.jointRanges.count)].map(\.angle),
                                 jointVelocities: jointStates[
                                    j..<(j + Self.jointRanges.count)].map(\.velocity))
        }
    }

    public func boxStates() -> [GPUSolver.RigidBodyState] {
        precondition(refs.allSatisfy { $0.projectile != nil },
                     "boxStates requires one task-owned box per environment")
        return solver.bodyStates(refs.map { $0.projectile! })
    }

    public func manipulationStates() -> [HumanoidManipulationState] {
        precondition(refs.allSatisfy { $0.projectile != nil },
                     "manipulationStates requires one box per environment")
        var bodyIDs = [Int]()
        bodyIDs.reserveCapacity(numEnvironments * 3)
        for ref in refs {
            bodyIDs.append(ref.projectile!)
            bodyIDs.append(ref.leftHand)
            bodyIDs.append(ref.rightHand)
        }
        let bodies = solver.bodyStates(bodyIDs)
        return (0..<numEnvironments).map { e in
            let base = e * 3
            return HumanoidManipulationState(
                object: bodies[base],
                leftHand: Self.linkState(
                    bodies[base + 1], refs[e].leftHandFrame),
                rightHand: Self.linkState(
                    bodies[base + 2], refs[e].rightHandFrame))
        }
    }

    private static func linkState(_ body: GPUSolver.RigidBodyState,
                                  _ frame: MJCFLinkFrame) -> GPUSolver.RigidBodyState {
        let offset = body.rotation.act(frame.position)
        return GPUSolver.RigidBodyState(
            position: body.position + offset,
            rotation: (body.rotation * frame.rotation).normalized,
            linearVelocity: body.linearVelocity + cross(body.angularVelocity, offset),
            angularVelocity: body.angularVelocity)
    }
}

/// Velocity-command locomotion task. Training is on-policy and reference-free.
/// Its locomotion objective is exact world-space progress, with the standard
/// contact air-time, upright, action-rate, and failure terms used by modern
/// legged-robot PPO tasks.
public final class HumanoidWalkTask: VectorizedRLTask, RLEvaluationCriteriaProviding,
                                    TrainingModeConfigurable, PolicySymmetryProviding,
                                    ObservationNormalizerTransferProviding,
                                    ObservationSchemaTransferProviding,
                                    PolicyStandExpertGateProviding {
    /// A short causal window filters AVBD projection jitter without delaying
    /// the locomotion signal across most of PPO's rollout. At the default
    /// 50 Hz control rate this is 0.1 s; the previous 0.5 s window consumed
    /// most of a 24-step rollout and made forward credit nearly non-local.
    private static let velocityWindowSteps = 5
    // A short causal history supplies acceleration/contact context without a
    // gait clock, reference pose, recurrent hidden state, or privileged
    // contacts.
    private static let observationHistorySteps = 9
    private static let observationFrameDimension =
        13 + 3 * HumanoidWalkEnv.jointRanges.count
    /// Training-only command curriculum. Evaluation and replay always sample
    /// the full configured range. The ramp is expressed in task control steps
    /// so its behavior is independent of PPO rollout/minibatch choices.
    public static let curriculumStartMinimumSpeed: Float = 0.20
    public static let curriculumStartMaximumSpeed: Float = 0.35
    private static let footContactClearance: Float = 0.025
    private static let h1FootCapsuleEndpoints: [F3] = [
        F3(-0.035, 0, -0.056), F3(0.020, 0, -0.045),
        F3(0.115, 0, -0.056),
        F3(0.140, -0.030, -0.056), F3(0.140, 0.030, -0.056),
    ]
    private static let h1FootCapsuleRadius: Float = 0.014
    public let spec: RLTaskSpec
    public let environment: HumanoidWalkEnv
    public let configuration: HumanoidWalkTaskConfig
    public let usesPointGoal: Bool
    public var evaluationCriteria: RLEvaluationCriteria {
        if usesPointGoal {
            let minimumAlternatingSteps: Float =
                configuration.minimumGoalDistanceMeters > 0
                ? max(8, 1.5 * configuration.minimumGoalDistanceMeters)
                : 15
            var minimumTaskMetrics: [String: Float] = [
                "episode/goal_reached": 0.70,
                // A metric route can terminate after a successful dwell, so
                // its gait requirement scales with the shortest commanded
                // route instead of inheriting the legacy 9--13 m threshold.
                "episode/alternating_steps": minimumAlternatingSteps,
                "episode/goal_front_success_rate": 0.75,
            ]
            if configuration.maximumGoalDirectionAngle > .pi / 4 + 1e-5 {
                minimumTaskMetrics["episode/goal_side_success_rate"] = 0.60
            }
            if configuration.maximumGoalDirectionAngle > 3 * .pi / 4 + 1e-5 {
                minimumTaskMetrics["episode/goal_rear_success_rate"] = 0.50
            }
            if configuration.projectileProbability < 1 {
                minimumTaskMetrics["episode/nominal_success_rate"] = 0.75
            }
            if configuration.projectileProbability > 0 {
                minimumTaskMetrics["episode/disturbed_success_rate"] = 0.50
            }
            return RLEvaluationCriteria(
                minimumSuccessRate: 0.70,
                // Successful navigation terminates after a stable dwell; a
                // short episode is therefore desirable, not evidence of a
                // fall. The dwell, gait, and geometric gates prevent an
                // immediate-termination exploit.
                minimumMeanEpisodeLengthFraction: 0.25,
                minimumTaskMetrics: minimumTaskMetrics,
                maximumTaskMetrics: [
                    "episode/goal_distance_error_m":
                        configuration.goalSlowdownDistance,
                ])
        }
        if configuration.standingCommandProbability > 0 {
            return RLEvaluationCriteria(
                minimumSuccessRate: 0.85,
                minimumMeanEpisodeLengthFraction: 0.90,
                minimumTaskMetrics: [
                    "episode/standing_success_rate": 0.80,
                    "episode/moving_success_rate": 0.90,
                    "episode/heading_alignment": 0.85,
                ],
                maximumTaskMetrics: [
                    "episode/speed_error_mps": 0.10,
                    "episode/lateral_distance_m":
                        HumanoidLocomotionObjective.maximumLateralDrift,
                ])
        }
        let meanCommand = 0.5 * (configuration.minimumCommandSpeed
            + configuration.maximumCommandSpeed)
        let duration = Float(configuration.maxEpisodeSteps) * spec.controlStep
        return RLEvaluationCriteria(
            minimumSuccessRate: 0.90,
            minimumMeanEpisodeLengthFraction: 0.90,
            minimumTaskMetrics: [
                "episode/forward_distance_m": 0.50 * meanCommand * duration,
                "episode/forward_speed_mps": 0.50 * meanCommand,
                "episode/heading_alignment": 0.85,
                "episode/alternating_steps": Float(HumanoidLocomotionObjective.minimumAlternatingSteps),
            ],
            maximumTaskMetrics: [
                "episode/speed_error_mps": 0.10,
                "episode/lateral_distance_m": HumanoidLocomotionObjective.maximumLateralDrift,
            ])
    }

    private let actionDimension = HumanoidWalkEnv.jointRanges.count
    private var commands: [Float]
    private var previousActions: ContiguousArray<Float>
    private var observationHistory: [[Float]]
    private var goalLateralVelocityHistory: [[Float]]
    private var observationHistoryInitialized: [Bool]
    private var previousJointAngles: [[Float]]
    private var jointVelocities: [[Float]]
    private var previousRootPositions: [F3]
    private var stepRootDisplacements: [F3]
    private var rootPositionHistory: [[F3]]
    private var rootHistoryIndex = 0
    private var measuredRootVelocities: [F3]
    private var episodeLengths: [Int]
    private var episodeReturns: [Float]
    private var episodeStartPositions: [F3]
    private var goalDirections: [F3]
    private var goalDistances: [Float]
    private struct GoalOverride {
        var direction: F3
        var distance: Float
    }
    /// Optional operator-selected goals used by replay and evaluation tools.
    /// Training never installs these overrides, so its reset distribution is
    /// unchanged. Applying an override through `reset` also rebuilds the
    /// policy's causal observation history from the selected command.
    private var goalOverrides: [GoalOverride?]
    private var goalDwellStepCounts: [Int]
    private var maximumGoalDwellStepCounts: [Int]
    /// Arrival diagnostics are episode state rather than terminal-position
    /// guesses. A fast fly-through can finish outside the radius, while a
    /// later fall can hide the speed at first entry; both cases matter when
    /// deciding whether PPO needs better braking or better balance.
    private var goalRadiusEntered: [Bool]
    private var goalRadiusEntrySpeeds: [Float]
    private var goalRadiusEntryForwardSpeeds: [Float]
    private var goalRadiusEntryLateralSpeeds: [Float]
    private var goalRadiusExitedAfterEntry: [Bool]
    private var minimumGoalRadiusSpeeds: [Float]
    private var goalRadiusInsideStepCounts: [Int]
    private var projectileLaunchSteps: [Int]
    private var projectileSpeeds: [Float]
    private var projectileSides: [Float]
    private var projectileLaunched: [Bool]
    private var disturbedEpisodes: [Bool]
    private var minimumRootHeights: [Float]
    private var headingAlignmentSums: [Float]
    private var maximumLateralDrifts: [Float]
    private var footAirTimes: [[Float]]
    private var footContactTimes: [[Float]]
    private var previousFootContacts: [[Bool]]
    private var previousFootPositions: [[F3]]
    private var lastTouchdownFoot: [Int]
    private var alternatingStepCounts: [Int]
    private var singleSupportStepCounts: [Int]
    private var doubleSupportStepCounts: [Int]
    private var flightStepCounts: [Int]
    private var maximumFootClearances: [[Float]]
    private var resetRNG: SplitMix64
    private var trainingMode = false
    private var trainingControlSteps = 0
    private var randomizeNextTrainingResetAges = false

    public init(configuration: HumanoidWalkTaskConfig,
                taskID: String = "humanoid-walk-v0",
                taskRevision: Int = 35) throws {
        guard configuration.numEnvironments > 0,
              configuration.maxEpisodeSteps > 0,
              configuration.controlDecimation > 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "environment count, episode length, and decimation must be positive")
        }
        guard configuration.minimumCommandSpeed >= 0,
              configuration.maximumCommandSpeed >= configuration.minimumCommandSpeed,
              configuration.commandCurriculumControlSteps >= 0,
              configuration.trainingInitialEpisodeAgeFraction >= 0,
              configuration.trainingInitialEpisodeAgeFraction <= 1,
              configuration.standingCommandProbability >= 0,
              configuration.standingCommandProbability <= 1,
              configuration.standingCommandCurriculumControlSteps >= 0,
              configuration.expertGateCommandSpeed > 0,
              configuration.expertGateBlendWidth >= 0,
              configuration.expertGateBlendWidth
                <= configuration.expertGateCommandSpeed,
              configuration.standExpertBlendStartSpeed >= 0,
              configuration.standExpertBlendWidth >= 0,
              ((configuration.standExpertBlendStartSpeed == 0
                    && configuration.standExpertBlendWidth == 0)
                || (configuration.standExpertBlendStartSpeed > 0
                    && configuration.standExpertBlendWidth > 0
                    && configuration.standExpertBlendWidth
                        <= configuration.standExpertBlendStartSpeed)),
              configuration.velocityTrackingStandardDeviation > 0,
              configuration.velocityTrackingErrorPenaltyWeight >= 0,
              configuration.standStillVelocityPenaltyWeight >= 0,
              configuration.standStillJointDeviationPenaltyWeight >= 0,
              configuration.standStillDoubleSupportRewardWeight >= 0,
              configuration.standStillFallPenalty >= 0,
              configuration.lateralPenaltyWarmupControlSteps >= 0,
              configuration.lateralPenaltyRampControlSteps >= 0,
              configuration.laneTrackingStandardDeviation > 0,
              configuration.alternatingTouchdownRewardWeight >= 0,
              configuration.flightPenaltyWeight >= 0,
              configuration.initialRollPitchRange >= 0,
              configuration.initialRollPitchRange <= 0.10,
              configuration.initialYawRange >= 0,
              configuration.initialYawRange <= 0.35,
              configuration.maximumGoalDirectionAngle >= 0,
              configuration.maximumGoalDirectionAngle <= .pi,
              configuration.initialGoalDirectionAngle >= 0,
              configuration.initialGoalDirectionAngle
                <= configuration.maximumGoalDirectionAngle,
              configuration.goalDirectionCurriculumControlSteps >= 0,
              configuration.goalRadius > 0,
              configuration.goalSlowdownDistance > configuration.goalRadius,
              configuration.minimumGoalDistanceMeters >= 0,
              configuration.maximumGoalDistanceMeters >= 0,
              ((configuration.minimumGoalDistanceMeters == 0
                    && configuration.maximumGoalDistanceMeters == 0)
                || (configuration.minimumGoalDistanceMeters
                        > configuration.goalRadius
                    && configuration.maximumGoalDistanceMeters
                        >= configuration.minimumGoalDistanceMeters)),
              configuration.initialGoalDistanceScale > 0,
              configuration.initialGoalDistanceScale <= 1,
              configuration.goalDistanceCurriculumControlSteps >= 0,
              configuration.goalDwellSteps > 0,
              configuration.maximumGoalArrivalSpeed > 0,
              configuration.goalBoundaryCommandSpeed >= 0,
              configuration.goalBoundaryCommandSpeed
                <= configuration.maximumGoalArrivalSpeed,
              configuration.goalStableDwellRewardWeight >= 0,
              configuration.projectileProbability >= 0,
              configuration.projectileProbability <= 1,
              configuration.projectileCurriculumControlSteps >= 0,
              configuration.minimumProjectileSpeed > 0,
              configuration.maximumProjectileSpeed
                >= configuration.minimumProjectileSpeed,
              configuration.minimumProjectileLaunchStep >= 0,
              configuration.maximumProjectileLaunchStep
                >= configuration.minimumProjectileLaunchStep,
              (configuration.projectileProbability == 0
                || configuration.maximumProjectileLaunchStep
                    < configuration.maxEpisodeSteps),
              configuration.actionTargetResponse > 0,
              configuration.actionTargetResponse <= 1 else {
            throw RLEnvironmentError.invalidConfiguration(
                "invalid humanoid command or actuator configuration")
        }
        guard taskID == "humanoid-walk-v0" || taskID == "humanoid-goal-v0" else {
            throw RLEnvironmentError.invalidConfiguration(
                "unsupported humanoid locomotion task id \(taskID)")
        }
        let pointGoal = taskID == "humanoid-goal-v0"
        if configuration.threeModeActor
            && !configuration.commandGatedActor {
            throw RLEnvironmentError.invalidConfiguration(
                "three-mode actor requires a command gate")
        }
        if configuration.freezeLowSpeedPolicyExpert
            && (!configuration.commandGatedActor
                || !configuration.threeModeActor) {
            throw RLEnvironmentError.invalidConfiguration(
                "freezing the low-speed expert requires a three-mode actor")
        }
        if configuration.standExpertBlendStartSpeed > 0
            && (!pointGoal || !configuration.commandGatedActor
                || !configuration.threeModeActor) {
            throw RLEnvironmentError.invalidConfiguration(
                "stand-expert speed blending requires a three-mode point-goal actor")
        }
        if configuration.standExpertRequiresDoubleSupport
            && (!pointGoal || !configuration.commandGatedActor
                || !configuration.threeModeActor) {
            throw RLEnvironmentError.invalidConfiguration(
                "contact-conditioned standing requires a three-mode point-goal actor")
        }
        if configuration.standExpertUsesPlanarSpeed
            && (!pointGoal || !configuration.commandGatedActor
                || !configuration.threeModeActor) {
            throw RLEnvironmentError.invalidConfiguration(
                "planar-speed standing requires a three-mode point-goal actor")
        }
        if configuration.goalObservationUsesLateralVelocity && !pointGoal {
            throw RLEnvironmentError.invalidConfiguration(
                "lateral goal-frame velocity requires a point-goal task")
        }
        if configuration.goalObservationIncludesLateralVelocity && !pointGoal {
            throw RLEnvironmentError.invalidConfiguration(
                "appended lateral velocity requires a point-goal task")
        }
        if configuration.goalObservationUsesLateralVelocity
            && configuration.goalObservationIncludesLateralVelocity {
            throw RLEnvironmentError.invalidConfiguration(
                "point-goal lateral velocity cannot replace and append the slot")
        }
        if configuration.trainBasePolicyExpert
            && (!configuration.commandGatedActor || pointGoal) {
            throw RLEnvironmentError.invalidConfiguration(
                "base-expert training requires a gated straight locomotion task")
        }
        if !pointGoal && (configuration.maximumGoalDirectionAngle != 0
            || configuration.projectileProbability != 0) {
            throw RLEnvironmentError.invalidConfiguration(
                "straight humanoid-walk-v0 cannot enable goal angles or projectiles")
        }
        let env = try HumanoidWalkEnv(
            numEnvironments: configuration.numEnvironments,
            seed: configuration.seed,
            includeProjectile: configuration.projectileProbability > 0)
        environment = env
        self.configuration = configuration
        usesPointGoal = pointGoal
        var configurationValues: [String: Float] = [
            "minimumCommandSpeed": configuration.minimumCommandSpeed,
            "maximumCommandSpeed": configuration.maximumCommandSpeed,
            "commandCurriculumControlSteps":
                Float(configuration.commandCurriculumControlSteps),
            "lateralPenaltyWarmupControlSteps":
                Float(configuration.lateralPenaltyWarmupControlSteps),
            "lateralPenaltyRampControlSteps":
                Float(configuration.lateralPenaltyRampControlSteps),
            "laneTrackingStandardDeviation":
                configuration.laneTrackingStandardDeviation,
            "alternatingTouchdownRewardWeight":
                configuration.alternatingTouchdownRewardWeight,
            "flightPenaltyWeight": configuration.flightPenaltyWeight,
            "initialRollPitchRange": configuration.initialRollPitchRange,
            "initialYawRange": configuration.initialYawRange,
            "actionTargetResponse": configuration.actionTargetResponse,
        ]
        if configuration.trainingInitialEpisodeAgeFraction > 0 {
            configurationValues["trainingInitialEpisodeAgeFraction"] =
                configuration.trainingInitialEpisodeAgeFraction
        }
        if configuration.minimumCommandSpeed < 0.20
            || configuration.standingCommandProbability > 0
            || configuration.commandGatedActor {
            configurationValues["standingCommandProbability"] =
                configuration.standingCommandProbability
            configurationValues["standStillVelocityPenaltyWeight"] =
                configuration.standStillVelocityPenaltyWeight
            configurationValues["standStillJointDeviationPenaltyWeight"] =
                configuration.standStillJointDeviationPenaltyWeight
            configurationValues["standStillDoubleSupportRewardWeight"] =
                configuration.standStillDoubleSupportRewardWeight
            if configuration.standStillFallPenalty > 0 {
                configurationValues["standStillFallPenalty"] =
                    configuration.standStillFallPenalty
            }
        }
        if configuration.standingCommandCurriculumControlSteps > 0 {
            configurationValues["standingCommandCurriculumControlSteps"] =
                Float(configuration.standingCommandCurriculumControlSteps)
        }
        if configuration.commandGatedActor {
            configurationValues["commandGatedActor"] = 1
            if configuration.threeModeActor {
                configurationValues["threeModeActor"] = 1
            }
            configurationValues["expertGateCommandSpeed"] =
                configuration.expertGateCommandSpeed
            if configuration.expertGateBlendWidth > 0 {
                configurationValues["expertGateBlendWidth"] =
                    configuration.expertGateBlendWidth
            }
            if configuration.standExpertBlendStartSpeed > 0 {
                configurationValues["standExpertBlendStartSpeed"] =
                    configuration.standExpertBlendStartSpeed
                configurationValues["standExpertBlendWidth"] =
                    configuration.standExpertBlendWidth
            }
            if configuration.standExpertRequiresDoubleSupport {
                configurationValues["standExpertRequiresDoubleSupport"] = 1
            }
            if configuration.standExpertUsesPlanarSpeed {
                configurationValues["standExpertUsesPlanarSpeed"] = 1
            }
            if pointGoal && configuration.freezeBasePolicyExpert {
                configurationValues["freezeBasePolicyExpert"] = 1
            }
            if configuration.freezeLowSpeedPolicyExpert {
                configurationValues["freezeLowSpeedPolicyExpert"] = 1
            }
            if configuration.trainBasePolicyExpert {
                configurationValues["trainBasePolicyExpert"] = 1
            }
        }
        if configuration.velocityTrackingStandardDeviation
            != HumanoidLocomotionObjective.velocityTrackingStandardDeviation {
            configurationValues["velocityTrackingStandardDeviation"] =
                configuration.velocityTrackingStandardDeviation
        }
        if configuration.velocityTrackingErrorPenaltyWeight > 0 {
            configurationValues["velocityTrackingErrorPenaltyWeight"] =
                configuration.velocityTrackingErrorPenaltyWeight
        }
        if pointGoal {
            configurationValues["maximumGoalDirectionAngle"] =
                configuration.maximumGoalDirectionAngle
            // Omit the legacy-equivalent zero so revision-1 checkpoints made
            // before staged transfers remain exactly evaluable. Non-zero
            // starts are serialized and therefore cannot be silently resumed
            // under a different curriculum.
            if configuration.initialGoalDirectionAngle > 0 {
                configurationValues["initialGoalDirectionAngle"] =
                    configuration.initialGoalDirectionAngle
            }
            configurationValues["goalDirectionCurriculumControlSteps"] =
                Float(configuration.goalDirectionCurriculumControlSteps)
            configurationValues["goalRadius"] = configuration.goalRadius
            configurationValues["goalSlowdownDistance"] =
                configuration.goalSlowdownDistance
            if configuration.goalObservationUsesLateralVelocity {
                configurationValues["goalObservationUsesLateralVelocity"] = 1
            }
            if configuration.goalObservationIncludesLateralVelocity {
                configurationValues["goalObservationIncludesLateralVelocity"] = 1
            }
            if configuration.minimumGoalDistanceMeters > 0 {
                configurationValues["minimumGoalDistanceMeters"] =
                    configuration.minimumGoalDistanceMeters
                configurationValues["maximumGoalDistanceMeters"] =
                    configuration.maximumGoalDistanceMeters
            }
            configurationValues["initialGoalDistanceScale"] =
                configuration.initialGoalDistanceScale
            configurationValues["goalDistanceCurriculumControlSteps"] =
                Float(configuration.goalDistanceCurriculumControlSteps)
            configurationValues["goalDwellSteps"] =
                Float(configuration.goalDwellSteps)
            configurationValues["maximumGoalArrivalSpeed"] =
                configuration.maximumGoalArrivalSpeed
            if configuration.goalBoundaryCommandSpeed > 0 {
                configurationValues["goalBoundaryCommandSpeed"] =
                    configuration.goalBoundaryCommandSpeed
            }
            if configuration.goalStableDwellRewardWeight > 0 {
                configurationValues["goalStableDwellRewardWeight"] =
                    configuration.goalStableDwellRewardWeight
            }
            configurationValues["projectileProbability"] =
                configuration.projectileProbability
            configurationValues["projectileCurriculumControlSteps"] =
                Float(configuration.projectileCurriculumControlSteps)
            configurationValues["minimumProjectileSpeed"] =
                configuration.minimumProjectileSpeed
            configurationValues["maximumProjectileSpeed"] =
                configuration.maximumProjectileSpeed
            configurationValues["minimumProjectileLaunchStep"] =
                Float(configuration.minimumProjectileLaunchStep)
            configurationValues["maximumProjectileLaunchStep"] =
                Float(configuration.maximumProjectileLaunchStep)
        }
        spec = RLTaskSpec(
            id: taskID,
            revision: RLPhysicsContract.fixedGainActuatorV2(taskRevision),
            numEnvironments: configuration.numEnvironments,
            observation: RLTensorSpec(
                name: "policy",
                shape: [Self.observationFrameDimension
                    * Self.observationHistorySteps
                    + (configuration.goalObservationIncludesLateralVelocity
                        ? Self.observationHistorySteps : 0)]),
            action: RLTensorSpec(name: "joint_position", shape: [actionDimension],
                                 lowerBound: [Float](repeating: -1, count: actionDimension),
                                 upperBound: [Float](repeating: 1, count: actionDimension)),
            maxEpisodeSteps: configuration.maxEpisodeSteps,
            simulationStep: env.scene.settings.dt,
            controlDecimation: configuration.controlDecimation,
            autoReset: configuration.autoReset,
            configurationValues: configurationValues)
        commands = [Float](repeating: configuration.minimumCommandSpeed,
                           count: configuration.numEnvironments)
        previousActions = ContiguousArray(repeating: 0,
                                          count: configuration.numEnvironments * actionDimension)
        observationHistory = [[Float]](
            repeating: [Float](
                repeating: 0,
                count: Self.observationFrameDimension
                    * Self.observationHistorySteps),
            count: configuration.numEnvironments)
        goalLateralVelocityHistory = [[Float]](
            repeating: [Float](
                repeating: 0, count: Self.observationHistorySteps),
            count: configuration.numEnvironments)
        observationHistoryInitialized = [Bool](
            repeating: false, count: configuration.numEnvironments)
        previousJointAngles = [[Float]](
            repeating: [Float](repeating: 0, count: actionDimension),
            count: configuration.numEnvironments)
        jointVelocities = previousJointAngles
        previousRootPositions = [F3](repeating: .zero,
                                     count: configuration.numEnvironments)
        stepRootDisplacements = previousRootPositions
        rootPositionHistory = [[F3]](
            repeating: [F3](repeating: .zero, count: Self.velocityWindowSteps),
            count: configuration.numEnvironments)
        measuredRootVelocities = previousRootPositions
        episodeLengths = [Int](repeating: 0, count: configuration.numEnvironments)
        episodeReturns = [Float](repeating: 0, count: configuration.numEnvironments)
        episodeStartPositions = [F3](repeating: .zero,
                                      count: configuration.numEnvironments)
        goalDirections = [F3](repeating: F3(1, 0, 0),
                              count: configuration.numEnvironments)
        goalDistances = [Float](repeating: 0,
                                count: configuration.numEnvironments)
        goalOverrides = [GoalOverride?](
            repeating: nil, count: configuration.numEnvironments)
        goalDwellStepCounts = [Int](repeating: 0,
                                    count: configuration.numEnvironments)
        maximumGoalDwellStepCounts = [Int](repeating: 0,
                                           count: configuration.numEnvironments)
        goalRadiusEntered = [Bool](repeating: false,
                                   count: configuration.numEnvironments)
        goalRadiusEntrySpeeds = [Float](repeating: 0,
                                        count: configuration.numEnvironments)
        goalRadiusEntryForwardSpeeds = [Float](repeating: 0,
                                               count: configuration.numEnvironments)
        goalRadiusEntryLateralSpeeds = [Float](repeating: 0,
                                               count: configuration.numEnvironments)
        goalRadiusExitedAfterEntry = [Bool](repeating: false,
                                            count: configuration.numEnvironments)
        minimumGoalRadiusSpeeds = [Float](repeating: .infinity,
                                          count: configuration.numEnvironments)
        goalRadiusInsideStepCounts = [Int](repeating: 0,
                                           count: configuration.numEnvironments)
        projectileLaunchSteps = [Int](repeating: .max,
                                       count: configuration.numEnvironments)
        projectileSpeeds = [Float](repeating: 0,
                                   count: configuration.numEnvironments)
        projectileSides = [Float](repeating: 1,
                                  count: configuration.numEnvironments)
        projectileLaunched = [Bool](repeating: false,
                                    count: configuration.numEnvironments)
        disturbedEpisodes = [Bool](repeating: false,
                                   count: configuration.numEnvironments)
        minimumRootHeights = [Float](repeating: .infinity,
                                     count: configuration.numEnvironments)
        headingAlignmentSums = [Float](repeating: 0,
                                       count: configuration.numEnvironments)
        maximumLateralDrifts = [Float](repeating: 0,
                                       count: configuration.numEnvironments)
        footAirTimes = [[Float]](repeating: [0, 0],
                                 count: configuration.numEnvironments)
        footContactTimes = [[Float]](repeating: [0, 0],
                                     count: configuration.numEnvironments)
        previousFootContacts = [[Bool]](repeating: [true, true],
                                        count: configuration.numEnvironments)
        previousFootPositions = [[F3]](
            repeating: [F3.zero, F3.zero],
            count: configuration.numEnvironments)
        lastTouchdownFoot = [Int](repeating: -1,
                                  count: configuration.numEnvironments)
        alternatingStepCounts = [Int](repeating: 0,
                                      count: configuration.numEnvironments)
        singleSupportStepCounts = [Int](repeating: 0,
                                        count: configuration.numEnvironments)
        doubleSupportStepCounts = [Int](repeating: 0,
                                        count: configuration.numEnvironments)
        flightStepCounts = [Int](repeating: 0,
                                 count: configuration.numEnvironments)
        maximumFootClearances = [[Float]](repeating: [0, 0],
                                           count: configuration.numEnvironments)
        resetRNG = SplitMix64(seed: configuration.seed &+ 0xE7037ED1A0B428DB)
        let states = env.states()
        initializeEpisodes(Array(0..<configuration.numEnvironments), states: states,
                           seeds: (0..<configuration.numEnvironments).map {
                               configuration.seed &+ UInt64($0)
                           })
    }

    public func currentCommandSpeed(environment: Int) -> Float {
        precondition(environment >= 0 && environment < spec.numEnvironments)
        return commands[environment]
    }

    public func currentGoalDirection(environment: Int) -> F3 {
        precondition(environment >= 0 && environment < spec.numEnvironments)
        return goalDirections[environment]
    }

    public func currentGoalPosition(environment: Int) -> F3 {
        precondition(environment >= 0 && environment < spec.numEnvironments)
        return episodeStartPositions[environment]
            + goalDirections[environment] * goalDistances[environment]
    }

    /// Select a deterministic point goal for subsequent resets of one replica.
    /// This is deliberately a task command, not a robot action: the learned
    /// policy remains the only source of joint targets in replay.
    public func setGoalOverride(environment: Int, direction: F3,
                                distance: Float) throws {
        guard usesPointGoal else {
            throw RLEnvironmentError.invalidConfiguration(
                "goal overrides require humanoid-goal-v0")
        }
        guard environment >= 0 && environment < spec.numEnvironments,
              direction.x.isFinite, direction.y.isFinite,
              distance.isFinite, distance > configuration.goalRadius else {
            throw RLEnvironmentError.invalidConfiguration(
                "goal override requires a valid replica, direction, and distance")
        }
        let horizontal = F3(direction.x, direction.y, 0)
        let length = simd_length(horizontal)
        guard length > 1e-6 else {
            throw RLEnvironmentError.invalidConfiguration(
                "goal override direction must be non-zero")
        }
        goalOverrides[environment] = GoalOverride(
            direction: horizontal / length, distance: distance)
    }

    public func clearGoalOverride(environment: Int) {
        precondition(environment >= 0 && environment < spec.numEnvironments)
        goalOverrides[environment] = nil
    }

    public func hasProjectile(environment: Int) -> Bool {
        precondition(environment >= 0 && environment < spec.numEnvironments)
        return self.environment.refs[environment].projectile != nil
    }

    public func currentMeasuredRootVelocity(environment: Int) -> F3 {
        precondition(environment >= 0 && environment < spec.numEnvironments)
        return measuredRootVelocities[environment]
    }

    public func currentAlternatingSteps(environment: Int) -> Int {
        precondition(environment >= 0 && environment < spec.numEnvironments)
        return alternatingStepCounts[environment]
    }

    private struct PointGoalNavigation {
        var direction: F3
        var normal: F3
        var remainingDistance: Float
        var commandedSpeed: Float
        var proximity: Float
    }

    /// Converts the remaining world-space goal vector into policy/task-space
    /// commands. Inside the goal circle heading is deliberately unconstrained:
    /// the only remaining behavior is braking and upright balance.
    private func pointGoalNavigation(
        environment e: Int, rootPosition: F3, heading: F3
    ) -> PointGoalNavigation {
        let target = episodeStartPositions[e]
            + goalDirections[e] * goalDistances[e]
        let offset = F3(target.x - rootPosition.x,
                        target.y - rootPosition.y, 0)
        let distance = simd_length(offset)
        let direction = distance > configuration.goalRadius
            ? offset / max(distance, 1e-6)
            : heading
        return PointGoalNavigation(
            direction: direction,
            normal: F3(-direction.y, direction.x, 0),
            remainingDistance: distance,
            commandedSpeed: HumanoidLocomotionObjective.pointGoalCommandSpeed(
                remainingDistance: distance, cruiseSpeed: commands[e],
                goalRadius: configuration.goalRadius,
                slowdownDistance: configuration.goalSlowdownDistance,
                boundaryCommandSpeed:
                    configuration.goalBoundaryCommandSpeed),
            proximity: HumanoidLocomotionObjective.pointGoalProximity(
                remainingDistance: distance,
                goalRadius: configuration.goalRadius,
            slowdownDistance: configuration.goalSlowdownDistance))
    }

    static func pointGoalAuxiliaryObservation(
        proximity: Float, lateralVelocity: Float,
        usesLateralVelocity: Bool
    ) -> Float {
        usesLateralVelocity ? lateralVelocity : proximity
    }

    /// The straight-walk checkpoint observed zero lateral offset and heading
    /// cosine/sine almost exclusively near (1, 0). Point navigation reuses
    /// those three slots for bounded goal proximity and relative bearing. The
    /// command channel also needs to cover zero-speed arrival commands: the
    /// source walker saw only positive cruise speeds. Joint, gait, action,
    /// height, and velocity statistics retain the accepted-walker prior.
    public var initializationObservationVarianceFloors: [Int: Double] {
        var floors = [Int: Double]()
        for history in 0..<Self.observationHistorySteps {
            let base = history * Self.observationFrameDimension
            if usesPointGoal {
                floors[base + 2] = 0.25
                floors[base + 7] = 0.25
                floors[base + 8] = 0.25
            }
            if usesPointGoal || configuration.minimumCommandSpeed < 0.20 {
                floors[base + 50] = 0.09
            }
        }
        if configuration.goalObservationIncludesLateralVelocity {
            let appendedBase = Self.observationFrameDimension
                * Self.observationHistorySteps
            for history in 0..<Self.observationHistorySteps {
                floors[appendedBase + history] = 0.04
            }
        }
        return floors
    }

    public func initializationObservationSourceIndices(
        sourceDimension: Int
    ) -> [Int?]? {
        guard configuration.goalObservationIncludesLateralVelocity else {
            return nil
        }
        let legacyDimension = Self.observationFrameDimension
            * Self.observationHistorySteps
        guard sourceDimension == legacyDimension,
              spec.observation.elementCount
                == legacyDimension + Self.observationHistorySteps else {
            return nil
        }
        return (0..<legacyDimension).map(Optional.some)
            + [Int?](repeating: nil, count: Self.observationHistorySteps)
    }

    /// Sagittal reflection for the official H1 joint order.  Rotations are
    /// axial vectors, so yaw/roll change sign while pitch/knee/ankle do not;
    /// left and right joints are exchanged.  Applying this transform to PPO
    /// samples is ordinary symmetry augmentation, not a gait reference.
    private static let mirroredJointSource = [
        5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 10,
        15, 16, 17, 18, 11, 12, 13, 14,
    ]
    private static let mirroredJointSign: [Float] = [
        -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1,
        1, -1, -1, 1, 1, -1, -1, 1,
    ]

    public var policyActionMirrorSourceIndices: [Int] {
        Self.mirroredJointSource
    }

    public var policyActionMirrorSigns: [Float] {
        Self.mirroredJointSign
    }

    public func mirrorPolicyActions(
        _ actions: ContiguousArray<Float>) -> ContiguousArray<Float> {
        precondition(actions.count.isMultiple(of: actionDimension))
        var mirrored = actions
        let rows = actions.count / actionDimension
        for row in 0..<rows {
            let base = row * actionDimension
            for j in 0..<actionDimension {
                mirrored[base + j] = Self.mirroredJointSign[j]
                    * actions[base + Self.mirroredJointSource[j]]
            }
        }
        return mirrored
    }

    public func mirrorPolicyObservations(
        _ observations: ContiguousArray<Float>) -> ContiguousArray<Float> {
        let observationDimension = spec.observation.elementCount
        precondition(observations.count.isMultiple(of: observationDimension))
        var mirrored = observations
        let rows = observations.count / observationDimension
        let frameDimension = Self.observationFrameDimension
        for row in 0..<rows {
            let rowBase = row * observationDimension
            for history in 0..<Self.observationHistorySteps {
                let base = rowBase + history * frameDimension
                // Point-goal proximity is a scalar, while lateral position
                // and the revised goal-frame lateral velocity are polar Y
                // components and therefore change sign under reflection.
                mirrored[base + 2] = usesPointGoal
                    && !configuration.goalObservationUsesLateralVelocity
                    ? observations[base + 2]
                    : -observations[base + 2]
                // Polar vectors: up and heading flip Y.
                mirrored[base + 5] = -observations[base + 5]
                mirrored[base + 8] = -observations[base + 8]
                // Angular velocity is an axial vector under reflection.
                mirrored[base + 9] = -observations[base + 9]
                mirrored[base + 10] = observations[base + 10]
                mirrored[base + 11] = -observations[base + 11]
                for tensorBase in [12, 31, 51] {
                    for j in 0..<actionDimension {
                        mirrored[base + tensorBase + j] =
                            Self.mirroredJointSign[j]
                            * observations[base + tensorBase
                                + Self.mirroredJointSource[j]]
                    }
                }
            }
            if configuration.goalObservationIncludesLateralVelocity {
                let appendedBase = rowBase + frameDimension
                    * Self.observationHistorySteps
                for history in 0..<Self.observationHistorySteps {
                    mirrored[appendedBase + history] =
                        -observations[appendedBase + history]
                }
            }
        }
        return mirrored
    }

    public func setTrainingMode(_ enabled: Bool) {
        trainingMode = enabled
        if enabled {
            trainingControlSteps = 0
            randomizeNextTrainingResetAges =
                configuration.trainingInitialEpisodeAgeFraction > 0
        } else {
            randomizeNextTrainingResetAges = false
        }
    }

    public func setTrainingProgress(environmentSteps: Int) {
        precondition(environmentSteps >= 0)
        trainingControlSteps = environmentSteps / spec.numEnvironments
    }

    public var trainingCurriculumProgress: Float {
        guard trainingMode, configuration.commandCurriculumControlSteps > 0 else {
            return 1
        }
        return simd_clamp(Float(trainingControlSteps)
            / Float(configuration.commandCurriculumControlSteps), 0, 1)
    }

    public var trainingLateralPenaltyScale: Float {
        guard trainingMode else { return 1 }
        let elapsed = trainingControlSteps
            - configuration.lateralPenaltyWarmupControlSteps
        guard elapsed > 0 else { return 0 }
        guard configuration.lateralPenaltyRampControlSteps > 0 else { return 1 }
        return simd_clamp(Float(elapsed)
            / Float(configuration.lateralPenaltyRampControlSteps), 0, 1)
    }

    public var trainingStandingCommandProbability: Float {
        guard trainingMode,
              configuration.standingCommandCurriculumControlSteps > 0 else {
            return configuration.standingCommandProbability
        }
        let progress = simd_clamp(Float(trainingControlSteps)
            / Float(configuration.standingCommandCurriculumControlSteps), 0, 1)
        return progress * configuration.standingCommandProbability
    }

    public var usesPolicyExpertGate: Bool { configuration.commandGatedActor }
    public var usesPolicyStandExpertGate: Bool {
        configuration.commandGatedActor && configuration.threeModeActor
    }
    public var freezesLowSpeedPolicyExpert: Bool {
        configuration.freezeLowSpeedPolicyExpert
    }
    /// Straight-walk specialization always protects the already accepted
    /// cruise expert. Point-goal curricula can explicitly preserve it for a
    /// straight arrival/recovery stage, then unfreeze it when steering begins.
    public var freezesBasePolicyExpert: Bool {
        configuration.commandGatedActor
            && (usesPointGoal
                ? configuration.freezeBasePolicyExpert
                : !configuration.trainBasePolicyExpert)
    }

    /// A newly introduced low-speed/standing branch must begin as an exact
    /// copy of the transferred locomotion actor. Activating the task-owned
    /// gate is therefore behavior-preserving before specialist PPO updates,
    /// instead of routing some commands through a randomly initialized MLP.
    public var initializesPolicyExpertFromBaseOnTransfer: Bool {
        configuration.commandGatedActor
    }

    public func policyExpertGates(
        _ observations: ContiguousArray<Float>
    ) -> ContiguousArray<Float> {
        let dimension = spec.observation.elementCount
        precondition(observations.count.isMultiple(of: dimension))
        let commandIndex = 12 + 2 * actionDimension
        let rows = observations.count / dimension
        var gates = ContiguousArray(repeating: Float(0), count: rows)
        guard configuration.commandGatedActor else { return gates }
        let contacts = configuration.standExpertRequiresDoubleSupport
            ? environment.groundContacts().feet : []
        for row in 0..<rows {
            let command = observations[row * dimension + commandIndex]
            if configuration.threeModeActor && command <= 1e-6 {
                // Keep the verified braking expert active immediately after
                // entering the goal. Only hand control to the stand expert as
                // measured radial speed falls through the configured band.
                gates[row] = 1 - standExpertGate(
                    observationRow: row, observations: observations,
                    dimension: dimension,
                    bothFeetInContact: !configuration
                        .standExpertRequiresDoubleSupport
                        || (contacts[row][0] && contacts[row][1]))
                continue
            }
            let width = configuration.expertGateBlendWidth
            if width > 0 {
                let linear = simd_clamp(
                    (configuration.expertGateCommandSpeed - command) / width,
                    0, 1)
                gates[row] = linear * linear * (3 - 2 * linear)
            } else {
                gates[row] = command < configuration.expertGateCommandSpeed
                    ? 1 : 0
            }
        }
        return gates
    }

    public func policyStandExpertGates(
        _ observations: ContiguousArray<Float>
    ) -> ContiguousArray<Float> {
        let dimension = spec.observation.elementCount
        precondition(observations.count.isMultiple(of: dimension))
        let commandIndex = 12 + 2 * actionDimension
        let rows = observations.count / dimension
        var gates = ContiguousArray(repeating: Float(0), count: rows)
        guard usesPolicyStandExpertGate else { return gates }
        let contacts = configuration.standExpertRequiresDoubleSupport
            ? environment.groundContacts().feet : []
        for row in 0..<rows {
            let command = observations[row * dimension + commandIndex]
            gates[row] = command <= 1e-6
                ? standExpertGate(observationRow: row,
                    observations: observations, dimension: dimension,
                    bothFeetInContact: !configuration
                        .standExpertRequiresDoubleSupport
                        || (contacts[row][0] && contacts[row][1]))
                : 0
        }
        return gates
    }

    /// Returns the exact-stand mixture weight for one zero-command row.
    private func standExpertGate(
        observationRow row: Int,
        observations: ContiguousArray<Float>,
        dimension: Int,
        bothFeetInContact: Bool
    ) -> Float {
        let projectedSpeed = observations[row * dimension + 1]
        let measuredSpeed = Self.standExpertMeasuredSpeed(
            projectedSpeed: projectedSpeed,
            measuredRootVelocity: measuredRootVelocities[row],
            usesPlanarSpeed: configuration.standExpertUsesPlanarSpeed)
        return Self.standExpertBlendWeight(
            measuredSpeed: measuredSpeed,
            start: configuration.standExpertBlendStartSpeed,
            width: configuration.standExpertBlendWidth,
            requiresDoubleSupport:
                configuration.standExpertRequiresDoubleSupport,
            bothFeetInContact: bothFeetInContact)
    }

    static func standExpertBlendWeight(
        measuredSpeed: Float, start: Float, width: Float,
        requiresDoubleSupport: Bool, bothFeetInContact: Bool
    ) -> Float {
        if requiresDoubleSupport && !bothFeetInContact { return 0 }
        guard start > 0, width > 0 else { return 1 }
        let linear = simd_clamp((start - abs(measuredSpeed)) / width, 0, 1)
        return linear * linear * (3 - 2 * linear)
    }

    static func standExpertMeasuredSpeed(
        projectedSpeed: Float, measuredRootVelocity: F3,
        usesPlanarSpeed: Bool
    ) -> Float {
        if usesPlanarSpeed {
            return sqrt(measuredRootVelocity.x * measuredRootVelocity.x
                + measuredRootVelocity.y * measuredRootVelocity.y)
        }
        return abs(projectedSpeed)
    }

    public var trainingGoalDirectionRange: Float {
        guard trainingMode,
              configuration.goalDirectionCurriculumControlSteps > 0 else {
            return configuration.maximumGoalDirectionAngle
        }
        let progress = simd_clamp(Float(trainingControlSteps)
            / Float(configuration.goalDirectionCurriculumControlSteps), 0, 1)
        return configuration.initialGoalDirectionAngle
            + progress * (configuration.maximumGoalDirectionAngle
                - configuration.initialGoalDirectionAngle)
    }

    public var trainingGoalDistanceScale: Float {
        guard trainingMode,
              configuration.goalDistanceCurriculumControlSteps > 0 else {
            return 1
        }
        let progress = simd_clamp(Float(trainingControlSteps)
            / Float(configuration.goalDistanceCurriculumControlSteps), 0, 1)
        return configuration.initialGoalDistanceScale
            + progress * (1 - configuration.initialGoalDistanceScale)
    }

    public var trainingProjectileProbability: Float {
        guard trainingMode,
              configuration.projectileCurriculumControlSteps > 0 else {
            return configuration.projectileProbability
        }
        let progress = simd_clamp(Float(trainingControlSteps)
            / Float(configuration.projectileCurriculumControlSteps), 0, 1)
        return progress * configuration.projectileProbability
    }

    public func reset(environments ids: [Int]?, seed: UInt64,
                      into observations: inout RLObservationBatch) throws {
        try observations.validate(for: spec)
        let envIDs = try checkedEnvironmentIDs(ids)
        let seeds = envIDs.map { seed &+ UInt64($0) &* 0x9E3779B97F4A7C15 }
        environment.reset(
            envIDs, seeds: seeds,
            initialRollPitchRange: configuration.initialRollPitchRange,
            initialYawRange: configuration.initialYawRange)
        let states = environment.states()
        initializeEpisodes(envIDs, states: states, seeds: seeds)
        fillObservations(states, into: &observations.policy,
                         advancingEnvironmentIDs: envIDs)
        try observations.validate(for: spec)
    }

    public func step(actions: RLActionBatch, into result: inout RLStepBatch) throws {
        try result.validate(for: spec)
        try actions.validate(for: spec)
        result.clearSignals()
        let n = spec.numEnvironments
        var applied = actions.values
        var actionRate = ContiguousArray(repeating: Float(0), count: n)
        let targetResponse = configuration.actionTargetResponse
        for e in 0..<n {
            for j in 0..<actionDimension {
                let i = e * actionDimension + j
                let requested = simd_clamp(applied[i], -1, 1)
                applied[i] = previousActions[i]
                    + targetResponse * (requested - previousActions[i])
                let delta = applied[i] - previousActions[i]
                actionRate[e] += delta * delta
                previousActions[i] = applied[i]
            }
        }
        launchScheduledProjectiles()
        environment.step(normalizedActions: applied,
                         decimation: configuration.controlDecimation)
        if trainingMode { trainingControlSteps += 1 }
        var states = environment.states()
        let physicalGroundContacts = environment.groundContacts()
        updateMeasuredRootVelocities(states)
        updateJointVelocities(states)
        fillObservations(states, into: &result.observations.policy)

        var velocityReward = ContiguousArray(repeating: Float(0), count: n)
        var uprightReward = ContiguousArray(repeating: Float(0), count: n)
        var runningReward = ContiguousArray(repeating: Float(0), count: n)
        var runningRewardPositive = ContiguousArray(repeating: Float(0), count: n)
        var progressReward = ContiguousArray(repeating: Float(0), count: n)
        var goalArrivalReward = ContiguousArray(repeating: Float(0), count: n)
        var goalStableDwellReward = ContiguousArray(
            repeating: Float(0), count: n)
        var feetAirTimeReward = ContiguousArray(repeating: Float(0), count: n)
        var alternatingTouchdownReward = ContiguousArray(repeating: Float(0), count: n)
        var feetFlightPenalty = ContiguousArray(repeating: Float(0), count: n)
        var feetSlidePenalty = ContiguousArray(repeating: Float(0), count: n)
        var feetClearancePenalty = ContiguousArray(repeating: Float(0), count: n)
        var lateralPositionPenalty = ContiguousArray(repeating: Float(0), count: n)
        var tiltPenaltyMetric = ContiguousArray(repeating: Float(0), count: n)
        var angularVelocityXYPenalty = ContiguousArray(repeating: Float(0), count: n)
        var yawAngularVelocityPenalty = ContiguousArray(repeating: Float(0), count: n)
        var verticalVelocityPenalty = ContiguousArray(repeating: Float(0), count: n)
        var headingErrorPenalty = ContiguousArray(repeating: Float(0), count: n)
        var jointDeviationPenalty = ContiguousArray(repeating: Float(0), count: n)
        var standJointDeviationPenalty = ContiguousArray(
            repeating: Float(0), count: n)
        var standDoubleSupportReward = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeReturnMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeLengthMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeDistanceMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeSpeedMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeCommandMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeSpeedErrorMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeHeadingMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeLateralMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeMinimumRootHeightMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeAlternatingStepsMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeSingleSupportMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeDoubleSupportMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeFlightMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeLeftFootClearanceMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeRightFootClearanceMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeFullHorizonPassMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeDistanceBandPassMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeLateralCorridorPassMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeHeadingPassMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeAlternatingGaitPassMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeLowCommandBinMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeLowCommandSuccessMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeMiddleCommandBinMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeMiddleCommandSuccessMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeHighCommandBinMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeHighCommandSuccessMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeStandingBinMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeStandingSuccessMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeStandingSurvivalBinMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeStandingSurvivalSuccessMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeStandingDriftBinMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeStandingDriftSuccessMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeMovingBinMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeMovingSuccessMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalAngleMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalDistanceErrorMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalReachedMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalEnteredMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalDwellStepsMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalRadiusEnteredBinMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalRadiusEntrySpeedMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalRadiusEntryForwardSpeedMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalRadiusEntryLateralSpeedMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalRadiusExitedAfterEntryMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalRadiusMinimumSpeedMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalRadiusInsideStepsMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalFellAfterEntryMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalFailureFallBeforeEntryMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalFailureTimeoutAfterEntryMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalFailureTimeoutWithoutEntryMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeNominalBinMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeNominalSuccessMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeDisturbedBinMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeDisturbedSuccessMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalFrontBinMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalFrontSuccessMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalSideBinMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalSideSuccessMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalRearBinMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalRearSuccessMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var resetIDs = [Int]()
        var resetSeeds = [UInt64]()
        let lateralPenaltyScale = trainingLateralPenaltyScale
        for e in 0..<n {
            let state = states[e]
            let up = state.torso.rotation.act(F3(0, 0, 1))
            let heading = Self.horizontalHeading(state.torso.rotation)
            let navigation = usesPointGoal
                ? pointGoalNavigation(
                    environment: e, rootPosition: state.root.position,
                    heading: heading)
                : nil
            let goalDirection = navigation?.direction ?? F3(1, 0, 0)
            let goalNormal = navigation?.normal ?? F3(0, 1, 0)
            let commandedSpeed = navigation?.commandedSpeed ?? commands[e]
            let relativeHeading = usesPointGoal ? F3(
                simd_dot(heading, goalDirection),
                simd_dot(heading, goalNormal), 0) : heading
            let rootVelocity = measuredRootVelocities[e]
            let goalFrameVelocity = usesPointGoal ? F3(
                simd_dot(rootVelocity, goalDirection),
                simd_dot(rootVelocity, goalNormal), rootVelocity.z) : rootVelocity
            let forwardDelta: Float
            if let navigation {
                let previousPosition = state.root.position
                    - stepRootDisplacements[e]
                let target = episodeStartPositions[e]
                    + goalDirections[e] * goalDistances[e]
                let previousOffset = F3(
                    target.x - previousPosition.x,
                    target.y - previousPosition.y, 0)
                forwardDelta = simd_length(previousOffset)
                    - navigation.remainingDistance
            } else {
                forwardDelta = stepRootDisplacements[e].x
            }
            velocityReward[e] = HumanoidLocomotionObjective.velocityTracking(
                measured: goalFrameVelocity.x, commanded: commandedSpeed,
                standardDeviation:
                    configuration.velocityTrackingStandardDeviation)
            uprightReward[e] = max(up.z, 0) * max(up.z, 0)
            progressReward[e] = forwardDelta
            let contacts = physicalGroundContacts.feet[e]
            standDoubleSupportReward[e] = HumanoidLocomotionObjective
                .standStillDoubleSupportReward(
                    commandedSpeed: commandedSpeed,
                    bothFeetInContact: contacts[0] && contacts[1])
            let footClearances = [Self.footGroundClearance(state.leftFoot),
                                  Self.footGroundClearance(state.rightFoot)]
            if contacts[0] && contacts[1] {
                doubleSupportStepCounts[e] += 1
            } else if contacts[0] || contacts[1] {
                singleSupportStepCounts[e] += 1
            } else {
                flightStepCounts[e] += 1
                feetFlightPenalty[e] = 1
            }
            maximumFootClearances[e][0] = max(
                maximumFootClearances[e][0],
                footClearances[0])
            maximumFootClearances[e][1] = max(
                maximumFootClearances[e][1],
                footClearances[1])
            let footPositions = [state.leftFoot.position, state.rightFoot.position]
            var touchdownMask = 0
            for foot in 0..<2 {
                let footVelocity = (footPositions[foot]
                    - previousFootPositions[e][foot]) / spec.controlStep
                let horizontalSpeedSquared = footVelocity.x * footVelocity.x
                    + footVelocity.y * footVelocity.y
                feetClearancePenalty[e] +=
                    HumanoidLocomotionObjective.footClearanceCost(
                        clearance: footClearances[foot],
                        horizontalSpeedSquared: horizontalSpeedSquared)
                if contacts[foot] {
                    feetSlidePenalty[e] += sqrt(horizontalSpeedSquared)
                }
                if contacts[foot] {
                    if !previousFootContacts[e][foot] {
                        if footAirTimes[e][foot] > 0.10 {
                            touchdownMask |= 1 << foot
                        }
                    }
                    footAirTimes[e][foot] = 0
                    footContactTimes[e][foot] += spec.controlStep
                } else {
                    footAirTimes[e][foot] += spec.controlStep
                    footContactTimes[e][foot] = 0
                }
                previousFootContacts[e][foot] = contacts[foot]
                previousFootPositions[e][foot] = footPositions[foot]
            }
            feetAirTimeReward[e] =
                HumanoidLocomotionObjective.positiveBipedAirTime(
                    leftInContact: contacts[0], rightInContact: contacts[1],
                    leftAirTime: footAirTimes[e][0],
                    rightAirTime: footAirTimes[e][1],
                    leftContactTime: footContactTimes[e][0],
                    rightContactTime: footContactTimes[e][1])
            // Count only a single-foot touchdown whose landing foot has
            // physically passed the other foot along the current heading.
            // Simultaneous landings and split-stance pogo contacts are not
            // walking steps. This specifies neither pose, phase, nor which
            // leg moves first.
            if touchdownMask == 1 || touchdownMask == 2 {
                let foot = touchdownMask == 1 ? 0 : 1
                let other = 1 - foot
                if HumanoidLocomotionObjective.isLeadingTouchdown(
                    touchdownFootPosition: footPositions[foot],
                    otherFootPosition: footPositions[other],
                    heading: heading) {
                    if lastTouchdownFoot[e] >= 0 && lastTouchdownFoot[e] != foot {
                        alternatingStepCounts[e] += 1
                        alternatingTouchdownReward[e] = 1
                    }
                    lastTouchdownFoot[e] = foot
                }
            } else if touchdownMask == 3 {
                lastTouchdownFoot[e] = -1
            }
            let angularPenalty = state.root.angularVelocity.x * state.root.angularVelocity.x
                + state.root.angularVelocity.y * state.root.angularVelocity.y
            let yawAngularPenalty = state.root.angularVelocity.z
                * state.root.angularVelocity.z
            let commandedYawRate = usesPointGoal
                ? HumanoidLocomotionObjective.pointGoalYawRate(
                    relativeHeading: relativeHeading)
                : 0
            let yawRateErrorSquared: Float
            if usesPointGoal {
                let error = state.root.angularVelocity.z - commandedYawRate
                yawRateErrorSquared = error * error
            } else {
                // Preserve the accepted humanoid-walk-v0 arithmetic exactly.
                yawRateErrorSquared = yawAngularPenalty
            }
            let tiltPenalty = up.x * up.x + up.y * up.y
            let headingError = (1 - relativeHeading.x)
                * (1 - relativeHeading.x)
                + relativeHeading.y * relativeHeading.y
            tiltPenaltyMetric[e] = tiltPenalty
            angularVelocityXYPenalty[e] = angularPenalty
            yawAngularVelocityPenalty[e] = yawAngularPenalty
            verticalVelocityPenalty[e] = goalFrameVelocity.z
                * goalFrameVelocity.z
            headingErrorPenalty[e] = headingError
            let currentLateralDrift = usesPointGoal
                ? abs(simd_dot(
                    state.root.position - episodeStartPositions[e], goalNormal))
                : abs(state.root.position.y - environment.refs[e].center.y)
            lateralPositionPenalty[e] = currentLateralDrift * currentLateralDrift
            headingAlignmentSums[e] += relativeHeading.x
            maximumLateralDrifts[e] = max(
                maximumLateralDrifts[e], currentLateralDrift)
            // Symmetric Isaac H1 nonessential-joint regularization in source
            // radians: -0.2 for hip yaw/roll and arms, -0.1 for the torso.
            // Do not divide by the policy action scale; doing that silently
            // makes these penalties two to four times stronger than the
            // published task when the action interface changes.
            var jointDeviation: Float = 0
            for j in [0, 1, 5, 6] {
                jointDeviation += 0.2 * abs(state.jointAngles[j]
                    - HumanoidWalkEnv.defaultJointPositions[j])
            }
            for j in 11...18 {
                jointDeviation += 0.2 * abs(state.jointAngles[j]
                    - HumanoidWalkEnv.defaultJointPositions[j])
            }
            jointDeviation += 0.1 * abs(state.jointAngles[10]
                - HumanoidWalkEnv.defaultJointPositions[10])
            jointDeviationPenalty[e] = jointDeviation
            var standJointDeviationAbsolute: Float = 0
            for j in 0..<actionDimension {
                let deviation = state.jointAngles[j]
                    - HumanoidWalkEnv.defaultJointPositions[j]
                standJointDeviationAbsolute += abs(deviation)
            }
            standJointDeviationPenalty[e] = HumanoidLocomotionObjective
                .standStillJointDeviationCost(
                    commandedSpeed: commandedSpeed,
                    jointDeviationAbsolute: standJointDeviationAbsolute)
            let finite = state.root.position.x.isFinite && state.root.position.z.isFinite
            minimumRootHeights[e] = min(minimumRootHeights[e],
                                        state.root.position.z)
            let fallen = !finite
                || physicalGroundContacts.torso[e]
                || state.root.position.z
                    < HumanoidLocomotionObjective.minimumPelvisHeight
                || up.z < 0.35
            let planarSpeed = sqrt(rootVelocity.x * rootVelocity.x
                + rootVelocity.y * rootVelocity.y)
            let insideGoalRadius = usesPointGoal
                && (navigation?.remainingDistance ?? .infinity)
                    <= configuration.goalRadius
            if insideGoalRadius {
                if !goalRadiusEntered[e] {
                    goalRadiusEntered[e] = true
                    goalRadiusEntrySpeeds[e] = planarSpeed
                    goalRadiusEntryForwardSpeeds[e] = abs(goalFrameVelocity.x)
                    goalRadiusEntryLateralSpeeds[e] = abs(goalFrameVelocity.y)
                }
                minimumGoalRadiusSpeeds[e] = min(
                    minimumGoalRadiusSpeeds[e], planarSpeed)
                goalRadiusInsideStepCounts[e] += 1
            } else if goalRadiusEntered[e] {
                goalRadiusExitedAfterEntry[e] = true
            }
            let stableAtGoal = usesPointGoal
                && insideGoalRadius
                && planarSpeed <= configuration.maximumGoalArrivalSpeed
                && up.z >= 0.80 && !fallen
            if stableAtGoal {
                goalDwellStepCounts[e] += 1
            } else {
                goalDwellStepCounts[e] = 0
            }
            maximumGoalDwellStepCounts[e] = max(
                maximumGoalDwellStepCounts[e], goalDwellStepCounts[e])
            let navigationSucceeded = usesPointGoal
                && goalDwellStepCounts[e] >= configuration.goalDwellSteps
            episodeLengths[e] += 1
            let timedOut = episodeLengths[e] >= configuration.maxEpisodeSteps
            // Isaac/MuJoCo-style measured-state terms. Exact world progress
            // is reserved for evaluation and is not separately rewarded. No
            // phase clock, reference pose, scripted gait, or support force is
            // available to either this objective or the policy.
            var reward = HumanoidLocomotionObjective.reward(
                controlStep: spec.controlStep, commandedSpeed: commandedSpeed,
                measuredVelocity: goalFrameVelocity,
                forwardDisplacement: forwardDelta, tiltSquared: tiltPenalty,
                angularVelocityXYSquared: angularPenalty,
                yawAngularVelocitySquared: yawRateErrorSquared,
                headingErrorSquared: headingError,
                actionRateSquared: actionRate[e], feetAirTime: feetAirTimeReward[e],
                alternatingTouchdown: alternatingTouchdownReward[e],
                alternatingTouchdownWeight:
                    configuration.alternatingTouchdownRewardWeight,
                feetFlight: feetFlightPenalty[e],
                feetFlightPenaltyWeight: configuration.flightPenaltyWeight,
                // Contact slip is an immediate physical exploit constraint,
                // not part of the lateral-position curriculum. The former
                // code multiplied both by `lateralPenaltyScale`: slip was
                // free for the first 10k control steps and only 20% active at
                // the end of a 500x24-step run, so PPO correctly converged to
                // permanent double-support gliding. Keep the published H1
                // coefficient active from the first transition; warm up only
                // the world-position term that can impede early exploration.
                feetSlideSpeed: feetSlidePenalty[e],
                jointDeviation: jointDeviation, fallen: fallen,
                standStillVelocityPenaltyWeight:
                    configuration.standStillVelocityPenaltyWeight,
                standStillJointDeviationAbsolute: standJointDeviationAbsolute,
                standStillJointDeviationPenaltyWeight:
                    configuration.standStillJointDeviationPenaltyWeight,
                bothFeetInContact: contacts[0] && contacts[1],
                standStillDoubleSupportRewardWeight:
                    configuration.standStillDoubleSupportRewardWeight,
                standStillFallPenalty:
                    configuration.standStillFallPenalty,
                velocityTrackingStandardDeviation:
                    configuration.velocityTrackingStandardDeviation,
                velocityTrackingErrorPenaltyWeight:
                    configuration.velocityTrackingErrorPenaltyWeight,
                // A point-goal command defines an endpoint, not a straight
                // world-space rail. Curved steering is therefore free to use
                // lateral displacement while the original forward-walk task
                // retains its exact lane-tracking objective.
                lateralDisplacementSquared: usesPointGoal
                    ? 0
                    : lateralPenaltyScale * lateralPositionPenalty[e],
                laneTrackingStandardDeviation:
                    configuration.laneTrackingStandardDeviation,
                feetClearanceCost: feetClearancePenalty[e])
            if usesPointGoal {
                // Signed potential progress cannot be farmed by circling or
                // overshooting: moving away from the fixed target repays it.
                reward += 2 * forwardDelta
                let arrivalQuality = HumanoidLocomotionObjective
                    .pointGoalArrivalQuality(
                        proximity: navigation?.proximity ?? 0,
                        planarSpeed: planarSpeed, upright: up.z,
                        arrivalSpeed: configuration.maximumGoalArrivalSpeed)
                // At most +2 reward/second. This is deliberately smaller
                // than the completion bonus, but dense enough to distinguish
                // controlled braking from repeatedly crossing the goal.
                goalArrivalReward[e] = 2 * spec.controlStep * arrivalQuality
                reward += goalArrivalReward[e]
                if stableAtGoal {
                    goalStableDwellReward[e] = spec.controlStep
                        * configuration.goalStableDwellRewardWeight
                    reward += goalStableDwellReward[e]
                }
                if navigationSucceeded { reward += 10 }
            }
            result.rewards[e] = reward
            // Expose the post-clip dense signal separately from termination.
            // A value pinned at zero is an immediate, machine-readable sign
            // that PPO is learning only from episode endings.
            runningReward[e] = reward
                + (fallen ? HumanoidLocomotionObjective.terminationPenalty : 0)
            runningRewardPositive[e] = runningReward[e] > 0 ? 1 : 0
            result.terminated[e] = fallen || navigationSucceeded
            result.truncated[e] = !fallen && !navigationSucceeded && timedOut
            episodeReturns[e] += reward
            if fallen || timedOut || navigationSucceeded {
                let elapsed = Float(episodeLengths[e]) * spec.controlStep
                let episodeDisplacement = state.root.position
                    - episodeStartPositions[e]
                let episodeGoalDirection = goalDirections[e]
                let forwardDistance = usesPointGoal
                    ? simd_dot(episodeDisplacement, episodeGoalDirection)
                    : state.root.position.x - episodeStartPositions[e].x
                let goalDistanceError = simd_length(F3(
                    episodeDisplacement.x
                        - episodeGoalDirection.x * goalDistances[e],
                    episodeDisplacement.y
                        - episodeGoalDirection.y * goalDistances[e], 0))
                let meanHeadingAlignment = headingAlignmentSums[e]
                    / Float(max(episodeLengths[e], 1))
                let successComponents =
                    HumanoidLocomotionObjective.successComponents(
                    timedOut: timedOut, fallen: fallen,
                    forwardDistance: forwardDistance,
                    lateralDistance: usesPointGoal
                        ? 0
                        : maximumLateralDrifts[e],
                    headingAlignment: meanHeadingAlignment,
                    alternatingSteps: alternatingStepCounts[e],
                    commandedSpeed: commands[e], elapsed: elapsed,
                    goalDistanceError: usesPointGoal ? goalDistanceError : nil,
                    goalRadius: configuration.goalRadius)
                let episodeSucceeded = usesPointGoal
                    ? navigationSucceeded
                    : successComponents.allPassed
                result.successes[e] = episodeSucceeded
                result.hasFinalObservation[e] = true
                let row = e * spec.observation.elementCount
                for j in 0..<spec.observation.elementCount {
                    result.finalObservations[row + j] = result.observations.policy[row + j]
                }
                episodeReturnMetric[e] = episodeReturns[e]
                episodeLengthMetric[e] = Float(episodeLengths[e])
                episodeDistanceMetric[e] = forwardDistance
                episodeSpeedMetric[e] = episodeDistanceMetric[e] / max(elapsed, 1e-6)
                episodeCommandMetric[e] = commands[e]
                episodeSpeedErrorMetric[e] = abs(episodeSpeedMetric[e] - commands[e])
                episodeHeadingMetric[e] = meanHeadingAlignment
                episodeLateralMetric[e] = maximumLateralDrifts[e]
                episodeMinimumRootHeightMetric[e] = minimumRootHeights[e]
                episodeAlternatingStepsMetric[e] = Float(alternatingStepCounts[e])
                let inverseLength = 1 / Float(max(episodeLengths[e], 1))
                episodeSingleSupportMetric[e] = Float(singleSupportStepCounts[e])
                    * inverseLength
                episodeDoubleSupportMetric[e] = Float(doubleSupportStepCounts[e])
                    * inverseLength
                episodeFlightMetric[e] = Float(flightStepCounts[e]) * inverseLength
                episodeLeftFootClearanceMetric[e] = maximumFootClearances[e][0]
                episodeRightFootClearanceMetric[e] = maximumFootClearances[e][1]
                episodeFullHorizonPassMetric[e] =
                    successComponents.fullHorizon ? 1 : 0
                episodeDistanceBandPassMetric[e] = usesPointGoal
                    ? (goalDistanceError <= configuration.goalRadius ? 1 : 0)
                    : (successComponents.distanceBand ? 1 : 0)
                episodeLateralCorridorPassMetric[e] =
                    successComponents.lateralCorridor ? 1 : 0
                episodeHeadingPassMetric[e] = successComponents.heading ? 1 : 0
                episodeAlternatingGaitPassMetric[e] =
                    successComponents.alternatingGait ? 1 : 0
                if commands[e] < 0.20 {
                    episodeStandingBinMetric[e] = 1
                    episodeStandingSuccessMetric[e] =
                        episodeSucceeded ? 1 : 0
                    episodeStandingSurvivalBinMetric[e] = 1
                    episodeStandingSurvivalSuccessMetric[e] =
                        successComponents.fullHorizon ? 1 : 0
                    episodeStandingDriftBinMetric[e] = 1
                    episodeStandingDriftSuccessMetric[e] =
                        successComponents.distanceBand ? 1 : 0
                } else {
                    episodeMovingBinMetric[e] = 1
                    episodeMovingSuccessMetric[e] =
                        episodeSucceeded ? 1 : 0
                }
                episodeGoalAngleMetric[e] = atan2(
                    episodeGoalDirection.y, episodeGoalDirection.x)
                episodeGoalDistanceErrorMetric[e] = goalDistanceError
                episodeGoalReachedMetric[e] = episodeSucceeded ? 1 : 0
                episodeGoalEnteredMetric[e] = goalDistanceError
                    <= configuration.goalRadius ? 1 : 0
                episodeGoalDwellStepsMetric[e] =
                    Float(maximumGoalDwellStepCounts[e])
                if goalRadiusEntered[e] {
                    episodeGoalRadiusEnteredBinMetric[e] = 1
                    episodeGoalRadiusEntrySpeedMetric[e] =
                        goalRadiusEntrySpeeds[e]
                    episodeGoalRadiusEntryForwardSpeedMetric[e] =
                        goalRadiusEntryForwardSpeeds[e]
                    episodeGoalRadiusEntryLateralSpeedMetric[e] =
                        goalRadiusEntryLateralSpeeds[e]
                    episodeGoalRadiusExitedAfterEntryMetric[e] =
                        goalRadiusExitedAfterEntry[e] ? 1 : 0
                    episodeGoalRadiusMinimumSpeedMetric[e] =
                        minimumGoalRadiusSpeeds[e]
                    episodeGoalRadiusInsideStepsMetric[e] =
                        Float(goalRadiusInsideStepCounts[e])
                    episodeGoalFellAfterEntryMetric[e] = fallen ? 1 : 0
                }
                episodeGoalFailureFallBeforeEntryMetric[e] =
                    fallen && !goalRadiusEntered[e] ? 1 : 0
                episodeGoalFailureTimeoutAfterEntryMetric[e] =
                    timedOut && goalRadiusEntered[e] ? 1 : 0
                episodeGoalFailureTimeoutWithoutEntryMetric[e] =
                    timedOut && !goalRadiusEntered[e] ? 1 : 0
                let absoluteGoalAngle = abs(episodeGoalAngleMetric[e])
                if absoluteGoalAngle <= .pi / 4 {
                    episodeGoalFrontBinMetric[e] = 1
                    episodeGoalFrontSuccessMetric[e] =
                        episodeSucceeded ? 1 : 0
                } else if absoluteGoalAngle <= 3 * .pi / 4 {
                    episodeGoalSideBinMetric[e] = 1
                    episodeGoalSideSuccessMetric[e] =
                        episodeSucceeded ? 1 : 0
                } else {
                    episodeGoalRearBinMetric[e] = 1
                    episodeGoalRearSuccessMetric[e] =
                        episodeSucceeded ? 1 : 0
                }
                if disturbedEpisodes[e] {
                    episodeDisturbedBinMetric[e] = 1
                    episodeDisturbedSuccessMetric[e] =
                        episodeSucceeded ? 1 : 0
                } else {
                    episodeNominalBinMetric[e] = 1
                    episodeNominalSuccessMetric[e] =
                        episodeSucceeded ? 1 : 0
                }
                let commandFraction = (commands[e]
                    - configuration.minimumCommandSpeed)
                    / max(configuration.maximumCommandSpeed
                        - configuration.minimumCommandSpeed, 1e-6)
                if commandFraction < 1 / 3 {
                    episodeLowCommandBinMetric[e] = 1
                    episodeLowCommandSuccessMetric[e] =
                        episodeSucceeded ? 1 : 0
                } else if commandFraction < 2 / 3 {
                    episodeMiddleCommandBinMetric[e] = 1
                    episodeMiddleCommandSuccessMetric[e] =
                        episodeSucceeded ? 1 : 0
                } else {
                    episodeHighCommandBinMetric[e] = 1
                    episodeHighCommandSuccessMetric[e] =
                        episodeSucceeded ? 1 : 0
                }
                if configuration.autoReset {
                    resetIDs.append(e)
                    resetSeeds.append(resetRNG.next())
                }
            }
        }
        result.metrics["reward/velocity_tracking"] = velocityReward
        result.metrics["reward/upright"] = uprightReward
        result.metrics["reward/running"] = runningReward
        result.metrics["reward/running_positive_fraction"] = runningRewardPositive
        result.metrics["reward/progress"] = progressReward
        if usesPointGoal {
            result.metrics["reward/goal_arrival"] = goalArrivalReward
            result.metrics["reward/goal_stable_dwell"] =
                goalStableDwellReward
        }
        result.metrics["reward/feet_air_time"] = feetAirTimeReward
        result.metrics["gait/alternating_touchdown"] = alternatingTouchdownReward
        result.metrics["penalty/feet_flight"] = feetFlightPenalty
        result.metrics["penalty/feet_slide"] = feetSlidePenalty
        result.metrics["penalty/feet_clearance"] = feetClearancePenalty
        result.metrics["penalty/lateral_position"] = lateralPositionPenalty
        result.metrics["penalty/action_rate"] = actionRate
        result.metrics["penalty/tilt"] = tiltPenaltyMetric
        result.metrics["penalty/angular_velocity_xy"] = angularVelocityXYPenalty
        result.metrics["penalty/yaw_angular_velocity"] = yawAngularVelocityPenalty
        result.metrics["penalty/vertical_velocity"] = verticalVelocityPenalty
        result.metrics["penalty/heading_error"] = headingErrorPenalty
        result.metrics["penalty/joint_deviation"] = jointDeviationPenalty
        result.metrics["penalty/stand_joint_deviation"] =
            standJointDeviationPenalty
        result.metrics["reward/stand_double_support"] =
            standDoubleSupportReward
        result.metrics["state/root_height_m"] = ContiguousArray(
            states.map { $0.root.position.z })
        result.metrics["episode/return"] = episodeReturnMetric
        result.metrics["episode/length"] = episodeLengthMetric
        result.metrics["episode/forward_distance_m"] = episodeDistanceMetric
        result.metrics["episode/forward_speed_mps"] = episodeSpeedMetric
        result.metrics["episode/command_speed_mps"] = episodeCommandMetric
        result.metrics["episode/speed_error_mps"] = episodeSpeedErrorMetric
        result.metrics["episode/heading_alignment"] = episodeHeadingMetric
        result.metrics["episode/lateral_distance_m"] = episodeLateralMetric
        result.metrics["episode/minimum_root_height_m"] =
            episodeMinimumRootHeightMetric
        result.metrics["episode/alternating_steps"] = episodeAlternatingStepsMetric
        result.metrics["episode/single_support_fraction"] = episodeSingleSupportMetric
        result.metrics["episode/double_support_fraction"] = episodeDoubleSupportMetric
        result.metrics["episode/flight_fraction"] = episodeFlightMetric
        result.metrics["episode/left_foot_max_clearance_m"] = episodeLeftFootClearanceMetric
        result.metrics["episode/right_foot_max_clearance_m"] = episodeRightFootClearanceMetric
        result.metrics["episode/pass_full_horizon"] = episodeFullHorizonPassMetric
        result.metrics["episode/pass_distance_band"] = episodeDistanceBandPassMetric
        result.metrics["episode/pass_lateral_corridor"] =
            episodeLateralCorridorPassMetric
        result.metrics["episode/pass_heading"] = episodeHeadingPassMetric
        result.metrics["episode/pass_alternating_gait"] =
            episodeAlternatingGaitPassMetric
        result.metrics["episode/command_low_bin"] = episodeLowCommandBinMetric
        result.metrics["episode/command_low_success"] =
            episodeLowCommandSuccessMetric
        result.metrics["episode/command_middle_bin"] =
            episodeMiddleCommandBinMetric
        result.metrics["episode/command_middle_success"] =
            episodeMiddleCommandSuccessMetric
        result.metrics["episode/command_high_bin"] = episodeHighCommandBinMetric
        result.metrics["episode/command_high_success"] =
            episodeHighCommandSuccessMetric
        result.metrics["episode/standing_bin"] = episodeStandingBinMetric
        result.metrics["episode/standing_success"] =
            episodeStandingSuccessMetric
        result.metrics["episode/standing_survival_bin"] =
            episodeStandingSurvivalBinMetric
        result.metrics["episode/standing_survival_success"] =
            episodeStandingSurvivalSuccessMetric
        result.metrics["episode/standing_drift_bin"] =
            episodeStandingDriftBinMetric
        result.metrics["episode/standing_drift_success"] =
            episodeStandingDriftSuccessMetric
        result.metrics["episode/moving_bin"] = episodeMovingBinMetric
        result.metrics["episode/moving_success"] = episodeMovingSuccessMetric
        if usesPointGoal {
            result.metrics["episode/goal_angle_rad"] = episodeGoalAngleMetric
            result.metrics["episode/goal_distance_error_m"] =
                episodeGoalDistanceErrorMetric
            result.metrics["episode/goal_reached"] = episodeGoalReachedMetric
            result.metrics["episode/goal_entered"] = episodeGoalEnteredMetric
            result.metrics["episode/goal_dwell_steps"] =
                episodeGoalDwellStepsMetric
            // Entry-speed metrics are emitted with an explicit cohort bin.
            // Aggregators can recover conditional means by dividing each
            // zero-masked metric by `goal_radius_entered_bin`.
            result.metrics["episode/goal_radius_entered_bin"] =
                episodeGoalRadiusEnteredBinMetric
            result.metrics["episode/goal_radius_entry_speed_mps"] =
                episodeGoalRadiusEntrySpeedMetric
            result.metrics["episode/goal_radius_entry_forward_speed_mps"] =
                episodeGoalRadiusEntryForwardSpeedMetric
            result.metrics["episode/goal_radius_entry_lateral_speed_mps"] =
                episodeGoalRadiusEntryLateralSpeedMetric
            result.metrics["episode/goal_radius_exited_after_entry"] =
                episodeGoalRadiusExitedAfterEntryMetric
            result.metrics["episode/goal_radius_minimum_speed_mps"] =
                episodeGoalRadiusMinimumSpeedMetric
            result.metrics["episode/goal_radius_inside_steps"] =
                episodeGoalRadiusInsideStepsMetric
            result.metrics["episode/goal_fell_after_entry"] =
                episodeGoalFellAfterEntryMetric
            result.metrics["episode/goal_failure_fall_before_entry"] =
                episodeGoalFailureFallBeforeEntryMetric
            result.metrics["episode/goal_failure_timeout_after_entry"] =
                episodeGoalFailureTimeoutAfterEntryMetric
            result.metrics["episode/goal_failure_timeout_without_entry"] =
                episodeGoalFailureTimeoutWithoutEntryMetric
            result.metrics["episode/nominal_bin"] = episodeNominalBinMetric
            result.metrics["episode/nominal_success"] = episodeNominalSuccessMetric
            result.metrics["episode/disturbed_bin"] = episodeDisturbedBinMetric
            result.metrics["episode/disturbed_success"] =
                episodeDisturbedSuccessMetric
            result.metrics["episode/goal_front_bin"] = episodeGoalFrontBinMetric
            result.metrics["episode/goal_front_success"] =
                episodeGoalFrontSuccessMetric
            result.metrics["episode/goal_side_bin"] = episodeGoalSideBinMetric
            result.metrics["episode/goal_side_success"] =
                episodeGoalSideSuccessMetric
            result.metrics["episode/goal_rear_bin"] = episodeGoalRearBinMetric
            result.metrics["episode/goal_rear_success"] =
                episodeGoalRearSuccessMetric
        }

        if !resetIDs.isEmpty {
            environment.reset(
                resetIDs, seeds: resetSeeds,
                initialRollPitchRange: configuration.initialRollPitchRange,
                initialYawRange: configuration.initialYawRange)
            states = environment.states()
            initializeEpisodes(resetIDs, states: states, seeds: resetSeeds)
            fillObservations(states, into: &result.observations.policy,
                             advancingEnvironmentIDs: resetIDs)
        }
        try result.observations.validate(for: spec)
    }

    private func initializeEpisodes(_ ids: [Int], states: [HumanoidState],
                                    seeds: [UInt64]) {
        let randomizeInitialAges = trainingMode
            && randomizeNextTrainingResetAges
            && configuration.trainingInitialEpisodeAgeFraction > 0
        let maximumInitialAge = min(
            configuration.maxEpisodeSteps - 1,
            Int(Float(configuration.maxEpisodeSteps)
                * configuration.trainingInitialEpisodeAgeFraction))
        var targetDistances = [Float]()
        var targetDirections = [F3]()
        targetDistances.reserveCapacity(ids.count)
        targetDirections.reserveCapacity(ids.count)
        for (offset, e) in ids.enumerated() {
            var rng = SplitMix64(seed: seeds[offset])
            let progress = trainingCurriculumProgress
            let minimumSpeed = trainingMode
                ? Self.curriculumStartMinimumSpeed
                    + progress * (configuration.minimumCommandSpeed
                        - Self.curriculumStartMinimumSpeed)
                : configuration.minimumCommandSpeed
            let maximumSpeed = trainingMode
                ? Self.curriculumStartMaximumSpeed
                    + progress * (configuration.maximumCommandSpeed
                        - Self.curriculumStartMaximumSpeed)
                : configuration.maximumCommandSpeed
            commands[e] = minimumSpeed
                + (maximumSpeed - minimumSpeed) * rng.nextFloat()
            if !usesPointGoal, trainingStandingCommandProbability > 0,
               rng.nextFloat() < trainingStandingCommandProbability {
                commands[e] = 0
            }
            let goalAngle = usesPointGoal
                ? (2 * rng.nextFloat() - 1) * trainingGoalDirectionRange
                : 0
            goalDirections[e] = F3(cos(goalAngle), sin(goalAngle), 0)
            if usesPointGoal && configuration.minimumGoalDistanceMeters > 0 {
                goalDistances[e] = (configuration.minimumGoalDistanceMeters
                    + (configuration.maximumGoalDistanceMeters
                        - configuration.minimumGoalDistanceMeters)
                        * rng.nextFloat()) * trainingGoalDistanceScale
            } else {
                goalDistances[e] = commands[e]
                    * Float(configuration.maxEpisodeSteps) * spec.controlStep
                    * trainingGoalDistanceScale
            }
            if let goalOverride = goalOverrides[e] {
                goalDirections[e] = goalOverride.direction
                goalDistances[e] = goalOverride.distance
            }
            disturbedEpisodes[e] = usesPointGoal
                && rng.nextFloat() < trainingProjectileProbability
            if disturbedEpisodes[e] {
                let span = configuration.maximumProjectileLaunchStep
                    - configuration.minimumProjectileLaunchStep + 1
                projectileLaunchSteps[e] =
                    configuration.minimumProjectileLaunchStep
                    + Int(rng.next() % UInt64(span))
                projectileSpeeds[e] = configuration.minimumProjectileSpeed
                    + (configuration.maximumProjectileSpeed
                        - configuration.minimumProjectileSpeed) * rng.nextFloat()
                projectileSides[e] = rng.nextFloat() < 0.5 ? -1 : 1
            } else {
                projectileLaunchSteps[e] = .max
                projectileSpeeds[e] = 0
                projectileSides[e] = 1
            }
            projectileLaunched[e] = false
            previousJointAngles[e] = states[e].jointAngles
            jointVelocities[e] = [Float](repeating: 0, count: actionDimension)
            previousRootPositions[e] = states[e].root.position
            stepRootDisplacements[e] = .zero
            rootPositionHistory[e] = [F3](
                repeating: states[e].root.position,
                count: Self.velocityWindowSteps)
            measuredRootVelocities[e] = .zero
            for j in 0..<actionDimension { previousActions[e * actionDimension + j] = 0 }
            episodeLengths[e] = randomizeInitialAges && maximumInitialAge > 0
                ? Int(rng.next() % UInt64(maximumInitialAge + 1))
                : 0
            episodeReturns[e] = 0
            episodeStartPositions[e] = states[e].root.position
            goalDwellStepCounts[e] = 0
            maximumGoalDwellStepCounts[e] = 0
            goalRadiusEntered[e] = false
            goalRadiusEntrySpeeds[e] = 0
            goalRadiusEntryForwardSpeeds[e] = 0
            goalRadiusEntryLateralSpeeds[e] = 0
            goalRadiusExitedAfterEntry[e] = false
            minimumGoalRadiusSpeeds[e] = .infinity
            goalRadiusInsideStepCounts[e] = 0
            minimumRootHeights[e] = states[e].root.position.z
            headingAlignmentSums[e] = 0
            maximumLateralDrifts[e] = 0
            footAirTimes[e] = [0, 0]
            footContactTimes[e] = [0, 0]
            previousFootContacts[e] = [Self.footInContact(states[e].leftFoot),
                                       Self.footInContact(states[e].rightFoot)]
            previousFootPositions[e] = [states[e].leftFoot.position,
                                        states[e].rightFoot.position]
            lastTouchdownFoot[e] = -1
            alternatingStepCounts[e] = 0
            singleSupportStepCounts[e] = 0
            doubleSupportStepCounts[e] = 0
            flightStepCounts[e] = 0
            maximumFootClearances[e] = [0, 0]
            observationHistoryInitialized[e] = false
            targetDistances.append(goalDistances[e])
            targetDirections.append(goalDirections[e])
        }
        if randomizeInitialAges { randomizeNextTrainingResetAges = false }
        environment.setGoalMarkers(
            environmentIDs: ids, directions: targetDirections,
            distances: targetDistances)
    }

    private func launchScheduledProjectiles() {
        guard configuration.projectileProbability > 0 else { return }
        var ids = [Int]()
        var positions = [F3]()
        var velocities = [F3]()
        var angularVelocities = [F3]()
        for e in 0..<spec.numEnvironments
            where disturbedEpisodes[e] && !projectileLaunched[e]
                && episodeLengths[e] >= projectileLaunchSteps[e] {
            let forward = goalDirections[e]
            let lateral = F3(-forward.y, forward.x, 0)
            let root = previousRootPositions[e]
            let launch = F3(root.x, root.y, 0.95)
                + lateral * (1.6 * projectileSides[e])
                + forward * 0.15
            let target = F3(root.x, root.y, 0.82)
            let direction = simd_normalize(target - launch)
            ids.append(e)
            positions.append(launch)
            velocities.append(direction * projectileSpeeds[e])
            angularVelocities.append(F3(
                projectileSides[e] * 2.5,
                -projectileSides[e] * 1.5,
                projectileSides[e] * 3.5))
            projectileLaunched[e] = true
        }
        if !ids.isEmpty {
            environment.throwProjectiles(
                environmentIDs: ids, positions: positions,
                velocities: velocities, angularVelocities: angularVelocities)
        }
    }

    private func updateJointVelocities(_ states: [HumanoidState]) {
        for e in 0..<spec.numEnvironments {
            for j in 0..<actionDimension {
                var delta = states[e].jointAngles[j] - previousJointAngles[e][j]
                delta -= 2 * .pi * (delta / (2 * .pi)).rounded()
                jointVelocities[e][j] = delta / spec.controlStep
            }
            previousJointAngles[e] = states[e].jointAngles
        }
    }

    /// AVBD is position based, so sub-frame constraint projection is noisy.
    /// Use net rendered displacement over a short causal window. This is the
    /// same world-space quantity used by episode success, and oscillation with
    /// zero net travel cannot imitate a locomotion velocity.
    private func updateMeasuredRootVelocities(_ states: [HumanoidState]) {
        for e in 0..<spec.numEnvironments {
            let displacement = states[e].root.position - previousRootPositions[e]
            stepRootDisplacements[e] = displacement
            let oldPosition = rootPositionHistory[e][rootHistoryIndex]
            measuredRootVelocities[e] =
                (states[e].root.position - oldPosition)
                / (Float(Self.velocityWindowSteps) * spec.controlStep)
            rootPositionHistory[e][rootHistoryIndex] = states[e].root.position
            previousRootPositions[e] = states[e].root.position
        }
        rootHistoryIndex = (rootHistoryIndex + 1) % Self.velocityWindowSteps
    }

    private func fillObservations(
        _ states: [HumanoidState],
        into output: inout ContiguousArray<Float>,
        advancingEnvironmentIDs: [Int]? = nil
    ) {
        var shouldAdvance = [Bool](repeating: advancingEnvironmentIDs == nil,
                                   count: spec.numEnvironments)
        if let ids = advancingEnvironmentIDs {
            for e in ids { shouldAdvance[e] = true }
        }
        let frameDimension = Self.observationFrameDimension
        let historySteps = Self.observationHistorySteps
        let legacyObservationDimension = frameDimension * historySteps
        let observationDimension = spec.observation.elementCount
        for e in 0..<spec.numEnvironments {
            if shouldAdvance[e] || !observationHistoryInitialized[e] {
                let s = states[e]
                let up = s.torso.rotation.act(F3(0, 0, 1))
                let heading = Self.horizontalHeading(s.torso.rotation)
                let navigation = usesPointGoal
                    ? pointGoalNavigation(
                        environment: e, rootPosition: s.root.position,
                        heading: heading)
                    : nil
                let goalDirection = navigation?.direction ?? F3(1, 0, 0)
                let goalNormal = navigation?.normal ?? F3(0, 1, 0)
                let goalLateralVelocity = simd_dot(
                    measuredRootVelocities[e], goalNormal)
                var frame = [Float](repeating: 0, count: frameDimension)
                frame[0] = s.root.position.z
                    - environment.scene.bodies[environment.refs[e].root].position.z
                frame[1] = usesPointGoal
                    ? simd_dot(measuredRootVelocities[e], goalDirection)
                    : measuredRootVelocities[e].x
                // Relative target bearing plus bounded proximity describes a
                // true point goal without changing the accepted policy shape.
                // Proximity remains zero during cruise, matching the nominal
                // walker's lateral-offset input until braking is required.
                frame[2] = usesPointGoal
                    ? Self.pointGoalAuxiliaryObservation(
                        proximity: navigation!.proximity,
                        lateralVelocity: goalLateralVelocity,
                        usesLateralVelocity: configuration
                            .goalObservationUsesLateralVelocity)
                    : s.root.position.y - environment.refs[e].center.y
                frame[3] = measuredRootVelocities[e].z
                frame[4] = usesPointGoal ? simd_dot(up, goalDirection) : up.x
                frame[5] = usesPointGoal ? simd_dot(up, goalNormal) : up.y
                frame[6] = up.z
                frame[7] = usesPointGoal
                    ? simd_dot(heading, goalDirection) : heading.x
                frame[8] = usesPointGoal
                    ? simd_dot(heading, goalNormal) : heading.y
                frame[9] = usesPointGoal
                    ? simd_dot(s.root.angularVelocity, goalDirection)
                    : s.root.angularVelocity.x
                frame[10] = usesPointGoal
                    ? simd_dot(s.root.angularVelocity, goalNormal)
                    : s.root.angularVelocity.y
                frame[11] = s.root.angularVelocity.z
                let jointBase = 12
                let velocityBase = jointBase + actionDimension
                let commandIndex = velocityBase + actionDimension
                let actionBase = commandIndex + 1
                for j in 0..<actionDimension {
                    frame[jointBase + j] = (s.jointAngles[j]
                        - HumanoidWalkEnv.defaultJointPositions[j])
                        / HumanoidWalkEnv.actionScales[j]
                    frame[velocityBase + j] = jointVelocities[e][j] * 0.1
                    frame[actionBase + j] = previousActions[
                        e * actionDimension + j]
                }
                frame[commandIndex] = navigation?.commandedSpeed ?? commands[e]

                if observationHistoryInitialized[e] {
                    for historyIndex in stride(from: historySteps - 1,
                                               through: 1, by: -1) {
                        let destination = historyIndex * frameDimension
                        let source = (historyIndex - 1) * frameDimension
                        for j in 0..<frameDimension {
                            observationHistory[e][destination + j] =
                                observationHistory[e][source + j]
                        }
                    }
                    if configuration.goalObservationIncludesLateralVelocity {
                        for historyIndex in stride(
                            from: historySteps - 1, through: 1, by: -1
                        ) {
                            goalLateralVelocityHistory[e][historyIndex] =
                                goalLateralVelocityHistory[e][historyIndex - 1]
                        }
                    }
                } else {
                    for historyIndex in 0..<historySteps {
                        let destination = historyIndex * frameDimension
                        for j in 0..<frameDimension {
                            observationHistory[e][destination + j] = frame[j]
                        }
                    }
                    if configuration.goalObservationIncludesLateralVelocity {
                        for historyIndex in 0..<historySteps {
                            goalLateralVelocityHistory[e][historyIndex] =
                                goalLateralVelocity
                        }
                    }
                }
                for j in 0..<frameDimension {
                    observationHistory[e][j] = frame[j]
                }
                if configuration.goalObservationIncludesLateralVelocity {
                    goalLateralVelocityHistory[e][0] = goalLateralVelocity
                }
                observationHistoryInitialized[e] = true
            }
            let outputBase = e * observationDimension
            for j in 0..<legacyObservationDimension {
                output[outputBase + j] = observationHistory[e][j]
            }
            if configuration.goalObservationIncludesLateralVelocity {
                for history in 0..<historySteps {
                    output[outputBase + legacyObservationDimension + history] =
                        goalLateralVelocityHistory[e][history]
                }
            }
        }
    }

    /// Horizontal torso +X in world coordinates. This is measured robot
    /// orientation—not a motion reference—and makes facing backward
    /// observable and explicitly unrewarded.
    private static func horizontalHeading(_ rotation: Quat) -> F3 {
        let forward = rotation.act(F3(1, 0, 0))
        let magnitude = sqrt(forward.x * forward.x + forward.y * forward.y)
        guard magnitude > 1e-6 else { return F3(1, 0, 0) }
        return F3(forward.x / magnitude, forward.y / magnitude, 0)
    }

    /// Conservative contact estimate for the oriented box foot. Center-height
    /// thresholds misclassify a pitched foot whose toe or heel is on the
    /// ground, suppressing both air-time reward and alternating-step metrics.
    static func footInContact(_ foot: GPUSolver.RigidBodyState) -> Bool {
        return footGroundClearance(foot) <= footContactClearance
    }

    /// Geometric clearance of the lowest oriented-box corner above z=0.
    /// This is diagnostic state only; it does not assume a gait phase or
    /// provide the policy with privileged contact impulses.
    static func footGroundClearance(_ foot: GPUSolver.RigidBodyState) -> Float {
        let minimumCenterZ = h1FootCapsuleEndpoints.reduce(Float.infinity) {
            min($0, (foot.position + foot.rotation.act($1)).z)
        }
        return minimumCenterZ - h1FootCapsuleRadius
    }
}
