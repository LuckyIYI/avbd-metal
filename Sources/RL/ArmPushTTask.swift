import simd
import SimCore
import PhysicsAVBD
import Robotics

public struct ArmPushTTaskConfig: Sendable {
    public var numEnvironments: Int
    public var seed: UInt64
    public var maxEpisodeSteps: Int
    public var controlDecimation: Int
    public var autoReset: Bool
    /// Normalized policy actions command a bounded increment from the
    /// measured joint position. ManiSkill's tested articulated Push-T PPO
    /// configuration uses the same ±0.1 rad PD joint-delta interface.
    public var jointDeltaActionScale: Float
    /// Optional task-space controller. A positive value switches the same
    /// two normalized actions to measured end-effector XY deltas in metres,
    /// resolved to physical joint targets with the arm's analytic IK. This is
    /// a controller choice, not a reference policy: the network still chooses
    /// every motion and all contacts remain simulated.
    public var endEffectorDeltaActionScale: Float
    /// Physical actuator and reduced-arm parameters. These are serialized in
    /// the task signature so a checkpoint cannot silently cross actuator
    /// plants used by the maintained Panda Push-T reference task.
    public var linkLength1: Float
    public var linkLength2: Float
    public var linkMass: Float
    public var tipMass: Float
    public var motorTorque: Float
    public var motorStiffness: Float
    public var motorDamping: Float
    public var motorArmature: Float
    public var blockMass: Float
    public var blockStaticFriction: Float
    public var blockDynamicFriction: Float
    /// Reset curriculum. Zero preserves the canonical origin-centered spawn;
    /// one centers the randomized spawn around the goal for precision-first
    /// training. Evaluation uses the default zero.
    public var blockSpawnGoalBlend: Float
    public var blockSpawnRadius: Float
    public var blockSpawnYawRange: Float
    /// Bias initial block offsets toward the left (+1) or right (-1) side of
    /// the nominal push direction. Zero preserves the historical full-circle
    /// sampler bit-for-bit; intermediate values mix the two half-planes.
    public var blockSpawnLateralBias: Float
    public var goalProgressWeight: Float
    public var reachProgressWeight: Float
    public var yawProgressWeight: Float
    public var coverageProgressWeight: Float
    /// Optional legacy overlap signal. Exact overlap remains the evaluation
    /// metric, but it is disabled by default for learning: articulated
    /// Push-T PPO is known to settle at a 50-75% overlap local optimum when
    /// trained directly on this piecewise geometric score.
    public var coverageRewardWeight: Float
    /// Difference of the smooth pose potential between consecutive states.
    /// Unlike an absolute dense reward, this pays nothing for waiting in a
    /// near-goal local optimum and penalizes losing previously gained pose
    /// quality. A positive value selects task revision 18.
    public var poseProgressRewardWeight: Float
    /// Smooth translation + rotation precision signal. This deliberately
    /// separates the two pose errors while leaving geometric coverage as the
    /// strict, independent success criterion.
    public var poseRewardWeight: Float
    /// Spatial frequency in inverse metres for the translation and reaching
    /// kernels. The maintained real-scale Push-T reward uses five.
    public var poseRewardDistanceScale: Float
    /// Small dense signal for bringing the end effector to the workpiece.
    public var reachingRewardWeight: Float
    /// Standard control-effort cost on the bounded normalized delta action.
    /// Joint-delta actions are velocities in effect: zero holds the measured
    /// pose, while a constant nonzero action keeps moving. This term lets a
    /// policy learn to stop after placement instead of only smoothing changes.
    public var actionMagnitudePenaltyWeight: Float
    public var actionRatePenaltyWeight: Float
    /// Route near-goal states to the architecture's independent expert actor.
    /// This separates high-energy acquisition from precision placement while
    /// leaving both action choices learned from the same measured state.
    public var precisionGatedActor: Bool
    public var precisionExpertGateCoverage: Float
    /// Release threshold for the precision option. Values below the entry
    /// threshold create hysteresis and prevent rapid expert/base switching at
    /// the geometric boundary; an equal value preserves the revision-20 gate.
    public var precisionExpertReleaseCoverage: Float
    public var freezeBasePolicyExpert: Bool
    /// Potential-based progress toward the goal-relative contact point. This
    /// is task-space shaping only: it never supplies an action or joint target.
    public var pushContactProgressWeight: Float
    public var reachDistancePenaltyWeight: Float
    public var goalDistancePenaltyWeight: Float
    public var yawErrorPenaltyWeight: Float
    public var pushContactDistancePenaltyWeight: Float
    public var successBonus: Float
    /// Optional canonical terminal reward. A positive value replaces the
    /// dense reward on a successful transition, matching environments such
    /// as ManiSkill PushT-v1; zero preserves additive-bonus behavior.
    public var successRewardOverride: Float
    /// Keep the physical episode running after first reaching the success
    /// region. While enabled, instantaneous success remains visible on every
    /// transition, episode reductions are reported at the time limit, and the
    /// terminal override is paid only on steps that remain successful.
    /// This removes the incentive to hover just outside the terminal set and
    /// also measures whether a policy can hold the solved pose.
    public var continueAfterSuccess: Bool
    /// Training curriculum threshold. Evaluation/replay use ManiSkill
    /// PushT-v1's maintained 90% intersection threshold by default.
    public var successCoverage: Float
    /// Optional curriculum guard for loose coverage stages. Canonical 90%
    /// coverage already implies alignment, so the default pi leaves final
    /// evaluation governed solely by geometric overlap.
    public var successYawTolerance: Float
    /// Optional approach-only curriculum. A positive value temporarily makes
    /// pusher-to-object distance the terminal criterion; zero preserves the
    /// canonical geometric-coverage task used by evaluation and replay.
    public var reachCurriculumSuccessDistance: Float
    /// Optional goal-directed contact curriculum. The target lies behind the
    /// object along the block-to-goal axis; it shapes contact selection but
    /// never commands joints or supplies a reference action.
    public var pushContactCurriculumSuccessDistance: Float
    public var pushContactOffset: Float
    /// A contact curriculum cannot claim success by pushing the workpiece
    /// away until its moving contact target reaches the tool. Zero requires
    /// non-regression relative to that episode's initial goal distance.
    public var pushContactCurriculumMaximumGoalRegression: Float
    /// Optional push curriculum layered on goal-directed contact. A positive
    /// value requires this much reduction from the episode's initial
    /// workpiece-to-goal distance before contact can terminate successfully.
    public var pushContactCurriculumMinimumGoalProgress: Float
    /// Absolute terminal pose curriculum. Unlike minimum progress, this
    /// remains feasible for every randomized reset, including workpieces that
    /// begin closer to the goal than a requested progress distance. A positive
    /// value takes precedence over the relative-progress criterion.
    public var pushContactCurriculumMaximumGoalDistance: Float

    public init(numEnvironments: Int, seed: UInt64 = 1,
                maxEpisodeSteps: Int = 100, controlDecimation: Int = 6,
                autoReset: Bool = true,
                jointDeltaActionScale: Float = 0.1,
                endEffectorDeltaActionScale: Float = 0,
                linkLength1: Float = ArmPushTEnv.linkLengths.x,
                linkLength2: Float = ArmPushTEnv.linkLengths.y,
                linkMass: Float = 2.7,
                tipMass: Float = 0.75,
                motorTorque: Float = 100,
                motorStiffness: Float = 1_000,
                motorDamping: Float = 100,
                motorArmature: Float = 0.1,
                blockMass: Float = 0.8,
                blockStaticFriction: Float = 3,
                blockDynamicFriction: Float = 3,
                blockSpawnGoalBlend: Float = 0,
                blockSpawnRadius: Float = 0.07,
                blockSpawnYawRange: Float = 0.35,
                blockSpawnLateralBias: Float = 0,
                goalProgressWeight: Float = 0,
                reachProgressWeight: Float = 0,
                yawProgressWeight: Float = 0,
                coverageProgressWeight: Float = 0,
                coverageRewardWeight: Float = 0,
                poseProgressRewardWeight: Float = 0,
                poseRewardWeight: Float = 1,
                poseRewardDistanceScale: Float = 5,
                reachingRewardWeight: Float = 0.05,
                actionMagnitudePenaltyWeight: Float = 0,
                actionRatePenaltyWeight: Float = 0,
                precisionGatedActor: Bool = false,
                precisionExpertGateCoverage: Float = 0.75,
                precisionExpertReleaseCoverage: Float = 0.75,
                freezeBasePolicyExpert: Bool = false,
                pushContactProgressWeight: Float = 0,
                reachDistancePenaltyWeight: Float = 0,
                goalDistancePenaltyWeight: Float = 0,
                yawErrorPenaltyWeight: Float = 0,
                pushContactDistancePenaltyWeight: Float = 0,
                successBonus: Float = 0,
                successRewardOverride: Float = 3,
                continueAfterSuccess: Bool = false,
                successCoverage: Float = ArmPushTEnv.successCoverage,
                successYawTolerance: Float = .pi,
                reachCurriculumSuccessDistance: Float = 0,
                pushContactCurriculumSuccessDistance: Float = 0,
                pushContactOffset: Float = 0.11,
                pushContactCurriculumMaximumGoalRegression: Float = 0,
                pushContactCurriculumMinimumGoalProgress: Float = 0,
                pushContactCurriculumMaximumGoalDistance: Float = 0) {
        self.numEnvironments = numEnvironments
        self.seed = seed
        self.maxEpisodeSteps = maxEpisodeSteps
        self.controlDecimation = controlDecimation
        self.autoReset = autoReset
        self.jointDeltaActionScale = jointDeltaActionScale
        self.endEffectorDeltaActionScale = endEffectorDeltaActionScale
        self.linkLength1 = linkLength1
        self.linkLength2 = linkLength2
        self.linkMass = linkMass
        self.tipMass = tipMass
        self.motorTorque = motorTorque
        self.motorStiffness = motorStiffness
        self.motorDamping = motorDamping
        self.motorArmature = motorArmature
        self.blockMass = blockMass
        self.blockStaticFriction = blockStaticFriction
        self.blockDynamicFriction = blockDynamicFriction
        self.blockSpawnGoalBlend = blockSpawnGoalBlend
        self.blockSpawnRadius = blockSpawnRadius
        self.blockSpawnYawRange = blockSpawnYawRange
        self.blockSpawnLateralBias = blockSpawnLateralBias
        self.goalProgressWeight = goalProgressWeight
        self.reachProgressWeight = reachProgressWeight
        self.yawProgressWeight = yawProgressWeight
        self.coverageProgressWeight = coverageProgressWeight
        self.coverageRewardWeight = coverageRewardWeight
        self.poseProgressRewardWeight = poseProgressRewardWeight
        self.poseRewardWeight = poseRewardWeight
        self.poseRewardDistanceScale = poseRewardDistanceScale
        self.reachingRewardWeight = reachingRewardWeight
        self.actionMagnitudePenaltyWeight = actionMagnitudePenaltyWeight
        self.actionRatePenaltyWeight = actionRatePenaltyWeight
        self.precisionGatedActor = precisionGatedActor
        self.precisionExpertGateCoverage = precisionExpertGateCoverage
        self.precisionExpertReleaseCoverage =
            precisionExpertReleaseCoverage
        self.freezeBasePolicyExpert = freezeBasePolicyExpert
        self.pushContactProgressWeight = pushContactProgressWeight
        self.reachDistancePenaltyWeight = reachDistancePenaltyWeight
        self.goalDistancePenaltyWeight = goalDistancePenaltyWeight
        self.yawErrorPenaltyWeight = yawErrorPenaltyWeight
        self.pushContactDistancePenaltyWeight = pushContactDistancePenaltyWeight
        self.successBonus = successBonus
        self.successRewardOverride = successRewardOverride
        self.continueAfterSuccess = continueAfterSuccess
        self.successCoverage = successCoverage
        self.successYawTolerance = successYawTolerance
        self.reachCurriculumSuccessDistance = reachCurriculumSuccessDistance
        self.pushContactCurriculumSuccessDistance =
            pushContactCurriculumSuccessDistance
        self.pushContactOffset = pushContactOffset
        self.pushContactCurriculumMaximumGoalRegression =
            pushContactCurriculumMaximumGoalRegression
        self.pushContactCurriculumMinimumGoalProgress =
            pushContactCurriculumMinimumGoalProgress
        self.pushContactCurriculumMaximumGoalDistance =
            pushContactCurriculumMaximumGoalDistance
    }
}

public struct ArmPushTState {
    public var tipPosition: SIMD2<Float>
    public var tipVelocity: SIMD2<Float>
    public var blockPosition: SIMD2<Float>
    public var blockVelocity: SIMD2<Float>
    public var blockYaw: Float
    public var blockAngularVelocity: Float
    public var jointAngles: [Float]
}

/// Two-link SCARA-style horizontal robot arm with a vertical compliant pusher. Arm
/// links run above the workpiece; only the tool contacts the T, giving a
/// clean first articulated manipulation benchmark without hiding IK in the
/// task. The policy commands normalized joint-position targets directly.
public final class ArmPushTEnv {
    public struct EnvRefs {
        public var center: F3
        public var bodies: [Int]
        public var motors: [Int]
        public var tip: Int
        public var blockBar: Int
        public var blockStem: Int
        public var goalPosition: SIMD2<Float>
        public var goalYaw: Float
    }

    public static let jointRanges: [(Float, Float)] = [(-2.9, 2.9), (0.0, 3.05)]
    public static let defaultJointPositions: [Float] = [0, 0.15]
    public static let actionScales: [Float] = [2.8, 2.8]
    /// Real-scale T geometry requires the pusher to reach the object's far
    /// face at the goal, not merely its centre. The original 0.64 m prototype
    /// arm could not do so and made a subset of canonical resets physically
    /// unrecoverable. These Panda-scale planar lengths retain the 20 x 5 x 4
    /// cm ManiSkill PushT-v1 workpiece while providing explicit workspace
    /// margin around every goal-face contact.
    public static let linkLengths: SIMD2<Float> = SIMD2(0.42, 0.40)
    public static let basePosition = SIMD2<Float>(-0.40, 0)
    public static let goalPosition = SIMD2<Float>(0.17, 0.17)
    public static let blockBarSize = F3(0.20, 0.05, 0.04)
    public static let blockStemSize = F3(0.05, 0.15, 0.04)
    public static let defaultBlockMass: Float = 0.8
    public static let defaultBlockFriction: Float = 3
    public static let blockVolume: Float = blockBarSize.x * blockBarSize.y
        * blockBarSize.z + blockStemSize.x * blockStemSize.y * blockStemSize.z
    /// ManiSkill PushT-v1 terminates at 90% geometric goal coverage. Keeping
    /// the threshold beside the geometry prevents a loose pose tolerance from
    /// being mistaken for the benchmark's normalized score.
    public static let successCoverage: Float = 0.90
    private static let goalArea: Float = blockBarSize.x * blockBarSize.y
        + blockStemSize.x * blockStemSize.y

    public let numEnvironments: Int
    public let scene: PhysicsScene
    public let solver: GPUSolver
    public private(set) var refs: [EnvRefs] = []
    public let blockSpawnGoalBlend: Float
    public let blockSpawnRadius: Float
    public let blockSpawnYawRange: Float
    public let blockSpawnLateralBias: Float
    public let configuredLinkLengths: SIMD2<Float>
    public let linkMass: Float
    public let tipMass: Float
    public let motorTorque: Float
    public let motorStiffness: Float
    public let motorDamping: Float
    public let motorArmature: Float
    public let blockMass: Float
    public let blockStaticFriction: Float
    public let blockDynamicFriction: Float
    private let spawnPoses: [(F3, Quat)]

    public init(numEnvironments: Int, seed: UInt64 = 1,
                blockSpawnGoalBlend: Float = 0,
                blockSpawnRadius: Float = 0.07,
                blockSpawnYawRange: Float = 0.35,
                blockSpawnLateralBias: Float = 0,
                linkLength1: Float = linkLengths.x,
                linkLength2: Float = linkLengths.y,
                linkMass: Float = 2.7,
                tipMass: Float = 0.75,
                motorTorque: Float = 100,
                motorStiffness: Float = 1_000,
                motorDamping: Float = 100,
                motorArmature: Float = 0.1,
                blockMass: Float = defaultBlockMass,
                blockStaticFriction: Float = defaultBlockFriction,
                blockDynamicFriction: Float = defaultBlockFriction) throws {
        precondition(numEnvironments > 0)
        precondition((0...1).contains(blockSpawnGoalBlend))
        precondition(blockSpawnRadius >= 0)
        precondition((0...Float.pi).contains(blockSpawnYawRange))
        precondition((-1...1).contains(blockSpawnLateralBias))
        precondition(linkLength1 > 0 && linkLength2 > 0)
        precondition(linkMass > 0 && tipMass > 0 && blockMass > 0)
        precondition(motorTorque > 0 && motorStiffness > 0)
        precondition(motorDamping >= 0 && motorArmature >= 0)
        precondition(blockStaticFriction >= 0 && blockDynamicFriction >= 0)
        self.numEnvironments = numEnvironments
        self.blockSpawnGoalBlend = blockSpawnGoalBlend
        self.blockSpawnRadius = blockSpawnRadius
        self.blockSpawnYawRange = blockSpawnYawRange
        self.blockSpawnLateralBias = blockSpawnLateralBias
        self.configuredLinkLengths = SIMD2(linkLength1, linkLength2)
        self.linkMass = linkMass
        self.tipMass = tipMass
        self.motorTorque = motorTorque
        self.motorStiffness = motorStiffness
        self.motorDamping = motorDamping
        self.motorArmature = motorArmature
        self.blockMass = blockMass
        self.blockStaticFriction = blockStaticFriction
        self.blockDynamicFriction = blockDynamicFriction
        var built = PhysicsScene(name: "arm-pusht")
        built.settings.dt = 1 / 120
        built.settings.iterations = 20
        built.settings.betaLin = 20_000
        built.settings.betaAng = 400
        built.settings.lambdaMax = 900
        // A close oblique default exposes the arm joints, end effector, T,
        // and target together. The replay view remains freely orbitable.
        built.settings.cameraDistance = 1.0
        built.settings.cameraTargetZ = 0.04
        Demos.addGround(&built, friction: 1.25)
        var rng = SplitMix64(seed: seed)
        let side = Int(ceil(Double(numEnvironments).squareRoot()))
        var builtRefs = [EnvRefs]()
        for e in 0..<numEnvironments {
            let center = F3(Float(e % side) * 1.4, Float(e / side) * 1.4, 0)
            builtRefs.append(Self.buildOne(
                &built, center: center, rng: &rng,
                blockSpawnGoalBlend: blockSpawnGoalBlend,
                blockSpawnRadius: blockSpawnRadius,
                blockSpawnYawRange: blockSpawnYawRange,
                blockSpawnLateralBias: blockSpawnLateralBias,
                linkLengths: configuredLinkLengths,
                linkMass: linkMass, tipMass: tipMass,
                motorTorque: motorTorque,
                motorStiffness: motorStiffness,
                motorDamping: motorDamping,
                motorArmature: motorArmature,
                blockMass: blockMass,
                blockStaticFriction: blockStaticFriction,
                blockDynamicFriction: blockDynamicFriction,
                goalMarkers: numEnvironments <= 4))
        }
        refs = builtRefs
        scene = built
        spawnPoses = built.bodies.map { ($0.position, $0.rotation) }
        solver = try GPUSolver(scene: built)
    }

    private static func spawnAngle(
        rng: inout SplitMix64, lateralBias: Float
    ) -> Float {
        if lateralBias == 0 {
            // Checkpoint and held-out seed compatibility: do not consume an
            // extra random number or perturb the legacy full-circle sampler.
            return 2 * .pi * rng.nextFloat()
        }
        let leftProbability = (lateralBias + 1) * 0.5
        let samplesLeft = rng.nextFloat() < leftProbability
        let nominalPushAngle = atan2(goalPosition.y, goalPosition.x)
        let lateralAngle = nominalPushAngle + .pi * 0.5
        let sideCenter = samplesLeft ? lateralAngle : lateralAngle + .pi
        return sideCenter + (rng.nextFloat() - 0.5) * .pi
    }

    private static func buildOne(_ s: inout PhysicsScene, center c: F3,
                                 rng: inout SplitMix64,
                                 blockSpawnGoalBlend: Float,
                                 blockSpawnRadius: Float,
                                 blockSpawnYawRange: Float,
                                 blockSpawnLateralBias: Float,
                                 linkLengths: SIMD2<Float>,
                                 linkMass: Float, tipMass: Float,
                                 motorTorque: Float,
                                 motorStiffness: Float,
                                 motorDamping: Float,
                                 motorArmature: Float,
                                 blockMass: Float,
                                 blockStaticFriction: Float,
                                 blockDynamicFriction: Float,
                                 goalMarkers: Bool) -> EnvRefs {
        let baseCenter = c + F3(basePosition.x, basePosition.y, 0.056)
        let base = s.addBody(size: F3(0.064, 0.064, 0.112), density: 0,
                             friction: 0.8, position: baseCenter)
        let lengths = [linkLengths.x, linkLengths.y]
        var links = [Int]()
        var x = baseCenter.x
        for length in lengths {
            let size = F3(length, 0.026, 0.022)
            let density = linkMass / (size.x * size.y * size.z)
            let link = s.addBody(size: size, density: density,
                                 friction: 0.35,
                                 position: F3(x + length * 0.5, c.y, 0.11))
            links.append(link)
            x += length
        }
        var motors = [Int]()
        motors.append(s.joints.count)
        s.addJoint(SceneJoint(bodyA: base, bodyB: links[0],
                              rA: F3(0, 0, 0.054),
                              rB: F3(-lengths[0] * 0.5, 0, 0),
                              stiffnessLin: .infinity, stiffnessAng: .infinity,
                              hingeAxis: F3(0, 0, 1),
                              motorTorque: motorTorque,
                              motorStiffness: motorStiffness,
                              motorDamping: motorDamping,
                              armature: motorArmature,
                              limitLo: jointRanges[0].0, limitHi: jointRanges[0].1))
        for j in 1..<2 {
            motors.append(s.joints.count)
            s.addJoint(SceneJoint(bodyA: links[j - 1], bodyB: links[j],
                                  rA: F3(lengths[j - 1] * 0.5, 0, 0),
                                  rB: F3(-lengths[j] * 0.5, 0, 0),
                                  stiffnessLin: .infinity, stiffnessAng: .infinity,
                                  hingeAxis: F3(0, 0, 1),
                                  motorTorque: motorTorque,
                                  motorStiffness: motorStiffness,
                                  motorDamping: motorDamping,
                                  armature: motorArmature,
                                  limitLo: jointRanges[j].0,
                                  limitHi: jointRanges[j].1))
        }
        let tipLength: Float = 0.116
        let tipRadius: Float = 0.016
        let tipVolume = Float.pi * tipRadius * tipRadius
            * (tipLength + 4 * tipRadius / 3)
        let tip = s.addCapsule(
            length: tipLength, radius: tipRadius,
            density: tipMass / tipVolume,
            friction: 0.45, position: F3(x, c.y, 0.084))
        s.addJoint(SceneJoint(bodyA: links[1], bodyB: tip,
                              rA: F3(lengths[1] * 0.5, 0, -0.026), rB: .zero,
                              stiffnessLin: .infinity, stiffnessAng: .infinity))

        let goalPosition = Self.goalPosition
        let goalYaw: Float = 0
        let angle = Self.spawnAngle(
            rng: &rng, lateralBias: blockSpawnLateralBias)
        let radius = blockSpawnRadius * sqrt(rng.nextFloat())
        let yaw = (2 * rng.nextFloat() - 1) * blockSpawnYawRange
        let q = Quat(angle: yaw, axis: F3(0, 0, 1))
        let anchor = goalPosition * blockSpawnGoalBlend
        let blockCenter = c + F3(anchor.x + cos(angle) * radius,
                                 anchor.y + sin(angle) * radius, 0)
        let blockDensity = blockMass / blockVolume
        let bar = s.addBody(
            size: blockBarSize, density: blockDensity,
            friction: blockStaticFriction,
            dynamicFriction: blockDynamicFriction,
            position: blockCenter + q.act(F3(0, 0.025, 0))
                + F3(0, 0, 0.02), rotation: q)
        let stem = s.addBody(
            size: blockStemSize, density: blockDensity,
            friction: blockStaticFriction,
            dynamicFriction: blockDynamicFriction,
            position: blockCenter + q.act(F3(0, -0.075, 0))
                + F3(0, 0, 0.02), rotation: q)
        let midpoint = (s.bodies[bar].position + s.bodies[stem].position) * 0.5
        s.addJoint(SceneJoint(
            bodyA: bar, bodyB: stem,
            rA: s.bodies[bar].rotation.inverse.act(midpoint - s.bodies[bar].position),
            rB: s.bodies[stem].rotation.inverse.act(midpoint - s.bodies[stem].position),
            stiffnessLin: .infinity, stiffnessAng: .infinity))
        s.addJoint(SceneJoint(bodyA: bar, bodyB: stem, rA: .zero, rB: .zero,
                              stiffnessLin: 0, stiffnessAng: 0))

        if goalMarkers {
            let goalRotation = Quat(angle: goalYaw, axis: F3(0, 0, 1))
            let goalCenter = c + F3(goalPosition.x, goalPosition.y, 0)
            let markerBar = s.addBody(
                size: F3(0.20, 0.05, 0.02), density: 0, friction: 0,
                position: goalCenter + goalRotation.act(F3(0, 0.025, 0))
                    + F3(0, 0, -0.0076), rotation: goalRotation)
            let markerStem = s.addBody(
                size: F3(0.05, 0.15, 0.02), density: 0, friction: 0,
                position: goalCenter + goalRotation.act(F3(0, -0.075, 0))
                    + F3(0, 0, -0.0076), rotation: goalRotation)
            // Zero-stiffness joints are the engine's collision-exclusion
            // mechanism. The visible target cannot push the T or tool and is
            // absent from batched training scenes.
            for marker in [markerBar, markerStem] {
                for body in [bar, stem, tip] {
                    s.addJoint(SceneJoint(
                        bodyA: marker, bodyB: body, rA: .zero, rB: .zero,
                        stiffnessLin: 0, stiffnessAng: 0))
                }
            }
        }

        for sign in [Float(-1), 1] {
            _ = s.addBody(size: F3(1.28, 0.024, 0.09), density: 0,
                          friction: 0.4,
                          position: c + F3(0, sign * 0.64, 0.044))
            _ = s.addBody(size: F3(0.024, 1.28, 0.09), density: 0,
                          friction: 0.4,
                          position: c + F3(sign * 0.64, 0, 0.044))
        }
        return EnvRefs(center: c, bodies: links + [tip, bar, stem], motors: motors,
                       tip: tip, blockBar: bar, blockStem: stem,
                       goalPosition: goalPosition, goalYaw: goalYaw)
    }

    public func step(normalizedActions: ContiguousArray<Float>, decimation: Int) {
        do {
            try stepChecked(
                normalizedActions: normalizedActions, decimation: decimation)
        } catch {
            fatalError("Arm Push-T simulation failed: \(error.localizedDescription)")
        }
    }

    public func stepChecked(normalizedActions: ContiguousArray<Float>,
                            decimation: Int) throws {
        precondition(normalizedActions.count == numEnvironments * 2)
        try solver.synchronize()
        var updates = [GPUSolver.MotorTargetUpdate]()
        updates.reserveCapacity(normalizedActions.count)
        for e in 0..<numEnvironments {
            for j in 0..<2 {
                let a = simd_clamp(normalizedActions[e * 2 + j], -1, 1)
                let range = Self.jointRanges[j]
                let target = simd_clamp(Self.defaultJointPositions[j]
                    + a * Self.actionScales[j], range.0, range.1)
                updates.append(.init(joint: refs[e].motors[j], angle: target))
            }
        }
        solver.setMotorTargets(updates)
        for _ in 0..<decimation { try solver.submitStep() }
        try solver.synchronize()
    }

    /// Analytic elbow-up IK for the benchmark's planar two-link arm. This is
    /// used by demonstration generators and interactive diagnostics only;
    /// learned policies still emit the same normalized joint targets directly.
    public static func normalizedJointTargets(
        tipTarget: SIMD2<Float>
    ) -> SIMD2<Float> {
        normalizedJointTargets(tipTarget: tipTarget, linkLengths: linkLengths)
    }

    /// IK against this environment's signed workspace geometry. Controllers
    /// must use the instance form so diagnostic overrides and serialized task
    /// configurations cannot silently resolve targets with different links.
    public func normalizedJointTargets(
        tipTarget: SIMD2<Float>
    ) -> SIMD2<Float> {
        Self.normalizedJointTargets(
            tipTarget: tipTarget, linkLengths: configuredLinkLengths)
    }

    public static func normalizedJointTargets(
        tipTarget: SIMD2<Float>, linkLengths: SIMD2<Float>
    ) -> SIMD2<Float> {
        precondition(linkLengths.x > 0 && linkLengths.y > 0)
        let relative = tipTarget - basePosition
        let l1 = linkLengths.x, l2 = linkLengths.y
        let minimumReach = abs(l1 - l2) + 1e-4
        let maximumReach = l1 + l2 - 1e-4
        let rawRadius = length(relative)
        let radius = simd_clamp(rawRadius, minimumReach, maximumReach)
        let direction = rawRadius > 1e-6
            ? relative / rawRadius : SIMD2<Float>(1, 0)
        let reachable = direction * radius
        let cosineElbow = simd_clamp(
            (radius * radius - l1 * l1 - l2 * l2) / (2 * l1 * l2),
            -1, 1)
        let q2 = acos(cosineElbow)
        var q1 = atan2(reachable.y, reachable.x)
            - atan2(l2 * sin(q2), l1 + l2 * cos(q2))
        q1 -= 2 * .pi * floor((q1 + .pi) / (2 * .pi))
        let angles = SIMD2(
            simd_clamp(q1, jointRanges[0].0, jointRanges[0].1),
            simd_clamp(q2, jointRanges[1].0, jointRanges[1].1))
        return simd_clamp(
            SIMD2((angles.x - defaultJointPositions[0]) / actionScales[0],
                  (angles.y - defaultJointPositions[1]) / actionScales[1]),
            SIMD2(repeating: -1), SIMD2(repeating: 1))
    }

    /// Convert normalized delta actions to the environment's normalized
    /// absolute motor-target interface. Keeping this conversion explicit and
    /// pure makes controller semantics independently testable and prevents a
    /// policy from accidentally commanding the joint limits on every step.
    public static func normalizedJointTargets(
        currentAngles: SIMD2<Float>, deltaActions: SIMD2<Float>,
        deltaScale: Float
    ) -> SIMD2<Float> {
        precondition(deltaScale > 0)
        var targets = SIMD2<Float>.zero
        for j in 0..<2 {
            let requested = simd_clamp(deltaActions[j], -1, 1)
            let range = jointRanges[j]
            let angle = simd_clamp(
                currentAngles[j] + requested * deltaScale,
                range.0, range.1)
            targets[j] = simd_clamp(
                (angle - defaultJointPositions[j]) / actionScales[j], -1, 1)
        }
        return targets
    }

    public func reset(_ environmentIDs: [Int], seeds: [UInt64]) {
        precondition(environmentIDs.count == seeds.count)
        var poses = [GPUSolver.BodyPoseUpdate]()
        var motors = [GPUSolver.MotorTargetUpdate]()
        for (offset, e) in environmentIDs.enumerated() {
            var rng = SplitMix64(seed: seeds[offset])
            let r = refs[e]
            for body in r.bodies.prefix(r.motors.count + 1) {
                let spawn = spawnPoses[body]
                poses.append(.init(body: body, position: spawn.0, rotation: spawn.1))
            }
            for (j, joint) in r.motors.enumerated() {
                motors.append(.init(joint: joint,
                                    angle: Self.defaultJointPositions[j]
                                        + (rng.nextFloat() - 0.5) * 0.02))
            }
            let angle = Self.spawnAngle(
                rng: &rng, lateralBias: blockSpawnLateralBias)
            let radius = blockSpawnRadius * sqrt(rng.nextFloat())
            let yaw = (2 * rng.nextFloat() - 1) * blockSpawnYawRange
            let q = Quat(angle: yaw, axis: F3(0, 0, 1))
            let anchor = r.goalPosition * blockSpawnGoalBlend
            let center = r.center + F3(anchor.x + cos(angle) * radius,
                                       anchor.y + sin(angle) * radius, 0)
            poses.append(.init(body: r.blockBar,
                               position: center + q.act(F3(0, 0.025, 0))
                                   + F3(0, 0, 0.02), rotation: q))
            poses.append(.init(body: r.blockStem,
                               position: center + q.act(F3(0, -0.075, 0))
                                   + F3(0, 0, 0.02), rotation: q))
        }
        solver.setBodyPoses(poses)
        solver.setMotorTargets(motors)
    }

    public func states() -> [ArmPushTState] {
        var bodyIDs = [Int](), motorIDs = [Int]()
        bodyIDs.reserveCapacity(numEnvironments * 3)
        motorIDs.reserveCapacity(numEnvironments * 2)
        for r in refs {
            bodyIDs.append(contentsOf: [r.tip, r.blockBar, r.blockStem])
            motorIDs.append(contentsOf: r.motors)
        }
        let bodies = solver.bodyStates(bodyIDs)
        let angles = solver.motorAngles(motorIDs)
        return (0..<numEnvironments).map { e in
            let r = refs[e]
            let tip = bodies[e * 3]
            let bar = bodies[e * 3 + 1]
            let forward = bar.rotation.act(F3(1, 0, 0))
            // The task origin is the junction between the bar and stem, not
            // the arithmetic mean of their unequal centers. The old midpoint
            // was displaced by 0.02 m and rotated with the block, corrupting
            // both pose observations and the reported goal error.
            let lateral = bar.rotation.act(F3(0, 1, 0))
            let origin = bar.position - lateral * 0.025 - r.center
            return ArmPushTState(
                tipPosition: SIMD2(tip.position.x - r.center.x,
                                   tip.position.y - r.center.y),
                tipVelocity: SIMD2(tip.linearVelocity.x, tip.linearVelocity.y),
                blockPosition: SIMD2(origin.x, origin.y),
                blockVelocity: SIMD2(bar.linearVelocity.x, bar.linearVelocity.y),
                blockYaw: atan2(forward.y, forward.x),
                blockAngularVelocity: bar.angularVelocity.z,
                jointAngles: Array(angles[(e * 2)..<(e * 2 + 2)]))
        }
    }

    public func coverage(_ e: Int, state: ArmPushTState) -> Float {
        Self.coverage(blockPosition: state.blockPosition,
                      blockYaw: state.blockYaw,
                      goalPosition: refs[e].goalPosition,
                      goalYaw: refs[e].goalYaw)
    }

    public func success(_ e: Int, state: ArmPushTState) -> Bool {
        coverage(e, state: state) > Self.successCoverage
    }

    /// Exact 2-D intersection-over-goal-area for the compound T. The bar and
    /// stem have disjoint interiors, so summing the four convex rectangle
    /// intersections is the union area without double counting.
    public static func coverage(blockPosition: SIMD2<Float>, blockYaw: Float,
                                goalPosition: SIMD2<Float>, goalYaw: Float) -> Float {
        let block = rectangles(position: blockPosition, yaw: blockYaw)
        let goal = rectangles(position: goalPosition, yaw: goalYaw)
        var intersection: Float = 0
        for subject in block {
            for clip in goal {
                intersection += polygonArea(clipPolygon(subject, by: clip))
            }
        }
        return simd_clamp(intersection / goalArea, 0, 1)
    }

    private static func rectangles(position: SIMD2<Float>, yaw: Float)
        -> [[SIMD2<Float>]] {
        [rectangle(position: position, localCenter: SIMD2(0, 0.025),
                   halfExtents: SIMD2(0.10, 0.025), yaw: yaw),
         rectangle(position: position, localCenter: SIMD2(0, -0.075),
                   halfExtents: SIMD2(0.025, 0.075), yaw: yaw)]
    }

    private static func rectangle(position: SIMD2<Float>,
                                  localCenter: SIMD2<Float>,
                                  halfExtents h: SIMD2<Float>, yaw: Float)
        -> [SIMD2<Float>] {
        let c = cos(yaw), s = sin(yaw)
        func transform(_ p: SIMD2<Float>) -> SIMD2<Float> {
            let q = p + localCenter
            return position + SIMD2(c * q.x - s * q.y,
                                    s * q.x + c * q.y)
        }
        return [transform(SIMD2(-h.x, -h.y)),
                transform(SIMD2( h.x, -h.y)),
                transform(SIMD2( h.x,  h.y)),
                transform(SIMD2(-h.x,  h.y))]
    }

    private static func cross(_ a: SIMD2<Float>, _ b: SIMD2<Float>) -> Float {
        a.x * b.y - a.y * b.x
    }

    /// Sutherland-Hodgman clipping for two counter-clockwise convex polygons.
    private static func clipPolygon(_ subject: [SIMD2<Float>],
                                    by clip: [SIMD2<Float>]) -> [SIMD2<Float>] {
        var output = subject
        for i in clip.indices {
            let a = clip[i]
            let b = clip[(i + 1) % clip.count]
            let edge = b - a
            let input = output
            output.removeAll(keepingCapacity: true)
            guard !input.isEmpty else { break }
            for j in input.indices {
                let p = input[j]
                let q = input[(j + 1) % input.count]
                let pSide = cross(edge, p - a)
                let qSide = cross(edge, q - a)
                let pInside = pSide >= -1e-6
                let qInside = qSide >= -1e-6
                if pInside != qInside {
                    let denominator = pSide - qSide
                    if abs(denominator) > 1e-8 {
                        output.append(p + (q - p) * (pSide / denominator))
                    }
                }
                if qInside { output.append(q) }
            }
        }
        return output
    }

    private static func polygonArea(_ polygon: [SIMD2<Float>]) -> Float {
        guard polygon.count >= 3 else { return 0 }
        var twiceArea: Float = 0
        for i in polygon.indices {
            twiceArea += cross(polygon[i], polygon[(i + 1) % polygon.count])
        }
        return abs(twiceArea) * 0.5
    }
}

/// Batched state-feedback demonstration policy for articulated Push-T.
///
/// The expert plans end-effector contact points, then uses the public IK above
/// to express them through the exact same two joint targets as an RL policy.
/// It is intentionally outside `ArmPushTTask`: observations, rewards,
/// evaluation, and replay expose no expert phase or action.
public final class ArmPushTGeometricExpert {
    /// Route transitions must be tight relative to the 16 mm pusher radius.
    /// The former 100 mm tolerance let the tool leave a waypoint while it was
    /// still an entire T half-width away, cutting directly through the object
    /// and applying an unintended impulse before the planned contact.
    private static let waypointTolerance: Float = 0.018
    private static let approachTolerance: Float = 0.012
    private static let toolRadius: Float = 0.016
    private static let barContactAnchors: [SIMD2<Float>] = [
        SIMD2(-0.072, 0.025), SIMD2(-0.054, 0.025),
        SIMD2(-0.036, 0.025), SIMD2(-0.018, 0.025),
        SIMD2( 0.000, 0.025), SIMD2( 0.018, 0.025),
        SIMD2( 0.036, 0.025), SIMD2( 0.054, 0.025),
        SIMD2( 0.072, 0.025),
    ]

    private let numEnvironments: Int
    private let contactPreload: Float
    private var routeModes: [Int8]
    private var routePaths: [[SIMD2<Float>]]
    private var routeIndices: [Int]
    private var routeCenters: [SIMD2<Float>]
    private var routeDirections: [SIMD2<Float>]

    public init(numEnvironments: Int, contactPreload: Float = 0.014) {
        precondition(numEnvironments > 0)
        precondition(contactPreload > 0
            && contactPreload < Self.toolRadius)
        self.numEnvironments = numEnvironments
        self.contactPreload = contactPreload
        routeModes = [Int8](repeating: 0, count: numEnvironments)
        routePaths = [[SIMD2<Float>]](
            repeating: [], count: numEnvironments)
        routeIndices = [Int](repeating: 0, count: numEnvironments)
        routeCenters = [SIMD2<Float>](repeating: .zero,
                                      count: numEnvironments)
        routeDirections = [SIMD2<Float>](repeating: .zero,
                                         count: numEnvironments)
    }

    public func actions(
        environment: ArmPushTEnv, states: [ArmPushTState]
    ) -> ContiguousArray<Float> {
        precondition(environment.numEnvironments == numEnvironments)
        precondition(states.count == numEnvironments)
        var result = ContiguousArray(repeating: Float(0),
                                     count: numEnvironments * 2)
        for e in 0..<numEnvironments {
            let desired = tipTarget(
                environment: e, state: states[e],
                goal: environment.refs[e].goalPosition,
                goalYaw: environment.refs[e].goalYaw)
            let delta = desired - states[e].tipPosition
            let goalDistance = length(
                environment.refs[e].goalPosition - states[e].blockPosition)
            let maximumTargetStep: Float = goalDistance < 0.08 ? 0.035 : 0.10
            let target = states[e].tipPosition
                + delta * min(
                    1, maximumTargetStep / max(length(delta), 1e-6))
            let action = environment.normalizedJointTargets(tipTarget: target)
            result[e * 2] = action.x
            result[e * 2 + 1] = action.y
        }
        return result
    }

    private func tipTarget(
        environment e: Int, state: ArmPushTState,
        goal: SIMD2<Float>, goalYaw: Float
    ) -> SIMD2<Float> {
        var yawError = state.blockYaw - goalYaw
        yawError -= 2 * .pi * (yawError / (2 * .pi)).rounded()
        let block = state.blockPosition
        let tip = state.tipPosition
        let delta = goal - block
        let goalDistance = length(delta)
        if ArmPushTEnv.coverage(
            blockPosition: block, blockYaw: state.blockYaw,
            goalPosition: goal, goalYaw: goalYaw) > 0.92 {
            return tip
        }
        let pushDirection = goalDistance > 1e-6
            ? delta / goalDistance
            : goal / max(length(goal), 1e-6)

        // Correct initial yaw, retain a useful translation contact through
        // the middle of the rollout, then tighten orientation only in the
        // final goal neighbourhood. Each phase transition uses a collision-
        // clear route instead of sliding a loaded pusher through the object.
        let translationYawLimit: Float = goalDistance > 0.04 ? 0.14 : 0.055
        let keepsTranslation = routeModes[e] == 3
            && abs(yawError) < translationYawLimit
        let rotates = !keepsTranslation && abs(yawError) > 0.025
        let mode: Int8 = rotates ? (yawError > 0 ? 1 : 2) : 3

        let u = SIMD2<Float>(cos(state.blockYaw), sin(state.blockYaw))
        let v = SIMD2<Float>(-u.y, u.x)
        let localCentreOfMass = SIMD2<Float>(0, -0.017857143)
        let centreOfMass = block + v * localCentreOfMass.y
        let contactSurface: SIMD2<Float>
        if rotates {
            let desiredMoment = simd_clamp(
                -0.20 * yawError, -0.06, 0.06)
            var bestSurface = Self.rearSurfacePoint(
                block: block, yaw: state.blockYaw,
                pushDirection: pushDirection,
                localAnchor: Self.barContactAnchors[0])
            var momentError = abs(Self.cross(
                bestSurface - centreOfMass, pushDirection) - desiredMoment)
            for anchor in Self.barContactAnchors.dropFirst() {
                let surface = Self.rearSurfacePoint(
                    block: block, yaw: state.blockYaw,
                    pushDirection: pushDirection, localAnchor: anchor)
                let candidateError = abs(Self.cross(
                    surface - centreOfMass, pushDirection) - desiredMoment)
                if candidateError < momentError {
                    bestSurface = surface
                    momentError = candidateError
                }
            }
            contactSurface = bestSurface
        } else {
            // A ray through the compound body's actual area centroid produces
            // zero moment by construction. Restricting this contact to the bar
            // created a systematic residual torque during goal translation.
            contactSurface = Self.rearSurfacePoint(
                block: block, yaw: state.blockYaw,
                pushDirection: pushDirection,
                localAnchor: localCentreOfMass)
        }
        // A deep preload is useful for overcoming static friction while far
        // from the goal, but it becomes an impulse launcher during the final
        // centimetres. Back it off with both pose error and outgoing object
        // velocity; zero means tangent contact, never attraction or teleport.
        let precisionPreload = min(
            contactPreload, 0.002 + 0.10 * goalDistance)
        let outgoingSpeed = max(0, dot(state.blockVelocity, pushDirection))
        let effectivePreload = goalDistance < 0.08
            ? max(0, precisionPreload - 0.04 * outgoingSpeed)
            : contactPreload
        let contact = contactSurface - pushDirection
            * (Self.toolRadius - effectivePreload)
        let standOff: Float = 0.045

        let approach = contact - pushDirection * standOff
        let centerMoved = length(routeCenters[e] - block) > 0.01
        let contactSideChanged = dot(routeDirections[e], pushDirection) < 0.95
        if routeModes[e] != mode
            || contactSideChanged
            || (routeIndices[e] < routePaths[e].count && centerMoved) {
            routeModes[e] = mode
            routePaths[e] = Self.collisionClearRoute(
                from: tip, to: approach, around: block)
            routeIndices[e] = 0
            routeCenters[e] = block
            routeDirections[e] = pushDirection
        }
        while routeIndices[e] < routePaths[e].count {
            let waypoint = routePaths[e][routeIndices[e]]
            let isApproach = routeIndices[e] == routePaths[e].count - 1
            let tolerance = isApproach
                ? Self.approachTolerance : Self.waypointTolerance
            if length(tip - waypoint) > tolerance { return waypoint }
            routeIndices[e] += 1
        }
        return contact
    }

    /// Construct a short polyline that remains outside a conservative circle
    /// enclosing the complete T plus the pusher radius. Chords are limited to
    /// 30 degrees so they cannot cut back through the obstacle. The final leg
    /// is radial into the intended contact face.
    private static func collisionClearRoute(
        from tip: SIMD2<Float>, to approach: SIMD2<Float>,
        around block: SIMD2<Float>
    ) -> [SIMD2<Float>] {
        let segment = approach - tip
        let projection = simd_clamp(
            dot(block - tip, segment) / max(dot(segment, segment), 1e-6),
            0, 1)
        let obstacleRadius: Float = 0.170
        if length(tip + projection * segment - block) >= obstacleRadius {
            return [approach]
        }

        let safeRadius: Float = 0.190
        var start = tip - block
        if length(start) < 1e-5 { start = SIMD2(1, 0) }
        let end = approach - block
        let startAngle = atan2(start.y, start.x)
        let endAngle = atan2(end.y, end.x)
        var delta = endAngle - startAngle
        delta -= 2 * .pi * (delta / (2 * .pi)).rounded()
        let arcSteps = max(1, Int(ceil(abs(delta) / (.pi / 6))))
        var route = [SIMD2<Float>]()
        route.reserveCapacity(arcSteps + 2)
        if length(start) < safeRadius - Self.waypointTolerance {
            route.append(block + safeRadius
                * SIMD2(cos(startAngle), sin(startAngle)))
        }
        for step in 1...arcSteps {
            let angle = startAngle + delta * Float(step) / Float(arcSteps)
            route.append(block + safeRadius
                * SIMD2(cos(angle), sin(angle)))
        }
        route.append(approach)
        return route
    }

    /// Rear boundary hit by a ray through the compound T's area centroid.
    /// Applying force on this line minimizes unwanted yaw while translating.
    /// A short deterministic march is clearer and less error-prone here than
    /// special-casing the two touching rectangles' ray intervals.
    private static func rearSurfacePoint(
        block: SIMD2<Float>, yaw: Float, pushDirection: SIMD2<Float>,
        localAnchor: SIMD2<Float> = SIMD2(0, -0.017857143)
    ) -> SIMD2<Float> {
        let u = SIMD2<Float>(cos(yaw), sin(yaw))
        let v = SIMD2<Float>(-u.y, u.x)
        let localDirection = SIMD2(
            dot(pushDirection, u), dot(pushDirection, v))
        func inside(_ point: SIMD2<Float>) -> Bool {
            let bar = abs(point.x) <= 0.10
                && point.y >= 0 && point.y <= 0.05
            let stem = abs(point.x) <= 0.025
                && point.y >= -0.15 && point.y <= 0
            return bar || stem
        }
        var insideDistance: Float = 0
        var outsideDistance: Float = 0.002
        while outsideDistance <= 0.25,
              inside(localAnchor - localDirection * outsideDistance) {
            insideDistance = outsideDistance
            outsideDistance += 0.002
        }
        for _ in 0..<12 {
            let middle = (insideDistance + outsideDistance) * 0.5
            if inside(localAnchor - localDirection * middle) {
                insideDistance = middle
            } else {
                outsideDistance = middle
            }
        }
        let localSurface = localAnchor - localDirection * insideDistance
        return block + u * localSurface.x + v * localSurface.y
    }

    private static func cross(
        _ lhs: SIMD2<Float>, _ rhs: SIMD2<Float>
    ) -> Float {
        lhs.x * rhs.y - lhs.y * rhs.x
    }

}

public final class ArmPushTTask: VectorizedRLTask, RLEvaluationCriteriaProviding,
    ObservationSchemaTransferProviding, PolicyExpertGateProviding
{
    public let spec: RLTaskSpec
    public let environment: ArmPushTEnv
    public let configuration: ArmPushTTaskConfig
    public let evaluationCriteria = RLEvaluationCriteria(
        minimumSuccessRate: 0.80,
        minimumTaskMetrics: ["episode/normalized_score": 0.80],
        maximumTaskMetrics: [
            "episode/goal_distance_m": 0.10,
            "episode/yaw_error_rad": 0.50,
        ])

    public var usesPolicyExpertGate: Bool {
        configuration.precisionGatedActor
    }

    public var freezesBasePolicyExpert: Bool {
        configuration.precisionGatedActor
            && configuration.freezeBasePolicyExpert
    }

    public var initializesPolicyExpertFromBaseOnTransfer: Bool {
        configuration.precisionGatedActor
    }

    public func policyExpertGates(
        _ observations: ContiguousArray<Float>
    ) -> ContiguousArray<Float> {
        let dimension = spec.observation.elementCount
        precondition(observations.count.isMultiple(of: dimension))
        let rows = observations.count / dimension
        var gates = ContiguousArray(repeating: Float(0), count: rows)
        guard configuration.precisionGatedActor else { return gates }
        for row in 0..<rows {
            let o = row * dimension
            let block = SIMD2<Float>(
                observations[o + 4] * 0.65,
                observations[o + 5] * 0.65)
            let goal = block + SIMD2<Float>(
                observations[o + 11] * 1.3,
                observations[o + 12] * 1.3)
            let yaw = atan2(observations[o + 8], observations[o + 9])
            let coverage = ArmPushTEnv.coverage(
                blockPosition: block, blockYaw: yaw,
                goalPosition: goal, goalYaw: 0)
            if precisionExpertActive[row] {
                if coverage < configuration.precisionExpertReleaseCoverage {
                    precisionExpertActive[row] = false
                }
            } else if coverage >= configuration.precisionExpertGateCoverage {
                precisionExpertActive[row] = true
            }
            gates[row] = precisionExpertActive[row] ? 1 : 0
        }
        return gates
    }

    private var previousActions: ContiguousArray<Float>
    private var previousTipPositions: [SIMD2<Float>]
    private var previousAngles: [[Float]]
    private var jointVelocities: [[Float]]
    private var previousGoalDistance: [Float]
    private var initialGoalDistance: [Float]
    private var initialYawError: [Float]
    private var initialBlockOffset: [SIMD2<Float>]
    private var previousReachDistance: [Float]
    private var previousPushContactDistance: [Float]
    private var previousYawError: [Float]
    private var previousCoverage: [Float]
    private var maximumNormalizedScores: [Float]
    private var episodeSucceeded: [Bool]
    private var precisionExpertActive: [Bool]
    private var episodeLengths: [Int]
    private var episodeReturns: [Float]
    private var resetRNG: SplitMix64

    public init(configuration: ArmPushTTaskConfig) throws {
        guard configuration.numEnvironments > 0,
              configuration.maxEpisodeSteps > 0,
              configuration.controlDecimation > 0,
              configuration.jointDeltaActionScale > 0,
              configuration.endEffectorDeltaActionScale >= 0,
              configuration.linkMass > 0,
              configuration.tipMass > 0,
              configuration.motorTorque > 0,
              configuration.motorStiffness > 0,
              configuration.motorDamping >= 0,
              configuration.motorArmature >= 0,
              configuration.blockMass > 0,
              configuration.blockStaticFriction >= 0,
              configuration.blockDynamicFriction >= 0,
              (0...1).contains(configuration.blockSpawnGoalBlend),
              configuration.blockSpawnRadius >= 0,
              (0...Float.pi).contains(configuration.blockSpawnYawRange),
              (-1...1).contains(configuration.blockSpawnLateralBias),
              configuration.linkLength1 > 0,
              configuration.linkLength2 > 0,
              configuration.goalProgressWeight >= 0,
              configuration.reachProgressWeight >= 0,
              configuration.yawProgressWeight >= 0,
              configuration.coverageProgressWeight >= 0,
              configuration.coverageRewardWeight >= 0,
              configuration.poseProgressRewardWeight >= 0,
              configuration.poseRewardWeight >= 0,
              configuration.poseRewardDistanceScale > 0,
              configuration.reachingRewardWeight >= 0,
              configuration.actionMagnitudePenaltyWeight >= 0,
              configuration.actionRatePenaltyWeight >= 0,
              configuration.precisionExpertGateCoverage > 0,
              configuration.precisionExpertGateCoverage <= 1,
              configuration.precisionExpertReleaseCoverage >= 0,
              configuration.precisionExpertReleaseCoverage
                <= configuration.precisionExpertGateCoverage,
              (!configuration.freezeBasePolicyExpert
                || configuration.precisionGatedActor),
              configuration.pushContactProgressWeight >= 0,
              configuration.reachDistancePenaltyWeight >= 0,
              configuration.goalDistancePenaltyWeight >= 0,
              configuration.yawErrorPenaltyWeight >= 0,
              configuration.pushContactDistancePenaltyWeight >= 0,
              configuration.successBonus >= 0,
              configuration.successRewardOverride >= 0,
              configuration.successCoverage > 0,
              configuration.successCoverage <= ArmPushTEnv.successCoverage,
              configuration.successYawTolerance > 0,
              configuration.successYawTolerance <= .pi,
              configuration.reachCurriculumSuccessDistance >= 0,
              configuration.pushContactCurriculumSuccessDistance >= 0,
              configuration.pushContactOffset > 0,
              configuration.pushContactCurriculumMaximumGoalRegression >= 0,
              configuration.pushContactCurriculumMinimumGoalProgress >= 0,
              configuration.pushContactCurriculumMaximumGoalDistance >= 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "arm task dimensions must be positive and reward weights nonnegative")
        }
        let env = try ArmPushTEnv(
            numEnvironments: configuration.numEnvironments,
            seed: configuration.seed,
            blockSpawnGoalBlend: configuration.blockSpawnGoalBlend,
            blockSpawnRadius: configuration.blockSpawnRadius,
            blockSpawnYawRange: configuration.blockSpawnYawRange,
            blockSpawnLateralBias: configuration.blockSpawnLateralBias,
            linkLength1: configuration.linkLength1,
            linkLength2: configuration.linkLength2,
            linkMass: configuration.linkMass,
            tipMass: configuration.tipMass,
            motorTorque: configuration.motorTorque,
            motorStiffness: configuration.motorStiffness,
            motorDamping: configuration.motorDamping,
            motorArmature: configuration.motorArmature,
            blockMass: configuration.blockMass,
            blockStaticFriction: configuration.blockStaticFriction,
            blockDynamicFriction: configuration.blockDynamicFriction)
        environment = env
        self.configuration = configuration
        var configurationValues: [String: Float] = [
            "jointDeltaActionScale": configuration.jointDeltaActionScale,
            "linkLength1": configuration.linkLength1,
            "linkLength2": configuration.linkLength2,
            "linkMass": configuration.linkMass,
            "tipMass": configuration.tipMass,
            "motorTorque": configuration.motorTorque,
            "motorStiffness": configuration.motorStiffness,
            "motorDamping": configuration.motorDamping,
            "motorArmature": configuration.motorArmature,
            "blockMass": configuration.blockMass,
            "blockStaticFriction": configuration.blockStaticFriction,
            "blockDynamicFriction": configuration.blockDynamicFriction,
            "blockSpawnGoalBlend": configuration.blockSpawnGoalBlend,
            "blockSpawnRadius": configuration.blockSpawnRadius,
            "blockSpawnYawRange": configuration.blockSpawnYawRange,
            "goalProgressWeight": configuration.goalProgressWeight,
            "reachProgressWeight": configuration.reachProgressWeight,
            "yawProgressWeight": configuration.yawProgressWeight,
            "coverageProgressWeight": configuration.coverageProgressWeight,
            "coverageRewardWeight": configuration.coverageRewardWeight,
            "poseRewardWeight": configuration.poseRewardWeight,
            "poseRewardDistanceScale": configuration.poseRewardDistanceScale,
            "reachingRewardWeight": configuration.reachingRewardWeight,
            "actionRatePenaltyWeight": configuration.actionRatePenaltyWeight,
            "pushContactProgressWeight": configuration.pushContactProgressWeight,
            "reachDistancePenaltyWeight": configuration.reachDistancePenaltyWeight,
            "goalDistancePenaltyWeight": configuration.goalDistancePenaltyWeight,
            "yawErrorPenaltyWeight": configuration.yawErrorPenaltyWeight,
            "pushContactDistancePenaltyWeight":
                configuration.pushContactDistancePenaltyWeight,
            "successBonus": configuration.successBonus,
            "successRewardOverride": configuration.successRewardOverride,
            "successCoverage": configuration.successCoverage,
            "successYawTolerance": configuration.successYawTolerance,
            "reachCurriculumSuccessDistance":
                configuration.reachCurriculumSuccessDistance,
            "pushContactCurriculumSuccessDistance":
                configuration.pushContactCurriculumSuccessDistance,
            "pushContactOffset": configuration.pushContactOffset,
            "pushContactCurriculumMaximumGoalRegression":
                configuration.pushContactCurriculumMaximumGoalRegression,
            "pushContactCurriculumMinimumGoalProgress":
                configuration.pushContactCurriculumMinimumGoalProgress,
        ]
        if configuration.pushContactCurriculumMaximumGoalDistance > 0 {
            configurationValues["pushContactCurriculumMaximumGoalDistance"] =
                configuration.pushContactCurriculumMaximumGoalDistance
        }
        if configuration.blockSpawnLateralBias != 0 {
            configurationValues["blockSpawnLateralBias"] =
                configuration.blockSpawnLateralBias
        }
        if configuration.endEffectorDeltaActionScale > 0 {
            configurationValues["endEffectorDeltaActionScale"] =
                configuration.endEffectorDeltaActionScale
        }
        if configuration.continueAfterSuccess {
            configurationValues["continueAfterSuccess"] = 1
        }
        if configuration.poseProgressRewardWeight > 0 {
            configurationValues["poseProgressRewardWeight"] =
                configuration.poseProgressRewardWeight
        }
        if configuration.actionMagnitudePenaltyWeight > 0 {
            configurationValues["actionMagnitudePenaltyWeight"] =
                configuration.actionMagnitudePenaltyWeight
        }
        if configuration.precisionGatedActor {
            configurationValues["precisionGatedActor"] = 1
            configurationValues["precisionExpertGateCoverage"] =
                configuration.precisionExpertGateCoverage
            if configuration.precisionExpertReleaseCoverage
                < configuration.precisionExpertGateCoverage {
                configurationValues["precisionExpertReleaseCoverage"] =
                    configuration.precisionExpertReleaseCoverage
            }
            if configuration.freezeBasePolicyExpert {
                configurationValues["freezeBasePolicyExpert"] = 1
            }
        }
        // Changing workspace geometry invalidates both policy observations and
        // joint-to-contact behavior. Offset the whole feature-revision family
        // rather than pretending a revision-14/19 checkpoint is compatible.
        let featureRevision = configuration.blockSpawnLateralBias != 0
            ? 22
            : (configuration.precisionGatedActor
                && configuration.precisionExpertReleaseCoverage
                    < configuration.precisionExpertGateCoverage
            ? 21
            : (configuration.precisionGatedActor
            ? 20
            : (configuration.actionMagnitudePenaltyWeight > 0
            ? 19
            : (configuration.poseProgressRewardWeight > 0
            ? 18
            : (configuration.continueAfterSuccess
            ? 17
            : (configuration.endEffectorDeltaActionScale > 0
            ? 16
            : (configuration.pushContactCurriculumMaximumGoalDistance > 0
                ? 15 : 14)))))))
        let taskRevision = featureRevision + 100
        spec = RLTaskSpec(
            id: "arm-pusht-v0",
            revision: RLPhysicsContract.deterministicColorSolveV1(taskRevision),
            numEnvironments: configuration.numEnvironments,
            observation: RLTensorSpec(name: "policy", shape: [19]),
            action: RLTensorSpec(
                name: configuration.endEffectorDeltaActionScale > 0
                    ? "end_effector_delta_position" : "joint_delta_position",
                shape: [2], lowerBound: [-1, -1], upperBound: [1, 1]),
            maxEpisodeSteps: configuration.maxEpisodeSteps,
            simulationStep: env.scene.settings.dt,
            controlDecimation: configuration.controlDecimation,
            autoReset: configuration.autoReset,
            configurationValues: configurationValues)
        let n = configuration.numEnvironments
        previousActions = ContiguousArray(repeating: 0, count: n * 2)
        previousTipPositions = [SIMD2<Float>](repeating: .zero, count: n)
        previousAngles = [[Float]](repeating: [0, 0], count: n)
        jointVelocities = previousAngles
        previousGoalDistance = [Float](repeating: 0, count: n)
        initialGoalDistance = [Float](repeating: 0, count: n)
        initialYawError = [Float](repeating: 0, count: n)
        initialBlockOffset = [SIMD2<Float>](repeating: .zero, count: n)
        previousReachDistance = [Float](repeating: 0, count: n)
        previousPushContactDistance = [Float](repeating: 0, count: n)
        previousYawError = [Float](repeating: 0, count: n)
        previousCoverage = [Float](repeating: 0, count: n)
        maximumNormalizedScores = [Float](repeating: 0, count: n)
        episodeSucceeded = [Bool](repeating: false, count: n)
        precisionExpertActive = [Bool](repeating: false, count: n)
        episodeLengths = [Int](repeating: 0, count: n)
        episodeReturns = [Float](repeating: 0, count: n)
        resetRNG = SplitMix64(seed: configuration.seed &+ 0x589965CC75374CC3)
        initializeEpisodes(Array(0..<n), states: env.states())
    }

    public func reset(environments ids: [Int]?, seed: UInt64,
                      into observations: inout RLObservationBatch) throws {
        try observations.validate(for: spec)
        let envIDs = try checkedEnvironmentIDs(ids)
        try environment.solver.synchronize()
        let seeds = envIDs.map { seed &+ UInt64($0) &* 0x9E3779B97F4A7C15 }
        environment.reset(envIDs, seeds: seeds)
        let states = environment.states()
        initializeEpisodes(envIDs, states: states)
        fillObservations(states, into: &observations.policy)
    }

    public func step(actions: RLActionBatch, into result: inout RLStepBatch) throws {
        try result.validate(for: spec)
        try actions.validate(for: spec)
        try environment.solver.synchronize()
        result.clearSignals()
        var applied = ContiguousArray(repeating: Float.zero,
                                      count: actions.values.count)
        var actionRate = ContiguousArray(repeating: Float(0),
                                         count: spec.numEnvironments)
        var actionMagnitude = ContiguousArray(repeating: Float(0),
                                              count: spec.numEnvironments)
        for e in 0..<spec.numEnvironments {
            let deltaAction = SIMD2(
                simd_clamp(actions.values[e * 2], -1, 1),
                simd_clamp(actions.values[e * 2 + 1], -1, 1))
            let target: SIMD2<Float>
            if configuration.endEffectorDeltaActionScale > 0 {
                let tipTarget = Self.endEffectorDeltaTarget(
                    currentPosition: previousTipPositions[e],
                    deltaActions: deltaAction,
                    deltaScale: configuration.endEffectorDeltaActionScale)
                target = environment.normalizedJointTargets(
                    tipTarget: tipTarget)
            } else {
                target = ArmPushTEnv.normalizedJointTargets(
                    currentAngles:
                        SIMD2(previousAngles[e][0], previousAngles[e][1]),
                    deltaActions: deltaAction,
                    deltaScale: configuration.jointDeltaActionScale)
            }
            for j in 0..<2 {
                let i = e * 2 + j
                let requested = deltaAction[j]
                applied[i] = target[j]
                actionMagnitude[e] += requested * requested
                let delta = requested - previousActions[i]
                actionRate[e] += delta * delta
                previousActions[i] = requested
            }
        }
        try environment.stepChecked(
            normalizedActions: applied,
            decimation: configuration.controlDecimation)
        var states = environment.states()
        updateJointVelocities(states)
        fillObservations(states, into: &result.observations.policy)

        let n = spec.numEnvironments
        var goalProgress = ContiguousArray(repeating: Float(0), count: n)
        var reachProgress = ContiguousArray(repeating: Float(0), count: n)
        var pushContactProgress = ContiguousArray(repeating: Float(0), count: n)
        var yawProgress = ContiguousArray(repeating: Float(0), count: n)
        var coverageProgress = ContiguousArray(repeating: Float(0), count: n)
        var coverageReward = ContiguousArray(repeating: Float(0), count: n)
        var poseProgressReward = ContiguousArray(repeating: Float(0), count: n)
        var poseReward = ContiguousArray(repeating: Float(0), count: n)
        var reachingReward = ContiguousArray(repeating: Float(0), count: n)
        var normalizedCoverage = ContiguousArray(repeating: Float(0), count: n)
        var reachDistancePenalty = ContiguousArray(repeating: Float(0), count: n)
        var goalDistancePenalty = ContiguousArray(repeating: Float(0), count: n)
        var yawErrorPenalty = ContiguousArray(repeating: Float(0), count: n)
        var pushContactDistancePenalty = ContiguousArray(
            repeating: Float(0), count: n)
        var actionMagnitudePenalty = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeReturnMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeLengthMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeGoalDistance = ContiguousArray(repeating: Float(0), count: n)
        var episodeReachDistance = ContiguousArray(repeating: Float(0), count: n)
        var episodePushContactDistance = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeYawError = ContiguousArray(repeating: Float(0), count: n)
        var episodeNormalizedScore = ContiguousArray(repeating: Float(0), count: n)
        var episodeSuccessOnce = ContiguousArray(repeating: Float(0), count: n)
        var episodeSuccessAtEnd = ContiguousArray(repeating: Float(0), count: n)
        var yawPositiveBin = ContiguousArray(repeating: Float(0), count: n)
        var yawPositiveSuccess = yawPositiveBin
        var yawNegativeBin = yawPositiveBin
        var yawNegativeSuccess = yawPositiveBin
        var yawLargeBin = yawPositiveBin
        var yawLargeSuccess = yawPositiveBin
        var yawSmallBin = yawPositiveBin
        var yawSmallSuccess = yawPositiveBin
        var spawnLeftBin = yawPositiveBin
        var spawnLeftSuccess = yawPositiveBin
        var spawnRightBin = yawPositiveBin
        var spawnRightSuccess = yawPositiveBin
        var resetIDs = [Int](), resetSeeds = [UInt64]()
        for e in 0..<n {
            let goalDistance = length(environment.refs[e].goalPosition
                                      - states[e].blockPosition)
            let reachDistance = length(states[e].tipPosition - states[e].blockPosition)
            let goalVector = environment.refs[e].goalPosition
                - states[e].blockPosition
            let goalDirection = goalVector / max(length(goalVector), 1e-6)
            let pushContactTarget = states[e].blockPosition
                - goalDirection * configuration.pushContactOffset
            let pushContactDistance = length(
                states[e].tipPosition - pushContactTarget)
            var yawError = states[e].blockYaw - environment.refs[e].goalYaw
            yawError -= 2 * .pi * (yawError / (2 * .pi)).rounded()
            yawError = abs(yawError)
            let coverage = environment.coverage(e, state: states[e])
            let score = min(coverage / ArmPushTEnv.successCoverage, 1)
            goalProgress[e] = previousGoalDistance[e] - goalDistance
            reachProgress[e] = previousReachDistance[e] - reachDistance
            pushContactProgress[e] = previousPushContactDistance[e]
                - pushContactDistance
            yawProgress[e] = previousYawError[e] - yawError
            coverageProgress[e] = coverage - previousCoverage[e]
            normalizedCoverage[e] = score
            coverageReward[e] = configuration.coverageRewardWeight * score
            let posePotential = Self.precisionPoseReward(
                goalDistance: goalDistance, yawError: yawError,
                distanceScale: configuration.poseRewardDistanceScale)
            let previousPosePotential = Self.precisionPoseReward(
                goalDistance: previousGoalDistance[e],
                yawError: previousYawError[e],
                distanceScale: configuration.poseRewardDistanceScale)
            poseProgressReward[e] = configuration.poseProgressRewardWeight
                * (posePotential - previousPosePotential)
            poseReward[e] = configuration.poseRewardWeight * posePotential
            reachingReward[e] = configuration.reachingRewardWeight
                * Self.reachingKernel(
                    distance: reachDistance,
                    distanceScale: configuration.poseRewardDistanceScale)
            reachDistancePenalty[e] = -configuration.reachDistancePenaltyWeight
                * reachDistance
            goalDistancePenalty[e] = -configuration.goalDistancePenaltyWeight
                * goalDistance
            yawErrorPenalty[e] = -configuration.yawErrorPenaltyWeight * yawError
            pushContactDistancePenalty[e] =
                -configuration.pushContactDistancePenaltyWeight
                * pushContactDistance
            actionMagnitudePenalty[e] =
                -configuration.actionMagnitudePenaltyWeight
                * actionMagnitude[e]
            maximumNormalizedScores[e] = max(maximumNormalizedScores[e], score)
            episodeLengths[e] += 1
            let success: Bool
            if configuration.pushContactCurriculumSuccessDistance > 0 {
                success = Self.pushContactCurriculumSucceeded(
                    pushContactDistance: pushContactDistance,
                    successDistance:
                        configuration.pushContactCurriculumSuccessDistance,
                    goalDistance: goalDistance,
                    initialGoalDistance: initialGoalDistance[e],
                    maximumGoalRegression: configuration
                        .pushContactCurriculumMaximumGoalRegression,
                    minimumGoalProgress: configuration
                        .pushContactCurriculumMinimumGoalProgress,
                    maximumGoalDistance: configuration
                        .pushContactCurriculumMaximumGoalDistance,
                    yawError: yawError,
                    maximumYawError: configuration.successYawTolerance)
            } else if configuration.reachCurriculumSuccessDistance > 0 {
                success = reachDistance
                    < configuration.reachCurriculumSuccessDistance
            } else {
                success = coverage > configuration.successCoverage
                    && yawError < configuration.successYawTolerance
            }
            let timeout = episodeLengths[e] >= configuration.maxEpisodeSteps
            episodeSucceeded[e] = episodeSucceeded[e] || success
            var reward = configuration.goalProgressWeight * goalProgress[e]
                + configuration.reachProgressWeight * reachProgress[e]
                + configuration.pushContactProgressWeight
                    * pushContactProgress[e]
                + configuration.yawProgressWeight * yawProgress[e]
                + configuration.coverageProgressWeight * coverageProgress[e]
                + coverageReward[e]
                + poseProgressReward[e]
                + poseReward[e]
                + reachingReward[e]
                + reachDistancePenalty[e]
                + goalDistancePenalty[e]
                + yawErrorPenalty[e]
                + pushContactDistancePenalty[e]
                + actionMagnitudePenalty[e]
                - configuration.actionRatePenaltyWeight * actionRate[e]
            if success {
                reward = configuration.successRewardOverride > 0
                    ? configuration.successRewardOverride
                    : reward + configuration.successBonus
            }
            result.rewards[e] = reward
            let done: Bool
            if configuration.continueAfterSuccess {
                result.terminated[e] = false
                result.truncated[e] = timeout
                result.successes[e] = success
                done = timeout
            } else {
                result.terminated[e] = success
                result.truncated[e] = !success && timeout
                result.successes[e] = success
                done = success || timeout
            }
            episodeReturns[e] += reward
            previousGoalDistance[e] = goalDistance
            previousReachDistance[e] = reachDistance
            previousPushContactDistance[e] = pushContactDistance
            previousYawError[e] = yawError
            previousCoverage[e] = coverage
            if done {
                result.hasFinalObservation[e] = true
                let row = e * spec.observation.elementCount
                for j in 0..<spec.observation.elementCount {
                    result.finalObservations[row + j] = result.observations.policy[row + j]
                }
                episodeReturnMetric[e] = episodeReturns[e]
                episodeLengthMetric[e] = Float(episodeLengths[e])
                episodeGoalDistance[e] = goalDistance
                episodeReachDistance[e] = reachDistance
                episodePushContactDistance[e] = pushContactDistance
                episodeYawError[e] = yawError
                episodeSuccessOnce[e] = episodeSucceeded[e] ? 1 : 0
                episodeSuccessAtEnd[e] = success ? 1 : 0
                let succeeded = episodeSucceeded[e] ? Float(1) : 0
                if initialYawError[e] >= 0 {
                    yawPositiveBin[e] = 1
                    yawPositiveSuccess[e] = succeeded
                } else {
                    yawNegativeBin[e] = 1
                    yawNegativeSuccess[e] = succeeded
                }
                if abs(initialYawError[e])
                    >= configuration.blockSpawnYawRange * 0.5 {
                    yawLargeBin[e] = 1
                    yawLargeSuccess[e] = succeeded
                } else {
                    yawSmallBin[e] = 1
                    yawSmallSuccess[e] = succeeded
                }
                let nominalPush = simd_normalize(
                    environment.refs[e].goalPosition)
                let lateral = SIMD2(-nominalPush.y, nominalPush.x)
                if dot(initialBlockOffset[e], lateral) >= 0 {
                    spawnLeftBin[e] = 1
                    spawnLeftSuccess[e] = succeeded
                } else {
                    spawnRightBin[e] = 1
                    spawnRightSuccess[e] = succeeded
                }
                // Canonical Push-T reports the best coverage reward achieved
                // during the episode, not a terminal pose-distance proxy.
                episodeNormalizedScore[e] = maximumNormalizedScores[e]
                if configuration.autoReset {
                    resetIDs.append(e)
                    resetSeeds.append(resetRNG.next())
                }
            }
        }
        result.metrics["reward/goal_progress"] = goalProgress
        result.metrics["reward/reach_progress"] = reachProgress
        result.metrics["reward/push_contact_progress"] = pushContactProgress
        result.metrics["reward/yaw_progress"] = yawProgress
        result.metrics["reward/coverage_progress"] = coverageProgress
        result.metrics["reward/coverage"] = coverageReward
        result.metrics["reward/pose_progress"] = poseProgressReward
        result.metrics["reward/pose_precision"] = poseReward
        result.metrics["reward/reaching"] = reachingReward
        result.metrics["task/normalized_coverage"] = normalizedCoverage
        result.metrics["penalty/reach_distance"] = reachDistancePenalty
        result.metrics["penalty/goal_distance"] = goalDistancePenalty
        result.metrics["penalty/yaw_error"] = yawErrorPenalty
        result.metrics["penalty/push_contact_distance"] =
            pushContactDistancePenalty
        result.metrics["penalty/action_magnitude"] = actionMagnitudePenalty
        result.metrics["penalty/action_rate"] = actionRate
        result.metrics["episode/return"] = episodeReturnMetric
        result.metrics["episode/length"] = episodeLengthMetric
        result.metrics["episode/goal_distance_m"] = episodeGoalDistance
        result.metrics["episode/reach_distance_m"] = episodeReachDistance
        result.metrics["episode/push_contact_distance_m"] =
            episodePushContactDistance
        result.metrics["episode/yaw_error_rad"] = episodeYawError
        result.metrics["episode/normalized_score"] = episodeNormalizedScore
        result.metrics["episode/success_once"] = episodeSuccessOnce
        result.metrics["episode/success_at_end"] = episodeSuccessAtEnd
        result.metrics["episode/yaw_positive_bin"] = yawPositiveBin
        result.metrics["episode/yaw_positive_success"] = yawPositiveSuccess
        result.metrics["episode/yaw_negative_bin"] = yawNegativeBin
        result.metrics["episode/yaw_negative_success"] = yawNegativeSuccess
        result.metrics["episode/yaw_large_bin"] = yawLargeBin
        result.metrics["episode/yaw_large_success"] = yawLargeSuccess
        result.metrics["episode/yaw_small_bin"] = yawSmallBin
        result.metrics["episode/yaw_small_success"] = yawSmallSuccess
        result.metrics["episode/spawn_left_bin"] = spawnLeftBin
        result.metrics["episode/spawn_left_success"] = spawnLeftSuccess
        result.metrics["episode/spawn_right_bin"] = spawnRightBin
        result.metrics["episode/spawn_right_success"] = spawnRightSuccess
        if !resetIDs.isEmpty {
            environment.reset(resetIDs, seeds: resetSeeds)
            states = environment.states()
            initializeEpisodes(resetIDs, states: states)
            fillObservations(states, into: &result.observations.policy)
        }
    }

    private func initializeEpisodes(_ ids: [Int], states: [ArmPushTState]) {
        for e in ids {
            for j in 0..<2 {
                previousActions[e * 2 + j] = 0
                jointVelocities[e][j] = 0
            }
            previousAngles[e] = states[e].jointAngles
            previousTipPositions[e] = states[e].tipPosition
            previousGoalDistance[e] = length(environment.refs[e].goalPosition
                                              - states[e].blockPosition)
            initialGoalDistance[e] = previousGoalDistance[e]
            let spawnAnchor = environment.refs[e].goalPosition
                * configuration.blockSpawnGoalBlend
            initialBlockOffset[e] = states[e].blockPosition - spawnAnchor
            previousReachDistance[e] = length(states[e].tipPosition
                                               - states[e].blockPosition)
            let goalVector = environment.refs[e].goalPosition
                - states[e].blockPosition
            let goalDirection = goalVector / max(length(goalVector), 1e-6)
            let pushContactTarget = states[e].blockPosition
                - goalDirection * configuration.pushContactOffset
            previousPushContactDistance[e] = length(
                states[e].tipPosition - pushContactTarget)
            var yawError = states[e].blockYaw - environment.refs[e].goalYaw
            yawError -= 2 * .pi * (yawError / (2 * .pi)).rounded()
            initialYawError[e] = yawError
            previousYawError[e] = abs(yawError)
            previousCoverage[e] = environment.coverage(e, state: states[e])
            maximumNormalizedScores[e] = min(
                previousCoverage[e] / ArmPushTEnv.successCoverage, 1)
            episodeSucceeded[e] = false
            precisionExpertActive[e] = false
            episodeLengths[e] = 0
            episodeReturns[e] = 0
        }
    }

    private func updateJointVelocities(_ states: [ArmPushTState]) {
        for e in 0..<spec.numEnvironments {
            for j in 0..<2 {
                var delta = states[e].jointAngles[j] - previousAngles[e][j]
                delta -= 2 * .pi * (delta / (2 * .pi)).rounded()
                jointVelocities[e][j] = delta / spec.controlStep
            }
            previousAngles[e] = states[e].jointAngles
            previousTipPositions[e] = states[e].tipPosition
        }
    }

    private func fillObservations(_ states: [ArmPushTState],
                                  into output: inout ContiguousArray<Float>) {
        let d = spec.observation.elementCount
        for e in 0..<spec.numEnvironments {
            let state = states[e]
            let goalDelta = environment.refs[e].goalPosition - state.blockPosition
            let o = e * d
            output[o] = state.tipPosition.x / 0.65
            output[o + 1] = state.tipPosition.y / 0.65
            output[o + 2] = state.tipVelocity.x
            output[o + 3] = state.tipVelocity.y
            output[o + 4] = state.blockPosition.x / 0.65
            output[o + 5] = state.blockPosition.y / 0.65
            // This benchmark moves a massive articulated object through
            // contact dynamics. Pose alone aliases states that need opposite
            // braking actions, so expose the object's full planar velocity.
            output[o + 6] = state.blockVelocity.x
            output[o + 7] = state.blockVelocity.y
            output[o + 8] = sin(state.blockYaw)
            output[o + 9] = cos(state.blockYaw)
            output[o + 10] = state.blockAngularVelocity * 0.1
            output[o + 11] = goalDelta.x / 1.3
            output[o + 12] = goalDelta.y / 1.3
            for j in 0..<2 {
                output[o + 13 + j] = (state.jointAngles[j]
                    - ArmPushTEnv.defaultJointPositions[j]) / ArmPushTEnv.actionScales[j]
                output[o + 15 + j] = jointVelocities[e][j] * 0.1
                output[o + 17 + j] = previousActions[e * 2 + j]
            }
        }
    }

    /// ManiSkill's maintained articulated Push-T task reports that directly
    /// rewarding overlap traps PPO around 50-75% coverage. Its replacement is
    /// a smooth, bounded pose reward with separate rotation and translation
    /// terms. The maintained reference uses tanh(5 * distance) for its
    /// 20-centimetre T, so distanceScale defaults to five in the task config.
    public static func precisionPoseReward(
        goalDistance: Float, yawError: Float, distanceScale: Float = 1
    ) -> Float {
        precondition(goalDistance >= 0 && distanceScale > 0)
        let alignment = 0.5 * (cos(yawError) + 1)
        let rotationReward = 0.5 * alignment * alignment
        let translation = 1 - tanh(distanceScale * goalDistance)
        return rotationReward + 0.5 * translation * translation
    }

    public static func reachingKernel(
        distance: Float, distanceScale: Float = 1
    ) -> Float {
        precondition(distance >= 0 && distanceScale > 0)
        return sqrt(max(1 - tanh(distanceScale * distance), 0))
    }

    /// Pure controller-space conversion used by both the batched task and
    /// tests. Clamping here guarantees that a policy cannot bypass the action
    /// contract before the target is resolved through the physical arm IK.
    public static func endEffectorDeltaTarget(
        currentPosition: SIMD2<Float>, deltaActions: SIMD2<Float>,
        deltaScale: Float
    ) -> SIMD2<Float> {
        precondition(deltaScale > 0)
        return currentPosition + simd_clamp(
            deltaActions, SIMD2(repeating: -1), SIMD2(repeating: 1))
            * deltaScale
    }

    public static func pushContactCurriculumSucceeded(
        pushContactDistance: Float, successDistance: Float,
        goalDistance: Float, initialGoalDistance: Float,
        maximumGoalRegression: Float, minimumGoalProgress: Float = 0,
        maximumGoalDistance: Float = 0,
        yawError: Float = 0, maximumYawError: Float = .pi
    ) -> Bool {
        precondition(pushContactDistance >= 0 && successDistance > 0)
        precondition(goalDistance >= 0 && initialGoalDistance >= 0)
        precondition(maximumGoalRegression >= 0)
        precondition(minimumGoalProgress >= 0)
        precondition(maximumGoalDistance >= 0)
        precondition(yawError >= 0 && maximumYawError > 0
            && maximumYawError <= .pi)
        let goalSucceeded = maximumGoalDistance > 0
            ? goalDistance < maximumGoalDistance
                && goalDistance <= initialGoalDistance + maximumGoalRegression
            : goalDistance <= initialGoalDistance - minimumGoalProgress
                + maximumGoalRegression
        return pushContactDistance < successDistance
            && goalSucceeded
            && yawError < maximumYawError
    }

    /// Revision 4 added measured object linear and angular velocity to the
    /// original 16-channel reference-free state. Explicitly remap the old
    /// actor's measured channels and initialize only the three new velocity
    /// columns to zero. This permits a canonical-coverage policy transfer
    /// without reintroducing the rejected controller-derived observations.
    public func initializationObservationSourceIndices(
        sourceDimension: Int
    ) -> [Int?]? {
        guard sourceDimension == 16,
              spec.observation.elementCount == 19 else { return nil }
        return [
            0, 1, 2, 3, 4, 5,
            nil, nil,
            6, 7,
            nil,
            8, 9, 10, 11, 12, 13, 14, 15,
        ]
    }
}
