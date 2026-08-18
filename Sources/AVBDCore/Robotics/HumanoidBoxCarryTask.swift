import Foundation
import simd

/// Whole-body Unitree H1 pickup-and-carry task.  Every success condition is
/// derived from simulated body state or contact manifolds: there are no
/// animation phases, welded grasps, object teleports, or reference motions.
public struct HumanoidBoxCarryTaskConfig: Sendable {
    public var numEnvironments: Int
    public var seed: UInt64
    public var maxEpisodeSteps: Int
    public var solverIterations: Int
    public var autoReset: Bool
    public var observationNoise: Bool
    public var minimumTrainingStationDistance: Float
    public var evaluationStationDistance: Float
    public var stationDistanceCurriculumControlSteps: Int
    public var boxMass: Float
    public var boxFriction: Float
    public var minimumTrainingLiftClearance: Float
    public var liftClearance: Float
    public var liftClearanceCurriculumControlSteps: Int
    public var minimumTrainingCarryDistance: Float
    public var carryDistance: Float
    public var carryDistanceCurriculumControlSteps: Int
    /// Training-only rotation of the loaded navigation waypoint from straight
    /// ahead to the final left-side receiving table. The actual source and
    /// destination tables never move, preserving pickup contact geometry.
    public var destinationBearingCurriculumControlSteps: Int
    public var minimumTrainingSuccessDwellSteps: Int
    public var successDwellSteps: Int
    public var successDwellCurriculumControlSteps: Int
    public var pregraspForwardOffset: Float
    public var pregraspLateralOffset: Float
    public var approachCommandSpeed: Float
    public var carryCommandSpeed: Float
    /// Use the imported H1 velocity policy's native planar x/y command while
    /// carrying. This lets a loaded robot translate toward a side table
    /// without first sweeping the held object through the source table during
    /// an in-place turn. Joint motion remains entirely policy controlled.
    public var carryHolonomicCommand: Bool
    /// Scale used by the explicit local navigation-goal observation. Keeping
    /// this configurable makes point-goal locomotion checkpoints transferable
    /// without aliasing their goal inputs onto manipulation state.
    public var navigationGoalObservationScale: Float
    public var manipulationHandoffSteps: Int
    public var carryHandoffSteps: Int
    public var carryCommandRampSteps: Int
    public var manipulationArmActionScaleMultiplier: Float
    public var carryBaseLegActionFraction: Float
    public var freezeBasePolicyExpert: Bool
    public var freezeManipulationPolicyExpert: Bool
    /// Freeze the post-lift waist/arm carry branch. Keeping this true protects
    /// a verified grasp during locomotion-only training; placement refinement
    /// can disable it without exposing the pre-lift pickup expert.
    public var freezeCarryPolicyExpert: Bool
    public var initializeManipulationExpertFromBaseOnTransfer: Bool
    public var initializeCarryExpertFromManipulationExpertOnTransfer: Bool
    /// Route frozen manipulation arms and trainable loaded-locomotion legs as
    /// disjoint actuator groups, matching modern locomanipulation stacks.
    public var compositionalCarryController: Bool
    /// After physical lift, hand the ten leg actions from the whole-body
    /// pickup expert to verified locomotion while waist plus both arms hand
    /// off to the carry expert. This preserves the learned pickup crouch and
    /// then applies the intervention-style lower/upper controller split.
    public var upperBodyCarryController: Bool
    /// Route H1's torso joint with the loaded-locomotion branch instead of
    /// freezing it with the arms. The imported gait policy actively uses this
    /// joint for balance; both arm groups remain protected by the carry actor.
    public var carryLocomotionControlsTorso: Bool
    public var initializeCarryExpertFromBaseOnTransfer: Bool
    /// Replace the loaded lower-body branch with an exact copy of the
    /// verified navigation actor when transferring a checkpoint. The branch
    /// remains independently trainable, so PPO learns load compensation
    /// without modifying the source walker or the load-bearing arms.
    public var initializeCarryLocomotionExpertFromBaseOnTransfer: Bool
    /// Fraction of training-only episode resets restored from a previously
    /// observed, physically unsupported lift. This is reference-state
    /// initialization at an episode boundary, not an in-episode teleport.
    public var carryStartReplayProbability: Float
    /// Advance the solver-complete carry reset snapshot on first physical
    /// destination-table contact. This is a training-only reverse curriculum;
    /// evaluation still begins from the authored pickup scene.
    public var advanceReplaySnapshotAtDestinationContact: Bool
    /// Relative reference-policy penalty retained on arm actions after a
    /// physical lift. Pickup remains protected by a separate frozen expert;
    /// lowering this lets the carry expert learn a sustained load-bearing
    /// pose instead of reproducing a finite pickup-and-lower trajectory.
    public var carryArmReferenceWeight: Float
    /// Reward-only clearance target relative to the physical lift-success
    /// gate. Values above one preserve a gradient for raising and holding the
    /// load after it first becomes unsupported.
    public var carryHoldClearanceMultiplier: Float
    /// Weight on the telescoping reduction in box-to-destination distance.
    /// This must be calibrated against the accumulated hold reward so a
    /// robust policy is still incentivized to transport rather than park.
    public var carryProgressRewardWeight: Float
    /// Scale on the ordinary measured command-tracking reward after lift.
    /// Loaded lower-body adaptation otherwise receives a much smaller signal
    /// than the sustained two-hand hold objective, especially while turning
    /// in place produces no immediate destination-distance progress.
    public var carryLocomotionRewardMultiplier: Float
    /// Squared-error temperature for loaded linear and yaw command tracking.
    /// The unloaded H1 task uses 0.25; a smaller loaded value prevents a
    /// stationary holder from receiving almost the same reward as walking.
    public var carryTrackingVariance: Float
    /// Couple loaded linear and yaw tracking multiplicatively. The historical
    /// additive objective pays almost half of its maximum when the robot
    /// stands still under a zero-yaw command, even if it ignores the planar
    /// velocity command completely. Coupling preserves the same maximum at
    /// perfect tracking while requiring both command components to be good.
    public var coupledCarryCommandTracking: Bool
    /// Weight on telescoping pelvis progress toward the loaded navigation
    /// stance. Object progress alone can be produced by reaching farther with
    /// the arms while both feet remain planted; this term makes translation
    /// of the load-bearing robot an explicit, independently measured goal.
    public var carryRootProgressRewardWeight: Float
    /// Event reward for an alternating physical foot exchange while the box
    /// is unsupported and held by both hands.
    public var carryAlternatingStepRewardWeight: Float
    /// Minimum measured alternating foot exchanges required before transport
    /// can satisfy the carry milestone. Zero preserves legacy checkpoints.
    public var minimumLoadedAlternatingSteps: Int

    public init(
        numEnvironments: Int, seed: UInt64 = 1,
        maxEpisodeSteps: Int = 600, solverIterations: Int = 20,
        autoReset: Bool = true, observationNoise: Bool = true,
        minimumTrainingStationDistance: Float = 0.55,
        evaluationStationDistance: Float = 1.40,
        stationDistanceCurriculumControlSteps: Int = 12_000,
        boxMass: Float = 2, boxFriction: Float = 1.2,
        minimumTrainingLiftClearance: Float = 0.001,
        liftClearance: Float = 0.04,
        liftClearanceCurriculumControlSteps: Int = 30_000,
        minimumTrainingCarryDistance: Float = 0.05,
        carryDistance: Float = 0.75,
        carryDistanceCurriculumControlSteps: Int = 30_000,
        destinationBearingCurriculumControlSteps: Int = 0,
        minimumTrainingSuccessDwellSteps: Int = 1,
        successDwellSteps: Int = 10,
        successDwellCurriculumControlSteps: Int = 30_000,
        pregraspForwardOffset: Float = HumanoidBoxCarryTask.pregraspOffset,
        pregraspLateralOffset: Float = 0,
        approachCommandSpeed: Float = 0.40,
        carryCommandSpeed: Float = 0.30,
        carryHolonomicCommand: Bool = false,
        navigationGoalObservationScale: Float = 8,
        manipulationHandoffSteps: Int = 24,
        carryHandoffSteps: Int = 12,
        carryCommandRampSteps: Int = 12,
        manipulationArmActionScaleMultiplier: Float = 2,
        carryBaseLegActionFraction: Float = 0.25,
        freezeBasePolicyExpert: Bool = true,
        freezeManipulationPolicyExpert: Bool = true,
        freezeCarryPolicyExpert: Bool = true,
        initializeManipulationExpertFromBaseOnTransfer: Bool = true,
        initializeCarryExpertFromManipulationExpertOnTransfer: Bool = true,
        compositionalCarryController: Bool = false,
        upperBodyCarryController: Bool = false,
        carryLocomotionControlsTorso: Bool = false,
        initializeCarryExpertFromBaseOnTransfer: Bool = false,
        initializeCarryLocomotionExpertFromBaseOnTransfer: Bool = false,
        carryStartReplayProbability: Float = 0,
        advanceReplaySnapshotAtDestinationContact: Bool = false,
        carryArmReferenceWeight: Float = 1,
        carryHoldClearanceMultiplier: Float = 1,
        carryProgressRewardWeight: Float = 25,
        carryLocomotionRewardMultiplier: Float = 1,
        carryTrackingVariance: Float = 0.25,
        coupledCarryCommandTracking: Bool = false,
        carryRootProgressRewardWeight: Float = 0,
        carryAlternatingStepRewardWeight: Float = 0,
        minimumLoadedAlternatingSteps: Int = 0
    ) {
        self.numEnvironments = numEnvironments
        self.seed = seed
        self.maxEpisodeSteps = maxEpisodeSteps
        self.solverIterations = solverIterations
        self.autoReset = autoReset
        self.observationNoise = observationNoise
        self.minimumTrainingStationDistance = minimumTrainingStationDistance
        self.evaluationStationDistance = evaluationStationDistance
        self.stationDistanceCurriculumControlSteps =
            stationDistanceCurriculumControlSteps
        self.boxMass = boxMass
        self.boxFriction = boxFriction
        self.minimumTrainingLiftClearance = minimumTrainingLiftClearance
        self.liftClearance = liftClearance
        self.liftClearanceCurriculumControlSteps =
            liftClearanceCurriculumControlSteps
        self.minimumTrainingCarryDistance = minimumTrainingCarryDistance
        self.carryDistance = carryDistance
        self.carryDistanceCurriculumControlSteps =
            carryDistanceCurriculumControlSteps
        self.destinationBearingCurriculumControlSteps =
            destinationBearingCurriculumControlSteps
        self.minimumTrainingSuccessDwellSteps =
            minimumTrainingSuccessDwellSteps
        self.successDwellSteps = successDwellSteps
        self.successDwellCurriculumControlSteps =
            successDwellCurriculumControlSteps
        self.pregraspForwardOffset = pregraspForwardOffset
        self.pregraspLateralOffset = pregraspLateralOffset
        self.approachCommandSpeed = approachCommandSpeed
        self.carryCommandSpeed = carryCommandSpeed
        self.carryHolonomicCommand = carryHolonomicCommand
        self.navigationGoalObservationScale = navigationGoalObservationScale
        self.manipulationHandoffSteps = manipulationHandoffSteps
        self.carryHandoffSteps = carryHandoffSteps
        self.carryCommandRampSteps = carryCommandRampSteps
        self.manipulationArmActionScaleMultiplier =
            manipulationArmActionScaleMultiplier
        self.carryBaseLegActionFraction = carryBaseLegActionFraction
        self.freezeBasePolicyExpert = freezeBasePolicyExpert
        self.freezeManipulationPolicyExpert = freezeManipulationPolicyExpert
        self.freezeCarryPolicyExpert = freezeCarryPolicyExpert
        self.initializeManipulationExpertFromBaseOnTransfer =
            initializeManipulationExpertFromBaseOnTransfer
        self.initializeCarryExpertFromManipulationExpertOnTransfer =
            initializeCarryExpertFromManipulationExpertOnTransfer
        self.compositionalCarryController = compositionalCarryController
        self.upperBodyCarryController = upperBodyCarryController
        self.carryLocomotionControlsTorso = carryLocomotionControlsTorso
        self.initializeCarryExpertFromBaseOnTransfer =
            initializeCarryExpertFromBaseOnTransfer
        self.initializeCarryLocomotionExpertFromBaseOnTransfer =
            initializeCarryLocomotionExpertFromBaseOnTransfer
        self.carryStartReplayProbability = carryStartReplayProbability
        self.advanceReplaySnapshotAtDestinationContact =
            advanceReplaySnapshotAtDestinationContact
        self.carryArmReferenceWeight = carryArmReferenceWeight
        self.carryHoldClearanceMultiplier = carryHoldClearanceMultiplier
        self.carryProgressRewardWeight = carryProgressRewardWeight
        self.carryLocomotionRewardMultiplier =
            carryLocomotionRewardMultiplier
        self.carryTrackingVariance = carryTrackingVariance
        self.coupledCarryCommandTracking = coupledCarryCommandTracking
        self.carryRootProgressRewardWeight = carryRootProgressRewardWeight
        self.carryAlternatingStepRewardWeight =
            carryAlternatingStepRewardWeight
        self.minimumLoadedAlternatingSteps = minimumLoadedAlternatingSteps
    }
}

public final class HumanoidBoxCarryTask: VectorizedRLTask,
    RLEvaluationCriteriaProviding, TrainingModeConfigurable,
    ObservationSchemaTransferProviding, PolicyReferenceRegularizationProviding,
    PolicyReferenceActionRegularizationProviding,
    PolicyActorTrainingWeightProviding, PolicyAuxiliaryExpertGateProviding
{
    public static let observationDimension = 105
    public static let controlDecimation = 4
    public static let boxDimensions = F3(0.22, 0.30, 0.24)
    /// A conventional table replaces the old floor-to-box solid plinth. The
    /// thin top and four lateral legs leave a real central approach lane for
    /// the humanoid's feet and shins; the box remains supported solely by
    /// ordinary collision manifolds.
    public static let pedestalDimensions = F3(0.52, 0.62, 0.06)
    public static let pedestalHeight: Float = 0.66
    public static let boxRestingHeight: Float = 0.78
    public static let tabletopForwardOffset: Float = 0.13
    public static let tabletopCenterHeight: Float = 0.63
    /// Final task success means transporting the box to this distinct table
    /// pose, not merely crossing a scalar X threshold in free space.
    public static let placementPlanarTolerance: Float = 0.11
    public static let placementApproachRadius: Float = 0.18
    public static let placementHeightTolerance: Float = 0.06
    public static let placementMaximumLinearSpeed: Float = 0.15
    public static let placementMaximumAngularSpeed: Float = 0.60
    /// First curriculum replay gate. Two consecutive control frames are long
    /// enough to reject a one-frame contact/separation artifact while still
    /// preserving rare early lifts for self-imitation. This is not task
    /// success and never terminates an episode.
    public static let imitationLiftDwellSteps = 2
    /// Pelvis target measured behind the box center. At the previous 0.40 m
    /// offset plus a 0.20 m entry radius, the near curriculum reset entered
    /// manipulation immediately with the box 0.55 m away. H1 could touch only
    /// the upper edges by crouching; the opposing face centers were outside a
    /// comfortable two-link arm workspace.
    public static let pregraspOffset: Float = 0.30
    /// Manipulation must not take control while the approach actor is still
    /// moving. The navigator commands zero inside the same five-centimeter
    /// radius; a short measured-speed dwell establishes an actual stance
    /// before the arms reach for the load.
    public static let manipulationEntryDistance: Float = 0.05
    public static let manipulationEntrySpeed: Float = 0.10
    public static let manipulationEntryDwellSteps = 8
    /// The H1 forearm collider is a 25 mm-radius capsule whose terminal sphere
    /// is the hand contact shape. A mechanically useful
    /// two-hand grasp may contact anywhere inside the two opposing side faces;
    /// requiring the sphere centers to coincide with the face centers rejects
    /// valid upper/front grasps even when their box-local transforms are stable.
    public static let handCollisionSphereRadius: Float = 0.025
    public static let loadBearingOpposingFaceQuality: Float = 0.5
    public static let loadBearingBoxUprightAlignment: Float = 0.80
    /// Checkpoint schema v0 encoded a centered-face proximity bit at policy
    /// input 91. Preserve that input exactly so immutable trained policies and
    /// residual-flow artifacts remain replayable. It is only an observation
    /// hint: physical milestones, rewards, metrics, and certification use
    /// `isOpposingFaceGrasp`.
    private static let checkpointFaceHintDistance: Float = 0.10
    /// The default final evaluation requires a visually unmistakable four-
    /// centimeter air gap. Training may begin with a smaller, still physical
    /// unsupported gap and ramp to this value; evaluation never relaxes it.
    public static let liftClearance: Float = 0.04
    public static let actionDimension = 19
    private static let trackingVariance: Float = 0.25
    /// Sustained clearance must outweigh simply squeezing the object against
    /// its support. Progress potentials alone telescope to zero and otherwise
    /// let a long, planted grasp dominate the physically useful air-gap state.
    private static let liftHoldRewardWeight: Float = 24
    private static let bilateralAcquisitionRewardWeight: Float = 8
    private static let unilateralContactPenaltyWeight: Float = 4
    /// Contact manifolds persist inside the narrowphase margin. A foot counts
    /// as load bearing only when it carries at least this share of the current
    /// total foot normal load.
    static let loadedFootMinimumNormalLoadShare: Float = 0.02
    /// Reject zero-force sole shuffles as swing: a touchdown must follow a
    /// measured sole clearance of at least 15 mm.
    static let loadedSwingMinimumClearance: Float = 0.015

    static func loadBearingFootContacts(
        manifoldContacts: [Bool], normalLoads: [Float]
    ) -> [Bool] {
        precondition(manifoldContacts.count == 2 && normalLoads.count == 2)
        let total = normalLoads[0] + normalLoads[1]
        guard total > 1e-6 else { return [false, false] }
        return (0..<2).map {
            manifoldContacts[$0]
                && normalLoads[$0] / total
                    >= loadedFootMinimumNormalLoadShare
        }
    }

    static func isStablePlacement(
        lifted: Bool, destinationContact: Bool,
        planarDistance: Float, heightError: Float, uprightAlignment: Float,
        leftHandContact: Bool, rightHandContact: Bool,
        linearSpeed: Float, angularSpeed: Float
    ) -> Bool {
        lifted && destinationContact
            && planarDistance <= placementPlanarTolerance
            && heightError <= placementHeightTolerance
            && uprightAlignment >= loadBearingBoxUprightAlignment
            && !leftHandContact && !rightHandContact
            && linearSpeed <= placementMaximumLinearSpeed
            && angularSpeed <= placementMaximumAngularSpeed
    }

    /// Whether opposing contact duals can supply enough Coulomb friction to
    /// support the box against gravity. A collision-margin manifold with a
    /// zero dual is contact geometry, not a load-bearing grasp.
    static func frictionGraspSupportsWeight(
        leftNormalLoad: Float, rightNormalLoad: Float,
        boxMass: Float, friction: Float,
        gravityMagnitude: Float = 9.81
    ) -> Bool {
        frictionGraspSupportFraction(
            leftNormalLoad: leftNormalLoad,
            rightNormalLoad: rightNormalLoad,
            boxMass: boxMass,
            friction: friction,
            gravityMagnitude: gravityMagnitude) >= 1
    }

    static func frictionGraspSupportFraction(
        leftNormalLoad: Float, rightNormalLoad: Float,
        boxMass: Float, friction: Float,
        gravityMagnitude: Float = 9.81
    ) -> Float {
        guard leftNormalLoad.isFinite, rightNormalLoad.isFinite,
              boxMass.isFinite, friction.isFinite,
              gravityMagnitude.isFinite,
              leftNormalLoad >= 0, rightNormalLoad >= 0,
              boxMass > 0, friction > 0, gravityMagnitude > 0 else {
            return 0
        }
        guard leftNormalLoad > 0, rightNormalLoad > 0 else { return 0 }
        return friction * (leftNormalLoad + rightNormalLoad)
            / (boxMass * gravityMagnitude)
    }

    /// Continuous contact geometry for the two opposing local-Y box faces.
    /// Bilateral collision remains a separate hard requirement at the call
    /// site. Normal-plane error measures whether each sphere is seated on its
    /// expected side; tangential error is charged only after the sphere center
    /// leaves the finite face expanded by the contact margin.
    static func opposingFaceGraspQuality(
        localLeftHand: F3, localRightHand: F3
    ) -> Float {
        guard localLeftHand.y > 0, localRightHand.y < 0 else { return 0 }
        let half = 0.5 * boxDimensions
        let normalTarget = half.y + handCollisionSphereRadius
        let normalError = max(
            abs(localLeftHand.y - normalTarget),
            abs(localRightHand.y + normalTarget))
        let xBound = half.x + handCollisionSphereRadius
        let zBound = half.z + handCollisionSphereRadius
        func tangentialExcess(_ hand: F3) -> Float {
            max(
                max(abs(hand.x) - xBound, 0),
                max(abs(hand.z) - zBound, 0))
        }
        let tangentialError = max(
            tangentialExcess(localLeftHand),
            tangentialExcess(localRightHand))
        let worstError = max(normalError, tangentialError)
        return exp(-400 * worstError * worstError)
    }

    static func isOpposingFaceGrasp(
        localLeftHand: F3, localRightHand: F3,
        bilateralContact: Bool, boxUprightAlignment: Float
    ) -> Bool {
        bilateralContact
            && opposingFaceGraspQuality(
                localLeftHand: localLeftHand,
                localRightHand: localRightHand)
                >= loadBearingOpposingFaceQuality
            && boxUprightAlignment >= loadBearingBoxUprightAlignment
    }

    static func commandTrackingReward(
        planarErrorSquared: Float, yawErrorSquared: Float,
        variance: Float, coupled: Bool
    ) -> Float {
        precondition(
            planarErrorSquared >= 0 && yawErrorSquared >= 0 && variance > 0)
        let linear = exp(-planarErrorSquared / variance)
        let yaw = exp(-yawErrorSquared / variance)
        // The factor of two preserves the additive objective's maximum, so
        // enabling coupling changes the loophole rather than reward scale.
        return coupled ? 2 * linear * yaw : linear + yaw
    }

    public let spec: RLTaskSpec
    public let environment: HumanoidWalkEnv
    public let configuration: HumanoidBoxCarryTaskConfig

    public var evaluationCriteria: RLEvaluationCriteria {
        RLEvaluationCriteria(
            minimumSuccessRate: 0.60,
            minimumTaskMetrics: [
                "episode/approached": 0.85,
                "episode/bilateral_grasped": 0.70,
                "episode/lifted": 0.65,
                "episode/carried": 0.60,
                "episode/placed": 0.60,
                "episode/survived": 0.85,
            ],
            maximumTaskMetrics: ["episode/dropped": 0.20])
    }

    private var trainingMode = false
    private var trainingControlSteps = 0
    private var commands: [F3]
    private var phases: [Int]
    private var stationDistances: [Float]
    private var stationCenters: [F3]
    private var destinationPedestalCenters: [F3]
    private var destinationBoxTargets: [F3]
    private var previousActions: ContiguousArray<Float>
    private var previousJointVelocities: [[Float]]
    private var episodeLengths: [Int]
    private var episodeReturns: [Float]
    private var previousPregraspDistances: [Float]
    private var previousReachDistances: [Float]
    private var previousBoxHeights: [Float]
    private var previousBoxClearances: [Float]
    private var previousPlacementDistances: [Float]
    private var previousRootPositions: [F3]
    private var maximumBoxHeights: [Float]
    private var maximumBoxClearances: [Float]
    private var maximumCarryDistances: [Float]
    private var maximumLoadedRootDisplacements: [Float]
    private var currentStableUnsupportedSteps: [Int]
    private var maximumStableUnsupportedSteps: [Int]
    private var liftOrigins: [F3]
    private var liftRootOrigins: [F3]
    private var loadedFootAirTimes: [[Float]]
    private var loadedFootMaximumClearances: [[Float]]
    private var previousLoadedFootContacts: [[Bool]]
    private var lastLoadedTouchdownFeet: [Int]
    private var loadedTouchdownCounts: [Int]
    private var loadedAlternatingStepCounts: [Int]
    private var approached: [Bool]
    private var approachSettleCounts: [Int]
    private var manipulationHandoffCounts: [Int]
    private var bilateralGrasped: [Bool]
    private var lifted: [Bool]
    private var dropped: [Bool]
    private var carryHandoffCounts: [Int]
    private var successDwellCounts: [Int]
    private var carryMilestoneReached: [Bool]
    private var placed: [Bool]
    private var missedContactCounts: [Int]
    private var latestHandContacts: (left: [Bool], right: [Bool])
    private var latestDestinationContacts: [Bool]
    private var commandRNGs: [SplitMix64]
    private var noiseRNGs: [SplitMix64]
    private var resetRNG: SplitMix64

    /// A solver-complete carry state captured only after the robot has lifted
    /// the box clear of the source table through ordinary contacts. Static
    /// support layout is stored alongside every dynamic body so a later reset
    /// recreates the exact physical problem instead of an approximate pose.
    private struct CarryStartSnapshot {
        var bodyStates: [GPUSolver.RigidBodyState]
        var previousAction: [Float]
        var stationDistance: Float
        var stationCenter: F3
        var destinationPedestalCenter: F3
        var destinationBoxTarget: F3
        var destinationContact: Bool
    }
    private var carryStartSnapshots: [CarryStartSnapshot?]

    /// Exact same-instance fork used by simulator-backed trajectory search.
    /// This is intentionally opaque: it captures both solver temporal state
    /// and every task variable that affects the next observation, reward, or
    /// termination. It is not a portable checkpoint and cannot be restored
    /// into another task or environment layout.
    public struct SpeculationSnapshot {
        fileprivate var solver: GPUSolver.RigidSpeculationSnapshot
        fileprivate var trainingMode: Bool
        fileprivate var trainingControlSteps: Int
        fileprivate var commands: [F3]
        fileprivate var phases: [Int]
        fileprivate var stationDistances: [Float]
        fileprivate var stationCenters: [F3]
        fileprivate var destinationPedestalCenters: [F3]
        fileprivate var destinationBoxTargets: [F3]
        fileprivate var previousActions: ContiguousArray<Float>
        fileprivate var previousJointVelocities: [[Float]]
        fileprivate var episodeLengths: [Int]
        fileprivate var episodeReturns: [Float]
        fileprivate var previousPregraspDistances: [Float]
        fileprivate var previousReachDistances: [Float]
        fileprivate var previousBoxHeights: [Float]
        fileprivate var previousBoxClearances: [Float]
        fileprivate var previousPlacementDistances: [Float]
        fileprivate var previousRootPositions: [F3]
        fileprivate var maximumBoxHeights: [Float]
        fileprivate var maximumBoxClearances: [Float]
        fileprivate var maximumCarryDistances: [Float]
        fileprivate var maximumLoadedRootDisplacements: [Float]
        fileprivate var currentStableUnsupportedSteps: [Int]
        fileprivate var maximumStableUnsupportedSteps: [Int]
        fileprivate var liftOrigins: [F3]
        fileprivate var liftRootOrigins: [F3]
        fileprivate var loadedFootAirTimes: [[Float]]
        fileprivate var loadedFootMaximumClearances: [[Float]]
        fileprivate var previousLoadedFootContacts: [[Bool]]
        fileprivate var lastLoadedTouchdownFeet: [Int]
        fileprivate var loadedTouchdownCounts: [Int]
        fileprivate var loadedAlternatingStepCounts: [Int]
        fileprivate var approached: [Bool]
        fileprivate var approachSettleCounts: [Int]
        fileprivate var manipulationHandoffCounts: [Int]
        fileprivate var bilateralGrasped: [Bool]
        fileprivate var lifted: [Bool]
        fileprivate var dropped: [Bool]
        fileprivate var carryHandoffCounts: [Int]
        fileprivate var successDwellCounts: [Int]
        fileprivate var carryMilestoneReached: [Bool]
        fileprivate var placed: [Bool]
        fileprivate var missedContactCounts: [Int]
        fileprivate var latestLeftHandContacts: [Bool]
        fileprivate var latestRightHandContacts: [Bool]
        fileprivate var latestDestinationContacts: [Bool]
        fileprivate var commandRNGs: [SplitMix64]
        fileprivate var noiseRNGs: [SplitMix64]
        fileprivate var resetRNG: SplitMix64
    }

    /// A cold-start physical/task boundary that can be restored into the same
    /// task revision with a different environment count. Unlike
    /// `SpeculationSnapshot`, solver buffers are not transplanted: authored
    /// body state and task memory are projected into the destination layout,
    /// then incident constraint warm starts are cleared by `setBodyStates`.
    ///
    /// The payload is intentionally opaque. Callers may retain it only while
    /// the originating task/configuration remains alive; durable artifacts
    /// continue to record the complete action lineage used to certify it.
    public struct PortableSpeculationState {
        fileprivate var sourceEnvironment: Int
        fileprivate var task: SpeculationSnapshot
        fileprivate var bodyStates: [GPUSolver.RigidBodyState]
    }

    public init(configuration: HumanoidBoxCarryTaskConfig) throws {
        guard configuration.numEnvironments > 0,
              configuration.maxEpisodeSteps > 0,
              configuration.solverIterations > 0,
              configuration.minimumTrainingStationDistance >= 0.52,
              configuration.evaluationStationDistance
                >= configuration.minimumTrainingStationDistance,
              configuration.stationDistanceCurriculumControlSteps >= 0,
              configuration.boxMass > 0,
              configuration.boxFriction >= 0,
              configuration.minimumTrainingLiftClearance > 0,
              configuration.liftClearance > 0,
              configuration.minimumTrainingLiftClearance
                <= configuration.liftClearance,
              configuration.liftClearanceCurriculumControlSteps >= 0,
              configuration.minimumTrainingCarryDistance > 0,
              configuration.carryDistance > 0,
              configuration.minimumTrainingCarryDistance
                <= configuration.carryDistance,
              configuration.carryDistanceCurriculumControlSteps >= 0,
              configuration.destinationBearingCurriculumControlSteps >= 0,
              configuration.minimumTrainingSuccessDwellSteps > 0,
              configuration.successDwellSteps > 0,
              configuration.minimumTrainingSuccessDwellSteps
                <= configuration.successDwellSteps,
              configuration.successDwellCurriculumControlSteps >= 0,
              configuration.pregraspForwardOffset >= 0.25,
              configuration.pregraspForwardOffset <= 0.50,
              abs(configuration.pregraspLateralOffset) <= 0.20,
              configuration.approachCommandSpeed > 0,
              configuration.carryCommandSpeed > 0,
              configuration.manipulationHandoffSteps > 0,
              configuration.carryHandoffSteps > 0,
              configuration.carryCommandRampSteps > 0,
              configuration.navigationGoalObservationScale > 0,
              configuration.manipulationArmActionScaleMultiplier > 0,
              (0...1).contains(configuration.carryBaseLegActionFraction),
              (0...1).contains(configuration.carryStartReplayProbability),
              !configuration.advanceReplaySnapshotAtDestinationContact
                || configuration.carryStartReplayProbability > 0,
              (0...1).contains(configuration.carryArmReferenceWeight),
              configuration.carryHoldClearanceMultiplier >= 1,
              configuration.carryHoldClearanceMultiplier <= 4,
              configuration.carryProgressRewardWeight > 0,
              configuration.carryProgressRewardWeight <= 1_000,
              configuration.carryLocomotionRewardMultiplier >= 1,
              configuration.carryLocomotionRewardMultiplier <= 50,
              configuration.carryTrackingVariance >= 0.005,
              configuration.carryTrackingVariance <= 1,
              configuration.carryRootProgressRewardWeight >= 0,
              configuration.carryRootProgressRewardWeight <= 1_000,
              configuration.carryAlternatingStepRewardWeight >= 0,
              configuration.carryAlternatingStepRewardWeight <= 100,
              (0...100).contains(configuration.minimumLoadedAlternatingSteps),
              !configuration.compositionalCarryController
                || !configuration.upperBodyCarryController,
              !configuration.carryLocomotionControlsTorso
                || configuration.upperBodyCarryController,
              !configuration
                    .initializeCarryLocomotionExpertFromBaseOnTransfer
                || configuration.upperBodyCarryController,
              !configuration.initializeCarryExpertFromBaseOnTransfer
                || !configuration
                    .initializeCarryExpertFromManipulationExpertOnTransfer else {
            throw RLEnvironmentError.invalidConfiguration(
                "invalid Unitree H1 box-carry configuration")
        }
        let env = try HumanoidWalkEnv(
            numEnvironments: configuration.numEnvironments,
            seed: configuration.seed,
            includeProjectile: true,
            projectileDimensions: Self.boxDimensions,
            projectileMass: configuration.boxMass,
            projectileFriction: configuration.boxFriction,
            carryPedestalSize: Self.pedestalDimensions,
            carryPedestalCenter: F3(
                configuration.minimumTrainingStationDistance
                    + Self.tabletopForwardOffset,
                0, Self.tabletopCenterHeight),
            carryPedestalLegs: true,
            carryDestinationPedestalSize: Self.pedestalDimensions,
            carryDestinationPedestalCenter: F3(
                configuration.minimumTrainingStationDistance
                    + Self.tabletopForwardOffset,
                configuration.carryDistance, Self.tabletopCenterHeight),
            carryDestinationPedestalLegs: true,
            controlProfile: .isaacLab,
            solverIterations: configuration.solverIterations)
        environment = env
        self.configuration = configuration
        spec = RLTaskSpec(
            id: "humanoid-box-carry-v0",
            revision: RLPhysicsContract.deterministicColorSolveV1(40),
            numEnvironments: configuration.numEnvironments,
            observation: RLTensorSpec(
                name: "policy", shape: [Self.observationDimension]),
            action: RLTensorSpec(
                name: "joint_position_offset", shape: [Self.actionDimension]),
            maxEpisodeSteps: configuration.maxEpisodeSteps,
            simulationStep: env.scene.settings.dt,
            controlDecimation: Self.controlDecimation,
            autoReset: configuration.autoReset,
            configurationValues: {
                var values: [String: Float] = [
                "solverIterations": Float(configuration.solverIterations),
                "observationNoise": configuration.observationNoise ? 1 : 0,
                "minimumTrainingStationDistance":
                    configuration.minimumTrainingStationDistance,
                "evaluationStationDistance":
                    configuration.evaluationStationDistance,
                "stationDistanceCurriculumControlSteps": Float(
                    configuration.stationDistanceCurriculumControlSteps),
                "boxMass": configuration.boxMass,
                "boxFriction": configuration.boxFriction,
                "minimumTrainingLiftClearance":
                    configuration.minimumTrainingLiftClearance,
                "liftClearance": configuration.liftClearance,
                "liftClearanceCurriculumControlSteps": Float(
                    configuration.liftClearanceCurriculumControlSteps),
                "minimumTrainingCarryDistance":
                    configuration.minimumTrainingCarryDistance,
                "carryDistance": configuration.carryDistance,
                "carryDistanceCurriculumControlSteps": Float(
                    configuration.carryDistanceCurriculumControlSteps),
                "minimumTrainingSuccessDwellSteps": Float(
                    configuration.minimumTrainingSuccessDwellSteps),
                "successDwellSteps": Float(configuration.successDwellSteps),
                "successDwellCurriculumControlSteps": Float(
                    configuration.successDwellCurriculumControlSteps),
                "pregraspForwardOffset":
                    configuration.pregraspForwardOffset,
                "pregraspLateralOffset":
                    configuration.pregraspLateralOffset,
                "approachCommandSpeed": configuration.approachCommandSpeed,
                "carryCommandSpeed": configuration.carryCommandSpeed,
                "navigationGoalObservationScale":
                    configuration.navigationGoalObservationScale,
                "manipulationHandoffSteps": Float(
                    configuration.manipulationHandoffSteps),
                "carryHandoffSteps": Float(configuration.carryHandoffSteps),
                "carryCommandRampSteps": Float(
                    configuration.carryCommandRampSteps),
                "manipulationArmActionScaleMultiplier":
                    configuration.manipulationArmActionScaleMultiplier,
                "carryBaseLegActionFraction":
                    configuration.carryBaseLegActionFraction,
                "manipulationGatedActor": 1,
                "freezeBasePolicyExpert":
                    configuration.freezeBasePolicyExpert ? 1 : 0,
                "freezeManipulationPolicyExpert":
                    configuration.freezeManipulationPolicyExpert ? 1 : 0,
                "initializeManipulationExpertFromBaseOnTransfer":
                    configuration.initializeManipulationExpertFromBaseOnTransfer
                        ? 1 : 0,
                "initializeCarryExpertFromManipulationExpertOnTransfer":
                        configuration
                            .initializeCarryExpertFromManipulationExpertOnTransfer
                                ? 1 : 0,
                ]
                // Omit disabled new options so checkpoints written before the
                // compositional controller remain exactly replay-compatible.
                if configuration.compositionalCarryController {
                    values["compositionalCarryController"] = 1
                }
                if !configuration.freezeCarryPolicyExpert {
                    values["freezeCarryPolicyExpert"] = 0
                }
                if configuration.upperBodyCarryController {
                    values["upperBodyCarryController"] = 1
                }
                if configuration.carryLocomotionControlsTorso {
                    values["carryLocomotionControlsTorso"] = 1
                }
                if configuration.initializeCarryExpertFromBaseOnTransfer {
                    values["initializeCarryExpertFromBaseOnTransfer"] = 1
                }
                if configuration
                        .initializeCarryLocomotionExpertFromBaseOnTransfer {
                    values[
                        "initializeCarryLocomotionExpertFromBaseOnTransfer"] = 1
                }
                if configuration.carryStartReplayProbability > 0 {
                    values["carryStartReplayProbability"] =
                        configuration.carryStartReplayProbability
                }
                if configuration.advanceReplaySnapshotAtDestinationContact {
                    values["advanceReplaySnapshotAtDestinationContact"] = 1
                }
                if configuration.carryArmReferenceWeight != 1 {
                    values["carryArmReferenceWeight"] =
                        configuration.carryArmReferenceWeight
                }
                if configuration.carryHoldClearanceMultiplier != 1 {
                    values["carryHoldClearanceMultiplier"] =
                        configuration.carryHoldClearanceMultiplier
                }
                if configuration.carryProgressRewardWeight != 25 {
                    values["carryProgressRewardWeight"] =
                        configuration.carryProgressRewardWeight
                }
                if configuration.carryLocomotionRewardMultiplier != 1 {
                    values["carryLocomotionRewardMultiplier"] =
                        configuration.carryLocomotionRewardMultiplier
                }
                if configuration.destinationBearingCurriculumControlSteps > 0 {
                    values["destinationBearingCurriculumControlSteps"] = Float(
                        configuration.destinationBearingCurriculumControlSteps)
                }
                if configuration.carryHolonomicCommand {
                    values["carryHolonomicCommand"] = 1
                }
                if configuration.carryTrackingVariance != Self.trackingVariance {
                    values["carryTrackingVariance"] =
                        configuration.carryTrackingVariance
                }
                if configuration.coupledCarryCommandTracking {
                    values["coupledCarryCommandTracking"] = 1
                }
                if configuration.carryRootProgressRewardWeight > 0 {
                    values["carryRootProgressRewardWeight"] =
                        configuration.carryRootProgressRewardWeight
                }
                if configuration.carryAlternatingStepRewardWeight > 0 {
                    values["carryAlternatingStepRewardWeight"] =
                        configuration.carryAlternatingStepRewardWeight
                }
                if configuration.minimumLoadedAlternatingSteps > 0 {
                    values["minimumLoadedAlternatingSteps"] = Float(
                        configuration.minimumLoadedAlternatingSteps)
                }
                return values
            }())

        let n = configuration.numEnvironments
        commands = [F3](repeating: .zero, count: n)
        phases = [Int](repeating: 0, count: n)
        stationDistances = [Float](repeating: 0, count: n)
        stationCenters = [F3](repeating: .zero, count: n)
        destinationPedestalCenters = [F3](repeating: .zero, count: n)
        destinationBoxTargets = [F3](repeating: .zero, count: n)
        previousActions = ContiguousArray(
            repeating: 0, count: n * Self.actionDimension)
        previousJointVelocities = [[Float]](
            repeating: [Float](repeating: 0, count: Self.actionDimension),
            count: n)
        episodeLengths = [Int](repeating: 0, count: n)
        episodeReturns = [Float](repeating: 0, count: n)
        previousPregraspDistances = [Float](repeating: 0, count: n)
        previousReachDistances = [Float](repeating: 0, count: n)
        previousBoxHeights = [Float](repeating: Self.boxRestingHeight, count: n)
        previousBoxClearances = [Float](repeating: 0, count: n)
        previousPlacementDistances = [Float](repeating: 0, count: n)
        previousRootPositions = [F3](repeating: .zero, count: n)
        maximumBoxHeights = [Float](repeating: Self.boxRestingHeight, count: n)
        maximumBoxClearances = [Float](repeating: 0, count: n)
        maximumCarryDistances = [Float](repeating: 0, count: n)
        maximumLoadedRootDisplacements = [Float](repeating: 0, count: n)
        currentStableUnsupportedSteps = [Int](repeating: 0, count: n)
        maximumStableUnsupportedSteps = [Int](repeating: 0, count: n)
        liftOrigins = [F3](repeating: .zero, count: n)
        liftRootOrigins = [F3](repeating: .zero, count: n)
        loadedFootAirTimes = [[Float]](
            repeating: [Float](repeating: 0, count: 2), count: n)
        loadedFootMaximumClearances = [[Float]](
            repeating: [Float](repeating: 0, count: 2), count: n)
        previousLoadedFootContacts = [[Bool]](
            repeating: [Bool](repeating: false, count: 2), count: n)
        lastLoadedTouchdownFeet = [Int](repeating: -1, count: n)
        loadedTouchdownCounts = [Int](repeating: 0, count: n)
        loadedAlternatingStepCounts = [Int](repeating: 0, count: n)
        approached = [Bool](repeating: false, count: n)
        approachSettleCounts = [Int](repeating: 0, count: n)
        manipulationHandoffCounts = [Int](repeating: 0, count: n)
        bilateralGrasped = [Bool](repeating: false, count: n)
        lifted = [Bool](repeating: false, count: n)
        dropped = [Bool](repeating: false, count: n)
        carryHandoffCounts = [Int](repeating: 0, count: n)
        successDwellCounts = [Int](repeating: 0, count: n)
        carryMilestoneReached = [Bool](repeating: false, count: n)
        placed = [Bool](repeating: false, count: n)
        missedContactCounts = [Int](repeating: 0, count: n)
        latestHandContacts = (
            [Bool](repeating: false, count: n),
            [Bool](repeating: false, count: n))
        latestDestinationContacts = [Bool](repeating: false, count: n)
        commandRNGs = (0..<n).map {
            SplitMix64(seed: configuration.seed
                &+ UInt64($0) &* 0x9E3779B97F4A7C15)
        }
        noiseRNGs = (0..<n).map {
            SplitMix64(seed: configuration.seed
                ^ (UInt64($0) &* 0xD1B54A32D192ED03))
        }
        resetRNG = SplitMix64(seed: configuration.seed
            &+ 0xA0761D6478BD642F)
        carryStartSnapshots = [CarryStartSnapshot?](repeating: nil, count: n)

        let ids = Array(0..<n)
        let seeds = ids.map { configuration.seed &+ UInt64($0) }
        env.reset(ids, seeds: seeds, initialRollPitchRange: 0,
                  initialYawRange: 0)
        initializeEpisodes(ids, states: env.states(), seeds: seeds)
    }

    public func setTrainingMode(_ enabled: Bool) {
        trainingMode = enabled
        if enabled { trainingControlSteps = 0 }
    }

    public func captureSpeculationSnapshot() -> SpeculationSnapshot {
        SpeculationSnapshot(
            solver: environment.solver.captureRigidSpeculationSnapshot(),
            trainingMode: trainingMode,
            trainingControlSteps: trainingControlSteps,
            commands: commands, phases: phases,
            stationDistances: stationDistances,
            stationCenters: stationCenters,
            destinationPedestalCenters: destinationPedestalCenters,
            destinationBoxTargets: destinationBoxTargets,
            previousActions: previousActions,
            previousJointVelocities: previousJointVelocities,
            episodeLengths: episodeLengths,
            episodeReturns: episodeReturns,
            previousPregraspDistances: previousPregraspDistances,
            previousReachDistances: previousReachDistances,
            previousBoxHeights: previousBoxHeights,
            previousBoxClearances: previousBoxClearances,
            previousPlacementDistances: previousPlacementDistances,
            previousRootPositions: previousRootPositions,
            maximumBoxHeights: maximumBoxHeights,
            maximumBoxClearances: maximumBoxClearances,
            maximumCarryDistances: maximumCarryDistances,
            maximumLoadedRootDisplacements:
                maximumLoadedRootDisplacements,
            currentStableUnsupportedSteps: currentStableUnsupportedSteps,
            maximumStableUnsupportedSteps: maximumStableUnsupportedSteps,
            liftOrigins: liftOrigins,
            liftRootOrigins: liftRootOrigins,
            loadedFootAirTimes: loadedFootAirTimes,
            loadedFootMaximumClearances: loadedFootMaximumClearances,
            previousLoadedFootContacts: previousLoadedFootContacts,
            lastLoadedTouchdownFeet: lastLoadedTouchdownFeet,
            loadedTouchdownCounts: loadedTouchdownCounts,
            loadedAlternatingStepCounts: loadedAlternatingStepCounts,
            approached: approached,
            approachSettleCounts: approachSettleCounts,
            manipulationHandoffCounts: manipulationHandoffCounts,
            bilateralGrasped: bilateralGrasped, lifted: lifted,
            dropped: dropped, carryHandoffCounts: carryHandoffCounts,
            successDwellCounts: successDwellCounts,
            carryMilestoneReached: carryMilestoneReached, placed: placed,
            missedContactCounts: missedContactCounts,
            latestLeftHandContacts: latestHandContacts.left,
            latestRightHandContacts: latestHandContacts.right,
            latestDestinationContacts: latestDestinationContacts,
            commandRNGs: commandRNGs, noiseRNGs: noiseRNGs,
            resetRNG: resetRNG)
    }

    public func restoreSpeculationSnapshot(_ snapshot: SpeculationSnapshot) {
        precondition(snapshot.commands.count == spec.numEnvironments
            && snapshot.phases.count == spec.numEnvironments,
            "speculation snapshot belongs to another task shape")
        environment.solver.restoreRigidSpeculationSnapshot(snapshot.solver)
        trainingMode = snapshot.trainingMode
        trainingControlSteps = snapshot.trainingControlSteps
        commands = snapshot.commands
        phases = snapshot.phases
        stationDistances = snapshot.stationDistances
        stationCenters = snapshot.stationCenters
        destinationPedestalCenters = snapshot.destinationPedestalCenters
        destinationBoxTargets = snapshot.destinationBoxTargets
        previousActions = snapshot.previousActions
        previousJointVelocities = snapshot.previousJointVelocities
        episodeLengths = snapshot.episodeLengths
        episodeReturns = snapshot.episodeReturns
        previousPregraspDistances = snapshot.previousPregraspDistances
        previousReachDistances = snapshot.previousReachDistances
        previousBoxHeights = snapshot.previousBoxHeights
        previousBoxClearances = snapshot.previousBoxClearances
        previousPlacementDistances = snapshot.previousPlacementDistances
        previousRootPositions = snapshot.previousRootPositions
        maximumBoxHeights = snapshot.maximumBoxHeights
        maximumBoxClearances = snapshot.maximumBoxClearances
        maximumCarryDistances = snapshot.maximumCarryDistances
        maximumLoadedRootDisplacements =
            snapshot.maximumLoadedRootDisplacements
        currentStableUnsupportedSteps = snapshot.currentStableUnsupportedSteps
        maximumStableUnsupportedSteps = snapshot.maximumStableUnsupportedSteps
        liftOrigins = snapshot.liftOrigins
        liftRootOrigins = snapshot.liftRootOrigins
        loadedFootAirTimes = snapshot.loadedFootAirTimes
        loadedFootMaximumClearances =
            snapshot.loadedFootMaximumClearances
        previousLoadedFootContacts = snapshot.previousLoadedFootContacts
        lastLoadedTouchdownFeet = snapshot.lastLoadedTouchdownFeet
        loadedTouchdownCounts = snapshot.loadedTouchdownCounts
        loadedAlternatingStepCounts = snapshot.loadedAlternatingStepCounts
        approached = snapshot.approached
        approachSettleCounts = snapshot.approachSettleCounts
        manipulationHandoffCounts = snapshot.manipulationHandoffCounts
        bilateralGrasped = snapshot.bilateralGrasped
        lifted = snapshot.lifted
        dropped = snapshot.dropped
        carryHandoffCounts = snapshot.carryHandoffCounts
        successDwellCounts = snapshot.successDwellCounts
        carryMilestoneReached = snapshot.carryMilestoneReached
        placed = snapshot.placed
        missedContactCounts = snapshot.missedContactCounts
        latestHandContacts = (
            snapshot.latestLeftHandContacts,
            snapshot.latestRightHandContacts)
        latestDestinationContacts = snapshot.latestDestinationContacts
        commandRNGs = snapshot.commandRNGs
        noiseRNGs = snapshot.noiseRNGs
        resetRNG = snapshot.resetRNG
    }

    public func capturePortableSpeculationState(
        sourceEnvironment source: Int = 0
    ) -> PortableSpeculationState {
        precondition((0..<spec.numEnvironments).contains(source))
        let refs = environment.refs[source]
        guard let box = refs.projectile else {
            preconditionFailure("carry speculation requires a box body")
        }
        return PortableSpeculationState(
            sourceEnvironment: source,
            task: captureSpeculationSnapshot(),
            bodyStates: environment.solver.bodyStates(refs.bodies + [box]))
    }

    /// Restore one certified row into every row of this task, preserving the
    /// measured controller/task memory but deliberately starting physics from
    /// a portable cold constraint state.
    public func restorePortableSpeculationState(
        _ portable: PortableSpeculationState
    ) {
        let source = portable.sourceEnvironment
        let saved = portable.task
        precondition(saved.commands.indices.contains(source))
        let n = spec.numEnvironments

        func repeated<T>(_ values: [T]) -> [T] {
            [T](repeating: values[source], count: n)
        }
        func repeatedRows<T>(_ values: [[T]]) -> [[T]] {
            [[T]](repeating: values[source], count: n)
        }

        // Keep the destination solver's own buffer shape, but replace every
        // task-owned row with the certified source row.
        var local = captureSpeculationSnapshot()
        local.trainingMode = saved.trainingMode
        local.trainingControlSteps = saved.trainingControlSteps
        local.commands = repeated(saved.commands)
        local.phases = repeated(saved.phases)
        local.stationDistances = repeated(saved.stationDistances)
        local.stationCenters = repeated(saved.stationCenters)
        local.destinationPedestalCenters = repeated(
            saved.destinationPedestalCenters)
        local.destinationBoxTargets = repeated(saved.destinationBoxTargets)
        let actionBase = source * Self.actionDimension
        let sourceAction = Array(saved.previousActions[
            actionBase..<(actionBase + Self.actionDimension)])
        local.previousActions = ContiguousArray(
            (0..<n).flatMap { _ in sourceAction })
        local.previousJointVelocities = repeatedRows(
            saved.previousJointVelocities)
        local.episodeLengths = repeated(saved.episodeLengths)
        local.episodeReturns = repeated(saved.episodeReturns)
        local.previousPregraspDistances = repeated(
            saved.previousPregraspDistances)
        local.previousReachDistances = repeated(saved.previousReachDistances)
        local.previousBoxHeights = repeated(saved.previousBoxHeights)
        local.previousBoxClearances = repeated(saved.previousBoxClearances)
        local.previousPlacementDistances = repeated(
            saved.previousPlacementDistances)
        local.previousRootPositions = repeated(saved.previousRootPositions)
        local.maximumBoxHeights = repeated(saved.maximumBoxHeights)
        local.maximumBoxClearances = repeated(saved.maximumBoxClearances)
        local.maximumCarryDistances = repeated(saved.maximumCarryDistances)
        local.maximumLoadedRootDisplacements = repeated(
            saved.maximumLoadedRootDisplacements)
        local.currentStableUnsupportedSteps = repeated(
            saved.currentStableUnsupportedSteps)
        local.maximumStableUnsupportedSteps = repeated(
            saved.maximumStableUnsupportedSteps)
        local.liftOrigins = repeated(saved.liftOrigins)
        local.liftRootOrigins = repeated(saved.liftRootOrigins)
        local.loadedFootAirTimes = repeatedRows(saved.loadedFootAirTimes)
        local.loadedFootMaximumClearances = repeatedRows(
            saved.loadedFootMaximumClearances)
        local.previousLoadedFootContacts = repeatedRows(
            saved.previousLoadedFootContacts)
        local.lastLoadedTouchdownFeet = repeated(
            saved.lastLoadedTouchdownFeet)
        local.loadedTouchdownCounts = repeated(saved.loadedTouchdownCounts)
        local.loadedAlternatingStepCounts = repeated(
            saved.loadedAlternatingStepCounts)
        local.approached = repeated(saved.approached)
        local.approachSettleCounts = repeated(saved.approachSettleCounts)
        local.manipulationHandoffCounts = repeated(
            saved.manipulationHandoffCounts)
        local.bilateralGrasped = repeated(saved.bilateralGrasped)
        local.lifted = repeated(saved.lifted)
        local.dropped = repeated(saved.dropped)
        local.carryHandoffCounts = repeated(saved.carryHandoffCounts)
        local.successDwellCounts = repeated(saved.successDwellCounts)
        local.carryMilestoneReached = repeated(saved.carryMilestoneReached)
        local.placed = repeated(saved.placed)
        local.missedContactCounts = repeated(saved.missedContactCounts)
        local.latestLeftHandContacts = repeated(
            saved.latestLeftHandContacts)
        local.latestRightHandContacts = repeated(
            saved.latestRightHandContacts)
        local.latestDestinationContacts = repeated(
            saved.latestDestinationContacts)
        local.commandRNGs = repeated(saved.commandRNGs)
        local.noiseRNGs = repeated(saved.noiseRNGs)
        local.resetRNG = saved.resetRNG
        restoreSpeculationSnapshot(local)

        let stationCenters = [F3](repeating: saved.stationCenters[source],
                                  count: n)
        let destinationCenters = [F3](
            repeating: saved.destinationPedestalCenters[source], count: n)
        let boxCenter = portable.bodyStates.last!.position
        environment.placeCarryStations(
            environmentIDs: Array(0..<n),
            pedestalCenters: stationCenters,
            boxCenters: [F3](repeating: boxCenter, count: n),
            destinationPedestalCenters: destinationCenters)

        var bodyUpdates = [GPUSolver.BodyStateUpdate]()
        var motorTargets = [GPUSolver.MotorTargetUpdate]()
        bodyUpdates.reserveCapacity(n * portable.bodyStates.count)
        motorTargets.reserveCapacity(n * Self.actionDimension)
        for environmentIndex in 0..<n {
            let refs = environment.refs[environmentIndex]
            guard let box = refs.projectile else {
                preconditionFailure("carry speculation requires a box body")
            }
            let bodies = refs.bodies + [box]
            precondition(bodies.count == portable.bodyStates.count)
            for (body, state) in zip(bodies, portable.bodyStates) {
                bodyUpdates.append(.init(
                    body: body, position: state.position,
                    rotation: state.rotation,
                    linearVelocity: state.linearVelocity,
                    angularVelocity: state.angularVelocity))
            }
            for jointIndex in 0..<Self.actionDimension {
                var action = sourceAction[jointIndex]
                if jointIndex >= 11 {
                    action *= configuration
                        .manipulationArmActionScaleMultiplier
                }
                let jointID = refs.motors[jointIndex]
                let joint = environment.scene.joints[jointID]
                let requested =
                    HumanoidWalkEnv.defaultJointPositions[jointIndex]
                    + action * HumanoidWalkEnv.actionScales[jointIndex]
                motorTargets.append(.init(
                    joint: jointID,
                    angle: simd_clamp(
                        requested, joint.limitLo, joint.limitHi)))
            }
            carryStartSnapshots[environmentIndex] = nil
        }
        environment.solver.setBodyStates(bodyUpdates)
        environment.solver.setMotorTargets(motorTargets)
    }

    /// Make every batched speculative replica an exact cold-start copy of one
    /// certified task row. Batched CEM must compare actions from the same
    /// physical branch; replaying a long contact-rich prefix independently
    /// in each replica permits tiny solver-order differences to become
    /// different grasp and foot-contact states before search even begins.
    ///
    /// `setBodyStates` deliberately clears incident joint/contact warm starts
    /// for every copied body, so all replicas share the same explicit
    /// branching boundary instead of inheriting different hidden multipliers.
    public func canonicalizeSpeculationReplicas(
        sourceEnvironment source: Int = 0
    ) {
        precondition((0..<spec.numEnvironments).contains(source))
        let sourceRefs = environment.refs[source]
        guard let sourceProjectile = sourceRefs.projectile else {
            preconditionFailure("carry speculation requires a box body")
        }
        let sourceBodyIDs = sourceRefs.bodies + [sourceProjectile]
        let sourceBodyStates = environment.solver.bodyStates(sourceBodyIDs)
        var bodyUpdates = [GPUSolver.BodyStateUpdate]()
        bodyUpdates.reserveCapacity(
            spec.numEnvironments * sourceBodyStates.count)
        var motorTargets = [GPUSolver.MotorTargetUpdate]()
        motorTargets.reserveCapacity(
            spec.numEnvironments * Self.actionDimension)
        // Include the source row itself. `setBodyStates` is also the declared
        // cold-start boundary: it clears incident joint/contact warm starts.
        // Returning early for a one-environment task previously made UI
        // validation continue with warm multipliers while batched planning
        // cold-started row zero, so a long contact lineage could not replay
        // across the two required execution regimes.
        for e in 0..<spec.numEnvironments {
            let refs = environment.refs[e]
            guard let projectile = refs.projectile else {
                preconditionFailure("carry speculation requires a box body")
            }
            let bodyIDs = refs.bodies + [projectile]
            precondition(bodyIDs.count == sourceBodyStates.count)
            for (body, state) in zip(bodyIDs, sourceBodyStates) {
                bodyUpdates.append(.init(
                    body: body, position: state.position,
                    rotation: state.rotation,
                    linearVelocity: state.linearVelocity,
                    angularVelocity: state.angularVelocity))
            }
            for j in 0..<Self.actionDimension {
                let jointID = refs.motors[j]
                var action = previousActions[
                    source * Self.actionDimension + j]
                if j >= 11 {
                    action *= configuration
                        .manipulationArmActionScaleMultiplier
                }
                let requested = HumanoidWalkEnv.defaultJointPositions[j]
                    + action * HumanoidWalkEnv.actionScales[j]
                let joint = environment.scene.joints[jointID]
                motorTargets.append(.init(
                    joint: jointID,
                    angle: simd_clamp(
                        requested, joint.limitLo, joint.limitHi)))
            }
        }
        environment.solver.setBodyStates(bodyUpdates)
        environment.solver.setMotorTargets(motorTargets)

        for e in 0..<spec.numEnvironments where e != source {
            commands[e] = commands[source]
            phases[e] = phases[source]
            stationDistances[e] = stationDistances[source]
            stationCenters[e] = stationCenters[source]
            destinationPedestalCenters[e] =
                destinationPedestalCenters[source]
            destinationBoxTargets[e] = destinationBoxTargets[source]
            previousJointVelocities[e] = previousJointVelocities[source]
            episodeLengths[e] = episodeLengths[source]
            episodeReturns[e] = episodeReturns[source]
            previousPregraspDistances[e] =
                previousPregraspDistances[source]
            previousReachDistances[e] = previousReachDistances[source]
            previousBoxHeights[e] = previousBoxHeights[source]
            previousBoxClearances[e] = previousBoxClearances[source]
            previousPlacementDistances[e] =
                previousPlacementDistances[source]
            previousRootPositions[e] = previousRootPositions[source]
            maximumBoxHeights[e] = maximumBoxHeights[source]
            maximumBoxClearances[e] = maximumBoxClearances[source]
            maximumCarryDistances[e] = maximumCarryDistances[source]
            maximumLoadedRootDisplacements[e] =
                maximumLoadedRootDisplacements[source]
            currentStableUnsupportedSteps[e] =
                currentStableUnsupportedSteps[source]
            maximumStableUnsupportedSteps[e] =
                maximumStableUnsupportedSteps[source]
            liftOrigins[e] = liftOrigins[source]
            liftRootOrigins[e] = liftRootOrigins[source]
            loadedFootAirTimes[e] = loadedFootAirTimes[source]
            loadedFootMaximumClearances[e] =
                loadedFootMaximumClearances[source]
            previousLoadedFootContacts[e] =
                previousLoadedFootContacts[source]
            lastLoadedTouchdownFeet[e] =
                lastLoadedTouchdownFeet[source]
            loadedTouchdownCounts[e] = loadedTouchdownCounts[source]
            loadedAlternatingStepCounts[e] =
                loadedAlternatingStepCounts[source]
            approached[e] = approached[source]
            approachSettleCounts[e] = approachSettleCounts[source]
            manipulationHandoffCounts[e] =
                manipulationHandoffCounts[source]
            bilateralGrasped[e] = bilateralGrasped[source]
            lifted[e] = lifted[source]
            dropped[e] = dropped[source]
            carryHandoffCounts[e] = carryHandoffCounts[source]
            successDwellCounts[e] = successDwellCounts[source]
            carryMilestoneReached[e] = carryMilestoneReached[source]
            placed[e] = placed[source]
            missedContactCounts[e] = missedContactCounts[source]
            latestHandContacts.left[e] = latestHandContacts.left[source]
            latestHandContacts.right[e] = latestHandContacts.right[source]
            latestDestinationContacts[e] =
                latestDestinationContacts[source]
            commandRNGs[e] = commandRNGs[source]
            noiseRNGs[e] = noiseRNGs[source]
            carryStartSnapshots[e] = carryStartSnapshots[source]
            for j in 0..<Self.actionDimension {
                previousActions[e * Self.actionDimension + j] =
                    previousActions[source * Self.actionDimension + j]
            }
        }
    }

    /// Canonicalize simulator/task state and the host-visible observation
    /// consumed by the next policy or flow action.
    public func canonicalizeSpeculationReplicas(
        observation: inout RLObservationBatch,
        sourceEnvironment source: Int = 0
    ) {
        precondition((0..<spec.numEnvironments).contains(source))
        canonicalizeSpeculationReplicas(sourceEnvironment: source)
        guard spec.numEnvironments > 1 else { return }

        broadcastSpeculationSourceRow(
            &observation.policy, sourceEnvironment: source)
        broadcastSpeculationSourceRow(
            &observation.privileged, sourceEnvironment: source)
    }

    /// Canonicalize both simulator/task state and every retained host-visible
    /// batch. Use this overload when the caller retains the previous step
    /// result across a speculative branch boundary.
    public func canonicalizeSpeculationReplicas(
        observation: inout RLObservationBatch,
        result: inout RLStepBatch,
        sourceEnvironment source: Int = 0
    ) {
        canonicalizeSpeculationReplicas(
            observation: &observation, sourceEnvironment: source)

        broadcastSpeculationSourceRow(
            &result.observations.policy, sourceEnvironment: source)
        broadcastSpeculationSourceRow(
            &result.observations.privileged, sourceEnvironment: source)
        broadcastSpeculationSourceRow(
            &result.rewards, sourceEnvironment: source)
        broadcastSpeculationSourceRow(
            &result.terminated, sourceEnvironment: source)
        broadcastSpeculationSourceRow(
            &result.truncated, sourceEnvironment: source)
        broadcastSpeculationSourceRow(
            &result.successes, sourceEnvironment: source)
        broadcastSpeculationSourceRow(
            &result.imitationMilestones, sourceEnvironment: source)
        broadcastSpeculationSourceRow(
            &result.finalObservations, sourceEnvironment: source)
        broadcastSpeculationSourceRow(
            &result.hasFinalObservation, sourceEnvironment: source)
        for key in result.metrics.keys {
            var values = result.metrics[key]!
            broadcastSpeculationSourceRow(
                &values, sourceEnvironment: source)
            result.metrics[key] = values
        }
    }

    private func broadcastSpeculationSourceRow<T>(
        _ values: inout ContiguousArray<T>,
        sourceEnvironment source: Int
    ) {
        guard !values.isEmpty else { return }
        precondition(values.count.isMultiple(of: spec.numEnvironments))
        let width = values.count / spec.numEnvironments
        let sourceBase = source * width
        let row = Array(values[sourceBase..<(sourceBase + width)])
        for environment in 0..<spec.numEnvironments
            where environment != source {
            let base = environment * width
            for component in 0..<width {
                values[base + component] = row[component]
            }
        }
    }

    public func setTrainingProgress(environmentSteps: Int) {
        precondition(environmentSteps >= 0)
        trainingControlSteps = environmentSteps / spec.numEnvironments
    }

    public var curriculumProgress: Float {
        curriculumProgress(
            duration: configuration.stationDistanceCurriculumControlSteps)
    }

    /// Evaluation always uses the final target. Training alone expands the
    /// measured carry boundary from an already reachable starting distance.
    public var currentCarryDistance: Float {
        let progress = curriculumProgress(
            duration: configuration.carryDistanceCurriculumControlSteps)
        return configuration.minimumTrainingCarryDistance
            + progress * (configuration.carryDistance
                - configuration.minimumTrainingCarryDistance)
    }

    /// A success always requires the oriented box to have no support contact.
    /// This curriculum changes only the measured air-gap threshold, beginning
    /// at a learnable millimeter-scale separation and ending at the exact
    /// evaluation requirement.
    public var currentLiftClearance: Float {
        let progress = curriculumProgress(
            duration: configuration.liftClearanceCurriculumControlSteps)
        return configuration.minimumTrainingLiftClearance
            + progress * (configuration.liftClearance
                - configuration.minimumTrainingLiftClearance)
    }

    public var currentSuccessDwellSteps: Int {
        let progress = curriculumProgress(
            duration: configuration.successDwellCurriculumControlSteps)
        let span = configuration.successDwellSteps
            - configuration.minimumTrainingSuccessDwellSteps
        return min(
            configuration.successDwellSteps,
            configuration.minimumTrainingSuccessDwellSteps
                + Int(floor(Float(span) * progress)))
    }

    private func curriculumProgress(duration: Int) -> Float {
        guard trainingMode, duration > 0 else { return 1 }
        return simd_clamp(
            Float(trainingControlSteps) / Float(duration), 0, 1)
    }

    private func carryNavigationTarget(environment e: Int) -> F3 {
        guard trainingMode,
              configuration.destinationBearingCurriculumControlSteps > 0 else {
            return destinationBoxTargets[e]
        }
        let progress = curriculumProgress(
            duration: configuration.destinationBearingCurriculumControlSteps)
        let bearing = 0.5 * Float.pi * progress
        return F3(
            stationDistances[e]
                + configuration.carryDistance * cos(bearing),
            configuration.carryDistance * sin(bearing),
            Self.boxRestingHeight)
    }

    public func reset(environments ids: [Int]?, seed: UInt64,
                      into observations: inout RLObservationBatch) throws {
        try observations.validate(for: spec)
        let envIDs = try checkedEnvironmentIDs(ids)
        try environment.solver.synchronize()
        let seeds = envIDs.map {
            seed &+ UInt64($0) &* 0x9E3779B97F4A7C15
        }
        environment.reset(envIDs, seeds: seeds, initialRollPitchRange: 0,
                          initialYawRange: 0)
        initializeEpisodes(envIDs, states: environment.states(), seeds: seeds)
        fillObservations(
            environment.states(), manipulation: environment.manipulationStates(),
            contacts: latestHandContacts, into: &observations.policy)
        try observations.validate(for: spec)
    }

    public func step(actions: RLActionBatch,
                     into result: inout RLStepBatch) throws {
        try actions.validate(for: spec)
        try result.validate(for: spec)
        try environment.solver.synchronize()
        result.clearSignals()
        let n = spec.numEnvironments
        let dt = spec.controlStep
        var actionRate = ContiguousArray(repeating: Float(0), count: n)
        for e in 0..<n {
            for j in 0..<Self.actionDimension {
                let index = e * Self.actionDimension + j
                let delta = actions.values[index] - previousActions[index]
                actionRate[e] += delta * delta
            }
        }

        // The walking backbone retains the published 0.5-radian action scale.
        // Once manipulation begins, a bounded specialist action must cover
        // the H1 arm workspace without relying on numerically unbounded motor
        // targets. The multiplier gives both arms a one-radian default range;
        // the instantiated joint limits remain the final physical safety
        // boundary. This keeps PPO likelihoods, replay actions, and the
        // commands applied to physics in the same bounded coordinate system.
        var appliedActions = actions.values
        for e in 0..<n where phases[e] >= 1 {
            let base = e * Self.actionDimension
            for j in 11..<Self.actionDimension {
                appliedActions[base + j] *=
                    configuration.manipulationArmActionScaleMultiplier
            }
        }
        try environment.stepChecked(
            normalizedActions: appliedActions,
            decimation: Self.controlDecimation,
            clampActions: false, clampTargetsToLimits: true)
        if trainingMode { trainingControlSteps += 1 }
        var states = environment.states()
        var manipulation = environment.manipulationStates()
        let groundContacts = environment.groundContacts()
        let groundContactNormalLoads =
            environment.groundContactNormalLoads()
        let loadBearingFootContacts = (0..<n).map {
            Self.loadBearingFootContacts(
                manifoldContacts: groundContacts.feet[$0],
                normalLoads: groundContactNormalLoads[$0])
        }
        let handContacts = environment.boxHandContacts()
        let handContactNormalLoads =
            environment.boxHandContactNormalLoads()
        let supportContacts = environment.boxCarrySupportContacts()
        let previousDestinationContacts = latestDestinationContacts
        latestHandContacts = handContacts
        latestDestinationContacts = supportContacts.destination

        var rewardLocomotion = ContiguousArray(repeating: Float(0), count: n)
        var rewardApproach = ContiguousArray(repeating: Float(0), count: n)
        var rewardReach = ContiguousArray(repeating: Float(0), count: n)
        var rewardContact = ContiguousArray(repeating: Float(0), count: n)
        var rewardLift = ContiguousArray(repeating: Float(0), count: n)
        var rewardRetention = ContiguousArray(repeating: Float(0), count: n)
        var rewardCarry = ContiguousArray(repeating: Float(0), count: n)
        var rewardCarryRootProgress = ContiguousArray(
            repeating: Float(0), count: n)
        var rewardLoadedAlternatingStep = ContiguousArray(
            repeating: Float(0), count: n)
        var rewardPlacement = ContiguousArray(repeating: Float(0), count: n)
        var penaltyBoxDescent = ContiguousArray(repeating: Float(0), count: n)
        var penaltyUnilateralContact = ContiguousArray(
            repeating: Float(0), count: n)
        var statePregraspDistance = ContiguousArray(
            repeating: Float(0), count: n)
        var stateReachDistance = ContiguousArray(repeating: Float(0), count: n)
        var stateBoxHeight = ContiguousArray(repeating: Float(0), count: n)
        var stateBoxClearance = ContiguousArray(repeating: Float(0), count: n)
        var stateCarryDistance = ContiguousArray(repeating: Float(0), count: n)
        var stateLoadedRootDisplacement = ContiguousArray(
            repeating: Float(0), count: n)
        var stateRootDestinationDistance = ContiguousArray(
            repeating: Float(0), count: n)
        var stateLoadedTouchdowns = ContiguousArray(
            repeating: Float(0), count: n)
        var stateLoadedAlternatingSteps = ContiguousArray(
            repeating: Float(0), count: n)
        var stateLeftLoadedFootAirTime = ContiguousArray(
            repeating: Float(0), count: n)
        var stateRightLoadedFootAirTime = ContiguousArray(
            repeating: Float(0), count: n)
        var stateMaximumLoadedFootAirTime = ContiguousArray(
            repeating: Float(0), count: n)
        var stateLeftFootNormalLoad = ContiguousArray(
            repeating: Float(0), count: n)
        var stateRightFootNormalLoad = ContiguousArray(
            repeating: Float(0), count: n)
        var stateFootUnloadingFraction = ContiguousArray(
            repeating: Float(0), count: n)
        var stateLeftFootGroundClearance = ContiguousArray(
            repeating: Float(0), count: n)
        var stateRightFootGroundClearance = ContiguousArray(
            repeating: Float(0), count: n)
        var stateMaximumLoadedSwingClearance = ContiguousArray(
            repeating: Float(0), count: n)
        var stateLeftLoadBearingFootContact = ContiguousArray(
            repeating: Float(0), count: n)
        var stateRightLoadBearingFootContact = ContiguousArray(
            repeating: Float(0), count: n)
        var stateLeftFootContact = ContiguousArray(
            repeating: Float(0), count: n)
        var stateRightFootContact = ContiguousArray(
            repeating: Float(0), count: n)
        var stateLiftFraction = ContiguousArray(repeating: Float(0), count: n)
        var stateLeftContact = ContiguousArray(repeating: Float(0), count: n)
        var stateRightContact = ContiguousArray(repeating: Float(0), count: n)
        var stateLeftHandNormalLoad = ContiguousArray(
            repeating: Float(0), count: n)
        var stateRightHandNormalLoad = ContiguousArray(
            repeating: Float(0), count: n)
        var stateGraspFrictionSupportFraction = ContiguousArray(
            repeating: Float(0), count: n)
        var stateGraspQuality = ContiguousArray(repeating: Float(0), count: n)
        var stateLoadBearingGrasp = ContiguousArray(
            repeating: Float(0), count: n)
        var stateFrictionLoadBearingGrasp = ContiguousArray(
            repeating: Float(0), count: n)
        var statePhase = ContiguousArray(repeating: Float(0), count: n)
        var stateBoxGroundContact = ContiguousArray(
            repeating: Float(0), count: n)
        var stateBoxPedestalContact = ContiguousArray(
            repeating: Float(0), count: n)
        var stateBoxDestinationContact = ContiguousArray(
            repeating: Float(0), count: n)
        var statePlacementDistance = ContiguousArray(
            repeating: Float(0), count: n)
        var stateReleased = ContiguousArray(repeating: Float(0), count: n)
        var stateMissedContactSteps = ContiguousArray(
            repeating: Float(0), count: n)
        var stateCarryHandoff = ContiguousArray(
            repeating: Float(0), count: n)
        var stateManipulationHandoff = ContiguousArray(
            repeating: Float(0), count: n)
        var stateCarryCommandRamp = ContiguousArray(
            repeating: Float(0), count: n)
        var stateStableUnsupportedSteps = ContiguousArray(
            repeating: Float(0), count: n)
        var stateMaximumActuatorTorqueRatio = ContiguousArray(
            repeating: Float(0), count: n)
        var stateMaximumArmActuatorTorqueRatio = ContiguousArray(
            repeating: Float(0), count: n)
        var stateSaturatedActuatorCount = ContiguousArray(
            repeating: Float(0), count: n)
        var stateSaturatedArmActuatorCount = ContiguousArray(
            repeating: Float(0), count: n)
        var stateMinimumJointLimitMargin = ContiguousArray(
            repeating: Float.infinity, count: n)
        var stateMaximumRequestedTargetClamp = ContiguousArray(
            repeating: Float(0), count: n)
        var taskImitationMilestone = ContiguousArray(
            repeating: Float(0), count: n)
        var referenceStateReset = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeReturnMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeLengthMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeApproachedMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeGraspedMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeLiftedMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeCarriedMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodePlacedMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeDroppedMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeSurvivedMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeFinalCarryDistanceMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeMaximumCarryDistanceMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeMaximumLoadedRootDisplacementMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeLoadedTouchdownsMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeLoadedAlternatingStepsMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeMaximumBoxHeightMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeMaximumBoxClearanceMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeMaximumStableUnsupportedStepsMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var resetIDs = [Int]()
        var resetSeeds = [UInt64]()
        var carryStartResetIDs = [Int]()
        var newCarryStartIDs = [Int]()
        var newDestinationStartIDs = [Int]()
        let carryTarget = currentCarryDistance
        let liftClearanceTarget = currentLiftClearance
        let carryHoldClearanceTarget = liftClearanceTarget
            * configuration.carryHoldClearanceMultiplier
        let dwellTarget = currentSuccessDwellSteps

        for e in 0..<n {
            let state = states[e]
            let item = manipulation[e]
            let footGroundClearances = [
                HumanoidWalkEnv.footGroundClearance(state.leftFoot),
                HumanoidWalkEnv.footGroundClearance(state.rightFoot),
            ]
            let pregrasp = pregraspPosition(environment: e)
            let pregraspDelta = pregrasp - state.root.position
            let pregraspDistance = sqrt(
                pregraspDelta.x * pregraspDelta.x
                    + pregraspDelta.y * pregraspDelta.y)
            let (leftTarget, rightTarget) = handTargets(item.object)
            let leftDistance = length(item.leftHand.position - leftTarget)
            let rightDistance = length(item.rightHand.position - rightTarget)
            let reachDistance = leftDistance + rightDistance
            let bilateral = handContacts.left[e] && handContacts.right[e]
            // Face-center proximity is a smooth reaching objective, not a
            // physical grasp predicate. Once collision occurs, the finite-face
            // geometry below decides whether the hold is load bearing.
            let leftHoldScore = exp(-40 * leftDistance * leftDistance)
            let rightHoldScore = exp(-40 * rightDistance * rightDistance)
            let worldBoxUp = item.object.rotation.act(F3(0, 0, 1))
            let inverseBox = item.object.rotation.conjugate
            let localLeftHand = inverseBox.act(
                item.leftHand.position - item.object.position)
            let localRightHand = inverseBox.act(
                item.rightHand.position - item.object.position)
            let twoHandHoldScore = Self.opposingFaceGraspQuality(
                localLeftHand: localLeftHand,
                localRightHand: localRightHand)
            let levelGripScore = max(worldBoxUp.z, 0)
            let geometricOpposingGrasp = Self.isOpposingFaceGrasp(
                localLeftHand: localLeftHand,
                localRightHand: localRightHand,
                bilateralContact: bilateral,
                boxUprightAlignment: worldBoxUp.z)
            let frictionLoadBearingGrasp = geometricOpposingGrasp
                && Self.frictionGraspSupportsWeight(
                    leftNormalLoad: handContactNormalLoads.left[e],
                    rightNormalLoad: handContactNormalLoads.right[e],
                    boxMass: configuration.boxMass,
                    friction: configuration.boxFriction)
            // Revision-40 checkpoints observed this historical geometric phase
            // signal. Keep it stable for exact checkpoint replay; new physical
            // certification consumes frictionLoadBearingGrasp separately.
            let loadBearingGrasp = geometricOpposingGrasp
            // Use the oriented cuboid's actual lowest corner.  Subtracting
            // only half the local Z size overestimates clearance whenever the
            // box pitches or rolls—the exact regime produced by a two-arm
            // friction grasp.
            let boxBottom = orientedBoxBottom(item.object)
            let boxClearance = boxBottom - Self.pedestalHeight
            maximumBoxHeights[e] = max(
                maximumBoxHeights[e], item.object.position.z)
            maximumBoxClearances[e] = max(
                maximumBoxClearances[e], boxClearance)
            let liftClear = boxClearance > liftClearanceTarget
            let physicallyClearOfPedestal = liftClear
                && !supportContacts.source[e]

            let planarSpeed = sqrt(
                state.root.linearVelocity.x * state.root.linearVelocity.x
                    + state.root.linearVelocity.y * state.root.linearVelocity.y)
            let settledAtPregrasp = pregraspDistance
                    <= Self.manipulationEntryDistance
                && planarSpeed <= Self.manipulationEntrySpeed
            approachSettleCounts[e] = settledAtPregrasp
                ? approachSettleCounts[e] + 1 : 0
            if approachSettleCounts[e] >= Self.manipulationEntryDwellSteps {
                approached[e] = true
            }
            // The locomotion and manipulation experts can command different
            // joint targets at the measured approach boundary. Blend them
            // over an observed, monotonic handoff instead of applying a
            // one-frame whole-body action discontinuity.
            if approached[e], manipulationHandoffCounts[e]
                    < configuration.manipulationHandoffSteps {
                manipulationHandoffCounts[e] += 1
            }
            let newlyAcquiredBilateralGrasp = loadBearingGrasp
                && !bilateralGrasped[e]
            if loadBearingGrasp { bilateralGrasped[e] = true }
            let newlyLifted = !lifted[e] && bilateralGrasped[e]
                && physicallyClearOfPedestal
            if newlyLifted {
                lifted[e] = true
                liftOrigins[e] = item.object.position
                liftRootOrigins[e] = state.root.position
                maximumLoadedRootDisplacements[e] = 0
                loadedFootAirTimes[e] = [0, 0]
                loadedFootMaximumClearances[e] = [0, 0]
                previousLoadedFootContacts[e] =
                    loadBearingFootContacts[e]
                lastLoadedTouchdownFeet[e] = -1
                loadedTouchdownCounts[e] = 0
                loadedAlternatingStepCounts[e] = 0
            }
            // A physical lift used to switch every arm action from the
            // manipulation expert to an independently learned carry expert
            // in one control frame. Even good endpoint policies can be far
            // apart at that boundary, turning a successful grasp into an
            // artificial action impulse. Advance a short, observable blend
            // monotonically after the measured unsupported-lift event. This
            // changes only policy composition and commanded walking speed;
            // object support remains entirely contact/friction based.
            if lifted[e], carryHandoffCounts[e]
                    < configuration.carryHandoffSteps
                        + configuration.carryCommandRampSteps {
                carryHandoffCounts[e] += 1
            }
            if lifted[e] {
                missedContactCounts[e] = bilateral
                    || supportContacts.destination[e]
                    ? 0 : missedContactCounts[e] + 1
            }
            let carryDelta = item.object.position - liftOrigins[e]
            let carryDistance = lifted[e]
                ? sqrt(carryDelta.x * carryDelta.x
                    + carryDelta.y * carryDelta.y) : 0
            maximumCarryDistances[e] = max(
                maximumCarryDistances[e], carryDistance)
            let loadedRootDelta = state.root.position - liftRootOrigins[e]
            let loadedRootDisplacement = lifted[e]
                ? hypot(loadedRootDelta.x, loadedRootDelta.y) : 0
            maximumLoadedRootDisplacements[e] = max(
                maximumLoadedRootDisplacements[e],
                loadedRootDisplacement)
            // A height-only test incorrectly extends the pedestal's top plane
            // across the entire world.  Once carried beyond the finite
            // pedestal, a box can dip below that plane while remaining held
            // well above the ground.  Terminate only on a real support
            // manifold or sustained loss of the bilateral grasp.
            let placementDelta = destinationBoxTargets[e]
                - item.object.position
            let placementPlanarDistance = sqrt(
                placementDelta.x * placementDelta.x
                    + placementDelta.y * placementDelta.y)
            let placementHeightError = abs(placementDelta.z)
            let placementDistance = sqrt(
                placementPlanarDistance * placementPlanarDistance
                    + placementHeightError * placementHeightError)
            let objectLinearSpeed = length(item.object.linearVelocity)
            let objectAngularSpeed = length(item.object.angularVelocity)
            let destinationSupported = supportContacts.destination[e]
                && placementPlanarDistance <= Self.placementPlanarTolerance
                && placementHeightError <= Self.placementHeightTolerance
                && worldBoxUp.z >= Self.loadBearingBoxUprightAlignment
            let released = !handContacts.left[e] && !handContacts.right[e]
            let stablePlacement = Self.isStablePlacement(
                lifted: lifted[e],
                destinationContact: supportContacts.destination[e],
                planarDistance: placementPlanarDistance,
                heightError: placementHeightError,
                uprightAlignment: worldBoxUp.z,
                leftHandContact: handContacts.left[e],
                rightHandContact: handContacts.right[e],
                linearSpeed: objectLinearSpeed,
                angularSpeed: objectAngularSpeed)
            let inPlacementRegion = lifted[e]
                && placementPlanarDistance <= Self.placementApproachRadius
            if configuration.advanceReplaySnapshotAtDestinationContact,
               trainingMode, lifted[e], bilateral,
               supportContacts.destination[e],
               !previousDestinationContacts[e] {
                newDestinationStartIDs.append(e)
            }

            let touchedSupport = supportContacts.ground[e]
                || supportContacts.source[e]
            if lifted[e], (touchedSupport || missedContactCounts[e] > 20) {
                dropped[e] = true
            }
            let stableUnsupported = lifted[e] && bilateral && !dropped[e]
                && !supportContacts.ground[e]
                && !supportContacts.source[e]
                && !supportContacts.destination[e]
            currentStableUnsupportedSteps[e] = stableUnsupported
                ? currentStableUnsupportedSteps[e] + 1 : 0
            maximumStableUnsupportedSteps[e] = max(
                maximumStableUnsupportedSteps[e],
                currentStableUnsupportedSteps[e])
            let reachedImitationMilestone = newlyAcquiredBilateralGrasp
                || currentStableUnsupportedSteps[e]
                    == Self.imitationLiftDwellSteps
            result.imitationMilestones[e] = reachedImitationMilestone
            if currentStableUnsupportedSteps[e]
                    == Self.imitationLiftDwellSteps {
                newCarryStartIDs.append(e)
            }

            // Count only alternating solver-contact touchdowns after a real
            // swing and while the load remains physically unsupported. A
            // planted shuffle, hop, simultaneous landing, or arm-only box
            // sweep cannot satisfy this locomotion evidence.
            var loadedTouchdownMask = 0
            if lifted[e] && !newlyLifted {
                for foot in 0..<2 {
                    let inContact = loadBearingFootContacts[e][foot]
                    if inContact {
                        if !previousLoadedFootContacts[e][foot]
                            && loadedFootAirTimes[e][foot] > 0.10
                            && loadedFootMaximumClearances[e][foot]
                                >= Self.loadedSwingMinimumClearance {
                            loadedTouchdownMask |= 1 << foot
                        }
                        loadedFootAirTimes[e][foot] = 0
                        loadedFootMaximumClearances[e][foot] = 0
                    } else {
                        loadedFootAirTimes[e][foot] += dt
                        loadedFootMaximumClearances[e][foot] = max(
                            loadedFootMaximumClearances[e][foot],
                            footGroundClearances[foot])
                    }
                    previousLoadedFootContacts[e][foot] = inContact
                }
            }
            if stableUnsupported
                && (loadedTouchdownMask == 1 || loadedTouchdownMask == 2) {
                let foot = loadedTouchdownMask == 1 ? 0 : 1
                let other = 1 - foot
                let footPositions = [
                    state.leftFoot.position, state.rightFoot.position,
                ]
                let goal = destinationPregraspPosition(environment: e)
                let travelHeading = goal - state.root.position
                if HumanoidLocomotionObjective.isLeadingTouchdown(
                    touchdownFootPosition: footPositions[foot],
                    otherFootPosition: footPositions[other],
                    heading: travelHeading) {
                    loadedTouchdownCounts[e] += 1
                    if lastLoadedTouchdownFeet[e] >= 0
                        && lastLoadedTouchdownFeet[e] != foot {
                        loadedAlternatingStepCounts[e] += 1
                        rewardLoadedAlternatingStep[e] =
                            configuration.carryAlternatingStepRewardWeight
                    }
                    lastLoadedTouchdownFeet[e] = foot
                }
            } else if loadedTouchdownMask == 3 {
                lastLoadedTouchdownFeet[e] = -1
            }

            if placed[e] { phases[e] = 3 }
            else if bilateralGrasped[e] { phases[e] = 2 }
            else if approached[e] { phases[e] = 1 }
            else { phases[e] = 0 }
            updateCommand(
                environment: e, state: state,
                bilateralHandContact: bilateral)

            let localVelocity = state.root.rotation.conjugate.act(
                state.root.linearVelocity)
            let localAngular = state.root.rotation.conjugate.act(
                state.root.angularVelocity)
            let command = commands[e]
            let trackingError =
                (localVelocity.x - command.x) * (localVelocity.x - command.x)
                + (localVelocity.y - command.y) * (localVelocity.y - command.y)
            let yawError = localAngular.z - command.z
            let up = state.root.rotation.act(F3(0, 0, 1))
            let orientationCost = up.x * up.x + up.y * up.y
            let angularCost = localAngular.x * localAngular.x
                + localAngular.y * localAngular.y
            var accelerationCost: Float = 0
            for j in 0..<Self.actionDimension {
                let joint = environment.scene.joints[
                    environment.refs[e].motors[j]]
                let requestedTarget = HumanoidWalkEnv.defaultJointPositions[j]
                    + appliedActions[e * Self.actionDimension + j]
                        * HumanoidWalkEnv.actionScales[j]
                let appliedTarget = simd_clamp(
                    requestedTarget, joint.limitLo, joint.limitHi)
                let rawTorque = joint.motorStiffness
                    * (state.jointAngles[j] - appliedTarget)
                    + joint.motorDamping * state.jointVelocities[j]
                let torqueRatio = abs(rawTorque)
                    / max(joint.motorTorque, Float.leastNormalMagnitude)
                stateMaximumActuatorTorqueRatio[e] = max(
                    stateMaximumActuatorTorqueRatio[e], torqueRatio)
                if j >= 11 {
                    stateMaximumArmActuatorTorqueRatio[e] = max(
                        stateMaximumArmActuatorTorqueRatio[e], torqueRatio)
                }
                if torqueRatio >= 1 {
                    stateSaturatedActuatorCount[e] += 1
                    if j >= 11 { stateSaturatedArmActuatorCount[e] += 1 }
                }
                stateMinimumJointLimitMargin[e] = min(
                    stateMinimumJointLimitMargin[e],
                    min(
                        state.jointAngles[j] - joint.limitLo,
                        joint.limitHi - state.jointAngles[j]))
                stateMaximumRequestedTargetClamp[e] = max(
                    stateMaximumRequestedTargetClamp[e],
                    abs(requestedTarget - appliedTarget))
                let acceleration = (state.jointVelocities[j]
                    - previousJointVelocities[e][j]) / dt
                accelerationCost += acceleration * acceleration
            }
            let fallen = groundContacts.torso[e]
                || state.torso.position.z < 0.55 || up.z < 0.35
            let trackingVariance = lifted[e]
                ? configuration.carryTrackingVariance : Self.trackingVariance
            let commandTrackingReward = Self.commandTrackingReward(
                planarErrorSquared: trackingError,
                yawErrorSquared: yawError * yawError,
                variance: trackingVariance,
                coupled: lifted[e]
                    && configuration.coupledCarryCommandTracking)
            let locomotionRate =
                commandTrackingReward
                - orientationCost - 0.05 * angularCost
                - 0.005 * actionRate[e]
                - 1.25e-7 * accelerationCost
                - (fallen ? 200 : 0)
            rewardLocomotion[e] = locomotionRate * dt
                * (lifted[e]
                    ? configuration.carryLocomotionRewardMultiplier : 1)

            // Progress terms are differences of measured distances, so an
            // agent cannot earn indefinitely by hovering in one state.
            rewardApproach[e] = phases[e] == 0
                ? 6 * (previousPregraspDistances[e] - pregraspDistance) : 0
            rewardReach[e] = phases[e] == 1
                ? 10 * (previousReachDistances[e] - reachDistance)
                    + 3 * (leftHoldScore + rightHoldScore) * dt
                : 0
            rewardContact[e] = phases[e] >= 1
                ? ((handContacts.left[e] ? leftHoldScore : 0)
                    + (handContacts.right[e] ? rightHoldScore : 0)) * dt : 0
            rewardContact[e] += loadBearingGrasp
                ? Self.bilateralAcquisitionRewardWeight * dt : 0
            penaltyUnilateralContact[e] = phases[e] >= 1
                && handContacts.left[e] != handContacts.right[e]
                ? Self.unilateralContactPenaltyWeight * dt : 0
            // Signed lowest-corner clearance remains the physical lift
            // potential and success gate. Center rise supplies a smoother
            // pre-separation gradient at the contact margin, but it is paired
            // with exact-clearance progress and an ongoing upright penalty so
            // pivoting around a planted edge is not a profitable solution.
            // Both progress terms telescope to zero for a bobbing cycle.
            let clearanceProgress = boxClearance
                - previousBoxClearances[e]
            let centerHeightProgress = item.object.position.z
                - previousBoxHeights[e]
            let liftFraction = simd_clamp(
                boxClearance / max(liftClearanceTarget, 1e-6),
                0, 1)
            let carryHoldFraction = simd_clamp(
                boxClearance / max(carryHoldClearanceTarget, 1e-6),
                0, 1)
            rewardLift[e] = loadBearingGrasp && !inPlacementRegion
                ? 150 * centerHeightProgress + 300 * clearanceProgress
                    + (Self.liftHoldRewardWeight * carryHoldFraction
                        + 2 * levelGripScore
                        - 10 * (1 - levelGripScore)) * dt
                    + (physicallyClearOfPedestal ? 4 * dt : 0)
                : 0
            // A contact bit alone gives PPO no guidance once a hand slips a
            // few millimeters off a face.  Shape both hands toward their
            // physical side-face targets after lift, while retaining an
            // explicit manifold-contact bonus.  The minimum makes this a
            // genuinely two-handed hold objective: parking one hand on the
            // box cannot compensate for losing the other.
            // A stable opposing grip is also a prerequisite before lift. If
            // retention shaping begins only after the sparse lift event, PPO
            // learns brief hand contacts that disappear before an upward
            // action can transfer frictional force to the box.
            rewardRetention[e] = phases[e] >= 1
                ? ((lifted[e] ? 8 : 3) * twoHandHoldScore
                    + (loadBearingGrasp ? 4 : 0)) * dt
                : 0
            // Optimize a physical destination pose rather than a hard-coded
            // world direction. This remains a telescoping potential: backing
            // away from the receiving table repays earlier progress.
            let carryGoalDistance = length(
                carryNavigationTarget(environment: e)
                    - item.object.position)
            let destinationProgress = previousPlacementDistances[e]
                - carryGoalDistance
            rewardCarry[e] = lifted[e] && bilateral
                ? configuration.carryProgressRewardWeight
                    * destinationProgress + 1.5 * dt : 0
            let rootGoal = destinationPregraspPosition(environment: e)
            let rootGoalProgress = planarDistance(
                previousRootPositions[e], rootGoal)
                - planarDistance(state.root.position, rootGoal)
            rewardCarryRootProgress[e] = stableUnsupported && !newlyLifted
                ? configuration.carryRootProgressRewardWeight
                    * rootGoalProgress : 0
            rewardPlacement[e] = lifted[e]
                ? (destinationSupported ? 5 * dt : 0)
                    + (stablePlacement ? 8 * dt : 0)
                : 0
            // Terminal drop alone delays credit until contact recovery is no
            // longer possible. Penalize measured downward object motion after
            // lift so PPO can assign the slip to the actions that caused it.
            // This remains pure physics shaping: no target trajectory, weld,
            // teleport, or hidden grasp state is introduced.
            let downwardProgress = max(
                previousBoxHeights[e] - item.object.position.z, 0)
            // Lowering is a failure while transporting through free space but
            // is the required motion once the box is over the receiving top.
            penaltyBoxDescent[e] = lifted[e] && !inPlacementRegion
                ? 80 * downwardProgress : 0

            let stableCarry = lifted[e] && bilateral && !dropped[e]
                && carryDistance >= carryTarget
                && loadedAlternatingStepCounts[e]
                    >= configuration.minimumLoadedAlternatingSteps
            if stableCarry { carryMilestoneReached[e] = true }
            successDwellCounts[e] = stablePlacement
                ? successDwellCounts[e] + 1 : 0
            let newlyPlaced = !placed[e]
                && successDwellCounts[e] >= dwellTarget
            if newlyPlaced { placed[e] = true }
            let succeeded = placed[e]
            let boxFellBeforeLift = !lifted[e] && boxBottom < 0.10
            let reward = rewardLocomotion[e] + rewardApproach[e]
                + rewardReach[e] + rewardContact[e] + rewardLift[e]
                + rewardRetention[e] + rewardCarry[e]
                + rewardCarryRootProgress[e]
                + rewardLoadedAlternatingStep[e] + rewardPlacement[e]
                + (newlyPlaced ? 24 : 0)
                - penaltyBoxDescent[e]
                - penaltyUnilateralContact[e]
                - (dropped[e] ? 8 : 0)
                - (boxFellBeforeLift ? 4 : 0)
            result.rewards[e] = reward
            episodeReturns[e] += reward
            episodeLengths[e] += 1

            statePregraspDistance[e] = pregraspDistance
            stateReachDistance[e] = reachDistance
            stateBoxHeight[e] = item.object.position.z
            stateBoxClearance[e] = boxClearance
            stateCarryDistance[e] = carryDistance
            stateLoadedRootDisplacement[e] = loadedRootDisplacement
            stateRootDestinationDistance[e] = planarDistance(
                state.root.position,
                destinationPregraspPosition(environment: e))
            stateLoadedTouchdowns[e] = Float(loadedTouchdownCounts[e])
            stateLoadedAlternatingSteps[e] = Float(
                loadedAlternatingStepCounts[e])
            stateLeftLoadedFootAirTime[e] = loadedFootAirTimes[e][0]
            stateRightLoadedFootAirTime[e] = loadedFootAirTimes[e][1]
            stateMaximumLoadedFootAirTime[e] = max(
                loadedFootAirTimes[e][0], loadedFootAirTimes[e][1])
            let leftFootNormalLoad = groundContactNormalLoads[e][0]
            let rightFootNormalLoad = groundContactNormalLoads[e][1]
            let totalFootNormalLoad = leftFootNormalLoad
                + rightFootNormalLoad
            stateLeftFootNormalLoad[e] = leftFootNormalLoad
            stateRightFootNormalLoad[e] = rightFootNormalLoad
            stateFootUnloadingFraction[e] = totalFootNormalLoad > 1e-6
                ? 1 - min(leftFootNormalLoad, rightFootNormalLoad)
                    / totalFootNormalLoad
                : 0
            stateLeftFootGroundClearance[e] = footGroundClearances[0]
            stateRightFootGroundClearance[e] = footGroundClearances[1]
            stateMaximumLoadedSwingClearance[e] = max(
                loadedFootMaximumClearances[e][0],
                loadedFootMaximumClearances[e][1])
            stateLeftLoadBearingFootContact[e] =
                loadBearingFootContacts[e][0] ? 1 : 0
            stateRightLoadBearingFootContact[e] =
                loadBearingFootContacts[e][1] ? 1 : 0
            stateLeftFootContact[e] = groundContacts.feet[e][0] ? 1 : 0
            stateRightFootContact[e] = groundContacts.feet[e][1] ? 1 : 0
            stateLiftFraction[e] = liftFraction
            stateLeftContact[e] = handContacts.left[e] ? 1 : 0
            stateRightContact[e] = handContacts.right[e] ? 1 : 0
            stateLeftHandNormalLoad[e] =
                handContactNormalLoads.left[e]
            stateRightHandNormalLoad[e] =
                handContactNormalLoads.right[e]
            stateGraspFrictionSupportFraction[e] =
                Self.frictionGraspSupportFraction(
                    leftNormalLoad: handContactNormalLoads.left[e],
                    rightNormalLoad: handContactNormalLoads.right[e],
                    boxMass: configuration.boxMass,
                    friction: configuration.boxFriction)
            stateGraspQuality[e] = twoHandHoldScore
            stateLoadBearingGrasp[e] = loadBearingGrasp ? 1 : 0
            stateFrictionLoadBearingGrasp[e] =
                frictionLoadBearingGrasp ? 1 : 0
            statePhase[e] = Float(phases[e])
            stateBoxGroundContact[e] = supportContacts.ground[e] ? 1 : 0
            stateBoxPedestalContact[e] = supportContacts.source[e] ? 1 : 0
            stateBoxDestinationContact[e] = supportContacts.destination[e] ? 1 : 0
            statePlacementDistance[e] = placementDistance
            stateReleased[e] = released ? 1 : 0
            stateMissedContactSteps[e] = Float(missedContactCounts[e])
            stateManipulationHandoff[e] =
                manipulationHandoffProgress(environment: e)
            stateCarryHandoff[e] = carryHandoffProgress(environment: e)
            stateCarryCommandRamp[e] = carryCommandProgress(environment: e)
            stateStableUnsupportedSteps[e] = Float(
                currentStableUnsupportedSteps[e])
            taskImitationMilestone[e] = reachedImitationMilestone ? 1 : 0
            previousPregraspDistances[e] = pregraspDistance
            previousReachDistances[e] = reachDistance
            previousBoxHeights[e] = item.object.position.z
            previousBoxClearances[e] = boxClearance
            previousPlacementDistances[e] = carryGoalDistance
            previousRootPositions[e] = state.root.position

            let timedOut = episodeLengths[e] >= configuration.maxEpisodeSteps
            let failed = fallen || dropped[e] || boxFellBeforeLift
            if failed || succeeded || timedOut {
                result.terminated[e] = failed || succeeded
                result.truncated[e] = !failed && !succeeded && timedOut
                result.successes[e] = succeeded && !fallen && !dropped[e]
                result.hasFinalObservation[e] = true
                episodeReturnMetric[e] = episodeReturns[e]
                episodeLengthMetric[e] = Float(episodeLengths[e])
                episodeApproachedMetric[e] = approached[e] ? 1 : 0
                episodeGraspedMetric[e] = bilateralGrasped[e] ? 1 : 0
                episodeLiftedMetric[e] = lifted[e] ? 1 : 0
                episodeCarriedMetric[e] = carryMilestoneReached[e] ? 1 : 0
                episodePlacedMetric[e] = succeeded ? 1 : 0
                episodeDroppedMetric[e] = dropped[e] ? 1 : 0
                episodeSurvivedMetric[e] = fallen ? 0 : 1
                episodeFinalCarryDistanceMetric[e] = carryDistance
                episodeMaximumCarryDistanceMetric[e] = maximumCarryDistances[e]
                episodeMaximumLoadedRootDisplacementMetric[e] =
                    maximumLoadedRootDisplacements[e]
                episodeLoadedTouchdownsMetric[e] = Float(
                    loadedTouchdownCounts[e])
                episodeLoadedAlternatingStepsMetric[e] = Float(
                    loadedAlternatingStepCounts[e])
                episodeMaximumBoxHeightMetric[e] = maximumBoxHeights[e]
                episodeMaximumBoxClearanceMetric[e] =
                    maximumBoxClearances[e]
                episodeMaximumStableUnsupportedStepsMetric[e] = Float(
                    maximumStableUnsupportedSteps[e])
                if configuration.autoReset {
                    resetIDs.append(e)
                    resetSeeds.append(resetRNG.next())
                    if trainingMode,
                       configuration.carryStartReplayProbability > 0,
                       carryStartSnapshots[e] != nil,
                       resetRNG.nextFloat()
                            < configuration.carryStartReplayProbability {
                        carryStartResetIDs.append(e)
                        referenceStateReset[e] = 1
                    }
                }
            }
        }

        if trainingMode, configuration.carryStartReplayProbability > 0,
           !newCarryStartIDs.isEmpty {
            captureCarryStartSnapshots(
                newCarryStartIDs, actions: actions.values)
        }
        if trainingMode,
           configuration.advanceReplaySnapshotAtDestinationContact,
           !newDestinationStartIDs.isEmpty {
            captureCarryStartSnapshots(
                newDestinationStartIDs, actions: actions.values)
        }

        for e in 0..<n {
            previousJointVelocities[e] = states[e].jointVelocities
            for j in 0..<Self.actionDimension {
                previousActions[e * Self.actionDimension + j] =
                    actions.values[e * Self.actionDimension + j]
            }
        }
        fillObservations(
            states, manipulation: manipulation, contacts: handContacts,
            into: &result.observations.policy)
        for e in 0..<n where result.hasFinalObservation[e] {
            let base = e * Self.observationDimension
            for j in 0..<Self.observationDimension {
                result.finalObservations[base + j] =
                    result.observations.policy[base + j]
            }
        }

        result.metrics["reward/locomotion"] = rewardLocomotion
        result.metrics["reward/approach_progress"] = rewardApproach
        result.metrics["reward/reach_progress"] = rewardReach
        result.metrics["reward/hand_contact"] = rewardContact
        result.metrics["reward/lift"] = rewardLift
        result.metrics["reward/grip_retention"] = rewardRetention
        result.metrics["reward/carry"] = rewardCarry
        result.metrics["reward/carry_root_progress"] =
            rewardCarryRootProgress
        result.metrics["reward/loaded_alternating_step"] =
            rewardLoadedAlternatingStep
        result.metrics["reward/placement"] = rewardPlacement
        result.metrics["penalty/box_descent"] = penaltyBoxDescent
        result.metrics["penalty/unilateral_contact"] =
            penaltyUnilateralContact
        result.metrics["state/pregrasp_distance_m"] = statePregraspDistance
        result.metrics["state/reach_distance_m"] = stateReachDistance
        result.metrics["state/box_height_m"] = stateBoxHeight
        result.metrics["state/box_clearance_m"] = stateBoxClearance
        result.metrics["state/carry_distance_m"] = stateCarryDistance
        result.metrics["state/loaded_root_displacement_m"] =
            stateLoadedRootDisplacement
        result.metrics["state/root_destination_distance_m"] =
            stateRootDestinationDistance
        result.metrics["state/loaded_touchdowns"] = stateLoadedTouchdowns
        result.metrics["state/loaded_alternating_steps"] =
            stateLoadedAlternatingSteps
        result.metrics["state/left_loaded_foot_air_time_s"] =
            stateLeftLoadedFootAirTime
        result.metrics["state/right_loaded_foot_air_time_s"] =
            stateRightLoadedFootAirTime
        result.metrics["state/maximum_loaded_foot_air_time_s"] =
            stateMaximumLoadedFootAirTime
        result.metrics["state/left_foot_normal_load"] =
            stateLeftFootNormalLoad
        result.metrics["state/right_foot_normal_load"] =
            stateRightFootNormalLoad
        result.metrics["state/foot_unloading_fraction"] =
            stateFootUnloadingFraction
        result.metrics["state/left_foot_ground_clearance_m"] =
            stateLeftFootGroundClearance
        result.metrics["state/right_foot_ground_clearance_m"] =
            stateRightFootGroundClearance
        result.metrics["state/maximum_loaded_swing_clearance_m"] =
            stateMaximumLoadedSwingClearance
        result.metrics["state/left_load_bearing_foot_contact"] =
            stateLeftLoadBearingFootContact
        result.metrics["state/right_load_bearing_foot_contact"] =
            stateRightLoadBearingFootContact
        result.metrics["state/left_foot_contact"] = stateLeftFootContact
        result.metrics["state/right_foot_contact"] = stateRightFootContact
        result.metrics["state/lift_fraction"] = stateLiftFraction
        result.metrics["state/left_hand_contact"] = stateLeftContact
        result.metrics["state/right_hand_contact"] = stateRightContact
        result.metrics["state/left_hand_normal_load"] =
            stateLeftHandNormalLoad
        result.metrics["state/right_hand_normal_load"] =
            stateRightHandNormalLoad
        result.metrics["state/grasp_friction_support_fraction"] =
            stateGraspFrictionSupportFraction
        result.metrics["state/grasp_quality"] = stateGraspQuality
        result.metrics["state/load_bearing_grasp"] = stateLoadBearingGrasp
        result.metrics["state/friction_load_bearing_grasp"] =
            stateFrictionLoadBearingGrasp
        result.metrics["state/task_phase"] = statePhase
        result.metrics["state/box_ground_contact"] = stateBoxGroundContact
        result.metrics["state/box_pedestal_contact"] = stateBoxPedestalContact
        result.metrics["state/box_destination_contact"] =
            stateBoxDestinationContact
        result.metrics["state/placement_distance_m"] = statePlacementDistance
        result.metrics["state/released"] = stateReleased
        result.metrics["state/missed_bilateral_contact_steps"] =
            stateMissedContactSteps
        result.metrics["state/manipulation_handoff_progress"] =
            stateManipulationHandoff
        result.metrics["state/carry_handoff_progress"] = stateCarryHandoff
        result.metrics["state/carry_command_progress"] = stateCarryCommandRamp
        result.metrics["state/stable_unsupported_steps"] =
            stateStableUnsupportedSteps
        result.metrics["state/maximum_actuator_torque_ratio"] =
            stateMaximumActuatorTorqueRatio
        result.metrics["state/maximum_arm_actuator_torque_ratio"] =
            stateMaximumArmActuatorTorqueRatio
        result.metrics["state/saturated_actuator_count"] =
            stateSaturatedActuatorCount
        result.metrics["state/saturated_arm_actuator_count"] =
            stateSaturatedArmActuatorCount
        result.metrics["state/minimum_joint_limit_margin_rad"] =
            stateMinimumJointLimitMargin
        result.metrics["state/maximum_requested_target_clamp_rad"] =
            stateMaximumRequestedTargetClamp
        result.metrics["task/imitation_milestone"] = taskImitationMilestone
        result.metrics["task/reference_state_reset"] = referenceStateReset
        result.metrics["curriculum/carry_target_m"] = ContiguousArray(
            repeating: carryTarget, count: n)
        result.metrics["curriculum/lift_clearance_m"] = ContiguousArray(
            repeating: liftClearanceTarget, count: n)
        result.metrics["curriculum/carry_hold_clearance_m"] = ContiguousArray(
            repeating: carryHoldClearanceTarget, count: n)
        result.metrics["curriculum/success_dwell_steps"] = ContiguousArray(
            repeating: Float(dwellTarget), count: n)
        result.metrics["episode/return"] = episodeReturnMetric
        result.metrics["episode/length"] = episodeLengthMetric
        result.metrics["episode/approached"] = episodeApproachedMetric
        result.metrics["episode/bilateral_grasped"] = episodeGraspedMetric
        result.metrics["episode/lifted"] = episodeLiftedMetric
        result.metrics["episode/carried"] = episodeCarriedMetric
        result.metrics["episode/placed"] = episodePlacedMetric
        result.metrics["episode/dropped"] = episodeDroppedMetric
        result.metrics["episode/survived"] = episodeSurvivedMetric
        result.metrics["episode/final_carry_distance_m"] =
            episodeFinalCarryDistanceMetric
        result.metrics["episode/maximum_carry_distance_m"] =
            episodeMaximumCarryDistanceMetric
        result.metrics["episode/maximum_loaded_root_displacement_m"] =
            episodeMaximumLoadedRootDisplacementMetric
        result.metrics["episode/loaded_touchdowns"] =
            episodeLoadedTouchdownsMetric
        result.metrics["episode/loaded_alternating_steps"] =
            episodeLoadedAlternatingStepsMetric
        result.metrics["episode/maximum_box_height_m"] =
            episodeMaximumBoxHeightMetric
        result.metrics["episode/maximum_box_clearance_m"] =
            episodeMaximumBoxClearanceMetric
        result.metrics["episode/maximum_stable_unsupported_steps"] =
            episodeMaximumStableUnsupportedStepsMetric

        if !resetIDs.isEmpty {
            environment.reset(resetIDs, seeds: resetSeeds,
                              initialRollPitchRange: 0, initialYawRange: 0)
            states = environment.states()
            initializeEpisodes(resetIDs, states: states, seeds: resetSeeds)
            if !carryStartResetIDs.isEmpty {
                restoreCarryStartSnapshots(carryStartResetIDs)
                states = environment.states()
            }
            manipulation = environment.manipulationStates()
            fillObservations(
                states, manipulation: manipulation,
                contacts: latestHandContacts,
                into: &result.observations.policy)
        }
        try result.observations.validate(for: spec)
    }

    private func captureCarryStartSnapshots(
        _ ids: [Int], actions: ContiguousArray<Float>
    ) {
        precondition(actions.count
            == spec.numEnvironments * Self.actionDimension)
        let uniqueIDs = Array(Set(ids)).sorted()
        var bodyIDs = [Int]()
        var bodyCounts = [Int]()
        for e in uniqueIDs {
            var ids = environment.refs[e].bodies
            guard let box = environment.refs[e].projectile else {
                preconditionFailure("box-carry environment has no box")
            }
            ids.append(box)
            bodyIDs.append(contentsOf: ids)
            bodyCounts.append(ids.count)
        }
        let states = environment.solver.bodyStates(bodyIDs)
        var stateOffset = 0
        for (offset, e) in uniqueIDs.enumerated() {
            let count = bodyCounts[offset]
            let actionBase = e * Self.actionDimension
            carryStartSnapshots[e] = CarryStartSnapshot(
                bodyStates: Array(states[stateOffset..<(stateOffset + count)]),
                previousAction: Array(
                    actions[actionBase..<(actionBase + Self.actionDimension)]),
                stationDistance: stationDistances[e],
                stationCenter: stationCenters[e],
                destinationPedestalCenter: destinationPedestalCenters[e],
                destinationBoxTarget: destinationBoxTargets[e],
                destinationContact: latestDestinationContacts[e])
            stateOffset += count
        }
    }

    /// Restore only at an episode reset. `environment.reset` and
    /// `initializeEpisodes` run first to clear all temporal task and solver
    /// state; this method then authors a previously observed physical state
    /// and matching controller targets before the first policy observation.
    private func restoreCarryStartSnapshots(_ ids: [Int]) {
        let uniqueIDs = Array(Set(ids)).sorted()
        var pedestalCenters = [F3]()
        var destinationPedestals = [F3]()
        var boxCenters = [F3]()
        pedestalCenters.reserveCapacity(uniqueIDs.count)
        destinationPedestals.reserveCapacity(uniqueIDs.count)
        boxCenters.reserveCapacity(uniqueIDs.count)
        for e in uniqueIDs {
            guard let snapshot = carryStartSnapshots[e],
                  let box = snapshot.bodyStates.last else {
                preconditionFailure("missing carry-start snapshot")
            }
            pedestalCenters.append(snapshot.stationCenter)
            destinationPedestals.append(snapshot.destinationPedestalCenter)
            boxCenters.append(box.position)
        }
        environment.placeCarryStations(
            environmentIDs: uniqueIDs,
            pedestalCenters: pedestalCenters,
            boxCenters: boxCenters,
            destinationPedestalCenters: destinationPedestals)

        var bodyUpdates = [GPUSolver.BodyStateUpdate]()
        var motorTargets = [GPUSolver.MotorTargetUpdate]()
        for e in uniqueIDs {
            guard let snapshot = carryStartSnapshots[e],
                  let boxID = environment.refs[e].projectile else {
                preconditionFailure("missing carry-start snapshot")
            }
            let bodyIDs = environment.refs[e].bodies + [boxID]
            precondition(bodyIDs.count == snapshot.bodyStates.count)
            for (body, state) in zip(bodyIDs, snapshot.bodyStates) {
                bodyUpdates.append(.init(
                    body: body, position: state.position,
                    rotation: state.rotation,
                    linearVelocity: state.linearVelocity,
                    angularVelocity: state.angularVelocity))
            }
            for j in 0..<Self.actionDimension {
                let jointID = environment.refs[e].motors[j]
                var action = snapshot.previousAction[j]
                if j >= 11 {
                    action *= configuration.manipulationArmActionScaleMultiplier
                }
                let requested = HumanoidWalkEnv.defaultJointPositions[j]
                    + action * HumanoidWalkEnv.actionScales[j]
                let joint = environment.scene.joints[jointID]
                motorTargets.append(.init(
                    joint: jointID,
                    angle: simd_clamp(
                        requested, joint.limitLo, joint.limitHi)))
            }
        }
        environment.solver.setBodyStates(bodyUpdates)
        environment.solver.setMotorTargets(motorTargets)

        let states = environment.states()
        let manipulation = environment.manipulationStates()
        for e in uniqueIDs {
            guard let snapshot = carryStartSnapshots[e] else {
                preconditionFailure("missing carry-start snapshot")
            }
            stationDistances[e] = snapshot.stationDistance
            stationCenters[e] = snapshot.stationCenter
            destinationPedestalCenters[e] =
                snapshot.destinationPedestalCenter
            destinationBoxTargets[e] = snapshot.destinationBoxTarget
            let item = manipulation[e]
            let targets = handTargets(item.object)
            previousPregraspDistances[e] = planarDistance(
                states[e].root.position, pregraspPosition(environment: e))
            previousReachDistances[e] =
                length(item.leftHand.position - targets.0)
                + length(item.rightHand.position - targets.1)
            previousBoxHeights[e] = item.object.position.z
            previousBoxClearances[e] = orientedBoxBottom(item.object)
                - Self.pedestalHeight
            previousPlacementDistances[e] = length(
                carryNavigationTarget(environment: e)
                    - item.object.position)
            previousRootPositions[e] = states[e].root.position
            maximumBoxHeights[e] = item.object.position.z
            maximumBoxClearances[e] = previousBoxClearances[e]
            maximumCarryDistances[e] = 0
            maximumLoadedRootDisplacements[e] = 0
            currentStableUnsupportedSteps[e] = 0
            maximumStableUnsupportedSteps[e] = 0
            liftOrigins[e] = item.object.position
            liftRootOrigins[e] = states[e].root.position
            loadedFootAirTimes[e] = [0, 0]
            loadedFootMaximumClearances[e] = [0, 0]
            previousLoadedFootContacts[e] = [
                HumanoidWalkTask.footInContact(states[e].leftFoot),
                HumanoidWalkTask.footInContact(states[e].rightFoot),
            ]
            lastLoadedTouchdownFeet[e] = -1
            loadedTouchdownCounts[e] = 0
            loadedAlternatingStepCounts[e] = 0
            approached[e] = true
            approachSettleCounts[e] = Self.manipulationEntryDwellSteps
            manipulationHandoffCounts[e] =
                configuration.manipulationHandoffSteps
            bilateralGrasped[e] = true
            lifted[e] = true
            dropped[e] = false
            carryHandoffCounts[e] = 0
            successDwellCounts[e] = 0
            carryMilestoneReached[e] = false
            placed[e] = false
            missedContactCounts[e] = 0
            phases[e] = 2
            episodeLengths[e] = 0
            episodeReturns[e] = 0
            previousJointVelocities[e] = states[e].jointVelocities
            latestHandContacts.left[e] = true
            latestHandContacts.right[e] = true
            latestDestinationContacts[e] = snapshot.destinationContact
            let base = e * Self.actionDimension
            for j in 0..<Self.actionDimension {
                previousActions[base + j] = snapshot.previousAction[j]
            }
            updateCommand(
                environment: e, state: states[e],
                bilateralHandContact: true)
        }
    }

    private func initializeEpisodes(_ ids: [Int], states: [HumanoidState],
                                    seeds: [UInt64]) {
        precondition(ids.count == seeds.count)
        var pedestalCenters = [F3]()
        var receivingPedestalCenters = [F3]()
        var boxCenters = [F3]()
        pedestalCenters.reserveCapacity(ids.count)
        receivingPedestalCenters.reserveCapacity(ids.count)
        boxCenters.reserveCapacity(ids.count)
        let progress = curriculumProgress
        for (offset, e) in ids.enumerated() {
            commandRNGs[e] = SplitMix64(seed: seeds[offset]
                ^ 0xA0761D6478BD642F)
            noiseRNGs[e] = SplitMix64(seed: seeds[offset]
                ^ 0xE7037ED1A0B428DB)
            let nominal = configuration.minimumTrainingStationDistance
                + progress * (configuration.evaluationStationDistance
                    - configuration.minimumTrainingStationDistance)
            let jitter: Float = trainingMode
                ? (2 * commandRNGs[e].nextFloat() - 1) * 0.06 : 0
            let distance = simd_clamp(
                nominal + jitter,
                configuration.minimumTrainingStationDistance,
                configuration.evaluationStationDistance)
            stationDistances[e] = distance
            stationCenters[e] = F3(
                distance + Self.tabletopForwardOffset, 0,
                Self.tabletopCenterHeight)
            destinationBoxTargets[e] = F3(
                distance, configuration.carryDistance,
                Self.boxRestingHeight)
            destinationPedestalCenters[e] = destinationBoxTargets[e]
                + F3(Self.tabletopForwardOffset, 0,
                     Self.tabletopCenterHeight - Self.boxRestingHeight)
            pedestalCenters.append(stationCenters[e])
            receivingPedestalCenters.append(destinationPedestalCenters[e])
            boxCenters.append(F3(distance, 0, Self.boxRestingHeight))
        }
        environment.placeCarryStations(
            environmentIDs: ids, pedestalCenters: pedestalCenters,
            boxCenters: boxCenters,
            destinationPedestalCenters: receivingPedestalCenters)
        let manipulation = environment.manipulationStates()
        for e in ids {
            let pregraspDistance = planarDistance(
                states[e].root.position, pregraspPosition(environment: e))
            let targets = handTargets(manipulation[e].object)
            previousPregraspDistances[e] = pregraspDistance
            previousReachDistances[e] =
                length(manipulation[e].leftHand.position - targets.0)
                + length(manipulation[e].rightHand.position - targets.1)
            previousBoxHeights[e] = manipulation[e].object.position.z
            previousBoxClearances[e] = orientedBoxBottom(
                manipulation[e].object) - Self.pedestalHeight
            previousPlacementDistances[e] = length(
                carryNavigationTarget(environment: e)
                    - manipulation[e].object.position)
            previousRootPositions[e] = states[e].root.position
            maximumBoxHeights[e] = manipulation[e].object.position.z
            maximumBoxClearances[e] = previousBoxClearances[e]
            maximumCarryDistances[e] = 0
            maximumLoadedRootDisplacements[e] = 0
            currentStableUnsupportedSteps[e] = 0
            maximumStableUnsupportedSteps[e] = 0
            liftOrigins[e] = manipulation[e].object.position
            liftRootOrigins[e] = states[e].root.position
            loadedFootAirTimes[e] = [0, 0]
            loadedFootMaximumClearances[e] = [0, 0]
            previousLoadedFootContacts[e] = [
                HumanoidWalkTask.footInContact(states[e].leftFoot),
                HumanoidWalkTask.footInContact(states[e].rightFoot),
            ]
            lastLoadedTouchdownFeet[e] = -1
            loadedTouchdownCounts[e] = 0
            loadedAlternatingStepCounts[e] = 0
            approached[e] = pregraspDistance
                <= Self.manipulationEntryDistance
                && sqrt(
                    states[e].root.linearVelocity.x
                        * states[e].root.linearVelocity.x
                        + states[e].root.linearVelocity.y
                            * states[e].root.linearVelocity.y)
                    <= Self.manipulationEntrySpeed
            approachSettleCounts[e] = approached[e]
                ? Self.manipulationEntryDwellSteps : 0
            manipulationHandoffCounts[e] = approached[e]
                ? configuration.manipulationHandoffSteps : 0
            bilateralGrasped[e] = false
            lifted[e] = false
            dropped[e] = false
            carryHandoffCounts[e] = 0
            successDwellCounts[e] = 0
            carryMilestoneReached[e] = false
            placed[e] = false
            missedContactCounts[e] = 0
            phases[e] = approached[e] ? 1 : 0
            episodeLengths[e] = 0
            episodeReturns[e] = 0
            previousJointVelocities[e] = states[e].jointVelocities
            latestHandContacts.left[e] = false
            latestHandContacts.right[e] = false
            latestDestinationContacts[e] = false
            for j in 0..<Self.actionDimension {
                previousActions[e * Self.actionDimension + j] = 0
            }
            updateCommand(
                environment: e, state: states[e],
                bilateralHandContact: false)
        }
    }

    private func updateCommand(
        environment e: Int, state: HumanoidState,
        bilateralHandContact: Bool
    ) {
        if phases[e] == 0 {
            let navigation = PointGoalNavigator.command(
                worldGoal: pregraspPosition(environment: e),
                bodyPosition: state.root.position,
                bodyRotation: state.root.rotation,
                parameters: PointGoalNavigationParameters(
                    goalRadius: 0.05, slowdownDistance: 0.35,
                    cruiseSpeed: configuration.approachCommandSpeed,
                    boundarySpeed: 0.05, yawGain: 0.5,
                    maximumYawRate: 1, mode: .forwardOnlyYaw))
            commands[e] = navigation.bodyTwist
        } else if phases[e] == 2 && lifted[e] && bilateralHandContact {
            // Navigate to a measured placement stance in front of the second
            // table. The same point-goal controller used by locomotion slows
            // the command continuously near the target; task logic never
            // moves the box or robot kinematically.
            let holonomic = configuration.carryHolonomicCommand
            let navigation = PointGoalNavigator.command(
                worldGoal: destinationPregraspPosition(environment: e),
                bodyPosition: state.root.position,
                bodyRotation: state.root.rotation,
                parameters: PointGoalNavigationParameters(
                    goalRadius: 0.05, slowdownDistance: 0.35,
                    cruiseSpeed: configuration.carryCommandSpeed,
                    boundarySpeed: 0.03, yawGain: 0.5,
                    maximumYawRate: holonomic ? 0 : 1,
                    forwardAlignmentSpeedExponent: 2,
                    minimumForwardAlignmentScale: 0.25,
                    mode: holonomic
                        ? .projectedBodyPlane : .forwardOnlyYaw))
            commands[e] = navigation.bodyTwist
                * carryCommandProgress(environment: e)
        } else {
            commands[e] = .zero
        }
    }

    private func pregraspPosition(environment e: Int) -> F3 {
        F3(stationDistances[e] - configuration.pregraspForwardOffset,
           configuration.pregraspLateralOffset, 0)
    }

    private func destinationPregraspPosition(environment e: Int) -> F3 {
        // The H1 starts facing +X, so +Y is its anatomical left. Approach the
        // receiving table from its open -Y side instead of walking through the
        // source table or placing the destination behind it.
        if !trainingMode
            || configuration.destinationBearingCurriculumControlSteps == 0 {
            return F3(destinationBoxTargets[e].x,
                      destinationBoxTargets[e].y - Self.pregraspOffset, 0)
        }
        let source = F3(stationDistances[e], 0, 0)
        let navigationTarget = carryNavigationTarget(environment: e)
        let target = F3(
            navigationTarget.x, navigationTarget.y, 0)
        let delta = target - source
        let distance = max(length(delta), 1e-6)
        return target - delta / distance * Self.pregraspOffset
    }

    public func currentPlacementTarget(environment e: Int) -> F3 {
        precondition((0..<spec.numEnvironments).contains(e))
        return destinationBoxTargets[e]
    }

    public func currentCarryNavigationTarget(environment e: Int) -> F3 {
        precondition((0..<spec.numEnvironments).contains(e))
        return carryNavigationTarget(environment: e)
    }

    private func carryHandoffProgress(environment e: Int) -> Float {
        simd_clamp(
            Float(carryHandoffCounts[e])
                / Float(configuration.carryHandoffSteps),
            0, 1)
    }

    private func manipulationHandoffProgress(environment e: Int) -> Float {
        simd_clamp(
            Float(manipulationHandoffCounts[e])
                / Float(configuration.manipulationHandoffSteps),
            0, 1)
    }

    private func carryCommandProgress(environment e: Int) -> Float {
        simd_clamp(
            Float(carryHandoffCounts[e] - configuration.carryHandoffSteps)
                / Float(configuration.carryCommandRampSteps),
            0, 1)
    }

    private func handTargets(
        _ box: GPUSolver.RigidBodyState
    ) -> (F3, F3) {
        let faceOffset = 0.5 * Self.boxDimensions.y
            + Self.handCollisionSphereRadius
        let left = box.position + box.rotation.act(F3(0, faceOffset, 0))
        let right = box.position + box.rotation.act(F3(0, -faceOffset, 0))
        return (left, right)
    }

    private func fillObservations(
        _ states: [HumanoidState],
        manipulation: [HumanoidManipulationState],
        contacts: (left: [Bool], right: [Bool]),
        into output: inout ContiguousArray<Float>
    ) {
        for e in 0..<spec.numEnvironments {
            let state = states[e]
            let item = manipulation[e]
            let inverseRoot = state.root.rotation.conjugate
            let localLinear = inverseRoot.act(state.root.linearVelocity)
            let localAngular = inverseRoot.act(state.root.angularVelocity)
            let gravity = inverseRoot.act(F3(0, 0, -1))
            let base = e * Self.observationDimension
            let noiseEnabled = trainingMode && configuration.observationNoise
            func noise(_ amplitude: Float) -> Float {
                guard noiseEnabled else { return 0 }
                return (2 * noiseRNGs[e].nextFloat() - 1) * amplitude
            }
            output[base] = localLinear.x + noise(0.1)
            output[base + 1] = localLinear.y + noise(0.1)
            output[base + 2] = localLinear.z + noise(0.1)
            output[base + 3] = localAngular.x + noise(0.2)
            output[base + 4] = localAngular.y + noise(0.2)
            output[base + 5] = localAngular.z + noise(0.2)
            output[base + 6] = gravity.x + noise(0.05)
            output[base + 7] = gravity.y + noise(0.05)
            output[base + 8] = gravity.z + noise(0.05)
            output[base + 9] = commands[e].x
            output[base + 10] = commands[e].y
            output[base + 11] = commands[e].z
            for j in 0..<Self.actionDimension {
                output[base + 12 + j] = state.jointAngles[j] + noise(0.01)
                output[base + 31 + j] = state.jointVelocities[j] + noise(1.5)
                output[base + 50 + j] = previousActions[
                    e * Self.actionDimension + j]
            }
            let localBox = inverseRoot.act(
                item.object.position - state.root.position)
            let localBoxVelocity = inverseRoot.act(item.object.linearVelocity)
            let localLeftHand = inverseRoot.act(
                item.leftHand.position - item.object.position)
            let localRightHand = inverseRoot.act(
                item.rightHand.position - item.object.position)
            let localBoxUp = inverseRoot.act(
                item.object.rotation.act(F3(0, 0, 1)))
            output[base + 69] = localBox.x / 2
            output[base + 70] = localBox.y / 2
            output[base + 71] = localBox.z / 1.5
            output[base + 72] = localBoxVelocity.x / 2
            output[base + 73] = localBoxVelocity.y / 2
            output[base + 74] = localBoxVelocity.z / 2
            output[base + 75] = localLeftHand.x
            output[base + 76] = localLeftHand.y
            output[base + 77] = localLeftHand.z
            output[base + 78] = localRightHand.x
            output[base + 79] = localRightHand.y
            output[base + 80] = localRightHand.z
            output[base + 81] = localBoxUp.x
            output[base + 82] = localBoxUp.y
            output[base + 83] = localBoxUp.z
            output[base + 84] = contacts.left[e] ? 1 : 0
            output[base + 85] = contacts.right[e] ? 1 : 0
            output[base + 86] = phases[e] == 0 ? 1 : 0
            // Cumulative phase coding preserves the learned reach behavior
            // when grasp adds the lift bit: phase 2 is [manipulate=1,
            // grasped=1], not a discontinuous switch to an unseen one-hot.
            output[base + 87] = phases[e] >= 1 ? 1 : 0
            output[base + 88] = phases[e] == 2 ? 1 : 0
            // This measured milestone is distinct from bilateral grasp. It
            // routes only a physically lifted box to carry control.
            output[base + 89] = lifted[e] ? 1 : 0
            // Continuous and fully observed expert-composition gate. The
            // physical lift bit above remains a separate measured input.
            output[base + 90] = carryHandoffProgress(environment: e)
            let worldBoxUp = item.object.rotation.act(F3(0, 0, 1))
            let (leftFaceTarget, rightFaceTarget) = handTargets(item.object)
            let checkpointFaceHint = contacts.left[e] && contacts.right[e]
                && length(item.leftHand.position - leftFaceTarget)
                    <= Self.checkpointFaceHintDistance
                && length(item.rightHand.position - rightFaceTarget)
                    <= Self.checkpointFaceHintDistance
                && worldBoxUp.z >= Self.loadBearingBoxUprightAlignment
            // Do not use this compatibility channel as physical evidence.
            // Current load-bearing state is emitted in the metrics and is
            // fully derivable from local hands 75...80 plus contacts 84...85.
            output[base + 91] = checkpointFaceHint ? 1 : 0
            // Expose signed progress toward the current physical curriculum
            // target. The source policy receives a zero-weight appended input,
            // so transfer remains behavior-identical before fine-tuning.
            output[base + 92] = simd_clamp(
                orientedBoxBottom(item.object) - Self.pedestalHeight,
                -currentLiftClearance, 2 * currentLiftClearance)
                / max(currentLiftClearance, 1e-6)
            let localDestination = inverseRoot.act(
                destinationBoxTargets[e] - state.root.position)
            let localPlacementDelta = inverseRoot.act(
                destinationBoxTargets[e] - item.object.position)
            output[base + 93] = localDestination.x / 2
            output[base + 94] = localDestination.y / 2
            output[base + 95] = localDestination.z / 1.5
            output[base + 96] = localPlacementDelta.x / 1.0
            output[base + 97] = localPlacementDelta.y / 1.0
            output[base + 98] = localPlacementDelta.z / 0.5
            output[base + 99] = latestDestinationContacts[e] ? 1 : 0
            output[base + 100] = carryMilestoneReached[e] ? 1 : 0
            output[base + 101] = (!contacts.left[e] && !contacts.right[e]) ? 1 : 0
            output[base + 102] = manipulationHandoffProgress(environment: e)
            // Keep the point-goal locomotion contract explicit instead of
            // aliasing its source channels 69...70 onto local box position.
            // The imported base legs are routed into the whole-body policy
            // only during physical carry, so the associated goal input uses
            // the same boundary. Manipulation sees an exact zero rather than
            // a phase-inactive target that would perturb its composed expert.
            let navigationGoal: F3
            if phases[e] == 2 && lifted[e]
                    && contacts.left[e] && contacts.right[e] {
                navigationGoal = destinationPregraspPosition(environment: e)
            } else {
                navigationGoal = state.root.position
            }
            let localNavigationGoal = inverseRoot.act(
                navigationGoal - state.root.position)
            output[base + 103] = localNavigationGoal.x
                / configuration.navigationGoalObservationScale
            output[base + 104] = localNavigationGoal.y
                / configuration.navigationGoalObservationScale
        }
    }

    public func initializationObservationSourceIndices(
        sourceDimension: Int
    ) -> [Int?]? {
        switch sourceDimension {
        case HumanoidIsaacVelocityTask.observationDimension,
             89, 90, 91, 93, 102, 103:
            return (0..<sourceDimension).map(Optional.some)
                + [Int?](repeating: nil,
                         count: Self.observationDimension - sourceDimension)
        case HumanoidIsaacVelocityTask.goalObservationDimension:
            // Carry channels 69...70 are object state, while 103...104 are the
            // semantically equivalent local navigation goal. Route the
            // imported point-goal weights there and leave all manipulation
            // channels zero-initialized.
            var mapping = [Int?](
                repeating: nil, count: Self.observationDimension)
            for index in 0..<HumanoidIsaacVelocityTask.observationDimension {
                mapping[index] = index
            }
            mapping[103] = 69
            mapping[104] = 70
            return mapping
        default:
            return nil
        }
    }

    public func policyReferenceRegularizationWeights(
        _ observations: ContiguousArray<Float>
    ) -> ContiguousArray<Float> {
        precondition(observations.count
            == spec.numEnvironments * Self.observationDimension)
        // Action-wise retention below keeps the complete verified behavior
        // before lift and only its load-bearing arm outputs during carry.
        return ContiguousArray(repeating: 1, count: spec.numEnvironments)
    }

    public func policyReferenceActionRegularizationWeights(
        _ observations: ContiguousArray<Float>,
        actionDimension: Int
    ) -> ContiguousArray<Float> {
        precondition(actionDimension == Self.actionDimension)
        precondition(observations.count
            == spec.numEnvironments * Self.observationDimension)
        var weights = ContiguousArray(
            repeating: Float(1),
            count: spec.numEnvironments * Self.actionDimension)
        for environment in 0..<spec.numEnvironments
            where observations[
                environment * Self.observationDimension + 89] > 0.5 {
            // The published intervention split treats H1's waist as upper
            // body. In compositional mode only the ten leg actions adapt;
            // legacy whole-body carry retains the historical torso split.
            let actionBase = environment * Self.actionDimension
            let lowerBodyActionCount = configuration
                    .compositionalCarryController
                || (configuration.upperBodyCarryController
                    && !configuration.carryLocomotionControlsTorso) ? 10 : 11
            for action in 0..<lowerBodyActionCount {
                weights[actionBase + action] = 0
            }
            for action in lowerBodyActionCount..<Self.actionDimension {
                weights[actionBase + action] =
                    configuration.carryArmReferenceWeight
            }
        }
        return weights
    }

    public func policyActorTrainingWeights(
        _ observations: ContiguousArray<Float>
    ) -> ContiguousArray<Float> {
        precondition(observations.count
            == spec.numEnvironments * Self.observationDimension)
        return ContiguousArray((0..<spec.numEnvironments).map { e in
            let base = e * Self.observationDimension
            // The base locomotion actor is independently frozen by the task.
            // Once the measured approach phase begins, the manipulation
            // expert must be trainable before contact; otherwise PPO can see
            // dense reach progress but receives zero actor gradient until it
            // has somehow already discovered a load-bearing grasp.
            return observations[base + 87] > 0.5 ? 1 : 0
        })
    }

    public var usesPolicyExpertGate: Bool { true }
    public var freezesBasePolicyExpert: Bool {
        configuration.freezeBasePolicyExpert
    }
    public var initializesPolicyExpertFromBaseOnTransfer: Bool {
        configuration.initializeManipulationExpertFromBaseOnTransfer
    }
    public var initializesPolicyExpertFromMirroredBaseOnTransfer: Bool { false }
    public var policyExpertActionMask: ContiguousArray<Float>? {
        guard configuration.compositionalCarryController else { return nil }
        // Published H1 intervention controllers group waist plus arms as upper
        // body; the loaded-locomotion branch owns only the ten leg joints.
        return ContiguousArray(
            [Float](repeating: 0, count: 10)
                + [Float](repeating: 1, count: 9))
    }

    public var usesPolicyStandExpertGate: Bool { true }
    public var freezesLowSpeedPolicyExpert: Bool {
        configuration.freezeManipulationPolicyExpert
    }
    public var initializesPolicyStandExpertFromPolicyExpertOnTransfer: Bool {
        configuration.initializeCarryExpertFromManipulationExpertOnTransfer
    }
    public var initializesPolicyStandExpertFromBaseOnTransfer: Bool {
        configuration.initializeCarryExpertFromBaseOnTransfer
    }

    /// H1 actions 0...10 are the two legs plus torso; 11...18 are the two
    /// four-DoF arms. Carry keeps a configurable frozen-walker contribution
    /// below the waist while the specialist retains full control of both arms.
    public var policyStandExpertActionMask: ContiguousArray<Float>? {
        if configuration.upperBodyCarryController {
            // Preserve the successful v129 load-bearing controller on waist
            // and arms. The independent auxiliary branch below owns the ten
            // leg actions, so loaded locomotion can adapt without changing the
            // grasp controller or the verified base walker.
            let locomotionActionCount = configuration
                    .carryLocomotionControlsTorso ? 11 : 10
            return ContiguousArray(
                [Float](repeating: 0, count: locomotionActionCount)
                    + [Float](repeating: 1,
                              count: Self.actionDimension
                                - locomotionActionCount))
        }
        let carryLegExpertFraction = 1
            - configuration.carryBaseLegActionFraction
        if configuration.compositionalCarryController {
            return ContiguousArray(
                [Float](repeating: carryLegExpertFraction, count: 10)
                    + [Float](repeating: 0, count: 9))
        }
        return ContiguousArray(
            [Float](repeating: carryLegExpertFraction, count: 11)
                + [Float](repeating: 1, count: 8))
    }

    public var usesPolicyAuxiliaryExpertGate: Bool {
        configuration.upperBodyCarryController
    }

    public var initializesPolicyAuxiliaryExpertFromBaseOnTransfer: Bool {
        configuration.initializeCarryLocomotionExpertFromBaseOnTransfer
    }

    public var policyAuxiliaryExpertZeroedObservationIndicesOnTransfer: [Int] {
        // The imported walker was trained around nominal arm coordinates. At
        // the physical lift boundary, the pickup branch instead leaves large
        // arm angles, velocities, and previous actions. Feeding that unrelated
        // upper-body state into a leg-only actor destroys the gait prior even
        // though its arm outputs are masked. Begin from the source policy's
        // nominal zero influence for those eight joints; PPO may learn useful
        // load coupling into the still-trainable columns afterward.
        let arms = 11..<Self.actionDimension
        return arms.map { 12 + $0 }
            + arms.map { 31 + $0 }
            + arms.map { 50 + $0 }
    }

    public var freezesStandPolicyExpert: Bool {
        configuration.upperBodyCarryController
            && configuration.freezeCarryPolicyExpert
    }

    public var policyAuxiliaryExpertActionMask: ContiguousArray<Float>? {
        guard configuration.upperBodyCarryController else { return nil }
        let locomotionActionCount = configuration
                .carryLocomotionControlsTorso ? 11 : 10
        return ContiguousArray(
            [Float](repeating: 1, count: locomotionActionCount)
                + [Float](repeating: 0,
                          count: Self.actionDimension
                            - locomotionActionCount))
    }

    public func policyAuxiliaryExpertGates(
        _ observations: ContiguousArray<Float>
    ) -> ContiguousArray<Float> {
        precondition(observations.count
            == spec.numEnvironments * Self.observationDimension)
        let learnedFraction = 1 - configuration.carryBaseLegActionFraction
        return ContiguousArray((0..<spec.numEnvironments).map { e in
            learnedFraction * simd_clamp(
                observations[e * Self.observationDimension + 90], 0, 1)
        })
    }

    public func policyExpertGates(
        _ observations: ContiguousArray<Float>
    ) -> ContiguousArray<Float> {
        precondition(observations.count
            == spec.numEnvironments * Self.observationDimension)
        return ContiguousArray((0..<spec.numEnvironments).map { e in
            let base = e * Self.observationDimension
            guard observations[base + 87] > 0.5 else { return 0 }
            let manipulation = simd_clamp(observations[base + 102], 0, 1)
            if configuration.compositionalCarryController {
                // Arms remain under the successful frozen manipulation expert
                // throughout transport; the disjoint stand mask independently
                // blends only loaded-locomotion actions below the waist.
                return manipulation
            }
            // Both the historical whole-body handoff and the upper-body-only
            // handoff cross-fade pickup to carry after a physical lift. Their
            // action masks determine whether the lower actor also changes.
            let carry = simd_clamp(observations[base + 90], 0, 1)
            return manipulation * (1 - carry)
        })
    }

    public func policyStandExpertGates(
        _ observations: ContiguousArray<Float>
    ) -> ContiguousArray<Float> {
        precondition(observations.count
            == spec.numEnvironments * Self.observationDimension)
        return ContiguousArray((0..<spec.numEnvironments).map { e in
            simd_clamp(
                observations[e * Self.observationDimension + 90], 0, 1)
        })
    }

    private func planarDistance(_ a: F3, _ b: F3) -> Float {
        let delta = a - b
        return sqrt(delta.x * delta.x + delta.y * delta.y)
    }

    private func orientedBoxBottom(
        _ box: GPUSolver.RigidBodyState
    ) -> Float {
        let half = 0.5 * Self.boxDimensions
        let xAxis = box.rotation.act(F3(1, 0, 0))
        let yAxis = box.rotation.act(F3(0, 1, 0))
        let zAxis = box.rotation.act(F3(0, 0, 1))
        let verticalExtent = abs(xAxis.z) * half.x
            + abs(yAxis.z) * half.y
            + abs(zAxis.z) * half.z
        return box.position.z - verticalExtent
    }
}
